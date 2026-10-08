"""The install interview: a numbered picture of the camera, the owner's answers, and the scene map they make.

Doc "ראיון ההתקנה ומפת הסצנה", v0. The owner draws nothing: the box numbers what it sees (Set-of-Mark: a number
on each object, so a question can point at "3" without confusion), asks whose each one is, and shows the map it
understood, coloured by ownership, for one "is this right?".

- **Numbering.** ``propose_regions`` uses SAM through ultralytics (FastSAM-s, downloaded by ultralytics on first
  use) when it can load; offline or without the weights it numbers a 3 x 4 grid instead and says so
  (``method``: ``sam`` or ``grid``). The picture (``interview_picture``) hides the scene map's black areas but
  shows what today's drawn zone blacks out, outlined in white: the owner is asked about it (stage 2c: the
  neighbour's side becomes seen-not-alerted once the owner confirms). Small specks and duplicates are dropped;
  the largest regions get the first numbers.
- **Questions.** At most MAX_QUESTIONS per camera, each with "skip" and "don't know" (``questions``).
- **Answers.** ``parse_answers`` reads the owner's words, English or Hebrew: each number (or "3,5", "3 ו-5") with
  the words after it ("1 שלי, 2 של השכן, 3 רחוב"; "7 is the neighbour's car"): mine -> ``mine``; the
  neighbour's -> ``watch_no_alert`` (neighbour); street/road/public -> ``watch_no_alert`` (public);
  hide/black/privacy -> ``black``. A quoted name ('the dog house') becomes the area's name; zone words (gate,
  driveway, lawn...) pick the taxonomy zone. Words with no ownership in them are not guessed. ``parse_lines``
  reads boundaries ("המעקה בין 2 ל-1"): ``line_between`` draws the line where the two regions meet, inward
  towards the one the owner called theirs.
- **Draft, then confirm.** ``answer`` makes a draft and the confirmation picture, and saves nothing;
  ``confirm`` (the owner's [save]) makes it the camera's whole truth (``scene_map.confirm_scene_map``).

UI contract (the Telegram flow is ``scene_chat.py``; an app dialog would use the same three steps):

1. ``propose(camera, picture, out_dir, watched=...)`` -> ``{"camera", "method", "image" (numbered JPEG),
   "regions_path", "regions": [{"number", "area"}], "questions": [{"number", "text", "options"}]}``.
2. ``answer(camera, text, regions_path, picture, out_dir)`` -> ``{"camera", "map" (the map as it will be
   confirmed), "image" (the ownership picture: mine blue, neighbour orange, public grey, black black), "notes",
   "draft"}``. Show the image with [save] / [fix]; [fix] asks for the words again.
3. ``confirm(camera, out_dir, known_cameras=...)`` -> ``{"camera", "map", "restart_needed", "stale_removed"}``.
   ``restart_needed``: the frame mask changed; restart the running mode (the CLI does it).

CLI (on the box; the app may call these over ssh, like find_cameras' zone commands)::

    python -m home_guard_project.box.scene_interview telegram --camera front     # change its map in the owner's chat
    python -m home_guard_project.box.scene_interview propose --camera front [--grid] [--json]
    python -m home_guard_project.box.scene_interview answer --camera front --text "1,3 mine; 2 hide it" [--json]
    python -m home_guard_project.box.scene_interview confirm --camera front [--no-restart]
    python -m home_guard_project.box.scene_interview show --camera front
    python -m home_guard_project.box.scene_interview clear --camera front
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import paths
from . import scene_map as sm
from . import taxonomy as tx

log = logging.getLogger("box.scene_interview")

INTERVIEW_DIR = paths.scene_interview_dir()
MAX_REGIONS = 12
MAX_QUESTIONS = 7
GRID_ROWS, GRID_COLS = 3, 4
MIN_AREA, MAX_AREA = 0.01, 0.7          # of the picture; smaller is a speck, larger is the whole view
DUPLICATE_IOU = 0.8
BLACK_LEVEL, BLACK_SHARE = 8, 0.9       # a region this much black is the masked outside: never numbered
FASTSAM_WEIGHTS = "FastSAM-s.pt"      # a bare name: a file in the box's models folder (paths.resolve_model)
OPTIONS = ("mine", "the neighbour's", "public / street", "hide it (black)", "skip", "don't know")

# BGR, for the ownership picture.
GROUND_COLOURS = {sm.MINE: (255, 120, 0), sm.NEIGHBOUR: (0, 140, 255), sm.PUBLIC: (160, 160, 160)}
_PALETTE = ((66, 135, 245), (245, 66, 230), (66, 245, 135), (245, 200, 66), (66, 230, 245), (180, 66, 245),
            (245, 105, 66), (120, 245, 66), (66, 66, 245), (245, 66, 120), (66, 180, 180), (200, 200, 66))

Segmenter = Callable[[np.ndarray], Sequence[np.ndarray]]      # picture -> boolean masks (H x W)


@dataclass(frozen=True)
class Region:
    number: int
    points: Tuple[sm.Point, ...]        # normalised polygon, like a watch zone
    area: float                          # share of the picture

    def to_dict(self) -> Dict[str, Any]:
        return {"number": self.number, "points": [list(p) for p in self.points], "area": round(self.area, 4)}


# ----------------------------------------------------------------------------
# Numbering the picture
# ----------------------------------------------------------------------------
def grid_segmenter(image: np.ndarray) -> List[np.ndarray]:
    """The fallback: GRID_ROWS x GRID_COLS cells, row by row."""
    h, w = image.shape[:2]
    masks = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            m = np.zeros((h, w), bool)
            m[r * h // GRID_ROWS:(r + 1) * h // GRID_ROWS, c * w // GRID_COLS:(c + 1) * w // GRID_COLS] = True
            masks.append(m)
    return masks


grid_segmenter.method = "grid"          # type: ignore[attr-defined]


def fastsam_segmenter(weights: str = FASTSAM_WEIGHTS) -> Segmenter:
    """SAM (FastSAM) through ultralytics. Raises when ultralytics or the weights are not available."""
    from ultralytics import FastSAM  # noqa: PLC0415

    model = FastSAM(paths.resolve_model(weights))

    def segment(image: np.ndarray) -> List[np.ndarray]:
        results = model(image, retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)
        masks = results[0].masks if results else None
        if masks is None:
            return []
        data = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else np.asarray(masks.data)
        return [m > 0.5 for m in data]

    segment.method = "sam"              # type: ignore[attr-defined]
    return segment


def _polygon(mask: np.ndarray) -> Optional[Tuple[sm.Point, ...]]:
    """The mask's outline as a normalised polygon of 3 to 32 corners (its largest part), or None."""
    import cv2  # noqa: PLC0415

    from ..data_collection.zones import MAX_POINTS  # noqa: PLC0415

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    while len(approx) > MAX_POINTS:
        eps *= 1.5
        approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 3:
        return None
    return tuple((round(min(1.0, max(0.0, float(x) / max(1, w - 1))), 4),
                  round(min(1.0, max(0.0, float(y) / max(1, h - 1))), 4)) for x, y in approx.reshape(-1, 2))


def propose_regions(image: np.ndarray, segmenter: Optional[Segmenter] = None,
                    max_regions: int = MAX_REGIONS) -> Tuple[List[Region], str]:
    """``(regions, method)``: numbered regions of *image*, largest first. Without a working SAM, a grid."""
    import cv2  # noqa: PLC0415

    if segmenter is None:
        try:
            segmenter = fastsam_segmenter()
        except Exception as exc:  # noqa: BLE001 - offline or no weights: the grid still lets the owner answer
            log.warning("SAM is not available (%s); numbering a grid instead", exc)
            segmenter = grid_segmenter
    method = getattr(segmenter, "method", "sam")
    try:
        masks = list(segmenter(image))
    except Exception as exc:  # noqa: BLE001
        log.warning("Segmentation failed (%s); numbering a grid instead", exc)
        masks, method = grid_segmenter(image), "grid"
    h, w = image.shape[:2]
    black = np.all(image[..., :3] < BLACK_LEVEL, axis=-1) if image.ndim == 3 else image < BLACK_LEVEL
    kept: List[Tuple[np.ndarray, float]] = []
    for m in masks:
        m = np.asarray(m).astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        size = int(m.sum())
        share = size / float(h * w)
        if not MIN_AREA <= share <= MAX_AREA or (black & m).sum() >= BLACK_SHARE * size:
            continue
        kept.append((m, share))
    kept.sort(key=lambda ms: -ms[1])
    regions: List[Region] = []
    taken: List[np.ndarray] = []
    for m, share in kept:
        if any((m & t).sum() / float((m | t).sum()) > DUPLICATE_IOU for t in taken):
            continue
        polygon = _polygon(m)
        if polygon is None:
            continue
        taken.append(m)
        regions.append(Region(len(regions) + 1, polygon, share))
        if len(regions) >= max_regions:
            break
    return regions, method


def _fill(shape: Tuple[int, int], points: Sequence[sm.Point]) -> np.ndarray:
    import cv2  # noqa: PLC0415

    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    corners = np.array([[int(round(x * (w - 1))), int(round(y * (h - 1)))] for x, y in points], np.int32)
    cv2.fillPoly(mask, [corners], 255)
    return mask


def _label_point(mask: np.ndarray) -> Tuple[int, int]:
    """The point deepest inside the region, where its number reads best."""
    import cv2  # noqa: PLC0415

    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


def numbered_overlay(image: np.ndarray, regions: Sequence[Region]) -> np.ndarray:
    """The picture with each region tinted, outlined and numbered (a new array)."""
    import cv2  # noqa: PLC0415

    out = image.copy()
    tint = image.copy()
    h, w = image.shape[:2]
    scale = max(0.4, min(h, w) / 600.0)
    for r in regions:
        colour = _PALETTE[(r.number - 1) % len(_PALETTE)]
        mask = _fill((h, w), r.points)
        tint[mask > 0] = colour
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    masks = [_fill((h, w), r.points) for r in regions]
    for i, r in enumerate(regions):
        colour = _PALETTE[(r.number - 1) % len(_PALETTE)]
        mask = masks[i]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, max(1, int(2 * scale)))
        # The number goes on the part no later (smaller) region covers, so "1" never sits on top of "7".
        visible = mask.copy()
        for later in masks[i + 1:]:
            visible[later > 0] = 0
        x, y = _label_point(visible if visible.any() else mask)
        text = str(r.number)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(2 * scale)))
        radius = int(max(tw, th) * 0.8) + 2
        cv2.circle(out, (x, y), radius, (255, 255, 255), -1)
        cv2.circle(out, (x, y), radius, (0, 0, 0), 1)
        cv2.putText(out, text, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    max(1, int(2 * scale)), cv2.LINE_AA)
    return out


def ownership_overlay(image: np.ndarray, scene: sm.SceneMap) -> np.ndarray:
    """The confirmation picture: mine blue, neighbour orange, public grey, black black; lines in white with an
    arrow towards the owner's side (a new array)."""
    import cv2  # noqa: PLC0415

    h, w = image.shape[:2]
    tint = image.copy()
    implicit = [a for a in scene.all_areas() if a.implicit]
    stored = sorted((a for a in scene.areas), key=lambda a: -sm.polygon_area(a.points))   # innermost on top
    for area in implicit + stored:
        colour = (0, 0, 0) if area.kind == sm.BLACK else GROUND_COLOURS.get(area.ground, (160, 160, 160))
        tint[_fill((h, w), area.points) > 0] = colour
    out = cv2.addWeighted(tint, 0.45, image, 0.55, 0)
    for area in scene.areas:
        if area.kind == sm.BLACK:
            out[_fill((h, w), area.points) > 0] = 0
    for line in scene.lines:
        a = (int(round(line.a[0] * (w - 1))), int(round(line.a[1] * (h - 1))))
        b = (int(round(line.b[0] * (w - 1))), int(round(line.b[1] * (h - 1))))
        cv2.line(out, a, b, (255, 255, 255), 2, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = max(1.0, float(np.hypot(dx, dy)))
        # "left of travel" on the picture (y down) is (dy, -dx).
        sx, sy = (dy / norm, -dx / norm) if line.inward == "left" else (-dy / norm, dx / norm)
        tip = (int(mid[0] + sx * 0.06 * w), int(mid[1] + sy * 0.06 * h))
        cv2.arrowedLine(out, mid, tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
    return out


def questions(regions: Sequence[Region], limit: int = MAX_QUESTIONS) -> List[Dict[str, Any]]:
    """One short question per region, the first *limit* regions (largest first)."""
    return [{"number": r.number, "text": f"What is {r.number}, and whose is it?", "options": list(OPTIONS)}
            for r in list(regions)[:limit]]


# ----------------------------------------------------------------------------
# The owner's answers
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Answer:
    number: int
    kind: str
    owner: str = ""
    name: str = ""
    zone: str = "other"


def _words(*words: str) -> re.Pattern:
    english = [w for w in words if re.match(r"[A-Za-z]", w)]
    hebrew = [w for w in words if not re.match(r"[A-Za-z]", w)]
    parts = ([r"\b(?:" + "|".join(english) + r")\b"] if english else []) + hebrew
    return re.compile("|".join(parts), re.IGNORECASE)


# Checked in this order: "my neighbour's" is the neighbour's, not mine.
_SKIP = _words("skip", "don't know", "dont know", "not sure", "דלג", "לא יודע", "לא יודעת", "לא בטוח")
_BLACK = _words("black", "hide", "hidden", "privacy", "blur", "don't look", "שחור", "להסתיר", "הסתר", "תסתיר", "תסתירו", "פרטיות")
_NEIGHBOUR = _words("neighbou?r'?s?", "next door", "שכן", "שכנה", "שכנים")
_PUBLIC = _words("street", "road", "sidewalk", "pavement", "public", "כביש", "רחוב", "מדרכה", "ציבורי")
_MINE = _words("mine", "my", "ours", "our", "שלי", "שלנו")
_ZONE_WORDS = (
    ("parking", _words("parking", "driveway", "garage", "carport", "car park", "חניה", "חנייה", "חניון")),
    ("street", _words("street", "road", "sidewalk", "pavement", "כביש", "רחוב", "מדרכה")),
    ("car", _words("car", "vehicle", "van", "truck", "רכב", "אוטו", "מכונית")),
    ("gate", _words("gate", "שער")),
    ("fence", _words("fence", "railing", "wall", "hedge", "גדר", "מעקה", "חומה")),
    ("window", _words("window", "חלון")),
    ("entrance", _words("door", "entrance", "stairs", "steps", "porch", "כניסה", "דלת", "מדרגות")),
    ("roof", _words("roof", "גג")),
    ("yard", _words("yard", "garden", "lawn", "grass", "pergola", "patio", "חצר", "גינה", "דשא", "פרגולה")),
)
_FILLER = {"is", "are", "it", "the", "a", "an", "and", "of", "זה", "זאת", "הם", "הן", "הוא", "היא", "את"}
_KIND_ONLY = {"mine", "ours", "black", "hide", "skip", "public", "שלי", "שלנו", "שחור", "להסתיר", "תסתיר", "דלג"}
_CLAUSES = re.compile(r"[;\n.]|\band\b", re.IGNORECASE)
# A quoted name starts after a space and ends before one, so an apostrophe ("neighbour's") never opens one.
_QUOTED = re.compile(r"(?:^|(?<=\s))['\"“”׳]([^'\"“”׳]{1,40})['\"“”׳]"
                     r"(?=\s|$|[,;.])")


def _kind_of(clause: str) -> Optional[Tuple[str, str]]:
    if _SKIP.search(clause):
        return None
    if _BLACK.search(clause):
        return sm.BLACK, ""
    if _NEIGHBOUR.search(clause):
        return sm.WATCH, sm.NEIGHBOUR
    if _PUBLIC.search(clause):
        return sm.WATCH, sm.PUBLIC
    if _MINE.search(clause):
        return sm.MINE, ""
    return None


def _zone_of(text: str) -> str:
    return next((zone for zone, pattern in _ZONE_WORDS if pattern.search(text)), "other")


_PUNCT = ",-;.!?"


def _name_of(clause: str) -> str:
    quoted = _QUOTED.search(clause)
    if quoted:
        return sm.plain_name(quoted.group(1))
    words = [w for w in re.sub(r"\d+", " ", clause).replace(":", " ").replace("=", " ").split()
             if w.strip(_PUNCT).lower() not in _KIND_ONLY]
    while words and words[0].strip(_PUNCT).lower() in _FILLER:
        words.pop(0)
    while words and words[-1].strip(_PUNCT).lower() in _FILLER:
        words.pop()
    name = " ".join(words).strip(" " + _PUNCT)
    return sm.plain_name(name) if name.lower() not in _KIND_ONLY else ""


# Clauses end at ; . a line break, or a comma / "and" that is not followed by another number ("3,5" stays one group).
_CLAUSE_SPLIT = re.compile(r"[;\n.]|,(?!\s*\d)|\band\b(?!\s*\d)", re.IGNORECASE)
_GROUP = re.compile(r"\d+(?:\s*(?:,|ו-?|\band\b|&)\s*\d+)*", re.IGNORECASE)
# "המעקה בין 2 ל-1", "the railing between 2 and 1": a boundary line where two numbered regions meet.
_LINE_CLAUSE = re.compile(r"(?P<name>[^\d,;.\n]{1,40}?)\s*(?:\bbetween\b|בין)\s*(?P<a>\d+)\s*(?:ל-?|ו-?|\band\b|&|,|-)\s*"
                          r"(?P<b>\d+)", re.IGNORECASE)


def _hide_quotes(text: str) -> str:
    """*text* with every quoted name blanked out (same length), so its digits are never region numbers."""
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), text)


def parse_lines(text: str) -> List[Tuple[str, int, int]]:
    """``(name, a, b)`` for each "the railing between 2 and 1" in the owner's words."""
    out = []
    for m in _LINE_CLAUSE.finditer(_hide_quotes(str(text or ""))):
        words = m.group("name").split()
        while words and words[0].strip(",-").lower() in _FILLER:
            words.pop(0)
        out.append((sm.plain_name(" ".join(words)) or "line", int(m.group("a")), int(m.group("b"))))
    return out


def parse_answers(text: str) -> List[Answer]:
    """The owner's answers: each number (or group, "3,5" / "3 ו-5") with the words after it, up to the next number
    ("1 שלי, 2 של השכן, 3 רחוב"). Boundary lines ("X בין 2 ל-1") are left to ``parse_lines``; skips and words with
    no ownership in them are left out."""
    text = str(text or "")
    hidden = _hide_quotes(text)
    for m in _LINE_CLAUSE.finditer(hidden):                    # the line clauses' numbers are not answers
        text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
        hidden = hidden[:m.start()] + " " * (m.end() - m.start()) + hidden[m.end():]
    out: List[Answer] = []
    start = 0
    for cut in list(_CLAUSE_SPLIT.finditer(hidden)) + [None]:
        end = cut.start() if cut is not None else len(text)
        clause, shown = text[start:end], hidden[start:end]
        start = cut.end() if cut is not None else len(text)
        groups = list(_GROUP.finditer(shown))
        for i, m in enumerate(groups):
            words = clause[m.end():groups[i + 1].start() if i + 1 < len(groups) else len(clause)]
            kind = _kind_of(words)
            if kind is None and i == 0:
                words = clause[:m.start()]                     # "להסתיר את 9": the words come first
                kind = _kind_of(words)
            if kind is None:
                continue
            name, zone = _name_of(words), _zone_of(words)
            out.extend(Answer(int(n), kind[0], kind[1], name, zone) for n in re.findall(r"\d+", m.group(0)))
    return out


def _mostly_inside(inner: Sequence[sm.Point], outer: Sequence[sm.Point], share: float = 0.8) -> bool:
    """At least *share* of polygon *inner* lies inside polygon *outer*."""
    size = 200
    a, b = _fill((size, size), inner) > 0, _fill((size, size), outer) > 0
    total = int(a.sum())
    return total > 0 and int((a & b).sum()) >= share * total


def _work_mask(region: Region, size: int = 200) -> np.ndarray:
    return _fill((size, size), region.points) > 0


def line_between(name: str, a: Region, b: Region, ours: Optional[Region] = None) -> sm.Line:
    """The boundary where regions *a* and *b* meet, its inward side towards *ours* (one of them). ValueError in
    plain words when they do not touch, or when neither is the owner's."""
    import cv2  # noqa: PLC0415

    if ours is None:
        raise ValueError(f"which side of {name} is yours? Say {a.number} or {b.number} is yours")
    size = 200
    ma, mb = _work_mask(a, size).astype(np.uint8), _work_mask(b, size).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    border = np.zeros_like(ma)
    for grow in (1, 3, 6, 10):
        border = cv2.dilate(ma, kernel, iterations=grow) & cv2.dilate(mb, kernel, iterations=grow)
        if int(border.sum()) >= 3:
            break
    else:
        raise ValueError(f"{a.number} and {b.number} do not touch, so there is no line between them")
    ys, xs = np.nonzero(border)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1))
    t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy

    def norm(v: float) -> float:
        return round(min(1.0, max(0.0, v / (size - 1))), 4)

    p = (norm(x0 + vx * float(t.min())), norm(y0 + vy * float(t.min())))
    q = (norm(x0 + vx * float(t.max())), norm(y0 + vy * float(t.max())))
    if p == q:
        raise ValueError(f"{a.number} and {b.number} touch at a single point only")
    x, y = _label_point(_work_mask(ours, size).astype(np.uint8))
    return sm.Line.toward(name, p, q, (norm(x), norm(y)))


def apply_answers(scene: sm.SceneMap, regions: Sequence[Region], answers: Sequence[Answer],
                  lines: Sequence[Tuple[str, int, int]] = ()) -> Tuple[sm.SceneMap, List[str]]:
    """The map with an area for each answered region (an area answered again is replaced) and a boundary for
    each "X between 2 and 1", and notes for the owner about what could not be used."""
    by_number = {r.number: r for r in regions}
    areas = list(scene.areas)
    notes: List[str] = []
    ours: set = set()
    for ans in answers:
        region = by_number.get(ans.number)
        if region is None:
            notes.append(f"there is no {ans.number} in the picture")
            continue
        name = ans.name or (ans.zone if ans.zone != "other" else f"area {ans.number}")
        area = sm.Area(name, ans.kind, ans.zone if ans.zone in tx.ZONES else "other", region.points,
                       owner=ans.owner if ans.kind == sm.WATCH else "")
        # The newest word wins where it was said: an area of the map that lies (mostly) inside the answered region
        # is replaced; the others keep their kind (the change merges, it never resets the map). A black area is
        # never lifted by a word about something else: privacy goes only when the owner removes it.
        areas = [a for a in areas if a.points != area.points
                 and (a.kind == sm.BLACK or not _mostly_inside(a.points, area.points))] + [area]
        if ans.kind == sm.MINE:
            ours.add(ans.number)
    found = list(scene.lines)
    for name, n1, n2 in lines:
        if n1 not in by_number or n2 not in by_number:
            notes.append(f"there is no {n1 if n1 not in by_number else n2} in the picture")
            continue
        mine = n1 if n1 in ours else n2 if n2 in ours else None
        try:
            line = line_between(name, by_number[n1], by_number[n2], by_number[mine] if mine else None)
        except ValueError as exc:
            notes.append(str(exc))
            continue
        found = [ln for ln in found if ln.name != line.name] + [line]
    return sm.SceneMap(scene.camera, tuple(areas), tuple(found), scene.watched, rest=scene.rest,
                       rest_owner=scene.rest_owner), notes


# ----------------------------------------------------------------------------
# The steps, with their files
# ----------------------------------------------------------------------------
def _stem(camera: str) -> str:
    return re.sub(r"[^\w.-]", "_", str(camera)) or "camera"


def propose(camera: str, picture: np.ndarray, out_dir: str = INTERVIEW_DIR, segmenter: Optional[Segmenter] = None,
            picture_path: str = "", watched: Optional[Sequence[sm.Point]] = None) -> Dict[str, Any]:
    """Step 1: number the camera's picture and write the numbered JPEG and the regions file. *watched* (today's
    drawn zone) is outlined, so the owner sees what was watched until now."""
    import cv2  # noqa: PLC0415

    regions, method = propose_regions(picture, segmenter)
    os.makedirs(out_dir, exist_ok=True)
    image = numbered_overlay(picture, regions)
    if watched:
        h, w = image.shape[:2]
        contours, _ = cv2.findContours(_fill((h, w), watched), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    image_path = os.path.join(out_dir, f"{_stem(camera)}_numbered.jpg")
    if not cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"could not write {image_path}")
    regions_path = os.path.join(out_dir, f"{_stem(camera)}_regions.json")
    with open(regions_path, "w", encoding="utf-8") as f:
        json.dump({"camera": camera, "method": method, "picture": picture_path,
                   "regions": [r.to_dict() for r in regions]}, f, ensure_ascii=False, indent=1)
    return {"camera": camera, "method": method, "image": image_path, "regions_path": regions_path,
            "regions": [{"number": r.number, "area": round(r.area, 4)} for r in regions],
            "questions": questions(regions)}


def load_regions(regions_path: str, camera: str) -> List[Region]:
    with open(regions_path, encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("camera")) != str(camera):
        raise ValueError(f"those numbers belong to camera {data.get('camera')!r}, not {camera!r}")
    return [Region(int(r["number"]), tuple(tuple(p) for p in r["points"]), float(r.get("area") or 0.0))
            for r in data.get("regions") or []]


def draft_path(camera: str, out_dir: str = INTERVIEW_DIR) -> str:
    return os.path.join(out_dir, f"{_stem(camera)}_draft.json")


def answer(camera: str, text: str, regions_path: str, picture: Optional[np.ndarray] = None,
           out_dir: str = INTERVIEW_DIR, zones_path: Optional[str] = None) -> Dict[str, Any]:
    """Step 2: the owner's answers into a DRAFT of the camera's map (nothing is saved: the owner confirms it with
    ``confirm``), and the ownership picture when *picture* is given."""
    import cv2  # noqa: PLC0415

    regions = load_regions(regions_path, camera)
    before = sm.load_scene_map(camera, zones_path)
    scene, notes = apply_answers(before, regions, parse_answers(text), parse_lines(text))
    if not scene.areas and not scene.lines:
        notes.append("nothing to save yet: say whose each number is")
    # What the confirmed map will look like: today's zone as ours, the rest the neighbour's.
    preview = confirmed_preview(scene)
    os.makedirs(out_dir, exist_ok=True)
    with open(draft_path(camera, out_dir), "w", encoding="utf-8") as f:
        json.dump(scene.to_dict(), f, ensure_ascii=False, indent=1)
    image_path = None
    if picture is not None:
        image_path = os.path.join(out_dir, f"{_stem(camera)}_map.jpg")
        if not cv2.imwrite(image_path, ownership_overlay(picture, preview), [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"could not write {image_path}")
    return {"camera": camera, "map": preview.to_dict(), "image": image_path, "notes": notes,
            "draft": draft_path(camera, out_dir)}


def confirmed_preview(scene: sm.SceneMap) -> sm.SceneMap:
    """The map as ``scene_map.confirm_scene_map`` will make it (for the confirmation picture), without saving."""
    areas = list(scene.areas)
    rest, owner = scene.rest, scene.rest_owner
    if scene.watched:
        if not any(a.points == tuple(scene.watched) for a in areas):
            areas.append(sm.Area(sm.WATCHED_NAME, sm.MINE, "other", tuple(scene.watched)))
        if not rest:
            rest, owner = sm.WATCH, sm.NEIGHBOUR
    return sm.SceneMap(scene.camera, tuple(areas), scene.lines, None, rest=rest, rest_owner=owner)


def confirm(camera: str, out_dir: str = INTERVIEW_DIR, zones_path: Optional[str] = None,
            known_cameras: Sequence[str] = ()) -> Dict[str, Any]:
    """Step 3, the owner's [save]: the draft becomes the camera's confirmed map (``scene_map.confirm_scene_map``).
    Stale keys of the same channel (an old site name, not among *known_cameras*) are dropped with it.
    ``restart_needed`` when the frame mask changed."""
    path = draft_path(camera, out_dir)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("camera")) != str(camera):
        raise ValueError(f"that draft belongs to camera {data.get('camera')!r}, not {camera!r}")
    draft = sm.SceneMap.from_dict(camera, data, data.get("watched") or None)
    stale = stale_keys(camera, known_cameras, zones_path) if known_cameras else []
    saved, changed = sm.confirm_scene_map(draft, zones_path, stale=stale)
    try:
        os.remove(path)
    except OSError:
        pass
    return {"camera": camera, "map": saved.to_dict(), "restart_needed": changed, "stale_removed": stale}


def stale_keys(camera: str, known_cameras: Sequence[str], zones_path: Optional[str] = None) -> List[str]:
    """Zone and scene-map keys that are no camera of this box but share *camera*'s channel (``ameer_test_ch2``
    next to ``ameer_week_0_1_ch2``): left over from a site rename."""
    from ..data_collection.zones import ZONES_PATH, _read_raw, read_scene_maps, scene_maps_path_for  # noqa: PLC0415
    from .camera_names import channel_of  # noqa: PLC0415

    zp = zones_path or ZONES_PATH
    keys = set(_read_raw(zp)) | set(read_scene_maps(scene_maps_path_for(zp)))
    channel = channel_of(camera)
    known = {str(c) for c in known_cameras}
    return sorted(k for k in keys if k not in known and channel is not None and channel_of(k) == channel)


def interview_picture(camera: str, cameras_path: str, out_dir: str = INTERVIEW_DIR) -> Tuple[np.ndarray, str]:
    """A current picture for the interview: the scene map's black areas hidden, today's black outside shown (the
    owner is asked about it). Raises RuntimeError in plain words."""
    from .live_view import grab_masked  # noqa: PLC0415

    grabbed = grab_masked(camera, cameras_path, out_dir, keep_zone=False)
    if "error" in grabbed:
        raise RuntimeError(grabbed["error"])
    picture = _read_picture(grabbed["image"])
    if picture is None:
        raise RuntimeError(f"could not read the picture from {camera}")
    return picture, grabbed["image"]


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def _read_picture(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    import cv2  # noqa: PLC0415

    return cv2.imread(path)


def _known_cameras(cameras_path: str) -> List[str]:
    from .find_cameras import _read_cameras_raw  # noqa: PLC0415

    raw = _read_cameras_raw(cameras_path)
    return list(dict(raw.get("cameras") or {})) + list(dict(raw.get("disabled") or {}))


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .find_cameras import CAMERAS_PATH, _known_camera, _restart_running_mode  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="scene_interview", description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("propose", "answer", "confirm", "show", "clear", "telegram"):
        p = sub.add_parser(name)
        p.add_argument("--camera", required=name != "telegram", default="")
        p.add_argument("--json", action="store_true")
        if name in ("propose", "answer", "confirm"):
            p.add_argument("--out", default=INTERVIEW_DIR)
        if name == "propose":
            p.add_argument("--grid", action="store_true", help="number a grid instead of using SAM")
        if name == "answer":
            p.add_argument("--text", required=True)
        if name == "confirm":
            p.add_argument("--no-restart", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "telegram":
            from .scene_chat import start_from_cli  # noqa: PLC0415

            result = start_from_cli(args.camera or "")
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else json.dumps(result, ensure_ascii=False))
            return 0 if "error" not in result else 1
        camera = _known_camera(args.camera, CAMERAS_PATH)
        if args.command == "propose":
            picture, picture_path = interview_picture(camera, CAMERAS_PATH, args.out)
            result = propose(camera, picture, args.out, grid_segmenter if args.grid else None, picture_path,
                             watched=sm.load_scene_map(camera).watched)
        elif args.command == "answer":
            regions_path = os.path.join(args.out, f"{_stem(camera)}_regions.json")
            with open(regions_path, encoding="utf-8") as f:
                picture_path = json.load(f).get("picture") or ""
            result = answer(camera, args.text, regions_path, _read_picture(picture_path), args.out)
        elif args.command == "confirm":
            result = confirm(camera, args.out, known_cameras=_known_cameras(CAMERAS_PATH))
            if result["restart_needed"] and not args.no_restart:
                _restart_running_mode()
        elif args.command == "show":
            result = sm.load_scene_map(camera).to_dict()
        else:
            result = {"camera": camera, "cleared": sm.clear_scene_map(camera)}
            _restart_running_mode()
    except Exception as exc:  # noqa: BLE001 - the app needs a plain error, never a traceback
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else f"could not do that: {exc}"
        print(json.dumps({"error": message}) if args.json else f"Error: {message}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
