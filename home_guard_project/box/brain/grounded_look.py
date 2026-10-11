"""A grounded live look: the detector and the owner's scene map say WHAT is there and WHOSE ground it is on; the
vision model only describes. Code checks its answer against them.

2026-10-10 17:03, camera 2 (the owner had just drawn the map with the boundary to the neighbour): "מה קורה במצלמה
2" got "יש אדם בלבוש כהה הולך ליד צד הבית" while the detector found no person (yolo11s: 0.06 on that frame), and "של
מי הרכב" got a description instead of the map's answer (the pickup stands beyond the boundary, at the neighbour's).

1. ``detect(...)``: the objects (people, vehicles, animals) with boxes and confidence on THAT frame - the box's own
   detector run on the photo (``Detector``), else the live detector's look (logs/ai_status.json) when it is at most
   ``FRESH_SEC`` old.
2. ``place(...)``: each object's area on the camera's scene map, through the scene map's own ``area_at`` (no
   geometry here): a person by the feet, a vehicle by the lower part of its box. Areas of structures (a fence, a
   window) only count when no ground area does - a pickup behind the owner's wall is not on the wall. A point in no
   area asks ``scene_map.boundary_ground(scene, point)`` (the side of the boundary lines) when that exists; else it
   is "not mapped". The map is found by the camera's name, else by its channel (a site rename: ameer_v2_ch2 keeps
   the map drawn as ameer_week_0_1_ch2).
3. ``facts_text(...)``: the facts block for the vision prompt.
4. ``enforce(...)``: the post-check. With no person detected, an answer that states a person is rewritten from the
   facts ("לא רואה אנשים במצלמה 2 עכשיו"); a person the model still thinks it saw is not told at all (2026-10-11, the
   owner: no assumption the detector did not make). A vehicle it alone saw is kept only as "ייתכן שיש רכב, לא בטוח".
   Logged.
6. The second look (2026-10-11): a person the detector is not sure of (below PERSON_SURE) on the full-resolution
   picture is looked at again on a crop around it (``Detector.zoom``); unconfirmed it is no fact.
5. ``map_info`` / ``where_is``: the map tools' answers ("של מי הרכב?" -> "הטנדר חונה בשטח של השכן (לפי המפה
   שציירת)"); without a map an honest "no map yet" and the offer to draw it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...prompts import load
from .. import scene_map as sm

log = logging.getLogger("box.brain.grounded_look")

FRESH_SEC = 2.0                     # the live detector's look counts for a photo taken this close to it
LOOK_CONF = {"person": 0.35, "vehicle": 0.4, "animal": 0.4}   # what this look counts as seen
# 2026-10-11 23:54, the pergola at night: "בפרגולה יש אדם ליד נגרר" - the detector gave stacked chairs "person" 0.52
# on the 960-wide picture and the vision model was told it as a fact. The owner: a person is said only when the
# detector is sure on the full-resolution picture. Below PERSON_SURE a person needs the second look (zoom) to find
# one at ZOOM_CONF or more; else it is not a fact, and nothing tells it (calibrated on the box, see the commit).
# On the box (yolo11s OpenVINO, 2026-10-11): the back yard's garden lamps and a wall fitting come out "person" up to
# 0.80 on the whole picture and real people from 0.31; on a full-resolution crop the lamps read 0.00-0.59 and the
# chairs of 23:54 0.00, while every person seen at 0.85+ read 0.88+. ZOOM_CONF 0.60 keeps every lamp out at the cost
# of a few dim or far people (0.37-0.50), which the owner prefers to a made-up one. On a small picture (no
# full-resolution copy: 704-960 wide) the second look still passes lamps at up to 0.77, so it needs ZOOM_CONF_SMALL.
PERSON_SURE = 0.85
ZOOM_CONF = 0.60
ZOOM_CONF_SMALL = 0.80
ZOOM_MIN_CONF = 0.25                # what the second look reads at all
ZOOM_PAD = 3.0                      # the crop: this many times the box's long side ...
ZOOM_MIN_SIDE = 320                 # ... and at least this many pixels
MAX_ZOOMS = 4                       # second looks per photo (~0.4 s each on the N150), the surest first
PERSON, VEHICLE, ANIMAL = "person", "vehicle", "animal"
_KIND_OF = {"person": PERSON, "car": VEHICLE, "truck": VEHICLE, "bus": VEHICLE, "motorcycle": VEHICLE,
            "bicycle": VEHICLE, "dog": ANIMAL, "cat": ANIMAL, "bird": ANIMAL}
STRUCTURE_ZONES = frozenset({"fence", "window"})   # things in the picture, not ground anyone stands on
MIN_SHARE = 2.0 / 3.0               # this share of an object's placed points decides its ground


@dataclass(frozen=True)
class Seen:
    label: str                      # the detector's word: person, car, truck, dog ...
    conf: float
    box: Tuple[float, float, float, float]      # 0..1 of the picture
    ground: str = ""                # mine | neighbour | public | border | "" (not mapped)
    area: str = ""                  # the owner's name of the area ("" when none)

    @property
    def kind(self) -> str:
        return _KIND_OF.get(self.label, "")

    def record(self) -> Dict[str, Any]:
        return {"label": self.label, "kind": self.kind, "conf": round(self.conf, 2),
                "box": [round(v, 4) for v in self.box], "ground": self.ground, "area": self.area}


@dataclass
class Facts:
    camera: str
    source: str = ""                # "detector" (this photo) | "live" (the live detector, fresh) | "" (none)
    objects: List[Seen] = field(default_factory=list)
    mapped: bool = False            # the camera has an informative scene map
    areas: Dict[str, List[str]] = field(default_factory=dict)    # ground -> area names
    doubtful: List[Seen] = field(default_factory=list)            # "people" the second look did not confirm: no fact

    @property
    def known(self) -> bool:
        return bool(self.source)

    def count(self, kind: str) -> int:
        return sum(1 for o in self.objects if o.kind == kind)

    def record(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"source": self.source or "none", "people": self.count(PERSON),
                               "vehicles": self.count(VEHICLE), "animals": self.count(ANIMAL),
                               "objects": [o.record() for o in self.objects], "map": "yes" if self.mapped else "none"}
        if self.doubtful:
            out["doubtful_people"] = [o.record() for o in self.doubtful]
        return out


# ---------------------------------------------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------------------------------------------
class Detector:
    """yolo11s on one photo, loaded once on first use in the assistant's own thread (the live detector's model is
    not shared: it is not safe across threads). The OpenVINO copy on the CPU when there is one, else PyTorch."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model: Any = None
        self._device: Optional[str] = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is not None or self._failed:
            return self._model
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            stem = os.path.splitext(self.model_path)[0]
            ov_dir = stem + "_openvino_model"
            if os.path.isfile(os.path.join(ov_dir, "metadata.yaml")):
                try:
                    model = YOLO(ov_dir, task="detect")
                    import numpy as np  # noqa: PLC0415

                    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device="intel:cpu", verbose=False)
                    self._model, self._device = model, "intel:cpu"
                    return model
                except Exception as exc:  # noqa: BLE001 - the PyTorch copy always works
                    log.info("Look detector: OpenVINO on the CPU not used (%s)", exc)
            self._model, self._device = YOLO(self.model_path), "cpu"
        except Exception as exc:  # noqa: BLE001 - no detector: the look says so
            log.warning("Look detector not loaded (%s); live looks go without detector facts", exc)
            self._failed = True
        return self._model

    def __call__(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """``[{"label", "conf", "box"}]`` of the shown kinds, or None when the detector cannot run."""
        with self._lock:
            model = self._load()
            if model is None:
                return None
            try:
                result = model.predict(image_path, conf=min(LOOK_CONF.values()), device=self._device,
                                       verbose=False)[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("Look detector failed on %s: %s", image_path, exc)
                return None
        from ..ai_status import objects_from_result  # noqa: PLC0415

        return objects_from_result(result)

    def zoom(self, image_path: str, box: Sequence[float]) -> Optional[float]:
        """The second look (2026-10-11): the detector on a crop of *image_path* around *box* (0..1), so a small
        far-off "person" is seen at the detector's full input size. The confidence of the most sure person whose box
        holds the candidate's centre (0.0 when none), or None when the detector cannot run."""
        with self._lock:
            model = self._load()
            if model is None:
                return None
            try:
                import cv2  # noqa: PLC0415

                img = cv2.imread(image_path)
                if img is None:
                    return None
                crop, (cx, cy) = zoom_crop(img, box)
                result = model.predict(crop, conf=ZOOM_MIN_CONF, device=self._device, verbose=False)[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("Look detector's second look failed on %s: %s", image_path, exc)
                return None
        best = 0.0
        names = getattr(result, "names", {}) or {}
        for b in getattr(result, "boxes", None) or []:
            if names.get(int(b.cls[0])) != "person":
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                best = max(best, float(b.conf[0]))
        return best


def zoom_crop(img: Any, box: Sequence[float]) -> Tuple[Any, Tuple[float, float]]:
    """A square-ish crop of *img* (pixels) around *box* (0..1): ZOOM_PAD times the box's long side, at least
    ZOOM_MIN_SIDE pixels, kept inside the picture. ``(crop, the box's centre in the crop)``."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(ZOOM_PAD * max(x2 - x1, y2 - y1), ZOOM_MIN_SIDE)
    sw, sh = int(min(side, w)), int(min(side, h))
    left = int(max(0, min(w - sw, cx - sw / 2)))
    top = int(max(0, min(h - sh, cy - sh / 2)))
    return img[top:top + sh, left:left + sw], (cx - left, cy - top)


def _from_live(camera: str, taken: float, status_path: str) -> Optional[List[Dict[str, Any]]]:
    """The live detector's look at *camera* (ai_status.json) when it was made within FRESH_SEC of *taken*."""
    try:
        with open(status_path, encoding="utf-8") as f:
            entry = (json.load(f).get("cameras") or {}).get(camera) or {}
        checked = float(entry.get("checked_ts") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if not checked or abs(taken - checked) > FRESH_SEC:
        return None
    if entry.get("ts") != entry.get("checked_ts"):         # the latest look found nothing
        return []
    return [o for o in entry.get("objects") or [] if isinstance(o, dict)]


def detect(camera: str, image_path: str, taken: float, detector: Optional[Callable[[str], Any]] = None,
           status_path: str = "", full_path: str = "",
           doubtful: Optional[List[Seen]] = None) -> Tuple[str, List[Seen]]:
    """``(source, objects)``: the detector on this photo (its full-resolution copy *full_path* when there is one),
    else the live detector's fresh look, else ``("", [])``. A person below PERSON_SURE stays only when the
    detector's second look (``detector.zoom``) confirms it - ZOOM_CONF on the full-resolution copy, ZOOM_CONF_SMALL
    without one; the others go to *doubtful* (not facts)."""
    raw: Optional[List[Dict[str, Any]]] = None
    source = ""
    full = bool(full_path) and os.path.isfile(full_path)
    path = full_path if full else image_path
    bar = ZOOM_CONF if full else ZOOM_CONF_SMALL
    if detector is not None:
        try:
            raw = detector(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Look detector failed: %s", exc)
            raw = None
        source = "detector" if raw is not None else ""
    if raw is None and status_path:
        raw = _from_live(camera, taken, status_path)
        source = "live" if raw is not None else ""
    zoom = getattr(detector, "zoom", None) if source == "detector" else None
    zooms = 0
    seen: List[Seen] = []
    for obj in sorted(raw or [], key=_conf_of, reverse=True):
        try:
            label = str(obj["label"])
            conf = float(obj["conf"])
            box = tuple(float(v) for v in obj["box"][:4])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        kind = _KIND_OF.get(label, "")
        if not (kind and len(box) == 4 and conf >= LOOK_CONF[kind]):
            continue
        item = Seen(label, conf, box)   # type: ignore[arg-type]
        if kind == PERSON and conf < PERSON_SURE:
            second = _second_look(zoom if zooms < MAX_ZOOMS else None, path, box)
            zooms += 1
            log.info("Look %s: person %.2f -> second look %s", camera, conf,
                     "none" if second is None else f"{second:.2f}")
            if second is None or second < bar:
                if doubtful is not None:
                    doubtful.append(item)
                continue
        seen.append(item)
    return source, seen


def _conf_of(obj: Any) -> float:
    try:
        return float(obj["conf"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _second_look(zoom: Any, path: str, box: Sequence[float]) -> Optional[float]:
    if not callable(zoom):
        return None
    try:
        got = zoom(path, box)
        return None if got is None else float(got)
    except Exception as exc:  # noqa: BLE001 - no second look: not confirmed
        log.warning("Look detector's second look failed: %s", exc)
        return None


# ---------------------------------------------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------------------------------------------
_CHANNEL = re.compile(r"_ch(\d+)$")


def load_map(camera: str, zones_path: Optional[str] = None) -> Optional[sm.SceneMap]:
    """The camera's informative scene map, else the one drawn under an older name of its channel; None without."""
    try:
        scene = sm.load_scene_map(camera, zones_path)
        if scene.informative:
            return scene
        match = _CHANNEL.search(str(camera))
        if not match:
            return None
        from ...data_collection.zones import read_scene_maps  # noqa: PLC0415

        _zp, sp = sm._paths(zones_path)
        for key in sorted(read_scene_maps(sp)):
            if key != camera and _CHANNEL.search(str(key)) and _CHANNEL.search(str(key)).group(1) == match.group(1):
                other = sm.load_scene_map(str(key), zones_path)
                if other.informative:
                    log.info("Scene map of %s read from %s (same channel)", camera, key)
                    return other
    except Exception as exc:  # noqa: BLE001 - a map is never worth a failed look
        log.warning("Scene map of %s not read: %s", camera, exc)
    return None


def _ground_scene(scene: sm.SceneMap) -> sm.SceneMap:
    return replace(scene, areas=tuple(a for a in scene.areas if a.zone not in STRUCTURE_ZONES))


def _samples(box: Sequence[float], kind: str) -> List[sm.Point]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w, h = x2 - x1, y2 - y1
    if kind == VEHICLE:      # where its wheels stand: the lower part of the box, not its bottom corner
        return [(x1 + fx * w, y1 + fy * h) for fy in (0.6, 0.7, 0.8, 0.9, 1.0) for fx in (0.2, 0.35, 0.5, 0.65, 0.8)]
    return [(x1 + fx * w, y2) for fx in (0.3, 0.5, 0.7)]      # the feet


def _at(scene: sm.SceneMap, ground_only: sm.SceneMap, p: sm.Point) -> Tuple[Optional[sm.Area], bool]:
    """The area under *p* (by ``area_at``) and whether it is a structure area (weak)."""
    p = (min(1.0, max(0.0, p[0])), min(1.0, max(0.0, p[1])))
    area = ground_only.area_at(p)
    if area is not None:
        return area, False
    area = scene.area_at(p)
    return area, area is not None


def place(scene: Optional[sm.SceneMap], seen: Seen) -> Seen:
    """*seen* with its ground and area on *scene* (unchanged without a map)."""
    if scene is None or not seen.kind:
        return seen
    ground_only = _ground_scene(scene)
    beyond = getattr(sm, "boundary_ground", None)      # the side of the boundary lines (scene_map, when it has it)
    strong: List[Tuple[str, str]] = []
    weak: List[Tuple[str, str]] = []
    for p in _samples(seen.box, seen.kind):
        area, is_weak = _at(scene, ground_only, p)
        if area is not None and area.ground:
            (weak if is_weak else strong).append((area.ground, area.name))
        elif area is None and callable(beyond):
            try:
                g = str(beyond(scene, p) or "")
            except Exception:  # noqa: BLE001
                g = ""
            if g:
                strong.append((g, ""))
    votes = strong or weak
    if not votes:
        return seen
    grounds = [g for g, _ in votes]
    best = max(set(grounds), key=grounds.count)
    if grounds.count(best) < MIN_SHARE * len(grounds):
        return replace(seen, ground="border")
    names = [n for g, n in votes if g == best and n]
    return replace(seen, ground=best, area=max(set(names), key=names.count) if names else "")


def area_names(scene: Optional[sm.SceneMap]) -> Dict[str, List[str]]:
    """``{mine|neighbour|public|black: [area names]}``, each name once."""
    out: Dict[str, List[str]] = {}
    for a in scene.areas if scene is not None else ():
        key = a.ground or "black"
        if a.name not in out.setdefault(key, []):
            out[key].append(a.name)
    return out


def grounded_facts(camera: str, image_path: str, taken: float, detector: Optional[Callable[[str], Any]] = None,
                   status_path: str = "", zones_path: Optional[str] = None, full_path: str = "") -> Facts:
    """Everything code knows about this photo before the vision model looks. Never raises."""
    try:
        doubtful: List[Seen] = []
        source, seen = detect(camera, image_path, taken, detector, status_path, full_path, doubtful)
        scene = load_map(camera, zones_path)
        return Facts(camera, source, [place(scene, s) for s in seen], scene is not None, area_names(scene),
                     doubtful)
    except Exception as exc:  # noqa: BLE001 - without facts the look goes on as before
        log.warning("Grounded facts for %s failed: %s", camera, exc)
        return Facts(camera)


# ---------------------------------------------------------------------------------------------------------------
# What the vision model is told
# ---------------------------------------------------------------------------------------------------------------
_GROUND_EN = {sm.MINE: "the owner's own ground", sm.NEIGHBOUR: "the neighbour's ground", sm.PUBLIC: "public ground",
              "border": "on the boundary between grounds", "": "a spot the map does not cover"}
_WORD = {"truck": "truck/pickup"}


def _where_en(o: Seen, mapped: bool) -> str:
    if not mapped:
        return ""
    text = _GROUND_EN.get(o.ground, _GROUND_EN[""])
    if o.ground in (sm.MINE, sm.NEIGHBOUR, sm.PUBLIC):
        return f" on {text}" + (f' (area "{o.area}")' if o.area else "")
    return f" - {text}"


def facts_text(facts: Facts) -> str:
    """The facts block for the vision prompt; "" when code knows nothing."""
    if not facts.known and not facts.mapped:
        return ""
    lines = []
    if facts.known:
        lines.append("DETECTOR FACTS (code; the object detector on this exact picture):")
        for kind, plural in ((PERSON, "people"), (VEHICLE, "vehicles"), (ANIMAL, "animals")):
            items = [o for o in facts.objects if o.kind == kind]
            parts = [f"{_WORD.get(o.label, o.label)} ({o.conf:.2f}){_where_en(o, facts.mapped)}" for o in items]
            lines.append(f"- {plural}: {len(items)}" + (": " + "; ".join(parts) if parts else ""))
    if facts.mapped:
        names = facts.areas
        lines.append("THE OWNER'S SCENE MAP: ours = " + (", ".join(names.get(sm.MINE, [])) or "-")
                     + "; neighbour's = " + (", ".join(names.get(sm.NEIGHBOUR, [])) or "-")
                     + "; public = " + (", ".join(names.get(sm.PUBLIC, [])) or "-") + ".")
    lines.append(load("brain_vision_facts_rules.prompt"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# The post-check
# ---------------------------------------------------------------------------------------------------------------
_HE_PREFIX = r"(?<![א-ת])[ושהבלמכ]{0,2}"
_PERSON_WORDS = re.compile(
    r"\b(?:person|persons|people|man|men|woman|women|boy|girl|child|children|kid|kids|someone|somebody|figure|"
    r"pedestrian|worker|workers|individual|individuals|visitor|guy)\b|"
    + _HE_PREFIX + r"(?:אדם|אנשים|איש|אישה|נשים|גבר|גברים|ילד|ילדה|ילדים|מישהו|מישהי|דמות|דמויות|עובד|עובדים|פועל|"
                   r"פועלים|הולך רגל|הולכי רגל)(?![א-ת])", re.IGNORECASE)
_VEHICLE_WORDS = re.compile(
    r"\b(?:car|cars|truck|trucks|pickup|pick-up|van|vans|vehicle|vehicles|bus|motorcycle|motorbike|scooter|suv|"
    r"jeep|lorry)\b|"
    + _HE_PREFIX + r"(?:רכב|רכבים|מכונית|מכוניות|טנדר|טנדרים|משאית|משאיות|אוטובוס|אופנוע|קטנוע|ג'יפ|ג׳יפ|ואן)"
                   r"(?![א-ת])", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:no|not|none|nobody|without|empty|neither|nor|isn't|aren't|no one)\b|"
    + r"(?<![א-ת])(?:אין|לא|ללא|בלי|ריק|ריקה|אף)(?![א-ת])", re.IGNORECASE)
_HEDGE = re.compile(r"\b(?:may|might|possibly|perhaps|not certain|unclear)\b|ייתכן|יתכן|אולי|לא בטוח|לא ברור",
                    re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r",\s*|;\s*|\s+(?:while|and|whereas)\s+|\s+(?=בזמן ש|כאשר|ובזמן ש|וכן )", re.IGNORECASE)
_LEAD = re.compile(r"^(?:בזמן ש|ובזמן ש|כאשר |וכן |(?:while|and|whereas)\s+)", re.IGNORECASE)


def _asserts(clause: str, words: "re.Pattern[str]") -> bool:
    return bool(words.search(clause)) and not _NEGATION.search(clause) and not _HEDGE.search(clause)


def _strip(text: str, words: "re.Pattern[str]") -> Tuple[str, bool]:
    """*text* without the clauses that state one of *words*; and whether any was dropped."""
    kept_sentences, dropped = [], False
    for sentence in _SENTENCE.split(" ".join(str(text or "").split())):
        if not _asserts(sentence, words):
            kept_sentences.append(sentence)
            continue
        clauses = [c for c in _CLAUSE.split(sentence.rstrip(".!?")) if c and c.strip()]
        keep = [_LEAD.sub("", c.strip()) for c in clauses if not _asserts(c, words)]
        dropped = True
        if keep:
            kept = ", ".join(keep).strip()
            kept_sentences.append(kept[:1].upper() + kept[1:] + ".")     # "while a pickup ..." -> "A pickup ..."
    return " ".join(s for s in kept_sentences if s.strip(" .")), dropped


def _hebrew(text: str) -> bool:
    return bool(re.search(r"[א-ת]", str(text or "")))


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def enforce(look: Dict[str, Any], facts: Facts, name: str = "") -> Dict[str, Any]:
    """*look* (``Vision.look``'s answer) checked against the detector. With no person detected, a stated person is
    taken out of the description and replaced by the fact, with no "maybe" (the model's count goes to
    ``model_people``, for the log). A vehicle only the model saw stays as "ייתכן שיש רכב, לא בטוח". Adds
    ``unsure_people`` / ``grounded`` / ``corrected``."""
    if not isinstance(look, dict) or not look.get("ok") or not facts.known:
        return look
    out = dict(look)
    text = str(out.get("description") or "")
    he = _hebrew(text)
    # The camera's name only in its own script: "במצלמה 2", never "at מצלמה 2" inside an English answer.
    where = f" ב{name}" if (he and name) else (f" at {name}" if name and not _hebrew(name) else "")
    corrected: List[str] = []
    unsure_people = _int(out.get("unsure_people"))
    if facts.count(PERSON) == 0:
        stated = _int(out.get("people"))
        text, dropped = _strip(text, _PERSON_WORDS)
        if stated or dropped:
            # 2026-10-11 (the owner): no "maybe a person" either - a person the detector did not confirm is not
            # told at all. The model's count stays in the log only.
            unsure_people = max(unsure_people, stated, 1 if dropped else 0)
            fact = (f"לא רואה אנשים{where} עכשיו." if he else f"No people seen{where} now.")
            text = (fact + (" " + text if text else "")).strip()
            corrected.append("people")
        out["people"] = 0
    elif _int(out.get("people")) == 0:
        n = facts.count(PERSON)
        text = (text + (f" הגלאי מצא {n} אנשים בתמונה." if he else f" The detector found {n} person(s) here.")).strip()
        out["people"] = n
        corrected.append("people_missed")
    if facts.count(VEHICLE) == 0:
        text2, dropped = _strip(text, _VEHICLE_WORDS)
        if dropped:
            text = (text2 + (" ייתכן שיש רכב, לא בטוח." if he else " There may be a vehicle, not certain.")).strip()
            corrected.append("vehicles")
    out["description"] = text or ("לא רואה אנשים עכשיו." if he else "No people seen now.")
    out["unsure_people"] = 0 if facts.count(PERSON) == 0 else unsure_people
    if unsure_people and facts.count(PERSON) == 0:
        out["model_people"] = unsure_people          # for the log and the replay, never for the owner
    out["grounded"] = facts.record()
    if corrected:
        out["corrected"] = corrected
        log.warning("Grounded look %s: the vision answer contradicted the detector (%s); rewritten: %r -> %r",
                    facts.camera, ", ".join(corrected), look.get("description"), out["description"])
    return out


def ground_line(facts: Facts, lang: str = "he") -> str:
    """One owner-language line of whose ground the detected vehicles and people are on ("" when the map says
    nothing about them)."""
    if not facts.mapped:
        return ""
    parts = [whose_text(o, lang) for o in facts.objects if o.kind in (PERSON, VEHICLE) and o.ground]
    return " ".join(dict.fromkeys(p for p in parts if p))


# ---------------------------------------------------------------------------------------------------------------
# The map tools
# ---------------------------------------------------------------------------------------------------------------
_THING_HE = {"person": "האדם", "car": "הרכב", "truck": "הטנדר", "bus": "האוטובוס", "motorcycle": "האופנוע",
             "bicycle": "האופניים", "dog": "הכלב", "cat": "החתול", "bird": "הציפור"}
_GROUND_HE = {sm.MINE: "בשטח שלך", sm.NEIGHBOUR: "בשטח של השכן", sm.PUBLIC: "בשטח ציבורי / ברחוב"}


def whose_text(o: Seen, lang: str = "he") -> str:
    """"הטנדר בשטח של השכן (לפי המפה שציירת)." for one placed object; "" when the map does not place it."""
    he = str(lang).startswith("he")
    thing = _THING_HE.get(o.label, "הדבר") if he else f"the {o.label}"
    if o.ground == "border":
        return (f"{thing} על הגבול בין השטח שלך לשטח של השכן, לפי המפה שציירת." if he
                else f"{thing.capitalize()} is on the boundary between your ground and the neighbour's, by your map.")
    if o.ground in _GROUND_HE:
        return (f"{thing} {_GROUND_HE[o.ground]} (לפי המפה שציירת)." if he
                else f"{thing.capitalize()} is on {_GROUND_EN[o.ground]} (by the map you drew).")
    return ""


def no_map_text(lang: str = "he") -> str:
    return ("אין עדיין מפה למצלמה הזו, אז אני לא יודע של מי השטח. אפשר לסמן אותה באפליקציה (מפת הסצנה) ואז אדע."
            if str(lang).startswith("he") else
            "This camera has no map yet, so I cannot tell whose ground it is. You can draw it in the app (scene map).")


def map_summary(camera: str, zones_path: Optional[str] = None) -> Dict[str, Any]:
    """The camera's map in a few words for the model: areas by whose ground, the boundary lines, the role, when
    it was saved. ``{"map": "none"}`` without one."""
    scene = load_map(camera, zones_path)
    if scene is None:
        return {"map": "none"}
    names = area_names(scene)
    out: Dict[str, Any] = {"map": "yes", "ours": names.get(sm.MINE, []), "neighbour": names.get(sm.NEIGHBOUR, []),
                           "public": names.get(sm.PUBLIC, []), "blacked_out": names.get("black", []),
                           "boundary_lines": [ln.name for ln in scene.lines],
                           "outside_the_areas": {"unmapped": "not mapped", sm.WATCH: (scene.rest_owner or "public")
                                                 + "'s ground", sm.BLACK: "blacked out"}.get(scene.outside, ""),
                           "role": scene.camera_role()}
    if scene.confirmed:
        out["saved"] = dt.datetime.fromtimestamp(scene.confirmed).strftime("%d/%m %H:%M")
    if scene.camera != camera:
        out["drawn_as"] = "the same channel before the cameras were renamed"
    return out


_ASK_VEHICLE = re.compile(_VEHICLE_WORDS.pattern + r"|" + _HE_PREFIX + r"(?:אוטו|רכב)", re.IGNORECASE)
_ASK_PERSON = _PERSON_WORDS
_ASK_ANIMAL = re.compile(r"\b(?:dog|cat|animal|bird)s?\b|" + _HE_PREFIX + r"(?:כלב|חתול|חיה|ציפור)", re.IGNORECASE)
_ENTITY = re.compile(r"\b(?:P|CAR|C|V)(\d+)\b", re.IGNORECASE)


def wanted_kind(what: str) -> str:
    what = str(what or "")
    if _ASK_VEHICLE.search(what) or re.search(r"\bCAR\d+\b", what, re.IGNORECASE):
        return VEHICLE
    if _ASK_PERSON.search(what) or re.search(r"\bP\d+\b", what):
        return PERSON
    if _ASK_ANIMAL.search(what):
        return ANIMAL
    return ""


def where_is(objects: Sequence[Seen], what: str, scene: Optional[sm.SceneMap], lang: str = "he") -> Dict[str, Any]:
    """Which ground each object the owner asks about stands on, from *objects* (placed on *scene*)."""
    he = str(lang).startswith("he")
    if scene is None:
        return {"map": "none", "say": no_map_text(lang)}
    kind = wanted_kind(what)
    # An area named by the owner ("השער", "area 7"): whose it is.
    for a in scene.areas:
        if a.name and len(a.name) > 2 and a.name.casefold() in str(what or "").casefold():
            g = a.ground or "black"
            return {"map": "yes", "area": a.name, "whose": g,
                    "say": (f"{a.name}: " + {sm.MINE: "השטח שלך", sm.NEIGHBOUR: "השטח של השכן",
                                             sm.PUBLIC: "שטח ציבורי", "black": "מוסתר במפה"}[g] + " (לפי המפה שציירת)."
                            if he else f"{a.name}: {_GROUND_EN.get(g, 'blacked out')} (by your map).")}
    hits = [o for o in objects if (o.kind == kind if kind else o.kind in (PERSON, VEHICLE))]
    if not hits:
        thing = {VEHICLE: "רכב", PERSON: "אדם", ANIMAL: "חיה"}.get(kind, "משהו") if he else (kind or "anything")
        return {"map": "yes", "found": [],
                "say": (f"לא רואה {thing} במצלמה הזו עכשיו, אז אין מה לבדוק במפה." if he
                        else f"I do not see {thing} on this camera now.")}
    found = [o.record() for o in hits]
    lines = [whose_text(o, lang) for o in hits]
    if not any(lines):
        lines = ["המקום שבו הוא עומד לא מסומן במפה שציירת, אז אני לא יודע של מי השטח שם." if he
                 else "The spot where it stands is not on the map you drew, so I cannot tell whose ground it is."]
    return {"map": "yes", "found": found, "say": " ".join(dict.fromkeys(x for x in lines if x)),
            "note": load("brain_tool_map_ground.prompt")}
