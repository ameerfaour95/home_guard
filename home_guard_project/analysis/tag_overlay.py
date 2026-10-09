"""Set-of-Mark tags (P1, P2, CAR1) drawn on the frames the box would send: benchmark only.

The box's :func:`data_collection.model_input.render_model_input` still refuses an overlay (it raises
NotImplementedError), so the box's input stays byte-identical. This module takes the box's own
:class:`ModelInput` (the exact frames, their clip-frame indices and crop boxes) and draws on COPIES of those
frames, in each sent frame's own pixels: crop coordinates when the frame was cut from the clip.

Style (Set-of-Mark / NumPro / MarkIt; filled masks hurt): an outline-only box about 3 px thick, a solid label
chip at the box's top-left (font about 3.5% of the frame height, black or white text, whichever contrasts),
one fixed colour per id across all frames, and ``#<frame> · <mm:ss>`` in the bottom-left corner. No masks.

IDs: P1, P2... for people and CAR1, CAR2... for vehicles, in order of first appearance in the SENT frames
(ties: left to right). An entity never visible in a sent frame gets no id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..data_collection.model_input import ModelInput

Box = Tuple[float, float, float, float]          # x1, y1, x2, y2

PERSON_KINDS = ("person",)
VEHICLE_KINDS = ("car", "truck", "bus", "motorcycle", "bicycle", "vehicle")

# BGR, high contrast on both day and night (IR) footage; one per id in id order, people first.
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (0, 255, 0),      # green
    (0, 128, 255),    # orange
    (255, 128, 0),    # azure
    (0, 0, 255),      # red
    (255, 255, 255),  # white
    (180, 105, 255),  # pink
    (128, 0, 128),    # purple
)

BOX_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_HEIGHT_FRAC = 0.035        # label text height / frame height
MIN_FONT_PX = 10


@dataclass
class Entity:
    """One human-tracked person or vehicle. *box_at(i)* is its box in clip-frame pixels at clip frame i, or None."""
    key: str
    kind: str
    box_at: Callable[[int], Optional[Box]]


@dataclass
class Mark:
    id: str          # the entity's true id (P1, CAR1, ...)
    label: str       # the string drawn (the id, or another one in a swap test)
    box: Tuple[int, int, int, int]
    color: Tuple[int, int, int]


@dataclass
class Tagged:
    frames: List[np.ndarray]                    # the overlaid copies, one per sent frame
    marks: List[List[Mark]]                     # what was drawn on each sent frame
    ids: Dict[str, str] = field(default_factory=dict)            # entity key -> id
    boxes: Dict[str, List[Optional[Tuple[int, int, int, int]]]] = field(default_factory=dict)  # id -> box per frame


def to_frame_coords(box: Box, crop: Optional[Tuple[int, int, int, int]], size: Tuple[int, int]) -> Optional[
        Tuple[int, int, int, int]]:
    """*box* (clip pixels) in a sent frame's own pixels: shifted and scaled into *crop* when the frame was cut
    (resized to *size*, width x height), clipped to the frame; None when nothing of it is inside."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = size
    if crop is not None:
        cx1, cy1, cx2, cy2 = crop
        sx = w / max(1e-6, float(cx2 - cx1))
        sy = h / max(1e-6, float(cy2 - cy1))
        x1, x2 = (x1 - cx1) * sx, (x2 - cx1) * sx
        y1, y2 = (y1 - cy1) * sy, (y2 - cy1) * sy
    x1, x2 = max(0.0, x1), min(float(w - 1), x2)
    y1, y2 = max(0.0, y1), min(float(h - 1), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def assign_ids(per_frame: Sequence[Mapping[str, Tuple[str, Tuple[int, int, int, int]]]]) -> Dict[str, str]:
    """Entity key -> id. *per_frame*: for each sent frame, {key: (kind, box)} of the entities visible in it.
    People get P1.., vehicles CAR1.., by first sent frame they appear in, then left to right."""
    first: Dict[str, Tuple[int, int, str]] = {}
    for k, visible in enumerate(per_frame):
        for key, (kind, box) in visible.items():
            if key not in first:
                first[key] = (k, box[0], kind)
    ids: Dict[str, str] = {}
    counters = {"P": 0, "CAR": 0}
    for key in sorted(first, key=lambda s: (first[s][0], first[s][1], s)):
        kind = first[key][2]
        prefix = "P" if kind in PERSON_KINDS else "CAR" if kind in VEHICLE_KINDS else None
        if prefix is None:
            continue
        counters[prefix] += 1
        ids[key] = f"{prefix}{counters[prefix]}"
    return ids


def color_of(entity_id: str, ids_in_order: Sequence[str]) -> Tuple[int, int, int]:
    return PALETTE[list(ids_in_order).index(entity_id) % len(PALETTE)]


def _font(frame_h: int) -> Tuple[float, int, int]:
    """(scale, thickness, pixel height) for text about FONT_HEIGHT_FRAC of the frame height."""
    px = max(MIN_FONT_PX, int(round(frame_h * FONT_HEIGHT_FRAC)))
    (_, base_h), _ = cv2.getTextSize("P", FONT, 1.0, 1)
    scale = px / float(base_h)
    return scale, max(1, int(round(scale * 2))), px


def _text_color(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)


def _chip(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    """A solid chip with *text* whose bottom-left sits at (x, y) (kept inside the frame)."""
    h, w = img.shape[:2]
    scale, thick, px = _font(h)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    pad = max(2, px // 5)
    cw, ch = tw + 2 * pad, th + base + 2 * pad
    x = int(min(max(0, x), max(0, w - cw)))
    top = int(min(max(0, y - ch), max(0, h - ch)))
    cv2.rectangle(img, (x, top), (x + cw, top + ch), color, -1)
    cv2.putText(img, text, (x + pad, top + pad + th), FONT, scale, _text_color(color), thick, cv2.LINE_AA)


def _stamp(img: np.ndarray, number: int, seconds: Optional[float]) -> None:
    """``#<frame> · <mm:ss>`` in the bottom-left corner on a black chip (the dot is drawn: no font has it)."""
    h, w = img.shape[:2]
    scale, thick, px = _font(h)
    s = 0 if seconds is None else int(seconds)
    left, right = f"#{number}", f"{s // 60:02d}:{s % 60:02d}"
    (lw, th), base = cv2.getTextSize(left, FONT, scale, thick)
    (rw, _), _ = cv2.getTextSize(right, FONT, scale, thick)
    pad, gap = max(2, px // 5), max(6, px)
    cw, ch = lw + gap + rw + 2 * pad, th + base + 2 * pad
    x0, y0 = 0, h - ch
    cv2.rectangle(img, (x0, y0), (min(w - 1, x0 + cw), h - 1), (0, 0, 0), -1)
    ty = y0 + pad + th
    cv2.putText(img, left, (x0 + pad, ty), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.circle(img, (x0 + pad + lw + gap // 2, ty - th // 2), max(1, thick), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, right, (x0 + pad + lw + gap, ty), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def tag_model_input(sent: ModelInput, entities: Sequence[Entity],
                    relabel: Optional[Mapping[str, str]] = None, stamp: bool = True) -> Tagged:
    """Copies of *sent*'s frames with each entity's box, label chip and the frame stamp drawn on.

    *relabel* maps a true id to the string drawn instead (the swap test: {"P1": "P2", "P2": "P1"}); the colour
    stays with the entity, so only the text changes. Nothing is drawn but the stamp when no entity is visible."""
    per_frame: List[Dict[str, Tuple[str, Tuple[int, int, int, int]]]] = []
    for k, frame in enumerate(sent.frames):
        h, w = frame.shape[:2]
        crop = sent.crops[k] if k < len(sent.crops) else None
        visible: Dict[str, Tuple[str, Tuple[int, int, int, int]]] = {}
        for e in entities:
            raw = e.box_at(sent.frame_indices[k])
            if raw is None:
                continue
            fb = to_frame_coords(raw, crop, (w, h))
            if fb is not None:
                visible[e.key] = (e.kind, fb)
        per_frame.append(visible)
    ids = assign_ids(per_frame)
    order = sorted(ids.values(), key=lambda s: (0 if s.startswith("P") else 1, int(s.lstrip("PCAR"))))
    relabel = dict(relabel or {})
    out_frames: List[np.ndarray] = []
    marks: List[List[Mark]] = []
    boxes: Dict[str, List[Optional[Tuple[int, int, int, int]]]] = {i: [None] * len(sent.frames) for i in order}
    for k, frame in enumerate(sent.frames):
        img = frame.copy()
        drawn: List[Mark] = []
        for key, (kind, fb) in per_frame[k].items():
            if key not in ids:
                continue
            eid = ids[key]
            drawn.append(Mark(eid, relabel.get(eid, eid), fb, color_of(eid, order)))
            boxes[eid][k] = fb
        drawn.sort(key=lambda m: order.index(m.id))
        for m in drawn:
            cv2.rectangle(img, (m.box[0], m.box[1]), (m.box[2], m.box[3]), m.color, BOX_THICKNESS)
        for m in drawn:      # chips after every outline, so no outline crosses a chip
            _chip(img, m.label, m.box[0], m.box[1], m.color)
        if stamp:
            _stamp(img, k + 1, sent.times[k] if k < len(sent.times) else None)
        out_frames.append(img)
        marks.append(drawn)
    return Tagged(frames=out_frames, marks=marks, ids=ids, boxes=boxes)


def contact_sheet(frames: Sequence[np.ndarray], cols: int = 5, width: int = 1600) -> Optional[np.ndarray]:
    """The frames tiled into one picture (for a quick look); None without frames."""
    if not frames:
        return None
    cell_w = max(1, width // cols)
    h0, w0 = frames[0].shape[:2]
    cell_h = max(1, int(round(h0 * cell_w / float(w0))))
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cv2.resize(f, (cell_w, cell_h))
    return sheet
