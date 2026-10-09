"""Set-of-Mark tags (P1, P2, CAR1) drawn on copies of the frames the vision model was sent.

Moved here from analysis/tag_overlay.py (the stage-3 benchmark, which imports it back) so the box's alert describer
(describer.py) draws exactly what the benchmark measured: an outline-only box about 3 px thick, a solid label chip at
the box's top-left (font about 3.5% of the frame height, black or white text, whichever contrasts), one fixed colour
per id across all frames, and ``#<frame> · <mm:ss>`` in the bottom-left corner. No masks (filled masks hurt).

Never used on the alert call itself: the Eye's frames stay byte-identical (data_collection.model_input refuses an
overlay); the benchmark found drawn tags there cost real alerts.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

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


def draw(frame: np.ndarray, marks: Sequence[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int]]],
         number: Optional[int] = None, seconds: Optional[float] = None) -> np.ndarray:
    """A copy of *frame* with each ``(label, box, colour)`` outlined and chipped (chips after every outline, so no
    outline crosses a chip), and the ``#number · mm:ss`` stamp when *number* is given."""
    img = frame.copy()
    for _, box, color in marks:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, BOX_THICKNESS)
    for label, box, color in marks:
        _chip(img, label, box[0], box[1], color)
    if number is not None:
        _stamp(img, number, seconds)
    return img
