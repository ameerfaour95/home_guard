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

from ..box.marks import (  # noqa: F401 - moved to the box (describer.py); re-exported for the benchmark
    BOX_THICKNESS, FONT, FONT_HEIGHT_FRAC, MIN_FONT_PX, PALETTE, PERSON_KINDS, VEHICLE_KINDS, Box, _chip, _font,
    _stamp, _text_color, assign_ids, color_of, to_frame_coords,
)
from ..data_collection.model_input import ModelInput


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
