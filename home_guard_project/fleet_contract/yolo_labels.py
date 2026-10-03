"""The one strict validator of YOLO label files (`cls xc yc w h` per line, normalised). Stdlib only.

Used by every reader of a label file -- the admin detections view and the training exporter -- so a file that
one of them refuses is refused by the other too. A file is either valid (possibly empty: a real negative) or
malformed as a whole; a malformed file is never shown or trained on as "no objects".

A row must have exactly 5 fields: a non-negative integer class id (written as an integer, or a float with no
fractional part), and finite xc, yc in [0, 1] and w, h in (0, 1]. A rounding overshoot of up to 1e-6 past a
bound is accepted (the value is clipped). Blank lines are ignored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

EPS = 1e-6
INVALID_ROW = "invalid_row"


@dataclass(frozen=True)
class LabelRow:
    cls: int
    xc: float
    yc: float
    w: float
    h: float
    raw: tuple  # the four coordinate fields as written (the exporter keeps them verbatim when not clipped)


def _clip(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def parse_label(text: str) -> tuple[list[LabelRow], Optional[str]]:
    """(rows, None) for a valid file (rows may be empty), ([], "invalid_row") when any row is malformed."""
    rows: list[LabelRow] = []
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            return [], INVALID_ROW
        try:
            cls = float(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except (ValueError, OverflowError):
            return [], INVALID_ROW
        if not all(math.isfinite(v) for v in (cls, xc, yc, w, h)) or not cls.is_integer() or cls < 0:
            return [], INVALID_ROW
        if not (-EPS <= xc <= 1 + EPS and -EPS <= yc <= 1 + EPS and 0 < w <= 1 + EPS and 0 < h <= 1 + EPS):
            return [], INVALID_ROW
        rows.append(LabelRow(int(cls), _clip(xc), _clip(yc), _clip(w), _clip(h), tuple(parts[1:])))
    return rows, None
