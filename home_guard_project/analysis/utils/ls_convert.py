"""Label Studio videorectangle -> YOLO format conversion with keyframe interpolation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def ls_rect_to_yolo(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Convert Label Studio percentage (0-100) top-left-xywh
    to YOLO normalised (0-1) centre-xywh.

    Inverse of labeling/utils/yolo.py:yolo_to_ls_rect.
    """
    xc = (x + w / 2.0) / 100.0
    yc = (y + h / 2.0) / 100.0
    wn = w / 100.0
    hn = h / 100.0
    return (
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, wn)),
        max(0.0, min(1.0, hn)),
    )


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


_TIME_EPS = 1e-6
_KEYS = ("x", "y", "width", "height")


def _box(kf: Dict[str, Any]) -> Dict[str, float]:
    return {k: kf[k] for k in _KEYS}


def _has_times(seq: List[Dict[str, Any]]) -> bool:
    return all(kf.get("time") is not None for kf in seq)


def box_at(
    sequence: List[Dict[str, Any]],
    frame: int,
    fps: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    Box visible at LS ``frame`` (1-based), or None.  Same semantics as the Admin
    Center's ``fleet_contract.tracks.box_at``:

    * linear interpolation by TIME when every keyframe carries ``time`` (and
      ``fps`` is known; the query time is ``(frame - 1) / fps``), else by frame;
    * a segment starting at an enabled keyframe moves to the next keyframe's
      position even when that next keyframe is disabled;
    * a disabled keyframe hides the box until the next keyframe;
    * hidden before the first keyframe; held after the last one if enabled.
    """
    if not sequence:
        return None
    by_time = bool(fps) and _has_times(sequence)
    if by_time:
        kfs = sorted(sequence, key=lambda kf: float(kf["time"]))
        q = (frame - 1) / float(fps)
        pos = lambda kf: float(kf["time"])
        eps = _TIME_EPS
    else:
        kfs = sorted(sequence, key=lambda kf: kf.get("frame", 0))
        q = float(frame)
        pos = lambda kf: float(kf.get("frame", 0))
        eps = 0.0
    if q < pos(kfs[0]) - eps:
        return None
    prev = kfs[0]
    for nxt in kfs[1:]:
        if q < pos(nxt) - eps:
            if not prev.get("enabled", True):
                return None
            span = pos(nxt) - pos(prev)
            w = (q - pos(prev)) / span if span > 0 else 0.0
            w = max(0.0, min(1.0, w))
            return {k: _lerp(prev[k], nxt[k], w) for k in _KEYS}
        prev = nxt
    return _box(prev) if prev.get("enabled", True) else None


def interpolate_keyframes(
    sequence: List[Dict[str, Any]],
    frames_count: int,
    fps: Optional[float] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Box for every visible LS frame ``1..frames_count`` as
    ``{frame_number: {"x","y","width","height"}}`` in LS percentage coords.
    Hidden frames are omitted.  See :func:`box_at` for the semantics.
    """
    result: Dict[int, Dict[str, float]] = {}
    for f in range(1, int(frames_count) + 1):
        b = box_at(sequence, f, fps)
        if b is not None:
            result[f] = b
    return result
