"""Label Studio videorectangle -> YOLO format conversion with keyframe interpolation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


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


def interpolate_keyframes(
    sequence: List[Dict[str, Any]],
    frames_count: int,
) -> Dict[int, Dict[str, float]]:
    """
    Interpolate LS videorectangle keyframes to produce a box for every frame.

    Returns ``{frame_number: {"x": ..., "y": ..., "width": ..., "height": ...}}``
    in LS percentage coords.  Frames where `enabled=False` are omitted.
    """
    if not sequence:
        return {}

    sorted_kfs = sorted(sequence, key=lambda kf: kf.get("frame", 0))

    result: Dict[int, Dict[str, float]] = {}

    for i, kf in enumerate(sorted_kfs):
        if not kf.get("enabled", True):
            continue

        frame = int(kf["frame"])

        if i + 1 < len(sorted_kfs):
            nxt = sorted_kfs[i + 1]
            end_frame = int(nxt["frame"])
        else:
            end_frame = frames_count

        for f in range(frame, end_frame + 1):
            if f > frames_count:
                break

            if frame == end_frame:
                t = 0.0
            else:
                t = (f - frame) / (end_frame - frame)

            if i + 1 < len(sorted_kfs):
                nxt = sorted_kfs[i + 1]
                if not nxt.get("enabled", True):
                    if f != frame:
                        continue
                    t = 0.0
                    box = {
                        "x": kf["x"],
                        "y": kf["y"],
                        "width": kf["width"],
                        "height": kf["height"],
                    }
                else:
                    box = {
                        "x": _lerp(kf["x"], nxt["x"], t),
                        "y": _lerp(kf["y"], nxt["y"], t),
                        "width": _lerp(kf["width"], nxt["width"], t),
                        "height": _lerp(kf["height"], nxt["height"], t),
                    }
            else:
                box = {
                    "x": kf["x"],
                    "y": kf["y"],
                    "width": kf["width"],
                    "height": kf["height"],
                }

            result[f] = box

    return result
