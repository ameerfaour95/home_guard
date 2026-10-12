"""Is the main stream's picture damaged? Each main-stream frame the Eye would see is checked against the sub stream at
the same moment (the same camera, the same view, a small stream that arrives whole).

2026-10-12: the 22:02 🔴 "a person in a hoodie covers the camera lens" was a woman at the door with a bag. Every one
of the 17 main-stream frames of her crop was broken (ghost copies of her, then a frozen grey field), while the sub
stream of the same seconds was clean. A sample of the day's alerts showed the same in about a third of them (purple
frames, ghost cars, smeared legs): the high-bitrate main stream loses pieces on the box's Wi-Fi link
("error while decoding MB ..., bytestream -8"; "hardware accelerator failed to decode picture"), the sub stream does
not. A damaged alert is described from the whole sub-stream frames instead: a little less detail, never a ghost.

The comparison: the crop's region of the main frame and of the sub frames nearest in time, both shrunk to 64x64 grey,
blurred, brightness-normalised; the share of pixels that differ by more than 0.8 standard deviations, at the best
sub frame. A whole frame scores 0.000-0.02 (a little more where someone moves fast); a damaged one 0.03-0.29.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("box.frame_check")

SIZE = 64
CELL = 0.8              # a pixel differs when it is this many standard deviations off
FRAME_BAD = 0.05        # one frame this far from the sub stream is damaged
FRAME_ODD = 0.025       # ... and two frames this far are too
FLAT = 0.35             # a main region with under this share of the sub's contrast is a lost (flat) picture
MIN_CONTRAST = 6.0      # ... when the sub region has some contrast (grey levels, std) to lose
NEAR_SEC = 0.6          # the sub frames compared: those this close in time (else the nearest one)


def _small(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or image.size == 0 or min(image.shape[:2]) < 2:
        return None
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g


def _times(n: int, start: float, end: float) -> List[float]:
    if n <= 1:
        return [float(start)] * n
    step = (float(end) - float(start)) / (n - 1)
    return [float(start) + i * step for i in range(n)]


def _region(crop: Any, main_size: Tuple[int, int], sub_size: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """The crop on the main frame and the same place on the sub frame (pixels, clipped, at least 4 wide)."""
    mw, mh = main_size
    sw, sh = sub_size
    if crop is None:
        x1, y1, x2, y2 = 0, 0, mw, mh
    else:
        x1, y1, x2, y2 = (int(v) for v in crop)
    x1, x2 = max(0, min(x1, mw - 1)), max(1, min(x2, mw))
    y1, y2 = max(0, min(y1, mh - 1)), max(1, min(y2, mh))
    s = (x1 * sw // mw, y1 * sh // mh, max(x1 * sw // mw + 4, x2 * sw // mw), max(y1 * sh // mh + 4, y2 * sh // mh))
    return (x1, y1, x2, y2), (s[0], s[1], min(s[2], sw), min(s[3], sh))


def scores(main_frames: Sequence[np.ndarray], main_start: float, main_end: float, crops: Sequence[Any],
           sub_frames: Sequence[np.ndarray], sub_start: float, sub_end: float) -> List[float]:
    """For each main frame: how far its crop is from the sub stream at the same moment (0: the same picture)."""
    if not main_frames or not sub_frames:
        return []
    mh, mw = main_frames[0].shape[:2]
    sh, sw = sub_frames[0].shape[:2]
    main_t = _times(len(main_frames), main_start, main_end)
    sub_t = _times(len(sub_frames), sub_start, sub_end)
    cache: Dict[Tuple[int, Tuple[int, int, int, int]], Optional[Tuple[np.ndarray, np.ndarray, float]]] = {}
    out: List[float] = []
    fallback = next((c for c in crops if c is not None), None) if crops else None
    for i, frame in enumerate(main_frames):
        crop = crops[i] if crops and i < len(crops) and crops[i] is not None else fallback
        (x1, y1, x2, y2), sub_box = _region(crop, (mw, mh), (sw, sh))
        raw = _small(frame[y1:y2, x1:x2])
        if raw is None:
            continue
        a = _norm(raw)
        near = [j for j, t in enumerate(sub_t) if abs(t - main_t[i]) <= NEAR_SEC] or \
            [min(range(len(sub_t)), key=lambda j: abs(sub_t[j] - main_t[i]))]
        lo = hi = None
        contrast = float("inf")
        for j in near:
            key = (j, sub_box)
            if key not in cache:
                sx1, sy1, sx2, sy2 = sub_box
                cache[key] = _envelope(_small(sub_frames[j][sy1:sy2, sx1:sx2]))
            got = cache[key]
            if got is not None:
                lo = got[0] if lo is None else np.minimum(lo, got[0])
                hi = got[1] if hi is None else np.maximum(hi, got[1])
                contrast = min(contrast, got[2])     # the flattest the sub shows around then
        if lo is None:
            continue
        if contrast >= MIN_CONTRAST and float(raw.std()) < FLAT * contrast:
            out.append(1.0)                       # a flat field where the sub stream shows a scene: the stream was lost
            continue
        # a pixel is wrong only outside what the sub stream shows around it, in space (1 pixel) and in time (the
        # sub frames around the moment): someone walking stays inside, a ghost or a grey field does not
        off = np.maximum(np.maximum(lo - a, a - hi), 0.0)
        out.append(float(np.mean(off > CELL)))
    return out


def _norm(g: np.ndarray) -> np.ndarray:
    return (g - g.mean()) / (g.std() + 8.0)


def _envelope(raw: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    if raw is None:
        return None
    b = _norm(raw)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(b, kernel), cv2.dilate(b, kernel), float(raw.std())


def damaged(values: Sequence[float]) -> bool:
    return bool(values) and (max(values) >= FRAME_BAD or sum(v >= FRAME_ODD for v in values) >= 2)
