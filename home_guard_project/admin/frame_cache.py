"""Every frame of a clip, decoded once when it opens (a background thread) and scaled for the screen, so a frame
step in Tag · YOLO is a lookup instead of a decoder round trip (2026-10-09: 110-280 ms per step on a 2592x1520 clip,
the owner's "every command responds with a delay").

Memory is tight on the founder laptop: frames are kept as RGB (3 bytes a pixel), their long side at most MAX_SIDE,
and the whole clip within BUDGET_BYTES (the long side shrinks further for a long clip). 61 frames of a 2592x1520 clip
take 200 MB instead of 720 MB at full size. A zoomed-in view asks the decoder for the full-size frame on top
(label_view.refine), so detail is never lost where it matters.

Over the network the clip is downloaded once, into memory (fetch): the player plays those bytes and the decoder reads
them, instead of each streaming the presigned URL on its own (2026-10-09 on the owner's clips: 1.3 s from the media
grant to the first frame, mostly HTTPS range requests). Nothing is written to disk.

Frames are usable as they arrive: *started* gets the ClipFrames as soon as the clip is open, and the list grows while
the rest decode (a list append is atomic; the view only reads frames below len()).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlparse

MAX_SIDE = 1600
BUDGET_BYTES = 200 * 1024 * 1024
MAX_FRAMES = 2000
MAX_FETCH_BYTES = 64 * 1024 * 1024     # a bigger file streams from its URL as before


@dataclass
class ClipFrames:
    frames: List = field(default_factory=list)     # QImage per decoded frame, in order (frame index = position)
    size: Tuple[int, int] = (0, 0)                # the clip's own (width, height): boxes are in its coordinates
    scale: float = 1.0                            # cached frame size / clip size
    token: object = None                          # which opened clip these frames belong to
    done: bool = False                            # every frame is in

    def __len__(self):
        return len(self.frames)

    def nbytes(self) -> int:
        return sum(f.sizeInBytes() for f in self.frames)

    def get(self, frame: int):
        return self.frames[frame] if 0 <= frame < len(self.frames) else None


def source_of(url: str) -> str:
    """What cv2 opens: a local path for a file:// URL, else the URL itself (an https media grant)."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        return path[1:] if len(path) > 2 and path[0] == "/" and path[2] == ":" else path
    return url


def fetch(backend, url: str) -> Optional[bytes]:
    """The clip's bytes for an http(s) media URL (one GET), None for a local file or when it cannot be fetched (the
    caller then streams the URL as before)."""
    if urlparse(url).scheme not in ("http", "https") or not hasattr(backend, "media_bytes"):
        return None
    try:
        data = backend.media_bytes(url)
    except Exception:  # noqa: BLE001 - streaming the URL still works
        return None
    return data if data and len(data) <= MAX_FETCH_BYTES else None


def frame_scale(width: int, height: int, count: int, max_side: int = MAX_SIDE, budget: int = BUDGET_BYTES) -> float:
    """The factor every frame is scaled by: long side at most *max_side*, all *count* frames within *budget*."""
    if width <= 0 or height <= 0:
        return 1.0
    scale = min(1.0, max_side / float(max(width, height)))
    per_frame = budget / float(max(1, count))
    if width * height * 3 * scale * scale > per_frame:
        scale = (per_frame / (width * height * 3)) ** 0.5
    return max(0.05, scale)


def decode(url, cancelled=lambda: False, max_side: Optional[int] = None, budget: Optional[int] = None,
           token=None, started=lambda clip: None) -> Optional[ClipFrames]:
    """Every frame of the clip at *url* (or the clip's bytes) as screen-sized QImages; None when it cannot be opened."""
    max_side, budget = max_side or MAX_SIDE, budget or BUDGET_BYTES
    import cv2  # noqa: PLC0415
    from PySide6.QtGui import QImage  # noqa: PLC0415

    stream = io.BytesIO(url) if isinstance(url, (bytes, bytearray)) else None   # held until release: cv2 reads it
    if stream is not None:
        cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG, [])
    else:
        cap = cv2.VideoCapture(source_of(url))
    if not cap.isOpened():
        return None
    try:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 200
        scale = frame_scale(width, height, min(count, MAX_FRAMES), max_side, budget)
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        out, used = ClipFrames(size=(width, height), scale=scale, token=token), 0
        started(out)
        while len(out.frames) < MAX_FRAMES and not cancelled():
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            out.frames.append(QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy())
            used += w * h * 3
            if used > budget * 1.1:              # a clip longer than its header said: stop at the budget
                break
        out.done = True
        return out if out.frames else None
    finally:
        cap.release()
