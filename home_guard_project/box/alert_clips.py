"""Save the video of an alert, in the same layout the collector uses.

Inference mode keeps a few seconds of every camera in memory as JPEGs. When an
alert fires, the seconds before and after it are written as one clip under the
production folder, with a ``.meta.json`` that carries the alert. From there the
upload task and the owner's assistant treat it like any other clip.

The clip is written as H.264 when ffmpeg is on the box, because Telegram does
not play OpenCV's own mp4 codec.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

log = logging.getLogger("box.alert_clips")

CLIP_FPS = 5.0          # frames kept per second, per camera
PRE_SECONDS = 4.0       # how much before the alert goes into the clip
POST_SECONDS = 6.0      # and how much after it
JPEG_QUALITY = 80


class ClipRing:
    """The last seconds of one camera, as ``(time, jpeg bytes)``. Safe to read from another thread."""

    def __init__(self, seconds: float = PRE_SECONDS + POST_SECONDS + 5.0, fps: float = CLIP_FPS) -> None:
        self.fps = fps
        self._frames: Deque[Tuple[float, bytes]] = deque(maxlen=max(1, int(seconds * fps)))
        self._lock = threading.Lock()
        self._last = 0.0

    def wants(self, now: float) -> bool:
        """True when enough time has passed for the next frame."""
        return now - self._last >= 1.0 / self.fps

    def add(self, now: float, jpeg: bytes) -> None:
        if jpeg:
            with self._lock:
                self._frames.append((now, jpeg))
                self._last = now

    def between(self, start: float, end: float) -> List[Tuple[float, bytes]]:
        with self._lock:
            return [(ts, data) for ts, data in self._frames if start <= ts <= end]


def encode_frame(frame_bgr: Any) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def alert_stem(camera: str, ts: float) -> str:
    """The file stem of an alert's clip, and the id the owner's answers are filed under."""
    return f"{camera}_{int(ts)}_alert"


def false_positive_stem(camera: str, ts: float) -> str:
    """The file stem of a clip the VLM dismissed: the detector fired, nobody and nothing moving was there."""
    return f"{camera}_{int(ts)}_fp"


def _to_h264(path: str) -> bool:
    """Re-encode *path* in place with the pipeline's own ffmpeg helper. False if ffmpeg is missing or fails."""
    try:
        from home_guard_project.labeling.utils.ffmpeg import _reencode_one, derive_ffprobe, detect_ffmpeg  # noqa: PLC0415

        ffmpeg = detect_ffmpeg()
        return bool(ffmpeg) and _reencode_one(path, ffmpeg, derive_ffprobe(ffmpeg), False) == "ok"
    except Exception as exc:  # noqa: BLE001 - the clip is still usable as it is
        log.warning("Could not re-encode %s: %s", path, exc)
        return False


def clip_file(root_dir: str, meta_path: str) -> str:
    """The video that a meta file under *root_dir* describes, as a path on this machine."""
    with open(meta_path, encoding="utf-8") as f:
        relative = str(json.load(f)["clip_path"])
    return os.path.join(root_dir, *relative.replace("\\", "/").split("/"))


def write_alert_clip(
    root_dir: str,
    camera: str,
    stem: str,
    frames: List[Tuple[float, bytes]],
    alert: Dict[str, Any],
    h264: bool = True,
    kind: str = "alert",
) -> Optional[str]:
    """Write the clip and then its meta under *root_dir*. Returns the meta path, or None with no frames.

    The meta is written last and through a temp file: a meta on disk means
    the clip is complete.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if not frames:
        log.warning("[%s] no frames to save for %s", camera, stem)
        return None
    start_ts, end_ts = frames[0][0], frames[-1][0]
    day = dt.datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
    clip_rel = os.path.join("clips", camera, day, f"{stem}.mp4")
    clip_path = os.path.join(root_dir, clip_rel)
    meta_path = os.path.join(root_dir, "meta", camera, day, f"{stem}.meta.json")
    os.makedirs(os.path.dirname(clip_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    duration = max(end_ts - start_ts, 1e-3)
    fps = max(1.0, round((len(frames) - 1) / duration, 1)) if len(frames) > 1 else 1.0
    writer = None
    written = 0
    for _, data in frames:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                log.warning("[%s] could not open a video writer for %s", camera, clip_path)
                return None
        elif image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        writer.write(image)
        written += 1
    if writer is None:
        return None
    writer.release()
    codec = "h264" if h264 and _to_h264(clip_path) else "mp4v"

    local = lambda ts: dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    meta = {
        "camera_name": camera,
        "kind": kind,
        "clip_path": clip_rel.replace("/", "\\"),
        "clip_start_ts": start_ts,
        "clip_end_ts": end_ts,
        "clip_start_local": local(start_ts),
        "clip_end_local": local(end_ts),
        "duration_sec": end_ts - start_ts,
        "frames_written": written,
        "fps_estimated": fps,
        "codec": codec,
        "buffer": {"store_fps": fps, "store_size": [width, height]},
        # What the detector fired on, in the shape the tagging tools read.
        "yolo": {"class_counts": {label: 1 for label in alert.get("labels", [])},
                 "trigger_classes": list(alert.get("labels", [])), "trigger_detected": True},
        "alert": alert,
    }
    tmp = f"{meta_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_path)
    return meta_path
