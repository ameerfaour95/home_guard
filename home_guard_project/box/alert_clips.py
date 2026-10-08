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
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

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


def quiet_stem(camera: str, ts: float) -> str:
    """Quiet events are saved outside the alert hours and never sent."""
    return f"{camera}_{int(ts)}_quiet"


def trim_quiet(roots: Sequence[str], max_bytes: int) -> int:
    """Delete oldest quiet videos and their metas across roots; never count failed deletions."""
    clips = []
    seen = set()
    for root in roots:
        for dirpath, _, names in os.walk(os.path.join(root, "clips")):
            for name in names:
                if not name.endswith("_quiet.mp4"):
                    continue
                path = os.path.join(dirpath, name)
                key = os.path.normcase(os.path.abspath(path))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    stat = os.stat(path)
                    clips.append((stat.st_mtime, stat.st_size, root, path))
                except OSError:
                    continue
    total, deleted = sum(size for _, size, _, _ in clips), 0
    for _, size, root, path in sorted(clips):
        if total <= max_bytes:
            break
        rel = os.path.relpath(path, os.path.join(root, "clips"))
        meta = os.path.join(root, "meta", rel[:-len(".mp4")] + ".meta.json")
        try:
            try:
                os.remove(meta)   # no meta means it is no longer a complete clip for the uploader
            except FileNotFoundError:
                pass
            try:
                os.remove(path)
            except FileNotFoundError:
                # It disappeared after the scan; its bytes are already freed.
                total -= size
                continue
        except OSError as exc:
            log.warning("Quiet clip not deleted: %s: %s", path, exc)
            continue
        total -= size
        deleted += 1
    return deleted


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


def teacher_record(root_dir: str, camera: str, day: str, stem: str, teacher: Dict[str, Any]) -> Dict[str, Any]:
    """Save the VLM's inputs and answer next to the clip; return the meta fields that point at them.

    *teacher* holds ``model``, ``prompt_version``, ``prompt``, ``frames``
    (the JPEG bytes sent, in order), ``raw`` (the answer verbatim),
    ``parsed`` and optionally ``tracker`` (kept in the meta's ``teacher``). The pictures become ``vlm_crops/<camera>/<day>/<stem>_f<i>.jpg``
    and the raw answer ``responses/<camera>/<day>/<stem>.model_raw.txt``, the
    layout the collector already uses, so the uploader and the tagging tools
    take them as they are.
    """
    inputs = []
    for i, data in enumerate(teacher.get("frames") or []):
        rel = os.path.join("vlm_crops", camera, day, f"{stem}_f{i}.jpg")
        path = os.path.join(root_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        inputs.append(rel.replace("/", "\\"))
    raw_rel = None
    if teacher.get("raw"):
        raw_rel = os.path.join("responses", camera, day, f"{stem}.model_raw.txt")
        path = os.path.join(root_dir, raw_rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(teacher["raw"]))
        raw_rel = raw_rel.replace("/", "\\")
    out = {
        "model_response": teacher.get("parsed"),
        "teacher": {
            "model": teacher.get("model"),
            "prompt_version": teacher.get("prompt_version"),
            "prompt": teacher.get("prompt"),
            "input_frames": inputs,
            "raw_path": raw_rel,
            "temperature": 0,
        },
    }
    if teacher.get("tracker"):
        # What the tracker measured over the visit (tracker.TrackerFacts.record), whether or not the prompt had it.
        out["teacher"]["tracker"] = teacher["tracker"]
    return out


def _write_crop(root_dir: str, camera: str, day: str, stem: str,
                crop: Any, settings: Any, fps: float) -> Dict[str, Any]:
    """Save the same pre-encode crop frames and mp4v format as data collection."""
    import cv2
    from ..data_collection import vlm_crop

    rel = os.path.join("vlm_crops", camera, day, f"{stem}.mp4")
    path = os.path.join(root_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (crop.width, crop.height))
    try:
        if not writer.isOpened():
            raise RuntimeError("Failed to open VLM crop VideoWriter")
        for frame in crop.frames:
            writer.write(frame)
    finally:
        writer.release()
    return vlm_crop.crop_meta(crop, settings, rel, fps)


def write_alert_clip(
    root_dir: str,
    camera: str,
    stem: str,
    frames: List[Tuple[float, Any]],
    alert: Dict[str, Any],
    h264: bool = True,
    kind: str = "alert",
    teacher: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    fps: Optional[float] = None,
    crop: Any = None,
    crop_settings: Any = None,
    crop_fps: Optional[float] = None,
) -> Optional[str]:
    """Write the clip and then its meta under *root_dir*. Returns the meta path, or None with no frames.

    *extra* fields are merged into the meta (``trigger_ts``, ``mode``).
    Frames may be JPEG bytes (older callers) or the collector's decoded sub
    frames. Inference supplies their configured *fps* and the shared *crop*.

    The meta is written last and through a temp file: a meta on disk means
    the clip is complete. With a *teacher* record (what the VLM was asked and
    answered, see :func:`teacher_record`) the pictures it saw go to
    ``vlm_crops/`` and its raw answer to ``responses/``, named after the clip
    so they travel with it, and the meta carries ``model_response``.
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
    if fps is None:
        fps = max(1.0, round((len(frames) - 1) / duration, 1)) if len(frames) > 1 else 1.0
    writer = None
    written = 0
    for _, data in frames:
        image = data if isinstance(data, np.ndarray) else cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
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
    meta.update(extra or {})
    if crop is not None:
        try:
            meta["vlm_crop"] = _write_crop(root_dir, camera, day, stem, crop, crop_settings, crop_fps)
        except Exception as exc:  # the owner's whole-frame clip must survive a crop disk/codec failure
            log.warning("[%s] could not save VLM crop: %s", camera, exc)
            meta["vlm_crop_save_error"] = str(exc)
    if teacher:
        try:
            fields = teacher_record(root_dir, camera, day, stem, teacher)
            json.dumps(fields)
            meta.update(fields)
        except Exception as exc:  # noqa: BLE001 - the owner's clip must survive a teacher record that cannot be saved
            log.warning("[%s] could not save the teacher record: %s", camera, exc)
            meta["teacher_save_error"] = str(exc)
    tmp = f"{meta_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_path)
    return meta_path
