# home_guard_project/box/brain/media.py
"""Pictures and video for the assistant: a live photo, a new live recording, frames
of a saved clip, and a part of a saved clip ("the 4 seconds before").

Every picture is masked to the camera's watch zone before it is kept, and the
recording fails closed: a frame that cannot be masked stops the recording.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import subprocess
import threading
import time
import uuid
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..live_view import _camera_url, grab_masked

log = logging.getLogger("box.brain.media")

_busy: set = set()
_busy_lock = threading.Lock()


def grab_photo(camera: str, out_dir: str, cameras_path: str, zones_path: Optional[str] = None,
               now: Callable[[], float] = time.time, grab: Any = None) -> dict:
    return grab_masked(camera, cameras_path, out_dir, now=now, grab=grab, zones_path=zones_path)


def bounds_text(start: float, end: float) -> str:
    try:
        fmt = lambda ts: dt.datetime.fromtimestamp(_finite(ts)).strftime("%H:%M:%S")  # noqa: E731
        return f"{fmt(start)}–{fmt(end)}"
    except (TypeError, ValueError, OverflowError, OSError) as exc:
        log.warning("Could not format clip bounds: %s", exc)
        return ""


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def clip_frames(clip_path: str, count: int = 4, open_video: Any = None) -> List[bytes]:
    """*count* evenly spaced frames of a saved clip as JPEG bytes (empty on any failure)."""
    try:
        import cv2  # noqa: PLC0415

        count = int(count)
        if count <= 0:
            return []
        cap = (open_video or cv2.VideoCapture)(clip_path)
        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        if not frames:
            return []
        step = max(1, len(frames) // max(1, count))
        out = []
        for frame in frames[::step][:count]:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                out.append(buf.tobytes())
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read frames from %s: %s", clip_path, exc)
        return []


def record_live(camera: str, seconds: float, out_dir: str, cameras_path: str, zones_path: Optional[str] = None,
                now: Callable[[], float] = time.time, open_capture: Any = None,
                clock: Callable[[], float] = time.monotonic, fps: float = 5.0, h264: bool = True) -> dict:
    """Record *seconds* of *camera* now, masked to its watch zone. Never raises."""
    acquired = False
    try:
        url = _camera_url(camera, cameras_path)
        if not url:
            return {"ok": False, "error": "camera_unknown"}
        seconds, fps = _finite(seconds), _finite(fps)
        if seconds <= 0 or fps <= 0:
            raise ValueError("seconds and fps must be positive")
        with _busy_lock:
            if camera in _busy:
                return {"ok": False, "error": "busy"}
            _busy.add(camera)
            acquired = True
        return _record(camera, url, seconds, out_dir, zones_path, now, open_capture, clock, fps, h264)
    except Exception as exc:  # noqa: BLE001
        log.warning("Live recording from %s failed: %s", camera, exc)
        return {"ok": False, "error": "error"}
    finally:
        if acquired:
            with _busy_lock:
                _busy.discard(camera)


def _record(camera: str, url: str, seconds: float, out_dir: str, zones_path: Optional[str],
            now: Callable[[], float], open_capture: Any, clock: Callable[[], float], fps: float,
            h264: bool) -> dict:
    import cv2  # noqa: PLC0415

    from ...data_collection.zones import ZoneMask  # noqa: PLC0415
    from ..live_view import strict_black, strict_zone  # noqa: PLC0415

    polygon, readable = strict_zone(camera, zones_path)
    black, black_readable = strict_black(camera, zones_path)
    if not (readable and black_readable):
        return {"ok": False, "error": "error"}       # a configured zone we cannot read: never record unmasked
    mask = ZoneMask(polygon, black) if (polygon or black) else None
    last_kept = -1.0
    os.makedirs(out_dir, exist_ok=True)
    cap = (open_capture or (lambda u: cv2.VideoCapture(u, cv2.CAP_FFMPEG)))(url)
    path = None
    writer = None
    written = 0
    finished = False
    try:
        start_wall = _finite(now())
        end_wall = _finite(start_wall + seconds)
        began = _finite(clock())
        path = os.path.join(out_dir, f"{camera}_{int(start_wall)}_{uuid.uuid4().hex[:8]}_live.mp4")
        while True:
            elapsed = _finite(clock()) - began
            if elapsed >= seconds:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            if last_kept >= 0 and elapsed - last_kept < 1.0 / fps:
                continue
            if mask is not None:
                frame = mask.apply(frame)        # raises on failure: the recording stops, nothing unmasked is kept
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError("could not open live recording writer")
            writer.write(frame)
            written += 1
            last_kept = elapsed
        finished = True
    finally:
        try:
            try:
                cap.release()
            finally:
                if writer is not None:
                    writer.release()
        finally:
            if path is not None and (not finished or written == 0):
                try:
                    os.remove(path)
                except OSError:
                    pass
    if written == 0:
        return {"ok": False, "error": "camera_offline"}
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise OSError("live recording was not written")
    if h264:
        from ..alert_clips import _to_h264  # noqa: PLC0415

        _to_h264(path)
    return {"ok": True, "path": path, "start": start_wall, "end": end_wall}


def _run(cmd: Sequence[str]) -> int:
    return subprocess.run(list(cmd), capture_output=True, timeout=60).returncode


def cut_segment(clip_path: str, clip_start_ts: float, clip_end_ts: float, start_ts: float, seconds: float,
                out_path: str, run: Optional[Callable[[Sequence[str]], int]] = None,
                ffmpeg: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Cut the part of ``[start_ts, start_ts + seconds]`` that lies inside the clip into *out_path* (H.264).

    "The 5 seconds before" never grows into footage after the trigger: the request is intersected with the
    clip, not shifted. Returns the real ``(start, end)``, or None if nothing overlaps or ffmpeg fails.
    """
    try:
        clip_start_ts, clip_end_ts, start_ts, seconds = map(
            _finite, (clip_start_ts, clip_end_ts, start_ts, seconds))
        begin = max(start_ts, clip_start_ts)
        end = min(_finite(start_ts + seconds), clip_end_ts)
    except (TypeError, ValueError, OverflowError) as exc:
        log.warning("Invalid clip segment bounds: %s", exc)
        return None
    if end - begin < 0.5:
        return None
    if ffmpeg is None:
        try:
            from home_guard_project.labeling.utils.ffmpeg import detect_ffmpeg  # noqa: PLC0415

            ffmpeg = detect_ffmpeg()
        except Exception:  # noqa: BLE001
            ffmpeg = None
    if not ffmpeg:
        return None
    offset = begin - clip_start_ts
    cmd = [ffmpeg, "-y", "-ss", f"{offset:.2f}", "-i", clip_path, "-t", f"{end - begin:.2f}",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", out_path]
    try:
        code = (run or _run)(cmd)
        if code != 0 or not os.path.isfile(out_path):
            _remove_quietly(out_path)
            return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Cutting %s failed: %s", clip_path, exc)
        _remove_quietly(out_path)
        return None
    return (begin, end)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
