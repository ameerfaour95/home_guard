"""Shared, masked RTSP readers for collection and inference."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __package__:
    from .config import Config
    from .zones import ZoneMask
else:
    from config import Config
    from zones import ZoneMask

log = logging.getLogger(__name__)


@dataclass
class _MaskFailState:
    """Per-reader memory for :func:`_masked_or_none`: who is reading, and whether it already warned."""

    label: str
    warned: bool = False


def _masked_or_none(mask: ZoneMask, frame: np.ndarray, state: _MaskFailState) -> Optional[np.ndarray]:
    """The frame with the watch zone applied, or None when masking fails.

    A failed mask drops the frame (the caller skips it): the unmasked picture
    is never stored or published, and the reader thread keeps running. The
    first failure per reader is logged; repeats are silent.
    """
    try:
        return mask.apply(frame)
    except Exception as exc:  # noqa: BLE001 - any mask failure must cost one frame, not the reader thread
        if not state.warned:
            state.warned = True
            log.warning("%s: watch zone could not be applied (%s); frame dropped", state.label, exc)
        return None


class SubStreamThread:
    """
    Read sub-stream RTSP continuously, keep JPEG-compressed frames at
    STORE_FPS in a rolling buffer.  ``latest_frame`` is kept as raw numpy
    for YOLO detection; the buffer stores JPEG bytes to save ~10-20x RAM
    compared to raw ndarrays.

    If ``cfg.STORE_SIZE`` is *None* the native camera resolution is kept;
    otherwise frames are resized before buffering.
    """

    _BUF_JPEG_QUALITY = 92

    def __init__(self, cfg: Config, src: str, mask: Optional[ZoneMask] = None, name: str = ""):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", cfg.OPENCV_FFMPEG_CAPTURE_OPTIONS)
        self.cfg = cfg
        self.src = src
        self.mask = mask or ZoneMask(None)   # blacks out everything outside the camera's watch zone
        self._mask_state = _MaskFailState(f"{name} sub-stream".strip())   # name only: src holds the password
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.buf: deque[Tuple[float, bytes]] = deque()
        self.buf_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.last_frame_ts = 0.0
        self.reconnect_backoff = float(cfg.RECONNECT_BACKOFF_START)
        self.store_interval = 1.0 / max(1e-6, float(cfg.STORE_FPS))
        self.last_store_ts = 0.0
        self.keep_seconds = float(cfg.CLIP_SECONDS) + float(cfg.POST_ROLL_SEC) + 2.0
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._BUF_JPEG_QUALITY]
        self._open_capture()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    # â”€â”€ internal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _open_capture(self) -> None:
        with self.lock:
            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.cfg.OPEN_TIMEOUT_MSEC))
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.cfg.READ_TIMEOUT_MSEC))
            except Exception:
                pass
            self.capture = cap
            self.last_frame_ts = time.time()

    def _reconnect(self) -> None:
        time.sleep(self.reconnect_backoff)
        self.reconnect_backoff = min(
            self.reconnect_backoff * 1.5,
            float(self.cfg.RECONNECT_BACKOFF_MAX),
        )
        self._open_capture()

    def _reader(self) -> None:
        store_size = self.cfg.STORE_SIZE
        while self.running:
            with self.lock:
                cap = self.capture
            if cap is None or not cap.isOpened():
                self._reconnect()
                continue

            ret, frame = cap.read()
            now = time.time()

            if not ret or frame is None:
                if now - self.last_frame_ts > float(self.cfg.FREEZE_RECONNECT_AFTER_SEC):
                    self._reconnect()
                else:
                    time.sleep(0.01)
                continue

            self.last_frame_ts = now
            self.reconnect_backoff = float(self.cfg.RECONNECT_BACKOFF_START)

            if (now - self.last_store_ts) < self.store_interval:
                continue
            self.last_store_ts = now
            # Before resize, latest_frame and the buffer: nothing sees the outside.
            frame = _masked_or_none(self.mask, frame, self._mask_state)
            if frame is None:
                continue

            if store_size is not None:
                frame = cv2.resize(frame, store_size, interpolation=cv2.INTER_AREA)

            self.latest_frame = frame

            ok, encoded = cv2.imencode(".jpg", frame, self._jpeg_params)
            if not ok:
                continue
            jpeg_bytes = encoded.tobytes()

            cutoff = now - self.keep_seconds
            with self.buf_lock:
                self.buf.append((now, jpeg_bytes))
                while self.buf and self.buf[0][0] < cutoff:
                    self.buf.popleft()

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.latest_frame is None:
            return False, None
        return True, self.latest_frame

    def get_clip_last_seconds(
        self, clip_seconds: float,
    ) -> Tuple[List[np.ndarray], float, float, float]:
        """Decompress JPEG buffer and return frames for the last N seconds."""
        now = time.time()
        cutoff = now - float(clip_seconds)
        with self.buf_lock:
            items = [(t, b) for (t, b) in self.buf if t >= cutoff]
        if len(items) < 2:
            return [], now, now, float(self.cfg.STORE_FPS)
        frames = []
        for _, jpeg_bytes in items:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        if len(frames) < 2:
            return [], now, now, float(self.cfg.STORE_FPS)
        return (
            frames,
            items[0][0],
            items[-1][0],
            float(self.cfg.STORE_FPS),
        )

    def is_opened(self) -> bool:
        with self.lock:
            return self.capture is not None and self.capture.isOpened()

    def release(self) -> None:
        self.running = False
        try:
            self.thread.join(timeout=2)
        except Exception:
            pass
        with self.lock:
            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Threaded RTSP capture â€” main-stream (JPEG-compressed buffer for VLM crops)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MainStreamThread:
    """
    Read main-stream RTSP continuously, store JPEG-compressed frames in a
    rolling buffer.  Decompression only happens at save time, keeping memory
    usage ~30-60x lower than raw numpy buffers for high-res streams.
    """

    def __init__(self, cfg: Config, src: str, mask: Optional[ZoneMask] = None, name: str = ""):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", cfg.OPENCV_FFMPEG_CAPTURE_OPTIONS)
        self.cfg = cfg
        self.src = src
        self.mask = mask or ZoneMask(None)   # blacks out everything outside the camera's watch zone
        self._mask_state = _MaskFailState(f"{name} main-stream".strip())  # name only: src holds the password
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.buf: deque[Tuple[float, bytes]] = deque()
        self.buf_lock = threading.Lock()
        self.running = True
        self.last_frame_ts = 0.0
        self.reconnect_backoff = float(cfg.RECONNECT_BACKOFF_START)
        self.store_interval = 1.0 / max(1e-6, float(cfg.MAIN_STORE_FPS))
        self.last_store_ts = 0.0
        self.keep_seconds = float(cfg.CLIP_SECONDS) + float(cfg.POST_ROLL_SEC) + 2.0
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.MAIN_JPEG_QUALITY)]
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._open_capture()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    # â”€â”€ internal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _open_capture(self) -> None:
        with self.lock:
            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.cfg.OPEN_TIMEOUT_MSEC))
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.cfg.READ_TIMEOUT_MSEC))
            except Exception:
                pass
            self.capture = cap
            self.last_frame_ts = time.time()

    def _reconnect(self) -> None:
        time.sleep(self.reconnect_backoff)
        self.reconnect_backoff = min(
            self.reconnect_backoff * 1.5,
            float(self.cfg.RECONNECT_BACKOFF_MAX),
        )
        self._open_capture()

    def _reader(self) -> None:
        while self.running:
            with self.lock:
                cap = self.capture
            if cap is None or not cap.isOpened():
                self._reconnect()
                continue

            ret, frame = cap.read()
            now = time.time()

            if not ret or frame is None:
                if now - self.last_frame_ts > float(self.cfg.FREEZE_RECONNECT_AFTER_SEC):
                    self._reconnect()
                else:
                    time.sleep(0.01)
                continue

            self.last_frame_ts = now
            self.reconnect_backoff = float(self.cfg.RECONNECT_BACKOFF_START)

            if (now - self.last_store_ts) < self.store_interval:
                continue
            self.last_store_ts = now
            frame = _masked_or_none(self.mask, frame, self._mask_state)
            if frame is None:
                continue

            self._frame_shape = (frame.shape[0], frame.shape[1])
            ok, encoded = cv2.imencode(".jpg", frame, self._jpeg_params)
            if not ok:
                continue
            jpeg_bytes = encoded.tobytes()

            cutoff = now - self.keep_seconds
            with self.buf_lock:
                self.buf.append((now, jpeg_bytes))
                while self.buf and self.buf[0][0] < cutoff:
                    self.buf.popleft()

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def frame_shape(self) -> Optional[Tuple[int, int]]:
        """(height, width) of the native main-stream frames, or None."""
        return self._frame_shape

    def get_clip_frames(
        self, clip_seconds: float,
    ) -> Tuple[List[np.ndarray], float, float, float]:
        """Decompress JPEG buffer and return frames for the last N seconds."""
        now = time.time()
        cutoff = now - float(clip_seconds)
        with self.buf_lock:
            items = [(t, b) for (t, b) in self.buf if t >= cutoff]
        if len(items) < 2:
            return [], now, now, float(self.cfg.MAIN_STORE_FPS)
        frames = []
        for _, jpeg_bytes in items:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        if len(frames) < 2:
            return [], now, now, float(self.cfg.MAIN_STORE_FPS)
        return (
            frames,
            items[0][0],
            items[-1][0],
            float(self.cfg.MAIN_STORE_FPS),
        )

    def buffer_duration(self) -> float:
        """Return the time span (seconds) currently held in the buffer."""
        with self.buf_lock:
            if len(self.buf) < 2:
                return 0.0
            return self.buf[-1][0] - self.buf[0][0]

    def is_opened(self) -> bool:
        with self.lock:
            return self.capture is not None and self.capture.isOpened()

    def release(self) -> None:
        self.running = False
        try:
            self.thread.join(timeout=2)
        except Exception:
            pass
        with self.lock:
            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None

