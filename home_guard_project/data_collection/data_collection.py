"""
Multi-camera RTSP data-collection pipeline.

Captures frames from RTSP cameras, runs YOLO for person/car detection,
and saves high-quality clips + metadata when triggered (or at random
intervals).  Optionally exports YOLO weak-label training data.

Configuration lives in config.yaml + cameras.yaml (loaded by config.py).
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import random
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import COCO_NAMES, Config, load_config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Suppress FFmpeg stderr noise
# ─────────────────────────────────────────────────────────────────────────────

def _silence_ffmpeg_stderr() -> None:
    """Redirect C-level stderr (fd 2) to devnull, silencing FFmpeg warnings.

    Python's sys.stderr is re-pointed to a dup of the original fd so that
    logging and tracebacks still print normally.
    """
    real_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    sys.stderr = io.TextIOWrapper(
        io.FileIO(real_stderr_fd, closefd=False),
        encoding="utf-8",
        errors="replace",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs(cfg: Config) -> None:
    root = cfg.OUT_DIR
    for sub in ("clips", "meta", "responses"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    if cfg.EXPORT_YOLO_TRAINING_DATA:
        base = os.path.join(root, cfg.YOLO_EXPORT_SUBDIR)
        os.makedirs(os.path.join(base, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, "labels"), exist_ok=True)


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _local_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _jittered_interval(base: float, frac: float) -> float:
    lo, hi = 1.0 - frac, 1.0 + frac
    return base * (lo + (hi - lo) * random.random())


def _write_mp4_clip(frames: List[np.ndarray], fps: float) -> str:
    """Write *frames* to a temp MP4 (mp4v codec) and return the path."""
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter.")
    for f in frames:
        if f is None:
            continue
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()
    return path


def _summarize_yolo(results) -> Tuple[Dict[int, int], Dict[int, float]]:
    counts: Dict[int, int] = defaultdict(int)
    max_conf: Dict[int, float] = defaultdict(float)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {}, {}
    for b in boxes:
        cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
        conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
        counts[cls_id] += 1
        if conf > max_conf[cls_id]:
            max_conf[cls_id] = conf
    return dict(counts), dict(max_conf)


def _xyxy_to_yolo_norm(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int,
) -> Tuple[float, float, float, float]:
    """Convert pixel xyxy box to normalised YOLO centre-xywh."""
    x1 = max(0.0, min(float(x1), float(w - 1)))
    x2 = max(0.0, min(float(x2), float(w - 1)))
    y1 = max(0.0, min(float(y1), float(h - 1)))
    y2 = max(0.0, min(float(y2), float(h - 1)))
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    xc, yc = x1 + bw / 2.0, y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


# ─────────────────────────────────────────────────────────────────────────────
# ROI polygon helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_polygon_to_px(
    norm_pts: List[Tuple[float, float]], w: int, h: int,
) -> np.ndarray:
    """Convert normalised (0-1) polygon to pixel int32 contour for cv2."""
    return np.array(
        [[int(x * w), int(y * h)] for x, y in norm_pts], dtype=np.int32,
    )


def _any_center_in_polygon(
    boxes: Any, class_ids: List[int], polygon: np.ndarray,
) -> bool:
    """True if any bbox whose class is in *class_ids* has its centre inside *polygon*."""
    if boxes is None or len(boxes) == 0:
        return False
    id_set = set(class_ids)
    for b in boxes:
        cid = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
        if cid not in id_set:
            continue
        xyxy = b.xyxy[0].tolist()
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# YOLO weak-label export
# ─────────────────────────────────────────────────────────────────────────────

def _export_yolo_frames(
    cfg: Config,
    detector: YOLO,
    frames: List[np.ndarray],
    camera_name: str,
    day: str,
    clip_id: str,
) -> List[Dict[str, Any]]:
    """Export sampled frames + YOLO .txt labels for training."""
    if not cfg.EXPORT_YOLO_TRAINING_DATA or not frames:
        return []

    base_dir = os.path.join(cfg.OUT_DIR, cfg.YOLO_EXPORT_SUBDIR)
    img_dir = os.path.join(base_dir, "images", camera_name, day)
    lbl_dir = os.path.join(base_dir, "labels", camera_name, day)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    store_fps = float(cfg.STORE_FPS)
    step = max(1, int(round(store_fps / max(0.1, float(cfg.YOLO_EXPORT_FPS)))))
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.YOLO_EXPORT_JPEG_QUALITY)]

    exported: List[Dict[str, Any]] = []
    for i in range(0, len(frames), step):
        frame = frames[i]
        if frame is None:
            continue
        h, w = frame.shape[:2]
        stem = f"{clip_id}_f{i:04d}"
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

        if not cv2.imwrite(img_path, frame, jpeg_params):
            continue

        results = detector(frame, verbose=False, conf=float(cfg.YOLO_EXPORT_CONF), imgsz=cfg.YOLO_IMGSZ)
        lines: List[str] = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                xyxy = b.xyxy[0].tolist()
                xc, yc, bw, bh = _xyxy_to_yolo_norm(*xyxy, w=w, h=h)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        exported.append({
            "frame_index": int(i),
            "approx_time_offset_sec": float(i / store_fps),
            "image_path": os.path.relpath(img_path, start=cfg.OUT_DIR),
            "label_path": os.path.relpath(lbl_path, start=cfg.OUT_DIR),
            "num_boxes": len(lines),
        })
    return exported


# ─────────────────────────────────────────────────────────────────────────────
# Threaded RTSP capture (per camera)
# ─────────────────────────────────────────────────────────────────────────────

class VideoCaptureThread:
    """
    Read RTSP continuously, keep frames at STORE_FPS in a rolling buffer.

    If ``cfg.STORE_SIZE`` is *None* the native camera resolution is kept;
    otherwise frames are resized before buffering.
    """

    def __init__(self, cfg: Config, src: str):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", cfg.OPENCV_FFMPEG_CAPTURE_OPTIONS)
        self.cfg = cfg
        self.src = src
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.buf: deque[Tuple[float, np.ndarray]] = deque()
        self.buf_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.last_frame_ts = 0.0
        self.reconnect_backoff = float(cfg.RECONNECT_BACKOFF_START)
        self.store_interval = 1.0 / max(1e-6, float(cfg.STORE_FPS))
        self.last_store_ts = 0.0
        self.keep_seconds = float(cfg.CLIP_SECONDS) + float(cfg.POST_ROLL_SEC) + 2.0
        self._open_capture()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    # ── internal ──────────────────────────────────────────────────────────

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

            if store_size is not None:
                frame = cv2.resize(frame, store_size, interpolation=cv2.INTER_AREA)

            self.latest_frame = frame
            cutoff = now - self.keep_seconds
            with self.buf_lock:
                self.buf.append((now, frame))
                while self.buf and self.buf[0][0] < cutoff:
                    self.buf.popleft()

    # ── public API ────────────────────────────────────────────────────────

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.latest_frame is None:
            return False, None
        return True, self.latest_frame

    def get_clip_last_seconds(
        self, clip_seconds: float,
    ) -> Tuple[List[np.ndarray], float, float, float]:
        now = time.time()
        cutoff = now - float(clip_seconds)
        with self.buf_lock:
            items = [(t, f) for (t, f) in self.buf if t >= cutoff]
        if len(items) < 2:
            return [], now, now, float(self.cfg.STORE_FPS)
        return (
            [f for _, f in items],
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


# ─────────────────────────────────────────────────────────────────────────────
# VLM worker (optional, loaded lazily)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ClipJob:
    camera_name: str
    clip_path: str
    meta_path: str
    raw_path: str
    meta: Dict[str, Any]


class VLMWorker:
    def __init__(self, cfg: Config):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.VLM_MODEL_ID, use_fast=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.VLM_MODEL_ID,
            torch_dtype=cfg.DTYPE,
            low_cpu_mem_usage=True,
        ).to(cfg.DEVICE)
        self.q: queue.Queue[_ClipJob] = queue.Queue(maxsize=200)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        log.info("VLM loaded: %s on %s (%s)", cfg.VLM_MODEL_ID, cfg.DEVICE, cfg.DTYPE)

    def submit(self, job: _ClipJob) -> bool:
        try:
            self.q.put_nowait(job)
            return True
        except queue.Full:
            log.warning("VLM queue full; dropping job")
            return False

    def _loop(self) -> None:
        while self.running:
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run(job)
            except Exception:
                log.exception("VLM job failed")
            finally:
                self.q.task_done()

    @torch.no_grad()
    def _run(self, job: _ClipJob) -> None:
        prompt = self.cfg.VLM_PROMPT.format(camera_name=job.camera_name)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "path": job.clip_path, "target_fps": self.cfg.VLM_SAMPLE_FPS},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.cfg.DEVICE) for k, v in inputs.items()}
        if self.cfg.DEVICE == "cuda":
            for k, v in list(inputs.items()):
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    inputs[k] = v.to(self.cfg.DTYPE)

        gen = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        out_text = self.processor.batch_decode(
            gen[:, prompt_len:], skip_special_tokens=True,
        )[0].strip()

        with open(job.raw_path, "w", encoding="utf-8") as f:
            f.write(out_text)

        job.meta["prompt_camera_name"] = job.camera_name
        job.meta["prompt_used"] = prompt
        job.meta["model_raw_text_path"] = os.path.relpath(job.raw_path, start=self.cfg.OUT_DIR)
        job.meta["model_response"] = out_text
        job.meta["model_response_path"] = os.path.relpath(job.raw_path, start=self.cfg.OUT_DIR)

        with open(job.meta_path, "w", encoding="utf-8") as f:
            json.dump(job.meta, f, ensure_ascii=False, indent=2)

        log.info("VLM done: %s -> %s", job.camera_name, os.path.basename(job.clip_path))
        if self.cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()

    def stop(self) -> None:
        self.running = False
        try:
            self.thread.join(timeout=2)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera runtime state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraState:
    name: str
    rtsp: str
    cap: VideoCaptureThread
    detection_score: float
    last_score_ts: float
    last_trigger_time: float
    next_random_time: float
    frame_i: int
    last_yolo: Any
    trigger_detected: bool
    trigger_max_conf: float
    yolo_class_counts: Dict[int, int]
    yolo_class_max_conf: Dict[int, float]
    is_recording: bool
    trigger_ts: float
    roi_polygon: Optional[np.ndarray]
    roi_norm: Optional[List[Tuple[float, float]]]


# ─────────────────────────────────────────────────────────────────────────────
# Clip saving
# ─────────────────────────────────────────────────────────────────────────────

def _save_clip(
    cfg: Config,
    detector: YOLO,
    vlm: Optional[VLMWorker],
    st: CameraState,
    kind: str,
    frames: List[np.ndarray],
    start_ts: float,
    end_ts: float,
    fps: float,
) -> None:
    if not frames:
        return

    day = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
    clip_id = f"{st.name}_{int(end_ts)}_{kind}"

    clips_dir = os.path.join(cfg.OUT_DIR, "clips", st.name, day)
    meta_dir = os.path.join(cfg.OUT_DIR, "meta", st.name, day)
    resp_dir = os.path.join(cfg.OUT_DIR, "responses", st.name, day)
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    if cfg.SAVE_WITH_PLOTTED_BOXES:
        plotted = []
        for f in frames:
            if f is None:
                continue
            res = detector(f, verbose=False, conf=cfg.YOLO_TRIGGER_CONF, imgsz=cfg.YOLO_IMGSZ)
            plotted.append(res[0].plot())
        tmp = _write_mp4_clip(plotted, fps=fps)
    else:
        tmp = _write_mp4_clip(frames, fps=fps)
    final_mp4 = os.path.join(clips_dir, f"{clip_id}.mp4")
    shutil.move(tmp, final_mp4)

    meta_path = os.path.join(meta_dir, f"{clip_id}.meta.json")
    raw_path = os.path.join(resp_dir, f"{clip_id}.model_raw.txt")

    meta: Dict[str, Any] = {
        "camera_name": st.name,
        "kind": kind,
        "clip_path": os.path.relpath(final_mp4, start=cfg.OUT_DIR),

        "clip_start_ts": float(start_ts),
        "clip_end_ts": float(end_ts),
        "clip_start_utc": _utc_iso(start_ts),
        "clip_end_utc": _utc_iso(end_ts),
        "clip_start_local": _local_iso(start_ts),
        "clip_end_local": _local_iso(end_ts),

        "duration_sec": float(end_ts - start_ts),
        "frames_written": len(frames),
        "fps_estimated": float(fps),
        "vlm_sample_fps": int(cfg.VLM_SAMPLE_FPS),

        "buffer": {
            "store_fps": float(fps),
            "store_size": [int(frames[0].shape[1]), int(frames[0].shape[0])],
        },

        "yolo": {
            "class_counts": {
                COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid): cnt
                for cid, cnt in st.yolo_class_counts.items()
            },
            "class_max_conf": {
                COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid): conf
                for cid, conf in st.yolo_class_max_conf.items()
            },
            "trigger_classes": [
                COCO_NAMES[cid] for cid in cfg.TRIGGER_CLASS_IDS
                if cid < len(COCO_NAMES)
            ],
            "trigger_detected": bool(st.trigger_detected),
            "trigger_conf_max": float(st.trigger_max_conf),
            "detection_score": float(st.detection_score),
        },

        "roi": {
            "active": st.roi_polygon is not None,
            "polygon_normalized": (
                [[float(x), float(y)] for x, y in st.roi_norm]
                if st.roi_norm else None
            ),
        },
    }

    if cfg.EXPORT_YOLO_TRAINING_DATA:
        exported = _export_yolo_frames(
            cfg=cfg, detector=detector, frames=frames,
            camera_name=st.name, day=day, clip_id=clip_id,
        )
        meta["yolo_export"] = {
            "enabled": True,
            "export_fps": float(cfg.YOLO_EXPORT_FPS),
            "conf": float(cfg.YOLO_EXPORT_CONF),
            "images_root": os.path.relpath(
                os.path.join(cfg.OUT_DIR, cfg.YOLO_EXPORT_SUBDIR, "images"),
                start=cfg.OUT_DIR,
            ),
            "labels_root": os.path.relpath(
                os.path.join(cfg.OUT_DIR, cfg.YOLO_EXPORT_SUBDIR, "labels"),
                start=cfg.OUT_DIR,
            ),
            "exported_frames": exported,
        }

    if vlm is None:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log.info("[%s] %s saved: %s (no VLM)", st.name, kind, final_mp4)
        return

    job = _ClipJob(
        camera_name=st.name,
        clip_path=final_mp4,
        meta_path=meta_path,
        raw_path=raw_path,
        meta=meta,
    )
    if vlm.submit(job):
        log.info("[%s] %s saved: %s (VLM queued)", st.name, kind, final_mp4)
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log.info("[%s] %s saved: %s (VLM full; meta written)", st.name, kind, final_mp4)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _silence_ffmpeg_stderr()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    _ensure_dirs(cfg)

    log.info("Device=%s  dtype=%s", cfg.DEVICE, cfg.DTYPE)
    trigger_names = [COCO_NAMES[i] for i in cfg.TRIGGER_CLASS_IDS if i < len(COCO_NAMES)]
    log.info("Trigger classes: %s", ", ".join(trigger_names) or "(none)")

    if not cfg.CAMERAS:
        log.error("No cameras configured in cameras.yaml — exiting.")
        return

    log.info("Starting %d camera threads...", len(cfg.CAMERAS))
    cameras: Dict[str, CameraState] = {}
    now = time.time()
    for name, rtsp in cfg.CAMERAS.items():
        cap = VideoCaptureThread(cfg, rtsp)
        norm_pts = cfg.ROI_ZONES.get(name)
        if norm_pts:
            log.info("%s: ROI zone active (%d vertices)", name, len(norm_pts))
        cameras[name] = CameraState(
            name=name, rtsp=rtsp, cap=cap,
            detection_score=0.0, last_score_ts=now,
            last_trigger_time=0.0,
            next_random_time=now + _jittered_interval(
                cfg.RANDOM_CLIP_INTERVAL_SEC, cfg.RANDOM_JITTER_FRAC,
            ),
            frame_i=0, last_yolo=None,
            trigger_detected=False, trigger_max_conf=0.0,
            yolo_class_counts={}, yolo_class_max_conf={},
            is_recording=False, trigger_ts=0.0,
            roi_polygon=None,
            roi_norm=list(norm_pts) if norm_pts else None,
        )

    ready = 0
    for name, st in cameras.items():
        ok = False
        for _ in range(30):
            ret, frame = st.cap.get_latest()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.2)
        if ok and st.cap.is_opened():
            ready += 1
            log.info("%s: connected", name)
        else:
            log.error("%s: could not read frames (check RTSP)", name)

    if ready == 0:
        log.error("No cameras producing frames — exiting.")
        for st in cameras.values():
            st.cap.release()
        return

    detector = YOLO(cfg.YOLO_MODEL)
    vlm: Optional[VLMWorker] = VLMWorker(cfg) if cfg.RUN_VLM_ON_SAVED_CLIPS else None

    if cfg.SHOW_WINDOWS:
        try:
            cv2.namedWindow("__gui_test__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__gui_test__")
            cv2.waitKey(1)
            for name in cameras:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        except cv2.error:
            log.warning("OpenCV GUI not available (headless build). Disabling display windows.")
            cfg.SHOW_WINDOWS = False

    log.info("Running.%s", " Press 'q' in any window to quit." if cfg.SHOW_WINDOWS else " Press Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            for name, st in cameras.items():
                ok, frame = st.cap.get_latest()
                if not ok or frame is None:
                    continue
                st.frame_i += 1

                # ── YOLO detection ────────────────────────────────────
                run_yolo = True
                if cfg.DEVICE == "cpu" and cfg.YOLO_EVERY_N_FRAMES_CPU > 1:
                    run_yolo = st.frame_i % cfg.YOLO_EVERY_N_FRAMES_CPU == 0

                if run_yolo:
                    results = detector(
                        frame, verbose=False,
                        conf=cfg.YOLO_TRIGGER_CONF,
                        imgsz=cfg.YOLO_IMGSZ,
                    )
                    st.last_yolo = results
                    st.yolo_class_counts, st.yolo_class_max_conf = _summarize_yolo(results)

                    st.trigger_detected = any(
                        cid in st.yolo_class_counts for cid in cfg.TRIGGER_CLASS_IDS
                    )
                    st.trigger_max_conf = max(
                        (st.yolo_class_max_conf.get(cid, 0.0) for cid in cfg.TRIGGER_CLASS_IDS),
                        default=0.0,
                    )

                    if st.roi_norm is not None:
                        if st.roi_polygon is None:
                            fh, fw = frame.shape[:2]
                            st.roi_polygon = _norm_polygon_to_px(st.roi_norm, fw, fh)
                        st.trigger_detected = _any_center_in_polygon(
                            results[0].boxes, class_ids=cfg.TRIGGER_CLASS_IDS,
                            polygon=st.roi_polygon,
                        )
                else:
                    results = st.last_yolo

                # ── Hysteresis score ─────────────────────────────────
                dt = max(0.0, now - st.last_score_ts)
                st.last_score_ts = now
                if st.trigger_detected:
                    st.detection_score = min(
                        st.detection_score + cfg.SCORE_REWARD * dt, cfg.SCORE_MAX,
                    )
                else:
                    st.detection_score = max(
                        st.detection_score - cfg.SCORE_PENALTY * dt, 0.0,
                    )

                # ── Trigger: arm ──────────────────────────────────────
                if (
                    not st.is_recording
                    and st.detection_score >= cfg.SCORE_MAX
                    and now - st.last_trigger_time > cfg.COOLDOWN_TRIGGER_SEC
                ):
                    st.is_recording = True
                    st.trigger_ts = now
                    log.info("[%s] Trigger armed, recording post-roll...", st.name)

                # ── Trigger: save after post-roll ────────────────────
                if st.is_recording and (now - st.trigger_ts >= cfg.POST_ROLL_SEC):
                    clip_frames, start_ts, end_ts, write_fps = (
                        st.cap.get_clip_last_seconds(cfg.CLIP_SECONDS)
                    )
                    if clip_frames:
                        _save_clip(cfg, detector, vlm, st, "trigger",
                                   clip_frames, start_ts, end_ts, write_fps)
                    st.is_recording = False
                    st.last_trigger_time = now

                # ── Random save ──────────────────────────────────────
                if (
                    cfg.RANDOM_CLIP_ENABLED
                    and now >= st.next_random_time
                ):
                    clip_frames, start_ts, end_ts, write_fps = (
                        st.cap.get_clip_last_seconds(cfg.CLIP_SECONDS)
                    )
                    has_full_clip = len(clip_frames) >= int(
                        cfg.CLIP_SECONDS * cfg.STORE_FPS * 0.7,
                    )
                    if has_full_clip and (cfg.RANDOM_ALLOW_PERSON or not st.trigger_detected):
                        _save_clip(cfg, detector, vlm, st, "random",
                                   clip_frames, start_ts, end_ts, write_fps)
                    st.next_random_time = now + _jittered_interval(
                        cfg.RANDOM_CLIP_INTERVAL_SEC, cfg.RANDOM_JITTER_FRAC,
                    )

                # ── Display ──────────────────────────────────────────
                if cfg.SHOW_WINDOWS:
                    if results is not None and cfg.SHOW_PLOTTED_BOXES:
                        disp = results[0].plot()
                    else:
                        disp = frame.copy()
                    if max(disp.shape[:2]) > 720:
                        scale = 720.0 / max(disp.shape[:2])
                        disp = cv2.resize(
                            disp, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    if st.roi_norm is not None:
                        dh, dw = disp.shape[:2]
                        roi_disp = _norm_polygon_to_px(st.roi_norm, dw, dh)
                        overlay = disp.copy()
                        cv2.fillPoly(overlay, [roi_disp], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.15, disp, 0.85, 0, disp)
                        cv2.polylines(disp, [roi_disp], True, (0, 255, 0), 2)
                    cv2.imshow(st.name, disp)

            if cfg.SHOW_WINDOWS and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

    finally:
        for st in cameras.values():
            st.cap.release()
        if cfg.SHOW_WINDOWS:
            cv2.destroyAllWindows()
        if vlm is not None:
            vlm.stop()
        log.info("Stopped.")


if __name__ == "__main__":
    main()
