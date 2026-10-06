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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import COCO_NAMES, Config, load_config
if __package__:
    from .detector import load_detector
    from .zones import ZoneMask, mask_for   # package mode (the preview test loads this file under the package)
    from .streams import SubStreamThread, MainStreamThread, _MaskFailState, _masked_or_none, log
    from . import vlm_crop                  # the crop, shared with inference
else:
    from detector import load_detector
    from zones import ZoneMask, mask_for    # script mode: run_collector.sh puts this dir on sys.path
    import vlm_crop
    from streams import SubStreamThread, MainStreamThread, _MaskFailState, _masked_or_none, log

# Keep the shared readers and the collector on the same logger.


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Suppress FFmpeg stderr noise
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _silence_ffmpeg_stderr() -> None:
    """No-op kept for backward compatibility.

    Previously redirected fd 2 to /dev/null to suppress FFmpeg warnings, but
    that broke output capture under Git Bash + uv run + redirection and twice
    masked real crashes (torchvision::nms NotImplementedError, etc.) by eating
    Python tracebacks. FFmpeg's stderr noise is harmless; leave it visible.
    """
    return


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Utilities
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _ensure_dirs(cfg: Config) -> None:
    root = cfg.OUT_DIR
    for sub in ("clips", "meta", "responses"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    if cfg.MAIN_STREAM_ENABLED:
        os.makedirs(os.path.join(root, "vlm_crops"), exist_ok=True)
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ROI polygon helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _norm_polygon_to_px(
    norm_pts: List[Tuple[float, float]], w: int, h: int,
) -> np.ndarray:
    """Convert normalised (0-1) polygon to pixel int32 contour for cv2."""
    return np.array(
        [[int(x * w), int(y * h)] for x, y in norm_pts], dtype=np.int32,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Trigger-class crop helpers (for VLM main-stream clips)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# The crop helpers live in vlm_crop.py, shared with inference (the AI sees this exact crop).
_scale_boxes = vlm_crop.scale_boxes
_compute_trigger_crop = vlm_crop.compute_trigger_crop
_smooth_crops = vlm_crop.smooth_crops


def _smooth_crops(
    crops: List[Optional[Tuple[int, int, int, int]]],
    alpha: float,
) -> List[Optional[Tuple[int, int, int, int]]]:
    """Apply EMA smoothing to a sequence of crop regions for temporal stability."""
    if not crops:
        return []
    smoothed: List[Optional[Tuple[int, int, int, int]]] = []
    prev: Optional[Tuple[float, float, float, float]] = None
    for crop in crops:
        if crop is None:
            smoothed.append(prev if prev else None)
            continue
        if prev is None:
            prev = tuple(float(v) for v in crop)  # type: ignore[assignment]
            smoothed.append(crop)
        else:
            s = tuple(
                alpha * float(c) + (1.0 - alpha) * p
                for c, p in zip(crop, prev)
            )
            prev = s
            smoothed.append((int(s[0]), int(s[1]), int(s[2]), int(s[3])))
    return smoothed


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# YOLO weak-label export
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Threaded RTSP capture â€” sub-stream (raw numpy buffer for YOLO)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# VLM worker (optional, loaded lazily)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        log.info("VLM response [%s]: %s", job.camera_name, out_text)
        if self.cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()

    def stop(self) -> None:
        self.running = False
        try:
            self.thread.join(timeout=2)
        except Exception:
            pass


@dataclass
class _VlmCropJob:
    cfg: Config
    detector: YOLO
    st_name: str
    sub_frames: List[np.ndarray]
    sub_start_ts: float
    sub_end_ts: float
    main_clip_data: Tuple[List[np.ndarray], float, float, float]
    day: str
    clip_id: str
    meta_path: str
    meta: Dict[str, Any]


class _VlmCropWorker:
    """Background thread that generates VLM crop clips without blocking the
    detection loop.  Follows the same queue pattern as ``VLMWorker``."""

    def __init__(self) -> None:
        self.q: queue.Queue[_VlmCropJob] = queue.Queue(maxsize=50)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="vlm-crop")
        self.thread.start()

    def submit(self, job: _VlmCropJob) -> bool:
        try:
            self.q.put_nowait(job)
            return True
        except queue.Full:
            log.warning("VLM-crop queue full; skipping crop for %s", job.clip_id)
            return False

    def _loop(self) -> None:
        while self.running:
            try:
                job = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(job)
            except Exception:
                log.exception("VLM crop job failed for %s", job.clip_id)
            finally:
                self.q.task_done()

    def _process(self, job: _VlmCropJob) -> None:
        # Build a minimal CameraState-like object for the existing function
        # signature â€” we only need `.name`.
        class _Stub:
            pass
        stub = _Stub()
        stub.name = job.st_name  # type: ignore[attr-defined]

        vlm_crop_meta = _save_vlm_crop_clip(
            cfg=job.cfg,
            detector=job.detector,
            st=stub,  # type: ignore[arg-type]
            sub_frames=job.sub_frames,
            sub_start_ts=job.sub_start_ts,
            sub_end_ts=job.sub_end_ts,
            main_clip_data=job.main_clip_data,
            day=job.day,
            clip_id=job.clip_id,
        )
        if vlm_crop_meta is not None:
            job.meta["vlm_crop"] = vlm_crop_meta
            with open(job.meta_path, "w", encoding="utf-8") as f:
                json.dump(job.meta, f, ensure_ascii=False, indent=2)

    def stop(self) -> None:
        self.running = False
        try:
            self.q.join()
        except Exception:
            pass
        try:
            self.thread.join(timeout=5)
        except Exception:
            pass


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Per-camera runtime state
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class CameraState:
    name: str
    rtsp: str
    rtsp_main: str
    cap: SubStreamThread
    main_cap: Optional[MainStreamThread]
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
    main_connect_ts: float
    roi_norm: Optional[List[Tuple[float, float]]]
    roi_black: Optional[List[List[Tuple[float, float]]]] = None   # scene-map black areas, blacked out too
    last_score_log: float = 0.0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Clip saving
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _save_vlm_crop_clip(
    cfg: Config,
    detector: YOLO,
    st: CameraState,
    sub_frames: List[np.ndarray],
    sub_start_ts: float,
    sub_end_ts: float,
    main_clip_data: Optional[Tuple[List[np.ndarray], float, float, float]],
    day: str,
    clip_id: str,
) -> Optional[Dict[str, Any]]:
    """Extract main-stream frames, crop around trigger-class detections, save.

    YOLO runs on the lightweight *sub_frames* (already in memory from the
    sub-stream clip save) and the detections are scaled to main-stream
    coordinates.  This avoids running YOLO on high-res frames entirely.

    *main_clip_data* must be pre-fetched by the caller at the moment the
    save decision is made, before any expensive I/O.  This prevents the
    rolling buffer's time window from shifting past the event.
    """
    if not cfg.MAIN_STREAM_ENABLED or main_clip_data is None:
        return None

    main_frames, m_start, m_end, m_fps = main_clip_data
    settings = vlm_crop.settings_from_config(cfg)
    result = vlm_crop.crop_clip(detector, settings, sub_frames, sub_start_ts, sub_end_ts,
                                main_frames, m_start, name=st.name)
    if result is None:
        return None
    cropped_frames = result.frames
    median_w, median_h = result.width, result.height

    vlm_dir = os.path.join(cfg.OUT_DIR, "vlm_crops", st.name, day)
    os.makedirs(vlm_dir, exist_ok=True)
    tmp = _write_mp4_clip(cropped_frames, fps=m_fps)
    vlm_mp4 = os.path.join(vlm_dir, f"{clip_id}.mp4")
    shutil.move(tmp, vlm_mp4)

    log.info("[%s] VLM crop saved: %s (%dx%d, %d frames, per-frame tracking)",
             st.name, vlm_mp4, median_w, median_h, len(cropped_frames))

    return vlm_crop.crop_meta(result, settings, os.path.relpath(vlm_mp4, start=cfg.OUT_DIR), m_fps)


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
    main_clip_data: Optional[Tuple[List[np.ndarray], float, float, float]] = None,
    vlm_crop_worker: Optional[_VlmCropWorker] = None,
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
            "active": st.roi_norm is not None,            # the clip was masked to this zone
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

    # Offload VLM crop to background thread so the detection loop isn't blocked
    if vlm_crop_worker is not None and main_clip_data is not None:
        crop_job = _VlmCropJob(
            cfg=cfg, detector=detector, st_name=st.name,
            sub_frames=list(frames), sub_start_ts=start_ts, sub_end_ts=end_ts,
            main_clip_data=main_clip_data,
            day=day, clip_id=clip_id,
            meta_path=meta_path, meta=meta,
        )
        vlm_crop_worker.submit(crop_job)

    if vlm is None:
        # Write meta now; the crop worker will re-write it with vlm_crop when done.
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    _silence_ffmpeg_stderr()

    # Force logging to stdout (line-buffered under `python -u`) instead of the
    # default stderr, which Git Bash + mintty block-buffer unpredictably.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        pass
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=logging.INFO, handlers=[_stdout_handler], force=True)

    cfg = load_config()
    preview = None
    if cfg.PREVIEW_ENABLED:
        # The legacy entry point is a script, so only its own directory is on
        # sys.path. Add the checkout root for the optional box bridge.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from home_guard_project.box.preview import PreviewWriter
        preview = PreviewWriter(os.path.join(project_root, "logs", "preview"), enabled=True)
        preview.set_cameras(cfg.CAMERAS)
    _ensure_dirs(cfg)

    log.info("Device=%s  dtype=%s", cfg.DEVICE, cfg.DTYPE)
    trigger_names = [COCO_NAMES[i] for i in cfg.TRIGGER_CLASS_IDS if i < len(COCO_NAMES)]
    log.info("Trigger classes: %s", ", ".join(trigger_names) or "(none)")

    if not cfg.CAMERAS:
        log.error("No cameras configured in cameras.yaml â€” exiting.")
        return

    log.info("Starting %d camera threads (sub-stream)...", len(cfg.CAMERAS))
    if cfg.MAIN_STREAM_ENABLED:
        log.info("Main-stream VLM crops enabled (on-demand, %.0f fps, q=%d)",
                 cfg.MAIN_STORE_FPS, cfg.MAIN_JPEG_QUALITY)
    cameras: Dict[str, CameraState] = {}
    now = time.time()
    for name, rtsp_sub in cfg.CAMERAS.items():
        rtsp_main = cfg.CAMERAS_MAIN.get(name, "")
        cap = SubStreamThread(cfg, rtsp_sub, mask=mask_for(cfg.ROI_ZONES, name, cfg.ROI_BLACK), name=name)
        norm_pts = cfg.ROI_ZONES.get(name)
        if norm_pts:
            log.info("%s: watch zone active (%d corners); everything outside is blacked out", name, len(norm_pts))
        cameras[name] = CameraState(
            name=name, rtsp=rtsp_sub, rtsp_main=rtsp_main,
            cap=cap, main_cap=None,
            detection_score=0.0, last_score_ts=now,
            last_trigger_time=0.0,
            next_random_time=now + _jittered_interval(
                cfg.RANDOM_CLIP_INTERVAL_SEC, cfg.RANDOM_JITTER_FRAC,
            ),
            frame_i=0, last_yolo=None,
            trigger_detected=False, trigger_max_conf=0.0,
            yolo_class_counts={}, yolo_class_max_conf={},
            is_recording=False, trigger_ts=0.0, main_connect_ts=0.0,
            roi_norm=list(norm_pts) if norm_pts else None,
            roi_black=cfg.ROI_BLACK.get(name) or None,
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
            log.info("%s: sub-stream connected", name)
        else:
            log.error("%s: could not read sub-stream frames (check RTSP)", name)

    if ready == 0:
        log.error("No cameras producing frames â€” exiting.")
        for st in cameras.values():
            st.cap.release()
        return

    # An NVIDIA card (the laptop) beats the Intel graphics chip: plain PyTorch already uses it.
    detector = load_detector(cfg.YOLO_MODEL, cfg.YOLO_DEVICE if cfg.DEVICE == "cpu" else "cpu")
    vlm: Optional[VLMWorker] = VLMWorker(cfg) if cfg.RUN_VLM_ON_SAVED_CLIPS else None
    crop_worker: Optional[_VlmCropWorker] = _VlmCropWorker() if cfg.MAIN_STREAM_ENABLED else None

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

                # â”€â”€ YOLO detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                run_yolo = True
                # Every picture on the graphics chip; on the CPU only every Nth, it cannot keep up.
                if cfg.DEVICE == "cpu" and not detector.on_graphics_chip and cfg.YOLO_EVERY_N_FRAMES_CPU > 1:
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
                else:
                    results = st.last_yolo

                # â”€â”€ Hysteresis score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

                if st.trigger_detected and now - st.last_score_log > 1.0:
                    counts = ", ".join(
                        f"{COCO_NAMES[cid]}x{n}"
                        for cid, n in st.yolo_class_counts.items()
                        if cid in cfg.TRIGGER_CLASS_IDS and n > 0
                    ) or "?"
                    log.info("[%s] detecting %s â€” score=%.1f/%.1f",
                             st.name, counts, st.detection_score, cfg.SCORE_MAX)
                    st.last_score_log = now

                # â”€â”€ Main-stream pre-connect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Start the main-stream RTSP as soon as the first
                # detection appears so it accumulates a full
                # CLIP_SECONDS of footage by save time.
                if (
                    cfg.MAIN_STREAM_ENABLED
                    and st.rtsp_main
                    and st.main_cap is None
                    and not st.is_recording
                    and st.trigger_detected
                    and st.detection_score > 0
                ):
                    st.main_cap = MainStreamThread(cfg, st.rtsp_main, mask=ZoneMask(st.roi_norm, st.roi_black), name=st.name)
                    st.main_connect_ts = now
                    log.info("[%s] Pre-connecting main-stream (score=%.1f)",
                             st.name, st.detection_score)

                # Release pre-connected main-stream if detection fades
                if (
                    st.main_cap is not None
                    and not st.is_recording
                    and st.detection_score <= 0
                ):
                    st.main_cap.release()
                    st.main_cap = None
                    st.main_connect_ts = 0.0
                    log.debug("[%s] Released idle main-stream", st.name)

                # â”€â”€ Trigger: arm â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if (
                    not st.is_recording
                    and st.detection_score >= cfg.SCORE_MAX
                    and now - st.last_trigger_time > cfg.COOLDOWN_TRIGGER_SEC
                ):
                    st.is_recording = True
                    st.trigger_ts = 0.0
                    if cfg.MAIN_STREAM_ENABLED and st.rtsp_main:
                        if st.main_cap is None:
                            st.main_cap = MainStreamThread(cfg, st.rtsp_main, mask=ZoneMask(st.roi_norm, st.roi_black), name=st.name)
                            st.main_connect_ts = now
                        if st.main_cap.frame_shape is not None:
                            st.trigger_ts = now
                            log.info("[%s] Trigger armed (main-stream already connected)",
                                     st.name)
                        else:
                            log.info("[%s] Trigger armed â€” waiting for main-stream...",
                                     st.name)
                    else:
                        st.trigger_ts = now
                        log.info("[%s] Trigger armed, recording post-roll...", st.name)

                # Wait for main-stream to deliver its first frame
                # before starting the post-roll timer.  Give up after
                # OPEN_TIMEOUT_MSEC + 2s so we don't block forever.
                if (
                    st.is_recording
                    and st.trigger_ts == 0.0
                    and st.main_cap is not None
                    and st.main_connect_ts > 0.0
                ):
                    if st.main_cap.is_opened() and st.main_cap.frame_shape is not None:
                        st.trigger_ts = now
                        log.info("[%s] Main-stream connected, recording post-roll...", st.name)
                    elif now - st.main_connect_ts > cfg.OPEN_TIMEOUT_MSEC / 1000.0 + 2.0:
                        log.warning("[%s] Main-stream failed to connect â€” saving without VLM crop", st.name)
                        st.main_cap.release()
                        st.main_cap = None
                        st.trigger_ts = now

                # â”€â”€ Trigger: save after post-roll â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                post_roll_elapsed = (
                    st.is_recording
                    and st.trigger_ts > 0.0
                    and (now - st.trigger_ts >= cfg.POST_ROLL_SEC)
                )
                if post_roll_elapsed:
                    main_ready = True
                    if st.main_cap is not None:
                        buf_dur = st.main_cap.buffer_duration()
                        max_extra = cfg.CLIP_SECONDS
                        if (
                            buf_dur < cfg.CLIP_SECONDS
                            and now - st.trigger_ts < cfg.POST_ROLL_SEC + max_extra
                        ):
                            main_ready = False
                    if main_ready:
                        main_clip_data = None
                        if st.main_cap is not None:
                            main_clip_data = st.main_cap.get_clip_frames(cfg.CLIP_SECONDS)
                            if len(main_clip_data[0]) < 2:
                                main_clip_data = None
                        clip_frames, start_ts, end_ts, write_fps = (
                            st.cap.get_clip_last_seconds(cfg.CLIP_SECONDS)
                        )
                        if clip_frames:
                            _save_clip(cfg, detector, vlm, st, "trigger",
                                       clip_frames, start_ts, end_ts, write_fps,
                                       main_clip_data=main_clip_data,
                                       vlm_crop_worker=crop_worker)
                        if st.main_cap is not None:
                            st.main_cap.release()
                            st.main_cap = None
                        st.is_recording = False
                        st.last_trigger_time = now

                # â”€â”€ Random save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                                   clip_frames, start_ts, end_ts, write_fps,
                                   vlm_crop_worker=crop_worker)
                    st.next_random_time = now + _jittered_interval(
                        cfg.RANDOM_CLIP_INTERVAL_SEC, cfg.RANDOM_JITTER_FRAC,
                    )

                # â”€â”€ Display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if cfg.SHOW_WINDOWS or (preview is not None and preview.wanted(st.name, frame)):
                    if results is not None and cfg.SHOW_PLOTTED_BOXES:
                        disp = results[0].plot()
                    else:
                        disp = frame.copy()
                    preview_limit = 1280 if preview is not None else 720
                    if max(disp.shape[:2]) > preview_limit:
                        scale = float(preview_limit) / max(disp.shape[:2])
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
                    if preview is not None:
                        preview.publish(st.name, disp, source=frame)
                    if cfg.SHOW_WINDOWS:
                        cv2.imshow(st.name, disp)

            if cfg.SHOW_WINDOWS and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

    finally:
        for st in cameras.values():
            st.cap.release()
            if st.main_cap is not None:
                st.main_cap.release()
        if cfg.SHOW_WINDOWS:
            cv2.destroyAllWindows()
        if crop_worker is not None:
            crop_worker.stop()
        if vlm is not None:
            vlm.stop()
        log.info("Stopped.")


if __name__ == "__main__":
    main()
