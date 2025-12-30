from __future__ import annotations

import os
import cv2
import json
import time
import queue
import torch
import random
import shutil
import datetime
import threading
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from collections import deque, defaultdict

import numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText


# =========================
# 0) PROMPT
# =========================
PROMPT_TEMPLATE = """
You are a security camera assistant.

Camera name: {camera_name}

Task:
Describe what happens in the video

Alert policy:
- If no person is visible: alert_command = "[none]"
- If any person is visible: alert_command = "[send_message]"
- If suspicious behavior is visible alert_command = "[call_owner]"

Output format:
{{
  "summary": "<long summary of the video explaining what is happening in detail>",
  "alert_reason": "<reason what alert_command you chose and why>",
  "alert_command": "[none] or [send_message] or [call_owner]"
}}
""".strip()


def log(*args):
    import sys
    print(*args, file=sys.stderr, flush=True)


# =========================
# 1) CONFIG
# =========================
@dataclass
class Config:
    OUT_DIR: str = "./dataset_multi"

    CAMERAS: Dict[str, str] = None  # set in __post_init__

    YOLO_MODEL: str = "yolov8n.pt"
    VLM_MODEL_ID: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Clip settings
    CLIP_SECONDS: float = 10.0

    # Critical performance knobs
    STORE_FPS: float = 10.0                   # how many frames/sec we KEEP per camera
    # STORE_SIZE: Tuple[int, int] = (640, 360)
    STORE_SIZE: Tuple[int, int] = (960, 540)

    # Trigger hysteresis (time-based)
    SCORE_MAX: float = 10.0       # points needed to trigger
    SCORE_REWARD: float = 3.0     # points/sec added while person present
    SCORE_PENALTY: float = 2.0    # points/sec subtracted while absent
    COOLDOWN_TRIGGER_SEC: float = 5.0

    # Random sampling
    RANDOM_CLIP_INTERVAL_SEC: float = 1800.0
    RANDOM_JITTER_FRAC: float = 0.25
    RANDOM_ALLOW_PERSON: bool = True

    # VLM
    RUN_VLM_ON_SAVED_CLIPS: bool = True
    VLM_SAMPLE_FPS: int = 1

    # YOLO export for training (NEW)
    EXPORT_YOLO_TRAINING_DATA: bool = True
    YOLO_EXPORT_FPS: float = 2.0            # how many frames/sec to export from each saved clip
    YOLO_EXPORT_CONF: float = 0.25          # YOLO confidence threshold for weak labels
    YOLO_EXPORT_JPEG_QUALITY: int = 90
    YOLO_EXPORT_SUBDIR: str = "yolo"        # saved under OUT_DIR/yolo/images + OUT_DIR/yolo/labels

    # RTSP stability
    OPENCV_FFMPEG_CAPTURE_OPTIONS: str = (
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer"
    )
    OPEN_TIMEOUT_MSEC: int = 5000
    READ_TIMEOUT_MSEC: int = 5000
    FREEZE_RECONNECT_AFTER_SEC: float = 3.0
    RECONNECT_BACKOFF_START: float = 1.0
    RECONNECT_BACKOFF_MAX: float = 10.0

    # Performance
    YOLO_EVERY_N_FRAMES_CPU: int = 6
    SHOW_WINDOWS: bool = True
    SHOW_PLOTTED_BOXES: bool = False

    def __post_init__(self):
        if self.CAMERAS is None:
            self.CAMERAS = {
                "main_door": "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c6/s1/live",
                "back_door": "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c1/s1/live",
                "left_side_1": "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c2/s1/live",
                "front_side": "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c3/s1/live",
                "left_side_2": "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c5/s1/live",
                "right_side": "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c8/s1/live",
            }
            # self.CAMERAS = {
            #     "main_door": "rtsp://admin:Aa123123%40@192.168.68.103:554/unicast/c2/s1/live",
            #     "back_door": "rtsp://admin:Aa123123%40@192.168.68.103:554/unicast/c1/s1/live",
            #     "right_side": "rtsp://admin:Aa123123%40@192.168.68.103:554/unicast/c3/s1/live",
            # }


# =========================
# 2) UTILITIES
# =========================
def ensure_dirs(cfg: Config):
    root = cfg.OUT_DIR
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "clips"), exist_ok=True)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    os.makedirs(os.path.join(root, "responses"), exist_ok=True)

    # YOLO export dirs (NEW)
    if cfg.EXPORT_YOLO_TRAINING_DATA:
        base = os.path.join(root, cfg.YOLO_EXPORT_SUBDIR)
        os.makedirs(os.path.join(base, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, "labels"), exist_ok=True)


def utc_iso(ts: float) -> str:
    return (
        datetime.datetime.utcfromtimestamp(ts)
        .replace(microsecond=0)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def local_iso(ts: float) -> str:
    return (
        datetime.datetime.fromtimestamp(ts)
        .replace(microsecond=0)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def jittered_interval(base: float, frac: float) -> float:
    lo = 1.0 - frac
    hi = 1.0 + frac
    return base * (lo + (hi - lo) * random.random())


def write_mp4_clip(frames: List[np.ndarray], fps: float) -> str:
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


def summarize_yolo(results) -> Tuple[Dict[int, int], Dict[int, float]]:
    counts = defaultdict(int)
    max_conf = defaultdict(float)

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


def xyxy_to_yolo_norm(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    """
    Convert pixel xyxy box to normalized YOLO xywh.
    """
    x1 = max(0.0, min(float(x1), float(w - 1)))
    x2 = max(0.0, min(float(x2), float(w - 1)))
    y1 = max(0.0, min(float(y1), float(h - 1)))
    y2 = max(0.0, min(float(y2), float(h - 1)))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    # Normalize
    return (xc / w, yc / h, bw / w, bh / h)


def export_yolo_frames_and_labels(
    cfg: Config,
    detector: YOLO,
    frames: List[np.ndarray],
    camera_name: str,
    day: str,
    clip_id: str,
) -> List[Dict[str, Any]]:
    """
    NEW: Export frames + YOLO-format .txt labels as weak labels.

    Output structure:
      OUT_DIR/yolo/images/<camera>/<day>/<clip_id>_f0000.jpg
      OUT_DIR/yolo/labels/<camera>/<day>/<clip_id>_f0000.txt

    Returns a list of exported items for meta.json.
    """
    if not cfg.EXPORT_YOLO_TRAINING_DATA:
        return []

    if not frames:
        return []

    base_dir = os.path.join(cfg.OUT_DIR, cfg.YOLO_EXPORT_SUBDIR)
    img_dir = os.path.join(base_dir, "images", camera_name, day)
    lbl_dir = os.path.join(base_dir, "labels", camera_name, day)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Choose which frames to export (e.g., 2 fps from 8 fps buffer)
    store_fps = float(cfg.STORE_FPS)
    export_fps = max(0.1, float(cfg.YOLO_EXPORT_FPS))
    step = max(1, int(round(store_fps / export_fps)))

    exported: List[Dict[str, Any]] = []
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.YOLO_EXPORT_JPEG_QUALITY)]

    for i in range(0, len(frames), step):
        frame = frames[i]
        if frame is None:
            continue

        h, w = frame.shape[:2]
        img_name = f"{clip_id}_f{i:04d}.jpg"
        lbl_name = f"{clip_id}_f{i:04d}.txt"

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        # Save image (raw frame, no boxes drawn!)
        ok = cv2.imwrite(img_path, frame, jpeg_params)
        if not ok:
            continue

        # Run YOLO to get weak boxes (conf threshold)
        # (This does not affect your trigger loop; it's only at save time.)
        results = detector(frame, verbose=False, conf=float(cfg.YOLO_EXPORT_CONF))

        lines: List[str] = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                xc, yc, bw, bh = xyxy_to_yolo_norm(x1, y1, x2, y2, w=w, h=h)

                # YOLO label line: class x_center y_center width height (all normalized)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Save label file (empty file is OK if no objects)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        exported.append({
            "frame_index": int(i),
            "approx_time_offset_sec": float(i / store_fps),
            "image_path": os.path.relpath(img_path, start=cfg.OUT_DIR),
            "label_path": os.path.relpath(lbl_path, start=cfg.OUT_DIR),
            "num_boxes": int(len(lines)),
        })

    return exported


# =========================
# 3) THREADED VIDEO CAPTURE (per camera) — only resize/store at STORE_FPS
# =========================
class VideoCaptureThread:
    """
    Reads RTSP continuously but only *keeps* frames at STORE_FPS.
    Kept frames are resized to STORE_SIZE and stored in a rolling buffer.
    """

    def __init__(self, cfg: Config, src: str):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", cfg.OPENCV_FFMPEG_CAPTURE_OPTIONS)

        self.cfg = cfg
        self.src = src

        self.capture: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()

        self.buf: deque[Tuple[float, np.ndarray]] = deque()
        self.buf_lock = threading.Lock()

        self.latest_frame: Optional[np.ndarray] = None  # latest stored frame (downscaled)

        self.running = True
        self.last_frame_ts = 0.0
        self.reconnect_backoff = float(cfg.RECONNECT_BACKOFF_START)

        self.store_interval = 1.0 / max(1e-6, float(cfg.STORE_FPS))
        self.last_store_ts = 0.0

        self.keep_seconds = float(cfg.CLIP_SECONDS) + 2.0

        self._open_capture()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _open_capture(self):
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

    def _reconnect(self):
        time.sleep(self.reconnect_backoff)
        self.reconnect_backoff = min(self.reconnect_backoff * 1.5, float(self.cfg.RECONNECT_BACKOFF_MAX))
        self._open_capture()

    def _reader(self):
        target_w, target_h = self.cfg.STORE_SIZE

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

            # Only keep/store frames at STORE_FPS
            if (now - self.last_store_ts) < self.store_interval:
                continue
            self.last_store_ts = now

            if target_w and target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            self.latest_frame = frame

            cutoff = now - self.keep_seconds
            with self.buf_lock:
                self.buf.append((now, frame))
                while self.buf and self.buf[0][0] < cutoff:
                    self.buf.popleft()

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.latest_frame is None:
            return False, None
        return True, self.latest_frame

    def get_clip_last_seconds(self, clip_seconds: float) -> Tuple[List[np.ndarray], float, float, float]:
        now = time.time()
        cutoff = now - float(clip_seconds)

        with self.buf_lock:
            items = [(t, f) for (t, f) in self.buf if t >= cutoff]

        if len(items) < 2:
            return [], now, now, float(self.cfg.STORE_FPS)

        start_ts = items[0][0]
        end_ts = items[-1][0]
        frames = [f for _, f in items]

        # We write at STORE_FPS for stable playback
        return frames, start_ts, end_ts, float(self.cfg.STORE_FPS)

    def isOpened(self) -> bool:
        with self.lock:
            return self.capture is not None and self.capture.isOpened()

    def release(self):
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


# =========================
# 4) SINGLE VLM WORKER (RAW output only)
# =========================
@dataclass
class ClipJob:
    camera_name: str
    clip_path: str
    meta_path: str
    raw_path: str
    meta: Dict[str, Any]


class SmolVLM2Worker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.VLM_MODEL_ID, use_fast=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.VLM_MODEL_ID,
            torch_dtype=cfg.DTYPE,
            low_cpu_mem_usage=True,
        ).to(cfg.DEVICE)

        self.q: "queue.Queue[ClipJob]" = queue.Queue(maxsize=200)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        log(f"[VLM] Loaded {cfg.VLM_MODEL_ID} on {cfg.DEVICE} ({cfg.DTYPE}).")

    def submit(self, job: ClipJob) -> bool:
        try:
            self.q.put_nowait(job)
            return True
        except queue.Full:
            log("[VLM] Queue full; dropping job.")
            return False

    def _loop(self):
        while self.running:
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run(job)
            except Exception as e:
                log("[VLM] job failed:", e)
            finally:
                self.q.task_done()

    @torch.no_grad()
    def _run(self, job: ClipJob):
        prompt = PROMPT_TEMPLATE.format(camera_name=job.camera_name)

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

        gen = self.model.generate(**inputs, max_new_tokens=220, do_sample=False)

        prompt_len = inputs["input_ids"].shape[1]
        out_text = self.processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0].strip()

        with open(job.raw_path, "w", encoding="utf-8") as f:
            f.write(out_text)

        job.meta["prompt_camera_name"] = job.camera_name
        job.meta["prompt_used"] = prompt
        job.meta["model_raw_text_path"] = os.path.relpath(job.raw_path, start=self.cfg.OUT_DIR)
        job.meta["model_response"] = out_text
        job.meta["model_response_path"] = os.path.relpath(job.raw_path, start=self.cfg.OUT_DIR)

        with open(job.meta_path, "w", encoding="utf-8") as f:
            json.dump(job.meta, f, ensure_ascii=False, indent=2)

        log(f"[VLM] done: {job.camera_name} -> {os.path.basename(job.clip_path)}")

        if self.cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=2)
        except Exception:
            pass


# =========================
# 5) PER-CAMERA STATE
# =========================
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
    last_person: bool
    last_car: bool
    last_person_conf: float
    last_car_conf: float

    yolo_class_counts: Dict[int, int]
    yolo_class_max_conf: Dict[int, float]


# =========================
# 6) MAIN
# =========================
def main():
    cfg = Config()
    ensure_dirs(cfg)

    log(f"[System] Device={cfg.DEVICE} dtype={cfg.DTYPE}")
    log("[System] Starting camera threads...")

    cameras: Dict[str, CameraState] = {}
    for name, rtsp in cfg.CAMERAS.items():
        cap = VideoCaptureThread(cfg, rtsp)
        now = time.time()
        st = CameraState(
            name=name,
            rtsp=rtsp,
            cap=cap,
            detection_score=0.0,
            last_score_ts=now,
            last_trigger_time=0.0,
            next_random_time=now + jittered_interval(cfg.RANDOM_CLIP_INTERVAL_SEC, cfg.RANDOM_JITTER_FRAC),
            frame_i=0,
            last_yolo=None,
            last_person=False,
            last_car=False,
            last_person_conf=0.0,
            last_car_conf=0.0,
            yolo_class_counts={},
            yolo_class_max_conf={},
        )
        cameras[name] = st

    # readiness check
    ready = 0
    for name, st in cameras.items():
        ok = False
        for _ in range(30):
            ret, frame = st.cap.get_latest()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.2)
        if ok and st.cap.isOpened():
            ready += 1
            log(f"[System] {name}: connected")
        else:
            log(f"[Error] {name}: could not read frames (check RTSP)")

    if ready == 0:
        log("[Error] No cameras are producing frames; exiting.")
        for st in cameras.values():
            st.cap.release()
        return

    detector = YOLO(cfg.YOLO_MODEL)
    vlm = SmolVLM2Worker(cfg) if cfg.RUN_VLM_ON_SAVED_CLIPS else None

    if cfg.SHOW_WINDOWS:
        for name in cameras.keys():
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def save_job(st: CameraState, kind: str, frames: List[np.ndarray], start_ts: float, end_ts: float, fps: float):
        if not frames:
            return

        day = datetime.datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
        clip_id = f"{st.name}_{int(end_ts)}_{kind}"

        clips_dir = os.path.join(cfg.OUT_DIR, "clips", st.name, day)
        meta_dir = os.path.join(cfg.OUT_DIR, "meta", st.name, day)
        resp_dir = os.path.join(cfg.OUT_DIR, "responses", st.name, day)
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(resp_dir, exist_ok=True)

        tmp = write_mp4_clip(frames, fps=fps)
        final_mp4 = os.path.join(clips_dir, clip_id + ".mp4")
        shutil.move(tmp, final_mp4)

        meta_path = os.path.join(meta_dir, clip_id + ".meta.json")
        raw_path = os.path.join(resp_dir, clip_id + ".model_raw.txt")

        meta: Dict[str, Any] = {
            "camera_name": st.name,
            "kind": kind,
            "clip_path": os.path.relpath(final_mp4, start=cfg.OUT_DIR),

            "clip_start_ts": float(start_ts),
            "clip_end_ts": float(end_ts),
            "clip_start_utc": utc_iso(start_ts),
            "clip_end_utc": utc_iso(end_ts),
            "clip_start_local": local_iso(start_ts),
            "clip_end_local": local_iso(end_ts),

            "duration_sec": float(end_ts - start_ts),
            "frames_written": int(len(frames)),
            "fps_estimated": float(fps),
            "vlm_sample_fps": int(cfg.VLM_SAMPLE_FPS),

            "buffer": {
                "store_fps": float(cfg.STORE_FPS),
                "store_size": list(cfg.STORE_SIZE),
            },

            "yolo": {
                "class_counts": st.yolo_class_counts,
                "class_max_conf": st.yolo_class_max_conf,
                "person_detected": bool(st.last_person),
                "car_detected": bool(st.last_car),
                "person_conf_max": float(st.last_person_conf),
                "car_conf_max": float(st.last_car_conf),
                "detection_score": float(st.detection_score),
            },
        }

        # NEW: export frames + YOLO labels for training
        if cfg.EXPORT_YOLO_TRAINING_DATA:
            exported = export_yolo_frames_and_labels(
                cfg=cfg,
                detector=detector,
                frames=frames,
                camera_name=st.name,
                day=day,
                clip_id=clip_id,
            )
            meta["yolo_export"] = {
                "enabled": True,
                "export_fps": float(cfg.YOLO_EXPORT_FPS),
                "conf": float(cfg.YOLO_EXPORT_CONF),
                "images_root": os.path.relpath(os.path.join(cfg.OUT_DIR, cfg.YOLO_EXPORT_SUBDIR, "images"), start=cfg.OUT_DIR),
                "labels_root": os.path.relpath(os.path.join(cfg.OUT_DIR, cfg.YOLO_EXPORT_SUBDIR, "labels"), start=cfg.OUT_DIR),
                "exported_frames": exported,
            }

        if vlm is None:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            log(f"[Save] {st.name} {kind}: {final_mp4} (no VLM)")
            return

        job = ClipJob(
            camera_name=st.name,
            clip_path=final_mp4,
            meta_path=meta_path,
            raw_path=raw_path,
            meta=meta,
        )
        if vlm.submit(job):
            log(f"[Save] {st.name} {kind}: {final_mp4} (queued VLM)")
        else:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            log(f"[Save] {st.name} {kind}: {final_mp4} (VLM queue full; meta saved)")

    log("[System] Running. Press 'q' in any window to quit.")
    try:
        while True:
            now = time.time()

            for name, st in cameras.items():
                ok, frame = st.cap.get_latest()
                if not ok or frame is None:
                    continue

                st.frame_i += 1

                # YOLO throttling
                run_yolo = True
                if cfg.DEVICE == "cpu" and cfg.YOLO_EVERY_N_FRAMES_CPU > 1:
                    run_yolo = (st.frame_i % cfg.YOLO_EVERY_N_FRAMES_CPU == 0)

                if run_yolo:
                    results = detector(frame, verbose=False)
                    st.last_yolo = results
                    st.yolo_class_counts, st.yolo_class_max_conf = summarize_yolo(results)

                    st.last_person = (0 in st.yolo_class_counts)
                    st.last_car = (2 in st.yolo_class_counts)
                    st.last_person_conf = float(st.yolo_class_max_conf.get(0, 0.0))
                    st.last_car_conf = float(st.yolo_class_max_conf.get(2, 0.0))
                else:
                    results = st.last_yolo

                # Time-based score update
                dt = max(0.0, now - st.last_score_ts)
                st.last_score_ts = now

                if st.last_person:
                    st.detection_score = min(st.detection_score + cfg.SCORE_REWARD * dt, cfg.SCORE_MAX)
                else:
                    st.detection_score = max(st.detection_score - cfg.SCORE_PENALTY * dt, 0.0)

                clip_frames, start_ts, end_ts, write_fps = st.cap.get_clip_last_seconds(cfg.CLIP_SECONDS)
                has_full_clip = (len(clip_frames) >= int(cfg.CLIP_SECONDS * cfg.STORE_FPS * 0.7))

                should_trigger = (
                    has_full_clip
                    and (st.detection_score >= cfg.SCORE_MAX)
                    and (now - st.last_trigger_time > cfg.COOLDOWN_TRIGGER_SEC)
                )
                if should_trigger:
                    st.last_trigger_time = now
                    save_job(st, "trigger", clip_frames, start_ts, end_ts, write_fps)

                if has_full_clip and (now >= st.next_random_time):
                    if cfg.RANDOM_ALLOW_PERSON or (not st.last_person):
                        save_job(st, "random", clip_frames, start_ts, end_ts, write_fps)
                    st.next_random_time = now + jittered_interval(cfg.RANDOM_CLIP_INTERVAL_SEC, cfg.RANDOM_JITTER_FRAC)

                if cfg.SHOW_WINDOWS:
                    if results is not None and cfg.SHOW_PLOTTED_BOXES:
                        disp = results[0].plot()
                    else:
                        disp = frame
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
        log("[System] Stopped.")


if __name__ == "__main__":
    main()
