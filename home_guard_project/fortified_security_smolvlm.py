"""
Fortified Security (VLM) — FULL SCRIPT (10s clip via real MP4, “as-trained” style)

Goal (Option 1):
- Buffer the last 10 seconds from RTSP
- When triggered, write those frames to a temporary .mp4
- Feed the MP4 path to the VLM processor: videos=[clip_path]
  (no custom frame sampling, no frames_indices / video_metadata)

Stability:
- RTSP watchdog + reconnect to avoid freezes when OpenCV/FFmpeg stalls
- Optional YOLO throttling on CPU so the pipeline doesn’t starve

Output:
- EXACTLY one JSON object printed (3 keys) after extraction + sanitizer.

IMPORTANT:
- You previously printed your RTSP credentials publicly. Consider changing them.
"""

from __future__ import annotations

import os
import cv2
import json
import re
import time
import queue
import torch
import datetime
import threading
import numpy as np
import tempfile
from typing import List, Callable, Optional, Dict, Any
from PIL import Image  # kept (not used in MP4-path mode, but harmless)

from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText


# ==========================================================
# 0) PROMPT
# ==========================================================
PROMPT = """
You are a security camera assistant watching a {camera_name} camera.

Task:
Describe what happens in the video

Alert policy:
- If no person is visible: alert_command = "[none]"
- If any person is visible: alert_command = "[send_message]"
- If suspicious behavior is visible alert_command = "[call_owner]"

Output format:
{{
  "summary": "<one sentence describing the whole clip>",
  "alert_reason": "<short reason>",
  "alert_command": "[none] or [send_message] or [call_owner]"
}}
"""


# ==========================================================
# 1) CONFIG
# ==========================================================
class Config:
    # --- Models ---
    YOLO_MODEL = "yolov8n.pt"
    VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # --- Stream ---
    DEFAULT_RTSP = "rtsp://admin:amer1967%40@192.168.68.110:554/unicast/c6/s1/live"
    FPS_CAM = 20

    # --- Buffering (10 seconds) ---
    BUFFER_DURATION = 10
    BUFFER_SIZE = FPS_CAM * BUFFER_DURATION  # 150 at 15fps

    # --- Detection Hysteresis / Trigger ---
    SCORE_MAX = 30
    SCORE_PENALTY = 1.0
    SCORE_REWARD = 1.0
    COOLDOWN = 10  # seconds between VLM calls

    # --- Schedule ---
    SCHEDULE_ENABLED = False
    TIME_START = "23:00"
    TIME_END = "07:00"

    # --- RTSP/FFmpeg stability ---
    OPENCV_FFMPEG_CAPTURE_OPTIONS = (
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer"
    )
    OPEN_TIMEOUT_MSEC = 5000
    READ_TIMEOUT_MSEC = 5000
    FREEZE_RECONNECT_AFTER_SEC = 3.0
    RECONNECT_BACKOFF_START = 1.0
    RECONNECT_BACKOFF_MAX = 10.0

    # --- Performance ---
    # If CPU: YOLO can be heavy; run every N frames.
    YOLO_EVERY_N_FRAMES_CPU = 2  # 1 = every frame


# ==========================================================
# 2) UTILITIES
# ==========================================================
def is_within_schedule() -> bool:
    if not Config.SCHEDULE_ENABLED:
        return True

    now = datetime.datetime.now().time()
    start = datetime.datetime.strptime(Config.TIME_START, "%H:%M").time()
    end = datetime.datetime.strptime(Config.TIME_END, "%H:%M").time()

    if start < end:
        return start <= now <= end
    return now >= start or now <= end


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    # Quick path: entire string is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first { ... } block
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def sanitize_report(obj: Dict[str, Any], person_seen: bool) -> Dict[str, str]:
    summary = str(obj.get("summary", "")).strip()
    alert_reason = str(obj.get("alert_reason", "")).strip()
    alert_command = str(obj.get("alert_command", "")).strip()

    cmd_low = alert_command.lower()
    if "call" in cmd_low:
        alert_command = "[call_owner]"
    elif "send" in cmd_low or "message" in cmd_low:
        alert_command = "[send_message]"

    if alert_command not in ("[call_owner]", "[send_message]"):
        alert_command = "[send_message]"

    # Policy: if person is visible -> at least send_message (unless call_owner)
    if person_seen and alert_command != "[call_owner]":
        alert_command = "[send_message]"

    if not summary:
        summary = "Activity detected at the door camera."
    if not alert_reason:
        alert_reason = "Detection triggered."

    return {
        "summary": summary,
        "alert_reason": alert_reason,
        "alert_command": alert_command,
    }


def write_mp4_clip(frames: List[np.ndarray], fps: float) -> str:
    """
    Write frames (BGR np.ndarray) to a temporary mp4 file and return path.
    Windows-safe: we create a temp filename via mkstemp and close fd first.
    """
    if not frames:
        raise ValueError("No frames to write.")

    h, w = frames[0].shape[:2]

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter for mp4.")

    for f in frames:
        if f is None:
            continue
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)

    writer.release()
    return path


# ==========================================================
# 3) THREADED VIDEO CAPTURE (watchdog + reconnect)
# ==========================================================
class VideoCaptureThread:
    def __init__(self, src: str = Config.DEFAULT_RTSP):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", Config.OPENCV_FFMPEG_CAPTURE_OPTIONS)

        self.src = src
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()

        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.running = True

        self.last_frame_ts = 0.0
        self.reconnect_backoff = float(Config.RECONNECT_BACKOFF_START)

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

            # Supported on some OpenCV builds; ignored on others (ok).
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(Config.OPEN_TIMEOUT_MSEC))
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(Config.READ_TIMEOUT_MSEC))
            except Exception:
                pass

            self.capture = cap
            self.last_frame_ts = time.time()

    def _reconnect(self):
        time.sleep(self.reconnect_backoff)
        self.reconnect_backoff = min(self.reconnect_backoff * 1.5, float(Config.RECONNECT_BACKOFF_MAX))
        self._open_capture()

    def _reader(self):
        while self.running:
            with self.lock:
                cap = self.capture

            if cap is None or not cap.isOpened():
                self._reconnect()
                continue

            ret, frame = cap.read()
            now = time.time()

            if not ret or frame is None:
                if now - self.last_frame_ts > float(Config.FREEZE_RECONNECT_AFTER_SEC):
                    self._reconnect()
                else:
                    time.sleep(0.02)
                continue

            # Good frame
            self.last_frame_ts = now
            self.reconnect_backoff = float(Config.RECONNECT_BACKOFF_START)

            # Keep only latest frame
            if not self.q.empty():
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def read(self):
        try:
            return True, self.q.get(timeout=1)
        except queue.Empty:
            return False, None

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


# ==========================================================
# 4) SMART SECURITY AGENT (VLM, MP4-path input)
# ==========================================================
class SmartSecurityAgent:
    def __init__(self):
        print(f"[System] Loading VLM ({Config.VLM_MODEL_ID}) on {Config.DEVICE} with {Config.DTYPE}...")

        # use_fast=False to avoid the LANCZOS/BICUBIC warning and match reference behavior
        self.processor = AutoProcessor.from_pretrained(Config.VLM_MODEL_ID, use_fast=False)

        self.model = AutoModelForImageTextToText.from_pretrained(
            Config.VLM_MODEL_ID,
            dtype=Config.DTYPE,
            low_cpu_mem_usage=True,
        ).to(Config.DEVICE)

        self.is_busy = False
        print("[System] VLM Loaded Successfully.")

    def analyze_async(
        self,
        frames: List[np.ndarray],
        person_seen: bool,
        callback: Callable[[str], None],
        camera_name: str = "door",
    ):
        if self.is_busy:
            return
        self.is_busy = True
        t = threading.Thread(
            target=self._run_inference,
            args=(frames, person_seen, callback, camera_name),
            daemon=True,
        )
        t.start()

    def _run_inference(
        self,
        frames: List[np.ndarray],
        person_seen: bool,
        callback: Callable[[str], None],
        camera_name: str,
    ):
        clip_path: Optional[str] = None
        try:
            # 1) Write the buffered 10s frames to a real mp4
            clip_path = write_mp4_clip(frames, fps=float(Config.FPS_CAM))

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "path": clip_path},   # <-- FIX (as-trained)
                    {"type": "text", "text": PROMPT},
                ],
            }]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

            if Config.DEVICE == "cuda":
                for k, v in list(inputs.items()):
                    if hasattr(v, "is_floating_point") and v.is_floating_point():
                        inputs[k] = v.to(Config.DTYPE)
            
            gen = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

            # Decode ONLY newly generated tokens
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = gen[:, prompt_len:]
            out_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

            obj = extract_first_json_object(out_text) or {}
            strict = sanitize_report(obj, person_seen=person_seen)
            callback(json.dumps(strict, ensure_ascii=False))

        except Exception as e:
            print(f"[Error] Inference failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_busy = False
            if clip_path is not None:
                try:
                    os.remove(clip_path)
                except Exception:
                    pass
            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()


# ==========================================================
# 5) MAIN
# ==========================================================
def main():
    print("[System] Stream Thread Starting. Connecting to camera...")
    cap = VideoCaptureThread()
    # Wait for first frames
    attempts = 0
    got_first = False
    while attempts < 12:
        ret, frame = cap.read()
        if ret and frame is not None:
            got_first = True
            break
        attempts += 1
        print(f"[System] Waiting for stream... ({attempts}/12)")
        time.sleep(0.5)

    if not got_first:
        print("[Error] Could not receive frames. Check RTSP URL / network / password.")
        cap.release()
        return

    if not cap.isOpened():
        print("[Error] Could not open RTSP stream.")
        cap.release()
        return

    print("[System] Connection established.")
    detector = YOLO(Config.YOLO_MODEL)
    agent = SmartSecurityAgent()

    frame_buffer: List[np.ndarray] = []
    detection_score = 0.0
    last_trigger_time = 0.0

    last_results = None
    last_person_detected = False

    def on_vlm_result(text_json: str):
        nonlocal last_trigger_time
        print(f"\n>>> [SECURITY REPORT]: {text_json}\n")
        last_trigger_time = time.time()

    print("[System] Main Loop Started. Press 'q' to quit.")

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Buffer last 10 seconds
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > Config.BUFFER_SIZE:
            frame_buffer.pop(0)

        frame_i += 1

        # YOLO throttling on CPU
        run_yolo = True
        if Config.DEVICE == "cpu" and Config.YOLO_EVERY_N_FRAMES_CPU > 1:
            run_yolo = (frame_i % Config.YOLO_EVERY_N_FRAMES_CPU == 0)

        if run_yolo:
            results = detector(frame, classes=[0], verbose=False)
            person_detected = len(results[0].boxes) > 0
            last_results = results
            last_person_detected = person_detected
        else:
            results = last_results
            person_detected = last_person_detected

        # If YOLO hasn't run yet, display raw
        if results is None:
            display_frame = cv2.resize(frame, (1024, 576))
            cv2.imshow("Security Agent", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Hysteresis score update
        if person_detected:
            detection_score = min(detection_score + Config.SCORE_REWARD, Config.SCORE_MAX)
        else:
            detection_score = max(detection_score - Config.SCORE_PENALTY, 0.0)

        schedule_active = is_within_schedule()

        # Require a full 10-second buffer before triggering
        has_full_clip = len(frame_buffer) >= Config.BUFFER_SIZE

        should_trigger = (
            has_full_clip and
            detection_score >= Config.SCORE_MAX and
            (time.time() - last_trigger_time > Config.COOLDOWN) and
            schedule_active and
            not agent.is_busy
        )

        if should_trigger:
            print("[System] Trigger matched. Analyzing 10s clip...")
            last_trigger_time = time.time()

            clip_frames = list(frame_buffer)

            agent.analyze_async(
                frames=clip_frames,
                person_seen=person_detected,
                callback=on_vlm_result,
                camera_name="door",
            )

        # Display annotated frame
        annotated_frame = results[0].plot()
        display_frame = cv2.resize(annotated_frame, (1024, 576))
        cv2.imshow("Security Agent", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
