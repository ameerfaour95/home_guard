from __future__ import annotations

import os
import base64
import cv2
import math
import time
import json
import logging
import threading
import ctypes
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# OS / ENV SETUP
# -----------------------------
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# Force FFmpeg to use TCP and low latency internally as a backup
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer"

load_dotenv("api_key.env")

# -----------------------------
# LOGGING
# -----------------------------
logger = logging.getLogger("security_escalation")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

# COCO animals (YOLOv8n COCO80)
ANIMAL_LABELS = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class Settings:
    camera_name: str = "door"
    conf: float = 0.25
    move_pixels: float = 25.0
    alert_start_hour: int = 20
    alert_end_hour: int = 6
    llm_window_sec: int = 5
    llm_fps: int = 2
    print_every_sec: int = 5
    llm_cooldown_sec: int = 30
    force_100_zoom: bool = True


# -----------------------------
# THREADED CAMERA CLASS (New Fix)
# -----------------------------
class ThreadedCamera:
    """
    Reads frames in a separate thread to prevent RTSP buffer overflows
    when main processing loop is slower than camera FPS.
    """
    def __init__(self, src):
        self.src = src
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        # Try to set small buffer (backend dependent)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.status, self.frame = self.capture.read()
        self.lock = threading.Lock()
        self.running = True
        
        # Start the thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    with self.lock:
                        self.status = status
                        self.frame = frame
                else:
                    # Optional: Add reconnection logic here if needed
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            # Return a copy to ensure thread safety
            if self.frame is not None and self.status:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()
        
    def isOpened(self):
        return self.capture.isOpened()


# -----------------------------
# HELPERS
# -----------------------------
def is_in_alert_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    h = now.hour
    if start_hour <= end_hour:
        return start_hour <= h < end_hour
    return h >= start_hour or h < end_hour

def format_window(start_hour: int, end_hour: int) -> str:
    return f"{start_hour:02d}:00–{end_hour:02d}:00"

def detect_human(result) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False
    for b in result.boxes:
        cls_id = int(b.cls[0])
        if result.names.get(cls_id) == "person":
            return True
    return False

def detect_person_and_animals(result) -> Tuple[bool, List[str]]:
    if result.boxes is None or len(result.boxes) == 0:
        return False, []
    has_person = False
    animals = set()
    for b in result.boxes:
        cls_id = int(b.cls[0])
        name = result.names.get(cls_id, str(cls_id))
        if name == "person":
            has_person = True
        if name in ANIMAL_LABELS:
            animals.add(name)
    return has_person, sorted(animals)

def detect_moving_car(result, last_center_by_id: Dict[int, Tuple[float, float]], move_pixels: float) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False
    if getattr(result.boxes, "id", None) is None:
        return False

    moving_car_found = False
    for b in result.boxes:
        cls_id = int(b.cls[0])
        name = result.names.get(cls_id, str(cls_id))
        if name != "car":
            continue
        
        track_id = int(b.id[0])
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if track_id in last_center_by_id:
            px, py = last_center_by_id[track_id]
            dist = math.hypot(cx - px, cy - py)
            if dist > move_pixels:
                moving_car_found = True
        
        last_center_by_id[track_id] = (cx, cy)
    return moving_car_found

# -----------------------------
# LLM LOGIC
# -----------------------------
def build_gpt_prompt(*, camera_name: str, t_sec: int, local_time_str: str, alert_start_hour: int, alert_end_hour: int) -> str:
    window_str = format_window(alert_start_hour, alert_end_hour)
    return f"""
You are a security camera assistant. You receive MULTIPLE sequential frames (~5 seconds) from camera "{camera_name}".
Treat them as a SHORT VIDEO CLIP.

Return EXACTLY ONE STRICT JSON OBJECT (NOT an array) and NOTHING else:
{{
  "time_sec": {t_sec},
  "camera": "{camera_name}",
  "summary": "<one sentence describing the ENTIRE clip>",
  "alert_command": "[none]" or "[call_owner]" or "[send_message]",
  "alert_reason": "<short reason, or empty string if alert_command is [none]>"
}}

HARD RULES:
- Single JSON Object only.
- Summary describes the 5-sec clip activity.
- If person detected: describe appearance and action.

Alert policy:
- Alert window: {window_str}. Current time: {local_time_str}.
- If OUTSIDE window: alert_command MUST be "[none]".
- If INSIDE window:
  * Person/Car visible? alert_command MUST be "[send_message]" (minimum).
  * Suspicious (forced entry, hiding, loitering)? alert_command MUST be "[call_owner]".
""".strip()

def _frame_to_jpeg_b64(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok: return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def call_gpt_on_frames(*, client: OpenAI, model_name: str, settings: Settings, t_sec: int, frames_bgr: List) -> Tuple[str, Optional[dict]]:
    local_time_str = datetime.now().strftime("%H:%M:%S")
    prompt = build_gpt_prompt(
        camera_name=settings.camera_name,
        t_sec=t_sec,
        local_time_str=local_time_str,
        alert_start_hour=settings.alert_start_hour,
        alert_end_hour=settings.alert_end_hour,
    )
    
    content = [{"type": "text", "text": prompt}]
    for fr in frames_bgr:
        b64 = _frame_to_jpeg_b64(fr)
        if b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        parsed = json.loads(raw)
        return raw, parsed
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return str(e), None

def activate_llm(*, person: bool, moving_car: bool, now: datetime, settings: Settings) -> bool:
    if not is_in_alert_window(now, settings.alert_start_hour, settings.alert_end_hour):
        return False
    return person or moving_car

# -----------------------------
# MAIN LOOP
# -----------------------------
def run(source: str | int, *, settings: Settings, openai_api_key: str, llm_model: str = "gpt-4o-mini"):
    logger.info(f"START run() camera={settings.camera_name} source={source}")

    client = OpenAI(api_key=openai_api_key)
    yolo = YOLO("yolov8n.pt")  # Consider 'yolov8s.pt' if GPU allows for better accuracy
    
    # --- USE THREADED CAMERA ---
    cap = ThreadedCamera(source)
    time.sleep(1.0) # Let buffer fill slightly
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    last_center_by_id = {}
    start_ts = time.time()
    
    # LLM Buffers
    frame_buffer = []
    buffer_start_ts = time.time()
    sample_interval = 1.0 / max(1, settings.llm_fps)
    last_sample_ts = 0.0
    
    # Rate limits
    last_llm_ts = 0.0
    last_person_print_ts = 0.0
    last_animal_print_ts = 0.0

    frame_idx = 0
    window_name = f"preview:{settings.camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False

    while True:
        # 1. Read latest frame (Thread Safe)
        ret, frame = cap.read()
        if not ret or frame is None:
            # If tracking is lost momentarily, just skip loop iter
            time.sleep(0.01)
            continue

        frame_idx += 1
        now = datetime.now()
        t_sec = int(time.time() - start_ts)
        now_ts = time.time()

        # 2. YOLO Tracking
        # 'persist=True' is crucial for ID tracking
        results = yolo.track(
            frame, 
            conf=settings.conf, 
            persist=True, 
            verbose=False,
            tracker="bytetrack.yaml" 
        )
        result = results[0]

        # 3. Logic & Signals
        has_person_any, animals_any = detect_person_and_animals(result)
        moving_car = detect_moving_car(result, last_center_by_id, settings.move_pixels)
        person = detect_human(result)

        # Print logs
        if has_person_any and (now_ts - last_person_print_ts >= settings.print_every_sec):
            print(f"{settings.camera_name}: person detected")
            last_person_print_ts = now_ts
        
        if animals_any and (now_ts - last_animal_print_ts >= settings.print_every_sec):
            print(f"{settings.camera_name}: animals: {animals_any}")
            last_animal_print_ts = now_ts

        # 4. Maintain Buffer
        if (now_ts - last_sample_ts) >= sample_interval:
            frame_buffer.append(frame.copy())
            last_sample_ts = now_ts
        
        # Trim buffer to window size
        if (now_ts - buffer_start_ts) >= settings.llm_window_sec:
            keep = int(settings.llm_window_sec * settings.llm_fps)
            frame_buffer = frame_buffer[-keep:]
            buffer_start_ts = now_ts # slight drift acceptible for this logic

        # 5. Escalation
        should_escalate = activate_llm(person=person, moving_car=moving_car, now=now, settings=settings)
        min_frames = int(settings.llm_window_sec * settings.llm_fps)

        if should_escalate and len(frame_buffer) >= min_frames:
            if (now_ts - last_llm_ts) >= settings.llm_cooldown_sec:
                last_llm_ts = now_ts
                
                # Double check window policy before sending
                in_window = is_in_alert_window(now, settings.alert_start_hour, settings.alert_end_hour)
                
                print(f"[ESCALATE] Sending frames to GPT... (Time: {now.strftime('%H:%M:%S')})")
                frames_for_llm = frame_buffer[-min_frames:]
                
                # Run LLM (Blocking for simplicity, could be threaded too)
                raw, parsed = call_gpt_on_frames(
                    client=client,
                    model_name=llm_model,
                    settings=settings,
                    t_sec=t_sec,
                    frames_bgr=frames_for_llm
                )

                if parsed:
                    # Policy enforcement override
                    if in_window and (person or moving_car) and parsed.get("alert_command") == "[none]":
                        parsed["alert_command"] = "[send_message]"
                        parsed["alert_reason"] = "Forced override: Person/Car in restricted window."
                    
                    print(json.dumps(parsed, indent=2))
                else:
                    print(f"[ERROR] LLM Raw: {raw}")

                frame_buffer.clear()

        # 6. Visualization
        annotated = result.plot()
        
        if settings.force_100_zoom and not window_initialized:
            h, w = annotated.shape[:2]
            cv2.resizeWindow(window_name, w, h)
            window_initialized = True

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("STOP run()")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    settings = Settings(
        camera_name="door",
        conf=0.25,
        move_pixels=25,
        alert_start_hour=18, # Adjusted for testing (current time is ~18:45)
        alert_end_hour=6,
        llm_window_sec=5,
        llm_fps=2,
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in api_key.env")

    # RTSP URL Handling
    base_url = os.getenv("RTSP_DOOR", "rtsp://admin:amer1967%40@192.168.68.105:554/unicast/c6/s0/live")
    
    # FORCE TCP IN URL (Crucial for Reolink/H.265 stability)
    if "?" not in base_url:
        rtsp_url = f"{base_url}?rtsp_transport=tcp"
    else:
        rtsp_url = f"{base_url}&rtsp_transport=tcp"

    run(
        source=rtsp_url,
        settings=settings,
        llm_model="gpt-4o-mini",
        openai_api_key=api_key,
    )