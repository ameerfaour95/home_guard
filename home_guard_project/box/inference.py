"""Headless inference (production) runner for a collector box.

Launched by run_collector.sh when box.yaml has ``mode: inference``:

    python -u -m home_guard_project.box.inference

Pipeline, per camera: YOLO gate (is anyone/anything there?) -> only inside the
owner's alert time window, buffer a few frames -> a VLM backend (GPT-4V now,
pluggable) returns {summary, alert_command, alert_reason} -> the alert is sent
to the owner over the configured channel (Telegram by default). The VLM call
runs off the capture loop, and a per-camera cooldown bounds cost and spam.

Design notes (N150): YOLO already saturates the CPU, so the gate uses a small
model, the VLM runs in the cloud on a worker thread, and we send few frames.
No cv2 windows. Logs to stdout. Exits non-zero on a fatal error so the runner
restarts it. The process is killed with TerminateProcess, so nothing here
relies on cleanup handlers for correctness.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("box.inference")

# Classes that open the gate (match the alert policy: person / vehicle).
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
TRIGGER_CLASSES = PERSON_CLASSES | VEHICLE_CLASSES


# ----------------------------------------------------------------------------
# Pure helpers (no heavy imports) - unit-tested without a camera or a model.
# ----------------------------------------------------------------------------
def in_alert_window(hour: int, start_hour: int, end_hour: int) -> bool:
    """True if *hour* is inside [start, end). start == end means 24h."""
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour  # wraps midnight


def parse_vlm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the VLM's JSON object. Tolerates leading/trailing text."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(raw[start:end + 1])
            except ValueError:
                return None
        return None


def apply_policy_override(parsed: Dict[str, Any], in_window: bool, person: bool, vehicle: bool) -> Dict[str, Any]:
    """Enforce the alert policy in code, redundant with the prompt.

    Outside the window -> [none]. Inside the window with a person/vehicle but a
    [none] verdict -> upgrade to [send_message].
    """
    out = dict(parsed)
    cmd = out.get("alert_command", "[none]")
    if not in_window:
        out["alert_command"] = "[none]"
        return out
    if (person or vehicle) and cmd == "[none]":
        out["alert_command"] = "[send_message]"
        out["alert_reason"] = out.get("alert_reason") or "Person/vehicle in the alert window."
    return out


def build_prompt(camera_name: str, t_sec: int, local_time_str: str, start_hour: int, end_hour: int) -> str:
    window = f"{start_hour:02d}:00-{end_hour:02d}:00"
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
- Alert window: {window}. Current time: {local_time_str}.
- If OUTSIDE window: alert_command MUST be "[none]".
- If INSIDE window:
  * Person/Car visible? alert_command MUST be "[send_message]" (minimum).
  * Suspicious (forced entry, hiding, loitering)? alert_command MUST be "[call_owner]".
""".strip()


@dataclass(frozen=True)
class AlertSettings:
    alert_start_hour: int = 0
    alert_end_hour: int = 0          # 0,0 -> always in window (handy for testing)
    cooldown_sec: float = 120.0      # min seconds between VLM calls per camera
    clip_frames: int = 5             # frames sent to the VLM per escalation
    frame_interval_sec: float = 1.0  # spacing of buffered frames
    model: str = "yolo11s.pt"        # the model shipped in the bundle (avoids a download on the box)
    conf: float = 0.4
    vlm_backend: str = "gpt"
    vlm_model: str = "gpt-4o"
    alert_channel: str = "telegram"  # telegram | twilio | both
    dry_run: bool = False

    @classmethod
    def from_box_settings(cls, s: Dict[str, Any]) -> "AlertSettings":
        g = s.get
        return cls(
            alert_start_hour=int(g("alert_start_hour", 0)),
            alert_end_hour=int(g("alert_end_hour", 0)),
            cooldown_sec=float(g("alert_cooldown_sec", 120.0)),
            clip_frames=int(g("alert_clip_frames", 5)),
            frame_interval_sec=float(g("alert_frame_interval_sec", 1.0)),
            model=str(g("inference_yolo_model", "yolo11s.pt")),
            conf=float(g("inference_conf", 0.4)),
            vlm_backend=str(g("vlm_backend", "gpt")),
            vlm_model=str(g("vlm_model", "gpt-4o")),
            alert_channel=str(g("alert_channel", "telegram")),
            dry_run=bool(g("notify_dry_run", False)),
        )


def frame_to_jpeg_bytes(frame_bgr: Any) -> bytes:
    """JPEG-encode a BGR frame to raw bytes. Imports cv2 lazily."""
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else b""


def frame_to_jpeg_b64(frame_bgr: Any) -> str:
    """JPEG-encode a BGR frame to base64. Imports cv2 lazily."""
    data = frame_to_jpeg_bytes(frame_bgr)
    return base64.b64encode(data).decode("utf-8") if data else ""


# ----------------------------------------------------------------------------
# VLM backends
# ----------------------------------------------------------------------------
class NullBackend:
    """Used when no real backend is configured (e.g. missing key, or dry-run).

    Returns a benign [none] so the pipeline runs end to end without a cloud call.
    """

    def analyze(self, frames_bgr: List[Any], camera_name: str, t_sec: int,
                start_hour: int, end_hour: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        parsed = {"summary": "(no VLM backend configured)", "alert_command": "[none]", "alert_reason": ""}
        return json.dumps(parsed), parsed


class GptBackend:
    """GPT-4V backend. Imports the OpenAI client lazily."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        from openai import OpenAI  # noqa: PLC0415

        # Use the OS trust store so the call still works where TLS is
        # intercepted (a laptop antivirus, a corporate/home MITM proxy); the
        # default certifi bundle would not trust those roots.
        http_client = None
        try:
            import ssl  # noqa: PLC0415
            import httpx  # noqa: PLC0415

            http_client = httpx.Client(verify=ssl.create_default_context())
        except Exception:  # noqa: BLE001
            http_client = None
        self._client = OpenAI(api_key=api_key, http_client=http_client) if http_client else OpenAI(api_key=api_key)
        self._model = model

    def analyze(self, frames_bgr: List[Any], camera_name: str, t_sec: int,
                start_hour: int, end_hour: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = build_prompt(camera_name, t_sec, datetime.now().strftime("%H:%M:%S"), start_hour, end_hour)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for fr in frames_bgr:
            b64 = frame_to_jpeg_b64(fr)
            if b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        return raw, parse_vlm_json(raw)


def make_backend(settings: AlertSettings, env: Dict[str, str]):
    """Pick a VLM backend from settings; fall back to NullBackend when a real
    one cannot be built (missing key, dry-run, or unknown name)."""
    if settings.dry_run:
        log.info("dry_run on: using NullBackend (no VLM calls).")
        return NullBackend()
    if settings.vlm_backend == "gpt":
        key = env.get("OPENAI_API_KEY", "")
        if not key:
            log.warning("vlm_backend=gpt but OPENAI_API_KEY is missing; using NullBackend.")
            return NullBackend()
        try:
            return GptBackend(key, settings.vlm_model)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not create GptBackend (%s); using NullBackend.", exc)
            return NullBackend()
    log.warning("Unknown vlm_backend '%s'; using NullBackend.", settings.vlm_backend)
    return NullBackend()


# ----------------------------------------------------------------------------
# Alert dispatch (channel-agnostic)
# ----------------------------------------------------------------------------
def dispatch_alert(box_settings: Dict[str, Any], env: Dict[str, str],
                   command: str, summary: str, reason: str,
                   image: Optional[bytes] = None) -> Dict[str, Any]:
    """Send the alert over the configured channel(s). Never raises.

    *image* (JPEG bytes) is the camera snapshot; Telegram sends it as a photo.
    """
    channel = str(box_settings.get("alert_channel", "telegram"))
    results: Dict[str, Any] = {"channel": channel}
    try:
        if channel in ("telegram", "both"):
            from . import telegram_notify  # noqa: PLC0415

            cfg = telegram_notify.load_telegram_config(box_settings, env)
            results["telegram"] = telegram_notify.notify(cfg, command, summary, reason, image=image)
        if channel in ("twilio", "both"):
            from . import notify as twilio_notify  # noqa: PLC0415

            cfg = twilio_notify.load_notify_config(box_settings, env)
            results["twilio"] = twilio_notify.notify(cfg, command, summary, reason)
    except Exception as exc:  # noqa: BLE001
        log.warning("Alert dispatch error: %s", exc)
        results["error"] = str(exc)
    return results


# ----------------------------------------------------------------------------
# Detection helpers (operate on an ultralytics result; no import needed here)
# ----------------------------------------------------------------------------
def detect_trigger(result) -> Tuple[bool, bool, List[str]]:
    """Return (person_present, vehicle_present, labels) from a YOLO result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False, False, []
    names = getattr(result, "names", {})
    person = vehicle = False
    labels: set = set()
    for b in boxes:
        cls_id = int(b.cls[0])
        name = names.get(cls_id, str(cls_id))
        if name in PERSON_CLASSES:
            person = True
        if name in VEHICLE_CLASSES:
            vehicle = True
        if name in TRIGGER_CLASSES:
            labels.add(name)
    return person, vehicle, sorted(labels)


# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
class _Stream:
    """Threaded RTSP reader keeping only the latest frame (drops stale frames)."""

    def __init__(self, name: str, url: str) -> None:
        import cv2  # noqa: PLC0415

        self.name = name
        self.url = url
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        import cv2  # noqa: PLC0415

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.5)
                self._cap.release()
                self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def _worker(backend, box_settings, env, settings: AlertSettings,
            camera_name: str, frames: List[Any]) -> None:
    """Run the VLM call + dispatch off the capture loop. Never raises out."""
    try:
        now = datetime.now()
        in_win = in_alert_window(now.hour, settings.alert_start_hour, settings.alert_end_hour)
        if not in_win:
            return
        raw, parsed = backend.analyze(frames, camera_name, int(time.time()),
                                      settings.alert_start_hour, settings.alert_end_hour)
        if not parsed:
            log.warning("[%s] VLM returned unparseable output: %s", camera_name, raw[:200])
            return
        # person/vehicle already true at the gate, so pass True for the override's benefit.
        parsed = apply_policy_override(parsed, in_win, person=True, vehicle=True)
        cmd = parsed.get("alert_command", "[none]")
        summary = str(parsed.get("summary", ""))
        reason = str(parsed.get("alert_reason", ""))
        log.info("[%s] verdict=%s summary=%s", camera_name, cmd, summary)
        if cmd in ("[send_message]", "[call_owner]"):
            # Attach the most recent frame of the clip as the alert snapshot.
            image = b""
            try:
                image = frame_to_jpeg_bytes(frames[-1]) if frames else b""
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] could not encode snapshot: %s", camera_name, exc)
            res = dispatch_alert(box_settings, env, cmd, f"{camera_name}: {summary}", reason,
                                 image=image or None)
            log.info("[%s] alert dispatched: %s", camera_name, res)
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] worker error: %s", camera_name, exc)


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()  # api_key.env at the repo root
    except Exception:  # noqa: BLE001
        pass
    env = dict(os.environ)

    from .boxconfig import load_box_settings  # noqa: PLC0415
    from ..data_collection.config import load_config  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    box_settings = load_box_settings()
    settings = AlertSettings.from_box_settings(box_settings)
    cam_cfg = load_config()

    log.info("Inference mode starting. window=%02d:00-%02d:00 channel=%s backend=%s dry_run=%s",
             settings.alert_start_hour, settings.alert_end_hour, settings.alert_channel,
             settings.vlm_backend, settings.dry_run)

    backend = make_backend(settings, env)
    model = YOLO(settings.model)

    # Camera sub-stream URLs from the data_collection config.
    cameras: Dict[str, str] = dict(getattr(cam_cfg, "CAMERAS", {}) or {})
    if not cameras:
        log.error("No cameras configured (cameras.yaml). Nothing to watch; exiting.")
        return 1
    streams = {name: _Stream(name, url) for name, url in cameras.items()}
    buffers: Dict[str, deque] = {name: deque(maxlen=settings.clip_frames) for name in cameras}
    last_buf_ts: Dict[str, float] = {name: 0.0 for name in cameras}
    last_alert_ts: Dict[str, float] = {name: 0.0 for name in cameras}
    worker = {"t": None}  # single in-flight VLM call across cameras (N150 budget)

    log.info("Watching %d camera(s): %s", len(cameras), ", ".join(cameras))
    while True:
        now_ts = time.time()
        for name in cameras:
            frame = streams[name].read()
            if frame is None:
                continue
            # Maintain a rolling buffer, one frame every frame_interval_sec.
            if now_ts - last_buf_ts[name] >= settings.frame_interval_sec:
                buffers[name].append(frame)
                last_buf_ts[name] = now_ts

            if now_ts - last_alert_ts[name] < settings.cooldown_sec:
                continue
            now = datetime.now()
            if not in_alert_window(now.hour, settings.alert_start_hour, settings.alert_end_hour):
                continue
            if worker["t"] is not None and worker["t"].is_alive():
                continue  # a VLM call is already running

            results = model.predict(frame, conf=settings.conf, verbose=False)
            person, vehicle, labels = detect_trigger(results[0]) if results else (False, False, [])
            if not (person or vehicle):
                continue
            if len(buffers[name]) == 0:
                buffers[name].append(frame)
            frames = list(buffers[name])
            last_alert_ts[name] = now_ts
            log.info("[%s] escalating (labels=%s), calling VLM", name, labels)
            t = threading.Thread(target=_worker,
                                 args=(backend, box_settings, env, settings, name, frames),
                                 daemon=True)
            t.start()
            worker["t"] = t
        time.sleep(0.05)


def main() -> None:
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("box.inference").exception("Fatal: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
