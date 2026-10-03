"""Headless inference (production) runner for a collector box.

Launched by run_collector.sh when box.yaml has ``mode: inference``:

    python -u -m home_guard_project.box.inference

Pipeline, per camera: YOLO gate (is anyone/anything there?) -> only inside the
owner's alert time window, buffer a few frames -> a VLM backend (GPT-4V now,
pluggable) returns {summary, alert_command, alert_reason} -> the alert is sent
to the owner over the configured channel (Telegram by default). The VLM call
runs off the capture loop, and a per-camera cooldown bounds cost and spam.
A vehicle that has not moved since the camera's previous look does not open
the gate (a parked car is not an event); a person always does.

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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# Bumped whenever the prompt or the answer's schema changes, so training records can be told apart.
PROMPT_VERSION = "2026-10-03.tagged-style-label"

# The three labels the model gives a scene, and what the box does with each. The owner
# chose them (this is also what a student model will be trained to answer):
#   normal      ordinary activity -> a message
#   suspicious  worth a look      -> a message marked suspicious (more may follow later)
#   escalation  danger or a crime -> the urgent alert
LABELS = ("normal", "suspicious", "escalation")
LABEL_COMMANDS = {"normal": "[send_message]", "suspicious": "[send_message]", "escalation": "[call_owner]"}

# The answer's shape, enforced on the model (structured output) and checked on the way back.
VLM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "label": {"type": "string", "enum": list(LABELS)},
        "people": {"type": "integer"},
        "vehicle_moving": {"type": "boolean"},
    },
    "required": ["summary", "label", "people", "vehicle_moving"],
    "additionalProperties": False,
}


def label_of(parsed: Optional[Dict[str, Any]]) -> str:
    """The scene's label from the model's answer; ``normal`` when it gave none or an unknown one."""
    label = str((parsed or {}).get("label") or "").strip().lower()
    return label if label in LABELS else "normal"


def alert_summary(label: str, summary: str) -> str:
    """The sentence the owner reads: a suspicious or escalated scene says so first."""
    if label == "suspicious":
        return f"Suspicious: {summary}"
    if label == "escalation":
        return f"Escalation: {summary}"
    return summary
VLM_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "camera_report", "strict": True, "schema": VLM_SCHEMA},
}


def build_prompt(camera_name: str, t_sec: int, local_time_str: str, start_hour: int, end_hour: int) -> str:
    # The summary is written the way our taggers wrote the training descriptions
    # (tagging/*/analysis_output/vlm_training.jsonl): what happens, in order, with what
    # people wear and hold, and "No special activity." for an empty scene. The examples
    # are real tagged descriptions. The label replaces the taggers' "[alert]" mark and
    # decides what the box does (LABEL_COMMANDS); people/vehicle_moving decide whether
    # anything is sent at all (vlm_confirms).
    return f"""
You are the eyes of a home security system. These are sequential frames (one short clip of a few
seconds) from the homeowner's own camera "{camera_name}", local time {local_time_str}.

Write "summary": what happens in the clip, in one to three short sentences (usually 10 to 25 words).
- Say who is there and what they do, in the order it happens: "Two men are standing near the steps,
  appearing to be talking." / "A man takes a mop and looks at his phone while walking to the step."
- Mention what matters for safety: clothing that hides the face (hood, mask, covered face), dark or
  covering clothes, and objects in the hands (phone, bag, tool, hammer, knife, gun, baby, mop).
- Where something is uncertain, say "appears to" or "seems to".
- Say "a man", "a woman", "a person", "two men", "a group of people"; never guess names, age,
  ethnicity or who the person is.
- If nobody is there and nothing moves (parked cars, plants, light changes), write exactly:
  "No special activity."

Examples of good summaries:
- "Two men are standing next to each other, appearing to be engaged in conversation."
- "A group of people are holding babies, standing outside, and talking."
- "A woman enters the house while holding two babies simultaneously."
- "A cat in the yard."
- "No special activity."
- "Three men walk slowly toward the entrance, obscuring their faces with hats, while the rear
  individual uses his shirt to fully cover his face."
- "A covered person is near a white car with an open door. He appears to be doing something
  suspicious, looking around cautiously."
- "Two people dressed in black try to break inside. They appear to be using a tool to break in."
- "A man in a black hooded sweatshirt obscuring his face walks slowly, looking behind him while
  holding a knife in his hand."

Then give the clip ONE "label":
- "normal": everyday life - family and visitors, people talking, walking, smoking, cleaning, carrying
  babies or bags into the house, deliveries, cars parking or leaving, pets, or no special activity.
- "suspicious": something the homeowner should look at - faces hidden by hoods, masks or clothing,
  lingering or loitering, looking around cautiously, looking into windows or cars, trying doors,
  gates or car doors, walking around the property at night, hiding, or a vehicle waiting with no
  clear purpose.
- "escalation": a crime or danger in progress - a break-in or forced entry, breaking a door, window
  or car, stealing and carrying things away, climbing a fence or wall into the property, a fight or
  attack, a knife, gun or other weapon in hand, fire or smoke, a crash.

Reply with EXACTLY ONE strict JSON object and nothing else:
{{"summary": "<one to three short sentences>",
  "label": "normal" | "suspicious" | "escalation",
  "people": <how many people are visible in the frames, as a number; 0 if none>,
  "vehicle_moving": <true if a vehicle is driving, arriving or leaving; false if vehicles are only parked or there are none>}}
""".strip()


def vlm_confirms(parsed: Optional[Dict[str, Any]]) -> Optional[bool]:
    """What the VLM saw, for the decision to alert.

    True: a person, or a vehicle on the move. False: it looked and saw neither,
    so the detector's trigger was a false positive. None: it did not say (no
    answer, or an answer without these fields); the caller then trusts the
    detector, because a missed alert is worse than a needless one.
    """
    if not parsed or ("people" not in parsed and "vehicle_moving" not in parsed):
        return None
    try:
        people = int(parsed.get("people") or 0)
    except (TypeError, ValueError):
        return None
    return people > 0 or parsed.get("vehicle_moving") is True


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

    def live_values(self) -> Dict[str, Any]:
        """The values the window shows and the owner can change while the program runs."""
        return {"conf": self.conf, "alert_start_hour": self.alert_start_hour,
                "alert_end_hour": self.alert_end_hour, "cooldown_sec": self.cooldown_sec}

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
        parsed = {"summary": ""}
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
        self.model_name = model
        self.last_prompt = ""           # what the last call asked, kept for the training record
        self._response_format: Dict[str, Any] = VLM_RESPONSE_FORMAT

    def analyze(self, frames_bgr: List[Any], camera_name: str, t_sec: int,
                start_hour: int, end_hour: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = build_prompt(camera_name, t_sec, datetime.now().strftime("%H:%M:%S"), start_hour, end_hour)
        self.last_prompt = prompt
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for fr in frames_bgr:
            b64 = frame_to_jpeg_b64(fr)
            if b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        try:
            resp = self._complete(content, self._response_format)
        except Exception as exc:  # noqa: BLE001
            # A model without structured output refuses the schema: ask for plain JSON from now on.
            if self._response_format.get("type") == "json_schema" and "response_format" in str(exc):
                log.warning("%s does not take a JSON schema (%s); asking for a JSON object instead.", self._model, exc)
                self._response_format = {"type": "json_object"}
                resp = self._complete(content, self._response_format)
            else:
                raise
        raw = resp.choices[0].message.content or ""
        return raw, parse_vlm_json(raw)

    def _complete(self, content: List[Dict[str, Any]], response_format: Dict[str, Any]) -> Any:
        return self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format=response_format,
        )


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
                   image: Optional[bytes] = None,
                   assistant: Any = None, alert: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send the alert over the configured channel(s). Never raises.

    *image* (JPEG bytes) is the camera snapshot; Telegram sends it as a photo.
    With an *assistant* (telegram_agent.OwnerAssistant) the Telegram alert goes
    out with the feedback question and buttons, filed under *alert*.
    """
    channel = str(box_settings.get("alert_channel", "telegram"))
    results: Dict[str, Any] = {"channel": channel}
    try:
        if channel in ("telegram", "both"):
            from . import telegram_notify  # noqa: PLC0415

            text = telegram_notify.alert_text(command, summary, reason)
            if assistant is not None and alert is not None and text is not None:
                results["telegram"] = {"command": command, "telegram": assistant.send_alert(alert, text, image)}
            else:
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
# Parked vehicles: a vehicle-only trigger is worth a VLM call only if it moved
# ----------------------------------------------------------------------------
Box = Tuple[float, float, float, float]   # normalised xyxy, as results[0].boxes.xyxyn gives it

VEHICLE_SAME_PLACE_IOU = 0.7              # a vehicle box overlapping its old box this much has not moved


def vehicle_boxes(result) -> List[Box]:
    """Normalised xyxy boxes of the vehicles in a YOLO result, in detection order."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(result, "names", {})
    out: List[Box] = []
    for b in boxes:
        if names.get(int(b.cls[0]), "") not in VEHICLE_CLASSES:
            continue
        xyxyn = getattr(b, "xyxyn", None)
        if xyxyn is None or len(xyxyn) == 0:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxyn[0])
        out.append((x1, y1, x2, y2))
    return out


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def vehicles_moved(previous_boxes: Optional[Sequence[Box]], current_boxes: Sequence[Box],
                   iou_threshold: float = VEHICLE_SAME_PLACE_IOU) -> bool:
    """Did the vehicles in view change since *previous_boxes* was taken?

    True on the first look (*previous_boxes* is None), when the number of
    vehicles changed (one arrived or left), or when some vehicle has no old box
    overlapping it with IoU >= *iou_threshold* (it drove on). The detector's box
    wobbling on a parked car stays well above the threshold and does not count.

    Policy: a moving vehicle inside the alert window is still at least
    [send_message]; a parked one is nothing - no VLM call, no alert, no clip.
    """
    if previous_boxes is None:
        return True
    if len(previous_boxes) != len(current_boxes):
        return True
    return any(all(_iou(cur, old) < iou_threshold for old in previous_boxes) for cur in current_boxes)


class VehicleMemory:
    """Where a camera's vehicles were the last time they moved.

    One per camera, fed on every look at the camera - the once-a-second looks
    during the cooldown too - so a car that arrives and parks during a cooldown
    is compared, two minutes later, with its own parked position and stays
    quiet. The reference is replaced only when the vehicles moved: comparing
    each look only with the one before would let a car creep in unnoticed, a
    few centimetres per look.
    """

    def __init__(self) -> None:
        self._reference: Optional[List[Box]] = None

    def look(self, boxes: Sequence[Box]) -> bool:
        """Record one look; True if the vehicles moved since the reference."""
        moved = vehicles_moved(self._reference, boxes)
        if moved:
            self._reference = list(boxes)
        return moved


def should_escalate(person: bool, vehicle: bool, vehicles_moved: bool) -> bool:
    """A person always goes to the VLM; a vehicle only when it moved since the camera's previous look."""
    return person or (vehicle and vehicles_moved)


# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
STATUS_LOOK_SEC = 1.0   # how often the detector looks at a camera that cannot alert right now
LIVE_SETTINGS_POLL_SEC = 2.0   # how often box.yaml is checked for a change made while running


def apply_live_settings(settings: AlertSettings, box_settings: Dict[str, Any]) -> List[str]:
    """Take over the values the program re-reads while running. Returns the names that changed."""
    fresh = AlertSettings.from_box_settings(box_settings)
    changed = []
    for name in ("alert_start_hour", "alert_end_hour", "cooldown_sec", "conf"):
        if getattr(settings, name) != getattr(fresh, name):
            # The settings object is frozen and shared with the worker threads: the same
            # instance must carry the new value, so the one write goes around the freeze.
            object.__setattr__(settings, name, getattr(fresh, name))
            changed.append(name)
    return changed


class LiveSettings:
    """Re-reads box.yaml while the program runs, so the owner's changes apply without a restart.

    The alert hours, the cooldown and the detector's threshold (boxconfig.LIVE_OPTIONS)
    are taken over within a couple of seconds of the file changing; everything
    else still needs a restart.
    """

    def __init__(self, settings: AlertSettings, path: Optional[str] = None,
                 poll_sec: float = LIVE_SETTINGS_POLL_SEC, now: float = 0.0) -> None:
        from .boxconfig import BOX_YAML  # noqa: PLC0415

        self.settings = settings
        self.path = path or BOX_YAML
        self.poll_sec = poll_sec
        self._mtime = self._stat()
        self._checked = now

    def _stat(self) -> Optional[float]:
        try:
            return os.stat(self.path).st_mtime
        except OSError:
            return None

    def check(self, now: float) -> List[str]:
        """Apply a change if the file changed since the last look. Returns the names that changed."""
        if now - self._checked < self.poll_sec:
            return []
        self._checked = now
        mtime = self._stat()
        if mtime == self._mtime:
            return []
        self._mtime = mtime
        try:
            from .boxconfig import load_box_settings  # noqa: PLC0415

            box_settings = load_box_settings(self.path)
        except Exception as exc:  # noqa: BLE001 - a half-written file: keep the current values
            log.warning("Settings file not readable (%s); keeping the current values.", exc)
            return []
        changed = apply_live_settings(self.settings, box_settings)
        if changed:
            log.info("Settings changed while running: %s",
                     ", ".join(f"{name}={getattr(self.settings, name)}" for name in changed))
        return changed


class _Stream:
    """Threaded RTSP reader keeping only the latest frame (drops stale frames).

    With a *ring* (alert_clips.ClipRing) it also keeps the last seconds for the
    alert clip. That happens here, at the camera's own pace: the detection loop
    visits a camera only every second or two, too rarely for a video.
    """

    def __init__(self, name: str, url: str, ring: Any = None, mask: Any = None) -> None:
        import cv2  # noqa: PLC0415

        self.name = name
        self.url = url
        self._ring = ring
        self._mask = mask                    # zones.ZoneMask, or None to watch the whole picture
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
            self._ingest(frame, time.time())

    def _ingest(self, frame: Any, now: float) -> None:
        """One decoded frame: masked to the watch zone first, then kept as the latest frame and offered to the clip ring."""
        if self._mask is not None:
            frame = self._mask.apply(frame)
        with self._lock:
            self._frame = frame
        if self._ring is not None and self._ring.wants(now):
            from .alert_clips import encode_frame  # noqa: PLC0415

            self._ring.add(now, encode_frame(frame))

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


@dataclass
class AlertJob:
    """One alert on its way: the worker fills in what was decided, the clip writer saves it with the video."""

    camera: str
    stem: str                        # the clip's file stem, and the id the owner's answers are filed under
    ts: float                        # when the gate opened
    labels: List[str] = field(default_factory=list)
    alert: Dict[str, Any] = field(default_factory=dict)
    false_positive: bool = False     # the VLM saw nothing: not sent, saved for training instead
    paused: bool = False             # the owner had paused alerts: the AI was not asked, saved for training
    teacher: Dict[str, Any] = field(default_factory=dict)   # what the VLM was asked and answered (alert_clips.teacher_record)
    ready: threading.Event = field(default_factory=threading.Event)


def delivery(res: Dict[str, Any]) -> Tuple[bool, str]:
    """Did the alert reach anyone, and if not, why: ``(sent, reason)``.

    A dispatch result is nested per channel and per chat; one delivery anywhere
    counts as sent. The reason is the first one a channel gave.
    """
    sent = False
    reasons: List[str] = []

    def walk(node: Any) -> None:
        nonlocal sent
        if isinstance(node, dict):
            if node.get("sent") is True:
                sent = True
            for key in ("error", "reason"):
                if isinstance(node.get(key), str) and node[key]:
                    reasons.append(node[key])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(res)
    return sent, "" if sent else (reasons[0] if reasons else "not delivered")


def _jpegs(frames: List[Any]) -> List[bytes]:
    """The frames as JPEG bytes, exactly as they are sent to the VLM; a frame that cannot be encoded is skipped."""
    out = []
    for frame in frames:
        try:
            data = frame_to_jpeg_bytes(frame)
        except Exception:  # noqa: BLE001
            data = b""
        if data:
            out.append(data)
    return out


def _worker(backend, box_settings, env, settings: AlertSettings,
            camera_name: str, frames: List[Any],
            assistant: Any = None, job: Optional[AlertJob] = None, status: Any = None) -> None:
    """Run the VLM call + dispatch off the capture loop. Never raises out.

    With an *assistant*, a camera the owner has paused is not analysed at all:
    the clip is kept for training (the owner usually paused because they know
    who it is), nothing is sent. *job* receives the outcome for the clip's
    meta, and *status* (ai_status.AiStatus) what the box's window shows.
    """
    labels = job.labels if job is not None else []
    try:
        now = datetime.now()
        in_win = in_alert_window(now.hour, settings.alert_start_hour, settings.alert_end_hour)
        if not in_win:
            return
        if assistant is not None and assistant.is_muted(camera_name):
            log.info("[%s] alerts are paused; the AI was not asked (labels=%s)", camera_name, labels)
            if job is not None:
                job.paused = True
                job.alert = {"summary": "", "alert_command": "[none]", "alert_reason": "alerts paused by the owner",
                             "labels": labels, "muted": True, "paused": True}
            if status is not None:
                status.decision(camera_name, labels, "Alerts are paused; the AI was not asked.", "[none]",
                                sent=False, muted=True)
            return
        raw, parsed = backend.analyze(frames, camera_name, int(time.time()),
                                      settings.alert_start_hour, settings.alert_end_hour)
        if job is not None and raw:
            # Everything a student model needs to learn this answer: the exact pictures,
            # the question, and the answer word for word.
            job.teacher = {
                "model": getattr(backend, "model_name", settings.vlm_model),
                "prompt_version": PROMPT_VERSION,
                "prompt": getattr(backend, "last_prompt", ""),
                "frames": _jpegs(frames),
                "raw": raw,
                "parsed": parsed,
            }
        summary = ""
        if parsed:
            summary = str(parsed.get("summary", "")).strip()
        else:
            log.warning("[%s] VLM returned no usable output: %s", camera_name, (raw or "")[:200])
        # The YOLO gate already confirmed a person/vehicle inside the window, so this is
        # at least a [send_message]; the model's label can raise it (LABEL_COMMANDS).
        label = label_of(parsed)
        cmd = LABEL_COMMANDS[label]
        if not summary:
            summary = "a person or vehicle was detected"
        reason = str(parsed.get("alert_reason", "")) if parsed else ""
        if vlm_confirms(parsed) is False:
            # The detector fired, the VLM looked and saw no person and nothing moving: no
            # message to the owner. The clip is kept as a false positive, for training.
            log.info("[%s] no alert: the VLM saw nobody and nothing moving (%s)", camera_name, summary)
            if job is not None:
                job.false_positive = True
                job.alert = {"summary": summary, "label": label, "alert_command": "[none]", "alert_reason": "",
                             "labels": job.labels, "false_positive": True,
                             "vlm": {"people": parsed.get("people"), "vehicle_moving": parsed.get("vehicle_moving")}}
            if status is not None:
                status.decision(camera_name, labels, summary, "[none]", sent=False, false_positive=True, label=label)
            return
        log.info("[%s] alert=%s label=%s summary=%s", camera_name, cmd, label, summary)
        # Attach the most recent frame of the clip as the alert snapshot.
        image = b""
        try:
            image = frame_to_jpeg_bytes(frames[-1]) if frames else b""
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] could not encode snapshot: %s", camera_name, exc)
        muted = bool(assistant is not None and assistant.is_muted(camera_name))
        if muted:
            res: Dict[str, Any] = {"sent": False, "reason": "paused by the owner"}
            log.info("[%s] alert not sent: the owner paused alerts", camera_name)
        else:
            alert_ref = None
            if job is not None:
                alert_ref = {"alert_id": job.stem, "camera": camera_name, "summary": summary, "label": label,
                             "ts": job.ts}
            res = dispatch_alert(box_settings, env, cmd, f"{camera_name}: {alert_summary(label, summary)}", reason,
                                 image=image or None, assistant=assistant, alert=alert_ref)
            log.info("[%s] alert dispatched: %s", camera_name, res)
        if status is not None:
            sent, why_not = delivery(res)
            status.decision(camera_name, labels, summary, cmd, sent=sent, muted=muted, error=why_not, label=label)
        if job is not None:
            job.alert = {"summary": summary, "label": label, "alert_command": cmd, "alert_reason": reason,
                         "labels": job.labels, "muted": muted, "dispatch": res}
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] worker error: %s", camera_name, exc)
    finally:
        if job is not None:
            job.ready.set()


def _save_clip(job: AlertJob, frames: List[Any], production_dir: str, training_dir: str,
               assistant: Any = None) -> None:
    """Write the alert's clip once the worker has decided what the alert was. Never raises out.

    A real alert goes to *production_dir* (kept two weeks, where the owner can
    ask for it). A false positive goes to *training_dir*, the collector's own
    dataset folder, and is uploaded with the clips for tagging. With an
    *assistant*, the video of an alert that was sent follows it in Telegram.
    """
    try:
        from .alert_clips import clip_file, false_positive_stem, write_alert_clip  # noqa: PLC0415

        job.ready.wait(timeout=90)
        alert = job.alert or {"summary": "", "alert_command": "[none]", "alert_reason": "", "labels": job.labels}
        if job.false_positive:
            meta = write_alert_clip(training_dir, job.camera, false_positive_stem(job.camera, job.ts), frames,
                                    alert, kind="false_positive", teacher=job.teacher)
        elif job.paused:
            meta = write_alert_clip(training_dir, job.camera, f"{job.camera}_{int(job.ts)}_paused", frames,
                                    alert, kind="paused")
        else:
            meta = write_alert_clip(production_dir, job.camera, job.stem, frames, alert)
            # The owner's copy above expires in two weeks; the training set keeps every
            # alert with the teacher's answer, so a student model can be trained on it.
            write_alert_clip(training_dir, job.camera, job.stem, frames, alert, kind="alert", teacher=job.teacher)
        if meta:
            log.info("[%s] clip saved: %s (%d frames)", job.camera, os.path.basename(meta), len(frames))
        if (meta and assistant is not None and not job.false_positive and not job.paused
                and delivery(alert.get("dispatch") or {})[0]):
            res = assistant.send_clip(job.stem, clip_file(production_dir, meta))
            log.info("[%s] video %s", job.camera, "sent" if res.get("sent") else f"not sent: {res}")
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] could not save the clip %s: %s", job.camera, job.stem, exc)


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    # Use the OS certificate store for all HTTPS (urllib + httpx), so Telegram
    # and OpenAI work on networks that intercept TLS (an antivirus / proxy whose
    # root is installed in the Windows store). Harmless where there is no proxy.
    try:
        import truststore  # noqa: PLC0415

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        from .boxconfig import PROJECT_ROOT  # noqa: PLC0415

        # Secrets live in api_key.env (NOT .env), at the repo root.
        load_dotenv(os.path.join(PROJECT_ROOT, "api_key.env"))
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
    # Every alert is saved as a clip (the seconds around it) in the production folder,
    # and the owner can answer it in Telegram. Neither may stop the alerts themselves.
    from .alert_clips import POST_SECONDS, PRE_SECONDS, ClipRing, alert_stem  # noqa: PLC0415
    from .ai_status import AiStatus, objects_from_result  # noqa: PLC0415
    from .boxconfig import LIVE_DIR, LOG_DIR, PRODUCTION_LIVE_DIR  # noqa: PLC0415

    rings: Dict[str, ClipRing] = {name: ClipRing() for name in cameras}
    from ..data_collection.zones import mask_for  # noqa: PLC0415

    zones = dict(getattr(cam_cfg, "ROI_ZONES", {}) or {})
    for name in cameras:
        if name in zones:
            log.info("[%s] watch zone active (%d corners); everything outside is blacked out", name, len(zones[name]))
    streams = {name: _Stream(name, url, ring=rings[name], mask=mask_for(zones, name)) for name, url in cameras.items()}
    buffers: Dict[str, deque] = {name: deque(maxlen=settings.clip_frames) for name in cameras}
    last_buf_ts: Dict[str, float] = {name: 0.0 for name in cameras}
    last_alert_ts: Dict[str, float] = {name: 0.0 for name in cameras}
    last_look_ts: Dict[str, float] = {name: 0.0 for name in cameras}
    vehicles: Dict[str, VehicleMemory] = {name: VehicleMemory() for name in cameras}
    parked: Dict[str, bool] = {name: False for name in cameras}  # "have not moved" already logged
    status = AiStatus(os.path.join(LOG_DIR, "ai_status.json"))  # what the box's window shows
    live = LiveSettings(settings, now=time.time())
    status.settings(settings.live_values())
    worker = {"t": None}  # single in-flight VLM call across cameras (N150 budget)
    pending: List[AlertJob] = []
    assistant = None
    try:
        from . import telegram_agent  # noqa: PLC0415

        assistant = telegram_agent.start(box_settings, env, list(cameras))
    except Exception as exc:  # noqa: BLE001
        log.warning("Owner assistant not started (%s); alerts go out without feedback buttons.", exc)

    log.info("Watching %d camera(s): %s", len(cameras), ", ".join(cameras))
    while True:
        now_ts = time.time()
        if live.check(now_ts):
            status.settings(settings.live_values())
        for job in [j for j in pending if now_ts >= j.ts + POST_SECONDS]:
            pending.remove(job)
            clip = rings[job.camera].between(job.ts - PRE_SECONDS, job.ts + POST_SECONDS)
            threading.Thread(target=_save_clip, args=(job, clip, PRODUCTION_LIVE_DIR, LIVE_DIR, assistant),
                             daemon=True).start()
        for name in cameras:
            frame = streams[name].read()
            if frame is None:
                continue
            # Maintain a rolling buffer, one frame every frame_interval_sec.
            if now_ts - last_buf_ts[name] >= settings.frame_interval_sec:
                buffers[name].append(frame)
                last_buf_ts[name] = now_ts

            now = datetime.now()
            if not in_alert_window(now.hour, settings.alert_start_hour, settings.alert_end_hour):
                continue
            # No new alert for this camera during its cooldown, or while a VLM call is
            # running. The detector still looks about once a second then, only so the
            # window can show what it sees.
            waiting = (now_ts - last_alert_ts[name] < settings.cooldown_sec
                       or (worker["t"] is not None and worker["t"].is_alive()))
            if waiting and now_ts - last_look_ts[name] < STATUS_LOOK_SEC:
                continue
            last_look_ts[name] = now_ts

            results = model.predict(frame, conf=settings.conf, verbose=False)
            try:
                status.detection(name, objects_from_result(results[0]) if results else [], now=now_ts)
            except Exception as exc:  # noqa: BLE001 - what the window shows must never stop the alerts
                log.debug("[%s] status not updated: %s", name, exc)
            # Every look feeds the camera's vehicle memory - the once-a-second looks of the
            # cooldown too - so a car that arrives and parks during the cooldown is compared
            # with its own parked position afterwards, and stays quiet.
            moved = vehicles[name].look(vehicle_boxes(results[0]) if results else [])
            if waiting:
                continue
            person, vehicle, labels = detect_trigger(results[0]) if results else (False, False, [])
            if not (person or vehicle):
                parked[name] = False
                continue
            if not should_escalate(person, vehicle, moved):
                if not parked[name]:  # one line per parked spell, not one per look
                    log.info("[%s] vehicles have not moved; no alert", name)
                    parked[name] = True
                continue
            parked[name] = False
            if len(buffers[name]) == 0:
                buffers[name].append(frame)
            frames = list(buffers[name])
            last_alert_ts[name] = now_ts
            log.info("[%s] escalating (labels=%s), calling VLM", name, labels)
            status.thinking(name, labels, now=now_ts)
            job = AlertJob(camera=name, stem=alert_stem(name, now_ts), ts=now_ts, labels=labels)
            pending.append(job)
            t = threading.Thread(target=_worker,
                                 args=(backend, box_settings, env, settings, name, frames, assistant, job, status),
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
