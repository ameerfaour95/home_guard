"""Headless inference (production) runner for a collector box.

Launched by run_collector.sh when box.yaml has ``mode: inference``:

    python -u -m home_guard_project.box.inference

Pipeline, per camera: YOLO gate (is anyone/anything there?) -> only inside the
owner's alert time window, wait for the complete crop clip -> a VLM backend (GPT-4o now,
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("box.inference")

# Classes that open the gate (match the alert policy: person / vehicle / animal).
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
# Birds are left out: they would wake the AI all day.
ANIMAL_CLASSES = {"cat", "dog", "horse", "sheep", "cow", "bear"}
TRIGGER_CLASSES = PERSON_CLASSES | VEHICLE_CLASSES | ANIMAL_CLASSES
# What the owner can be alerted about (box.yaml alert_on is the house default, a camera can
# have its own in camera_alerts.yaml), and the default: people only.
ALERT_ON_CHOICES = ("person", "vehicle", "animal")
DEFAULT_ALERT_ON = ("person",)


def parse_alert_on(value: Any) -> Tuple[str, ...]:
    """box.yaml's alert_on ("person,vehicle" or a list) as a tuple in fixed order; anything unusable -> people only."""
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        return DEFAULT_ALERT_ON
    picked = {p.strip().lower() for p in parts if p.strip()}
    if not picked or not picked <= set(ALERT_ON_CHOICES):
        return DEFAULT_ALERT_ON
    return tuple(c for c in ALERT_ON_CHOICES if c in picked)


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
PROMPT_VERSION = "2026-10-03.tagged-rules-label-animals-why-owner"

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
        "animals": {"type": "integer"},
        "why": {"type": "string"},
        "summary_owner": {"type": "string"},
    },
    "required": ["summary", "label", "people", "vehicle_moving", "animals", "why", "summary_owner"],
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


def is_silent(label: str) -> bool:
    """Only a normal scene is delivered without a sound. Suspicious and escalation are always loud."""
    return label == "normal"


def owner_language() -> str:
    """The box language (alerts and announcements), read from box.yaml each time: it changes without a restart."""
    try:
        from .boxconfig import load_box_settings  # noqa: PLC0415

        return "he" if str(load_box_settings().get("owner_language") or "en") == "he" else "en"
    except Exception:  # noqa: BLE001
        return "en"


_PLACEHOLDERS = ("an empty string", "empty string")


def owner_summary(summary: str, summary_owner: str, lang: str) -> str:
    """The summary the owner reads: the model's *summary_owner* in the box language when it is not English and the
    model really wrote one; otherwise *summary*. An English prompt asks for summary_owner as "<an empty string>",
    and a model sometimes copies that placeholder word for word, so it never reaches the owner."""
    text = (summary_owner or "").strip()
    if lang == "en" or not text:
        return summary
    if (text.startswith("<") and text.endswith(">")) or text.strip("<>.\"' ").lower() in _PLACEHOLDERS:
        return summary
    return text


def _int_or_none(value: Any) -> Optional[int]:
    """A count from the model's answer, or None when it gave something that is not a number."""
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return None


VLM_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "camera_report", "strict": True, "schema": VLM_SCHEMA},
}


# The tagging rules for the three labels, shared by the guard loop's prompt and the
# assistant's guard-mode look at a saved clip, so both judge a scene the same way.
LABEL_RULES = """
- "normal": everyday life - family and visitors, people talking, walking, standing or waiting, looking
  at a phone, smoking, cleaning, carrying babies or bags into the house, deliveries, cars parking or
  leaving, pets, or no special activity. A person standing still is normal unless they hide their face
  or do something from the "suspicious" list.
- "suspicious": something the homeowner should look at - faces hidden by hoods, masks or clothing,
  lingering or loitering, looking around cautiously, looking into windows or cars, trying doors,
  gates or car doors, walking around the property at night, hiding, or a vehicle waiting with no
  clear purpose.
- "escalation": a crime or danger in progress - a break-in or forced entry, breaking a door, window
  or car, stealing and carrying things away, climbing a fence or wall into the property, a fight or
  attack, a knife, gun or other weapon in hand, fire or smoke, a crash.
""".strip()


def build_prompt(camera_name: str, t_sec: int, local_time_str: str, start_hour: int, end_hour: int,
                 owner_language: str = "en") -> str:
    language = "Hebrew" if owner_language == "he" else "English"
    owner_rule = "the same summary, translated into Hebrew" if owner_language == "he" else "an empty string"
    # The summary follows the rules our taggers wrote by (tagging/*/analysis_output/
    # vlm_training.jsonl): what happens, in order, with what people wear and hold, and
    # "No special activity." for an empty scene. No example sentences, so the model does
    # not copy their wording. The label replaces the taggers' "[alert]" mark and
    # decides what the box does (LABEL_COMMANDS); people/vehicle_moving decide whether
    # anything is sent at all (vlm_confirms).
    return f"""
You are the eyes of a home security system. These are sequential frames (one short clip of a few
seconds) from the homeowner's own camera "{camera_name}", local time {local_time_str}.

Write "summary": what happens in the clip, in one to three short sentences (usually 10 to 25 words).
- Say who is there and what they do, in the order it happens.
- Mention what matters for safety: clothing that hides the face (hood, mask, covered face), dark or
  covering clothes, and objects in the hands (phone, bag, tool, hammer, knife, gun, baby, mop).
- Where something is uncertain, say "appears to" or "seems to".
- Describe only what is there and what happens. Do not mention what is absent ("no faces are
  obscured", "no movement") or the background (parked cars, walls, plants) unless someone acts on it.
- Say "a man", "a woman", "a person", "two men", "a group of people"; never guess names, age,
  ethnicity or who the person is.
- If nobody is there and nothing moves (parked cars, plants, light changes), write exactly:
  "No special activity."

Then give the clip ONE "label":
{LABEL_RULES}
Dark clothing alone never makes a scene suspicious; judge what people do.

Reply with EXACTLY ONE strict JSON object and nothing else:
{{"summary": "<one to three short sentences>",
  "label": "normal" | "suspicious" | "escalation",
  "people": <how many people are visible in the frames, as a number; 0 if none>,
  "vehicle_moving": <true if a vehicle is driving, arriving or leaving; false if vehicles are only parked or there are none>,
  "animals": <how many animals (cats, dogs and other animals, not birds) are visible, as a number; 0 if none>,
  "why": "<one short clause in {language} naming the behaviour behind a suspicious or escalation label; empty for normal>",
  "summary_owner": "<{owner_rule}>"}}
""".strip()


def vlm_confirms(parsed: Optional[Dict[str, Any]],
                 alert_on: Sequence[str] = ALERT_ON_CHOICES) -> Optional[bool]:
    """What the VLM saw, for the decision to alert.

    True: a person, a vehicle on the move, or an animal - each only if the
    camera alerts on it (*alert_on*). False: it looked and saw nothing the owner alerts on, so the
    detector's trigger was a false positive (a passing car when the owner wants
    people only, too). None: it did not say (no
    answer, or an answer without these fields); the caller then trusts the
    detector, because a missed alert is worse than a needless one.
    """
    if not parsed or not any(k in parsed for k in ("people", "vehicle_moving", "animals")):
        return None
    try:
        people = int(parsed.get("people") or 0)
        animals = int(parsed.get("animals") or 0)
    except (TypeError, ValueError):
        return None
    return (("person" in alert_on and people > 0)
            or ("vehicle" in alert_on and parsed.get("vehicle_moving") is True)
            or ("animal" in alert_on and animals > 0))


@dataclass(frozen=True)
class AlertSettings:
    alert_start_hour: int = 0
    alert_end_hour: int = 0          # 0,0 -> always in window (handy for testing)
    cooldown_sec: float = 120.0      # min seconds between VLM calls per camera
    model: str = "yolo11s.pt"        # the model shipped in the bundle (avoids a download on the box)
    device: str = "auto"             # auto: the Intel graphics chip when there is one, else the CPU; cpu: always the CPU
    conf: float = 0.4
    vlm_backend: str = "gpt"
    vlm_model: str = "gpt-4o"
    alert_channel: str = "telegram"  # telegram | twilio | both
    dry_run: bool = False
    alert_on: Tuple[str, ...] = DEFAULT_ALERT_ON   # what reaches the owner: people, vehicles or both
    # The detector's certainty per type; None -> conf (the single house threshold).
    conf_person: Optional[float] = None
    conf_vehicle: Optional[float] = None
    conf_animal: Optional[float] = None

    def thresholds(self) -> Dict[str, float]:
        """The house's certainty per type (person / vehicle / animal)."""
        own = {"person": self.conf_person, "vehicle": self.conf_vehicle, "animal": self.conf_animal}
        return {kind: float(self.conf if value is None else value) for kind, value in own.items()}

    def live_values(self) -> Dict[str, Any]:
        """The values the window shows and the owner can change while the program runs."""
        return {"conf": self.conf, "alert_start_hour": self.alert_start_hour,
                "alert_end_hour": self.alert_end_hour, "cooldown_sec": self.cooldown_sec,
                "alert_on": list(self.alert_on), "sensitivity": self.thresholds()}

    @classmethod
    def from_box_settings(cls, s: Dict[str, Any]) -> "AlertSettings":
        g = s.get
        return cls(
            alert_start_hour=int(g("alert_start_hour", 0)),
            alert_end_hour=int(g("alert_end_hour", 0)),
            cooldown_sec=float(g("alert_cooldown_sec", 120.0)),
            model=str(g("inference_yolo_model", "yolo11s.pt")),
            device=str(g("inference_device", "auto")).strip().lower(),
            conf=float(g("inference_conf", 0.4)),
            vlm_backend=str(g("vlm_backend", "gpt")),
            vlm_model=str(g("vlm_model", "gpt-4o")),
            alert_channel=str(g("alert_channel", "telegram")),
            dry_run=bool(g("notify_dry_run", False)),
            alert_on=parse_alert_on(g("alert_on", ",".join(DEFAULT_ALERT_ON))),
            conf_person=_optional_float(g("conf_person")),
            conf_vehicle=_optional_float(g("conf_vehicle")),
            conf_animal=_optional_float(g("conf_animal")),
        )


def _optional_float(value: Any) -> Optional[float]:
    """A box.yaml number, or None when it is unset or not a number (then the single threshold applies)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
                start_hour: int, end_hour: int, owner_language: str = "en") -> Tuple[str, Optional[Dict[str, Any]]]:
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
                start_hour: int, end_hour: int, owner_language: str = "en") -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = build_prompt(camera_name, t_sec, datetime.now().strftime("%H:%M:%S"), start_hour, end_hour,
                              owner_language=owner_language)
        self.last_prompt = prompt
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        self.last_frame_jpegs = []
        for fr in frames_bgr:
            data = frame_to_jpeg_bytes(fr)
            if data:
                self.last_frame_jpegs.append(data)
                b64 = base64.b64encode(data).decode("utf-8")
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
                   assistant: Any = None, alert: Optional[Dict[str, Any]] = None,
                   graded: Optional[str] = None, silent: bool = False, lang: str = "en") -> Dict[str, Any]:
    """Send the alert over the configured channel(s). Never raises.

    *image* (JPEG bytes) is the camera snapshot; Telegram sends it as a photo.
    With an *assistant* (telegram_agent.OwnerAssistant) the Telegram alert goes
    out with the feedback question and buttons, filed under *alert*: as the
    *graded* text (telegram_notify.graded_alert_text) when given, without a
    sound when *silent*, with the question and buttons in *lang*.
    """
    channel = str(box_settings.get("alert_channel", "telegram"))
    results: Dict[str, Any] = {"channel": channel}
    try:
        if channel in ("telegram", "both"):
            from . import telegram_notify  # noqa: PLC0415

            text = graded or telegram_notify.alert_text(command, summary, reason)
            if assistant is not None and alert is not None and text is not None:
                results["telegram"] = {"command": command,
                                       "telegram": assistant.send_alert(alert, text, image, silent=silent, lang=lang)}
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


def kind_of(label: str) -> Optional[str]:
    """person / vehicle / animal for a detector class name; None for anything else (and birds)."""
    if label in PERSON_CLASSES:
        return "person"
    if label in VEHICLE_CLASSES:
        return "vehicle"
    if label in ANIMAL_CLASSES:
        return "animal"
    return None


class _Kept:
    """A detector result holding only the finds that passed their type's certainty."""

    def __init__(self, boxes: List[Any], names: Dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


def filter_by_thresholds(result: Any, thresholds: Dict[str, float], other: float) -> Any:
    """Keep each find only if the detector is as sure as its type requires (*other* for the rest).

    The detector is asked down to the lowest certainty in use (:func:`detector_floor`),
    so a person the owner wants caught at 50% is not lost while cars need 80%.
    """
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {})
    if boxes is None or len(boxes) == 0:
        return _Kept([], names)
    kept = []
    for b in boxes:
        kind = kind_of(names.get(int(b.cls[0]), ""))
        if float(b.conf[0]) >= thresholds.get(kind, other) if kind else float(b.conf[0]) >= other:
            kept.append(b)
    return _Kept(kept, names)


def detector_floor(thresholds: Dict[str, float], other: float) -> float:
    """The certainty the detector itself is run at: the lowest any type needs."""
    return min([other, *thresholds.values()])


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


VEHICLE_MOVE_CONFIRM_SEC = 1.0   # a change must last this long, over two looks or more, to be movement


class VehicleMemory:
    """Where a camera's vehicles were the last time they moved.

    One per camera, fed on every look at the camera - the once-a-second looks
    during the cooldown too - so a car that arrives and parks during a cooldown
    is compared, two minutes later, with its own parked position and stays
    quiet. The reference is replaced only when the vehicles moved: comparing
    each look only with the one before would let a car creep in unnoticed, a
    few centimetres per look.

    A change counts as movement only once it has lasted *confirm_sec* over two
    looks or more. On the graphics chip each camera is looked at several times
    a second, and the detector missing a parked car for a look or two must not
    read as "a car left".
    """

    def __init__(self, confirm_sec: float = VEHICLE_MOVE_CONFIRM_SEC) -> None:
        self._reference: Optional[List[Box]] = None
        self._confirm_sec = confirm_sec
        self._changed_since: Optional[float] = None   # first look of the current change
        self._changed_looks = 0

    def look(self, boxes: Sequence[Box], now: Optional[float] = None) -> bool:
        """Record one look; True if the vehicles moved since the reference."""
        if self._reference is None:   # the first look: whatever is there counts as arrived
            self._reference = list(boxes)
            return True
        if not vehicles_moved(self._reference, boxes):
            self._changed_since, self._changed_looks = None, 0
            return False
        now = time.monotonic() if now is None else now
        if self._changed_since is None:
            self._changed_since = now
        self._changed_looks += 1
        if self._changed_looks < 2 or now - self._changed_since < self._confirm_sec:
            return False
        self._reference = list(boxes)
        self._changed_since, self._changed_looks = None, 0
        return True


def has_animal(labels: Sequence[str]) -> bool:
    """Did the detector see an animal (not a bird)?"""
    return any(label in ANIMAL_CLASSES for label in labels)


def quiet_reason(vehicle: bool, vehicles_moved: bool, alert_on: Sequence[str]) -> str:
    """Why the detector's find did not wake the AI, for the log."""
    if vehicle and not vehicles_moved and "vehicle" in alert_on:
        return "vehicles have not moved"
    return f"this camera alerts only on {', '.join(alert_on)}"


def should_escalate(person: bool, vehicle: bool, vehicles_moved: bool,
                    alert_on: Sequence[str] = ALERT_ON_CHOICES, animal: bool = False) -> bool:
    """A person or an animal goes to the VLM; a vehicle only when it moved since the camera's previous look.

    Only what the camera alerts on (*alert_on*) counts: with people only, a car
    or a cat never wakes the AI.
    """
    return (("person" in alert_on and person)
            or ("vehicle" in alert_on and vehicle and vehicles_moved)
            or ("animal" in alert_on and animal))


# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
STATUS_LOOK_SEC = 1.0   # how often the detector looks at a camera that cannot alert right now
LIVE_SETTINGS_POLL_SEC = 2.0   # how often box.yaml is checked for a change made while running


def apply_live_settings(settings: AlertSettings, box_settings: Dict[str, Any]) -> List[str]:
    """Take over the values the program re-reads while running. Returns the names that changed."""
    fresh = AlertSettings.from_box_settings(box_settings)
    changed = []
    for name in ("alert_start_hour", "alert_end_hour", "cooldown_sec", "conf", "alert_on",
                 "conf_person", "conf_vehicle", "conf_animal"):
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
    """Preview-compatible adapter over the collector's sub reader.

    The legacy constructor/ingest path remains for existing stream consumers;
    run() always supplies sub_cap and opens no capture through this adapter.
    """

    def __init__(self, name: str, url: str, ring: Any = None, mask: Any = None, sub_cap: Any = None) -> None:
        import cv2  # noqa: PLC0415

        self.name = name
        self.sub_cap = sub_cap
        if sub_cap is not None:
            # Runtime uses the collector reader; retain read() for the preview adapter.
            self._frame = None
            return
        self.url = url
        self._ring = ring
        self._mask = mask                    # zones.ZoneMask, or None to watch the whole picture
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._frame = None
        self.last_ts = 0.0                   # when the camera last delivered a picture; 0.0 until the first
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
            try:
                self._ingest(frame, time.time())
            except Exception as exc:  # noqa: BLE001 - a bad frame is dropped (never stored unmasked); the camera keeps running
                if not getattr(self, "_ingest_failed", False):
                    log.warning("[%s] frame dropped: %s", self.name, exc)
                self._ingest_failed = True
            else:
                self._ingest_failed = False

    def _ingest(self, frame: Any, now: float) -> None:
        """One decoded frame: masked to the watch zone first, then kept as the latest frame and offered to the clip ring."""
        if self._mask is not None:
            frame = self._mask.apply(frame)
        with self._lock:
            self._frame = frame
        self.last_ts = now          # before the ring: a failing encode must not freeze the camera's clock
        if self._ring is not None and self._ring.wants(now):
            from .alert_clips import encode_frame  # noqa: PLC0415

            self._ring.add(now, encode_frame(frame))

    @property
    def last_ts(self):
        if getattr(self, "sub_cap", None) is not None:
            # The reader also updates last_frame_ts on reconnect: an empty camera
            # must not count as having delivered its first picture.
            with self.sub_cap.buf_lock:
                return self.sub_cap.buf[-1][0] if self.sub_cap.buf else 0.0
        return self._last_ts

    @last_ts.setter
    def last_ts(self, value):
        self._last_ts = value

    def read(self):
        if getattr(self, "sub_cap", None) is not None:
            ok, self._frame = self.sub_cap.get_latest()
            return self._frame.copy() if ok else None
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
    snapshot: Any = None             # whole sub frame for Telegram, never the VLM crop
    alert_on: Optional[Sequence[str]] = None
    clip_fps: Optional[float] = None
    crop: Any = None
    crop_settings: Any = None
    crop_fps: Optional[float] = None
    input_meta: Dict[str, Any] = field(default_factory=dict)


def _camera_streams(cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Exactly one collector sub reader and one main reader per configured camera."""
    from ..data_collection.streams import SubStreamThread, MainStreamThread
    from ..data_collection.zones import mask_for

    streams, main_caps = {}, {}
    for name, url in cfg.CAMERAS.items():
        sub = SubStreamThread(cfg, url, mask=mask_for(cfg.ROI_ZONES, name), name=name)
        streams[name] = _Stream(name, url, sub_cap=sub)
        main_url = cfg.CAMERAS_MAIN.get(name)
        main_caps[name] = (MainStreamThread(cfg, main_url, mask=mask_for(cfg.ROI_ZONES, name), name=name)
                           if cfg.MAIN_STREAM_ENABLED and main_url else None)
    return streams, main_caps


def _prepare_alert(job: AlertJob, cfg: Any, detector: Any, sub_cap: Any, main_cap: Any,
                   predict_args: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    """Freeze the completed window, then use the collector's crop and sampling verbatim.

    Called on the detection loop so the shared YOLO model is never used concurrently.
    Reader threads keep buffering while this per-alert work runs.
    """
    from ..data_collection import vlm_crop

    reason = ""
    try:
        sub_frames, start, end, sub_fps = sub_cap.get_clip_last_seconds(cfg.CLIP_SECONDS)
    except Exception:
        log.exception("[%s] sub-stream window unavailable; retaining the trigger snapshot", job.camera)
        sub_frames, start, end, sub_fps = [], job.ts, job.ts, cfg.STORE_FPS
        reason = "sub_stream_read_error"
    try:
        main = main_cap.get_clip_frames(cfg.CLIP_SECONDS) if main_cap is not None else None
    except Exception:
        log.exception("[%s] main-stream window unavailable; using whole sub frames", job.camera)
        main, reason = None, "main_stream_read_error"
    job.clip_fps = sub_fps
    if len(sub_frames) < 2:
        reason = reason or "too_few_sub_frames"
        # A reconnect may leave an empty ring. Preserve the triggering picture
        # so a capture gap cannot swallow the alert.
        if not sub_frames and job.snapshot is not None:
            sub_frames, start, end = [job.snapshot], job.ts, job.ts
    if sub_frames:
        job.snapshot = sub_frames[-1]
    clip = [(start + i * (end - start) / max(1, len(sub_frames) - 1), frame)
            for i, frame in enumerate(sub_frames)]
    job.crop_settings = vlm_crop.settings_from_config(cfg)
    if not reason:
        if main is None:
            reason = "no_main_stream"
        elif len(main[0]) < 2:
            reason = "too_few_main_frames"
        else:
            try:
                # The crop supplies conf/imgsz from cfg; only device comes from inference.
                def crop_detector(frame, **kwargs):
                    return detector(frame, **kwargs, **predict_args)

                job.crop = vlm_crop.crop_clip(crop_detector, job.crop_settings, sub_frames, start, end,
                                               main[0], main[1], name=job.camera)
                job.crop_fps = main[3]
                if job.crop is None:
                    reason = "no_trigger_class_detection_or_usable_crop"
            except Exception:
                log.exception("[%s] crop failed; using the whole sub-stream window", job.camera)
                reason = "crop_error"
    if job.crop is not None:
        job.input_meta = {"vlm_input": "crop"}
        frames = vlm_crop.sample_for_vlm(job.crop.frames, fps=job.crop_fps, sample_fps=cfg.VLM_SAMPLE_FPS)
    else:
        job.input_meta = {"vlm_input": "whole_frame_fallback", "vlm_fallback_reason": reason}
        log.warning("[%s] VLM whole_frame_fallback: %s", job.camera, reason)
        frames = vlm_crop.sample_for_vlm(sub_frames, fps=sub_fps, sample_fps=cfg.VLM_SAMPLE_FPS)
    return frames, clip


def _start_due_alerts(pending: List[AlertJob], now: float, cfg: Any, detector: Any,
                      streams: Dict[str, Any], main_caps: Dict[str, Any], predict_args: Dict[str, Any],
                      backend: Any, box_settings: Dict[str, Any], env: Dict[str, str], settings: AlertSettings,
                      assistant: Any, status: Any, production_dir: str, training_dir: str) -> Any:
    """Start the reserved VLM call once its post-roll is complete; consume each job once."""
    from .alert_clips import POST_SECONDS

    for job in list(pending):
        if now < job.ts + POST_SECONDS:
            continue
        frames, clip = _prepare_alert(job, cfg, detector, streams[job.camera].sub_cap,
                                      main_caps[job.camera], predict_args)
        pending.remove(job)
        thread = threading.Thread(target=_worker,
                                  args=(backend, box_settings, env, settings, job.camera, frames,
                                        assistant, job, status, job.alert_on), daemon=True)
        thread.start()
        threading.Thread(target=_save_clip, args=(job, clip, production_dir, training_dir, assistant),
                         daemon=True).start()
        return thread
    return None


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
            assistant: Any = None, job: Optional[AlertJob] = None, status: Any = None,
            alert_on: Optional[Sequence[str]] = None) -> None:
    """Run the VLM call + dispatch off the capture loop. Never raises out.

    With an *assistant*, a camera the owner has paused is not analysed at all:
    the clip is kept for training (the owner usually paused because they know
    who it is), nothing is sent. *job* receives the outcome for the clip's
    meta, and *status* (ai_status.AiStatus) what the box's window shows.
    """
    labels = job.labels if job is not None else []
    alert_on = tuple(alert_on or settings.alert_on)   # this camera's own choice, else the house default
    try:
        # A job admitted before the alert window closes keeps its reserved call
        # after post-roll; the six-second wait must not silently discard it.
        now = datetime.fromtimestamp(job.ts) if job is not None and job.input_meta else datetime.now()
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
        lang = owner_language()
        raw, parsed = backend.analyze(frames, camera_name, int(time.time()),
                                      settings.alert_start_hour, settings.alert_end_hour, owner_language=lang)
        if job is not None and raw:
            # Everything a student model needs to learn this answer: the exact pictures,
            # the question, and the answer word for word.
            job.teacher = {
                "model": getattr(backend, "model_name", settings.vlm_model),
                "prompt_version": PROMPT_VERSION,
                "prompt": getattr(backend, "last_prompt", ""),
                "frames": list(backend.last_frame_jpegs) if isinstance(backend, GptBackend) else _jpegs(frames),
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
        why = str(parsed.get("why") or "").strip() if parsed else ""
        summary_owner = str(parsed.get("summary_owner") or "").strip() if parsed else ""
        people = _int_or_none(parsed.get("people")) if parsed else None
        raw_label = str((parsed or {}).get("label") or "").strip().lower()
        shown_label = raw_label if raw_label in LABELS else ""    # no valid label: "activity", loud
        if vlm_confirms(parsed, alert_on) is False:
            # The detector fired, the VLM looked and saw nothing the owner alerts on (no
            # person, nothing moving, or only a car when the owner wants people): no
            # message to the owner. The clip is kept as a false positive, for training.
            log.info("[%s] no alert: the VLM saw nothing to alert on (alert_on=%s): %s",
                     camera_name, ",".join(alert_on), summary)
            if job is not None:
                job.false_positive = True
                job.alert = {"summary": summary, "label": label, "alert_command": "[none]", "alert_reason": "",
                             "labels": job.labels, "false_positive": True,
                             "alert_on": list(alert_on),
                             "vlm": {"people": parsed.get("people"), "vehicle_moving": parsed.get("vehicle_moving"),
                                     "animals": parsed.get("animals")}}
            if status is not None:
                status.decision(camera_name, labels, summary, "[none]", sent=False, false_positive=True, label=label)
            return
        log.info("[%s] alert=%s label=%s summary=%s", camera_name, cmd, label, summary)
        # Attach the most recent frame of the clip as the alert snapshot.
        image = b""
        try:
            snapshot = job.snapshot if job is not None and job.snapshot is not None else (frames[-1] if frames else None)
            image = frame_to_jpeg_bytes(snapshot) if snapshot is not None else b""
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] could not encode snapshot: %s", camera_name, exc)
        muted = bool(assistant is not None and assistant.is_muted(camera_name))
        silent = is_silent(shown_label)
        if muted:
            res: Dict[str, Any] = {"sent": False, "reason": "paused by the owner"}
            log.info("[%s] alert not sent: the owner paused alerts", camera_name)
        else:
            from .telegram_notify import graded_alert_text  # noqa: PLC0415

            alert_ref = None
            if job is not None:
                alert_ref = {"alert_id": job.stem, "camera": camera_name, "summary": summary, "label": label,
                             "ts": job.ts}
            graded = graded_alert_text(shown_label, camera_name, owner_summary(summary, summary_owner, lang),
                                       why, lang)
            res = dispatch_alert(box_settings, env, cmd, f"{camera_name}: {alert_summary(label, summary)}", reason,
                                 image=image or None, assistant=assistant, alert=alert_ref, graded=graded,
                                 silent=silent, lang=lang)
            log.info("[%s] alert dispatched: %s", camera_name, res)
            if label == "escalation" and assistant is not None and alert_ref and delivery(res)[0]:
                try:
                    assistant.remind_if_silent(alert_ref, graded, lang)
                except Exception as exc:  # noqa: BLE001 - a missed reminder must not lose the alert's record
                    log.warning("[%s] escalation reminder not scheduled: %s", camera_name, exc)
        if status is not None:
            sent, why_not = delivery(res)
            status.decision(camera_name, labels, summary, cmd, sent=sent, muted=muted, error=why_not, label=label)
        if job is not None:
            job.alert = {"summary": summary, "label": label, "alert_command": cmd, "alert_reason": reason,
                         "labels": job.labels, "muted": muted, "dispatch": res,
                         "why": why, "summary_owner": summary_owner, "silent": silent, "people": people}
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
        clip_options = dict(fps=job.clip_fps, crop=job.crop, crop_settings=job.crop_settings, crop_fps=job.crop_fps)
        if job.false_positive:
            meta = write_alert_clip(training_dir, job.camera, false_positive_stem(job.camera, job.ts), frames,
                                    alert, kind="false_positive", teacher=job.teacher, extra=job.input_meta, **clip_options)
        elif job.paused:
            meta = write_alert_clip(training_dir, job.camera, f"{job.camera}_{int(job.ts)}_paused", frames,
                                    alert, kind="paused", extra=job.input_meta, **clip_options)
        else:
            meta = write_alert_clip(production_dir, job.camera, job.stem, frames, alert,
                                    extra={"trigger_ts": job.ts, "mode": "guard", **job.input_meta}, **clip_options)
            # The owner's copy above expires in two weeks; the training set keeps every
            # alert with the teacher's answer, so a student model can be trained on it.
            write_alert_clip(training_dir, job.camera, job.stem, frames, alert, kind="alert", teacher=job.teacher,
                             extra=job.input_meta, **clip_options)
        if meta:
            log.info("[%s] clip saved: %s (%d frames)", job.camera, os.path.basename(meta), len(frames))
        if (meta and assistant is not None and not job.false_positive and not job.paused
                and delivery(alert.get("dispatch") or {})[0]):
            res = assistant.send_clip(job.stem, clip_file(production_dir, meta), silent=bool(alert.get("silent")))
            log.info("[%s] video %s", job.camera, "sent" if res.get("sent") else f"not sent: {res}")
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] could not save the clip %s: %s", job.camera, job.stem, exc)


def load_detector(model_path: str, device: str = "auto") -> Tuple[Any, Optional[str]]:
    """The YOLO detector, and the device to pass to predict(): the Intel graphics chip when it works, else the CPU.

    On the N150 the graphics chip (through OpenVINO) runs yolo11s in ~65 ms a picture, the
    CPU (PyTorch) in ~550 ms: with five cameras that is a look at each camera about every
    0.35 s instead of every 3.5 s, and the CPU is left for the cameras' video. The OpenVINO
    copy of the model is made once, next to the .pt file. Anything that goes wrong (no
    openvino, no graphics driver, a broken copy) leaves the detector on the CPU.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    if device == "auto" and model_path.endswith(".pt"):
        ov_dir = os.path.join(os.path.dirname(model_path), os.path.splitext(os.path.basename(model_path))[0] + "_openvino_model")
        copied = False   # the failure came from the OpenVINO copy itself (not a missing chip or package)
        try:
            import numpy as np  # noqa: PLC0415
            import openvino as ov  # noqa: PLC0415

            if not any(d.startswith("GPU") for d in ov.Core().available_devices):
                raise RuntimeError("no Intel graphics chip")
            copied = True
            if not os.path.isfile(os.path.join(ov_dir, "metadata.yaml")):   # written last by the export
                log.info("Preparing %s for the Intel graphics chip (once, about 10 s)...", model_path)
                YOLO(model_path).export(format="openvino", imgsz=640, verbose=False)
            model = YOLO(ov_dir, task="detect")
            model.predict(np.zeros((576, 704, 3), dtype=np.uint8), device="intel:gpu", verbose=False)  # compile + warm up
            log.info("Detector: %s on the Intel graphics chip (OpenVINO)", model_path)
            return model, "intel:gpu"
        except Exception as exc:  # noqa: BLE001 - the CPU always works
            log.warning("Intel graphics chip not used (%s); the detector runs on the CPU", exc)
            if copied and os.path.isdir(ov_dir):
                import shutil  # noqa: PLC0415
                shutil.rmtree(ov_dir, ignore_errors=True)   # a broken copy: made again at the next start
    log.info("Detector: %s on the CPU", model_path)
    return YOLO(model_path), None


def serve_without_cameras(box_settings: Dict[str, Any], env: Dict[str, str], turned_off: Sequence[str],
                          start_assistant: Optional[Callable[..., Any]] = None,
                          keep_running: Callable[[], bool] = lambda: True,
                          sleep: Callable[[float], None] = time.sleep) -> int:
    """No camera is on: run only the owner's assistant, until the runner restarts us.

    *turned_off* are the cameras the owner can turn back on. Turning one on (Telegram,
    the app, the setup program) asks for a restart, and the program comes back
    watching it.
    """
    log.warning("No camera is on (turned off: %s). Nothing to watch; the assistant keeps listening "
                "so a camera can be turned back on.", ", ".join(turned_off) or "none")
    if start_assistant is None:
        from . import telegram_agent  # noqa: PLC0415

        start_assistant = telegram_agent.start
    try:
        start_assistant(box_settings, env, list(turned_off))
    except Exception as exc:  # noqa: BLE001 - without the assistant there is still nothing to watch
        log.warning("Owner assistant not started (%s).", exc)
    while keep_running():
        sleep(5.0)
    return 0


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
    box_settings = load_box_settings()
    settings = AlertSettings.from_box_settings(box_settings)
    cam_cfg = load_config()

    log.info("Inference mode starting. window=%02d:00-%02d:00 channel=%s backend=%s dry_run=%s",
             settings.alert_start_hour, settings.alert_end_hour, settings.alert_channel,
             settings.vlm_backend, settings.dry_run)

    backend = make_backend(settings, env)
    model, device = load_detector(settings.model, settings.device)
    predict_args = {"device": device} if device else {}

    # Camera sub-stream URLs from the data_collection config.
    cameras: Dict[str, str] = dict(getattr(cam_cfg, "CAMERAS", {}) or {})
    if not cameras:
        # Every camera is turned off (or none was found yet). Keep the owner's assistant
        # listening, so "turn the cameras back on" in Telegram still works: exiting here
        # left the owner without an answer while the runner restarted us every 15 s.
        from .alert_settings import camera_names  # noqa: PLC0415

        return serve_without_cameras(box_settings, env, camera_names())
    # Every alert is saved as a clip (the seconds around it) in the production folder,
    # and the owner can answer it in Telegram. Neither may stop the alerts themselves.
    from .alert_clips import POST_SECONDS, PRE_SECONDS, alert_stem  # noqa: PLC0415
    from .ai_status import AiStatus, objects_from_result  # noqa: PLC0415
    from .boxconfig import LIVE_DIR, LOG_DIR, PRODUCTION_LIVE_DIR  # noqa: PLC0415

    assert PRE_SECONDS + POST_SECONDS == cam_cfg.CLIP_SECONDS == 10, "alert and collector windows must match"

    zones = dict(getattr(cam_cfg, "ROI_ZONES", {}) or {})
    for name in cameras:
        if name in zones:
            log.info("[%s] watch zone active (%d corners); everything outside is blacked out", name, len(zones[name]))
    streams, main_caps = _camera_streams(cam_cfg)
    last_alert_ts: Dict[str, float] = {name: 0.0 for name in cameras}
    last_look_ts: Dict[str, float] = {name: 0.0 for name in cameras}
    vehicles: Dict[str, VehicleMemory] = {name: VehicleMemory() for name in cameras}
    parked: Dict[str, bool] = {name: False for name in cameras}  # "have not moved" already logged
    status = AiStatus(os.path.join(LOG_DIR, "ai_status.json"))  # what the box's window shows
    live = LiveSettings(settings, now=time.time())
    from .camera_alerts import LiveCameraAlerts  # noqa: PLC0415

    camera_alerts = LiveCameraAlerts(now=time.time())   # each camera's own alert types, re-read while running

    def reported_settings() -> Dict[str, Any]:
        return {**settings.live_values(),
                "camera_alert_on": {c: list(t) for c, t in sorted(camera_alerts.overrides.items())},
                "camera_sensitivity": {c: dict(t) for c, t in sorted(camera_alerts.sensitivity.items())}}

    status.settings(reported_settings())
    from .brain.mode import ModeWatch, status_line, switch_announcement  # noqa: PLC0415

    mode_watch = ModeWatch()
    worker = {"t": None}  # single in-flight VLM call across cameras (N150 budget)
    pending: List[AlertJob] = []
    assistant = None
    try:
        from . import telegram_agent  # noqa: PLC0415

        assistant = telegram_agent.start(box_settings, env, list(cameras))
    except Exception as exc:  # noqa: BLE001
        log.warning("Owner assistant not started (%s); alerts go out without feedback buttons.", exc)

    log.info("Watching %d camera(s): %s", len(cameras), ", ".join(cameras))
    started = time.time()   # no camera counts as offline before it had a minute to deliver its first picture
    while True:
        now_ts = time.time()
        settings_changed = bool(live.check(now_ts))
        if camera_alerts.check(now_ts) or settings_changed:
            status.settings(reported_settings())
        if mode_watch.due(now_ts):
            try:
                start, end = settings.alert_start_hour, settings.alert_end_hour
                switched = mode_watch.update(now_ts, start, end)
                for cam_name, stream in streams.items():
                    if stream.last_ts:
                        status.frame_seen(cam_name, stream.last_ts)
                offline = status.offline(now_ts, cameras=list(cameras), since=started)
                mute = getattr(assistant, "mute", None)
                paused = [(c, mute.muted_until(now_ts, c)) for c in cameras if mute and mute.muted_until(now_ts, c)]
                logging_on = bool(getattr(settings, "quiet_log", False))      # the setting arrives in Task 19
                status.mode(mode_watch.mode, status_line(mode_watch.mode, now_ts, start, end, paused, offline,
                                                         logging_on), now=now_ts)
                if switched and assistant is not None:
                    assistant.announce(switch_announcement(switched, now_ts, start, end, len(cameras) - len(offline),
                                                           len(cameras), owner_language(), logging_on))
            except Exception as exc:  # noqa: BLE001 - the status line must never stop the alerts
                log.warning("Mode status not updated: %s", exc)
        due_worker = _start_due_alerts(pending, now_ts, cam_cfg, model, streams, main_caps, predict_args,
                                       backend, box_settings, env, settings, assistant, status,
                                       PRODUCTION_LIVE_DIR, LIVE_DIR)
        if due_worker is not None:
            worker["t"] = due_worker
        for name in cameras:
            frame = streams[name].read()
            if frame is None:
                continue
            if time.time() - streams[name].last_ts > 5:   # a frozen camera: its last picture is not seen again
                continue
            now = datetime.now()
            if not in_alert_window(now.hour, settings.alert_start_hour, settings.alert_end_hour):
                continue
            # No new alert for this camera during its cooldown, or while a VLM call is
            # running. The detector still looks about once a second then, only so the
            # window can show what it sees.
            waiting = (now_ts - last_alert_ts[name] < settings.cooldown_sec
                       or bool(pending) or (worker["t"] is not None and worker["t"].is_alive()))
            if waiting and now_ts - last_look_ts[name] < STATUS_LOOK_SEC:
                continue
            last_look_ts[name] = now_ts

            # Each type is held to its own certainty (house value, or the camera's own);
            # the detector runs at the lowest of them and the rest are filtered here.
            thresholds = camera_alerts.thresholds_for(name, settings.thresholds())
            raw = model.predict(frame, conf=detector_floor(thresholds, settings.conf), verbose=False,
                                **predict_args)
            results = [filter_by_thresholds(raw[0], thresholds, settings.conf)] if raw else []
            seen_ts = time.time()   # when the picture was looked at, not when this round over the cameras began
            try:
                status.detection(name, objects_from_result(results[0]) if results else [], now=seen_ts)
            except Exception as exc:  # noqa: BLE001 - what the window shows must never stop the alerts
                log.debug("[%s] status not updated: %s", name, exc)
            # Every look feeds the camera's vehicle memory - the once-a-second looks of the
            # cooldown too - so a car that arrives and parks during the cooldown is compared
            # with its own parked position afterwards, and stays quiet.
            moved = vehicles[name].look(vehicle_boxes(results[0]) if results else [], now=seen_ts)
            if waiting:
                continue
            person, vehicle, labels = detect_trigger(results[0]) if results else (False, False, [])
            animal = has_animal(labels)
            if not (person or vehicle or animal):
                parked[name] = False
                continue
            alert_on = camera_alerts.for_camera(name, settings.alert_on)
            if not should_escalate(person, vehicle, moved, alert_on, animal=animal):
                if not parked[name]:  # one line per quiet spell, not one per look
                    log.info("[%s] %s; no alert", name, quiet_reason(vehicle, moved, alert_on))
                    parked[name] = True
                continue
            parked[name] = False
            trigger_ts = time.time()
            last_alert_ts[name] = trigger_ts
            log.info("[%s] escalating (labels=%s), waiting for the complete crop window", name, labels)
            status.thinking(name, labels, now=trigger_ts)
            job = AlertJob(camera=name, stem=alert_stem(name, trigger_ts), ts=trigger_ts, labels=labels,
                           snapshot=frame, alert_on=tuple(alert_on))
            pending.append(job)   # reserves the single VLM slot throughout post-roll
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
