"""Tolerant readers for the box's unversioned metadata documents."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal, Optional

from .keys import normalize_rel, parse_key, stem_kind


@dataclass
class AiRecord:
    status: Literal["real", "fallback", "failed", "none"]
    model: Optional[str]
    prompt_version: Optional[str]
    prompt: Optional[str]
    parsed: Optional[dict]
    raw_rel: Optional[str]
    input_frames_rel: list[str]


@dataclass
class ClipRecord:
    site: str
    root: Literal["dataset", "production"]
    camera: str
    stem: str
    kind: str
    start_ts: Optional[float]
    end_ts: Optional[float]
    trigger_ts: Optional[int]
    clip_rel: Optional[str]
    duration_sec: Optional[float]
    fps: Optional[float]
    frame_size: Optional[list[int]]
    codec: Optional[str]
    detected: list[str]
    class_max_conf: dict[str, float]
    summary: str
    label: Optional[str]
    alert_command: Optional[str]
    alert_reason: str
    dispatch: Optional[dict]
    muted: bool
    paused: bool
    ai: AiRecord
    sampled_frames: list[dict]
    owner_feedback: list[dict]
    problems: list[str]


@dataclass
class FeedbackRecord:
    site: str
    alert_id: Optional[str]
    camera: Optional[str]
    verdict: str
    action: str
    note: str
    raw_text: str
    source: str
    time_utc: Optional[datetime]
    scope_camera: Optional[str]


@dataclass
class Heartbeat:
    site: str
    mode: Optional[str]
    host: Optional[str]
    time_utc: Optional[datetime]
    collector_running: Optional[bool]
    stopped: Optional[bool]
    disk_free_gb: Optional[float]
    newest_clip_utc: Optional[datetime]
    cameras: dict[str, Optional[datetime]]
    clips_outbox: int


def _number(value) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (ValueError, OverflowError):
            pass
    return None


def _utc(value) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


class _Fields:
    """Read typed JSON fields while collecting diagnostics for malformed values."""

    def __init__(self, body, problems: list[str], prefix: str = ""):
        self.problems = problems
        self.prefix = prefix
        if not isinstance(body, dict):
            if body is not None:
                problems.append(f"invalid {prefix.rstrip('.') or 'body'}")
            body = {}
        self.body = body

    def typed(self, field, expected, default=None):
        value = self.body.get(field)
        if value is None:
            return default
        if isinstance(value, expected):
            return value
        self.problems.append(f"invalid {self.prefix}{field}")
        return default

    def text(self, field, default=None):
        return self.typed(field, str, default)

    def number(self, field, required=False):
        value = self.body.get(field)
        result = _number(value)
        if result is None:
            if value is not None:
                self.problems.append(f"invalid {self.prefix}{field}")
            elif required:
                self.problems.append(f"missing {self.prefix}{field}")
        return result

    def child(self, field):
        return _Fields(self.body.get(field), self.problems, self.prefix + field + ".")

    def path(self, field):
        value = self.body.get(field)
        if value is None:
            return None
        result = normalize_rel(value)
        if result is None:
            self.problems.append(f"{self.prefix}{field} outside site")
        return result


def parse_meta(key: str, body: dict) -> ClipRecord:
    problems: list[str] = []
    fields = _Fields(body, problems)
    info = parse_key(key)
    if info is None or info.area != "meta":
        problems.append("invalid meta key")
    alert = fields.child("alert")
    yolo = fields.child("yolo")
    buffer = fields.child("buffer")
    teacher = fields.child("teacher")
    export = fields.child("yolo_export")
    response = fields.body.get("model_response")
    if response is not None and not isinstance(response, (dict, str)):
        problems.append("invalid model_response")
    if fields.body.get("teacher") is None and response is None:
        status = "none"
    elif (isinstance(response, dict) and response.get("summary") == ""
          and not any(value for name, value in response.items() if name != "summary")
          and teacher.body.get("raw_path") is None):
        status = "fallback"
    elif response is None:
        status = "failed"
    else:
        status = "real"
    parsed = {"text": response} if isinstance(response, str) else response if isinstance(response, dict) else None

    input_frames = []
    for index, path in enumerate(teacher.typed("input_frames", list, [])):
        normalized = normalize_rel(path)
        if normalized is None:
            problems.append(f"teacher.input_frames[{index}] outside site")
        else:
            input_frames.append(normalized)
    raw_rel = teacher.path("raw_path")
    if "raw_path" not in teacher.body:
        raw_rel = fields.path("model_raw_text_path")
        if "model_raw_text_path" not in fields.body:
            raw_rel = fields.path("model_response_path")
    prompt = teacher.text("prompt")
    if prompt is None:
        prompt = fields.text("prompt_used")
    ai = AiRecord(status, teacher.text("model"), teacher.text("prompt_version"),
                  prompt, parsed, raw_rel, input_frames)

    summary = alert.text("summary")
    if summary is None and isinstance(response, dict):
        summary = _Fields(response, problems, "model_response.").text("summary")
    if summary is None:
        summary = response if isinstance(response, str) else ""

    sampled_frames = []
    for index, frame in enumerate(export.typed("exported_frames", list, [])):
        prefix = f"yolo_export.exported_frames[{index}]"
        if not isinstance(frame, dict):
            problems.append(f"invalid {prefix}")
            continue
        normalized = dict(frame)
        frame_fields = _Fields(frame, problems, prefix + ".")
        for field in ("image_path", "label_path"):
            if field in frame:
                normalized[field] = frame_fields.path(field)
        sampled_frames.append(normalized)

    owner_feedback = []
    for index, feedback in enumerate(fields.typed("owner_feedback", list, [])):
        if isinstance(feedback, dict):
            owner_feedback.append(dict(feedback))
        else:
            problems.append(f"invalid owner_feedback[{index}]")
    frame_size = buffer.typed("store_size", list)
    if frame_size is not None and (len(frame_size) != 2 or any(type(v) is not int or v <= 0 for v in frame_size)):
        problems.append("invalid buffer.store_size")
        frame_size = None
    confidence = {}
    for name, value in yolo.typed("class_max_conf", dict, {}).items():
        number = _number(value)
        if isinstance(name, str) and number is not None:
            confidence[name] = number
        else:
            problems.append(f"invalid yolo.class_max_conf.{name}")
    detected = [name for name in yolo.typed("class_counts", dict, {}) if isinstance(name, str)]
    stem = info.stem or "" if info else ""
    camera = fields.text("camera_name") or (info.camera if info else None) or ""
    if not camera:
        problems.append("missing camera_name")
    kind = fields.text("kind") or (info.kind if info else None) or ""
    if kind == "fp":
        kind = "false_positive"
    if fields.body.get("clip_path") is None:
        problems.append("missing clip_path")
    return ClipRecord(
        site=info.site if info else "", root=info.root if info else "dataset",
        camera=camera, stem=stem, kind=kind,
        start_ts=fields.number("clip_start_ts", required=True),
        end_ts=fields.number("clip_end_ts", required=True), trigger_ts=stem_kind(stem)[1],
        clip_rel=fields.path("clip_path"), duration_sec=fields.number("duration_sec"),
        fps=fields.number("fps_estimated"), frame_size=frame_size,
        codec=fields.text("codec"), detected=detected, class_max_conf=confidence,
        summary=summary, label=alert.text("label"), alert_command=alert.text("alert_command"),
        alert_reason=alert.text("alert_reason", ""), dispatch=alert.typed("dispatch", dict),
        muted=alert.typed("muted", bool, False), paused=alert.typed("paused", bool, False),
        ai=ai, sampled_frames=sampled_frames, owner_feedback=owner_feedback, problems=problems,
    )


def parse_feedback(key: str, body: dict) -> FeedbackRecord:
    fields = _Fields(body, [])
    alert = fields.child("alert")
    info = parse_key(key)
    return FeedbackRecord(
        site=info.site if info else "", alert_id=alert.text("alert_id"),
        camera=alert.text("camera"), verdict=fields.text("verdict", "none"),
        action=fields.text("action", "none"), note=fields.text("note", ""),
        raw_text=fields.text("raw_text", ""), source=fields.text("source", ""),
        time_utc=_utc(fields.body.get("time_utc")), scope_camera=fields.text("camera"),
    )


def parse_heartbeat(body: dict) -> Heartbeat:
    fields = _Fields(body, [])
    cameras = {}
    for name, camera in fields.typed("cameras", dict, {}).items():
        if isinstance(name, str):
            cameras[name] = _utc(camera.get("newest_clip_utc")) if isinstance(camera, dict) else None
    outbox = fields.body.get("clips_outbox")
    return Heartbeat(
        site=fields.text("site", ""), mode=fields.text("mode"), host=fields.text("host"),
        time_utc=_utc(fields.body.get("time_utc")),
        collector_running=fields.typed("collector_running", bool),
        stopped=fields.typed("stopped", bool), disk_free_gb=fields.number("disk_free_gb"),
        newest_clip_utc=_utc(fields.body.get("newest_clip_utc")), cameras=cameras,
        clips_outbox=outbox if type(outbox) is int and outbox >= 0 else 0,
    )
