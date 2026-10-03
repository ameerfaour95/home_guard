"""Plain dataclasses matching phase 1 Task 2. No cloud/box imports.

The decoder validates the wire shape, including nested types and literals, and
ignores additional fields so a newer server can extend a response safely.
"""
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from datetime import datetime
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

Role = Literal['admin', 'support', 'labeler']
Verdict = Literal['healthy', 'warning', 'critical', 'offline', 'unknown']
AiStatus = Literal['real', 'fallback', 'failed', 'none']
BoxesStatus = Literal['captured', 'sampled', 'recomputed', 'none']
EventKind = Literal['alert', 'false_positive', 'paused', 'owner_feedback', 'trigger', 'random', 'unknown']


@dataclass
class StaffOut:
    id: int
    email: str
    name: str
    role: Role


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int
    staff: StaffOut


@dataclass
class HealthReason:
    code: str
    message: str
    severity: Verdict


@dataclass
class CameraHealth:
    name: str
    newest_clip_utc: datetime | None
    stale: bool
    clips_waiting: int = 0


@dataclass
class DeviceSummary:
    device_id: str
    site: str
    customer_id: int
    customer_name: str
    verdict: Verdict
    reasons: list[HealthReason]
    last_seen_utc: datetime | None
    mode: str | None
    host: str | None
    cameras_total: int
    cameras_stale: int
    disk_free_gb: float | None
    collector_running: bool | None
    stopped: bool | None
    newest_clip_utc: datetime | None
    events_24h: int
    alerts_24h: int
    false_alarms_7d: int


@dataclass
class FleetResponse:
    devices: list[DeviceSummary]
    generated_utc: datetime


@dataclass
class CustomerOut:
    name: str
    id: int
    timezone: str = 'Asia/Jerusalem'
    consent_live: bool = False
    consent_recordings: bool = False
    consent_training: bool = False
    notes: str = ''
    devices: list[DeviceSummary] = field(default_factory=list)


@dataclass
class Completeness:
    video: bool
    boxes: BoxesStatus
    ai: AiStatus
    owner_feedback: bool
    expired: bool
    copies: list[Literal['production', 'training']]


@dataclass
class EventSummary:
    id: int
    site: str
    customer_id: int
    customer_name: str
    camera: str
    kind: EventKind
    start_utc: datetime
    end_utc: datetime | None
    summary: str
    label: str | None
    alert_command: str | None
    detected: list[str]
    owner_verdicts: list[str]
    completeness: Completeness
    reviewed: bool
    flagged: bool
    thumbnail_url: str | None


@dataclass
class EventPage:
    items: list[EventSummary]
    next_cursor: str | None


@dataclass
class AiRunOut:
    id: int
    purpose: Literal['guard']
    status: AiStatus
    model: str | None
    prompt_version: str | None
    prompt: str | None
    parsed: dict | None
    raw_text_artifact_id: int | None
    input_frame_artifact_ids: list[int]


@dataclass
class FeedbackOut:
    id: int
    verdict: str
    action: str
    note: str
    raw_text: str
    source: str
    received_utc: datetime


@dataclass
class ArtifactOut:
    id: int
    role: str
    s3_key: str
    bytes: int | None
    available: bool
    provenance: str


@dataclass
class DispatchOut:
    channel: str | None
    sent: bool | None
    detail: dict


@dataclass
class EventDetail(EventSummary):
    clip_start_local: str | None
    duration_sec: float | None
    fps: float | None
    frame_size: list[int] | None
    alert_reason: str
    dispatch: DispatchOut | None
    ai_runs: list[AiRunOut]
    feedback: list[FeedbackOut]
    artifacts: list[ArtifactOut]
    raw_meta: dict


def decode(cls, value):
    """Parse JSON into a model, rejecting malformed or ambiguous wire values."""
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, UnionType):
        for option in args:
            try:
                return decode(option, value)
            except (ValueError, TypeError):
                pass
        raise ValueError('Invalid optional value')
    if origin is Literal:
        if value not in args:
            raise ValueError('Invalid enum value')
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ValueError('Expected array')
        return [decode(args[0], item) for item in value]
    if is_dataclass(cls):
        if not isinstance(value, dict):
            raise ValueError('Expected object')
        hints = get_type_hints(cls)
        result = {}
        for f in fields(cls):
            if f.name in value:
                result[f.name] = decode(hints[f.name], value[f.name])
            elif f.default is MISSING and f.default_factory is MISSING:
                raise ValueError(f'Missing field: {f.name}')
        return cls(**result)
    if cls is datetime:
        if not isinstance(value, str):
            raise ValueError('Expected ISO timestamp')
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None:
            raise ValueError('Timestamp must include timezone')
        return result
    if cls is float and type(value) in (int, float):
        return float(value)
    if type(value) is not cls:
        raise ValueError(f'Expected {cls.__name__}')
    return value
