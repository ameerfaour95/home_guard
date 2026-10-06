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
AiStatus = Literal['real', 'fallback', 'failed', 'none', 'unknown']
BoxesStatus = Literal['captured', 'sampled', 'recomputed', 'none', 'unknown']
EventKind = Literal['alert', 'false_positive', 'paused', 'owner_feedback', 'trigger', 'random', 'unknown']


@dataclass
class IndexProblem:
    s3_key: str
    reason: str
    seen_utc: datetime


@dataclass
class StaffOut:
    id: int
    email: str
    name: str
    role: Role


@dataclass
class TokenPair:
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
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
    consent_proposed: dict | None = None  # what the box's setup recorded (information only)
    consent_source: str = 'contract'      # 'contract', or 'withdrawn' once an admin switched a consent off


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
    timezone: str = field(default='UTC', kw_only=True)
    display_name: str | None = field(default=None, kw_only=True)
    annotation_status: str | None = field(default=None, kw_only=True)


@dataclass
class EventPage:
    items: list[EventSummary]
    next_cursor: str | None
    total: int | None = None
    total_capped: bool = False


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
    detail: dict | None = None


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


@dataclass
class Box:
    cls: int
    label: str
    conf: float | None
    xyxy: list[float]


@dataclass
class FrameBoxes:
    frame_index: int
    t_sec: float
    status: Literal['ran', 'ran_empty', 'not_run']
    boxes: list[Box]


@dataclass
class DetectionsOut:
    provenance: BoxesStatus
    model: str | None
    frames: list[FrameBoxes]


@dataclass
class MediaAccess:
    url: str
    expires_utc: datetime
    mime: str


@dataclass
class SavedFilter:
    key: str
    title: str
    description: str
    builtin: bool
    query: dict


@dataclass
class CollectionOut:
    name: str
    id: int
    event_count: int
    created_by: str
    created_utc: datetime
    description: str = ''


@dataclass
class ExportOut:
    id: int
    name: str
    version: int
    state: Literal['queued', 'running', 'ready', 'failed', 'partial']
    item_count: int
    s3_prefix: str
    manifest_url: str | None
    error: str | None
    created_utc: datetime
    created_by: str


@dataclass
class ExportExclusion:
    event_id: int
    reason: str


@dataclass
class ExportPreview:
    included_ids: list[int]
    excluded: list[ExportExclusion]
    split_counts: dict
    groups: int
    warnings: list[str]


@dataclass
class AuditEntry:
    id: int
    ts: datetime
    staff: str
    action: str
    customer_id: int | None
    device_id: str | None
    target: str
    reason: str
    detail: dict | None = None


@dataclass
class AuditPage:
    items: list[AuditEntry]
    next_cursor: str | None


@dataclass
class DensityRow:
    camera: str
    events: list[int]
    alerts: list[int]
    false_alarms: list[int]


@dataclass
class DensityOut:
    bucket: Literal['hour', 'day']
    starts_utc: list[datetime]
    timezone: str
    rows: list[DensityRow]


@dataclass
class ReviewCount:
    unreviewed_24h: int
    flagged_open: int


@dataclass
class Keyframe:
    frame: int
    t_sec: float
    xyxy: list[float]
    enabled: bool = True


@dataclass
class Track:
    track_id: str
    label: str
    keyframes: list[Keyframe]
    source: Literal['human', 'yolo', 'suggestion'] = 'human'  # yolo: preloaded detector boxes nobody checked yet


@dataclass
class AnnotationIn:
    base_version: int
    tracks: list[Track]
    description: str
    drop_clip: bool = False
    needs_review: bool = False
    status: Literal['edited', 'submitted'] = 'edited'


@dataclass
class AnnotationOut:
    event_id: int
    version: int
    status: str
    tracks: list[Track]
    description: str
    ai_description: str
    ai_status: AiStatus
    ai_model: str | None
    ai_prompt_version: str | None
    drop_clip: bool
    needs_review: bool
    author: str | None
    updated_utc: datetime | None
    fps: float | None
    frame_count: int | None
    frame_size: list[int] | None
    suggestions_used: bool
    review_note: str = ''
    review_frame: int | None = None


@dataclass
class ReviewDecision:
    decision: Literal['accept', 'reject']
    note: str = ''
    frame: int | None = None
    version: int | None = None  # the annotation version on screen; the server answers 409 when it is not current


@dataclass
class AnnotationVersion:
    version: int
    status: str
    author: str | None
    created_utc: datetime
    tracks_count: int
    description_changed: bool


@dataclass
class PublishMissing:
    event_id: int
    reason: str


@dataclass
class PublishOut:
    batch_name: str
    s3_prefix: str
    state: str
    tasks: int
    yolo_frames: int
    vlm_lines: int
    missing: list[PublishMissing]
    created_utc: datetime
    created_by: str


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
            if cls is not Role and isinstance(value, str):
                return 'unknown'
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
