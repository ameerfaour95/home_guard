"""Frozen API contract for the Home Guard Admin Center. Names and types are law."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["admin", "support", "labeler"]
Verdict = Literal["healthy", "warning", "critical", "offline", "unknown"]
EventKind = Literal["alert", "false_positive", "paused", "owner_feedback", "trigger", "random", "unknown"]
AiStatus = Literal["real", "fallback", "failed", "none"]
BoxesStatus = Literal["captured", "sampled", "recomputed", "none"]


class LoginRequest(BaseModel):
    email: str
    password: str
    totp: str


class StaffOut(BaseModel):
    id: int
    email: str
    name: str
    role: Role


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    staff: StaffOut


class RefreshRequest(BaseModel):
    refresh_token: str


class HealthReason(BaseModel):
    code: str
    message: str
    severity: Verdict


class CameraHealth(BaseModel):
    name: str
    newest_clip_utc: Optional[datetime]
    clips_waiting: int = 0
    stale: bool


class DeviceSummary(BaseModel):
    device_id: str
    site: str
    customer_id: int
    customer_name: str
    verdict: Verdict
    reasons: list[HealthReason]
    last_seen_utc: Optional[datetime]
    mode: Optional[str]
    host: Optional[str]
    cameras_total: int
    cameras_stale: int
    disk_free_gb: Optional[float]
    collector_running: Optional[bool]
    stopped: Optional[bool]
    newest_clip_utc: Optional[datetime]
    events_24h: int
    alerts_24h: int
    false_alarms_7d: int


class FleetResponse(BaseModel):
    devices: list[DeviceSummary]
    generated_utc: datetime


class CustomerIn(BaseModel):
    name: str
    timezone: str = "Asia/Jerusalem"
    consent_live: bool = False
    consent_recordings: bool = False
    consent_training: bool = False
    notes: str = ""


class CustomerOut(CustomerIn):
    id: int
    devices: list[DeviceSummary] = []


class EnrollRequest(BaseModel):
    customer_id: int
    site: str = Field(pattern=r"^[a-z0-9_]+$")
    tailscale_host: str = ""
    ssh_user: str = "ameer"


class Completeness(BaseModel):
    video: bool
    boxes: BoxesStatus
    ai: AiStatus
    owner_feedback: bool
    expired: bool
    copies: list[Literal["production", "training"]]


class EventSummary(BaseModel):
    id: int
    site: str
    customer_id: int
    customer_name: str
    camera: str
    kind: EventKind
    start_utc: datetime
    end_utc: Optional[datetime]
    summary: str
    label: Optional[str]
    alert_command: Optional[str]
    detected: list[str]
    owner_verdicts: list[str]
    completeness: Completeness
    reviewed: bool
    flagged: bool
    thumbnail_url: Optional[str]
    timezone: str = "UTC"


class EventPage(BaseModel):
    items: list[EventSummary]
    next_cursor: Optional[str]


class AiRunOut(BaseModel):
    id: int
    purpose: Literal["guard"]
    status: AiStatus
    model: Optional[str]
    prompt_version: Optional[str]
    prompt: Optional[str]
    parsed: Optional[dict]
    raw_text_artifact_id: Optional[int]
    input_frame_artifact_ids: list[int]


class FeedbackOut(BaseModel):
    id: int
    verdict: str
    action: str
    note: str
    raw_text: str
    source: str
    received_utc: datetime


class ArtifactOut(BaseModel):
    id: int
    role: str
    s3_key: str
    bytes: Optional[int]
    available: bool
    provenance: str
    detail: Optional[dict] = None


class DispatchOut(BaseModel):
    channel: Optional[str]
    sent: Optional[bool]
    detail: dict


class EventDetail(EventSummary):
    clip_start_local: Optional[str]
    duration_sec: Optional[float]
    fps: Optional[float]
    frame_size: Optional[list[int]]
    alert_reason: str
    dispatch: Optional[DispatchOut]
    ai_runs: list[AiRunOut]
    feedback: list[FeedbackOut]
    artifacts: list[ArtifactOut]
    raw_meta: dict  # newest raw revision, for the "raw" tab


class Box(BaseModel):
    cls: int
    label: str
    conf: Optional[float]
    xyxy: list[float]  # normalised 0..1


class FrameBoxes(BaseModel):
    frame_index: int
    t_sec: float
    status: Literal["ran", "ran_empty", "not_run"]
    boxes: list[Box]


class DetectionsOut(BaseModel):
    provenance: BoxesStatus
    model: Optional[str]
    frames: list[FrameBoxes]


class MediaAccessRequest(BaseModel):
    purpose: Literal["review", "support", "training"]


class MediaAccess(BaseModel):
    url: str
    expires_utc: datetime
    mime: str


class ReviewUpdate(BaseModel):
    reviewed: Optional[bool] = None
    flagged: Optional[bool] = None


class SavedFilter(BaseModel):
    key: str
    title: str
    description: str
    builtin: bool
    query: dict


class CollectionIn(BaseModel):
    name: str
    description: str = ""


class CollectionOut(CollectionIn):
    id: int
    event_count: int
    created_by: str
    created_utc: datetime


class CollectionItems(BaseModel):
    event_ids: list[int]


class ExportRequest(BaseModel):
    collection_id: int
    name: str = Field(pattern=r"^[a-z0-9_-]+$")
    formats: list[Literal["yolo", "vlm_jsonl", "clips"]]
    split: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
    include_fallback_ai: bool = False


class ExportOut(BaseModel):
    id: int
    name: str
    version: int
    state: Literal["queued", "running", "ready", "failed", "partial"]
    item_count: int
    s3_prefix: str
    manifest_url: Optional[str]
    error: Optional[str]
    created_utc: datetime
    created_by: str


class AuditEntry(BaseModel):
    id: int
    ts: datetime
    staff: str
    action: str
    customer_id: Optional[int]
    device_id: Optional[str]
    target: str
    reason: str


class AuditPage(BaseModel):
    items: list[AuditEntry]
    next_cursor: Optional[str]


class IndexProblem(BaseModel):
    s3_key: str
    reason: str
    seen_utc: datetime


class DensityRow(BaseModel):
    camera: str
    events: list[int]
    alerts: list[int]
    false_alarms: list[int]


class DensityOut(BaseModel):
    bucket: Literal["hour", "day"]
    starts_utc: list[datetime]
    timezone: str
    rows: list[DensityRow]


class ReviewCount(BaseModel):
    unreviewed_24h: int
    flagged_open: int
