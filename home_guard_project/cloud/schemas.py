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


class CameraOut(BaseModel):
    """One camera of a house as staff read it. `name` is the family's name when the box sent one, else "Camera N"
    from the channel (never the raw id); `camera` is the id, for filters. `current` is false for retired ids (a site
    rename, a removed camera): they are listed, never warned about."""
    customer_id: int
    device_id: str
    site: str
    camera: str
    name: str
    owner_named: bool
    current: bool
    newest_clip_utc: Optional[datetime]
    enabled: bool = True  # false: the box lists it but the owner switched it off


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
    needs_details: bool = False
    enrolled_by: Literal["admin", "setup", "discovered"] = "admin"
    app_version: Optional[str] = None
    # One box, many site names (cloud/boxes.box_lineage): an old site's row names the device it became; the current
    # row lists its old site names. The Fleet shows one row per box.
    replaced_by: Optional[str] = None
    replaced_by_site: Optional[str] = None
    old_sites: list[str] = []


class FleetResponse(BaseModel):
    devices: list[DeviceSummary]
    generated_utc: datetime


class CustomerIn(BaseModel):
    name: str
    timezone: str = "Asia/Jerusalem"
    consent_live: bool = True       # from the sales contract; an admin switches one off when the customer withdraws it
    consent_recordings: bool = True
    consent_training: bool = True
    notes: str = ""


class ConsentProposal(BaseModel):
    live: bool
    recordings: bool
    training: bool
    recorded_utc: datetime
    installer: str


class CustomerOut(CustomerIn):
    id: int
    consent_proposed: Optional[ConsentProposal] = None  # admin/support only: what the box's setup recorded (information)
    consent_source: Literal["contract", "withdrawn"] = "contract"
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
    annotation_status: Optional[str] = None  # new|edited|submitted|reviewed|rejected (None = never opened)
    # The box's event layer (fleet_contract.event_outcome): the session the clip belongs to, what happened to it
    # ("Sent", "Kept in the event, not sent (normal)", "Not sent: owner said known (...)", ...; null for a clip that
    # predates events) with its code (sent|undelivered|held|known|not_ours|muted|dismissed|none), and whether the
    # baseline in shadow mode would have raised it ("rare for this camera").
    session_id: Optional[str] = None
    outcome: Optional[str] = None
    outcome_code: Optional[str] = None
    would_raise: Optional[bool] = None
    ai_flags: list[str] = []  # what happened to the AI call: "Rescued at 768 px", "AI answer rescued", "AI failed"


class EventSession(BaseModel):
    """One event (the box's session: one ongoing activity at one camera) over its clips."""
    session_id: str
    site: str
    camera: str
    first_utc: datetime
    last_utc: datetime
    clips: int
    sent: int


class EventPage(BaseModel):
    items: list[EventSummary]
    next_cursor: Optional[str]
    total: Optional[int] = None  # only with with_total=true; capped at 10000 (then total_capped is true)
    total_capped: bool = False


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
    # the owner's Telegram tag (box feedback.py OWNER_LABELS), the owner's own words and a voice answer's
    # transcript (0017); a labeler reads the tag only
    owner_label: str = ""
    owner_text: str = ""
    transcript: str = ""


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


class ChatLine(BaseModel):
    """One line of the owner's Telegram conversation (the box's ChatFeed), camera ids replaced by owner names."""
    ts: datetime
    site: str
    who: str            # "box" | "owner" | "assistant"
    name: str
    kind: str           # "alert" | "video" | "message" | "button" | "answer"
    text: str
    camera: str
    camera_name: str
    alert_id: str
    image: str          # a picture's file name under production_<site>/chat/images/, or ""
    delivered: bool
    error: str


class ChatDay(BaseModel):
    day: Optional[str]  # the day shown; null for a search across days
    days: list[str]     # days uploaded, newest first
    messages: list[ChatLine]


class ChatImageRequest(BaseModel):
    site: str
    image: str


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


class ExportExclusion(BaseModel):
    event_id: int
    reason: Literal["no_training_consent", "video_unavailable", "no_real_ai", "expired", "dropped_by_labeler"]


class ExportPreview(BaseModel):
    included_ids: list[int]
    excluded: list[ExportExclusion]
    split_counts: dict[str, int]  # events per split after the group-aware split
    groups: int  # number of (site, day) groups
    warnings: list[str]


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
    detail: Optional[dict] = None


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


# ---------------------------------------------------------------- contract 2f: in-app labeling

class Keyframe(BaseModel):
    frame: int
    t_sec: float
    xyxy: list[float]
    enabled: bool = True


class Track(BaseModel):
    track_id: str
    label: str
    keyframes: list[Keyframe]
    # "yolo": preloaded detector boxes nobody has checked yet; "human" once a person edited (or kept) them;
    # "suggestion" is the older name of "yolo", still accepted
    source: Literal["human", "yolo", "suggestion"] = "human"
    # the object's stable name: P1 / CAR1 / A1 (the box's entity ids); never part of a YOLO training label
    entity: Optional[str] = Field(default=None, max_length=8, pattern=r"^(P|CAR|A)\d{1,3}$")


class AnnotationIn(BaseModel):
    base_version: int
    tracks: list[Track]
    description: str
    drop_clip: bool = False
    needs_review: bool = False
    status: Literal["edited", "submitted"]


class AnnotationOut(BaseModel):
    event_id: int
    version: int
    status: Literal["new", "edited", "submitted", "reviewed", "rejected"]
    tracks: list[Track]
    description: str
    ai_description: str
    ai_status: AiStatus
    ai_model: Optional[str]
    ai_prompt_version: Optional[str]
    drop_clip: bool
    needs_review: bool
    author: Optional[str]
    updated_utc: Optional[datetime]
    review_note: str = ""
    review_frame: Optional[int] = None
    fps: Optional[float]
    frame_count: Optional[int]
    frame_size: Optional[list[int]]
    suggestions_used: bool
    # what a never-saved clip's boxes were preloaded from: "tracker" (the box tracker's <stem>.tracks.json, P1 stays
    # P1), "dataset" (the unified dataset's labels), "yolo" (the box's weak YOLO labels linked by IoU); None when
    # nothing was preloaded or the clip was saved since
    preload_source: Optional[str] = None


class ReviewDecision(BaseModel):
    decision: Literal["accept", "reject"]
    note: str = ""
    frame: Optional[int] = None
    # the annotation version the reviewer looked at; the server requires it (422 without, 409 when it is not the
    # current version any more)
    version: Optional[int] = None


class AnnotationVersion(BaseModel):
    version: int
    status: str
    author: Optional[str]
    created_utc: datetime
    tracks_count: int
    description_changed: bool


class PublishRequest(BaseModel):
    batch_name: str = Field(pattern=r"^[a-z0-9_]+$")


class PublishMissing(BaseModel):
    event_id: int
    reason: str


class PublishOut(BaseModel):
    batch_name: str
    s3_prefix: str
    state: Literal["queued", "running", "ready", "failed", "partial"]
    tasks: int
    yolo_frames: int
    vlm_lines: int
    missing: list[PublishMissing]
    created_utc: datetime
    created_by: str


# ---------------------------------------------------------------- tagging studio (cloud/tagstudio)
# Clips are named by a key across sources ("ds:<clip id>", "ev:<event id>", "of:<prefix>/<stem>"). Opinions, tags and
# the taxonomy are open JSON objects: their vocabulary is fleet_contract/taxonomy.py, not this file.

class TaggingState(BaseModel):
    taxonomy: dict
    fields: dict[str, str]
    counts: dict[str, int]
    total: int
    sources: list[dict]
    teachers: list[dict]
    can_ask_teacher: bool
    paths: dict[str, str]


class TaggingQueueItem(BaseModel):
    key: str
    clip_id: str
    origin: str
    source: str = ""
    batch: str = ""
    camera: str = ""
    camera_display: str = ""            # what staff read: the box's camera name (an old id by its channel)
    date: str = ""
    duration_sec: Optional[float] = None
    local_time: Optional[str] = None
    sort_ts: float = 0.0
    event_id: Optional[int] = None
    tier: int
    tier_name: Literal["contradiction", "check", "untagged", "done"]
    reasons: list[str]
    alert: bool
    conflicts: list[dict]
    labels: dict[str, dict]
    has_media: bool = True


class TaggingQueue(BaseModel):
    items: list[TaggingQueueItem]
    count: int


class TaggingClip(BaseModel):
    item: dict
    media: dict[str, bool]
    media_reasons: dict[str, str] = {}   # why a video kind cannot be opened ("" when it can)
    fps: Optional[float] = None
    opinions: dict[str, dict]
    tag: Optional[dict] = None
    form: dict
    prefilled_from: str
    assessment: dict
    history: list[dict]
    prompt_version: str = ""            # the clip's prompt version: the tag follows its answer schema
    answer_schema: dict = {}            # {"kind": "legacy" | "eye", "name", "fields": the answer's fields in order}
    ai_badges: list[str] = []           # what happened to the AI call: "Rescued at 768 px", "AI failed", ...


class TagSave(BaseModel):
    key: str = Field(max_length=512)
    fields: dict


class TagSuggestRequest(BaseModel):
    key: str = Field(max_length=512)
    refresh: bool = False


class TagSuggestion(BaseModel):
    key: str
    model: str
    prompt_version: str = ""
    at: str
    fields: dict
    cached: bool


class ClipAnnotationIn(AnnotationIn):
    key: str = Field(max_length=512)


class TagSaved(BaseModel):
    tag: dict
    assessment: dict
    next_key: str


class TaggingMediaRequest(BaseModel):
    key: str = Field(max_length=512)
    kind: Literal["clip", "crop"]


class TaggingKey(BaseModel):
    key: str = Field(max_length=512)


class TagConvertRequest(BaseModel):
    key: str = Field(max_length=512)
    words: str = Field(max_length=2000)
    # whose words they are; the server checks it (customer words need that customer's training consent)
    words_source: Literal["staff", "owner_answer", "transcript"] = "staff"
    feedback_id: Optional[int] = None   # the owner answer the words came from (owner_answer / transcript)


class TagConversion(BaseModel):
    """"In my words" restructured into the clip's answer schema (tagstudio/convert.py): a suggestion for the form."""
    key: str
    model: str
    language: str
    prompt_version: str
    schema_name: str
    fields: dict
    raw: str
    model_id: str = ""                  # the exact model id that answered
    words_source: str = "staff"         # whose words, as the server decided
    words_sent: str = ""                # the words as sent: camera, house and customer names replaced


class ModelInputView(BaseModel):
    """What the AI saw of a clip (tagstudio/model_view.py): the JPEG frames, base64, in the order the model got them."""
    key: str
    label: str                          # "What the AI sees: crop · 1 fps · 10 frames · 384×384"
    source: Literal["sent", "recipe", "rendered", "whole"]
    source_title: str
    vlm_input: str
    times: list[Optional[float]]
    size: Optional[list[int]]
    sample_fps: float
    record: dict
    frames: list[str]
    prompt_version: str = ""
    max_side: Optional[int] = None      # the long-side cap the box applied (box.yaml vlm_max_side), when it did
    badges: list[str] = []              # "Rescued at 768 px", "AI answer rescued", "AI failed"


class TaggingExportRequest(BaseModel):
    include_needs_check: bool = False


class TaggingExportOut(BaseModel):
    training_path: str
    eval_path: str
    counts: dict[str, int]
    sharegpt_path: str = ""             # LLaMA-Factory sharegpt rows: the tag as the box's JSON answer


class TeacherAnswer(BaseModel):
    suggestion: Optional[dict] = None


# ---------------------------------------------------------------- the Inbox: owner answers from Telegram (0017)

class InboxItem(BaseModel):
    feedback_id: int
    event_id: int
    clip_key: str                      # the Tag · AI key of the clip ("ev:<event id>")
    customer_id: int
    customer: str
    site: str
    camera: str
    camera_name: Optional[str]         # the owner's name for the camera, when the box reported one
    received_utc: Optional[datetime]
    owner_label: str                   # normal / suspicious / escalation / empty / other / rule_mismatch, or ""
    owner_text: str
    transcript: str
    raw_text: str
    note: str
    verdict: str
    source: str
    tagged_by: str
    model_label: Optional[str]         # what the model said (the event's label) and its summary
    model_summary: str
    model: Optional[str]
    prompt_version: Optional[str]      # the prompt the clip's AI answer came from (meta teacher.prompt_version)
    probably_not_label: bool           # no tag, and the box's rule calls the words a question / complaint / command
    consent_training: bool
    decision: Optional[Literal["accepted", "fixed", "not_label"]]
    decided_by: Optional[str]
    decided_utc: Optional[datetime]
    decision_note: str = ""
    # the clip's earlier tags, each replaced by a later one: [{feedback_id, owner_label, owner_text, transcript,
    # received_utc, superseded_by}], oldest first
    history: list[dict] = []


class InboxDecisionIn(BaseModel):
    decision: Literal["accepted", "fixed", "not_label"]
    note: str = ""
