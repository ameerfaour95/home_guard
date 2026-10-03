"""SQLAlchemy 2 models (spec section 5, phase-1 subset)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (BigInteger, Boolean, Computed, DateTime, Float, ForeignKey, Index, Integer, JSON,
                        String, Text, UniqueConstraint, text)
from sqlalchemy import false as sa_false, true as sa_true
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSONType = JSON().with_variant(postgresql.JSONB, "postgresql")
TS = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class Staff(Base):
    __tablename__ = "staff"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32))
    password_hash: Mapped[str] = mapped_column(Text)
    totp_secret: Mapped[str] = mapped_column(String(64))
    disabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    totp_last_counter: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    staff_id: Mapped[int] = mapped_column(ForeignKey("staff.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True)
    family: Mapped[str] = mapped_column(String(64), index=True, default="", server_default="")
    expires_at: Mapped[datetime] = mapped_column(TS)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    family_started_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Jerusalem", server_default="Asia/Jerusalem")
    consent_live: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    consent_recordings: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    consent_training: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    notes: Mapped[str] = mapped_column(Text, default="", server_default="")


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(primary_key=True)
    device_id: Mapped[str] = mapped_column(String(36), unique=True)
    site: Mapped[str] = mapped_column(String(64), unique=True)
    tailscale_host: Mapped[str] = mapped_column(String(255), default="", server_default="")
    ssh_user: Mapped[str] = mapped_column(String(64), default="ameer", server_default="ameer")
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    enrolled_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    last_heartbeat: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    last_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (UniqueConstraint("device_pk", "name"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    device_pk: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("device_pk", "camera", "stem"),
        Index("ix_events_device_start", "device_pk", text("start_ts DESC"), text("id DESC")),
        Index("ix_events_camera_start", "camera", text("start_ts DESC"), text("id DESC")),
        Index("ix_events_completeness", "completeness", postgresql_using="gin"),
        Index("ix_events_search", "search", postgresql_using="gin"),
        Index("ix_events_search_redacted", "search_redacted", postgresql_using="gin"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    device_pk: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
    site: Mapped[str] = mapped_column(String(64))
    camera: Mapped[str] = mapped_column(String(128))
    stem: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(32), default="unknown", server_default="unknown")
    start_ts: Mapped[float] = mapped_column(Float)
    end_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trigger_ts: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    day: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="", server_default="")
    label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    alert_command: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    alert_reason: Mapped[str] = mapped_column(Text, default="", server_default="")
    detected: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    class_max_conf: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    owner_verdicts: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    dispatch: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    muted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # production copy only; None = unknown
    completeness: Mapped[Any] = mapped_column(JSONType, default=dict, server_default=text("'{}'::jsonb"))
    clip_start_local: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frame_size: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    search: Mapped[Optional[Any]] = mapped_column(
        postgresql.TSVECTOR, Computed("to_tsvector('simple', coalesce(summary,''))", persisted=True), nullable=True)
    # the summary with the household's names replaced (redact.py); labelers search this, never `search`
    summary_redacted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    search_redacted: Mapped[Optional[Any]] = mapped_column(
        postgresql.TSVECTOR, Computed("to_tsvector('simple', coalesce(summary_redacted,''))", persisted=True),
        nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        Index("ix_artifacts_s3_key_prefix", "s3_key", postgresql_ops={"s3_key": "text_pattern_ops"}),
        Index("ix_artifacts_camera_stem", "camera", "stem"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), nullable=True, index=True)
    # role: original_video|thumbnail|rendition|filmstrip|meta|raw_answer|teacher_frame|yolo_image|yolo_label|feedback
    role: Mapped[str] = mapped_column(String(32))
    s3_key: Mapped[str] = mapped_column(String(1024), unique=True)
    etag: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    mime: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    available: Mapped[bool] = mapped_column(Boolean, default=True, server_default=sa_true())
    provenance: Mapped[str] = mapped_column(String(16), default="box", server_default="box")  # box|cloud
    detail: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    camera: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    stem: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_modified: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    # etag = what S3 lists now; applied_etag = the JSON revision whose content is in effect
    applied_etag: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    etag_mismatches: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))


class RawRevision(Base):
    __tablename__ = "raw_revisions"
    __table_args__ = (UniqueConstraint("s3_key", "etag"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    s3_key: Mapped[str] = mapped_column(String(1024))
    etag: Mapped[str] = mapped_column(String(128))
    fetched_at: Mapped[datetime] = mapped_column(TS)
    body: Mapped[Any] = mapped_column(JSONType)


class AiRun(Base):
    __tablename__ = "ai_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    purpose: Mapped[str] = mapped_column(String(32), default="guard", server_default="guard")
    status: Mapped[str] = mapped_column(String(16), default="none", server_default="none")
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    raw_artifact_id: Mapped[Optional[int]] = mapped_column(ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True)
    input_artifact_ids: Mapped[Any] = mapped_column(JSONType, default=list, server_default=text("'[]'::jsonb"))
    ai_source_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    ai_source_etag: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    device_pk: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    alert_stem: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"), nullable=True, index=True)
    verdict: Mapped[str] = mapped_column(String(64), default="", server_default="")
    action: Mapped[str] = mapped_column(String(64), default="", server_default="")
    note: Mapped[str] = mapped_column(Text, default="", server_default="")
    raw_text: Mapped[str] = mapped_column(Text, default="", server_default="")
    source: Mapped[str] = mapped_column(String(64), default="", server_default="")
    scope_camera: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    received_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    s3_key: Mapped[str] = mapped_column(String(1024), unique=True)


class ReviewState(Base):
    __tablename__ = "review_state"
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    by: Mapped[Optional[int]] = mapped_column(ForeignKey("staff.id", ondelete="SET NULL"), nullable=True)
    at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class Collection(Base):
    __tablename__ = "collections"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_by: Mapped[Optional[int]] = mapped_column(ForeignKey("staff.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    # migration 0008: set at creation when a labeler made it (their id), never changed afterwards. A private
    # collection is seen only by that staff member and by non-labelers, whatever roles change later. No FK: it
    # must stay private even if the staff row goes.
    private_to_staff_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class CollectionItem(Base):
    __tablename__ = "collection_items"
    collection_id: Mapped[int] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    added_by: Mapped[Optional[int]] = mapped_column(ForeignKey("staff.id", ondelete="SET NULL"), nullable=True)
    added_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class Export(Base):
    __tablename__ = "exports"
    __table_args__ = (UniqueConstraint("name", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    state: Mapped[str] = mapped_column(String(16), default="queued", server_default="queued")
    item_count: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    s3_prefix: Mapped[str] = mapped_column(String(1024), default="", server_default="")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request: Mapped[Any] = mapped_column(JSONType, default=dict, server_default=text("'{}'::jsonb"))
    created_by: Mapped[int] = mapped_column(ForeignKey("staff.id"))
    created_at: Mapped[datetime] = mapped_column(TS)
    # the lease of the worker building it (migration 0006): claimed atomically, refreshed while it runs
    worker_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class AuditLog(Base):
    """Append-only: DB triggers reject UPDATE, DELETE and TRUNCATE."""
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_action_ts", "action", "ts"),
        # view de-duplication: "has this staff seen this target in the last window?" (migration 0007)
        Index("ix_audit_log_staff_action_target_ts", "staff_id", "action", "target", "ts"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(TS)
    staff_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # no FK: log outlives staff
    staff_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # snapshot at write time
    action: Mapped[str] = mapped_column(String(64))
    customer_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    device_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    target: Mapped[str] = mapped_column(String(1024), default="", server_default="")
    reason: Mapped[str] = mapped_column(Text, default="", server_default="")
    detail: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)


class IndexProblem(Base):
    __tablename__ = "index_problems"
    __table_args__ = (
        Index("ix_index_problems_s3_key_prefix", "s3_key", postgresql_ops={"s3_key": "text_pattern_ops"}),
    )
    s3_key: Mapped[str] = mapped_column(String(1024), primary_key=True)
    reason: Mapped[str] = mapped_column(Text)
    seen_at: Mapped[datetime] = mapped_column(TS)
    etag: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)  # the revision it is about
    # media retries (migration 0007): attempts so far; next_retry_at NULL = not retried (permanent or given up)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class IdentityAlias(Base):
    """Every name a device's household has been known by (migration 0007): the customer name before and after
    each rename, the site, every camera name and owner display name, every host. Labeler redaction uses the union,
    so a rename never brings an old name back into what labelers see."""
    __tablename__ = "identity_aliases"
    __table_args__ = (UniqueConstraint("device_pk", "kind", "value"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    device_pk: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # customer|site|camera|display_name|host (|_scanned: marker)
    value: Mapped[str] = mapped_column(Text)
    first_seen: Mapped[datetime] = mapped_column(TS)


class S3Cursor(Base):
    __tablename__ = "s3_cursors"
    prefix: Mapped[str] = mapped_column(String(512), primary_key=True)
    last_full_scan: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class OwnerNotice(Base):
    __tablename__ = "owner_notices"
    id: Mapped[int] = mapped_column(primary_key=True)
    device_pk: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    staff_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    kind: Mapped[str] = mapped_column(String(64))
    cameras: Mapped[Any] = mapped_column(JSONType, default=list, server_default=text("'[]'::jsonb"))
    first_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s3_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    # True while the S3 notice object could not be written; cleared when a retry uploads it
    pending_upload: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
