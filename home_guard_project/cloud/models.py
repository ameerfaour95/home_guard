"""SQLAlchemy 2 models (spec section 5, phase-1 subset)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (BigInteger, Boolean, Computed, DateTime, Float, ForeignKey, Index, Integer, JSON,
                        String, Text, UniqueConstraint)
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
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    staff_id: Mapped[int] = mapped_column(ForeignKey("staff.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True)
    family: Mapped[str] = mapped_column(String(64), index=True, default="")
    expires_at: Mapped[datetime] = mapped_column(TS)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Jerusalem")
    consent_live: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_recordings: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_training: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(primary_key=True)
    device_id: Mapped[str] = mapped_column(String(36), unique=True)
    site: Mapped[str] = mapped_column(String(64), unique=True)
    tailscale_host: Mapped[str] = mapped_column(String(255), default="")
    ssh_user: Mapped[str] = mapped_column(String(64), default="ameer")
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
        Index("ix_events_device_start", "device_pk", "start_ts"),
        Index("ix_events_camera_start", "camera", "start_ts"),
        Index("ix_events_completeness", "completeness", postgresql_using="gin"),
        Index("ix_events_search", "search", postgresql_using="gin"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    device_pk: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
    site: Mapped[str] = mapped_column(String(64))
    camera: Mapped[str] = mapped_column(String(128))
    stem: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(32), default="unknown")
    start_ts: Mapped[float] = mapped_column(Float)
    end_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trigger_ts: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    day: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    alert_command: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    alert_reason: Mapped[str] = mapped_column(Text, default="")
    detected: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    class_max_conf: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    owner_verdicts: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    dispatch: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    completeness: Mapped[Any] = mapped_column(JSONType, default=dict)
    clip_start_local: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frame_size: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    search: Mapped[Optional[Any]] = mapped_column(
        postgresql.TSVECTOR, Computed("to_tsvector('simple', coalesce(summary,''))", persisted=True), nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), nullable=True, index=True)
    # role: original_video|thumbnail|rendition|filmstrip|meta|raw_answer|teacher_frame|yolo_image|yolo_label|feedback
    role: Mapped[str] = mapped_column(String(32))
    s3_key: Mapped[str] = mapped_column(String(1024), unique=True)
    etag: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    mime: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    available: Mapped[bool] = mapped_column(Boolean, default=True)
    provenance: Mapped[str] = mapped_column(String(16), default="box")  # box|cloud
    detail: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    camera: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    stem: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_modified: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


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
    purpose: Mapped[str] = mapped_column(String(32), default="guard")
    status: Mapped[str] = mapped_column(String(16), default="none")
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    raw_artifact_id: Mapped[Optional[int]] = mapped_column(ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True)
    input_artifact_ids: Mapped[Any] = mapped_column(JSONType, default=list)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    device_pk: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    alert_stem: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"), nullable=True, index=True)
    verdict: Mapped[str] = mapped_column(String(64), default="")
    action: Mapped[str] = mapped_column(String(64), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(64), default="")
    scope_camera: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    received_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    s3_key: Mapped[str] = mapped_column(String(1024), unique=True)


class ReviewState(Base):
    __tablename__ = "review_state"
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    by: Mapped[Optional[int]] = mapped_column(ForeignKey("staff.id", ondelete="SET NULL"), nullable=True)
    at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


class Collection(Base):
    __tablename__ = "collections"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[Optional[int]] = mapped_column(ForeignKey("staff.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)


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
    version: Mapped[int] = mapped_column(Integer, default=1)
    state: Mapped[str] = mapped_column(String(16), default="queued")
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    s3_prefix: Mapped[str] = mapped_column(String(1024), default="")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request: Mapped[Any] = mapped_column(JSONType, default=dict)
    created_by: Mapped[int] = mapped_column(ForeignKey("staff.id"))
    created_at: Mapped[datetime] = mapped_column(TS)


class AuditLog(Base):
    """Append-only: a DB trigger rejects UPDATE and DELETE."""
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(TS)
    staff_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # no FK: log outlives staff
    action: Mapped[str] = mapped_column(String(64))
    customer_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    device_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    target: Mapped[str] = mapped_column(String(1024), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)


class IndexProblem(Base):
    __tablename__ = "index_problems"
    s3_key: Mapped[str] = mapped_column(String(1024), primary_key=True)
    reason: Mapped[str] = mapped_column(Text)
    seen_at: Mapped[datetime] = mapped_column(TS)


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
    cameras: Mapped[Any] = mapped_column(JSONType, default=list)
    first_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s3_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
