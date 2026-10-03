"""Presigned media access. Every view is audited; recordings viewed by staff are also announced to the owner.

Presigned URLs are secrets for their lifetime: they are returned to the caller and never logged or audited.

A storage key names the household (`dataset_<site>/clips/<camera>/...`), and so would a URL presigned for it.
Labelers therefore never get the real key: on first access the file is copied server-side to
`admin_cache/opaque/<hmac(artifact id, etag)>.<ext>` (recorded as an `opaque_copy` artifact with provenance
"cloud", never shown to labelers) and that copy is presigned, without any filename hint. Owner documents and raw
AI text (meta, feedback, raw answers) are not media and are refused to labelers outright.

For a labeler, an artifact they may not see (no training consent, an owner document, an opaque copy, no known
household) is indistinguishable from a missing one: visibility is decided before existence, availability or any
consent-specific message, and the answer is always the same 404. The real reason is audited (`media_denied`);
the audit is not visible to labelers.
"""
from __future__ import annotations

import hashlib
import hmac
import mimetypes
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .. import audit
from ..deps import SessionDep, current_staff
from ..models import Artifact, AuditLog, Customer, Device, Event, Staff
from ..schemas import MediaAccess, MediaAccessRequest
from .events import _Viewer, _load_one

router = APIRouter(tags=["media"], dependencies=[Depends(current_staff)])

URL_TTL_SECONDS = 300
OPAQUE_PREFIX = "admin_cache/opaque/"
_OPAQUE_DOMAIN = b"home-guard-admin/opaque-media/v1"
LABELER_HIDDEN_ROLES = frozenset({"meta", "feedback", "raw_answer", "opaque_copy"})
THUMBNAIL_AUDIT_WINDOW = timedelta(minutes=10)
_MIME = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".json": "application/json", ".txt": "text/plain"}


def _mime(key: str, fallback: Optional[str]) -> str:
    name = key.rsplit("/", 1)[-1]
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _MIME.get(ext) or mimetypes.guess_type(name)[0] or fallback or "application/octet-stream"


def _s3(request: Request):
    s3 = request.app.state.s3
    if s3 is None:
        raise HTTPException(status_code=503, detail="Storage is not configured")
    return s3


def opaque_key(secret: str, art: Artifact) -> str:
    """Where a labeler's copy of `art` lives: a keyed digest of (artifact id, etag) and the extension only."""
    sub_key = hmac.new(secret.encode("utf-8"), _OPAQUE_DOMAIN, hashlib.sha256).digest()
    digest = hmac.new(sub_key, f"{art.id}:{art.etag or ''}".encode("utf-8"), hashlib.sha256).hexdigest()[:40]
    name = art.s3_key.rsplit("/", 1)[-1]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return f"{OPAQUE_PREFIX}{digest}.{ext if ext.isalnum() and len(ext) <= 8 else 'bin'}"


def _labeler_url(session: Session, request: Request, s3, art: Artifact, mime: str) -> str:
    """A presigned URL of the labeler's opaque copy of `art`, copying it on first access."""
    from botocore.exceptions import ClientError

    key = opaque_key(request.app.state.settings.jwt_secret, art)
    copy = session.scalar(select(Artifact).where(Artifact.s3_key == key))
    if copy is None or not copy.available:
        try:
            s3.copy(art.s3_key, key, content_type=mime)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                raise HTTPException(status_code=410, detail="This file is no longer available")
            raise
        values = dict(role="opaque_copy", s3_key=key, provenance="cloud", detail={"source_artifact_id": art.id},
                      bytes=art.bytes, mime=mime, available=True)
        session.execute(pg_insert(Artifact).values(**values).on_conflict_do_update(
            index_elements=[Artifact.s3_key], set_={"available": True, "detail": values["detail"]}))
    return s3.presign(key, URL_TTL_SECONDS)


def _device_of(session: Session, art: Artifact, ev: Optional[Event]) -> Optional[Device]:
    if ev is not None:
        return session.get(Device, ev.device_pk)
    device_pk = getattr(art, "device_pk", None)
    if device_pk is not None:
        return session.get(Device, device_pk)
    top = art.s3_key.split("/", 1)[0]  # dataset_<site> / production_<site>
    site = top.split("_", 1)[1] if "_" in top else None
    if not site:
        return None
    found = session.scalars(select(Device).where(Device.site == site).limit(2)).all()
    return found[0] if len(found) == 1 else None  # none or ambiguous: the caller answers 404


_NOT_FOUND = "Artifact not found"


def _deny(session: Session, staff: Staff, target: str, customer: Optional[Customer], device: Optional[Device],
          purpose: str, reason: str, status: int = 403, shown: Optional[str] = None):
    """Audit a refused access (committed now: the raise below would roll the request back), then refuse with
    `shown` (default: the reason itself)."""
    audit.record(session, staff.id, "media_denied", target=target, reason=reason,
                 customer_id=customer.id if customer is not None else None,
                 device_id=device.device_id if device is not None else None,
                 detail={"reason": reason, "purpose": purpose})
    session.commit()
    raise HTTPException(status_code=status, detail=shown or reason)


def _labeler_lookup(session: Session, artifact_id: int):
    """(artifact, event, device, customer) when a labeler may see the artifact, else None: one SELECT with the
    visibility predicates (event, device, training consent, openable role) in its WHERE, so a hidden artifact and a
    missing one cost exactly the same."""
    row = session.execute(
        select(Artifact, Event, Device, Customer)
        .join(Event, Event.id == Artifact.event_id)
        .join(Device, Device.id == Event.device_pk)
        .join(Customer, Customer.id == Device.customer_id)
        .where(Artifact.id == artifact_id, Customer.consent_training.is_(True),
               Artifact.role.not_in(sorted(LABELER_HIDDEN_ROLES)))).first()
    return tuple(row) if row is not None else None


@router.post("/artifacts/{artifact_id}/access", response_model=MediaAccess)
def artifact_access(artifact_id: int, body: MediaAccessRequest, request: Request,
                    staff: Staff = Depends(current_staff), session: Session = SessionDep):
    if staff.role == "labeler":  # visibility first: hidden and missing artifacts get the same 404, the same work
        found = _labeler_lookup(session, artifact_id)
        if found is None:
            _deny(session, staff, f"artifact/{artifact_id}", None, None, body.purpose, "not_visible", status=404,
                  shown=_NOT_FOUND)
        art, ev, device, customer = found
    else:
        art = session.get(Artifact, artifact_id)
        ev = session.get(Event, art.event_id) if art is not None and art.event_id is not None else None
        device = _device_of(session, art, ev) if art is not None else None
        customer = session.get(Customer, device.customer_id) if device is not None else None
    if art is None or customer is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    if not art.available:
        raise HTTPException(status_code=410, detail="This file is no longer available")
    if body.purpose == "training":
        if staff.role not in ("labeler", "admin"):
            _deny(session, staff, art.s3_key, customer, device, body.purpose, "Your role cannot open training data")
        if not customer.consent_training:
            _deny(session, staff, art.s3_key, customer, device, body.purpose,
                  "This customer has not agreed to training use")
    else:
        if staff.role == "labeler":
            _deny(session, staff, art.s3_key, customer, device, body.purpose, "Your role cannot do this")
        if not customer.consent_recordings:
            _deny(session, staff, art.s3_key, customer, device, body.purpose,
                  "This customer has not agreed to recordings access")
    s3 = _s3(request)
    now: datetime = request.app.state.clock()
    camera = art.camera or (ev.camera if ev is not None else None)
    audit.record(session, staff.id, "media_view", target=art.s3_key, reason=body.purpose,
                 customer_id=customer.id, device_id=device.device_id, detail={"role": art.role, "camera": camera},
                 ts=now)
    mime = _mime(art.s3_key, art.mime)
    url = (_labeler_url(session, request, s3, art, mime) if staff.role == "labeler"
           else s3.presign(art.s3_key, URL_TTL_SECONDS))
    if body.purpose != "training":
        cameras = [audit.camera_label(session, device.id, camera)] if camera else []
        # All DB writes above are flushed; the S3 put happens last (see audit.owner_notice).
        audit.owner_notice(session, s3, device, staff, kind="recording", cameras=cameras, now=now)
    return MediaAccess(url=url, expires_utc=now + timedelta(seconds=URL_TTL_SECONDS), mime=mime)


@router.get(
    "/events/{event_id}/thumbnail",
    status_code=307,
    responses={307: {"description": "Redirect to a presigned thumbnail URL"}},
)
def event_thumbnail(event_id: int, request: Request, staff: Staff = Depends(current_staff),
                    session: Session = SessionDep):
    viewer = _Viewer(staff, request)
    row = _load_one(session, viewer, event_id)  # 404 for events a labeler may not see
    ev, customer_id = row[0], row[1]
    art = session.scalar(select(Artifact).where(Artifact.event_id == ev.id, Artifact.role == "thumbnail",
                                                Artifact.available.is_(True)).order_by(Artifact.id).limit(1))
    if art is None:
        raise HTTPException(status_code=404, detail="No thumbnail")
    s3 = _s3(request)
    now: datetime = request.app.state.clock()
    target = f"event/{ev.id}"
    last = session.scalar(select(AuditLog.ts).where(
        AuditLog.action == "thumbnail_view", AuditLog.staff_id == staff.id, AuditLog.target == target)
        .order_by(AuditLog.ts.desc()).limit(1))
    if last is None or last <= now - THUMBNAIL_AUDIT_WINDOW:
        device_id = session.scalar(select(Device.device_id).where(Device.id == ev.device_pk))
        audit.record(session, staff.id, "thumbnail_view", target=target, customer_id=customer_id,
                     device_id=device_id, ts=now)
    location = (_labeler_url(session, request, s3, art, _mime(art.s3_key, art.mime)) if viewer.labeler
                else s3.presign(art.s3_key, URL_TTL_SECONDS))
    return Response(status_code=307, headers={"Location": location, "Cache-Control": "no-store"})
