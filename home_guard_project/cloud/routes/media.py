"""Presigned media access. Every view is audited; recordings viewed by staff are also announced to the owner.

Presigned URLs are secrets for their lifetime: they are returned to the caller and never logged or audited.
"""
from __future__ import annotations

import mimetypes
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit
from ..deps import current_staff, get_session
from ..models import Artifact, AuditLog, Customer, Device, Event, Staff
from ..schemas import MediaAccess, MediaAccessRequest
from .events import _Viewer, _load_one

router = APIRouter(tags=["media"], dependencies=[Depends(current_staff)])

URL_TTL_SECONDS = 300
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


def _device_of(session: Session, art: Artifact, ev: Optional[Event]) -> Optional[Device]:
    if ev is not None:
        return session.get(Device, ev.device_pk)
    top = art.s3_key.split("/", 1)[0]  # dataset_<site> / production_<site>
    site = top.split("_", 1)[1] if "_" in top else None
    return session.scalar(select(Device).where(Device.site == site)) if site else None


@router.post("/artifacts/{artifact_id}/access", response_model=MediaAccess)
def artifact_access(artifact_id: int, body: MediaAccessRequest, request: Request,
                    staff: Staff = Depends(current_staff), session: Session = Depends(get_session)):
    art = session.get(Artifact, artifact_id)
    if art is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not art.available:
        raise HTTPException(status_code=410, detail="This file is no longer available")
    ev = session.get(Event, art.event_id) if art.event_id is not None else None
    device = _device_of(session, art, ev)
    customer = session.get(Customer, device.customer_id) if device is not None else None
    if customer is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if body.purpose == "training":
        if staff.role not in ("labeler", "admin"):
            raise HTTPException(status_code=403, detail="Your role cannot open training data")
        if not customer.consent_training:
            raise HTTPException(status_code=403, detail="This customer has not agreed to training use")
    else:
        if staff.role == "labeler":
            raise HTTPException(status_code=403, detail="Your role cannot do this")
        if not customer.consent_recordings:
            raise HTTPException(status_code=403, detail="This customer has not agreed to recordings access")
    s3 = _s3(request)
    now: datetime = request.app.state.clock()
    camera = art.camera or (ev.camera if ev is not None else None)
    audit.record(session, staff.id, "media_view", target=art.s3_key, reason=body.purpose,
                 customer_id=customer.id, device_id=device.device_id, detail={"role": art.role, "camera": camera},
                 ts=now)
    url = s3.presign(art.s3_key, URL_TTL_SECONDS)
    if body.purpose != "training":
        label = audit.camera_label(session, device.id, camera) if camera else "a camera"
        audit.owner_notice(session, s3, device, staff, kind="recording", cameras=[label], now=now)
    return MediaAccess(url=url, expires_utc=now + timedelta(seconds=URL_TTL_SECONDS), mime=_mime(art.s3_key, art.mime))


@router.get(
    "/events/{event_id}/thumbnail",
    status_code=307,
    responses={307: {"description": "Redirect to a presigned thumbnail URL"}},
)
def event_thumbnail(event_id: int, request: Request, staff: Staff = Depends(current_staff),
                    session: Session = Depends(get_session)):
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
    return Response(status_code=307, headers={"Location": s3.presign(art.s3_key, URL_TTL_SECONDS),
                                              "Cache-Control": "no-store"})
