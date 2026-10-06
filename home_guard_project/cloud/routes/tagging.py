from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit
from ..access import media_refusal
from ..deps import SessionDep, require_role
from ..models import Customer, Device, Staff
from ..schemas import (
    AnnotationOut,
    ClipAnnotationIn,
    TagSuggestion,
    TagSuggestRequest,
    TagSave,
    TagSaved,
    TaggingClip,
    TaggingExportOut,
    TaggingExportRequest,
    TaggingKey,
    TaggingMediaRequest,
    TaggingQueue,
    TaggingState,
    MediaAccess,
    TeacherAnswer,
)
from ..tagstudio.config import StudioPaths
from ..tagstudio.service import MEDIA_TTL_SECONDS, StudioError, TagStudio

# The tagging studio is admin-only for now: it shows clips from every household together with the owners' own
# words, and the labeler privacy rules (pseudonyms, opaque media copies) are not applied to it yet.
router = APIRouter(prefix="/tagging", tags=["tagging"], dependencies=[Depends(require_role("admin"))])
# Media files are fetched by the desktop's video player, which cannot send a bearer token: the URL itself is a
# short-lived HMAC-signed grant for one file, issued by POST /tagging/media to a signed-in admin.
files = APIRouter(prefix="/tagging", tags=["tagging"])

_admin = require_role("admin")


def studio_of(request: Request) -> TagStudio:
    studio = getattr(request.app.state, "tagstudio", None)
    if studio is None:
        studio = TagStudio(StudioPaths.resolve(), bucket=request.app.state.settings.bucket)
        request.app.state.tagstudio = studio
    return studio


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except StudioError as e:
        raise HTTPException(status_code=e.status, detail=str(e)) from None


@router.get("/state", response_model=TaggingState)
def tagging_state(request: Request, session: Session = SessionDep):
    return studio_of(request).state(session)


@router.get("/queue", response_model=TaggingQueue)
def tagging_queue(request: Request, session: Session = SessionDep,
                  tier: str = Query("open", max_length=32), origin: str = Query("", max_length=32),
                  q: str = Query("", max_length=200), limit: int = Query(2000, ge=1, le=10000)):
    return studio_of(request).queue(session, tier=tier, origin=origin, text=q, limit=limit)


@router.get("/clip", response_model=TaggingClip)
def tagging_clip(request: Request, key: str = Query(..., max_length=512), session: Session = SessionDep):
    return _call(studio_of(request).detail, session, key)


@router.post("/tag", response_model=TagSaved)
def tagging_save(body: TagSave, request: Request, staff: Staff = Depends(_admin), session: Session = SessionDep):
    return _call(studio_of(request).save, session, staff, body.key, body.fields, request.app.state.clock())


def _household(session: Session, item) -> tuple[Optional[Customer], Optional[Device]]:
    """The household a customer clip belongs to (None for our own dataset clips)."""
    site = item.info.get("site") or (item.source if item.origin != "dataset" else "")
    if item.origin == "dataset" and not item.event_id:
        return None, None
    device = session.scalar(select(Device).where(Device.site == site)) if site else None
    customer = session.get(Customer, device.customer_id) if device is not None else None
    return customer, device


@router.post("/media", response_model=MediaAccess)
def tagging_media(body: TaggingMediaRequest, request: Request, staff: Staff = Depends(_admin),
                  session: Session = SessionDep):
    studio = studio_of(request)
    item, _ = _call(studio._item, session, body.key)
    customer, device = _household(session, item)
    now: datetime = request.app.state.clock()
    if customer is not None:
        refusal = media_refusal(staff.role, "training", customer.consent_recordings, customer.consent_training)
        if refusal is not None:
            raise HTTPException(status_code=403, detail=refusal)
    path = studio.local_media(item, body.kind)
    if path is not None:
        url = str(request.url_for("tagging_file", token=studio.sign(request.app.state.settings.jwt_secret, path,
                                                                     now.timestamp())))
        target = path
    else:
        art_id = item.artifacts.get(body.kind)
        s3 = request.app.state.s3
        if art_id is None or s3 is None:
            raise HTTPException(status_code=404, detail=f"No {body.kind} video for this clip on this computer")
        from ..models import Artifact  # noqa: PLC0415

        art = session.get(Artifact, art_id)
        url, target = s3.presign(art.s3_key, MEDIA_TTL_SECONDS), art.s3_key
    if customer is not None:
        audit.record(session, staff.id, "media_view", target=target, reason="training", customer_id=customer.id,
                     device_id=device.device_id if device is not None else None,
                     detail={"via": "tagging", "kind": body.kind}, ts=now)
    return MediaAccess(url=url, expires_utc=now + timedelta(seconds=MEDIA_TTL_SECONDS), mime="video/mp4")


@files.get("/file/{token}", name="tagging_file", response_class=FileResponse)
def tagging_file(token: str, request: Request):
    path = TagStudio.verify(request.app.state.settings.jwt_secret, token, request.app.state.clock().timestamp())
    if path is None:
        raise HTTPException(status_code=404, detail="This link has expired")
    return FileResponse(path, media_type="video/mp4")


@router.post("/teach", response_model=TeacherAnswer)
def tagging_teach(body: TaggingKey, request: Request, session: Session = SessionDep):
    return _call(studio_of(request).teach, session, body.key)


@router.post("/export", response_model=TaggingExportOut)
def tagging_export(body: TaggingExportRequest, request: Request, staff: Staff = Depends(_admin),
                   session: Session = SessionDep):
    result = studio_of(request).export(session, include_needs_check=body.include_needs_check)
    audit.record(session, staff.id, "tagging_export", target=result["training_path"], reason="training",
                 detail={"counts": result["counts"]}, ts=request.app.state.clock())
    return result


@router.post("/suggest", response_model=TagSuggestion)
def tagging_suggest(body: TagSuggestRequest, request: Request, staff: Staff = Depends(_admin),
                    session: Session = SessionDep):
    studio = studio_of(request)
    item, _ = _call(studio._item, session, body.key)
    customer, device = _household(session, item)
    if customer is not None:
        refusal = media_refusal(staff.role, "training", customer.consent_recordings, customer.consent_training)
        if refusal is not None:
            raise HTTPException(status_code=403, detail=refusal)
    result = _call(studio.suggest, session, body.key, body.refresh, request.app.state.s3)
    if not result.get("cached"):
        audit.record(session, staff.id, "tag_suggested", target=item.clip_id, reason="training",
                     customer_id=customer.id if customer is not None else None,
                     device_id=device.device_id if device is not None else None,
                     detail={"model": result["model"]}, ts=request.app.state.clock())
    return result


@router.get("/boxes", response_model=AnnotationOut)
def tagging_boxes(request: Request, key: str = Query(..., max_length=512), session: Session = SessionDep):
    return _call(studio_of(request).clip_boxes, session, key)


@router.put("/boxes", response_model=AnnotationOut)
def tagging_save_boxes(body: ClipAnnotationIn, request: Request, staff: Staff = Depends(_admin),
                       session: Session = SessionDep):
    return _call(studio_of(request).save_clip_boxes, session, staff, body, request.app.state.clock())
