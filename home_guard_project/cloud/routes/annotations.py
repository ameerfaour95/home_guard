"""In-app labeling routes (contract 2f): a clip's annotation, its versions and review, and the tagging publish.

Visibility is the event history's: a labeler reaches only events of customers who agreed to training use (one
uniform 404 otherwise) and reads every free text redacted (descriptions, the AI summary, model names). Saves are
append-only versions guarded by `base_version` (409 when someone saved since). Every save and review is audited.
"""
from __future__ import annotations

import functools
from typing import get_args

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .. import audit, labeling, pseudonym, tagging
from ..deps import NOTES_MAX, SessionDep, check_length, current_staff, require_id, require_role
from ..models import Annotation, AnnotationHead, AnnotationReview, Event, Staff, TaggingPublish
from ..schemas import (AiStatus, AnnotationIn, AnnotationOut, AnnotationVersion, PublishMissing, PublishOut,
                       PublishRequest, ReviewDecision)
from .events import _device_id, _load_one, _now, _Viewer

router = APIRouter(tags=["annotations"], dependencies=[Depends(current_staff)])

# Route functions carry no docstrings on purpose (they would become the OpenAPI description).

_AI_STATUSES = set(get_args(AiStatus))
NOT_FOUND = "Event not found"
CONFLICT = "Someone saved this clip since you opened it (now version {v}). Reload it, then save again."
NOTHING_TO_REVIEW = "This clip has no saved annotation to review"
VERSION_REQUIRED = "version required"
REVIEW_CONFLICT = "This clip changed since you opened it (now version {v}). Reload it, then review again."


def _lock(session: Session, event_id: int) -> None:
    """Serialise saves and reviews of one clip (transaction-scoped)."""
    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": f"annotation:{event_id}"})


def _ai(session: Session, viewer: _Viewer, ev: Event) -> tuple:
    """(ai_status, ai_model, ai_prompt_version) as this viewer may read them."""
    run = labeling.guard_run(session, ev.id)
    comp = ev.completeness if isinstance(ev.completeness, dict) else {}
    status = run.status if run is not None else comp.get("ai", "none")
    status = status if status in _AI_STATUSES else "none"
    shown = lambda v: viewer.text(session, ev.device_pk, v) if v is not None else None  # noqa: E731
    return status, shown(run.model if run else None), shown(run.prompt_version if run else None)


def _author(viewer: _Viewer, author_id, author_name):
    """Who saved a version, as this viewer may read it: a labeler only ever reads staff pseudonyms."""
    if not viewer.labeler:
        return author_name
    return pseudonym.staff(viewer.secret, author_id) if author_id is not None else None


def _out(session: Session, request: Request, viewer: _Viewer, ev: Event) -> AnnotationOut:
    fps, frame_count, _ = labeling.clip_timing(ev, labeling.meta_body(session, ev.id))
    ai_status, ai_model, ai_prompt_version = _ai(session, viewer, ev)
    shown = lambda v: viewer.text(session, ev.device_pk, v)  # noqa: E731
    frame_size = ev.frame_size if isinstance(ev.frame_size, list) else None
    head = labeling.head(session, ev.id)
    row = labeling.version_row(session, ev.id, head.version) if head is not None else None
    common = dict(event_id=ev.id, ai_status=ai_status, ai_model=ai_model, ai_prompt_version=ai_prompt_version,
                  fps=fps, frame_count=frame_count, frame_size=frame_size)
    if row is None:  # never saved: the YOLO boxes as editable tracks, the AI summary as the starting description
        tracks = preloaded_tracks(session, request, ev, fps)
        summary = shown(ev.summary)
        return AnnotationOut(**common, version=0, status="new", tracks=labeling.track_dicts(tracks),
                             description=summary, ai_description=summary, drop_clip=False, needs_review=False,
                             author=None, updated_utc=None, suggestions_used=bool(tracks))
    review = labeling.latest_review(session, ev.id, row.version)
    return AnnotationOut(**common, version=row.version, status=labeling.effective_status(row, review),
                         tracks=row.tracks or [], description=shown(row.description),
                         ai_description=shown(row.ai_description), drop_clip=row.drop_clip,
                         needs_review=row.needs_review, author=_author(viewer, row.author_id, row.author_name),
                         updated_utc=review.created_at if review is not None else row.created_at,
                         review_note=shown(review.note) if review is not None else "",
                         review_frame=review.frame if review is not None else None,
                         suggestions_used=row.suggestions_used)


def preloaded_tracks(session: Session, request: Request, ev: Event, fps) -> list:
    """The boxes a never-saved clip opens with, editable at once (source "yolo" until a person edits them): the
    unified dataset's YOLO labels for this clip when it has them, else the event's own weak labels."""
    from .tagging import studio_of  # noqa: PLC0415
    from ..tagstudio.boxes import dataset_tracks  # noqa: PLC0415

    try:
        tracks = dataset_tracks(str(studio_of(request).paths.dataset), ev.stem, fps)
    except Exception:  # noqa: BLE001 - an unreadable dataset never blocks the clip
        tracks = []
    if not tracks:
        tracks = labeling.suggestions(session, request.app.state.s3, ev, fps, _now(request))
    for tr in tracks:
        tr.source = "yolo"
    return tracks


def _event(session: Session, request: Request, staff: Staff, event_id: int):
    """(viewer, event, customer id) of a visible event; one 404 for absent and hidden alike."""
    require_id(event_id, NOT_FOUND)
    viewer = _Viewer(staff, request)
    row = _load_one(session, viewer, event_id)
    return viewer, row[0], row[1]


def _set_head(session: Session, event_id: int, version: int, status: str, needs_review: bool, drop_clip: bool,
              now) -> None:
    values = dict(event_id=event_id, version=version, status=status, needs_review=needs_review,
                  drop_clip=drop_clip, updated_at=now)
    session.execute(pg_insert(AnnotationHead).values(**values).on_conflict_do_update(
        index_elements=[AnnotationHead.event_id], set_={k: v for k, v in values.items() if k != "event_id"}))


@router.get("/events/{event_id}/annotation", response_model=AnnotationOut,
            dependencies=[Depends(require_role("admin", "labeler", "support"))])
def get_annotation(event_id: int, request: Request, staff: Staff = Depends(current_staff),
                   session: Session = SessionDep):
    viewer, ev, _ = _event(session, request, staff, event_id)
    return _out(session, request, viewer, ev)


@router.put("/events/{event_id}/annotation", response_model=AnnotationOut,
            dependencies=[Depends(require_role("admin", "labeler"))])
def save_annotation(event_id: int, body: AnnotationIn, request: Request, staff: Staff = Depends(current_staff),
                    session: Session = SessionDep):
    check_length("description", body.description, labeling.DESCRIPTION_MAX)
    viewer, ev, customer_id = _event(session, request, staff, event_id)
    _, frame_count, duration = labeling.clip_timing(ev, labeling.meta_body(session, ev.id))
    tracks = labeling.to_tracks([t.model_dump() for t in body.tracks])
    _lock(session, ev.id)
    head = labeling.head(session, ev.id)
    current = head.version if head is not None else 0
    if body.base_version != current:
        raise HTTPException(status_code=409, detail=CONFLICT.format(v=current))
    labeling.assign_track_ids(tracks, labeling.versions(session, ev.id), current)  # opaque, server-given ids
    problems = labeling.problems(tracks, duration, frame_count)
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    previous = labeling.version_row(session, ev.id, current) if current else None
    run = labeling.guard_run(session, ev.id)
    now = _now(request)
    row = Annotation(event_id=ev.id, version=current + 1, status=body.status, tracks=labeling.track_dicts(tracks),
                     description=body.description, ai_description=ev.summary or "",
                     ai_run_id=run.id if run is not None else None, drop_clip=body.drop_clip,
                     needs_review=body.needs_review,
                     suggestions_used=any(t.source in ("yolo", "suggestion") for t in tracks)
                     or bool(previous is not None and previous.suggestions_used),
                     author_id=staff.id, author_name=staff.name, created_at=now)
    session.add(row)
    session.flush()
    _set_head(session, ev.id, row.version, body.status, body.needs_review, body.drop_clip, now)
    audit.record(session, staff.id, "annotation_save", target=f"event/{ev.id}", customer_id=customer_id,
                 device_id=_device_id(session, ev), ts=now,
                 detail={"version": row.version, "status": body.status, "tracks": len(tracks),
                         "drop_clip": body.drop_clip, "needs_review": body.needs_review})
    return _out(session, request, viewer, ev)


@router.post("/events/{event_id}/annotation/review", response_model=AnnotationOut)
def review_annotation(event_id: int, body: ReviewDecision, request: Request,
                      staff: Staff = Depends(require_role("admin")), session: Session = SessionDep):
    check_length("note", body.note, NOTES_MAX)
    viewer, ev, customer_id = _event(session, request, staff, event_id)
    if body.version is None:  # optional in the contract (additive), required here: never approve an unseen version
        raise HTTPException(status_code=422, detail=VERSION_REQUIRED)
    _, frame_count, _ = labeling.clip_timing(ev, labeling.meta_body(session, ev.id))
    if body.frame is not None and (body.frame < 0 or (frame_count and body.frame >= frame_count)):
        raise HTTPException(status_code=422, detail="frame is outside the clip")
    _lock(session, ev.id)
    head = labeling.head(session, ev.id)
    if head is None:
        raise HTTPException(status_code=400, detail=NOTHING_TO_REVIEW)
    if head.version != body.version:
        raise HTTPException(status_code=409, detail=REVIEW_CONFLICT.format(v=head.version))
    row = labeling.version_row(session, ev.id, head.version)
    now = _now(request)
    session.add(AnnotationReview(event_id=ev.id, version=row.version, decision=body.decision, note=body.note,
                                 frame=body.frame, reviewer_id=staff.id, reviewer_name=staff.name, created_at=now))
    status = "reviewed" if body.decision == "accept" else "rejected"
    _set_head(session, ev.id, row.version, status, row.needs_review, row.drop_clip, now)
    audit.record(session, staff.id, "annotation_review", target=f"event/{ev.id}", customer_id=customer_id,
                 device_id=_device_id(session, ev), ts=now,
                 detail={"version": row.version, "decision": body.decision, "frame": body.frame})
    session.flush()
    return _out(session, request, viewer, ev)


@router.get("/events/{event_id}/annotation/history", response_model=list[AnnotationVersion],
            dependencies=[Depends(require_role("admin", "labeler", "support"))])
def annotation_history(event_id: int, request: Request, staff: Staff = Depends(current_staff),
                       session: Session = SessionDep):
    # newest version first; a version's status has its review applied
    viewer, ev, _ = _event(session, request, staff, event_id)
    rows = labeling.versions(session, ev.id)
    reviews = labeling.reviews_by_version(session, ev.id)
    out = []
    for i, row in enumerate(rows):
        before = rows[i - 1].description if i else row.ai_description
        out.append(AnnotationVersion(
            version=row.version, status=labeling.effective_status(row, reviews.get(row.version)),
            author=_author(viewer, row.author_id, row.author_name), created_utc=row.created_at,
            tracks_count=len(row.tracks or []),
            description_changed=(row.description or "") != (before or "")))
    return list(reversed(out))


# ---------------------------------------------------------------- publish as a tagging batch (tagging.py)

def _publish_out(session: Session, pub: TaggingPublish) -> PublishOut:
    creator = session.scalar(select(Staff.name).where(Staff.id == pub.created_by)) or ""
    missing = [PublishMissing(event_id=e["event_id"], reason=str(e.get("reason", "")))
               for e in (pub.missing or []) if isinstance(e, dict) and isinstance(e.get("event_id"), int)]
    return PublishOut(batch_name=pub.batch_name, s3_prefix=pub.s3_prefix, state=pub.state, tasks=pub.tasks,
                      yolo_frames=pub.yolo_frames, vlm_lines=pub.vlm_lines, missing=missing,
                      created_utc=pub.created_at, created_by=creator)


@router.post("/studio/collections/{collection_id}/publish", response_model=PublishOut,
             dependencies=[Depends(require_role("admin"))])
def publish_collection(collection_id: int, body: PublishRequest, request: Request,
                       staff: Staff = Depends(current_staff), session: Session = SessionDep):
    # runs in the training-export worker pool; 409 for a read-only batch, a folder not made here, a published
    # batch or a run in flight; 400 over the frame/byte budget
    check_length("batch_name", body.batch_name, 64)
    from .studio import _load_collection, _run_job

    col = _load_collection(session, collection_id, staff)
    s3 = request.app.state.s3
    if s3 is None:
        raise HTTPException(status_code=503, detail="Storage is not configured")
    now = _now(request)
    try:
        pub = tagging.create_publish(session, s3, col, body.batch_name, staff, now)
    except tagging.PublishRefused as e:
        raise HTTPException(status_code=409, detail=str(e))
    except tagging.PublishTooBig as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit.record(session, staff.id, "tagging_publish", target=f"tagging/{body.batch_name}", ts=now,
                 detail={"collection_id": col.id, "publish_id": pub.id,
                         "events": len((pub.snapshot or {}).get("events", []))})
    session.commit()  # the job reads the row from its own session
    _run_job(request, functools.partial(tagging.run_publish_job, request.app.state.sessionmaker, s3, pub.id))
    session.refresh(pub)
    return _publish_out(session, pub)


@router.get("/studio/publishes", response_model=list[PublishOut], dependencies=[Depends(require_role("admin"))])
def list_publishes(session: Session = SessionDep):
    rows = session.scalars(select(TaggingPublish).order_by(TaggingPublish.id.desc()).limit(500)).all()
    return [_publish_out(session, p) for p in rows]
