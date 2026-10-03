"""In-app labeling routes (contract 2f): a clip's annotation, its versions and review, and the tagging publish.

Visibility is the event history's: a labeler reaches only events of customers who agreed to training use (one
uniform 404 otherwise) and reads every free text redacted (descriptions, the AI summary, model names). Saves are
append-only versions guarded by `base_version` (409 when someone saved since). Every save and review is audited.
"""
from __future__ import annotations

from typing import get_args

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .. import audit, labeling
from ..deps import NOTES_MAX, SessionDep, check_length, current_staff, require_id, require_role
from ..models import Annotation, AnnotationHead, AnnotationReview, Event, Staff
from ..schemas import (AiStatus, AnnotationIn, AnnotationOut, AnnotationVersion, PublishOut, PublishRequest,
                       ReviewDecision)
from .events import _device_id, _load_one, _now, _Viewer

router = APIRouter(tags=["annotations"], dependencies=[Depends(current_staff)])

# Route functions carry no docstrings on purpose (they would become the OpenAPI description).

_AI_STATUSES = set(get_args(AiStatus))
NOT_FOUND = "Event not found"
CONFLICT = "Someone saved this clip since you opened it (now version {v}). Reload it, then save again."
NOTHING_TO_REVIEW = "This clip has no saved annotation to review"
_NOT_BUILT = "Not implemented yet"


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


def _out(session: Session, request: Request, viewer: _Viewer, ev: Event) -> AnnotationOut:
    fps, frame_count, _ = labeling.clip_timing(ev, labeling.meta_body(session, ev.id))
    ai_status, ai_model, ai_prompt_version = _ai(session, viewer, ev)
    shown = lambda v: viewer.text(session, ev.device_pk, v)  # noqa: E731
    frame_size = ev.frame_size if isinstance(ev.frame_size, list) else None
    head = labeling.head(session, ev.id)
    row = labeling.version_row(session, ev.id, head.version) if head is not None else None
    common = dict(event_id=ev.id, ai_status=ai_status, ai_model=ai_model, ai_prompt_version=ai_prompt_version,
                  fps=fps, frame_count=frame_count, frame_size=frame_size)
    if row is None:  # never saved: suggestions from the weak labels, the AI summary as the starting description
        tracks = labeling.suggestions(session, request.app.state.s3, ev, fps, _now(request))
        summary = shown(ev.summary)
        return AnnotationOut(**common, version=0, status="new", tracks=labeling.track_dicts(tracks),
                             description=summary, ai_description=summary, drop_clip=False, needs_review=False,
                             author=None, updated_utc=None, suggestions_used=bool(tracks))
    review = labeling.latest_review(session, ev.id, row.version)
    return AnnotationOut(**common, version=row.version, status=labeling.effective_status(row, review),
                         tracks=row.tracks or [], description=shown(row.description),
                         ai_description=shown(row.ai_description), drop_clip=row.drop_clip,
                         needs_review=row.needs_review, author=row.author_name,
                         updated_utc=review.created_at if review is not None else row.created_at,
                         review_note=shown(review.note) if review is not None else "",
                         review_frame=review.frame if review is not None else None,
                         suggestions_used=row.suggestions_used)


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
    problems = labeling.problems(tracks, duration, frame_count)
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    _lock(session, ev.id)
    head = labeling.head(session, ev.id)
    current = head.version if head is not None else 0
    if body.base_version != current:
        raise HTTPException(status_code=409, detail=CONFLICT.format(v=current))
    previous = labeling.version_row(session, ev.id, current) if current else None
    run = labeling.guard_run(session, ev.id)
    now = _now(request)
    row = Annotation(event_id=ev.id, version=current + 1, status=body.status, tracks=labeling.track_dicts(tracks),
                     description=body.description, ai_description=ev.summary or "",
                     ai_run_id=run.id if run is not None else None, drop_clip=body.drop_clip,
                     needs_review=body.needs_review,
                     suggestions_used=any(t.source == "suggestion" for t in tracks)
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
    _, frame_count, _ = labeling.clip_timing(ev, labeling.meta_body(session, ev.id))
    if body.frame is not None and (body.frame < 0 or (frame_count and body.frame >= frame_count)):
        raise HTTPException(status_code=422, detail="frame is outside the clip")
    _lock(session, ev.id)
    head = labeling.head(session, ev.id)
    if head is None:
        raise HTTPException(status_code=400, detail=NOTHING_TO_REVIEW)
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
    _, ev, _ = _event(session, request, staff, event_id)
    rows = labeling.versions(session, ev.id)
    reviews = labeling.reviews_by_version(session, ev.id)
    out = []
    for i, row in enumerate(rows):
        before = rows[i - 1].description if i else row.ai_description
        out.append(AnnotationVersion(
            version=row.version, status=labeling.effective_status(row, reviews.get(row.version)),
            author=row.author_name, created_utc=row.created_at, tracks_count=len(row.tracks or []),
            description_changed=(row.description or "") != (before or "")))
    return list(reversed(out))


@router.post("/studio/collections/{collection_id}/publish", response_model=PublishOut,
             dependencies=[Depends(require_role("admin"))])
def publish_collection(collection_id: int, body: PublishRequest):
    raise HTTPException(status_code=501, detail=_NOT_BUILT)


@router.get("/studio/publishes", response_model=list[PublishOut], dependencies=[Depends(require_role("admin"))])
def list_publishes():
    raise HTTPException(status_code=501, detail=_NOT_BUILT)
