"""The Inbox (admin only): owner answers from Telegram across all customers, filtered by date, customer, camera,
the owner's tag and whether an admin handled them; and the admin's decision on each one (accepted as a tag's
starting point, opened to fix, not a label). Every decision is audited; reading the list is audited once per
request with the number of answers shown, like a staff view of customer words."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from .. import audit, inbox
from ..deps import NOTES_MAX, SessionDep, check_length, current_staff, id_in_range, require_role
from ..models import Feedback, Staff
from ..schemas import InboxDecisionIn, InboxItem

router = APIRouter(tags=["inbox"], dependencies=[Depends(current_staff)])
NOT_FOUND = "Answer not found"

# Route functions carry no docstrings on purpose (they would become the OpenAPI description).


def _item(fb, ev, cu, dec, camera_name, model) -> InboxItem:
    return InboxItem(feedback_id=fb.id, event_id=ev.id, clip_key=f"ev:{ev.id}", customer_id=cu.id, customer=cu.name,
                     site=ev.site, camera=ev.camera, camera_name=camera_name, received_utc=fb.received_at,
                     owner_label=fb.owner_label, owner_text=fb.owner_text, transcript=fb.transcript,
                     raw_text=fb.raw_text, note=fb.note, verdict=fb.verdict, source=fb.source, tagged_by=fb.tagged_by,
                     model_label=ev.label, model_summary=ev.summary or "", model=model,
                     consent_training=bool(cu.consent_training),
                     decision=dec.decision if dec is not None else None,
                     decided_by=dec.staff_name if dec is not None else None,
                     decided_utc=dec.decided_at if dec is not None else None,
                     decision_note=dec.note if dec is not None else "")


def _one(session: Session, feedback_id: int):
    rows = [r for r in inbox.query(session, before_id=feedback_id + 1, limit=1) if r[0].id == feedback_id]
    if not rows:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return rows[0]


@router.get("/inbox", response_model=list[InboxItem])
def list_inbox(request: Request, staff: Staff = Depends(require_role("admin")), session: Session = SessionDep,
               from_utc: Optional[datetime] = None, to_utc: Optional[datetime] = None,
               customer_id: Optional[int] = None, camera: Optional[str] = Query(None, max_length=128),
               owner_label: Optional[str] = Query(None, max_length=32),
               handled: Literal["all", "handled", "unhandled"] = "unhandled",
               before_id: Optional[int] = None, limit: int = Query(200, ge=1, le=inbox.MAX_LIMIT)):
    # newest first; page with before_id = the last feedback_id shown
    for value in (customer_id, before_id):
        if value is not None and not id_in_range(value):
            return []
    rows = inbox.query(session, from_utc=from_utc, to_utc=to_utc, customer_id=customer_id, camera=camera,
                       owner_label=owner_label, handled={"handled": True, "unhandled": False}.get(handled),
                       before_id=before_id, limit=limit)
    audit.record(session, staff.id, "inbox_view", target="inbox", ts=request.app.state.clock(),
                 detail={"answers": len(rows), "customers": sorted({r[2].id for r in rows})})
    return [_item(*r) for r in rows]


@router.post("/inbox/{feedback_id}/decision", response_model=InboxItem)
def decide(feedback_id: int, body: InboxDecisionIn, request: Request, staff: Staff = Depends(require_role("admin")),
           session: Session = SessionDep):
    check_length("note", body.note, NOTES_MAX)
    if not id_in_range(feedback_id):
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    fb, ev, cu, _, _, _ = _one(session, feedback_id)
    if body.decision == "accepted" and not cu.consent_training:
        raise HTTPException(status_code=409, detail=f"{cu.name} has withdrawn consent to training use: this answer "
                                                    "cannot become a training label")
    now = request.app.state.clock()
    inbox.decide(session, session.get(Feedback, fb.id), body.decision, body.note, staff.id, staff.name, now)
    audit.record(session, staff.id, "inbox_decision", target=f"feedback/{fb.id}", customer_id=cu.id, ts=now,
                 detail={"decision": body.decision, "event_id": ev.id, "owner_label": fb.owner_label})
    return _item(*_one(session, feedback_id))


@router.delete("/inbox/{feedback_id}/decision", response_model=InboxItem)
def reopen(feedback_id: int, request: Request, staff: Staff = Depends(require_role("admin")),
           session: Session = SessionDep):
    # back to the waiting list (an admin's mistake); audited like a decision
    if not id_in_range(feedback_id):
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    fb, ev, cu, dec, _, _ = _one(session, feedback_id)
    if dec is not None:
        session.delete(dec)
        session.flush()
        audit.record(session, staff.id, "inbox_reopen", target=f"feedback/{fb.id}", customer_id=cu.id,
                     ts=request.app.state.clock(), detail={"was": dec.decision, "event_id": ev.id})
    return _item(*_one(session, feedback_id))
