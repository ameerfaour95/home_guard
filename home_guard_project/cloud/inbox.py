"""The Inbox: every owner answer from Telegram (a tag, words or a voice answer to an alert), once, with the clip and
the model's tag, waiting for an admin.

An answer is a candidate label until an admin decides (the Frigate+ pattern): "accepted" (the owner's tag is the
starting point of a Tag · AI tag the admin then saves), "fixed" (opened in Tag · AI to tag it differently) or
"not_label" (a complaint or a question, not about what the clip shows). Only a saved Tag · AI tag is training data;
the decision records what was done with the answer, and every decision is audited.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .models import AiRun, Camera, Customer, Device, Event, Feedback, InboxDecision

OWNER_LABELS = ("normal", "suspicious", "escalation", "empty", "other", "rule_mismatch")  # box feedback.py
DECISIONS = ("accepted", "fixed", "not_label")
MAX_LIMIT = 500


def answers_filter():
    """Feedback rows that are an owner's answer about an alert: a tag, words or a voice answer, or a verdict button
    (a pause or a search alone is not)."""
    return (Feedback.event_id.is_not(None),
            or_(Feedback.owner_label != "", Feedback.owner_text != "", Feedback.transcript != "",
                Feedback.verdict.not_in(("", "none"))))


def query(session: Session, *, from_utc: Optional[datetime] = None, to_utc: Optional[datetime] = None,
          customer_id: Optional[int] = None, camera: Optional[str] = None, owner_label: Optional[str] = None,
          handled: Optional[bool] = None, before_id: Optional[int] = None, limit: int = 200) -> list[tuple]:
    """[(feedback, event, customer, decision or None, camera display name or None, model name or None)], newest
    first. `handled`: True = decided, False = waiting, None = both. `owner_label` "" = answers without a tag."""
    q = (select(Feedback, Event, Customer, InboxDecision, Camera.display_name)
         .join(Event, Event.id == Feedback.event_id)
         .join(Device, Device.id == Event.device_pk)
         .join(Customer, Customer.id == Device.customer_id)
         .outerjoin(InboxDecision, InboxDecision.feedback_id == Feedback.id)
         .outerjoin(Camera, (Camera.device_pk == Event.device_pk) & (Camera.name == Event.camera))
         .where(*answers_filter()))
    if from_utc is not None:
        q = q.where(Feedback.received_at >= from_utc)
    if to_utc is not None:
        q = q.where(Feedback.received_at < to_utc)
    if customer_id is not None:
        q = q.where(Customer.id == customer_id)
    if camera:
        q = q.where(Event.camera == camera)
    if owner_label is not None:
        q = q.where(Feedback.owner_label == owner_label)
    if handled is True:
        q = q.where(InboxDecision.feedback_id.is_not(None))
    elif handled is False:
        q = q.where(InboxDecision.feedback_id.is_(None))
    if before_id is not None:
        q = q.where(Feedback.id < before_id)
    rows = session.execute(q.order_by(Feedback.id.desc()).limit(max(1, min(limit, MAX_LIMIT)))).all()
    ids = [ev.id for _, ev, _, _, _ in rows]
    models = {}
    if ids:
        for run in session.scalars(select(AiRun).where(AiRun.event_id.in_(ids), AiRun.purpose == "guard")
                                   .order_by(AiRun.id)):
            if run.status == "real" or run.event_id not in models:
                models[run.event_id] = run.model
    return [(fb, ev, cu, dec, name, models.get(ev.id)) for fb, ev, cu, dec, name in rows]


def decide(session: Session, feedback: Feedback, decision: str, note: str, staff_id: int, staff_name: str,
           now: datetime) -> InboxDecision:
    """Record (or replace) the admin's decision on one answer."""
    row = session.get(InboxDecision, feedback.id)
    if row is None:
        row = InboxDecision(feedback_id=feedback.id)
        session.add(row)
    row.decision, row.note, row.staff_id, row.staff_name, row.decided_at = decision, note, staff_id, staff_name, now
    session.flush()
    return row
