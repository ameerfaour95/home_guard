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

from datetime import timezone

from sqlalchemy import and_, exists, func, not_, or_, select
from sqlalchemy.orm import Session, aliased

from home_guard_project.fleet_contract.judgement import not_a_judgement

from .models import AiRun, Camera, Customer, Device, Event, Feedback, InboxDecision

OWNER_LABELS = ("normal", "suspicious", "escalation", "empty", "other", "rule_mismatch")  # box feedback.py
DECISIONS = ("accepted", "fixed", "not_label")
MAX_LIMIT = 500


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
GENERAL = "%/feedback/_general/%"   # chat about no clip: the box no longer writes it; older files stay hidden


def superseded():
    """SQL: a tag row replaced by a later one. Explicitly (the box wrote ``superseded_by`` into it when the clip was
    retagged), or because a newer, non-superseded tag of the same clip exists (an older box, or a rewrite not
    uploaded yet). A clip has ONE current tag; rows without a tag word are never superseded by recency."""
    later = aliased(Feedback)
    newer = or_(func.coalesce(later.received_at, EPOCH) > func.coalesce(Feedback.received_at, EPOCH),
                and_(func.coalesce(later.received_at, EPOCH) == func.coalesce(Feedback.received_at, EPOCH),
                     later.id > Feedback.id))
    return or_(Feedback.superseded_by != "",
               and_(Feedback.owner_label.in_(OWNER_LABELS),
                    exists().where(later.event_id == Feedback.event_id, later.owner_label.in_(OWNER_LABELS),
                                   later.superseded_by == "", newer)))


def answers_filter():
    """Feedback rows the Inbox lists: an owner's CURRENT tag of a clip, a real tag word (box OWNER_LABELS). Action
    records (a pause, a search), plain chat and the old _general files are never owner tags; a superseded tag is
    history, shown under the tag that replaced it."""
    return (Feedback.event_id.is_not(None), Feedback.owner_label.in_(OWNER_LABELS),
            not_(Feedback.s3_key.like(GENERAL)), not_(superseded()))


def query(session: Session, *, from_utc: Optional[datetime] = None, to_utc: Optional[datetime] = None,
          customer_id: Optional[int] = None, camera: Optional[str] = None, owner_label: Optional[str] = None,
          handled: Optional[bool] = None, before_id: Optional[int] = None, limit: int = 200) -> list[tuple]:
    """[(feedback, event, customer, decision or None, camera display name or None, guard AI run or None)], newest
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
    runs = {}
    if ids:
        for run in session.scalars(select(AiRun).where(AiRun.event_id.in_(ids), AiRun.purpose == "guard")
                                   .order_by(AiRun.id)):
            if run.status == "real" or run.event_id not in runs:
                runs[run.event_id] = run
    return [(fb, ev, cu, dec, name, runs.get(ev.id)) for fb, ev, cu, dec, name in rows]


def history(session: Session, event_ids: list[int]) -> dict[int, list[Feedback]]:
    """{event id: the clip's earlier tags, replaced by a later one}, oldest first."""
    if not event_ids:
        return {}
    out: dict[int, list[Feedback]] = {}
    for fb in session.scalars(select(Feedback).where(Feedback.event_id.in_(event_ids),
                                                      Feedback.owner_label.in_(OWNER_LABELS), superseded())
                              .order_by(func.coalesce(Feedback.received_at, EPOCH), Feedback.id)):
        out.setdefault(fb.event_id, []).append(fb)
    return out


def owner_words(fb: Feedback) -> str:
    """What the owner said in words: their text, else a voice answer's transcript, else a typed message."""
    return fb.owner_text or fb.transcript or fb.raw_text or ""


def probably_not_label(fb: Feedback) -> bool:
    """An "other" tag (or a word answer without a tag) whose words the box's own rule (feedback.not_a_judgement,
    vendored) calls a question, a complaint or a command: most likely not about what the clip shows. A tag button
    (normal / suspicious / escalation / empty / rule_mismatch) is always a judgement."""
    words = fb.owner_text or fb.transcript or ("" if fb.owner_label else fb.raw_text) or ""
    return fb.owner_label in ("", "other") and bool(words.strip()) and not_a_judgement(words)


def decide(session: Session, feedback: Feedback, decision: str, note: str, staff_id: int, staff_name: str,
           now: datetime, prompt_version: Optional[str] = None) -> InboxDecision:
    """Record (or replace) the admin's decision on one answer, with the prompt version the clip's AI answer came
    from (a tag made from it follows that prompt's schema)."""
    row = session.get(InboxDecision, feedback.id)
    if row is None:
        row = InboxDecision(feedback_id=feedback.id)
        session.add(row)
    row.decision, row.note, row.staff_id, row.staff_name, row.decided_at = decision, note, staff_id, staff_name, now
    row.prompt_version = prompt_version
    session.flush()
    return row
