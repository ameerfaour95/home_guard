"""The Keeper: keeps the memory tidy and honest, always through the owner.

- Saving an interview: a new case, or, when it is the same routine as a live case, a proposal to merge into one
  memory with several examples (Mem0's ADD / UPDATE / NOOP, decided in code). An explanation already covered by a
  case is a confirmation of that case (NOOP plus an example).
- Owner corrections: a narrowing applies at once; a widening ("also on Saturday") is a proposal with the diff.
- Never deletes history: "delete", "moved away" and merges set ``invalid_at`` with the reason.
- Forgetting: a case not seen for 30 days pauses with one question, "still relevant?".
- Transparency: "what do you remember about the gate?" lists the cases with a delete button each.
- Buttons: one dispatcher for the buttons the policy and the listings attach.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import texts
from .gates import circular_distance
from .interviewer import Outcome
from .models import Case, Example, Scope, Signature, hhmm, window_minutes
from .store import CaseStore, narrows, scope_diff

STALE_DAYS = 30
MERGE_GAP_MIN = 60           # windows closer than this on the same camera and path are one routine


@dataclass(frozen=True)
class SaveResult:
    kind: str                          # saved / reinforced / merge_proposed / expecting / cancelled / failed
    case: Optional[Case] = None
    target: Optional[Case] = None      # merge_proposed / reinforced: the existing case
    merged_scope: Optional[Scope] = None
    text_he: str = ""
    text_en: str = ""


# -- scope algebra -------------------------------------------------------------------------------------------------

def window_gap(a: Tuple[str, str], b: Tuple[str, str]) -> int:
    """Minutes between two windows (0 when they overlap)."""
    a_start, a_len = window_minutes(a)
    b_start, b_len = window_minutes(b)
    if a_len >= 1440 or b_len >= 1440:
        return 0
    if (b_start - a_start) % 1440 <= a_len or (a_start - b_start) % 1440 <= b_len:
        return 0
    a_end, b_end = (a_start + a_len) % 1440, (b_start + b_len) % 1440
    return min(circular_distance(a_end, b_start), circular_distance(b_end, a_start))


def union_window(a: Tuple[str, str], b: Tuple[str, str]) -> Tuple[str, str]:
    """The shortest window covering both."""
    a_start, a_len = window_minutes(a)
    b_start, b_len = window_minutes(b)
    if a_len >= 1440 or b_len >= 1440:
        return ("00:00", "00:00")
    options = []
    for s1, l1, s2, l2 in ((a_start, a_len, b_start, b_len), (b_start, b_len, a_start, a_len)):
        options.append((max(l1, (s2 - s1) % 1440 + l2), s1))
    length, start = min(options)
    return (hhmm(start), hhmm(start + length)) if length < 1440 else ("00:00", "00:00")


def mergeable(a: Scope, b: Scope) -> bool:
    return (a.camera == b.camera and a.people == b.people and a.vehicles == b.vehicles and a.path == b.path
            and a.entry_edge == b.entry_edge and a.exit_edge == b.exit_edge
            and window_gap(a.hours, b.hours) <= MERGE_GAP_MIN)


def union_scope(a: Scope, b: Scope) -> Scope:
    dwell = None if a.max_dwell_s is None or b.max_dwell_s is None else max(a.max_dwell_s, b.max_dwell_s)
    return replace(a, hours=union_window(a.hours, b.hours), weekdays=tuple(sorted(set(a.weekdays) | set(b.weekdays))),
                   house_states=tuple(sorted(set(a.house_states) | set(b.house_states))), night=a.night or b.night,
                   categories=tuple(sorted(set(a.categories) | set(b.categories))) if a.categories and b.categories
                   else (), max_dwell_s=dwell)


def with_embedding(example: Example, embed: Optional[Callable[[str], Optional[Sequence[float]]]]) -> Example:
    if embed is None or example.embedding:
        return example
    try:
        vec = embed(example.signature.template)
    except Exception:  # noqa: BLE001 - an example without a vector still counts on the other components
        vec = None
    return replace(example, embedding=tuple(vec) if vec else None)


# -- saving an interview -------------------------------------------------------------------------------------------

def save_interview(store: CaseStore, outcome: Outcome, by: str,
                   embed: Optional[Callable[[str], Optional[Sequence[float]]]] = None,
                   now: Optional[float] = None) -> SaveResult:
    """Turn the interview's outcome into memory. Only the owner (``by``) gets here."""
    if outcome.kind == "cancelled":
        return SaveResult("cancelled")
    if outcome.kind == "expecting":
        note = store.add_expecting(outcome.camera, outcome.expecting, by, ts=now)
        return SaveResult("expecting" if note else "failed")
    draft = outcome.case
    if draft is None:
        return SaveResult("failed")
    draft = replace(draft, examples=[with_embedding(e, embed) for e in draft.examples])
    for case in store.live_cases(draft.camera):
        if narrows(case.scope, draft.scope) and draft.effect == case.effect:
            confirmed = store.confirm(case.id, by, draft.source.get("alert_id", ""),
                                      draft.examples[0] if draft.examples else None)
            return SaveResult("reinforced", case=confirmed, target=confirmed,
                              text_he=f"זה כבר בזיכרון: {texts.case_line(case, 'he')}. הוספתי את האירוע כדוגמה.",
                              text_en=f"Already remembered: {texts.case_line(case, 'en')}. I added this event as an example.")
    for case in store.live_cases(draft.camera):
        if mergeable(case.scope, draft.scope):
            merged = union_scope(case.scope, draft.scope)
            return SaveResult("merge_proposed", case=draft, target=case, merged_scope=merged,
                              text_he=f"זה נראה כמו {texts.case_title(case, 'he')} שכבר בזיכרון. לאחד לזיכרון אחד: "
                                      f"{texts.days_text(merged.weekdays, 'he')} {texts.hours_text(merged.hours)}?",
                              text_en=f"This looks like {texts.case_title(case, 'en')}, already remembered. Combine into "
                                      f"one memory: {texts.days_text(merged.weekdays, 'en')} "
                                      f"{texts.hours_text(merged.hours)}?")
    saved = store.add(draft, by)
    return SaveResult("saved" if saved else "failed", case=saved)


def confirm_merge(store: CaseStore, result: SaveResult, by: str, combine: bool) -> Optional[Case]:
    """The owner's answer to a merge proposal: combine (the draft is saved, then folded into the target and
    invalidated as "merged into"), or keep separate (saved as its own case)."""
    if result.kind != "merge_proposed" or result.case is None or result.target is None:
        return None
    saved = store.add(result.case, by)
    if saved is None or not combine:
        return saved
    return store.merge(saved.id, result.target.id, result.merged_scope or result.target.scope, by)


# -- corrections ---------------------------------------------------------------------------------------------------

def correct(store: CaseStore, case_id: str, by: str, **changes: Any) -> Tuple[str, Any]:
    """Apply an owner correction to a case's scope: ``("narrowed", case)`` at once, or
    ``("proposed", {wid, id, diff})`` for a widening that waits for the owner's yes."""
    case = store.get(case_id)
    if case is None:
        return "missing", None
    data = case.scope.to_dict()
    data.update(changes)
    new = Scope.from_dict(data)
    if new == case.scope:
        return "unchanged", case
    if narrows(case.scope, new):
        return "narrowed", store.narrow(case_id, new, by)
    return "proposed", store.propose_widen(case_id, new, by)


def widen_for_event(store: CaseStore, case_id: str, sig: Signature, by: str) -> Optional[Dict[str, Any]]:
    """ "That's fine too" on a similar-but-different alert: propose the scope that would also cover *sig*.
    Counts and the camera never widen this way (a second person is a different situation)."""
    case = store.get(case_id)
    if case is None or sig.camera != case.camera or sig.people != case.scope.people:
        return None
    s = case.scope
    hours = s.hours
    if not (window_minutes(hours)[1] >= 1440):
        hours = union_window(hours, (hhmm(sig.minute), hhmm(sig.minute + 1)))
    path = s.path
    new = replace(s, hours=hours, weekdays=tuple(sorted(set(s.weekdays) | {sig.weekday})),
                  house_states=tuple(sorted(set(s.house_states) | {sig.house_state})),
                  path=() if sig.path and path and sig.path != path else path,
                  entry_edge=s.entry_edge if sig.entry_edge == s.entry_edge else "",
                  exit_edge=s.exit_edge if sig.exit_edge == s.exit_edge else "",
                  max_dwell_s=None if s.max_dwell_s is None or sig.dwell_s is None else max(s.max_dwell_s, sig.dwell_s))
    if new == s:
        return None
    return store.propose_widen(case_id, new, by)


# -- forgetting and transparency -----------------------------------------------------------------------------------

def last_activity(case: Case) -> float:
    return max(x for x in (case.created_at, case.last_seen_at, case.last_confirmed_at) if x is not None)


def due_reviews(store: CaseStore, now: Optional[float] = None, days: int = STALE_DAYS) -> List[Dict[str, Any]]:
    """Live cases not seen for *days*: ask "still relevant?" (``store.ask_review`` pauses them until answered)."""
    now = time.time() if now is None else now
    out = []
    for case in store.cases():
        if not case.live or now - last_activity(case) < days * 86400:
            continue
        out.append({"case_id": case.id, "text_he": texts.review_question(case, "he"),
                    "text_en": texts.review_question(case, "en"),
                    "buttons": [texts.button("keep", case.id), texts.button("forget", case.id)]})
    return out


def remember_listing(store: CaseStore, camera: str, now: Optional[float] = None) -> Dict[str, Any]:
    """ "What do you remember about <camera>?": every case and today's expecting notes, each with delete."""
    items = []
    for case in sorted(store.cases(camera), key=lambda c: c.created_at):
        status_he = {"shadow": "לומד", "active": "פעיל", "paused": "ממתין לתשובה"}.get(case.status, case.status)
        items.append({"case_id": case.id, "status": case.status,
                      "text_he": f"{texts.case_line(case, 'he')} ({status_he}, {case.confirmations} אישורים)",
                      "text_en": f"{texts.case_line(case, 'en')} ({case.status}, {case.confirmations} confirmations)",
                      "button": texts.button("delete", case.id)})
    for note in store.expecting_for(camera, now):
        items.append({"expecting_id": note.id, "status": "today", "text_he": f"היום בלבד: {note.text}",
                      "text_en": f"Today only: {note.text}", "button": texts.button("delete", "", expecting_id=note.id)})
    if not items:
        return {"items": [], "text_he": f"אין לי זיכרונות על מצלמת {camera}.",
                "text_en": f"I don't remember anything about camera {camera}."}
    return {"items": items, "text_he": f"מה שאני זוכר על מצלמת {camera}:", "text_en": f"What I remember about {camera}:"}


def on_button(store: CaseStore, action: str, case_id: str, by: str, event_id: str = "",
              example: Optional[Example] = None, expecting_id: str = "") -> Any:
    """Apply one owner button. The caller checks that *by* is the verified owner before calling."""
    if action == "confirm":
        return store.confirm(case_id, by, event_id, example)
    if action == "not_them":
        return store.contradict(case_id, by, event_id)
    if action == "keep_alerting":
        return store.keep_alerting(case_id, by)
    if action == "keep":
        return store.answer_review(case_id, True, by)
    if action == "forget":
        return store.answer_review(case_id, False, by)
    if action == "delete":
        if expecting_id:
            return store.cancel_expecting(expecting_id, by)
        return store.invalidate(case_id, "deleted by the owner", by)
    raise ValueError(f"unknown action {action!r}")


def describe_diff(diff: Dict[str, Tuple[Any, Any]]) -> str:
    """ "hours: 07:10-08:10 -> 07:10-09:00" lines for a widening proposal."""
    lines = []
    for key, (old, new) in diff.items():
        fmt = (lambda v: "-".join(v)) if key == "hours" else (lambda v: ",".join(map(str, v)) if isinstance(v, list) else str(v))
        lines.append(f"{key}: {fmt(old)} -> {fmt(new)}")
    return "\n".join(lines)


__all__ = ["SaveResult", "save_interview", "confirm_merge", "correct", "widen_for_event", "due_reviews",
           "remember_listing", "on_button", "union_scope", "union_window", "window_gap", "mergeable",
           "describe_diff", "scope_diff"]
