"""The Keeper: keeps the memory tidy and honest, always through the owner.

- Saving an interview: a new case, or, when it is the same routine as a live case, a proposal to merge into one
  memory with several examples (Mem0's ADD / UPDATE / NOOP, decided in code). An explanation already covered by a
  case is a confirmation of that case (NOOP plus an example).
- Owner corrections: a narrowing applies at once; a widening ("also on Saturday") is a proposal with the diff.
- Never deletes history: "delete", "moved away" and merges set ``invalid_at`` with the reason.
- Forgetting: a case not seen for 30 days pauses with one question, "still relevant?".
- Transparency: "what do you remember about the gate?" lists the cases with a delete button each.
- Buttons: one dispatcher for the buttons the policy and the listings attach.
- Owner memories from the chat (2026-10-09, :func:`remember`): an explained action or a known mark becomes a
  precedent too. ADD a new one; UPDATE the one from the same source (its fact or mark: the owner's own words, applied
  at once with the old scope in ``history``) or leave it (NOOP); REINFORCE one the owner explains again from a new
  source (one confirmation a day: the trust ladder's step). "Not them" narrows at once; a widening the box infers
  by itself still waits for a yes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import texts
from .gates import circular_distance, in_window
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
        out = store.contradict(case_id, by, event_id)
        if out is not None and example is not None:
            out = narrow_for_event(store, case_id, example.signature, by) or out
        return out
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


# -- owner memories from the chat: one model for the week-long and the long-term (2026-10-09) -----------------------

def _day(ts: Optional[float]) -> Any:
    return datetime.fromtimestamp(float(ts or 0)).date()


def same_source(case: Case, draft: Case) -> bool:
    """*draft* comes from the same owner memory as *case*: the same explained-action fact, or the same mark (or one
    it replaces)."""
    a, b = case.source or {}, draft.source or {}
    if not a.get("origin") or a.get("origin") != b.get("origin"):
        return False
    if a["origin"] == "activity":
        return bool(a.get("fact_id")) and (a.get("fact_id") == b.get("fact_id")
                                           or a.get("fact_id") in (b.get("facts") or ()))
    if a["origin"] == "known":
        ids = {b.get("known_id"), *(b.get("replaces") or ())}
        return bool(a.get("known_id")) and a.get("known_id") in ids
    return False


def same_activity(case: Case, draft: Case, same_people: Optional[Callable[[str, str], bool]] = None) -> bool:
    """*draft* explains the same thing as *case* from a new source (another day's explanation of the same work at
    the same camera, the same people marked again)."""
    a, b = case.scope, draft.scope
    if a.camera != b.camera or window_gap(a.hours, b.hours) > MERGE_GAP_MIN:
        return False
    if a.actions or b.actions:
        return bool(set(a.actions) & set(b.actions)) and (a.place == b.place or not a.place or not b.place)
    if (case.source or {}).get("origin") != "known" or (draft.source or {}).get("origin") != "known":
        return False
    same = same_people or (lambda x, y: " ".join(x.split()) == " ".join(y.split()))
    return bool(case.title and draft.title and same(case.title, draft.title))


def remember(store: CaseStore, draft: Case, by: str, now: Optional[float] = None,
             same_people: Optional[Callable[[str, str], bool]] = None) -> SaveResult:
    """Keep one precedent per owner memory (Mem0's ADD / UPDATE / NOOP, Zep's history): ``saved``, ``updated``,
    ``narrowed``, ``unchanged``, ``reinforced`` or ``failed``. Only the owner (*by*) gets here."""
    if not by:
        raise ValueError("only the owner's words make a memory")
    now = time.time() if now is None else now
    live = store.live_cases(draft.camera)
    for case in live:
        if same_source(case, draft):
            return _update(store, case, draft, by)
    for case in live:
        if same_activity(case, draft, same_people):
            return _reinforce(store, case, draft, by, now)
    saved = store.add(draft, by)
    return SaveResult("saved" if saved else "failed", case=saved)


def _update(store: CaseStore, case: Case, draft: Case, by: str) -> SaveResult:
    source = {**case.source, **{k: v for k, v in draft.source.items() if k != "replaces" and v not in ("", None)}}
    if draft.scope == case.scope and (draft.title or case.title) == case.title and source == case.source:
        return SaveResult("unchanged", case=case, target=case)
    if narrows(case.scope, draft.scope) and draft.scope != case.scope and (draft.title or case.title) == case.title \
            and source == case.source:
        out = store.narrow(case.id, draft.scope, by, reason="the owner's correction")
        return SaveResult("narrowed" if out else "failed", case=out, target=out)
    out = store.revise(case.id, draft.scope, by, reason="the owner's words", title=draft.title or None,
                       note=draft.note or None, source=source)
    return SaveResult("updated" if out else "failed", case=out, target=out)


def _reinforce(store: CaseStore, case: Case, draft: Case, by: str, now: float) -> SaveResult:
    """The owner explained it again from a new source: one confirmation a day, and the end follows the newest."""
    out: Optional[Case] = case
    newest = max(case.created_at or 0.0, case.last_confirmed_at or 0.0)
    if _day(newest) < _day(now):
        out = store.confirm(case.id, by, str(draft.source.get("alert_id") or ""),
                            draft.examples[0] if draft.examples else None) or out
    scope = case.scope
    if scope.until is not None and (draft.scope.until is None or draft.scope.until > scope.until):
        scope = replace(scope, until=draft.scope.until)
    facts = list(dict.fromkeys([*(case.source.get("facts") or ()), case.source.get("fact_id"),
                                draft.source.get("fact_id")]))
    source = {k: v for k, v in {"fact_id": draft.source.get("fact_id") or case.source.get("fact_id"),
                                "facts": [f for f in facts if f],
                                "known_id": draft.source.get("known_id") or case.source.get("known_id")}.items() if v}
    if scope != case.scope or any(case.source.get(k) != v for k, v in source.items()):
        out = store.revise(case.id, scope, by, reason="explained again", source=source) or out
    return SaveResult("reinforced", case=out, target=out)


def narrow_for_event(store: CaseStore, case_id: str, sig: Signature, by: str) -> Optional[Case]:
    """ "Not them" narrows at once where the event shows how: an event in the outer third of the hours cuts the
    window short of it; one with the most people the case allows lowers that limit. Otherwise only the ladder steps
    back (``contradict``)."""
    case = store.get(case_id)
    if case is None:
        return None
    s = case.scope
    start, length = window_minutes(s.hours)
    new = s
    if length < 1440 and in_window(sig.minute, s.hours):
        offset = (sig.minute - start) % 1440
        if offset >= length * 2 / 3 and offset >= 10:
            new = replace(s, hours=(s.hours[0], hhmm(sig.minute)))
        elif offset <= length / 3 and length - offset - 1 >= 10:
            new = replace(s, hours=(hhmm(sig.minute + 1), s.hours[1]))
    if new == s and s.people_range and len(s.people_range) == 2:
        low, high = s.people_range
        if high and sig.people == high and high - 1 >= max(1, low):
            new = replace(s, people_range=(low, high - 1))
    if new == s or not narrows(s, new):
        return None
    return store.narrow(case_id, new, by, reason=f"not them ({hhmm(sig.minute)}, {sig.people} people)")


def describe_diff(diff: Dict[str, Tuple[Any, Any]]) -> str:
    """ "hours: 07:10-08:10 -> 07:10-09:00" lines for a widening proposal."""
    lines = []
    for key, (old, new) in diff.items():
        fmt = (lambda v: "-".join(v)) if key == "hours" else (lambda v: ",".join(map(str, v)) if isinstance(v, list) else str(v))
        lines.append(f"{key}: {fmt(old)} -> {fmt(new)}")
    return "\n".join(lines)


__all__ = ["SaveResult", "save_interview", "confirm_merge", "correct", "widen_for_event", "due_reviews",
           "remember_listing", "on_button", "union_scope", "union_window", "window_gap", "mergeable",
           "describe_diff", "scope_diff", "remember", "same_source", "same_activity", "narrow_for_event"]
