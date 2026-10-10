"""One memory for the owner (2026-10-09): what he explains in the chat is remembered for the week AND learned for
the long term.

The week-long memories already exist and act today: an explained action (activity_memory.ActivityFact: "the lying
down on the entrance stairs is the electricians installing LEDs", until about a week) and a known mark (events.Known:
"the pergola workers", a daily window for a crew's week). Each of them now also writes a precedent here (a
:class:`Case`), through the Keeper (:func:`keeper.remember`): its camera, hours, days, head-count range, the actions
and the place, the owner's words, its end. A precedent starts in SHADOW like every case: the guard loop logs "would
quiet (case C1, score 1.00)" and changes nothing until the owner confirmed it three times (the trust ladder).

Then the precedent outlives the week only when the owner says so. One question, at most once per explanation, when
the explained action is about to end (its last day, at 18:00 or its end when that is earlier) or when the same kind
of event comes back on a later day after it ended:

    החשמלאים שמתקינים לדים במדרגות של הכניסה הראשית: עוד פעילים אחרי 15.10?
    [עוד שבוע] [זה נגמר] [קבוע: כל יום חול]

- "עוד שבוע": the explanation and its precedent go on one more week (an owner confirmation).
- "זה נגמר": the explanation ends now; the precedent is marked "not valid since" (its history stays).
- "קבוע": the precedent becomes standing (every weekday, the same hours, no end) and gets an owner confirmation.
  After three, a matching suspicious or normal alert goes out as a quiet message instead of an alert. An escalation
  is never touched (memory is not even consulted for it).

The words the owner reads are the brain's (brain/case_chat.py); this module is plain data and never raises into the
chat or the guard loop's callers (each entry point catches and logs).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .keeper import SaveResult, remember
from .models import ALL_DAYS, QUIET, WORKWEEK, Case, Example, Scope, Signature, hhmm
from .signature import build_signature
from .store import CaseStore

log = logging.getLogger("box.case_memory.link")

ORIGIN_ACTIVITY, ORIGIN_KNOWN = "activity", "known"
EXTRA_PEOPLE = 2                 # a crew's head-count jumps around (events.KNOWN_EXTRA_PEOPLE)
CREW_DAYS = 7                    # a crew works about a week (owner, 2026-10-09 10:15)
ASK_AT = "18:00"                 # the last day's question: at 18:00, or at the end when that is earlier
DAY_STATES = ("away", "home_awake")
_WORK = re.compile(r"(?<![א-ת])[ושהבלמכ]{0,2}(?:עובד|עובדים|פועל|פועלים|חשמלאי|חשמלאים|טכנאי|טכנאים|גנן|קבלן|"
                   r"אינסטלטור|צבעי|בנאי|בנאים|מתקינים|מרכיבים)(?![א-ת])|\b(?:work\w*|electrician\w*|technician\w*|"
                   r"gardener\w*|contractor\w*|plumber\w*|install\w*)\b", re.IGNORECASE)

# The owner's answers to the one question (the brain's buttons send these codes).
WEEK, ENDED, STANDING = "week", "ended", "standing"
ANSWERS = (WEEK, ENDED, STANDING)


def _clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def _date(ts: float) -> dt.date:
    return dt.datetime.fromtimestamp(ts).date()


def _at(day: dt.date, clock: str) -> float:
    hour, minute = (int(x) for x in clock.split(":"))
    return dt.datetime.combine(day, dt.time(hour % 24, minute)).timestamp()


def _who(text: str) -> str:
    return "worker" if _WORK.search(str(text or "")) else "other"


# -- the precedent of an explained action -------------------------------------------------------------------------

def fact_hours(fact: Any, first_ts: Optional[float] = None) -> tuple:
    """The precedent's hours: the fact's daily window; for a fact of one day, from the hour it was first seen
    (never before 06:00) to its end; else the whole day. Never a time nobody said or saw."""
    if getattr(fact, "daily_from", "") and getattr(fact, "daily_to", ""):
        return (str(fact.daily_from), str(fact.daily_to))
    until, at = float(fact.until), float(fact.at or until)
    if _date(until) == _date(at):
        seen = min(x for x in (first_ts, at) if x) if first_ts else at
        hour = max(6, dt.datetime.fromtimestamp(seen).hour)
        start = f"{hour:02d}:00"
        if start < _clock(until):
            return (start, _clock(until))
    return ("00:00", "00:00")


def event_signature(camera: str, entry: Optional[Mapping[str, Any]]) -> Optional[Signature]:
    """The alert the owner explained (the brain's handle entry: ``ts``, ``observation`` / ``summary``, ``label``),
    as a signature; None without one."""
    if not entry or not entry.get("ts"):
        return None
    words = f'{entry.get("observation") or ""} {entry.get("summary") or ""} {entry.get("why") or ""}'
    people = entry.get("people")
    obs = {"people": int(people)} if isinstance(people, (int, float)) else {}
    return build_signature(camera, float(entry["ts"]), obs, None, None, str(entry.get("label") or ""), text=words)


def precedent_from_fact(fact: Any, camera: str, entry: Optional[Mapping[str, Any]] = None, chat_id: str = "",
                        people: int = 0) -> Case:
    """The draft precedent of an explained action at *camera* (one per camera of the fact)."""
    sig = event_signature(camera, entry)
    seen = max([people, sig.people if sig else 0])
    scope = Scope(camera=camera, hours=fact_hours(fact, sig.ts if sig else None), weekdays=ALL_DAYS,
                  house_states=DAY_STATES, night=False, people=seen, vehicles=0,
                  people_range=(1, seen + EXTRA_PEOPLE if seen else 0), actions=tuple(fact.actions),
                  place=str(fact.place or ""), until=float(fact.until))
    examples = [Example(str((entry or {}).get("alert_id") or (entry or {}).get("ref") or ""), sig)] if sig else []
    source = {"origin": ORIGIN_ACTIVITY, "fact_id": fact.id, "known_id": str(fact.known_id or ""),
              "alert_id": str(getattr(fact, "alert_id", "") or ""), "chat_id": str(chat_id or "")}
    return Case(id="", scope=scope, who=_who(f"{fact.cause} {fact.cause_en}"), title=str(fact.cause),
                note=str(fact.owner_words or fact.cause), effect=QUIET, examples=examples,
                created_at=float(fact.at or time.time()), source={k: v for k, v in source.items() if v})


def on_fact(store: CaseStore, fact: Any, by: str, entry: Optional[Mapping[str, Any]] = None, chat_id: str = "",
            now: Optional[float] = None, same_people: Optional[Callable[[str, str], bool]] = None) -> List[SaveResult]:
    """An explained action was saved or changed: its precedent(s) follow (one per camera). A camera the owner
    moved it away from keeps its precedent as history, marked not valid since now. Never raises."""
    out: List[SaveResult] = []
    try:
        now = time.time() if now is None else now
        cameras = list(dict.fromkeys(fact.cameras or ()))
        for case in store.cases():
            if case.source.get("fact_id") == fact.id and case.camera not in cameras and case.invalid_at is None:
                store.invalidate(case.id, "the owner moved it to another camera", by)
        for camera in cameras:
            draft = precedent_from_fact(fact, camera, entry, chat_id)
            out.append(remember(store, draft, by, now, same_people))
            log.info("case memory: %s %s from explained action %s", out[-1].kind,
                     out[-1].case.id if out[-1].case else "-", fact.id)
    except Exception as exc:  # noqa: BLE001 - the week-long memory is saved; the long-term one waits for next time
        log.warning("Precedent of explained action not saved: %s", exc)
    return out


def on_fact_cancelled(store: CaseStore, fact_id: str, by: str, reason: str = "cancelled by the owner") -> int:
    """[↩ ביטול] under an explained action: its precedents are not valid since now (history kept)."""
    done = 0
    try:
        for case in store.cases():
            if case.source.get("fact_id") == fact_id or fact_id in (case.source.get("facts") or ()):
                if case.invalid_at is None and len(case.source.get("facts") or ()) <= 1:
                    done += bool(store.invalidate(case.id, reason, by))
    except Exception as exc:  # noqa: BLE001
        log.warning("Precedent of explained action not cancelled: %s", exc)
    return done


# -- the precedent of a known mark ----------------------------------------------------------------------------------

def precedent_from_mark(mark: Mapping[str, Any], camera: str, chat_id: str = "",
                        replaces: Sequence[str] = ()) -> Case:
    """The draft precedent of the owner's "these are my workers" at *camera* (a house-wide mark: one per camera)."""
    until, at = float(mark.get("until") or 0), float(mark.get("at") or time.time())
    if mark.get("daily_from") and mark.get("daily_to"):
        hours = (str(mark["daily_from"]), str(mark["daily_to"]))
    elif _date(until) == _date(at):
        hours = (f"{max(6, dt.datetime.fromtimestamp(at).hour):02d}:00", _clock(until))
        hours = hours if hours[0] < hours[1] else ("00:00", "00:00")
    else:
        hours = ("00:00", "00:00")
    people = int(mark.get("people") or 0)
    scope = Scope(camera=camera, hours=hours, weekdays=ALL_DAYS, house_states=DAY_STATES, night=False, people=people,
                  vehicles=0, people_range=(1, people + EXTRA_PEOPLE if people else 0), until=until)
    source = {"origin": ORIGIN_KNOWN, "known_id": str(mark.get("id") or ""), "chat_id": str(chat_id or ""),
              "replaces": [str(r) for r in replaces if r]}
    text = str(mark.get("text") or "")
    return Case(id="", scope=scope, who=_who(text), title=text, note=text, effect=QUIET, created_at=at,
                source={k: v for k, v in source.items() if v})


def on_mark(store: CaseStore, mark: Mapping[str, Any], by: str, cameras: Sequence[str], chat_id: str = "",
            replaces: Sequence[str] = (), now: Optional[float] = None,
            same_people: Optional[Callable[[str, str], bool]] = None) -> List[SaveResult]:
    """A known mark was saved (or replaced *replaces*): its precedents follow, one per camera it covers (*cameras*:
    its camera, or every camera for a house-wide mark). Never raises."""
    out: List[SaveResult] = []
    try:
        now = time.time() if now is None else now
        camera = str(mark.get("camera") or "")
        wanted = [camera] if camera else list(dict.fromkeys(c for c in cameras if c))
        old_ids = {str(r) for r in replaces if r}
        for case in store.cases():
            if case.source.get("known_id") in old_ids and case.camera not in wanted and case.invalid_at is None:
                store.invalidate(case.id, "the owner corrected the mark", by)
        for cam in wanted:
            out.append(remember(store, precedent_from_mark(mark, cam, chat_id, replaces), by, now, same_people))
    except Exception as exc:  # noqa: BLE001
        log.warning("Precedent of known mark not saved: %s", exc)
    return out


def on_mark_cancelled(store: CaseStore, known_id: str, by: str) -> int:
    done = 0
    try:
        for case in store.cases():
            if case.source.get("origin") == ORIGIN_KNOWN and case.source.get("known_id") == known_id \
                    and case.invalid_at is None:
                done += bool(store.invalidate(case.id, "cancelled by the owner", by))
    except Exception as exc:  # noqa: BLE001
        log.warning("Precedent of known mark not cancelled: %s", exc)
    return done


def sync(store: CaseStore, facts: Iterable[Any], marks: Iterable[Mapping[str, Any]], cameras: Sequence[str],
         now: Optional[float] = None) -> int:
    """The owner memories saved before this link existed (the box's 2026-10-09 explanation and marks): each live
    one without a precedent gets it, written as its own author's. Idempotent. Returns how many were added."""
    now = time.time() if now is None else now
    added = 0
    try:
        sources = {(c.source.get("origin"), c.source.get("fact_id") or c.source.get("known_id"))
                   for c in store.cases(include_invalid=True)}
        sources |= {(ORIGIN_ACTIVITY, f) for c in store.cases(include_invalid=True)
                    for f in c.source.get("facts") or ()}
        for fact in facts:
            if (ORIGIN_ACTIVITY, fact.id) not in sources and fact.live(now):
                added += sum(r.kind == "saved" for r in on_fact(store, fact, str(fact.by or "owner"), now=now))
        for mark in marks:
            if (ORIGIN_KNOWN, str(mark.get("id"))) not in sources and float(mark.get("until") or 0) > now:
                added += sum(r.kind == "saved" for r in on_mark(store, mark, str(mark.get("by") or "owner"),
                                                                 cameras, now=now))
    except Exception as exc:  # noqa: BLE001
        log.warning("Owner memories not synced into the case memory: %s", exc)
    return added


# -- "also on future days?" ------------------------------------------------------------------------------------------

def _fact_case(store: CaseStore, fact_id: str) -> Optional[Case]:
    cases = [c for c in store.cases() if c.source.get("origin") == ORIGIN_ACTIVITY
             and (c.source.get("fact_id") == fact_id)]
    return cases[0] if cases else None


def _last_day_due(first_at: float, until: float, now: float) -> bool:
    """A memory of more than one day, on its last day, at 18:00 or at its end when that is earlier."""
    last = _date(until)
    return last > _date(first_at) and _date(now) == last and now >= min(_at(last, ASK_AT), until)


def _mark_info(mark: Mapping[str, Any]) -> Dict[str, Any]:
    keep = ("id", "camera", "text", "until", "at", "people", "daily_from", "daily_to")
    return {k: mark.get(k) for k in keep}


def mark_groups(events: Any, now: float) -> List[List[Dict[str, Any]]]:
    """The live crew marks (a daily window) grouped by the same people and the same end: the pergola workers
    marked at the pergola and at the main entrance are one crew, and get one question."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    try:
        marks = events.list_known(now) if events is not None else []
    except Exception as exc:  # noqa: BLE001
        log.warning("Marks not read for the question: %s", exc)
        marks = []
    for m in marks:
        if isinstance(m, dict) and m.get("daily_from") and m.get("daily_to") and m.get("until"):
            groups.setdefault((" ".join(str(m.get("text") or "").split()), round(float(m["until"]))), []).append(m)
    return list(groups.values())


def questions_due(store: CaseStore, activities: Any, now: Optional[float] = None,
                  events: Any = None) -> List[Dict[str, Any]]:
    """The one question per week-long memory that is due now: ``{key, ident, kind, case_id, fact_id, facts,
    marks, who, chat_id, last_day, seen_at}``. *kind* ``ending``: its last day (a memory of more than one day), at
    18:00 or its end when earlier - a crew's marks (*events*) as one question, which also covers the explained
    actions tied to them (``known_id``); an explained action of its own. *kind* ``again``: an explained action that
    ended and whose kind of event came back on a later day. Never one asked before."""
    now = time.time() if now is None else now
    out: List[Dict[str, Any]] = []
    folded: set = set()
    facts_now = []
    if activities is not None:
        try:
            facts_now = [f for f in activities.live(now - 2 * 86400) if not f.cancelled_at]
        except Exception as exc:  # noqa: BLE001
            log.warning("Explained actions not read for the question: %s", exc)
    for group in mark_groups(events, now - 6 * 3600):    # a mark that ended at 18:00 is still asked at 18:05
        until = float(group[0]["until"])
        if not _last_day_due(min(float(m.get("at") or until) for m in group), until, now):
            continue
        ids = sorted(str(m["id"]) for m in group)
        tied = [f for f in facts_now if f.known_id in ids and _date(f.until) == _date(until)]
        folded |= {f.id for f in tied}
        ident = f"mk_{ids[0]}"
        if store.asked(f"extend:{ident}") is not None:
            continue
        cases = [c for c in store.cases() if c.source.get("known_id") in ids or c.source.get("fact_id") in
                 {f.id for f in tied}]
        out.append({"key": f"extend:{ident}", "ident": ident, "kind": "ending",
                    "case_id": cases[0].id if cases else "", "fact_id": "", "facts": [f.id for f in tied],
                    "marks": [_mark_info(m) for m in group], "who": str(group[0].get("text") or ""),
                    "chat_id": next((str(c.source.get("chat_id")) for c in cases if c.source.get("chat_id")), ""),
                    "last_day": until, "seen_at": 0.0})
    for case in store.cases():
        fact_id = str(case.source.get("fact_id") or "")
        if case.source.get("origin") != ORIGIN_ACTIVITY or not fact_id or case.scope.until is None \
                or fact_id in folded:
            continue
        key = f"extend:{fact_id}"
        if store.asked(key) is not None:
            continue
        fact = activities.get(fact_id) if activities is not None else None
        if fact is None or fact.cancelled_at:
            continue
        common = {"key": key, "ident": fact_id, "case_id": case.id, "fact_id": fact_id, "facts": [fact_id],
                  "marks": [], "who": str(fact.cause), "chat_id": str(case.source.get("chat_id") or ""),
                  "last_day": float(fact.until)}
        last = _date(fact.until)
        if _date(now) == last and last > _date(fact.at or fact.until):
            if _last_day_due(float(fact.at or fact.until), float(fact.until), now):
                out.append({**common, "kind": "ending", "seen_at": 0.0})
            continue
        if fact.live(now):
            continue
        again = [m for m in store.seen_after_end(case.id)
                 if float(m.get("event_ts") or 0) > float(fact.until) and _date(float(m["event_ts"])) != _date(fact.at)]
        if again:
            out.append({**common, "kind": "again", "seen_at": float(again[-1]["event_ts"])})
    return out


def _crew_until(until: float, daily_to: str, now: float, days: int = CREW_DAYS) -> float:
    """The end of one more crew week: the daily window's end (or the old end's clock), *days* days on."""
    clock = str(daily_to or _clock(until))
    if _date(until) < _date(now):                       # it had ended: a new week from today
        return _at(_date(now) + dt.timedelta(days=days - 1), clock)
    return _at(_date(until) + dt.timedelta(days=days), clock)


def answer(store: CaseStore, activities: Any, key: str, choice: str, by: str,
           now: Optional[float] = None, events: Any = None) -> Dict[str, Any]:
    """The owner's tap on the question *key*: for its explained actions and its crew marks (*events*) alike.
    Returns ``{ok, kind, case, fact, until, who}``; ``ok`` False when it was answered before or its memory is gone
    (the brain then says nothing new)."""
    now = time.time() if now is None else now
    asked = store.asked(key)
    if asked is None or asked.get("answer") or choice not in ANSWERS:
        return {"ok": False, "kind": "answered" if asked and asked.get("answer") else "gone"}
    fact_ids = [str(f) for f in (asked.get("facts") or [asked.get("fact_id")]) if f]
    facts = [f for f in (activities.get(i) for i in fact_ids) if f is not None] if activities is not None else []
    marks = [m for m in asked.get("marks") or [] if isinstance(m, dict)]
    if not facts and not (marks and events is not None):
        return {"ok": False, "kind": "gone"}
    store.answer_asked(key, choice, by)
    mark_ids = {str(m.get("id")) for m in marks}
    mine = [c for c in store.cases() if c.source.get("fact_id") in fact_ids or c.source.get("known_id") in mark_ids]
    first = facts[0] if facts else None
    out: Dict[str, Any] = {"ok": True, "kind": choice, "fact": first, "who": str(asked.get("who") or ""),
                           "until": None}
    if choice == ENDED:
        for fact in facts:
            if not fact.cancelled_at:
                activities.cancel(fact.id, now)
        for m in marks:
            events.cancel_known(str(m.get("id")))
        for c in mine:
            store.invalidate(c.id, "ended (the owner)", by)
        out.update(until=now, case=store.get(mine[0].id) if mine else None)
        return out
    for c in mine:
        store.confirm(c.id, by, f"answer:{key}")
    if choice == WEEK:
        until = None
        for fact in facts:
            until = _crew_until(float(fact.until), fact.daily_to, now)
            extended = activities.update(fact.id, now, allow_ended=True, until=until)
            if extended is None:                # pruned from the book: the same explanation, written again
                extended = activities.add(fact.cameras, fact.actions, fact.cause, until, now, cause_en=fact.cause_en,
                                          place=fact.place, place_words=fact.place_words,
                                          owner_words=fact.owner_words, who=fact.who, known_id=fact.known_id,
                                          daily_from=fact.daily_from, daily_to=fact.daily_to, by=by,
                                          alert_id=fact.alert_id)
            for c in mine:
                if c.source.get("fact_id") == fact.id:
                    store.revise(c.id, replace(c.scope, until=until), by, "one more week (the owner)",
                                 source={"fact_id": extended.id, "facts": list(dict.fromkeys(
                                     [*(c.source.get("facts") or ()), fact.id, extended.id]))})
            out["fact"] = out["fact"] if out["fact"] is not first else extended
        for m in marks:
            until = _crew_until(float(m.get("until") or now), str(m.get("daily_to") or ""), now)
            saved = events.replace_known(str(m.get("id")), str(m.get("camera") or ""), str(m.get("text") or ""),
                                         by=by, until=until, now=now, people=m.get("people"),
                                         daily_from=str(m.get("daily_from") or ""),
                                         daily_to=str(m.get("daily_to") or ""))
            for c in mine:
                if c.source.get("known_id") == str(m.get("id")):
                    store.revise(c.id, replace(c.scope, until=until), by, "one more week (the owner)",
                                 source={"known_id": str(saved.get("id") or m.get("id"))})
        out.update(until=until, case=store.get(mine[0].id) if mine else None)
        return out
    # STANDING: every weekday, the same hours, no end. The week-long memories keep their own end.
    for c in mine:
        hours = c.scope.hours
        if hours == ("00:00", "00:00") and first is not None:
            hours = fact_hours(first)
        store.revise(c.id, replace(c.scope, until=None, weekdays=WORKWEEK, hours=hours), by,
                     "standing: every weekday (the owner)")
    out.update(case=store.get(mine[0].id) if mine else None)
    return out


# -- the nightly routine proposals ------------------------------------------------------------------------------------

ROUTINE_MODES = ("off", "shadow", "on")


def routine_mode(settings: Mapping[str, Any]) -> str:
    """box.yaml ``routine_proposals: off|shadow|on``; off by default (the owner sees an example first)."""
    value = settings.get("routine_proposals", "off") if isinstance(settings, Mapping) else "off"
    if value is False:
        return "off"
    if value is True:
        return "on"
    text = str(value or "off").strip().lower()
    return text if text in ROUTINE_MODES else "off"


def event_records(events_path: str, since: float) -> List[Dict[str, Any]]:
    """The event archive (events.jsonl: one closed session per line) as the routine proposer's records: one per
    observation since *since* (``camera, ts, final_label, people, event_id``)."""
    out: List[Dict[str, Any]] = []
    try:
        with open(events_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(s, dict) or float(s.get("last_active") or s.get("opened") or 0) < since:
                    continue
                for o in s.get("observations") or ():
                    if isinstance(o, dict) and float(o.get("ts") or 0) >= since:
                        out.append({"camera": str(s.get("camera") or ""), "ts": float(o["ts"]),
                                    "final_label": str(o.get("label") or ""), "people": int(o.get("people") or 0),
                                    "event_id": str(o.get("alert_id") or s.get("id") or "")})
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("Event archive not read for routines: %s", exc)
    return out


def nightly_routines(store: CaseStore, events_path: str, mode: str, now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Run the routine proposer over the archive's last 28 days. ``shadow``: each new proposal is logged once
    (journal + log line), nothing is sent. ``on``: the same, returned for the brain to send once with buttons.
    ``off``: nothing. Returns ``[{rid, proposal}]`` of the new ones."""
    from . import routines  # noqa: PLC0415

    if mode not in ("shadow", "on"):
        return []
    now = time.time() if now is None else now
    records = event_records(events_path, now - routines.HISTORY_DAYS * 86400)
    out = []
    for proposal in routines.propose(records, store, now):
        rid = store.note_routine(proposal, mode)
        if rid:
            log.info("routine proposal %s (%s): %s", rid, mode, proposal.get("text_en"))
            out.append({"rid": rid, "proposal": proposal})
    return out


def answer_routine(store: CaseStore, rid: str, yes: bool, by: str, embed=None) -> Optional[Case]:
    """The owner's yes / no to a sent routine proposal (once)."""
    from . import routines  # noqa: PLC0415

    item = store.routine_proposals().get(rid)
    if item is None or item["proposal"].get("key") in store.routine_answers():
        return None
    return routines.answer(store, item["proposal"], yes, by, embed) if yes else (
        routines.answer(store, item["proposal"], False, by) or None)


# -- "what do you remember?" ---------------------------------------------------------------------------------------

def long_term(store: CaseStore, now: Optional[float] = None) -> List[Case]:
    """The live precedents, oldest first (shadow and active; an ended week is history, not listed)."""
    now = time.time() if now is None else now
    return sorted((c for c in store.cases() if c.live and (c.scope.until is None or c.scope.until > now)),
                  key=lambda c: c.created_at)


__all__ = ["on_fact", "on_fact_cancelled", "on_mark", "on_mark_cancelled", "sync", "questions_due", "answer",
           "precedent_from_fact", "precedent_from_mark", "routine_mode", "nightly_routines", "answer_routine",
           "event_records", "long_term", "WEEK", "ENDED", "STANDING", "ANSWERS"]
