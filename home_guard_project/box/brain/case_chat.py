"""The chat side of the long-term memory (case_memory/link.py, 2026-10-09).

- What the owner explains is saved for the week (activity_chat, the known marks) and, through these hooks, as a
  precedent that learns in shadow: :func:`after_fact`, :func:`after_fact_cancelled`, :func:`after_mark`,
  :func:`after_mark_cancelled`. The owner reads nothing new for it (owner rule: no new message types).
- "מה אתה זוכר?" lists the precedents too, under their own heading (:func:`memory_section`).
- One question per explained action, when it is about to end or comes back after it ended (:func:`tick`, every few
  minutes in a background thread), with three buttons. The owner's tap (:func:`button`) extends it a week, ends it,
  or makes it standing.
- Nightly routine proposals: box.yaml ``routine_proposals: off|shadow|on`` (off by default; shadow only logs).

Buttons ride the existing ``kn:x:<id>`` callback (telegram_agent routes it to ``known_button``) with ids
``ce.<w|e|s>.<fact id>`` and ``ce.<ry|rn>.<routine id>``; ``x`` also clears the keyboard after the tap.
Every entry point catches and logs: the week-long memory and the chat never fail because of this one.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..case_memory import link
from ..case_memory import texts as cm_texts
from . import known_memory as km
from .i18n import t
from .registry import display

log = logging.getLogger("box.brain.case_chat")

TICK_SEC = 300.0
ROUTINES_AFTER = "03:00"             # the nightly run: the first tick after this, once a day
_CHOICES = {"w": link.WEEK, "e": link.ENDED, "s": link.STANDING}


def store_of(services: Any) -> Any:
    return getattr(services, "cases", None) if services is not None else None


def _by(who: Any) -> str:
    who = who if isinstance(who, dict) else {}
    return str(who.get("name") or who.get("user_id") or "owner")


# -- the hooks: every owner memory also learns for the long term -------------------------------------------------

def after_fact(services: Any, fact: Any, entry: Optional[Dict[str, Any]], chat_id: str, who: Any,
               now: Optional[float] = None) -> List[Any]:
    store = store_of(services)
    if store is None or fact is None:
        return []
    return link.on_fact(store, fact, _by(who), entry, str(chat_id or ""), now, km.same_people)


def after_fact_cancelled(services: Any, fact_id: str, who: Any) -> int:
    store = store_of(services)
    return link.on_fact_cancelled(store, fact_id, _by(who)) if store is not None else 0


def after_mark(services: Any, mark: Dict[str, Any], snapshot: Any, chat_id: str, who: Any,
               replaces: Sequence[str] = (), now: Optional[float] = None) -> List[Any]:
    """*mark*: the book's receipt or entry (``id, camera, text, until, daily_from, daily_to, people``)."""
    store = store_of(services)
    if store is None or not isinstance(mark, dict):
        return []
    cameras = [c.name for c in getattr(snapshot, "cameras", ()) or ()]
    mark = dict(mark, at=mark.get("at") or (time.time() if now is None else now))
    return link.on_mark(store, mark, _by(who), cameras, str(chat_id or ""), replaces, now, km.same_people)


def after_mark_cancelled(services: Any, known_id: str, who: Any) -> int:
    store = store_of(services)
    return link.on_mark_cancelled(store, known_id, _by(who)) if store is not None else 0


# -- the words ----------------------------------------------------------------------------------------------------

def _day(ts: float, lang: str) -> str:
    return km.day_text(ts, lang)


def _date(ts: float) -> str:
    moment = dt.datetime.fromtimestamp(ts)
    return f"{moment.day}.{moment.month}"


def _where(case: Any, fact: Any, snapshot: Any, lang: str) -> str:
    """"במדרגות של הכניסה הראשית" - or only the camera when the owner's words already say the place."""
    from .activity_chat import definite_he, where_text  # noqa: PLC0415

    if fact is not None:
        place_words = [w for w in (fact.place_words, (link_place_he(fact.place) if lang == "he" else "")) if w]
        if not any(w and w in str(fact.cause or "") for w in place_words):
            return where_text(fact, snapshot, lang)
    name = display(snapshot, case.camera, lang)
    if str(lang).startswith("he"):
        name = definite_he(name)
        return f"ב{name[1:]}" if name.startswith("ה") else f"ב{name}"
    return f"at {name}"


def link_place_he(place: str) -> str:
    from .. import activity_memory as am  # noqa: PLC0415

    return str(am.PLACES.get(place, {}).get("he") or "")


def _actions(case: Any, lang: str) -> str:
    from .. import activity_memory as am  # noqa: PLC0415

    return am.actions_text(case.scope.actions, lang, short=True)


def case_row(case: Any, snapshot: Any, lang: str, now: float, activities: Any = None,
             live_source: bool = False) -> str:
    """"🗂️ בכניסה הראשית: שכיבה/כיפוף במדרגות = החשמלאים מתקינים לדים, כל יום 07:00–18:00, עד יום ה׳ 15.10 · לומד, 0
    מתוך 3 אישורים"."""
    s = case.scope
    fact = activities.get(case.source.get("fact_id")) if activities is not None and case.source.get("fact_id") else None
    if s.actions:
        what = f"{_where(case, fact, snapshot, lang)}: {_actions(case, lang)} = {case.title}"
    else:
        what = km.who_where(case.title, case.camera, snapshot, lang)
    status = t("cm_active", lang) if case.status == "active" else t("cm_learning", lang, n=case.confirmations)
    if live_source:
        # The week-long item above already says its hours and end: only what the long term adds (no repeats).
        return f"🗂️ {what} · {status}"
    days = cm_texts.days_text(s.weekdays, "he" if str(lang).startswith("he") else "en")
    hours = "" if tuple(s.hours) == ("00:00", "00:00") else f" {cm_texts.hours_text(s.hours)}"
    end = t("cm_until", lang, day=_day(s.until, lang)) if s.until is not None else t("cm_standing", lang)
    return f"🗂️ {what}, {days}{hours}, {end} · {status}"


def memory_section(services: Any, snapshot: Any, lang: str, now: float) -> str:
    """The long-term precedents for "מה אתה זוכר?", under their own heading ("" when there are none)."""
    store = store_of(services)
    if store is None:
        return ""
    try:
        activities, events = getattr(services, "activities", None), getattr(services, "events", None)
        live = {f.id for f in (activities.live(now) if activities is not None else [])}
        live |= {str(k.get("id")) for k in (events.list_known(now) if events is not None else [])}
        rows = [case_row(c, snapshot, lang, now, activities, live_source=bool(
                    c.scope.until is not None and {c.source.get("fact_id"), c.source.get("known_id")} & live))
                for c in link.long_term(store, now)]
    except Exception as exc:  # noqa: BLE001
        log.warning("Long-term memory not listed: %s", exc)
        return ""
    return t("cm_long_term", lang, rows="\n".join(rows)) if rows else ""


def _cameras_where(cameras: Sequence[str], snapshot: Any, lang: str) -> str:
    """"(בפרגולה ובכניסה הראשית)" / "(at the pergola and at the main entrance)": a crew's marked cameras."""
    from .activity_chat import definite_he  # noqa: PLC0415

    names = list(dict.fromkeys(display(snapshot, c, lang) for c in cameras if c))
    if not names:
        return ""
    if str(lang).startswith("he"):
        parts = []
        for n in names:
            d = definite_he(n)
            parts.append(f"ב{d[1:]}" if d.startswith("ה") else f"ב{d}")
        return "(" + " ו".join(parts) + ")"
    return "(" + " and ".join(f"at {n}" for n in names) + ")"


def _question_where(q: Dict[str, Any], services: Any, snapshot: Any, lang: str) -> str:
    store, activities = store_of(services), getattr(services, "activities", None)
    if q.get("marks"):
        return _cameras_where([str(m.get("camera") or "") for m in q["marks"]], snapshot, lang)
    case = store.get(q["case_id"])
    fact = activities.get(q["fact_id"]) if activities is not None and q.get("fact_id") else None
    return _where(case, fact, snapshot, lang)


def question(q: Dict[str, Any], services: Any, snapshot: Any, lang: str) -> Tuple[str, Tuple[Tuple[Tuple[str, str], ...], ...]]:
    """The one question and its three buttons (an explained action, or a crew's marks with what is tied to them)."""
    who = str(q.get("who") or "")
    where = _question_where(q, services, snapshot, lang)
    if q["kind"] == "again":
        text = t("cm_ask_again", lang, who=who, where=where,
                 time=dt.datetime.fromtimestamp(q["seen_at"]).strftime("%H:%M"))
    else:
        text = t("cm_ask_ending", lang, who=who, where=where, date=_date(q["last_day"]))
    text = " ".join(text.split())
    ident = q.get("ident") or q["fact_id"]
    rows = (((t("cm_btn_week", lang), f"kn:x:ce.w.{ident}"), (t("cm_btn_ended", lang), f"kn:x:ce.e.{ident}")),
            ((t("cm_btn_standing", lang), f"kn:x:ce.s.{ident}"),))
    return text, rows


def routine_question(item: Dict[str, Any], snapshot: Any, lang: str) -> Tuple[str, Tuple[Tuple[Tuple[str, str], ...], ...]]:
    p, rid = item["proposal"], item["rid"]
    who = t("cm_vehicle", lang) if p.get("vehicles") and not p.get("people") else t("cm_someone", lang)
    clock = f'{int(p["minute"]) // 60:02d}:{int(p["minute"]) % 60:02d}'
    text = t("cm_routine_ask", lang, who=who, camera=display(snapshot, p["camera"], lang), days=p["days_seen"],
             time=clock)
    return text, (((t("cm_btn_routine_yes", lang), f"kn:x:ce.ry.{rid}"), (t("cm_btn_routine_no", lang),
                                                                          f"kn:x:ce.rn.{rid}")),)


# -- the owner's tap ------------------------------------------------------------------------------------------------

def button(services: Any, snapshot: Any, code: str, who: Any, lang: str, now: Optional[float] = None) -> Optional[str]:
    """``ce.<w|e|s>.<fact id>`` / ``ce.<ry|rn>.<routine id>``: the answer line, or None when the question was
    answered before or its memory is gone (nothing is sent again)."""
    store, activities = store_of(services), getattr(services, "activities", None)
    if store is None:
        return None
    now = time.time() if now is None else now
    try:
        _, kind, ident = code.split(".", 2)
        if kind in ("ry", "rn"):
            item = store.routine_proposals().get(ident)
            if item is None or item["proposal"].get("key") in store.routine_answers():
                return None
            saved = link.answer_routine(store, ident, kind == "ry", _by(who))
            return t("cm_routine_saved", lang) if kind == "ry" and saved is not None else t("cm_routine_no", lang)
        choice = _CHOICES.get(kind)
        if choice is None:
            return None
        asked = store.asked(f"extend:{ident}") or {}
        out = link.answer(store, activities, f"extend:{ident}", choice, _by(who), now,
                          events=getattr(services, "events", None))
        if not out.get("ok"):
            return None
        case, fact = out.get("case"), out.get("fact")
        q = dict(asked, case_id=case.id if case is not None else "", fact_id=fact.id if fact is not None else "")
        where = _question_where(q, services, snapshot, lang) if (asked.get("marks") or case is not None) else ""
        name = str(asked.get("who") or (fact.cause if fact is not None else ""))
        if choice == link.ENDED:
            if asked.get("marks") or case is None or not case.scope.actions:
                return " ".join(t("cm_ended_marks", lang, who=name, where=where).split())
            return t("cm_ended_saved", lang, actions=_actions(case, lang), where=where)
        if choice == link.WEEK:
            return " ".join(t("cm_week_saved", lang, who=name, where=where, day=_day(out["until"], lang)).split())
        hours = case.scope.hours if case is not None else (str((asked.get("marks") or [{}])[0].get("daily_from")),
                                                           str((asked.get("marks") or [{}])[0].get("daily_to")))
        return " ".join(t("cm_standing_saved", lang, who=name, where=where, hours=cm_texts.hours_text(hours)).split())
    except Exception as exc:  # noqa: BLE001
        log.warning("Case memory answer failed: %s", exc)
        return None


# -- the background keeper -----------------------------------------------------------------------------------------

def tick(services: Any, send: Callable[..., Dict[str, Any]], chat_ids: Sequence[str], settings: Dict[str, Any],
         snapshot: Any, now: Optional[float] = None) -> Dict[str, int]:
    """One round: older owner memories get their precedent, due questions are sent once, and the nightly routine
    run happens once a day after 03:00. *send(chat_id, text, rows=...)* is the deliverer's ``text``. Never raises."""
    now = time.time() if now is None else now
    done = {"synced": 0, "asked": 0, "routines": 0}
    store, activities = store_of(services), getattr(services, "activities", None)
    if store is None:
        return done
    lang = str((settings or {}).get("owner_language") or "en")
    default_chat = str(chat_ids[0]) if chat_ids else ""
    try:
        facts = activities.live(now) if activities is not None else []
        events = getattr(services, "events", None)
        marks = events.list_known(now) if events is not None else []
        cameras = [c.name for c in getattr(snapshot, "cameras", ()) or ()]
        done["synced"] = link.sync(store, facts, marks, cameras, now)
    except Exception as exc:  # noqa: BLE001
        log.warning("Owner memories not synced: %s", exc)
    try:
        for q in link.questions_due(store, activities, now, getattr(services, "events", None)):
            chat = q["chat_id"] or default_chat
            if not chat:
                continue
            text, rows = question(q, services, snapshot, lang)
            result = send(chat, text, rows=rows)
            if isinstance(result, dict) and result.get("ok"):
                store.note_asked(q["key"], case_id=q["case_id"], fact_id=q["fact_id"], kind=q["kind"], chat_id=chat,
                                 facts=q.get("facts") or [], marks=q.get("marks") or [], who=q.get("who") or "",
                                 ident=q.get("ident") or "",
                                 message_id=result.get("message_id"))
                done["asked"] += 1
                log.info("case memory asked once (%s, %s): %s", q["kind"], q["fact_id"], text)
    except Exception as exc:  # noqa: BLE001
        log.warning("Case memory question not sent: %s", exc)
    try:
        mode = link.routine_mode(settings or {})
        day = dt.datetime.fromtimestamp(now).date().isoformat()
        if mode != "off" and dt.datetime.fromtimestamp(now).strftime("%H:%M") >= ROUTINES_AFTER \
                and store.asked(f"routines:{day}") is None:
            store.note_asked(f"routines:{day}", mode=mode)
            events_path = getattr(getattr(services, "events", None), "events_path", "")
            for item in link.nightly_routines(store, events_path, mode, now):
                done["routines"] += 1
                if mode == "on" and default_chat:
                    text, rows = routine_question(item, snapshot, lang)
                    send(default_chat, text, rows=rows)
    except Exception as exc:  # noqa: BLE001
        log.warning("Routine proposals not run: %s", exc)
    return done


def start(agent: Any, deliverer: Any, chat_ids: Sequence[str], read_settings: Callable[[], Dict[str, Any]],
          interval: float = TICK_SEC, stop: Optional[threading.Event] = None) -> Optional[threading.Thread]:
    """The keeper's background thread (inference starts it next to the assistant). None without a case store."""
    services = getattr(agent, "services", None)
    if store_of(services) is None or deliverer is None:
        return None

    def loop() -> None:
        while not (stop and stop.is_set()):
            try:
                settings = read_settings() or {}
                try:
                    snapshot = agent.registry.snapshot()
                except Exception:  # noqa: BLE001
                    snapshot = None
                tick(services, deliverer.text, list(chat_ids), settings, snapshot)
            except Exception as exc:  # noqa: BLE001
                log.warning("Case memory keeper round failed: %s", exc)
            if stop is not None:
                if stop.wait(interval):
                    return
            else:
                time.sleep(interval)

    thread = threading.Thread(target=loop, name="case-keeper", daemon=True)
    thread.start()
    log.info("Case memory keeper on (questions once, routine proposals: %s)", link.routine_mode(read_settings() or {}))
    return thread
