"""The owner explains an ACTION seen in an alert, in plain text (2026-10-09): the conversation half of
activity_memory.

13:54 (no reply, nine minutes after the last alert) "אלה מקרים תקינים הם מתקינים זה שני אנשים שמרכיבים את
הלדים במדרגות", then "אני מדבר על שני האנשים שאחד מהם התכופף זה רגיל", then a reply to the 13:44 pergola alert
"חשמלאים שמנסים להתקין לד למדרגות" and "לא בפרגולה אני מתכוון בכניסה הראשית במדרגות". The old bot answered with
the pergola workers' mark, then baseline counts ("בערך 9 פעמים"), then the mark again.

Now a plain message about a recent alert (a reply to it, or a statement while alerts of the last hour are open)
that says what was seen is normal or explains it goes to ONE model call (the big model) that picks the alert it is
about and the actions, the place and the cause in the owner's words. The code saves an ActivityFact
(activity_memory) and answers about THAT explanation in one natural line, with [↩ ביטול]:

    🧠 הבנתי, השכיבה, הזחילה והכיפוף במדרגות של הכניסה הראשית זה החשמלאים שמתקינים לדים. עד 18:00 זה ייחשב
    אצלי רגיל.

A follow-up adds to or corrects the same fact (never a second one). The window: the owner's own "עד 18", else the
live known mark of that camera (its daily hours and end), else one question "עד מתי?". A "🏷️ תיוג אחר" answer is
a TAG and never reaches this module.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import activity_memory as am
from . import known_memory as km
from .i18n import t
from .registry import display, resolve_camera

log = logging.getLogger("box.brain.activity_chat")

EXPLAIN_WINDOW_SEC = 3600.0     # an alert this recent may be the one explained (13:54 explained the 13:25 red)
FOLLOW_SEC = 900.0              # a fact this recent may be added to or corrected by the next messages

_NORMAL = re.compile(
    r"(?<![א-ת])[ושהכ]{0,2}(?:תקין|תקינה|תקינים|רגיל|רגילה|רגילים|בסדר|לא חשוד|לא בעיה|אין בעיה|נורמלי|טבעי)(?![א-ת])|"
    r"\b(?:normal|fine|ok|okay|nothing wrong|not suspicious|no problem)\b", re.IGNORECASE)
_EXPLAINS = re.compile(
    r"(?<![א-ת])[ושה]{0,2}(?:עובד|עובדים|עובדת|פועל|פועלים|חשמלאי|חשמלאים|אינסטלטור|טכנאי|טכנאים|קבלן|גנן|צבעי|"
    r"מתקין|מתקינים|מרכיב|מרכיבים|מתקן|מתקנים|בונה|בונים|מנקה|מנקים|מסדר|מסדרים|עבודה|עבודות|שיפוץ|תיקון|התקנה|"
    r"להתקין|לתקן|להרכיב|מתכוון|מתכוונת|מדבר על)(?![א-ת])|"
    r"\b(?:work(?:er|ers|ing)?|electrician|plumber|technician|contractor|install\w*|fix\w*|repair\w*|I mean)\b",
    re.IGNORECASE)
_QUESTION = re.compile(r"\?|(?<![א-ת])ו?(?:האם|מי|מה|למה|איך|מתי|איפה|כמה)(?![א-ת])|"
                       r"^\s*(?:who|what|why|how|when|where|is|are|do|does|did|can)\b", re.IGNORECASE)
_ACK = re.compile(r"^\s*(?:(?:סבבה|בסדר|הבנתי|תודה|תודה רבה|אוקי|אוקיי|אוקיי?|מעולה|יופי|קיבלתי|טוב|נהדר|אחלה|"
                  r"ok|okay|thanks|thank you|got it|cool|great|fine|sure|👍|🙏|👌)[\s.!,]*)+$", re.IGNORECASE)


_TAG = re.compile(r"(?<![א-ת])[ולשה]?(?:תיוג|התיוג|תייג|תתייג|לתייג|תג|תווית)(?![א-ת])|\b(?:tag|tags|tagged|retag|label)\b",
                  re.IGNORECASE)


def asks_to_tag(text: str) -> bool:
    """The owner explicitly speaks of the clip's tag ("שנה תיוג", "תתייג את זה כ..."): only then a plain message
    may become a tag."""
    return bool(_TAG.search(str(text or "")))


_STOP_REPEATING = re.compile(r"(?<![א-ת])(?:לא צריך|אין צורך|אל ת|די|תפסיק|מספיק)\s*(?:\S+\s+){0,2}?ל?(?:חזור|לחזור|תחזור|חוזר)|"
                             r"\b(?:stop|no need to|don'?t)\s+(?:\w+\s+){0,2}?repeat", re.IGNORECASE)


def stop_repeating(text: str) -> bool:
    """"הבנתי, אתה לא צריך לחזור על זה" (13:55:40): a short "בסדר." - never the thing again."""
    text = str(text or "")
    return "?" not in text and bool(_STOP_REPEATING.search(text))


def is_ack(text: str) -> bool:
    """"סבבה", "בסדר הבנתי", "תודה", "👍": an acknowledgement and nothing else."""
    return bool(_ACK.match(str(text or "")))


def recent_alerts(state: Any, now: float, within: float = EXPLAIN_WINDOW_SEC) -> List[Tuple[str, Dict[str, Any]]]:
    """``(handle, entry)`` of the alerts of the last *within* seconds, newest first."""
    rows = []
    for handle, entry in (getattr(state, "handles", None) or {}).items():
        if not isinstance(entry, dict) or entry.get("kind") != "event":
            continue
        try:
            ts = float(entry.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if 0 <= now - ts <= within:
            rows.append((handle, entry))
    return sorted(rows, key=lambda kv: float(kv[1].get("ts") or 0), reverse=True)


def recent_facts(activities: Any, now: float, within: float = FOLLOW_SEC) -> List[am.ActivityFact]:
    if activities is None:
        return []
    return [f for f in activities.live(now) if 0 <= now - float(f.at or 0) <= within
            or any(now - float(h.get("at") or 0) <= within for h in f.history)]


def candidate(text: str, state: Any, alert_handle: Optional[str], now: float, activities: Any) -> bool:
    """Cheap: may *text* explain an action of a recent alert (or follow up a fact just said)? Then one model call
    decides. Never a question, an acknowledgement or a command."""
    text = str(text or "").strip()
    if activities is None or len(text) < 4 or text.startswith("/") or is_ack(text) or _QUESTION.search(text):
        return False
    if not (alert_handle or recent_alerts(state, now) or recent_facts(activities, now)):
        return False
    return bool(_NORMAL.search(text) or _EXPLAINS.search(text) or am.actions_in(text) or am.places_in(text)
                or (recent_facts(activities, now) and re.search(r"(?<![א-ת])לא ב|\bnot (?:at|in|on)\b", text)))


def definite_he(name: str) -> str:
    """"כניסה ראשית" -> "הכניסה הראשית", "פרגולה" -> "הפרגולה"; a number ("מצלמה 6") stays."""
    name = " ".join(str(name or "").split())
    if not name or re.search(r"\d", name) or not re.search(r"[א-ת]", name):
        return name
    return " ".join(w if w.startswith("ה") else "ה" + w for w in name.split())


def where_text(fact: am.ActivityFact, snapshot: Any, lang: str) -> str:
    """"במדרגות של הכניסה הראשית" / "בכניסה הראשית" / "on the stairs at the main entrance"."""
    names = [display(snapshot, c, lang) for c in fact.cameras]
    place = am.place_text(fact, lang)
    if str(lang).startswith("he"):
        cams = " ו".join(definite_he(n) for n in names)
        if place:
            return f"{place} של {cams}" if cams else place
        return f"ב{cams[1:]}" if cams.startswith("ה") else (f"ב{cams}" if cams else "")
    cams = " and ".join(names)
    return " ".join(x for x in (place, f"at {cams}" if cams else "") if x)


def when_text(fact: am.ActivityFact, lang: str) -> str:
    end = fact.daily_to or dt.datetime.fromtimestamp(fact.until).strftime("%H:%M")
    return t("act_until", lang, end=end)


def memory_rows(activities: Any, snapshot: Any, lang: str, now: float) -> List[str]:
    """For "מה אתה זוכר?": "בכניסה הראשית: שכיבה/כיפוף במדרגות = החשמלאים מתקינים לדים, כל יום עד 18:00"."""
    rows = []
    for f in (activities.live(now) if activities is not None else []):
        names = [display(snapshot, c, lang) for c in f.cameras]
        cams = " ו".join(definite_he(n) for n in names) if str(lang).startswith("he") else " and ".join(names)
        at_cams = (f"ב{cams[1:]}" if cams.startswith("ה") else f"ב{cams}") if str(lang).startswith("he") else f"at {cams}"
        acts = am.actions_text(f.actions, lang, short=True)
        place = am.place_text(f, lang)
        if f.daily_to and dt.datetime.fromtimestamp(f.until).date() > dt.datetime.fromtimestamp(now).date():
            when = t("act_daily_until", lang, end=f.daily_to)
        else:
            when = t("act_until", lang, end=f.daily_to or dt.datetime.fromtimestamp(f.until).strftime("%H:%M"))
        rows.append(f"{at_cams}: {acts}{' ' + place if place else ''} = {f.cause}, {when}")
    return rows


def context_lines(activities: Any, snapshot: Any, lang: str, now: float) -> List[str]:
    """[OWNER EXPLAINED ACTIONS] for the model: what the owner said an action means, and that it is saved."""
    rows = []
    for f in (activities.live(now) if activities is not None else []):
        names = ", ".join(display(snapshot, c, lang) for c in f.cameras) or "all cameras"
        rows.append(f'{names}: {"/".join(f.actions)}{" at " + f.place if f.place else ""} = "{f.cause}" '
                    f'(daily {f.daily_from or "-"}-{f.daily_to or "-"}, until '
                    f'{dt.datetime.fromtimestamp(f.until).strftime("%a %d.%m %H:%M")}, id {f.id})')
    if not rows:
        return []
    return ["[OWNER EXPLAINED ACTIONS] saved; alerts with these actions at these cameras in their hours are kept "
            "quiet (a red gets a second look first): " + "; ".join(rows) + ". Never offer to save them again."]


_SYSTEM = """You read one message the owner of a home-security box wrote in the family's chat, and decide whether it
explains an ACTION seen in one of the recent alerts as normal (work, a known person, a routine), so the box can
remember what that action means at that camera.

Answer with ONE JSON object and nothing else:
{"kind": "explain" | "correct" | "none",
 "event": "<the handle of the alert it is about, e.g. E4, or empty>",
 "camera": "<a camera the owner NAMES in this message, as he wrote it, or empty>",
 "fact": "<the id of a fact said in the last minutes that this message repeats, adds to or corrects, or empty>",
 "actions": ["<from ACTIONS: what the owner says is normal, or what the alert shows that he explains>"],
 "place": "<from PLACES, or empty>",
 "place_words": "<the owner's own words for the place, or empty>",
 "cause": "<a short noun phrase in the owner's language, from his words: who and what work, e.g. החשמלאים
           שמתקינים לדים - never a copy of his whole message>",
 "cause_en": "<the same in English>",
 "who_mark": "<the id of a live mark when he says these are those people, else empty>",
 "until": "<HH:MM only if he said until when, else empty>"}

Rules
- "explain": he says something seen in an alert is normal or explains why it happened ("אלה מקרים תקינים הם
  מתקינים...", "אחד מהם התכופף זה רגיל", "חשמלאים שמנסים להתקין לד"). "correct": he corrects the camera, the
  place or the actions of a fact said in the last minutes ("לא בפרגולה, אני מתכוון בכניסה הראשית").
  "none": anything else - a question, a complaint about the box's messages, an acknowledgement, a request, a
  remark that explains no action.
- event: a reply is usually about the alert it replies to, but when his words fit another recent alert better
  (the camera or place he names, the number of people, the action he names), choose that one.
- When the message repeats, adds to or corrects a fact said in the last minutes (same people, same work), give
  that fact's id; never a new fact for the same work.
- actions only from ACTIONS. Never "car_door" unless he speaks about a car door.
- cause in his words; never invent who they are."""


def _parse(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def ask_model(model: Any, text: str, state: Any, snapshot: Any, alert_handle: Optional[str], activities: Any,
              marks: Sequence[Dict[str, Any]], now: float, lang: str, usage: Dict[str, List[int]]) -> Dict[str, Any]:
    """The one model call. {} when it fails."""
    hm = lambda ts: dt.datetime.fromtimestamp(float(ts or 0)).strftime("%H:%M")  # noqa: E731
    alerts = []
    for handle, e in recent_alerts(state, now)[:6]:
        alerts.append(f'{handle} {hm(e.get("ts"))} {display(snapshot, str(e.get("camera") or ""), lang)}'
                      f'{" (" + str(e.get("label")) + ")" if e.get("label") else ""}: '
                      f'{str(e.get("observation") or e.get("summary") or "")[:400]}')
    facts = [f'{f.id}: {", ".join(display(snapshot, c, lang) for c in f.cameras)}; place {f.place or "-"}; actions '
             f'{"/".join(f.actions)}; cause "{f.cause}"' for f in recent_facts(activities, now)]
    live = [f'{k.get("id")}: "{k.get("text")}" at {display(snapshot, str(k.get("camera") or ""), lang) if k.get("camera") else "the whole house"}'
            f' daily {k.get("daily_from") or "-"}-{k.get("daily_to") or "-"}' for k in marks]
    said = [f'{hm(turn.get("ts"))} "{str(turn.get("text"))[:200]}"' for turn in (getattr(state, "turns", None) or [])[-8:]
            if isinstance(turn, dict) and turn.get("kind") != "alert" and turn.get("text")
            and now - float(turn.get("ts") or 0) <= FOLLOW_SEC]
    cams = ", ".join(display(snapshot, c.name, lang) for c in getattr(snapshot, "cameras", ()) or ())
    user = "\n".join([
        f"[NOW] {hm(now)}",
        "[RECENT ALERTS] newest first:\n" + ("\n".join(alerts) or "none"),
        f"[REPLIES TO] {alert_handle or 'nothing'}",
        "[FACTS SAID IN THE LAST MINUTES]\n" + ("\n".join(facts) or "none"),
        "[LIVE MARKS]\n" + ("\n".join(live) or "none"),
        "[HIS LAST MESSAGES]\n" + ("\n".join(said) or "none"),
        f"[CAMERAS] {cams}",
        f"[ACTIONS] {', '.join(am.ACTION_KEYS)}",
        f"[PLACES] {', '.join(am.PLACES)}",
        f'[OWNER NOW] "{text}"'])
    try:
        msg = model.chat([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}], [])
        spent = usage.setdefault("big", [0, 0])
        spent[0] += int(msg.usage[0] or 0)
        spent[1] += int(msg.usage[1] or 0)
        if getattr(msg, "error", ""):
            raise RuntimeError(msg.error)
        return _parse(msg.content or "")
    except Exception as exc:  # noqa: BLE001 - the message then goes the usual way
        log.warning("Activity explanation not read: %s", exc)
        return {}


def window(book: Any, camera: str, words: str, now: float) -> Optional[Dict[str, Any]]:
    """``{until, daily_from, daily_to, who, known_id}``: the owner's own end today, else the live mark that covers
    the camera (its daily hours and end), else None (ask "עד מתי?")."""
    said = km.until_from_words(words, now)
    if said is not None and said > now:
        return {"until": said, "daily_from": "", "daily_to": "", "who": "", "known_id": ""}
    for k in km.live_marks(book, now):
        if not km.covers(k, camera):
            continue
        out = {"until": float(k.get("until") or 0), "daily_from": str(k.get("daily_from") or ""),
               "daily_to": str(k.get("daily_to") or ""), "who": str(k.get("text") or ""), "known_id": str(k.get("id") or "")}
        if out["until"] > now:
            return out
    return None


def handle(model: Any, ctx: Any, activities: Any, events_book: Any, alert_handle: Optional[str], snapshot: Any,
           now: float, usage: Dict[str, List[int]]) -> Optional[Dict[str, Any]]:
    """An explanation of an action, saved: ``{"text", "rows", "pending", "note"}``; None when the message is not
    one (the usual path answers it). Never raises."""
    try:
        text, state, lang = ctx.text, ctx.state, ctx.lang
        if not candidate(text, state, alert_handle, now, activities):
            return None
        marks = km.live_marks(events_book, now)
        got = ask_model(model, text, state, snapshot, alert_handle, activities, marks, now, lang, usage)
        kind = str(got.get("kind") or "none")
        if kind not in ("explain", "correct"):
            return None
        handles = dict(recent_alerts(state, now))
        event = str(got.get("event") or "").strip().upper()
        if event not in handles:
            event = alert_handle if alert_handle in handles else (next(iter(handles)) if handles else "")
        entry = handles.get(event) or {}
        event_text = f'{entry.get("observation") or ""} {entry.get("summary") or ""}'
        camera = ""
        named = str(got.get("camera") or "").strip()
        if named and snapshot is not None:
            camera = resolve_camera(snapshot, named).camera or ""
        said = am.actions_in(text)
        wanted = [a for a in (got.get("actions") or []) if isinstance(a, str) and a in am.ACTIONS]
        if "car_door" in wanted and "car_door" not in said and not re.search(r"דלת|door", text, re.IGNORECASE):
            wanted.remove("car_door")
        place = str(got.get("place") or "")
        place = place if place in am.PLACES else next(iter(am.places_in(text)), "")
        place_words = str(got.get("place_words") or "")[:60]
        cause = " ".join(str(got.get("cause") or "").split())[:120] or " ".join(text.split())[:120]
        for said_place in {p for p in (am.PLACES.get(place, {}).get("he"), str(got.get("place_words") or "").strip()) if p}:
            if cause.endswith(" " + said_place):         # "...את הלדים במדרגות" + "במדרגות של הכניסה": once
                cause = cause[: -len(said_place) - 1].rstrip()
        cause_en = " ".join(str(got.get("cause_en") or "").split())[:160]
        by = str((ctx.speaker or {}).get("name") or "owner")
        fact_id = str(got.get("fact") or "")
        fact = activities.get(fact_id) if fact_id else None
        if fact is None or not fact.live(now):
            fact = next(iter(recent_facts(activities, now)), None) if kind == "correct" else None
        if fact is not None:
            return _follow_up(ctx, activities, fact, camera, entry, event_text, said, wanted, place, place_words,
                              snapshot, now, kind)
        # A new fact: the actions of the alert he answered and the ones he names.
        camera = camera or str(entry.get("camera") or "")
        if not camera:
            return None
        actions = list(dict.fromkeys(am.actions_in(event_text) + said + wanted))
        if "car_door" in actions and "car_door" not in said and "car_door" not in wanted:
            actions.remove("car_door")
        if not actions:
            return None
        win = window(events_book, camera, text, now)
        ask = win is None
        if ask:
            end_of_day = dt.datetime.fromtimestamp(now).replace(hour=23, minute=59).timestamp()
            win = {"until": end_of_day, "daily_from": "", "daily_to": "", "who": "", "known_id": ""}
        who_mark = str(got.get("who_mark") or "")
        if who_mark and who_mark != win.get("known_id"):
            mark = next((k for k in marks if str(k.get("id")) == who_mark), None)
            if mark is not None:
                win["who"], win["known_id"] = str(mark.get("text") or ""), who_mark
        fact = activities.add([camera], actions, cause, win["until"], now, cause_en=cause_en, place=place,
                              place_words=place_words, owner_words=" ".join(text.split())[:300], by=by,
                              daily_from=win["daily_from"], daily_to=win["daily_to"], who=win["who"],
                              known_id=win["known_id"], alert_id=str(entry.get("ref") or ""))
        _learn(ctx, fact, entry, now)
        line = t("act_new", lang, actions=am.actions_text(fact.actions, lang), where=where_text(fact, snapshot, lang),
                 cause=fact.cause)
        out = {"text": line + " " + (t("act_ask_until", lang) if ask else t("act_normal_until", lang,
                                                                              when=when_text(fact, lang))),
               "rows": (((t("btn_cancel_activity", lang), f"kn:x:{fact.id}"),),),
               "note": f"activity fact {fact.id} saved: {'/'.join(fact.actions)} at {camera} "
                       f"{fact.place} = {fact.cause}", "fact": fact.id}
        if ask:
            out["pending"] = {"question": t("act_ask_until", lang), "choices": ["16:00", "17:00", "18:00"],
                              "ts": now, "kind": "activity_until", "id": fact.id}
        return out
    except Exception as exc:  # noqa: BLE001 - the usual path answers it
        log.warning("Activity explanation failed: %s", exc)
        return None


def _follow_up(ctx: Any, activities: Any, fact: am.ActivityFact, camera: str, entry: Dict[str, Any],
               event_text: str, said: List[str], wanted: List[str], place: str, place_words: str, snapshot: Any,
               now: float, kind: str = "explain") -> Dict[str, Any]:
    """The message repeats, adds to or corrects *fact*: more actions (named by the owner), the camera he names
    (instead), the place he names. The reply is about what changed, or that it is already so."""
    lang = ctx.lang
    changes: Dict[str, Any] = {}
    new_actions = [a for a in dict.fromkeys(said + [w for w in wanted if w in said or (
        str(entry.get("camera") or "") in fact.cameras and w in am.actions_in(event_text))]) if a not in fact.actions
                   and a != "car_door"]
    if new_actions:
        changes["actions"] = fact.actions + new_actions
    if camera and camera not in fact.cameras:
        changes["cameras"] = [camera]               # "לא בפרגולה, בכניסה הראשית": instead, not as well
    if place and place != fact.place:
        changes["place"], changes["place_words"] = place, place_words
    if not changes and kind == "correct":
        # "לא בפרגולה, אני מתכוון בכניסה הראשית" when that is what is saved: say so, in other words than before.
        return {"text": t("act_right", lang, where=where_text(fact, snapshot, lang)), "rows": (),
                "note": f"activity fact {fact.id} confirmed", "fact": fact.id}
    if not changes:
        line = t("act_same", lang, actions=am.actions_text(fact.actions, lang), where=where_text(fact, snapshot, lang),
                 cause=fact.cause)
        return {"text": line, "rows": (), "note": f"activity fact {fact.id} unchanged", "fact": fact.id}
    fact = activities.update(fact.id, now, **changes) or fact
    _learn(ctx, fact, entry, now)
    if set(changes) == {"actions"}:
        line = t("act_more", lang, actions=am.actions_text(new_actions, lang), where=where_text(fact, snapshot, lang),
                 cause=fact.cause)
    else:
        line = t("act_fixed", lang, actions=am.actions_text(fact.actions, lang), where=where_text(fact, snapshot, lang),
                 cause=fact.cause)
    return {"text": line, "rows": (((t("btn_cancel_activity", lang), f"kn:x:{fact.id}"),),),
            "note": f"activity fact {fact.id} updated: {changes}", "fact": fact.id}


def _learn(ctx: Any, fact: am.ActivityFact, entry: Optional[Dict[str, Any]], now: float) -> None:
    """The explanation also learns for the long term: its precedent, in shadow (case_chat). Never raises."""
    try:
        from . import case_chat  # noqa: PLC0415

        case_chat.after_fact(getattr(ctx, "services", None), fact, entry or None, str(getattr(ctx, "chat_id", "")),
                             getattr(ctx, "speaker", None), now)
    except Exception as exc:  # noqa: BLE001 - the week-long memory is saved either way
        log.warning("Precedent not saved: %s", exc)


def answer_until(activities: Any, pending: Dict[str, Any], text: str, choice: Optional[Tuple[str, int]], now: float,
                 lang: str) -> Optional[str]:
    """The answer to "עד מתי?" (a tapped hour or typed words). None when it is not one."""
    fact = activities.get(str(pending.get("id") or "")) if activities is not None else None
    if fact is None:
        return None
    words = text
    if choice is not None:
        choices = pending.get("choices") or []
        if 0 <= choice[1] < len(choices):
            words = f"עד {choices[choice[1]]}"
    until = km.until_from_words(words, now)
    if until is None or until <= now:
        return None
    activities.update(fact.id, now, until=until)
    return t("act_until_saved", lang, end=dt.datetime.fromtimestamp(until).strftime("%H:%M"))
