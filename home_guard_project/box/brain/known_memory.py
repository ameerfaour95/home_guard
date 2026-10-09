# home_guard_project/box/brain/known_memory.py
"""What the owner told the box about who is around, read before every answer (2026-10-09).

The owner's morning chat: the workers' clip explanation was saved as a tag only; the mark got 23:59 that nobody said;
the 18:00 correction added a second mark instead of replacing the first; the workers walked from the pergola to the
main entrance, which alerted, and the assistant then OFFERED to mark the pergola - already marked - and answered the
owner's "don't you read the history?" with "אני מבין את התסכול שלך." This module holds the code-side answers:

- ``lasting_fact`` / ``work_group`` / ``who_label``: does an explanation state a lasting fact about who is around
  (workers, the gardener, ours), and who, in a short label ("העובדים");
- ``until_from_words`` / ``scope_from_words``: the end time and the scope (the whole house or one camera) the owner
  actually said - never invented;
- ``same_people``: is a new mark a correction of a live one (same people);
- ``live_lines`` / ``today_lines`` / ``gap_line``: the context lines the model reads every turn - the live marks,
  what the owner said today, and the camera of this event that no mark covers.

Never raises from the context helpers: a failure is logged and the line is left out.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

log = logging.getLogger("box.brain.known_memory")

_HE_PREFIX = r"(?<![א-ת])[ושהבלמכ]{0,2}"
_HE_END = r"(?![א-ת])"

# People who work at the house and move around it (owner, 2026-10-09: "עובדים אצלי" usually means the whole house).
_GROUPS = (
    ("העובדים", "the workers", r"עובד|עובדים|פועל|פועלים|בנאי|בנאים|צוות"),
    ("הגנן", "the gardener", r"גנן|גננים"),
    ("המנקה", "the cleaner", r"מנקה|מנקים|מנקות|עוזרת"),
    ("הטכנאי", "the technician", r"טכנאי|טכנאים|שרברב|חשמלאי|אינסטלטור"),
    ("הקבלן", "the contractor", r"קבלן|קבלנים"),
)
_GROUPS_EN = (
    ("העובדים", "the workers", r"workers?|builders?|crew|labou?rers?"),
    ("הגנן", "the gardener", r"gardeners?"),
    ("המנקה", "the cleaner", r"cleaners?|housekeeper"),
    ("הטכנאי", "the technician", r"technicians?|plumber|electrician"),
    ("הקבלן", "the contractor", r"contractors?"),
)
_GROUP_RES = [(he, en, re.compile(_HE_PREFIX + "(?:" + words + ")" + _HE_END)) for he, en, words in _GROUPS] + \
             [(he, en, re.compile(r"\b(?:" + words + r")\b", re.IGNORECASE)) for he, en, words in _GROUPS_EN]
# Ours / family / neighbours: a lasting fact about who is around, but not a group that moves around the house.
_OURS = re.compile(_HE_PREFIX + r"(?:שלנו|שלי|אצלי|אצלנו|השכן|שכן|שכנה|השכנים|שכנים|המשפחה|הילדים|אשתי|בעלי|"
                   r"אמא שלי|אבא שלי|סבא|סבתא)" + _HE_END + r"|\b(?:our|ours|my|neighbou?rs?|family|kids)\b",
                   re.IGNORECASE)
# A one-off visitor: a tag, not something to remember ("זה הדוור").
_ONE_OFF = re.compile(_HE_PREFIX + r"(?:דוור|הדוור|שליח|שליחה|השליח|משלוח|שליחויות|דואר|חבילה|מוביל)" + _HE_END +
                      r"|\b(?:postman|mailman|mail|courier|delivery|parcel|package)\b", re.IGNORECASE)
_QUESTION = re.compile(r"\?|(?<![א-ת])ו?(?:האם|מי|מה|למה|איך|מתי)(?![א-ת])|\b(?:who|what|why|is it|are they)\b",
                       re.IGNORECASE)


def _group(text: str) -> Optional[tuple]:
    for he, en, pattern in _GROUP_RES:
        if pattern.search(str(text or "")):
            return he, en
    return None


def work_group(text: str) -> bool:
    """The text names people who work at the house (workers, the gardener, the cleaner): they move around it."""
    return _group(text) is not None


def lasting_fact(text: str) -> bool:
    """A clip explanation that states a lasting fact about who is around ("זה תקין עובדים על הפרגולה בשעות אלה",
    "הגנן שלנו", "אדם עם כובע עובד ליד הכניסה"), not a one-off ("זה הדוור") and not a question."""
    text = str(text or "")
    if not text.strip() or _QUESTION.search(text) or _ONE_OFF.search(text):
        return False
    return work_group(text) or bool(_OURS.search(text))


def who_label(text: str, lang: str = "he") -> str:
    """Who the people are, short: "העובדים" / "the workers"; else the owner's words, clipped."""
    found = _group(text)
    if found:
        return found[0] if str(lang).startswith("he") else found[1]
    words = " ".join(str(text or "").split())
    return words if len(words) <= 60 else words[:59].rstrip() + "…"


_STOP = frozenset({"זה", "זאת", "הם", "הן", "של", "על", "עם", "את", "יש", "גם", "כאן", "פה", "רק", "עד", "אצלי",
                   "אצלנו", "שלי", "שלנו", "האנשים", "אנשים", "אדם", "ליד", "תקין", "בסדר", "the", "are", "is", "at",
                   "my", "our", "people", "here", "they", "these", "those", "fine", "ok"})


def _content_words(text: str) -> set:
    out = set()
    for word in re.findall(r"[\w']+", str(text or "").lower()):
        word = re.sub(r"^[ושהבלמכ]{1,2}(?=[א-ת]{3,})", "", word)
        if len(word) >= 3 and word not in _STOP:
            out.add(word)
    return out


def same_people(a: str, b: str) -> bool:
    """Two marks' words are about the same people: the same group ("העובדים על הפרגולה" / "עובדים אצלי"), or they
    share a content word ("עמיר")."""
    ga, gb = _group(a), _group(b)
    if ga and gb:
        return ga == gb
    return bool(_content_words(a) & _content_words(b))


# -- the end time, only from the owner's words ------------------------------------------------------------------
_HE_HOURS = {"אחת": 1, "שתיים": 2, "שתים": 2, "שלוש": 3, "ארבע": 4, "חמש": 5, "שש": 6, "שבע": 7, "שמונה": 8,
             "תשע": 9, "עשר": 10, "אחת עשרה": 11, "אחת-עשרה": 11, "שתים עשרה": 12, "שתים-עשרה": 12}
_CLOCK = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)")
_UNTIL_HOUR = re.compile(r"(?:(?<![א-ת])עד|\buntil|\btill|\bto)\s+(?:ה?שעה\s+|about\s+|around\s+|בערך\s+)?"
                         r"([01]?\d|2[0-3])(?![\d:.])\s*(pm|am|בערב|אחה\"צ|אחר הצהריים|בלילה|בבוקר)?",
                         re.IGNORECASE)
_UNTIL_HE_WORD = re.compile(r"(?<![א-ת])עד\s+(?:ה?שעה\s+)?(" + "|".join(sorted(_HE_HOURS, key=len, reverse=True))
                            + r")(?![א-ת])\s*(בערב|אחה\"צ|אחר הצהריים|בלילה|בבוקר)?")
_HOURS_FOR = re.compile(r"(?<![א-ת])ל(?:-)?\s*(\d{1,2})\s+שעות(?![א-ת])|\bfor\s+(\d{1,2})\s+hours?\b", re.IGNORECASE)
_ONE_HOUR = re.compile(r"(?<![א-ת])(?:לשעה|לשעה אחת|רק עכשיו|רק כרגע|לעכשיו)(?![א-ת])|\b(?:for an hour|for one hour|"
                       r"only now|just now|for now)\b", re.IGNORECASE)
_TWO_HOURS = re.compile(r"(?<![א-ת])לשעתיים(?![א-ת])|\bfor two hours\b", re.IGNORECASE)
_WEEK = re.compile(r"(?<![א-ת])(?:כל השבוע|לשבוע|השבוע)(?![א-ת])|\b(?:all week|this week|for a week)\b",
                   re.IGNORECASE)
_TOMORROW = re.compile(r"(?<![א-ת])עד מחר(?![א-ת])|\buntil tomorrow\b", re.IGNORECASE)


def _at(now: float, hour: int, minute: int = 0) -> float:
    day = dt.datetime.fromtimestamp(now)
    moment = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if moment.timestamp() <= now:
        moment += dt.timedelta(days=1)
    return moment.timestamp()


def _hour_today(now: float, hour: int, part: str) -> float:
    """"עד 6" in the morning is 18:00 when 06:00 is over and 18:00 is not; "בערב" / "pm" always add 12."""
    part = str(part or "").lower()
    if hour < 12 and part in ("pm", "בערב", 'אחה"צ', "אחר הצהריים", "בלילה"):
        hour += 12
    elif hour <= 12 and not part:
        today = dt.datetime.fromtimestamp(now)
        if today.hour >= hour and hour + 12 < 24 and today.hour < hour + 12:
            hour += 12
    return _at(now, hour % 24)


def until_from_words(text: str, now: float) -> Optional[float]:
    """The end time the owner's words give: "עד 18:00", "עד 18", "עד שש", "לשעתיים", "for 3 hours", "רק עכשיו" (an
    hour), "כל השבוע", "עד מחר". None when they give none - "today" alone is not a time (owner, 2026-10-09: "מי אמר
    עד 23:59?"): the box asks."""
    text = str(text or "")
    m = _CLOCK.search(text)
    if m:
        return _at(now, int(m.group(1)), int(m.group(2)))
    m = _UNTIL_HOUR.search(text)
    if m:
        return _hour_today(now, int(m.group(1)), m.group(2) or "")
    m = _UNTIL_HE_WORD.search(text)
    if m:
        return _hour_today(now, _HE_HOURS[m.group(1)], m.group(2) or "")
    m = _HOURS_FOR.search(text)
    if m:
        hours = int(m.group(1) or m.group(2))
        if 0 < hours <= 72:
            return now + hours * 3600.0
    if _TWO_HOURS.search(text):
        return now + 7200.0
    if _ONE_HOUR.search(text):
        return now + 3600.0
    if _WEEK.search(text):
        return now + 7 * 86400.0
    if _TOMORROW.search(text):
        day = dt.datetime.fromtimestamp(now) + dt.timedelta(days=1)
        return day.replace(hour=23, minute=59, second=0, microsecond=0).timestamp()
    return None


# -- the scope: the whole house or one camera ---------------------------------------------------------------------
_HOUSE = re.compile(r"(?<![א-ת])[בל]?(?:כל הבית|כל המצלמות|כל מקום|כל החצר|כל השטח)(?![א-ת])|"
                    r"(?<![א-ת])(?:בכל|לכל)\s+(?:ה)?(?:בית|מצלמות|מקום|חצר)(?![א-ת])|"
                    r"\b(?:everywhere|all (?:the )?cameras|(?:the )?whole house|house-wide|all over)\b", re.IGNORECASE)
_ONLY = re.compile(r"(?<![א-ת])רק\s+[בל]|\bonly\s+(?:at|on|in)\b", re.IGNORECASE)
_NO = re.compile(r"^\s*(?:לא|לא צריך|אל תזכור|אל תסמן|no|nope|don'?t)\s*[.!]?\s*$", re.IGNORECASE)
HOUSE = "house"


def scope_from_words(text: str, snapshot: Any = None) -> Optional[str]:
    """``"house"`` when the owner said the whole house / all the cameras; a camera when they named exactly one
    ("רק בפרגולה"); None otherwise."""
    text = str(text or "")
    if _HOUSE.search(text):
        return HOUSE
    if snapshot is not None:
        try:
            from .registry import mentioned_cameras  # noqa: PLC0415

            named = list(dict.fromkeys(camera for _, camera in mentioned_cameras(snapshot, text)))
            if len(named) == 1:
                return named[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("Scope camera not read: %s", exc)
    return None


def declined(text: str) -> bool:
    """A plain no to "remember this too?"."""
    return bool(_NO.match(str(text or "")))


# -- the context lines ------------------------------------------------------------------------------------------
def _hhmm(ts: Any) -> str:
    try:
        return dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def where_text(snapshot: Any, camera: str, lang: str) -> str:
    """The family's name for a mark's camera; "all the cameras" for a house-wide mark."""
    from .i18n import t  # noqa: PLC0415
    from .registry import display  # noqa: PLC0415

    return t("known_all_cameras", lang) if not camera else display(snapshot, camera, lang)


def live_marks(book: Any, now: float) -> List[Dict[str, Any]]:
    if book is None:
        return []
    try:
        return [k for k in book.list_known(now) if isinstance(k, dict)]
    except Exception as exc:  # noqa: BLE001
        log.warning("Live marks not read: %s", exc)
        return []


def live_lines(book: Any, snapshot: Any, lang: str, now: float) -> List[str]:
    """[LIVE MARKS]: what the box keeps right now from the owner's words, with the cameras' names and end times."""
    if book is None:
        return []
    marks = live_marks(book, now)
    if not marks:
        return ["[LIVE MARKS] none - the owner marked nobody as known right now"]
    rows = [f'"{k.get("text")}" at {where_text(snapshot, str(k.get("camera") or ""), lang)} until '
            f'{_hhmm(k.get("until"))} (said {_hhmm(k.get("at"))}, id {k.get("id")})' for k in marks]
    return ["[LIVE MARKS] the box silences suspicious alerts about these people (never an escalation): "
            + "; ".join(rows) + ". They are already saved: never offer to mark them again; a correction of one is "
                                "mark_known with the new time / camera / words, and it replaces the old mark."]


def today_lines(state: Any, now: float, limit: int = 8) -> List[str]:
    """[OWNER SAID TODAY]: today's clip explanations (✏️) and what was saved from the owner's words, oldest first."""
    try:
        day = dt.datetime.fromtimestamp(now).date()
        rows = []
        for turn in getattr(state, "turns", []) or []:
            if not isinstance(turn, dict) or turn.get("kind") == "alert":
                continue
            ts = float(turn.get("ts") or 0)
            if dt.datetime.fromtimestamp(ts).date() != day:
                continue
            receipts = [r for r in turn.get("receipts") or [] if isinstance(r, str)
                        and any(w in r for w in ("mark_known", "retag_clip", "record_verdict", "tag"))]
            if turn.get("kind") != "tag" and not receipts:
                continue
            said = " ".join(str(turn.get("text") or "").split())[:160]
            rows.append(f'{_hhmm(ts)} "{said}"' + (f" -> {'; '.join(receipts)}" if receipts else ""))
        if not rows:
            return []
        return ["[OWNER SAID TODAY] " + " | ".join(rows[-limit:])]
    except Exception as exc:  # noqa: BLE001
        log.warning("Today's owner lines not built: %s", exc)
        return []


def covers(mark: Dict[str, Any], camera: str) -> bool:
    return not mark.get("camera") or str(mark.get("camera")) == str(camera)


def gap_line(book: Any, snapshot: Any, camera: str, when: str, lang: str, now: float) -> str:
    """The event being discussed is at *camera*, live marks exist, and none covers it (2026-10-09: the workers were
    marked at the pergola, the alert came from the main entrance). "" otherwise."""
    if not camera:
        return ""
    marks = live_marks(book, now)
    if not marks or any(covers(k, camera) for k in marks):
        return ""
    try:
        from .registry import display  # noqa: PLC0415

        here = display(snapshot, camera, lang)
        there = ", ".join(f'"{k.get("text")}" at {where_text(snapshot, str(k.get("camera") or ""), lang)} until '
                          f'{_hhmm(k.get("until"))}' for k in marks)
    except Exception as exc:  # noqa: BLE001
        log.warning("Gap line not built: %s", exc)
        return ""
    return (f"[NOT COVERED] the event being discussed ({when}) is at {here}; the live marks cover only: {there}. "
            f"If the owner says these are the same people, this is the gap: name it in one line and call "
            f"mark_known for them with camera \"all\" (the whole house) or {here}, the same until - it replaces the "
            f"old mark. Never offer what is already marked.")


def receipt_note(receipt: Any, snapshot: Any = None) -> str:
    """A receipt as the history keeps it: its summary plus, for a mark or a tag, what it said (so a later turn
    reads "mark_known done (העובדים at all the cameras until 18:00)", not only an id)."""
    text = receipt.summary()
    d = receipt.detail if isinstance(getattr(receipt, "detail", None), dict) else {}
    try:
        if receipt.tool == "mark_known" and d.get("until_ts"):
            where = where_text(snapshot, str(d.get("camera") or ""), "en")
            extra = f'"{d.get("who")}" at {where} until {_hhmm(d.get("until_ts"))}'
            if d.get("replaced"):
                extra += "; replaced " + ", ".join(
                    f'{where_text(snapshot, str(r.get("camera") or ""), "en")} until {_hhmm(r.get("until"))}'
                    for r in d["replaced"] if isinstance(r, dict))
            if d.get("already"):
                extra += "; it was already saved"
            return f"{text} ({extra})"
        if receipt.tool == "retag_clip" and d.get("tag"):
            return f'{text} (tag: "{d.get("tag")}")'
    except Exception as exc:  # noqa: BLE001
        log.warning("Receipt note not built: %s", exc)
    return text


def pick(marks: Iterable[Dict[str, Any]], who: str, camera: str) -> List[Dict[str, Any]]:
    """The live marks a new mark of *who* at *camera* ("" = the whole house) corrects: the same people, where the
    scopes overlap."""
    return [k for k in marks if same_people(who, str(k.get("text") or ""))
            and (not camera or not k.get("camera") or str(k.get("camera")) == camera)]




# -- a work crew's working days (owner, 2026-10-09 10:15: "it can assume that they also might work for a week") ------
CREW_DAYS = 7
_ONE_DAY = re.compile(_HE_PREFIX + r"(?:רק היום|היום בלבד|רק לעכשיו|רק עכשיו|רק כרגע|לשעה|לשעתיים)" + _HE_END +
                      r"|\b(?:only today|just today|today only|only now|for now|for an hour|for (?:\d+|two) hours)\b",
                      re.IGNORECASE)


def one_day(text: str) -> bool:
    """The owner limited it to today / now ("רק היום", "לשעתיים"): not a weekly window."""
    return bool(_ONE_DAY.search(str(text or "")))


def first_seen_today(book: Any, camera: str, now: float, fallback: Optional[float] = None) -> str:
    """When the people were first seen today at *camera* (the box's events), rounded down to the hour and never
    before 06:00: the start of a crew's daily window ("HH:MM")."""
    day = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
    first = [float(fallback)] if fallback else []
    try:
        if book is not None and camera:
            first += [float(r.get("opened") or 0) for r in book.recent(day.timestamp(), camera) if r.get("opened")]
    except Exception as exc:  # noqa: BLE001
        log.warning("First sighting not read: %s", exc)
    first = [x for x in first if x >= day.timestamp()] or [now]
    hour = max(6, dt.datetime.fromtimestamp(min(first)).hour)
    return f"{hour:02d}:00"


def crew_window(book: Any, camera: str, until: float, now: float,
                fallback: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """A crew's mark as a daily window: from their first sighting today to the hour they end, every day for
    CREW_DAYS days. None when the end is not today or comes before the start."""
    end = dt.datetime.fromtimestamp(until)
    if end.date() != dt.datetime.fromtimestamp(now).date():
        return None
    start = first_seen_today(book, camera, now, fallback)
    stop = end.strftime("%H:%M")
    if start >= stop:
        return None
    last = (end + dt.timedelta(days=CREW_DAYS - 1)).timestamp()
    return {"daily_from": start, "daily_to": stop, "until": last}


def likely_hours(now: float) -> List[str]:
    """Button answers for "until what time?": the end of a working day, or the next hours late in the day."""
    hour = dt.datetime.fromtimestamp(now).hour
    hours = [16, 17, 18] if hour < 15 else [h for h in range(hour + 1, 24)][:3]
    return [f"{h:02d}:00" for h in hours] or ["23:00"]


def until_said_today(state: Any, who: str, now: float) -> Optional[float]:
    """An end time the owner already gave today for the same people (never ask what today's chat says)."""
    try:
        day = dt.datetime.fromtimestamp(now).date()
        for turn in reversed(getattr(state, "turns", []) or []):
            if not isinstance(turn, dict) or turn.get("kind") == "alert":
                continue
            ts = float(turn.get("ts") or 0)
            if dt.datetime.fromtimestamp(ts).date() != day:
                break
            text = str(turn.get("text") or "")
            if same_people(who, text) or work_group(text) and work_group(who):
                until = until_from_words(text, ts)
                if until is not None and until > now:
                    return until
    except Exception as exc:  # noqa: BLE001
        log.warning("Today's chat not read for an end time: %s", exc)
    return None


_HE_DAYS = {"א": 6, "ב": 0, "ג": 1, "ד": 2, "ה": 3, "ו": 4, "שבת": 5, "ראשון": 6, "שני": 0, "שלישי": 1, "רביעי": 2,
            "חמישי": 3, "שישי": 4}
_DATE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?!\d)")
_HE_WEEKDAY = re.compile(r"(?<![א-ת])(?:עד\s+)?יום\s+(ראשון|שני|שלישי|רביעי|חמישי|שישי|שבת|[אבגדהו])['׳]?(?![א-ת])")


def last_day_from_words(text: str, now: float) -> Optional[dt.date]:
    """The last day the owner gives ("עד יום ה׳", "עד 20.10", "מחר", "שבוע", "רק היום"); None when none."""
    text = str(text or "")
    today = dt.datetime.fromtimestamp(now).date()
    m = _DATE.search(text)
    if m:
        year = int(m.group(3)) if m.group(3) else today.year
        year = year + 2000 if year < 100 else year
        try:
            day = dt.date(year, int(m.group(2)), int(m.group(1)))
            return day if day >= today else None
        except ValueError:
            return None
    m = _HE_WEEKDAY.search(text)
    if m:
        ahead = (_HE_DAYS[m.group(1)] - today.weekday()) % 7
        return today + dt.timedelta(days=ahead)
    if one_day(text) or re.search(r"(?<![א-ת])היום(?![א-ת])|\btoday\b", text, re.IGNORECASE):
        return today
    if re.search(r"(?<![א-ת])מחר(?![א-ת])|\btomorrow\b", text, re.IGNORECASE):
        return today + dt.timedelta(days=1)
    if _WEEK.search(text):
        return today + dt.timedelta(days=CREW_DAYS - 1)
    return None


def mark_when(mark: Dict[str, Any], now: float, lang: str) -> str:
    """"כל יום עד 18:00, עד יום ה׳ 15.10" / "היום עד 18:00" / "עד יום ה׳ 15.10 18:00": a mark's time as the owner
    reads it."""
    from .i18n import t  # noqa: PLC0415

    until = float(mark.get("until") or now)
    if mark.get("daily_from") and mark.get("daily_to"):
        return t("known_when_daily", lang, start=str(mark["daily_from"]), end=str(mark["daily_to"]),
                 day=day_text(until, lang))
    end = dt.datetime.fromtimestamp(until)
    if end.date() == dt.datetime.fromtimestamp(now).date():
        return t("known_when_today", lang, end=end.strftime("%H:%M"))
    return t("known_when_until", lang, end=f"{day_text(until, lang)} {end.strftime('%H:%M')}")


def day_text(ts: float, lang: str) -> str:
    """"יום ה׳ 15.10" / "Thu 15.10"."""
    day = dt.datetime.fromtimestamp(ts)
    if str(lang).startswith("he"):
        names = ("ב׳", "ג׳", "ד׳", "ה׳", "ו׳", "שבת", "א׳")
        head = names[day.weekday()]
        return f"{head if head == 'שבת' else 'יום ' + head} {day.strftime('%d.%m')}"
    return day.strftime("%a %d.%m")


# -- TAG vs MEMORY (owner, 2026-10-09 10:05: "a clear distinction between tagging for Qwen and the knowledge for the
# memory"). A tag is about ONE clip (feedback/, training data, never changes behaviour); a memory is what the box knows
# and acts on (known marks, house facts, camera profiles; never feedback/). ----------------------------------------
_NORMAL = re.compile(r"(?<![א-ת])(?:תקין|תקינה|זה בסדר|הכל בסדר|הכול בסדר|בסדר גמור|לא חשוד|לא חשודה|רגיל|"
                     r"זה רגיל|הכל טוב|אין בעיה|לא אירוע|סתם)(?![א-ת])|"
                     r"\b(?:normal|fine|false alarm|not suspicious|all good|nothing wrong|no problem)\b", re.IGNORECASE)


def tag_label(text: str) -> str:
    """The owner's label in a clip explanation: "normal" when it says the scene is fine ("זה תקין", "זה בסדר",
    "לא חשוד"), else "other" (their own description)."""
    return "normal" if _NORMAL.search(str(text or "")) else "other"


def tag_line(time: str, camera_name: str, label: str, words: str, lang: str) -> str:
    """"🏷️ תיוג לסרטון 08:01 (פרגולה): תקין: עובדים ליד הטנדר"."""
    from .i18n import TEMPLATES, t  # noqa: PLC0415

    key = f"label_{label}"
    name = t(key, lang) if key in TEMPLATES else ""
    words = " ".join(str(words or "").split())
    what = f"{name}: {words}" if name and words else (name or words)
    return t("tag_line", lang, time=time, camera=camera_name, what=what)


_ASKS_MEMORY = re.compile(r"(?<![א-ת])מה\s+(?:אתה\s+|את\s+)?(?:זוכר|זוכרת|יודע עלינו|שמור אצלך|בזיכרון)(?![א-ת])|"
                          r"(?<![א-ת])מה\s+יש\s+(?:לך\s+)?בזיכרון(?![א-ת])|"
                          r"\bwhat\s+do\s+you\s+(?:remember|know about us)\b|\bwhat(?:'s| is)\s+in\s+your\s+memory\b",
                          re.IGNORECASE)
_ASKS_TAGS = re.compile(r"(?<![א-ת])(?:מה|אילו|איזה)\s+(?:\S+\s+)?(?:תייגתי|תיוגים|תייגנו)(?![א-ת])|"
                        r"\bwhat\s+did\s+I\s+tag\b|\bmy\s+tags\b", re.IGNORECASE)


def asks_memory(text: str) -> bool:
    """"מה אתה זוכר?" / "what do you remember?"."""
    return bool(_ASKS_MEMORY.search(str(text or "")))


def asks_tags(text: str) -> bool:
    """"מה תייגתי היום?" / "what did I tag today?"."""
    return bool(_ASKS_TAGS.search(str(text or "")))


def memory_text(book: Any, snapshot: Any, lang: str, now: float, facts: Sequence[str] = ()) -> str:
    """The live memories, one per line: "העובדים בכל הבית: כל יום 08:00–18:00, עד יום ה׳ 15.10"."""
    from .i18n import t  # noqa: PLC0415

    rows = []
    for k in live_marks(book, now):
        camera = str(k.get("camera") or "")
        where = t("known_where_house", lang) if not camera else t(
            "known_where_camera", lang, camera=_in_place(where_text(snapshot, camera, lang), lang))
        rows.append(f"🧠 {k.get('text')} {where}: {mark_when(k, now, lang)}")
    rows += [f"🧠 {f}" for f in facts if f]
    if not rows:
        return f"{t('memory_list_none', lang)}\n{t('memory_tags_note', lang)}"
    return f"{t('memory_list', lang, rows=chr(10).join(rows))}\n{t('memory_tags_note', lang)}"


def _in_place(name: str, lang: str) -> str:
    name = str(name or "").strip()
    return name[1:] if str(lang).startswith("he") and len(name) > 2 and name.startswith("ה") else name


def tags_today(roots: Sequence[str], snapshot: Any, lang: str, now: float) -> str:
    """Today's tags of clips (feedback/ records with the owner's label), oldest first."""
    import glob  # noqa: PLC0415
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    from .i18n import t  # noqa: PLC0415
    from .registry import display  # noqa: PLC0415

    day = dt.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    seen, rows = set(), []
    for root in roots:
        for path in glob.glob(os.path.join(glob.escape(root), "feedback", "*", day, "*.feedback.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, ValueError):
                continue
            label = str(rec.get("owner_label") or "")
            alert = rec.get("alert") if isinstance(rec.get("alert"), dict) else {}
            if not label or not alert.get("alert_id"):
                continue
            stamp = os.path.basename(path).rsplit("_", 1)[-1].split(".", 1)[0]
            try:
                when = dt.datetime.fromtimestamp(float(alert.get("ts") or 0)).strftime("%H:%M")
            except (TypeError, ValueError, OverflowError, OSError):
                when = "?"
            line = tag_line(when, display(snapshot, str(alert.get("camera") or ""), lang), label,
                            str(rec.get("owner_text") or ""), lang)
            key = (alert.get("alert_id"), stamp)
            if key not in seen:
                seen.add(key)
                rows.append((stamp, line))
    if not rows:
        return t("tags_list_none", lang)
    return t("tags_list", lang, rows="\n".join(line for _, line in sorted(rows)))
