# home_guard_project/box/brain/house.py
"""The house state from Telegram: "going to sleep", "we left", "on vacation until 20.10", "a plumber at 10".

The owner's short commands are read here in code, before any model, so they work the same in Guard and Assistant
mode and whichever model would have answered. Each one goes through ``house_state``'s writer (never the file),
is read back, and gets a receipt in the owner's language that names the state and when it ends, with Undo. The
status line ("מה המצב?") is code-written too: the state with its source and end, what the family expects, today's
pauses and the cameras that watch, and the changes that wait for the owner's yes or no.

Anything the parser does not take (a vacation without a date, a free-form "we're off to Eilat till the weekend")
goes to the big model, which has the same actions as tools (``house_state``, ``house_expect``, ``house_cancel``,
``house_status`` in tools.py).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .profiles import hebrew_words

log = logging.getLogger("box.brain.house")

DEFAULT_SCHEDULE = ("00:00", "06:00")
EXPECT_HOURS = 3.0               # "the plumber at 10" holds until 13:00
LAST_COMMAND_MINUTES = 30        # a bare "cancel" undoes a house command this recent

# -- what the owner writes -----------------------------------------------------------------------------------------
_EN_APOS = "['’]?"


def _en(pattern: str) -> "re.Pattern[str]":
    return re.compile(r"\b(?:" + pattern.replace("'", _EN_APOS) + r")\b", re.IGNORECASE)


_SLEEP = (hebrew_words(("הולכים לישון", "הולך לישון", "הולכת לישון", "הלכנו לישון", "נכנסים לישון", "אנחנו ישנים",
                        "הולכים למיטה", "לילה טוב", "מצב לילה")),
          _en(r"going to (?:sleep|bed)|off to bed|heading to bed|good ?night|we're (?:going to )?sleep(?:ing)?|"
              r"bed ?time|night mode"))
_UP = (hebrew_words(("קמנו", "קמתי", "התעוררנו", "בוקר טוב", "אנחנו ערים")),
       _en(r"we're up|we are up|we're awake|we are awake|good morning|woke up|we got up"))
_LEFT = (hebrew_words(("יצאנו", "יוצאים", "אנחנו יוצאים", "עזבנו", "נסענו", "אנחנו לא בבית")),
         _en(r"we left|we're leaving|we are leaving|leaving (?:now|the house|home)|we're (?:out|away)|"
             r"we are (?:out|away)|nobody's home|nobody is home|away mode"))
_BACK = (hebrew_words(("חזרנו", "חזרתי", "הגענו הביתה", "הגעתי הביתה", "אנחנו בבית")),
         _en(r"we're back|we are back|we're home|we are home|back home|i'm (?:home|back)|i am (?:home|back)|"
             r"we got (?:back|home)"))
_VACATION = (hebrew_words(("חופשה", 'חו"ל', "חול")), _en(r"vacation|holiday|holidays"))
_EXPECT = (re.compile(r"(?<!\w)[ו]?(?:מצפים|מחכים|מצפה|מחכה)\s+ל(?P<what>\S+(?:\s+\S+){0,3})"),
           re.compile(r"\b(?:we're |we are |i'm |i am )?(?:expecting|waiting for)\s+(?P<what>.+)", re.IGNORECASE))
_ARRIVES = (re.compile(r"(?P<what>(?:\S+\s+){0,2}\S+)\s+(?:מגיע|מגיעה|מגיעים|יגיע|תגיע|יגיעו|יבוא|תבוא|יבואו|"
                       r"אמור להגיע|אמורה להגיע|אמורים להגיע)(?!\w)"),
            re.compile(r"(?P<what>(?:\S+\s+){0,2}\S+)\s+(?:is coming|will come|comes|arrives|is arriving|"
                       r"will arrive|is due|should come|is supposed to come)\b", re.IGNORECASE))
_CANCEL = (re.compile(r"(?<!\w)[ולשהתמכ]{0,2}בטל[ויה]?(?!\w)"),
           re.compile(r"\b(?:cancel|never ?mind|scratch that)\b", re.IGNORECASE))
_STATUS = re.compile(
    r"^(?:מה\s+המצב(?:\s+בבית)?|מה\s+מצב\s+הבית|מצב\s+הבית|סטטוס|מה\s+הסטטוס|status|house\s+status|"
    r"what" + _EN_APOS + r"s\s+the\s+status|what\s+is\s+the\s+status|what" + _EN_APOS +
    r"s\s+the\s+house\s+state)$", re.IGNORECASE)

_STATE_OBJECTS = (hebrew_words(("חופשה", "נסיעה", "יציאה", "מצב", 'חו"ל')),
                  _en(r"vacation|holiday|trip|away|the state|house state|sleep|night mode"))
_OTHER_OBJECTS = (hebrew_words(("השתקה", "ההשתקה", "התראות", "מצלמה", "השהיה", "הגדרה")),
                  _en(r"pause|mute|alerts?|camera|setting|sensitivity|recording|video"))
_FILLER = {
    "אנחנו", "אני", "עכשיו", "כבר", "תודה", "בבקשה", "אוקיי", "אוקי", "סבבה", "טוב", "יאללה", "את", "זה", "כן",
    "הביתה", "מהבית", "הבית", "בבית", "ok", "okay", "now", "we", "i", "just", "thanks", "thank", "you", "please",
    "the", "house", "home", "all", "so", "yes", "bye", "everyone", "guys", "חבר'ה", "כולם", "לכולם", "ביי",
}
_DAY = (re.compile(r"(?<!\w)(?:היום|הערב)(?!\w)|\b(?:today|tonight|this evening)\b", re.IGNORECASE),
        re.compile(r"(?<!\w)מחר(?!\w)|\btomorrow\b", re.IGNORECASE))
_TIME_HE = re.compile(r"(?<!\w)(?:ב-?|בשעה\s+|ב\s+)(?P<h>[01]?\d|2[0-3])(?::(?P<m>[0-5]\d))?(?![\d.:/])")
_TIME_EN = re.compile(r"\bat\s+(?P<h>[01]?\d|2[0-3])(?::(?P<m>[0-5]\d))?\s*(?P<ap>am|pm)?\b", re.IGNORECASE)
_UNTIL = re.compile(r"(?<!\w)(?:עד|until|till|til|through|thru)\s+(?P<rest>.+)$", re.IGNORECASE)
_MONTHS_EN = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTHS_HE = ("ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר",
              "דצמבר")
_WEEKDAYS_EN = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAYS_HE = ("שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון")      # Python order: Monday first


@dataclass(frozen=True)
class HouseCommand:
    kind: str                        # sleep / up / left / back / vacation / expect / cancel / status
    until: Optional[float] = None    # when the state or the note ends (None: away until the owner says)
    note: str = ""                   # expect: the note as the owner will read it ("שרברב ב-10:00")
    what: str = ""                   # cancel: "state", "expect", or "" (a bare cancel: the last house command)
    words: str = ""                  # cancel expect: the words to find the note by
    rest: bool = False               # the message asks for more than this: the model answers the rest


def _any(patterns: Sequence["re.Pattern[str]"], text: str) -> Optional["re.Match[str]"]:
    for p in patterns:
        m = p.search(text)
        if m:
            return m
    return None


def _end_of_day(day: dt.date) -> float:
    return dt.datetime(day.year, day.month, day.day, 23, 59).timestamp()


def next_at(now: float, hhmm: str) -> float:
    """The next *hhmm* (local) after *now*."""
    moment = dt.datetime.fromtimestamp(now)
    hour, minute = map(int, hhmm.split(":"))
    target = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= moment:
        target += dt.timedelta(days=1)
    return target.timestamp()


def _date(text: str, now: float) -> Optional[dt.date]:
    """The first date in *text*: 20.10 / 20/10/2026 / ה-20 / October 20 / 20 באוקטובר / Sunday / יום ראשון /
    tomorrow. A day already past this year means next year."""
    today = dt.datetime.fromtimestamp(now).date()

    def ahead(day: int, month: int, year: Optional[int] = None) -> Optional[dt.date]:
        try:
            found = dt.date(year or today.year, month, day)
        except ValueError:
            return None
        if year is None and found < today:
            found = dt.date(today.year + 1, month, day)
        return found

    m = re.search(r"(?<!\d)(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?(?!\d)", text)
    if m:
        year = int(m.group(3)) if m.group(3) else None
        return ahead(int(m.group(1)), int(m.group(2)), year + 2000 if year and year < 100 else year)
    for i, name in enumerate(_MONTHS_EN):
        m = re.search(rf"\b{name}[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
                      rf"{name}[a-z]*\b", text, re.IGNORECASE)
        if m:
            return ahead(int(m.group(1) or m.group(2)), i + 1)
    for i, name in enumerate(_MONTHS_HE):
        m = re.search(rf"(?<!\d)(\d{{1,2}})\s+(?:ב|ל)?{name}(?!\w)", text)
        if m:
            return ahead(int(m.group(1)), i + 1)
    if re.search(r"(?<!\w)מחרתיים(?!\w)|\bday after tomorrow\b", text, re.IGNORECASE):
        return today + dt.timedelta(days=2)
    if _DAY[1].search(text):
        return today + dt.timedelta(days=1)
    for names in (_WEEKDAYS_EN, _WEEKDAYS_HE):
        for i, name in enumerate(names):
            pattern = rf"\b{name}\b" if names is _WEEKDAYS_EN else rf"(?<!\w)(?:יום\s+)?ה?{name}(?!\w)"
            if re.search(pattern, text, re.IGNORECASE):
                days = (i - today.weekday()) % 7 or 7
                return today + dt.timedelta(days=days)
    m = re.search(r"(?<!\w)ה-?(\d{1,2})(?!\d)|\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", text, re.IGNORECASE)
    if m:
        day = int(m.group(1) or m.group(2))
        found = ahead(day, today.month)
        if found is not None and found < today:
            found = None
        if found is None:
            month = today.month % 12 + 1
            found = ahead(day, month, today.year + (1 if month == 1 else 0))
        return found
    return None


def _time(text: str) -> Optional[Tuple[int, int]]:
    m = _TIME_EN.search(text)
    if m:
        hour, minute = int(m.group("h")), int(m.group("m") or 0)
        ap = (m.group("ap") or "").lower()
        if ap == "pm" and hour < 12:
            hour += 12
        elif ap == "am" and hour == 12:
            hour = 0
        elif not ap and not m.group("m") and 1 <= hour <= 6:
            hour += 12                     # "at 4": the afternoon
        return hour, minute
    m = _TIME_HE.search(text)
    if m:
        hour, minute = int(m.group("h")), int(m.group("m") or 0)
        if not m.group("m") and 1 <= hour <= 6:
            hour += 12
        return hour, minute
    return None


def _strip_spans(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    out, last = [], 0
    for start, end in sorted(spans):
        out.append(text[last:start])
        last = max(last, end)
    out.append(text[last:])
    return " ".join(out)


def _rest(text: str, spans: Sequence[Tuple[int, int]]) -> bool:
    """True when two or more words that are not filler are left once the command is taken out."""
    words = [w.strip(".,!?;:-–—\"'()") for w in _strip_spans(text, spans).split()]
    return len([w for w in words if w and w.casefold() not in _FILLER and re.search(r"\w", w)]) >= 2


def _note_words(what: str) -> str:
    """The thing expected, without the day, the time and filler ("a package", "השרברב")."""
    what = _TIME_EN.sub(" ", _TIME_HE.sub(" ", what))
    for p in _DAY:
        what = p.sub(" ", what)
    words = [w.strip(".,!?;:-–—\"'()") for w in what.split()]
    words = [w for w in words if w and w.casefold() not in ("today", "tomorrow", "now", "please", "בבקשה", "עכשיו")]
    return " ".join(words[:4])


def parse_command(text: Any, now: float, schedule: Optional[Tuple[str, str]] = DEFAULT_SCHEDULE
                  ) -> Optional[HouseCommand]:
    """The house command in an owner's message, or None. Never raises."""
    try:
        return _parse(text, now, schedule)
    except Exception as exc:  # noqa: BLE001 - an odd message goes to the model instead
        log.warning("House command not parsed: %s", exc)
        return None


def _parse(text: Any, now: float, schedule: Optional[Tuple[str, str]]) -> Optional[HouseCommand]:
    if not isinstance(text, str) or not text.strip():
        return None
    clean = " ".join(text.replace("’", "'").replace("״", '"').split())
    bare = re.sub(r"[^\w\s'\"-]", " ", clean).strip()
    if _STATUS.match(" ".join(bare.split())):
        return HouseCommand("status")
    if clean.rstrip().endswith("?"):
        return None                       # a question about the house is not a command
    asleep_from, asleep_until = schedule or DEFAULT_SCHEDULE

    cancel = _any(_CANCEL, clean)
    if cancel:
        tail = clean[cancel.end():]
        if _any(_OTHER_OBJECTS, tail):
            return None
        if _any(_STATE_OBJECTS, tail):
            return HouseCommand("cancel", what="state")
        words = _note_words(re.sub(r"(?<!\w)(?:את|the|my|our)(?!\w)", " ", tail, flags=re.IGNORECASE))
        words = " ".join(w[1:] if len(w) > 2 and w[0] == "ה" else w for w in words.split())
        if words:
            return HouseCommand("cancel", what="expect", words=words)
        return HouseCommand("cancel", what="", rest=_rest(clean, [cancel.span()]))

    until_m = _UNTIL.search(clean)
    vacation = _any(_VACATION, clean)
    left = _any(_LEFT, clean)
    if vacation or (left and until_m):
        day = _date(until_m.group("rest"), now) if until_m else None
        if day is None:
            return None                   # no end date: the model asks for one
        spans = [m.span() for m in (vacation, left, until_m) if m]
        return HouseCommand("vacation", until=_end_of_day(day), rest=_rest(clean, spans))

    for kind, patterns in (("sleep", _SLEEP), ("up", _UP), ("left", _LEFT), ("back", _BACK)):
        m = _any(patterns, clean)
        if m:
            if kind == "sleep":
                until: Optional[float] = next_at(now, asleep_until)
            elif kind == "up":
                until = next_at(now, asleep_from)
            else:
                until = None              # left: away until they are back; back: the executor decides
            return HouseCommand(kind, until=until, rest=_rest(clean, [m.span()]))

    expect = _any(_EXPECT, clean)
    arrives = None if expect else _any(_ARRIVES, clean)
    found = expect or arrives
    if found:
        hhmm = _time(clean)
        day_word = _any(_DAY, clean)
        if arrives and not (hhmm or day_word):
            return None                   # "the car comes" is not a note; "the plumber comes at 10" is
        what = _note_words(found.group("what"))
        if not what:
            return None
        today = dt.datetime.fromtimestamp(now).date()
        tomorrow = bool(_DAY[1].search(clean))
        if hhmm:
            day = today + dt.timedelta(days=1 if tomorrow else 0)
            moment = dt.datetime(day.year, day.month, day.day, *hhmm).timestamp()
            if moment < now - 3600 and not tomorrow:
                moment += 86400           # 10:00 already passed: tomorrow at 10
            until = moment + EXPECT_HOURS * 3600
            hebrew = bool(re.search(r"[א-ת]", clean))
            note = f"{what} {'ב-' if hebrew else 'at '}{hhmm[0]:02d}:{hhmm[1]:02d}"
        else:
            until = _end_of_day(today + dt.timedelta(days=1 if tomorrow else 0))
            note = what
        return HouseCommand("expect", until=until, note=note)
    return None


# -- running a command ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HouseResult:
    handled: bool = True             # False: not a house command after all (the model gets the message)
    text: str = ""                   # code-written answer (the status line); receipts are rendered by the agent
    rows: Tuple[Tuple[Tuple[str, str], ...], ...] = ()     # button rows: ((label, callback), ...)
    undo: str = ""                   # a bare cancel: the turn token of the house command to undo
    rest: bool = False


NOT_OURS = HouseResult(handled=False)
_STATE_OF = {"sleep": "home_asleep", "up": "home_awake", "left": "away", "vacation": "vacation"}


def _ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return dt.datetime.fromisoformat(str(value)).timestamp()


def _schedule(now_state: Any) -> Optional[Tuple[str, str]]:
    sched = getattr(now_state, "schedule", None) or {}
    frm, until = sched.get("asleep_from"), sched.get("asleep_until")
    return (frm, until) if frm and until else None


def schedule_of(store: Any, now: float) -> Optional[Tuple[str, str]]:
    """The night schedule the store holds now (None when the owner turned it off)."""
    return _schedule(store.current(now))


def _issue(ctx: Any, tool: str, status: str, target: str, detail: Dict[str, Any], reason: str = "") -> Any:
    from .tools import _issue as issue  # noqa: PLC0415 - tools imports this module

    return issue(ctx, tool, status, target, detail, reason)


def _by(ctx: Any) -> str:
    return str((ctx.speaker or {}).get("name") or "")


def _before(now_state: Any) -> Dict[str, Any]:
    return {"state": now_state.state, "source": now_state.source, "entry_id": now_state.entry_id,
            "until": _ts(now_state.expires_at)}


def set_state(ctx: Any, kind: str, until: Optional[float] = None) -> Any:
    """sleep / up / left / back / vacation through the writer, read back before the receipt says so."""
    store, now = ctx.services.house, float(ctx.services.now())
    before = store.current(now)
    schedule = _schedule(before)
    if kind == "back":                        # home again: asleep or awake as the night schedule says now
        from ..house_state import scheduled  # noqa: PLC0415

        state = scheduled(now, schedule).state
        if schedule:
            until = next_at(now, schedule[1] if state == "home_asleep" else schedule[0])
    else:
        state = _STATE_OF[kind]
    detail: Dict[str, Any] = {"state": state, "before": _before(before)}
    try:
        entry = store.set_house_state(state, "owner", until=until, by=_by(ctx), now=now)
    except (ValueError, TypeError) as exc:
        return _issue(ctx, "house_state", "failed", state, detail, str(exc))
    after = store.current(now)
    if entry.get("status") != "applied" or after.entry_id != entry.get("id") or after.state != state:
        log.warning("House state %s was not confirmed by the writer: %s", state, entry)
        return _issue(ctx, "house_state", "failed", state, detail, "not_confirmed")
    detail.update(until=_ts(after.expires_at), entry_id=after.entry_id)
    return _issue(ctx, "house_state", "done", state, detail)


def add_expect(ctx: Any, note: str, until: float, camera: Optional[str] = None) -> Any:
    store, now = ctx.services.house, float(ctx.services.now())
    detail: Dict[str, Any] = {"text": note, "camera": camera or ""}
    try:
        entry = store.add_expecting(note, camera=camera, until=until, source="owner", by=_by(ctx), now=now)
    except (ValueError, TypeError) as exc:
        return _issue(ctx, "house_expect", "failed", note, detail, str(exc))
    live = next((e for e in store.current(now).expecting if e["id"] == entry.get("id")), None)
    if entry.get("status") != "applied" or live is None:
        return _issue(ctx, "house_expect", "failed", note, detail, "not_confirmed")
    detail.update(id=live["id"], text=live["text"], until=_ts(live["expires_at"]))
    return _issue(ctx, "house_expect", "done", live["text"], detail)


def _matches(note: str, words: str) -> bool:
    have = {w.strip(".,!?").casefold().lstrip("ה") for w in note.split()}
    want = [w.strip(".,!?").casefold().lstrip("ה") for w in words.split()]
    return bool(want) and all(any(w == h or (len(w) > 2 and (w in h or h in w)) for h in have) for w in want)


def cancel(ctx: Any, what: str, words: str = "") -> List[Any]:
    store, now = ctx.services.house, float(ctx.services.now())
    current = store.current(now)
    if what == "state":
        detail: Dict[str, Any] = {"kind": "state", "old": current.state}
        if current.source == "schedule" or not current.entry_id:
            return [_issue(ctx, "house_cancel", "failed", current.state, detail, "nothing_to_cancel")]
        detail.update(entry_id=current.entry_id, old_until=_ts(current.expires_at), old_source=current.source)
        done = store.cancel_entry(current.entry_id, source="owner", by=_by(ctx), now=now)
        after = store.current(now)
        if not done or after.entry_id == current.entry_id:
            return [_issue(ctx, "house_cancel", "failed", current.state, detail, "not_confirmed")]
        detail.update(state=after.state, until=_ts(after.expires_at))
        return [_issue(ctx, "house_cancel", "done", current.state, detail)]
    notes = [e for e in current.expecting if _matches(e["text"], words)] if words else list(current.expecting)
    if not notes or (not words and len(notes) > 1):
        return [_issue(ctx, "house_cancel", "failed", words, {"kind": "expect", "text": words}, "no_such_note")]
    out = []
    for note in notes:
        detail = {"kind": "expect", "id": note["id"], "text": note["text"], "camera": note.get("camera") or "",
                  "until": _ts(note["expires_at"])}
        ok = store.cancel_expecting(note["id"], by=_by(ctx), now=now)
        gone = ok and all(e["id"] != note["id"] for e in store.current(now).expecting)
        out.append(_issue(ctx, "house_cancel", "done" if gone else "failed", note["text"], detail,
                          "" if gone else "not_confirmed"))
    return out


def describe_proposal(p: Dict[str, Any], lang: str) -> str:
    from .i18n import t  # noqa: PLC0415

    if p.get("kind") == "state":
        return t("proposal_state", lang, state=t(f"state_{p.get('state')}", lang))
    if p.get("kind") == "expect":
        return t("proposal_expect", lang, text=str(p.get("text") or ""))
    return t("proposal_cancel", lang)


def answer_proposal(ctx: Any, proposal_id: str, approve: bool) -> Any:
    """The owner's yes or no to a change that would relax the house and did not come from them."""
    store, now = ctx.services.house, float(ctx.services.now())
    proposal = next((p for p in store.pending_proposals(now) if p.get("id") == proposal_id), None)
    detail: Dict[str, Any] = {"approved": bool(approve), "proposal": proposal_id}
    if proposal is None:
        return _issue(ctx, "house_answer", "failed", proposal_id, detail, "proposal_gone")
    detail["what"] = describe_proposal(proposal, ctx.lang)
    ok = (store.approve if approve else store.reject)(proposal_id, by=_by(ctx), now=now)
    if not ok:
        return _issue(ctx, "house_answer", "failed", proposal_id, detail, "proposal_gone")
    return _issue(ctx, "house_answer", "done", proposal_id, detail)


def run_command(ctx: Any, cmd: Optional[HouseCommand]) -> HouseResult:
    """Carry out one parsed command. Receipts go on *ctx*; the status line comes back as text."""
    if cmd is None or getattr(ctx.services, "house", None) is None:
        return NOT_OURS
    now = float(ctx.services.now())
    if cmd.kind == "status":
        text, rows = status(ctx)
        return HouseResult(text=text, rows=rows)
    if cmd.kind == "cancel" and not cmd.what:
        last = ctx.state.house_last if isinstance(getattr(ctx.state, "house_last", None), dict) else {}
        if last.get("token") and now - float(last.get("ts") or 0) <= LAST_COMMAND_MINUTES * 60:
            return HouseResult(undo=str(last["token"]), rest=cmd.rest)
        return NOT_OURS                   # a bare cancel long after: it may mean a pause, the model decides
    if cmd.kind == "cancel":
        cancel(ctx, cmd.what, cmd.words)
    elif cmd.kind == "expect":
        add_expect(ctx, cmd.note, float(cmd.until or now))
    else:
        set_state(ctx, cmd.kind, cmd.until)
    return HouseResult(rest=cmd.rest)


# -- what the owner reads ------------------------------------------------------------------------------------------
def when(ts: float, ref: float, lang: str) -> str:
    """A moment as the owner reads it: "06:00" today, "tomorrow 06:00", "20.10 23:59"; midnight tonight is
    "00:00"."""
    from .i18n import t  # noqa: PLC0415

    moment, base = dt.datetime.fromtimestamp(ts), dt.datetime.fromtimestamp(ref)
    days = (moment.date() - base.date()).days
    clock = moment.strftime("%H:%M")
    if days == 0 or (days == 1 and clock == "00:00"):
        return clock
    if days == 1:
        return t("house_tomorrow_at", lang, time=clock)
    return moment.strftime("%d.%m %H:%M")


def until_text(ts: Optional[float], ref: float, lang: str) -> str:
    from .i18n import t  # noqa: PLC0415

    return t("house_until_said", lang) if ts is None else t("house_until", lang, when=when(ts, ref, lang))


def receipt_line(receipt: Any, lang: str) -> str:
    """A done house receipt (a failure uses the shared failed line)."""
    from .i18n import t  # noqa: PLC0415

    d, ref = receipt.detail, float(receipt.ts or 0)

    def state(key: str) -> str:
        return t(f"state_{d.get(key)}", lang)

    if receipt.tool == "house_state":
        key = "house_back_to" if d.get("undo_of") else "house_set"
        return t(key, lang, state=state("state"), until=until_text(d.get("until"), ref, lang))
    if receipt.tool == "house_expect":
        return t("house_expecting", lang, text=str(d.get("text") or ""), until=until_text(d.get("until"), ref, lang))
    if receipt.tool == "house_cancel":
        if d.get("kind") == "state":
            return t("house_state_cancelled", lang, old=state("old"), state=state("state"),
                     until=until_text(d.get("until"), ref, lang))
        return t("house_expect_cancelled", lang, text=str(d.get("text") or ""))
    if receipt.tool == "house_answer":
        return t("house_approved" if d.get("approved") else "house_rejected", lang, what=str(d.get("what") or ""))
    raise ValueError("not a house receipt")


def _source(now_state: Any, lang: str) -> str:
    from .i18n import t  # noqa: PLC0415

    if now_state.source == "owner":
        return t("source_owner_by", lang, by=now_state.by) if now_state.by else t("source_owner", lang)
    if now_state.source == "schedule":
        sched = _schedule(now_state)
        return t("source_schedule", lang, start=sched[0], end=sched[1]) if sched else t("source_schedule_off", lang)
    return t(f"source_{now_state.source}", lang) if now_state.source in ("proposal", "system") else now_state.source


def _asker(p: Dict[str, Any], lang: str) -> str:
    from .i18n import t  # noqa: PLC0415

    return t("source_proposal" if p.get("source") == "proposal" else "source_system", lang)   # never the owner


def status(ctx: Any) -> Tuple[str, Tuple[Tuple[Tuple[str, str], ...], ...]]:
    """The status line: the house state (source, end), what is expected, today's pauses, the cameras that watch,
    and the requests waiting for the owner - each with a yes and a no button."""
    from .i18n import t  # noqa: PLC0415

    store, now, lang = ctx.services.house, float(ctx.services.now()), ctx.lang
    house = store.current(now)
    none = t("status_none", lang)
    lines = [t("status_house", lang, state=t(f"state_{house.state}", lang), source=_source(house, lang),
               until=until_text(_ts(house.expires_at), now, lang))]
    notes = [t("status_item_until", lang, what=e["text"], when=when(_ts(e["expires_at"]), now, lang))
             for e in house.expecting]
    lines.append(t("status_expecting", lang, items="; ".join(notes) or none))
    mutes = [t("status_item_until", lang, what=m["camera"] or t("status_whole_house", lang),
               when=when(_ts(m["expires_at"]), now, lang)) for m in house.mutes]
    lines.append(t("status_mutes", lang, items="; ".join(mutes) or none))
    snap = ctx.snapshot
    if snap is not None:
        watching = [c.name for c in snap.cameras if c.enabled and c.live is not False]
        line = t("status_cameras", lang, names=", ".join(watching) or none, live=len(watching),
                 total=len(snap.cameras))
        off = [c.name for c in snap.cameras if not c.enabled]
        down = [c.name for c in snap.cameras if c.enabled and c.live is False]
        if off:
            line += " · " + t("status_cameras_off", lang, names=", ".join(off))
        if down:
            line += " · " + t("status_cameras_down", lang, names=", ".join(down))
        lines.append(line)
    rows = []
    for p in store.pending_proposals(now):
        lines.append(t("status_pending", lang, what=describe_proposal(p, lang), source=_asker(p, lang),
                       when=when(_ts(p.get("expires_at")), now, lang)))
        rows.append(((t("btn_approve", lang), f"hs:{p['id']}:y"), (t("btn_reject", lang), f"hs:{p['id']}:n")))
    return "\n".join(lines), tuple(rows)


def context_line(store: Any, now: float) -> str:
    """The house state as the model reads it in the context block (English, code-written)."""
    house = store.current(now)
    end = when(_ts(house.expires_at), now, "en") if house.expires_at else "the owner says otherwise"
    parts = [f"[HOUSE STATE] {house.state} ({house.source}) until {end}"]
    if house.expecting:
        parts.append("expecting: " + "; ".join(f"{e['text']} until {when(_ts(e['expires_at']), now, 'en')}"
                                               for e in house.expecting))
    if house.pending:
        parts.append(f"{house.pending} change(s) wait for the owner's yes or no (the status command shows them)")
    return " · ".join(parts)


# -- Undo ----------------------------------------------------------------------------------------------------------
class ChangedSince(Exception):
    """What the house command changed no longer holds (another command, the schedule, an approved request)."""


def _restore(ctx: Any, before: Dict[str, Any], now: float) -> Any:
    """Put back the state that held before a command: an owner's or an approved state is set again with its old
    end (if that end is still ahead); otherwise the night schedule simply takes over again."""
    store = ctx.services.house
    until = before.get("until")
    if before.get("entry_id") and before.get("source") not in ("", None, "schedule") and (
            until is None or float(until) > now):
        entry = store.set_house_state(before["state"], "owner", until=until, by=_by(ctx), now=now)
        if entry.get("status") != "applied":
            raise ValueError("the earlier house state could not be set again")
    after = store.current(now)
    return _issue(ctx, "house_state", "done", after.state,
                  {"state": after.state, "until": _ts(after.expires_at), "entry_id": after.entry_id,
                   "undo_of": "house_state"})


def undo(ctx: Any, receipt: Any, now: float) -> None:
    """Reverse one house receipt while what it did still holds; raises ChangedSince when it no longer does."""
    store, d = ctx.services.house, receipt.detail
    current = store.current(now)
    if receipt.tool == "house_state":
        if not d.get("entry_id") or current.entry_id != d["entry_id"]:
            raise ChangedSince()
        if not store.cancel_entry(d["entry_id"], source="owner", by=_by(ctx), now=now):
            raise ValueError("the house state could not be ended")
        _restore(ctx, d.get("before") or {}, now)
    elif receipt.tool == "house_expect":
        if all(e["id"] != d.get("id") for e in current.expecting):
            raise ChangedSince()
        if not store.cancel_expecting(d["id"], by=_by(ctx), now=now):
            raise ValueError("the note could not be cancelled")
        _issue(ctx, "house_cancel", "done", str(d.get("text") or ""),
               {"kind": "expect", "id": d["id"], "text": str(d.get("text") or ""), "undo_of": "house_expect"})
    elif receipt.tool == "house_cancel" and d.get("kind") == "state":
        old_until = d.get("old_until")
        if current.entry_id or (old_until is not None and float(old_until) <= now):
            raise ChangedSince()                  # something else holds now, or it would have ended anyway
        _restore(ctx, {"state": d.get("old"), "source": d.get("old_source") or "owner", "entry_id": d.get("entry_id"),
                       "until": old_until}, now)
    elif receipt.tool == "house_cancel":
        until = d.get("until")
        if until is None or float(until) <= now or any(e["text"] == d.get("text") for e in current.expecting):
            raise ChangedSince()
        add_expect(ctx, str(d.get("text") or ""), float(until), d.get("camera") or None)
        ctx.receipts[-1].detail["undo_of"] = "house_cancel"
    else:
        raise ValueError("not a house receipt")
