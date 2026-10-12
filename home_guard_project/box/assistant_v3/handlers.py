"""ACT: the deterministic handlers (report §8.3 [2]). Everything that changes state runs here, in code, by the act's
TYPE: memory writes with their type's validity policy, tags only on an explicit tag act, commands with receipts.
A handler never writes the reply; it adds what it DID (``plan.done``, owner-facing Hebrew facts), what it SAW
(``plan.evidence``) and at most one question with button answers (``plan.ask``). The writer phrases it.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .. import activity_memory as am
from ..brain import known_memory as km
from ..brain.receipts import DONE, FAILED
from ..brain.registry import current_camera
from ..brain.tools import TOOLS, ToolContext, _issue, quoted_from, retag_words
from .acts import Act, Understanding
from .context import Cameras, hhmm
from . import past_photo
from .memory_view import MemoryView
from ...prompts import load, render
from .skills import Toolbox, internal
from .timeparse import clock, parse_until

log = logging.getLogger("box.assistant_v3.handlers")

RECENT_EVENT_SEC = 3 * 3600.0
PHOTO_FRESH_SEC = 15 * 60.0
MAX_PAST_VIDEOS = 2


@dataclass
class Plan:
    done: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    ask: Optional[Dict[str, Any]] = None
    template: Optional[str] = None
    rows: List[Any] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    cited: List[str] = field(default_factory=list)      # memory lines the writer may use for this turn
    notes: List[str] = field(default_factory=list)      # how to phrase THIS turn (for the writer, never sent)
    long_ok: bool = False
    final: Optional[str] = None                         # a reply written by code that goes out as is (day story)


@dataclass
class Turn:
    ctx: ToolContext
    mem: MemoryView
    cams: Cameras
    text: str
    now: float
    alert: Optional[Dict[str, Any]]
    und: Understanding
    called: List[str]
    box: Toolbox
    plan: Plan = field(default_factory=Plan)

    @property
    def state(self) -> Any:
        return self.ctx.state

    @property
    def snapshot(self) -> Any:
        return self.ctx.snapshot

    # -- what an act points at ----------------------------------------------------------------------------------
    def events(self, within: float = RECENT_EVENT_SEC) -> List[Tuple[str, Dict[str, Any]]]:
        rows = [(h, e) for h, e in (self.state.handles or {}).items() if isinstance(e, dict)
                and e.get("kind") == "event" and 0 <= self.now - float(e.get("ts") or 0) <= within]
        return sorted(rows, key=lambda kv: float(kv[1].get("ts") or 0), reverse=True)

    def event_of(self, act: Optional[Act]) -> Tuple[str, Dict[str, Any]]:
        for handle in ([act.event] if act and act.event else []) + [self.ctx.alert_handle or "",
                                                                    self.state.topic_event(self.now) or ""]:
            entry = self.state.resolve(handle) if handle else None
            if isinstance(entry, dict) and entry.get("kind") == "event":
                return handle, entry
        rows = self.events()
        return rows[0] if rows else ("", {})

    def camera_now(self, raw: str) -> str:
        return (current_camera(self.snapshot, raw) or raw) if raw else ""

    def camera_of(self, act: Optional[Act], use_event: bool = True) -> str:
        """The act's camera: named, else the alert's / the event being discussed, else the topic camera, else the
        last photo's. "" when nothing points at one."""
        if act and act.camera and act.camera != "house":
            return act.camera
        if use_event:
            _, entry = self.event_of(act)
            if entry.get("camera"):
                return self.camera_now(str(entry["camera"]))
        topic = self.state.topic_camera(self.now)
        if topic:
            return topic[0]
        return self.last_photo_camera() or ""

    def last_photo(self, within: float = PHOTO_FRESH_SEC) -> Tuple[str, Dict[str, Any]]:
        """The newest live photo (of the newest ones, the one with the most people in it)."""
        rows = [(h, e) for h, e in (self.state.handles or {}).items() if isinstance(e, dict)
                and e.get("kind") == "photo" and 0 <= self.now - float(e.get("ts") or 0) <= within]
        if not rows:
            return ("", {})
        newest = max(float(e.get("ts") or 0) for _, e in rows)
        latest = [(h, e) for h, e in rows if newest - float(e.get("ts") or 0) <= 120]
        return max(latest, key=lambda kv: (int(kv[1].get("people") or 0) or _people_in(kv[1].get("observation")),
                                           float(kv[1].get("ts") or 0)))

    def last_photo_camera(self, within: float = PHOTO_FRESH_SEC) -> str:
        _, e = self.last_photo(within)
        return self.camera_now(str(e.get("camera") or "")) if e else ""

    def name(self, camera: str) -> str:
        return self.cams.name(camera) if camera else "כל הבית"

    def outcome(self, entry: Dict[str, Any], what: str) -> None:
        if isinstance(entry, dict) and entry:
            entry["v3_outcome"] = what[:160]

    def mem_owner_words(self) -> List[str]:
        """His own messages of the live window, newest first (what "she" and "it" point at)."""
        from .context import earlier_owner_words  # noqa: PLC0415

        return list(reversed(earlier_owner_words(self.state, self.now, self.text)))

    def did(self, line: str) -> None:
        self.plan.done.append(line)


# ---------------------------------------------------------------------------------------------------------------------
_COUNT = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "אחד": 1, "שני": 2, "שניים": 2,
          "שלושה": 3, "ארבעה": 4, "חמישה": 5}


def _people_in(text: Any) -> int:
    """How many people a description names ("Three men work..." -> 3, "A man kneels..." -> 1, "no people" -> 0)."""
    text = str(text or "")
    if re.search(r"\bno (?:people|one)\b|אין אנשים|אין אף אחד", text, re.IGNORECASE):
        return 0
    m = re.search(r"\b(a|an|one|two|three|four|five|\d)\s+(?:\w+\s+)?(?:man|men|woman|women|person|people|workers?)\b",
                  text, re.IGNORECASE)
    if m:
        word = m.group(1).lower()
        return int(word) if word.isdigit() else _COUNT.get(word, 1)
    return 1 if re.search(r"\b(?:man|men|woman|person|people|workers?)\b|אדם|אנשים|גבר|עובד|אישה", text,
                          re.IGNORECASE) else 0


def place_fact(t: Turn, act: Act) -> None:
    from .. import place_facts  # noqa: PLC0415

    said = f"{act.subject} {act.quote}"
    if _crew(said) and not re.search(r"בית|חצר|חניה|שטח|מגרש|גינה|רחוב|בניין|שכנ", said):
        return person_mark(t, act)             # "המידע זה עובדים אצלי על הפרגולה" is people, not a place
    handle, entry = t.event_of(act)
    camera = t.camera_of(act)
    words = act.quote or act.subject
    said = place_facts.place_statement(words) or place_facts.place_statement(act.subject or "") or {}
    owner = said.get("owner") or ("neighbour" if re.search(r"שכנ", words) else "other")
    event = None
    if entry and t.camera_now(str(entry.get("camera") or "")) == camera:
        event = {"alert_id": str(entry.get("ref") or ""), "camera": str(entry.get("camera") or ""),
                 "ts": float(entry.get("ts") or 0)}
    fact = t.mem.save_place(t.ctx, camera, said.get("words") or act.subject or words, owner, event,
                            whole_house=not camera or act.camera == "house")
    if fact is None:
        t.plan.trace.append("place_fact: not saved")
        return
    where = t.name(camera)
    if fact.get("already"):
        t.did(f"ב{where} — {fact['text']}: זה כבר היה ידוע לי, ומה שקורה שם לא מקפיץ התראה.")
    else:
        t.did(f"ב{where} — {fact['text']} (לתמיד, בלי תאריך): מה שקורה שם לא יקפיץ התראה, רק מי שעובר לשטח "
              f"של הבית.")
    if event:
        t.outcome(entry, f"בעל הבית: {fact['text']} → נסגרה כרגילה")
        t.did(f"ההתראה של {hhmm(entry.get('ts'))} שם נסגרה כרגילה.")


def _crew(text: str) -> bool:
    return km.work_group(text) or bool(re.search(r"פועל|עובד|קבלן|חשמלא|אינסטלט|גנן|צבע|שיפוצ|צוות|טכנאי", text))


def _shared(a: str, b: str) -> bool:
    from .memory_view import words  # noqa: PLC0415

    return bool(words(a) & words(b))


def person_mark(t: Turn, act: Act, until_words: str = "") -> None:
    who = act.subject or act.quote
    if not who:
        return
    if act.routine:
        return camera_fact(t, act)
    if km._ONE_OFF.search(f"{who} {act.quote}"):
        return alert_feedback(t, act)          # the postman, a delivery: a one-off closes the alert, never a mark
    house_wide = act.scope == "house" or act.camera == "house"
    camera = "" if house_wide else t.camera_of(act)
    handle, entry = t.event_of(act)
    book = getattr(t.ctx.services, "events", None)
    live = km.live_marks(book, t.now) if book is not None else []
    crew = _crew(f"{who} {act.quote}")
    same = [k for k in live if km.same_people(who, str(k.get("text") or "")) or
            (crew and _crew(str(k.get("text") or "")) and _shared(who, str(k.get("text") or "")))]
    if crew and same and not house_wide:
        # A crew moves around (2026-10-09 09:46): he explains people at a camera their mark does not cover yet.
        recent = [e for _, e in t.events(3600)]
        if recent:
            cam_now = t.camera_now(str(recent[0].get("camera") or ""))
            if cam_now and not any(not k.get("camera") or str(k.get("camera")) == cam_now for k in same):
                house_wide, camera = True, ""
    from .acts import quoted  # noqa: PLC0415

    # an hour from an EARLIER message with the same people already marked: widen, never a new window
    words = until_words or ("" if (act.earlier and same and not quoted(act.until_quote, t.text)) else act.until_quote)
    until = parse_until(words, t.now) if words else None
    covered = [k for k in same if not k.get("camera") or str(k.get("camera")) == camera]
    if until is None and same:
        # Re-stated or widened: keep the window he already gave (never a new one he did not say).
        k = max(same, key=lambda x: float(x.get("at") or 0))      # his latest word on them (a correction wins)
        if covered and not house_wide:
            t.did(f"{who} כבר מסומנים אצלי ב{t.name(camera) if camera else 'כל הבית'} {_when_text(k, t.now)}; "
                  f"לא שיניתי כלום.")
            t.outcome(entry, f"בעל הבית: אלה {who} (כבר מסומנים)")
            return
        wide = house_wide or len(same) > 1 or not camera
        saved = t.mem.mark_people(t.ctx, str(k.get("text") or who), "" if wide else camera,
                                  float(k.get("until") or 0), str(k.get("daily_from") or ""),
                                  str(k.get("daily_to") or ""), replaces=same)
        if saved:
            before = ", ".join(sorted({t.name(str(x.get("camera") or "")) for x in same}))
            t.did(f"הרחבתי: {saved.get('text') or who} מסומנים עכשיו {'בכל הבית' if wide else 'ב' + t.name(camera)}, "
                  f"באותן שעות כמו קודם (קודם רק ב{before}).")
            t.outcome(entry, f"בעל הבית: אלה {who} → נסגרה כרגילה")
            if entry:
                t.did(f"ההתראה של {hhmm(entry.get('ts'))} ב{t.name(t.camera_now(str(entry.get('camera') or '')))} "
                      f"נסגרה, אלה הם.")
        return
    if until is None:
        until = km.until_said_today(t.state, who, t.now)
    if until is None:
        where = f"ב{t.name(camera)}" if camera else "בכל הבית"
        quiet = act.repaired or t.und.emotion == "angry" or t.und.has("activity_explain")
        if quiet:
            # He is complaining or this repairs an earlier miss: no question now; kept for today, said as "today".
            end = dt.datetime.fromtimestamp(t.now).replace(hour=23, minute=59, second=0).timestamp()
            if t.mem.mark_people(t.ctx, who, camera, end, replaces=[]) is not None:
                t.outcome(entry, f"בעל הבית: אלה {who} → נסגרה כרגילה")
                t.did(f"סימנתי את {who} {where} להיום, כדי שלא תגיע עליהם התראה.")
            return
        # The owner's own way (2026-10-09): "אה, הם של הפרגולה? עד איזו שעה הם עובדים?" - nothing saved with a
        # time he did not say; the tapped or typed hour completes it in code.
        choices = km.likely_hours(t.now) + ["אחר…"]
        question = "עד איזו שעה הם עובדים?" if crew else "עד מתי הם פה היום?"
        t.plan.ask = {"question": question, "choices": choices, "kind": "v3_mark",
                      "args": {"who": who, "camera": camera, "house": house_wide, "handle": handle,
                               "quote": act.quote[:200], "crew": crew}}
        t.outcome(entry, f"בעל הבית: אלה {who}")
        t.plan.notes.append(f"{who}: עוד לא נשמרו (חסרה שעה). אם עשית בתור הזה משהו אחר (למשל תיוג) תגיד אותו "
                            f"בקצרה קודם; ואז הד קצר כמו 'אה, אלה {who}?' והשאלה. אל תגיד שסימנת אותם.")
        return
    daily_from = daily_to = ""
    span = until
    one_day = km.one_day(act.quote) or km.one_day(t.text)
    daily_old = [k for k in same if k.get("daily_from")]
    if crew and not one_day:
        if daily_old:
            k = daily_old[-1]
            daily_from = str(k.get("daily_from"))
            end = dt.datetime.fromtimestamp(until)
            daily_to = end.strftime("%H:%M") if end.date() == dt.datetime.fromtimestamp(t.now).date() else str(k.get("daily_to"))
            span = max(float(k.get("until") or 0), until)
            if daily_from >= daily_to:
                daily_from = daily_to = ""
                span = until
        else:
            win = km.crew_window(book, camera or str((entry or {}).get("camera") or ""), until, t.now,
                                 fallback=float(entry.get("ts")) if entry and entry.get("ts") else None)
            if win:
                daily_from, daily_to, span = win["daily_from"], win["daily_to"], float(win["until"])
    replaces = [k for k in same if not camera or not k.get("camera") or str(k.get("camera")) == camera] if same else []
    saved = t.mem.mark_people(t.ctx, who, camera, span, daily_from, daily_to, replaces=replaces)
    if not saved:
        t.plan.trace.append("person_mark: not saved")
        return
    where = f"ב{t.name(camera)}" if camera else "בכל הבית"
    when = _when_text(saved, t.now)
    line = f"{who} {where} {when}: לא תגיע עליהם התראה בשעות האלה."
    if daily_from and not daily_old:
        line += f" (הנחה: עבודה של כמה ימים, כל יום {daily_from}–{daily_to}.)"
        t.plan.notes.append("יש כפתורים לתיקון (רק היום / שבוע); אל תשאל על זה ואל תציע במילים.")
    if replaces:
        olds = ", ".join(sorted({clock(float(k.get('until') or 0)) for k in replaces}))
        line += f" זה מחליף את הסימון הקודם (עד {olds})."
    t.did(line)
    t.outcome(entry, f"בעל הבית: אלה {who} → נסגרה כרגילה")


def _when_text(mark: Dict[str, Any], now: float) -> str:
    until = float(mark.get("until") or 0)
    if mark.get("daily_from"):
        last = km.day_text(until, "he")
        return f"כל יום {mark.get('daily_from')}–{mark.get('daily_to')} עד {last}"
    same_day = dt.datetime.fromtimestamp(until).date() == dt.datetime.fromtimestamp(now).date()
    return f"עד {clock(until)}" + ("" if same_day else f" ({km.day_text(until, 'he')})")


# ---------------------------------------------------------------------------------------------------------------------
def activity_explain(t: Turn, act: Act) -> None:
    words = " ".join(x for x in (act.subject, act.quote) if x)
    named_places = am.places_in(words)
    if not am.actions_in(words) and not named_places and _crew(words):
        return person_mark(t, act)             # "הם עובדים בחוץ": who they are, not what an action means
    candidates = []
    for handle, entry in t.events():
        text = f"{entry.get('observation') or ''} {entry.get('summary') or ''}"
        acts = am.actions_in(text)
        if not acts:
            continue
        score = 0.0
        if act.event == handle:
            score += 3
        cam = t.camera_now(str(entry.get("camera") or ""))
        if act.camera and act.camera != "house" and cam == act.camera:
            score += 4
        if set(named_places) & set(am.places_in(text)):
            score += 2
        if set(am.actions_in(words)) & set(acts):
            score += 1
        if "car_door" in acts and not re.search(r"רכב|אוטו|מכונית|דלת|car|door", words, re.IGNORECASE):
            score -= 10                    # a car door stays red (owner, 2026-10-09): work never explains it
        candidates.append((score, float(entry.get("ts") or 0), handle, entry))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    handle, entry = (candidates[0][2], candidates[0][3]) if candidates else t.event_of(act)
    camera = t.camera_now(str((entry or {}).get("camera") or "")) if candidates else ""
    camera = camera or (act.camera if act.camera and act.camera != "house" else "")
    if not camera:
        t.plan.trace.append("activity: no camera")
        return
    event_text = f"{(entry or {}).get('observation') or ''} {(entry or {}).get('summary') or ''}"
    actions = list(dict.fromkeys(am.actions_in(event_text) + am.actions_in(words)))
    if "car_door" in actions and not re.search(r"דלת|door", words, re.IGNORECASE):
        actions.remove("car_door")
    if not actions:
        t.plan.trace.append("activity: no action named")
        return person_mark(t, act) if _crew(words) else None
    book = getattr(t.ctx.services, "events", None)
    from ..brain import activity_chat  # noqa: PLC0415

    win = activity_chat.window(book, camera, act.until_quote, t.now)
    today = False
    if win is None:
        today = True
        win = {"until": dt.datetime.fromtimestamp(t.now).replace(hour=23, minute=59).timestamp(), "daily_from": "",
               "daily_to": "", "who": "", "known_id": ""}
    place = next(iter(named_places), "") or next(iter(am.places_in(event_text)), "")
    cause = act.subject or act.quote
    fact = t.mem.add_activity(t.ctx, [camera], actions, cause, float(win["until"]), place=place,
                              daily_from=win["daily_from"], daily_to=win["daily_to"], who=win["who"],
                              known_id=win["known_id"], owner_words=t.text,
                              alert_id=str((entry or {}).get("ref") or ""))
    if fact is None:
        return
    where = f"{am.place_text(fact, 'he')} ב{t.name(camera)}" if fact.place else f"ב{t.name(camera)}"
    if today:
        when = "להיום (לא נאמר עד מתי)"
    elif fact.daily_from:
        when = f"כל יום {fact.daily_from}–{fact.daily_to} עד {km.day_text(fact.until, 'he')}"
    else:
        when = f"עד {clock(fact.until)}"
    t.did(f"{am.actions_text(fact.actions, 'he')} {where} = {fact.cause}: זה ייחשב רגיל {when}"
          f"{' (אותו חלון כמו הסימון של ' + win['who'] + ')' if win.get('who') else ''}; פריצה, פגיעה או דלת רכב "
          f"עדיין יתריעו.")
    t.plan.rows.append((("↩ ביטול", f"kn:x:{fact.id}"),))
    if entry:
        t.outcome(entry, f"בעל הבית: {fact.cause} → נסגרה כרגילה")
        t.did(f"ההתראה של {hhmm(entry.get('ts'))} ב{t.name(camera)} נסגרה כרגילה.")


def camera_fact(t: Turn, act: Act) -> None:
    camera = "" if act.camera == "house" else t.camera_of(act, use_event=bool(t.alert))
    words = act.quote or act.subject
    fact = t.mem.save_camera_fact(t.ctx, camera, words)
    if fact is None:
        return
    where = f"ב{t.name(camera)}" if camera else "בבית"
    t.did(f"{where}: {fact['text']} ({'כבר היה ידוע' if fact.get('already') else 'לתמיד'}).")
    t.plan.notes.append(load("assistant_v3_note_fact_saved.prompt"))


def tag_only(t: Turn, act: Act) -> None:
    handle, entry = t.event_of(act)
    tag = act.value if act.value and quoted_from(act.value, t.text) else (retag_words(t.text) or "")
    if not tag and act.quote and quoted_from(act.quote, t.text):
        tag = act.quote
    if not handle or not tag:
        t.plan.trace.append("tag: no clip or no words")
        return
    t.called.append("retag_clip")
    out = TOOLS["retag_clip"](t.ctx, {"handle": handle, "tag": tag})
    if out.get("ok"):
        t.did(f"🏷️ התיוג של הסרטון מ-{hhmm(entry.get('ts'))} ({t.name(t.camera_now(str(entry.get('camera') or '')))}) "
              f"עודכן: {tag}. (תיוג משמש רק לאימון המודל.)")
    else:
        t.plan.trace.append(f"tag failed: {out.get('error')}")


def alert_feedback(t: Turn, act: Act) -> None:
    handle, entry = t.event_of(act)
    if not entry:
        return
    verdict = {"normal": "תקינה", "false_alarm": "התראת שווא", "real": "אמיתית"}.get(act.verdict or "normal", "תקינה")
    t.outcome(entry, f"בעל הבית: {verdict}")
    t.did(f"ההתראה של {hhmm(entry.get('ts'))} ב{t.name(t.camera_now(str(entry.get('camera') or '')))} סומנה אצלי "
          f"כ{verdict}.")


_NAMING = re.compile(r"תקרא|קרא ל|נקרא|תקראו|שם ל?ה?מצלמה|השם של|שם חדש|תשנה את השם|תן לה שם|\bcall\b|\bname\b|rename",
                     re.IGNORECASE)


def preference(t: Turn, act: Act) -> None:
    if act.command == "alias" and act.value and not _NAMING.search(f"{act.quote} {t.text}"):
        # 2026-10-11 07:57 "מעולה ברגע שהיא יוצאת תן לי דיווח" became the camera name "אשת בעל הבית" (twice): a name
        # is saved only when he says to call / name the camera.
        t.plan.trace.append(f"alias without naming words dropped: {act.value!r}")
        return
    if act.command == "alias" and act.value:
        camera = act.camera if act.camera and act.camera != "house" else (t.last_photo_camera(3600) or t.camera_of(act))
        if not camera:
            t.plan.trace.append("alias: no camera")
            return
        t.called.append("set_alias")
        before = t.cams.name(camera)
        out = TOOLS["set_alias"](t.ctx, {"camera": camera, "alias": act.value})
        if out.get("ok"):
            t.did(f"השם החדש של המצלמה ({before}) הוא '{act.value}'." if before != act.value
                  else f"המצלמה נקראת '{act.value}'.")
        return
    value = act.value or act.quote
    if value:
        t.mem.add_preference(value)
        t.did(f"מעכשיו: {value}")
        t.plan.notes.append("תאשר במשפט קצר וטבעי מה ישתנה מעכשיו; בלי 'נשמר' ובלי להזכיר את הזיכרון.")


def command(t: Turn, act: Act) -> None:
    kind = act.command
    if kind == "pause":
        words = act.until_quote or act.quote or t.text
        until = parse_until(words, t.now) or parse_until(t.text, t.now)
        if until is None:
            t.plan.ask = {"question": "עד מתי להשתיק?", "choices": ["שעה", "עד הערב", "עד מחר בבוקר"],
                          "kind": "v3_pause", "args": {"camera": act.camera}}
            return
        camera = act.camera if act.camera and act.camera != "house" else None
        t.called.append("pause_alerts")
        if pause(t, camera, until):
            where = f"ב{t.name(camera)}" if camera else "בכל המצלמות"
            t.did(f"ההתראות מושתקות {where} עד {clock(until)}.")
    elif kind == "resume":
        t.called.append("resume_alerts")
        camera = act.camera if act.camera and act.camera != "house" else None
        if resume(t, camera):
            t.did("ההתראות חזרו לפעול.")
    elif kind == "camera_off" and (act.until_quote or not act.camera or act.camera == "house"):
        # "תכבה את המצלמות עד ..." means quiet until then (the cameras keep recording): a pause
        act.command = "pause"
        return command(t, act)
    elif kind in ("camera_off", "camera_on"):
        camera = t.camera_of(act)
        if not camera:
            return
        t.called.append("set_camera_active")
        out = TOOLS["set_camera_active"](t.ctx, {"camera": camera, "active": kind == "camera_on"})
        if out.get("ok"):
            t.did(f"המצלמה {t.name(camera)} {'מודלקת' if kind == 'camera_on' else 'מכובה'} (לוקח כחצי דקה).")
    elif kind == "send_video":
        send_video(t, act)
    elif kind == "send_photo":
        send_photo(t, act)
    elif kind in ("house_mode", "setting"):
        from ..brain import house  # noqa: PLC0415

        services = t.ctx.services
        if services.house is not None:
            schedule = house.schedule_of(services.house, t.now) or house.DEFAULT_SCHEDULE
            out = house.run_command(t.ctx, house.parse_command(t.text, t.now, schedule))
            if out.handled and out.text:
                t.plan.final = out.text
                t.plan.rows.extend(out.rows or ())


_PHOTO_ASK = re.compile(r"(?<![א-ת])(?:תמונה+|תמונות|תצלם|צלם|צילום|סנאפ)(?![א-ת])|\b(?:photo|picture|snapshot)\b",
                        re.IGNORECASE)


def photo_request(t: Turn) -> None:
    """"תן לי תמונה" / "נוווו תן לי פאקינג תמונהה" always gets a picture (2026-10-11 10:11-10:14: four asks, the first
    three read as a complaint or answered "no live picture from the main camera"). A message that asks for a picture
    and has no send_photo / send_video act gets one, at the camera he names now or named in his last messages."""
    if any(a.act == "command" and a.command in ("send_photo", "send_video") for a in t.und.acts):
        return
    if not _PHOTO_ASK.search(t.text):
        # "לא של עכשיו יא מטומטם אני מתכוון שהם היו" (2026-10-12 11:44): no picture word, but it corrects the
        # picture he just asked for: the past one is sent
        earlier = t.mem_owner_words()[:1]
        if not (past_photo.asks_past(t.text) and earlier and _PHOTO_ASK.search(earlier[0])):
            return
        t.und.acts.append(Act(act="command", command="send_photo", quote=t.text[:120]))
        t.plan.trace.append("photo asked again (past): send_photo added")
        return
    if t.und.has("question_live") and not t.und.has("complaint"):
        return                                   # "יש תמונה מהכניסה?" as a live question: prefetch looks already
    t.und.acts.append(Act(act="command", command="send_photo", quote=t.text[:120]))
    t.plan.trace.append("photo asked: send_photo added")


def _named_camera(t: Turn, *texts: str) -> str:
    from ..brain.registry import mentioned_cameras  # noqa: PLC0415

    for text in texts:
        named = [cam for _, cam in mentioned_cameras(t.snapshot, text or "")]
        if named:
            return t.camera_now(named[-1])
    return ""


def send_photo(t: Turn, act: Act) -> None:
    """A live picture: of the camera he names (in this message or his last two), else the one being discussed, else
    of every camera that is on. A camera that is off is said, and the others are shown instead."""
    enabled = {c.name for c in t.snapshot.cameras if c.enabled}
    owner = t.mem_owner_words()[:2]
    camera = act.camera if act.camera and act.camera != "house" else ""
    # what he names now, else what was just talked about (an alert or the box's line: the topic), else his last words
    others = [a.camera for a in t.und.acts if a.camera and a.camera != "house"]
    camera = _named_camera(t, t.text) or camera or (others[0] if others else "") or t.camera_of(act) or         _named_camera(t, *owner)
    if past_photo.asks_past(t.text):
        send_past_photo(t, camera)                # "תמונה שהם היו שם ב-10:24": the recording, never a live picture
        return
    if camera and camera not in enabled:
        t.did(f"{t.name(camera)} כבויה, אז שלחתי מהמצלמות שפועלות.")
        camera = ""
    cams = [camera] if camera else [c.name for c in t.snapshot.cameras if c.enabled]
    sent = []
    for cam in cams:
        t.box.run("look_now", {"camera": cam})
        if any(r.tool == "check_camera" and r.status == DONE and r.target == cam for r in t.ctx.receipts):
            sent.append(t.name(cam))
    if sent:
        t.did(f"נשלחה תמונה חיה מ{', '.join(sent)}.")
        t.plan.notes.append(load("assistant_v3_note_photo_sent.prompt"))
    else:
        t.did("לא הצלחתי לקבל תמונה מהמצלמות כרגע.")


def _past_events(t: Turn, camera: str) -> Tuple[List[Tuple[Dict[str, Any], float]], List[float], str]:
    """The journal events he asks to see (past_photo.pick_events), the moments he named, and where (for the "none
    found" line)."""
    from ..brain.tools import _memory  # noqa: PLC0415

    memory = _memory(t.ctx)
    records: List[Dict[str, Any]] = []
    if memory is not None:
        try:
            records = list(memory.records())
            book = getattr(t.ctx.services, "events", None)
            if book is not None:
                records += [memory.live_record(s) for s in book.recent(t.now - 86400.0, "") if not s.get("closed")]
        except Exception as exc:  # noqa: BLE001
            log.warning("v3: event memory not read for a past photo: %s", exc)
    from .context import last_replies  # noqa: PLC0415

    replies = last_replies(t.state, 1)
    camera = camera or _named_camera(t, replies[-1] if replies else "")    # "בשעות האלה" after "נראו בפרגולה ..."
    if camera:
        records = [r for r in records if t.camera_now(str(r.get("camera") or "")) == camera]
    at_times = past_photo.moments(t.text, t.mem_owner_words(), replies[-1] if replies else "", t.now)
    return past_photo.pick_events(records, at_times, t.now), at_times, (t.name(camera) if camera else "המצלמות")


def _none_found(t: Turn, at_times: List[float], where: str) -> None:
    asked = ", ".join(clock(a) for a in at_times)
    t.did(f"לא מצאתי הקלטה עם אנשים ב{where}" + (f" סביב {asked}" if asked else "") + ".")


def send_past_photo(t: Turn, camera: str) -> None:
    """A picture of a moment that has passed, from the box's recordings (past_photo.py): the event with people at
    the time he gives (else the times the assistant just cited, else the newest), a still of its clip, with the
    camera's name and the time it shows."""
    picked, at_times, where = _past_events(t, camera)
    sent = []
    for record, at in picked:
        path, shown = past_photo.still_of(record, at, t.ctx.services.roots(), t.ctx.services.work_dir)
        if not path:
            continue
        cam = t.camera_now(str(record.get("camera") or ""))
        handle = t.state.add_handle("photo", path, cam, shown)
        seen = past_photo.observation_at(record, shown)
        t.state.note_observation(handle, str(seen.get("summary") or record.get("caption") or "")[:300])
        t.called.append("send_media")
        with internal(t.ctx):
            out = TOOLS["send_media"](t.ctx, {"handle": handle})
        if out.get("ok") and out.get("status", DONE) == DONE:
            sent.append(f"{t.name(cam)} {clock(shown)}")
            t.plan.evidence.append(f"תמונה מההקלטה (לא חיה) מ{t.name(cam)} בשעה {clock(shown)}, מתוך האירוע "
                                   f"{clock(float(record.get('start') or shown))}–{clock(float(record.get('end') or shown))}: "
                                   f"{seen.get('summary') or record.get('caption') or ''}")
    if sent:
        t.did(f"נשלחו תמונות מההקלטות של הקופסה (לא חיות): {', '.join(sent)}.")
        t.plan.notes.append(load("assistant_v3_note_past_photo_sent.prompt"))
    else:
        _none_found(t, at_times, where)


def send_past_video(t: Turn, camera: str) -> None:
    """"תביא לי סרטון של מישהו שהיה פה בפרגולה" (2026-10-12 12:33): the recorded video of that event (its alert clip
    nearest the moment), never 10 live seconds of an empty place."""
    picked, at_times, where = _past_events(t, camera)
    sent = []
    for record, at in picked[:MAX_PAST_VIDEOS]:
        cam = t.camera_now(str(record.get("camera") or ""))
        ids = sorted([a for a in record.get("alert_ids") or [] if a], key=lambda a: abs(past_photo._alert_ts(a) - at))
        for alert_id in ids:
            seen = past_photo.observation_at(record, at)
            handle = t.state.add_handle("event", alert_id, cam, past_photo._alert_ts(alert_id) or at,
                                        str(seen.get("summary") or record.get("caption") or "")[:200])
            t.called.append("send_media")
            with internal(t.ctx):
                out = TOOLS["send_media"](t.ctx, {"handle": handle})
            if out.get("ok") and out.get("status", DONE) == DONE:
                when = past_photo._alert_ts(alert_id) or at
                sent.append(f"{t.name(cam)} {clock(when)}")
                t.plan.evidence.append(f"סרטון מההקלטה (לא חי) מ{t.name(cam)} בשעה {clock(when)}, מתוך האירוע "
                                       f"{clock(float(record.get('start') or when))}–{clock(float(record.get('end') or when))}: "
                                       f"{seen.get('summary') or record.get('caption') or ''}")
                break
    if sent:
        t.did(f"נשלחו סרטונים מההקלטות של הקופסה (לא חיים): {', '.join(sent)}.")
        t.plan.notes.append(load("assistant_v3_note_past_photo_sent.prompt"))
    else:
        _none_found(t, at_times, where)


def pause(t: Turn, camera: Optional[str], until: float) -> bool:
    """Alerts paused until *until* (one camera or all), with v2's receipt so Undo puts the old state back. Unlike
    v2's tool it files nothing in feedback/: a command in the chat is conversation, not a tag (owner, 2026-10-09)."""
    from ..feedback import Feedback  # noqa: PLC0415

    mute = t.ctx.services.mute
    until = min(until, t.now + float(getattr(t.ctx.services, "max_mute_hours", 24.0)) * 3600)
    before = mute.snapshot()
    after = {"cameras": {camera: until}} if camera else {"all": until}
    detail = {"camera": camera or "", "until": clock(until), "before": before, "after": after}
    try:
        mute.apply(Feedback(action="mute", mute_until=until, camera=camera), t.now)
    except Exception as exc:  # noqa: BLE001
        log.warning("v3 pause failed: %s", exc)
        _issue(t.ctx, "pause_alerts", FAILED, camera or "all", detail, "error")
        return False
    _issue(t.ctx, "pause_alerts", DONE, camera or "all", detail)
    return True


def resume(t: Turn, camera: Optional[str]) -> bool:
    try:
        t.ctx.services.mute.resume(camera, t.now, cameras=t.snapshot.names)
    except Exception as exc:  # noqa: BLE001
        log.warning("v3 resume failed: %s", exc)
        _issue(t.ctx, "resume_alerts", FAILED, camera or "all", {"camera": camera or ""}, "error")
        return False
    _issue(t.ctx, "resume_alerts", DONE, camera or "all", {"camera": camera or ""})
    return True


def send_video(t: Turn, act: Act) -> None:
    entry = t.state.resolve(act.event) if act.event else None
    photo_handle, photo = t.last_photo()
    if isinstance(entry, dict) and entry.get("kind") == "event":
        out = t.box.run("send_clip", {"handle": act.event})
        if out.get("ok"):
            t.did(f"נשלח הסרטון של ההתראה מ-{hhmm(entry.get('ts'))} ב{t.name(t.camera_now(str(entry.get('camera') or '')))}.")
        return
    camera = act.camera if act.camera and act.camera != "house" else (t.last_photo_camera() or "")
    if past_photo.asks_past(t.text):
        send_past_video(t, _named_camera(t, t.text) or camera or t.camera_of(act))   # a recording, not live seconds
        return
    if camera:
        out = t.box.run("record_clip", {"camera": camera})
        if any(r.tool == "record_clip" and r.status == DONE for r in t.ctx.receipts):
            what = photo.get("observation") if photo and t.camera_now(str(photo.get("camera") or "")) == camera else ""
            t.did(f"נשלחו 10 שניות חיות מ{t.name(camera)} עכשיו." + (f" (בתמונה הקודמת משם: {what})" if what else ""))
        else:
            t.plan.trace.append(f"record failed: {out}")
        return
    handle, entry = t.event_of(act)
    if handle:
        out = t.box.run("send_clip", {"handle": handle})
        if out.get("ok"):
            t.did(f"נשלח הסרטון של ההתראה מ-{hhmm(entry.get('ts'))} ב{t.name(t.camera_now(str(entry.get('camera') or '')))}.")


def watch_request(t: Turn, act: Act) -> None:
    """"ברגע שהיא יוצאת תן לי דיווח" (2026-10-11): a one-shot watch on the vehicles at the camera (watches.py), checked
    every minute by the box's own detector; one photo when it happens. The baseline is the vehicles on a picture
    taken now."""
    from .. import watches  # noqa: PLC0415

    store = getattr(t.ctx.services, "watches", None) or watches.current()
    change = act.change or "leaves"
    car = vehicle_words(act.subject, t.text) if change == "arrives" else \
        vehicle_words(act.subject, t.text, *t.mem_owner_words())
    if not car and store is not None:
        return _event_watch(t, act, store)
    camera = t.camera_of(act)
    if not camera:
        t.plan.notes.append(load("assistant_v3_note_watch_no_camera.prompt"))
        return
    name = t.name(camera)
    if store is None:
        t.plan.trace.append("watch: watches are off")
        t.plan.notes.append(render("assistant_v3_note_watch_off.prompt", camera=name))
        return
    seen = store.look(camera)
    if seen is None:
        t.plan.trace.append("watch: no picture")
        t.plan.notes.append(render("assistant_v3_note_watch_no_picture.prompt", camera=name))
        return
    if change == "leaves" and not seen["vehicles"]:
        t.plan.notes.append(render("assistant_v3_note_watch_no_vehicle.prompt", camera=name))
        return
    until = parse_until(act.until_quote, t.now) if act.until_quote else None
    subject = vehicle_words(act.subject, t.text, *t.mem_owner_words()) if len(seen["vehicles"]) <= 1 else ""
    row = store.add(t.ctx.chat_id, camera, name, change, subject, seen["vehicles"], until)
    _issue(t.ctx, "watch", DONE, camera, {"camera": camera, "change": change, "subject": subject,
                                          "until": clock(row["until"]), "watch_id": row["id"]})
    what = subject or ("הרכב" if len(seen["vehicles"]) == 1 else f"אחד מ-{len(seen['vehicles'])} הרכבים שעומדים שם")
    verb = "יוצא מ" if change == "leaves" else "מגיע ל"
    if change == "arrives":
        what = vehicle_words(act.subject, t.text) or "רכב"
    t.did(f"אשלח לך תמונה ברגע ש{what} {verb}{name}.")
    t.plan.notes.append(load("assistant_v3_note_watch_set.prompt"))


def _event_watch(t: Turn, act: Act, store: Any) -> None:
    """"אם מגיע שליח לקחת את החבילות תעדכן אותי" (2026-10-11 13:00): no vehicle in his words, so not a car watch on
    the camera of the alert he answered (it watched camera 8 for a car while the courier's van stood at the main
    entrance). Every event the box records, at the camera he names or anywhere, is put to the judge with his words
    (watches.on_event); the first match sends its picture."""
    from ..watches import EVENT  # noqa: PLC0415

    camera = _named_camera(t, t.text)
    about = t.text.strip()
    if act.subject and act.subject not in about:
        about = f"{about} ({act.subject})"
    until = parse_until(act.until_quote, t.now) if act.until_quote else None
    row = store.add(t.ctx.chat_id, camera, t.name(camera) if camera else "", EVENT, act.subject or "", [],
                    until, about=about)
    _issue(t.ctx, "watch", DONE, camera, {"camera": camera or "house", "change": "event", "about": about[:120],
                                          "until": clock(row["until"]), "watch_id": row["id"]})
    where = f"ב{t.name(camera)}" if camera else ", בכל מצלמה שהיא (לא רק במצלמה של ההודעה)"
    t.did(f"אשלח לך תמונה ברגע שאראה {act.subject or 'את מה שביקשת'}{' ' if camera else ''}{where}.")
    t.plan.notes.append(load("assistant_v3_note_watch_set.prompt"))


_VEHICLE = re.compile(r"(?<![א-ת])(?:[ובכלמש]{1,3})?[ה]?(רכב|אוטו|מכונית|טנדר|משאית|ג'יפ|ג׳יפ|אופנוע)(?![א-ת])(?:\s+[ה]?(לבן|לבנה|שחור|שחורה|"
                      r"כסוף|כסופה|אפור|אפורה|אדום|אדומה|כחול|כחולה|ירוק|ירוקה|צהוב|צהובה))?")


def vehicle_words(*texts: str) -> str:
    """"הרכב הלבן" from the first of *texts* that names a vehicle (his words: "יש רכב לבן שם"); "" when none does.
    The understanding's subject can be a person ("אשת בעל הבית", 2026-10-11) while what is watched is her car."""
    for text in texts:
        m = _VEHICLE.search(str(text or ""))
        if m:
            return f"ה{m.group(1)}" + (f" ה{m.group(2)}" if m.group(2) else "")
    return ""


DAY_EVENTS = 8


def day_events(t: Turn, camera: str = "", hours: float = 12.0) -> List[str]:
    """The box's events of the last *hours* (at *camera*, else every camera), oldest first, each one line with whether
    the owner was told: the event memory's archive and the sessions still open. [] when there is no memory."""
    from ..brain.tools import _memory, memory_event_text  # noqa: PLC0415

    memory = _memory(t.ctx)
    if memory is None:
        return []
    since = t.now - hours * 3600.0
    rows: List[Dict[str, Any]] = []
    try:
        rows = [r for r in memory.records() if float(r.get("end") or r.get("start") or 0) >= since]
        book = getattr(t.ctx.services, "events", None)
        if book is not None:
            rows += [memory.live_record(s) for s in book.recent(since, camera) if not s.get("closed")]
    except Exception as exc:  # noqa: BLE001 - the chat's own alerts still answer
        log.warning("v3: event memory not read: %s", exc)
        return []
    if camera:
        rows = [r for r in rows if t.camera_now(str(r.get("camera") or "")) == camera]
    rows = sorted({str(r.get("event_id") or id(r)): r for r in rows}.values(), key=lambda r: float(r.get("start") or 0))
    out = []
    for r in rows[-DAY_EVENTS:]:
        e = memory_event_text(r, t.snapshot, t.ctx.lang, t.now)
        told = "נשלח לך" if e["told_the_owner"] else "לא נשלח לך (נשמר ביומן)"
        out.append(f"מהיומן של הקופסה: {e['from']}–{e['to']} ב{e['camera']} ({told}): {cx_clip(e['what'], 260)}")
    if out:
        t.plan.notes.append(load("assistant_v3_note_day_events.prompt"))
    return out


def cx_clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rsplit(" ", 1)[0] + "…"


HANDLERS = {"place_fact": place_fact, "person_mark": person_mark, "activity_explain": activity_explain,
            "camera_fact": camera_fact, "tag_only": tag_only, "alert_feedback": alert_feedback,
            "preference": preference, "command": command, "watch_request": watch_request}


# ---------------------------------------------------------------------------------------------------------------------
def prefetch(t: Turn) -> None:
    """The obvious first look, run in code before the writer (one round trip saved): a live question looks at the
    camera named (or every camera); a disputed clip is looked at again."""
    live = t.und.first("question_live")
    if live is not None:
        entry = t.state.resolve(live.event) if live.event else None
        fresh_photo = isinstance(entry, dict) and entry.get("kind") == "photo" and \
            t.now - float(entry.get("ts") or 0) <= 5 * 60
        if not fresh_photo:
            cam = live.camera if live.camera and live.camera != "house" else ""
            t.box.run("look_now", {"camera": cam or "house"})
        else:
            # "?" right after a photo: the answer is what that photo showed (one look, kept on its handle)
            t.plan.evidence.append(f"התמונה החיה מ-{hhmm(entry.get('ts'))} מ{t.name(t.camera_now(str(entry.get('camera') or '')))}"
                                   f" (נשלחה לפני רגע): {entry.get('observation') or 'בלי תיאור'}")
            t.plan.notes.append("ענה מהתמונה הזאת: מה רואים בה ואיפה. אל תגיד שלא בדקת.")
    if t.und.has("question_history") or any(a.act == "complaint" and a.issue in ("ignored_memory", "wrong_fact", "other")
                                            for a in t.und.acts):
        # 2026-10-11 09:20 "האם היא עזבה? זה מתועד אצלך?" / "בכניסה הראשית אין תיעוד כזה?": the answer came only from
        # the alerts sent in the chat, while the box's event memory held 08:15 "a woman walks out holding a child
        # ... towards a parked white car" (normal: kept, not sent). What happened is read from the event memory.
        history = t.und.first("question_history", "complaint")
        cam = history.camera if history and history.camera and history.camera != "house" else ""
        for line in day_events(t, cam):
            t.plan.evidence.append(line)
    if t.und.has("question_history") and not any(a.look_again for a in t.und.acts):
        rows = t.events(12 * 3600)[:4]
        for handle, e in rows:
            t.plan.evidence.append(f"התראה {hhmm(e.get('ts'))} ב{t.name(t.camera_now(str(e.get('camera') or '')))}: "
                                   f"{(e.get('observation') or e.get('summary') or '')[:200]}")
        if rows:
            t.plan.notes.append("ענה מההתראות האלה (שעה, מצלמה, מה נראה); אל תגיד שאין לך ראיות.")
    if any(a.act == "complaint" and a.issue == "no_explanation" for a in t.und.acts) or \
            (live is not None and not live.camera):
        photos = sorted([e for e in (t.state.handles or {}).values() if isinstance(e, dict) and e.get("kind") == "photo"
                         and 0 <= t.now - float(e.get("ts") or 0) <= 15 * 60], key=lambda e: float(e.get("ts") or 0))
        if photos and t.und.has("complaint"):
            for e in photos[-3:]:
                t.plan.evidence.append(f"תמונה ששלחת ב-{hhmm(e.get('ts'))} מ{t.name(t.camera_now(str(e.get('camera') or '')))}: "
                                       f"{e.get('observation') or 'בלי תיאור'}")
            t.plan.notes.append("הסבר עכשיו מה רואים בתמונות האלה, מצלמה מצלמה, במשפט או שניים.")
    for act in t.und.acts:
        if act.look_again:
            handle, entry = t.event_of(act)
            if handle:
                t.box.run("look_again", {"handle": handle, "question": act.quote or t.text})
