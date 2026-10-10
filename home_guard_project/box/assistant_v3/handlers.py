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
from .memory_view import MemoryView
from .skills import Toolbox, internal
from .timeparse import clock, parse_until

log = logging.getLogger("box.assistant_v3.handlers")

RECENT_EVENT_SEC = 3 * 3600.0
PHOTO_FRESH_SEC = 15 * 60.0


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
        rows = [(h, e) for h, e in (self.state.handles or {}).items() if isinstance(e, dict)
                and e.get("kind") == "photo" and 0 <= self.now - float(e.get("ts") or 0) <= within]
        return max(rows, key=lambda kv: float(kv[1].get("ts") or 0)) if rows else ("", {})

    def last_photo_camera(self, within: float = PHOTO_FRESH_SEC) -> str:
        _, e = self.last_photo(within)
        return self.camera_now(str(e.get("camera") or "")) if e else ""

    def name(self, camera: str) -> str:
        return self.cams.name(camera) if camera else "כל הבית"

    def outcome(self, entry: Dict[str, Any], what: str) -> None:
        if isinstance(entry, dict) and entry:
            entry["v3_outcome"] = what[:160]

    def did(self, line: str) -> None:
        self.plan.done.append(line)


# ---------------------------------------------------------------------------------------------------------------------
def place_fact(t: Turn, act: Act) -> None:
    from .. import place_facts  # noqa: PLC0415

    if _crew(act.subject or act.quote) and not re.search(r"בית|חצר|חניה|שטח|מגרש|גינה|רחוב|בניין", act.subject or act.quote):
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
        t.did(f"זה כבר שמור אצלי לתמיד: ב{where} {fact['text']}.")
    else:
        t.did(f"שמרתי לתמיד (מקום, בלי תאריך): ב{where} — {fact['text']}. מה שקורה שם לא יקפיץ התראה, רק מי "
              f"שעובר לשטח של הבית.")
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
        k = max(same, key=lambda x: float(x.get("until") or 0))
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
        quiet = act.repaired or t.und.emotion == "angry"
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
        t.plan.notes.append(f"עוד לא נשמר כלום (חסרה שעה): כתוב הד קצר כמו 'אה, אלה {who}?' ואז את השאלה; אל "
                            f"תגיד שסימנת או שמרת.")
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
    line = f"נשמר: {who} {where} {when}."
    if daily_from and not daily_old:
        line += f" הנחתי שהעבודה נמשכת כמה ימים (כל יום {daily_from}–{daily_to}, שבוע); יש כפתור 'רק היום'."
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
    t.did(f"נשמר: {am.actions_text(fact.actions, 'he')} {where} = {fact.cause}. זה ייחשב רגיל {when}"
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
    t.did(f"{'זה כבר ידוע לי' if fact.get('already') else 'רשמתי לתמיד'} {where}: {fact['text']}.")


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


def preference(t: Turn, act: Act) -> None:
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
        camera = t.camera_of(act)
        if camera:
            out = t.box.run("look_now", {"camera": camera})
            if out.get("ok", True):
                t.did(f"נשלחה תמונה חיה מ{t.name(camera)}.")
    elif kind in ("house_mode", "setting"):
        from ..brain import house  # noqa: PLC0415

        services = t.ctx.services
        if services.house is not None:
            schedule = house.schedule_of(services.house, t.now) or house.DEFAULT_SCHEDULE
            out = house.run_command(t.ctx, house.parse_command(t.text, t.now, schedule))
            if out.handled and out.text:
                t.plan.final = out.text
                t.plan.rows.extend(out.rows or ())


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


HANDLERS = {"place_fact": place_fact, "person_mark": person_mark, "activity_explain": activity_explain,
            "camera_fact": camera_fact, "tag_only": tag_only, "alert_feedback": alert_feedback,
            "preference": preference, "command": command}


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
    for act in t.und.acts:
        if act.look_again:
            handle, entry = t.event_of(act)
            if handle:
                t.box.run("look_again", {"handle": handle, "question": act.quote or t.text})
