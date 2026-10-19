"""Day activity threads: by day an ordinary activity on our ground is ONE neutral message, then only its changes.

Why (owner, 2026-10-19): the box sent 8 "🟡 חשוד" by day, all about workers (a ladder, a cap, their own car's trunk).
He said by day don't call it suspicious, say what they do, and:

> "send natural message but don't continue sending it if nothing has changed, meaning don't send me every hour the
> worker outside if they are still outside"
> "you send and then you ask me (once a day for each unique activity) if you want me to keep updating you. if no then
> you keep in log. if yes, if let's say the workers left then update, the workers came back update, etc. if a totally
> different thing happens then you update"

Scope (the caller, inference._day_activity_look, checks it): daylight, the Eye said "normal", people in view and the
camera alerts on people, not off our ground, not muted. Night and every suspicious / escalation are untouched.

- **Thread** = one activity of today ("העובדים בפרגולה"), across cameras. A new daytime normal event joins a thread
  still present (seen within ``PRESENT_SEC``) on the same camera or a box.yaml ``camera_neighbours`` camera, with no
  model call; otherwise, when today has threads, the cheap text model (the v3 "understand" model) says whether it is
  the same activity or a totally different thing, and names a new one. A failed or missing model: the same thread
  when one was at this camera today, else a new one.
- **New thread** -> one message (the photo, "⚪ פעילות · <camera> · HH:MM", what happens, why) with the question
  "להמשיך לעדכן אותך על זה היום?" and two buttons. Asked once per day per thread.
- **follow** (his answer): None (no answer) or False -> nothing more today, the events stay in the journal as always.
  True -> only changes: they LEFT (no person on the thread's cameras for ``LEFT_SEC``, by the detector's looks), they
  CAME BACK (a left thread seen again), or clearly MORE people (``MORE_PEOPLE`` above what he was told).
- The owner's "these are my workers" (events.Known) covering the camera: the thread is follow=False (log only); the
  existing "they arrived" line is left as it is.

Code decides; the models never decide whether to send. One JSON file per box (the registry folder), rewritten with an
atomic replace + fsync, holding today only (a new day starts empty).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

log = logging.getLogger("box.day_activity")

PRESENT_SEC = 15 * 60.0        # a thread seen this recently is still here: a new event near it joins without a model
LEFT_SEC = 20 * 60.0           # no person on its cameras this long: the activity left
MORE_PEOPLE = 2                # this many more people than he was told about is a change worth one update
MODEL_CALLS_PER_DAY = 40       # then the rules alone (each call is a fraction of a cent; owner budget $10/box/month)
MODEL_TIMEOUT_SEC = 12.0
SAVE_EVERY_SEC = 60.0          # detector looks only move last_seen; written at most this often
KEEP_EVENTS = 60               # alert ids kept per thread
OWNER_WORDS_MAX = 8            # "תפסיק לעדכן על העובדים" is short; a longer message goes to the assistant as usual

ASK, CAME_BACK, MORE, LEFT, QUIET = "ask", "came_back", "more_people", "left", "quiet"

_TEXTS = {
    "why_one": {"he": "אדם בשטח שלנו ביום", "en": "a person on our ground by day"},
    "why_many": {"he": "אנשים בשטח שלנו ביום", "en": "people on our ground by day"},
    "why_one_seen": {"he": "אדם נראה במצלמה ביום", "en": "a person seen by the camera by day"},
    "why_many_seen": {"he": "אנשים נראו במצלמה ביום", "en": "people seen by the camera by day"},
    "question": {"he": "להמשיך לעדכן אותך על זה היום?", "en": "Keep updating you about this today?"},
    "btn_yes": {"he": "🔔 כן, רק כשמשהו משתנה", "en": "🔔 Yes, only when something changes"},
    "btn_no": {"he": "🔕 לא, רק ביומן", "en": "🔕 No, just log it"},
    "receipt_yes": {"he": "סבבה, אעדכן רק כשהם יוצאים או חוזרים.",
                    "en": "OK, I'll only tell you when they leave or come back."},
    "receipt_no": {"he": "סבבה, היום רק ביומן.", "en": "OK, just the log today."},
    "other_day": {"he": "זה כבר לא מהיום.", "en": "That one is not from today."},
    "left": {"he": "⚪ {name} {verb} · {camera} · {clock} ({seen} לאחרונה ב-{last})",
             "en": "⚪ {name} left · {camera} · {clock} (last seen {last})"},
    "back": {"he": "⚪ {name} {verb} · {camera} · {clock}", "en": "⚪ {name} came back · {camera} · {clock}"},
    "more": {"he": "⚪ {name}: {more} · {camera} · {clock}", "en": "⚪ {name}: {more} · {camera} · {clock}"},
    "scene": {"he": "מה קורה: {text}", "en": "What's happening: {text}"},
    "one_person": {"he": "אדם ב{place}", "en": "a person at {place}"},
    "people_at": {"he": "אנשים ב{place}", "en": "people at {place}"},
}


def _t(key: str, lang: str, **values: Any) -> str:
    row = _TEXTS[key]
    return row["he" if str(lang or "").startswith("he") else "en"].format(**values)


def _clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M")


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _dark(ts: float) -> bool:
    from .situation import phase_of  # noqa: PLC0415

    return phase_of(ts)[1]


@dataclass
class Thread:
    id: str
    day: str
    started: float
    last_seen: float
    name_he: str
    cameras: List[str] = field(default_factory=list)
    last_camera: str = ""
    present: bool = True
    plural: bool = False
    first_alert_id: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)   # {chat_id, message_id, ts, kind}
    follow: Optional[bool] = None
    follow_by: str = ""
    asked_at: float = 0.0
    people_told: int = 0
    people_max: int = 0
    left_at: float = 0.0          # when it was last seen before it was marked gone
    left_told_at: float = 0.0
    came_back_at: float = 0.0
    known: str = ""               # the owner's mark that covered it ("these are my workers")
    summary: str = ""             # the latest observation (English, the Eye's)
    events: List[str] = field(default_factory=list)
    matched_by: str = ""          # how its latest event joined: new | present | model | fallback

    def first_message(self) -> Optional[Dict[str, Any]]:
        return self.messages[0] if self.messages else None


@dataclass
class Step:
    """What one daytime normal event means for the owner: ``action`` ask | came_back | more_people | quiet."""

    action: str
    thread: Dict[str, Any]
    reason: str
    new: bool = False

    @property
    def sends(self) -> bool:
        return self.action != QUIET

    def record(self) -> Dict[str, Any]:
        t = self.thread
        return {"action": self.action, "reason": self.reason, "new": self.new, "thread": t.get("id"),
                "name": t.get("name_he"), "follow": t.get("follow"), "matched_by": t.get("matched_by")}


def neighbours_of(raw: Any) -> Dict[str, Set[str]]:
    from .events import neighbours_of as both_ways  # noqa: PLC0415

    return both_ways(raw)


class DayActivity:
    """Today's activity threads. *judge(threads_text, camera_name, clock, people, summary)* -> ``{"same": id | "",
    "name": str, "plural": bool}`` or None (the cheap text model; None = rules only). *names(camera)* -> the name the
    owner reads (never an id). *is_dark(ts)* -> True at night."""

    def __init__(self, path: str, neighbours: Any = None, judge: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
                 names: Optional[Callable[[str], str]] = None, lang: str = "he",
                 is_dark: Callable[[float], bool] = _dark, now: Callable[[], float] = time.time,
                 present_sec: float = PRESENT_SEC, left_sec: float = LEFT_SEC) -> None:
        self.path = path
        self.neighbours = neighbours_of(neighbours or {})
        self.judge = judge
        self.names = names or (lambda c: c)
        self.lang = lang
        self.is_dark = is_dark
        self.now = now
        self.present_sec = float(present_sec)
        self.left_sec = float(left_sec)
        self._lock = threading.RLock()
        self.day = ""
        self.threads: List[Thread] = []
        self.model_calls = 0
        self._dirty = False
        self._saved_at = 0.0
        self._load()

    # ---------- the file ----------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        today = _day(self.now())
        if not isinstance(data, dict) or data.get("day") != today:
            self.day, self.threads, self.model_calls = today, [], 0
            return
        fields = set(Thread.__dataclass_fields__)
        rows = [r for r in data.get("threads") or [] if isinstance(r, dict)]
        try:
            self.threads = [Thread(**{k: v for k, v in r.items() if k in fields}) for r in rows]
        except TypeError:
            self.threads = []
        self.day, self.model_calls = today, _int(data.get("model_calls"))

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"day": self.day, "model_calls": self.model_calls,
                           "threads": [asdict(t) for t in self.threads]}, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._dirty, self._saved_at = False, float(self.now())
        except OSError as exc:
            log.warning("day activity not saved: %s", exc)

    def _roll(self, ts: float) -> None:
        """A new day starts empty (yesterday's threads, asked or not, are over)."""
        day = _day(ts)
        if day > self.day:
            if self.threads:
                log.info("day activity: new day %s; %d thread(s) of %s closed", day, len(self.threads), self.day)
            self.day, self.threads, self.model_calls = day, [], 0
            self._save()

    # ---------- reading ----------
    def find(self, thread_id: str) -> Optional[Thread]:
        with self._lock:
            return next((t for t in self.threads if t.id == thread_id), None)

    def today(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._roll(float(self.now()))
            return [asdict(t) for t in self.threads]

    # ---------- the detector's looks ----------
    def seen(self, camera: str, ts: float, people: int) -> None:
        """One detector look with *people* at *camera*: every thread still here on that camera stays here."""
        if not people:
            return
        with self._lock:
            for t in self.threads:
                if t.present and camera in t.cameras and ts > t.last_seen:
                    t.last_seen, t.last_camera = float(ts), camera
                    self._dirty = True

    def tick(self, now: float) -> List[Dict[str, Any]]:
        """Mark threads not seen for ``LEFT_SEC`` as gone. Returns the "they left" notices to send (only for a
        thread he follows, by day, that was announced): ``{"kind", "thread", "text", "reply_to"}``."""
        out: List[Dict[str, Any]] = []
        with self._lock:
            self._roll(now)
            for t in self.threads:
                if not t.present or now - t.last_seen <= self.left_sec:
                    continue
                t.present, t.left_at, self._dirty = False, t.last_seen, True
                if t.follow is True and t.messages and not self.is_dark(now):
                    out.append({"kind": LEFT, "thread": asdict(t), "text": self.left_text(t, now),
                                "reply_to": t.first_message()})
                    log.info("day activity %s (%s): left at %s; he follows it, a line goes out", t.id, t.name_he,
                             _clock(t.last_seen))
                else:
                    log.info("day activity %s (%s): left at %s; not sent (%s)", t.id, t.name_he, _clock(t.last_seen),
                             "night" if t.follow is True and t.messages else f"follow={t.follow}")
            if self._dirty and (out or now - self._saved_at >= SAVE_EVERY_SEC):
                self._save()
        return out

    # ---------- one daytime normal event ----------
    def look(self, camera: str, ts: float, people: int, summary: str, alert_id: str = "",
             known_text: str = "") -> Step:
        """A daytime "normal" with people at *camera*: which of today's threads it is, and what the owner gets."""
        people = max(1, _int(people))
        with self._lock:
            self._roll(ts)
            thread, how = self._near(camera, ts), "present"
            briefs = self._briefs(ts) if thread is None and self.threads else ""
            today_here = next((t for t in reversed(self.threads) if camera in t.cameras), None)
        name, plural = "", people > 1
        if thread is None:
            answer = self._ask_model(briefs, camera, ts, people, summary)
            with self._lock:
                if answer is not None and answer.get("same"):
                    thread, how = self.find(str(answer["same"])), "model"
                    if thread is None:
                        log.info("day activity: the model named an unknown thread %r; rules", answer.get("same"))
                if thread is None and answer is None and today_here is not None:
                    thread, how = today_here, "fallback"
                if answer is not None:
                    name = str(answer.get("name") or "").strip()
                    plural = bool(answer.get("plural")) if "plural" in answer else plural
        with self._lock:
            if thread is None:
                return self._new(camera, ts, people, summary, alert_id, known_text, name, plural, how="model"
                                 if name else "new")
            return self._join(thread, camera, ts, people, summary, alert_id, known_text, how)

    def _near(self, camera: str, ts: float) -> Optional[Thread]:
        around = {camera} | self.neighbours.get(camera, set())
        here = [t for t in self.threads if t.present and ts - t.last_seen <= self.present_sec
                and around & set(t.cameras)]
        return max(here, key=lambda t: t.last_seen) if here else None

    def _briefs(self, ts: float) -> str:
        rows = []
        for t in self.threads[-8:]:
            where = ", ".join(self.names(c) for c in t.cameras)
            state = "still here" if t.present and ts - t.last_seen <= self.left_sec else "gone"
            rows.append(f'- {t.id}: "{t.name_he}" (cameras: {where}; seen {_clock(t.started)}-{_clock(t.last_seen)}, '
                        f"{state}; up to {t.people_max} people; latest: {t.summary[:200]})")
        return "\n".join(rows)

    def _ask_model(self, briefs: str, camera: str, ts: float, people: int, summary: str) -> Optional[Dict[str, Any]]:
        if self.judge is None:
            return None
        with self._lock:
            if self.model_calls >= MODEL_CALLS_PER_DAY:
                log.info("day activity: %d model calls today; the rules decide", self.model_calls)
                return None
            self.model_calls += 1
            self._dirty = True
        got: Dict[str, Any] = {}

        def call() -> None:
            try:
                got["answer"] = self.judge(briefs or "(none yet today)", self.names(camera), _clock(ts), people,
                                           str(summary or "")[:400])
            except Exception as exc:  # noqa: BLE001 - the rules decide
                got["error"] = str(exc)

        worker = threading.Thread(target=call, name="day-activity-judge", daemon=True)
        worker.start()
        worker.join(MODEL_TIMEOUT_SEC)
        answer = got.get("answer")
        if worker.is_alive() or not isinstance(answer, dict):
            log.warning("day activity: no model answer (%s); the rules decide",
                        "timeout" if worker.is_alive() else got.get("error") or "no JSON")
            return None
        return answer

    def _fallback_name(self, camera: str, people: int) -> str:
        from .brain.tools import in_place  # noqa: PLC0415

        place = in_place(self.names(camera), self.lang)
        return _t("people_at" if people > 1 else "one_person", self.lang, place=place)

    def _new(self, camera: str, ts: float, people: int, summary: str, alert_id: str, known_text: str, name: str,
             plural: bool, how: str) -> Step:
        t = Thread(id=uuid.uuid4().hex[:8], day=self.day, started=float(ts), last_seen=float(ts),
                   name_he=(name or self._fallback_name(camera, people))[:60], cameras=[camera], last_camera=camera,
                   plural=bool(plural), first_alert_id=alert_id, people_max=people, summary=str(summary or "")[:400],
                   events=[alert_id] if alert_id else [], matched_by=how)
        self.threads.append(t)
        self._dirty = True
        if known_text:
            t.known, t.follow, t.follow_by = known_text[:200], False, "mark"
            self._save()
            log.info("day activity %s (%s) new at %s: the owner's mark %r covers it; log only", t.id, t.name_he,
                     camera, known_text)
            return Step(QUIET, asdict(t), f"the owner's mark covers it ({known_text})", new=True)
        self._save()
        log.info("day activity %s (%s) new at %s (%s): one message with the question", t.id, t.name_he, camera, how)
        return Step(ASK, asdict(t), "a new activity today: one message, and the question", new=True)

    def _join(self, t: Thread, camera: str, ts: float, people: int, summary: str, alert_id: str, known_text: str,
              how: str) -> Step:
        was_present = t.present
        t.present, t.matched_by = True, how
        t.last_seen, t.last_camera = max(t.last_seen, float(ts)), camera
        if camera not in t.cameras:
            t.cameras.append(camera)
        t.people_max = max(t.people_max, people)
        t.summary = str(summary or "")[:400] or t.summary
        if alert_id and alert_id not in t.events:
            t.events = (t.events + [alert_id])[-KEEP_EVENTS:]
        if known_text and not t.known:
            t.known = known_text[:200]
            if t.follow is None:
                t.follow, t.follow_by = False, "mark"
        self._dirty = True
        step: Step
        if t.follow is None and not t.asked_at:
            step = Step(ASK, asdict(t), "the question did not reach him yet: asked again")
        elif t.follow is not True:
            why = ("he said log only" if t.follow is False and t.follow_by != "mark" else
                   "the owner's mark covers it" if t.follow is False else "asked, no answer yet")
            step = Step(QUIET, asdict(t), f"same activity ({how}); {why}")
        elif not was_present:
            t.came_back_at = float(ts)
            step = Step(CAME_BACK, asdict(t), f"he follows it; it came back (gone since {_clock(t.left_at)})")
        elif people >= t.people_told + MORE_PEOPLE:
            step = Step(MORE, asdict(t), f"he follows it; {people} people, he was told about {t.people_told}")
        else:
            step = Step(QUIET, asdict(t), f"same activity ({how}); nothing changed")
        if not was_present or step.sends:
            self._save()
        log.info("day activity %s (%s) at %s: %s (%s)", t.id, t.name_he, camera, step.action, step.reason)
        return step

    # ---------- after a send ----------
    def sent(self, thread_id: str, action: str, people: int, ts: float, chat_id: Any = None,
             message_id: Any = None) -> None:
        """*action*'s message reached the owner (call only after a successful delivery)."""
        with self._lock:
            t = self.find(thread_id)
            if t is None:
                return
            if action == ASK:
                t.asked_at = float(ts)
            if action == LEFT:
                t.left_told_at = float(ts)
            if action in (ASK, CAME_BACK):
                t.people_told = _int(people)            # who is there now is what he knows
            elif action == MORE:
                t.people_told = max(t.people_told, _int(people))
            if message_id is not None:
                t.messages.append({"chat_id": str(chat_id), "message_id": int(message_id), "ts": float(ts),
                                   "kind": action})
            self._save()

    def set_follow(self, thread_id: str, follow: bool, by: str = "") -> Dict[str, Any]:
        """His answer to "keep updating you?" (a button or his words). ``found``: a thread of today; ``changed``: the
        answer differs from the one he gave before (the receipt is said only then)."""
        with self._lock:
            self._roll(float(self.now()))
            t = self.find(thread_id)
            if t is None:
                return {"found": False, "changed": False, "receipt": _t("other_day", self.lang)}
            changed = t.follow is not follow or t.follow_by == "mark"
            t.follow, t.follow_by = bool(follow), str(by or "owner")
            self._save()
            log.info("day activity %s (%s): follow=%s by %s", t.id, t.name_he, follow, by or "owner")
            return {"found": True, "changed": changed, "thread": asdict(t),
                    "receipt": _t("receipt_yes" if follow else "receipt_no", self.lang)}

    # ---------- the owner's texts ----------
    def _verb(self, t: Thread, kind: str) -> str:
        if not str(self.lang).startswith("he"):
            return ""
        return {"left": "יצאו" if t.plural else "יצא", "back": "חזרו" if t.plural else "חזר"}[kind]

    def left_text(self, t: Thread, now: float) -> str:
        return _t("left", self.lang, name=t.name_he, verb=self._verb(t, "left"), camera=self.names(t.last_camera),
                  clock=_clock(now), last=_clock(t.last_seen), seen="נראו" if t.plural else "נראה")

    def update_text(self, step: Step, camera: str, ts: float, scene: str, people: int) -> str:
        """The text of a came_back / more_people update: the line, then "מה קורה:"."""
        t = self.find(step.thread["id"]) or Thread(**step.thread)
        if step.action == CAME_BACK:
            head = _t("back", self.lang, name=t.name_he, verb=self._verb(t, "back"), camera=self.names(camera),
                      clock=_clock(ts))
        else:
            from .alert_texts import more_people  # noqa: PLC0415

            head = _t("more", self.lang, name=t.name_he, more=more_people(max(1, people - t.people_told), self.lang),
                      camera=self.names(camera), clock=_clock(ts))
        scene = " ".join(str(scene or "").split())
        return f"{head}\n{_t('scene', self.lang, text=scene)}" if scene else head

    def why_line(self, people: int, ours: bool) -> str:
        key = ("why_many" if people > 1 else "why_one") + ("" if ours else "_seen")
        return _t(key, self.lang)

    def question(self) -> str:
        return _t("question", self.lang)

    def buttons(self, thread_id: str, chosen: Optional[bool] = None) -> Dict[str, Any]:
        """The yes / no buttons under the first message; after an answer the chosen one carries a ✓ (the other one
        stays: tapping it changes his mind)."""
        yes, no = _t("btn_yes", self.lang), _t("btn_no", self.lang)
        if chosen is True:
            yes = f"✓ {yes}"
        elif chosen is False:
            no = f"✓ {no}"
        return {"inline_keyboard": [[{"text": yes, "callback_data": f"da:y:{thread_id}"}],
                                    [{"text": no, "callback_data": f"da:n:{thread_id}"}]]}

    # ---------- his words in the chat ----------
    def owner_words(self, text: str, alert: Optional[Mapping[str, Any]] = None, by: str = "") -> Optional[str]:
        """"תפסיק לעדכן על העובדים" / "תעדכן אותי על העובדים": the receipt when his words clearly answer for one or
        more of today's threads (a reply to its message, its name's words, or the only thread today), else None (the
        assistant handles the message as usual). Never a "tell me WHEN ..." watch request ("כש", "ברגע ש")."""
        words = " ".join(str(text or "").split())
        follow = wants_follow(words)
        if follow is None or len(words.split()) > OWNER_WORDS_MAX:
            return None             # a longer message says more (a mark, a time): the assistant reads all of it
        with self._lock:
            self._roll(float(self.now()))
            if not self.threads:
                return None
            picked: List[Thread] = []
            tid = str((alert or {}).get("day_thread") or "")
            if tid:
                picked = [t for t in self.threads if t.id == tid]
            if not picked:
                said = _stems(words)
                picked = [t for t in self.threads if _stems(t.name_he) & said]
            if not picked and len(self.threads) == 1 and (alert or {}).get("day_thread") is None and _about_them(words):
                picked = list(self.threads)
            if not picked:
                return None
        receipt = ""
        for t in picked:
            receipt = self.set_follow(t.id, follow, by or "owner words").get("receipt") or receipt
        return receipt or None


_STOP = re.compile(r"(תפסיק|תפסיקו|הפסק|די|אל|לא צריך|בלי)\s+(ל|את ה|עם ה)?(עדכן|לעדכן|תעדכן|עדכונים|העדכונים|"
                   r"להודיע|תודיע|הודעות|ההודעות)")
_START = re.compile(r"(תמשיך|תמשיכו|כן)\s+(ל|את ה)?(עדכן|לעדכן|תעדכן|עדכונים|להודיע)|תעדכן אותי על|תעדכן על")
_WHEN = re.compile(r"(^|\s)(כש|כאשר|ברגע ש|אם )")
_THEM = re.compile(r"(עליהם|עליו|עליה|על זה|על העובדים|על הפועלים)")
_FILLER = {"של", "על", "את", "עם", "ליד", "אדם", "אנשים", "אותי", "לעדכן", "תעדכן", "תפסיק", "עדכונים"}


def wants_follow(words: str) -> Optional[bool]:
    """True: "keep telling me", False: "stop telling me", None: neither (or a "tell me WHEN" request)."""
    if _STOP.search(words):
        return False
    if _START.search(words) and not _WHEN.search(words):
        return True
    return None


def _about_them(words: str) -> bool:
    return bool(_THEM.search(words))


def _stems(text: str) -> Set[str]:
    """Content words without the Hebrew prefixes ה / ב / ל / ו ("העובדים בפרגולה" -> {"עובדים", "פרגולה"})."""
    out: Set[str] = set()
    for w in re.findall(r"[\w֐-׿]+", str(text or "")):
        if w in _FILLER:
            continue
        while len(w) > 3 and w[0] in "הבלו":
            w = w[1:]
        if len(w) >= 3 and w not in _FILLER:
            out.add(w)
    return out


# ---------- the model ----------
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["same", "name", "plural"],
                "properties": {"same": {"type": "string"}, "name": {"type": "string"}, "plural": {"type": "boolean"}}}


def model_judge(model: Any) -> Callable[..., Optional[Dict[str, Any]]]:
    """The judge on a chat model (the v3 understanding model): ``{"same", "name", "plural"}``, or None on any failure
    (then the rules decide)."""
    from ..prompts import render  # noqa: PLC0415

    def judge(threads: str, camera: str, clock: str, people: int, summary: str) -> Optional[Dict[str, Any]]:
        msg = model.chat([{"role": "user", "content": render("day_activity_match.prompt", threads=threads,
                                                             camera=camera, clock=clock, people=people,
                                                             summary=summary)}],
                         None, json_schema=JUDGE_SCHEMA, max_tokens=120)
        if getattr(msg, "error", ""):
            log.warning("day activity judge failed: %s", msg.error)
            return None
        text = str(getattr(msg, "content", "") or "").strip()
        try:
            out = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except ValueError:
            log.warning("day activity judge answered no JSON: %r", text[:200])
            return None
        if not isinstance(out, dict):
            return None
        return {"same": str(out.get("same") or "").strip(), "name": str(out.get("name") or "").strip()[:60],
                "plural": out.get("plural") is True}

    return judge


# ---------- the box's one store ----------
_current: Optional[DayActivity] = None


def current() -> Optional[DayActivity]:
    return _current


def set_current(store: Optional[DayActivity]) -> None:
    global _current
    _current = store


def on_of(box_settings: Mapping[str, Any]) -> bool:
    """box.yaml ``day_activity`` (default on: the owner asked for it, 2026-10-19)."""
    value = (box_settings or {}).get("day_activity", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("off", "false", "no", "0")


def minutes_of(box_settings: Mapping[str, Any], key: str, default_sec: float) -> float:
    try:
        value = float((box_settings or {}).get(key)) * 60.0
    except (TypeError, ValueError):
        return default_sec
    return value if 60.0 <= value <= 6 * 3600.0 else default_sec


def default_path() -> str:
    from . import paths  # noqa: PLC0415

    return os.path.join(paths.state_dir(), "day_activity.json")


def build(box_settings: Mapping[str, Any], env: Mapping[str, str], names: Callable[[str], str], lang: str,
          path: Optional[str] = None) -> Optional[DayActivity]:
    """The box's store, or None with ``day_activity: off``. The model: box.yaml ``day_activity_model`` (default the
    v3 understand model; "off" = the rules alone); usage is logged as ``day_activity``."""
    if not on_of(box_settings):
        log.info("Day activity threads off (day_activity: off)")
        set_current(None)
        return None
    judge = None
    spec = str((box_settings or {}).get("day_activity_model") or (box_settings or {}).get("v3_understand_model")
               or "qwen/qwen3.7-flash").strip()
    if spec.lower() != "off":
        try:
            from .assistant_v3.llm import make_v3_model  # noqa: PLC0415

            model = make_v3_model(spec, dict(env or {}), usage_agent="day_activity", temperature=0.0)
            judge = model_judge(model) if model is not None else None
        except Exception as exc:  # noqa: BLE001 - the rules alone
            log.warning("Day activity model not made (%s); the rules decide", exc)
    store = DayActivity(path or default_path(), neighbours=(box_settings or {}).get("camera_neighbours") or {},
                        judge=judge, names=names, lang=lang,
                        present_sec=minutes_of(box_settings, "day_activity_present_min", PRESENT_SEC),
                        left_sec=minutes_of(box_settings, "day_activity_left_min", LEFT_SEC))
    set_current(store)
    log.info("Day activity threads on (model %s; present %d min, left %d min; %d thread(s) today)",
             spec if judge is not None else "off", store.present_sec // 60, store.left_sec // 60, len(store.threads))
    return store
