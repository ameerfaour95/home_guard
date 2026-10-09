"""What an ACTION means at a camera, learned from the owner's own words (2026-10-09).

13:25 a red at the main entrance: "a man lies on the ground while another crawls nearby holding a long metal
bar". 13:54 the owner, in plain text: "אלה מקרים תקינים ... שני אנשים שמרכיבים את הלדים במדרגות", "אחד מהם
התכופף זה רגיל". Then 14:03, 14:41 and 15:44 the same lying-down went out red again. The owner: "MAP THE ACTIONS -
the lying down is the work outside, so the next such action of the same people in the same place is not a
problem."

An :class:`ActivityFact` is that map: the camera(s), the place (optional: stairs, a door...), the actions
(normalised verbs: lying, kneeling, bending, crawling, holding a tool...), the cause in the owner's words
("החשמלאים שמתקינים לדים"), who (a live known mark, when it is those people), the window (a daily window and an
end), by, at, and cancel (the [↩ ביטול] under the confirmation).

- CREATED by the assistant (brain/activity_chat.py) when the owner answers an alert in plain text - never from the
  "🏷️ תיוג אחר" words, which are a tag for the detection model only.
- APPLIED by the guard loop (inference.py, two small hooks) for a new alert on that camera, inside the window,
  whose why + summary names one of the fact's actions (and no other way-in place than the fact's):
  1. a suspicious is kept, not sent ("owner explained: <cause>"), like a known mark;
  2. a red gets a second look WITH the owner's context (:func:`context_question`), and only when the red names
     nothing clear or dangerous (:func:`red_blocked`: a CLEAR phrase, a weapon, violence, a break-in or theft,
     harm, a car door - owner decision 2026-10-09: a car-door red always stays red). Consistent with the work:
     lowered to suspicious, and so not sent. Unsure, no answer, or an answer that names harm: the red stays.
  3. The Eye's prompt is not changed.

Stored as JSON in the state dir (``activity.json`` next to ``events/``). Never raises into the guard loop.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence

log = logging.getLogger("box.activity_memory")

_FLAGS = re.IGNORECASE


def _any(patterns: Sequence[str]) -> "re.Pattern[str]":
    return re.compile("|".join(f"(?:{p})" for p in patterns), _FLAGS)


# ---------- the actions: (Hebrew noun with ה, Hebrew noun, English, patterns) ----------
ACTIONS: Dict[str, Dict[str, Any]] = {
    "lying": {"he": "השכיבה", "short": "שכיבה", "en": "lying on the ground", "re": _any([
        r"\bl(?:ie|ies|ying|ay|ays|aying|ain)\b", r"\b(?:person|man|woman|someone|worker|he|she|people|men|workers|they)"
        r"\b (?:\w+ ){0,3}?on the (?:ground|floor|pavement)\b",
        r"שוכב", r"שוכבת", r"שוכבים", r"(?<![א-ת])שכב", r"שכיבה",
        r"(?:אדם|גבר|אישה|מישהו|פועל|עובד|אנשים|פועלים|הוא|היא|איש)(?: \S+){0,2} על (?:הקרקע|הרצפה|האדמה|המדרכה)"])},
    "kneeling": {"he": "הכריעה", "short": "כריעה", "en": "kneeling or crouching", "re": _any([
        r"\bkneel", r"\bknelt\b", r"\bcrouch", r"\bsquat", r"\bon (?:his|her|their|all) (?:knees|fours)\b",
        r"כורע", r"כורעת", r"כורעים", r"כריעה", r"על הברכיים", r"על ברכיו", r"(?<![א-ת])כרע"])},
    "bending": {"he": "הכיפוף", "short": "כיפוף", "en": "bending down", "re": _any([
        r"\bbend", r"\bbent\b", r"\bstoop", r"\blean(?:s|ed|ing)? (?:down|over|in|inside|into)\b",
        r"התכופף", r"מתכופף", r"מתכופפת", r"מתכופפים", r"כיפוף", r"התכופפות", r"(?<![א-ת])רכון", r"רכונה", r"רכונים",
        r"נשען פנימה", r"מתכופף פנימה"])},
    "crawling": {"he": "הזחילה", "short": "זחילה", "en": "crawling", "re": _any([
        r"\bcrawl", r"זוחל", r"זוחלת", r"זוחלים", r"זחילה", r"(?<![א-ת])זחל"])},
    "sitting_ground": {"he": "הישיבה על הקרקע", "short": "ישיבה על הקרקע", "en": "sitting on the ground", "re": _any([
        r"\bsit(?:s|ting)? (?:\w+ ){0,2}on the (?:ground|floor|steps|stairs|pavement)", r"\bsat (?:\w+ ){0,2}on the",
        r"יושב(?:ת|ים)? על (?:הקרקע|הרצפה|המדרגות|האדמה)"])},
    "working_ground": {"he": "העבודה על הקרקע", "short": "עבודה על הקרקע", "en": "working on the ground", "re": _any([
        r"\bwork\w* (?:\w+ ){0,2}on the (?:ground|floor|pavement|stairs|steps)", r"עובד(?:ת|ים)? על (?:הקרקע|הרצפה|המדרגות)"])},
    "holding_tool": {"he": "כלי העבודה ביד", "short": "כלי עבודה ביד", "en": "holding a tool or a long bar", "re": _any([
        r"\b(?:hold|carr|grip)\w* (?:\w+ ){0,4}(?:bars?|rods?|poles?|pipes?|tools?|sticks?|planks?|cables?|wires?)\b",
        r"\b(?:metal|long|iron) (?:bar|rod|pole|pipe)\b",
        r"(?:מחזיק|מחזיקה|מחזיקים|נושא|נושאים|סוחב|אוחז)\S* (?:\S+ ){0,3}(?:מוט|צינור|כלי|קרש|מקל|כבל|חוט)",
        r"מוט מתכת", r"מוט ארוך"])},
    "ladder": {"he": "הסולם", "short": "סולם", "en": "carrying or using a ladder", "re": _any([r"\bladder", r"סולם"])},
    "digging": {"he": "החפירה", "short": "חפירה", "en": "digging", "re": _any([r"\bdig(?:s|ging)?\b", r"\bdug\b", r"חופר",
                                                                                   r"חפירה"])},
    "car_door": {"he": "פתיחת דלת הרכב", "short": "דלת רכב", "en": "opening a car door", "re": _any([
        r"\b(?:car|vehicle|van|truck|sedan)(?:'s)? (?:\w+ )?doors?\b", r"\bdoor of (?:a |the )?(?:\w+ )?(?:car|vehicle|van)\b",
        r"\b(?:car|vehicle|van|truck|sedan)\b.{0,50}\bopen\w* (?:the |its |a )?doors?\b",
        r"\bopen\w* (?:the |a )?doors? (?:of|to) (?:the |a )?(?:\w+ )?(?:car|vehicle|van)\b",
        r"דלת (?:ה)?(?:רכב|מכונית)", r"(?:רכב|מכונית).{0,40}פות(?:ח|חת|חים) (?:את )?(?:ה)?דלת",
        r"פות(?:ח|חת|חים) (?:את )?דלת (?:ה)?(?:רכב|מכונית)"])},
}
ACTION_KEYS = tuple(ACTIONS)

# ---------- places ----------
PLACES: Dict[str, Dict[str, Any]] = {
    "stairs": {"he": "במדרגות", "en": "on the stairs", "re": _any([r"\bstairs?\b", r"\bsteps\b", r"\bstaircase",
                                                                   r"מדרגות", r"מדרגה"])},
    "door": {"he": "ליד הדלת", "en": "at the door", "re": _any([
        r"(?<!car )(?<!vehicle )\bdoors?\b(?! of (?:a |the )?(?:\w+ )?(?:car|vehicle|van))",
        r"(?<![א-ת])ה?דלת(?! (?:ה)?(?:רכב|מכונית))"])},
    "window": {"he": "ליד החלון", "en": "at the window", "re": _any([r"\bwindows?\b", r"חלון"])},
    "gate": {"he": "בשער", "en": "at the gate", "re": _any([r"\bgates?\b", r"(?<![א-ת])ה?שער(?![א-ת])"])},
    "fence": {"he": "ליד הגדר", "en": "at the fence", "re": _any([r"\bfence", r"גדר"])},
    "roof": {"he": "על הגג", "en": "on the roof", "re": _any([r"\broof", r"(?<![א-ת])[בעה]?ה?גג(?![א-ת])"])},
    "yard": {"he": "בחצר", "en": "in the yard", "re": _any([r"\byard\b", r"\bgarden\b", r"חצר", r"גינה"])},
    "parking": {"he": "בחניה", "en": "in the parking", "re": _any([r"\bparking\b", r"\bdriveway", r"חניה", r"חנייה"])},
}
WAY_IN = ("door", "window", "gate", "fence", "roof")      # a different way in is never covered by the owner's words


def actions_in(text: str) -> List[str]:
    """The normalised actions *text* names (an alert's why + summary, or the owner's words), in ACTIONS order."""
    text = str(text or "")
    return [key for key, spec in ACTIONS.items() if spec["re"].search(text)]


def places_in(text: str) -> List[str]:
    text = str(text or "")
    return [key for key, spec in PLACES.items() if spec["re"].search(text)]


# ---------- what keeps a red red ----------
BREAK_IN = _any([
    r"\bbreak", r"\bbroke", r"\bsmash", r"\bshatter", r"\bforc", r"\bpr(?:y|ies|ied|ying)\b", r"\bburgl", r"\bsteal",
    r"\bstole", r"\btheft", r"\bthie", r"\brob", r"\btamper", r"\bpick(?:s|ed|ing)? (?:the |a )?lock",
    r"\btak(?:e|es|ing|en)\b (?:\w+ ){0,3}(?:from|out of) (?:inside|the car|the vehicle|a car|a vehicle|the trunk)",
    r"\btak(?:e|es|ing|en) items\b",
    r"פורץ", r"פורצים", r"פריצה", r"(?<![א-ת])פרץ", r"שובר", r"(?<![א-ת])שבר", r"מנפץ", r"ניפץ", r"גונב", r"גניבה",
    r"(?<![א-ת])גנב", r"שודד", r"לקחת חפצים", r"לוקח חפצים", r"לקח חפצים", r"מחטט", r"נכנס לבית", r"מטפס",
])


# Harm a "not" cannot take back ("lying on the ground, not moving" is the danger itself), and the violence words
# alert_guards does not list.
STILL_OR_HURT = _any([
    r"\bnot moving", r"\bmotionless", r"\bunresponsive", r"\bimmobile", r"\bnot breathing", r"\bstab", r"\bshoot",
    r"\bshot\b", r"\bstrangl", r"\bchok", r"\bblood", r"\bbleed", r"לא זז", r"ללא תנועה", r"לא נע(?![א-ת])", r"לא נושם",
    r"דוקר", r"דקר", r"יורה", r"(?<![א-ת])ירה", r"חונק", r"(?<![א-ת])ה?דם(?![א-ת])", r"מדמם",
])


def red_blocked(text: str) -> str:
    """Why an explained action can never lower this red ("" when a context look may be asked): a CLEAR serious
    thing, a weapon, violence, a break-in or a theft, harm to a person, or a car door (owner decision 2026-10-09:
    car-door reds always stay red)."""
    from .alert_guards import CLEAR, DOWN_HARM, VERIFY_PATTERNS, _NEGATED  # noqa: PLC0415

    text = str(text or "")
    if CLEAR.search(text):
        return "clear"
    if STILL_OR_HURT.search(text):
        return "harm"
    for name in ("weapon", "vehicle", "violence"):
        if VERIFY_PATTERNS[name].search(text):
            return name
    if BREAK_IN.search(text):
        return "break-in"
    if ACTIONS["car_door"]["re"].search(text):
        return "car door"
    if DOWN_HARM.search(_NEGATED.sub(" ", text)):
        return "harm"
    return ""


# ---------- the fact ----------
@dataclass
class ActivityFact:
    id: str
    cameras: List[str]
    actions: List[str]
    cause: str                          # in the owner's language, from their words ("החשמלאים שמתקינים לדים")
    cause_en: str = ""
    place: str = ""                     # a PLACES key, or ""
    place_words: str = ""               # the owner's own words for the place ("במדרגות")
    owner_words: str = ""
    who: str = ""                       # the live known mark's people, when it is them
    known_id: str = ""
    daily_from: str = ""
    daily_to: str = ""
    until: float = 0.0
    by: str = ""
    at: float = 0.0
    alert_id: str = ""
    cancelled_at: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def live(self, now: float) -> bool:
        return not self.cancelled_at and float(self.until or 0) > now

    def in_window(self, ts: float) -> bool:
        """Inside the end and, with a daily window, inside its hours of that day."""
        if self.cancelled_at or ts > float(self.until or 0):
            return False
        if self.daily_from and self.daily_to:
            hm = dt.datetime.fromtimestamp(ts).strftime("%H:%M")
            return self.daily_from <= hm <= self.daily_to
        return True

    def covers(self, camera: str) -> bool:
        return not self.cameras or camera in self.cameras


def _fact(row: Dict[str, Any]) -> Optional[ActivityFact]:
    try:
        names = set(ActivityFact.__dataclass_fields__)
        fact = ActivityFact(**{k: v for k, v in row.items() if k in names})
        fact.cameras = [str(c) for c in (fact.cameras or []) if c]
        fact.actions = [a for a in (fact.actions or []) if a in ACTIONS]
        fact.until, fact.at, fact.cancelled_at = float(fact.until or 0), float(fact.at or 0), float(fact.cancelled_at or 0)
        return fact if fact.id and fact.actions else None
    except Exception:  # noqa: BLE001 - one bad row is skipped
        return None


class ActivityBook:
    """The owner's explained actions, one JSON file. Re-read when another process wrote it. Thread-safe."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._facts: List[ActivityFact] = []
        self._mtime = -1.0

    def _load(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self._facts, self._mtime = [], -1.0
            return
        if mtime == self._mtime:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                rows = json.load(f)
            self._facts = [x for x in (_fact(r) for r in rows if isinstance(r, dict)) if x is not None]
        except Exception as exc:  # noqa: BLE001 - a damaged file is an empty book
            log.warning("Activity memory not read: %s", exc)
            self._facts = []
        self._mtime = mtime

    def _save(self, now: float) -> None:
        keep = [f for f in self._facts if f.until > now - 7 * 86400]        # a week after the end, it goes
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in keep], f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)
        self._facts = keep
        try:
            self._mtime = os.path.getmtime(self.path)
        except OSError:
            self._mtime = -1.0

    def add(self, cameras: Iterable[str], actions: Iterable[str], cause: str, until: float, now: float,
            **extra: Any) -> ActivityFact:
        acts = [a for a in dict.fromkeys(actions) if a in ACTIONS]
        if not acts:
            raise ValueError("an activity fact needs at least one known action")
        if not str(cause or "").strip():
            raise ValueError("an activity fact needs the owner's cause")
        if until <= now:
            raise ValueError("the window has already ended")
        with self._lock:
            self._load()
            fact = ActivityFact(id="act_" + uuid.uuid4().hex[:8], cameras=list(dict.fromkeys(c for c in cameras if c)),
                                actions=acts, cause=" ".join(str(cause).split()), until=float(until), at=float(now),
                                **{k: v for k, v in extra.items() if k in ActivityFact.__dataclass_fields__})
            self._facts.append(fact)
            self._save(now)
            log.info("activity: %s at %s = %s until %s", ",".join(fact.actions), ",".join(fact.cameras) or "all",
                     fact.cause, dt.datetime.fromtimestamp(fact.until).strftime("%a %H:%M"))
            return fact

    def update(self, fact_id: str, now: float, **changes: Any) -> Optional[ActivityFact]:
        """Changes one live fact (more actions, another camera or place, a new end); the old values go to its
        history. None when there is no such live fact."""
        with self._lock:
            self._load()
            fact = next((f for f in self._facts if f.id == fact_id and f.live(now)), None)
            if fact is None:
                return None
            before = {k: getattr(fact, k) for k in changes if k in ActivityFact.__dataclass_fields__}
            for key, value in changes.items():
                if key in ("id", "history") or key not in ActivityFact.__dataclass_fields__:
                    continue
                if key == "actions":
                    value = [a for a in dict.fromkeys(value) if a in ACTIONS] or fact.actions
                setattr(fact, key, value)
            fact.history = (fact.history + [{"at": now, "before": before}])[-10:]
            self._save(now)
            return fact

    def cancel(self, fact_id: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            self._load()
            fact = next((f for f in self._facts if f.id == fact_id and not f.cancelled_at), None)
            if fact is None:
                return False
            fact.cancelled_at = now
            self._save(now)
            return True

    def live(self, now: float) -> List[ActivityFact]:
        with self._lock:
            self._load()
            return [f for f in self._facts if f.live(now)]

    def get(self, fact_id: str) -> Optional[ActivityFact]:
        with self._lock:
            self._load()
            return next((f for f in self._facts if f.id == fact_id), None)

    def match(self, camera: str, ts: float, text: str, for_red: bool = False) -> Optional[ActivityFact]:
        """The live fact that explains an alert at *camera* and *ts* whose why + summary is *text*: inside its
        window, naming one of its actions (for a red, not the car door), and no way in other than its own place."""
        named = set(actions_in(text))
        if for_red:
            named.discard("car_door")
        if not named:
            return None
        ways = [p for p in places_in(text) if p in WAY_IN]
        with self._lock:
            self._load()
            for fact in reversed(self._facts):                 # the newest first
                if not fact.covers(camera) or not fact.in_window(ts) or not named & set(fact.actions):
                    continue
                if ways and fact.place not in ways:
                    continue
                return fact
        return None


_BOOK: Optional[ActivityBook] = None
_BOOK_LOCK = threading.Lock()


def book(path: Optional[str] = None) -> Optional[ActivityBook]:
    """The box's one activity book (state dir / activity.json); None when the state dir cannot be read."""
    global _BOOK
    if path is not None:
        return ActivityBook(path)
    with _BOOK_LOCK:
        if _BOOK is None:
            try:
                from . import paths  # noqa: PLC0415

                _BOOK = ActivityBook(os.path.join(paths.state_dir(), "activity.json"))
            except Exception as exc:  # noqa: BLE001 - the box runs without it
                log.warning("Activity memory not available: %s", exc)
                return None
        return _BOOK


# ---------- the words ----------
def join_he(words: Sequence[str]) -> str:
    """"השכיבה", "השכיבה והכיפוף", "השכיבה, הזחילה והכיפוף"."""
    words = [w for w in words if w]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " ו" + words[-1]


def actions_text(actions: Sequence[str], lang: str = "he", short: bool = False) -> str:
    acts = [a for a in actions if a in ACTIONS]
    if str(lang).startswith("he"):
        if short:
            return "/".join(ACTIONS[a]["short"] for a in acts)
        return join_he([ACTIONS[a]["he"] for a in acts])
    names = [ACTIONS[a]["en"] for a in acts]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""


def place_text(fact: ActivityFact, lang: str = "he") -> str:
    if fact.place in PLACES:
        return PLACES[fact.place]["he" if str(lang).startswith("he") else "en"]
    return str(fact.place_words or "")


def context_question(fact: ActivityFact) -> str:
    """The red's second look with the owner's context (a "true" lowers the red; unclear is false and keeps it)."""
    cause = fact.cause_en or fact.cause
    where = place_text(fact, "en")
    return (f'The owner of the house said: "{cause}". So {actions_text(fact.actions, "en")}'
            f'{" " + where if where else ""} here is part of that work. Is what you see consistent with that work? '
            "Answer false if anyone looks hurt, unconscious or in pain, is attacked, threatened or held down, or if "
            "something is being broken into or stolen.")


VERIFY_TIMEOUT_SEC = 15.0


def red_look(backend: Any, frames: List[Any], camera: str, ts: float, text: str, lang: str = "he",
             timeout: Optional[float] = None, activities: Optional[ActivityBook] = None) -> Optional[Dict[str, Any]]:
    """The context look for a red at *camera* (*text*: its why + reason + summary), or None when no live fact of
    the owner explains it or the red names something that is never lowered (:func:`red_blocked`). Never raises.

    The record: ``fact_id``, ``cause``, ``question``, ``answered``, ``consistent``, ``what_it_is``,
    ``evidence_frame``, ``lowered`` (consistent, and the answer names no harm), ``reason``, ``seconds``."""
    try:
        store = activities if activities is not None else book()
        if store is None:
            return None
        fact = store.match(camera, ts, text, for_red=True)
        if fact is None:
            return None
        blocked = red_blocked(text)
        if blocked:
            log.info("[%s] activity %s not asked: the red names %s", camera, fact.id, blocked)
            return {"fact_id": fact.id, "cause": fact.cause, "question": "", "answered": False, "consistent": None,
                    "what_it_is": "", "evidence_frame": 0, "lowered": False, "reason": f"the red names {blocked}"}
        question = context_question(fact)
        record: Dict[str, Any] = {"fact_id": fact.id, "cause": fact.cause, "question": question, "answered": False,
                                  "consistent": None, "what_it_is": "", "evidence_frame": 0, "lowered": False,
                                  "reason": ""}
        verify = getattr(backend, "verify", None)
        if not callable(verify):
            record["reason"] = "the vision model cannot take a second look"
            return record
        timeout = VERIFY_TIMEOUT_SEC if timeout is None else timeout
        box: Dict[str, Any] = {}

        def call() -> None:
            try:
                box["answer"] = verify(frames, question, language="Hebrew" if lang == "he" else "English",
                                       timeout=timeout)
            except TypeError:
                try:
                    box["answer"] = verify(frames, question)
                except Exception as exc:  # noqa: BLE001
                    box["error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001
                box["error"] = f"{type(exc).__name__}: {exc}"

        started = time.monotonic()
        thread = threading.Thread(target=call, name="activity-look", daemon=True)
        thread.start()
        thread.join(timeout)
        record["seconds"] = round(time.monotonic() - started, 2)
        answer = box.get("answer")
        if thread.is_alive():
            record["reason"] = f"no answer within {timeout:.0f} s"
        elif "error" in box:
            record["reason"] = box["error"][:300]
        elif not isinstance(answer, dict) or not isinstance(answer.get("confirmed"), bool):
            record["reason"] = "no usable answer"
        else:
            what = str(answer.get("what_it_is") or "")
            record.update(answered=True, consistent=answer["confirmed"], what_it_is=what,
                          evidence_frame=answer.get("evidence_frame") or 0)
            from .alert_guards import _NEGATED  # noqa: PLC0415

            if answer["confirmed"] and not _harm(what) and not BREAK_IN.search(_NEGATED.sub(" ", what)):
                record["lowered"] = True
            elif answer["confirmed"]:
                record["reason"] = "the answer itself names harm"
        return record
    except Exception as exc:  # noqa: BLE001 - the red goes out as it is
        log.warning("[%s] activity look failed; the red stays: %s", camera, exc)
        return None


def _harm(text: str) -> bool:
    from .alert_guards import CLEAR, DOWN_HARM, VERIFY_PATTERNS, _NEGATED  # noqa: PLC0415

    kept = _NEGATED.sub(" ", str(text or ""))
    return bool(DOWN_HARM.search(kept) or CLEAR.search(kept) or VERIFY_PATTERNS["violence"].search(kept)
                or VERIFY_PATTERNS["weapon"].search(kept) or STILL_OR_HURT.search(kept))


def explained(camera: str, ts: float, text: str, activities: Optional[ActivityBook] = None) -> Optional[ActivityFact]:
    """The live fact that explains a suspicious at *camera* (kept, not sent), or None. Never raises."""
    try:
        store = activities if activities is not None else book()
        return store.match(camera, ts, text) if store is not None else None
    except Exception as exc:  # noqa: BLE001 - the alert goes on as before
        log.warning("[%s] activity memory not read: %s", camera, exc)
        return None


def quiet_reason(fact: ActivityFact) -> str:
    return f"owner explained: {fact.cause}"
