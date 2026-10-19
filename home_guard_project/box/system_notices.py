"""Three system notices to the owner over Telegram: the box was down, the AI is not available, no camera picture.

2026-10-10 the box ran with no vision AI from 00:25 to 09:05 (the OpenRouter credit ran out: every Eye call got 402),
was stopped 09:06-12:33, and a power cut hit at 09:23. Nobody knew. The owner then approved exactly these owner
messages (and no others; his rule: never a message type he did not agree to):

1. Box down, once when inference starts again after a gap of DOWN_GAP_SEC or more::

       ⚠️ המערכת לא פעלה בין 02:10 ל-02:47 (הקופסה כבתה, כנראה הפסקת חשמל).

   The cause is said only when it is known: Windows booted after the last alive time. Nothing is sent on the very
   first run (no alive file) or when the gap began with the owner's own stop (control.stop records it, OWNER_STOP).
2. AI not available, once per outage: no Eye answer for AI_OUTAGE_SEC since the first failed call AND at least
   AI_OUTAGE_FAILS failed calls in a row (a failed call: the main model, the fallback and the 768 rescue all failed)::

       ⚠️ ה-AI לא זמין מ-00:25 (נגמר הקרדיט ב-OpenRouter). הקופסה ממשיכה להקליט ולשמור, אבל בלי בדיקת AI לא נשלחות
       התרעות עד שזה יחזור.

   and ONE line when the first Eye call answers again::

       ✅ ה-AI חזר לעבוד. לא היה זמין 00:25–09:05.

3. No picture from any camera (approved 2026-10-19, after the recorder came back from a power cut on a new address and
   the box saw nothing for a day and a half without telling anyone), once per outage: every watched camera without a
   picture for CAMERAS_OUTAGE_SEC::

       ⚠️ אין תמונה מאף מצלמה מ-06:59. הקופסה פועלת, אבל בלי תמונה לא נשלחות התרעות. כדאי לבדוק שמכשיר ההקלטה דלוק
       ומחובר לרשת.

   and ONE line when a camera sends a picture again (the clause only when recorder_finder.py moved the address)::

       ✅ המצלמות חזרו לעבוד. לא הייתה תמונה 06:59–10:45 (מכשיר ההקלטה קיבל כתובת חדשה ברשת והקופסה מצאה אותו לבד).

   An outage that began before the box was down for DOWN_GAP_SEC or more is not carried over: the box-down notice
   tells that part.

All are fixed texts (no model is asked to write or translate them), Hebrew on a Hebrew box and English otherwise,
sent as a normal (not silent) message with no buttons. box.yaml ``system_notices: on | off`` (default on). A send
never blocks or stops inference: one background thread sends them in order, retrying while the network is not up yet
(the boot after a power cut), and logs what happened. Each decision is logged in one ASCII line ("Box-down notice:",
"AI-down notice:", "AI-back notice:", "Cameras-down notice:", "Cameras-back notice:"), which the runner's log filter keeps (it keeps every line with our time stamp).

Files: ALIVE_NAME and STATE_NAME in the box state folder (paths.state_dir), OWNER_STOP_NAME next to the runner's
flags in the logs folder (written by control.stop). All are written through a synced temp file (2026-10-10 09:23 the
power cut left files that were replaced but never synced as zero bytes); an unreadable alive file falls back to its
modification time.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional

log = logging.getLogger("box.system_notices")

ALIVE_EVERY_SEC = 60.0          # how often a running inference writes its alive time
DOWN_GAP_SEC = 600.0            # a shorter gap (a restart, a deploy) is not news
AI_OUTAGE_SEC = 600.0           # since the first failed Eye call of the streak
AI_OUTAGE_FAILS = 3             # failed Eye calls in a row
CAMERAS_OFFLINE_SEC = 60.0      # a camera with no picture this long is offline (as ai_status.offline)
CAMERAS_OUTAGE_SEC = 600.0      # every camera offline this long: tell the owner
# The owner's stop is written up to ALIVE_EVERY_SEC after the last alive time; the runner needs a couple of seconds
# to end the program, so an alive time may also land just after the stop.
STOP_SLACK_SEC = 30.0
SEND_TRIES = 20                 # about ten minutes of tries: the network may come up well after Windows does
SEND_WAIT_SEC = 30.0

ALIVE_NAME = "system_alive.json"
STATE_NAME = "system_notices.json"
OWNER_STOP_NAME = "owner_stop.json"

BOX_DOWN, AI_DOWN, AI_BACK = "box_down", "ai_down", "ai_back"
CAMERAS_DOWN, CAMERAS_BACK = "cameras_down", "cameras_back"

# The causes of a failed Eye call, from its error text. Mirrors brain/models._NO_ACCESS, split in two: money first.
_CREDIT = re.compile(r"\b402\b|payment required|insufficient_quota|no credits|more credits|"
                     r"credits? (?:remaining|exhausted)", re.IGNORECASE)
_KEY = re.compile(r"\b401\b|invalid api key|incorrect api key|unauthori[sz]ed", re.IGNORECASE)
CREDIT, KEY, OTHER = "credit", "key", "other"

_PROVIDER_NAMES = {"openrouter": "OpenRouter", "openai": "OpenAI", "dashscope-intl": "DashScope"}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ----------------------------------------------------------------------------
# Switch and texts
# ----------------------------------------------------------------------------
def enabled(box_settings: Mapping[str, Any]) -> bool:
    """box.yaml ``system_notices`` (default on). YAML reads on/off as booleans; an unknown value is on."""
    value = (box_settings or {}).get("system_notices", True)
    if isinstance(value, bool):
        return value
    text = str("" if value is None else value).strip().lower()
    if text in ("off", "false", "no", "0"):
        return False
    if text not in ("on", "true", "yes", "1", ""):
        log.warning("Unknown system_notices '%s'; using on.", value)
    return True


def _he(lang: str) -> bool:
    return str(lang or "").startswith("he")


def _with_date(start: float, end: float) -> bool:
    """Times alone are enough unless the span crosses midnight or is longer than a day."""
    return (datetime.fromtimestamp(start).date() != datetime.fromtimestamp(end).date()) or end - start > 86400


def _when(ts: float, lang: str, with_date: bool) -> str:
    """Local box time: "02:10", or "10.10 02:10" / "Oct 10 02:10" when the date is needed."""
    moment = datetime.fromtimestamp(ts)
    clock = moment.strftime("%H:%M")
    if not with_date:
        return clock
    if _he(lang):
        return f"{moment.day}.{moment.month} {clock}"
    return f"{_MONTHS[moment.month - 1]} {moment.day} {clock}"


def down_text(start: float, end: float, lang: str, power_cut: bool = False) -> str:
    """The box-down message for the gap *start*..*end*; the cause only when Windows booted inside the gap."""
    dated = _with_date(start, end)
    a, b = _when(start, lang, dated), _when(end, lang, dated)
    if _he(lang):
        return f"⚠️ המערכת לא פעלה בין {a} ל-{b}" + (" (הקופסה כבתה, כנראה הפסקת חשמל)" if power_cut else "") + "."
    return (f"⚠️ The system was not running between {a} and {b}"
            + (" (the box was off, probably a power cut)" if power_cut else "") + ".")


def ai_cause(error: str) -> str:
    """CREDIT for 402 / no credits, KEY for 401 / a bad key, OTHER for anything else."""
    text = str(error or "")
    if _CREDIT.search(text):
        return CREDIT
    if _KEY.search(text):
        return KEY
    return OTHER


def _cause_words(cause: str, provider: str, lang: str) -> str:
    name = _PROVIDER_NAMES.get(str(provider or "").strip().lower(), "")
    if cause == CREDIT:
        if _he(lang):
            return f"נגמר הקרדיט ב-{name}" if name else "נגמר הקרדיט אצל ספק ה-AI"
        return f"the {name} credit ran out" if name else "the AI provider's credit ran out"
    if cause == KEY:
        return "בעיה במפתח ה-API" if _he(lang) else "a problem with the API key"
    return "הספק לא עונה" if _he(lang) else "the provider is not answering"


def ai_down_text(since: float, now: float, cause: str, provider: str, lang: str) -> str:
    """The AI-not-available message, for an outage that began at *since* (told at *now*)."""
    when = _when(since, lang, _with_date(since, now))
    why = _cause_words(cause, provider, lang)
    if _he(lang):
        return (f"⚠️ ה-AI לא זמין מ-{when} ({why}). הקופסה ממשיכה להקליט ולשמור, אבל בלי בדיקת AI לא נשלחות התרעות "
                "עד שזה יחזור.")
    return (f"⚠️ The AI has not been available since {when} ({why}). The box keeps recording and saving, but without "
            "the AI check no alerts are sent until it is back.")


def ai_back_text(since: float, until: float, kept: int, lang: str) -> str:
    """The one line when the Eye answers again; *kept*: the events saved without an AI check (none: no clause)."""
    dated = _with_date(since, until)
    a, b = _when(since, lang, dated), _when(until, lang, dated)
    kept = max(0, int(kept or 0))
    if _he(lang):
        head = f"✅ ה-AI חזר לעבוד. לא היה זמין {a}–{b}"
        if not kept:
            return head + "."
        saved = "נשמר אירוע אחד" if kept == 1 else f"נשמרו {kept} אירועים"
        return f"{head}, ובזמן הזה {saved} בלי בדיקה."
    head = f"✅ The AI is working again. It was not available {a}–{b}"
    if not kept:
        return head + "."
    saved = "1 event was saved" if kept == 1 else f"{kept} events were saved"
    return f"{head}, and {saved} without a check in that time."


def cameras_down_text(since: float, now: float, lang: str) -> str:
    """No picture from any camera since *since* (told at *now*)."""
    when = _when(since, lang, _with_date(since, now))
    if _he(lang):
        return (f"⚠️ אין תמונה מאף מצלמה מ-{when}. הקופסה פועלת, אבל בלי תמונה לא נשלחות התרעות. "
                "כדאי לבדוק שמכשיר ההקלטה דלוק ומחובר לרשת.")
    return (f"⚠️ No camera has sent a picture since {when}. The box is running, but without pictures no alerts are "
            "sent. Please check that the recorder is on and connected to the network.")


def cameras_back_text(since: float, until: float, moved: bool, lang: str) -> str:
    """The one line when a camera sends a picture again; *moved*: the box found the recorder at a new address."""
    dated = _with_date(since, until)
    a, b = _when(since, lang, dated), _when(until, lang, dated)
    if _he(lang):
        return (f"✅ המצלמות חזרו לעבוד. לא הייתה תמונה {a}–{b}"
                + (" (מכשיר ההקלטה קיבל כתובת חדשה ברשת והקופסה מצאה אותו לבד)" if moved else "") + ".")
    return (f"✅ The cameras are working again. There was no picture {a}–{b}"
            + (" (the recorder got a new network address and the box found it by itself)" if moved else "") + ".")


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def _write_json(path: str, data: Any) -> None:
    """Write through a temp file that is synced to disk before it replaces the old one."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number > 0 else None


def record_owner_stop(log_dir: str, now: Optional[float] = None) -> None:
    """The owner stopped the box (the app, Telegram, ``python -m home_guard_project.box stop``): kept after the start
    clears the stop flag, so the next start can tell a stop from a death. Never raises."""
    try:
        _write_json(os.path.join(log_dir, OWNER_STOP_NAME), {"at": time.time() if now is None else now})
    except Exception as exc:  # noqa: BLE001 - the stop itself must go on
        log.warning("Owner stop time not recorded: %s", exc)


def owner_stop_at(log_dir: str) -> Optional[float]:
    return _number((_read_json(os.path.join(log_dir, OWNER_STOP_NAME)) or {}).get("at"))


def last_alive(state_dir: str) -> Optional[float]:
    """The last time a running inference said it was alive; the file's time when its body is unreadable (a power
    cut); None when there is no file (the very first run)."""
    path = os.path.join(state_dir, ALIVE_NAME)
    if not os.path.exists(path):
        return None
    ts = _number((_read_json(path) or {}).get("at"))
    if ts is not None:
        return ts
    try:
        ts = os.path.getmtime(path)
        log.warning("Alive file unreadable; using its time %s", datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"))
        return ts
    except OSError:
        return None


def boot_time() -> Optional[float]:
    """When Windows (or Linux) last started, as an epoch; None when it cannot be told."""
    try:
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415

            ticks = ctypes.windll.kernel32.GetTickCount64
            ticks.restype = ctypes.c_ulonglong
            return time.time() - ticks() / 1000.0
        with open("/proc/uptime", encoding="ascii") as f:
            return time.time() - float(f.read().split()[0])
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------------
def down_decision(alive: Optional[float], now: float, stopped_at: Optional[float],
                  booted_at: Optional[float]) -> Dict[str, Any]:
    """Whether a start at *now* after the last *alive* time is news: ``{"send", "why", "owner_stop", "power_cut"}``."""
    if alive is None:
        return {"send": False, "why": "first run (no alive file)", "owner_stop": False, "power_cut": False}
    owner_stop = stopped_at is not None and alive - STOP_SLACK_SEC <= stopped_at <= now
    power_cut = booted_at is not None and alive < booted_at <= now
    gap = now - alive
    if gap < DOWN_GAP_SEC:
        why = f"gap {int(gap // 60)} min, below {DOWN_GAP_SEC / 60:.0f} min"
        return {"send": False, "why": why, "owner_stop": owner_stop, "power_cut": power_cut}
    if owner_stop:
        return {"send": False, "why": "the owner stopped the box", "owner_stop": True, "power_cut": power_cut}
    return {"send": True, "why": "", "owner_stop": False, "power_cut": power_cut}


class SystemNotices:
    """The box-down check at start, the alive time while running, and the Eye's outage. Safe from several threads.

    *send(kind, text)* hands a message on (Sender.put on the box); *lang()* is the box language now; *clock* and
    *boot* are injected by the tests."""

    def __init__(self, state_dir: str, log_dir: str, send: Callable[[str, str], None], lang: Callable[[], str],
                 provider: str = "", on: bool = True, clock: Callable[[], float] = time.time,
                 boot: Callable[[], Optional[float]] = boot_time) -> None:
        self.state_dir, self.log_dir = state_dir, log_dir
        self.send, self.lang, self.provider, self.on = send, lang, provider, on
        self.clock, self.boot = clock, boot
        self._lock = threading.Lock()
        self._alive_written = 0.0
        self._eye = self._load("eye")
        self._cams = self._load("cameras")

    # -- the box was down --------------------------------------------------------------------------
    def box_started(self, now: Optional[float] = None) -> Dict[str, Any]:
        """At inference start: tell the owner about a gap since the last alive time, then write the alive time."""
        now = self.clock() if now is None else now
        alive = last_alive(self.state_dir)
        booted = self.boot()
        decision = down_decision(alive, now, owner_stop_at(self.log_dir), booted)
        if alive is None:
            log.info("Box-down notice: no earlier alive time (first run) -> nothing to send")
        else:
            lang = self._lang()
            gap = f"{_when(alive, 'en', _with_date(alive, now))}-{_when(now, 'en', _with_date(alive, now))}"
            facts = (f"owner stop: {'yes' if decision['owner_stop'] else 'no'}, "
                     f"power cut: {'unknown' if booted is None else 'yes' if decision['power_cut'] else 'no'}")
            if not decision["send"]:
                outcome = f"not sent ({decision['why']})"
            elif not self.on:
                outcome = "not sent (system_notices off)"
            else:
                self._hand_on(BOX_DOWN, down_text(alive, now, lang, power_cut=decision["power_cut"]))
                outcome = "sent"
            log.info("Box-down notice: gap %s (%s) -> %s", gap, facts, outcome)
        if (alive is None or now - alive >= DOWN_GAP_SEC) and self._cams and not self._cams.get("announced"):
            self._cams = {}             # the box was down: the box-down notice covers that time
            self._save()
        self.alive(now, force=True)
        return decision

    def alive(self, now: Optional[float] = None, force: bool = False) -> None:
        """Write the alive time, at most once every ALIVE_EVERY_SEC unless *force*. Never raises."""
        now = self.clock() if now is None else now
        if not force and now - self._alive_written < ALIVE_EVERY_SEC:
            return
        self._alive_written = now
        try:
            _write_json(os.path.join(self.state_dir, ALIVE_NAME), {"at": now})
        except Exception as exc:  # noqa: BLE001
            log.warning("Alive time not written: %s", exc)

    def tick(self, now: Optional[float] = None) -> None:
        """From the guard loop: the alive time, and an outage that passed its time with no new Eye call."""
        now = self.clock() if now is None else now
        self.alive(now)
        if not self._eye.get("since") or self._eye.get("announced"):
            return                       # no outage to look at: the guard loop's beat costs nothing more
        try:
            with self._lock:
                self._maybe_announce(now)
        except Exception as exc:  # noqa: BLE001
            log.warning("AI outage not checked: %s", exc)

    # -- the AI is not available ------------------------------------------------------------------
    def eye(self, answered: bool, error: str = "", now: Optional[float] = None) -> None:
        """One Eye call ended: *answered*, or failed after the fallback and the rescue with *error*. Never raises."""
        now = self.clock() if now is None else now
        try:
            with self._lock:
                if answered:
                    self._eye_answered(now)
                else:
                    self._eye_failed(error, now)
        except Exception as exc:  # noqa: BLE001
            log.warning("AI outage state not updated: %s", exc)

    def _eye_failed(self, error: str, now: float) -> None:
        eye = self._eye
        if not eye.get("since"):
            eye["since"] = now
        eye["fails"] = int(eye.get("fails") or 0) + 1
        eye["cause"] = ai_cause(error)
        log.info("AI outage: %d failed Eye call(s) in a row since %s (cause: %s)", eye["fails"],
                 _when(eye["since"], "en", _with_date(eye["since"], now)), eye["cause"])
        self._maybe_announce(now)
        self._save_eye()

    def _maybe_announce(self, now: float) -> None:
        eye = self._eye
        since = _number(eye.get("since"))
        if since is None or eye.get("announced"):
            return
        if now - since < AI_OUTAGE_SEC or int(eye.get("fails") or 0) < AI_OUTAGE_FAILS:
            return
        when = _when(since, "en", _with_date(since, now))
        facts = f"no answer since {when}, {eye['fails']} failed Eye calls, cause: {eye.get('cause') or OTHER}"
        if not self.on:
            log.info("AI-down notice: %s -> not sent (system_notices off)", facts)
            return
        eye["announced"] = now
        self._save_eye()
        self._hand_on(AI_DOWN, ai_down_text(since, now, eye.get("cause") or OTHER, self.provider, self._lang()))
        log.info("AI-down notice: %s -> sent", facts)

    def _eye_answered(self, now: float) -> None:
        eye = self._eye
        if not eye.get("since") and not eye.get("fails"):
            return
        since = _number(eye.get("since")) or now
        kept = int(eye.get("fails") or 0)
        span = f"{_when(since, 'en', _with_date(since, now))}-{_when(now, 'en', _with_date(since, now))}"
        if eye.get("announced"):
            # No count for the owner: failed calls are not events (one event can fail several calls), and a wrong
            # number is worse than none. The log keeps it.
            self._hand_on(AI_BACK, ai_back_text(since, now, 0, self._lang()))
            log.info("AI-back notice: not available %s, %d failed Eye call(s) -> sent", span, kept)
        else:
            log.info("AI outage over: %d failed Eye call(s) %s, never announced -> nothing to send", kept, span)
        self._eye = {}
        self._save_eye()

    # -- no picture from any camera ----------------------------------------------------------------
    def cameras(self, now: float, last_frames: Mapping[str, float], started: float) -> None:
        """From the guard loop: each watched camera's last picture time (0: none yet since *started*). Never raises."""
        try:
            with self._lock:
                self._cameras(now, last_frames, started)
        except Exception as exc:  # noqa: BLE001
            log.warning("Camera outage not checked: %s", exc)

    def _cameras(self, now: float, last_frames: Mapping[str, float], started: float) -> None:
        if not last_frames:
            return
        picture = max(float(ts or 0.0) for ts in last_frames.values())
        latest = max(picture, float(started))   # a camera that sent nothing yet had since the start to do so
        cams = self._cams
        if picture and now - picture < CAMERAS_OFFLINE_SEC:          # a camera is sending pictures
            if not cams.get("since"):
                return
            since = _number(cams.get("since")) or now
            dated = _with_date(since, now)
            span = f"{_when(since, 'en', dated)}-{_when(now, 'en', dated)}"
            moved = bool(cams.get("moved"))
            if cams.get("told"):
                self._hand_on(CAMERAS_BACK, cameras_back_text(since, now, moved, self._lang()))
                log.info("Cameras-back notice: no picture %s (recorder moved: %s) -> sent", span,
                         "yes" if moved else "no")
            else:
                log.info("Camera outage over: no picture %s, never announced -> nothing to send", span)
            self._cams = {}
            self._save()
            return
        if now - latest < CAMERAS_OFFLINE_SEC:          # just started: too early to tell
            return
        if not cams.get("since"):
            cams["since"] = latest
            log.info("Camera outage: no picture from any of %d camera(s) since %s", len(last_frames),
                     _when(latest, "en", _with_date(latest, now)))
            self._save()
        since = _number(cams.get("since")) or latest
        if cams.get("announced") or now - since < CAMERAS_OUTAGE_SEC:
            return
        when = _when(since, "en", _with_date(since, now))
        cams["announced"] = now         # once per outage, also when switched off ...
        cams["told"] = self.on          # ... and the back line only after a sent notice
        self._save()
        if not self.on:
            log.info("Cameras-down notice: no picture since %s -> not sent (system_notices off)", when)
            return
        self._hand_on(CAMERAS_DOWN, cameras_down_text(since, now, self._lang()))
        log.info("Cameras-down notice: no picture from %d camera(s) since %s -> sent", len(last_frames), when)

    def recorder_moved(self, old: str, new: str, count: int) -> None:
        """recorder_finder.py pointed the cameras at a new address (the restart follows): kept for the back line."""
        with self._lock:
            self._cams["moved"] = True
            self._save()
        log.info("Camera outage: recorder moved %s -> %s (%d URL(s)); noted for the back line", old, new, count)

    # -- helpers ----------------------------------------------------------------------------------
    def _lang(self) -> str:
        try:
            return "he" if _he(self.lang()) else "en"
        except Exception:  # noqa: BLE001
            return "en"

    def _hand_on(self, kind: str, text: str) -> None:
        try:
            self.send(kind, text)
        except Exception as exc:  # noqa: BLE001
            log.warning("System notice (%s) not handed on: %s", kind, exc)

    def _load(self, key: str) -> Dict[str, Any]:
        value = (_read_json(os.path.join(self.state_dir, STATE_NAME)) or {}).get(key)
        return dict(value) if isinstance(value, dict) else {}

    def _save_eye(self) -> None:
        self._save()

    def _save(self) -> None:
        """Both outages in one file (an older file holds only "eye")."""
        try:
            _write_json(os.path.join(self.state_dir, STATE_NAME), {"eye": self._eye, "cameras": self._cams})
        except Exception as exc:  # noqa: BLE001
            log.warning("System notice state not saved: %s", exc)


# ----------------------------------------------------------------------------
# Sending
# ----------------------------------------------------------------------------
class Sender:
    """Sends the notices in order on one background thread, so a slow or absent network never holds inference up.

    *send(text)* returns True (sent), False (try again: no network yet) or None (never: Telegram not set up, dry
    run)."""

    def __init__(self, send: Callable[[str], Optional[bool]], tries: int = SEND_TRIES, wait: float = SEND_WAIT_SEC,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._send, self.tries, self.wait, self._sleep = send, max(1, int(tries)), wait, sleep
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def put(self, kind: str, text: str) -> None:
        self._queue.put((kind, text))
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="system-notices", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                kind, text = self._queue.get(timeout=5.0)
            except queue.Empty:
                return
            self.deliver(kind, text)

    def deliver(self, kind: str, text: str) -> bool:
        """Send one notice, with up to *tries* attempts. Never raises."""
        for attempt in range(1, self.tries + 1):
            try:
                result = self._send(text)
            except Exception as exc:  # noqa: BLE001
                log.warning("System notice (%s) send error: %s", kind, exc)
                result = False
            if result is True:
                log.info("System notice (%s) delivered (try %d)", kind, attempt)
                return True
            if result is None:
                log.warning("System notice (%s) not delivered: Telegram is not set up (or dry run)", kind)
                return False
            if attempt < self.tries:
                self._sleep(self.wait)
        log.warning("System notice (%s) not delivered after %d tries", kind, self.tries)
        return False


def telegram_send(box_settings: Mapping[str, Any], env: Mapping[str, str]) -> Callable[[str], Optional[bool]]:
    """The owner's Telegram chats, as the alerts reach them (telegram_notify.send_message: a normal message, no
    buttons). None from the result when Telegram is not the alert channel or is not set up."""
    from . import telegram_notify  # noqa: PLC0415

    channel = str((box_settings or {}).get("alert_channel", "telegram"))
    cfg = telegram_notify.load_telegram_config(dict(box_settings or {}), dict(env or {}))

    def send(text: str) -> Optional[bool]:
        if channel not in ("telegram", "both"):
            return None
        res = telegram_notify.send_message(cfg, text)
        if res.get("reason") in ("dry_run", "not_configured"):
            return None
        return bool(res.get("sent"))

    return send


def start(box_settings: Mapping[str, Any], env: Mapping[str, str], lang: Callable[[], str], provider: str = "",
          state_dir: Optional[str] = None, log_dir: Optional[str] = None) -> Optional[SystemNotices]:
    """At inference start: the notices with the box's folders and Telegram, and the box-down check done. None with
    ``HOMEGUARD_SYSTEM_NOTICES=off`` in the environment (the test suite: no alive file on the developer's disk)."""
    from . import paths  # noqa: PLC0415

    if str(os.environ.get("HOMEGUARD_SYSTEM_NOTICES") or "").strip().lower() in ("0", "off", "false", "no"):
        log.info("System notices not started (HOMEGUARD_SYSTEM_NOTICES off)")
        return None

    on = enabled(box_settings)
    sender = Sender(telegram_send(box_settings, env))
    notices = SystemNotices(state_dir or paths.state_dir(), log_dir or paths.logs_dir(), sender.put, lang,
                            provider=provider, on=on)
    log.info("System notices %s (box down after %.0f min; AI down after %.0f min and %d failed calls)",
             "on" if on else "off", DOWN_GAP_SEC / 60, AI_OUTAGE_SEC / 60, AI_OUTAGE_FAILS)
    notices.box_started()
    return notices
