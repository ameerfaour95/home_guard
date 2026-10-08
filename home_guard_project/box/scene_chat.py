"""Re-defining one camera's map over Telegram, on demand (owner's decisions 2026-10-08 22:55 and 23:10).

The map itself is set up in the box's setup (the app). Telegram only changes it when the owner asks:

    owner:  אני רוצה להגדיר מחדש את האזור של השכן במצלמה 3     (or "תגדיר אזור חדש", "מפה פרגולה", "/map")
    bot:    "אתה בטוח שאתה רוצה להגדיר מחדש את חניה? המפה הנוכחית ממשיכה לעבוד עד שתאשר את החדשה."
            [כן, להגדיר מחדש] [לא]
    bot:    [numbered picture] "חניה: מה משתנה? למשל: 4 של השכן, להסתיר את 9, המעקה בין 4 ל-9"   [בטל]
    owner:  4 של השכן
    bot:    [coloured picture] "ככה? כחול שלנו, כתום של השכן, ..."   [שמור] [תקן] [בטל]
    owner:  [שמור]   -> scene_interview.confirm: the change merges into the camera's map; the old one is the backup

The owner's rules (23:10), each tested:

1. It starts only when the owner ASKS to define or re-define an area ("תגדיר", "להגדיר מחדש", "לשנות את האזור",
   "מפה <camera>", "/map"). A message that only mentions a camera never starts it. With no camera named: one
   button per camera (camera_names), never a walk over all of them.
2. Before anything, "are you sure" with [כן, להגדיר מחדש] [לא]. Nothing starts without [כן].
3. Exit at any point: "עצור", "בטל", "לא משנה", or the [בטל] button every step carries. Exiting changes nothing,
   and the exit message says the current map stays as it is.
4. Until [שמור] the camera's live map and its mask are untouched: answers only write the draft.
5. Only [שמור] on the final coloured picture replaces the map; the map it replaced is kept (``scene_interview.
   restore_previous``), and "תחזיר את המפה הקודמת" brings it back after its own [כן, להחזיר] [לא], with a receipt.
6. A flow nobody moves on expires after SESSION_TTL (30 min); after that nothing is taken. Within it, a message
   is an answer only when it parses as a change of this picture (a number of it with an ownership, hide or line
   word); everything else goes to the assistant, with one short reminder at most.

From the owner's camera-1 answer (stage 2c): while the coloured picture waits, a message with a fix word
("שכחת", "תתקן", "לא נכון", "חסר") or a short one naming a number of the picture is a correction (added to the
answer, or the bot asks what that number is); a wall "between us" stays ours with a boundary, its side asked with
buttons when unclear; a number that made nothing is said back ("לא הבנתי את 7").

State: a small JSON file in the box's state folder (version 2, one session per chat). Anything else in it (the
multi-camera sessions of stage 2c, a damaged file) is dropped harmlessly. A change of the frame mask restarts the
running mode once, after the session is closed. The box CLI (``scene_interview telegram --camera X``) sends the
"are you sure" for camera X; the running inbox carries on.
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
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import paths
from . import scene_interview as si
from . import scene_map as sm

log = logging.getLogger("box.scene_chat")

STATE_NAME = "scene_chat.json"
STATE_VERSION = 2
SESSION_TTL = 1800.0              # a flow nobody moved on for this long is over (owner: 30 minutes)
CAPTION_LIMIT = 1024
SAVE, FIX, STOP, CHOOSE, SIDE, SURE, RESTORE = "sm:s", "sm:f", "sm:x", "sm:c", "sm:d", "sm:y", "sm:r"
FIX_WORDS_SHORT = 3               # a correction without a fix word is at most this many words ("7", "גם 7 שלי")

_DEFINE = re.compile(
    r"^\s*/map\b|^\s*(?:מפה|map)(?:\s+\S+){0,3}\s*[?!.]?\s*$"
    r"|(?:תגדיר|להגדיר|נגדיר|הגדר|תגדירו|תשנה|לשנות|נשנה|שנה|תעדכן|לעדכן|תסמן|לסמן)\s+(?:\S+\s+){0,4}?"
    r"(?:אזור|האזור|אזורים|מפה|המפה|מפת|מחדש)"
    r"|\b(?:set up|define|redefine|re-define|change|edit|update|mark)\s+(?:\S+\s+){0,4}?(?:area|zone|map)\b",
    re.IGNORECASE)
_RESTORE = re.compile(r"(?:תחזיר|להחזיר|החזר|תשחזר|לשחזר)\s+(?:לי\s+)?(?:את\s+)?(?:ה)?מפה\s+(?:ה)?קודמת"
                      r"|\b(?:restore|bring back)\s+the\s+(?:previous|old)\s+map\b", re.IGNORECASE)
_STOP_WORDS = re.compile(r"^\s*(?:עצור|תעצור|בטל|תבטל|לא משנה|די|stop|cancel|never ?mind)\s*[.!]?\s*$",
                         re.IGNORECASE)
_FIX_WORDS = re.compile(r"שכחת|תתקן|לתקן|לא נכון|חסר|טעות|\b(?:forgot|fix|wrong|missing)\b", re.IGNORECASE)
_CAMERA_NUMBER = re.compile(r"(?:מצלמה|camera|cam|ch)\s*-?\s*(\d+)", re.IGNORECASE)

TEXTS = {
    "he": {
        "which": "איזו מצלמה?",
        "sure": "אתה בטוח שאתה רוצה להגדיר מחדש את {name}? המפה הנוכחית ממשיכה לעבוד עד שתאשר את החדשה.",
        "yes_define": "כן, להגדיר מחדש", "no": "לא", "cancel": "בטל",
        "ask": "{name}: מה משתנה?\nלמשל: 4 של השכן, להסתיר את 9, המעקה בין 4 ל-9\nמה שלא תזכירו נשאר כמו שהוא.",
        "zone": "\nהקו הלבן: מה שנצפה עד היום.",
        "grid": "\n(המספרים על רשת: זיהוי האובייקטים לא זמין עכשיו.)",
        "confirm": "{name}: ככה? כחול שלנו, כתום של השכן, אפור רחוב, שחור מוסתר.",
        "ours": "שלנו", "neighbour": "של השכן", "public": "רחוב", "black": "מוסתר", "lines": "קווים",
        "rest": "כל השאר: {who}",
        "save": "שמור", "fix": "תקן",
        "saved": "נשמר: המפה החדשה של {name} פועלת. המפה הקודמת שמורה: אפשר לכתוב 'תחזיר את המפה הקודמת'.",
        "fixing": "בסדר. כתבו שוב מה משתנה ב{name}.",
        "stopped": "עצרתי. לא שיניתי כלום: המפה של {name} נשארת כמו שהיא.",
        "stopped_any": "עצרתי. לא שיניתי כלום.",
        "restart": " הצפייה מתחילה מחדש כדי שהשינוי ייכנס (כדקה).",
        "expired": "פג הזמן לשינוי הזה. לא שיניתי כלום: המפה של {name} נשארת כמו שהיא.",
        "no_picture": "לא הצלחתי לקבל תמונה מ{name} עכשיו. לא שיניתי כלום.",
        "no_cameras": "לא מצאתי מצלמות.",
        "reminder": "(הגדרת המפה של {name} עדיין פתוחה: כתבו למשל '4 של השכן', או 'בטל'.)",
        "which_side": "באיזה צד של {name} ({n}) השטח שלכם?",
        "what_is": "מה {nums}? כתבו למשל '{n} שלי' או '{n} של השכן'.", "what_fix": "מה לתקן? כתבו למשל '7 שלי'.",
        "restore_q": "להחזיר את המפה הקודמת של {name} (מ-{when})? המפה הנוכחית תישמר במקומה כגיבוי.",
        "yes_restore": "כן, להחזיר", "restored": "החזרתי את המפה הקודמת של {name}. היא פועלת עכשיו.",
        "no_previous": "אין מפה קודמת של {name}.",
        "notes": {"no_region": "אין {n} בתמונה", "side": "איזה צד של {name} שלכם? כתבו למשל '{a} שלי'",
                  "not_understood": "לא הבנתי את {n}",
                  "touch": "{a} ו-{b} לא נוגעים, אז אין ביניהם קו", "nothing": "עוד אין מה לשמור: כתבו מה משתנה"},
    },
    "en": {
        "which": "Which camera?",
        "sure": "Are you sure you want to re-define {name}? The current map keeps working until you confirm the new "
                "one.",
        "yes_define": "Yes, re-define", "no": "No", "cancel": "Cancel",
        "ask": "{name}: what changes?\nFor example: 4 the neighbour's, hide 9, the railing between 4 and 9\n"
               "What you do not mention stays as it is.",
        "zone": "\nThe white line: what was watched until now.",
        "grid": "\n(The numbers are on a grid: object finding is not available now.)",
        "confirm": "{name}: like this? Blue ours, orange the neighbour's, grey street, black hidden.",
        "ours": "ours", "neighbour": "the neighbour's", "public": "street", "black": "hidden", "lines": "lines",
        "rest": "everything else: {who}",
        "save": "Save", "fix": "Fix",
        "saved": "Saved: {name}'s new map is working. The previous one is kept: write 'restore the previous map'.",
        "fixing": "OK. Write again what changes in {name}.",
        "stopped": "Stopped. Nothing changed: {name}'s map stays as it is.", "stopped_any": "Stopped. Nothing changed.",
        "restart": " Watching restarts so the change takes hold (about a minute).",
        "expired": "This change timed out. Nothing changed: {name}'s map stays as it is.",
        "no_picture": "I could not get a picture from {name} now. Nothing changed.",
        "no_cameras": "I found no cameras.",
        "reminder": "(Re-defining {name}'s map is still open: write for example '4 the neighbour's', or 'cancel'.)",
        "which_side": "Which side of {name} ({n}) is yours?",
        "what_is": "What is {nums}? Write for example '{n} mine' or '{n} the neighbour's'.",
        "what_fix": "What should change? Write for example '7 mine'.",
        "restore_q": "Bring back {name}'s previous map (from {when})? The current one is kept as the backup instead.",
        "yes_restore": "Yes, bring it back", "restored": "{name}'s previous map is back and working.",
        "no_previous": "There is no previous map of {name}.",
        "notes": {"no_region": "there is no {n} in the picture", "not_understood": "I did not understand {n}",
                  "side": "which side of {name} is yours? Write for example '{a} mine'",
                  "touch": "{a} and {b} do not touch, so there is no line between them",
                  "nothing": "nothing to save yet: say what changes"},
    },
}


def _t(lang: str) -> Dict[str, Any]:
    return TEXTS["he"] if str(lang).startswith("he") else TEXTS["en"]


def is_trigger(text: str) -> bool:
    """The owner asks to define or re-define a camera's area ("תגדיר אזור חדש", "להגדיר מחדש", "מפה פרגולה",
    "/map"). Only mentioning a camera is not asking."""
    return bool(_DEFINE.search(str(text or "")))


def is_restore(text: str) -> bool:
    """"תחזיר את המפה הקודמת": the owner wants the map before the last [שמור] back."""
    return bool(_RESTORE.search(str(text or "")))


def owner_notes(notes: Sequence[str], lang: str) -> List[str]:
    """The interview's notes (English, from scene_interview) in the owner's language."""
    words = _t(lang)["notes"]
    out = []
    for note in notes:
        m = re.match(r"there is no (\d+) in the picture", note)
        if m:
            out.append(words["no_region"].format(n=m.group(1)))
            continue
        m = re.match(r"which side of (.+) is yours\? Say (\d+) or (\d+) is yours", note)
        if m:
            out.append(words["side"].format(name=m.group(1), a=m.group(2)))
            continue
        m = re.match(r"(\d+) and (\d+) do not touch", note)
        if m:
            out.append(words["touch"].format(a=m.group(1), b=m.group(2)))
            continue
        m = re.match(r"I did not understand (\d+)", note)
        if m:
            out.append(words["not_understood"].format(n=m.group(1)))
            continue
        if re.match(r"which side of .+ \(\d+\) is yours\?", note):
            continue                                   # asked with buttons
        out.append(words["nothing"] if note.startswith("nothing to save") else note)
    return out


def map_lines(scene: sm.SceneMap, lang: str) -> List[str]:
    """The confirmation picture's words: which areas are whose, in the owner's own names."""
    words = _t(lang)
    groups: Dict[str, List[str]] = {"ours": [], "neighbour": [], "public": [], "black": []}
    for a in scene.areas:
        if a.kind == sm.BLACK:
            groups["black"].append(a.name)
        elif a.name not in (sm.WATCHED_NAME,):
            groups["ours" if a.ground == sm.MINE else a.ground].append(a.name)
    out = [f"{words[k]}: {', '.join(v)}" for k, v in groups.items() if v]
    if scene.lines:
        out.append(f"{words['lines']}: {', '.join(ln.name for ln in scene.lines)}")
    if scene.rest:
        out.append(words["rest"].format(who=words["neighbour" if scene.rest_owner == sm.NEIGHBOUR else "public"]))
    return out


def is_answer(text: str, numbers: Sequence[int]) -> bool:
    """*text* is a change of the picture whose regions are *numbers*: every number it gives an ownership, hide or
    line word to is one of them ("4 של השכן", "להסתיר את 9", "המעקה בין 4 ל-9"). "יש 2 אנשים בחוץ?" is not."""
    known = {int(n) for n in numbers}
    said = [a.number for a in si.parse_answers(text)] + [n for _, a, b in si.parse_lines(text) for n in (a, b)]
    return bool(said) and all(n in known for n in said)


class SceneChat:
    """One re-definition of one camera's map per chat; the state in a JSON file (the box's CLI can start one)."""

    def __init__(self, send_photo: Callable[[str, str, str, List[List[tuple]]], Any],
                 send_text: Callable[[str, str, List[List[tuple]]], Any], cameras: Callable[[], Sequence[str]],
                 lang: Callable[[], str] = lambda: "he", state_path: Optional[str] = None,
                 out_dir: Optional[str] = None, cameras_path: Optional[str] = None, zones_path: Optional[str] = None,
                 picture: Optional[Callable[[str], Any]] = None, segmenter: Any = None,
                 restart: Optional[Callable[[], None]] = None, background: bool = True,
                 now: Callable[[], float] = time.time) -> None:
        self.send_photo, self.send_text, self.cameras, self.lang = send_photo, send_text, cameras, lang
        self.state_path = state_path or os.path.join(paths.state_dir(), STATE_NAME)
        self.out_dir = out_dir or si.INTERVIEW_DIR
        self.cameras_path, self.zones_path = cameras_path, zones_path
        self._picture = picture
        self.segmenter = segmenter
        self._restart = restart
        self.background = background
        self.now = now
        self._lock = threading.RLock()

    # ---------- state ----------
    def _load(self) -> Dict[str, Any]:
        """The sessions of this version; anything else (stage 2c's multi-camera sessions, a damaged file) is
        dropped."""
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): v for k, v in data.items()
                if isinstance(v, dict) and v.get("v") == STATE_VERSION and isinstance(v.get("camera"), str)}

    def _save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.state_path)

    def session(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """The chat's open session, or None (none, an old format, or nobody moved it on for SESSION_TTL)."""
        s = self._load().get(str(chat_id))
        if s is None:
            return None
        try:
            if self.now() - float(s.get("last_bot") or 0) > SESSION_TTL:
                return None
        except (TypeError, ValueError):
            return None
        return s

    def _put(self, chat_id: str, s: Optional[Dict[str, Any]]) -> None:
        try:
            data = self._load()
            if s is None:
                data.pop(str(chat_id), None)
            else:
                data[str(chat_id)] = dict(s, v=STATE_VERSION)
            self._save(data)
        except OSError as exc:
            log.warning("interview state not saved: %s", exc)

    # ---------- what the owner sends ----------
    def on_text(self, chat_id: str, text: str) -> bool:
        """True when *text* belonged to the map flow (a request, an answer, an exit); else the assistant reads it."""
        chat_id, text = str(chat_id), str(text or "").strip()
        with self._lock:
            s = self.session(chat_id)
            if s is not None and _STOP_WORDS.match(text):
                self._cancel(chat_id, s)
                return True
            if is_restore(text):
                self._request(chat_id, text, "restore")
                return True
            if is_trigger(text):
                self._request(chat_id, text, "define")
                return True
            if s is None:
                return False
            numbers = s.get("numbers") or []
            if s.get("stage") in ("answer", "side") and is_answer(text, numbers):
                self._answer(chat_id, s, text, append=bool(s.get("append")) or s.get("stage") == "side")
                return True
            if s.get("stage") == "confirm" and self._correction(text, numbers):
                if is_answer(text, numbers):
                    self._answer(chat_id, s, text, append=True)
                else:
                    named = [n for n in si.mentioned_numbers(text) if n in numbers]
                    s.update(stage="answer", append=True, last_bot=self.now())
                    self._put(chat_id, s)
                    words = _t(self.lang())
                    self._say(chat_id, words["what_is"].format(nums=", ".join(map(str, named)), n=named[0])
                              if named else words["what_fix"], self._cancel_row(s))
                return True
            if s.get("camera") and not s.get("reminded"):
                s["reminded"] = True
                self._put(chat_id, s)
                self._say(chat_id, _t(self.lang())["reminder"].format(name=self._name(s["camera"])))
            return False

    @staticmethod
    def _correction(text: str, numbers: Sequence[int]) -> bool:
        """While the coloured picture waits: a fix word, or a short message naming a number of the picture."""
        if _FIX_WORDS.search(text) or is_answer(text, numbers):
            return True
        named = [n for n in si.mentioned_numbers(text) if n in set(numbers)]
        return bool(named) and len(text.split()) <= FIX_WORDS_SHORT

    def on_button(self, chat_id: str, code: str) -> bool:
        chat_id = str(chat_id)
        parts = str(code or "").split(":")
        if len(parts) < 3 or parts[0] != "sm":
            return False
        with self._lock:
            s = self.session(chat_id)
            words = _t(self.lang())
            if s is None or parts[2] != s.get("token"):
                old = self._load().get(chat_id)
                if old is not None and parts[2] == old.get("token"):
                    self._put(chat_id, None)            # this session's own button, after its time
                    self._say(chat_id, words["expired"].format(name=self._name(old.get("camera") or "")))
                return True
            action = f"sm:{parts[1]}"
            stage = s.get("stage")
            if action == STOP:
                self._cancel(chat_id, s)
            elif action == CHOOSE and stage == "choose" and len(parts) == 4:
                try:
                    camera = (s.get("choices") or [])[int(parts[3])]
                except (ValueError, IndexError):
                    return True
                self._after_choice(chat_id, camera, s.get("purpose") or "define")
            elif action == SURE and stage == "sure":
                self.begin(chat_id, s["camera"])
            elif action == RESTORE and stage == "restore":
                self._restore(chat_id, s)
            elif action == SIDE and stage == "side" and len(parts) == 5 and parts[4] in sm.SIDES:
                sides = dict(s.get("sides") or {})
                sides[str(parts[3])] = parts[4]
                s["sides"] = sides
                self._answer(chat_id, s, "", append=True)
            elif action == SAVE and stage == "confirm":
                self._confirm(chat_id, s)
            elif action == FIX and stage == "confirm":
                s.update(stage="answer", last_bot=self.now(), append=False, text="", sides={})
                self._put(chat_id, s)
                self._say(chat_id, words["fixing"].format(name=self._name(s["camera"])), self._cancel_row(s))
            return True

    # ---------- the steps ----------
    def _request(self, chat_id: str, text: str, purpose: str) -> None:
        """A request to re-define (or restore) a map: the camera named, the box's only one, or a choice."""
        named = self._named(text) or [c for c in self.cameras() if c]
        if len(named) == 1:
            self._after_choice(chat_id, named[0], purpose)
        else:
            self.ask_camera(chat_id, named, purpose)

    def ask_camera(self, chat_id: str, cameras: Sequence[str], purpose: str = "define") -> None:
        """Which camera? One button each, by the family's names, and [בטל]."""
        words = _t(self.lang())
        cameras = [c for c in cameras if c]
        if not cameras:
            self._say(chat_id, words["no_cameras"])
            return
        s = {"camera": "", "choices": list(cameras), "purpose": purpose, "stage": "choose",
             "token": uuid.uuid4().hex[:8], "last_bot": self.now()}
        self._put(chat_id, s)
        rows = [[(self._name(c), f"{CHOOSE}:{s['token']}:{i}")] for i, c in enumerate(cameras)]
        self._say(chat_id, words["which"], rows + self._cancel_row(s))

    def _after_choice(self, chat_id: str, camera: str, purpose: str) -> None:
        if purpose == "restore":
            self.ask_restore(chat_id, camera)
        else:
            self.ask_sure(chat_id, camera)

    def ask_sure(self, chat_id: str, camera: str) -> None:
        """Rule 2: are you sure? Nothing starts without [כן]."""
        words = _t(self.lang())
        s = {"camera": camera, "stage": "sure", "token": uuid.uuid4().hex[:8], "last_bot": self.now()}
        self._put(chat_id, s)
        self._say(chat_id, words["sure"].format(name=self._name(camera)),
                  [[(words["yes_define"], f"{SURE}:{s['token']}"), (words["no"], f"{STOP}:{s['token']}")]])

    def ask_restore(self, chat_id: str, camera: str) -> None:
        """Rule 5: bring the previous map back? Its own [כן, להחזיר] [לא]."""
        words = _t(self.lang())
        previous = si.previous_map(camera, self.zones_path)
        if previous is None:
            self._put(chat_id, None)
            self._say(chat_id, words["no_previous"].format(name=self._name(camera)))
            return
        try:
            when = dt.datetime.fromtimestamp(float(previous.get("saved_at") or 0)).strftime("%d.%m %H:%M")
        except (TypeError, ValueError, OverflowError, OSError):
            when = "?"
        s = {"camera": camera, "stage": "restore", "token": uuid.uuid4().hex[:8], "last_bot": self.now()}
        self._put(chat_id, s)
        self._say(chat_id, words["restore_q"].format(name=self._name(camera), when=when),
                  [[(words["yes_restore"], f"{RESTORE}:{s['token']}"), (words["no"], f"{STOP}:{s['token']}")]])

    def begin(self, chat_id: str, camera: str) -> None:
        """After [כן]: the numbered picture (made in the background). The live map is not touched."""
        s = {"camera": camera, "stage": "preparing", "token": uuid.uuid4().hex[:8], "last_bot": self.now(),
             "started": self.now()}
        self._put(chat_id, s)
        if self.background:
            threading.Thread(target=self._ask, args=(chat_id, dict(s)), name="scene-interview", daemon=True).start()
        else:
            self._ask(chat_id, dict(s))

    def _ask(self, chat_id: str, s: Dict[str, Any]) -> None:
        camera, words = s["camera"], _t(self.lang())
        try:
            picture, picture_path = (self._picture or self._grab)(camera)
            watched = sm.load_scene_map(camera, self.zones_path).watched
            proposal = si.propose(camera, picture, self.out_dir, self.segmenter, picture_path, watched=watched)
        except Exception as exc:  # noqa: BLE001 - a camera that cannot be seen ends this change, nothing else
            log.warning("[%s] interview picture not made: %s", camera, exc)
            with self._lock:
                current = self.session(chat_id)
                if current is not None and current.get("token") == s["token"]:
                    self._put(chat_id, None)
            self._say(chat_id, words["no_picture"].format(name=self._name(camera)))
            return
        caption = words["ask"].format(name=self._name(camera))
        if watched:
            caption += words["zone"]
        if proposal["method"] == "grid":
            caption += words["grid"]
        with self._lock:
            current = self.session(chat_id)
            if current is None or current.get("token") != s["token"]:
                return                          # cancelled while the picture was made
            current.update(stage="answer", regions_path=proposal["regions_path"], picture=picture_path,
                           numbers=[r["number"] for r in proposal["regions"]], last_bot=self.now())
            self._put(chat_id, current)
        self.send_photo(chat_id, proposal["image"], caption[:CAPTION_LIMIT], self._cancel_row(s))

    def _answer(self, chat_id: str, s: Dict[str, Any], text: str, append: bool = False) -> None:
        """The owner's words (added to what they already said when *append*) into the DRAFT only; then the side
        of a wall when unclear, else the coloured picture with [שמור] / [תקן] / [בטל]."""
        camera, words = s["camera"], _t(self.lang())
        said = "\n".join(t for t in ((s.get("text") or "") if append else "", text) if t)
        s.update(text=said, append=False)
        sides = {int(k): v for k, v in (s.get("sides") or {}).items() if str(k).isdigit()}
        picture = si._read_picture(s.get("picture") or "")
        result = si.answer(camera, said, s["regions_path"], picture, self.out_dir, self.zones_path, sides=sides)
        if result.get("sides_needed"):
            need = result["sides_needed"][0]
            names = si.side_words(tuple(need["line"][0]), tuple(need["line"][1]), self.lang())
            s.update(stage="side", last_bot=self.now())
            self._put(chat_id, s)
            rows = [[(names["left"], f"{SIDE}:{s['token']}:{need['number']}:left"),
                     (names["right"], f"{SIDE}:{s['token']}:{need['number']}:right")]] + self._cancel_row(s)
            question = words["which_side"].format(name=need["name"], n=need["number"])
            if result["image"]:
                self.send_photo(chat_id, result["image"], question, rows)
            else:
                self._say(chat_id, question, rows)
            return
        preview = sm.SceneMap.from_dict(camera, result["map"])
        lines = [words["confirm"].format(name=self._name(camera))] + map_lines(preview, self.lang())
        lines += owner_notes(result["notes"], self.lang())
        s.update(stage="confirm", last_bot=self.now())
        self._put(chat_id, s)
        rows = [[(words["save"], f"{SAVE}:{s['token']}"), (words["fix"], f"{FIX}:{s['token']}")]] + self._cancel_row(s)
        caption = "\n".join(lines)[:CAPTION_LIMIT]
        if result["image"]:
            self.send_photo(chat_id, result["image"], caption, rows)
        else:
            self._say(chat_id, caption, rows)

    def _confirm(self, chat_id: str, s: Dict[str, Any]) -> None:
        """Rule 5: only [שמור] on the final picture replaces the map (the old one becomes the backup)."""
        camera, words = s["camera"], _t(self.lang())
        result = si.confirm(camera, self.out_dir, self.zones_path, known_cameras=self._all_cameras())
        self._put(chat_id, None)
        text = words["saved"].format(name=self._name(camera))
        if result["restart_needed"]:
            text += words["restart"]
        self._say(chat_id, text)
        if result["restart_needed"]:
            (self._restart or _request_restart)()

    def _restore(self, chat_id: str, s: Dict[str, Any]) -> None:
        camera, words = s["camera"], _t(self.lang())
        self._put(chat_id, None)
        try:
            result = si.restore_previous(camera, self.zones_path)
        except LookupError:
            self._say(chat_id, words["no_previous"].format(name=self._name(camera)))
            return
        text = words["restored"].format(name=self._name(camera))
        if result["restart_needed"]:
            text += words["restart"]
        self._say(chat_id, text)
        if result["restart_needed"]:
            (self._restart or _request_restart)()

    def _cancel(self, chat_id: str, s: Dict[str, Any]) -> None:
        """Rule 3: exiting changes nothing; the draft is dropped and the message says the map stays."""
        words = _t(self.lang())
        self._put(chat_id, None)
        camera = s.get("camera") or ""
        if camera:
            try:
                os.remove(si.draft_path(camera, self.out_dir))
            except OSError:
                pass
        self._say(chat_id, words["stopped"].format(name=self._name(camera)) if camera else words["stopped_any"])

    # ---------- the Telegram inbox's one routing line each ----------
    def inbox_text(self, inbox: Any, chat_id: str, sender: Dict[str, Any], text: str) -> bool:
        """``on_text`` for the inbox: True when the message was the map flow's (noted in the chat feed). Never
        raises: a failing flow hands the message to the assistant."""
        try:
            taken = self.on_text(chat_id, text)
        except Exception as exc:  # noqa: BLE001 - the flow must never lose the owner's message
            log.warning("Map change failed: %s", exc)
            return False
        if taken:
            name = str(sender.get("first_name") or sender.get("username") or "") if isinstance(sender, dict) else ""
            inbox._note("owner", "message", text, name)
        return taken

    def inbox_button(self, inbox: Any, query: Dict[str, Any], chat_id: str, code: str) -> None:
        """An "sm:" button in the inbox: stop the spinner, note it, act. Never raises."""
        try:
            inbox._post(inbox.cfg.bot_token, "answerCallbackQuery", {"callback_query_id": str(query.get("id"))})
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not stop the button's spinner: %s", exc)
        try:
            inbox._note("owner", "button", code, str((query.get("from") or {}).get("first_name") or ""))
            self.on_button(chat_id, code)
        except Exception as exc:  # noqa: BLE001
            log.warning("Map change button failed: %s", exc)

    # ---------- helpers ----------
    def _cancel_row(self, s: Dict[str, Any]) -> List[List[tuple]]:
        return [[(_t(self.lang())["cancel"], f"{STOP}:{s['token']}")]]

    def _say(self, chat_id: str, text: str, rows: Optional[List[List[tuple]]] = None) -> None:
        try:
            self.send_text(chat_id, text, rows or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("map change message not sent: %s", exc)

    def _name(self, camera: str) -> str:
        from .camera_names import display_name  # noqa: PLC0415

        return display_name(camera, self.lang())

    def _named(self, text: str) -> List[str]:
        """The camera(s) the owner named: by the family's name, or by number ("מצלמה 3", "camera 3")."""
        from .camera_names import channel_of, family_names  # noqa: PLC0415

        cameras = list(self.cameras())
        numbers = {m.group(1) for m in _CAMERA_NUMBER.finditer(text)}
        return [c for c in cameras
                if channel_of(c) in numbers or any(n and n in text for n in family_names(c) + [self._name(c)])]

    def _all_cameras(self) -> List[str]:
        """Every camera of the box, the disabled ones too (their zones are not stale)."""
        try:
            from .find_cameras import CAMERAS_PATH  # noqa: PLC0415

            known = si._known_cameras(self.cameras_path or CAMERAS_PATH)
        except Exception:  # noqa: BLE001
            known = []
        return sorted(set(known) | set(self.cameras()))

    def _grab(self, camera: str) -> Any:
        from .find_cameras import CAMERAS_PATH  # noqa: PLC0415

        return si.interview_picture(camera, self.cameras_path or CAMERAS_PATH, self.out_dir)


def _request_restart() -> None:
    try:
        from . import control  # noqa: PLC0415

        control.request_restart()
    except Exception as exc:  # noqa: BLE001
        log.warning("restart not requested: %s", exc)


def telegram_senders(token: str, post: Callable[..., Any], post_multipart: Callable[..., Any]) -> tuple:
    """``(send_photo, send_text)`` over the Bot API with *post* / *post_multipart* (telegram_notify's)."""

    def markup(rows: List[List[tuple]]) -> Dict[str, str]:
        if not rows:
            return {}
        return {"reply_markup": json.dumps({"inline_keyboard": [[{"text": str(label)[:60], "callback_data": str(code)}
                                                                  for label, code in row] for row in rows]})}

    def send_photo(chat_id: str, path: str, caption: str, rows: List[List[tuple]]) -> Any:
        with open(path, "rb") as f:
            data = f.read()
        return post_multipart(token, "sendPhoto", {"chat_id": str(chat_id), "caption": caption, **markup(rows)},
                              {"photo": (os.path.basename(path), data, "image/jpeg")}, timeout=60.0)

    def send_text(chat_id: str, text: str, rows: List[List[tuple]]) -> Any:
        return post(token, "sendMessage", {"chat_id": str(chat_id), "text": text, **markup(rows)})

    return send_photo, send_text


def for_inbox(inbox: Any, cameras: Callable[[], Sequence[str]], lang: Callable[[], str]) -> SceneChat:
    """The map flow the running Telegram inbox carries on (telegram_agent.start)."""
    send_photo, _ = telegram_senders(inbox.cfg.bot_token, inbox._post, inbox._post_multipart)

    def send_text(chat_id: str, text: str, rows: List[List[tuple]]) -> Any:
        return inbox._say(chat_id, text, rows=rows)

    return SceneChat(send_photo, send_text, cameras, lang)


def start_from_cli(camera: str = "") -> Dict[str, Any]:
    """The box's CLI: ask the owner whether to re-define *camera*'s map (one camera, always named). The running
    inbox carries on with the owner's [כן] and answer."""
    if not camera:
        return {"error": "name the camera: --camera X"}
    from dotenv import load_dotenv  # noqa: PLC0415

    from . import telegram_notify  # noqa: PLC0415
    from .boxconfig import load_box_settings  # noqa: PLC0415
    from .find_cameras import CAMERAS_PATH, _known_camera, _read_cameras_raw  # noqa: PLC0415
    from .telegram_agent import box_language  # noqa: PLC0415

    load_dotenv(paths.secrets_env())
    cfg = telegram_notify.load_telegram_config(load_box_settings(), dict(os.environ))
    if not cfg.enabled or not cfg.chat_ids:
        return {"error": "Telegram is not set up on this box"}
    active = list(dict(_read_cameras_raw(CAMERAS_PATH).get("cameras") or {}))
    name = _known_camera(camera, CAMERAS_PATH)
    send_photo, send_text = telegram_senders(cfg.bot_token, telegram_notify._http_post,
                                             telegram_notify._http_post_multipart)
    SceneChat(send_photo, send_text, lambda: active, box_language, background=False).ask_sure(cfg.chat_ids[0], name)
    return {"chat": cfg.chat_ids[0], "camera": name}
