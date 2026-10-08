"""The install interview over Telegram (scene map stage 2c): per camera a numbered picture, the owner's words, the
coloured picture back with [שמור] [תקן]. Nothing is saved before [שמור].

    owner:  מפה                      (or "בוא נגדיר את המצלמות", "/map"; with a camera's name: only that camera)
    bot:    [numbered picture] "פרגולה (1/6): מה כל מספר ושל מי? למשל: 1 שלי, 2 של השכן, 3 רחוב, המעקה בין 2 ל-1"
    owner:  1 שלי, 2 של השכן, 3 רחוב, המעקה בין 2 ל-1
    bot:    [coloured picture] "ככה? כחול שלנו, כתום של השכן, אפור רחוב, שחור מוסתר" [שמור] [תקן]
    owner:  [שמור]  -> scene_interview.confirm (the map becomes the camera's whole truth), next camera

"דלג" skips a camera, "עצור" ends the interview. Camera names only through ``camera_names.display_name`` (owner
rule: never an id). A confirmed map that changes the frame mask (a zone's black outside opened, black areas
changed) needs the running mode restarted; that waits until the interview ends, so the inbox is not restarted in
the middle of it.

The state is a small JSON file in the box's state folder, so the CLI on the box
(``python -m home_guard_project.box.scene_interview telegram [--camera X]``) can start an interview that the
running inbox then carries on: the CLI sends the first picture and exits; the owner's answers reach the inbox.
"""

from __future__ import annotations

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
SESSION_TTL = 6 * 3600.0          # an interview nobody answered for this long is over
CAPTION_LIMIT = 1024
SAVE, FIX, SKIP, STOP = "sm:s", "sm:f", "sm:k", "sm:x"

_TRIGGER = re.compile(r"(?:^|\s|/)(?:מפה|מפת המצלמות|map)(?=\s|$|[?!.,])|בוא(?:ו)? נגדיר את המצלמות|"
                      r"נגדיר את המצלמות|להגדיר את המצלמות|scene map|set up the cameras", re.IGNORECASE)
_STOP_WORDS = re.compile(r"^\s*(?:עצור|תעצור|בטל|די|stop|cancel)\s*[.!]?\s*$", re.IGNORECASE)
_SKIP_WORDS = re.compile(r"^\s*(?:דלג|תדלג|skip)\s*[.!]?\s*$", re.IGNORECASE)

TEXTS = {
    "he": {
        "intro": "נגדיר מה כל מצלמה רואה: מה שלנו, מה של השכן ומה רחוב. {n} מצלמות. בכל שלב אפשר לכתוב 'דלג' או 'עצור'.",
        "ask": "{name} ({i}/{n}): מה כל מספר, ושל מי?\nלמשל: 1 שלי, 2 של השכן, 3 רחוב, המעקה בין 2 ל-1\n"
               "אפשר גם להסתיר: 4 להסתיר.",
        "zone": "\nהקו הלבן: מה שנצפה עד היום. מה שמחוצה לו ולא תסמנו יהיה של השכן: רואים, לא מתריעים.",
        "grid": "\n(המספרים על רשת: זיהוי האובייקטים לא זמין עכשיו.)",
        "confirm": "{name}: ככה? כחול שלנו, כתום של השכן, אפור רחוב, שחור מוסתר.",
        "ours": "שלנו", "neighbour": "של השכן", "public": "רחוב", "black": "מוסתר", "lines": "קווים",
        "rest": "כל השאר: {who}",
        "save": "שמור", "fix": "תקן", "skip_btn": "דלג", "stop_btn": "עצור",
        "saved": "נשמר: {name}.", "fixing": "בסדר. כתבו שוב את התשובה ל{name}.",
        "skipped": "דילגתי על {name}.", "stopped": "עצרתי. מה שכבר נשמר נשאר.",
        "done": "סיימנו. המפה בתוקף מעכשיו.", "restart": " הצפייה מתחילה מחדש כדי שהשינוי ייכנס (כדקה).",
        "no_picture": "לא הצלחתי לקבל תמונה מ{name} עכשיו. ממשיך.",
        "no_cameras": "לא מצאתי מצלמות להגדיר.", "preparing": "מכין תמונה של {name}...",
        "notes": {"no_region": "אין {n} בתמונה", "side": "איזה צד של {name} שלכם? כתבו למשל '{a} שלי'",
                  "touch": "{a} ו-{b} לא נוגעים, אז אין ביניהם קו", "nothing": "עוד אין מה לשמור: כתבו של מי כל מספר"},
    },
    "en": {
        "intro": "Let's set what each camera sees: what is ours, what is the neighbour's and what is the street. "
                 "{n} cameras. Write 'skip' or 'stop' at any time.",
        "ask": "{name} ({i}/{n}): what is each number, and whose is it?\nFor example: 1 mine, 2 the neighbour's, "
               "3 street, the railing between 2 and 1\nYou can also hide one: 4 hide.",
        "zone": "\nThe white line: what was watched until now. What is outside it and not marked becomes the "
                "neighbour's: seen, no alerts.",
        "grid": "\n(The numbers are on a grid: object finding is not available now.)",
        "confirm": "{name}: like this? Blue ours, orange the neighbour's, grey street, black hidden.",
        "ours": "ours", "neighbour": "the neighbour's", "public": "street", "black": "hidden", "lines": "lines",
        "rest": "everything else: {who}",
        "save": "Save", "fix": "Fix", "skip_btn": "Skip", "stop_btn": "Stop",
        "saved": "Saved: {name}.", "fixing": "OK. Write the answer for {name} again.",
        "skipped": "Skipped {name}.", "stopped": "Stopped. What was saved stays.",
        "done": "Done. The map is in force from now.", "restart": " Watching restarts so the change takes hold "
                                                                  "(about a minute).",
        "no_picture": "I could not get a picture from {name} now. Moving on.",
        "no_cameras": "I found no cameras to set up.", "preparing": "Preparing a picture of {name}...",
        "notes": {"no_region": "there is no {n} in the picture", "side": "which side of {name} is yours? Write "
                  "for example '{a} mine'", "touch": "{a} and {b} do not touch, so there is no line between them",
                  "nothing": "nothing to save yet: say whose each number is"},
    },
}


def _t(lang: str) -> Dict[str, Any]:
    return TEXTS["he"] if str(lang).startswith("he") else TEXTS["en"]


def is_trigger(text: str) -> bool:
    """The owner asks to set up the cameras' map ("מפה", "בוא נגדיר את המצלמות", "/map")."""
    return bool(_TRIGGER.search(str(text or "")))


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


class SceneChat:
    """One interview per chat, its state in a JSON file (so the box's CLI can start one the inbox carries on)."""

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
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.state_path)

    def session(self, chat_id: str) -> Optional[Dict[str, Any]]:
        s = self._load().get(str(chat_id))
        if not isinstance(s, dict) or self.now() - float(s.get("touched") or 0) > SESSION_TTL:
            return None
        return s

    def _put(self, chat_id: str, s: Optional[Dict[str, Any]]) -> None:
        data = self._load()
        if s is None:
            data.pop(str(chat_id), None)
        else:
            s["touched"] = self.now()
            data[str(chat_id)] = s
        self._save(data)

    # ---------- what the owner sends ----------
    def on_text(self, chat_id: str, text: str) -> bool:
        """True when *text* belonged to the interview (a start, an answer, skip or stop); else the assistant reads it."""
        chat_id, text = str(chat_id), str(text or "").strip()
        with self._lock:
            s = self.session(chat_id)
            if s is None:
                if not is_trigger(text):
                    return False
                self.start(chat_id, self._named(text))
                return True
            if _STOP_WORDS.match(text):
                self._finish(chat_id, s, stopped=True)
                return True
            if _SKIP_WORDS.match(text):
                self._say(chat_id, _t(self.lang())["skipped"].format(name=self._name(s["camera"])))
                self._next(chat_id, s)
                return True
            if is_trigger(text):
                self.start(chat_id, self._named(text))
                return True
            if s.get("stage") == "answer" and re.search(r"\d", text):
                self._answer(chat_id, s, text)
                return True
            return False

    def on_button(self, chat_id: str, code: str) -> bool:
        chat_id = str(chat_id)
        parts = str(code or "").split(":")
        if len(parts) != 3 or parts[0] != "sm":
            return False
        with self._lock:
            s = self.session(chat_id)
            if s is None or parts[2] != s.get("token"):
                return True                     # an old button: nothing to do
            action = f"sm:{parts[1]}"
            words = _t(self.lang())
            if action == SAVE and s.get("stage") == "confirm":
                self._confirm(chat_id, s)
            elif action == FIX and s.get("stage") == "confirm":
                s["stage"] = "answer"
                self._put(chat_id, s)
                self._say(chat_id, words["fixing"].format(name=self._name(s["camera"])))
            elif action == SKIP:
                self._say(chat_id, words["skipped"].format(name=self._name(s["camera"])))
                self._next(chat_id, s)
            elif action == STOP:
                self._finish(chat_id, s, stopped=True)
            return True

    # ---------- the steps ----------
    def start(self, chat_id: str, cameras: Optional[Sequence[str]] = None) -> None:
        chat_id = str(chat_id)
        queue = [c for c in (cameras or list(self.cameras())) if c]
        words = _t(self.lang())
        if not queue:
            self._say(chat_id, words["no_cameras"])
            return
        s = {"cameras": queue, "index": -1, "camera": "", "stage": "", "token": "", "restart": False,
             "started": self.now()}
        self._put(chat_id, s)
        if len(queue) > 1:
            self._say(chat_id, words["intro"].format(n=len(queue)))
        self._next(chat_id, s)

    def _next(self, chat_id: str, s: Dict[str, Any]) -> None:
        s["index"] = int(s.get("index", -1)) + 1
        if s["index"] >= len(s["cameras"]):
            self._finish(chat_id, s)
            return
        s.update(camera=s["cameras"][s["index"]], stage="preparing", token=uuid.uuid4().hex[:8])
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
        except Exception as exc:  # noqa: BLE001 - one camera that cannot be seen must not end the interview
            log.warning("[%s] interview picture not made: %s", camera, exc)
            with self._lock:
                self._say(chat_id, words["no_picture"].format(name=self._name(camera)))
                current = self.session(chat_id)
                if current is not None and current.get("token") == s["token"]:
                    self._next(chat_id, current)
            return
        caption = words["ask"].format(name=self._name(camera), i=s["index"] + 1, n=len(s["cameras"]))
        if watched:
            caption += words["zone"]
        if proposal["method"] == "grid":
            caption += words["grid"]
        with self._lock:
            current = self.session(chat_id)
            if current is None or current.get("token") != s["token"]:
                return                          # stopped or skipped while the picture was made
            current.update(stage="answer", regions_path=proposal["regions_path"], picture=picture_path)
            self._put(chat_id, current)
        self.send_photo(chat_id, proposal["image"], caption[:CAPTION_LIMIT],
                        [[(words["skip_btn"], f"{SKIP}:{s['token']}"), (words["stop_btn"], f"{STOP}:{s['token']}")]])

    def _answer(self, chat_id: str, s: Dict[str, Any], text: str) -> None:
        camera, words = s["camera"], _t(self.lang())
        picture = si._read_picture(s.get("picture") or "")
        result = si.answer(camera, text, s["regions_path"], picture, self.out_dir, self.zones_path)
        preview = sm.SceneMap.from_dict(camera, result["map"])
        lines = [words["confirm"].format(name=self._name(camera))] + map_lines(preview, self.lang())
        lines += owner_notes(result["notes"], self.lang())
        s["stage"] = "confirm"
        self._put(chat_id, s)
        rows = [[(words["save"], f"{SAVE}:{s['token']}"), (words["fix"], f"{FIX}:{s['token']}")]]
        caption = "\n".join(lines)[:CAPTION_LIMIT]
        if result["image"]:
            self.send_photo(chat_id, result["image"], caption, rows)
        else:
            self._say(chat_id, caption, rows)

    def _confirm(self, chat_id: str, s: Dict[str, Any]) -> None:
        camera = s["camera"]
        result = si.confirm(camera, self.out_dir, self.zones_path, known_cameras=self._all_cameras())
        s["restart"] = bool(s.get("restart")) or bool(result["restart_needed"])
        self._say(chat_id, _t(self.lang())["saved"].format(name=self._name(camera)))
        self._next(chat_id, s)

    def _finish(self, chat_id: str, s: Dict[str, Any], stopped: bool = False) -> None:
        words = _t(self.lang())
        text = words["stopped"] if stopped else words["done"]
        restart = bool(s.get("restart"))
        if restart:
            text += words["restart"]
        self._put(chat_id, None)
        self._say(chat_id, text)
        if restart:
            (self._restart or _request_restart)()

    # ---------- helpers ----------
    def _say(self, chat_id: str, text: str, rows: Optional[List[List[tuple]]] = None) -> None:
        try:
            self.send_text(chat_id, text, rows or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("interview message not sent: %s", exc)

    def _name(self, camera: str) -> str:
        from .camera_names import display_name  # noqa: PLC0415

        return display_name(camera, self.lang())

    def _named(self, text: str) -> Optional[List[str]]:
        """Only the camera(s) whose family name the owner wrote ("מפה לפרגולה"); None for all."""
        named = [c for c in self.cameras() if self._name(c) and self._name(c) in text]
        return named or None

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
    """The interview the running Telegram inbox carries on (telegram_agent.start)."""
    send_photo, _ = telegram_senders(inbox.cfg.bot_token, inbox._post, inbox._post_multipart)

    def send_text(chat_id: str, text: str, rows: List[List[tuple]]) -> Any:
        return inbox._say(chat_id, text, rows=rows)

    return SceneChat(send_photo, send_text, cameras, lang)


def start_from_cli(camera: str = "") -> Dict[str, Any]:
    """The box's CLI: start the interview in the owner's chat (all active cameras, or *camera*). The first
    picture is sent from here; the running inbox carries on with the owner's answers."""
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
    cameras = [_known_camera(camera, CAMERAS_PATH)] if camera else active
    send_photo, send_text = telegram_senders(cfg.bot_token, telegram_notify._http_post,
                                             telegram_notify._http_post_multipart)
    chat = SceneChat(send_photo, send_text, lambda: active, box_language, background=False)
    chat.start(cfg.chat_ids[0], cameras)
    return {"chat": cfg.chat_ids[0], "cameras": cameras}
