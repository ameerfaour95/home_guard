"""The Telegram side of the owner's assistant: alerts go out with a question, answers come back in.

    send_alert()      sends an alert with the feedback question and buttons, and
                      remembers which message carries which alert
    TelegramInbox     long-polls the bot's updates and hands each one to the
                      buttons or to the agent, then answers in the chat

Only chats listed in ``telegram_chat_ids`` are listened to: nobody else can
pause the alerts or ask for a video. In a group, Telegram's default privacy
mode already limits a bot to replies to its own messages, mentions and
commands, so the box does not read the family's other conversation.

One bot token can have only one poller. Each box therefore needs its own bot
(or, later, one central service that polls for all boxes).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import threading
import time
import urllib.error
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import telegram_notify
from .agent import UNAVAILABLE_REPLY, AgentContext, OwnerAgent, make_chat_model
from .chat_feed import ChatFeed
from .brain.i18n import LANGS
from .brain.i18n import t as tr
from .brain.deliver import choice_keyboard
from .boxconfig import LOG_DIR, PRODUCTION_ARCHIVE_DIR, PRODUCTION_LIVE_DIR, PRODUCTION_RETENTION_DAYS
from .feedback import (
    FEEDBACK_BUTTONS,
    MAX_TAG_TEXT_CHARS,
    OWNER_LABELS,
    TAG_UNDONE_NOTE,
    AlertIndex,
    Feedback,
    MuteState,
    _read_json,
    _write_json,
    button_feedback,
    save_feedback,
    undo_training_tag,
    verdict_for,
)
from .telegram_notify import TelegramConfig

log = logging.getLogger("box.telegram_agent")

POLL_SECONDS = 25
CAPTION_LIMIT = 1024  # Telegram's limit for a photo caption

Post = Callable[..., Dict[str, Any]]


BUTTON_LABELS = {code: label for row in FEEDBACK_BUTTONS for label, code in row}

REMIND_SEC = 300.0   # an escalation nobody answered is sent once more, loud, after this long
TAG_WAIT_SEC = 600.0  # how long "Other…" waits for the owner's words
CALLBACK_DATA_LIMIT = 64  # Telegram refuses a button whose callback data is longer (bytes)

# The buttons under an alert: (emoji, i18n key, callback code). The owner tags the clip with one of
# OWNER_LABELS; "Other…" waits for their own words; fb:mute60 stays the pause.
_ALERT_BUTTONS = (
    (("🟢", "btn_tag_normal", "tag:normal"), ("🟡", "btn_tag_suspicious", "tag:suspicious"),
     ("🔴", "btn_tag_escalation", "tag:escalation")),
    (("⚪", "btn_tag_empty", "tag:empty"), ("✏️", "btn_tag_other", "tag:other"), ("⏸", "btn_mute60", "fb:mute60")),
)


def feedback_keyboard(lang: str = "en", ai_label: str = "") -> str:
    """The buttons under an alert, as Telegram's ``reply_markup`` JSON, in the box language.

    The label the AI gave the clip (*ai_label*) carries a leading "✓ ", so tagging what the AI already
    said is one glance away.
    """
    def text(emoji: str, key: str, code: str) -> str:
        words = f"{emoji} {tr(key, lang)}"
        return f"✓ {words}" if ai_label and code == f"tag:{ai_label}" else words

    return json.dumps({"inline_keyboard": [
        [{"text": text(*button), "callback_data": button[2]} for button in row] for row in _ALERT_BUTTONS
    ]})


def box_language() -> str:
    """The box language (``owner_language`` in box.yaml), read each time: it changes without a restart."""
    try:
        from .boxconfig import load_box_settings  # noqa: PLC0415

        code = str(load_box_settings().get("owner_language") or "en")
    except Exception:  # noqa: BLE001
        return "en"
    return code if code in LANGS else "en"


class PendingTags:
    """Who tapped "Other…" on which alert: that person's next message in that chat is the tag.

    Kept as JSON (``<feedback_dir>/.pending_tags.json``) so a restart does not lose a wait. A wait
    ends after *ttl* seconds; the question it asked is remembered for a day, so a late reply to it
    can be told that it expired. Never raises: a damaged file is an empty store.
    """

    KEEP_EXPIRED_SEC = 86400.0
    KEEP_EXPIRED_MAX = 100

    def __init__(self, path: str, ttl: float = TAG_WAIT_SEC) -> None:
        self.path, self.ttl = path, float(ttl)
        data = _read_json(path)
        pending = data.get("pending") if isinstance(data.get("pending"), dict) else {}
        expired = data.get("expired_prompts") if isinstance(data.get("expired_prompts"), dict) else {}
        self._pending: Dict[str, Dict[str, Any]] = {
            str(k): v for k, v in pending.items() if isinstance(v, dict) and self._valid(v)}
        self._expired: Dict[str, float] = {}
        for key, ts in expired.items():
            try:
                stamp = float(ts)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(stamp):
                self._expired[str(key)] = stamp

    @staticmethod
    def _valid(entry: Dict[str, Any]) -> bool:
        try:
            return isinstance(entry.get("alert_id"), str) and bool(entry["alert_id"]) and math.isfinite(
                float(entry.get("ts")))
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _key(chat_id: Any, user_id: Any) -> str:
        return f"{chat_id}:{user_id}"

    def _save(self) -> None:
        try:
            _write_json(self.path, {"pending": self._pending, "expired_prompts": self._expired})
        except Exception as exc:  # noqa: BLE001 - a tag wait must never stop the inbox
            log.warning("Pending tags not saved: %s", exc)

    def _purge(self, now: float) -> bool:
        changed = False
        for key, entry in list(self._pending.items()):
            if now - float(entry["ts"]) > self.ttl:
                del self._pending[key]
                if entry.get("prompt_id") is not None:
                    self._expired[f"{entry.get('chat_id')}:{entry.get('prompt_id')}"] = now
                changed = True
        keep = sorted(((k, v) for k, v in self._expired.items() if now - v <= self.KEEP_EXPIRED_SEC),
                      key=lambda kv: kv[1])[-self.KEEP_EXPIRED_MAX:]
        if len(keep) != len(self._expired):
            self._expired = dict(keep)
            changed = True
        return changed

    def add(self, chat_id: Any, user_id: Any, alert_id: str, now: float, prompt_id: Optional[int] = None) -> None:
        self._purge(now)
        self._pending[self._key(chat_id, user_id)] = {"chat_id": str(chat_id), "alert_id": str(alert_id),
                                                      "ts": float(now), "prompt_id": prompt_id}
        self._save()

    def take(self, chat_id: Any, user_id: Any, now: float) -> Optional[str]:
        """The alert *user_id* is tagging in *chat_id*, removed from the store; None when nothing waits."""
        changed = self._purge(now)
        entry = self._pending.pop(self._key(chat_id, user_id), None)
        if changed or entry is not None:
            self._save()
        return str(entry["alert_id"]) if entry else None

    def cancel(self, chat_id: Any, user_id: Any) -> None:
        if self._pending.pop(self._key(chat_id, user_id), None) is not None:
            self._save()

    def expired_prompt(self, chat_id: Any, message_id: Any, now: float) -> bool:
        """True (once) when *message_id* is an "Other…" question whose wait ran out."""
        changed = self._purge(now)
        found = self._expired.pop(f"{chat_id}:{message_id}", None) is not None
        if changed or found:
            self._save()
        return found


def owner_reacted(feedback_dir: str, alert_id: str) -> bool:
    """True once any answer to *alert_id* was saved (a button, a reply, a message bound to it)."""
    import glob  # noqa: PLC0415

    if not feedback_dir or not alert_id:
        return False
    pattern = os.path.join(glob.escape(feedback_dir), "feedback", "*", "*", f"{glob.escape(alert_id)}_*.feedback.json")
    prefix, suffix = f"{alert_id}_", ".feedback.json"
    # The file is <alert_id>_<milliseconds>.feedback.json: an alert whose id merely starts the same does not count.
    return any(os.path.basename(p)[len(prefix):-len(suffix)].isdigit() for p in glob.glob(pattern))


def not_them_button(alert: Dict[str, Any], lang: str = "en") -> Optional[Dict[str, str]]:
    """The note owner's callback, only for a softened alert and within Telegram's byte limit."""
    if alert.get("softened") is True and alert.get("applied_fact_id") and alert.get("alert_id"):
        callback = f"nt:{alert['applied_fact_id']}:{alert['alert_id']}"
        if len(callback.encode("utf-8")) <= 64:
            return {"text": tr("house_fact_not_them", lang), "callback_data": callback}
    return None


def send_alert(
    cfg: TelegramConfig,
    index: AlertIndex,
    alert: Dict[str, Any],
    text: str,
    image: Optional[bytes] = None,
    post: Post = telegram_notify._http_post,
    post_multipart: Post = telegram_notify._http_post_multipart,
    feed: Optional[ChatFeed] = None,
    silent: bool = False,
    lang: str = "en",
) -> Dict[str, Any]:
    """Send one alert to every chat, with the feedback question and buttons. Never raises.

    *alert* is what an answer will be filed under: ``{"alert_id", "camera",
    "summary", "ts"}``. Each chat's message id is stored in *index*. With a
    *feed*, the alert also appears in the box's window, delivered or not.
    *silent* delivers it without a sound (a normal scene); the question and the
    buttons are in *lang*, the box language.
    """
    if cfg.dry_run or not cfg.enabled:
        reason = "dry_run" if cfg.dry_run else "not_configured"
        log.info("Telegram alert not sent (%s): %s", reason, text)
        return {"sent": False, "reason": reason}
    body = f"{text}\n\n{tr('feedback_question', lang)}"
    keyboard = feedback_keyboard(lang, ai_label=str((alert.get("label") if isinstance(alert, dict) else "") or ""))
    button = not_them_button(alert, lang)
    if button:
        markup = json.loads(keyboard)
        markup["inline_keyboard"].append([button])
        keyboard = json.dumps(markup)
    quiet = {"disable_notification": "true"} if silent else {}
    results = []
    for chat_id in cfg.chat_ids:
        try:
            if image:
                resp = post_multipart(
                    cfg.bot_token, "sendPhoto",
                    {"chat_id": chat_id, "caption": body[:CAPTION_LIMIT], "reply_markup": keyboard, **quiet},
                    {"photo": ("alert.jpg", image, "image/jpeg")},
                )
            else:
                resp = post(cfg.bot_token, "sendMessage",
                            {"chat_id": chat_id, "text": body, "reply_markup": keyboard, **quiet})
            ok = bool(resp.get("ok"))
            message_id = (resp.get("result") or {}).get("message_id")
            if ok and message_id is not None:
                index.remember(chat_id, message_id, alert)
            else:
                log.warning("Telegram alert not ok for %s: %s", chat_id, resp.get("description"))
            results.append({"chat_id": chat_id, "ok": ok, "message_id": message_id})
        except (urllib.error.URLError, OSError) as exc:
            reason = telegram_error(exc)
            log.warning("Telegram alert failed for %s: %s", chat_id, reason)
            results.append({"chat_id": chat_id, "ok": False, "error": reason})
    sent = any(r["ok"] for r in results)
    if feed is not None:
        errors = [str(r.get("error") or "") for r in results if not r["ok"]]
        feed.add("box", "alert", text, camera=str(alert.get("camera") or ""), alert_id=str(alert.get("alert_id") or ""),
                 image=image, delivered=sent, error="" if sent else next((e for e in errors if e), "not delivered"))
    return {"sent": sent, "results": results}


def telegram_error(exc: BaseException) -> str:
    """Why Telegram refused, in its own words when it gave any.

    A refusal arrives as an HTTP error whose body says what is wrong ("Forbidden:
    bot was kicked from the group chat"); the status line alone ("403 Forbidden")
    does not tell an installer what to fix.
    """
    if isinstance(exc, urllib.error.HTTPError):
        try:
            description = json.loads(exc.read().decode("utf-8", "replace")).get("description")
        except (ValueError, OSError, AttributeError):
            description = None
        if description:
            hint = ""
            if exc.code == 403:
                hint = " - add the bot to the Telegram group again"
            return f"{description}{hint}"
    return str(exc)


def send_clip(
    cfg: TelegramConfig,
    index: AlertIndex,
    alert_id: str,
    clip_path: str,
    post_multipart: Post = telegram_notify._http_post_multipart,
    feed: Optional[ChatFeed] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """Send an alert's video as a reply under the alert it belongs to, in every chat that got the alert. Never raises.

    *silent*: no sound, as for the alert itself (a normal scene).
    """
    if cfg.dry_run or not cfg.enabled:
        return {"sent": False, "reason": "dry_run" if cfg.dry_run else "not_configured"}
    targets = index.messages(alert_id)
    if not targets:
        return {"sent": False, "reason": "the alert was not delivered"}
    results = []
    try:
        with open(clip_path, "rb") as f:
            data = f.read()
    except OSError as exc:
        return {"sent": False, "reason": str(exc)}
    for chat_id, message_id in targets:
        try:
            resp = post_multipart(
                cfg.bot_token, "sendVideo",
                {"chat_id": chat_id, "reply_to_message_id": str(message_id),
                 "allow_sending_without_reply": "true", "supports_streaming": "true",
                 **({"disable_notification": "true"} if silent else {})},
                {"video": (os.path.basename(clip_path), data, "video/mp4")},
                timeout=120.0,
            )
            ok = bool(resp.get("ok"))
            if not ok:
                log.warning("Telegram video not ok for %s: %s", chat_id, resp.get("description"))
            results.append({"chat_id": chat_id, "ok": ok})
        except (urllib.error.URLError, OSError) as exc:
            reason = telegram_error(exc)
            log.warning("Telegram video failed for %s: %s", chat_id, reason)
            results.append({"chat_id": chat_id, "ok": False, "error": reason})
    sent = any(r["ok"] for r in results)
    if feed is not None:
        errors = [str(r.get("error") or "") for r in results if not r["ok"]]
        feed.add("box", "video", "Video of the alert", alert_id=alert_id, delivered=sent,
                 error="" if sent else next((e for e in errors if e), "not delivered"))
    return {"sent": sent, "results": results}


@dataclass
class OwnerAssistant:
    """What inference mode holds on to: send alerts through it, and ask it whether a camera is paused."""

    cfg: TelegramConfig
    index: AlertIndex
    mute: MuteState
    inbox: "TelegramInbox"
    thread: Optional[threading.Thread] = None
    feed: Optional[ChatFeed] = None

    deliverer: Any = None
    feedback_dir: str = ""
    archive_dir: str = PRODUCTION_ARCHIVE_DIR   # the upload moves answers here, one folder per site

    def answered(self, alert_id: str) -> bool:
        """True once the owner answered *alert_id*: seen by the inbox in this run, or saved in the live folder or
        in any site folder of the archive (the upload moves answers there every 15 minutes)."""
        if alert_id in (getattr(self.inbox, "answered", None) or ()):
            return True
        return any(owner_reacted(root, alert_id) for root in alert_roots(self.feedback_dir, self.archive_dir))

    def announce(self, text: str) -> None:
        """One message to every configured chat (mode switches), sent on its own thread so a slow network never
        holds up the detector loop. Never raises."""
        try:
            if self.cfg.dry_run or not self.cfg.enabled:
                return

            def send() -> None:
                for chat_id in self.cfg.chat_ids:
                    try:
                        telegram_notify._http_post(self.cfg.bot_token, "sendMessage", {"chat_id": chat_id, "text": text})
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Announcement not sent to %s: %s", chat_id, exc)

            threading.Thread(target=send, name="announce", daemon=True).start()
        except Exception as exc:
            log.warning("Announcement not started: %s", exc)

    def is_muted(self, camera: str) -> bool:
        return self.mute.is_muted(time.time(), camera)

    def send_alert(self, alert: Dict[str, Any], text: str, image: Optional[bytes] = None, silent: bool = False,
                   lang: str = "en") -> Dict[str, Any]:
        return send_alert(self.cfg, self.index, alert, text, image, feed=self.feed, silent=silent, lang=lang)

    def remind_if_silent(self, alert: Dict[str, Any], text: str, lang: str = "en", delay: float = REMIND_SEC) -> None:
        """For an escalation: if nobody answered within *delay*, send it once more, loud. Never raises."""
        def remind() -> None:
            try:
                if self.answered(str(alert.get("alert_id") or "")):
                    return
                camera = str(alert.get("camera") or "")
                if self.mute is not None and self.mute.is_muted(time.time(), camera):
                    log.info("[%s] escalation reminder not sent: the owner paused alerts", camera)
                    return
                self.send_alert(alert, f"{tr('alert_reminder', lang)}\n{text}", lang=lang)
            except Exception as exc:  # noqa: BLE001
                log.warning("Escalation reminder not sent: %s", exc)

        try:
            timer = threading.Timer(delay, remind)
            timer.daemon = True
            timer.start()
        except Exception as exc:  # noqa: BLE001 - the alert loop must never stop for a reminder
            log.warning("Escalation reminder not scheduled: %s", exc)

    def send_clip(self, alert_id: str, clip_path: str, silent: bool = False) -> Dict[str, Any]:
        """The alert's video, as a reply under the alert. Call it once the clip has been written."""
        return send_clip(self.cfg, self.index, alert_id, clip_path, feed=self.feed, silent=silent)


def alert_roots(live_dir: str = PRODUCTION_LIVE_DIR, archive_dir: str = PRODUCTION_ARCHIVE_DIR) -> List[str]:
    """The folders that hold saved alerts: the live one, and one per site in the archive."""
    sites = sorted(os.listdir(archive_dir)) if os.path.isdir(archive_dir) else []
    return [live_dir] + [os.path.join(archive_dir, s) for s in sites if os.path.isdir(os.path.join(archive_dir, s))]


def start(
    box_settings: Dict[str, Any],
    env: Dict[str, str],
    camera_names: Sequence[str],
    log_dir: str = LOG_DIR,
    live_dir: str = PRODUCTION_LIVE_DIR,
    archive_dir: str = PRODUCTION_ARCHIVE_DIR,
) -> OwnerAssistant:
    """Build the owner's assistant for this box and start listening on a background thread.

    Without an OpenAI key the buttons still work and written messages are saved
    unread. Without a Telegram token or chat ids nothing is started.
    """
    if not isinstance(box_settings, dict) or not isinstance(env, dict):
        log.warning("Ignoring malformed assistant configuration")
        box_settings = box_settings if isinstance(box_settings, dict) else {}
        env = env if isinstance(env, dict) else {}
    cfg = telegram_notify.load_telegram_config(box_settings, env)
    mute = MuteState(os.path.join(log_dir, "alert_mute.json"))
    index = AlertIndex(os.path.join(log_dir, "alert_index.json"))
    feed = ChatFeed(os.path.join(log_dir, "telegram_chat.jsonl"))
    agent = None
    deliverer = None
    try:
        version = int(str(box_settings.get("agent_version", 1) or 1))
    except (ValueError, TypeError, OverflowError):
        log.warning("Invalid agent_version; using v1")
        version = 1
    try:
        if version == 2:
            from .brain.agent import build_owner_agent, follow_up_camera_receipts  # noqa: PLC0415

            agent, deliverer = build_owner_agent(box_settings, env, mute, cfg, live_dir, archive_dir, log_dir, feed)
            if agent is not None:
                threading.Thread(target=follow_up_camera_receipts, args=(agent.book, agent.registry, deliverer),
                                 name="camera-follow-up", daemon=True).start()
        else:
            model = make_chat_model(env, str(box_settings.get("agent_model", "gpt-4o-mini")))
            if model is not None:
                agent = OwnerAgent(model, AgentContext(
                    camera_names=list(camera_names), mute_state=mute, feedback_dir=live_dir,
                    roots=lambda: alert_roots(live_dir, archive_dir),
                    retention_days=PRODUCTION_RETENTION_DAYS,
                ))
    except Exception as exc:  # noqa: BLE001 - a missing library must not stop the alerts
        log.warning("Owner agent not available (%s); buttons still work.", exc)
    inbox = TelegramInbox(cfg, agent, index, mute, live_dir, os.path.join(log_dir, "telegram_offset.json"),
                          feed=feed, deliverer=deliverer, archive_dir=archive_dir, lang=box_language)
    assistant = OwnerAssistant(cfg=cfg, index=index, mute=mute, inbox=inbox, feed=feed, deliverer=deliverer,
                               feedback_dir=live_dir, archive_dir=archive_dir)
    if cfg.enabled and not cfg.dry_run:
        assistant.thread = threading.Thread(target=inbox.run, name="telegram-inbox", daemon=True)
        assistant.thread.start()
    return assistant


def _who(sender: Dict[str, Any]) -> Dict[str, Any]:
    name = " ".join(x for x in (sender.get("first_name"), sender.get("last_name")) if x)
    return {"user_id": sender.get("id"), "name": name or sender.get("username") or ""}


def _button_text(message: Dict[str, Any], code: str, default: str) -> str:
    """The label of the tapped button, read from the message's keyboard (for the box's chat window)."""
    try:
        for row in (message.get("reply_markup") or {}).get("inline_keyboard") or []:
            for button in row:
                if isinstance(button, dict) and button.get("callback_data") == code and button.get("text"):
                    return str(button["text"])
    except (AttributeError, TypeError):
        pass
    return default


class TelegramInbox:
    """Receives what the owner taps and writes, and answers."""

    def __init__(
        self,
        cfg: TelegramConfig,
        agent: Optional[OwnerAgent],
        index: AlertIndex,
        mute: MuteState,
        feedback_dir: str,
        offset_path: str,
        post: Post = telegram_notify._http_post,
        post_multipart: Post = telegram_notify._http_post_multipart,
        now: Callable[[], float] = time.time,
        feed: Optional[ChatFeed] = None,
        deliverer: Any = None,
        training_dir: Optional[str] = None,
        archive_dir: Optional[str] = None,
        lang: Optional[Callable[[], str]] = None,
    ) -> None:
        self.feed = feed
        self.deliverer = deliverer
        # Where a tagged clip is copied for training, and the archive searched for it (None: the box's own).
        self.training_dir, self.archive_dir = training_dir, archive_dir
        self._lang_source = lang          # the box language for what the code writes; None: English
        self.pending = PendingTags(os.path.join(feedback_dir, ".pending_tags.json"))
        self._seen = deque(maxlen=200)
        self.answered: set = set()       # alert ids the owner answered in this run (the reminder checks it first)
        self.cfg, self.agent, self.index, self.mute = cfg, agent, index, mute
        self.feedback_dir, self.offset_path = feedback_dir, offset_path
        self._post, self._post_multipart, self._now = post, post_multipart, now
        self._offset = self._load_offset()

    # -- offset: which updates were already handled ---------------------------
    def _load_offset(self) -> int:
        try:
            with open(self.offset_path, encoding="utf-8") as f:
                return int(json.load(f)["offset"])
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            return 0

    def _save_offset(self) -> None:
        os.makedirs(os.path.dirname(self.offset_path) or ".", exist_ok=True)
        tmp = f"{self.offset_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"offset": self._offset}, f)
        os.replace(tmp, self.offset_path)

    # -- sending ----------------------------------------------------------------
    def _say(self, chat_id: str, text: str, reply_to: Optional[int] = None,
             buttons: Sequence[str] = (), undo_token: str = "", lang: str = "en",
             question_token: str = "", undo_data: str = "", markup: Optional[Dict[str, Any]] = None,
             entities: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send *text* (with retries) and return Telegram's answer. *undo_data* is a whole Undo callback
        code (``tu:...``); *markup* is used when there are no buttons (a ``force_reply``)."""
        fields = {"chat_id": chat_id, "text": text}
        if reply_to is not None:
            fields["reply_to_message_id"] = str(reply_to)
            fields["allow_sending_without_reply"] = "true"
        if entities:
            fields["entities"] = json.dumps(entities)
        undo_code = undo_data or (f"u:{undo_token}" if undo_token else "")
        undo_row = [{"text": tr("undo_button", lang), "callback_data": undo_code}] if undo_code else None
        if buttons:
            # A question asked after something was already changed: the choices, then the Undo row.
            markup = json.loads(choice_keyboard(buttons, question_token))
            if undo_row:
                markup.setdefault("inline_keyboard", []).append(undo_row)
            fields["reply_markup"] = json.dumps(markup)
        elif undo_row:
            fields["reply_markup"] = json.dumps({"inline_keyboard": [undo_row]})
        elif markup:
            fields["reply_markup"] = json.dumps(markup)
        # A dropped connection (WinError 10054) once swallowed the confirmation of a pause:
        # the owner never learned the house was unwatched. Try again before giving up.
        resp: Any = {}
        for attempt in range(3):
            try:
                resp = self._post(self.cfg.bot_token, "sendMessage", fields)
                break
            except (urllib.error.URLError, OSError) as exc:
                if attempt == 2:
                    raise
                log.warning("Telegram answer not sent (%s); trying again.", exc)
                time.sleep(2 * (attempt + 1))
        self._note("assistant", "answer", text)
        return resp if isinstance(resp, dict) else {}

    def _after(self, reply: Any) -> None:
        for action in getattr(reply, "after", ()) or ():
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                log.warning("After-reply action failed: %s", exc)

    def _send_v2(self, chat_id: str, reply: Any, reply_to: Optional[int]) -> None:
        try:
            self._say(chat_id, reply.text, reply_to=reply_to, buttons=getattr(reply, "buttons", ()),
                      undo_token=getattr(reply, "undo_token", ""), lang=getattr(reply, "lang", "en"),
                      question_token=getattr(reply, "question_token", ""))
        finally:
            self._after(reply)       # a camera change must be applied even if the answer could not be sent

    def _note(self, who: str, kind: str, text: str, name: str = "", alert: Optional[Dict[str, Any]] = None) -> None:
        """Add a line to the conversation the box's window shows."""
        if self.feed is not None:
            alert = alert or {}
            self.feed.add(who, kind, text, name=name, camera=str(alert.get("camera") or ""),
                          alert_id=str(alert.get("alert_id") or ""), now=self._now())

    def _send_clip(self, chat_id: str, path: str) -> None:
        with open(path, "rb") as f:
            data = f.read()
        self._post_multipart(self.cfg.bot_token, "sendVideo", {"chat_id": chat_id},
                             {"video": (os.path.basename(path), data, "video/mp4")}, timeout=120.0)
        self._note("assistant", "video", os.path.basename(path))

    def _send_photo(self, chat_id: str, photo: Any) -> None:
        """A picture the assistant took just now: JPEG bytes, or the path of a JPEG file."""
        if isinstance(photo, (bytes, bytearray)):
            data, name = bytes(photo), "camera.jpg"
        else:
            with open(str(photo), "rb") as f:
                data = f.read()
            name = os.path.basename(str(photo))
        self._post_multipart(self.cfg.bot_token, "sendPhoto", {"chat_id": chat_id},
                             {"photo": (name, data, "image/jpeg")}, timeout=60.0)
        self._note("assistant", "photo", name)

    # -- one update ---------------------------------------------------------------
    def _allowed(self, chat_id: str) -> bool:
        return chat_id in self.cfg.chat_ids

    def _mark_answered(self, alert: Optional[Dict[str, Any]]) -> None:
        alert_id = str((alert or {}).get("alert_id") or "")
        if alert_id:
            self.answered.add(alert_id)

    def _on_button(self, query: Dict[str, Any]) -> None:
        message = query.get("message") or {}
        chat_id = str((message.get("chat") or {}).get("id"))
        if not self._allowed(chat_id):
            return
        code = str(query.get("data") or "")
        if code.startswith(("tag:", "tu:")):          # tagging: before the v1/v2 split, both agents share it
            self._on_tag_button(query, message, chat_id, code)
            return
        self._cancel_tag(chat_id, (query.get("from") or {}).get("id"))   # any other tap ends an "Other…" wait
        if getattr(self.agent, "version", 1) == 2 and (code.startswith("cl:") or code.startswith("u:")):
            try:   # only stops the button's spinner: a dropped connection here must not lose the tap
                self._post(self.cfg.bot_token, "answerCallbackQuery", {"callback_query_id": str(query.get("id"))})
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not stop the button's spinner: %s", exc)
            who = _who(query.get("from") or {})
            if code.startswith("cl:"):
                try:
                    _, token, index = code.split(":", 2)
                    index = int(index)
                except ValueError:
                    log.warning("Ignoring malformed clarification callback")
                    return
                self._note("owner", "button", _button_text(message, code, code), who["name"])
                reply = self.agent.handle_choice(chat_id, token, index, who)
            else:
                self._note("owner", "button", _button_text(message, code, tr("undo_button", "en")), who["name"])
                reply = self.agent.undo_turn(chat_id, code[2:], who)
            if reply is not None:
                self._send_v2(chat_id, reply, message.get("message_id"))
            return
        feedback = button_feedback(code, self._now())
        answer = self._button_answer(feedback, self._language())
        tapped_alert = self.index.lookup(chat_id, message.get("message_id"))
        self._note("owner", "button", BUTTON_LABELS.get(code, code), _who(query.get("from") or {})["name"], tapped_alert)
        if feedback:
            alert = tapped_alert
            self._mark_answered(alert)
            self.mute.apply(feedback, self._now())
            save_feedback(self.feedback_dir, alert, feedback, "", _who(query.get("from") or {}), chat_id, self._now(),
                          training_dir=self.training_dir, archive_dir=self.archive_dir)
        # Stops the button's spinner, then leaves a visible line in the chat.
        self._post(self.cfg.bot_token, "answerCallbackQuery", {"callback_query_id": str(query.get("id"))})
        self._say(chat_id, answer, reply_to=message.get("message_id"))

    # -- tagging an alert ------------------------------------------------------------
    def _language(self) -> str:
        try:
            code = self._lang_source() if self._lang_source is not None else "en"
        except Exception as exc:  # noqa: BLE001
            log.warning("Box language not read: %s", exc)
            return "en"
        return code if code in LANGS else "en"

    @staticmethod
    def _button_answer(feedback: Optional[Feedback], lang: str) -> str:
        """The confirmation of an old ``fb:`` button, in the box language."""
        if feedback is None:
            return tr("button_unused", lang)
        if feedback.action == "mute" and feedback.mute_until:
            return tr("paused_all", lang, until=dt.datetime.fromtimestamp(feedback.mute_until).strftime("%H:%M"))
        if feedback.verdict in ("true_alert", "false_alarm", "real_but_wrong", "expected", "missed_event"):
            return tr("verdict_saved", lang, verdict=tr(f"verdict_{feedback.verdict}", lang))
        return tr("button_unused", lang)

    def _cancel_tag(self, chat_id: str, user_id: Any) -> None:
        try:
            self.pending.cancel(chat_id, user_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Pending tag not cancelled: %s", exc)

    @staticmethod
    def _undo_code(alert_id: str) -> str:
        code = f"tu:{alert_id}"
        return code if len(code.encode("utf-8")) <= CALLBACK_DATA_LIMIT else ""

    def _answer_callback(self, query: Dict[str, Any], text: str = "") -> None:
        fields = {"callback_query_id": str(query.get("id"))}
        if text:
            fields["text"] = text[:200]
        self._post(self.cfg.bot_token, "answerCallbackQuery", fields)

    def _ask_for_tag(self, chat_id: str, sender: Dict[str, Any], reply_to: Any, lang: str) -> Dict[str, Any]:
        """Ask the tapping person for their own words, as a forced reply only they are prompted for.

        Telegram's ``selective`` targets the people mentioned in the text, so the question names them.
        """
        ask = tr("tag_ask_text", lang)
        force = {"force_reply": True, "selective": True}
        username = str(sender.get("username") or "")
        if username:
            return self._say(chat_id, f"@{username} {ask}", reply_to=reply_to, markup=force)
        name = str(sender.get("first_name") or "") or "🙂"
        mention = {"type": "text_mention", "offset": 0, "length": len(name.encode("utf-16-le")) // 2,
                   "user": {"id": sender.get("id"), "is_bot": False, "first_name": name}}
        return self._say(chat_id, f"{name} {ask}", reply_to=reply_to, markup=force, entities=[mention])

    def _on_tag_button(self, query: Dict[str, Any], message: Dict[str, Any], chat_id: str, code: str) -> None:
        """``tag:<label>`` and ``tu:<alert_id>``. A tag never pauses or changes a setting, and never raises."""
        lang = self._language()
        sender = query.get("from") or {}
        who = _who(sender)
        user_id = sender.get("id")
        answered = False
        try:
            now = self._now()
            if code.startswith("tu:"):
                self._cancel_tag(chat_id, user_id)
                alert_id = code[3:]
                alert = self.index.alert(alert_id) or {"alert_id": alert_id}
                self._note("owner", "button", _button_text(message, code, tr("undo_button", lang)), who["name"], alert)
                save_feedback(self.feedback_dir, alert, Feedback(verdict="none", note=TAG_UNDONE_NOTE, source="button"),
                              "", who, chat_id, now, training_dir=self.training_dir, archive_dir=self.archive_dir)
                undo_training_tag(alert, "", who, now, self.training_dir)
                self._answer_callback(query)
                answered = True
                self._say(chat_id, tr("tag_undone", lang), reply_to=message.get("message_id"))
                return
            label = code[4:]
            known = label in OWNER_LABELS
            alert = self.index.lookup(chat_id, message.get("message_id")) if known else None
            self._note("owner", "button", _button_text(message, code, tr(f"btn_tag_{label}", lang) if known else code),
                       who["name"], alert)
            if not alert or not alert.get("alert_id"):
                self._cancel_tag(chat_id, user_id)
                self._answer_callback(query)
                answered = True
                self._say(chat_id, tr("button_unused", lang), reply_to=message.get("message_id"))
                return
            alert_id = str(alert["alert_id"])
            self._mark_answered(alert)
            if label == "other":
                self._answer_callback(query)
                answered = True
                resp = self._ask_for_tag(chat_id, sender, message.get("message_id"), lang)
                self.pending.add(chat_id, user_id, alert_id, now, prompt_id=(resp.get("result") or {}).get("message_id"))
                return
            self._cancel_tag(chat_id, user_id)
            feedback = Feedback(verdict=verdict_for(label, str(alert.get("label") or "")), owner_label=label,
                                tagged_by=who["name"] or str(user_id or ""), source="button")
            save_feedback(self.feedback_dir, alert, feedback, "", who, chat_id, now,
                          training_dir=self.training_dir, archive_dir=self.archive_dir)
            self._answer_callback(query)
            answered = True
            self._say(chat_id, tr("tag_saved", lang, label=tr(f"btn_tag_{label}", lang)),
                      reply_to=message.get("message_id"), undo_data=self._undo_code(alert_id), lang=lang)
        except Exception as exc:  # noqa: BLE001 - a tag must never stop the inbox
            log.warning("Tag %s not saved: %s", code, exc)
            if not answered:
                try:
                    self._answer_callback(query, tr("failed", lang, what=tr("what_record_verdict", lang),
                                                    reason=tr("reason_error", lang)))
                except Exception as exc2:  # noqa: BLE001
                    log.warning("Could not answer the tag button: %s", exc2)

    def _take_tag_text(self, chat_id: str, sender: Dict[str, Any], text: str, message: Dict[str, Any]) -> bool:
        """True when *text* was handled as the words of an "Other…" tag (or a late reply to its question):
        then neither agent sees it. A command ("/...") ends the wait and goes on as usual."""
        user_id = sender.get("id")
        if text.startswith("/"):
            self._cancel_tag(chat_id, user_id)
            return False
        lang = self._language()
        taken = False
        try:
            now = self._now()
            alert_id = self.pending.take(chat_id, user_id, now)
            who = _who(sender)
            if not alert_id:
                replied = (message.get("reply_to_message") or {}).get("message_id")
                if replied is not None and self.pending.expired_prompt(chat_id, replied, now):
                    self._note("owner", "message", text, who["name"])
                    self._say(chat_id, tr("tag_expired", lang), reply_to=message.get("message_id"))
                    return True
                return False
            taken = True
            alert = self.index.alert(alert_id) or {"alert_id": alert_id}
            self._note("owner", "message", text, who["name"], alert)
            self._mark_answered(alert)
            words = text[:MAX_TAG_TEXT_CHARS]
            feedback = Feedback(verdict=verdict_for("other", str(alert.get("label") or "")), owner_label="other",
                                owner_text=words, tagged_by=who["name"] or str(user_id or ""), source="text")
            save_feedback(self.feedback_dir, alert, feedback, text, who, chat_id, now,
                          training_dir=self.training_dir, archive_dir=self.archive_dir)
            self._say(chat_id, tr("tag_saved_text", lang, text=words), reply_to=message.get("message_id"),
                      undo_data=self._undo_code(alert_id), lang=lang)
            return True
        except Exception as exc:  # noqa: BLE001 - a tag must never stop the inbox
            log.warning("Tag text not saved: %s", exc)
            if taken:
                try:
                    self._say(chat_id, tr("failed", lang, what=tr("what_record_verdict", lang),
                                          reason=tr("reason_error", lang)), reply_to=message.get("message_id"))
                except Exception as exc2:  # noqa: BLE001
                    log.warning("Could not say the tag failed: %s", exc2)
            return taken

    def _on_message(self, message: Dict[str, Any]) -> None:
        chat_id = str((message.get("chat") or {}).get("id"))
        sender = message.get("from") or {}
        text = str(message.get("text") or "").strip()
        if not self._allowed(chat_id) or sender.get("is_bot") or not text:
            return
        if self._take_tag_text(chat_id, sender, text, message):
            return
        replied = (message.get("reply_to_message") or {}).get("message_id")
        alert = self.index.lookup(chat_id, replied) if replied is not None else None
        threaded = alert is not None
        v2 = getattr(self.agent, "version", 1) == 2
        if alert is None:
            if v2:
                recent = self.index.recent(chat_id, self._now())
                alert = recent[0] if len(recent) == 1 else None
            else:
                alert = self.index.latest(chat_id, self._now())
        self._note("owner", "message", text, _who(sender)["name"], alert)
        self._mark_answered(alert)
        if self.agent is None:
            save_feedback(self.feedback_dir, alert, Feedback(), text, _who(sender), chat_id, self._now())
            self._say(chat_id, UNAVAILABLE_REPLY, reply_to=message.get("message_id"))
            return
        if v2:
            if self.deliverer is not None:
                self.deliverer.typing(chat_id)
            reply = self.agent.handle(text, chat_id, _who(sender), alert, threaded)
            self._send_v2(chat_id, reply, message.get("message_id"))
            return
        reply = self.agent.handle(text, chat_id, _who(sender), alert)
        self._say(chat_id, reply.text, reply_to=message.get("message_id"))
        for photo in getattr(reply, "photos", None) or ():        # a live picture the assistant took
            try:
                self._send_photo(chat_id, photo)
            except (urllib.error.URLError, OSError) as exc:
                log.warning("Could not send a live picture: %s", exc)
                self._say(chat_id, "I could not send that picture right now.")
        for path in reply.clips:
            try:
                self._send_clip(chat_id, path)
            except (urllib.error.URLError, OSError) as exc:
                log.warning("Could not send %s: %s", path, exc)
                self._say(chat_id, "I could not send that video right now.")
        if getattr(reply, "restart", False):
            from . import control  # noqa: PLC0415

            control.request_restart()        # a camera was turned on/off; the answer is out now

    def handle_update(self, update: Dict[str, Any]) -> None:
        """Act on one update. Errors are logged, never raised: one bad update must not stop the inbox."""
        try:
            if not isinstance(update, dict):
                raise ValueError("update must be an object")
            uid = update.get("update_id")
            if uid is not None:
                if type(uid) is not int:
                    raise ValueError("update_id must be an integer")
                if uid in self._seen:
                    return
                self._seen.append(uid)
            if "callback_query" in update:
                self._on_button(update["callback_query"])
            elif "message" in update:
                self._on_message(update["message"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not handle Telegram update %s: %s", update.get("update_id") if isinstance(update, dict) else None, exc)

    # -- the loop -----------------------------------------------------------------
    def poll_once(self, timeout: int = POLL_SECONDS) -> int:
        """Fetch and handle the waiting updates. Returns how many there were."""
        resp = self._post(
            self.cfg.bot_token, "getUpdates",
            {"offset": str(self._offset), "timeout": str(timeout),
             "allowed_updates": json.dumps(["message", "callback_query"])},
            timeout=timeout + 15.0,
        )
        updates: List[Dict[str, Any]] = resp.get("result") or []
        if not isinstance(updates, list):
            log.warning("Ignoring malformed Telegram updates collection")
            return 0
        warned = False
        for update in updates:
            if not isinstance(update, dict) or type(update.get("update_id")) is not int:
                if not warned:
                    log.warning("Ignoring malformed polled update")
                    warned = True
                continue
            # Mark it handled first: an update that crashes the box must not be replayed forever.
            self._offset = max(self._offset, update["update_id"] + 1)
            try:
                self._save_offset()
            except Exception as exc:  # noqa: BLE001 - a disk failure must not discard the rest of this batch
                log.warning("Could not save Telegram offset: %s", exc)
            self.handle_update(update)
        return len(updates)

    def run(self, stop: Optional[threading.Event] = None) -> None:
        """Poll until *stop* is set. Network errors wait and retry."""
        if not self.cfg.enabled or self.cfg.dry_run:
            log.info("Telegram inbox not started (not configured, or dry run).")
            return
        log.info("Telegram inbox started for %d chat(s).", len(self.cfg.chat_ids))
        while not (stop and stop.is_set()):
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - offline, a timeout, a 409 from a second poller
                log.warning("Telegram inbox: %s; retrying in 15 s.", exc)
                time.sleep(15)
