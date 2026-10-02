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

import json
import logging
import os
import threading
import time
import urllib.error
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import telegram_notify
from .agent import UNAVAILABLE_REPLY, AgentContext, OwnerAgent, make_chat_model
from .chat_feed import ChatFeed
from .boxconfig import LOG_DIR, PRODUCTION_ARCHIVE_DIR, PRODUCTION_LIVE_DIR, PRODUCTION_RETENTION_DAYS
from .feedback import (
    FEEDBACK_BUTTONS,
    FEEDBACK_QUESTION,
    AlertIndex,
    Feedback,
    MuteState,
    button_feedback,
    confirmation_text,
    save_feedback,
)
from .telegram_notify import TelegramConfig

log = logging.getLogger("box.telegram_agent")

POLL_SECONDS = 25
CAPTION_LIMIT = 1024  # Telegram's limit for a photo caption

Post = Callable[..., Dict[str, Any]]


BUTTON_LABELS = {code: label for row in FEEDBACK_BUTTONS for label, code in row}


def feedback_keyboard() -> str:
    """The buttons under an alert, as Telegram's ``reply_markup`` JSON."""
    return json.dumps({"inline_keyboard": [
        [{"text": label, "callback_data": code} for label, code in row] for row in FEEDBACK_BUTTONS
    ]})


def send_alert(
    cfg: TelegramConfig,
    index: AlertIndex,
    alert: Dict[str, Any],
    text: str,
    image: Optional[bytes] = None,
    post: Post = telegram_notify._http_post,
    post_multipart: Post = telegram_notify._http_post_multipart,
    feed: Optional[ChatFeed] = None,
) -> Dict[str, Any]:
    """Send one alert to every chat, with the feedback question and buttons. Never raises.

    *alert* is what an answer will be filed under: ``{"alert_id", "camera",
    "summary", "ts"}``. Each chat's message id is stored in *index*. With a
    *feed*, the alert also appears in the box's window, delivered or not.
    """
    if cfg.dry_run or not cfg.enabled:
        reason = "dry_run" if cfg.dry_run else "not_configured"
        log.info("Telegram alert not sent (%s): %s", reason, text)
        return {"sent": False, "reason": reason}
    body = f"{text}\n\n{FEEDBACK_QUESTION}"
    results = []
    for chat_id in cfg.chat_ids:
        try:
            if image:
                resp = post_multipart(
                    cfg.bot_token, "sendPhoto",
                    {"chat_id": chat_id, "caption": body[:CAPTION_LIMIT], "reply_markup": feedback_keyboard()},
                    {"photo": ("alert.jpg", image, "image/jpeg")},
                )
            else:
                resp = post(cfg.bot_token, "sendMessage",
                            {"chat_id": chat_id, "text": body, "reply_markup": feedback_keyboard()})
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
) -> Dict[str, Any]:
    """Send an alert's video as a reply under the alert it belongs to, in every chat that got the alert. Never raises."""
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
                 "allow_sending_without_reply": "true", "supports_streaming": "true"},
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

    def is_muted(self, camera: str) -> bool:
        return self.mute.is_muted(time.time(), camera)

    def send_alert(self, alert: Dict[str, Any], text: str, image: Optional[bytes] = None) -> Dict[str, Any]:
        return send_alert(self.cfg, self.index, alert, text, image, feed=self.feed)

    def send_clip(self, alert_id: str, clip_path: str) -> Dict[str, Any]:
        """The alert's video, as a reply under the alert. Call it once the clip has been written."""
        return send_clip(self.cfg, self.index, alert_id, clip_path, feed=self.feed)


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
    cfg = telegram_notify.load_telegram_config(box_settings, env)
    mute = MuteState(os.path.join(log_dir, "alert_mute.json"))
    index = AlertIndex(os.path.join(log_dir, "alert_index.json"))
    agent = None
    try:
        model = make_chat_model(env, str(box_settings.get("agent_model", "gpt-4o-mini")))
        if model is not None:
            agent = OwnerAgent(model, AgentContext(
                camera_names=list(camera_names), mute_state=mute, feedback_dir=live_dir,
                roots=lambda: alert_roots(live_dir, archive_dir),
                retention_days=PRODUCTION_RETENTION_DAYS,
            ))
    except Exception as exc:  # noqa: BLE001 - a missing library must not stop the alerts
        log.warning("Owner agent not available (%s); buttons still work.", exc)
    feed = ChatFeed(os.path.join(log_dir, "telegram_chat.jsonl"))
    inbox = TelegramInbox(cfg, agent, index, mute, live_dir, os.path.join(log_dir, "telegram_offset.json"), feed=feed)
    assistant = OwnerAssistant(cfg=cfg, index=index, mute=mute, inbox=inbox, feed=feed)
    if cfg.enabled and not cfg.dry_run:
        assistant.thread = threading.Thread(target=inbox.run, name="telegram-inbox", daemon=True)
        assistant.thread.start()
    return assistant


def _who(sender: Dict[str, Any]) -> Dict[str, Any]:
    name = " ".join(x for x in (sender.get("first_name"), sender.get("last_name")) if x)
    return {"user_id": sender.get("id"), "name": name or sender.get("username") or ""}


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
    ) -> None:
        self.feed = feed
        self.cfg, self.agent, self.index, self.mute = cfg, agent, index, mute
        self.feedback_dir, self.offset_path = feedback_dir, offset_path
        self._post, self._post_multipart, self._now = post, post_multipart, now
        self._offset = self._load_offset()

    # -- offset: which updates were already handled ---------------------------
    def _load_offset(self) -> int:
        try:
            with open(self.offset_path, encoding="utf-8") as f:
                return int(json.load(f)["offset"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def _save_offset(self) -> None:
        os.makedirs(os.path.dirname(self.offset_path) or ".", exist_ok=True)
        tmp = f"{self.offset_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"offset": self._offset}, f)
        os.replace(tmp, self.offset_path)

    # -- sending ----------------------------------------------------------------
    def _say(self, chat_id: str, text: str, reply_to: Optional[int] = None) -> None:
        fields = {"chat_id": chat_id, "text": text}
        if reply_to is not None:
            fields["reply_to_message_id"] = str(reply_to)
            fields["allow_sending_without_reply"] = "true"
        self._post(self.cfg.bot_token, "sendMessage", fields)
        self._note("assistant", "answer", text)

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

    def _on_button(self, query: Dict[str, Any]) -> None:
        message = query.get("message") or {}
        chat_id = str((message.get("chat") or {}).get("id"))
        if not self._allowed(chat_id):
            return
        code = str(query.get("data") or "")
        feedback = button_feedback(code, self._now())
        answer = confirmation_text(feedback) if feedback else "That button is no longer in use."
        tapped_alert = self.index.lookup(chat_id, message.get("message_id"))
        self._note("owner", "button", BUTTON_LABELS.get(code, code), _who(query.get("from") or {})["name"], tapped_alert)
        if feedback:
            alert = tapped_alert
            self.mute.apply(feedback, self._now())
            save_feedback(self.feedback_dir, alert, feedback, "", _who(query.get("from") or {}), chat_id, self._now())
        # Stops the button's spinner, then leaves a visible line in the chat.
        self._post(self.cfg.bot_token, "answerCallbackQuery", {"callback_query_id": str(query.get("id"))})
        self._say(chat_id, answer, reply_to=message.get("message_id"))

    def _on_message(self, message: Dict[str, Any]) -> None:
        chat_id = str((message.get("chat") or {}).get("id"))
        sender = message.get("from") or {}
        text = str(message.get("text") or "").strip()
        if not self._allowed(chat_id) or sender.get("is_bot") or not text:
            return
        replied = (message.get("reply_to_message") or {}).get("message_id")
        alert = self.index.lookup(chat_id, replied) if replied is not None else None
        if alert is None:
            alert = self.index.latest(chat_id, self._now())
        self._note("owner", "message", text, _who(sender)["name"], alert)
        if self.agent is None:
            save_feedback(self.feedback_dir, alert, Feedback(), text, _who(sender), chat_id, self._now())
            self._say(chat_id, UNAVAILABLE_REPLY, reply_to=message.get("message_id"))
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

    def handle_update(self, update: Dict[str, Any]) -> None:
        """Act on one update. Errors are logged, never raised: one bad update must not stop the inbox."""
        try:
            if "callback_query" in update:
                self._on_button(update["callback_query"])
            elif "message" in update:
                self._on_message(update["message"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not handle Telegram update %s: %s", update.get("update_id"), exc)

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
        for update in updates:
            # Mark it handled first: an update that crashes the box must not be replayed forever.
            self._offset = int(update["update_id"]) + 1
            self._save_offset()
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
