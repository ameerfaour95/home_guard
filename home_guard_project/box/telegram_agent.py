"""The Telegram side of the owner's assistant: alerts go out with a question, answers come back in.

    send_alert()      sends an alert as one message - its video, a short caption
                      and the tag buttons - and remembers which message carries
                      which alert
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
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import telegram_notify, voice
from .agent import UNAVAILABLE_REPLY, AgentContext, OwnerAgent, make_chat_model
from .chat_feed import ChatFeed
from .brain.i18n import LANGS, detect_language
from .brain.i18n import t as tr
from .brain.deliver import choice_keyboard
from .brain.style import clean_outgoing, replace_camera_ids
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
    not_a_judgement,
    save_feedback,
    saved_tag_requests,
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
TAG_WAIT_SEC = 86400.0     # how long "Other…" waits for the owner's words, as a reply to the question or video
# How long a message that replies to nothing is taken as those words (2026-10-08: it was 30 minutes, and
# "איזה תזכורת, אתה מטומטם ומגזים" was saved as a clip's explanation). Only a statement within this time counts;
# a question, a complaint or a command goes to the assistant while the wait stays open.
TAG_IMPLICIT_SEC = 180.0
CALLBACK_DATA_LIMIT = 64  # Telegram refuses a button whose callback data is longer (bytes)
VIDEO_WAIT_SEC = 60.0  # an alert waits this long for its video; then it goes out with the picture

# The buttons under an alert: (emoji, i18n key, callback code). The owner tags the clip with one of
# OWNER_LABELS; "Other…" waits for their own words. Alerts sent before carry tag:escalation, tag:empty and
# fb:mute60, which are still accepted.
_ALERT_BUTTONS = (
    (("🟡", "btn_tag_suspicious", "tag:suspicious"), ("🟢", "btn_tag_normal", "tag:normal"),
     ("✏️", "btn_tag_other", "tag:other")),
)
# Under an alert that a house rule (an owner's "raise" note) made suspicious.
_RULE_BUTTON = ("📏", "btn_tag_rule_mismatch", "tag:rule_mismatch")


def feedback_keyboard(lang: str = "en", ai_label: str = "", rule: bool = False) -> str:
    """The buttons under an alert, as Telegram's ``reply_markup`` JSON, in the box language.

    The label the AI gave the clip (*ai_label*) carries a leading "✓ ", so tagging what the AI already
    said is one glance away. *rule*: the alert went out because of a house rule, which the owner can say
    does not match.
    """
    def text(emoji: str, key: str, code: str) -> str:
        words = f"{emoji} {tr(key, lang)}"
        return f"✓ {words}" if ai_label and code == f"tag:{ai_label}" else words

    rows = _ALERT_BUTTONS + (((_RULE_BUTTON,),) if rule else ())
    return json.dumps({"inline_keyboard": [
        [{"text": text(*button), "callback_data": button[2]} for button in row] for row in rows
    ]})


def raised_by_rule(alert: Any) -> bool:
    """The alert was made suspicious by a house note (the owner's own rule), not lowered by one."""
    return bool(isinstance(alert, dict) and alert.get("applied_fact_id") and alert.get("softened") is False)


def alert_keyboard(alert: Any, lang: str = "en") -> str:
    """Everything under one alert: the tag buttons, and the rule and "not them" buttons when they apply."""
    alert = alert if isinstance(alert, dict) else {}
    keyboard = feedback_keyboard(lang, ai_label=str(alert.get("label") or ""), rule=raised_by_rule(alert))
    button = not_them_button(alert, lang)
    if button:
        markup = json.loads(keyboard)
        markup["inline_keyboard"].append([button])
        keyboard = json.dumps(markup)
    return keyboard


def _with_clock(text: str, alert: Dict[str, Any]) -> str:
    """*text* with the alert's local time at the end of its first line ("🟡 Suspicious · door · 02:14")."""
    try:
        clock = dt.datetime.fromtimestamp(float(alert.get("ts"))).strftime("%H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return text
    first, sep, rest = text.partition("\n")
    return f"{first} · {clock}{sep}{rest}"


def box_language() -> str:
    """The box language (``owner_language`` in box.yaml), read each time: it changes without a restart."""
    try:
        from .boxconfig import load_box_settings  # noqa: PLC0415

        code = str(load_box_settings().get("owner_language") or "en")
    except Exception:  # noqa: BLE001
        return "en"
    return code if code in LANGS else "en"


class PendingTags:
    """Who tapped "Other…" on which alert: that person's words in that chat become the tag.

    A person may wait on several alerts at once (one entry per chat, user and alert). A text is
    taken only when it is meant for a wait: a reply to that wait's question or to its alert's
    message goes to that alert; only the newest wait can take text that replies to nothing.
    Opening a newer prompt permanently makes older waits reply-only, even if sending it fails.
    Kept as JSON (``<feedback_dir>/.pending_tags.json``) so a restart does not lose a wait. A wait
    ends after *ttl* seconds; its question is remembered for a day with who was asked, so that
    person's late reply can be told that it expired. Never raises: a damaged file is an empty store.
    """

    KEEP_EXPIRED_SEC = 86400.0
    KEEP_EXPIRED_MAX = 100
    MAX_PROMPTS = 5

    def __init__(self, path: str, ttl: float = TAG_WAIT_SEC, implicit_ttl: float = TAG_IMPLICIT_SEC) -> None:
        self.path, self.ttl = path, float(ttl)
        # A wait takes a message that replies to nothing only this long; later, "it's me, pause until six"
        # is a request for the assistant, and only a reply to the question or to the clip is the answer.
        self.implicit_ttl = min(float(implicit_ttl), self.ttl)
        data = _read_json(path)
        pending = data.get("pending") if isinstance(data.get("pending"), dict) else {}
        expired = data.get("expired_prompts") if isinstance(data.get("expired_prompts"), dict) else {}
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._completed = saved_tag_requests(os.path.dirname(path)) if pending else set()
        for entry in pending.values():
            if isinstance(entry, dict) and self._valid(entry):
                entry["prompt_ids"] = self._prompt_ids(entry)
                # Old files have no id. Derive a stable one so a failed migration write cannot
                # change the request's identity on restart.
                if not entry.get("request_id"):
                    entry["request_id"] = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(entry, sort_keys=True)).hex
                self._pending[self._key(entry["chat_id"], entry["user_id"], entry["alert_id"])] = entry
        for chat_id, user_id in {(e["chat_id"], e["user_id"]) for e in self._pending.values()}:
            for _, entry in self._mine(chat_id, user_id)[1:]:
                entry["reply_only"] = True
        self._pending = {k: e for k, e in self._pending.items() if e["request_id"] not in self._completed}
        self._expired: Dict[str, Dict[str, Any]] = {}
        for key, value in expired.items():
            # {"ts", "user_id"}; an older file kept only a time: whose question it was is unknown, so drop it.
            if not isinstance(value, dict) or value.get("user_id") is None:
                continue
            try:
                stamp = float(value.get("ts"))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(stamp):
                self._expired[str(key)] = {"ts": stamp, "user_id": str(value["user_id"])}

    @staticmethod
    def _valid(entry: Dict[str, Any]) -> bool:
        try:
            return (isinstance(entry.get("alert_id"), str) and bool(entry["alert_id"])
                    and entry.get("chat_id") is not None and entry.get("user_id") is not None
                    and math.isfinite(float(entry.get("ts"))))
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _prompt_ids(entry: Dict[str, Any]) -> List[str]:
        ids = entry.get("prompt_ids") if isinstance(entry.get("prompt_ids"), list) else []
        return [str(i) for i in ids if i is not None]

    @staticmethod
    def _key(chat_id: Any, user_id: Any, alert_id: str) -> str:
        return f"{chat_id}:{user_id}:{alert_id}"

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
                for prompt_id in entry["prompt_ids"]:
                    self._expired[f"{entry['chat_id']}:{prompt_id}"] = {"ts": now, "user_id": str(entry["user_id"])}
                changed = True
        keep = sorted(((k, v) for k, v in self._expired.items() if now - v["ts"] <= self.KEEP_EXPIRED_SEC),
                      key=lambda kv: kv[1]["ts"])[-self.KEEP_EXPIRED_MAX:]
        if len(keep) != len(self._expired):
            self._expired = dict(keep)
            changed = True
        return changed

    def _mine(self, chat_id: Any, user_id: Any) -> List[Tuple[str, Dict[str, Any]]]:
        """This person's waits in this chat, newest first."""
        found = [(k, e) for k, e in self._pending.items()
                 if e["chat_id"] == str(chat_id) and e["user_id"] == str(user_id)]
        return sorted(reversed(found), key=lambda kv: float(kv[1]["ts"]), reverse=True)   # a tie: the later added

    def add(self, chat_id: Any, user_id: Any, alert_id: str, now: float, prompt_id: Optional[int] = None) -> None:
        self._purge(now)
        self.demote(chat_id, user_id)
        key = self._key(chat_id, user_id, alert_id)
        # "Other…" on the same alert again: a reply to either of its questions still counts.
        prompts = self._prompt_ids(self._pending.pop(key, None) or {})     # popped: it is the newest now
        if prompt_id is not None:
            prompts.append(str(prompt_id))
        self._pending[key] = {"chat_id": str(chat_id), "user_id": str(user_id), "alert_id": str(alert_id),
                              "ts": float(now), "prompt_ids": prompts[-self.MAX_PROMPTS:], "reply_only": False,
                              "request_id": uuid.uuid4().hex}
        self._save()

    def demote(self, chat_id: Any, user_id: Any, request_id: Optional[str] = None) -> None:
        """Disable implicit answers for superseded waits or a tag whose save is being attempted."""
        changed = False
        for _, entry in self._mine(chat_id, user_id):
            if not entry.get("reply_only") and (request_id is None or entry["request_id"] == request_id):
                entry["reply_only"] = True
                changed = True
        if changed:
            self._save()

    def match(self, chat_id: Any, user_id: Any, now: float, reply_to: Any = None,
              reply_alert: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find a wait without consuming it; completion follows the durable feedback write.

        *reply_to* is the id of the message the text replies to (None: it replies to nothing) and
        *reply_alert* the alert that message carried, if any.
        """
        changed = self._purge(now)
        mine = self._mine(chat_id, user_id)
        chosen = None
        if reply_to is None:
            open_waits = [kv for kv in mine if not kv[1].get("reply_only")
                          and now - float(kv[1]["ts"]) <= self.implicit_ttl]
            if open_waits:
                chosen = open_waits[0]
        else:
            chosen = next((kv for kv in mine if str(reply_to) in kv[1]["prompt_ids"]), None)
            if chosen is None and reply_alert:
                chosen = next((kv for kv in mine if kv[1]["alert_id"] == str(reply_alert)), None)
        if changed:
            self._save()
        return dict(chosen[1]) if chosen else None

    def complete(self, request_id: str) -> None:
        """Remember completion in memory even if saving the pending file fails."""
        self._completed.add(request_id)
        self._pending = {k: e for k, e in self._pending.items() if e["request_id"] != request_id}
        self._save()

    def take(self, chat_id: Any, user_id: Any, now: float, reply_to: Any = None,
             reply_alert: Optional[str] = None) -> Optional[str]:
        """Compatibility API: remove a matching wait and return its alert id."""
        entry = self.match(chat_id, user_id, now, reply_to, reply_alert)
        if entry:
            self.complete(entry["request_id"])
        return str(entry["alert_id"]) if entry else None

    def needs_text(self, chat_id: Any, user_id: Any, now: float, reply_to: Any = None) -> bool:
        """Media from the asker needs a text hint, without consuming any wait."""
        if self._purge(now):
            self._save()
        return any(reply_to is None or str(reply_to) in e["prompt_ids"]
                   for _, e in self._mine(chat_id, user_id))

    def cancel(self, chat_id: Any, user_id: Any) -> None:
        """End every wait of *user_id* in *chat_id*."""
        mine = self._mine(chat_id, user_id)
        for key, _ in mine:
            del self._pending[key]
        if mine:
            self._save()

    def expired_prompt(self, chat_id: Any, message_id: Any, user_id: Any, now: float) -> bool:
        """True (once) when *message_id* is an "Other…" question to *user_id* whose wait ran out.

        Someone else's reply to that question is not an answer to it: False, and it stays remembered.
        """
        changed = self._purge(now)
        key = f"{chat_id}:{message_id}"
        found = key in self._expired and self._expired[key]["user_id"] == str(user_id)
        if found:
            del self._expired[key]
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


def reply_fields(chat_id: Any, reply_to: Any) -> Dict[str, str]:
    """Telegram's fields that put a message in the thread of *reply_to* ({"chat_id", "message_id"}: the event's
    first alert), only in the chat that owns that message. {} otherwise. The guard's telegram_notify.reply_fields
    when it exists. Never raises."""
    try:
        own = getattr(telegram_notify, "reply_fields", None)
        if callable(own):
            return dict(own(chat_id, reply_to) or {})
        if (not isinstance(reply_to, dict) or reply_to.get("message_id") is None
                or str(reply_to.get("chat_id")) != str(chat_id)):
            return {}
        return {"reply_to_message_id": str(int(reply_to["message_id"])), "allow_sending_without_reply": "true"}
    except Exception as exc:  # noqa: BLE001 - an alert goes out without its thread rather than not at all
        log.warning("Reply fields not built: %s", exc)
        return {}


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
    video: Optional[str] = None,
    reply_to: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send one alert to every chat, with its buttons. Never raises.

    *reply_to* ({"chat_id", "message_id"}): an update of an event goes out as a reply to its first alert, in the
    chat that has it.

    *alert* is what an answer will be filed under: ``{"alert_id", "camera",
    "summary", "ts"}``. Each chat's message id is stored in *index*. With a
    *feed*, the alert also appears in the box's window, delivered or not.
    *silent* delivers it without a sound (a normal scene); the buttons are in
    *lang*, the box language. With *video* (the alert's clip) the alert is that
    video with a short caption: the text, with the time. A video that cannot be
    read or is refused falls back to the picture (or the text) with the
    feedback question, as without one.
    """
    if cfg.dry_run or not cfg.enabled:
        reason = "dry_run" if cfg.dry_run else "not_configured"
        log.info("Telegram alert not sent (%s): %s", reason, text)
        return {"sent": False, "reason": reason}
    body = f"{text}\n\n{tr('feedback_question', lang)}"
    keyboard = alert_keyboard(alert, lang)
    quiet = {"disable_notification": "true"} if silent else {}
    clip = None
    if video:
        try:
            with open(video, "rb") as f:
                clip = (os.path.basename(video), f.read(), "video/mp4")
        except OSError as exc:
            log.warning("Alert video not readable (%s); sending the picture", exc)
    results = []
    for chat_id in cfg.chat_ids:
        try:
            resp = None
            thread = reply_fields(chat_id, reply_to)
            if clip is not None:
                try:
                    resp = post_multipart(
                        cfg.bot_token, "sendVideo",
                        {"chat_id": chat_id, "caption": _with_clock(text, alert)[:CAPTION_LIMIT],
                         "reply_markup": keyboard, "supports_streaming": "true", **quiet, **thread},
                        {"video": clip}, timeout=120.0,
                    )
                except (urllib.error.URLError, OSError) as exc:
                    log.warning("Alert video not sent to %s (%s); sending the picture", chat_id, telegram_error(exc))
                if resp is not None and not resp.get("ok"):
                    log.warning("Alert video not ok for %s (%s); sending the picture", chat_id, resp.get("description"))
                    resp = None
            if resp is None and image:
                resp = post_multipart(
                    cfg.bot_token, "sendPhoto",
                    {"chat_id": chat_id, "caption": body[:CAPTION_LIMIT], "reply_markup": keyboard, **quiet,
                     **thread},
                    {"photo": ("alert.jpg", image, "image/jpeg")},
                )
            elif resp is None:
                resp = post(cfg.bot_token, "sendMessage",
                            {"chat_id": chat_id, "text": body, "reply_markup": keyboard, **quiet, **thread})
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
    video_wait: float = VIDEO_WAIT_SEC
    # Alerts waiting for their video (alert id -> what send_alert was given), so alert and clip are one message.
    _held: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _held_lock: Any = field(default_factory=threading.Lock)

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
                   lang: str = "en", reply_to: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send an alert. A new alert waits for its video (:meth:`send_clip`), at most ``video_wait`` seconds,
        so the owner gets one message: the clip, the caption and the buttons. Never raises.

        The answer says it is on its way (``sent``); the picture goes out instead if the video never comes.
        An alert already delivered (the escalation reminder) goes out at once. *reply_to* ({"chat_id",
        "message_id"}, the event's first alert) threads it under that alert, held or not.
        """
        alert_id = str((alert or {}).get("alert_id") or "") if isinstance(alert, dict) else ""
        delivered = bool(alert_id and self.index is not None and self.index.messages(alert_id))
        if not alert_id or delivered or self.cfg.dry_run or not self.cfg.enabled:
            return self._deliver(alert, text, image, silent, lang, reply_to=reply_to)
        try:
            with self._held_lock:
                if alert_id in self._held:
                    return {"sent": True, "held": "waiting for the video"}
                self._held[alert_id] = {"alert": alert, "text": text, "image": image, "silent": silent, "lang": lang,
                                        "reply_to": reply_to}
            timer = threading.Timer(self.video_wait, self._release, args=(alert_id, ""))
            timer.daemon = True
            timer.start()
        except Exception as exc:  # noqa: BLE001 - without a timer the alert goes out now, with its picture
            log.warning("Alert %s not held for its video (%s); sending it now", alert_id, exc)
            return self._release(alert_id, "") or self._deliver(alert, text, image, silent, lang, reply_to=reply_to)
        return {"sent": True, "held": "waiting for the video"}

    def _deliver(self, alert: Dict[str, Any], text: str, image: Optional[bytes], silent: bool, lang: str,
                 video: Optional[str] = None, reply_to: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = send_alert(self.cfg, self.index, alert, text, image, post=telegram_notify._http_post,
                         post_multipart=telegram_notify._http_post_multipart, feed=self.feed, silent=silent,
                         lang=lang, video=video, reply_to=reply_to)
        self._note_alert(alert, res)  # into the chat history only once it has really gone out
        return res

    def _release(self, alert_id: str, video: str) -> Optional[Dict[str, Any]]:
        """Send the held alert *alert_id*, as *video* when it can be read, else with its picture. None when nothing
        was held (already sent, or never held). Never raises."""
        try:
            with self._held_lock:
                held = self._held.pop(alert_id, None)
            if held is None:
                return None
            if not video:
                log.info("Alert %s goes out without its video", alert_id)
            res = self._deliver(held["alert"], held["text"], held["image"], held["silent"], held["lang"], video or None,
                                reply_to=held.get("reply_to"))
            if not res.get("sent"):
                log.warning("Alert %s not delivered: %s", alert_id, res)
            return res
        except Exception as exc:  # noqa: BLE001 - runs on a timer thread too
            log.warning("Held alert %s not sent: %s", alert_id, exc)
            return {"sent": False, "error": str(exc)}

    def _note_alert(self, alert: Dict[str, Any], res: Dict[str, Any]) -> None:
        """The v2 assistant keeps a delivered alert in each chat's history as text (its observation), so a follow-up
        question knows the clip. On its own thread: a turn in progress must not hold up the detector. Never raises."""
        try:
            agent = getattr(self.inbox, "agent", None)
            if getattr(agent, "version", 1) != 2 or not isinstance(alert, dict) or not isinstance(res, dict):
                return
            chats = [str(r.get("chat_id")) for r in res.get("results") or [] if isinstance(r, dict) and r.get("ok")]

            def note() -> None:
                for chat_id in chats:
                    try:
                        agent.note_alert(chat_id, alert)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Alert not noted in the chat history: %s", exc)

            if chats:
                threading.Thread(target=note, name="note-alert", daemon=True).start()
        except Exception as exc:  # noqa: BLE001 - the alert itself is already out
            log.warning("Alert not noted in the chat history: %s", exc)

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
        """The alert's video. Call it once the clip has been written (with ``""`` when it could not be).

        An alert still waiting for it goes out now as one message, the video with the alert's caption and
        buttons; an alert already sent with its picture gets the video as a reply under it.
        """
        released = self._release(alert_id, clip_path)
        if released is not None:
            return released
        return send_clip(self.cfg, self.index, alert_id, clip_path,
                         post_multipart=telegram_notify._http_post_multipart, feed=self.feed, silent=silent)


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
    version = agent_version(box_settings)
    if version == 2:
        try:
            from .brain.agent import build_owner_agent, follow_up_camera_receipts  # noqa: PLC0415

            agent, deliverer = build_owner_agent(box_settings, env, mute, cfg, live_dir, archive_dir, log_dir, feed)
            if agent is not None:
                threading.Thread(target=follow_up_camera_receipts, args=(agent.book, agent.registry, deliverer),
                                 name="camera-follow-up", daemon=True).start()
                log.info("Owner assistant v2 (brain) is on.")
        except Exception as exc:  # noqa: BLE001 - the inbox must start whatever the brain does
            log.warning("Owner assistant v2 not available (%s); using v1.", exc)
            agent, deliverer = None, None
    if agent is None:
        try:
            # v1 reads agent_model as a bare OpenAI name; as the fallback of v2 (whose agent_model is
            # "provider:model") it runs on its own default.
            model = make_chat_model(env, str(box_settings.get("agent_model", "gpt-4o-mini")) if version == 1
                                    else "gpt-4o-mini")
            if model is not None:
                agent = OwnerAgent(model, AgentContext(
                    camera_names=list(camera_names), mute_state=mute, feedback_dir=live_dir,
                    roots=lambda: alert_roots(live_dir, archive_dir),
                    retention_days=PRODUCTION_RETENTION_DAYS,
                ))
                if version == 2:
                    log.warning("Owner assistant runs as v1 (the v2 brain could not be built).")
        except Exception as exc:  # noqa: BLE001 - a missing library must not stop the alerts
            log.warning("Owner agent not available (%s); buttons still work.", exc)
    transcriber = voice.make_transcriber(env, str(box_settings.get("transcribe_model") or voice.DEFAULT_MODEL))
    names = list(camera_names)

    def cameras() -> List[str]:
        """The cameras whose ids an answer must never show: the box's list, and the brain's when it has one."""
        try:
            return names + list(agent.registry.snapshot().names) if getattr(agent, "version", 1) == 2 else names
        except Exception:  # noqa: BLE001
            return names

    inbox = TelegramInbox(cfg, agent, index, mute, live_dir, os.path.join(log_dir, "telegram_offset.json"),
                          feed=feed, deliverer=deliverer, archive_dir=archive_dir, lang=box_language,
                          transcriber=transcriber, cameras=cameras)
    try:
        from .scene_chat import for_inbox  # noqa: PLC0415

        inbox.scene = for_inbox(inbox, lambda: list(names), box_language)
    except Exception as exc:  # noqa: BLE001 - without the interview the assistant still works
        log.warning("Install interview not available (%s)", exc)
    assistant = OwnerAssistant(cfg=cfg, index=index, mute=mute, inbox=inbox, feed=feed, deliverer=deliverer,
                               feedback_dir=live_dir, archive_dir=archive_dir)
    if cfg.enabled and not cfg.dry_run:
        assistant.thread = threading.Thread(target=inbox.run, name="telegram-inbox", daemon=True)
        assistant.thread.start()
    return assistant


def is_tag_answer(text: str, replied: Any, request: Dict[str, Any]) -> bool:
    """Is *text* the owner's words for the "Other…" wait *request*? A reply to its question ("מה קורה בסרטון?") always
    is. A reply to the alert, or the first message within ``TAG_IMPLICIT_SEC`` that replies to nothing, is only when
    it is a statement - not a question, a complaint or a command (2026-10-07: "איזה תזכורת, אתה מטומטם ומגזים" was
    saved as the 12:46 clip's explanation)."""
    prompts = [str(p) for p in (request or {}).get("prompt_ids") or []]
    if replied is not None and str(replied) in prompts:
        return True
    return not not_a_judgement(text)


def agent_version(box_settings: Dict[str, Any]) -> int:
    """box.yaml ``agent_version``: 2 (the brain) unless it says 1. Before 2026-10-08 the default was 1, and the box
    ran the old assistant with none of the brain's checks (aliases, claims, the topic camera)."""
    value = box_settings.get("agent_version") if isinstance(box_settings, dict) else None
    try:
        return 1 if int(str(value).strip()) == 1 else 2
    except (TypeError, ValueError, OverflowError):
        if value not in (None, ""):
            log.warning("Invalid agent_version %r; using v2", value)
        return 2


def _clock_of(alert: Dict[str, Any]) -> str:
    """The alert's local time, "02:14"; the alert id ends ``_<epoch>_alert`` when its record has no time."""
    stamp = alert.get("ts")
    if stamp is None:
        parts = str(alert.get("alert_id") or "").split("_")
        stamp = next((p for p in reversed(parts) if p.isdigit()), None)
    try:
        return dt.datetime.fromtimestamp(float(stamp)).strftime("%H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


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
        transcriber: Optional[voice.Transcriber] = None,
        fetch_voice: Optional[Callable[[str], Tuple[bytes, str]]] = None,
        cameras: Optional[Callable[[], Sequence[str]]] = None,
        scene: Any = None,
    ) -> None:
        self.feed = feed
        # The install interview (scene_chat.SceneChat): "מפה" starts it; its answers and buttons never reach the agent.
        self.scene = scene
        # The camera ids no answer may show (brain/style.py swaps them for the family's names).
        self._cameras = cameras
        # A spoken answer to "Other…": fetched from Telegram, then transcribed (None: asked for in writing).
        self.transcriber = transcriber
        self._fetch_voice = fetch_voice or (lambda file_id: voice.download(self.cfg.bot_token, file_id, post=self._post))
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
             entities: Optional[List[Dict[str, Any]]] = None,
             rows: Sequence[Sequence[Sequence[str]]] = ()) -> Dict[str, Any]:
        """Send *text* (with retries) and return Telegram's answer. *undo_data* is a whole Undo callback
        code (``tu:...``); *markup* is used when there are no buttons (a ``force_reply``). *rows* are more button
        rows of ``(label, callback)`` (the yes / no of a house-state request), after the others.

        Every text goes out cleaned (brain/style.py): no closing offers ("אם יש משהו נוסף… אני כאן"), no camera
        ids. A text with *entities* (a mention at a fixed offset) is sent as it is."""
        cameras = self._camera_list()
        if not entities:
            text = clean_outgoing(text, cameras)
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
            # The labels name cameras by the family's names; the choice itself (the id) stays in the agent's state.
            said = detect_language(text) or self._language()
            markup = json.loads(choice_keyboard([replace_camera_ids(str(b), cameras, said) for b in buttons],
                                                question_token))
            if undo_row:
                markup.setdefault("inline_keyboard", []).append(undo_row)
            fields["reply_markup"] = json.dumps(markup)
        elif undo_row:
            fields["reply_markup"] = json.dumps({"inline_keyboard": [undo_row]})
        elif markup:
            fields["reply_markup"] = json.dumps(markup)
        extra = [[{"text": str(label)[:60], "callback_data": str(data)} for label, data in row] for row in rows or ()]
        if extra:
            keyboard = json.loads(fields.get("reply_markup") or "{}")
            keyboard.setdefault("inline_keyboard", []).extend(extra)
            fields["reply_markup"] = json.dumps(keyboard)
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

    def _camera_list(self) -> List[str]:
        try:
            return [str(c) for c in (self._cameras() if self._cameras is not None else ()) if c]
        except Exception as exc:  # noqa: BLE001
            log.warning("Camera list for the answer not read: %s", exc)
            return []

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
                      question_token=getattr(reply, "question_token", ""), rows=getattr(reply, "rows", ()))
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
        if code.startswith("sm:") and self.scene is not None:      # the install interview's [שמור] [תקן] (scene_chat)
            return self.scene.inbox_button(self, query, chat_id, code)
        if code.startswith(("tag:", "tu:")):          # tagging: before the v1/v2 split, both agents share it
            self._on_tag_button(query, message, chat_id, code)
            return
        self._cancel_tag(chat_id, (query.get("from") or {}).get("id"))   # any other tap ends an "Other…" wait
        if getattr(self.agent, "version", 1) == 2 and code.startswith(("cl:", "u:", "hs:", "kn:")):
            try:   # only stops the button's spinner: a dropped connection here must not lose the tap
                self._post(self.cfg.bot_token, "answerCallbackQuery", {"callback_query_id": str(query.get("id"))})
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not stop the button's spinner: %s", exc)
            who = _who(query.get("from") or {})
            if code.startswith("kn:"):            # Cancel / All week under "these are my workers" (Memory Keeper)
                parts = code.split(":")
                if len(parts) != 3 or parts[1] not in ("x", "w") or not parts[2]:
                    log.warning("Ignoring malformed known-people callback")
                    return
                self._note("owner", "button", _button_text(message, code, code), who["name"])
                reply = self.agent.known_button(chat_id, parts[1], parts[2], who)
                if reply is not None and parts[1] == "x":
                    self._edit_buttons(chat_id, message.get("message_id"), {"inline_keyboard": []})
            elif code.startswith("hs:"):            # yes / no to a house-state request (brain/house.py)
                parts = code.split(":")
                if len(parts) != 3 or parts[2] not in ("y", "n") or not parts[1]:
                    log.warning("Ignoring malformed house-state callback")
                    return
                self._note("owner", "button", _button_text(message, code, code), who["name"])
                reply = self.agent.answer_proposal(chat_id, parts[1], parts[2] == "y", who)
            elif code.startswith("cl:"):
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

    def _edit_buttons(self, chat_id: str, message_id: Any, markup: Dict[str, Any]) -> bool:
        """Replace the buttons under *message_id*. False when Telegram would not (too old, gone, offline)."""
        if message_id is None:
            return False
        try:
            resp = self._post(self.cfg.bot_token, "editMessageReplyMarkup",
                              {"chat_id": chat_id, "message_id": str(message_id), "reply_markup": json.dumps(markup)})
        except Exception as exc:  # noqa: BLE001 - the tag is saved; the receipt falls back to a message
            log.warning("Buttons of message %s not replaced: %s", message_id, exc)
            return False
        return bool(isinstance(resp, dict) and resp.get("ok"))

    def _receipt(self, chat_id: str, message_id: Any, label: str, alert_id: str, lang: str) -> bool:
        """The alert's buttons become "✓ Saved: <label>" and Undo."""
        row = [{"text": tr("tag_receipt", lang, label=tr(f"btn_tag_{label}", lang)), "callback_data": "tag:noop"}]
        undo = self._undo_code(alert_id)
        if undo:
            row.append({"text": tr("undo_button", lang), "callback_data": undo})
        return self._edit_buttons(chat_id, message_id, {"inline_keyboard": [row]})

    def _alert_messages(self, chat_id: str, alert_id: str) -> List[int]:
        return [message_id for chat, message_id in self.index.messages(alert_id) if chat == str(chat_id)]

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

    def _undo_tag(self, alert: Dict[str, Any], who: Dict[str, Any], chat_id: str, now: float) -> bool:
        """Take a tag back: an undo record (the newest answer wins), then the training copy the tag made.

        Each step is tried even when the other fails; success requires both stores to agree.
        """
        record_saved = training_done = False
        try:
            save_feedback(self.feedback_dir, alert, Feedback(verdict="none", note=TAG_UNDONE_NOTE, source="button"),
                          "", who, chat_id, now, training_dir=self.training_dir, archive_dir=self.archive_dir)
            record_saved = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Undo record of %s not saved: %s", alert.get("alert_id"), exc)
        try:
            training_done = undo_training_tag(alert, "", who, now, self.training_dir) in ("removed", "noted", "none")
        except Exception as exc:  # noqa: BLE001
            log.warning("Training copy of %s not taken back: %s", alert.get("alert_id"), exc)
        return record_saved and training_done

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
                undone = self._undo_tag(alert, who, chat_id, now)
                on_alert = message.get("message_id") in self._alert_messages(chat_id, alert_id)
                if undone:
                    for alert_message in self._alert_messages(chat_id, alert_id):   # the tag buttons come back
                        self._edit_buttons(chat_id, alert_message, json.loads(alert_keyboard(alert, lang)))
                if undone and on_alert:
                    answered = True
                    self._answer_callback(query, tr("tag_undone", lang))
                    return
                self._answer_callback(query)
                answered = True
                self._say(chat_id, tr("tag_undone" if undone else "tag_undo_partial", lang),
                          reply_to=message.get("message_id"),
                          undo_data="" if undone else self._undo_code(alert_id), lang=lang)
                return
            label = code[4:]
            if label == "noop":                         # the receipt itself: nothing to do
                answered = True
                self._answer_callback(query)
                return
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
                self.pending.demote(chat_id, user_id)
                try:
                    resp = self._ask_for_tag(chat_id, sender, message.get("message_id"), lang)
                except Exception as exc:  # noqa: BLE001 - the owner must learn the question never came
                    log.warning("Tag question not sent: %s", exc)
                    answered = True
                    self._answer_callback(query, tr("tag_ask_failed", lang))
                    return
                self.pending.add(chat_id, user_id, alert_id, now, prompt_id=(resp.get("result") or {}).get("message_id"))
                answered = True                  # the question is out: a lost spinner answer is no failure
                try:
                    self._answer_callback(query)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not stop the button's spinner: %s", exc)
                return
            self._cancel_tag(chat_id, user_id)
            feedback = Feedback(verdict=verdict_for(label, str(alert.get("label") or "")), owner_label=label,
                                tagged_by=who["name"] or str(user_id or ""), source="button")
            save_feedback(self.feedback_dir, alert, feedback, "", who, chat_id, now,
                          training_dir=self.training_dir, archive_dir=self.archive_dir)
            self._answer_callback(query)
            answered = True
            if not self._receipt(chat_id, message.get("message_id"), label, alert_id, lang):
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

    def _wait_for(self, chat_id: str, user_id: Any, message: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
        """The "Other…" wait *message* answers (a reply to its question or alert, or the newest open wait)."""
        replied = (message.get("reply_to_message") or {}).get("message_id")
        reply_alert = None
        if replied is not None:
            replied_alert = self.index.lookup(chat_id, replied)
            reply_alert = str((replied_alert or {}).get("alert_id") or "") or None
        return self.pending.match(chat_id, user_id, now, reply_to=replied, reply_alert=reply_alert)

    def _take_voice(self, chat_id: str, sender: Dict[str, Any], message: Dict[str, Any]) -> Any:
        """True when *message* is a voice answer to an "Other…" wait: it is transcribed and saved as the
        words; when it cannot be, the owner is asked to write them (the wait stays). The transcript (a str) when
        it was heard but is not the clip's words (a question, a complaint): the assistant reads it. False when no
        wait is open. Never raises."""
        spoken = message.get("voice") or message.get("audio")
        if not isinstance(spoken, dict) or not spoken.get("file_id"):
            return False
        lang = self._language()
        try:
            if not self._wait_for(chat_id, sender.get("id"), message, self._now()):
                return False
        except Exception as exc:  # noqa: BLE001
            log.warning("Voice message not matched to a tag: %s", exc)
            return False
        heard = ""
        try:
            if self.transcriber is not None:
                audio, name = self._fetch_voice(str(spoken["file_id"]))
                heard = " ".join(str(self.transcriber(audio, name, lang) or "").split())
        except Exception as exc:  # noqa: BLE001 - the owner is asked to write instead
            log.warning("Voice message not transcribed: %s", exc)
        if heard:
            return self._take_tag_text(chat_id, sender, heard, message, spoken=True) or heard
        try:
            self._say(chat_id, tr("tag_voice_failed", lang), reply_to=message.get("message_id"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not ask for the tag in writing: %s", exc)
        return True

    def _take_tag_text(self, chat_id: str, sender: Dict[str, Any], text: str, message: Dict[str, Any],
                       spoken: bool = False) -> bool:
        """True when *text* was handled as the words of an "Other…" tag (or its asker's late reply to the
        question): then neither agent sees it. Only a text meant for the wait is taken - one that replies
        to nothing, to the question, or to the alert's own message; a reply to anything else (another
        alert, an answer of the assistant) goes to the agent and the wait stays. A command ("/...") ends
        the waits and goes on as usual. *spoken*: *text* is the transcript of a voice message."""
        user_id = sender.get("id")
        if text.startswith("/") and not spoken:
            self._cancel_tag(chat_id, user_id)
            return False
        lang = self._language()
        taken = False
        try:
            now = self._now()
            replied = (message.get("reply_to_message") or {}).get("message_id")
            request = self._wait_for(chat_id, user_id, message, now)
            who = _who(sender)
            if not request:
                if replied is not None and self.pending.expired_prompt(chat_id, replied, user_id, now):
                    self._note("owner", "message", text, who["name"])
                    self._say(chat_id, tr("tag_expired", lang), reply_to=message.get("message_id"))
                    return True
                return False
            if not is_tag_answer(text, replied, request):
                return False          # meant for the assistant; the wait stays open
            taken = True
            self.pending.demote(chat_id, user_id, request["request_id"])
            alert_id = request["alert_id"]
            alert = self.index.alert(alert_id) or {"alert_id": alert_id}
            self._note("owner", "message", text, who["name"], alert)
            self._mark_answered(alert)
            words = text[:MAX_TAG_TEXT_CHARS]
            feedback = Feedback(verdict=verdict_for("other", str(alert.get("label") or "")), owner_label="other",
                                owner_text=words, tagged_by=who["name"] or str(user_id or ""),
                                source="voice" if spoken else "text", request_id=request["request_id"],
                                transcript=words if spoken else "")
            save_feedback(self.feedback_dir, alert, feedback, text, who, chat_id, now,
                          training_dir=self.training_dir, archive_dir=self.archive_dir)
            self.pending.complete(request["request_id"])
            for alert_message in self._alert_messages(chat_id, alert_id):
                self._receipt(chat_id, alert_message, "other", alert_id, lang)
            self._say(chat_id, tr("tag_saved_explanation", lang, time=_clock_of(alert)),
                      reply_to=message.get("message_id"), undo_data=self._undo_code(alert_id), lang=lang)
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
        if not self._allowed(chat_id) or sender.get("is_bot"):
            return
        if text and self.scene is not None and self.scene.inbox_text(self, chat_id, sender, text):   # "מפה" (scene_chat)
            return
        if not text:
            heard = self._take_voice(chat_id, sender, message)
            if heard is True:
                return
            if not (isinstance(heard, str) and heard.strip()):
                replied = (message.get("reply_to_message") or {}).get("message_id")
                if self.pending.needs_text(chat_id, sender.get("id"), self._now(), replied):
                    self._say(chat_id, tr("tag_need_text", self._language()), reply_to=message.get("message_id"))
                return
            text = heard.strip()          # spoken, but not the clip's words: the assistant reads it
        elif self._take_tag_text(chat_id, sender, text, message):
            return
        replied = (message.get("reply_to_message") or {}).get("message_id")
        alert = self.index.lookup(chat_id, replied) if replied is not None else None
        threaded = alert is not None
        v2 = getattr(self.agent, "version", 1) == 2
        # A message that replies to nothing is bound to the newest alert only when it may be a judgement of it: a
        # question, a complaint or a command never is (2026-10-07: "על איזה סרטון אתה מדבר", "יא מטומטם" and
        # "די עם ההודעה" were each filed as a verdict on the newest alert).
        if alert is None and not not_a_judgement(text):
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
