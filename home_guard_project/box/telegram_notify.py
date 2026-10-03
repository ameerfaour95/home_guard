"""Owner notifications over Telegram for a collector box in inference mode.

Free alternative to Twilio: a Telegram bot posts the alert to one or more
chats (a family group counts as one chat_id, so the whole family is reached
with a single free message). Uses only the standard library.

Secret (from api_key.env via the environment): ``TELEGRAM_BOT_TOKEN``.
Non-secret box setting: ``telegram_chat_ids`` (a list, or a comma-separated
string). Delivery is dry-run-safe: nothing is sent unless a token and at least
one chat id are configured and ``dry_run`` is off.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("box.telegram")

_API = "https://api.telegram.org/bot{token}/{method}"


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = ""
    chat_ids: List[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_ids)


def _as_chat_ids(value: Any) -> List[str]:
    """Accept a list, or a comma/space-separated string, of chat ids."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).replace(",", " ").split() if part.strip()]


def load_telegram_config(settings: Dict[str, Any], env: Optional[Dict[str, str]] = None) -> TelegramConfig:
    env = dict(os.environ if env is None else env)
    return TelegramConfig(
        bot_token=env.get("TELEGRAM_BOT_TOKEN", ""),
        chat_ids=_as_chat_ids(settings.get("telegram_chat_ids")),
        dry_run=bool(settings.get("notify_dry_run", False)),
    )


def _http_post(token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
    """POST form fields to a Telegram Bot API method. Isolated for testing."""
    url = _API.format(token=token, method=method)
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_get(token: str, method: str, timeout: float = 15.0) -> Dict[str, Any]:
    url = _API.format(token=token, method=method)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _multipart(fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]) -> Tuple[str, bytes]:
    """Build a multipart/form-data body. files: name -> (filename, data, content_type)."""
    boundary = "----HomeGuard" + uuid.uuid4().hex
    crlf = b"\r\n"
    bb = boundary.encode()
    body = b""
    for name, value in fields.items():
        body += b"--" + bb + crlf
        body += f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf
        body += str(value).encode("utf-8") + crlf
    for name, (filename, data, ctype) in files.items():
        body += b"--" + bb + crlf
        body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode() + crlf
        body += f"Content-Type: {ctype}".encode() + crlf + crlf
        body += data + crlf
    body += b"--" + bb + b"--" + crlf
    return "multipart/form-data; boundary=" + boundary, body


def _http_post_multipart(token: str, method: str, fields: Dict[str, str],
                         files: Dict[str, Tuple[str, bytes, str]], timeout: float = 20.0) -> Dict[str, Any]:
    """POST multipart form data (for file uploads such as sendPhoto). Isolated for testing."""
    url = _API.format(token=token, method=method)
    ctype, body = _multipart(fields, files)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def send_photo(cfg: TelegramConfig, image_bytes: bytes, caption: str = "") -> Dict[str, Any]:
    """Send a JPEG photo with an optional caption to every chat. Never raises."""
    if cfg.dry_run or not cfg.enabled:
        reason = "dry_run" if cfg.dry_run else "not_configured"
        log.info("Telegram photo not sent (%s): %s", reason, caption)
        return {"sent": False, "reason": reason}
    results = []
    for chat_id in cfg.chat_ids:
        try:
            resp = _http_post_multipart(
                cfg.bot_token, "sendPhoto",
                {"chat_id": chat_id, "caption": caption},
                {"photo": ("alert.jpg", image_bytes, "image/jpeg")},
            )
            ok = bool(resp.get("ok"))
            results.append({"chat_id": chat_id, "ok": ok})
            if not ok:
                log.warning("Telegram sendPhoto not ok for %s: %s", chat_id, resp.get("description"))
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Telegram photo failed for %s: %s", chat_id, exc)
            results.append({"chat_id": chat_id, "ok": False, "error": str(exc)})
    return {"sent": any(r["ok"] for r in results), "results": results}


def send_message(cfg: TelegramConfig, text: str) -> Dict[str, Any]:
    """Send *text* to every configured chat. Never raises; returns a status dict."""
    if cfg.dry_run or not cfg.enabled:
        reason = "dry_run" if cfg.dry_run else "not_configured"
        log.info("Telegram not sent (%s): %s", reason, text)
        return {"sent": False, "reason": reason}
    results = []
    for chat_id in cfg.chat_ids:
        try:
            resp = _http_post(cfg.bot_token, "sendMessage", {"chat_id": chat_id, "text": text})
            ok = bool(resp.get("ok"))
            results.append({"chat_id": chat_id, "ok": ok})
            if not ok:
                log.warning("Telegram sendMessage not ok for %s: %s", chat_id, resp.get("description"))
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Telegram send failed for %s: %s", chat_id, exc)
            results.append({"chat_id": chat_id, "ok": False, "error": str(exc)})
    return {"sent": any(r["ok"] for r in results), "results": results}


def alert_text(command: str, summary: str = "", reason: str = "") -> Optional[str]:
    """The message for an alert_command, or None for a command that sends nothing."""
    detail = summary if not reason else f"{summary} - {reason}"
    detail = detail.strip() or "activity detected"
    if command == "[send_message]":
        return f"\U0001F7E1 Home Guard: {detail}"
    if command == "[call_owner]":
        return f"\U0001F6A8 Home Guard ALERT: {detail}"
    return None


def graded_alert_text(label: str, camera: str, summary: str, why: str = "", lang: str = "en") -> str:
    """The alert the owner reads: the label and camera first, then what happened, then why (not for normal)."""
    from .brain.i18n import t  # noqa: PLC0415

    key = {"normal": "alert_normal", "suspicious": "alert_suspicious",
           "escalation": "alert_escalation"}.get(label, "alert_unclassified")
    lines = [t(key, lang, camera=camera), (summary or "").strip() or "activity detected"]
    if label in ("suspicious", "escalation") and (why or "").strip():
        lines.append(t("alert_why", lang, why=why.strip()))
    return "\n".join(lines)


def notify(cfg: TelegramConfig, command: str, summary: str = "", reason: str = "",
           image: Optional[bytes] = None) -> Dict[str, Any]:
    """Dispatch on an alert_command. Telegram cannot place calls, so
    ``[call_owner]`` is sent as a prominent urgent message. When *image* (JPEG
    bytes) is given, the alert is sent as a photo with the text as its caption.
    """
    text = alert_text(command, summary, reason)
    if text is None:
        return {"command": command, "sent": False}
    if image:
        return {"command": command, "telegram": send_photo(cfg, image, text)}
    return {"command": command, "telegram": send_message(cfg, text)}


def discover_chats(token: str) -> List[Dict[str, str]]:
    """List chats the bot can currently see (from recent updates), for setup.

    Returns one entry per distinct chat: ``{id, type, title}``. A chat only
    appears after someone messages the bot or adds it to a group and a message
    is posted there.
    """
    resp = _http_get(token, "getUpdates")
    seen: Dict[str, Dict[str, str]] = {}
    for upd in resp.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid is None:
            continue
        title = chat.get("title") or " ".join(
            x for x in (chat.get("first_name"), chat.get("last_name")) if x
        ) or chat.get("username") or ""
        seen[str(cid)] = {"id": str(cid), "type": chat.get("type", ""), "title": title}
    return list(seen.values())


def _api_get(token: str, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 15.0) -> Dict[str, Any]:
    """GET a Bot API method, with optional query params. Isolated for testing."""
    url = _API.format(token=token, method=method)
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def check_group_readiness(token: str, chat_id: str, get: Any = _api_get) -> Dict[str, Any]:
    """Whether the assistant can actually work in *chat_id*.

    Two things must hold: the bot must be IN the chat (a left/kicked bot cannot
    post, so alerts get 403), AND it must be able to read the owner's typed
    messages - which in a group means its privacy mode is off
    (``can_read_all_group_messages``) or it is an administrator; otherwise it
    receives only button taps, replies to its own messages and @mentions.
    Privacy off does NOT imply membership, so membership is always checked.
    Uses getMe and getChatMember only - never getUpdates - so it never disturbs
    a running poller. Returns ``{"ready", "reason", "bot", "status"}``; never raises.
    """
    try:
        me = (get(token, "getMe") or {}).get("result") or {}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ready": False, "reason": f"cannot reach Telegram ({exc})", "bot": "", "status": ""}
    bot = me.get("username") or ""
    can_read_all = bool(me.get("can_read_all_group_messages"))
    try:
        member = (get(token, "getChatMember", {"chat_id": chat_id, "user_id": me.get("id")}) or {}).get("result") or {}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ready": False, "reason": f"cannot check the bot's membership ({exc})", "bot": bot, "status": ""}
    status = str(member.get("status") or "")
    if status in ("administrator", "creator"):
        return {"ready": True, "reason": "the bot is a group admin", "bot": bot, "status": status}
    if status == "member":
        if can_read_all:
            return {"ready": True, "reason": "the bot is in the group and its privacy mode is off",
                    "bot": bot, "status": status}
        return {"ready": False, "bot": bot, "status": status,
                "reason": "the bot is in the group but privacy mode is on and it is not an admin - "
                          "make it an admin, or disable privacy in BotFather and re-add it"}
    if status == "restricted":
        return {"ready": False, "bot": bot, "status": status,
                "reason": "the bot is restricted in the group - allow it to send messages, or make it an admin"}
    return {"ready": False, "bot": bot, "status": status,
            "reason": f"the bot is not in the group (status={status or 'unknown'}) - add it back to the chat"}
