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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


def notify(cfg: TelegramConfig, command: str, summary: str = "", reason: str = "") -> Dict[str, Any]:
    """Dispatch on an alert_command. Telegram cannot place calls, so
    ``[call_owner]`` is sent as a prominent urgent message.
    """
    detail = summary if not reason else f"{summary} - {reason}"
    detail = detail.strip() or "activity detected"
    if command == "[send_message]":
        return {"command": command, "telegram": send_message(cfg, f"\U0001F7E1 Home Guard: {detail}")}
    if command == "[call_owner]":
        return {"command": command, "telegram": send_message(cfg, f"\U0001F6A8 Home Guard ALERT: {detail}")}
    return {"command": command, "sent": False}


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
