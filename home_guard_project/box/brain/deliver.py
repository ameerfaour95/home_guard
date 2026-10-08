# home_guard_project/box/brain/deliver.py
"""Telegram sends for the assistant that say whether they arrived.

Version 1 queued a video and then told the owner "here it is" without checking
Telegram's answer. Every send here returns ``{"ok", "message_id", "error"}``
from Telegram's own reply; a photo or video is retried once. Nothing here
raises.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

from .. import telegram_notify

log = logging.getLogger("box.brain.deliver")

CAPTION_LIMIT = 1024


def choice_keyboard(choices: Sequence[str], token: str = "") -> str:
    try:
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ValueError("choices must be a sequence of labels")
        return json.dumps({"inline_keyboard": [[{"text": str(c)[:60], "callback_data": f"cl:{token}:{i}"}]
                                               for i, c in enumerate(choices)]})
    except Exception as exc:  # noqa: BLE001 - malformed tool arguments must not stop polling
        log.warning("Invalid choice keyboard: %s", exc)
        return '{"inline_keyboard": []}'


def _why(exc: BaseException) -> str:
    try:
        from ..telegram_agent import telegram_error  # noqa: PLC0415 - avoids an import cycle

        return telegram_error(exc)
    except Exception:  # noqa: BLE001
        return str(exc)


class Deliverer:
    def __init__(self, cfg: Any, post: Callable[..., Dict[str, Any]] = telegram_notify._http_post,
                 post_multipart: Callable[..., Dict[str, Any]] = telegram_notify._http_post_multipart,
                 feed: Any = None, retries: int = 1) -> None:
        self.cfg = cfg
        self._post = post
        self._post_multipart = post_multipart
        self.feed = feed
        try:
            self.retries = max(0, int(retries))
        except (TypeError, ValueError, OverflowError):
            log.warning("Invalid media retry count; using one retry")
            self.retries = 1

    def _blocked(self) -> Optional[Dict[str, Any]]:
        if getattr(self.cfg, "dry_run", False):
            return {"ok": False, "message_id": None, "error": "dry_run"}
        if not getattr(self.cfg, "bot_token", ""):
            return {"ok": False, "message_id": None, "error": "not_configured"}
        return None

    def _call(self, send: Callable[[], Dict[str, Any]], attempts: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {"ok": False, "message_id": None, "error": "not sent"}
        for _ in range(max(1, attempts)):
            try:
                resp = send()
                if not isinstance(resp, dict) or type(resp.get("ok")) is not bool:
                    raise ValueError("invalid Telegram response")
                if resp["ok"]:
                    message = resp.get("result")
                    if not isinstance(message, dict):
                        raise ValueError("invalid Telegram message")
                    message_id = message.get("message_id")
                    if message_id is not None and type(message_id) is not int:
                        raise ValueError("invalid Telegram message_id")
                    return {"ok": True, "message_id": message_id, "error": ""}
                result = {"ok": False, "message_id": None, "error": str(resp.get("description") or "not ok")}
            except Exception as exc:  # noqa: BLE001 - includes malformed JSON and injected clients
                result = {"ok": False, "message_id": None, "error": _why(exc)}
        log.warning("Telegram send failed: %s", result["error"])
        return result

    def _note(self, kind: str, text: str, result: Dict[str, Any]) -> None:
        if self.feed is not None:
            try:
                self.feed.add("assistant", kind, text, delivered=bool(result.get("ok")), error=result.get("error", ""))
            except Exception as exc:  # noqa: BLE001 - delivery already happened; never resend for a feed failure
                log.warning("Telegram feed entry not saved: %s", exc)

    def text(self, chat_id: str, text: str, reply_to: Optional[int] = None,
             buttons: Optional[Sequence[str]] = None, silent: bool = False) -> Dict[str, Any]:
        try:
            blocked = self._blocked()
            if blocked:
                return blocked
            from .style import clean_outgoing  # noqa: PLC0415 - no closing offers, no camera ids (2026-10-08)

            fields: Dict[str, str] = {"chat_id": str(chat_id), "text": clean_outgoing(str(text))}
            if reply_to is not None:
                fields["reply_to_message_id"] = str(reply_to)
                fields["allow_sending_without_reply"] = "true"
            if buttons:
                fields["reply_markup"] = choice_keyboard(buttons)
            if silent:
                fields["disable_notification"] = "true"
            result = self._call(lambda: self._post(self.cfg.bot_token, "sendMessage", fields), attempts=1)
            self._note("answer", fields["text"], result)
            return result
        except Exception as exc:  # noqa: BLE001 - bad configuration or tool arguments
            log.warning("Telegram text not sent: %s", exc)
            return {"ok": False, "message_id": None, "error": _why(exc)}

    def _file(self, chat_id: str, path: str, method: str, field: str, ctype: str, caption: str,
              kind: str) -> Dict[str, Any]:
        try:
            blocked = self._blocked()
            if blocked:
                return blocked
            path = os.fspath(path)
            with open(path, "rb") as f:
                data = f.read()
            fields = {"chat_id": str(chat_id)}
            if caption:
                from .style import replace_camera_ids  # noqa: PLC0415

                fields["caption"] = replace_camera_ids(str(caption))[:CAPTION_LIMIT]
            if method == "sendVideo":
                fields["supports_streaming"] = "true"
            result = self._call(lambda: self._post_multipart(
                self.cfg.bot_token, method, fields, {field: (os.path.basename(path), data, ctype)},
                timeout=120.0), attempts=1 + self.retries)
            self._note(kind, os.path.basename(path), result)
            return result
        except Exception as exc:  # noqa: BLE001 - includes invalid paths and captions
            log.warning("Telegram media not sent: %s", exc)
            return {"ok": False, "message_id": None, "error": _why(exc)}

    def photo(self, chat_id: str, path: str, caption: str = "") -> Dict[str, Any]:
        return self._file(chat_id, path, "sendPhoto", "photo", "image/jpeg", caption, "photo")

    def video(self, chat_id: str, path: str, caption: str = "") -> Dict[str, Any]:
        return self._file(chat_id, path, "sendVideo", "video", "video/mp4", caption, "video")

    def typing(self, chat_id: str) -> None:
        try:
            if self._blocked():
                return
            response = self._post(self.cfg.bot_token, "sendChatAction", {"chat_id": str(chat_id), "action": "typing"})
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise ValueError("invalid or unsuccessful Telegram typing response")
        except Exception as exc:  # noqa: BLE001 - a missing typing dot must never matter
            log.warning("Telegram typing failed: %s", exc)
