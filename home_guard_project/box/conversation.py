"""Per-chat conversation history for the owner's assistant.

The box restarts whenever a setting changes, so history kept only in memory
would be lost between the owner's messages. This keeps one small JSON file per
chat - the running conversation, oldest first - so the assistant always sees
the thread it is in, across restarts. Only the user's and the assistant's text
is kept (not tool calls), so the model reads a clean conversation, and the file
is capped so it cannot grow without bound.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List

log = logging.getLogger("box.conversation")

WINDOW = 40            # how many recent messages the model is shown (~20 turns)
MAX_PERSIST = 1000     # how many messages are kept on disk per chat

_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _chat_file(directory: str, chat_id: Any) -> str:
    name = _SAFE.sub("_", str(chat_id)) or "chat"
    return os.path.join(directory, f"{name}.json")


class ConversationStore:
    """The running conversation per chat, persisted as JSON so it survives a restart."""

    def __init__(self, directory: str, window: int = WINDOW, max_persist: int = MAX_PERSIST) -> None:
        self._dir = directory
        self._window = window
        self._max = max_persist

    def _load(self, chat_id: Any) -> List[Dict[str, Any]]:
        try:
            with open(_chat_file(self._dir, chat_id), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return []
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            return []
        return [m for m in messages if isinstance(m, dict) and m.get("role") and m.get("content")]

    def history(self, chat_id: Any) -> List[Dict[str, str]]:
        """The recent window as chat messages (role/content), oldest first - for the model's context."""
        return [{"role": m["role"], "content": m["content"]} for m in self._load(chat_id)[-self._window:]]

    def append(self, chat_id: Any, user_text: str, assistant_text: str) -> None:
        """Add one turn - the owner's message and the assistant's reply - and persist it. Never raises."""
        now = time.time()
        messages = self._load(chat_id)
        messages.append({"role": "user", "content": str(user_text), "ts": now})
        messages.append({"role": "assistant", "content": str(assistant_text), "ts": now})
        messages = messages[-self._max:]
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = _chat_file(self._dir, chat_id)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"messages": messages}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("Could not save conversation for %s: %s", chat_id, exc)
