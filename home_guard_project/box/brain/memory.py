# home_guard_project/box/brain/memory.py
"""The conversation in one chat, with what it pointed at.

Version 1 kept only the words, so later turns trusted the assistant's own
earlier claims. The model now sees every turn of the last 24 hours, and every turn also keeps the event handles it showed (E1,
E2 ... -> real alert ids or files), the receipts of what it did, the question
it is waiting on, and each family member's language. "Both" and "the second
one" are resolved against handles in code, and a short reply like "8" is
answered in the language its writer used last. Same file names as version 1;
a version-1 file is upgraded when read.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..conversation import _chat_file
from .i18n import DEFAULT_LANG, SUPPORTED_LANGS, detect_language, language_override

log = logging.getLogger("box.brain.memory")

VERSION = 2
_HANDLE_RE = re.compile(r"^E\d+$")
HISTORY_HOURS = 24.0  # the model sees the whole conversation of the last day...
MAX_HISTORY_TURNS = 60  # ...but never more than this many turns
KEEP_TURNS = 500      # turns kept on disk
KEEP_HANDLES = 150    # more than one turn can show (a summary details up to 60 events)


@dataclass
class ChatState:
    turns: List[Dict[str, Any]] = field(default_factory=list)
    handles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    next_handle: int = 1
    pending: Optional[Dict[str, Any]] = None
    languages: Dict[str, str] = field(default_factory=dict)
    overrides: Dict[str, str] = field(default_factory=dict)

    def add_handle(self, kind: str, ref: str, camera: str = "", ts: float = 0.0, summary: str = "") -> str:
        for handle, entry in self.handles.items():
            if isinstance(entry, dict) and entry.get("kind") == kind and entry.get("ref") == ref:
                return handle
        handle = f"E{self.next_handle}"
        self.next_handle += 1
        self.handles[handle] = {"kind": kind, "ref": ref, "camera": camera, "ts": ts, "summary": summary}
        while len(self.handles) > KEEP_HANDLES:
            oldest = min(self.handles, key=lambda h: int(h[1:]))
            del self.handles[oldest]
        return handle

    def resolve(self, handle: str) -> Optional[Dict[str, Any]]:
        return self.handles.get(str(handle or "").strip().upper())

    def last_turn_handles(self) -> List[str]:
        if not self.turns or not isinstance(self.turns[-1], dict):
            return []
        handles = self.turns[-1].get("handles")
        return list(handles) if isinstance(handles, list) else []

    def language_for(self, speaker: str, text: str, default: str = DEFAULT_LANG) -> str:
        """The reply language: what this person explicitly asked for, else the language of the message, else the
        language they used last, else *default* (the box's configured language). Only supported languages."""
        chosen = language_override(text)
        if chosen in SUPPORTED_LANGS:
            self.overrides[speaker] = chosen
        if self.overrides.get(speaker) in SUPPORTED_LANGS:
            return self.overrides[speaker]
        found = detect_language(text)
        if found in SUPPORTED_LANGS:
            self.languages[speaker] = found
            return found
        last = self.languages.get(speaker)
        if last in SUPPORTED_LANGS:
            return last
        return default if default in SUPPORTED_LANGS else DEFAULT_LANG

    def add_turn(self, speaker: str, text: str, reply: str, handles: List[str], receipts: List[str],
                 ts: float) -> None:
        self.turns.append({"speaker": speaker, "text": text, "reply": reply, "handles": list(handles),
                           "receipts": list(receipts), "ts": ts})
        self.turns = self.turns[-KEEP_TURNS:]

    def _handle_note(self, handle: str) -> str:
        entry = self.handles.get(handle)
        if not entry:
            return f"{handle}=(forgotten)"
        when = ""
        if entry.get("ts"):
            try:
                when = dt.datetime.fromtimestamp(entry["ts"]).strftime("%a %d %b %H:%M")
            except (ValueError, OverflowError, OSError, TypeError):
                when = ""
        return f"{handle}={entry.get('kind')} {entry.get('camera') or ''} {when}".rstrip()

    def history_messages(self, now: float, hours: float = HISTORY_HOURS,
                         max_turns: int = MAX_HISTORY_TURNS) -> List[Dict[str, str]]:
        """Every turn of the last *hours* (at most *max_turns*), with its handles and receipts."""
        out: List[Dict[str, str]] = []
        recent = [turn for turn in self.turns
                  if isinstance(turn, dict) and now - _num(turn.get("ts")) <= hours * 3600]
        for turn in recent[-max_turns:]:
            out.append({"role": "user", "content": str(turn.get("text") or "")})
            notes = []
            handles = turn.get("handles")
            if isinstance(handles, list) and handles:
                notes.append("handles: " + ", ".join(self._handle_note(str(h)) for h in handles))
            receipts = turn.get("receipts")
            if isinstance(receipts, list) and receipts:
                notes.append("receipts: " + "; ".join(str(r) for r in receipts))
            reply = str(turn.get("reply") or "")
            out.append({"role": "assistant", "content": reply + (f"\n[{' | '.join(notes)}]" if notes else "")})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"version": VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatState":
        """Build a state from saved data, dropping anything malformed."""
        turns: List[Dict[str, Any]] = []
        raw_turns = data.get("turns")
        for t in raw_turns if isinstance(raw_turns, list) else []:
            if not isinstance(t, dict):
                continue
            turn = dict(t)
            turn["ts"] = _num(t.get("ts"))
            turn["text"] = _text(t.get("text"))
            turn["reply"] = _text(t.get("reply"))
            turn["handles"] = _str_list(t.get("handles"))
            turn["receipts"] = _str_list(t.get("receipts"))
            turns.append(turn)
        handles: Dict[str, Dict[str, Any]] = {}
        raw_handles = data.get("handles")
        for key, entry in (raw_handles.items() if isinstance(raw_handles, dict) else []):
            if isinstance(key, str) and _HANDLE_RE.match(key) and isinstance(entry, dict):
                handles[key] = {**entry, "ts": _num(entry.get("ts"))}
        try:
            next_handle = int(data.get("next_handle") or 1)
        except (TypeError, ValueError, OverflowError):
            next_handle = 1
        highest = max((int(h[1:]) for h in handles), default=0)
        pending = data.get("pending")
        if not (isinstance(pending, dict) and isinstance(pending.get("question"), str)
                and isinstance(pending.get("choices"), list)):
            pending = None
        return cls(turns=turns, handles=handles, next_handle=max(next_handle, highest + 1, 1), pending=pending,
                   languages=_lang_map(data.get("languages")), overrides=_lang_map(data.get("overrides")))


def _num(value: Any) -> float:
    """A finite float, else 0.0."""
    try:
        number = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _str_list(value: Any) -> List[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _lang_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str) and v in SUPPORTED_LANGS}


def _from_v1(messages: Any) -> ChatState:
    state = ChatState()
    if not isinstance(messages, list):
        return state
    question: Optional[Dict[str, Any]] = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            question = m
        elif m.get("role") == "assistant" and question is not None:
            state.add_turn("", str(question.get("content") or ""), str(m.get("content") or ""), [], [],
                           _num(m.get("ts")))
            question = None
    return state


class ChatMemory:
    """One JSON file per chat. Never raises: a damaged file is a fresh chat, a write error is logged."""

    def __init__(self, directory: str) -> None:
        self._dir = directory

    def load(self, chat_id: Any) -> ChatState:
        try:
            with open(_chat_file(self._dir, chat_id), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return ChatState()
        if not isinstance(data, dict):
            return ChatState()
        try:
            if data.get("version") == VERSION:
                return ChatState.from_dict(data)
            return _from_v1(data.get("messages"))
        except Exception as exc:  # a hand-edited or damaged file is a fresh chat
            log.warning("Ignoring the damaged chat file for %s: %s", chat_id, exc)
            return ChatState()

    def save(self, chat_id: Any, state: ChatState) -> None:
        tmp = ""
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = _chat_file(self._dir, chat_id)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as exc:
            log.warning("Could not save the chat %s: %s", chat_id, exc)
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
