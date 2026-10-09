# home_guard_project/box/brain/tools.py
"""The assistant's tools.

Each tool takes the turn's :class:`ToolContext` and the model's arguments and
returns a JSON-serialisable dict. Cameras are resolved through the house
registry (any alias works; an ambiguous name comes back with its candidates,
an unknown one with the list of cameras). Events are shown to the model as
handles (E1, E2 ...) kept in the chat memory. Tools that act write a receipt
(see receipts.py); the reply's confirmations come from those receipts only.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import unicodedata
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..alert_clips import PRE_SECONDS
from ..archive import AlertRecord
from ..feedback import (MAX_MUTE_HOURS, VERDICTS, Feedback, MuteState, feedback_from_fields, is_insult,
                        is_question, not_a_judgement, save_feedback, verdict_for)
from . import house, media
from . import known_memory as km
from .aliases import normalize
from .events import (
    coverage,
    event_doc,
    filter_events,
    load_events,
    local,
    order_for_mode,
    rank_events,
    read_desc,
    write_desc,
)
from .i18n import LANGUAGE_NAMES, t
from .memory import ChatState
from .media import bounds_text
from .mode import GUARD, hhmm
from .receipts import DONE, FAILED, REQUESTED, Receipt, ReceiptBook
from .registry import HouseSnapshot, Resolution, current_camera, display, resolve_camera
from .vision import QUALITIES, VISION_VERSION

log = logging.getLogger("box.brain.tools")

MAX_FOUND = 8
SUMMARY_MAX_EVENTS = 60
MAX_DESCRIBE_PER_TURN = 5
MAX_MEDIA_PER_TURN = 3
MAX_ASK_FRAMES = 8          # a follow-up question looks at twice the frames of a first look


@dataclass
class Services:
    """What the tools reach outside the turn. Production values are built in agent.build_owner_agent."""

    roots: Callable[[], List[str]]
    desc_dir: str
    feedback_dir: str
    work_dir: str
    mute: Any
    deliver: Any
    vision: Any = None
    grab_photo: Optional[Callable[[str], Dict[str, Any]]] = None
    record_live: Optional[Callable[[str, float], Dict[str, Any]]] = None
    cut_segment: Optional[Callable[..., Any]] = None
    clip_frames: Callable[[str], List[bytes]] = media.clip_frames
    clip_frames_at: Callable[..., List[Tuple[float, bytes]]] = media.clip_frames_at
    set_camera: Optional[Callable[[str, bool], Dict[str, Any]]] = None
    add_alias: Optional[Callable[[str, str, Sequence[str]], List[str]]] = None
    remove_alias: Optional[Callable[[str, str], List[str]]] = None
    request_restart: Optional[Callable[[], None]] = None
    embedder: Any = None
    now: Callable[[], float] = time.time
    retention_days: float = 14.0
    max_mute_hours: float = MAX_MUTE_HOURS
    set_option: Optional[Callable[[str, str], Any]] = None
    read_settings: Optional[Callable[[], Dict[str, Any]]] = None
    alert_settings: Any = None
    house: Any = None          # house_state.HouseStateStore: the one writer of the house state (brain/house.py)
    events: Any = None         # events.EventBook: sessions per camera and the owner's "these are my workers"


@dataclass
class ToolContext:
    turn_id: str
    chat_id: str
    speaker: Dict[str, Any]
    text: str
    lang: str
    mode: str
    snapshot: HouseSnapshot
    state: ChatState
    services: Services
    book: ReceiptBook
    alert_handle: Optional[str] = None
    threaded: bool = False
    receipts: List[Receipt] = field(default_factory=list)
    shown: List[str] = field(default_factory=list)
    sent_paths: List[str] = field(default_factory=list)   # media files already sent this turn
    clarification: Optional[Dict[str, Any]] = None
    after_reply: List[Callable[[], None]] = field(default_factory=list)
    call_key: str = ""
    described: int = 0
    saved: int = 0
    done_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # idempotency within one turn
    camera_states: Dict[str, bool] = field(default_factory=dict)          # cameras changed earlier this turn
    vision_notes: List[str] = field(default_factory=list)                 # vision answers, kept in the history
    results: List[str] = field(default_factory=list)                      # every tool result of the turn, as JSON
    extra_rows: List[Any] = field(default_factory=list)                   # more button rows for the reply (↩ תיוג)


def _err(message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "error": message, **extra}


def _safe_tool(function):
    """Contain malformed model arguments and service failures at the poll-loop boundary."""
    @wraps(function)
    def wrapped(ctx, args):
        try:
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be an object")
            json.dumps(args, allow_nan=False)
            out = function(ctx, args)
            json.dumps(out, allow_nan=False)
            return out
        except Exception as exc:
            log.warning("%s failed: %s", function.__name__, exc)
            return _err(f"{function.__name__} could not complete; check the arguments or try again")
        finally:
            ctx.call_key = ""      # a key never carries over to a later call
    return wrapped


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def _camera_error(ctx: ToolContext, words: Any, res: Resolution) -> Dict[str, Any]:
    if res.candidates:
        return _err(f"{str(words)!r} could mean {', '.join(res.candidates)}; ask the owner which one",
                    candidates=list(res.candidates))
    return _err(f"unknown camera {str(words)!r}; the cameras are: {', '.join(ctx.snapshot.names) or 'none'}")


def _cameras_arg(ctx: ToolContext, value: Any) -> Tuple[Optional[Set[str]], Optional[Dict[str, Any]]]:
    """``(cameras, None)`` - None meaning every camera - or ``(None, error)``."""
    if value in (None, "", []):
        return None, None
    out: Set[str] = set()
    for item in value if isinstance(value, list) else [value]:
        res = resolve_camera(ctx.snapshot, str(item))
        if res.camera is None:
            return None, _camera_error(ctx, item, res)
        out.add(res.camera)
    return out, None


def _range(ctx: ToolContext, args: Dict[str, Any]) -> Tuple[float, float]:
    hours = args.get("last_hours")
    if hours is not None:
        hours = _finite(hours)
    fb = feedback_from_fields({"action": "find", "find": {
        "day": args.get("day"), "from": args.get("time_from"), "to": args.get("time_to"),
        "last_hours": hours, "latest": bool(args.get("latest")),
    }}, _finite(ctx.services.now()), [], _finite(ctx.services.max_mute_hours),
                              _finite(ctx.services.retention_days))
    return fb.query.start_ts, fb.query.end_ts


def _show(ctx: ToolContext, record: AlertRecord) -> str:
    handle = ctx.state.add_handle("event", record.alert_id, record.camera, record.ts, record.summary)
    if handle not in ctx.shown:
        ctx.shown.append(handle)
    return handle


def _record_for(ctx: ToolContext, handle: Any) -> Tuple[Optional[AlertRecord], Optional[Dict[str, Any]]]:
    entry = ctx.state.resolve(str(handle or ""))
    if not entry or entry.get("kind") != "event":
        return None, _err(f"unknown handle {handle!r}; use a handle from find_events or summarize_period")
    record = next((r for r in load_events(ctx.services.roots(), ctx.services.desc_dir)
                   if r.alert_id == entry["ref"]), None)
    if record is None:
        return None, _err("that event is no longer on the box")
    ctx.state.set_topic_event(str(handle).strip().upper(), _finite(ctx.services.now()))
    return record, None


# -- looking things up -----------------------------------------------------------
@_safe_tool
def find_events(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    cameras, bad = _cameras_arg(ctx, args.get("cameras"))
    if bad:
        return bad
    start, end = _range(ctx, args)
    everything = load_events(ctx.services.roots(), ctx.services.desc_dir)
    kinds = {args["kind"]} if args.get("kind") in ("alert", "quiet") else None
    labels = {args["label"]} if args.get("label") in ("normal", "suspicious", "escalation") else None
    hits = filter_events(everything, start, end, cameras, kinds, labels)
    what = str(args.get("what") or "").strip()
    if args.get("latest"):
        hits = sorted(hits, key=lambda r: r.ts, reverse=True)
    elif what:
        hits = rank_events(hits, what, ctx.services.embedder)
    else:
        hits = order_for_mode(hits, ctx.mode)
    shown = hits[:MAX_FOUND]
    return {
        "ok": True,
        "count": len(hits),
        "more": max(0, len(hits) - len(shown)),
        "events": [event_doc(r, _show(ctx, r)) for r in shown],
        "coverage": coverage(ctx.snapshot, everything, start, end),
    }


@_safe_tool
def summarize_period(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    cameras, bad = _cameras_arg(ctx, args.get("cameras"))
    if bad:
        return bad
    start, end = _range(ctx, {"day": args.get("day"), "last_hours": args.get("last_hours")})
    everything = load_events(ctx.services.roots(), ctx.services.desc_dir)
    records = filter_events(everything, start, end, cameras)
    notable = order_for_mode([r for r in records if r.label in ("suspicious", "escalation")], GUARD)
    others = [r for r in records if r.label not in ("suspicious", "escalation")]
    detailed = (notable + others[: max(0, 10 - len(notable))])[:SUMMARY_MAX_EVENTS]
    return {
        "ok": True,
        "total": len(records),
        "by_camera": dict(Counter(r.camera for r in records)),
        "by_label": dict(Counter(r.label or "not assessed" for r in records)),
        "by_kind": dict(Counter(r.kind for r in records)),
        "events": [event_doc(r, _show(ctx, r)) for r in detailed],
        "truncated": len(records) > len(detailed),
        "coverage": coverage(ctx.snapshot, everything, start, end),
    }


def _valid_description(value: Any, guard: bool) -> bool:
    """Only complete, usable descriptions can be reused or persisted."""
    if not isinstance(value, dict):
        return False
    try:
        json.dumps(value, allow_nan=False)
        _finite(value.get("ts"))
        return (isinstance(value.get("text"), str) and bool(value["text"].strip())
                and value.get("quality") in QUALITIES
                and type(value.get("people")) is int and value["people"] >= 0
                and (not guard or (value.get("label") in ("normal", "suspicious", "escalation")
                                   and isinstance(value.get("why"), str))))
    except (TypeError, ValueError, OverflowError):
        return False


def _look_clip(ctx: ToolContext, record: AlertRecord, guard: bool, question: str) -> Dict[str, Any]:
    key = f"{'guard' if guard else 'assistant'}:{VISION_VERSION}:{question.casefold().strip()}"
    cached = read_desc(ctx.services.desc_dir, record.alert_id).get(key)
    if cached is not None:
        if _valid_description(cached, guard):
            return dict(cached, ok=True, cached=True)
        log.warning("Skipping malformed cached description of %s", record.alert_id)
    if ctx.services.vision is None:
        return _err("the vision model is not available on this box")
    if not record.clip_path:
        return _err("the video of that event is no longer on the box")
    if ctx.described >= MAX_DESCRIBE_PER_TURN:
        return _err(f"only {MAX_DESCRIBE_PER_TURN} saved videos can be looked at per message; "
                    "tell the owner how many were not checked")
    ctx.described += 1
    try:
        out = ctx.services.vision.look(record.camera, ctx.services.clip_frames(record.clip_path), guard=guard,
                                       question=question, what="frames from a saved video")
        if not isinstance(out, dict) or type(out.get("ok")) is not bool:
            raise ValueError("vision result must contain a boolean ok field")
        json.dumps(out, allow_nan=False)
        if not out["ok"]:
            return out
        value: Dict[str, Any] = {"text": out["description"], "quality": out["quality"], "people": out["people"],
                                 "ts": ctx.services.now()}
        if guard:
            value.update(label=out.get("label", ""), why=out.get("why", ""))
        if not _valid_description(value, guard):
            raise ValueError("vision returned a malformed description")
    except Exception as exc:
        log.warning("Saved-video vision failed: %s", exc)
        return _err("vision_failed")
    write_desc(ctx.services.desc_dir, record.alert_id, key, value)
    return dict(value, ok=True)


def _keep_description(ctx: ToolContext, handle: str, question: str, text: str) -> None:
    """What a look at a clip found stays with the event in the chat, so a later answer can rely on it."""
    entry = ctx.state.resolve(handle) or {}
    if question.strip():
        ctx.state.add_answer(handle, question, text)
    elif not entry.get("observation"):
        ctx.state.note_observation(handle, text, str(entry.get("visibility") or ""), str(entry.get("label") or ""))
    else:
        ctx.state.add_answer(handle, "what happened?", text)


@_safe_tool
def describe_event(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    handle = str(args.get("handle") or "").strip().upper()
    record, bad = _record_for(ctx, handle)
    if bad:
        return bad
    question = str(args.get("question") or "")[:300]
    out = _look_clip(ctx, record, guard=False, question=question)
    if not out.get("ok"):
        return out
    _keep_description(ctx, handle, question, out["text"])
    return {"ok": True, "handle": handle, "camera": record.camera, "time": local(record.ts),
            "description": out["text"], "quality": out["quality"], "people": out["people"]}


@_safe_tool
def ask_vision(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """A follow-up question about a saved event ("what was in his hand?"): the vision model looks again at more
    frames of the clip (or of one part of it) and answers with the frame that shows it. The chat keeps only the
    answer, as text tied to the event; the pictures stay on the box."""
    now = _finite(ctx.services.now())
    handle = (str(args.get("event_id") or "").strip().upper() or ctx.state.topic_event(now)
              or str(ctx.alert_handle or ""))
    if not handle:
        return _err("no event is being talked about; find it with find_events or ask the owner which event")
    if ctx.services.vision is None:
        return _err("the vision model is not available on this box")
    record, bad = _record_for(ctx, handle)
    if bad:
        return bad
    question = (str(args.get("question") or "").strip() or ctx.text.strip())[:300]
    window = args.get("time_range")
    if window is not None and not isinstance(window, dict):
        return _err("time_range must be {\"from_sec\": number, \"to_sec\": number}")
    entry = ctx.state.resolve(handle) or {}
    for item in entry.get("answers") or [] if not window else []:
        if isinstance(item, dict) and str(item.get("q") or "").casefold() == question.casefold():
            return {"ok": True, "handle": handle, "camera": record.camera, "answer": item.get("a"),
                    "frame": item.get("frame", 0), "time": item.get("at", ""), "cached": True}
    if not record.clip_path:
        return _err("the video of that event is no longer on the box")
    if ctx.described >= MAX_DESCRIBE_PER_TURN:
        return _err(f"only {MAX_DESCRIBE_PER_TURN} saved videos can be looked at per message")
    clip_start = record.clip_start_ts if record.clip_start_ts else record.ts - 10.0
    trigger = record.trigger_ts if record.trigger_ts else clip_start + PRE_SECONDS
    start = end = None
    if window:
        if window.get("from_sec") is not None:
            start = max(0.0, trigger + _finite(window["from_sec"]) - clip_start)
        if window.get("to_sec") is not None:
            end = max(0.0, trigger + _finite(window["to_sec"]) - clip_start)
    ctx.described += 1
    frames = ctx.services.clip_frames_at(record.clip_path, MAX_ASK_FRAMES, start, end)
    if not frames:
        return _err("the video of that event could not be read")
    out = ctx.services.vision.ask(record.camera, [jpg for _, jpg in frames], question,
                                  LANGUAGE_NAMES.get(ctx.lang, "English"))
    if not isinstance(out, dict) or type(out.get("ok")) is not bool:
        raise ValueError("vision result must contain a boolean ok field")
    if not out["ok"]:
        return {"ok": False, "error": str(out.get("error") or "vision_failed"), "refused": bool(out.get("refused"))}
    frame = int(out.get("frame") or 0)
    at = (dt.datetime.fromtimestamp(clip_start + _finite(frames[frame - 1][0])).strftime("%H:%M:%S")
          if 1 <= frame <= len(frames) else "")
    answer = str(out.get("answer") or "")
    ctx.state.add_answer(handle, question, answer, frame, at)
    ctx.vision_notes.append(f'{handle} asked "{question}": {answer}' + (f" (frame {frame}, {at})" if at else ""))
    return {"ok": True, "handle": handle, "camera": record.camera, "answer": answer, "seen": out.get("seen") is True,
            "frame": frame, "time": at, "frames_looked_at": len(frames)}


@_safe_tool
def assess_event(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    handle = str(args.get("handle") or "").strip().upper()
    record, bad = _record_for(ctx, handle)
    if bad:
        return bad
    out = _look_clip(ctx, record, guard=True, question="")
    if not out.get("ok"):
        if out.get("refused") or out.get("error") in ("vision_failed", "no_answer"):
            return {"ok": True, "handle": handle, "assessment": "unavailable",
                    "reason": "refused" if out.get("refused") else "failed",
                    "note": "Say: activity detected; assessment unavailable. This never means normal."}
        return out
    _keep_description(ctx, handle, "", out["text"])
    return {"ok": True, "handle": handle, "camera": record.camera, "time": local(record.ts),
            "label": out.get("label", ""), "why": out.get("why", ""), "description": out["text"],
            "quality": out["quality"]}


@_safe_tool
def ask_clarification(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    question = str(args.get("question") or "").strip()[:300]
    raw_choices = args.get("choices")
    if raw_choices is not None and not isinstance(raw_choices, list):
        raise ValueError("choices must be a list")
    choices = [str(c).strip()[:60] for c in (raw_choices or []) if str(c).strip()][:5]
    if not question or len(choices) < 2:
        return _err("give one short question and 2 to 5 choices")
    ctx.clarification = {"question": question, "choices": choices, "ts": _finite(ctx.services.now())}
    return {"ok": True, "message": "The question will be sent with buttons; end the turn now."}


# -- acting ----------------------------------------------------------------------
_FILLER = frozenset({
    "this", "that", "one", "it", "here", "there", "the", "a", "yes", "no", "ok", "okay", "these", "those",
    "הזה", "הזאת", "זה", "זאת", "הזו", "זו", "הנה", "פה", "כאן", "כן", "לא", "אוקיי",
    "هذا", "هذه", "ذلك", "تلك", "هنا", "نعم", "لا",
    "אחד", "האחד", "את", "גם", "ההוא", "ההיא", "שם", "אותו", "אותה", "ההם",
    "هاد", "هيدا", "هاي", "هون", "هذي", "هيك",
    "is", "it's", "that's", "what", "about", "and", "too", "also", "please", "him", "them", "so", "just", "me",
})

_ARABIC_MARKS = {c: None for c in list(range(0x064B, 0x0660)) + [0x0670]}


def _edge_ok(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LNM"


def _clean_word(word: str) -> str:
    start, end = 0, len(word)
    while start < end and not _edge_ok(word[start]):
        start += 1
    while end > start and not _edge_ok(word[end - 1]):
        end -= 1
    return word[start:end]


def _words(value: Any) -> List[str]:
    text = str(value).casefold().replace("’", "'").replace("‘", "'").translate(_ARABIC_MARKS)
    return [w for w in (_clean_word(w) for w in text.split()) if w]


def quoted_from(quote: str, text: str) -> bool:
    """True if *quote* is a run of whole words of *text*: two or more, at least one not a pointer word."""
    q, t = _words(quote), _words(text)
    if len(q) < 2 or all(w in _FILLER for w in q):
        return False
    return any(t[i:i + len(q)] == q for i in range(len(t) - len(q) + 1))


def _issue(ctx: ToolContext, tool: str, status: str, target: str = "", detail: Optional[Dict[str, Any]] = None,
           reason: str = "") -> Receipt:
    key, ctx.call_key = ctx.call_key, ""       # the agent's idempotency key goes on the call's first receipt
    detail = dict(detail or {})
    if ctx.speaker.get("name"):
        detail["by"] = str(ctx.speaker["name"])  # which family member asked for it
    receipt = ctx.book.issue(ctx.turn_id, tool, status, target=target, detail=detail, reason=reason, key=key)
    ctx.receipts.append(receipt)
    return receipt


def _result(receipt: Receipt, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": receipt.status != FAILED, "receipt": receipt.id, "status": receipt.status}
    if receipt.reason:
        out["reason"] = receipt.reason
    out.update(extra)
    return out


def _service_result(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("ok")) is not bool:
        raise ValueError("service result must contain a boolean ok field")
    json.dumps(value, allow_nan=False)
    return value


def _media_sent(ctx: ToolContext) -> int:
    return sum(1 for r in ctx.receipts if r.tool in ("send_media", "record_clip", "check_camera")
               and r.status == DONE)


MAX_CAMERA_CHOICES = 8


def _owner_word(ctx: ToolContext, camera: str, words: Any) -> str:
    """The owner's own name for *camera* in *words* ("פרגולה"), or the one already known for the topic."""
    cam = ctx.snapshot.camera(camera)
    wanted = normalize(str(words or ""))
    for alias in cam.aliases if cam is not None else ():
        key = normalize(alias)
        if key and (wanted == key or (len(wanted) > len(key) and wanted.endswith(key) and wanted[0] in "הבלמושכ")):
            return alias
    topic = ctx.state.topic_camera(_finite(ctx.services.now()))
    return topic[1] if topic and topic[0] == camera else ""


def _one_camera(ctx: ToolContext, words: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    res = resolve_camera(ctx.snapshot, str(words or ""))
    if res.camera is None:
        return None, _camera_error(ctx, words, res)
    # Every camera the turn identifies becomes the one being talked about ("give me a picture" next).
    ctx.state.set_topic_camera(res.camera, _owner_word(ctx, res.camera, words), _finite(ctx.services.now()))
    return res.camera, None


def _camera_or_topic(ctx: ToolContext, words: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """A camera the model named, else the camera being talked about, else the owner is asked - with a button per
    camera - instead of a guess (2026-10-05: "give me a picture" sent an unrelated camera)."""
    if words is not None and str(words).strip():
        return _one_camera(ctx, words)
    topic = ctx.state.topic_camera(_finite(ctx.services.now()))
    if topic and ctx.snapshot.camera(topic[0]) is not None:
        return topic[0], None
    choices = (ctx.snapshot.enabled_names or ctx.snapshot.names)[:MAX_CAMERA_CHOICES]
    if len(choices) == 1:
        return choices[0], None
    ctx.clarification = {"question": t("which_camera", ctx.lang), "choices": list(choices),
                         "ts": _finite(ctx.services.now())}
    return None, _err("no camera was named and none is being talked about: the owner is asked which one with "
                      "buttons; end the turn now")


def _aka(ctx: ToolContext, camera: str) -> str:
    """The owner's word for the camera, for the receipt: "camera_3 (פרגולה)"."""
    topic = ctx.state.topic_camera(_finite(ctx.services.now()))
    return topic[1] if topic and topic[0] == camera else ""


@_safe_tool
def check_camera(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    camera, bad = _camera_or_topic(ctx, args.get("camera"))
    if bad:
        return bad
    implied = not str(args.get("camera") or "").strip()
    if not ctx.snapshot.camera(camera).enabled:
        return _result(_issue(ctx, "check_camera", FAILED, camera, {"camera": camera}, "camera_off"))
    if _media_sent(ctx) >= MAX_MEDIA_PER_TURN:
        return _result(_issue(ctx, "check_camera", FAILED, camera, {"camera": camera}, "too_many"))
    shot = ctx.services.grab_photo(camera) if ctx.services.grab_photo else {"error": "no live view"}
    if not isinstance(shot, dict):
        raise ValueError("photo result must be an object")
    if shot.get("error"):
        return _result(_issue(ctx, "check_camera", FAILED, camera, {"camera": camera}, "camera_offline"))
    if not isinstance(shot["image"], str):
        raise ValueError("photo path must be a string")
    sent = _service_result(ctx.services.deliver.photo(ctx.chat_id, shot["image"]))
    detail = {"camera": camera, "message_id": sent.get("message_id")}
    if _aka(ctx, camera):
        detail["aka"] = _aka(ctx, camera)
    receipt = _issue(ctx, "check_camera", DONE if sent.get("ok") else FAILED, camera, detail,
                     "" if sent.get("ok") else "telegram")
    handle = ctx.state.add_handle("photo", shot["image"], camera, ctx.services.now())
    ctx.shown.append(handle)
    ctx.state.topic_event_ref = {}        # the talk is about the live picture now, not an earlier alert's clip
    out = _result(receipt, camera=camera, handle=handle)
    if implied:
        out["used_camera_being_discussed"] = True
    look: Dict[str, Any] = {}
    if ctx.services.vision is not None:
        try:
            with open(shot["image"], "rb") as f:
                look = ctx.services.vision.look(camera, [f.read()], guard=ctx.mode == GUARD)
        except Exception as exc:
            log.warning("Live-photo vision failed: %s", exc)
            look = {"ok": False}
    if ctx.services.vision is not None and (not isinstance(look, dict)
                                           or type(look.get("ok")) is not bool):
        log.warning("Live-photo vision returned a malformed result")
        return dict(out, description_error="the picture could not be described")
    if isinstance(look, dict) and look.get("ok"):
        value = dict(text=look.get("description"), quality=look.get("quality"), people=look.get("people"),
                     label=look.get("label"), why=look.get("why"), ts=ctx.services.now())
        if not _valid_description(value, ctx.mode == GUARD):
            log.warning("Live-photo vision returned a malformed description")
            return dict(out, description_error="the picture could not be described")
        out.update(description=look["description"], quality=look["quality"], people=look["people"])
        ctx.state.note_observation(handle, look["description"])
        if ctx.mode == GUARD:
            out.update(label=look.get("label", ""), why=look.get("why", ""))
    else:
        out["description_error"] = "the picture was taken but could not be described" + (
            " (the model declined)" if isinstance(look, dict) and look.get("refused") else "")
    return out


LOOK_AROUND_PHOTOS = 2


@_safe_tool
def look_around(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """A live look at every camera that is on ("anything outside?", "what's around the house?" with no camera named).

    2026-10-08: the owner's "יש משהו מעניין בחוץ?" got "on which camera?" back. Every camera is looked at; a photo is
    sent only of cameras where people are seen (at most LOOK_AROUND_PHOTOS), the rest is told in words."""
    cams = [c.name for c in ctx.snapshot.cameras if c.enabled]
    if not cams:
        return _err("every camera is off")
    rows: List[Dict[str, Any]] = []
    for cam in cams:
        name = display(ctx.snapshot, cam, ctx.lang)
        shot = ctx.services.grab_photo(cam) if ctx.services.grab_photo else {"error": "no live view"}
        if not isinstance(shot, dict) or shot.get("error") or not isinstance(shot.get("image"), str):
            rows.append({"camera": name, "error": "no picture from this camera now"})
            continue
        look: Dict[str, Any] = {}
        if ctx.services.vision is not None:
            try:
                with open(shot["image"], "rb") as f:
                    look = ctx.services.vision.look(cam, [f.read()], guard=ctx.mode == GUARD) or {}
            except Exception as exc:  # noqa: BLE001 - one camera's failure must not end the look
                log.warning("look_around: vision failed on %s: %s", cam, exc)
                look = {}
        if not (isinstance(look, dict) and look.get("ok")):
            rows.append({"camera": name, "error": "the picture could not be described"})
            continue
        people = look.get("people") if isinstance(look.get("people"), int) else 0
        row: Dict[str, Any] = {"camera": name, "description": look.get("description"), "people": people,
                               "quality": look.get("quality")}
        if ctx.mode == GUARD:
            row.update(label=look.get("label", ""), why=look.get("why", ""))
        handle = ctx.state.add_handle("photo", shot["image"], cam, ctx.services.now())
        ctx.state.note_observation(handle, str(look.get("description") or ""))
        row["handle"] = handle
        if people and _media_sent(ctx) < LOOK_AROUND_PHOTOS and ctx.services.deliver is not None:
            sent = _service_result(ctx.services.deliver.photo(ctx.chat_id, shot["image"]))
            _issue(ctx, "check_camera", DONE if sent.get("ok") else FAILED, cam,
                   {"camera": cam, "message_id": sent.get("message_id")}, "" if sent.get("ok") else "telegram")
            ctx.shown.append(handle)
            row["photo_sent"] = bool(sent.get("ok"))
        rows.append(row)
    seen = [row for row in rows if "error" not in row]
    if not seen:
        # 2026-10-08 replay: with no picture at all the model still answered "all quiet". Nothing seen is not quiet.
        return _err("No camera gave a picture now, so nothing is known about outside. Say exactly that in one line; "
                    "never say it is quiet.", cameras=rows)
    blind = [row["camera"] for row in rows if "error" in row]
    return {"ok": True, "cameras": rows,
            "note": ("Answer in one or two sentences: where people are and what they do; call quiet only the cameras "
                     "that gave a picture." + (f" No picture from: {', '.join(blind)} - say so." if blind else ""))}


# 2026-10-08 live: "שכחת את 7" (a correction meant for the install interview) made the model record a new clip of
# the entrance and send it twice. A new recording only when the owner's message asks for a video or a picture.
_ASKS_FOR_MEDIA = re.compile(
    r"(?<![א-ת])[והשב]?(?:סרטון|סרטונים|וידאו|וידיאו|קליפ|הקלטה|תקליט|הקלט|צלם|תצלם|צילום|תמונה|תראה|הראה|תשלח|שלח|"
    r"שניות|דקה)(?![א-ת])|\b(?:video|clip|record|film|footage|seconds|show me|send me|picture|photo)\b", re.IGNORECASE)


def asks_for_media(text: str) -> bool:
    return bool(_ASKS_FOR_MEDIA.search(str(text or "")))


@_safe_tool
def record_clip(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if str(ctx.text or "").strip() and not asks_for_media(ctx.text):     # no text: an internal call (tests, tools)
        return _err("Not recorded: this message does not ask for a video. If you are not sure what the owner means, "
                    "ask one short question instead of acting.")
    camera, bad = _camera_or_topic(ctx, args.get("camera"))
    if bad:
        return bad
    seconds = int(min(30, max(1, _finite(args.get("seconds") if args.get("seconds") is not None else 10))))
    if not ctx.snapshot.camera(camera).enabled:
        return _result(_issue(ctx, "record_clip", FAILED, camera, {"camera": camera}, "camera_off"))
    if _media_sent(ctx) >= MAX_MEDIA_PER_TURN:
        return _result(_issue(ctx, "record_clip", FAILED, camera, {"camera": camera}, "too_many"))
    rec = ctx.services.record_live(camera, seconds) if ctx.services.record_live else {"ok": False, "error": "error"}
    _service_result(rec)
    if not rec.get("ok"):
        reason = rec.get("error") if rec.get("error") in ("camera_offline", "busy", "camera_unknown") else "error"
        return _result(_issue(ctx, "record_clip", FAILED, camera, {"camera": camera}, reason))
    bounds = bounds_text(_finite(rec["start"]), _finite(rec["end"]))
    if not isinstance(rec["path"], str):
        raise ValueError("recording path must be a string")
    sent = _service_result(ctx.services.deliver.video(ctx.chat_id, rec["path"], caption=f"{display(ctx.snapshot, camera, ctx.lang)} · {bounds}"))
    detail = {"camera": camera, "seconds": seconds, "bounds": bounds, "message_id": sent.get("message_id")}
    if _aka(ctx, camera):
        detail["aka"] = _aka(ctx, camera)
    receipt = _issue(ctx, "record_clip", DONE if sent.get("ok") else FAILED, camera, detail,
                     "" if sent.get("ok") else "telegram")
    handle = ctx.state.add_handle("clip", rec["path"], camera, rec["start"])
    ctx.shown.append(handle)
    if sent.get("ok"):
        ctx.sent_paths.append(str(rec["path"]))
    ctx.state.topic_event_ref = {}
    return _result(receipt, handle=handle, bounds=bounds)


@_safe_tool
def send_media(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    handle = str(args.get("handle") or "").strip().upper()
    entry = ctx.state.resolve(handle)
    if not entry:
        return _err(f"unknown handle {handle!r}; use a handle from an earlier tool result")
    if _media_sent(ctx) >= MAX_MEDIA_PER_TURN:
        return _result(_issue(ctx, "send_media", FAILED, handle, {"kind": "video"}, "too_many"))
    if entry.get("kind") in ("photo", "clip"):
        path, camera = str(entry["ref"]), str(entry.get("camera") or "")
        if path in ctx.sent_paths:
            return {"ok": True, "already_sent": True, "note": "This exact file was already sent in this turn; "
                    "do not send it again and do not mention it twice."}
        if not os.path.isfile(path):
            return _result(_issue(ctx, "send_media", FAILED, handle, {"kind": entry["kind"]}, "not_on_box"))
        if entry["kind"] == "photo":
            sent = _service_result(ctx.services.deliver.photo(ctx.chat_id, path))
            return _result(_issue(ctx, "send_media", DONE if sent.get("ok") else FAILED, handle,
                                  {"kind": "photo", "camera": camera, "message_id": sent.get("message_id")},
                                  "" if sent.get("ok") else "telegram"))
        bounds = ""
    else:
        record, bad = _record_for(ctx, handle)
        if bad:
            return bad
        camera = record.camera
        if not record.clip_path:
            return _result(_issue(ctx, "send_media", FAILED, handle, {"kind": "video", "camera": camera},
                                  "not_on_box"))
        path = record.clip_path
        clip_start = record.clip_start_ts if record.clip_start_ts else record.ts - 10.0
        bounds = bounds_text(clip_start, record.ts)
        if args.get("from_sec") is not None or args.get("seconds") is not None:
            trigger = record.trigger_ts if record.trigger_ts else clip_start + PRE_SECONDS
            try:
                start = trigger + _finite(args.get("from_sec") if args.get("from_sec") is not None else -PRE_SECONDS)
                seconds = min(60.0, max(1.0, _finite(args.get("seconds") if args.get("seconds") is not None else 10)))
            except (TypeError, ValueError) as exc:
                log.warning("Invalid saved-video segment arguments: %s", exc)
                return _err("from_sec and seconds must be numbers")
            os.makedirs(ctx.services.work_dir, exist_ok=True)
            out_path = os.path.join(ctx.services.work_dir, f"{record.alert_id}_{int(start)}_{int(seconds)}.mp4")
            span = ctx.services.cut_segment(path, clip_start, record.ts, start, seconds, out_path) \
                if ctx.services.cut_segment else None
            if not span:
                return _result(_issue(ctx, "send_media", FAILED, handle, {"kind": "video", "camera": camera},
                                      "error"))
            path, bounds = out_path, bounds_text(*[_finite(v) for v in span])
    sent = _service_result(ctx.services.deliver.video(ctx.chat_id, path, caption=f"{display(ctx.snapshot, camera, ctx.lang)} · {bounds}".strip(" ·")))
    return _result(_issue(ctx, "send_media", DONE if sent.get("ok") else FAILED, handle,
                          {"kind": "video", "camera": camera, "bounds": bounds, "message_id": sent.get("message_id")},
                          "" if sent.get("ok") else "telegram"))


def _alert_of(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {"alert_id": entry.get("ref"), "camera": entry.get("camera"), "summary": entry.get("summary", ""),
            "ts": entry.get("ts")}


def _stored(ctx: ToolContext) -> Optional[MuteState]:
    """The pause state as a restart would load it, or None when the mute service has no file."""
    path = getattr(ctx.services.mute, "path", None)
    return MuteState(path) if isinstance(path, str) else None


def _pause_on_disk(ctx: ToolContext, now: float, camera: Optional[str], until: float) -> bool:
    stored = _stored(ctx)
    if stored is None:
        return True
    targets = [camera] if camera else list(ctx.snapshot.names)
    return all((stored.muted_until(now, c) or 0.0) >= until for c in targets)


def _resume_not_on_disk(ctx: ToolContext, now: float, camera: Optional[str]) -> bool:
    stored = _stored(ctx)
    if stored is None:
        return False
    targets = [camera] if camera else list(ctx.snapshot.names)
    mute = ctx.services.mute
    return any(stored.is_muted(now, c) and not mute.is_muted(now, c) for c in targets)


@_safe_tool
def pause_alerts(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if not quoted_from(str(args.get("owner_words") or ""), ctx.text):
        return _err("Not paused: pause only when this message asks for it, and owner_words must be copied "
                    "from it (two words or more).")
    cameras, bad = _cameras_arg(ctx, args.get("cameras") or args.get("camera"))
    if bad:
        return bad
    now = _finite(ctx.services.now())
    if args.get("minutes") is not None:
        _finite(args["minutes"])
    fb = feedback_from_fields({"action": "mute", "mute_until": args.get("until"),
                               "mute_minutes": args.get("minutes")}, now, [], _finite(ctx.services.max_mute_hours))
    out = []
    for camera in sorted(cameras) if cameras else [None]:
        feedback = Feedback(action="mute", mute_until=fb.mute_until, camera=camera)
        before = ctx.services.mute.snapshot()            # what Undo puts back (an earlier pause survives)
        # before: what Undo puts back; after: the one entry this call sets, so Undo can tell whether it still holds
        after = {"cameras": {camera: fb.mute_until}} if camera else {"all": fb.mute_until}
        detail = {"camera": camera or "", "until": hhmm(fb.mute_until), "before": before, "after": after}
        try:
            ctx.services.mute.apply(feedback, now)
        except Exception as exc:  # noqa: BLE001 - memory may have changed even though saving failed
            log.warning("Pause apply failed: %s", exc)
            targets = [camera] if camera else list(ctx.snapshot.names)
            held = bool(targets) and all(
                (ctx.services.mute.muted_until(now, c) or 0.0) >= fb.mute_until for c in targets)
            if not held:
                out.append(_issue(ctx, "pause_alerts", FAILED, camera or "all", detail, "error"))
                continue
            detail["saved"] = False
        if "saved" not in detail and not _pause_on_disk(ctx, now, camera, fb.mute_until):
            detail["saved"] = False
        out.append(_issue(ctx, "pause_alerts", DONE, camera or "all", detail))
        alert = _alert_of(ctx.state.resolve(ctx.alert_handle) or {}) if ctx.alert_handle else None
        try:
            save_feedback(ctx.services.feedback_dir, alert, feedback, ctx.text, ctx.speaker, ctx.chat_id, now)
            ctx.saved += 1
        except Exception as exc:  # noqa: BLE001 - the pause happened and has its receipt
            log.warning("Pause applied but its feedback file was not saved: %s", exc)
    ok = any(r.status == DONE for r in out)
    result = {"ok": ok, "status": DONE if ok else FAILED, "receipts": [r.id for r in out],
              "until": hhmm(fb.mute_until)}
    if not ok:
        result["reason"] = "error"
    return result


@_safe_tool
def resume_alerts(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    cameras, bad = _cameras_arg(ctx, args.get("cameras") or args.get("camera"))
    if bad:
        return bad
    now = _finite(ctx.services.now())
    out = []
    for camera in sorted(cameras) if cameras else [None]:
        detail = {"camera": camera or ""}
        try:
            ctx.services.mute.resume(camera, now, cameras=ctx.snapshot.names)
        except Exception as exc:  # noqa: BLE001
            log.warning("Resume failed: %s", exc)
            targets = [camera] if camera else list(ctx.snapshot.names)
            if any(ctx.services.mute.is_muted(now, c) for c in targets):
                out.append(_issue(ctx, "resume_alerts", FAILED, camera or "all", detail, "error"))
                continue
            detail["saved"] = False
        if "saved" not in detail and _resume_not_on_disk(ctx, now, camera):
            detail["saved"] = False
        out.append(_issue(ctx, "resume_alerts", DONE, camera or "all", detail))
    try:
        save_feedback(ctx.services.feedback_dir, None, Feedback(action="resume"), ctx.text, ctx.speaker,
                      ctx.chat_id, now)
        ctx.saved += 1
    except Exception as exc:  # noqa: BLE001
        log.warning("Alerts resumed but the feedback file was not saved: %s", exc)
    ok = any(r.status == DONE for r in out)
    result = {"ok": ok, "status": DONE if ok else FAILED, "receipts": [r.id for r in out]}
    if not ok:
        result["reason"] = "error"
    return result


@_safe_tool
def record_verdict(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    handle = str(args.get("handle") or ctx.alert_handle or "").strip().upper()
    if not handle:
        return _err("no alert is bound to this message; ask the owner which alert they mean")
    entry = ctx.state.resolve(handle)
    if not entry or entry.get("kind") != "event":
        return _err(f"unknown handle {handle!r}")
    verdict = str(args.get("verdict") or "")
    if verdict not in VERDICTS or verdict == "none":
        return _err("verdict must be one of true_alert, false_alarm, real_but_wrong, expected, missed_event")
    if not quoted_from(str(args.get("owner_words") or ""), ctx.text):
        return _err("Not saved: owner_words must quote the owner's judgement from this message (two words or "
                    "more). If the message only points at an event, ask what they want to say about it.")
    if (not_a_judgement(str(args.get("owner_words") or "")) or is_insult(ctx.text)
            or (is_question(ctx.text) and not ctx.threaded)):
        # 2026-10-07: "על איזה סרטון דיברת", "יא מטומטם" and "די עם ההודעה" were each filed as a verdict.
        return _err("Not saved: a question, a complaint or a command is not a judgement of the alert. Answer what "
                    "the owner asked or said; if they ask which alert you meant, name it (time, camera, what was "
                    "seen).")
    feedback = Feedback(verdict=verdict, note=str(args.get("note") or "")[:300])
    save_feedback(ctx.services.feedback_dir, _alert_of(entry), feedback, ctx.text, ctx.speaker, ctx.chat_id,
                  ctx.services.now())
    ctx.saved += 1
    return _result(_issue(ctx, "record_verdict", DONE, handle, {"verdict": verdict}))


@_safe_tool
def set_camera_active(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    camera, bad = _one_camera(ctx, args.get("camera"))
    if bad:
        return bad
    active = args.get("active")
    if type(active) is not bool:
        return _err("active must be a boolean")
    detail = {"camera": camera, "active": active, "chat_id": ctx.chat_id, "lang": ctx.lang}
    current = {c.name: ctx.camera_states.get(c.name, c.enabled) for c in ctx.snapshot.cameras}
    if current[camera] == active:
        if ctx.camera_states.get(camera) is active:   # only because an earlier call this turn asked for it
            return _result(_issue(ctx, "set_camera_active", REQUESTED, camera, dict(detail, already=True)))
        return _result(_issue(ctx, "set_camera_active", DONE, camera, dict(detail, already=True)))
    if not active and sum(1 for on in current.values() if on) <= 1:
        # The last camera stays on: with none, the box would stop listening to this chat.
        return _result(_issue(ctx, "set_camera_active", FAILED, camera, detail, "last_camera"))
    try:
        changed = ctx.services.set_camera(camera, active) if ctx.services.set_camera else {"error": "unavailable"}
        _service_result(changed)
    except Exception as exc:  # noqa: BLE001
        log.warning("Camera change failed: %s", exc)
        return _result(_issue(ctx, "set_camera_active", FAILED, camera, detail, "error"))
    if not changed.get("ok") or changed.get("error"):
        return _result(_issue(ctx, "set_camera_active", FAILED, camera, detail, "error"))
    ctx.camera_states[camera] = active
    if ctx.services.request_restart and ctx.services.request_restart not in ctx.after_reply:
        ctx.after_reply.append(ctx.services.request_restart)
    return _result(_issue(ctx, "set_camera_active", REQUESTED, camera, detail))


@_safe_tool
def set_alias(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    camera, bad = _one_camera(ctx, args.get("camera"))
    if bad:
        return bad
    alias = str(args.get("alias") or "").strip()
    cam = ctx.snapshot.camera(camera)
    already = normalize(alias) in [normalize(a) for a in (cam.aliases if cam else ())] + [normalize(camera)]
    try:
        ctx.services.add_alias(camera, alias, ctx.snapshot.names)
    except (ValueError, TypeError) as exc:
        return _result(_issue(ctx, "set_alias", FAILED, camera, {"camera": camera, "alias": alias}, str(exc)))
    ctx.state.set_topic_camera(camera, alias, _finite(ctx.services.now()))
    # The receipt shows the camera itself, so a wrong camera is seen at once (and undone with one tap).
    detail: Dict[str, Any] = {"camera": camera, "alias": alias, "already": already,
                              "photo": _alias_photo(ctx, camera, alias)}
    return _result(_issue(ctx, "set_alias", DONE, camera, detail))


def _alias_photo(ctx: ToolContext, camera: str, alias: str) -> bool:
    """A live photo of the newly named camera, sent with the receipt. A failed photo never fails the save."""
    cam = ctx.snapshot.camera(camera)
    if ctx.services.grab_photo is None or ctx.services.deliver is None or cam is None or not cam.enabled:
        return False
    try:
        shot = ctx.services.grab_photo(camera)
        if not isinstance(shot, dict) or shot.get("error") or not isinstance(shot.get("image"), str):
            return False
        from .render import _name_before  # noqa: PLC0415 - the camera by its other name, never its id

        label = _name_before(ctx.snapshot, camera, alias, ctx.lang)
        sent = _service_result(ctx.services.deliver.photo(ctx.chat_id, shot["image"], caption=f"{label} = {alias}"))
        return bool(sent.get("ok"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Alias photo of %s not sent: %s", camera, exc)
        return False

# -- settings --------------------------------------------------------------------
SETTING_NAMES = ("alert_hours", "cooldown_minutes", "sensitivity", "language", "quiet_log")
SENSITIVITY_LEVELS = {"low": 0.6, "medium": 0.4, "high": 0.25}   # detector confidence: lower = more sensitive
LANGUAGE_WORDS = {"en": "en", "english": "en", "אנגלית": "en", "he": "he", "hebrew": "he", "עברית": "he"}


def settings_view(settings: Dict[str, Any], lang: str = "en") -> Dict[str, str]:
    malformed = not isinstance(settings, dict)
    source = settings if isinstance(settings, dict) else {}
    def number(key, default, low, high):
        nonlocal malformed
        value = source.get(key, default)
        try:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError("invalid number")
            result = _finite(value)
            if not low <= result <= high:
                raise ValueError("out of range")
            return result
        except (ValueError, TypeError, OverflowError):
            malformed = True
            return default
    start = int(number("alert_start_hour", 0, 0, 23))
    end = int(number("alert_end_hour", 0, 0, 23))
    conf = number("inference_conf", 0.4, 0.05, 0.95)
    cooldown = number("alert_cooldown_sec", 120, 0, 86400)
    code = source.get("owner_language", "en")
    if code is None or code == "":
        code = "en"
    if not isinstance(code, str) or code not in LANGUAGE_NAMES:
        malformed = True
        code = "en"
    if malformed:
        log.warning("Skipped malformed settings while formatting")
    level = min(SENSITIVITY_LEVELS, key=lambda k: abs(SENSITIVITY_LEVELS[k] - conf))
    return {
        "alert_hours": "all day" if start == end else f"{start:02d}:00–{end:02d}:00",
        "cooldown_minutes": f"{cooldown / 60:g} min",
        "sensitivity": f"{level} ({conf:.2f})",
        "language": LANGUAGE_NAMES[code],
        "quiet_log": t("setting_on" if source.get("quiet_log", False) else "setting_off", lang),
    }


def settings_line(settings: Dict[str, Any], lang: str = "en") -> str:
    v = settings_view(settings, lang)
    return (f"SETTINGS: alert hours {v['alert_hours']} · time between alerts per camera {v['cooldown_minutes']} · "
            f"detector sensitivity {v['sensitivity']} · box language {v['language']} (alerts and announcements)"
            f" · {t('settings_quiet_log', lang, state=v['quiet_log'])}")


_ALL_DAY_WORDS = ("all day", "24h", "always", "כל היום", "طوال اليوم")


def _hours(value: Any) -> Optional[Tuple[int, int]]:
    text = str(value or "").strip().lower()
    if text in _ALL_DAY_WORDS:
        return (0, 0)
    match = re.fullmatch(r"(\d{1,2})(?::00)?\s*(?:-|–|to)\s*(\d{1,2})(?::00)?", text)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if start > 24 or end > 24:
        return None                         # "99-88" is a mistake, not 03:00-16:00
    if start == 0 and end == 24:
        return (0, 0)                       # "0-24" is all day
    if start % 24 == end % 24:
        return None                         # "6-6" would silently mean all day; the owner must say so
    return (start % 24, end % 24)            # 24 means midnight


KEYS = {"alert_hours": ("alert_start_hour", "alert_end_hour"), "cooldown_minutes": ("alert_cooldown_sec",),
        "sensitivity": ("inference_conf",), "language": ("owner_language",), "quiet_log": ("quiet_log",)}
DEFAULTS = {"alert_start_hour": 0, "alert_end_hour": 0, "alert_cooldown_sec": 120,
            "inference_conf": 0.4, "owner_language": "en", "quiet_log": False}


@_safe_tool
def change_setting(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if not quoted_from(str(args.get("owner_words") or ""), ctx.text):
        return _err("Not changed: change a setting only when this message asks for it; owner_words must be "
                    "copied from it (two words or more).")
    name = str(args.get("setting") or "")
    if name not in SETTING_NAMES:
        return _err("setting must be one of alert_hours, cooldown_minutes, sensitivity, language, quiet_log")
    if ctx.services.set_option is None or ctx.services.read_settings is None:
        return _err("settings cannot be changed on this box")
    value = args.get("value")
    seen: Dict[str, Any] = {}
    def read_view():
        settings = ctx.services.read_settings()
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")
        json.dumps(settings, allow_nan=False)
        for key, low, high in (("alert_start_hour", 0, 23), ("alert_end_hour", 0, 23),
                               ("inference_conf", 0.05, 0.95), ("alert_cooldown_sec", 0, 86400)):
            if key in settings:
                raw = settings[key]
                if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                    raise ValueError("invalid setting number")
                if not low <= _finite(raw) <= high:
                    raise ValueError("setting number out of range")
        code = settings.get("owner_language", "en")
        if code is not None and (not isinstance(code, str) or code not in ("", "en", "he", "ar")):
            raise ValueError("invalid setting language")
        seen.setdefault("first", settings)
        return settings_view(settings)
    before = read_view()[name]
    before_raw = seen["first"]
    restore = {k: before_raw.get(k, DEFAULTS[k]) for k in KEYS[name]}
    wrote: Dict[str, str] = {}              # the raw values this call writes: Undo acts only while they still hold
    try:
        if name == "alert_hours":
            hours = _hours(value)
            if hours is None:
                return _err('alert_hours must look like "22-06" (whole hours) or "all day"')
            raw = seen["first"].get("alert_start_hour", 0)
            old_start = int(_finite(raw))
            wrote = {"alert_start_hour": str(hours[0]), "alert_end_hour": str(hours[1])}
            ctx.services.set_option("alert_start_hour", str(hours[0]))
            try:
                ctx.services.set_option("alert_end_hour", str(hours[1]))
            except Exception as exc:  # noqa: BLE001 - put the start hour back so the change is all or nothing
                log.warning("End hour write failed: %s", exc)
                try:
                    ctx.services.set_option("alert_start_hour", str(old_start))
                except Exception as back:  # noqa: BLE001
                    log.warning("Start hour rollback failed: %s", back)
                    raise ValueError(f"{exc} (start hour changed to {hours[0]:02d}, end hour unchanged)") from back
                raise
            try:
                after = read_view()[name]
            except Exception as exc:  # noqa: BLE001 - both writes happened; only the confirmation read failed
                log.warning("Hours read-back failed: %s", exc)
                after = settings_view({"alert_start_hour": hours[0], "alert_end_hour": hours[1]})[name]
            return _result(_issue(ctx, "change_setting", DONE, name,
                                  {"setting": name, "old": before, "new": after, "restore": restore,
                                   "wrote": wrote}))
        elif name == "cooldown_minutes":
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError("cooldown_minutes must be a number")
            seconds = int(round(_finite(value) * 60))
            if not 10 <= seconds <= 86400:
                return _err("cooldown_minutes must be between 0.2 and 1440")
            wrote = {"alert_cooldown_sec": str(seconds)}
            ctx.services.set_option("alert_cooldown_sec", str(seconds))
        elif name == "quiet_log":
            word = str(value).strip().lower()
            words = {"on": "true", "true": "true", "yes": "true", "כן": "true", "להפעיל": "true",
                     "off": "false", "false": "false", "no": "false", "לא": "false", "לכבות": "false"}
            if word not in words:
                return _err('quiet_log must be "on" or "off"')
            wrote = {"quiet_log": words[word]}
            ctx.services.set_option("quiet_log", wrote["quiet_log"])
        elif name == "sensitivity":
            text = str(value).strip().lower()
            conf = SENSITIVITY_LEVELS.get(text)
            wrote = {"inference_conf": str(conf if conf is not None else _finite(text))}
            ctx.services.set_option("inference_conf", wrote["inference_conf"])
        else:
            code = LANGUAGE_WORDS.get(str(value).strip().lower())
            if code is None:
                return _err('language must be "en" (English) or "he" (Hebrew)')
            wrote = {"owner_language": code}
            ctx.services.set_option("owner_language", code)
        after = read_view()[name]
    except Exception as exc:  # noqa: BLE001 - BoxConfigError / ValueError: the owner's value was refused
        log.warning("Setting change failed: %s", exc)
        return _result(_issue(ctx, "change_setting", FAILED, name, {"setting": name}, str(exc)))
    return _result(_issue(ctx, "change_setting", DONE, name, {"setting": name, "old": before, "new": after,
                                                             "restore": restore, "wrote": wrote}))

# -- alert types and sensitivity (logic in box/alert_settings.py, shared with v1) ---------------------------
def _alert_types(value: Any) -> List[str]:
    words = value.split(",") if isinstance(value, str) else value
    if not isinstance(words, list) or not words or not all(isinstance(w, str) for w in words):
        raise ValueError("types must be a list or comma-separated string")
    words = [w.strip().casefold() for w in words]
    if any(w not in ("person", "vehicle", "animal", "default", "+person", "-person",
                     "+vehicle", "-vehicle", "+animal", "-animal") for w in words):
        raise ValueError("types must be person, vehicle or animal, changes, or default")
    if any(w.startswith(("+", "-")) for w in words):
        if not all(w.startswith(("+", "-")) for w in words) or any(
                "+" + k in words and "-" + k in words for k in ("person", "vehicle", "animal")):
            raise ValueError("use unambiguous changes or a complete type list")
    elif "default" in words and words != ["default"]:
        raise ValueError("default must be used alone")
    return list(dict.fromkeys(words))


def _alert_values(value: Any) -> Any:
    if isinstance(value, str) and value.strip().casefold() == "default":
        return "default"
    if not isinstance(value, dict) or not value:
        raise ValueError("values must be a type-to-number object or default")
    out = {}
    for key, raw in value.items():
        if not isinstance(key, str) or key.strip().casefold() not in ("person", "vehicle", "animal"):
            raise ValueError("unknown sensitivity type")
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise ValueError("sensitivity must be a number")
        number = _finite(str(raw).strip().rstrip("%"))
        number = number / 100 if number > 1 else number
        if not 0.05 <= number <= 0.95:
            raise ValueError("sensitivity must be between 5 and 95 percent")
        kind = key.strip().casefold()
        if kind in out and out[kind] != number:
            raise ValueError("conflicting sensitivity values")
        out[kind] = number
    return out


def _alert_state(state: Any, camera: Optional[str]) -> Dict[str, Any]:
    """Validate and detach API state before using it as restore data or a confirmation."""
    state = json.loads(json.dumps(state, allow_nan=False))
    if not isinstance(state, dict):
        raise ValueError("alert settings must be an object")
    rows = [state] if camera is not None else [state["house"], *state["cameras"]]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("alert settings row must be an object")
        for key in ("alert_on", "own_alert_on"):
            value = row.get(key)
            if key == "own_alert_on" and value is None:
                continue
            if not isinstance(value, list) or not value or any(
                    k not in ("person", "vehicle", "animal") for k in value):
                raise ValueError("invalid alert types in state")
        for key in ("sensitivity", "own_sensitivity"):
            value = row.get(key)
            if key == "own_sensitivity" and value is None:
                continue
            if not isinstance(value, dict) or not value or _alert_values(value) != value:
                raise ValueError("invalid sensitivity in state")
    return state


def _alert_target(ctx: ToolContext, words: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """``(camera name or None for the house, None)`` or ``(None, error)``. "default" is not the house."""
    if words is not None and not isinstance(words, str):
        return None, _err("camera must be a camera name or house")
    text = str(words or "").strip()
    if not text or text.casefold() in ("house", "the house", "בית", "הבית", "כל הבית", "المنزل"):
        return None, None
    camera, bad = _one_camera(ctx, text)
    return camera, bad


_PATH_LIKE = re.compile(
    r"""'[^'\n]*[\\/][^'\n]*'|"[^"\n]*[\\/][^"\n]*"|[A-Za-z]:[\\/]\S*|(?<!\w)/(?:[\w.\-~]+/)*[\w.\-~]+""")


def _no_paths(message: str) -> str:
    """The owner sees what went wrong, never where on the box's disk."""
    return " ".join(_PATH_LIKE.sub("the file", message).split())


def _intended(tool: str, camera: Optional[str], value: Any, old: Any, own_before: Any) -> Optional[Tuple[Any, Any]]:
    """What a saved change should read back as ``(new, own_after)``; None when it cannot be known without reading."""
    if tool == "set_alert_types":
        if value == ["default"]:
            return None
        if all(w.startswith(("+", "-")) for w in value):
            new = list(old)
            for w in value:
                if w[0] == "+" and w[1:] not in new:
                    new.append(w[1:])
                elif w[0] == "-" and w[1:] in new:
                    new.remove(w[1:])
        else:
            new = list(value)
        return new, (new if camera is not None else None)
    if value == "default":
        return None
    return {**old, **value}, ({**(own_before or {}), **value} if camera is not None else None)


@_safe_tool
def get_alert_settings(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if ctx.services.alert_settings is None:
        return _err("alert settings are not available on this box")
    camera, bad = _alert_target(ctx, args.get("camera"))
    if bad:
        return bad
    try:
        return dict(_alert_state(ctx.services.alert_settings.get_alert_settings(camera), camera), ok=True)
    except ValueError as exc:
        return _err(str(exc))


def _change_alerts(ctx: ToolContext, args: Dict[str, Any], tool: str, key: str, value: Any) -> Dict[str, Any]:
    if not quoted_from(str(args.get("owner_words") or ""), ctx.text):
        return _err("Not changed: change alert settings only when this message asks for it; owner_words must be "
                    "copied from it (two words or more).")
    if ctx.services.alert_settings is None:
        return _err("alert settings are not available on this box")
    raw = args.get("camera")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _result(_issue(ctx, tool, FAILED, "", {"camera": ""}, "which_camera"))
    camera, bad = _alert_target(ctx, raw)
    if bad:
        return bad
    api = ctx.services.alert_settings
    own_key = "own_alert_on" if key == "alert_on" else "own_sensitivity"
    detail: Dict[str, Any] = {"camera": camera or ""}
    try:
        value = _alert_types(value) if key == "alert_on" else _alert_values(value)
        before = _alert_state(api.get_alert_settings(camera), camera)
        old = before["house"][key] if camera is None else before[key]
        detail.update(old=old, own_before=None if camera is None else before.get(own_key))
        if camera is None:
            detail["house_before"] = old
        saved = api.set_alert_types(camera, value) if tool == "set_alert_types" else api.set_sensitivity(camera, value)
    except Exception as exc:
        log.warning("Alert setting change failed: %s", exc)
        return _result(_issue(ctx, tool, FAILED, camera or "house", detail, _no_paths(str(exc))))
    try:
        after = _alert_state(saved, camera)
        detail["new"] = after["house"][key] if camera is None else after[key]
        if camera is not None:
            detail["own_after"] = after.get(own_key)
    except Exception as exc:
        # The change is saved; only the confirmation read failed, so report what was asked for (like change_setting).
        log.warning("Alert setting saved but could not be read back: %s", exc)
        intended = _intended(tool, camera, value, old, detail.get("own_before"))
        if intended is None:
            return _result(_issue(ctx, tool, FAILED, camera or "house", detail, _no_paths(str(exc))))
        detail["new"] = intended[0]
        if camera is not None:
            detail["own_after"] = intended[1]
        return _result(_issue(ctx, tool, DONE, camera or "house", detail))
    return _result(_issue(ctx, tool, DONE, camera or "house", detail), state=after)


@_safe_tool
def set_alert_types(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return _change_alerts(ctx, args, "set_alert_types", "alert_on", args.get("types"))


@_safe_tool
def set_sensitivity(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return _change_alerts(ctx, args, "set_sensitivity", "sensitivity", args.get("values"))


# -- the house state (the commands in code are in house.py; these are for what the parser does not take) ---------
_HOUSE_KINDS = {"asleep": "sleep", "awake": "up", "away": "left", "back": "back", "vacation": "vacation"}


def _until_arg(value: Any, now: float) -> Optional[float]:
    """"2026-10-20" (the end of that day), "2026-10-20 18:00", or "18:00" (the next one); None when not given."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    for fmt, whole_day in (("%Y-%m-%d %H:%M", False), ("%Y-%m-%dT%H:%M", False), ("%Y-%m-%d", True)):
        try:
            moment = dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
        return (moment.replace(hour=23, minute=59) if whole_day else moment).timestamp()
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        return house.next_at(now, text.zfill(5))
    raise ValueError("until must look like 2026-10-20, 2026-10-20 18:00 or 18:00")


def _house_ready(ctx: ToolContext, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if ctx.services.house is None:
        return _err("the house state is not available on this box")
    if not quoted_from(str(args.get("owner_words") or ""), ctx.text):
        return _err("Not changed: change the house state only when this message says so; owner_words must be "
                    "copied from it (two words or more).")
    return None


def _clock_text(ts: Any) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="minutes") if ts else "until the owner says otherwise"


@_safe_tool
def house_state(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    bad = _house_ready(ctx, args)
    if bad:
        return bad
    kind = _HOUSE_KINDS.get(str(args.get("state") or "").strip().lower())
    if kind is None:
        return _err("state must be asleep, awake, away, back or vacation")
    now = _finite(ctx.services.now())
    try:
        until = _until_arg(args.get("until"), now)
    except ValueError as exc:
        return _err(str(exc))
    if kind == "vacation" and until is None:
        return _err("a vacation needs until (the date they are back); ask the owner")
    schedule = house.schedule_of(ctx.services.house, now) or house.DEFAULT_SCHEDULE
    if until is None and kind in ("sleep", "up"):
        until = house.next_at(now, schedule[1] if kind == "sleep" else schedule[0])
    receipt = house.set_state(ctx, kind, until)
    return _result(receipt, state=receipt.detail.get("state"), until=_clock_text(receipt.detail.get("until")))


@_safe_tool
def house_expect(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    bad = _house_ready(ctx, args)
    if bad:
        return bad
    text = str(args.get("text") or "").strip()
    if not text:
        return _err("give the note text, e.g. \"a package\" or \"the plumber at 10:00\"")
    now = _finite(ctx.services.now())
    try:
        until = _until_arg(args.get("until"), now)
    except ValueError as exc:
        return _err(str(exc))
    if until is None:
        today = dt.datetime.fromtimestamp(now)
        until = today.replace(hour=23, minute=59, second=0, microsecond=0).timestamp()
    camera = None
    if str(args.get("camera") or "").strip():
        camera, bad = _one_camera(ctx, args["camera"])
        if bad:
            return bad
    receipt = house.add_expect(ctx, text, until, camera)
    return _result(receipt, text=receipt.detail.get("text"), until=_clock_text(receipt.detail.get("until")))


@_safe_tool
def house_cancel(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    bad = _house_ready(ctx, args)
    if bad:
        return bad
    what = str(args.get("what") or "").strip().lower()
    if what not in ("state", "expect"):
        return _err('what must be "state" (end the current house state) or "expect" (an expecting note)')
    receipts = house.cancel(ctx, what, str(args.get("words") or "").strip())
    ok = any(r.status == DONE for r in receipts)
    return {"ok": ok, "status": DONE if ok else FAILED, "receipts": [r.id for r in receipts],
            **({} if ok else {"reason": receipts[0].reason if receipts else "error"})}


@_safe_tool
def house_status(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if ctx.services.house is None:
        return _err("the house state is not available on this box")
    text, _rows = house.status(ctx)
    return {"ok": True, "status": text, "note": "Pending requests are answered with buttons; tell the owner to "
                                                  "type status to see them."}


# -- the Memory Keeper: "these are my workers" (events.EventBook.mark_known) ------------------------------------
# 2026-10-07: "זה בסדר זה עובדים אצלי שעובדים על הפרגולה" was answered "רשמתי את זה כהתרעה צפויה" and 142 more
# alerts about the same workers followed. The owner's words about who is there now silence that camera's
# "suspicious" alerts (never an escalation) until a time, with a receipt the code writes and a Cancel button.
KNOWN_NOW_SEC = 3600.0
KNOWN_WEEK_SEC = 7 * 86400.0
_KNOWN_NOW = ("now", "only now", "just now", "hour", "an hour", "1h", "עכשיו", "רק עכשיו", "רק כרגע", "כרגע", "שעה")
_KNOWN_WEEK = ("week", "all week", "this week", "7d", "שבוע", "כל השבוע", "השבוע")
_KNOWN_TODAY = ("today", "all day", "end of day", "היום", "כל היום")
_KNOWN_TOMORROW = ("tomorrow", "מחר", "עד מחר")


def end_of_day(now: float, days: int = 0) -> float:
    day = dt.datetime.fromtimestamp(now) + dt.timedelta(days=days)
    return day.replace(hour=23, minute=59, second=0, microsecond=0).timestamp()


def known_until(value: Any, now: float) -> float:
    """When the owner's "they are known" ends: what they said ("only now" = an hour, "all week" = 7 days, a time or
    a date), by default the end of today (23:59 local; an hour when today is already over). Raises ValueError."""
    text = " ".join(str(value or "").strip().lower().split())
    if text in _KNOWN_NOW:
        return now + KNOWN_NOW_SEC
    if text in _KNOWN_WEEK:
        return now + KNOWN_WEEK_SEC
    if text in _KNOWN_TOMORROW:
        return end_of_day(now, 1)
    if not text or text in _KNOWN_TODAY:
        until = end_of_day(now)
        return until if until > now + 60 else now + KNOWN_NOW_SEC
    until = _until_arg(text, now)
    if until is None or until <= now:
        raise ValueError("until must be in the future")
    return until


def until_text(until: float, now: float, lang: str) -> str:
    """"23:59" today, "יום ה' 15.10 18:00" / "Thu 15.10 18:00" on another day."""
    day = dt.datetime.fromtimestamp(until)
    if day.date() == dt.datetime.fromtimestamp(now).date():
        return day.strftime("%H:%M")
    if str(lang).startswith("he"):
        names = ("ב'", "ג'", "ד'", "ה'", "ו'", "שבת", "א'")   # Monday first, as datetime.weekday()
        head = names[day.weekday()]
        head = head if head == "שבת" else f"יום {head}"
        return f"{head} {day.strftime('%d.%m %H:%M')}"
    return day.strftime("%a %d.%m %H:%M")


def in_place(name: str, lang: str) -> str:
    """*name* after the Hebrew "ב" ("ב" + "הפרגולה" is "בפרגולה")."""
    name = str(name or "").strip()
    if str(lang).startswith("he") and len(name) > 2 and name.startswith("ה"):
        return name[1:]
    return name


def _said(quote: Any, text: str) -> bool:
    """*quote* is a run of whole words of *text* (one word is enough: "it's me" is all filler words)."""
    q, words = _words(quote), _words(text)
    return bool(q) and any(words[i:i + len(q)] == q for i in range(len(words) - len(q) + 1))


def _known_camera(ctx: ToolContext, words: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """The camera the owner's "they are known" is about: the one named, else the replied-to alert's camera (after a
    rename too), else the camera being discussed, else the owner is asked with a button per camera."""
    if words is not None and str(words).strip():
        return _one_camera(ctx, words)
    entry = ctx.state.resolve(str(ctx.alert_handle or "")) if ctx.alert_handle else None
    if isinstance(entry, dict) and entry.get("camera"):
        camera = current_camera(ctx.snapshot, str(entry["camera"]))
        if camera:
            return camera, None
    return _camera_or_topic(ctx, None)


def known_where(snapshot: Any, camera: str, lang: str) -> str:
    """"בכל הבית" / "בפרגולה" / "at the pergola": where a mark holds."""
    if not camera:
        return t("known_where_house", lang)
    return t("known_where_camera", lang, camera=in_place(display(snapshot, camera, lang), lang))


def known_line(receipt: Receipt, lang: str, snapshot: Any = None) -> str:
    """The keeper's MEMORY receipt, written by code (never by the model), with the 🧠 of a memory (a tag of a clip
    is 🏷️): "🧠 זכרתי: העובדים בכל הבית, כל יום 08:00–18:00, עד יום ה׳ 15.10", a correction "🧠 עדכנתי: ... (במקום
    23:59)", or "🧠 כבר זוכר: ..."."""
    d = receipt.detail
    house_wide = bool(d.get("house"))
    camera = "" if house_wide else str(d.get("camera") or receipt.target)
    at = _finite(d.get("at") or receipt.ts or time.time())
    mark = {"until": _finite(d["until_ts"]), "daily_from": str(d.get("daily_from") or ""),
            "daily_to": str(d.get("daily_to") or "")}
    who = str(d.get("who") or "")
    where = known_where(snapshot, camera, lang)
    when = km.mark_when(mark, at, lang)
    if d.get("already"):
        return t("mem_already", lang, who=who, where=where, when=when)
    replaced = [r for r in d.get("replaced") or [] if isinstance(r, dict)]
    if replaced:
        olds = []
        for r in replaced:
            parts = []
            old_camera = str(r.get("camera") or "")
            if old_camera != camera:
                parts.append(t("known_only_at", lang, camera=in_place(display(snapshot, old_camera, lang), lang))
                             if old_camera else t("known_where_house", lang))
            old = {"until": _finite(r.get("until") or mark["until"]), "daily_from": str(r.get("daily_from") or ""),
                   "daily_to": str(r.get("daily_to") or "")}
            if abs(old["until"] - mark["until"]) >= 60 or old["daily_from"] != mark["daily_from"]:
                same_day = (not old["daily_from"] and dt.datetime.fromtimestamp(old["until"]).date()
                            == dt.datetime.fromtimestamp(at).date())
                parts.append(dt.datetime.fromtimestamp(old["until"]).strftime("%H:%M") if same_day
                             else km.mark_when(old, at, lang))
            olds.append(", ".join(parts) if parts else when)
        return t("mem_updated", lang, who=who, where=where, when=when, old=", ".join(dict.fromkeys(olds)))
    return t("mem_saved", lang, who=who, where=where, when=when)


def known_rows(receipts: Sequence[Receipt], lang: str) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    """The buttons under a saved memory: a crew's week window gets the one-tap corrections [רק היום] [שבוע ✓]
    [אחר…] and [↩ זיכרון] (``kn:d`` / ``kn:w`` / ``kn:o`` / ``kn:x``); any other mark [ביטול] [כל השבוע]."""
    rows = []
    for r in receipts:
        d = r.detail if isinstance(r.detail, dict) else {}
        known_id = str(d.get("known_id") or "")
        if r.tool != "mark_known" or r.status != DONE or not known_id:
            continue
        if d.get("daily_from"):
            rows.append(((t("btn_today_only", lang), f"kn:d:{known_id}"), (t("btn_week", lang), f"kn:w:{known_id}"),
                         (t("btn_other", lang), f"kn:o:{known_id}")))
            rows.append(((t("btn_undo_memory", lang), f"kn:x:{known_id}"),))
        else:
            rows.append(((t("known_cancel_button", lang), f"kn:x:{known_id}"),
                         (t("known_week_button", lang), f"kn:w:{known_id}")))
    return tuple(rows)


# mark_known is for WHO the people are (2026-10-08 replay: "מדי פעם אני יוצא החוצה" silenced the entrance, and a
# description of what a man did - "he stands by the car talking with his hands" - was saved as "known people").
_IDENTITY = re.compile(
    r"(?<![א-ת])[והשב]?(?:עובד|עובדים|עובדת|פועל|פועלים|אני|אנחנו|הגנן|גנן|השכן|שכן|שכנה|השכנים|משפחה|המשפחה|הילדים|"
    r"ילדים שלי|אשתי|בעלי|אמא|אבא|אחי|אחותי|סבא|סבתא|אורח|אורחים|חברים|חבר שלי|שליח|השליח|מנקה|המנקה|טכנאי|"
    r"הטכנאי|קבלן|הקבלן|מוכר|מוכרים|שלי|שלנו|אצלי|צפוי|מצפה|מצפים)(?![א-ת])|"
    r"\b(?:workers?|my|me|mine|our|we|gardener|neighbou?rs?|family|kids|wife|husband|mom|dad|guests?|friends?|"
    r"delivery|courier|cleaner|technician|contractor|known|expected)\b", re.IGNORECASE)
_ROUTINE = re.compile(r"(?<![א-ת])(?:מדי פעם|לפעמים|בדרך כלל|תמיד|כל יום|כל בוקר|כל ערב|כל לילה)(?![א-ת])|"
                      r"\b(?:sometimes|usually|always|every (?:day|morning|evening|night)|often)\b", re.IGNORECASE)


def identifies_people(text: str) -> bool:
    """The owner's message says who the people are (workers, me, the gardener), not only what they did."""
    return bool(_IDENTITY.search(str(text or "")))


def is_routine(text: str) -> bool:
    """"Sometimes", "every day": a routine, which the box cannot learn yet (case memory, stage 2)."""
    return bool(_ROUTINE.search(str(text or "")))


_HOUSE_ARGS = frozenset({"all", "house", "the house", "whole house", "the whole house", "all cameras", "everywhere",
                         "כל הבית", "בכל הבית", "הבית", "כל המצלמות", "בכל המצלמות", "כולן", "הכל"})


def _house_arg(value: Any) -> bool:
    """The model's camera / scope argument means the whole house ("all", "כל הבית")."""
    text = " ".join(str(value or "").strip().lower().split())
    return bool(text) and (text in _HOUSE_ARGS or km.scope_from_words(text) == km.HOUSE)


def _owner_until(ctx: ToolContext, given: Any, now: float, old: Sequence[Dict[str, Any]], who: str) -> Optional[float]:
    """The end time the OWNER gave (2026-10-09: "מי אמר עד 23:59?"): from their words in this message (with the
    questions it answers), else the model's value when it keeps the time of the mark it corrects ("extend it to
    the entrance" keeps 18:00), else the corrected mark's own time, else a time the owner gave today for the same
    people. None: nobody said - ask."""
    said = km.until_from_words(ctx.text, now)
    if said is not None:
        return said
    if given is not None and str(given).strip():
        try:
            until = known_until(given, now)
        except ValueError:
            until = None
        if until is not None and any(abs(until - _finite(k.get("until"))) < 60 for k in old):
            return until
    if old:
        today = [k for k in old if k.get("daily_to")]
        if today:                                   # a daily window: today's end, not the week's
            return km.until_from_words(str(today[-1]["daily_to"]), now)
        return max(_finite(k.get("until")) for k in old)
    return km.until_said_today(ctx.state, who, now)


def ask_known(ctx: ToolContext, args: Dict[str, Any], need: List[str], now: float) -> Dict[str, Any]:
    """ONE short natural question for the first thing the owner did not say, with button answers (2026-10-09:
    "אה, הם של הפרגולה? עד איזו שעה הם עובדים?" [16:00] [17:00] [18:00] [אחר…]; then, for a crew, "והם עובדים
    רק בפרגולה, או בכל הבית?" [כל הבית] [רק פרגולה]). The answer completes the mark (agent, kind "known"); at most
    these two questions are ever asked about one mark."""
    lang = ctx.lang
    who, camera = str(args.get("who") or ""), str(args.get("camera") or "")
    if camera == "all":
        camera = ""
    name = in_place(display(ctx.snapshot, camera, lang), lang) if camera else ""
    first = need[0]
    if first == "until":
        crew = km.work_group(f"{who} {args.get('owner_words') or ''}")
        question = (t("ask_until_crew", lang, camera=name) if crew and camera
                    else t("ask_until_other", lang, who=who))
        choices = km.likely_hours(now) + [t("btn_other", lang)]
    elif first == "scope":
        question = t("ask_scope", lang, camera=name)
        choices = [t("known_btn_house", lang), t("known_btn_camera", lang, camera=name)]
    else:                                                       # one obvious answer: one confirm
        question = str(args.get("confirm") or "")
        choices = [t("btn_yes", lang), t("btn_no", lang)]
    asked = int(args.get("asked") or 0) + 1
    ctx.clarification = {"question": question, "choices": choices, "ts": now, "kind": "known", "need": list(need),
                         "args": dict(args, asked=asked)}
    return {"ok": False, "asked": question,
            "note": "Nothing is saved yet: the owner is asked this one question. End the turn now."}


@_safe_tool
def mark_known(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    book = ctx.services.events
    if book is None:
        return _err("the box's event book is not available; tell the owner it was not saved")
    who = " ".join(str(args.get("who") or "").split())[:120]
    if not who:
        return _err("who: who the people are, in the owner's words (\"the workers\", \"Ameer\")")
    if not _said(args.get("owner_words"), ctx.text):
        return _err("Not saved: owner_words must be copied exactly from this message.")
    if not identifies_people(ctx.text):
        return _err("Not saved: this message says what happened, not who the people are. Do not call mark_known; "
                    "answer the message itself.")
    if is_routine(ctx.text) and not ctx.alert_handle and not km.work_group(ctx.text):
        return _err("Not saved: a routine ('sometimes', 'every day') is not learned yet. Tell the owner in one line "
                    "that you cannot learn routines yet, and that replying 'זה אני' / 'these are mine' to an alert "
                    "about them stops the alerts about them for that day.")
    now = _finite(ctx.services.now())
    scope_arg = str(args.get("scope") or "").strip().lower()
    house_wide = _house_arg(args.get("camera")) or scope_arg in ("house", "all")
    camera = ""
    if not house_wide:
        found, bad = _known_camera(ctx, args.get("camera"))
        if bad:
            return bad
        camera = str(found)
    seen_at = camera or (ctx.state.topic_camera(now) or ("",))[0]
    marks = km.live_marks(book, now)
    wanted = args.get("replaces")
    wanted = [wanted] if isinstance(wanted, str) else [str(x) for x in wanted or () if x]
    old = [k for k in marks if k.get("id") in wanted] or km.pick(marks, who, camera)
    # A work crew moves around the house: asked once (house-wide first), and the answer is kept for this chat.
    said_scope = km.scope_from_words(ctx.text, ctx.snapshot, strict=True)
    crew = km.work_group(f"{who} {ctx.text}")
    need: List[str] = []
    prefs = ctx.state.prefs if isinstance(ctx.state.prefs, dict) else {}
    if crew and not house_wide and scope_arg != "camera" and not old:
        if said_scope == km.HOUSE or prefs.get("group_scope") == km.HOUSE:
            house_wide = True
        elif said_scope is None and prefs.get("group_scope") != "camera":
            need.append("scope")
    if crew and said_scope is not None:
        prefs["group_scope"] = km.HOUSE if said_scope == km.HOUSE else "camera"
    if not house_wide and any(not k.get("camera") for k in old) and said_scope in (None, km.HOUSE):
        # A whole-house mark of these people stays whole-house: only the owner's "רק ב..." narrows it (2026-10-09
        # replay: "המידע זה עובדים אצלי על הפרגולה" said where they work, not "only the pergola").
        house_wide = True
    if house_wide:
        camera = ""
        old = [k for k in marks if k.get("id") in wanted] or km.pick(marks, who, "")
    until = _owner_until(ctx, args.get("until"), now, old, who)
    if until is None:
        need.insert(0, "until")
    if need:
        return ask_known(ctx, {"who": who, "owner_words": str(args.get("owner_words") or ""),
                               "camera": camera or ("all" if house_wide else ""), "handle": str(ctx.alert_handle or "")},
                         need, now)
    if until <= now:
        return _err("until must be in the future")
    # A crew works more than one day (owner, 2026-10-09 10:15): their hours every day for a week, said plainly in
    # the receipt with one-tap corrections - unless the owner limited it ("רק היום", "לשעתיים").
    window: Dict[str, Any] = {}
    daily_old = [k for k in old if k.get("daily_from")]
    if crew and not km.one_day(ctx.text):
        if daily_old:
            end = dt.datetime.fromtimestamp(until)
            window = {"daily_from": daily_old[-1]["daily_from"], "until": _finite(daily_old[-1]["until"]),
                      "daily_to": end.strftime("%H:%M") if end.date() == dt.datetime.fromtimestamp(now).date()
                      else daily_old[-1]["daily_to"]}
        else:
            entry = ctx.state.resolve(str(ctx.alert_handle or ctx.state.topic_event(now) or ""))
            window = km.crew_window(book, seen_at, until, now,
                                    fallback=_finite(entry.get("ts")) if isinstance(entry, dict) and entry.get("ts")
                                    else None) or {}
        last = km.last_day_from_words(ctx.text, now)
        if window and last is not None and not km._WEEK.search(ctx.text):
            stop = dt.datetime.combine(last, dt.datetime.strptime(window["daily_to"], "%H:%M").time())
            window["until"] = max(stop.timestamp(), until)
        if window and window["daily_from"] >= window["daily_to"]:
            window = {}
    span = _finite(window.get("until") or until)
    daily = {"daily_from": str(window.get("daily_from") or ""), "daily_to": str(window.get("daily_to") or "")}
    detail: Dict[str, Any] = {"camera": camera, "who": who, "until_ts": span, "at": now, "house": house_wide, **daily}
    same = [k for k in old if str(k.get("camera") or "") == camera and abs(_finite(k.get("until")) - span) < 60
            and str(k.get("daily_from") or "") == daily["daily_from"] and str(k.get("daily_to") or "") == daily["daily_to"]]
    if same and len(old) == 1:
        detail.update(known_id=str(same[0].get("id") or ""), already=True, people=int(same[0].get("people") or 0))
        return _result(_issue(ctx, "mark_known", DONE, camera or "house", detail),
                       note="It was already saved like this; nothing changed. Reply with an empty answer.")
    try:
        by = str(ctx.speaker.get("name") or "owner")
        if old:
            saved = book.replace_known([str(k.get("id")) for k in old], camera, who, by=by, until=span, now=now,
                                       **daily)
        else:
            saved = book.mark_known(camera, who, by=by, until=span, now=now, **daily)
    except ValueError as exc:
        log.warning("mark_known refused: %s", exc)
        return _result(_issue(ctx, "mark_known", FAILED, camera or "house", detail, "error"))
    detail.update(known_id=str(saved.get("id") or ""), people=int(saved.get("people") or 0))
    if old:
        detail["replaced"] = [{"id": str(k.get("id") or ""), "camera": str(k.get("camera") or ""),
                               "until": _finite(k.get("until")), "text": str(k.get("text") or ""),
                               "daily_from": str(k.get("daily_from") or ""), "daily_to": str(k.get("daily_to") or "")}
                              for k in old]
    entry = ctx.state.resolve(str(ctx.alert_handle or "")) if ctx.alert_handle else None
    if isinstance(entry, dict) and entry.get("kind") == "event" and not old:
        # Said in reply to an alert: that clip's TAG too (normal: the owner's people), its own 🏷️ line and store.
        try:
            file_tag(ctx, entry, "normal", str(args.get("owner_words") or ""), now)
            detail["verdict_filed"] = str(ctx.alert_handle)
        except Exception as exc:  # noqa: BLE001 - the memory is saved; the tag is extra
            log.warning("Tag not saved with the memory: %s", exc)
    receipt = _issue(ctx, "mark_known", DONE, camera or "house", detail)
    return _result(receipt, until=dt.datetime.fromtimestamp(span).isoformat(timespec="minutes"),
                   note="The box writes the confirmation; reply with an empty answer.")


# -- the clip's tag, changed from the chat (2026-10-09: "שמור מידע ושנה תיוג / התיוג זה אדם עם חולצה לבנה..." saved
# the fact and dropped the tag change) ---------------------------------------------------------------------------------
_RETAG_ASK = re.compile(r"(?<![א-ת])[ולש]?(?:ה?תיוג|תייג|תתייג|לתייג|תג)(?![א-ת])|\b(?:re-?tag|the tag|tag it)\b",
                        re.IGNORECASE)
_RETAG_TEXT = re.compile(r"(?<![א-ת])(?:התיוג|התג|תיוג)\s*(?:הנכון\s*)?(?:זה|הוא|יהיה|צריך להיות|:|-)\s*:?\s*"
                         r"([^\n]{4,300})|\bthe (?:new )?tag (?:is|should be)\s*:?\s*([^\n]{4,300})", re.IGNORECASE)


def asks_retag(text: str) -> bool:
    """The message asks to change a clip's tag ("שנה תיוג", "התיוג זה ...", "the tag is ...")."""
    return bool(_RETAG_ASK.search(str(text or "")))


def retag_words(text: str) -> str:
    """The new tag the message gives ("התיוג זה אדם עם חולצה לבנה..." -> "אדם עם חולצה לבנה..."), or ""."""
    m = _RETAG_TEXT.search(str(text or ""))
    return " ".join((m.group(1) or m.group(2) or "").split()).strip(" .") if m else ""


def file_tag(ctx: ToolContext, entry: Dict[str, Any], label: str, words: str, now: float) -> Receipt:
    """A TAG of one clip (training data for the vision model, feedback/): the owner's label and words. It never
    changes what the box does - that is a memory (mark_known, camera_fact). Its receipt is a 🏷️ line."""
    label = label if label in ("normal", "suspicious", "escalation", "empty", "rule_mismatch") else "other"
    feedback = Feedback(verdict=verdict_for(label, str(entry.get("label") or "")), owner_label=label,
                        owner_text=words, tagged_by=str(ctx.speaker.get("name") or ""), source="text")
    save_feedback(ctx.services.feedback_dir, _alert_of(entry), feedback, ctx.text, ctx.speaker, ctx.chat_id, now)
    ctx.saved += 1
    try:
        when = hhmm(_finite(entry.get("ts")))
    except (TypeError, ValueError):
        when = "?"
    handle = next((h for h, e in ctx.state.handles.items() if e is entry), str(ctx.alert_handle or ""))
    return _issue(ctx, "retag_clip", DONE, handle, {"handle": handle, "camera": str(entry.get("camera") or ""),
                                                    "alert_id": str(entry.get("ref") or ""), "time": when,
                                                    "label": label, "tag": words})


@_safe_tool
def retag_clip(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    now = _finite(ctx.services.now())
    handle = str(args.get("handle") or ctx.alert_handle or ctx.state.topic_event(now) or "").strip().upper()
    if not handle:
        return _err("no clip is being discussed; ask the owner which clip (time and camera)")
    entry = ctx.state.resolve(handle)
    if not entry or entry.get("kind") != "event":
        return _err(f"unknown handle {handle!r}")
    tag = " ".join(str(args.get("tag") or "").split())[:300]
    if not quoted_from(tag, ctx.text):
        return _err("Not saved: tag must be the owner's new description of the clip, copied exactly from this "
                    "message (two words or more).")
    return _result(file_tag(ctx, entry, km.tag_label(tag), tag, now),
                   note="The box writes the confirmation of the new tag.")


def session_text(session: Dict[str, Any], snapshot: Any, lang: str, now: float) -> Dict[str, Any]:
    """One event (a camera's session) as the model reads it: names and HH:MM, never ids or epoch seconds."""
    observations = [o for o in session.get("observations") or [] if isinstance(o, dict)]
    closed = float(session.get("closed") or 0.0)
    end = closed or float(session.get("last_active") or session.get("opened") or now)
    return {
        "camera": display(snapshot, str(session.get("camera") or ""), lang),
        "from": hhmm(float(session.get("opened") or now)),
        "to": hhmm(end) if closed else f"{hhmm(end)} (still going on)",
        "most_people": int(session.get("people_max") or 0),
        "told_the_owner": str(session.get("reported_level") or "none") != "none",
        "seen": [{"at": hhmm(float(o.get("ts") or 0)), "label": str(o.get("label") or ""),
                  "people": int(o.get("people") or 0), "what": str(o.get("summary") or "")[:300]}
                 for o in observations[-5:]],
        "owner_said": [{"who": str(k.get("text") or ""), "until": hhmm(float(k.get("until") or 0))}
                       for k in session.get("known") or [] if isinstance(k, dict)],
    }


@_safe_tool
def recent_activity(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    book = ctx.services.events
    if book is None:
        return _err("the box's event history is not available; use find_events")
    now = _finite(ctx.services.now())
    camera = ""
    if str(args.get("camera") or "").strip():
        camera, bad = _one_camera(ctx, args["camera"])
        if bad:
            return bad
    start = str(args.get("time_from") or "").strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", start):
        since = house.next_at(now, start.zfill(5)) - 86400.0       # the last time the clock read that
    else:
        minutes = args.get("minutes")
        minutes = 60.0 if minutes in (None, "") else min(1440.0, max(1.0, _finite(minutes)))
        since = now - minutes * 60.0
    rows = book.recent(since, camera)
    known = [{"camera": display(ctx.snapshot, str(k["camera"]), ctx.lang) if k.get("camera") else "all cameras",
              "who": str(k.get("text") or ""), "until": hhmm(float(k.get("until") or now))}
             for k in book.list_known(now) if not camera or not k.get("camera") or k.get("camera") == camera]
    return {"ok": True, "since": hhmm(since), "events": [session_text(r, ctx.snapshot, ctx.lang, now)
                                                        for r in rows[-10:]],
            "owner_said": known,
            "note": "Events are per camera: who was there, from when to when, what the vision model saw. "
                    "For what is happening at this moment, also look live with check_camera."}


# -- the event memory (event_memory.py): "what happened at the pergola at noon?", "were the workers here yesterday?" --
def _memory(ctx: ToolContext) -> Any:
    book = ctx.services.events
    if book is None or not getattr(book, "directory", ""):
        return None
    from ..event_memory import memory_for  # noqa: PLC0415

    return memory_for(book.directory)


def _day_text(ts: float, now: float, lang: str) -> str:
    """"today", "yesterday" (היום / אתמול), else "Tue 06.10"."""
    day = dt.datetime.fromtimestamp(ts).date()
    today = dt.datetime.fromtimestamp(now).date()
    if day == today:
        return "היום" if lang == "he" else "today"
    if day == today - dt.timedelta(days=1):
        return "אתמול" if lang == "he" else "yesterday"
    return dt.datetime.fromtimestamp(ts).strftime("%a %d.%m")


def _memory_range(args: Dict[str, Any], now: float) -> Tuple[Optional[float], Optional[float]]:
    """``(since, until)`` from day / time_from / time_to / last_hours; (None, None) when none is given (the query's
    own day words, "אתמול", then decide)."""
    hours = args.get("last_hours")
    if hours not in (None, ""):
        return now - min(24.0 * 30, max(0.1, _finite(hours))) * 3600.0, now
    day_arg = str(args.get("day") or "").strip().lower()
    t_from, t_to = str(args.get("time_from") or "").strip(), str(args.get("time_to") or "").strip()
    if not (day_arg or t_from or t_to):
        return None, None
    base = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
    if day_arg in ("yesterday", "אתמול"):
        base -= dt.timedelta(days=1)
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_arg):
        base = dt.datetime.strptime(day_arg, "%Y-%m-%d")
    since, until = base.timestamp(), (base + dt.timedelta(days=1)).timestamp()
    for value, which in ((t_from, "from"), (t_to, "to")):
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
            ts = (base + dt.timedelta(hours=int(m.group(1)), minutes=int(m.group(2)))).timestamp()
            if which == "from":
                since = ts
            else:
                until = ts
    return since, until


def memory_event_text(record: Dict[str, Any], snapshot: Any, lang: str, now: float, full: bool = False) -> Dict[str, Any]:
    """One archived event as the model reads it: the camera's name, the day and HH:MM, never ids or epoch seconds."""
    camera = str(record.get("camera") or "")
    name = display(snapshot, camera, lang)
    caption = str(record.get("caption") or "")
    stored = str(record.get("camera_name") or "")
    if stored and stored != name:
        caption = caption.replace(f" at {stored}.", f" at {name}.", 1)     # renamed since: today's name
    start = float(record.get("start") or now)
    end = float(record.get("end") or start)
    out: Dict[str, Any] = {
        "camera": name,
        "day": _day_text(start, now, lang),
        "from": hhmm(start),
        "to": hhmm(end) if record.get("outcome") != "open" else f"{hhmm(end)} (still going on)",
        "what": caption,
        "most_people": int(record.get("people_max") or 0),
        "told_the_owner": bool(record.get("reported")),
        "owner_said": list(record.get("owner_known") or []),
        "change_from_previous": str(record.get("change_from_previous") or ""),
    }
    if full:
        out["change_to_next"] = str(record.get("change_to_next") or "")
        out["seen"] = [{"at": hhmm(float(o.get("ts") or start)), "label": str(o.get("label") or ""),
                        "people": int(o.get("people") or 0), "what": str(o.get("summary") or "")[:300]}
                       for o in record.get("observations") or [] if isinstance(o, dict)]
        out["continues_an_earlier_event"] = bool(record.get("parent"))
    return out


@_safe_tool
def search_events(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """The box's memory of past events (EventMemAgent-style): the best 3 for the owner's words, a time, a camera."""
    memory = _memory(ctx)
    if memory is None:
        return _err("the box's event memory is not available; use find_events")
    query = " ".join(str(args.get("query") or "").split())[:300] or ctx.text
    now = _finite(ctx.services.now())
    camera: Optional[str] = None
    if str(args.get("camera") or "").strip():
        camera, bad = _one_camera(ctx, args["camera"])
        if bad:
            return bad
    from .registry import mentioned_cameras  # noqa: PLC0415

    named = [cam for _, cam in mentioned_cameras(ctx.snapshot, query)]
    names = {c.name: [c.name, *c.aliases, display(ctx.snapshot, c.name, ctx.lang)] for c in ctx.snapshot.cameras}
    since, until = _memory_range(args, now)
    k = int(min(5, max(1, _finite(args.get("k") if args.get("k") not in (None, "") else 3))))
    live: List[Dict[str, Any]] = []
    try:
        live = [memory.live_record(s) for s in ctx.services.events.recent(now - 86400.0, camera or "")
                if not s.get("closed")]
    except Exception as exc:  # noqa: BLE001 - the archive alone still answers
        log.debug("open events not searched: %s", exc)
    hits = memory.search(query, since=since, until=until, camera=camera or None, k=k,
                         embedder=ctx.services.embedder, names=names, boost=named, extra=live, now=now)
    rows = []
    for hit in hits:
        handle = ctx.state.add_handle("memory", str(hit["event_id"]), str(hit.get("camera") or ""),
                                      float(hit.get("start") or 0.0), str(hit.get("caption") or "")[:200])
        rows.append(dict(memory_event_text(hit, ctx.snapshot, ctx.lang, now), handle=handle))
    if not rows:
        return {"ok": True, "found": 0, "events": [],
                "note": "Nothing in the box's memory matches. Say that nothing matching was recorded (not that "
                        "nothing happened). The memory keeps 30 days of events."}
    return {"ok": True, "found": len(rows), "events": rows,
            "note": "Past events from the box's memory, best match first. Answer from 'what'; get_event gives the "
                    "details, its picture and its video."}


@_safe_tool
def get_event(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """One event of the memory in full, with its picture (a photo handle) and its first alert's video (an event
    handle) when they are still on the box."""
    memory = _memory(ctx)
    if memory is None:
        return _err("the box's event memory is not available")
    handle = str(args.get("handle") or "").strip().upper()
    entry = ctx.state.resolve(handle)
    if not entry or entry.get("kind") != "memory":
        return _err(f"unknown handle {handle!r}; use a handle from search_events")
    now = _finite(ctx.services.now())
    record = memory.get(str(entry["ref"]))
    if record is None and ctx.services.events is not None:
        record = next((memory.live_record(s) for s in ctx.services.events.recent(now - 86400.0)
                       if s.get("id") == entry["ref"]), None)
    if record is None:
        return _err("that event is no longer in the box's memory")
    out: Dict[str, Any] = {"ok": True, **memory_event_text(record, ctx.snapshot, ctx.lang, now, full=True)}
    camera = str(record.get("camera") or "")
    from ..event_memory import keyframe_path  # noqa: PLC0415

    picture = str(record.get("keyframe") or "") or keyframe_path(memory.directory, str(record.get("event_id") or ""))
    if os.path.isfile(picture):
        photo = ctx.state.add_handle("photo", picture, camera, float(record.get("start") or now))
        ctx.state.note_observation(photo, str(record.get("caption") or "")[:300])
        out["picture"] = photo
    alert_ids = [a for a in record.get("alert_ids") or [] if a]
    if alert_ids:
        saved = {r.alert_id: r for r in load_events(ctx.services.roots(), ctx.services.desc_dir)
                 if r.alert_id in set(alert_ids)}
        first = next((saved[a] for a in alert_ids if a in saved), None)
        if first is not None:
            out["video"] = _show(ctx, first)
    out["note"] = ("To show it: send_media with 'picture' (its first picture) or 'video' (its first alert's video). "
                   "Without them the picture or video is no longer on the box.")
    return out


# -- what is usual at each camera (baseline.py, camera_profiles.py; task 2.9) --------------------------------------
_PHASE_WORDS = {"late_night": ("late night", "לפנות בוקר", "באמצע הלילה", "אחרי חצות"),
                "night": ("night", "tonight", "לילה", "בלילה", "בלילות"),
                "evening": ("evening", "ערב", "בערב", "בערבים"),
                "day": ("day", "daytime", "morning", "noon", "afternoon", "יום", "ביום", "בוקר", "בבוקר", "צהריים",
                        "בצהריים", "אחה\"צ")}
_PHASE_AT = {"late_night": "03:00", "night": "22:30", "evening": "19:00", "day": "12:00"}


def profiles_for(services: Services) -> Any:
    """The camera personalities beside the event book (``<events>/camera_profiles.json``), or None."""
    book = services.events
    if book is None or not getattr(book, "directory", ""):
        return None
    from ..camera_profiles import PROFILES_NAME, CameraProfiles  # noqa: PLC0415

    return CameraProfiles(os.path.join(book.directory, PROFILES_NAME))


def _historian(ctx: ToolContext) -> Any:
    book = ctx.services.events
    if book is None or not getattr(book, "directory", ""):
        return None
    from types import SimpleNamespace  # noqa: PLC0415

    from ..baseline import BASELINE_NAME, Baseline, Historian  # noqa: PLC0415
    from ..camera_profiles import house_profile  # noqa: PLC0415

    store = profiles_for(ctx.services)
    settings: Dict[str, Any] = {}
    try:
        settings = dict(ctx.services.read_settings() or {}) if ctx.services.read_settings else {}
    except Exception:  # noqa: BLE001
        settings = {}

    def house_now(ts: float) -> Dict[str, Any]:
        state = ctx.services.house.current(ts) if ctx.services.house is not None else \
            SimpleNamespace(state="", expecting=[])
        return house_profile(now=ts, events_dir=book.directory, profiles=store, house_now=state)

    return Historian(Baseline(os.path.join(book.directory, BASELINE_NAME), clock=ctx.services.now), store,
                     names=lambda cam, lang: display(ctx.snapshot, cam, lang), house=house_now, settings=settings)


def _usual_time(value: Any, now: float, default: float) -> Tuple[float, str]:
    """``(ts, part of day)`` for how_usual's time: "HH:MM" (the last time the clock read it), a part of the day
    ("בלילה", "evening"), "now", or *default*."""
    text = str(value or "").strip().lower()
    if not text:
        return default, ""
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
        return house.next_at(now, f"{int(m.group(1)):02d}:{m.group(2)}") - 86400.0, ""
    for phase in ("late_night", "night", "evening", "day"):
        if any(w in text for w in _PHASE_WORDS[phase]):
            return house.next_at(now, _PHASE_AT[phase]) - 86400.0, phase
    return now, ""


def _at_place(name: str, lang: str) -> str:
    """"בכניסה הראשית" / "at the main entrance"."""
    if lang != "he":
        return f"at {name}"
    return f"ב{name[1:]}" if name.startswith("ה") and len(name) > 2 else f"ב{name}"


def usual_answer(d: Dict[str, Any], lang: str, phase: str = "", tag: str = "") -> str:
    """The code-written answer of how_usual, in the owner's language. Names only, never ids."""
    he = lang == "he"
    name = d["camera"]
    parts: List[str] = []
    asked = d.get("asked_activity")
    if tag and asked:
        from ..activities import name_at  # noqa: PLC0415

        parts.append(f"{name_at(tag, name, lang)} {_at_place(name, lang)}: {asked.get('often')}")
        for other, often in list((asked.get("elsewhere") or {}).items())[:5]:
            parts.append(f"{_at_place(other, lang)}: {often}")
    elif phase and d.get("asked_part_of_day"):
        p = d["asked_part_of_day"]
        parts.append(f"{name} {p['name']}: " + (f"אנשים {p['often']}" if he else f"people {p['often']}"))
    else:
        busiest = ", ".join(d.get("busiest_hours") or []) or ("אין" if he else "none")
        parts.append((f"{name}: הכי עמוס {busiest}" if he else f"{name}: busiest {busiest}"))
        top = sorted((v for v in (d.get("activities") or {}).values() if v.get("events")),
                     key=lambda v: -v["events"])[:3]
        if top:
            parts.append("; ".join(f"{v['name']} - {v['often']}" for v in top))
    if d.get("at_that_time", {}).get("says"):
        parts.append(d["at_that_time"]["says"])
    if d.get("owner_facts"):
        parts.append(("מה שלימדת: " if he else "What you taught me: ") + "; ".join(d["owner_facts"][:3]))
    if not d.get("enough_data"):
        parts.append(f"(יש לי רק {d['days_of_data']} ימים של היסטוריה, אז זו הערכה זהירה)" if he
                     else f"(only {d['days_of_data']} days of history so far, so this is a careful guess)")
    return "; ".join(p for p in parts if p).replace(";;", ";")


@_safe_tool
def how_usual(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """What is usual at a camera ("זה רגיל?", "כמה פעמים זה קורה?", "מה רגיל בחצר האחורית בלילה?"): its history
    by hour and by activity, its role and what the owner taught about it, from the box's own counts."""
    historian = _historian(ctx)
    if historian is None:
        return _err("the box's history is not available")
    camera, bad = _known_camera(ctx, args.get("camera"))
    if bad:
        return bad
    now = _finite(ctx.services.now())
    entry = ctx.state.resolve(str(ctx.alert_handle or "")) if ctx.alert_handle else None
    alert_ts = float(entry["ts"]) if isinstance(entry, dict) and entry.get("ts") else None
    ts, phase = _usual_time(args.get("time"), now, alert_ts if alert_ts is not None else now)
    from ..activities import TAGS, tags_of  # noqa: PLC0415

    tag = str(args.get("activity") or "").strip().lower()
    if tag not in TAGS:
        found = tags_of(tag) if tag else []
        if not found and isinstance(entry, dict) and not args.get("activity"):
            found = tags_of(str(entry.get("summary") or ""))
        tag = found[0] if found else ""
    asked_time = bool(str(args.get("time") or "").strip()) or alert_ts is not None
    d = historian.describe(camera, ts if asked_time and not phase else None, phase=phase, tag=tag, lang=ctx.lang)
    if asked_time and phase:
        d["at_that_time"] = {"says": ""}
    out = {"ok": True, **d, "answer": usual_answer(d, ctx.lang, phase, tag),
           "note": "Answer from 'answer' (written by the box from its own counts); say how many days of history "
                   "it has when it is short. Never invent a number. Only the owner's own words go into "
                   "camera_fact."}
    return out


@_safe_tool
def camera_fact(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """The owner teaches (or removes) a fact about one camera or the whole house ("בדלת האחורית משתמשים רק אנחנו",
    "שליחים מגיעים רק לכניסה הראשית", "יש לנו כלב"), or confirms the camera's role. Kept until removed."""
    store = profiles_for(ctx.services)
    if store is None:
        return _err("the box's camera memory is not available; tell the owner it was not saved")
    words = " ".join(str(args.get("owner_words") or "").split())[:200]
    if not _said(words, ctx.text):
        return _err("Not saved: owner_words must be copied exactly from this message.")
    whole_house = bool(args.get("whole_house"))
    camera = ""
    if not whole_house:
        camera, bad = _known_camera(ctx, args.get("camera"))
        if bad:
            return bad
    by = str(ctx.speaker.get("name") or "owner")
    now = _finite(ctx.services.now())
    role = str(args.get("role") or "").strip().lower()
    if role:
        from ..camera_profiles import ROLES  # noqa: PLC0415

        if whole_house or role not in ROLES:
            return _err(f"role must be one of {', '.join(ROLES)}, for one camera")
        old = store.set_role(camera, role, by=by, now=now)
        return _result(_issue(ctx, "camera_fact", DONE, camera,
                              {"camera": camera, "role": role, "old_role": old, "already": old == role}),
                       note="The box writes the confirmation; reply with an empty answer.")
    if args.get("remove"):
        wanted = " ".join(str(args.get("fact") or words).split()).casefold()
        rows = store.house_facts() if whole_house else store.facts(camera)
        hits = [f for f in rows if f.get("id") == args.get("fact") or wanted in str(f.get("text") or "").casefold()
                or str(f.get("text") or "").casefold() in wanted]
        if len(hits) != 1:
            return _err("which fact? the saved ones are: " + " | ".join(str(f.get("text")) for f in rows)
                        if rows else "nothing is saved there", saved=[str(f.get("text")) for f in rows])
        removed = store.remove_fact(str(hits[0]["id"]))
        if removed is None:
            return _err("that fact is no longer saved")
        key, fact = removed
        detail = {"camera": camera, "fact": str(fact.get("text") or ""), "fact_id": str(fact.get("id")),
                  "removed": True, "whole_house": whole_house, "key": key,
                  "restore": {k: fact[k] for k in ("id", "text", "by", "at") if k in fact}}
        return _result(_issue(ctx, "camera_fact", DONE, camera, detail),
                       note="The box writes the confirmation; reply with an empty answer.")
    try:
        fact = store.add_fact(camera, words, by=by, now=now)
    except ValueError as exc:
        return _result(_issue(ctx, "camera_fact", FAILED, camera, {"camera": camera, "fact": words}, str(exc)))
    detail = {"camera": camera, "fact": fact["text"], "fact_id": fact["id"], "removed": False,
              "whole_house": whole_house, "rule": (fact.get("rule") or {}).get("kind", ""),
              "already": bool(fact.get("already"))}
    return _result(_issue(ctx, "camera_fact", DONE, camera, detail),
                   note="The box writes the confirmation; reply with an empty answer.")


TOOLS: Dict[str, Callable[[ToolContext, Dict[str, Any]], Dict[str, Any]]] = {
    "find_events": find_events,
    "summarize_period": summarize_period,
    "describe_event": describe_event,
    "assess_event": assess_event,
    "ask_vision": ask_vision,
    "ask_clarification": ask_clarification,
    "check_camera": check_camera,
    "look_around": look_around,
    "record_clip": record_clip,
    "send_media": send_media,
    "pause_alerts": pause_alerts,
    "resume_alerts": resume_alerts,
    "record_verdict": record_verdict,
    "set_camera_active": set_camera_active,
    "set_alias": set_alias,
    "change_setting": change_setting,
    "get_alert_settings": get_alert_settings,
    "set_alert_types": set_alert_types,
    "set_sensitivity": set_sensitivity,
    "house_state": house_state,
    "house_expect": house_expect,
    "house_cancel": house_cancel,
    "house_status": house_status,
    "mark_known": mark_known,
    "retag_clip": retag_clip,
    "recent_activity": recent_activity,
    "search_events": search_events,
    "get_event": get_event,
    "how_usual": how_usual,
    "camera_fact": camera_fact,
}
