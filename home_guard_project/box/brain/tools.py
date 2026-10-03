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

import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..archive import AlertRecord
from ..feedback import MAX_MUTE_HOURS, feedback_from_fields
from . import media
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
from .memory import ChatState
from .mode import GUARD
from .receipts import Receipt, ReceiptBook
from .registry import HouseSnapshot, Resolution, resolve_camera
from .vision import QUALITIES, VISION_VERSION

log = logging.getLogger("box.brain.tools")

MAX_FOUND = 8
SUMMARY_MAX_EVENTS = 60
MAX_DESCRIBE_PER_TURN = 5
MAX_MEDIA_PER_TURN = 3


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
    set_camera: Optional[Callable[[str, bool], Dict[str, Any]]] = None
    add_alias: Optional[Callable[[str, str, Sequence[str]], List[str]]] = None
    request_restart: Optional[Callable[[], None]] = None
    embedder: Any = None
    now: Callable[[], float] = time.time
    retention_days: float = 14.0
    max_mute_hours: float = MAX_MUTE_HOURS


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
    clarification: Optional[Dict[str, Any]] = None
    after_reply: List[Callable[[], None]] = field(default_factory=list)
    call_key: str = ""
    described: int = 0
    saved: int = 0
    done_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # idempotency within one turn
    camera_states: Dict[str, bool] = field(default_factory=dict)          # cameras changed earlier this turn


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


@_safe_tool
def describe_event(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    handle = str(args.get("handle") or "").strip().upper()
    record, bad = _record_for(ctx, handle)
    if bad:
        return bad
    out = _look_clip(ctx, record, guard=False, question=str(args.get("question") or "")[:300])
    if not out.get("ok"):
        return out
    return {"ok": True, "handle": handle, "camera": record.camera, "time": local(record.ts),
            "description": out["text"], "quality": out["quality"], "people": out["people"]}


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


TOOLS: Dict[str, Callable[[ToolContext, Dict[str, Any]], Dict[str, Any]]] = {
    "find_events": find_events,
    "summarize_period": summarize_period,
    "describe_event": describe_event,
    "assess_event": assess_event,
    "ask_clarification": ask_clarification,
}
