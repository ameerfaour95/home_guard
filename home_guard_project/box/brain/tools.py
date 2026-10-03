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
from ..feedback import MAX_MUTE_HOURS, VERDICTS, Feedback, MuteState, feedback_from_fields, save_feedback
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
from .i18n import LANGUAGE_NAMES
from .memory import ChatState
from .media import bounds_text
from .mode import GUARD, hhmm
from .receipts import DONE, FAILED, REQUESTED, Receipt, ReceiptBook
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
    set_option: Optional[Callable[[str, str], Any]] = None
    read_settings: Optional[Callable[[], Dict[str, Any]]] = None


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


def _one_camera(ctx: ToolContext, words: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    res = resolve_camera(ctx.snapshot, str(words or ""))
    if res.camera is None:
        return None, _camera_error(ctx, words, res)
    return res.camera, None


@_safe_tool
def check_camera(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    camera, bad = _one_camera(ctx, args.get("camera"))
    if bad:
        return bad
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
    receipt = _issue(ctx, "check_camera", DONE if sent.get("ok") else FAILED, camera,
                     {"camera": camera, "message_id": sent.get("message_id")}, "" if sent.get("ok") else "telegram")
    handle = ctx.state.add_handle("photo", shot["image"], camera, ctx.services.now())
    ctx.shown.append(handle)
    out = _result(receipt, camera=camera, handle=handle)
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
        if ctx.mode == GUARD:
            out.update(label=look.get("label", ""), why=look.get("why", ""))
    else:
        out["description_error"] = "the picture was taken but could not be described" + (
            " (the model declined)" if isinstance(look, dict) and look.get("refused") else "")
    return out


@_safe_tool
def record_clip(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    camera, bad = _one_camera(ctx, args.get("camera"))
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
    sent = _service_result(ctx.services.deliver.video(ctx.chat_id, rec["path"], caption=f"{camera} · {bounds}"))
    receipt = _issue(ctx, "record_clip", DONE if sent.get("ok") else FAILED, camera,
                     {"camera": camera, "seconds": seconds, "bounds": bounds, "message_id": sent.get("message_id")},
                     "" if sent.get("ok") else "telegram")
    handle = ctx.state.add_handle("clip", rec["path"], camera, rec["start"])
    ctx.shown.append(handle)
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
    sent = _service_result(ctx.services.deliver.video(ctx.chat_id, path, caption=f"{camera} · {bounds}".strip(" ·")))
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
        detail = {"camera": camera or "", "until": hhmm(fb.mute_until), "before": before}
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
    try:
        ctx.services.add_alias(camera, alias, ctx.snapshot.names)
    except (ValueError, TypeError) as exc:
        return _result(_issue(ctx, "set_alias", FAILED, camera, {"camera": camera, "alias": alias}, str(exc)))
    return _result(_issue(ctx, "set_alias", DONE, camera, {"camera": camera, "alias": alias}))

# -- settings --------------------------------------------------------------------
SETTING_NAMES = ("alert_hours", "cooldown_minutes", "sensitivity", "language")
SENSITIVITY_LEVELS = {"low": 0.6, "medium": 0.4, "high": 0.25}   # detector confidence: lower = more sensitive
LANGUAGE_WORDS = {"en": "en", "english": "en", "אנגלית": "en", "he": "he", "hebrew": "he", "עברית": "he"}


def settings_view(settings: Dict[str, Any]) -> Dict[str, str]:
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
    }


def settings_line(settings: Dict[str, Any]) -> str:
    v = settings_view(settings)
    return (f"SETTINGS: alert hours {v['alert_hours']} · time between alerts per camera {v['cooldown_minutes']} · "
            f"detector sensitivity {v['sensitivity']} · box language {v['language']} (alerts and announcements)")


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
    if start % 24 == end % 24:
        return None                         # "6-6" would silently mean all day; the owner must say so
    return (start % 24, end % 24)            # 24 means midnight


@_safe_tool
def change_setting(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if not quoted_from(str(args.get("owner_words") or ""), ctx.text):
        return _err("Not changed: change a setting only when this message asks for it; owner_words must be "
                    "copied from it (two words or more).")
    name = str(args.get("setting") or "")
    if name not in SETTING_NAMES:
        return _err("setting must be one of alert_hours, cooldown_minutes, sensitivity, language")
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
    try:
        if name == "alert_hours":
            hours = _hours(value)
            if hours is None:
                return _err('alert_hours must look like "22-06" (whole hours) or "all day"')
            raw = seen["first"].get("alert_start_hour", 0)
            old_start = int(_finite(raw))
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
                                  {"setting": name, "old": before, "new": after}))
        elif name == "cooldown_minutes":
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError("cooldown_minutes must be a number")
            seconds = int(round(_finite(value) * 60))
            if not 10 <= seconds <= 86400:
                return _err("cooldown_minutes must be between 0.2 and 1440")
            ctx.services.set_option("alert_cooldown_sec", str(seconds))
        elif name == "sensitivity":
            text = str(value).strip().lower()
            conf = SENSITIVITY_LEVELS.get(text)
            ctx.services.set_option("inference_conf", str(conf if conf is not None else _finite(text)))
        else:
            code = LANGUAGE_WORDS.get(str(value).strip().lower())
            if code is None:
                return _err('language must be "en" (English) or "he" (Hebrew)')
            ctx.services.set_option("owner_language", code)
        after = read_view()[name]
    except Exception as exc:  # noqa: BLE001 - BoxConfigError / ValueError: the owner's value was refused
        log.warning("Setting change failed: %s", exc)
        return _result(_issue(ctx, "change_setting", FAILED, name, {"setting": name}, str(exc)))
    return _result(_issue(ctx, "change_setting", DONE, name, {"setting": name, "old": before, "new": after}))

TOOLS: Dict[str, Callable[[ToolContext, Dict[str, Any]], Dict[str, Any]]] = {
    "find_events": find_events,
    "summarize_period": summarize_period,
    "describe_event": describe_event,
    "assess_event": assess_event,
    "ask_clarification": ask_clarification,
    "check_camera": check_camera,
    "record_clip": record_clip,
    "send_media": send_media,
    "pause_alerts": pause_alerts,
    "resume_alerts": resume_alerts,
    "record_verdict": record_verdict,
    "set_camera_active": set_camera_active,
    "set_alias": set_alias,
    "change_setting": change_setting,
}
