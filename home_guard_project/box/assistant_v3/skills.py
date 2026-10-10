"""The agentic part: the grounded READ tools the writer may call before it answers, and the loop with budgets.

Six tools, all read-only or reversible media sends (report §8.5 class R / W1), each a thin adapter over the v2
implementation in ``brain/tools.py`` (so receipts, the photo captions, the grounded live look and the media limits
are the same code the box runs today):

- ``look_now(camera)``: a fresh picture + grounded description of one camera, or every camera ("house");
- ``look_again(handle, question)``: the vision model looks again at a saved alert clip OR a live photo already
  sent, for one question; the answer is kept on the handle (never re-analysed on the next message);
- ``recent_activity(camera, minutes)``: the box's sessions (who was where, from when to when);
- ``search_history(query)``: 30 days of events by meaning;
- ``send_clip(handle)`` / ``record_clip(camera)``: the video of an alert, or 10 live seconds.

Budgets are code: a chat turn may make 3 tool calls in 8 s, an investigation 6 in 20 s; on the limit the writer
answers with what it has.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..brain.receipts import DONE
from ..brain.tools import TOOLS, ToolContext

log = logging.getLogger("box.assistant_v3.skills")

CHAT_BUDGET = (3, 8.0)
INVESTIGATE_BUDGET = (6, 20.0)


def _fn(name: str, description: str, props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


SCHEMAS = [
    _fn("look_now", "Take a fresh picture now and describe it. camera: a camera name, or 'house' for every camera. "
        "The owner gets the photo when people are in it.", {"camera": {"type": "string"}}, ["camera"]),
    _fn("look_again", "Look again at a saved alert clip or a live photo already sent (handle E#) for one question.",
        {"handle": {"type": "string"}, "question": {"type": "string"}}, ["handle", "question"]),
    _fn("recent_activity", "The box's events (who was seen where, from when to when) in the last minutes, at one "
        "camera or all.", {"camera": {"type": "string"}, "minutes": {"type": "number"}}, []),
    _fn("search_history", "Search 30 days of events by meaning (e.g. 'a man with a ladder at night').",
        {"query": {"type": "string"}}, ["query"]),
    _fn("send_clip", "Send the owner the video of a saved alert (handle E#).", {"handle": {"type": "string"}},
        ["handle"]),
    _fn("record_clip", "Record and send 10 live seconds from one camera.", {"camera": {"type": "string"}},
        ["camera"]),
]


@contextmanager
def internal(ctx: ToolContext) -> Iterator[None]:
    """A media tool run by the code's own decision: v2's quote checks read the owner's words of v2's own path,
    which the understanding step already replaced."""
    text, ctx.text = ctx.text, ""
    try:
        yield
    finally:
        ctx.text = text


class Toolbox:
    def __init__(self, ctx: ToolContext, called: List[str], resolve_camera: Callable[[str], str]) -> None:
        self.ctx, self.called, self.resolve_camera = ctx, called, resolve_camera
        self.evidence: List[str] = []

    def run(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            out = getattr(self, "_" + name)(args) if name in {s["function"]["name"] for s in SCHEMAS} else \
                {"ok": False, "error": f"no tool {name}"}
        except Exception as exc:  # noqa: BLE001 - a tool bug never ends the turn
            log.warning("v3 tool %s failed: %s", name, exc)
            out = {"ok": False, "error": "that could not be done"}
        return out

    # -- the tools -----------------------------------------------------------------------------------------------
    def _look_now(self, args: Dict[str, Any]) -> Dict[str, Any]:
        cam = str(args.get("camera") or "").strip()
        camera = self.resolve_camera(cam) if cam and cam.casefold() not in ("house", "all", "כל הבית", "הכל") else ""
        if camera:
            self.called.append("check_camera")
            with internal(self.ctx):
                out = TOOLS["check_camera"](self.ctx, {"camera": camera})
            self.evidence.append(_look_line(self.ctx, out))
            return _brief(out, ("camera", "description", "people", "detector", "whose_ground", "note", "error",
                                "description_error", "handle", "status", "reason"))
        self.called.append("look_around")
        with internal(self.ctx):
            out = TOOLS["look_around"](self.ctx, {})
        for row in out.get("cameras") or []:
            self.evidence.append(_row_line(row))
        if not out.get("ok"):
            self.evidence.append(str(out.get("error") or "no picture"))
        return _brief(out, ("cameras", "error", "note"))

    def _look_again(self, args: Dict[str, Any]) -> Dict[str, Any]:
        handle = str(args.get("handle") or "").strip().upper()
        question = str(args.get("question") or self.ctx.text or "").strip()[:300]
        entry = self.ctx.state.resolve(handle) or {}
        self.called.append("ask_vision")
        if entry.get("kind") == "photo":
            return self._look_again_photo(handle, entry, question)
        out = TOOLS["ask_vision"](self.ctx, {"event_id": handle, "question": question})
        if out.get("ok"):
            self.evidence.append(f"{handle} נבדק שוב (\"{question[:80]}\"): {out.get('answer')}")
        return _brief(out, ("handle", "answer", "frame", "time", "error", "cached"))

    def _look_again_photo(self, handle: str, entry: Dict[str, Any], question: str) -> Dict[str, Any]:
        """look_again on a live photo already sent (v2's ask_vision reads alert clips only)."""
        vision = self.ctx.services.vision
        path = str(entry.get("ref") or "")
        if vision is None or not os.path.isfile(path):
            return {"ok": False, "error": "that photo is no longer on the box"}
        with open(path, "rb") as f:
            data = f.read()
        out = vision.ask(str(entry.get("camera") or ""), [data], question, "Hebrew")
        if not isinstance(out, dict) or not out.get("ok"):
            return {"ok": False, "error": "the photo could not be checked"}
        self.ctx.state.add_answer(handle, question, str(out.get("answer") or ""))
        self.evidence.append(f"{handle} נבדק שוב (\"{question[:80]}\"): {out.get('answer')}")
        return {"ok": True, "handle": handle, "answer": out.get("answer")}

    def _recent_activity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.called.append("recent_activity")
        payload: Dict[str, Any] = {"minutes": args.get("minutes") or 240}
        if str(args.get("camera") or "").strip():
            payload["camera"] = self.resolve_camera(str(args["camera"])) or str(args["camera"])
        out = TOOLS["recent_activity"](self.ctx, payload)
        if out.get("ok"):
            for e in out.get("events") or []:
                self.evidence.append(f"{e.get('camera')} {e.get('from')}–{e.get('to')}: עד {e.get('most_people')} "
                                     f"אנשים; " + "; ".join(f"{s.get('at')} {s.get('what')}" for s in e.get("seen") or [])[:300])
        out.pop("owner_said", None)          # the marks are memory: the writer gets the relevant ones only
        return _brief(out, ("since", "events", "error"))

    def _search_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.called.append("search_events")
        out = TOOLS["search_events"](self.ctx, {"query": str(args.get("query") or self.ctx.text)})
        self.evidence.append("חיפוש: " + json.dumps(out, ensure_ascii=False, default=str)[:600])
        return out

    def _send_clip(self, args: Dict[str, Any]) -> Dict[str, Any]:
        handle = str(args.get("handle") or "").strip().upper()
        self.called.append("send_media")
        with internal(self.ctx):
            out = TOOLS["send_media"](self.ctx, {"handle": handle})
        if out.get("ok"):
            self.evidence.append(f"נשלח הסרטון של {handle}")
        return _brief(out, ("ok", "status", "reason", "error"))

    def _record_clip(self, args: Dict[str, Any]) -> Dict[str, Any]:
        camera = self.resolve_camera(str(args.get("camera") or ""))
        self.called.append("record_clip")
        with internal(self.ctx):
            out = TOOLS["record_clip"](self.ctx, {"camera": camera, "seconds": 10})
        if out.get("ok"):
            self.evidence.append(f"נשלחו 10 שניות חיות מ{camera}")
        return _brief(out, ("ok", "status", "reason", "error", "bounds"))


def _brief(out: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    return {k: out[k] for k in keys if k in out} or dict(out)


def _look_line(ctx: ToolContext, out: Dict[str, Any]) -> str:
    from ..brain.registry import display  # noqa: PLC0415

    name = display(ctx.snapshot, str(out.get("camera") or ""), ctx.lang)
    if out.get("description"):
        people = out.get("people")
        return f"מבט חי עכשיו ב{name}: {out['description']}" + (f" (אנשים: {people})" if people is not None else "")
    return f"מבט חי ב{name} נכשל: {out.get('reason') or out.get('error') or out.get('description_error') or ''}"


def _row_line(row: Dict[str, Any]) -> str:
    if row.get("error"):
        return f"מבט חי ב{row.get('camera')}: אין תמונה ({row['error']})"
    sent = " · התמונה נשלחה לבעל הבית" if row.get("photo_sent") else ""
    return f"מבט חי עכשיו ב{row.get('camera')}: {row.get('description')} (אנשים: {row.get('people')}){sent}"


def loop(model: Any, messages: List[Dict[str, Any]], box: Toolbox, budget: Tuple[int, float], usage: List[int],
         max_tokens: int = 400) -> str:
    """The writer with tools: it may call up to *budget* tools inside the time budget, then answers. Returns the
    reply text ("" when the model failed)."""
    calls, seconds = budget
    started = time.monotonic()
    used = 0
    for _ in range(calls + 1):
        over = used >= calls or time.monotonic() - started > seconds
        msg = model.chat(messages, SCHEMAS, "none" if over else "auto", max_tokens=max_tokens)
        usage[0] += int(msg.usage[0] or 0)
        usage[1] += int(msg.usage[1] or 0)
        if msg.error:
            log.warning("v3 writer failed: %s", msg.error)
            return ""
        if not msg.tool_calls or over:
            return (msg.content or "").strip()
        messages.append({"role": "assistant", "content": msg.content or None,
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                                        for c in msg.tool_calls]})
        for c in msg.tool_calls:
            used += 1
            result = box.run(c.name, c.arguments) if used <= calls else {"ok": False, "error": "tool budget used up"}
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": json.dumps(result, ensure_ascii=False, default=str)[:3000]})
    return ""


def sent_media(ctx: ToolContext) -> bool:
    return any(r.tool in ("check_camera", "record_clip", "send_media") and r.status == DONE for r in ctx.receipts)
