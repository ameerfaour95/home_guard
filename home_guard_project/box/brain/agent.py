# home_guard_project/box/brain/agent.py
"""One owner message, start to finish.

The fast model answers first (a picture, a video, a pause, a greeting). It
hands the turn to the big model when the message needs judgement, when it
fails or runs out of steps, or when its answer claims an action that no
receipt backs. The big model starts from the same message; anything already
done stays done and is never repeated, because every acting call carries an
idempotency key and the receipts of the turn are shown to it. The big model
gets one rewrite if its own answer claims an action with no receipt; after
that only the code-written lines go out. The reply is the model's facts plus
one line per receipt, in the owner's language. Every message is saved.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..feedback import Feedback, save_feedback
from .claims import unbacked_claims
from .grounding import evidence_text, ungrounded_details
from .i18n import LANGUAGE_NAMES, t
from .memory import ChatMemory, ChatState
from .aliases import normalize
from .profiles import asks_about_now, needs_big, system_prompt, tool_names, tools_for
from .receipts import ACTING_TOOLS, DONE, FAILED, REQUESTED, UNDONE, Receipt, ReceiptBook
from .mode import hhmm
from .registry import mentioned_cameras, render_block, resolve_camera
from .render import receipt_line, render_reply, undo_what
from .tools import DEFAULTS, TOOLS, KEYS, Services, ToolContext, _issue, settings_line
from .tools import _alert_state, _alert_target, _alert_types, _alert_values

log = logging.getLogger("box.brain.agent")

FAST, BIG = "fast", "big"
UNDOABLE = ("pause_alerts", "set_camera_active", "change_setting", "set_alert_types", "set_sensitivity", "set_alias")


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def _object(value: Any) -> Dict[str, Any]:
    """Drop malformed metadata before it reaches feedback or chat persistence."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        log.warning("Ignoring malformed turn metadata")
        return {}
    out = {}
    skipped = False
    for key, item in value.items():
        try:
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            json.dumps(item, allow_nan=False)
            out[key] = item
        except (TypeError, ValueError, OverflowError, RecursionError):
            skipped = True
    if skipped:
        log.warning("Ignoring malformed turn metadata entries")
    return out


def _check_message(msg: Any, tier: str, usage: Dict[str, List[int]]) -> None:
    spent = usage.setdefault(tier, [0, 0])
    for i in range(2):
        spent[i] += max(0, int(_finite(msg.usage[i])))
    if msg.error:
        raise RuntimeError(msg.error)
    if msg.refused:
        raise RuntimeError("the model declined")


def _valid_args(args: Any) -> bool:
    try:
        if not isinstance(args, dict):
            return False
        json.dumps(args, allow_nan=False)
        return True
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False


@dataclass(frozen=True)
class AgentReply:
    text: str
    buttons: Tuple[str, ...] = ()
    question_token: str = ""                 # the clarification's token, for its buttons (cl:<token>:<index>)
    after: Tuple[Callable[[], None], ...] = ()
    lang: str = "en"
    receipts: Tuple[Receipt, ...] = ()
    tier: str = BIG
    escalated: bool = False
    guard_hits: int = 0
    usage: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    tools_called: Tuple[str, ...] = ()
    answer: str = ""
    undo_token: str = ""
    clips: Tuple[str, ...] = ()
    photos: Tuple[str, ...] = ()


class _HandOff(Exception):
    """The fast model asked for the big one, failed, or ran out of steps."""


class _ChangedSince(Exception):
    """Undo: what the turn changed no longer holds (another turn, button or person changed it since)."""


def _same_value(current: Any, wrote: Any) -> bool:
    """A setting still holds the raw value a turn wrote (the store may keep "23" as 23)."""
    if not isinstance(current, bool) and not isinstance(wrote, bool):
        try:
            return float(current) == float(wrote)
        except (TypeError, ValueError):
            pass
    return str(current).strip().lower() == str(wrote).strip().lower()


def _pause_entries(value: Any) -> List[Tuple[Optional[str], float]]:
    """The pause entries in a receipt's ``before``/``after``: (camera, until), camera None for the whole house."""
    if not isinstance(value, dict) or not isinstance(value.get("cameras", {}), dict):
        raise ValueError("invalid pause entries")
    out: List[Tuple[Optional[str], float]] = []
    if "all" in value:
        out.append((None, _finite(value["all"])))
    for name, until in value.get("cameras", {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid pause camera")
        out.append((name, _finite(until)))
    return out


def _touches_pause(entry: Optional[str], later: Receipt) -> bool:
    """Did a later pause/resume receipt set or clear this pause entry (a camera's, or the house's when None)?"""
    camera = later.detail.get("camera") or None
    if later.tool == "pause_alerts":
        return camera == entry
    if later.tool == "resume_alerts":
        # A one-camera resume can replace a house pause with entries for every other camera.
        return True
    return False


def context_block(snapshot: Any, settings_text: str, now: float, lang: str, alert_handle: Optional[str],
                  alert: Optional[Dict[str, Any]], pending_answer: Optional[Tuple[Dict[str, Any], str]],
                  text: str, box_lang: str = "en", focus: Sequence[str] = ()) -> str:
    lines = ["[HOUSE]", render_block(snapshot)]
    if settings_text:
        lines.append(settings_text)
    when = dt.datetime.fromtimestamp(now).strftime("%A %Y-%m-%d %H:%M")
    lines.append(f"[NOW] {when} · mode: {'Guard' if snapshot.mode == 'guard' else 'Assistant'}")
    if alert_handle and alert:
        alert_time = dt.datetime.fromtimestamp(float(alert.get("ts") or now)).strftime("%a %d %b %H:%M")
        lines.append(f"[ALERT THIS MESSAGE ANSWERS] {alert_handle} {alert.get('camera')} {alert_time}: "
                     f"{alert.get('summary') or 'no description'}")
    else:
        lines.append("[ALERT THIS MESSAGE ANSWERS] none - if the owner judges an alert, ask which one")
    lines.append(f"[ANSWER IN] {LANGUAGE_NAMES.get(lang, 'English')}")
    lines.append(f"[BOX LANGUAGE] {LANGUAGE_NAMES.get(box_lang, 'English')} (alerts and announcements)")
    if pending_answer:
        question, answer = pending_answer
        lines.append(f'[YOUR QUESTION] You asked: "{question.get("question")}" with the choices '
                     f'{", ".join(question.get("choices") or [])}. The owner answered: "{answer}".')
    lines += list(focus)
    lines += ["[MESSAGE]", text]
    return "\n".join(lines)


def focus_lines(state: ChatState, snapshot: Any, text: str, now: float, alert_handle: Optional[str] = None,
                alert: Optional[Dict[str, Any]] = None) -> List[str]:
    """What this message is about, worked out in code before the model reads it (the pergola bug, 2026-10-05):
    the cameras it names by the owner's own words, the camera and the event being talked about (kept in the chat
    state across turns), and whether it asks about right now. Updates the state's topics."""
    mentions = mentioned_cameras(snapshot, text)
    named = list(dict.fromkeys(camera for _, camera in mentions))
    if len(named) == 1:
        cam = snapshot.camera(named[0])
        own = [w for w, c in mentions if cam is not None and w in cam.aliases]
        old = state.topic_camera(now)
        state.set_topic_camera(named[0], own[0] if own else (old[1] if old and old[0] == named[0] else ""), now)
    if alert_handle and alert:
        state.note_observation(alert_handle, str(alert.get("observation") or alert.get("summary") or ""),
                               str(alert.get("visibility") or ""), str(alert.get("label") or ""))
        state.set_topic_event(alert_handle, now)
        old = state.topic_camera(now)
        if not named and alert.get("camera"):
            camera = str(alert["camera"])
            state.set_topic_camera(camera, old[1] if old and old[0] == camera else "", now)
    lines = []
    said = [f"{w} = {c}" for w, c in mentions if normalize(w) != normalize(c)]
    if said:
        lines.append("[CAMERAS IN THIS MESSAGE] " + ", ".join(said))
    topic = state.topic_camera(now)
    if topic:
        lines.append(f"[CAMERA BEING DISCUSSED] {topic[0]}" + (f" ({topic[1]})" if topic[1] else "")
                     + " - a request that names no camera means this one: leave camera out of check_camera / "
                       "record_clip and the box uses it")
    else:
        lines.append("[CAMERA BEING DISCUSSED] none - if the owner names no camera, leave camera out and the box "
                     "asks them; never guess one")
    event = state.topic_event(now)
    if event:
        lines.append(f"[EVENT BEING DISCUSSED] {state.event_text(event)}")
    if asks_about_now(text):
        lines.append("[RIGHT NOW] the message asks what is happening now: look live with check_camera, "
                     "not find_events")
    return lines


def _default_run_tool(ctx: ToolContext, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return TOOLS[name](ctx, args)


_NOT_PART_OF_THE_ACTION = ("owner_words", "note", "uses", "reason")


def _effective(ctx: ToolContext, name: str, args: Dict[str, Any]) -> str:
    """The operation an acting call performs, for idempotency: empty optionals dropped, camera words resolved to a
    sorted list of camera names (``camera`` and ``cameras`` are one operation), defaults filled in, and the
    explanatory fields left out - so "front" and "the front camera" are the same action."""
    out: Dict[str, Any] = {}
    cameras: set = set()
    alert_change = name in ("set_alert_types", "set_sensitivity")
    for key, value in args.items():
        if key in _NOT_PART_OF_THE_ACTION or value is None or value == "" or value == []:
            continue
        if key in ("camera", "cameras"):
            items = value if isinstance(value, list) else [value]
            for v in items:
                if alert_change:
                    camera, bad = _alert_target(ctx, v)
                    if not bad:
                        if camera:
                            cameras.add(camera)
                        continue
                if ctx.snapshot is not None:
                    cameras.add(resolve_camera(ctx.snapshot, str(v)).camera or str(v))
                else:
                    cameras.add(str(v))
            continue
        if key == "handle":
            value = str(value).strip().upper()
        if alert_change:
            try:
                if key == "types":
                    value = sorted(_alert_types(value))
                elif key == "values":
                    value = _alert_values(value)
            except (TypeError, ValueError, OverflowError):
                pass  # Invalid arguments stay distinct and are refused by the tool.
        out[key] = value
    if cameras:
        out["cameras"] = sorted(cameras)
    if name == "record_verdict" and not out.get("handle") and ctx.alert_handle:
        out["handle"] = str(ctx.alert_handle).strip().upper()
    if name == "record_clip" and "seconds" not in out:
        out["seconds"] = 10
    if name == "send_media" and ("seconds" in out or "from_sec" in out):
        from ..alert_clips import PRE_SECONDS

        out.setdefault("seconds", 10)
        out.setdefault("from_sec", -PRE_SECONDS)
    for key in ("seconds", "from_sec"):
        if key in out:
            try:
                value = _finite(out[key])
                if key == "seconds" and name == "record_clip":
                    value = float(int(min(30, max(1, value))))
                elif key == "seconds" and name == "send_media":
                    value = min(60.0, max(1.0, value))
                out[key] = value
            except (TypeError, ValueError, OverflowError):
                pass  # Keep invalid values distinct; the tool will refuse them.
    return json.dumps(out, sort_keys=True, ensure_ascii=False, default=str)


def _undo_token(ctx: ToolContext) -> str:
    return ctx.turn_id.rsplit(":", 1)[-1] if any(
        r.tool in UNDOABLE and r.status in (DONE, REQUESTED) and isinstance(r.detail, dict)
        and not r.detail.get("already") and not r.detail.get("undo_of") for r in ctx.receipts) else ""


def _fallback_reply(ctx: Optional[ToolContext], lang: str) -> AgentReply:
    text_out = t("unavailable", lang)
    if ctx is None:
        return AgentReply(text=text_out, lang=lang)
    try:
        lines = [receipt_line(r, lang) for r in ctx.receipts]
        text_out = "\n".join(line for line in lines if line) or text_out
    except Exception:
        pass
    try:
        token = _undo_token(ctx) if not ctx.turn_id.endswith(":undo") else ""
    except Exception:
        token = ""
    return AgentReply(text=text_out, after=tuple(ctx.after_reply), receipts=tuple(ctx.receipts),
                      lang=lang, undo_token=token)


class OwnerAgentV2:
    version = 2

    def __init__(self, model: Any, registry: Any, memory: ChatMemory, book: ReceiptBook, services: Services,
                 fast_model: Any = None, retention_days: float = 14.0, max_rounds: int = 5,
                 run_tool: Optional[Callable[[ToolContext, str, Dict[str, Any]], Dict[str, Any]]] = None,
                 now: Callable[[], float] = time.time) -> None:
        self.model, self.fast_model = model, fast_model
        self.registry, self.memory, self.book, self.services = registry, memory, book, services
        self.retention_days, self.max_rounds = retention_days, max_rounds
        self._run_tool = run_tool or _default_run_tool
        self._now = now
        self._lock = threading.Lock()

    # -- one tool call ---------------------------------------------------------------
    def _dispatch(self, ctx: ToolContext, name: str, args: Dict[str, Any], valid: bool,
                  allowed: Sequence[str]) -> Dict[str, Any]:
        if not valid or not _valid_args(args):
            log.warning("Ignoring malformed tool arguments")
            return {"ok": False, "error": "the tool arguments were not valid JSON; send the call again"}
        if name not in allowed or name not in TOOLS:
            return {"ok": False, "error": f"{name} is not available now"}
        key = ""
        try:
            key = f"{ctx.turn_id}:{name}:{_effective(ctx, name, args)}"
            if name in ACTING_TOOLS:
                if key in ctx.done_calls:
                    return dict(ctx.done_calls[key], note="already done in this turn; do not repeat it")
                ctx.call_key = key
            result = self._run_tool(ctx, name, args)
            if not isinstance(result, dict):
                raise ValueError("tool result must be an object")
            json.dumps(result, allow_nan=False)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not end the turn
            log.warning("Tool %s failed: %s", name, exc)
            result = {"ok": False, "error": "that could not be done"}
        finally:
            ctx.call_key = ""
        if name in ACTING_TOOLS and (result.get("receipt") or result.get("receipts")):
            ctx.done_calls[key] = result
        ctx.results.append(json.dumps(result, ensure_ascii=False, default=str))     # what an answer may rely on
        return result

    # -- visual details must come from the clip ------------------------------------------
    def _check_the_clip(self, ctx: ToolContext, answer: str, missing: List[str], now: float, lang: str,
                        called: List[str]) -> str:
        """The answer gives visual details (a colour, clothing, what is in a hand) that neither the observation nor
        a vision answer backs: ask the clip in code and send what it shows instead of the model's words."""
        handle = ctx.state.topic_event(now) or ctx.alert_handle
        if not handle:
            log.warning("grounding: %s is in no observation, and no event is being discussed", missing)
            return answer
        log.warning("grounding: %s is not in the observation; asking the clip", missing)
        called.append("ask_vision")
        result = self._dispatch(ctx, "ask_vision", {"event_id": handle, "question": ctx.text}, True, ["ask_vision"])
        if not result.get("ok"):
            return f"{t('checking_clip', lang)}\n{t('clip_check_failed', lang)}"
        where = (f" ({t('clip_frame', lang, frame=result['frame'], time=result['time'])})"
                 if result.get("frame") and result.get("time") else "")
        return f"{t('checking_clip', lang)}\n{result.get('answer')}{where}"

    def note_alert(self, chat_id: Any, alert: Dict[str, Any]) -> Optional[str]:
        """An alert the box sent to this chat goes into its history as the vision agent's observation (text, about
        150 tokens, with the event handle); the clip stays on the box. It becomes the event - and its camera the
        camera - being talked about, so "what was in his hand?" knows which clip. Returns the handle; never raises."""
        with self._lock:
            try:
                alert = _object(alert)
                if not alert.get("alert_id"):
                    return None
                chat_id, now = str(chat_id), _finite(self._now())
                try:
                    ts = _finite(alert.get("ts") or now)
                    dt.datetime.fromtimestamp(ts)
                except (TypeError, ValueError, OverflowError, OSError):
                    ts = now
                camera = str(alert.get("camera") or "")
                state = self.memory.load(chat_id)
                handle = state.add_handle("event", str(alert["alert_id"]), camera, ts, str(alert.get("summary") or ""))
                state.note_observation(handle, str(alert.get("observation") or alert.get("summary") or ""),
                                       str(alert.get("visibility") or ""), str(alert.get("label") or ""))
                noted = any(isinstance(turn, dict) and turn.get("kind") == "alert"
                            and handle in (turn.get("handles") or []) for turn in state.turns[-20:])
                if not noted:                                      # a reminder of the same alert is not a new one
                    state.add_event_turn(handle, now)
                state.set_topic_event(handle, now)
                if camera:
                    old = state.topic_camera(now)
                    state.set_topic_camera(camera, old[1] if old and old[0] == camera else "", now)
                self.memory.save(chat_id, state)
                return handle
            except Exception as exc:  # noqa: BLE001 - an alert must never fail because of the chat history
                log.warning("Could not note the alert in the chat history: %s", exc)
                return None

    # -- the model/tool loop -----------------------------------------------------------
    def _loop(self, ctx: ToolContext, model: Any, messages: List[Dict[str, Any]], tier: str,
              usage: Dict[str, List[int]], called: List[str], only: Optional[Sequence[str]] = None) -> str:
        allowed = list(only) if only else tool_names(ctx.mode, tier)
        schemas = [s for s in tools_for(ctx.mode, tier) if s["function"]["name"] in allowed]
        for _ in range(self.max_rounds):
            msg = model.chat(messages, schemas)
            _check_message(msg, tier, usage)
            if not msg.tool_calls:
                if tier == FAST and not (msg.content or "").strip():
                    raise _HandOff("empty answer")
                return (msg.content or "").strip()
            messages.append({
                "role": "assistant", "content": msg.content or None, "_raw": msg.raw,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": c.raw_arguments or
                                             (json.dumps(c.arguments) if _valid_args(c.arguments) else "{}")}}
                               for c in msg.tool_calls],
            })
            final: Optional[str] = None
            for c in msg.tool_calls:
                if ctx.clarification is not None:      # a question was asked: nothing after it runs
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(
                        {"ok": False, "error": "not run: you asked the owner a question; wait for the answer"})})
                    continue
                valid = c.valid and _valid_args(c.arguments)
                if c.name == "hand_off" and tier == FAST and valid:
                    raise _HandOff(str(c.arguments.get("reason") or ""))
                if c.name == "reply":
                    final = str(c.arguments.get("answer") or "") if valid else ""
                    result: Dict[str, Any] = {"ok": True}
                else:
                    called.append(c.name)
                    result = self._dispatch(ctx, c.name, c.arguments, c.valid, allowed)
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
            if ctx.clarification is not None:
                return ""
            if final is not None:
                return final.strip()
        if tier == FAST:
            raise _HandOff("out of steps")
        last = model.chat(messages, schemas, tool_choice="none")
        _check_message(last, tier, usage)
        return (last.content or "").strip()

    # -- one message -------------------------------------------------------------------
    def handle(self, text: str, chat_id: Any, who: Optional[Dict[str, Any]] = None,
               alert: Optional[Dict[str, Any]] = None, threaded: bool = False,
               choice: Optional[Tuple[str, int]] = None) -> Optional[AgentReply]:
        """Act on one owner message. Never raises; the message is always saved. ``choice`` (token, index) makes
        the message a tapped button: it is checked against the saved question under the lock, and a stale or
        already-used token does nothing (None)."""
        with self._lock:
            try:
                return self._handle(str(text or ""), str(chat_id), _object(who), _object(alert) or None, threaded,
                                    choice)
            except Exception as exc:  # the outer boundary also covers loading and saving state
                log.warning("Owner message failed at the poll boundary: %s", exc)
                lang = ChatState().language_for("", text if isinstance(text, str) else "")
                reply_text = t("unavailable", lang)
                try:
                    save_feedback(self.services.feedback_dir, None, Feedback(), str(text or ""),
                                  {}, str(chat_id), time.time())
                except Exception as save_exc:
                    log.warning("Could not save the owner's message: %s", save_exc)
                return AgentReply(text=reply_text, lang=lang)

    def handle_choice(self, chat_id: Any, token: str, index: int,
                      who: Optional[Dict[str, Any]] = None) -> Optional[AgentReply]:
        """A tapped clarification button (``cl:<token>:<index>``): the choice's text becomes the owner's message.
        A button from an older question (another token) does nothing."""
        try:
            if type(index) is not int or not isinstance(token, str) or not token:
                log.warning("Ignoring malformed clarification callback")
                return None
            state = self.memory.load(chat_id)
            pending = state.pending if isinstance(state.pending, dict) else {}
            choices = pending.get("choices") or []
            if (pending.get("token") != token or not isinstance(choices, list)
                    or not 0 <= index < len(choices) or not isinstance(choices[index], str)):
                return None
            return self.handle(choices[index], chat_id, who, choice=(token, index))
        except Exception as exc:
            log.warning("Could not handle clarification callback: %s", exc)
            return None

    def undo_turn(self, chat_id: Any, token: str, who: Optional[Dict[str, Any]] = None) -> AgentReply:
        """The Undo button: put back what one turn changed (pause, camera on/off, settings). Never raises."""
        with self._lock:
            ctx = None
            lang = "en"
            try:
                chat_id, who = str(chat_id), _object(who)
                if not isinstance(token, str) or not token.isascii() or not token.isdigit():
                    raise ValueError("invalid undo token")
                now = _finite(self._now())
                state = self.memory.load(chat_id)
                try:
                    settings = _object(self.services.read_settings()) if self.services.read_settings else {}
                except Exception:  # noqa: BLE001
                    settings = {}
                speaker = str(who.get("user_id") or "")
                lang = state.language_for(speaker, "", default=str(settings.get("owner_language") or "en"))
                try:
                    snapshot = self.registry.snapshot()
                except Exception:  # noqa: BLE001
                    return AgentReply(text=t("unavailable", lang), lang=lang)
                ctx = ToolContext(turn_id=f"{chat_id}:{token}:undo", chat_id=chat_id, speaker=who, text="", lang=lang,
                                  mode=snapshot.mode, snapshot=snapshot, state=state, services=self.services,
                                  book=self.book)
                processed = 0
                lines: List[str] = []
                for r in reversed(self.book.turn_receipts(f"{chat_id}:{token}")):     # newest first
                    if (r.tool not in UNDOABLE or r.status not in (DONE, REQUESTED)
                            or r.detail.get("already") or r.detail.get("undo_of")):
                        continue
                    processed += 1
                    first = len(ctx.receipts)
                    try:
                        self._undo_one(ctx, r, snapshot, now)
                    except _ChangedSince:
                        # Left as it is, not marked undone; the line says why nothing happened.
                        lines.append(t("undo_changed_since", lang, what=undo_what(r.tool, r.detail, r.target, lang)))
                        continue
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Undo of %s failed: %s", r.summary(), exc)
                        _issue(ctx, r.tool, FAILED, r.target,
                               {"undo_of": r.tool, "camera": str(r.detail.get("camera") or ""),
                                "setting": str(r.detail.get("setting") or "")}, "error")
                        lines += [receipt_line(x, lang, self.retention_days) for x in ctx.receipts[first:]]
                        continue                                   # not undone: it stays undoable
                    self.book.update(r, UNDONE)
                    lines += [receipt_line(x, lang, self.retention_days) for x in ctx.receipts[first:]]
                if processed:
                    text_out = "\n".join(line for line in lines if line) or t("unavailable", lang)
                else:
                    text_out = t("nothing_to_undo", lang)
                state.add_turn(speaker, "↩", text_out, [], [r.summary() for r in ctx.receipts], now)
                self.memory.save(chat_id, state)
                return AgentReply(text=text_out, after=tuple(ctx.after_reply), lang=lang, receipts=tuple(ctx.receipts))
            except Exception as exc:
                log.warning("Undo failed at the poll boundary: %s", exc)
                return _fallback_reply(ctx, lang)

    def _undo_one(self, ctx: ToolContext, r: Receipt, snapshot: Any, now: float) -> None:
        """Reverse one receipt - only while what it changed still holds. Raises _ChangedSince when it no longer
        does (nothing is touched), any other exception when the reversal itself fails."""
        d = r.detail
        later = [x for x in self.book.later(r.turn, r.id)
                 if x.status in (DONE, REQUESTED) and not x.detail.get("undo_of") and not x.detail.get("already")]
        if r.tool == "pause_alerts":
            before = d.get("before")
            if (not isinstance(before, dict) or "all" not in before
                    or not isinstance(before.get("cameras"), dict)):
                raise ValueError("invalid pause restore data")
            earlier = dict(_pause_entries(before))
            entries = _pause_entries(d.get("after"))
            if not entries:
                raise ValueError("missing pause undo data")
            current = self.services.mute.snapshot()
            held = []
            for entry, until in entries:
                now_until = current["all"] if entry is None else current["cameras"].get(entry, 0.0)
                if until > now and float(now_until) == until and not any(_touches_pause(entry, x) for x in later):
                    held.append(entry)
            if not held:
                raise _ChangedSince()
            for entry in held:                        # only this turn's entries; every other pause is untouched
                if self.services.mute.set_entry(entry, earlier.get(entry, 0.0), now) is False:
                    raise ValueError("pause restore was not saved")
            for entry in held:
                detail: Dict[str, Any] = {"camera": entry or "", "undo_of": "pause_alerts"}
                if entry:
                    still = self.services.mute.muted_until(now, entry)
                    detail["still_until"] = hhmm(still) if still else ""
                else:
                    left = self.services.mute.snapshot()
                    if float(left["all"]) > now:
                        detail["still_until"] = hhmm(float(left["all"]))
                    else:
                        detail["still_pauses"] = [[name, hhmm(until)] for name, until in sorted(left["cameras"].items())
                                                  if until > now]
                _issue(ctx, "resume_alerts", DONE, entry or "all", detail)
        elif r.tool == "set_camera_active":
            camera, active = d.get("camera"), d.get("active")
            if type(active) is not bool or not isinstance(camera, str):
                raise ValueError("invalid camera restore data")
            if not getattr(snapshot, "state_known", True):
                raise ValueError("the camera list could not be read")
            cam = snapshot.camera(camera)
            if (cam is None or cam.enabled is not active
                    or any(x.tool == "set_camera_active" and (x.detail.get("camera") or x.target) == camera
                           for x in later)):
                raise _ChangedSince()
            back = not active
            result = self.services.set_camera(camera, back)
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise ValueError("camera change was refused")
            _issue(ctx, "set_camera_active", REQUESTED, camera,
                   {"camera": camera, "active": back, "chat_id": ctx.chat_id, "lang": ctx.lang,
                    "undo_of": "set_camera_active"})
            if self.services.request_restart and self.services.request_restart not in ctx.after_reply:
                ctx.after_reply.append(self.services.request_restart)
        elif r.tool in ("set_alert_types", "set_sensitivity"):
            api = self.services.alert_settings
            camera = d.get("camera") or None
            key = "alert_on" if r.tool == "set_alert_types" else "sensitivity"
            own_key = "own_alert_on" if key == "alert_on" else "own_sensitivity"
            if camera is not None and not isinstance(camera, str):
                raise ValueError("invalid alert restore camera")
            if "new" not in d or "own_before" not in d or (camera is not None and "own_after" not in d):
                raise ValueError("missing alert restore data")
            back = d["house_before"] if camera is None else d["own_before"]
            if camera is None and back is None:
                raise ValueError("missing house restore data")
            if back is not None:
                # Receipts hold exact stored values, never commands such as default or +animal.
                if key == "alert_on":
                    if not isinstance(back, list) or not back or any(
                            kind not in ("person", "vehicle", "animal") for kind in back):
                        raise ValueError("invalid alert types restore data")
                elif not isinstance(back, dict) or _alert_values(back) != back:
                    raise ValueError("invalid sensitivity restore data")
            current = _alert_state(api.get_alert_settings(camera), camera)
            row = current["house"] if camera is None else current
            if (row[key] != d["new"] or (camera is not None and row.get(own_key) != d["own_after"])
                    or any(x.tool == r.tool and (x.detail.get("camera") or None) == camera for x in later)):
                raise _ChangedSince()
            if r.tool == "set_alert_types":
                after = api.set_alert_types(camera, ["default"] if back is None else back)
            elif camera is None:
                # Restore only the types this turn changed; the others keep following box.yaml's own default.
                if not isinstance(d["new"], dict):
                    raise ValueError("invalid sensitivity restore data")
                changed = {k: back[k] for k in d["new"] if k in back and d["new"][k] != back[k]}
                after = api.set_sensitivity(None, changed) if changed else api.get_alert_settings(None)
            elif back is None:
                after = api.set_sensitivity(camera, "default")
            elif not isinstance(d["own_after"], (dict, type(None))):
                raise ValueError("invalid sensitivity restore data")
            elif set(d["own_after"] or ()) <= set(back):   # None: the turn reset it to the house values
                after = api.set_sensitivity(camera, back)       # same keys: one merge call restores them
            else:
                # The turn added keys: updates merge, so clear first, then restore the exact partial override.
                api.set_sensitivity(camera, "default")
                try:
                    after = api.set_sensitivity(camera, back)
                except Exception:
                    try:
                        api.set_sensitivity(camera, d["own_after"])     # put the camera back as the turn left it
                    except Exception:   # noqa: BLE001 - the original failure is the one reported
                        log.warning("Could not put the camera sensitivity back after a failed undo")
                    raise
            after = _alert_state(after, camera)
            row = after["house"] if camera is None else after
            if (row[key] if camera is None else row.get(own_key)) != back:
                raise ValueError("alert restore was not saved")
            _issue(ctx, r.tool, DONE, camera or "house",
                   {"camera": camera or "", "old": d["new"], "new": row[key], "undo_of": r.tool})
        elif r.tool == "set_alias":
            camera, alias = d.get("camera"), d.get("alias")
            if not isinstance(camera, str) or not isinstance(alias, str) or not alias.strip():
                raise ValueError("invalid alias restore data")
            cam = snapshot.camera(camera)
            if (cam is None or normalize(alias) not in [normalize(a) for a in cam.aliases]
                    or any(x.tool == "set_alias" and normalize(str(x.detail.get("alias") or "")) == normalize(alias)
                           for x in later)):
                raise _ChangedSince()
            if self.services.remove_alias is None:
                raise ValueError("aliases cannot be removed on this box")
            self.services.remove_alias(camera, alias)
            _issue(ctx, "set_alias", DONE, camera, {"camera": camera, "alias": alias, "undo_of": "set_alias"})
        elif r.tool == "change_setting":
            setting = d.get("setting")
            restore, wrote = d.get("restore"), d.get("wrote")
            keys = KEYS.get(setting, ()) if isinstance(setting, str) else ()
            if (not isinstance(restore, dict) or not keys or set(restore) != set(keys)
                    or not isinstance(wrote, dict) or set(wrote) != set(keys)):
                raise ValueError("missing or invalid setting undo data")
            json.dumps([restore, wrote], allow_nan=False)
            if any(not isinstance(v, (str, int, float, bool)) for v in list(restore.values()) + list(wrote.values())):
                raise ValueError("invalid setting undo values")
            current = self.services.read_settings() if self.services.read_settings else None
            if not isinstance(current, dict):
                raise ValueError("settings could not be read")
            if (not all(_same_value(current.get(k, DEFAULTS[k]), wrote[k]) for k in keys)
                    or any(x.tool == "change_setting" and x.detail.get("setting") == setting for x in later)):
                raise _ChangedSince()
            for key in keys:
                value = restore[key]
                self.services.set_option(key, str(value).lower() if isinstance(value, bool) else str(value))
            _issue(ctx, "change_setting", DONE, setting,
                   {"setting": setting, "old": d.get("new", ""), "new": d.get("old", ""), "undo_of": "change_setting"})

    def _handle(self, text: str, chat_id: str, who: Dict[str, Any], alert: Optional[Dict[str, Any]],
                threaded: bool, choice: Optional[Tuple[str, int]] = None) -> Optional[AgentReply]:
        now = _finite(self._now())
        state: ChatState = self.memory.load(chat_id)
        if choice is not None:
            pending_now = state.pending if isinstance(state.pending, dict) else {}
            choices_now = pending_now.get("choices")
            if (pending_now.get("token") != choice[0] or not isinstance(choices_now, list)
                    or not 0 <= choice[1] < len(choices_now) or choices_now[choice[1]] != text):
                return None
        speaker = str(who.get("user_id") or "")
        try:
            settings = _object(self.services.read_settings()) if self.services.read_settings else {}
        except Exception:  # noqa: BLE001
            settings = {}
        box_lang = str(settings.get("owner_language") or "en")
        lang = state.language_for(speaker, text, default=box_lang)
        failed = False
        try:
            snapshot = self.registry.snapshot()
        except Exception as exc:  # noqa: BLE001
            log.warning("House snapshot failed: %s", exc)
            snapshot = None
        ctx = ToolContext(turn_id=f"{chat_id}:{int(now * 1000)}", chat_id=chat_id, speaker=who, text=text, lang=lang,
                          mode=getattr(snapshot, "mode", "guard"), snapshot=snapshot, state=state,
                          services=self.services, book=self.book, threaded=threaded)
        if alert and alert.get("alert_id"):
            try:
                alert["ts"] = _finite(alert.get("ts") or 0)
                dt.datetime.fromtimestamp(alert["ts"])
            except (TypeError, ValueError, OverflowError, OSError):
                log.warning("Ignoring malformed alert timestamp")
                alert["ts"] = 0.0
            ctx.alert_handle = state.add_handle("event", str(alert["alert_id"]), str(alert.get("camera") or ""),
                                                float(alert.get("ts") or 0), str(alert.get("summary") or ""))
        pending, state.pending = state.pending, None
        if pending and pending.get("request"):
            # The answer to a question completes the original request: quote checks (pause, settings, verdict)
            # read the owner's first message together with the answer.
            ctx.text = f"{pending['request']}\n{text}"
        answer, tier, escalated, guard_hits = "", BIG, False, 0
        usage: Dict[str, List[int]] = {}
        called: List[str] = []
        try:
            if snapshot is None:
                raise RuntimeError("no house snapshot")
            settings_text = settings_line(settings, lang) if settings else ""
            focus = focus_lines(state, snapshot, text, now, ctx.alert_handle, alert)
            block = context_block(snapshot, settings_text, now, lang, ctx.alert_handle, alert,
                                  (pending, text) if pending else None, text, box_lang, focus)
            history = state.history_messages(now)
            skip_fast = self.fast_model is None or needs_big(text, threaded or bool(alert)) or pending is not None
            tiers = [(BIG, self.model)] if skip_fast else [(FAST, self.fast_model), (BIG, self.model)]
            for tier, model in tiers:
                note = ""
                if tier == BIG and ctx.receipts:
                    note = "\n[ALREADY DONE THIS TURN] " + "; ".join(r.summary() for r in ctx.receipts)
                messages = [{"role": "system", "content": system_prompt(ctx.mode, self.retention_days, tier)},
                            *history, {"role": "user", "content": block + note}]
                try:
                    answer = self._loop(ctx, model, messages, tier, usage, called)
                except _HandOff as why:
                    log.info("Fast model handed off: %s", why)
                    escalated = True
                    continue
                except Exception as exc:  # noqa: BLE001
                    if tier == FAST:
                        log.warning("Fast model failed (%s); asking the big model.", exc)
                        escalated = True
                        continue
                    raise
                if ctx.clarification is not None:
                    break
                if tier == FAST and not answer:
                    log.info("Fast model gave an empty answer; asking the big model.")
                    escalated = True
                    continue
                bad = unbacked_claims(answer, ctx.receipts)
                if bad and tier == FAST:
                    log.warning("Fast answer claims %s with no receipt; asking the big model.", bad)
                    guard_hits += 1
                    escalated = True
                    continue
                if bad:
                    guard_hits += 1
                    if answer:
                        messages.append({"role": "assistant", "content": answer})
                    # A promise to remember with nothing saved (2026-10-05): the model may still save it now.
                    save = "save" in bad and "set_alias" in tool_names(ctx.mode, tier)
                    messages.append({"role": "user", "content": (
                        f"[BOX] Your answer describes actions that did not happen ({', '.join(bad)}). "
                        + ("If the owner asked you to remember a name for a camera, call set_alias now. " if save
                           else "") +
                        "Write the answer again with facts only and call reply. Do not call any other tool.")})
                    answer = self._loop(ctx, model, messages, tier, usage, called,
                                        only=["reply", "set_alias"] if save else ["reply"])
                    still = unbacked_claims(answer, ctx.receipts)
                    if still:
                        guard_hits += 1
                        log.warning("claim_guard: the answer still claims actions; sending only the receipt lines")
                        # Never "I'll remember" without a save receipt: say plainly that nothing was saved.
                        answer = t("not_saved_yet", lang) if "save" in still else ""
                break
            if ctx.clarification is None and answer:
                # The observation is partial: a visual detail it does not give was never checked (§10).
                evidence = evidence_text([render_block(snapshot), str((alert or {}).get("summary") or ""),
                                          str((alert or {}).get("observation") or ""), *focus, *ctx.results,
                                          *(state.event_text(h) for h, e in state.handles.items()
                                            if isinstance(e, dict) and e.get("kind") == "event")])
                missing = ungrounded_details(answer, evidence)
                if missing:
                    guard_hits += 1
                    answer = self._check_the_clip(ctx, answer, missing, now, lang, called)
        except Exception as exc:  # noqa: BLE001 - no network, a model error: the owner still gets an answer
            log.warning("The agent could not handle a message: %s", exc)
            failed = True
            answer = ""
        try:
            if not ctx.saved:
                save_feedback(self.services.feedback_dir, alert, Feedback(), text, who, chat_id, now)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save the owner's message: %s", exc)
        try:
            buttons: Tuple[str, ...] = ()
            if ctx.clarification is not None:
                reply_text = ctx.clarification["question"]
                if ctx.receipts:
                    done = render_reply("", ctx.receipts, lang, self.retention_days)
                    if done:
                        reply_text = f"{done}\n\n{reply_text}"
                buttons = tuple(ctx.clarification["choices"])
                state.pending = dict(ctx.clarification, request=ctx.text, speaker=speaker,
                                     token=uuid.uuid4().hex[:8])
            else:
                reply_text = render_reply(t("unavailable", lang) if failed else answer,
                                          ctx.receipts, lang, self.retention_days)
                if not reply_text:
                    reply_text = t("unavailable" if failed else "nothing_done", lang)
            try:
                state.add_turn(speaker, text, reply_text, ctx.shown, [r.summary() for r in ctx.receipts], now,
                               notes=ctx.vision_notes)
                self.memory.save(chat_id, state)
            except Exception as exc:  # noqa: BLE001 - what was done is still reported, and its restart still runs
                log.warning("Could not save the conversation: %s", exc)
            return AgentReply(text=reply_text, buttons=buttons,
                              question_token=(state.pending or {}).get("token", "") if buttons else "",
                              after=tuple(ctx.after_reply), lang=lang,
                              receipts=tuple(ctx.receipts), tier=tier, escalated=escalated, guard_hits=guard_hits,
                              usage={k: (v[0], v[1]) for k, v in usage.items()}, tools_called=tuple(called),
                              answer=answer, undo_token=_undo_token(ctx))
        except Exception as exc:
            log.warning("Could not finish owner reply: %s", exc)
            return _fallback_reply(ctx, lang)


def follow_up_camera_receipts(book: ReceiptBook, registry: Any, deliverer: Any, now: Optional[float] = None,
                              give_up_sec: float = 600.0) -> int:
    """After a restart: confirm camera changes the box asked for, or report them failed. Returns lines sent."""
    try:
        now = _finite(time.time() if now is None else now)
        give_up_sec = _finite(give_up_sec)
        snapshot = registry.snapshot()
        receipts = book.open_receipts("set_camera_active")
        if not isinstance(receipts, (list, tuple)):
            raise ValueError("expected a list of receipts")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load camera follow-ups: %s", exc)
        return 0
    sent = 0
    warned = False
    for receipt in receipts:
        try:
            if (not isinstance(receipt, Receipt) or not isinstance(receipt.detail, dict)
                    or type(receipt.detail.get("active")) is not bool):
                raise ValueError("invalid camera receipt")
            timestamp = _finite(receipt.ts)
            json.dumps(receipt.to_dict(), allow_nan=False)
            cam = snapshot.camera(receipt.target)
            lang = str(receipt.detail.get("lang") or "en")
            chat = str(receipt.detail.get("chat_id") or "")
            wanted = receipt.detail["active"]
            # The restarted detector loaded cameras.yaml: this is its running state.
            if cam is not None and cam.enabled == wanted:
                status, reason = DONE, None
                line = t("camera_on_done" if wanted else "camera_off_done", lang, camera=receipt.target)
            elif now - timestamp > give_up_sec:
                status, reason = FAILED, "error"
                line = t("failed", lang, what=t("what_set_camera_active", lang), reason=t("reason_error", lang))
            else:
                continue
            result = deliverer.text(chat, line) if chat else {"ok": True}
            if isinstance(result, dict) and result.get("ok") is True:
                book.update(receipt, status, reason=reason)       # closed only once the owner was told
                sent += 1 if chat else 0
        except Exception as exc:
            if not warned:
                log.warning("Could not complete camera follow-up: %s", exc)
                warned = True
    return sent


def _budgeted(vision: Any, limit: int, path: str, wrapper: Any) -> Any:
    return wrapper(vision, limit, path) if vision is not None else None


def build_owner_agent(box_settings: Dict[str, Any], env: Dict[str, str], mute: Any, cfg: Any, live_dir: str,
                      archive_dir: str, log_dir: str, feed: Any = None) -> Tuple[Optional[OwnerAgentV2], Any]:
    """The production agent and its Telegram deliverer (the agent is None when no model key is set)."""
    from .. import alert_settings, boxconfig  # noqa: PLC0415
    from ..embeddings import make_embedder  # noqa: PLC0415
    from ..find_cameras import _restart_running_mode, apply_changes  # noqa: PLC0415
    from ..telegram_agent import alert_roots  # noqa: PLC0415
    from . import aliases, media  # noqa: PLC0415
    from .deliver import Deliverer  # noqa: PLC0415
    from .models import make_model  # noqa: PLC0415
    from .registry import CAMERAS_PATH, STATUS_PATH, HouseRegistry, hours_from_box_yaml  # noqa: PLC0415
    from .vision import BudgetedVision, make_vision  # noqa: PLC0415

    box_settings = _object(box_settings)
    env = _object(env)
    deliverer = Deliverer(cfg, feed=feed)
    big = make_model(str(box_settings.get("agent_model") or "openai:gpt-4o"), env)
    fast_spec = str(box_settings.get("agent_fast_model", "openai:gpt-4o-mini") or "")
    fast = make_model(fast_spec, env) if fast_spec else None
    if big is None:
        big, fast = fast, None
    if big is None:
        return None, deliverer
    work_dir = os.path.join(live_dir, ".live")

    def set_camera(camera: str, active: bool) -> Dict[str, Any]:
        apply_changes({"cameras": [{"name": camera, "new_name": camera, "enabled": active}]}, CAMERAS_PATH,
                      restart=False)
        return {"ok": True}

    retention = float(boxconfig.PRODUCTION_RETENTION_DAYS)
    try:
        vision_budget = int(_finite(box_settings.get("vision_daily_budget", 300) or 300))
    except (TypeError, ValueError, OverflowError):
        log.warning("Invalid vision daily budget; using 300")
        vision_budget = 300
    services = Services(
        roots=lambda: alert_roots(live_dir, archive_dir), desc_dir=os.path.join(live_dir, ".desc"),
        feedback_dir=live_dir, work_dir=work_dir, mute=mute, deliver=deliverer,
        vision=_budgeted(make_vision(env, str(box_settings.get("vlm_model") or "gpt-4o")),
                         vision_budget,
                         os.path.join(live_dir, ".registry", "vision_budget.json"), BudgetedVision),
        grab_photo=lambda camera: media.grab_photo(camera, work_dir, CAMERAS_PATH),
        record_live=lambda camera, seconds: media.record_live(camera, seconds, work_dir, CAMERAS_PATH),
        cut_segment=media.cut_segment, set_camera=set_camera, add_alias=aliases.add_alias,
        remove_alias=aliases.remove_alias,
        request_restart=_restart_running_mode,
        embedder=make_embedder(env, os.path.join(live_dir, ".alert_embeddings.json")),
        retention_days=retention, set_option=boxconfig.set_option, read_settings=boxconfig.load_box_settings,
        alert_settings=alert_settings,
    )
    def quiet_log_on() -> bool:
        return bool(boxconfig.load_box_settings().get("quiet_log", False))

    registry = HouseRegistry(mute, hours_from_box_yaml(), CAMERAS_PATH, aliases.ALIASES_PATH, STATUS_PATH,
                             os.path.join(live_dir, ".registry", "sees.json"), retention_days=retention,
                             quiet_log=quiet_log_on,
                             quiet_since_path=os.path.join(live_dir, ".registry", "quiet_since.json"))
    # vision_daily_budget (box.yaml, default 300): the most vision calls the assistant may make per day.
    agent = OwnerAgentV2(big, registry, ChatMemory(os.path.join(live_dir, ".conversations")),
                         ReceiptBook(os.path.join(live_dir, ".receipts")), services, fast_model=fast,
                         retention_days=retention)
    return agent, deliverer
