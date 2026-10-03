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
from .i18n import LANGUAGE_NAMES, t
from .memory import ChatMemory, ChatState
from .profiles import needs_big, system_prompt, tool_names, tools_for
from .receipts import ACTING_TOOLS, DONE, FAILED, Receipt, ReceiptBook
from .registry import render_block, resolve_camera
from .render import render_reply
from .tools import TOOLS, Services, ToolContext, settings_line

log = logging.getLogger("box.brain.agent")

FAST, BIG = "fast", "big"


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
    clips: Tuple[str, ...] = ()
    photos: Tuple[str, ...] = ()


class _HandOff(Exception):
    """The fast model asked for the big one, failed, or ran out of steps."""


def context_block(snapshot: Any, settings_text: str, now: float, lang: str, alert_handle: Optional[str],
                  alert: Optional[Dict[str, Any]], pending_answer: Optional[Tuple[Dict[str, Any], str]],
                  text: str, box_lang: str = "en") -> str:
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
    lines += ["[MESSAGE]", text]
    return "\n".join(lines)


def _default_run_tool(ctx: ToolContext, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return TOOLS[name](ctx, args)


_NOT_PART_OF_THE_ACTION = ("owner_words", "note", "uses", "reason")


def _effective(ctx: ToolContext, args: Dict[str, Any]) -> str:
    """The operation an acting call performs, for idempotency: camera words resolved to camera names, and the
    explanatory fields left out - so "front" and "the front camera" are the same action."""
    out: Dict[str, Any] = {}
    for key, value in args.items():
        if key in _NOT_PART_OF_THE_ACTION:
            continue
        if key == "handle":
            value = str(value or "").strip().upper()
        if key in ("camera", "cameras") and ctx.snapshot is not None:
            items = value if isinstance(value, list) else [value]
            names = sorted({resolve_camera(ctx.snapshot, str(v)).camera or str(v) for v in items})
            value = names if key == "cameras" else names[0]
        out[key] = value
    return json.dumps(out, sort_keys=True, ensure_ascii=False, default=str)


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
            key = f"{ctx.turn_id}:{name}:{_effective(ctx, args)}"
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
        return result

    # -- the model/tool loop -----------------------------------------------------------
    def _loop(self, ctx: ToolContext, model: Any, messages: List[Dict[str, Any]], tier: str,
              usage: Dict[str, List[int]], called: List[str], only: Optional[Sequence[str]] = None) -> str:
        allowed = list(only) if only else tool_names(ctx.mode, tier)
        schemas = [s for s in tools_for(ctx.mode, tier) if s["function"]["name"] in allowed]
        for _ in range(self.max_rounds):
            msg = model.chat(messages, schemas)
            _check_message(msg, tier, usage)
            if not msg.tool_calls:
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
               alert: Optional[Dict[str, Any]] = None, threaded: bool = False) -> AgentReply:
        """Act on one owner message. Never raises; the message is always saved."""
        with self._lock:
            try:
                return self._handle(str(text or ""), str(chat_id), _object(who), _object(alert) or None, threaded)
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
            return self.handle(choices[index], chat_id, who)
        except Exception as exc:
            log.warning("Could not handle clarification callback: %s", exc)
            return None

    def _handle(self, text: str, chat_id: str, who: Dict[str, Any], alert: Optional[Dict[str, Any]],
                threaded: bool) -> AgentReply:
        now = _finite(self._now())
        state: ChatState = self.memory.load(chat_id)
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
            settings_text = settings_line(settings) if settings else ""
            block = context_block(snapshot, settings_text, now, lang, ctx.alert_handle, alert,
                                  (pending, text) if pending else None, text, box_lang)
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
                bad = unbacked_claims(answer, ctx.receipts)
                if bad and tier == FAST:
                    log.warning("Fast answer claims %s with no receipt; asking the big model.", bad)
                    guard_hits += 1
                    escalated = True
                    continue
                if bad:
                    guard_hits += 1
                    messages.append({"role": "user", "content": (
                        f"[BOX] Your answer describes actions that did not happen ({', '.join(bad)}). Write the "
                        "answer again with facts only and call reply. Do not call any other tool.")})
                    answer = self._loop(ctx, model, messages, tier, usage, called, only=["reply"])
                    if unbacked_claims(answer, ctx.receipts):
                        guard_hits += 1
                        log.warning("claim_guard: the answer still claims actions; sending only the receipt lines")
                        answer = ""
                break
        except Exception as exc:  # noqa: BLE001 - no network, a model error: the owner still gets an answer
            log.warning("The agent could not handle a message: %s", exc)
            failed = True
            answer = ""
        buttons: Tuple[str, ...] = ()
        if ctx.clarification is not None:
            reply_text = ctx.clarification["question"]
            buttons = tuple(ctx.clarification["choices"])
            state.pending = dict(ctx.clarification, request=ctx.text, speaker=speaker,
                                 token=uuid.uuid4().hex[:8])
        else:
            reply_text = render_reply(t("unavailable", lang) if failed else answer,
                                      ctx.receipts, lang, self.retention_days)
            if not reply_text:
                reply_text = t("unavailable" if failed else "nothing_done", lang)
        try:
            if not ctx.saved:
                save_feedback(self.services.feedback_dir, alert, Feedback(), text, who, chat_id, now)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save the owner's message: %s", exc)
        state.add_turn(speaker, text, reply_text, ctx.shown, [r.summary() for r in ctx.receipts], now)
        self.memory.save(chat_id, state)
        return AgentReply(text=reply_text, buttons=buttons,
                          question_token=(state.pending or {}).get("token", "") if buttons else "",
                          after=tuple(ctx.after_reply), lang=lang,
                          receipts=tuple(ctx.receipts), tier=tier, escalated=escalated, guard_hits=guard_hits,
                          usage={k: (v[0], v[1]) for k, v in usage.items()}, tools_called=tuple(called),
                          answer=answer)


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
    from .. import boxconfig  # noqa: PLC0415
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
        request_restart=_restart_running_mode,
        embedder=make_embedder(env, os.path.join(live_dir, ".alert_embeddings.json")),
        retention_days=retention, set_option=boxconfig.set_option, read_settings=boxconfig.load_box_settings,
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
