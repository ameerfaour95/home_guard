"""Assistant v3: understand -> act -> write -> supervise, one owner message at a time (report §8).

    message ─▶ [UNDERSTAND] cheap model, acts JSON with quoted slots (acts.py, prompts.UNDERSTAND)
            ─▶ [ACT] deterministic handlers for state (handlers.py) + the obvious first look (handlers.prefetch)
            ─▶ [WRITE] persona + examples; may call read tools within a budget (skills.py, prompts.WRITER)
            ─▶ [SUPERVISE] code checks + a small critic -> one rewrite -> Hebrew template (supervisor.py)
            ─▶ Telegram, the turn saved with its acts, receipts and supervisor trace

It is a subclass of the v2 agent so everything around the message (alerts noted into the chat, the 🏷️ tag path,
Undo, the memory buttons, proposals) is the same code the box runs; only ``_handle`` - the doom-loop chain of
regex short-circuits, two models and six post-hoc guards - is replaced. Selected by box.yaml ``assistant: v3``.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..brain import known_memory as km
from ..brain.agent import AgentReply, OwnerAgentV2, _object, _tag_undo_rows, _undo_token
from ..brain.memory import ChatState
from ..brain.receipts import DONE, REQUESTED, Receipt
from ..brain.registry import current_camera, resolve_camera
from ..brain.tools import Services, ToolContext, known_rows
from . import context as cx
from . import handlers as hd
from . import prompts
from .acts import SCHEMA, Act, Understanding, quoted, validate
from .llm import parse_json
from .memory_view import MemoryView
from .skills import CHAT_BUDGET, INVESTIGATE_BUDGET, Toolbox, loop
from .supervisor import (CRITIC_SCHEMA, code_checks, critic_messages, log_intervention, sanitize, times_in)

log = logging.getLogger("box.assistant_v3")

ACK = ("👍",)
GREETING = ("היי 👋", "היי, הכול שקט פה.", "היי 👋 הכול בסדר.")
ESCALATIONS_PER_DAY = 3
COMPLAINT_HINTS = {
    "repetition": "הוא אומר שחזרת על עצמך: \"צודק, לא אחזור על זה.\" או דומה, וזהו.",
    "irrelevant": "הוא אומר שהזכרת משהו לא קשור: \"צודק, זה לא קשור\" + סליחה קצרה. אל תסביר אותו ואל תזכיר אותו שוב.",
    "unwanted_question": "השאלה הקודמת שלך הייתה מיותרת: תגיד את זה במילים פשוטות.",
    "wrong_fact": "טעית בעובדה: תגיד מה היה לא נכון ומה נכון עכשיו.",
    "ignored_memory": "זה כבר היה אצלך: \"צודק\" + מה פספסת במשפט אחד + התיקון.",
    "too_many_alerts": "הוא מתלונן על הודעות מיותרות: \"צודק\" + על מה שתקין לא אמורה להגיע אליו הודעה.",
    "no_explanation": "הוא רוצה הסבר: תן אותו עכשיו מהראיות (מה רואים ואיפה).",
    "bad_wording": "הניסוח שלך היה גרוע: תגיד בפשטות מה התכוונת.",
    "other": "\"צודק\"/\"סליחה\" + מה היה לא בסדר במילים פשוטות (בלי לצטט את ההודעה הקודמת שלך).",
}
FIX_TEXT = {
    "english": "יש מילים באנגלית; כתוב הכול בעברית.",
    "internal_id": "יש מזהה פנימי; כתוב את שם המצלמה או את השעה במקום.",
    "closing_offer": "בלי משפט סיום/הצעה.",
    "generic_question": "בלי שאלה כללית או הצעה; תגיד מה עשית או מה ראית.",
    "empty_empathy": "בלי 'אני מבין'; תגיד מה היה לא בסדר ומה עשית.",
    "repetition": "חזרת על משפט ששלחת כבר; תגיד רק מה שחדש.",
    "memory_recital": "הזכרת משהו מהזיכרון שלא קשור למה שהוא אמר; תוריד את זה.",
    "invented_time": "כתבת שעה שאין לה מקור; תוריד אותה.",
    "unbacked_claim": "כתבת שעשית/שמרת/שלחת משהו שלא נעשה בתור הזה; אל תטען את זה.",
    "question": "אל תשאל שאלה.",
    "too_long": "ארוך מדי; משפט או שניים.",
    "jargon": "בלי ז'רגון ('התרעה צפויה', 'תיוג'); מילים פשוטות.",
}
NO_REPAIR = ("repetition", "too_many_alerts", "bad_wording", "no_explanation")
REPAIRABLE = ("place_fact", "person_mark", "activity_explain", "camera_fact")
_INVESTIGATE = re.compile(r"מה קרה|תבדוק|בדוק|מה היה|במשך|הלילה|אתמול|מהבוקר|כל היום|היום")


class AssistantV3(OwnerAgentV2):
    version = 3

    def __init__(self, model: Any, registry: Any, memory: Any, book: Any, services: Services,
                 understand_model: Any = None, critic_model: Any = None, escalation_model: Any = None,
                 retention_days: float = 14.0, now: Callable[[], float] = time.time, critic: bool = True) -> None:
        super().__init__(model, registry, memory, book, services, fast_model=None, retention_days=retention_days,
                         now=now)
        self.writer = model
        self.understander = understand_model or model
        self.critic = critic_model if critic else None
        self.escalation = escalation_model

    # ------------------------------------------------------------------------------------------------------------
    def _handle(self, text: str, chat_id: str, who: Dict[str, Any], alert: Optional[Dict[str, Any]],
                threaded: bool, choice: Optional[Tuple[str, int]] = None) -> Optional[AgentReply]:
        started = time.monotonic()
        now = float(self._now())
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
        lang = state.language_for(speaker, text, default=str(settings.get("owner_language") or "he"))
        snapshot = self.registry.snapshot()
        ctx = ToolContext(turn_id=f"{chat_id}:{int(now * 1000)}", chat_id=chat_id, speaker=who, text=text, lang=lang,
                          mode=getattr(snapshot, "mode", "guard"), snapshot=snapshot, state=state,
                          services=self.services, book=self.book, threaded=threaded)
        if alert and alert.get("alert_id"):
            ctx.alert_event = alert
            ctx.alert_handle = state.add_handle("event", str(alert["alert_id"]), str(alert.get("camera") or ""),
                                                float(alert.get("ts") or 0), str(alert.get("summary") or ""))
            state.set_topic_event(ctx.alert_handle, now)
        cams = cx.Cameras.of(snapshot, lang)
        mem = MemoryView(self.services, state, snapshot, lang, now)
        called: List[str] = []
        usage: Dict[str, List[int]] = {}
        trace: List[str] = []

        def resolve(words: str) -> str:
            key = str(words or "").strip()
            if key.casefold() in cams.keys:
                return cams.keys[key.casefold()]
            found = resolve_camera(snapshot, key).camera if key else None
            return found or ""

        box = Toolbox(ctx, called, resolve)
        pending = state.pending if isinstance(state.pending, dict) else None
        state.pending = None
        und = self._pending_answer(pending, text, choice, now)
        answered = und is not None
        if und is None:
            und = self._understand(text, ctx, cams, mem, now, usage, trace)
            self._repair_earlier(und, ctx, cams, mem, now, usage, trace)
        turn = hd.Turn(ctx=ctx, mem=mem, cams=cams, text=text, now=now, alert=alert, und=und, called=called, box=box)

        # ---- ACT ----
        for act in und.acts:
            fn = hd.HANDLERS.get(act.act)
            if fn is None:
                continue
            try:
                if answered and act.act == "person_mark":
                    fn(turn, act, until_words=text)
                else:
                    fn(turn, act)
            except Exception as exc:  # noqa: BLE001 - one handler's bug never ends the turn
                log.warning("v3 handler %s failed: %s", act.act, exc, exc_info=True)
                trace.append(f"handler {act.act} failed: {exc}")
        if und.has("question_history", "command") and not (alert or threaded):
            story = self._day_story(ctx, text, snapshot, now, usage)
            if story:
                turn.plan.final = story
            elif story == "":
                turn.plan.done.append("נשלח הסרטון שביקש מהסיפור של היום.")
        try:
            hd.prefetch(turn)
        except Exception as exc:  # noqa: BLE001
            log.warning("v3 prefetch failed: %s", exc)
        turn.plan.evidence.extend(box.evidence)

        # ---- WRITE + SUPERVISE ----
        reply = self._reply(turn, cams, mem, usage, trace)

        # ---- the turn ----
        buttons: Tuple[str, ...] = ()
        token = ""
        if turn.plan.ask:
            buttons = tuple(turn.plan.ask["choices"])
            token = uuid.uuid4().hex[:8]
            state.pending = {"question": turn.plan.ask["question"], "choices": list(buttons), "ts": now,
                             "kind": turn.plan.ask["kind"], "args": turn.plan.ask.get("args") or {}, "token": token,
                             "request": text, "speaker": speaker}
        undo = _undo_token(ctx)
        rows = tuple(known_rows(ctx.receipts, lang)) + tuple(turn.plan.rows) + _tag_undo_rows(ctx, lang)
        try:
            state.add_turn(speaker, text, reply, ctx.shown, [km.receipt_note(r, snapshot) for r in ctx.receipts],
                           now, notes=ctx.vision_notes)
            state.turns[-1]["v3"] = {"acts": [a.brief() for a in und.acts], "emotion": und.emotion,
                                     "supervisor": trace[-8:], "ms": int((time.monotonic() - started) * 1000),
                                     "usage": {k: v for k, v in usage.items()}}
            self.memory.save(chat_id, state)
        except Exception as exc:  # noqa: BLE001
            log.warning("v3: the conversation was not saved: %s", exc)
        log.info("v3 turn %s acts=%s tools=%s %.1fs trace=%s", ctx.turn_id, und.kinds(), called,
                 time.monotonic() - started, trace)
        after = list(ctx.after_reply) + [lambda: self._fold_summary(chat_id)]
        return AgentReply(text=reply, buttons=buttons, question_token=token if buttons else "", after=tuple(after),
                          lang=lang, receipts=tuple(ctx.receipts), tier="v3", guard_hits=len(trace),
                          usage={k: (v[0], v[1]) for k, v in usage.items()}, tools_called=tuple(called),
                          answer=reply, undo_token=undo, rows=rows)

    # ------------------------------------------------------------------------------------------------------------
    def _pending_answer(self, pending: Optional[Dict[str, Any]], text: str, choice: Optional[Tuple[str, int]],
                        now: float) -> Optional[Understanding]:
        """The answer to v3's own question ("עד איזו שעה?" [17:00] / typed "עד 6"): completes the act in code, no
        understanding call. None when the message is not an answer (he moved on: the question is dropped)."""
        if not pending or pending.get("kind") not in ("v3_mark", "v3_pause"):
            return None
        from .timeparse import parse_until  # noqa: PLC0415

        if parse_until(text, now) is None and not (choice is not None and pending.get("kind") == "v3_pause"):
            return None
        args = pending.get("args") or {}
        if pending["kind"] == "v3_mark":
            act = Act(act="person_mark", quote=str(args.get("quote") or ""), subject=str(args.get("who") or ""),
                      camera="house" if args.get("house") else str(args.get("camera") or ""),
                      event=str(args.get("handle") or ""), until_quote=text)
        else:
            words = {"שעה": "לשעה", "עד הערב": "עד 20:00", "עד מחר בבוקר": "עד 07:00"}.get(text, text)
            act = Act(act="command", command="pause", quote=text, until_quote=words,
                      camera=str(args.get("camera") or ""))
        return Understanding(acts=[act])

    def _repair_earlier(self, und: Understanding, ctx: ToolContext, cams: cx.Cameras, mem: MemoryView, now: float,
                        usage: Dict[str, List[int]], trace: List[str]) -> None:
        """A complaint or "what do you do with that?" right after a statement the assistant mishandled (it asked a
        wrong question or saved nothing, 2026-10-10 "זה הבית של השכן" -> "עד מתי?" -> "מה קשר?"): the owner's
        last statements of the live window that left no receipt are read again, and what they should have saved
        is done now. Only memory acts are taken from them."""
        if not und.has("complaint", "question_meta", "unclear"):
            # an "earlier" act re-does a mishandled statement; only a complaint or "what do you do with it" asks that
            und.acts = [a for a in und.acts if not (a.earlier and a.act in REPAIRABLE
                                                    and not quoted(a.quote, ctx.text))] or und.acts
        if any(a.act == "complaint" and a.issue == "irrelevant" for a in und.acts):
            # "מה הקשר העובדים?": the people he names are what he complains about, never a new memory of them
            und.acts = [a for a in und.acts if a.act not in ("person_mark", "activity_explain")]
        if not und.has("complaint", "unclear") or und.has(*REPAIRABLE):
            return
        if any(a.act == "complaint" and a.issue in NO_REPAIR for a in und.acts):
            return                                     # "why do you repeat", "too many messages": nothing to redo
        turns = [t for t in ctx.state.turns if isinstance(t, dict) and t.get("kind") not in ("alert", "tag")
                 and now - float(t.get("ts") or 0) <= cx.WINDOW_SEC and str(t.get("text") or "").strip()]
        for old in reversed(turns[-3:]):
            if old.get("receipts"):
                return                                 # the newest statement was handled: nothing to repair
            if "complaint" in [a.get("act") for a in (old.get("v3") or {}).get("acts", [])]:
                continue
            handles = [h for h in old.get("handles") or [] if (ctx.state.resolve(h) or {}).get("kind") == "event"]
            again = self._understand(str(old["text"]), ctx, cams, mem, now, usage, trace,
                                     reply_to=handles[0] if handles else "",
                                     note=f"(הודעה קודמת שלו מ-{cx.hhmm(old.get('ts'))} שלא טופלה כמו שצריך)")
            keep = [a for a in again.acts if a.act in REPAIRABLE]
            if keep:
                for a in keep:
                    a.earlier = True
                    a.repaired = True               # a repair never asks a question
                und.acts.extend(keep)
                trace.append(f"repaired earlier {cx.hhmm(old.get('ts'))}: {[a.act for a in keep]}")
                return

    def _understand(self, text: str, ctx: ToolContext, cams: cx.Cameras, mem: MemoryView, now: float,
                    usage: Dict[str, List[int]], trace: List[str], reply_to: Optional[str] = None,
                    note: str = "") -> Understanding:
        state = ctx.state
        relevant = mem.relevant(text, [c for c in [ctx.alert_event and current_camera(ctx.snapshot, str(
            ctx.alert_event.get("camera") or ""))] if c], limit=4)
        body = cx.block(ctx.snapshot, cams, state, getattr(self.services, "events", None), now,
                        memory_lines=[r.line(ctx.snapshot, ctx.lang, now) for r in relevant],
                        message=text + ("\n" + note if note else ""),
                        reply_to=(ctx.alert_handle or "") if reply_to is None else reply_to, for_understanding=True,
                        open_question=str((state.pending or {}).get("question") or ""))
        messages = [{"role": "system", "content": prompts.UNDERSTAND}, {"role": "user", "content": body}]
        raw: Optional[Dict[str, Any]] = None
        for attempt in range(2):
            msg = self.understander.chat(messages, None, json_schema=SCHEMA, max_tokens=700)
            spent = usage.setdefault("understand", [0, 0])
            spent[0] += int(msg.usage[0] or 0)
            spent[1] += int(msg.usage[1] or 0)
            raw = parse_json(msg.content)
            if raw is not None:
                break
            trace.append(f"understand retry: {msg.error or 'no JSON'}")
        handles = list((state.handles or {}).keys())
        und = validate(raw or {}, text, cx.earlier_owner_words(state, now, text), cams.keys, handles)
        if raw is None:
            und.error = "no understanding"
        for a in und.acts:
            if a.dropped:
                trace.append(f"dropped {a.act}.{','.join(a.dropped)}")
        return und

    # ------------------------------------------------------------------------------------------------------------
    def _reply(self, turn: hd.Turn, cams: cx.Cameras, mem: MemoryView, usage: Dict[str, List[int]],
               trace: List[str]) -> str:
        und, plan, ctx = turn.und, turn.plan, turn.ctx
        if plan.final is not None:
            return sanitize(plan.final, cams.ids, ctx.lang)
        kinds = set(und.kinds())
        if kinds <= {"ack", "greeting"}:
            return random.choice(GREETING if "greeting" in kinds else ACK)
        last = cx.last_replies(ctx.state)
        cameras_in_play = [c for c in [turn.camera_of(a) for a in und.acts] if c]
        relevant = mem.relevant(turn.text, cameras_in_play, limit=3) if not und.has("question_memory") else \
            mem.relevant(turn.text, [], limit=6, include_types=("person_mark", "activity_rule", "place_fact",
                                                                  "camera_fact"))
        if und.has("question_memory"):
            relevant = [r for r in mem.records() if r.type != "preference"][:8]
        memory_lines = [r.line(ctx.snapshot, ctx.lang, turn.now) for r in relevant]
        body = cx.block(ctx.snapshot, cams, ctx.state, getattr(self.services, "events", None), turn.now,
                        memory_lines=memory_lines, prefs=mem.prefs(), message=turn.text,
                        reply_to=ctx.alert_handle or "")
        tools_ok = bool(kinds & {"question_live", "question_history", "question_meta", "complaint", "chit_chat",
                                 "unclear", "question_memory"}) and not plan.ask
        budget = INVESTIGATE_BUDGET if (und.has("question_history") and _INVESTIGATE.search(turn.text)) else CHAT_BUDGET
        task = self._task(turn, last)
        system = prompts.WRITER + ("\n\n" + prompts.TOOLS_NOTE.format(budget=budget[0]) if tools_ok else "")
        model, tier = self.writer, "write"
        if self._escalate(turn, trace):
            model, tier = self.escalation, "escalate"
        draft = self._write(model, system, body, task, tools_ok, budget, turn, usage, tier)
        evidence = "\n".join(plan.done + plan.evidence + turn.box.evidence)
        allowed = times_in(turn.text, body, evidence, *cx.earlier_owner_words(ctx.state, turn.now))
        claim_receipts = list(ctx.receipts) + [Receipt(id="v3", turn="v3", tool=t, status=DONE) for t in
                                               _backing(ctx.receipts, plan)]
        subjects = [r.text for r in mem.records() if r.type in ("person_mark", "activity_rule") and r not in relevant]

        def check(text: str):
            return code_checks(text, lang=ctx.lang, last_replies=last, allowed_times=allowed, receipts=claim_receipts,
                               memory_subjects=subjects, owner_text=turn.text + " " + " ".join(
                                   cx.earlier_owner_words(ctx.state, turn.now)[-3:]),
                               may_ask=bool(plan.ask), act_kinds=list(kinds), strict_memory=not (kinds & {
                                   "question_live", "question_history", "question_memory"}), evidence_text=evidence + " ".join(memory_lines),
                               long_ok=plan.long_ok or (und.has("question_live") and len(turn.box.evidence) > 2)
                               or und.has("question_memory"))

        draft = sanitize(draft, cams.ids, ctx.lang)
        verdict = check(draft)
        fix = ""
        if not verdict.ok:
            fix = "הבעיה בטיוטה: " + " ".join(FIX_TEXT.get(c.name, c.name) + (f" ({c.detail})" if c.detail else "")
                                              for c in verdict.failed)
            log_intervention(trace, f"code: {verdict.reason()}")
        elif self.critic is not None and draft:
            fix = self._critique(turn, draft, evidence, last, usage, trace)
        if fix:
            second = sanitize(self._write(model, system, body, task + f"\n\nתקן את הטיוטה הקודמת: \"{draft}\"\n{fix}",
                                          False, budget, turn, usage, "rewrite"), cams.ids, ctx.lang)
            v2 = check(second)
            if v2.ok:
                draft = second
            else:
                log_intervention(trace, f"rewrite failed: {v2.reason()}")
                if verdict.ok:
                    pass                                  # the critic's wish failed; the first draft passed code
                else:
                    draft = self._template(turn)
                    log_intervention(trace, "template sent")
        if plan.ask and "?" not in draft:
            draft = (draft.rstrip() + " " + plan.ask["question"]).strip()
        return draft or self._template(turn)

    def _task(self, turn: hd.Turn, last: Sequence[str]) -> str:
        und, plan = turn.und, turn.plan
        lines = ["מה הבנתי מההודעה (לשימוש פנימי): " + json.dumps(public_acts(und, turn.cams), ensure_ascii=False)
                 + f" · רגש: {und.emotion}"]
        handle, entry = turn.event_of(None) if (turn.ctx.alert_handle or turn.state.topic_event(turn.now)) else ("", {})
        if entry:
            lines.append(f"ההתראה שמדברים עליה: {cx.hhmm(entry.get('ts'))} ב{turn.cams.name(turn.camera_now(str(entry.get('camera') or '')))}"
                         f" — {cx.clip(entry.get('observation') or entry.get('summary'), 200)}")
        if plan.done:
            lines.append("מה עשיתי בתור הזה (עובדות שאפשר לומר):\n" + "\n".join(f"- {d}" for d in plan.done))
        else:
            lines.append("בתור הזה לא שמרתי ולא שיניתי כלום.")
        ev = plan.evidence or []
        if ev:
            lines.append("ראיות שבדקתי עכשיו:\n" + "\n".join(f"- {e}" for e in ev))
        if plan.ask:
            lines.append(f"ASK: סיים בשאלה האחת הזאת בדיוק: {plan.ask['question']} (יש כפתורים: "
                         f"{' / '.join(plan.ask['choices'])})")
        if last:
            lines.append("התשובות האחרונות שלך (אל תחזור עליהן):\n" + "\n".join(f"- {cx.clip(x, 200)}" for x in last))
        for act in und.acts:
            if act.act == "complaint":
                lines.append("הוא מתלונן. " + COMPLAINT_HINTS.get(act.issue or "other", COMPLAINT_HINTS["other"])
                             + (" מה שתיקנת בתור הזה מופיע למעלה תחת 'מה עשיתי'; אל תמציא פעולה אחרת."
                                if plan.done else " אל תטען שתיקנת, מחקת או שינית משהו."))
                break
        lines.append("כתוב עכשיו רק את ההודעה לבעל הבית, בעברית, משפט או שניים.")
        return "\n\n".join(lines)

    def _write(self, model: Any, system: str, body: str, task: str, tools_ok: bool, budget: Tuple[int, float],
               turn: hd.Turn, usage: Dict[str, List[int]], tier: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": body + "\n\n" + task}]
        spent = usage.setdefault(tier, [0, 0])
        if tools_ok:
            return loop(model, messages, turn.box, budget, spent)
        msg = model.chat(messages, None, max_tokens=400)
        spent[0] += int(msg.usage[0] or 0)
        spent[1] += int(msg.usage[1] or 0)
        if msg.error:
            log.warning("v3 writer failed: %s", msg.error)
        return (msg.content or "").strip()

    def _critique(self, turn: hd.Turn, draft: str, evidence: str, last: Sequence[str], usage: Dict[str, List[int]],
                  trace: List[str]) -> str:
        msg = self.critic.chat(critic_messages(prompts.CRITIC, turn.text, public_acts(turn.und, turn.cams), last,
                                               evidence, draft), None, json_schema=CRITIC_SCHEMA, max_tokens=200)
        spent = usage.setdefault("critic", [0, 0])
        spent[0] += int(msg.usage[0] or 0)
        spent[1] += int(msg.usage[1] or 0)
        got = parse_json(msg.content) or {}
        if got.get("verdict") == "rewrite" and str(got.get("fix") or "").strip():
            log_intervention(trace, f"critic: {got.get('problem')} -> {got.get('fix')}")
            note = "הערת הבודק: " + str(got.get("problem") or "") + ". " + str(got.get("fix"))
            return sanitize(note, list(turn.cams.ids), turn.ctx.lang)
        return ""

    def _escalate(self, turn: hd.Turn, trace: List[str]) -> bool:
        """The bigger model, by code only: an angry complaint right after another complaint (the repair failed), at
        most ESCALATIONS_PER_DAY a day."""
        if self.escalation is None or not turn.und.has("complaint") or turn.und.emotion != "angry":
            return False
        prev = [t for t in turn.state.turns if isinstance(t, dict) and t.get("kind") != "alert"][-1:]
        if not prev or "complaint" not in [a.get("act") for a in (prev[0].get("v3") or {}).get("acts", [])]:
            return False
        day = dt.datetime.fromtimestamp(turn.now).strftime("%Y-%m-%d")
        used = str(turn.state.prefs.get("v3_escalations") or "")
        n = int(used.split(":")[1]) if used.startswith(day + ":") else 0
        if n >= ESCALATIONS_PER_DAY:
            return False
        turn.state.prefs["v3_escalations"] = f"{day}:{n + 1}"
        log_intervention(trace, "escalated to the big model")
        return True

    def _template(self, turn: hd.Turn) -> str:
        """The safe Hebrew reply built from what the turn did and saw (never English, never empty)."""
        plan = turn.plan
        if plan.done:
            return " ".join(re.sub(r"\s*\([^)]*\)", "", d) for d in plan.done[:2])
        if plan.evidence:
            first = re.sub(r"^מבט חי עכשיו ב", "", plan.evidence[0])
            if len(re.findall(r"[A-Za-z]{3,}", first)) > 2:
                return "בדקתי עכשיו, התמונה למעלה."
            return "בדקתי: " + first
        if turn.und.has("complaint"):
            return "צודק, סליחה. מה שכתבתי קודם לא היה במקום."
        return "קיבלתי."

    # ------------------------------------------------------------------------------------------------------------
    def _fold_summary(self, chat_id: str) -> None:
        """After the reply: when more than 16 turns of today are older than the live window, fold them into a
        rolling summary (decisions, corrections, promises) with the cheap model. Never raises."""
        try:
            with self._lock:
                state = self.memory.load(chat_id)
                now = float(self._now())
                upto = float(state.prefs.get("v3_summary_upto") or 0)
                older = [t for t in state.turns if isinstance(t, dict) and t.get("kind") != "alert"
                         and upto < float(t.get("ts") or 0) and now - float(t.get("ts") or 0) > cx.WINDOW_SEC
                         and now - float(t.get("ts") or 0) <= cx.DAY_SEC]
                if len(older) < 16:
                    return
                lines = [f"[{cx.hhmm(t.get('ts'))}] בעל הבית: {cx.clip(t.get('text'), 200)} | אתה: "
                         f"{cx.clip(t.get('reply'), 120)}" for t in older]
                prior = str(state.prefs.get("v3_summary") or "")
            msg = self.understander.chat([
                {"role": "system", "content": "Summarize this Hebrew chat between a homeowner and his security "
                 "assistant in at most 6 short Hebrew lines: what he decided, corrected, asked to remember or to stop, "
                 "and any open promise. No ids. Only facts from the chat."},
                {"role": "user", "content": (f"סיכום קודם:\n{prior}\n\n" if prior else "") + "\n".join(lines)}],
                None, max_tokens=300)
            if msg.error or not (msg.content or "").strip():
                return
            with self._lock:
                state = self.memory.load(chat_id)
                state.prefs["v3_summary"] = msg.content.strip()[:1200]
                state.prefs["v3_summary_upto"] = max(float(t.get("ts") or 0) for t in older)
                self.memory.save(chat_id, state)
        except Exception as exc:  # noqa: BLE001
            log.warning("v3 summary not folded: %s", exc)


def public_acts(und: Understanding, cams: cx.Cameras) -> List[Dict[str, Any]]:
    """The acts as the writer and the critic read them: camera NAMES, no internal ids."""
    out = []
    for a in und.acts:
        d = a.brief()
        d.pop("dropped", None)
        d.pop("repaired", None)
        if d.get("camera"):
            d["camera"] = "כל הבית" if d["camera"] == "house" else cams.name(d["camera"])
        d.pop("event", None)
        out.append(d)
    return out


def _backing(receipts: Sequence[Any], plan: hd.Plan) -> List[str]:
    """Receipts that back a claim word in v3's own terms: any memory write this turn (a mark, a place or camera
    fact, an activity rule, a preference) backs "רשמתי / שמרתי / הבנתי ש... השכן / זוכר"."""
    tools = {r.tool for r in receipts if getattr(r, "status", "") in (DONE, REQUESTED)}
    memory = bool(tools & {"mark_known", "camera_fact", "activity"}) or         any(d.startswith("העדפה קבועה נשמרה") for d in plan.done)
    return ["mark_known", "camera_fact", "house_expect"] if memory else []
