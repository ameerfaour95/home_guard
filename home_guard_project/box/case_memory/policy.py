"""The policy: what a matched case may do to one delivery. Code, not a model.

``apply_case_memory(event, decision) -> (delivery_level, note)``

- ``delivery_level`` is ``alert`` (the box's normal path, untouched), ``quiet`` (a message without sound) or
  ``digest`` (no message; a line in the daily digest). Memory lowers at most one step on the trust ladder.
- A call (``[call_owner]``) and an escalation are never touched, and memory is not even consulted for them.
- The veto (S/E category, a risk flag, serious behaviour, a multi-camera incident at night or away) means memory is
  not consulted: no memory ever cancels a suspicious sign.
- A ``suspicious`` label is never softened; a matching case only adds context ("similar to X, but this alert
  stands").
- Bands: a high score lowers by the ladder; a middle score goes to the judge (same / similar but different /
  unsure; unsure, invalid and timeout = alert); a low score is the normal flow. "Similar but different" goes out
  as a normal alert with context ("looks like the neighbour, but this time went to the door").
- A case in shadow decides and logs "would have silenced", still alerts, and asks the owner.
- ``note`` is ``None`` when memory has nothing to say; otherwise a :class:`CaseNote` with the owner text in
  Hebrew and English, the buttons, the digest line and an audit record for the event's meta.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import texts
from .gates import ENDED, gate_failures, veto
from .judge import SAME, SIMILAR, JudgeVerdict, decide
from .ladder import SHADOW_STAGE, TrustLadder
from .models import ALERT, DIGEST, QUIET, Case, Signature
from .scorer import DEFAULT_WEIGHTS, TIME_SIGMA_MIN, ScoreDetail, rank
from .signature import build_signature
from .store import CaseStore

log = logging.getLogger("box.case_memory")

HIGH, MID, LOW = "high", "mid", "low"


@dataclass(frozen=True)
class CaseMemoryConfig:
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    high: float = 0.85                 # at or above: lower one step without the judge
    mid: float = 0.60                  # at or above: the judge decides; below: normal flow
    high_requires: Tuple[str, ...] = ("path",)   # a high band needs these components; else it is judged
    hour_margin_min: int = 15
    max_path_distance: float = 0.34
    time_sigma_min: float = TIME_SIGMA_MIN
    top_k: int = 3
    few_cases: int = 5                 # with this many live cases or fewer on the camera, the judge sees all gated
    judge_timeout: float = 8.0
    ladder: TrustLadder = TrustLadder()


@dataclass(frozen=True)
class CaseEvent:
    """One event as memory sees it. ``event_id`` is the alert id (the clip stem)."""
    event_id: str
    signature: Signature

    @classmethod
    def build(cls, event_id: str, camera: str, ts: float, observation: Optional[Mapping[str, Any]] = None,
              tracker: Optional[Mapping[str, Any]] = None, situation: Optional[Mapping[str, Any]] = None,
              label: str = "", cameras_in_incident: int = 1, eye_model: str = "",
              prompt_version: str = "", text: str = "") -> "CaseEvent":
        """*text*: the alert's words (why + reason + summary), read for the actions an owner may have explained."""
        return cls(event_id, build_signature(camera, ts, observation, tracker, situation, label,
                                             cameras_in_incident, eye_model, prompt_version, text))

    @classmethod
    def coerce(cls, value: Union["CaseEvent", Mapping[str, Any]], label: str = "") -> "CaseEvent":
        if isinstance(value, CaseEvent):
            return value
        v = dict(value)
        return cls.build(str(v.get("event_id") or v.get("alert_id") or ""), str(v["camera"]), float(v["ts"]),
                         v.get("observation"), v.get("tracker"), v.get("situation"), label or v.get("label", ""),
                         int(v.get("cameras_in_incident") or 1), v.get("eye_model", ""), v.get("prompt_version", ""),
                         str(v.get("text") or ""))


@dataclass(frozen=True)
class CaseNote:
    kind: str                          # shadow / softened / keep_alerting / similar / context
    case_id: str
    text_he: str
    text_en: str
    buttons: Tuple[Dict[str, str], ...] = ()
    digest_he: str = ""
    digest_en: str = ""
    band: str = ""
    score: float = 0.0
    would_level: str = ""              # shadow: what the case would have done
    verdict: Optional[JudgeVerdict] = None

    def text(self, lang: str) -> str:
        return self.text_he if lang == "he" else self.text_en

    def owner_text(self, lang: str) -> str:
        """The line the owner reads under the alert. A shadow note has none: shadow logs "would quiet" and asks
        nothing until its buttons exist (owner rules: no new message types, no question without buttons)."""
        return "" if self.kind == "shadow" else self.text(lang)

    def digest(self, lang: str) -> str:
        return self.digest_he if lang == "he" else self.digest_en

    def record(self) -> Dict[str, Any]:
        """For the event's meta / decision record (``job.alert["case_memory"]``)."""
        return {"kind": self.kind, "case_id": self.case_id, "band": self.band, "score": round(self.score, 4),
                "would_level": self.would_level, "verdict": self.verdict.to_dict() if self.verdict else None,
                "text_en": self.text_en}


@dataclass(frozen=True)
class Assessment:
    """Everything memory looked at for one event (for logs, the eval and tests)."""
    level: str
    note: Optional[CaseNote] = None
    vetoed: Tuple[str, ...] = ()
    gated_out: Dict[str, List[str]] = field(default_factory=dict)
    ranked: Tuple[Tuple[str, float], ...] = ()
    band: str = ""
    verdict: Optional[JudgeVerdict] = None
    matched: str = ""
    shadow: bool = False


def _decision_label(decision: Mapping[str, Any]) -> Tuple[str, str]:
    label = str(decision.get("final_label") or decision.get("label") or "").strip().lower()
    return label, str(decision.get("alert_command") or "")


class CaseMemory:
    """Gates, scorer, bands, judge and ladder over one store. Never raises out of ``apply``."""

    def __init__(self, store: CaseStore, embed: Optional[Callable[[str], Optional[Sequence[float]]]] = None,
                 judge: Optional[Callable[[Dict[str, Any]], Any]] = None,
                 config: CaseMemoryConfig = CaseMemoryConfig(), rng: Optional[random.Random] = None,
                 log_matches: bool = True) -> None:
        self.store = store
        self.embed = embed
        self.judge = judge
        self.config = config
        self.rng = rng
        self.log_matches = log_matches

    # -- the pieces -----------------------------------------------------------------------------------------------
    def _embedding(self, sig: Signature) -> Optional[Sequence[float]]:
        if self.embed is None:
            return None
        try:
            return self.embed(sig.template)
        except Exception as exc:  # noqa: BLE001 - without a vector the text component is left out
            log.warning("Case memory embedding failed: %s", exc)
            return None

    def band_of(self, detail: ScoreDetail, case: Optional[Case] = None) -> str:
        # An explained action's case has no path: its named actions stand in for it.
        requires = ("actions",) if case is not None and case.scope.actions else self.config.high_requires
        if detail.score >= self.config.high and all(k in detail.components for k in requires):
            return HIGH
        if detail.score >= self.config.mid:
            return MID
        return LOW

    def _note_for_match(self, case: Case, sig: Signature, band: str, score: float,
                        verdict: Optional[JudgeVerdict]) -> Tuple[str, CaseNote, bool]:
        """``(level, note, shadow)`` for a case judged to be the same situation."""
        ladder = self.config.ladder
        stage = ladder.stage(case.streak, case.contradictions)
        level = ladder.level(case.streak, case.contradictions, case.effect)
        common = dict(case_id=case.id, band=band, score=score, verdict=verdict)
        if case.effect == ALERT:
            return ALERT, CaseNote("keep_alerting", text_he=texts.keep_alerting_text(case, "he"),
                                   text_en=texts.keep_alerting_text(case, "en"),
                                   buttons=(texts.button("not_them", case.id),), **common), False
        if stage == SHADOW_STAGE:
            would = QUIET
            return ALERT, CaseNote("shadow", text_he=texts.shadow_text(case, "he"), text_en=texts.shadow_text(case, "en"),
                                   buttons=(texts.button("confirm", case.id), texts.button("not_them", case.id),
                                            texts.button("keep_alerting", case.id)),
                                   would_level=would, **common), True
        return level, CaseNote("softened", text_he=texts.softened_text(case, "he"),
                               text_en=texts.softened_text(case, "en"),
                               buttons=(texts.button("not_them", case.id),),
                               digest_he=texts.digest_line(case, sig, "he"), digest_en=texts.digest_line(case, sig, "en"),
                               would_level=level, **common), False

    def _fits_but_its_end(self, case: Case, sig: Signature) -> bool:
        """The same kind of event after the case's end, at any hour: what it did fits, only the end (and the
        hour) do not."""
        whole = replace(case, scope=replace(case.scope, until=None, hours=("00:00", "00:00")))
        return not gate_failures(whole, sig, self.config.hour_margin_min, self.config.max_path_distance)

    # -- the decision ---------------------------------------------------------------------------------------------
    def assess(self, event: Union[CaseEvent, Mapping[str, Any]], decision: Mapping[str, Any],
               use_judge: bool = True) -> Assessment:
        label, command = _decision_label(decision)
        ev = CaseEvent.coerce(event, label)
        sig = ev.signature
        if label == "escalation" or command == "[call_owner]":
            return Assessment(ALERT, vetoed=("escalation or call",))
        if decision.get("serious_behaviour") is True and not sig.serious_behaviour:
            sig = Signature(**{**sig.__dict__, "serious_behaviour": True})
        reasons = veto(sig, label, command)
        cases = self.store.live_cases(sig.camera)
        explained_only = False
        if reasons:
            # Only a case of an explained action (activity_memory) may still look: the flags and S categories that
            # ARE the explained action, and a red its context look already lowered, are what the owner explained.
            lowered = decision.get("context_lowered") is True
            cases = [c for c in cases if c.scope.actions and label in ("normal", "suspicious")
                     and not veto(sig, label, command, explained=c.scope.actions, context_lowered=lowered)]
            if not cases:
                return Assessment(ALERT, vetoed=tuple(reasons))
            explained_only = True
        elif label not in ("normal", "suspicious"):
            return Assessment(ALERT, vetoed=(f"label {label or 'missing'}",))
        else:
            cases = [c for c in cases if not c.scope.actions
                     or not veto(sig, label, command, explained=c.scope.actions)]
        failures = {c.id: gate_failures(c, sig, self.config.hour_margin_min, self.config.max_path_distance)
                    for c in cases}
        gated = [c for c in cases if not failures[c.id]]
        gated_out = {k: v for k, v in failures.items() if v}
        if self.log_matches:
            for c in cases:
                if ENDED in failures[c.id] and self._fits_but_its_end(c, sig):
                    self.store.log_seen_after_end(c.id, ev.event_id, sig.ts)   # the keeper asks once
        if not gated:
            return Assessment(ALERT, gated_out=gated_out)
        ranked = rank(gated, sig, self._embedding(sig), self.config.weights, self.config.time_sigma_min)
        summary = tuple((c.id, round(d.score, 4)) for c, d in ranked)
        best_case, best = ranked[0]
        band = self.band_of(best, best_case)
        base = dict(gated_out=gated_out, ranked=summary, band=band)

        if label == "suspicious" and not best_case.scope.actions:
            if band == LOW:
                return Assessment(ALERT, **base)
            note = CaseNote("context", best_case.id, texts.context_text(best_case, "he"),
                            texts.context_text(best_case, "en"), band=band, score=best.score)
            return Assessment(ALERT, note=note, **base)
        if label == "suspicious":
            # Only an explained action may lower a suspicious: the judge never sees the other cases for it.
            ranked = [(c, d) for c, d in ranked if c.scope.actions]

        verdict: Optional[JudgeVerdict] = None
        matched: Optional[Tuple[Case, ScoreDetail]] = None
        if band == HIGH:
            matched = (best_case, best)
        elif band == MID and use_judge:
            pool = ranked if len(cases) <= self.config.few_cases else ranked[:self.config.top_k]
            verdict, _ = decide(self.judge, sig, pool, self.config.judge_timeout, self.rng)
            by_id = {c.id: (c, d) for c, d in pool}
            if verdict.verdict == SAME and verdict.valid:
                matched = by_id.get(verdict.case_id)
            elif verdict.verdict == SIMILAR and verdict.valid and verdict.case_id in by_id:
                case, detail = by_id[verdict.case_id]
                note = CaseNote("similar", case.id, texts.similar_text(case, sig, verdict.mismatched_fields, "he"),
                                texts.similar_text(case, sig, verdict.mismatched_fields, "en"),
                                buttons=(texts.button("fine_too", case.id),), band=band, score=detail.score,
                                verdict=verdict)
                return Assessment(ALERT, note=note, verdict=verdict, **base)
        if matched is None:
            return Assessment(ALERT, verdict=verdict, **base)

        case, detail = matched
        level, note, shadow = self._note_for_match(case, sig, band, detail.score, verdict)
        if (label == "suspicious" or explained_only) and level == DIGEST:
            level = QUIET                     # an explained action's suspicious: at most a quiet message
            note = replace(note, would_level=QUIET)
        if self.log_matches:
            self.store.log_match(case.id, ev.event_id, note.would_level or level, band, detail.score, shadow,
                                 verdict.verdict if verdict else "")
        if shadow:
            log.info("[%s] case memory, shadow: would quiet (case %s, score %.2f): %s", sig.camera, case.id,
                     detail.score, texts.case_title(case, "en"))
        return Assessment(level, note=note, verdict=verdict, matched=case.id, shadow=shadow, **base)

    def apply(self, event: Union[CaseEvent, Mapping[str, Any]], decision: Mapping[str, Any],
              shadow_only: bool = False) -> Tuple[str, Optional[CaseNote]]:
        """``(delivery_level, note)``. Any failure inside is logged and means ``alert`` with no note.
        *shadow_only*: an event the box already kept quiet for another reason (a live explained action, a known
        mark): matches are logged for learning, no judge is asked, and the level is always ``alert``."""
        label, command = _decision_label(decision)
        if label == "escalation" or command == "[call_owner]":
            return ALERT, None
        try:
            result = self.assess(event, decision, use_judge=not shadow_only)
            if shadow_only:
                return ALERT, result.note
        except Exception as exc:  # noqa: BLE001 - memory must never stop or soften an alert by failing
            log.warning("Case memory failed; alerting as usual: %s", exc)
            return ALERT, None
        if result.level not in (ALERT, QUIET, DIGEST):
            return ALERT, result.note
        return result.level, result.note


# -- the module-level entry point the guard loop calls ------------------------------------------------------------

_MEMORY: Optional[CaseMemory] = None


def configure(memory: Optional[CaseMemory]) -> None:
    """Install the box's memory (``None`` switches it off: every call returns ``("alert", None)``)."""
    global _MEMORY
    _MEMORY = memory


def current() -> Optional[CaseMemory]:
    return _MEMORY


def apply_case_memory(event: Union[CaseEvent, Mapping[str, Any]], decision: Mapping[str, Any],
                      memory: Optional[CaseMemory] = None, shadow_only: bool = False) -> Tuple[str, Optional[CaseNote]]:
    """``(delivery_level, note)`` for one event the box is about to deliver. Without a configured memory,
    ``("alert", None)``: exactly today's behaviour. *shadow_only*: see :meth:`CaseMemory.apply`."""
    mem = memory or _MEMORY
    if mem is None:
        return ALERT, None
    return mem.apply(event, decision, shadow_only=True) if shadow_only else mem.apply(event, decision)


def make_default(store_path: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
                 embed_cache_path: Optional[str] = None, judge_model: str = "gpt-6-luna",
                 judge_provider: str = "openai", config: CaseMemoryConfig = CaseMemoryConfig()) -> CaseMemory:
    """The memory a real box runs: the journal in the state folder (``cases.jsonl``), the existing
    embedder with the alerts' cache (``.alert_embeddings.json``, the assistant's) and the gpt-6-luna judge. Without an
    OpenAI key there is no embedder and no judge: high-band matches still work, the middle band alerts."""
    import os  # noqa: PLC0415

    from .. import paths  # noqa: PLC0415
    from ..embeddings import make_embedder  # noqa: PLC0415
    from .judge import OpenAICompatibleJudge  # noqa: PLC0415
    from .store import default_path  # noqa: PLC0415

    env = dict(os.environ if env is None else env)
    path = store_path or default_path()
    store = CaseStore.at(path, now=time.time, ladder=config.ladder)
    if embed_cache_path is None:
        # Next to the assistant's state: <live_dir> for a journal at <live_dir>/.registry/cases.jsonl.
        assistant_dir = paths.assistant_dir() if store_path is None else os.path.dirname(os.path.dirname(path))
        embed_cache_path = os.path.join(assistant_dir, ".alert_embeddings.json")
    embedder = make_embedder(env, embed_cache_path) if embed_cache_path else None
    judge = None
    try:
        from .. import providers  # noqa: PLC0415

        providers.resolve(judge_provider, env, judge_model)
        judge = OpenAICompatibleJudge(judge_model, judge_provider, env, timeout=config.judge_timeout)
    except Exception as exc:  # noqa: BLE001 - no key: the middle band alerts
        log.info("Case memory judge not available (%s); the middle band will alert", exc)
    return CaseMemory(store, embedder.embed_one if embedder else None, judge, config)
