"""The mid-band judge: a cheap text model decides "same case", "similar but different" or "unsure".

Text only, no pictures: the event's structured fields and the candidate cases (in random order, because LLM judges
favour positions). It must name the case and the fields that matched and did not. Code then checks the answer:
an unknown case id, a field that is not in the list, or a matched field the case does not have (matching on
"appearance" for a case with no appearance words) makes it invalid. ``unsure``, invalid, an error and a timeout all mean
"alert as usual". The owner's note is passed as data, never as instructions.

The judge is any callable ``judge(request) -> str | dict``; ``OpenAICompatibleJudge`` is the default for a
real box (gpt-6-luna through providers.py). Tests inject fakes; nothing here touches the network on import.
"""
from __future__ import annotations

import json
import logging
import random
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ...prompts import render
from .models import Case, Signature
from .scorer import ScoreDetail

log = logging.getLogger("box.case_memory.judge")

SAME, SIMILAR, UNSURE = "same", "similar_but_different", "unsure"
VERDICTS = (SAME, SIMILAR, UNSURE)
FIELDS = ("camera", "hour", "weekday", "house_state", "path", "entry_edge", "exit_edge", "movement", "category",
          "people", "vehicles", "dwell", "appearance")
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_TIMEOUT = 8.0
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SYSTEM_PROMPT = render("case_judge.system_prompt", fields=", ".join(FIELDS))

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "case_id": {"type": "string"},
        "matched_fields": {"type": "array", "items": {"type": "string", "enum": list(FIELDS)}},
        "mismatched_fields": {"type": "array", "items": {"type": "string", "enum": list(FIELDS)}},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "case_id", "matched_fields", "mismatched_fields", "reason"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "case_match", "strict": True,
                                                          "schema": RESPONSE_SCHEMA}}


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str
    case_id: str = ""
    matched_fields: Tuple[str, ...] = ()
    mismatched_fields: Tuple[str, ...] = ()
    reason: str = ""
    valid: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "case_id": self.case_id, "matched_fields": list(self.matched_fields),
                "mismatched_fields": list(self.mismatched_fields), "reason": self.reason, "valid": self.valid,
                "error": self.error}


def unsure(error: str) -> JudgeVerdict:
    return JudgeVerdict(UNSURE, valid=False, error=error)


def event_fields(sig: Signature) -> Dict[str, Any]:
    """The structured view both the event and a case's example are shown in."""
    return {
        "camera": sig.camera, "hour": f"{sig.minute // 60:02d}:{sig.minute % 60:02d}",
        "weekday": _DAYS[sig.weekday], "house_state": sig.house_state, "phase": sig.phase,
        "path": ">".join(sig.path), "entry_edge": sig.entry_edge, "exit_edge": sig.exit_edge,
        "movement": sig.movement, "category": sig.category, "people": sig.people, "vehicles": sig.vehicles,
        "dwell_s": None if sig.dwell_s is None else round(sig.dwell_s, 1), "appearance": list(sig.appearance),
        "description": sig.template,
    }


def case_fields(case: Case, detail: Optional[ScoreDetail] = None) -> Dict[str, Any]:
    s = case.scope
    closest = next((e for e in case.examples if detail and e.event_id == detail.example_id),
                   case.examples[0] if case.examples else None)
    return {
        "case_id": case.id, "who": case.who, "title": case.title, "owner_note": case.note,
        "scope": {"camera": s.camera, "hours": f"{s.hours[0]}-{s.hours[1]}",
                  "weekdays": [_DAYS[d] for d in s.weekdays], "house_states": list(s.house_states),
                  "path": ">".join(s.path), "entry_edge": s.entry_edge, "exit_edge": s.exit_edge,
                  "people": s.people, "vehicles": s.vehicles, "categories": list(s.categories),
                  "max_dwell_s": s.max_dwell_s},
        "recognise_by": list(case.recognise),
        "closest_example": event_fields(closest.signature) if closest else None,
    }


def available_fields(case: Case) -> Tuple[str, ...]:
    """Fields a verdict may cite for *case*: those the case actually has."""
    out = ["camera", "hour", "weekday", "house_state", "people", "vehicles", "movement", "category"]
    ex = [e.signature for e in case.examples]
    if case.scope.path or any(e.path for e in ex):
        out += ["path", "entry_edge", "exit_edge"]
    if case.scope.max_dwell_s is not None or any(e.dwell_s is not None for e in ex):
        out.append("dwell")
    if case.recognise or any(e.appearance for e in ex):
        out.append("appearance")
    return tuple(out)


def build_request(sig: Signature, candidates: Sequence[Tuple[Case, ScoreDetail]],
                  rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """``{"messages", "response_format", "case_ids", "fields"}``: one chat request, candidates shuffled."""
    order = list(candidates)
    (rng or random.Random()).shuffle(order)
    payload = {"new_event": event_fields(sig), "cases": [case_fields(c, d) for c, d in order]}
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "response_format": RESPONSE_FORMAT,
        "case_ids": [c.id for c, _ in order],
        "fields": {c.id: available_fields(c) for c, _ in order},
    }


def parse_verdict(raw: Any, request: Mapping[str, Any]) -> JudgeVerdict:
    """Check the judge's answer against the request; anything doubtful is an invalid ``unsure``."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return unsure("not JSON")
    if not isinstance(data, dict):
        return unsure("not an object")
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        return unsure(f"unknown verdict {verdict!r}")
    matched, mismatched = data.get("matched_fields"), data.get("mismatched_fields")
    if not isinstance(matched, list) or not isinstance(mismatched, list):
        return unsure("matched_fields and mismatched_fields must be lists")
    matched_t = tuple(str(f) for f in matched)
    mismatched_t = tuple(str(f) for f in mismatched)
    reason = str(data.get("reason") or "")[:300]
    if verdict == UNSURE:
        return JudgeVerdict(UNSURE, str(data.get("case_id") or ""), matched_t, mismatched_t, reason)
    case_id = str(data.get("case_id") or "")
    if case_id not in request.get("case_ids", ()):
        return unsure(f"case {case_id!r} was not offered")
    allowed = set(request.get("fields", {}).get(case_id, FIELDS))
    cited = set(matched_t) | set(mismatched_t)
    if not cited <= set(FIELDS):
        return unsure("cites an unknown field")
    if not set(matched_t) <= allowed:      # a match can only rest on what the case has; a difference can be anything
        return unsure("matches a field the case does not have: " + ",".join(sorted(set(matched_t) - allowed)))
    if set(matched_t) & set(mismatched_t):
        return unsure("a field both matched and mismatched")
    if verdict == SAME and (not matched_t or mismatched_t):
        return unsure("same needs matched fields and no mismatched ones")
    if verdict == SIMILAR and not mismatched_t:
        return unsure("similar_but_different needs the mismatched fields")
    return JudgeVerdict(verdict, case_id, matched_t, mismatched_t, reason)


def run_judge(judge: Callable[[Dict[str, Any]], Any], request: Dict[str, Any],
              timeout: float = DEFAULT_TIMEOUT) -> JudgeVerdict:
    """Call *judge* with a hard timeout. Errors and timeouts are an invalid ``unsure`` (alert as usual)."""
    box: Dict[str, Any] = {}

    def call() -> None:
        try:
            box["raw"] = judge(request)
        except Exception as exc:  # noqa: BLE001 - a failing judge must never silence anything
            box["error"] = exc

    from .. import usage_ledger  # noqa: PLC0415

    worker = threading.Thread(target=usage_ledger.carry(call), name="case-judge", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        log.warning("Case judge timed out after %.1fs", timeout)
        return unsure("timeout")
    if "error" in box:
        log.warning("Case judge failed: %s", box["error"])
        return unsure(f"error: {box['error']}")
    return parse_verdict(box.get("raw"), request)


class OpenAICompatibleJudge:
    """The default judge: one chat call with strict JSON output to a cheap text model (gpt-6-luna on OpenAI;
    any provider in providers.py works). The client is created lazily on the first call."""

    def __init__(self, model: str = DEFAULT_MODEL, provider: str = "openai", env: Optional[Mapping[str, str]] = None,
                 timeout: float = DEFAULT_TIMEOUT, client: Any = None) -> None:
        self.model = model
        self.provider = provider
        self._env = env
        self.timeout = timeout
        self._client = client
        self._extra: Optional[Dict[str, Any]] = None

    def _make_client(self) -> Any:
        import os  # noqa: PLC0415
        import ssl  # noqa: PLC0415

        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        from .. import providers  # noqa: PLC0415

        key, url, extra = providers.resolve(self.provider, self._env if self._env is not None else os.environ,
                                            self.model)
        self._extra = extra
        kwargs: Dict[str, Any] = {"api_key": key, "timeout": self.timeout, "max_retries": 0,
                                  "http_client": httpx.Client(verify=ssl.create_default_context())}
        if url:
            kwargs["base_url"] = url
        return OpenAI(**kwargs)

    def __call__(self, request: Dict[str, Any]) -> str:
        if self._client is None:
            self._client = self._make_client()
        # No temperature or token cap: newer small models refuse both; the schema keeps the answer short.
        kwargs: Dict[str, Any] = dict(model=self.model, messages=request["messages"],
                                      response_format=request["response_format"], max_tokens=800)
        if self._extra:
            kwargs["extra_body"] = self._extra
        from .. import usage_ledger  # noqa: PLC0415

        resp = usage_ledger.call("case_judge", lambda: self._client.chat.completions.create(**kwargs),
                                 client=self._client, provider=self.provider, model=self.model)
        return resp.choices[0].message.content or ""


def decide(judge: Optional[Callable[[Dict[str, Any]], Any]], sig: Signature,
           candidates: Sequence[Tuple[Case, ScoreDetail]], timeout: float = DEFAULT_TIMEOUT,
           rng: Optional[random.Random] = None) -> Tuple[JudgeVerdict, Optional[Dict[str, Any]]]:
    """Build the request and ask; no judge configured is ``unsure`` (alert)."""
    if judge is None or not candidates:
        return unsure("no judge" if judge is None else "no candidates"), None
    request = build_request(sig, candidates, rng)
    return run_judge(judge, request, timeout), request


__all__: List[str] = ["SAME", "SIMILAR", "UNSURE", "FIELDS", "JudgeVerdict", "build_request", "parse_verdict",
                      "run_judge", "OpenAICompatibleJudge", "decide", "available_fields", "event_fields"]
