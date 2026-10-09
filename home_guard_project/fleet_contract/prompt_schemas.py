"""The answer schemas of the box's prompts, by prompt version, vendored verbatim.

A tag follows the schema of the prompt the clip was answered with (meta ``teacher.prompt_version``): the form, the
"In my words" converter, Suggest and the export all read it from here. Two blocks are verbatim copies of the box's
code (branch beelink-collector-box), checked by tests/fleet_contract/test_prompt_schemas.py:

- the Oct 3 legacy prompt the box runs today: ``PROMPT_VERSION`` .. ``VLM_SCHEMA`` of box/inference.py;
- the Eye (eye-v3): ``EYE_PROMPT_VERSION`` .. ``response_format`` of box/eye_prompt.py (``tx`` is the vendored
  taxonomy, itself a verbatim copy of box/taxonomy.py).

Below the blocks, :func:`answer_schema` picks a clip's schema from its prompt version.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from . import taxonomy as tx

# >>> vendored from box/inference.py
# Bumped whenever the prompt or the answer's schema changes, so training records can be told apart.
PROMPT_VERSION = "2026-10-03.tagged-rules-label-animals-why-owner-facts"

# The three labels the model gives a scene, and what the box does with each. The owner
# chose them (this is also what a student model will be trained to answer):
#   normal      ordinary activity -> a message
#   suspicious  worth a look      -> a message marked suspicious (more may follow later)
#   escalation  danger or a crime -> the urgent alert
LABELS = ("normal", "suspicious", "escalation")
LABEL_COMMANDS = {"normal": "[send_message]", "suspicious": "[send_message]", "escalation": "[call_owner]"}

# The answer's shape, enforced on the model (structured output) and checked on the way back.
VLM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "label": {"type": "string", "enum": list(LABELS)},
        "raw_label": {"type": "string", "enum": list(LABELS)},
        "applied_fact_id": {"type": "string"},
        "serious_behaviour": {"type": "boolean"},
        "people": {"type": "integer"},
        "vehicle_moving": {"type": "boolean"},
        "animals": {"type": "integer"},
        "why": {"type": "string"},
        "summary_owner": {"type": "string"},
    },
    "required": ["summary", "label", "raw_label", "applied_fact_id", "serious_behaviour",
                 "people", "vehicle_moving", "animals", "why", "summary_owner"],
    "additionalProperties": False,
}


# <<< vendored


# >>> vendored from box/eye_prompt.py
EYE_PROMPT_VERSION = "2026-10-06.eye-v3"

# Flags that make a "normal" answer at least suspicious (eval 2026-10-06: the Eye called handle-trying normal).
RISK_FLAGS = frozenset({"touching_handle", "weapon_visible", "crouching", "flashlight"})

QUALITY = ("good", "limited", "unusable")
CONFIDENCE = ("clear", "partial", "cannot_tell")
YES_NO = ("yes", "no", "unclear")
VISIBLE = ("clear", "partial", "no")
MAX_QUESTIONS = 3


# ----------------------------------------------------------------------------
# Schemas (strict: every property required, nothing else allowed)
# ----------------------------------------------------------------------------
def _obj(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _enum(values: Sequence[str]) -> Dict[str, Any]:
    return {"type": "string", "enum": list(values)}


_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}

_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "alert_triage": _obj({
        "summary": _STR, "category": _enum(tx.CATEGORY_IDS), "other_text": _STR, "zone": _enum(tx.ZONES),
        "movement": _enum(tx.MOVEMENTS), "flags": {"type": "array", "items": _enum(tx.FLAGS)},
        "people": _INT, "vehicles": _INT, "vehicle_moving": _BOOL, "animals": _INT,
        "visibility": _enum(tx.VISIBILITY), "appearance": {"type": "array", "items": _STR},
        "evidence_frame": _INT, "raw_label": _enum(tx.LABELS), "label": _enum(tx.LABELS),
        "applied_fact_id": _STR, "serious_behaviour": _BOOL, "why": _STR,
    }),
    "snapshot": _obj({
        "description": _STR, "quality": _enum(QUALITY), "people": _INT, "vehicles": _INT, "animals": _INT,
        "safety_note": _STR,
    }),
    "event_question": _obj({"answer": _STR, "confidence": _enum(CONFIDENCE), "evidence_frame": _INT}),
    "follow_up": _obj({"answers": {"type": "array", "items": _obj({
        "question": _STR, "answer": _enum(YES_NO), "evidence_frame": _INT, "visible": _enum(VISIBLE)})}}),
}


def schema(intent: str) -> Dict[str, Any]:
    if intent not in _SCHEMAS:
        raise ValueError(f"intent must be one of {', '.join(tx.INTENTS)}")
    return _SCHEMAS[intent]


def response_format(intent: str) -> Dict[str, Any]:
    """The OpenAI-compatible ``response_format`` (strict json_schema; vLLM turns it into guided decoding)."""
    return {"type": "json_schema", "json_schema": {"name": f"eye_{intent}", "strict": True, "schema": schema(intent)}}


# <<< vendored


LEGACY = "legacy"
EYE = "eye"


def schema_kind(prompt_version: Optional[str]) -> str:
    """EYE for an Eye prompt version (with or without "+<tracker facts>") and for a clip no prompt answered (an old
    dataset clip: tagged with the studio's category form, the Eye's schema); LEGACY for any other recorded version
    (the box's own Oct 3 prompt, or an older one with the same fields)."""
    version = str(prompt_version or "")
    return EYE if not version or version.startswith(EYE_PROMPT_VERSION) else LEGACY


def answer_schema(prompt_version: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """(name, strict JSON schema) of the answer a clip's prompt version asks for."""
    if schema_kind(prompt_version) == EYE:
        return "eye_alert_triage", schema("alert_triage")
    return "legacy_alert", VLM_SCHEMA


def field_order(prompt_version: Optional[str]) -> Sequence[str]:
    """The answer's fields in the order the model writes them (the order a training row keeps)."""
    return tuple(answer_schema(prompt_version)[1]["properties"])
