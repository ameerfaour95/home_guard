"""The Eye's prompt, composed per look from the fixed taxonomy and the current situation.

Spec ``docs/superpowers/specs/2026-10-05-situation-aware-eye-and-investigator-design.md`` Part 2. The Eye sees
the same way always and judges by the situation: what a scene *is* (the category) never depends on the hour;
what it *means* (the label) does, and code (``taxonomy.contextual_label``) has the last word on that.

Modules, in prompt order:

1. Base: role, honesty ("appears to"; never names, age or ethnicity; appearance alone is never a category),
   English only (the owner's Hebrew comes from the messenger), and the taxonomy (``taxonomy.prompt_list``).
2. The situation header (``Situation.header()``), identical at training and inference, and right under it the
   scene map's ZONE FACTS line when the camera has a map (``zone_facts_block``; never inside the header, and
   kept in the training record's ``situation.scene`` so training prompts carry the same line), then the tracker's
   TRACKER FACTS line when ``eye_tracker_facts`` is on (``tracker.prompt_block``; the version gets ``+tf1``).
3. The expectations block: plain sentences built from the priors table, only what differs from a plain day.
4. The attention list: what to look for now (night: flashlights, hands on handles and windows, crouching,
   carrying things out; day: the act, not the clothes).
5. The intent module and its strict JSON schema (``schema`` / ``response_format``): ``alert_triage`` (the guard
   loop; observation first, label last), ``snapshot`` (the owner's "what's there now"), ``event_question``
   (the owner asks about a saved clip) and ``follow_up`` (the Investigator's yes/no questions).
   ``alert_triage`` keeps the house-notes (facts) block of ``inference.build_prompt``.

``postprocess(parsed, situation)`` turns an ``alert_triage`` answer into the fields inference already relies on
(label, raw_label, people, vehicle_moving, animals, why, summary, applied_fact_id, serious_behaviour) plus the
``situation`` / ``observation`` / ``judgement`` records for the training data and .meta.json. It never lowers the
Eye's own labels and never softens escalation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from . import inference
from ..prompts import load, render
from . import taxonomy as tx
from .situation import Situation
from .tracker import TRACKER_FACTS_VERSION, prompt_block

# Bumped whenever the wording or a schema changes, so training records and eval results can be told apart.
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


# ----------------------------------------------------------------------------
# Prompt modules
# ----------------------------------------------------------------------------
def _quoted(text: Any, limit: int = 200) -> str:
    """Owner or Investigator text placed in the prompt as data: one line, no quotes or backticks."""
    return " ".join(re.sub(r"[\"`]", "", str(text or "")).split())[:limit]


def _base(situation: Situation) -> str:
    return render("eye_v3_base.prompt", camera=_quoted(situation.camera, 40),
                  categories=tx.prompt_list(order=("E", "S", "N")))


# How a category reads when it is NOT expected now, per priors column (default: its name).
_NOT_EXPECTED_TEXT = {
    tx.NIGHT: {"N2": load("eye_v3_unexpected_n2.prompt"),
               "N6": load("eye_v3_unexpected_n6.prompt"),
               "N7": load("eye_v3_unexpected_n7.prompt"),
               "N9": load("eye_v3_unexpected_n9.prompt")},
    tx.AWAY: {"N7": load("eye_v3_unexpected_n7.prompt"),
              "N9": load("eye_v3_unexpected_n9.prompt")},
}


def _names(ids: Sequence[str], special: Optional[Dict[str, str]] = None) -> str:
    parts = [(special or {}).get(cid) or f"{tx.BY_ID[cid].name} ({cid})" for cid in ids]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


ZONE_FACTS_RULE = load("eye_v3_zone_facts_rule.prompt")


def zone_facts_block(situation: Situation) -> str:
    """The ZONE FACTS line and how to use it; "" when the camera's map has nothing to say."""
    if not situation.zone_facts:
        return ""
    return situation.zone_facts + "\n" + ZONE_FACTS_RULE


def expectations_block(situation: Situation) -> str:
    """What is expected now, in short plain sentences, from the priors table."""
    ctx = situation.to_taxonomy_context()
    col = tx.column(situation.phase, situation.house_state)
    if col == tx.AWAY:
        lead = load("eye_v3_now_away.prompt")
    elif situation.house_state == "home_asleep":
        lead = load("eye_v3_now_asleep.prompt")
    elif col == tx.NIGHT:
        lead = load("eye_v3_now_night.prompt")
    else:
        lead = render("eye_v3_now_home.prompt", part_of_day="evening" if situation.phase == "evening" else "daytime")
    lines = [lead]
    changes = tx.expectation_changes(ctx)
    unusual = [cid for cid, now, _ in changes if now == tx.UNUSUAL]
    serious = [cid for cid, now, _ in changes if now == tx.SERIOUS]
    if not changes:
        lines.append(load("eye_v3_expected_all.prompt"))
    if unusual and col == tx.AWAY:
        lines.append(render("eye_v3_not_expected_away.prompt", names=_names(unusual, _NOT_EXPECTED_TEXT[col])))
    elif unusual:
        lines.append(render("eye_v3_not_expected_hour.prompt", names=_names(unusual, _NOT_EXPECTED_TEXT.get(col))))
    if serious:
        lines.append(render("eye_v3_serious_now.prompt", names=_names(serious)))
    if situation.expecting:
        lines.append(render("eye_v3_owner_expects.prompt",
                            expected=", ".join(f"'{t}'" for t in situation.expecting)))
    return "\n".join(lines)


def attention_block(situation: Situation) -> str:
    """What to look for now."""
    if situation.dark or tx.column(situation.phase, situation.house_state) != tx.DAY:
        flashlight = (load("eye_v3_look_for_flashlight.prompt") + " ") if situation.dark else ""
        text = render("eye_v3_look_for.prompt", flashlight=flashlight)
    else:
        text = load("eye_v3_look_at_act.prompt")
    if situation.house_state == "away":
        text += " " + load("eye_v3_look_away.prompt")
    return text


def _facts_block(situation: Situation, facts: Sequence[Dict[str, Any]]) -> str:
    """The house notes, with the same rules as ``inference.build_prompt``."""
    live = inference._prompt_facts(facts, situation.camera, situation.ts)
    if not live:
        return ""
    lines = []
    for fact in live:
        hours = "-".join(fact["hours"]) if fact.get("hours") else "all day"
        line = (f"- {fact['id']}: {fact['kind']}, {fact['effect']}, {hours}, "
                f"at {fact.get('area') or situation.camera}: {fact['text']}")
        lines.append(inference._note_text(line)[:200])
    return render("eye_house_notes.prompt", notes="\n".join(lines))


def _alert_triage() -> str:
    return render("eye_v3_alert_triage.prompt", zones=" | ".join(tx.ZONES),
                  movements=" | ".join(tx.MOVEMENTS), flags=", ".join(tx.FLAGS))


def _snapshot() -> str:
    return render("eye_v3_snapshot.prompt", quality=" | ".join(QUALITY))


def _event_question(question: str) -> str:
    return render("eye_v3_event_question.prompt", question=_quoted(question),
                  confidence=" | ".join(CONFIDENCE))


def _follow_up(questions: Sequence[str]) -> str:
    asked = [_quoted(q, 160) for q in questions if _quoted(q, 160)][:MAX_QUESTIONS]
    numbered = ("\n".join(f"{i}. {q}" for i, q in enumerate(asked, 1))
                or "1. " + load("eye_v3_follow_up_default.prompt"))
    return render("eye_v3_follow_up.prompt", questions=numbered, answer=" | ".join(YES_NO),
                  visible=" | ".join(VISIBLE))


def build_prompt(situation: Situation, facts: Sequence[Dict[str, Any]] = (), question: str = "",
                 questions: Sequence[str] = ()) -> str:
    """The Eye's prompt for *situation* (its intent picks the module and schema)."""
    intent = situation.intent
    schema(intent)                      # unknown intents fail here
    under = [block for block in (zone_facts_block(situation), prompt_block(situation.tracker_facts)) if block]
    parts = [_base(situation), "\n".join([situation.header()] + under)]
    if intent != "snapshot":
        parts += [expectations_block(situation), attention_block(situation)]
    if intent == "alert_triage":
        parts.append(_alert_triage())
        notes = _facts_block(situation, facts)
        if notes:
            parts.append(notes)
    elif intent == "snapshot":
        parts.append(_snapshot())
    elif intent == "event_question":
        parts.append(_event_question(question))
    else:
        parts.append(_follow_up(questions))
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------
# After the answer
# ----------------------------------------------------------------------------
def _choice(value: Any, allowed: Sequence[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _higher(*labels: str) -> str:
    valid = [x for x in labels if x in tx.LABELS]
    return max(valid, key=tx.LABELS.index) if valid else "normal"


_WHEN = {tx.NIGHT: "at this hour of the night", tx.AWAY: "while nobody is home", tx.DAY: "now"}


def _situation_why(category: str, judgement: tx.Judgement, situation: Situation) -> str:
    cat = tx.get(category)
    what = cat.name if cat else "something the box has no category for"
    when = _WHEN[tx.column(situation.phase, situation.house_state)]
    if situation.house_state == "home_asleep" and when != _WHEN[tx.AWAY]:
        when = "while the family sleeps"
    if situation.crossed_in and when != _WHEN[tx.DAY] and judgement.expectation == tx.UNUSUAL:
        return f"crossed onto the owner's ground {when}"
    if judgement.expectation == tx.UNUSUAL:
        return f"{what} is not expected {when}"
    if judgement.expectation in (tx.SERIOUS, tx.ESCALATION):
        return f"{what} is serious {when}"
    return ""


def observation_of(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """The context-free observation, normalized to the taxonomy's words."""
    category = tx.normalize_id(parsed.get("category"))
    flags = parsed.get("flags") if isinstance(parsed.get("flags"), list) else []
    appearance = parsed.get("appearance") if isinstance(parsed.get("appearance"), list) else []
    try:
        frame = max(0, int(parsed.get("evidence_frame") or 0))
    except (TypeError, ValueError, OverflowError):
        frame = 0
    return {"category": category,
            "other_text": " ".join(str(parsed.get("other_text") or "").split())[:120] if category == tx.OTHER else "",
            "zone": _choice(parsed.get("zone"), tx.ZONES, "other"),
            "movement": _choice(parsed.get("movement"), tx.MOVEMENTS, "none"),
            "flags": [f for f in tx.FLAGS if f in {str(x).strip().lower() for x in flags}],
            # Unknown visibility counts as partial: on a serious category that opens a case.
            "visibility": _choice(parsed.get("visibility"), tx.VISIBILITY, "partial"),
            "evidence_frame": frame,
            # Recognising the same person or car again (case memory); clothing and vehicle words only.
            "appearance": [" ".join(str(a).split())[:40] for a in appearance if str(a).strip()][:4],
            "vehicles": _count(parsed.get("vehicles"))}


def postprocess(parsed: Any, situation: Situation) -> Optional[Dict[str, Any]]:
    """The answer as inference uses it, with the situation's judgement applied. None when it is not an object.

    ``raw_label`` stays the Eye's context-free label (the training target; the category's group when it gave
    none). ``label`` is the higher of the code's contextual label and the Eye's own labels. ``serious_behaviour``
    is forced true on a serious or escalation category, so no house note can soften it.
    """
    if not isinstance(parsed, dict):
        return None
    if situation.intent != "alert_triage" or "category" not in parsed:
        # Another intent, or an answer that is not the Eye's (NullBackend's {"summary": ""}): passed through as is.
        return dict(parsed, situation=situation.record())
    obs = observation_of(parsed)
    cat = tx.get(obs["category"])
    raw = _choice(parsed.get("raw_label"), tx.LABELS, cat.label if cat else "")
    model_label = _choice(parsed.get("label"), tx.LABELS, "")
    ctx = situation.to_taxonomy_context(obs["movement"], obs["zone"], obs["flags"])
    judgement = tx.contextual_label(obs["category"], raw, ctx, obs["visibility"])
    label = _higher(judgement.label, raw, model_label)
    risky = sorted(set(obs["flags"]) & RISK_FLAGS)
    if label == "normal" and risky and "key_or_door_opened_from_inside" not in obs["flags"]:
        # The Eye saw a hand on a handle, a weapon, crouching or a flashlight and still called it normal.
        label = "suspicious"
        judgement = tx.Judgement(label, judgement.expectation, judgement.escalation_candidate, True,
                                 judgement.reasons + (f"risk flag {', '.join(risky)} on a normal category",))
    serious = parsed.get("serious_behaviour") is True or judgement.expectation in (tx.SERIOUS, tx.ESCALATION)
    why = " ".join(str(parsed.get("why") or "").split())
    if label != "normal" and (not why or label != _higher(raw, model_label)):
        reason = _situation_why(obs["category"], judgement, situation)
        why = "; ".join(x for x in (why, reason) if x)
    if label == "normal":
        why = ""
    return dict(
        parsed,
        summary=str(parsed.get("summary") or "").strip(), label=label, raw_label=raw,
        applied_fact_id=str(parsed.get("applied_fact_id") or ""), serious_behaviour=serious,
        people=_count(parsed.get("people")), vehicle_moving=parsed.get("vehicle_moving") is True,
        animals=_count(parsed.get("animals")), why=why, category=obs["category"],
        situation=situation.record(), observation=obs,
        judgement={"expectation": judgement.expectation, "escalation_candidate": judgement.escalation_candidate,
                   "open_case": judgement.open_case, "reasons": list(judgement.reasons)},
    )


def records(processed: Optional[Dict[str, Any]], situation: Situation) -> Dict[str, Any]:
    """The fields the training record and .meta.json gain (situation always; observation and judgement when the
    Eye answered)."""
    version = EYE_PROMPT_VERSION + (f"+{TRACKER_FACTS_VERSION}" if situation.tracker_facts else "")
    out: Dict[str, Any] = {"situation": situation.record(), "prompt_version": version}
    for key in ("observation", "judgement"):
        if processed and isinstance(processed.get(key), dict):
            out[key] = processed[key]
    return out
