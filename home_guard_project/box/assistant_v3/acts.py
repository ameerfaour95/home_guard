"""The dialogue acts: what one owner message asks for or tells, as structured data (report §8.3 [1]).

The understanding call returns ``{"emotion", "acts": [...]}`` under :data:`SCHEMA`. Every act carries the EXACT
quote of the owner's words it came from, and every time slot quotes the words that gave it (``until_quote``).
:func:`validate` drops a quote that is not in the owner's words (this message, or an earlier one when the act
completes an earlier statement), so a time the owner never said cannot reach memory (the invented "23:59").
Routing is code: :data:`HANDLER_ACTS` change state in deterministic handlers; :data:`SKILL_ACTS` look and answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

ACTS = (
    "place_fact",        # whose place it is / what a place is - permanent ("זה הבית של השכן")
    "person_mark",       # who the people are (workers, family, the gardener) - a window and an end
    "activity_explain",  # what an ACTION seen in an alert means ("lying on the stairs = the electricians")
    "camera_fact",       # a standing fact or routine about a camera or the house ("only we use the back door")
    "tag_only",          # explicitly asks to change a clip's TAG / label ("שנה תיוג", "התיוג זה ...")
    "alert_feedback",    # a verdict on one alert ("זה תקין", "זה לא התרעה") with nothing to remember
    "question_live",     # what is happening NOW ("יש מישהו בחוץ?", "מה קורה")
    "question_history",  # what happened / which clip / same people / are you sure
    "question_memory",   # what do you remember
    "question_meta",     # about the assistant itself or its last message ("מה ההבדל בין...", "מה זה אומר?")
    "complaint",         # the owner is unhappy with the assistant (insult, "why do you repeat", "what's the link")
    "ack",               # "סבבה", "תודה", "בסדר הבנתי", "הבנתי אתה לא צריך לחזור על זה"
    "greeting",          # "היי", "בוקר טוב"
    "preference",        # how the assistant should behave from now on, or a name for a camera
    "command",           # do something now: pause, resume, camera off/on, send a video or photo, house mode
    "chit_chat",         # anything else conversational
    "unclear",           # cannot tell
)
HANDLER_ACTS = ("place_fact", "person_mark", "activity_explain", "camera_fact", "tag_only", "alert_feedback",
                "ack", "greeting", "preference", "command")
SKILL_ACTS = ("question_live", "question_history", "question_memory", "question_meta", "complaint", "chit_chat",
              "unclear")
COMMANDS = ("pause", "resume", "camera_off", "camera_on", "alias", "send_video", "send_photo", "house_mode",
            "setting")
ISSUES = ("repetition", "ignored_memory", "wrong_fact", "irrelevant", "unwanted_question", "bad_wording",
          "too_many_alerts", "no_explanation", "other")
MEMORY_ACTS = ("place_fact", "person_mark", "activity_explain", "camera_fact")
EMOTIONS = ("neutral", "annoyed", "angry", "confused", "happy")


def _nullable(kind: str, enum: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """A slot that may be empty: null for free text; "" for a closed list (Anthropic's structured output rejects a
    null inside an enum)."""
    if enum is not None:
        return {"type": "string", "enum": list(enum) + [""]}
    return {"type": [kind, "null"]}


ACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "act": {"type": "string", "enum": list(ACTS)},
        "quote": _nullable("string"),
        "camera": _nullable("string"),
        "event": _nullable("string"),
        "subject": _nullable("string"),
        "until_quote": _nullable("string"),
        "scope": _nullable("string", ("camera", "house")),
        "command": _nullable("string", COMMANDS),
        "value": _nullable("string"),
        "verdict": _nullable("string", ("normal", "false_alarm", "real")),
        "issue": _nullable("string", ISSUES),
        "routine": {"type": "boolean"},
        "look_again": {"type": "boolean"},
        "earlier": {"type": "boolean"},
    },
    "required": ["act", "quote", "camera", "event", "subject", "until_quote", "scope", "command", "value",
                 "verdict", "issue", "routine", "look_again", "earlier"],
    "additionalProperties": False,
}
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "emotion": {"type": "string", "enum": list(EMOTIONS)},
        "acts": {"type": "array", "items": ACT_SCHEMA},
    },
    "required": ["emotion", "acts"],
    "additionalProperties": False,
}


@dataclass
class Act:
    act: str
    quote: str = ""
    camera: str = ""          # the internal camera id ("" = none named; "house" = the whole house)
    event: str = ""           # a chat handle (E7) of an alert, photo or clip
    subject: str = ""
    until_quote: str = ""
    scope: str = ""
    command: str = ""
    value: str = ""
    verdict: str = ""
    issue: str = ""
    routine: bool = False
    look_again: bool = False
    earlier: bool = False
    repaired: bool = False    # re-read from an earlier mishandled message: done only when complete
    dropped: List[str] = field(default_factory=list)      # slots validate() removed, for the turn log

    def brief(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in ("", False, [], None)}


@dataclass
class Understanding:
    emotion: str = "neutral"
    acts: List[Act] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def kinds(self) -> List[str]:
        return [a.act for a in self.acts]

    def has(self, *kinds: str) -> bool:
        return any(a.act in kinds for a in self.acts)

    def first(self, *kinds: str) -> Optional[Act]:
        return next((a for a in self.acts if a.act in kinds), None)


_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[\"'“”„׳״`.,!?;:()\[\]{}\-–—]+")


def norm(text: str) -> str:
    return _SPACE.sub(" ", _PUNCT.sub(" ", str(text or ""))).strip().casefold()


def quoted(quote: str, *sources: str) -> bool:
    """*quote* appears (punctuation and spacing aside) in one of *sources*."""
    q = norm(quote)
    return bool(q) and any(q in norm(s) for s in sources if s)


def validate(raw: Dict[str, Any], message: str, earlier_owner: Sequence[str], cameras: Dict[str, str],
             handles: Sequence[str]) -> Understanding:
    """The parsed understanding with every slot checked against what is real: quotes against the owner's words
    (this message, or his earlier messages for an ``earlier`` act), cameras against the house's cameras (by key
    ``cam3`` or name), events against the chat's handles. A slot that fails is emptied, never guessed."""
    out = Understanding(raw=raw if isinstance(raw, dict) else {})
    emotion = str(out.raw.get("emotion") or "neutral")
    out.emotion = emotion if emotion in EMOTIONS else "neutral"
    for item in out.raw.get("acts") or []:
        if not isinstance(item, dict) or item.get("act") not in ACTS:
            continue
        act = Act(act=str(item["act"]))
        sources = [message] + (list(earlier_owner) if item.get("earlier") else [])
        for name in ("quote", "until_quote"):
            value = " ".join(str(item.get(name) or "").split())
            if value and quoted(value, *sources):
                setattr(act, name, value)
            elif value:
                act.dropped.append(name)
        for name in ("subject", "value"):
            setattr(act, name, " ".join(str(item.get(name) or "").split())[:300])
        cam = str(item.get("camera") or "").strip()
        if cam:
            key = cam.casefold()
            if key in ("house", "all", "הבית", "כל הבית"):
                act.camera = "house"
            elif key in cameras:
                act.camera = cameras[key]
            else:
                act.dropped.append("camera")
        ev = str(item.get("event") or "").strip().upper()
        if ev and ev in handles:
            act.event = ev
        elif ev:
            act.dropped.append("event")
        for name, allowed in (("scope", ("camera", "house")), ("command", COMMANDS),
                              ("verdict", ("normal", "false_alarm", "real")), ("issue", ISSUES)):
            value = str(item.get(name) or "")
            setattr(act, name, value if value in allowed else "")
        act.routine = item.get("routine") is True
        act.look_again = item.get("look_again") is True
        act.earlier = item.get("earlier") is True
        if act.act in MEMORY_ACTS and not act.quote:
            continue                           # memory only from words he really wrote (no quote: no write)
        out.acts.append(act)
    if not out.acts:
        out.acts.append(Act(act="unclear"))
    return out
