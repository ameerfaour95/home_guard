"""The records the case memory keeps: a case (a routine the owner explained), its scope, its examples.

A case remembers a *situation*, never a person: place, time, path, counts, clothing and vehicle words. No faces,
no gait or body shape, no image embeddings (plan page section 6, Privacy). Everything here is plain data with a
JSON round trip, so the store can live in its own file today and inside the house facts journal later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

# What memory may do to a delivery. ``alert`` is the box's normal path, untouched.
ALERT, QUIET, DIGEST = "alert", "quiet", "digest"
DELIVERY_LEVELS = (ALERT, QUIET, DIGEST)

# The owner's ceiling for a case ("what to do next time"). ``alert`` = remember for context, keep alerting.
EFFECTS = (ALERT, QUIET, DIGEST)

# Where a case stands. Shadow decides and logs but still alerts; active softens; paused waits for the owner's
# "still relevant?"; invalid is history (never deleted).
SHADOW, ACTIVE, PAUSED, INVALID = "shadow", "active", "paused", "invalid"
STATUSES = (SHADOW, ACTIVE, PAUSED, INVALID)

WHO = ("neighbour", "family", "worker", "courier", "other")

# Python's weekday numbers (Monday = 0). The Israeli work week is Sunday to Thursday.
SUNDAY, FRIDAY, SATURDAY = 6, 4, 5
WORKWEEK = (6, 0, 1, 2, 3)
ALL_DAYS = (0, 1, 2, 3, 4, 5, 6)

MAX_EXAMPLES = 5


def _known(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in names}


@dataclass(frozen=True)
class Signature:
    """What one event looked like, from the Eye's structured observation and the tracker. Built in code.

    ``path`` is the zone sequence (``("gate", "street")``); ``entry_edge``/``exit_edge`` are the first and last
    zones unless the tracker names the frame edges. ``template`` is the fixed-shape English sentence that is
    embedded for retrieval (never the model's free text: it keeps retrieval stable when the Eye model changes).
    """
    camera: str
    ts: float
    minute: int                      # local minute of day, 0..1439
    weekday: int                     # Python weekday, Monday = 0
    phase: str = ""                  # day / evening / late_night / dawn ("" when unknown)
    house_state: str = "home_awake"
    category: str = ""               # N1.. / S1.. / E1.. / other; "" when the Eye gave none (today's gpt-4o)
    movement: str = ""
    zone: str = ""
    path: Tuple[str, ...] = ()
    entry_edge: str = ""
    exit_edge: str = ""
    dwell_s: Optional[float] = None  # time in view, from the tracker
    people: int = 0
    vehicles: int = 0
    animals: int = 0
    flags: Tuple[str, ...] = ()
    appearance: Tuple[str, ...] = ()  # clothing, bag, vehicle colour/type words; never a face
    label: str = ""                  # the box's label after the priors: normal / suspicious / escalation
    serious_behaviour: bool = False
    cameras_in_incident: int = 1
    eye_model: str = ""
    prompt_version: str = ""
    template: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for key in ("path", "flags", "appearance"):
            out[key] = list(out[key])
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signature":
        data = _known(cls, data)
        for key in ("path", "flags", "appearance"):
            data[key] = tuple(data.get(key) or ())
        return cls(**data)


@dataclass(frozen=True)
class Scope:
    """Every field is a negative gate: an event outside it never matches (gates.py)."""
    camera: str
    hours: Tuple[str, str]           # ("07:10", "08:10"); may cross midnight
    weekdays: Tuple[int, ...] = ALL_DAYS
    house_states: Tuple[str, ...] = ("home_awake",)
    night: bool = False              # late_night allowed (only when the case was learned at night)
    people: int = 1
    vehicles: int = 0
    path: Tuple[str, ...] = ()
    entry_edge: str = ""
    exit_edge: str = ""
    categories: Tuple[str, ...] = ()  # allowed category ids; compared by family (gates.FAMILIES)
    max_dwell_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for key in ("hours", "weekdays", "house_states", "path", "categories"):
            out[key] = list(out[key])
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scope":
        data = _known(cls, data)
        data["hours"] = tuple(data.get("hours") or ("00:00", "00:00"))
        for key in ("weekdays", "house_states", "path", "categories"):
            if key in data:
                data[key] = tuple(data[key] or ())
        return cls(**data)


@dataclass(frozen=True)
class Example:
    """One confirmed occurrence. ``embedding`` is the vector of the signature's template (256 floats)."""
    event_id: str
    signature: Signature
    embedding: Optional[Tuple[float, ...]] = None
    clip: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"event_id": self.event_id, "signature": self.signature.to_dict(),
                "embedding": list(self.embedding) if self.embedding else None, "clip": self.clip}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Example":
        emb = data.get("embedding")
        return cls(event_id=str(data.get("event_id") or ""), signature=Signature.from_dict(data["signature"]),
                   embedding=tuple(float(x) for x in emb) if emb else None, clip=str(data.get("clip") or ""))


@dataclass
class Case:
    """A precedent: the owner's explanation as a scoped rule, plus 1-5 real examples.

    ``confirmations`` count only the owner's "yes, that's them"; automatic matches are logged as
    ``last_seen_at`` and never strengthen a case. ``streak`` is the confirmations the trust ladder counts;
    a contradiction steps it back one stage.
    """
    id: str
    scope: Scope
    who: str = "other"
    title: str = ""                  # the owner's words for who ("the neighbour going to work")
    note: str = ""                   # the owner's explanation, shown to the judge as data, never as instructions
    recognise: Tuple[str, ...] = ()  # appearance words the owner confirmed; () when "the clothes change"
    effect: str = DIGEST             # the owner's ceiling: alert / quiet / digest
    examples: List[Example] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)   # event ids the owner said were "not them"
    status: str = SHADOW
    confirmations: int = 0
    contradictions: int = 0
    streak: int = 0
    created_at: float = 0.0
    created_by: str = ""
    source: Dict[str, Any] = field(default_factory=dict)  # alert_id, chat_id, message_ids, origin
    last_confirmed_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    review_asked_at: Optional[float] = None
    invalid_at: Optional[float] = None
    invalid_reason: str = ""
    merged_into: str = ""
    revision: int = 1

    @property
    def camera(self) -> str:
        return self.scope.camera

    @property
    def live(self) -> bool:
        return self.status in (SHADOW, ACTIVE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "scope": self.scope.to_dict(), "who": self.who, "title": self.title, "note": self.note,
            "recognise": list(self.recognise), "effect": self.effect,
            "examples": [e.to_dict() for e in self.examples], "negatives": list(self.negatives),
            "status": self.status, "confirmations": self.confirmations, "contradictions": self.contradictions,
            "streak": self.streak, "created_at": self.created_at, "created_by": self.created_by,
            "source": dict(self.source), "last_confirmed_at": self.last_confirmed_at,
            "last_seen_at": self.last_seen_at, "review_asked_at": self.review_asked_at,
            "invalid_at": self.invalid_at, "invalid_reason": self.invalid_reason, "merged_into": self.merged_into,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Case":
        data = dict(data)
        scope = Scope.from_dict(data.pop("scope"))
        examples = [Example.from_dict(e) for e in data.pop("examples", []) or []]
        kept = _known(cls, data)
        kept.pop("scope", None)
        kept.pop("examples", None)
        kept["recognise"] = tuple(kept.get("recognise") or ())
        kept["negatives"] = list(kept.get("negatives") or [])
        kept["source"] = dict(kept.get("source") or {})
        return cls(scope=scope, examples=examples, **kept)


@dataclass(frozen=True)
class Expecting:
    """ "Only today": an expecting note that ends at *until* (end of that day). Not a case."""
    id: str
    camera: str
    text: str
    until: float
    created_at: float
    created_by: str = ""
    source: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Expecting":
        return cls(**_known(cls, data))


def hhmm(minute: int) -> str:
    minute %= 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def minutes_of(text: str) -> int:
    hour, minute = str(text).split(":")
    h, m = int(hour), int(minute)
    if not (0 <= h <= 24 and 0 <= m < 60) or (h == 24 and m):
        raise ValueError(f"bad time {text!r}")
    return (h * 60 + m) % 1440


def window_minutes(hours: Sequence[str]) -> Tuple[int, int]:
    """``(start, length)`` of an ``("HH:MM", "HH:MM")`` window; length 1440 when start equals end."""
    start, end = minutes_of(hours[0]), minutes_of(hours[1])
    length = (end - start) % 1440 or 1440
    return start, length
