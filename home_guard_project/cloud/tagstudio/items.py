"""What the studio knows about one clip, and what each party said about it.

Every source turns its own records into :class:`ClipItem` objects; every opinion about a
clip (the old tag, the customer's answer, the AI's label, a teacher's suggestion, our own
tag) becomes an :class:`Opinion` with the same few fields, so the queue and the screen can
compare them without knowing where they came from.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...fleet_contract import taxonomy

# Who an opinion comes from, in the order the screen shows them.
OLD, OWNER, AI, TEACHER, STUDIO = "old", "owner", "ai", "teacher", "studio"
WHO = (OLD, OWNER, AI, TEACHER, STUDIO)

# Opinion labels: the taxonomy's three plus "empty" (nothing there) and "alert" (an old tag's
# [alert]: suspicious or escalation, we do not know which).
EMPTY, ALERT = "empty", "alert"
OPINION_LABELS = taxonomy.LABELS + (EMPTY, ALERT)
ALERTISH = ("suspicious", "escalation", ALERT)
QUIET = ("normal", EMPTY)


@dataclass
class Opinion:
    who: str
    label: str = ""            # one of OPINION_LABELS, or "" when unknown
    category: str = ""         # a taxonomy id, or ""
    text: str = ""             # what they wrote (description, summary, the owner's words)
    at: str = ""               # ISO time (UTC) the opinion was given, "" when unknown
    disputes_ai: bool = False  # the owner said the AI was wrong without saying what it was
    knows_empty: bool = True   # False for a model asked only normal/suspicious/escalation: its "normal" may be empty
    detail: Dict[str, Any] = field(default_factory=dict)

    def effective_label(self) -> str:
        """The label, or the one its category implies (N10 "nothing" is empty)."""
        if self.label:
            return self.label
        if self.category == "N10":
            return EMPTY
        cat = taxonomy.get(self.category)
        return cat.label if cat else ""

    def as_dict(self) -> Dict[str, Any]:
        return {"who": self.who, "label": self.label, "effective_label": self.effective_label(),
                "category": self.category, "text": self.text, "at": self.at,
                "disputes_ai": self.disputes_ai, "knows_empty": self.knows_empty, "detail": self.detail}


@dataclass
class ClipItem:
    key: str                         # the studio's id of the clip: "ds:<clip_id>", "ev:<event id>", "of:<prefix>/<stem>"
    clip_id: str                     # the clip's own name (its file stem), what exports and the eval call it
    origin: str                      # "dataset" (old tags), "customer" (an indexed event), "owner_feedback" (local copy)
    source: str = ""                 # house / external / uca / smarthome, or the site of a customer clip
    batch: str = ""
    camera: str = ""
    date: str = ""
    duration_sec: Optional[float] = None
    video: str = ""                  # local path of the full clip, when this machine has it
    crop: str = ""                   # local path of the VLM crop
    video_s3: str = ""               # s3://... of the full clip
    crop_s3: str = ""
    meta_path: str = ""
    local_time: Optional[str] = None  # HH:MM:SS the clip was recorded, when known
    sort_ts: float = 0.0             # newest first among equals
    fps: Optional[float] = None
    event_id: Optional[int] = None   # the indexed event, for customer clips
    artifacts: Dict[str, int] = field(default_factory=dict)  # "clip"/"crop" -> artifact id of an indexed event
    info: Dict[str, Any] = field(default_factory=dict)   # num_persons, num_cars, kind, needs_check, consent, ...
    opinions: Dict[str, Opinion] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {"key": self.key, "clip_id": self.clip_id, "origin": self.origin, "source": self.source,
                "batch": self.batch, "camera": self.camera, "date": self.date, "duration_sec": self.duration_sec,
                "local_time": self.local_time, "sort_ts": self.sort_ts, "event_id": self.event_id}

    def merge(self, other: "ClipItem") -> None:
        """Fill what this item lacks from *other* (the same clip found by another source)."""
        for name in ("source", "batch", "camera", "date", "video", "crop", "video_s3", "crop_s3", "meta_path",
                     "local_time"):
            if not getattr(self, name) and getattr(other, name):
                setattr(self, name, getattr(other, name))
        for name in ("duration_sec", "fps", "event_id"):
            if getattr(self, name) is None:
                setattr(self, name, getattr(other, name))
        self.sort_ts = max(self.sort_ts, other.sort_ts)
        for key, value in other.info.items():
            self.info.setdefault(key, value)
        for who, op in other.opinions.items():
            self.opinions.setdefault(who, op)
        for kind, art in other.artifacts.items():
            self.artifacts.setdefault(kind, art)
        self.info.setdefault("also", []).append(other.key)


def iso_utc(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    try:
        return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def iso_ts(text: str) -> Optional[float]:
    """Epoch seconds of an ISO time (``Z`` or an offset), or None."""
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def alertish(label: str) -> bool:
    return label in ALERTISH


def present(opinions: Dict[str, Opinion], skip: tuple = ()) -> List[Opinion]:
    """The opinions that say something (a label, a category, or a dispute), in screen order."""
    return [opinions[w] for w in WHO if w in opinions and w not in skip
            and (opinions[w].effective_label() or opinions[w].disputes_ai)]
