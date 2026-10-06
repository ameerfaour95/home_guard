"""The tag form's fields, their validation against the taxonomy, and the fold of tag events into one tag per clip.

Tags are stored as append-only events (``tag_events``): every save adds one row with the fields it carries. The fold
merges a clip's events in order (a later event's fields replace earlier ones; fields it does not carry are kept),
so a partial save (only ``needs_check``) does not wipe the rest, and the whole history stays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from ...fleet_contract import taxonomy
from .items import STUDIO, Opinion

MAX_TEXT = 2000
MAX_SHORT_TEXT = 500

# name -> kind. Choices come from the taxonomy.
FIELDS: Dict[str, str] = {
    "category": "choice", "other_text": "short_text", "zone": "choice", "movement": "choice",
    "flags": "multi", "visibility": "choice", "evidence_sec": "number", "evidence_frame": "int",
    "raw_label": "choice", "description": "text", "notes": "text", "needs_check": "bool", "delete": "bool",
}
CHOICES: Dict[str, tuple] = {
    "category": taxonomy.CATEGORY_IDS, "zone": taxonomy.ZONES, "movement": taxonomy.MOVEMENTS,
    "visibility": taxonomy.VISIBILITY, "raw_label": taxonomy.LABELS, "flags": taxonomy.FLAGS,
}


class TagError(ValueError):
    """A tag field is unknown or its value is not allowed."""


def _clean_one(name: str, value: Any) -> Any:
    kind = FIELDS[name]
    if kind == "choice":
        text = str(value or "").strip()
        if name == "category" and text:
            text = taxonomy.normalize_id(text) if taxonomy.get(text) else text.lower()
        if text and text not in CHOICES[name]:
            raise TagError(f"{name}: {value!r} is not one of {', '.join(CHOICES[name])}")
        return text
    if kind == "multi":
        values = value if isinstance(value, (list, tuple)) else ([] if value in (None, "") else [value])
        chosen = []
        for v in values:
            v = str(v).strip()
            if v not in CHOICES[name]:
                raise TagError(f"{name}: {v!r} is not one of {', '.join(CHOICES[name])}")
            chosen.append(v)
        return [f for f in CHOICES[name] if f in chosen]          # the taxonomy's order, no repeats
    if kind in ("text", "short_text"):
        text = str(value or "").replace("\r\n", "\n").strip()
        limit = MAX_SHORT_TEXT if kind == "short_text" else MAX_TEXT
        if len(text) > limit:
            raise TagError(f"{name}: longer than {limit} characters")
        return text
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if value in (0, 1, "0", "1", "true", "false", "", None):
            return value in (1, "1", "true")
        raise TagError(f"{name}: {value!r} is not true/false")
    if value in (None, ""):                                          # number / int
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TagError(f"{name}: {value!r} is not a number") from None
    if not math.isfinite(number) or number < 0 or number > 86400:
        raise TagError(f"{name}: {value!r} must be between 0 and 86400")
    return int(round(number)) if kind == "int" else round(number, 3)


def clean_fields(fields: Any) -> Dict[str, Any]:
    """Validate a (possibly partial) tag. Raises :class:`TagError` naming the bad field."""
    if not isinstance(fields, dict):
        raise TagError("fields must be an object")
    unknown = sorted(set(fields) - set(FIELDS))
    if unknown:
        raise TagError(f"unknown field(s): {', '.join(unknown)}")
    return {name: _clean_one(name, value) for name, value in fields.items()}


def default_raw_label(category: str) -> str:
    cat = taxonomy.get(category)
    return cat.label if cat else ""


def empty_form() -> Dict[str, Any]:
    return {name: ([] if kind == "multi" else False if kind == "bool" else None if kind in ("number", "int") else "")
            for name, kind in FIELDS.items()}


@dataclass
class Tag:
    key: str
    fields: Dict[str, Any] = field(default_factory=dict)
    at: str = ""
    by: str = ""
    events: int = 0

    @property
    def raw_label(self) -> str:
        return self.fields.get("raw_label") or default_raw_label(self.fields.get("category", ""))

    def opinion(self) -> Opinion:
        return Opinion(STUDIO, label=self.raw_label, category=self.fields.get("category", ""),
                       text=self.fields.get("description", ""), at=self.at,
                       detail={"by": self.by, "needs_check": bool(self.fields.get("needs_check")),
                               "delete": bool(self.fields.get("delete"))})

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "fields": self.fields, "at": self.at, "by": self.by, "events": self.events,
                "raw_label": self.raw_label}


def fold(events: Iterable[Dict[str, Any]]) -> Dict[str, Tag]:
    """The latest tag per clip from events ``{"key", "at", "by", "fields"}``, in order."""
    tags: Dict[str, Tag] = {}
    for ev in events:
        key = str(ev.get("key") or "")
        if not key or not isinstance(ev.get("fields"), dict):
            continue
        tag = tags.setdefault(key, Tag(key))
        tag.fields.update(ev["fields"])
        tag.at = str(ev.get("at") or tag.at)
        tag.by = str(ev.get("by") or tag.by)
        tag.events += 1
    return tags
