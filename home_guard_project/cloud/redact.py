"""Household identity redaction for labelers.

Labelers label clips; they must never learn which household a clip comes from. Free text written by the box or
the AI (summaries, alert reasons, prompts, model output) can name the site, a camera, the owner's own camera
labels or the customer, so every such string a labeler sees is passed through `redact_text` first. Structured
data is never redacted by deleting fields: labeler responses are built from allowlisted projections, and only
their free-text values go through here.

Identity terms of a device: its site, every camera name (Camera rows, cameras seen in events, heartbeat
cameras), the owner's camera display names, the customer name and the Tailscale host; plus spelling variants
(`_`, space, `-` or nothing between the parts, so "bian_house" also catches "BianHouse") and each distinctive
part of at least 4 characters that is not a common word ("bian" of "bian_ch2", "Levi" of "Daniel Levi").
Matching is case-insensitive and whole-word-ish: a term must not continue an alphanumeric run, except at a
camelCase boundary ("MyBianHome" still loses "Bian"); `_` and punctuation count as boundaries, so
"dataset_bian" loses "bian" too.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract.legacy import parse_heartbeat

from . import pseudonym
from .models import Camera, Customer, Device, Event

# Parts of names that are ordinary words: redacting them would mangle every summary without hiding anyone.
STOPWORDS = frozenset({
    "front", "back", "side", "door", "main", "entrance", "left", "right", "garden", "driveway", "yard", "gate",
    "house", "home", "camera", "street", "garage", "porch", "parking",
    "rear", "room", "kitchen", "window", "inside", "outside", "north", "south", "east", "west", "balcony",
    "corner", "entry", "floor", "office", "roof", "stairs", "upper", "lower", "middle", "center",
})
MIN_PART = 4

# What a labeler sees instead of a term when no server secret is at hand (the stored search copy).
NEUTRAL_CUSTOMER = "customer-redacted"
NEUTRAL_CAMERA = "cam-redacted"

_SEPARATORS = re.compile(r"[\s_.\-]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_START = r"(?:(?<![A-Za-z0-9])|(?<=[a-z0-9])(?=[A-Z]))"
_END = r"(?:(?![A-Za-z0-9])|(?<=[a-z0-9])(?=[A-Z]))"


def variants(term: str) -> list[str]:
    """`term` and its spellings: separators swapped or dropped, and its distinctive parts."""
    term = (term or "").strip()
    if not term:
        return []
    out = {term}
    parts = [p for p in _SEPARATORS.split(term) if p]
    tokens = [t for p in parts for t in _CAMEL.split(p) if t]
    for seq in (parts, tokens):
        if len(seq) > 1:
            out.update(sep.join(seq) for sep in ("_", " ", "-", ""))
    for t in set(parts) | set(tokens):
        if len(t) >= MIN_PART and not t.isdigit() and t.lower() not in STOPWORDS:
            out.add(t)
    return sorted(v for v in out if len(v) >= 2)


def _sources(session: Session, device: Device) -> tuple[list[str], list[tuple[str, Optional[str]]]]:
    """(customer-level names, [(camera-level name, camera it stands for)])."""
    customer = session.get(Customer, device.customer_id) if device.customer_id is not None else None
    whole = [device.site, device.tailscale_host or "", customer.name if customer is not None else ""]
    cams: set[str] = set(session.scalars(select(Camera.name).where(Camera.device_pk == device.id)))
    cams |= set(session.scalars(select(Event.camera).where(Event.device_pk == device.id).distinct()))
    if isinstance(device.last_heartbeat, dict):
        cams |= set(parse_heartbeat(device.last_heartbeat).cameras)
    per_camera: list[tuple[str, Optional[str]]] = [(c, c) for c in sorted(cams) if c]
    for name, shown in session.execute(select(Camera.name, Camera.display_name).where(
            Camera.device_pk == device.id, Camera.display_name.is_not(None))).all():
        if shown and shown.strip():
            per_camera.append((shown.strip(), name))
    return [w for w in whole if w], per_camera


def identity_terms(session: Session, device: Device) -> list[str]:
    """Every identity term of the device, variants included, longest first."""
    return identity(session, device).terms


@dataclass(frozen=True)
class Identity:
    """The identity terms of one device and what each becomes for a labeler."""
    terms: list[str]
    mapping: dict[str, str]  # lower-cased term -> replacement

    def text(self, value: Optional[str]) -> str:
        return redact_text(value or "", self.terms, self.mapping)

    def json(self, value: Any) -> Any:
        return redact_json(value, self.terms, self.mapping)

    def mentions(self, value: str) -> bool:
        return bool(self.terms) and _pattern(tuple(self.terms)).search(value or "") is not None


def identity(session: Session, device: Device, secret: Optional[str] = None) -> Identity:
    """The device's identity terms; with `secret` they map to the labeler pseudonyms, else to neutral words."""
    whole, per_camera = _sources(session, device)
    customer_p = pseudonym.customer(secret, device.customer_id) if secret else NEUTRAL_CUSTOMER
    mapping: dict[str, str] = {}
    for name in whole:  # customer-level names first: a part shared with a camera name stays the customer's
        for v in variants(name):
            mapping.setdefault(v.lower(), customer_p)
    for name, camera in per_camera:
        cam_p = pseudonym.camera(secret, device.site, camera) if secret and camera else NEUTRAL_CAMERA
        for v in variants(name):
            mapping.setdefault(v.lower(), cam_p)
    terms = sorted(mapping, key=lambda t: (-len(t), t))
    return Identity(terms=terms, mapping=mapping)


@lru_cache(maxsize=256)
def _pattern(terms: tuple[str, ...]) -> re.Pattern:
    alternation = "|".join(re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t)))
    return re.compile(f"{_START}(?i:{alternation}){_END}")


def redact_text(text: str, terms: Iterable[str], mapping: dict[str, str]) -> str:
    """`text` with every identity term replaced by its pseudonym, longest terms first, in a single pass."""
    terms = tuple(terms)
    if not text or not terms:
        return text or ""
    return _pattern(terms).sub(lambda m: mapping.get(m.group(0).lower(), NEUTRAL_CUSTOMER), text)


def redact_json(value: Any, terms: Iterable[str], mapping: dict[str, str]) -> Any:
    """A copy of a JSON value with every string (dict keys included) redacted."""
    terms = tuple(terms)
    if isinstance(value, str):
        return redact_text(value, terms, mapping)
    if isinstance(value, dict):
        return {redact_text(str(k), terms, mapping): redact_json(v, terms, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_json(v, terms, mapping) for v in value]
    return value


# ---------------------------------------------------------------- the stored search copy

def backfill(session: Session, device: Device, everything: bool = False) -> int:
    """Fill `Event.summary_redacted` (the labelers' search text) where it is missing, or for every event of the
    device with `everything` (after a camera or customer rename). Returns how many events were written."""
    ident = identity(session, device)
    stmt = select(Event).where(Event.device_pk == device.id)
    if not everything:
        stmt = stmt.where(Event.summary_redacted.is_(None))
    n = 0
    for ev in session.scalars(stmt):
        ev.summary_redacted = ident.text(ev.summary)
        n += 1
    session.flush()
    return n
