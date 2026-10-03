"""Household identity redaction for labelers.

Labelers label clips; they must never learn which household a clip comes from. Free text written by the box or
the AI (summaries, alert reasons, prompts, model output) can name the site, a camera, the owner's own camera
labels or the customer, so every such string a labeler sees is passed through `redact_text` first. Structured
data is never redacted by deleting fields: labeler responses are built from allowlisted projections, and only
their free-text values go through here.

Identity terms of a device: its site, every camera name (Camera rows, cameras seen in events, heartbeat
cameras), the owner's camera display names, the customer name, the Tailscale host and the heartbeat host -- now
and in the past: every such name is also kept in `identity_aliases` (recorded at enrolment, on a customer rename,
and by every index pass), and the union is used, so a rename never brings an old name back. History older than that table (audit log
renames, raw JSON revisions, enrolment sites; `legacy_names`) is read into it by migration 0008, and lazily, once
per device, by `ensure_history` as a safety net; plus spelling variants
(`_`, space, `-` or nothing between the parts, so "bian_house" also catches "BianHouse") and each distinctive
part of at least 4 characters that is not a common word ("bian" of "bian_ch2", "Levi" of "Daniel Levi").
Matching is case-insensitive and whole-word-ish: a term must not continue an alphanumeric run, except at a
case or digit boundary ("MyBianHome", "MyBIANHome", "BIANhome" and "BIAN2" all lose "BIAN"); `_` and punctuation
count as boundaries, so "dataset_bian" loses "bian" too.

Redaction is defence in depth: labelers get no search and no prompt text, and the enum/format fields they see are
validated here (`alert_command`, `label`) rather than redacted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable, Optional

from sqlalchemy import column, select, table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract.legacy import parse_heartbeat

from . import pseudonym
from .models import Camera, Customer, Device, Event, IdentityAlias

# Parts of names that are ordinary words: redacting them would mangle every summary without hiding anyone.
STOPWORDS = frozenset({
    "front", "back", "side", "door", "main", "entrance", "left", "right", "garden", "driveway", "yard", "gate",
    "house", "home", "camera", "street", "garage", "porch", "parking",
    "rear", "room", "kitchen", "window", "inside", "outside", "north", "south", "east", "west", "balcony",
    "corner", "entry", "floor", "office", "roof", "stairs", "upper", "lower", "middle", "center",
})
MIN_PART = 4

# The only values these fields may take; anything else is shown (and stored) as null.
ALERT_COMMANDS = frozenset({"[none]", "[send_message]", "[call_owner]"})
LABELS = frozenset({"normal", "suspicious", "escalation"})
# Owner verdicts: stored values outside this set become "unknown" at ingestion (and are never matched by filters).
VERDICTS = frozenset({"true_alert", "false_alarm", "real_but_wrong", "expected", "missed_event", "none"})
UNKNOWN_VERDICT = "unknown"


def verdict(value: Any) -> str:
    """`value` when it is one of the owner verdicts, else "unknown"."""
    return value if isinstance(value, str) and value in VERDICTS else UNKNOWN_VERDICT
# The box writes "YYYY-MM-DD HH:MM:SS" (optionally ISO with T, fraction, offset); imports write "<seconds>s".
_CLIP_START_LOCAL = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?|\d{1,7}(?:\.\d{1,3})?s")


def alert_command(value: Optional[str]) -> Optional[str]:
    """`value` when it is one of the bracketed alert commands, else None."""
    return value if value in ALERT_COMMANDS else None


def label(value: Optional[str]) -> Optional[str]:
    """`value` when it is one of the AI's scene labels, else None."""
    return value if value in LABELS else None


def clip_start_local(value: Optional[str]) -> Optional[str]:
    """`value` when it is a local timestamp (or an import's offset in seconds), else None."""
    return value if isinstance(value, str) and _CLIP_START_LOCAL.fullmatch(value) else None

# What a labeler sees instead of a term when no server secret is at hand (the stored search copy).
NEUTRAL_CUSTOMER = "customer-redacted"
NEUTRAL_CAMERA = "cam-redacted"

_SEPARATORS = re.compile(r"[\s_.\-]+")
# Source tokenizer: every case or digit boundary of a name splits it ("OAKRIDGEHome" -> OAKRIDGE|Home,
# "BIAN2" -> BIAN|2, "myBian" -> my|Bian). An all-caps run followed by lower case with no capital ("BIANhome")
# has no unambiguous split; the matcher's `_INNER` boundaries still catch it inside text.
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
# Word boundaries inside an alphanumeric run: lower -> Upper ("myBian"), an acronym ending where a capitalised word
# starts ("BIAN|Home"), an upper-case run of two or more ending in lower case ("BIAN|home") and letter <-> digit
# ("BIAN|2"). A boundary only lets a term match there; it never removes a character.
_INNER = (r"(?<=[a-z0-9])(?=[A-Z])", r"(?<=[A-Z])(?=[A-Z][a-z])", r"(?<=[A-Z]{2})(?=[a-z])",
          r"(?<=[A-Za-z])(?=[0-9])", r"(?<=[0-9])(?=[A-Za-z])")
_START = "(?:" + "|".join((r"(?<![A-Za-z0-9])",) + _INNER) + ")"
_END = "(?:" + "|".join((r"(?![A-Za-z0-9])",) + _INNER) + ")"


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


def _current(session: Session, device: Device) -> tuple[set[tuple[str, str]], list[tuple[str, str]]]:
    """The device's identity names as of now: ({(kind, value)}, [(display name, camera it stands for)])."""
    customer = session.get(Customer, device.customer_id) if device.customer_id is not None else None
    names = {("site", device.site), ("host", device.tailscale_host or ""),
             ("customer", customer.name if customer is not None else "")}
    cams: set[str] = set(session.scalars(select(Camera.name).where(Camera.device_pk == device.id)))
    cams |= set(session.scalars(select(Event.camera).where(Event.device_pk == device.id).distinct()))
    if isinstance(device.last_heartbeat, dict):
        hb = parse_heartbeat(device.last_heartbeat)
        cams |= set(hb.cameras)
        names.add(("host", hb.host or ""))
    names |= {("camera", c) for c in cams}
    shown: list[tuple[str, str]] = []
    for name, display in session.execute(select(Camera.name, Camera.display_name).where(
            Camera.device_pk == device.id, Camera.display_name.is_not(None))).all():
        if display and display.strip():
            shown.append((display.strip(), name))
            names.add(("display_name", display.strip()))
    return {(k, v) for k, v in names if v}, shown


# ---------------------------------------------------------------- names in older history

# The marker row (kind, value) saying a device's older history has been read into identity_aliases.
SCANNED = ("_scanned", "legacy")
MAX_ALIAS = 200  # longer strings are not names
# Fields of the box's JSON bodies (meta, feedback, heartbeat) that name the household or a camera.
BODY_FIELDS = (("camera_name", "camera"), ("site", "site"), ("prompt_camera_name", "display_name"),
               ("host", "host"), ("camera", "camera"))


def _name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if 0 < len(value) <= MAX_ALIAS else None


def body_names(body: Any) -> set[tuple[str, str]]:
    """The (kind, name) pairs a box JSON body carries: camera_name, site, prompt_camera_name, host, a feedback's
    camera and alert camera, a heartbeat's camera names."""
    out: set[tuple[str, str]] = set()
    if not isinstance(body, dict):
        return out
    pairs = [(kind, body.get(field)) for field, kind in BODY_FIELDS]
    if isinstance(body.get("alert"), dict):
        pairs.append(("camera", body["alert"].get("camera")))
    if isinstance(body.get("cameras"), dict):
        pairs += [("camera", k) for k in body["cameras"]]
    for kind, value in pairs:
        name = _name(value)
        if name:
            out.add((kind, name))
    return out


def _like_prefix(prefix: str) -> str:
    """A LIKE pattern (escape character `!`) for keys starting with `prefix`."""
    return prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


# (kind, SQL of the JSON value, SQL of its text) for each name-bearing field of raw_revisions.body
_RAW_FIELDS = [(kind, f"r.body->'{field}'", f"r.body->>'{field}'") for field, kind in BODY_FIELDS] + [
    ("camera", "r.body->'alert'->'camera'", "r.body->'alert'->>'camera'")]
_UNDER = "(r.s3_key LIKE :p1 ESCAPE '!' OR r.s3_key LIKE :p2 ESCAPE '!')"
# the table as plain columns (not the model): a migration writes it too
_ALIASES = table("identity_aliases", column("device_pk"), column("kind"), column("value"), column("first_seen"))


def legacy_names(conn, device_pk: int) -> set[tuple[str, str]]:
    """Every (kind, name) the device's older history holds, read with plain SQL (`conn` is a Connection or a
    Session, so a migration can use it): the customer's names in the audit log (customer_create and
    customer_update targets, and the old and new names of a logged rename), the site of its enrolment, the raw
    JSON revisions under its prefixes (camera_name, site, prompt_camera_name, host, feedback cameras, heartbeat
    camera names), the camera of every indexed object, and a `camera_aliases` table when one exists."""
    dev = conn.execute(text("SELECT site, device_id, customer_id FROM devices WHERE id = :pk"),
                       {"pk": device_pk}).first()
    if dev is None:
        return set()
    site, device_id, customer_id = dev
    out: set[tuple[str, str]] = set()

    def add(kind: str, value: Any) -> None:
        name = _name(value)
        if name:
            out.add((kind, name))

    for action, target, detail in conn.execute(text(
            "SELECT action, target, detail FROM audit_log "
            "WHERE (customer_id = :c AND action IN ('customer_create', 'customer_update')) "
            "OR (device_id = :d AND action = 'device_enroll')"), {"c": customer_id, "d": device_id}):
        if action == "device_enroll":
            add("site", target)
            continue
        add("customer", target)
        changed = detail.get("changed") if isinstance(detail, dict) else None
        if isinstance(changed, dict) and isinstance(changed.get("name"), (list, tuple)):
            for value in changed["name"]:
                add("customer", value)
    prefixes = {"p1": _like_prefix(f"production_{site}/"), "p2": _like_prefix(f"dataset_{site}/")}
    columns = ", ".join(f"CASE WHEN jsonb_typeof({js}) = 'string' THEN {tx} END" for _, js, tx in _RAW_FIELDS)
    for row in conn.execute(text(f"SELECT DISTINCT {columns} FROM raw_revisions r WHERE {_UNDER}"), prefixes):
        for (kind, _, _), value in zip(_RAW_FIELDS, row):
            add(kind, value)
    for (name,) in conn.execute(text(
            "SELECT DISTINCT k FROM raw_revisions r, LATERAL jsonb_object_keys(CASE WHEN "
            "jsonb_typeof(r.body->'cameras') = 'object' THEN r.body->'cameras' ELSE '{}'::jsonb END) AS k "
            f"WHERE {_UNDER}"), prefixes):
        add("camera", name)
    for (name,) in conn.execute(text(f"SELECT DISTINCT r.camera FROM artifacts r WHERE {_UNDER}"), prefixes):
        add("camera", name)
    if conn.execute(text("SELECT to_regclass('camera_aliases')")).scalar() is not None:
        columns = dict(conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'camera_aliases'")
        ).all())
        texts = [c for c, kind in columns.items() if kind in ("text", "character varying")]
        if "device_pk" in columns and texts:
            cols = ", ".join('"' + c.replace('"', '""') + '"' for c in texts)
            for row in conn.execute(text(f"SELECT {cols} FROM camera_aliases WHERE device_pk = :pk"),
                                    {"pk": device_pk}):
                for value in row:
                    add("camera", value)
    return out


def scan_legacy(conn, device_pk: int, now: Optional[datetime] = None) -> int:
    """Read the device's older history (`legacy_names`) into identity_aliases and mark it read; returns how many
    names were new. When there are any, the stored labeler search text of its events is cleared, so the next
    index pass recomputes it with the whole history."""
    when = now or datetime.now(timezone.utc)
    rows = [{"device_pk": device_pk, "kind": k, "value": v, "first_seen": when}
            for k, v in sorted(legacy_names(conn, device_pk) | {SCANNED})]
    stmt = pg_insert(_ALIASES).values(rows).on_conflict_do_nothing().returning(_ALIASES.c.kind)
    new = sum(1 for (kind,) in conn.execute(stmt).all() if kind != SCANNED[0])
    if new:
        conn.execute(text("UPDATE events SET summary_redacted = NULL WHERE device_pk = :d"), {"d": device_pk})
    return new


def ensure_history(session: Session, device: Device) -> None:
    """The safety net behind migration 0008: a device whose older history was never read (no marker row) has it
    read now, once."""
    if device is None or device.id is None:
        return
    marked = session.scalar(select(IdentityAlias.id).where(
        IdentityAlias.device_pk == device.id, IdentityAlias.kind == SCANNED[0]).limit(1))
    if marked is None:
        scan_legacy(session, device.id)


def _history(session: Session, device: Device) -> list[tuple[str, str]]:
    """Every (kind, name) recorded for the device, plus the customer names recorded for the customer's other
    devices."""
    siblings = select(Device.id).where(Device.customer_id == device.customer_id)
    rows = session.execute(select(IdentityAlias.kind, IdentityAlias.value).where(
        (IdentityAlias.device_pk == device.id)
        | ((IdentityAlias.kind == "customer") & IdentityAlias.device_pk.in_(siblings)))).all()
    return sorted({(k, v) for k, v in rows})


def remember(session: Session, device: Device, extra: Iterable[tuple[str, str]] = (),
             now: Optional[datetime] = None) -> int:
    """Record the device's current identity names (and `extra` (kind, value) pairs) in `identity_aliases`;
    returns how many were new. One statement; existing aliases are left as they are."""
    names, _ = _current(session, device)
    names |= {(k, v.strip()) for k, v in extra if v and v.strip()}
    if not names:
        return 0
    when = now or datetime.now(timezone.utc)
    stmt = (pg_insert(IdentityAlias)
            .values([{"device_pk": device.id, "kind": k, "value": v, "first_seen": when} for k, v in sorted(names)])
            .on_conflict_do_nothing(index_elements=["device_pk", "kind", "value"])
            .returning(IdentityAlias.id))
    return len(session.execute(stmt).all())


def _sources(session: Session, device: Device) -> tuple[list[str], list[tuple[str, Optional[str]]]]:
    """(customer-level names, [(camera-level name, camera it stands for)]), current names first, then history
    (the device's older history is read into identity_aliases first if it never was)."""
    ensure_history(session, device)
    names, shown = _current(session, device)
    history = _history(session, device)
    whole_kinds = ("site", "customer", "host")
    whole = [v for k, v in sorted(names) if k in whole_kinds]
    whole += [v for k, v in history if k in whole_kinds and v not in whole]
    per_camera: list[tuple[str, Optional[str]]] = [(c, c) for k, c in sorted(names) if k == "camera"]
    per_camera += shown
    per_camera += [(c, c) for k, c in history if k == "camera"]
    per_camera += [(v, None) for k, v in history if k == "display_name"]  # a past display name: camera unknown
    return whole, per_camera


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
