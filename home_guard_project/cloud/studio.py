"""Training-set selection and splitting shared by the export preview and the export builder (Task 13).

Both functions are pure over database rows so the preview shows exactly what the builder will produce.
"""
from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import CollectionItem, Customer, Device, Event
from .schemas import ExportExclusion, ExportRequest

SPLIT_ORDER = ("train", "val", "test")


def _day(ev: Event) -> str:
    return ev.day or datetime.fromtimestamp(ev.start_ts, timezone.utc).strftime("%Y-%m-%d")


def select_export_items(session: Session, collection_id: int, request: ExportRequest, labeler: bool = False):
    """Events of the collection split into (included, excluded).

    Reasons, checked in this order: `no_training_consent` (always), `expired`, `video_unavailable`, and
    `no_real_ai` -- only for VLM-only exports (formats == ["vlm_jsonl"]) without `include_fallback_ai`.
    With other formats, fallback/failed AI only drops the event from vlm.jsonl; clips and YOLO labels stay.
    For a `labeler`, events they may not see (no training consent) are dropped before anything else: they appear
    neither in the result nor as an exclusion, so the preview is that of the visible events alone.
    Returns (list[Event] ordered by id, list[ExportExclusion] ordered by event id).
    """
    stmt = (select(Event, Customer.consent_training)
            .join(CollectionItem, CollectionItem.event_id == Event.id)
            .join(Device, Device.id == Event.device_pk).join(Customer, Customer.id == Device.customer_id)
            .where(CollectionItem.collection_id == collection_id))
    if labeler:
        stmt = stmt.where(Customer.consent_training.is_(True))
    rows = session.execute(stmt.order_by(Event.id)).all()
    vlm_only = list(request.formats) == ["vlm_jsonl"]
    included: list[Event] = []
    excluded: list[ExportExclusion] = []
    for ev, consent in rows:
        comp = ev.completeness if isinstance(ev.completeness, dict) else {}
        reason = None
        if not consent:
            reason = "no_training_consent"
        elif comp.get("expired"):
            reason = "expired"
        elif not comp.get("video"):
            reason = "video_unavailable"
        elif vlm_only and not request.include_fallback_ai and comp.get("ai") != "real":
            reason = "no_real_ai"
        if reason:
            excluded.append(ExportExclusion(event_id=ev.id, reason=reason))
        else:
            included.append(ev)
    return included, excluded


_SPLIT_DOMAIN = b"home-guard-admin/export-split/v1"  # not the pseudonym domain: the digests never coincide


def _group_key(ev: Any) -> str:
    return f"{ev.site}\x00{_day(ev)}"


def _split_key(secret: str) -> bytes:
    """A sub-key of the server secret used only for export splits."""
    return hmac.new(secret.encode("utf-8"), _SPLIT_DOMAIN, hashlib.sha256).digest()


def assign_splits(items: Iterable[Any], name: str, split: dict[str, float], *, secret: str) -> dict[int, str]:
    """Map event id -> split name. Events of one (site, day) group always share a split.

    The group's bucket is a deterministic number in [0, 1): HMAC-SHA256 under a sub-key of the server secret
    (`secret`, the JWT secret) over (export name, site, day), read as a fraction and compared against the
    cumulative split fractions in train, val, test order (then any other keys, sorted). Keyed, because an
    unkeyed hash lets anyone who can choose export names recompute the split offline for candidate sites and
    so identify the household of an event they only know by its pseudonym.
    """
    sub_key = _split_key(secret)
    names = [n for n in SPLIT_ORDER if n in split] + sorted(n for n in split if n not in SPLIT_ORDER)
    names = [n for n in names if split[n] > 0] or names
    total = sum(split[n] for n in names) or 1.0
    bounds, acc = [], 0.0
    for n in names:
        acc += split[n] / total
        bounds.append((acc, n))
    out: dict[int, str] = {}
    cache: dict[str, str] = {}
    for ev in items:
        key = _group_key(ev)
        if key not in cache:
            digest = hmac.new(sub_key, f"{name}\x00{key}".encode("utf-8"), hashlib.sha256).digest()
            u = int.from_bytes(digest[:8], "big") / 2 ** 64
            cache[key] = next((n for edge, n in bounds if u < edge), bounds[-1][1])
        out[ev.id] = cache[key]
    return out


def group_count(items: Iterable[Any]) -> int:
    return len({_group_key(ev) for ev in items})
