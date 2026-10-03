"""Training-set selection and splitting shared by the export preview and the export builder (Task 13).

Both functions are pure over database rows so the preview shows exactly what the builder will produce.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import CollectionItem, Customer, Device, Event
from .schemas import ExportExclusion, ExportRequest

SPLIT_ORDER = ("train", "val", "test")


def _day(ev: Event) -> str:
    return ev.day or datetime.fromtimestamp(ev.start_ts, timezone.utc).strftime("%Y-%m-%d")


def select_export_items(session: Session, collection_id: int, request: ExportRequest):
    """Events of the collection split into (included, excluded).

    Reasons, checked in this order: `no_training_consent` (always), `expired`, `video_unavailable`, and
    `no_real_ai` -- only for VLM-only exports (formats == ["vlm_jsonl"]) without `include_fallback_ai`.
    With other formats, fallback/failed AI only drops the event from vlm.jsonl; clips and YOLO labels stay.
    Returns (list[Event] ordered by id, list[ExportExclusion] ordered by event id).
    """
    rows = session.execute(
        select(Event, Customer.consent_training)
        .join(CollectionItem, CollectionItem.event_id == Event.id)
        .join(Device, Device.id == Event.device_pk).join(Customer, Customer.id == Device.customer_id)
        .where(CollectionItem.collection_id == collection_id).order_by(Event.id)).all()
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


def _group_key(ev: Any) -> str:
    return f"{ev.site}|{_day(ev)}"


def assign_splits(items: Iterable[Any], name: str, split: dict[str, float]) -> dict[int, str]:
    """Map event id -> split name. Events of one (site, day) group always share a split.

    The group's bucket is a deterministic number in [0, 1): sha256(name + group) read as a fraction, compared
    against the cumulative split fractions in train, val, test order (then any other keys, sorted).
    """
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
            digest = hashlib.sha256((name + key).encode("utf-8")).digest()
            u = int.from_bytes(digest[:8], "big") / 2 ** 64
            cache[key] = next((n for edge, n in bounds if u < edge), bounds[-1][1])
        out[ev.id] = cache[key]
    return out


def group_count(items: Iterable[Any]) -> int:
    return len({_group_key(ev) for ev in items})
