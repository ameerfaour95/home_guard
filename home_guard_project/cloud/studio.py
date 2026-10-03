"""Training studio: built-in filters, training-set selection and splitting, and the versioned export builder.

`select_export_items` and `assign_splits` are shared by the export preview and the builder, so the preview shows
exactly what an export will contain.

Exports are written to `training_exports/<name>/v<N>/` and never name a household: manifest items carry customer
and camera pseudonyms, an opaque group id and opaque artifact references; files are named by event id; free text
(the teacher prompt, the AI answer, owner verdicts) is redacted. Provenance (sites, cameras, stems, S3 keys) goes
only to `_private/mapping.json`, which no route serves.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import socket
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, get_args

from sqlalchemy import Float, and_, case, cast, func, or_, select, text, type_coerce, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract.classes import COCO_NAMES, CONTIGUOUS
from home_guard_project.fleet_contract.yolo_labels import parse_label

from . import labeling, media, pseudonym, redact
from .models import (AiRun, AnnotationHead, Artifact, CollectionItem, Customer, Device, Event, Export, Feedback,
                     RawRevision, Staff)
from .s3 import ETagMismatch
from .schemas import EventKind, ExportExclusion, ExportRequest, SavedFilter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- built-in filters

def has_verdict(verdict: str):
    """Condition: the owner gave `verdict` on the event."""
    return type_coerce(Event.owner_verdicts, JSONB).contains([verdict])


def max_class_conf():
    """The event's highest per-class detector confidence (NULL when there is none)."""
    conf = type_coerce(Event.class_max_conf, JSONB)
    each = func.jsonb_each_text(conf).table_valued("key", "value")
    best = select(func.max(cast(each.c.value, Float))).select_from(each).scalar_subquery()
    return case((func.jsonb_typeof(conf) == "object", best), else_=None)


LOW_CONF = 0.45
LABEL_DONE = labeling.DONE  # a clip counts as labeled once its current annotation is submitted or reviewed


def _head_where(*conds):
    """Condition over Event: its current annotation (annotation_heads) matches `conds`."""
    return Event.id.in_(select(AnnotationHead.event_id).where(*conds))


# key -> (title, description, SQL condition over Event)
_BUILTINS: dict[str, tuple[str, str, Callable[[], Any]]] = {
    "false_alarm": ("Owner said false alarm", "Events the owner marked as a false alarm.",
                    lambda: has_verdict("false_alarm")),
    "ai_dismissed_person": ("AI dismissed, YOLO saw a person",
                            "The AI called it a false positive although the detector saw a person.",
                            lambda: and_(Event.kind == "false_positive",
                                         type_coerce(Event.detected, JSONB).contains(["person"]))),
    "ai_failed": ("AI call failed or fell back", "The AI answer failed or a fallback answer was used.",
                  lambda: Event.completeness["ai"].as_string().in_(["failed", "fallback"])),
    "real_but_wrong": ("Owner said real but wrong",
                       "The owner said the alert was real but the description or action was wrong.",
                       lambda: has_verdict("real_but_wrong")),
    "low_conf": ("Low detector confidence", f"The detector's best class confidence is below {LOW_CONF}.",
                 lambda: max_class_conf() < LOW_CONF),
    "paused": ("Paused-camera footage", "Clips recorded while the camera's alerts were paused.",
               lambda: Event.kind == "paused"),
    "needs_labeling": ("Needs labeling", "Clips without a submitted or reviewed annotation.",
                       lambda: ~_head_where(AnnotationHead.status.in_(LABEL_DONE))),
    "to_review": ("Labels to review", "Submitted annotations, and annotations a labeler asked to have reviewed.",
                  lambda: _head_where(or_(AnnotationHead.status == "submitted",
                                          and_(AnnotationHead.needs_review.is_(True),
                                               AnnotationHead.status.notin_(("reviewed", "rejected")))))),
    "rejected": ("Labels rejected", "Annotations a reviewer sent back.",
                 lambda: _head_where(AnnotationHead.status == "rejected")),
}

BUILTIN_FILTERS: list[SavedFilter] = [
    SavedFilter(key=key, title=title, description=description, builtin=True, query={"filter": key})
    for key, (title, description, _) in _BUILTINS.items()
]


def builtin_condition(key: str):
    """The SQL condition of a built-in filter, or None for an unknown key."""
    entry = _BUILTINS.get(key)
    return entry[2]() if entry is not None else None

SPLIT_ORDER = ("train", "val", "test")


def _day(ev: Event) -> str:
    return ev.day or datetime.fromtimestamp(ev.start_ts, timezone.utc).strftime("%Y-%m-%d")


def select_export_items(session: Session, collection_id: int, request: ExportRequest, labeler: bool = False):
    """Events of the collection split into (included, excluded).

    Reasons, checked in this order: `no_training_consent` (always), `dropped_by_labeler` (the clip's submitted or
    reviewed annotation asks to drop it), `expired` (no copy of the video is left and
    the production retention has passed: a training copy that outlives the production one stays exportable),
    `video_unavailable`, and
    `no_real_ai` -- only for VLM-only exports (formats == ["vlm_jsonl"]) without `include_fallback_ai`, and never
    for a clip with a submitted or reviewed annotation (its human description is the answer).
    With other formats, fallback/failed AI only drops the event from vlm.jsonl; clips and YOLO labels stay.
    For a `labeler`, events they may not see (no training consent) are dropped before anything else: they appear
    neither in the result nor as an exclusion, so the preview is that of the visible events alone.
    Returns (list[Event] ordered by id, list[ExportExclusion] ordered by event id).
    """
    stmt = (select(Event, Customer.consent_training, AnnotationHead.status, AnnotationHead.drop_clip)
            .join(CollectionItem, CollectionItem.event_id == Event.id)
            .join(Device, Device.id == Event.device_pk).join(Customer, Customer.id == Device.customer_id)
            .outerjoin(AnnotationHead, AnnotationHead.event_id == Event.id)
            .where(CollectionItem.collection_id == collection_id))
    if labeler:
        stmt = stmt.where(Customer.consent_training.is_(True))
    rows = session.execute(stmt.order_by(Event.id)).all()
    vlm_only = list(request.formats) == ["vlm_jsonl"]
    included: list[Event] = []
    excluded: list[ExportExclusion] = []
    for ev, consent, label_status, drop_clip in rows:
        comp = ev.completeness if isinstance(ev.completeness, dict) else {}
        reason = None
        if not consent:
            reason = "no_training_consent"
        elif label_status in LABEL_DONE and drop_clip:
            reason = "dropped_by_labeler"
        elif comp.get("expired"):
            reason = "expired"
        elif not comp.get("video"):
            reason = "video_unavailable"
        elif (vlm_only and not request.include_fallback_ai and comp.get("ai") != "real"
              and label_status not in LABEL_DONE):  # a human description needs no AI answer
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


def group_id(ev: Any, name: str, secret: str) -> str:
    """Opaque id of the event's (site, day) group in export `name`: keyed, so it names no site."""
    digest = hmac.new(_split_key(secret), f"group\x00{name}\x00{_group_key(ev)}".encode("utf-8"), hashlib.sha256)
    return "group-" + digest.hexdigest()[:12]



# ---------------------------------------------------------------- request validation (preview and create share it)

NAME_IDENTIFIES = "Name must not identify a household"
SPLIT_KEYS_ERROR = "Split keys must be train, val and test"
SPLIT_VALUES_ERROR = "Split fractions must be finite, non-negative and add up to 1, with train above 0"
CLIPS_ADDED = "VLM lines need their clips: clips were added to the export."
MIN_GROUPS = 3
_SPLIT_EPS = 1e-6


class RequestError(ValueError):
    """A request the studio refuses (HTTP 400); the message is shown as is."""


def identifies_household(session: Session, value: Optional[str]) -> bool:
    """Whether `value` mentions an identity term of ANY household (every device, and customers without one)."""
    if not value or not value.strip():
        return False
    for device in session.scalars(select(Device).order_by(Device.id)):
        terms = redact.identity_terms(session, device)
        if terms and redact.redact_text(value, terms, {}) != value:
            return True
    lonely = select(Customer.name).where(~select(Device.id).where(Device.customer_id == Customer.id).exists())
    for name in session.scalars(lonely):
        terms = redact.variants(name or "")
        if terms and redact.redact_text(value, terms, {}) != value:
            return True
    return False


def effective_formats(formats: Iterable[str]) -> list[str]:
    """The formats an export writes: the requested ones, plus `clips` whenever `vlm_jsonl` is asked for (a VLM
    line points at its clip, which must be in the bundle)."""
    out = list(dict.fromkeys(formats))
    if "vlm_jsonl" in out and "clips" not in out:
        out.append("clips")
    return out


def validate_export_request(session: Session, body: ExportRequest, labeler: bool = False) -> list[str]:
    """Raise RequestError for a request neither the preview nor the export accepts; returns warnings.

    A labeler's export name is not checked against household identities: their exports are visible only to them
    and to admins, and a check that refuses names of households they cannot see would reveal those households."""
    if not body.formats:
        raise RequestError("Choose at least one format")
    # the name's length (at most deps.NAME_MAX, else 422) is checked by the routes, like collection names
    if not labeler and identifies_household(session, body.name):
        raise RequestError(NAME_IDENTIFIES)
    split = body.split or {}
    if not split or any(k not in SPLIT_ORDER for k in split):
        raise RequestError(SPLIT_KEYS_ERROR)
    values = list(split.values())
    if (any(not math.isfinite(v) or v < 0 for v in values)
            or abs(sum(values) - 1.0) > _SPLIT_EPS or not split.get("train", 0) > 0):
        raise RequestError(SPLIT_VALUES_ERROR)
    return [CLIPS_ADDED] if "vlm_jsonl" in body.formats and "clips" not in body.formats else []


def validate_collection_text(session: Session, *texts: Optional[str]) -> None:
    if any(identifies_household(session, t) for t in texts):
        raise RequestError(NAME_IDENTIFIES)


MAX_EXPORT_EVENTS = 20_000  # one export's selection, held in memory as its snapshot


def check_selection_size(session: Session, collection_id: int, labeler: bool = False) -> None:
    """Refuse a collection with more events (that this viewer may see) than one export can take."""
    stmt = (select(func.count()).select_from(CollectionItem).join(Event, Event.id == CollectionItem.event_id)
            .where(CollectionItem.collection_id == collection_id))
    if labeler:
        stmt = (stmt.join(Device, Device.id == Event.device_pk).join(Customer, Customer.id == Device.customer_id)
                .where(Customer.consent_training.is_(True)))
    if (session.scalar(stmt) or 0) > MAX_EXPORT_EVENTS:
        raise RequestError(f"An export can include at most {MAX_EXPORT_EVENTS} events; this collection has more. "
                           "Split it into smaller collections.")


def small_set_warnings(groups: int) -> list[str]:
    if groups < MIN_GROUPS:
        return [f"Only {groups} site+day groups: too few for train, val and test to be independent "
                f"(at least {MIN_GROUPS} are needed)."]
    return []


# ---------------------------------------------------------------- the snapshot taken at creation

EXPORT_ROOT = "training_exports/"
PLACEHOLDER_PROMPT = "<video>Describe what happens in this security camera clip."
YOLO_NAMES = [COCO_NAMES[coco] for coco in sorted(CONTIGUOUS, key=CONTIGUOUS.get)]
CLASS_MAP = {str(i): label for i, label in enumerate(YOLO_NAMES)}
CLASS_SCHEMA = "coco→contiguous-9"
SNAPSHOT_VERSION = 2  # 2: feedback values frozen with their source revision (1: feedback ids only)
_READABLE_SNAPSHOTS = (1, 2)
SPLIT_KEY_VERSION = 1
MANIFEST_SCHEMA = 2
_SNAP_ROLES = ("original_video", "rendition", "meta", "yolo_image", "yolo_label")
_KINDS = set(get_args(EventKind))
_FRAME = re.compile(r"_f([0-9]+)\.[A-Za-z0-9]+$")
_PATHISH = re.compile(r"\S*[\\/]\S*")
_MEDIA_TOKENS = re.compile(r"<(?:video|image|audio)>")
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_PAGE = 200


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _frame(key: str) -> Optional[int]:
    match = _FRAME.search(key)
    return int(match.group(1)) if match else None


def _training_first(art: Artifact):
    return (not art.s3_key.startswith("dataset_"), art.s3_key)


def _root(key: str) -> str:
    return key.split("/", 1)[0]


def _ref(art_id: int) -> str:
    return f"artifact-{art_id}"


def _art(art: Optional[Artifact]) -> Optional[dict]:
    if art is None:
        return None
    return {"id": art.id, "key": art.s3_key, "etag": art.etag, "bytes": art.bytes}


def _meta_body(session: Session, arts: list[Artifact]) -> dict:
    """The box's meta for the clip (newest revision, training copy first), or {}."""
    for art in sorted((a for a in arts if a.role == "meta"), key=_training_first):
        body = session.scalar(select(RawRevision.body).where(RawRevision.s3_key == art.s3_key)
                              .order_by(RawRevision.fetched_at.desc(), RawRevision.id.desc()).limit(1))
        if isinstance(body, dict):
            return body
    return {}


def _detector_model(meta: dict) -> Optional[str]:
    """The detector the box named in its meta, as a bare file name (a path could name a person)."""
    for holder, key in ((meta.get("yolo_export"), "model"), (meta.get("yolo"), "model"), (meta, "yolo_model"),
                        (meta.get("detector"), "model")):
        value = holder.get(key) if isinstance(holder, dict) else None
        if isinstance(value, str) and value.strip():
            return PurePosixPath(value.replace("\\", "/")).name[:64] or None
    return None


def _expected_frames(meta: dict) -> list[int]:
    export = meta.get("yolo_export") if isinstance(meta.get("yolo_export"), dict) else {}
    frames = export.get("exported_frames") if isinstance(export.get("exported_frames"), list) else []
    out = {f["frame_index"] for f in frames if isinstance(f, dict) and isinstance(f.get("frame_index"), int)
           and not isinstance(f.get("frame_index"), bool) and f["frame_index"] >= 0}
    return sorted(out)


def _choose_clip(arts: list[Artifact], meta: dict) -> Optional[dict]:
    """The original clip (training copy first); its H.264 rendition instead only when the original is not H.264
    and the rendition was made from exactly this original (`detail.src_etag` equals its current ETag)."""
    originals = sorted((a for a in arts if a.role == "original_video" and a.available), key=_training_first)
    if not originals:
        return None
    original = originals[0]
    codec = meta.get("codec") if isinstance(meta.get("codec"), str) else None
    if (codec or "").lower() != "h264" and original.etag:
        same = [a for a in arts if a.role == "rendition" and a.available
                and isinstance(a.detail, dict) and a.detail.get("src_etag") == original.etag]
        if same:
            return {**_art(max(same, key=lambda a: a.id)), "kind": "rendition"}
    return {**_art(original), "kind": "original"}


def _original_clip(arts: list[Artifact]) -> Optional[dict]:
    """The original clip (training copy first): human labels are per native frame of exactly this file."""
    originals = sorted((a for a in arts if a.role == "original_video" and a.available), key=_training_first)
    return _art(originals[0]) if originals else None


def _feedback_utc(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def _frozen_feedback(fb: Feedback, rev: Optional[str]) -> dict:
    """What the export will say about one piece of owner feedback, frozen at request time."""
    return {"id": fb.id, "verdict": fb.verdict, "action": fb.action, "received_utc": _feedback_utc(fb.received_at),
            "rev": rev}


def _run_digest(run: AiRun) -> str:
    blob = json.dumps([run.status, run.model, run.prompt_version, run.prompt, run.parsed, run.input_artifact_ids,
                       run.ai_source_etag], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def take_snapshot(session: Session, request: ExportRequest, *, labeler: bool, secret: str) -> dict:
    """Freeze what an export will contain: the included event ids and their splits (the same selection and split
    code as the preview), and per event the exact source revisions (artifact key + ETag), the AI run (id and
    digest) and the owner feedback (values and source revision). The build reads only this, so later changes
    cannot leak into a requested export: a source that changed is reported as `changed_since_request`."""
    included, excluded = select_export_items(session, request.collection_id, request, labeler=labeler)
    splits = assign_splits(included, request.name, request.split, secret=secret)
    formats = effective_formats(request.formats)
    events: list[dict] = []
    for chunk in _chunks(included, 500):
        ids = [ev.id for ev in chunk]
        arts: dict[int, list[Artifact]] = {}
        for art in session.scalars(select(Artifact).where(Artifact.event_id.in_(ids),
                                                          Artifact.role.in_(_SNAP_ROLES)).order_by(Artifact.id)):
            arts.setdefault(art.event_id, []).append(art)
        runs: dict[int, AiRun] = {}
        for run in session.scalars(select(AiRun).where(AiRun.event_id.in_(ids), AiRun.purpose == "guard")
                                   .order_by(AiRun.id)):
            runs[run.event_id] = run  # the newest wins
        feedback: dict[int, list[dict]] = {}
        for fb, rev in session.execute(select(Feedback, Artifact.applied_etag)
                                       .outerjoin(Artifact, Artifact.s3_key == Feedback.s3_key)
                                       .where(Feedback.event_id.in_(ids)).order_by(Feedback.id)):
            feedback.setdefault(fb.event_id, []).append(_frozen_feedback(fb, rev))
        customers = dict(session.execute(select(Device.id, Device.customer_id)
                                         .where(Device.id.in_({ev.device_pk for ev in chunk}))).all())
        heads = labeling.labeled_versions(session, ids)  # submitted or reviewed annotations win over weak labels
        for ev in chunk:
            mine = arts.get(ev.id, [])
            meta = _meta_body(session, mine)
            run = runs.get(ev.id)
            snap: dict[str, Any] = {
                "id": ev.id, "split": splits[ev.id], "customer_id": customers.get(ev.device_pk),
                "verdicts": [v for v in (ev.owner_verdicts or []) if isinstance(v, str)],
                "feedback": feedback.get(ev.id, []),
                "ai": ({"id": run.id, "status": run.status, "digest": _run_digest(run)}
                       if run is not None and run.status != "none" else None),
                "clip": _choose_clip(mine, meta) if "clips" in formats else None,
            }
            head = heads.get(ev.id)
            snap["annotation"] = ({"version": head.version, "status": head.status,
                                   "clip": _original_clip(mine) if "yolo" in formats else None}
                                  if head is not None and not head.drop_clip else None)
            if "yolo" in formats:
                labels: dict[int, Artifact] = {}
                images: dict[int, list[Artifact]] = {}
                for art in sorted(mine, key=_training_first):
                    frame = _frame(art.s3_key) if art.available else None
                    if frame is None:
                        continue
                    if art.role == "yolo_label":
                        labels.setdefault(frame, art)
                    elif art.role == "yolo_image":
                        images.setdefault(frame, []).append(art)
                expected = _expected_frames(meta) or sorted(labels)  # a meta naming no frames: the labels found
                pairs = []
                for frame in expected:
                    label = labels.get(frame)
                    cands = images.get(frame, [])
                    image = next((a for a in cands if label is not None and _root(a.s3_key) == _root(label.s3_key)),
                                 cands[0] if cands else None)
                    pairs.append({"frame": frame, "label": _art(label), "image": _art(image)})
                snap["frames"] = pairs
                snap["detector_model"] = _detector_model(meta)
            events.append(snap)
    return {"version": SNAPSHOT_VERSION, "split_key_version": SPLIT_KEY_VERSION, "formats": formats,
            "events": events, "excluded": [e.model_dump() for e in excluded]}


def _snapshot(export: Export) -> Optional[dict]:
    stored = export.request if isinstance(export.request, dict) else {}
    snap = stored.get("snapshot")
    return snap if isinstance(snap, dict) and snap.get("version") in _READABLE_SNAPSHOTS else None


def _consenting(session: Session, customer_ids: Iterable[Optional[int]]) -> set[int]:
    ids = {c for c in customer_ids if c is not None}
    if not ids:
        return set()
    return set(session.scalars(select(Customer.id).where(Customer.id.in_(ids),
                                                         Customer.consent_training.is_(True))))


def consent_withdrawn(session: Session, export: Export) -> bool:
    """Whether a household included in the export has since withdrawn training consent (or is gone)."""
    snap = _snapshot(export)
    if snap is None:
        return False
    customers = {e.get("customer_id") for e in snap["events"]}
    return bool(customers) and _consenting(session, customers) != customers


def export_visible_to(session: Session, staff: Staff, export: Export) -> bool:
    """Admins see every export. A labeler sees one only if they created it and every event it includes is
    visible to them now (its household consents to training); otherwise it is as if it did not exist."""
    if staff.role == "admin":
        return True
    if staff.role != "labeler" or export.created_by != staff.id:
        return False
    snap = _snapshot(export)
    if snap is None:
        return False
    ids = sorted({e["id"] for e in snap["events"]})
    if not ids:
        return True
    visible = session.scalar(select(func.count()).select_from(Event)
                             .join(Device, Device.id == Event.device_pk)
                             .join(Customer, Customer.id == Device.customer_id)
                             .where(Event.id.in_(ids), Customer.consent_training.is_(True)))
    return visible == len(ids)


CONSENT_WARNING = ("warning: a household in this export has withdrawn training consent since it was made; "
                   "do not train on it")


# ---------------------------------------------------------------- export builder

STALE_RUNNING_AFTER = timedelta(minutes=5)
STALE_QUEUED_AFTER = timedelta(minutes=30)
HEARTBEAT_EVERY = 30.0
MAX_COPY_BYTES = 5 * 1024 ** 3  # copy_object's limit
STALE_ERROR = "The export stopped before it finished (the server restarted or the job crashed). Start it again."
CONSENT_WITHDRAWN_ERROR = ("A household withdrew training consent while the export was being built; its files "
                           "were removed. Start it again.")
NO_SNAPSHOT_ERROR = "The export was requested by an older server version and has no snapshot. Start it again."


def export_prefix(name: str, version: int) -> str:
    return f"{EXPORT_ROOT}{name}/v{version}/"


def next_version(session: Session, s3, name: str) -> int:
    """The next version of export `name`: one more than any version in the database or under its S3 folder.

    Takes a transaction-scoped advisory lock on the name, so concurrent creates of one name get distinct
    versions (the caller commits the new row before the lock is released)."""
    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": f"export_version:{name}"})
    top = session.scalar(select(func.max(Export.version)).where(Export.name == name)) or 0
    if s3 is not None:
        base = f"{EXPORT_ROOT}{name}/"
        for sub in s3.list_dirs(base):
            match = re.fullmatch(r"v([0-9]{1,9})/", sub[len(base):])
            if match:
                top = max(top, int(match.group(1)))
    return top + 1


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def remap_label(label_text: str) -> tuple[str, int, Optional[str]]:
    """A YOLO label file with COCO-sparse class ids -> (file with contiguous 0-8 ids, rows dropped, problem).

    A row must have exactly 5 fields: an integer class and finite xc, yc in [0, 1] and w, h in (0, 1] (a
    rounding overshoot up to 1e-6 is clipped). Rows of a class with no contiguous id are dropped and counted.
    `problem` is "invalid_row" when any row is malformed, and "only_unmapped_classes" when a non-empty file
    would come out empty: either way the sample is quarantined (text ""), never turned into an empty negative.
    An empty source file is a negative and stays one."""
    rows, problem = parse_label(label_text)  # the one strict validator, shared with the detections view
    if problem is not None:
        return "", 0, problem
    out, dropped = [], 0
    for row in rows:
        if row.cls not in CONTIGUOUS:
            dropped += 1
            continue
        coords = [raw if float(raw) == v else _fmt(v) for raw, v in zip(row.raw, (row.xc, row.yc, row.w, row.h))]
        out.append(" ".join([str(CONTIGUOUS[row.cls]), *coords]))
    if not out and dropped:
        return "", dropped, "only_unmapped_classes"
    return ("\n".join(out) + "\n") if out else "", dropped, None


def strip_media_tokens(value: str) -> str:
    """`value` without LLaMA-Factory media placeholders (<video>, <image>, <audio>)."""
    return _MEDIA_TOKENS.sub("", value or "")


def _strip_tokens_json(value: Any) -> Any:
    if isinstance(value, str):
        return strip_media_tokens(value)
    if isinstance(value, dict):
        return {strip_media_tokens(str(k)): _strip_tokens_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_tokens_json(v) for v in value]
    return value


def vlm_prompt(ident, prompt: Optional[str]) -> tuple[str, bool]:
    """(user message content, whether privacy filtering changed or replaced the teacher prompt).

    The teacher prompt with the household's identity terms redacted and every media token removed, behind
    exactly one `<video>`; when there is no prompt, or the redacted text still mentions an identity term, the
    generic placeholder instead."""
    if not isinstance(prompt, str) or not strip_media_tokens(prompt).strip():
        return PLACEHOLDER_PROMPT, False
    shown = ident.text(prompt)
    if ident.mentions(shown):
        return PLACEHOLDER_PROMPT, True
    return "<video>" + strip_media_tokens(shown), shown != prompt


def error_text(exc: BaseException) -> str:
    """What an export's `error` shows: the exception type and first line, with anything path-like (file paths,
    storage keys -- which name sites) replaced."""
    first = (str(exc).strip().splitlines() or [""])[0]
    message = _PATHISH.sub("<path>", first)
    return f"{type(exc).__name__}: {message}"[:300] if message else type(exc).__name__


def _not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("NoSuchKey", "404", "NotFound") or status == 404


def _export_clip(s3, source_key: str, dest_key: str, etag: Optional[str]) -> None:
    """Server-side copy of a clip into the export, only of the snapshot's revision."""
    s3.copy(source_key, dest_key, content_type="video/mp4", if_match=etag)


class LeaseLost(Exception):
    """Another process (the sweep) took the export away from this worker."""


class ConsentWithdrawn(Exception):
    pass


def new_worker_id() -> str:
    return f"{socket.gethostname()[:30]}:{os.getpid()}:{uuid.uuid4().hex[:8]}"[:64]


def claim_export(session: Session, export_id: int, worker_id: str) -> bool:
    """Atomically take a queued export (queued -> running, owned by `worker_id`); False if it was not queued."""
    row = session.execute(text(
        "UPDATE exports SET state='running', worker_id=:w, heartbeat_at=now(), error=NULL "
        "WHERE id=:id AND state='queued' RETURNING id"), {"w": worker_id, "id": export_id}).first()
    session.commit()
    return row is not None


class _Lease:
    """Refreshes the export's heartbeat every HEARTBEAT_EVERY seconds on its own connection; notices when the
    export is no longer this worker's (swept as stale). `model` is the leased table: Export, or TaggingPublish for
    a tagging publish (same columns: state, worker_id, heartbeat_at)."""

    def __init__(self, engine, export_id: int, worker_id: str, model=Export):
        self.engine, self.export_id, self.worker_id, self.model = engine, export_id, worker_id, model
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"export-lease-{export_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5.0)

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_EVERY):
            try:
                self.beat()
            except Exception:  # noqa: BLE001 -- a missed beat is retried; the sweep allows 5 minutes
                log.warning("export %s heartbeat failed", self.export_id)

    def beat(self) -> None:
        m = self.model
        with self.engine.begin() as conn:
            n = conn.execute(update(m).where(m.id == self.export_id, m.worker_id == self.worker_id,
                                             m.state == "running")
                             .values(heartbeat_at=func.now())).rowcount
        if not n:
            self.lost.set()

    def check(self) -> None:
        if self.lost.is_set():
            raise LeaseLost(self.export_id)


def _finish(session: Session, export_id: int, worker_id: str, model=Export, **values) -> bool:
    """Set the final state only while this worker still owns the running export (or publish: `model`)."""
    n = session.execute(update(model).where(model.id == export_id, model.worker_id == worker_id,
                                            model.state == "running").values(**values)
                        .execution_options(synchronize_session=False)).rowcount
    session.commit()
    return bool(n)


class _Build:
    """One export run over its snapshot: writes objects under the export prefix and streams the manifest items,
    the VLM lines and the private mapping to temporary files."""

    def __init__(self, session: Session, s3, export: Export, request: ExportRequest, secret: str,
                 formats: list[str], tmp: Path, lease: _Lease):
        self.session, self.s3, self.req, self.secret, self.lease = session, s3, request, secret, lease
        self.prefix = export.s3_prefix
        self.formats = formats
        self.splits = [n for n in SPLIT_ORDER if n in request.split]
        self.items_file = (tmp / "items.jsonl").open("w", encoding="utf-8")
        self.private_file = (tmp / "mapping.json").open("w", encoding="utf-8")
        self.private_file.write("{")
        self.vlm_files = {sp: (tmp / f"vlm_{sp}.jsonl").open("w", encoding="utf-8") for sp in SPLIT_ORDER}
        self.item_count = 0
        self.missing: list[dict] = []
        self._missing_seen: set = set()
        self.quarantined: list[dict] = []
        self.no_weak_labels: list[int] = []
        self.dropped_boxes = 0
        self.groups: set[str] = set()
        self.split_counts = {n: 0 for n in request.split}
        self.counts: dict[str, dict] = {}
        if "yolo" in formats:
            self.counts["yolo"] = {n: {"images": 0, "boxes": 0} for n in SPLIT_ORDER}
        if "vlm_jsonl" in formats:
            self.counts["vlm"] = {n: 0 for n in SPLIT_ORDER}
        if "clips" in formats:
            self.counts["clips"] = {n: 0 for n in SPLIT_ORDER}
        self.referenced: list[str] = []  # every key the bundle points at, verified before publishing
        self.label_sources: set[str] = set()  # "human" and/or "detector_weak_label"
        self._identities: dict[int, redact.Identity] = {}

    def close(self) -> None:
        for f in (self.items_file, self.private_file, *self.vlm_files.values()):
            if not f.closed:
                f.close()

    def identity(self, device: Device) -> redact.Identity:
        if device.id not in self._identities:
            self._identities[device.id] = redact.identity(self.session, device, self.secret)
        return self._identities[device.id]

    def miss(self, event_id: int, reason: str, frame: Optional[int] = None, note: Optional[str] = None) -> None:
        mark = (event_id, reason, frame, note)
        if mark in self._missing_seen:
            return
        self._missing_seen.add(mark)
        entry: dict[str, Any] = {"event_id": event_id, "reason": reason}
        if frame is not None:
            entry["frame"] = frame
        if note is not None:
            entry["note"] = note
        self.missing.append(entry)

    def copy(self, event_id: int, art: dict, dest_key: str, missing_reason: str, copier,
             frame: Optional[int] = None) -> bool:
        """Copy the snapshot revision of `art` to `dest_key`; on any difference record why and return False."""
        info = self.s3.head(art["key"])
        if info is None:
            self.miss(event_id, missing_reason, frame)
            return False
        if art.get("etag") and info.etag != art["etag"]:
            self.miss(event_id, "changed_since_request", frame)
            return False
        if info.size > MAX_COPY_BYTES:
            self.miss(event_id, "too_large", frame, note="too large for server-side copy")
            return False
        try:
            copier(self.s3, art["key"], dest_key, art.get("etag"))
        except ETagMismatch:
            self.miss(event_id, "changed_since_request", frame)
            return False
        except Exception as e:
            if not _not_found(e):
                raise
            self.miss(event_id, missing_reason, frame)
            return False
        return True

    def event(self, ev: Event, snap: dict) -> None:
        split = snap["split"]
        device = self.session.get(Device, ev.device_pk)
        ident = self.identity(device)
        sources: list[str] = []
        private: dict[str, Any] = {"site": ev.site, "camera": ev.camera, "stem": ev.stem,
                                   "customer_id": device.customer_id, "device_id": device.device_id,
                                   "clip_key": None, "clip_source": None, "clip_etag": None, "yolo": [],
                                   "ai_run_id": (snap.get("ai") or {}).get("id"),
                                   "feedback_ids": [_feedback_id(x) for x in snap.get("feedback", [])]}

        clip_path, clip_sha, clip_hash, clip_ok = None, None, "not_exported", False
        if "clips" in self.formats:
            clip = snap.get("clip")
            dest = f"clips/{split}/{ev.id}.mp4"
            if clip is None:
                self.miss(ev.id, "clip_missing")
            elif self.copy(ev.id, clip, self.prefix + dest, "clip_missing", _export_clip):
                try:
                    clip_sha = self.s3.sha256(self.prefix + dest)  # streamed once; never the ETag
                    clip_ok = True
                except Exception as e:
                    if not _not_found(e):
                        raise
                    self.miss(ev.id, "clip_not_in_export")
            if clip_ok:
                clip_path, clip_hash = dest, "sha256"
                sources.append(_ref(clip["id"]))
                private.update(clip_key=clip["key"], clip_source=clip["kind"], clip_etag=clip.get("etag"))
                self.counts["clips"][split] += 1
                self.referenced.append(self.prefix + dest)

        ann = self.annotation(ev, snap)
        who = (pseudonym.staff(self.secret, ann.author_id) if ann is not None and ann.author_id is not None
               else "labeler-unknown")
        frames, yolo_prov = 0, None
        if "yolo" in self.formats and ann is not None:  # human labels win over the weak labels
            frames = self.human_yolo(ev, snap, split, sources, private, ann)
            yolo_prov = {"source": "human", "annotation_version": ann.version, "labeled_by": who,
                         "class_schema": CLASS_SCHEMA}
            self.label_sources.add("human")
        elif "yolo" in self.formats:
            frames = self.yolo(ev, snap, split, sources, private)
            if snap.get("frames"):
                yolo_prov = {"source": "detector_weak_label", "model": snap.get("detector_model") or "unknown",
                             "class_schema": CLASS_SCHEMA}
                self.label_sources.add("detector_weak_label")

        ai_snap = snap.get("ai")
        run = self.session.get(AiRun, ai_snap["id"]) if ai_snap else None
        if ai_snap and (run is None or _run_digest(run) != ai_snap["digest"]):
            self.miss(ev.id, "changed_since_request", note="AI answer")
            run = None
        ai_status = ai_snap["status"] if ai_snap else "none"
        ai_prov = None
        if run is not None:
            ai_prov = {"run_id": f"ai-run-{run.id}", "status": run.status,
                       "prompt_version": ident.text(run.prompt_version) if run.prompt_version else None,
                       "model": ident.text(run.model) if run.model else None,
                       "input": "sampled_frames" if run.input_artifact_ids else "clip"}

        wants_ai = ai_status == "real" or (self.req.include_fallback_ai and ai_status in ("fallback", "failed"))
        if ann is not None:  # the human description is the answer, whatever the AI said
            if "vlm_jsonl" in self.formats and clip_ok and (ann.description or "").strip():
                content, redacted = vlm_prompt(ident, run.prompt if run is not None else None)
                parsed = (_strip_tokens_json(ident.json(run.parsed))
                          if run is not None and isinstance(run.parsed, dict) else None)
                answer = labeling.assistant_answer(parsed, strip_media_tokens(ident.text(ann.description)))
                line = {
                    "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}],
                    "videos": [clip_path],
                    "event_id": ev.id, "split": split,
                    "owner_verdicts": [ident.text(v) for v in snap.get("verdicts", [])], "ai_status": ai_status,
                    "prompt_redacted": redacted,
                    "prompt_source": "placeholder" if content == PLACEHOLDER_PROMPT else "teacher",
                    "prompt_version": ident.text(run.prompt_version) if run and run.prompt_version else None,
                    "label_source": "human_corrected", "annotation_version": ann.version, "labeled_by": who,
                }
                self.vlm_files[split].write(json.dumps(line, ensure_ascii=False) + "\n")
                self.counts["vlm"][split] += 1
        elif ("vlm_jsonl" in self.formats and clip_ok and wants_ai and run is not None
                and isinstance(run.parsed, dict) and run.parsed):
            content, redacted = vlm_prompt(ident, run.prompt)
            answer = json.dumps(_strip_tokens_json(ident.json(run.parsed)), ensure_ascii=False)
            line = {
                "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}],
                "videos": [clip_path],
                "event_id": ev.id, "split": split,
                "owner_verdicts": [ident.text(v) for v in snap.get("verdicts", [])], "ai_status": ai_status,
                "prompt_redacted": redacted,
                "prompt_source": "placeholder" if content == PLACEHOLDER_PROMPT else "teacher",
                "prompt_version": ident.text(run.prompt_version) if run.prompt_version else None,
            }
            self.vlm_files[split].write(json.dumps(line, ensure_ascii=False) + "\n")
            self.counts["vlm"][split] += 1

        feedback = self.feedback(ev.id, snap.get("feedback", []), ident)

        group = group_id(ev, self.req.name, self.secret)
        self.groups.add(group)
        self.split_counts[split] = self.split_counts.get(split, 0) + 1
        item = {
            "event_id": ev.id,
            "customer": pseudonym.customer(self.secret, device.customer_id),
            "camera": pseudonym.camera(self.secret, ev.site, ev.camera),
            "group": group, "split": split,
            "kind": ev.kind if ev.kind in _KINDS else "unknown",
            "clip": clip_path, "clip_sha256": clip_sha, "clip_hash": clip_hash,
            "ai_status": ai_status,
            "owner_verdicts": [ident.text(v) for v in snap.get("verdicts", [])],
            "yolo_frames": frames, "sources": sources,
            "yolo": yolo_prov, "ai": ai_prov, "owner_feedback": feedback,
            "annotation": ({"version": ann.version, "labeled_by": who, "status": snap["annotation"].get("status")}
                           if ann is not None else None),
        }
        self.items_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.private_file.write(("," if self.item_count else "") + json.dumps(str(ev.id)) + ":"
                                + json.dumps(private, ensure_ascii=False))
        self.item_count += 1

    def annotation(self, ev: Event, snap: dict):
        """The annotation version frozen in the snapshot (versions are append-only), or None."""
        info = snap.get("annotation")
        if not info:
            return None
        row = labeling.version_row(self.session, ev.id, info["version"])
        if row is None:
            self.miss(ev.id, "changed_since_request", note="annotation")
        return row

    def human_yolo(self, ev: Event, snap: dict, split: str, sources: list[str], private: dict, ann) -> int:
        """EVERY frame of the clip: the image extracted with ffmpeg at its native index, and its label rows from the
        human tracks at the frame's own time (box_at); a frame with no box gets an empty label (a human negative)."""
        clip = snap["annotation"].get("clip")
        if clip is None:
            self.miss(ev.id, "clip_missing")
            return 0
        tracks = labeling.to_tracks(ann.tracks)
        written = 0
        with tempfile.TemporaryDirectory(prefix="hgframes_") as tmp:
            work = Path(tmp)
            try:
                self.s3.download_to(clip["key"], work / "src.mp4", if_match=clip.get("etag") or None)
            except ETagMismatch:
                self.miss(ev.id, "changed_since_request")
                return 0
            except Exception as e:
                if not _not_found(e):
                    raise
                self.miss(ev.id, "clip_missing")
                return 0
            try:
                frames = labeling.extract_frames(work / "src.mp4", work, ev.fps)
            except media.MediaError as e:
                self.miss(ev.id, "frames_unavailable", note=error_text(e))
                return 0
            for index, t_sec, path in frames:
                stem = f"{ev.id}_f{index:04d}"
                image_key = f"{self.prefix}yolo/images/{split}/{stem}.jpg"
                label_key = f"{self.prefix}yolo/labels/{split}/{stem}.txt"
                rows = labeling.yolo_rows(tracks, t_sec)
                self.s3.upload_file(path, image_key, "image/jpeg")
                self.s3.put_bytes(label_key, labeling.label_text(rows).encode("utf-8"), "text/plain")
                self.counts["yolo"][split]["images"] += 1
                self.counts["yolo"][split]["boxes"] += len(rows)
                self.referenced += [image_key, label_key]
                written += 1
        sources.append(_ref(clip["id"]))
        private["human_labels"] = {"clip_key": clip["key"], "clip_etag": clip.get("etag"),
                                   "annotation_version": ann.version, "frames": written}
        return written

    def feedback(self, event_id: int, frozen: list, ident: redact.Identity) -> list[dict]:
        """The owner feedback frozen in the snapshot. A piece whose row or source revision has changed since the
        request (or is gone) is left out and reported as `changed_since_request`."""
        ids = [_feedback_id(x) for x in frozen]
        if not ids:
            return []
        rows = {fb.id: (fb, rev) for fb, rev in self.session.execute(
            select(Feedback, Artifact.applied_etag).outerjoin(Artifact, Artifact.s3_key == Feedback.s3_key)
            .where(Feedback.id.in_(ids)))}
        out = []
        for want in frozen:
            found = rows.get(_feedback_id(want))
            if not isinstance(want, dict):  # a version-1 snapshot froze ids only: the row as it is now
                if found is None:
                    self.miss(event_id, "changed_since_request", note="owner feedback")
                    continue
                want = _frozen_feedback(*found)
            elif found is None or _frozen_feedback(*found) != want:
                self.miss(event_id, "changed_since_request", note="owner feedback")
                continue
            out.append({"feedback_id": f"feedback-{want['id']}", "verdict": ident.text(want["verdict"]),
                        "received_utc": want["received_utc"]})
        return out

    def yolo(self, ev: Event, snap: dict, split: str, sources: list[str], private: dict) -> int:
        """Every expected sampled frame: image copied and label remapped, or recorded as missing/quarantined."""
        pairs = snap.get("frames") or []
        if not pairs:
            self.no_weak_labels.append(ev.id)
            return 0
        written = 0
        for pair in pairs:
            frame, label, image = pair["frame"], pair.get("label"), pair.get("image")
            if label is None:
                self.miss(ev.id, "yolo_label_missing", frame)
            if image is None:
                self.miss(ev.id, "yolo_image_missing", frame)
            if label is None or image is None:
                continue
            try:
                label_text = self.s3.get_text(label["key"], if_match=label.get("etag") or None)
            except ETagMismatch:
                self.miss(ev.id, "changed_since_request", frame)
                continue
            except Exception as e:
                if not _not_found(e):
                    raise
                self.miss(ev.id, "yolo_label_missing", frame)
                continue
            remapped, dropped, problem = remap_label(label_text)
            if problem:
                self.quarantined.append({"event_id": ev.id, "frame": frame, "reason": problem})
                continue
            ext = PurePosixPath(image["key"]).suffix.lower() or ".jpg"
            stem = f"{ev.id}_f{frame:04d}"
            image_key = f"{self.prefix}yolo/images/{split}/{stem}{ext}"
            mime = _MIME.get(ext, "image/jpeg")
            if not self.copy(ev.id, image, image_key, "yolo_image_missing",
                             lambda s3, src, dest, etag: s3.copy(src, dest, content_type=mime, if_match=etag),
                             frame=frame):
                continue
            label_key = f"{self.prefix}yolo/labels/{split}/{stem}.txt"
            self.s3.put_bytes(label_key, remapped.encode("utf-8"), "text/plain")
            self.dropped_boxes += dropped
            self.counts["yolo"][split]["images"] += 1
            self.counts["yolo"][split]["boxes"] += len(remapped.splitlines())
            self.referenced += [image_key, label_key]
            sources += [_ref(image["id"]), _ref(label["id"])]
            private["yolo"].append({"frame": frame, "image_key": image["key"], "label_key": label["key"]})
            written += 1
        return written


def _feedback_id(entry: Any) -> Optional[int]:
    return entry.get("id") if isinstance(entry, dict) else entry


def _data_yaml(empty: set[str]) -> str:
    lines = ["path: .", "train: images/train"]
    for sp in ("val", "test"):
        lines.append(f"{sp}: images/{'train' if sp in empty else sp}")
    lines.append("names:")
    lines += [f"  {i}: {label}" for i, label in enumerate(YOLO_NAMES)]
    return "\n".join(lines) + "\n"


def dataset_info(splits_with_lines: Iterable[str]) -> dict:
    """LLaMA-Factory registrations (sharegpt with videos) of the non-empty per-split VLM files."""
    present = set(splits_with_lines)
    return {f"homeguard_{sp}": {
        "file_name": f"{sp}.jsonl", "formatting": "sharegpt",
        "columns": {"messages": "messages", "videos": "videos"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    } for sp in SPLIT_ORDER if sp in present}


DEST = "<DEST>"


def readme_text(export: Export, formats: list[str], datasets: list[str]) -> str:
    """README.txt at the export root: how to download it and the exact training commands."""
    tag = f"{export.name}_v{export.version}"
    out = [
        f"Home Guard training export {export.name} v{export.version}",
        "",
        "manifest.json lists every item, what is missing or quarantined, counts per split and warnings.",
        "No household identity is in this bundle: customers, cameras and site+day groups are pseudonyms.",
        "",
        "Download (admin, on a machine with the server's S3 credentials):",
        f"  python -m home_guard_project.cloud.manage export-download {export.id} --dest {DEST}",
        f"This copies the bundle (without _private/) into {DEST}, rewrites {DEST}/yolo/data.yaml so its",
        f"`path:` is the absolute {DEST}/yolo folder, and fills in {DEST} in this file.",
        "",
    ]
    if "yolo" in formats:
        out += [
            "YOLO (Ultralytics), contiguous class ids 0-8. A clip whose manifest item says yolo.source = human",
            "has human labels on EVERY frame (an empty label file is a human negative); the others carry",
            "detector weak labels (COCO ids remapped), not human ground truth.",
            "An empty val/test split points at train (see the manifest warnings).",
            f"  yolo detect train data={DEST}/yolo/data.yaml model=yolo11s.pt epochs=100 imgsz=640 batch=16 "
            f"project=runs/detect name={tag}",
            f"  yolo detect val data={DEST}/yolo/data.yaml model=runs/detect/{tag}/weights/best.pt imgsz=640",
            "",
        ]
    if "vlm_jsonl" in formats:
        train = "homeguard_train" if "homeguard_train" in datasets else (datasets[0] if datasets else "homeguard_train")
        out += [
            "Qwen2.5-VL (LLaMA-Factory). vlm/dataset_info.json registers: " + (", ".join(datasets) or "(none)"),
            "Video paths in the JSONL are relative to the export root, so media_dir is the export root.",
            "Save as hg_qwen.yaml:",
            "  model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct",
            "  stage: sft",
            "  do_train: true",
            "  finetuning_type: lora",
            "  lora_rank: 64",
            "  lora_alpha: 128",
            "  lora_target: all",
            "  template: qwen2_vl",
            f"  dataset_dir: {DEST}/vlm",
            f"  media_dir: {DEST}",
            f"  dataset: {train}",
        ]
        if "homeguard_val" in datasets:
            out.append("  eval_dataset: homeguard_val")
        out += [
            "  cutoff_len: 2048",
            f"  output_dir: runs/qwen25vl_{tag}",
            "  per_device_train_batch_size: 1",
            "  gradient_accumulation_steps: 8",
            "  learning_rate: 1.0e-5",
            "  num_train_epochs: 3",
            "  lr_scheduler_type: cosine",
            "  warmup_ratio: 0.1",
            "  bf16: true",
            "  video_fps: 2.0",
            "  video_maxlen: 64",
            "then run:",
            "  llamafactory-cli train hg_qwen.yaml",
            "",
        ]
    return "\n".join(out) + "\n"


def export_request(export: Export) -> ExportRequest:
    stored = export.request if isinstance(export.request, dict) else {}
    return ExportRequest(**{k: v for k, v in stored.items() if k in ExportRequest.model_fields})


def _run_build(session: Session, s3, export: Export, secret: str, lease: _Lease) -> tuple[str, int]:
    snap = _snapshot(export)
    if snap is None:
        raise RuntimeError(NO_SNAPSHOT_ERROR)
    if s3 is None:
        raise RuntimeError("Storage is not configured")
    req = export_request(export)
    prefix = export.s3_prefix
    if s3.any_under(prefix):  # never write into a version that already has files
        raise RuntimeError("The export folder already has files; nothing was overwritten")
    events = snap["events"]
    consenting = _consenting(session, {e.get("customer_id") for e in events})  # consent rechecked at the start
    kept = [e for e in events if e.get("customer_id") in consenting]
    excluded = list(snap.get("excluded", [])) + [{"event_id": e["id"], "reason": "no_training_consent"}
                                                  for e in events if e.get("customer_id") not in consenting]
    formats = list(snap.get("formats") or effective_formats(req.formats))

    with tempfile.TemporaryDirectory(prefix="hgexport_") as tmp_name:
        tmp = Path(tmp_name)
        build = _Build(session, s3, export, req, secret, formats, tmp, lease)
        try:
            for chunk in _chunks(kept, _PAGE):  # paged reads of the snapshot's events
                ids = [e["id"] for e in chunk]
                rows = {ev.id: ev for ev in session.scalars(select(Event).where(Event.id.in_(ids)))}
                for snap_ev in chunk:
                    lease.check()
                    ev = rows.get(snap_ev["id"])
                    if ev is None:
                        build.miss(snap_ev["id"], "event_deleted")
                        continue
                    build.event(ev, snap_ev)
            build.private_file.write("}")
        finally:
            build.close()

        warnings: list[str] = []
        for fmt, per in build.counts.items():
            for sp in build.splits:
                n = per[sp]["images"] if fmt == "yolo" else per[sp]
                if not n:
                    warnings.append(f"{fmt}: the {sp} split is empty.")
        warnings += small_set_warnings(len(build.groups))
        empty_yolo: set[str] = set()
        if "yolo" in formats:
            empty_yolo = {sp for sp in SPLIT_ORDER if not build.counts["yolo"][sp]["images"]}
            if "train" in empty_yolo:
                warnings.append("yolo: no training images.")
            if "val" in empty_yolo:
                warnings.append("yolo: the val split is empty, so validation reuses training data.")
            if "test" in empty_yolo:
                warnings.append("yolo: the test split is empty, so testing reuses training data.")
        if build.quarantined:
            warnings.append(f"{len(build.quarantined)} YOLO sample(s) quarantined for invalid labels.")
        dropped_consent = len(events) - len(kept)
        if dropped_consent:
            warnings.append(f"{dropped_consent} event(s) left out: training consent was withdrawn after the request.")
        if "vlm_jsonl" in req.formats and "clips" not in req.formats:
            warnings.append(CLIPS_ADDED)

        datasets: list[str] = []
        if "vlm_jsonl" in formats:
            info = dataset_info(sp for sp in SPLIT_ORDER if build.counts["vlm"][sp])
            datasets = sorted(info)
            for sp in SPLIT_ORDER:
                s3.upload_file(tmp / f"vlm_{sp}.jsonl", f"{prefix}vlm/{sp}.jsonl", "application/jsonl")
            s3.put_bytes(prefix + "vlm/dataset_info.json", json.dumps(info, indent=2).encode("utf-8"),
                         "application/json")
        if "yolo" in formats:
            s3.put_bytes(prefix + "yolo/data.yaml", _data_yaml(empty_yolo).encode("utf-8"), "application/yaml")
        s3.put_bytes(prefix + "README.txt", readme_text(export, formats, datasets).encode("utf-8"), "text/plain")
        s3.upload_file(tmp / "mapping.json", prefix + "_private/mapping.json", "application/json")
        s3.put_json(prefix + "_private/excluded.json", excluded)

        for key in build.referenced:  # every file the bundle points at must be in it
            if s3.head(key) is None:
                raise RuntimeError("An exported file is missing after it was copied")

        # consent again before publishing: a withdrawal during the build removes everything
        kept_customers = {e.get("customer_id") for e in kept}
        if _consenting(session, kept_customers) != {c for c in kept_customers if c is not None}:
            raise ConsentWithdrawn()
        lease.beat()
        lease.check()

        creator = session.scalar(select(Staff.name).where(Staff.id == export.created_by)) or ""
        excluded_counts: dict[str, int] = {}
        for e in excluded:
            excluded_counts[e["reason"]] = excluded_counts.get(e["reason"], 0) + 1
        head = {
            "schema_version": MANIFEST_SCHEMA, "name": export.name, "version": export.version,
            "created_utc": export.created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_by": creator, "collection_id": req.collection_id,
            "requested_formats": list(dict.fromkeys(req.formats)), "formats": formats,
            "split": dict(req.split), "split_key_version": snap.get("split_key_version", SPLIT_KEY_VERSION),
            "split_counts": build.split_counts, "groups": len(build.groups),
            "include_fallback_ai": req.include_fallback_ai, "class_map": CLASS_MAP, "class_schema": CLASS_SCHEMA,
            "counts": build.counts, "warnings": warnings,
            "missing": build.missing, "no_weak_labels": sorted(build.no_weak_labels),
            "quarantined_samples": {"count": len(build.quarantined), "samples": build.quarantined},
            "dropped_boxes": build.dropped_boxes,
            # exclusions of households without training consent are only counted (details in _private/)
            "excluded": [e for e in excluded if e["reason"] != "no_training_consent"],
            "excluded_counts": excluded_counts,
        }
        if "vlm_jsonl" in formats:
            head["vlm"] = {"jsonl_dir": "vlm", "media_dir": ".", "datasets": datasets}
        if "yolo" in formats:
            sources = build.label_sources
            head["yolo"] = {"data_yaml": "yolo/data.yaml",
                            "labels": "mixed" if len(sources) > 1 else next(iter(sources), "detector_weak_label")}
        manifest_path = tmp / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as out, \
                (tmp / "items.jsonl").open("r", encoding="utf-8") as items:
            out.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "items": [')
            for i, line in enumerate(items):
                out.write(("," if i else "") + line.rstrip("\n"))
            out.write("]}")
        s3.upload_file(manifest_path, prefix + "manifest.json", "application/json")  # last: the bundle is whole
    return ("partial" if build.missing else "ready"), build.item_count


def build_export(session: Session, s3, export_id: int, *, secret: str,
                 worker_id: Optional[str] = None) -> Optional[Export]:
    """Build a queued export from its snapshot (queued -> running -> ready | partial | failed).

    The worker first claims the export atomically; an export that is not queued (finished, or another worker's)
    is left alone. While building, a lease thread refreshes `heartbeat_at`; every final state is written only
    while this worker still owns the export. A crash ends in `failed` with a path-free error text."""
    worker_id = worker_id or new_worker_id()
    if not claim_export(session, export_id, worker_id):
        return session.get(Export, export_id, populate_existing=True)
    export = session.get(Export, export_id, populate_existing=True)
    lease = _Lease(session.get_bind(), export_id, worker_id)
    lease.start()
    try:
        state, count = _run_build(session, s3, export, secret, lease)
        if not _finish(session, export_id, worker_id, state=state, item_count=count, error=None,
                       heartbeat_at=func.now()):
            log.warning("export %s: lease lost before it could be marked %s", export_id, state)
    except LeaseLost:
        session.rollback()
        log.warning("export %s: lease lost (swept as stale); stopped without publishing", export_id)
    except ConsentWithdrawn:
        session.rollback()
        try:
            s3.delete_prefix(export.s3_prefix)
        finally:
            _finish(session, export_id, worker_id, state="failed", error=CONSENT_WITHDRAWN_ERROR)
    except Exception as e:  # noqa: BLE001 -- any failure ends the export as failed, never stuck in running
        log.exception("export %s failed", export_id)
        session.rollback()
        _finish(session, export_id, worker_id, state="failed", error=error_text(e))
    finally:
        lease.stop()
    return session.get(Export, export_id, populate_existing=True)


def run_export_job(sessionmaker, s3, export_id: int, secret: str) -> None:
    """Background entry point: builds the export in its own session."""
    session = sessionmaker()
    try:
        build_export(session, s3, export_id, secret=secret)
    except Exception:  # noqa: BLE001 -- the periodic sweep fails it later if even marking it failed broke
        log.exception("export job %s crashed", export_id)
    finally:
        session.close()


def sweep_stale_exports(session: Session, now: datetime) -> int:
    """Fail exports whose worker is gone: running with no heartbeat for 5 minutes (or, never having beaten,
    created over 5 minutes ago) and queued for over 30 minutes. A healthy long export keeps beating and is never
    swept for its age. Returns how many were failed."""
    beat = func.coalesce(Export.heartbeat_at, Export.created_at)
    stale = or_(and_(Export.state == "running", beat < now - STALE_RUNNING_AFTER),
                and_(Export.state == "queued", Export.created_at < now - STALE_QUEUED_AFTER))
    result = session.execute(update(Export).where(stale).values(state="failed", error=STALE_ERROR)
                             .execution_options(synchronize_session=False))
    session.flush()
    return result.rowcount or 0


# ---------------------------------------------------------------- download (admin CLI)

def download_export(s3, prefix: str, dest: Path) -> int:
    """Download an export (without `_private/`) into `dest`; make `yolo/data.yaml` point at the absolute
    `dest/yolo` and fill the destination into README.txt. Returns how many files were written."""
    dest = Path(dest).resolve()
    n = 0
    for obj in s3.list(prefix):
        rel = obj.key[len(prefix):]
        parts = rel.split("/")
        if not rel or parts[0] == "_private" or any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts):
            continue
        target = dest.joinpath(*parts)
        if dest not in target.resolve().parents:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(obj.key, target)
        n += 1
    yaml = dest / "yolo" / "data.yaml"
    if yaml.exists():
        lines = yaml.read_text(encoding="utf-8").splitlines()
        lines = [f"path: {(dest / 'yolo').as_posix()}" if line.startswith("path:") else line for line in lines]
        yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = dest / "README.txt"
    if readme.exists():
        readme.write_text(readme.read_text(encoding="utf-8").replace(DEST, str(dest)), encoding="utf-8")
    return n
