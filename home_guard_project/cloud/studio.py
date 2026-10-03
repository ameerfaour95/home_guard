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
import re
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Optional, get_args

from sqlalchemy import Float, and_, case, cast, func, select, text, type_coerce, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract.classes import COCO_NAMES, CONTIGUOUS

from . import pseudonym, redact
from .models import AiRun, Artifact, CollectionItem, Collection, Customer, Device, Event, Export, RawRevision, Staff
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


def group_id(ev: Any, name: str, secret: str) -> str:
    """Opaque id of the event's (site, day) group in export `name`: keyed, so it names no site."""
    digest = hmac.new(_split_key(secret), f"group\x00{name}\x00{_group_key(ev)}".encode("utf-8"), hashlib.sha256)
    return "group-" + digest.hexdigest()[:12]


# ---------------------------------------------------------------- export builder

EXPORT_ROOT = "training_exports/"
PLACEHOLDER_PROMPT = "<video>Describe what happens in this security camera clip."
YOLO_NAMES = [COCO_NAMES[coco] for coco in sorted(CONTIGUOUS, key=CONTIGUOUS.get)]
CLASS_MAP = {str(i): label for i, label in enumerate(YOLO_NAMES)}
STALE_AFTER = timedelta(hours=1)
STALE_ERROR = "The export stopped before it finished (the server restarted or the job crashed). Start it again."
_KINDS = set(get_args(EventKind))
_FRAME = re.compile(r"_f([0-9]+)\.[A-Za-z0-9]+$")
_PATHISH = re.compile(r"\S*[\\/]\S*")
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


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


def remap_label(label_text: str) -> tuple[str, int]:
    """A YOLO label file with COCO-sparse class ids -> (the file with contiguous 0-8 ids, rows dropped).

    Rows whose class has no contiguous id, and malformed rows, are dropped and counted."""
    out, dropped = [], 0
    for line in label_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = float(parts[0])
            coords = [float(p) for p in parts[1:5]]
        except (ValueError, OverflowError):
            dropped += 1
            continue
        finite = all(math.isfinite(v) for v in coords)
        if len(coords) != 4 or not finite or not math.isfinite(cls) or not cls.is_integer() \
                or int(cls) not in CONTIGUOUS:
            dropped += 1
            continue
        out.append(" ".join([str(CONTIGUOUS[int(cls)]), *parts[1:5]]))
    return ("\n".join(out) + "\n") if out else "", dropped


def vlm_prompt(ident, prompt: Optional[str]) -> tuple[str, bool]:
    """(user message content, whether redaction changed the teacher prompt).

    The teacher prompt with the household's identity terms redacted; when there is no prompt, or the redacted
    text still mentions an identity term, the generic placeholder instead."""
    if not isinstance(prompt, str) or not prompt.strip():
        return PLACEHOLDER_PROMPT, False
    shown = ident.text(prompt)
    if ident.mentions(shown):
        return PLACEHOLDER_PROMPT, False
    return "<video>" + shown, shown != prompt


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


def _frame(key: str) -> Optional[int]:
    match = _FRAME.search(key)
    return int(match.group(1)) if match else None


def _training_first(art: Artifact):
    return (not art.s3_key.startswith("dataset_"), art.s3_key)


def _root(key: str) -> str:
    return key.split("/", 1)[0]


def _meta_codec(session: Session, arts: list[Artifact]) -> Optional[str]:
    """The clip codec the box recorded in its meta (newest revision, training copy first), if any."""
    for art in sorted((a for a in arts if a.role == "meta"), key=_training_first):
        body = session.scalar(select(RawRevision.body).where(RawRevision.s3_key == art.s3_key)
                              .order_by(RawRevision.fetched_at.desc(), RawRevision.id.desc()).limit(1))
        if isinstance(body, dict) and isinstance(body.get("codec"), str):
            return body["codec"]
    return None


def _ref(art: Artifact) -> str:
    return f"artifact-{art.id}"


def _export_clip(s3, source_key: str, dest_key: str) -> Optional[str]:
    """Server-side copy of a clip into the export; returns its SHA-256 (hex) as computed by S3."""
    return s3.copy(source_key, dest_key, content_type="video/mp4", sha256=True)


class _Build:
    """One export run: writes objects under the export prefix and collects the manifest."""

    def __init__(self, session: Session, s3, export: Export, request: ExportRequest, secret: str):
        self.session, self.s3, self.req, self.secret = session, s3, request, secret
        self.prefix = export.s3_prefix
        self.formats = list(dict.fromkeys(request.formats))
        self.items: list[dict] = []
        self.missing: list[dict] = []
        self.vlm_lines: list[str] = []
        self.private: dict[str, dict] = {}
        self.dropped_boxes = 0
        self.frames_without_image = 0
        self._identities: dict[int, redact.Identity] = {}

    def identity(self, device: Device) -> redact.Identity:
        if device.id not in self._identities:
            self._identities[device.id] = redact.identity(self.session, device, self.secret)
        return self._identities[device.id]

    def miss(self, event_id: int, reason: str) -> None:
        entry = {"event_id": event_id, "reason": reason}
        if entry not in self.missing:
            self.missing.append(entry)

    def event(self, ev: Event, split: str) -> None:
        device = self.session.get(Device, ev.device_pk)
        ident = self.identity(device)
        arts = list(self.session.scalars(select(Artifact).where(Artifact.event_id == ev.id).order_by(Artifact.id)))
        run = self.session.scalar(select(AiRun).where(AiRun.event_id == ev.id, AiRun.purpose == "guard")
                                  .order_by(AiRun.id.desc()).limit(1))
        ai_status = run.status if run is not None else "none"
        verdicts = [ident.text(v) for v in (ev.owner_verdicts or []) if isinstance(v, str)]
        sources: list[str] = []
        private: dict[str, Any] = {"site": ev.site, "camera": ev.camera, "stem": ev.stem,
                                   "customer_id": device.customer_id, "device_id": device.device_id,
                                   "clip_key": None, "clip_source": None, "clip_etag": None, "yolo": [],
                                   "ai_source_key": run.ai_source_key if run is not None else None}

        clip_path, clip_sha, clip_ok = None, None, True
        if "clips" in self.formats:
            clip_ok = False
            src, kind = self.clip_source(arts)
            if src is not None:
                try:
                    clip_sha = _export_clip(self.s3, src.s3_key, f"{self.prefix}clips/{split}/{ev.id}.mp4")
                    clip_ok = True
                except Exception as e:
                    if not _not_found(e):
                        raise
            if clip_ok:
                clip_path = f"clips/{split}/{ev.id}.mp4"
                sources.append(_ref(src))
                private.update(clip_key=src.s3_key, clip_source=kind, clip_etag=src.etag)
            else:
                self.miss(ev.id, "clip_missing")

        frames = self.yolo(ev, split, arts, sources, private) if "yolo" in self.formats else 0

        wants_ai = ai_status == "real" or (self.req.include_fallback_ai and ai_status in ("fallback", "failed"))
        if ("vlm_jsonl" in self.formats and clip_ok and wants_ai and run is not None
                and isinstance(run.parsed, dict) and run.parsed):
            content, redacted = vlm_prompt(ident, run.prompt)
            line = {
                "messages": [{"role": "user", "content": content},
                             {"role": "assistant",
                              "content": json.dumps(ident.json(run.parsed), ensure_ascii=False)}],
                "videos": [f"clips/{split}/{ev.id}.mp4"],
                "event_id": ev.id, "split": split, "owner_verdicts": verdicts, "ai_status": ai_status,
                "prompt_redacted": redacted,
                "prompt_source": "placeholder" if content == PLACEHOLDER_PROMPT else "teacher",
                "prompt_version": ident.text(run.prompt_version) if run.prompt_version else None,
            }
            self.vlm_lines.append(json.dumps(line, ensure_ascii=False))

        self.items.append({
            "event_id": ev.id,
            "customer": pseudonym.customer(self.secret, device.customer_id),
            "camera": pseudonym.camera(self.secret, ev.site, ev.camera),
            "group": group_id(ev, self.req.name, self.secret),
            "split": split,
            "kind": ev.kind if ev.kind in _KINDS else "unknown",
            "clip": clip_path,
            "clip_sha256": clip_sha,
            "ai_status": ai_status,
            "owner_verdicts": verdicts,
            "yolo_frames": frames,
            "sources": sources,
        })
        self.private[str(ev.id)] = private

    def clip_source(self, arts: list[Artifact]) -> tuple[Optional[Artifact], Optional[str]]:
        """The original clip (training copy first); its H.264 rendition instead when the original is not H.264."""
        originals = sorted((a for a in arts if a.role == "original_video" and a.available), key=_training_first)
        renditions = [a for a in arts if a.role == "rendition" and a.available]
        if not originals:
            return (renditions[0], "rendition") if renditions else (None, None)
        codec = _meta_codec(self.session, arts)
        if renditions and (codec or "").lower() != "h264":
            return renditions[0], "rendition"
        return originals[0], "original"

    def yolo(self, ev: Event, split: str, arts: list[Artifact], sources: list[str], private: dict) -> int:
        """Copy each sampled frame's image and write its remapped label; returns frames written."""
        labels: dict[int, Artifact] = {}
        for art in sorted((a for a in arts if a.role == "yolo_label" and a.available), key=_training_first):
            frame = _frame(art.s3_key)
            if frame is not None:
                labels.setdefault(frame, art)
        images: dict[int, list[Artifact]] = {}
        for art in sorted((a for a in arts if a.role == "yolo_image" and a.available), key=_training_first):
            frame = _frame(art.s3_key)
            if frame is not None:
                images.setdefault(frame, []).append(art)
        written = 0
        for frame, label in sorted(labels.items()):
            candidates = images.get(frame, [])
            image = next((a for a in candidates if _root(a.s3_key) == _root(label.s3_key)),
                         candidates[0] if candidates else None)
            if image is None:  # the box never uploaded this frame's image: not a training sample
                self.frames_without_image += 1
                continue
            try:
                label_text = self.s3.get_text(label.s3_key)
            except Exception as e:
                if not _not_found(e):
                    raise
                self.miss(ev.id, "yolo_label_missing")
                continue
            ext = PurePosixPath(image.s3_key).suffix.lower() or ".jpg"
            stem = f"{ev.id}_f{frame:04d}"
            try:
                self.s3.copy(image.s3_key, f"{self.prefix}yolo/images/{split}/{stem}{ext}",
                             content_type=_MIME.get(ext, "image/jpeg"))
            except Exception as e:
                if not _not_found(e):
                    raise
                self.miss(ev.id, "yolo_image_missing")
                continue
            remapped, dropped = remap_label(label_text)
            self.dropped_boxes += dropped
            self.s3.put_bytes(f"{self.prefix}yolo/labels/{split}/{stem}.txt", remapped.encode("utf-8"),
                              "text/plain")
            sources += [_ref(image), _ref(label)]
            private["yolo"].append({"frame": frame, "image_key": image.s3_key, "label_key": label.s3_key})
            written += 1
        return written


def _data_yaml() -> str:
    lines = ["path: .", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines += [f"  {i}: {label}" for i, label in enumerate(YOLO_NAMES)]
    return "\n".join(lines) + "\n"


def export_request(export: Export) -> ExportRequest:
    stored = export.request if isinstance(export.request, dict) else {}
    return ExportRequest(**{k: v for k, v in stored.items() if k in ExportRequest.model_fields})


def build_export(session: Session, s3, export_id: int, *, secret: str) -> Optional[Export]:
    """Build a queued export (queued -> running -> ready | partial | failed); any other state is left alone.

    Uses `select_export_items` and `assign_splits(..., secret=...)` exactly as the preview does. Every state
    change is committed; a crash ends in `failed` with a path-free error text, never in `running`."""
    export = session.get(Export, export_id)
    if export is None or export.state != "queued":
        return export
    export.state, export.error = "running", None
    session.commit()
    try:
        if s3 is None:
            raise RuntimeError("Storage is not configured")
        req = export_request(export)
        if s3.any_under(export.s3_prefix):  # never write into a version that already has files
            raise RuntimeError("The export folder already has files; nothing was overwritten")
        if session.get(Collection, req.collection_id) is None:
            raise RuntimeError("The collection no longer exists")
        stored = export.request if isinstance(export.request, dict) else {}
        included, excluded = select_export_items(session, req.collection_id, req,
                                                 labeler=bool(stored.get("as_labeler")))
        splits = assign_splits(included, req.name, req.split, secret=secret)
        build = _Build(session, s3, export, req, secret)
        for ev in included:
            build.event(ev, splits[ev.id])

        counts = {n: 0 for n in req.split}
        for item in build.items:
            counts[item["split"]] = counts.get(item["split"], 0) + 1
        if "vlm_jsonl" in build.formats:
            body = "".join(line + "\n" for line in build.vlm_lines)
            s3.put_bytes(export.s3_prefix + "vlm.jsonl", body.encode("utf-8"), "application/jsonl")
        if "yolo" in build.formats:
            s3.put_bytes(export.s3_prefix + "yolo/data.yaml", _data_yaml().encode("utf-8"), "application/yaml")
        s3.put_json(export.s3_prefix + "_private/mapping.json", build.private)
        creator = session.scalar(select(Staff.name).where(Staff.id == export.created_by)) or ""
        manifest = {
            "schema_version": 1, "name": export.name, "version": export.version,
            "created_utc": export.created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_by": creator, "collection_id": req.collection_id, "formats": build.formats,
            "split": dict(req.split), "split_counts": counts, "groups": group_count(included),
            "include_fallback_ai": req.include_fallback_ai, "class_map": CLASS_MAP,
            "items": build.items, "missing": build.missing,
            "excluded": [e.model_dump() for e in excluded],
            "dropped_boxes": build.dropped_boxes, "frames_without_image": build.frames_without_image,
        }
        s3.put_json(export.s3_prefix + "manifest.json", manifest)  # last: a manifest means the export is whole
        export.item_count = len(build.items)
        export.state = "partial" if build.missing else "ready"
        session.commit()
    except Exception as e:  # noqa: BLE001 -- any failure ends the export as failed, never stuck in running
        log.exception("export %s failed", export_id)
        session.rollback()
        export = session.get(Export, export_id)
        export.state, export.error = "failed", error_text(e)
        session.commit()
    return export


def run_export_job(sessionmaker, s3, export_id: int, secret: str) -> None:
    """Background entry point: builds the export in its own session."""
    session = sessionmaker()
    try:
        build_export(session, s3, export_id, secret=secret)
    except Exception:  # noqa: BLE001 -- the startup sweep fails it later if even marking it failed broke
        log.exception("export job %s crashed", export_id)
    finally:
        session.close()


def sweep_stale_exports(session: Session, now: datetime) -> int:
    """Mark exports still queued or running an hour after creation as failed; returns how many."""
    result = session.execute(update(Export).where(Export.state.in_(["queued", "running"]),
                                                  Export.created_at < now - STALE_AFTER)
                             .values(state="failed", error=STALE_ERROR))
    session.flush()
    return result.rowcount or 0
