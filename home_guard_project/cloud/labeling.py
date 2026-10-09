"""In-app labeling: a clip's annotation versions, reviews and current head; suggestions from the YOLO weak labels;
the clip's timing; and what a human annotation gives the training data -- the YOLO rows of any frame (by
`fleet_contract.tracks.box_at`, contiguous class ids) and every decoded frame of the clip as an image.

Used by the annotation routes, the training export (studio.py) and the tagging publish (tagging.py), so all three
read a human annotation the same way.
"""
from __future__ import annotations

import json
import logging
import math
import re
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import exists, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import tracks as ft
from home_guard_project.fleet_contract.classes import COCO_NAMES, coco_to_contiguous, name_to_coco

from . import media
from .models import (AiRun, Annotation, AnnotationHead, AnnotationReview, AnnotationSuggestion, Artifact,
                     Event, RawRevision)

log = logging.getLogger(__name__)

DONE = ("submitted", "reviewed")  # a clip counts as labeled once its current annotation is one of these
MAX_TRACKS = 500
MAX_KEYFRAMES = 20_000  # over all tracks of one annotation
DESCRIPTION_MAX = 4000
SUGGESTION_MODEL = "yolov8n-weak-labels"
_KNOWN = set(COCO_NAMES.values())


# ---------------------------------------------------------------- stored state

def head(session: Session, event_id: int) -> Optional[AnnotationHead]:
    """The clip's current annotation state, read fresh (heads are upserted with plain SQL)."""
    return session.get(AnnotationHead, event_id, populate_existing=True)


def version_row(session: Session, event_id: int, version: int) -> Optional[Annotation]:
    return session.scalar(select(Annotation).where(Annotation.event_id == event_id, Annotation.version == version))


def versions(session: Session, event_id: int) -> list[Annotation]:
    return list(session.scalars(select(Annotation).where(Annotation.event_id == event_id)
                                .order_by(Annotation.version)))


def latest_review(session: Session, event_id: int, version: int) -> Optional[AnnotationReview]:
    return session.scalar(select(AnnotationReview).where(AnnotationReview.event_id == event_id,
                                                         AnnotationReview.version == version)
                          .order_by(AnnotationReview.id.desc()).limit(1))


def reviews_by_version(session: Session, event_id: int) -> dict[int, AnnotationReview]:
    """The newest review of each reviewed version."""
    out: dict[int, AnnotationReview] = {}
    for row in session.scalars(select(AnnotationReview).where(AnnotationReview.event_id == event_id)
                               .order_by(AnnotationReview.id)):
        out[row.version] = row
    return out


def effective_status(row: Annotation, review: Optional[AnnotationReview]) -> str:
    if review is not None and review.version == row.version:
        return "reviewed" if review.decision == "accept" else "rejected"
    return row.status


def guard_run(session: Session, event_id: int) -> Optional[AiRun]:
    """The event's AI (guard) run, the newest when there are several."""
    return session.scalar(select(AiRun).where(AiRun.event_id == event_id, AiRun.purpose == "guard")
                          .order_by(AiRun.id.desc()).limit(1))


def labeled_versions(session: Session, ids: list[int]) -> dict[int, AnnotationHead]:
    """{event id: head} for the events among `ids` whose current annotation is submitted or reviewed."""
    if not ids:
        return {}
    return {h.event_id: h for h in session.scalars(select(AnnotationHead).where(
        AnnotationHead.event_id.in_(ids), AnnotationHead.status.in_(DONE)))}


# ---------------------------------------------------------------- the clip's timing

def meta_body(session: Session, event_id: int) -> dict:
    """The box's meta for the clip (newest revision, training copy first), or {}."""
    arts = session.scalars(select(Artifact).where(Artifact.event_id == event_id, Artifact.role == "meta")).all()
    for art in sorted(arts, key=lambda a: (not a.s3_key.startswith("dataset_"), a.s3_key)):
        body = session.scalar(select(RawRevision.body).where(RawRevision.s3_key == art.s3_key)
                              .order_by(RawRevision.fetched_at.desc(), RawRevision.id.desc()).limit(1))
        if isinstance(body, dict):
            return body
    return {}


def _positive(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def clip_timing(ev: Event, meta: dict) -> tuple[Optional[float], Optional[int], Optional[float]]:
    """(fps, frame_count, duration_sec) of the clip.

    fps is the rate the box wrote the clip at (its decoded frame rate: frames / decoded duration), else frames
    written / duration. frame_count is the frames the box wrote, else duration x fps. duration_sec is the longer
    of the box's duration and frame_count / fps (the range keyframe times may take)."""
    frames = meta.get("frames_written")
    frames = frames if type(frames) is int and frames > 0 else None
    duration = _positive(ev.duration_sec)
    fps = _positive(ev.fps)
    if fps is None and frames and duration:
        fps = frames / duration
    if frames is None and fps and duration:
        frames = int(round(duration * fps))
    spans = [d for d in (duration, frames / fps if frames and fps else None) if d]
    return fps, frames, (max(spans) if spans else None)


# ---------------------------------------------------------------- tracks

def to_tracks(raw: Any) -> list[ft.Track]:
    """Stored/contract track dicts -> fleet_contract tracks (keyframes sorted by time)."""
    out = []
    for tr in raw or []:
        kfs = [ft.Keyframe(frame=int(k["frame"]), t_sec=float(k["t_sec"]), xyxy=[float(v) for v in k["xyxy"]],
                           enabled=bool(k.get("enabled", True))) for k in tr.get("keyframes", [])]
        kfs.sort(key=lambda k: k.t_sec)
        out.append(ft.Track(track_id=str(tr["track_id"]), label=str(tr["label"]), keyframes=kfs,
                            source=tr.get("source", "human"), entity=tr.get("entity")))
    return out


def track_dicts(tracks: list[ft.Track]) -> list[dict]:
    """Tracks as stored / sent; ``entity`` only when the track has one (an old save keeps its exact shape)."""
    return [{"track_id": t.track_id, "label": t.label, "source": t.source,
             **({"entity": t.entity} if getattr(t, "entity", None) else {}),
             "keyframes": [{"frame": k.frame, "t_sec": k.t_sec, "xyxy": list(k.xyxy), "enabled": k.enabled}
                           for k in t.keyframes]} for t in tracks]


_TRACK_ID = re.compile(r"t-([0-9]{1,9})")


def assign_track_ids(tracks: list[ft.Track], earlier: list[Annotation], current: int) -> None:
    """Give every track a server-assigned opaque id `t-<n>` (in place): a client may keep an id of the current
    version's tracks (the same object across saves); any other id -- free text that could carry a name, a
    suggestion's id, a duplicate -- becomes the next unused number of the clip."""
    used, keep = 0, set()
    for row in earlier:
        for tr in row.tracks or []:
            m = _TRACK_ID.fullmatch(str(tr.get("track_id", "")))
            if m:
                used = max(used, int(m.group(1)))
                if row.version == current:
                    keep.add(m.group(0))
    seen: set[str] = set()
    for tr in tracks:
        if tr.track_id not in keep or tr.track_id in seen:
            used += 1
            tr.track_id = f"t-{used}"
        seen.add(tr.track_id)


def problems(tracks: list[ft.Track], duration: Optional[float], frame_count: Optional[int]) -> list[str]:
    """Why a set of tracks cannot be saved (empty list = it can)."""
    out: list[str] = []
    if len(tracks) > MAX_TRACKS:
        return [f"at most {MAX_TRACKS} tracks"]
    if sum(len(t.keyframes) for t in tracks) > MAX_KEYFRAMES:
        return [f"at most {MAX_KEYFRAMES} keyframes in all"]
    seen: set[str] = set()
    for tr in tracks:
        if not tr.track_id or len(tr.track_id) > 64:
            out.append("every track needs an id of at most 64 characters")
        elif tr.track_id in seen:
            out.append(f"track {tr.track_id}: the id is used twice")
        seen.add(tr.track_id)
        if not tr.keyframes:
            out.append(f"track {tr.track_id}: no keyframes")
        for k in tr.keyframes:
            if k.frame < 0 or (frame_count and k.frame >= frame_count):
                out.append(f"track {tr.track_id}: frame {k.frame} is outside the clip")
            if not math.isfinite(k.t_sec):
                out.append(f"track {tr.track_id}: keyframe time must be a number")
    limit = duration if duration else float("inf")
    out += ft.validate_tracks(tracks, limit)
    return list(dict.fromkeys(out))


def suggestion_sources(session: Session, ev: Event, fps: Optional[float]) -> dict:
    """What a clip's suggestions are computed from (database only): fps, the applied meta revision, the label
    files' etags."""
    from .routes.events import weak_label_sources

    return {"fps": fps, **weak_label_sources(session, ev)}


def _compute_suggestions(session: Session, s3, ev: Event, fps: Optional[float],
                         now: datetime) -> tuple[list[ft.Track], bool]:
    """(tracks, complete): complete = every sampled frame's label file was read."""
    from .routes.events import weak_label_frames  # the one reader of weak labels (shared with /detections)

    frames, complete = [], True
    for f in weak_label_frames(session, s3, ev, now, fps):
        if f.status == "not_run":
            complete = False
            continue
        frames.append((f.frame_index, f.t_sec, [(b.label, b.xyxy) for b in f.boxes if b.label in _KNOWN]))
    tracks = ft.tracks_from_weak_labels(frames)
    for n, tr in enumerate(tracks, start=1):
        tr.track_id = f"t-{n}"
    return tracks, complete


def suggestions(session: Session, s3, ev: Event, fps: Optional[float], now: datetime,
                store: bool = True) -> list[ft.Track]:
    """Suggestion tracks linked from the event's YOLO weak labels (fleet_contract.tracks.tracks_from_weak_labels).
    Frames whose label file could not be read are left out (unknown, not empty); classes outside the nine are
    dropped. Precomputed by the media loop (annotation_suggestions) and used while their sources are current;
    otherwise computed now (label files read concurrently) and stored when every file was read."""
    comp = ev.completeness if isinstance(ev.completeness, dict) else {}
    if s3 is None or comp.get("boxes") != "sampled":
        return []
    sources = suggestion_sources(session, ev, fps)
    row = session.get(AnnotationSuggestion, ev.id, populate_existing=True)
    if row is not None and row.sources == sources:
        return to_tracks(row.tracks)
    tracks, complete = _compute_suggestions(session, s3, ev, fps, now)
    if complete and store:  # a long job (the publish) never holds the row: it only reads
        _store_suggestions(session, ev.id, tracks, sources, now)
    return tracks


def _store_suggestions(session: Session, event_id: int, tracks: list[ft.Track], sources: dict,
                       now: datetime) -> None:
    values = dict(event_id=event_id, tracks=track_dicts(tracks), sources=sources, computed_at=now)
    session.execute(pg_insert(AnnotationSuggestion).values(**values).on_conflict_do_update(
        index_elements=[AnnotationSuggestion.event_id], set_={k: v for k, v in values.items() if k != "event_id"}))


def precompute_suggestions(session: Session, s3, now: Optional[datetime] = None, limit: int = 25) -> int:
    """Media loop step: compute the suggestions of up to `limit` clips with weak labels that have none yet, or whose
    label/meta files changed since (newest clips first); commits per clip. Returns how many were stored."""
    now = now or datetime.now(timezone.utc)
    if s3 is None:
        return 0
    sug = AnnotationSuggestion
    changed = exists().where(Artifact.event_id == Event.id, Artifact.role.in_(("meta", "yolo_label")),
                             Artifact.last_modified > sug.computed_at)
    ids = list(session.scalars(
        select(Event.id).outerjoin(sug, sug.event_id == Event.id)
        .where(Event.completeness["boxes"].as_string() == "sampled", or_(sug.event_id.is_(None), changed))
        .order_by(Event.start_ts.desc(), Event.id.desc()).limit(limit)))
    stored = 0
    for event_id in ids:
        try:
            ev = session.get(Event, event_id)
            fps = clip_timing(ev, meta_body(session, ev.id))[0]
            sources = suggestion_sources(session, ev, fps)
            tracks, complete = _compute_suggestions(session, s3, ev, fps, now)
            if complete:
                _store_suggestions(session, ev.id, tracks, sources, now)
                stored += 1
            session.commit()
        except Exception:  # noqa: BLE001 -- one clip never stops the pass; the request path computes on demand
            log.exception("suggestions for event %s failed", event_id)
            session.rollback()
    return stored


TRACKS_CACHE: dict = {}  # (tracks.json key, etag) -> document or None


def tracker_tracks(session: Session, s3, ev: Event, fps: Optional[float]) -> list[ft.Track]:
    """The box tracker's tracks of the clip: its indexed ``responses/<camera>/<day>/<stem>.tracks.json`` (artifact
    role "tracks"; box clip_tracks.py, the format of tagstudio/tracks_file.py), the kept training copy first; []
    when there is none or it cannot be read. Read once per revision (etag): reopening a clip reads nothing."""
    from .tagstudio import tracks_file  # noqa: PLC0415

    if s3 is None:
        return []
    files = session.execute(select(Artifact.s3_key, Artifact.etag).where(
        Artifact.event_id == ev.id, Artifact.role == "tracks", Artifact.available.is_(True))).all()
    for key, etag in sorted(files, key=lambda r: (not r[0].startswith("dataset_"), r[0])):
        if (key, etag) not in TRACKS_CACHE:
            try:
                doc = s3.get_json(key)
            except Exception:  # noqa: BLE001 - unreadable: the weak labels are used instead
                doc = None
            if len(TRACKS_CACHE) > 5000:
                TRACKS_CACHE.clear()
            TRACKS_CACHE[(key, etag)] = doc if isinstance(doc, dict) else None
        doc = TRACKS_CACHE[(key, etag)]
        tracks = tracks_file.read_tracks(doc, fps) if doc else []
        if tracks:
            return tracks
    return []


def yolo_rows(tracks: list[ft.Track], t_sec: float) -> list[str]:
    """YOLO label rows (contiguous class ids, `cls xc yc w h` normalised) of every track visible at `t_sec`."""
    rows = []
    for label, box in ft.boxes_at(tracks, t_sec):
        coco = name_to_coco(label)
        index = coco_to_contiguous(coco) if coco is not None else None
        if index is None:
            continue
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in box)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        rows.append(f"{index} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}")
    return rows


def label_text(rows: list[str]) -> str:
    """A label file: one row per line; empty for a frame with no box (a human negative)."""
    return ("\n".join(rows) + "\n") if rows else ""


def assistant_answer(parsed: Any, description: str) -> str:
    """The VLM line's assistant text for a human description: the AI's JSON answer with `summary` replaced by the
    description (its other fields kept) when the AI answered in JSON, else the description itself."""
    if isinstance(parsed, dict) and parsed and set(parsed) != {"text"}:
        return json.dumps({**parsed, "summary": description}, ensure_ascii=False)
    return description


# ---------------------------------------------------------------- every frame of a clip

def frame_times(src: Path, ffprobe: str = "ffprobe") -> list[float]:
    """Presentation time (seconds from the first frame) of every decoded frame of the clip's video stream, exact:
    each frame's timestamp in stream time-base units times the time base (NaN for a frame without one)."""
    r = media._run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "frame=best_effort_timestamp:stream=time_base", "-of", "json", str(src)])
    data = json.loads(r.stdout or b"{}")
    base = Fraction(0)
    for stream in data.get("streams", []):
        try:
            base = Fraction(str(stream.get("time_base") or "0"))
        except (ValueError, ZeroDivisionError):
            base = Fraction(0)
    ticks = [f.get("best_effort_timestamp") for f in data.get("frames", [])]
    known = [t for t in ticks if isinstance(t, int)]
    if not known or base <= 0:
        return [math.nan] * len(ticks)
    t0 = min(known)
    return [float((t - t0) * base) if isinstance(t, int) else math.nan for t in ticks]


def extract_frames(src: Path, work: Path, fps: Optional[float], ffmpeg: str = "ffmpeg",
                   ffprobe: str = "ffprobe", times: Optional[list[float]] = None) -> list[tuple[int, float, Path]]:
    """Every decoded frame of the clip as a JPEG: [(native 0-based frame index, t_sec, image path)].

    One ffmpeg pass with `-fps_mode passthrough` writes decoded frame n (0-based) as f_<n+1>.jpg: exactly the frame
    `select=eq(n\\,i)` picks for i = n, without decoding the clip once per frame. Times are the frames' own
    presentation timestamps (ffprobe, or `times` when the caller already probed them); a frame without one gets
    index / fps."""
    out_dir = work / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    local = work / "src.mp4"
    if src.resolve() != local.resolve():
        local.write_bytes(src.read_bytes())
    media._run(media._ffmpeg(ffmpeg, "-i", "src.mp4", "-map", "0:v:0", "-fps_mode", "passthrough", "-q:v", "2",
                             "frames/f_%06d.jpg"), cwd=work)
    files = sorted(out_dir.glob("f_*.jpg"))
    if not files:
        raise media.MediaError("no frames decoded")
    times = frame_times(local, ffprobe) if times is None else times
    rate = fps if fps and fps > 0 else None
    result = []
    for n, path in enumerate(files):
        t = times[n] if n < len(times) and not math.isnan(times[n]) else (n / rate if rate else 0.0)
        result.append((n, float(t), path))
    return result
