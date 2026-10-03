"""In-app labeling: a clip's annotation versions, reviews and current head; suggestions from the YOLO weak labels;
the clip's timing; and what a human annotation gives the training data -- the YOLO rows of any frame (by
`fleet_contract.tracks.box_at`, contiguous class ids) and every decoded frame of the clip as an image.

Used by the annotation routes, the training export (studio.py) and the tagging publish (tagging.py), so all three
read a human annotation the same way.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import tracks as ft
from home_guard_project.fleet_contract.classes import COCO_NAMES, CONTIGUOUS, name_to_coco

from . import media
from .models import AiRun, Annotation, AnnotationHead, AnnotationReview, Artifact, Event, RawRevision

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
                            source=tr.get("source", "human")))
    return out


def track_dicts(tracks: list[ft.Track]) -> list[dict]:
    return [{"track_id": t.track_id, "label": t.label, "source": t.source,
             "keyframes": [{"frame": k.frame, "t_sec": k.t_sec, "xyxy": list(k.xyxy), "enabled": k.enabled}
                           for k in t.keyframes]} for t in tracks]


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


def suggestions(session: Session, s3, ev: Event, fps: Optional[float], now: datetime) -> list[ft.Track]:
    """Suggestion tracks linked from the event's YOLO weak labels (fleet_contract.tracks.tracks_from_weak_labels).
    Frames whose label file could not be read are left out (unknown, not empty); classes outside the nine are
    dropped."""
    comp = ev.completeness if isinstance(ev.completeness, dict) else {}
    if s3 is None or comp.get("boxes") != "sampled":
        return []
    from .routes.events import weak_label_frames  # the one reader of weak labels (shared with /detections)

    frames = []
    for f in weak_label_frames(session, s3, ev, now, fps):
        if f.status == "not_run":
            continue
        frames.append((f.frame_index, f.t_sec, [(b.label, b.xyxy) for b in f.boxes if b.label in _KNOWN]))
    return ft.tracks_from_weak_labels(frames)


def yolo_rows(tracks: list[ft.Track], t_sec: float) -> list[str]:
    """YOLO label rows (contiguous class ids, `cls xc yc w h` normalised) of every track visible at `t_sec`."""
    rows = []
    for label, box in ft.boxes_at(tracks, t_sec):
        coco = name_to_coco(label)
        if coco is None:
            continue
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in box)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        rows.append(f"{CONTIGUOUS[coco]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}")
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
    """Presentation time (seconds from the first frame) of every decoded frame of the clip's video stream."""
    r = media._run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(src)])
    out: list[Optional[float]] = []
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        value = line.strip().strip(",")
        if not value:
            continue
        try:
            out.append(float(value))
        except ValueError:
            out.append(None)
    known = [t for t in out if t is not None]
    t0 = min(known) if known else 0.0
    return [(t - t0) if t is not None else math.nan for t in out]


def extract_frames(src: Path, work: Path, fps: Optional[float], ffmpeg: str = "ffmpeg",
                   ffprobe: str = "ffprobe") -> list[tuple[int, float, Path]]:
    """Every decoded frame of the clip as a JPEG: [(native 0-based frame index, t_sec, image path)].

    One ffmpeg pass with `-fps_mode passthrough` writes decoded frame n (0-based) as f_<n+1>.jpg: exactly the frame
    `select=eq(n\\,i)` picks for i = n, without decoding the clip once per frame. Times are the frames' own
    presentation timestamps (ffprobe); a frame without one gets index / fps."""
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
    times = frame_times(local, ffprobe)
    rate = fps if fps and fps > 0 else None
    result = []
    for n, path in enumerate(files):
        t = times[n] if n < len(times) and not math.isnan(times[n]) else (n / rate if rate else 0.0)
        result.append((n, float(t), path))
    return result
