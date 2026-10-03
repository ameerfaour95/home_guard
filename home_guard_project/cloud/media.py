"""Media worker: thumbnails, filmstrip sprites and browser/Qt-safe H.264 renditions (via ffmpeg).

Failures never propagate: they are recorded as IndexProblem rows ("media: <error>") about the clip's ETag. Each
stage -- thumbnail, filmstrip, rendition -- has its own row, keyed by the stage's S3 key, so its own attempts and
next retry: a failed stage is skipped until its retry is due and never blocks the others, and an event is picked
for work only when some missing stage may run now. Every cloud artifact carries detail.src_etag (the clip it was
made from).

Failures come in two classes:
- transient (S3 or network errors, timeouts, ffmpeg missing, a full disk): retried with backoff -- attempt n waits
  2^n minutes (`next_retry_at`), and after MAX_ATTEMPTS the clip is left alone (`next_retry_at` NULL);
- permanent (a downloaded file ffmpeg cannot decode, a clip over the size or length caps): quarantined at once.
Either way the part is skipped until the clip changes or an operator runs `manage media-retry [--event ID]`.

Small-server caps: every ffmpeg run uses at most FFMPEG_THREADS threads, and clips over MAX_CLIP_BYTES or
MAX_CLIP_SECONDS are never downloaded or rendered (a permanent "too large").

Retention: media made in the cloud never outlives its source. When none of an event's clips is available any
more (the box's retention deleted them), its thumbnails, filmstrips and renditions -- and every labeler opaque
copy of the event's files -- are deleted from S3 and marked unavailable (`retire_orphans`, each media pass). Only keys S3 reports deleted
are marked; a key DeleteObjects refused stays available with a "media: delete failed" problem counting attempts,
and is retried with backoff.
"""
from __future__ import annotations

import json
import logging
import struct
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Integer, String, and_, case, cast, delete, exists, or_, select
from sqlalchemy.orm import Session, aliased

from .models import Artifact, Event, IndexProblem
from .s3 import S3

log = logging.getLogger(__name__)

TIMEOUT = 120
THUMB_W = 480
TILE_W = 160
MAX_TILES = 60
MAX_BOXES = 10000
FFMPEG_THREADS = "2"
MAX_CLIP_BYTES = 500 * 1024 ** 2
MAX_CLIP_SECONDS = 15 * 60
MAX_ATTEMPTS = 8
RETIRE_BATCH = 500
DERIVED_ROLES = ("thumbnail", "filmstrip", "rendition")
STAGES = DERIVED_ROLES  # the media stages of an event, each with its own retry state


class MediaError(Exception):
    """A media failure; `transient` ones are retried with backoff, the others quarantine the clip."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def is_transient(exc: BaseException) -> bool:
    """Whether a failure may go away by itself (retry later) rather than being a property of the clip."""
    if isinstance(exc, MediaError):
        return exc.transient
    return True  # S3/network/database errors, a full disk, ...: anything that is not a verdict on the file


def _run(cmd: list, timeout: int = TIMEOUT, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    name = Path(str(cmd[0])).name
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{name} timed out", transient=True) from e
    except OSError as e:  # not installed, not executable, out of resources
        raise MediaError(f"cannot run {name}: {e}", transient=True) from e
    if r.returncode != 0:
        stderr = r.stderr.decode("utf-8", "replace")
        tail = stderr.strip().splitlines()[-1:] or ["failed"]
        raise MediaError(f"{name} exit {r.returncode}: {tail[0][:200]}",
                         transient="No space left on device" in stderr)
    return r


def _ffmpeg(ffmpeg: str, *args: str) -> list:
    """An ffmpeg command line limited to FFMPEG_THREADS threads (decoding and filtering)."""
    return [ffmpeg, "-threads", FFMPEG_THREADS, "-filter_threads", FFMPEG_THREADS, "-v", "error", "-y", *args]


def probe(path: Path, ffprobe: str) -> dict:
    r = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(r.stdout or b"{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("no video stream")
    try:
        duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if duration <= 0 or width <= 0 or height <= 0:
        raise MediaError("unreadable duration or size")
    return {"codec": video.get("codec_name"), "pix_fmt": video.get("pix_fmt"), "duration": duration,
            "width": width, "height": height}


def moov_first(path: Path) -> bool:
    """True when the mp4's moov box precedes mdat (playable before the whole file downloads)."""
    try:
        with open(path, "rb") as f:
            pos = 0
            for _ in range(MAX_BOXES):
                f.seek(pos)
                head = f.read(16)
                if len(head) < 8:
                    return False
                size, kind = struct.unpack(">I4s", head[:8])
                if kind == b"moov":
                    return True
                if kind == b"mdat":
                    return False
                if size == 1 and len(head) == 16:
                    size = struct.unpack(">Q", head[8:16])[0]
                if size < 8:
                    return False
                pos += size
            return False
    except (OSError, OverflowError, ValueError):
        return False


def needs_rendition(info: dict, path: Path) -> bool:
    return info["codec"] != "h264" or info["pix_fmt"] != "yuv420p" or not moov_first(path)


# ffmpeg runs with cwd=work and relative names, so no path fragment can be misread as a pattern.
def _make_thumb(work: Path, duration: float, ffmpeg: str) -> Path:
    out = work / "thumb.jpg"
    for t in (duration * 0.4, 0.0):
        _run(_ffmpeg(ffmpeg, "-ss", f"{t:.3f}", "-i", "src.mp4", "-frames:v", "1",
                     "-vf", f"scale={THUMB_W}:-2", "-q:v", "3", "thumb.jpg"), cwd=work)
        if out.exists() and out.stat().st_size > 0:
            return out
    raise MediaError("no thumbnail frame")


def _image_size(path: Path, ffprobe: str) -> tuple:
    r = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(path)])
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def _make_filmstrip(work: Path, duration: float, ffmpeg: str, ffprobe: str) -> tuple:
    out = work / "strip.jpg"
    fps = 1.0 if duration <= MAX_TILES else MAX_TILES / duration
    (work / "frames").mkdir(exist_ok=True)
    _run(_ffmpeg(ffmpeg, "-i", "src.mp4", "-vf", f"fps={fps:.6f},scale={TILE_W}:-2",
                 "-frames:v", str(MAX_TILES), "-q:v", "4", "frames/f_%04d.jpg"), cwd=work)
    count = len(list((work / "frames").glob("f_*.jpg")))
    if count == 0:
        raise MediaError("no filmstrip frames")
    _run(_ffmpeg(ffmpeg, "-framerate", "1", "-i", "frames/f_%04d.jpg",
                 "-vf", f"tile={count}x1", "-frames:v", "1", "-q:v", "4", "strip.jpg"), cwd=work)
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError("filmstrip not written")
    height = _image_size(out, ffprobe)[1]
    return out, {"fps": 1 if fps == 1.0 else round(fps, 4), "tile_w": TILE_W, "tile_h": height, "count": count}


def _make_rendition(work: Path, ffmpeg: str) -> Path:
    _run(_ffmpeg(ffmpeg, "-i", "src.mp4", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26",
                 "-threads", FFMPEG_THREADS, "-movflags", "+faststart", "-an", "rendition.mp4"), cwd=work)
    return work / "rendition.mp4"


def _clip_for(session: Session, event: Event) -> Optional[Artifact]:
    clips = session.scalars(select(Artifact).where(
        Artifact.event_id == event.id, Artifact.role == "original_video", Artifact.available.is_(True))).all()
    clips.sort(key=lambda a: (not a.s3_key.startswith("production_"), a.s3_key))
    return clips[0] if clips else None


def _upsert(session: Session, s3: S3, event: Event, role: str, key: str, path: Path, mime: str,
            detail: dict) -> Artifact:
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    s3.upload_file(path, key, mime)
    head = s3.client.head_object(Bucket=s3.bucket, Key=key)
    # conflict-safe: a row for this key made by anyone else is taken over, never a unique violation
    session.execute(pg_insert(Artifact).values(s3_key=key, event_id=event.id, role=role, provenance="cloud")
                    .on_conflict_do_nothing(index_elements=[Artifact.s3_key]))
    art = session.scalars(select(Artifact).where(Artifact.s3_key == key)).one()
    art.event_id, art.role, art.provenance, art.available = event.id, role, "cloud", True
    art.camera, art.stem, art.mime, art.detail = event.camera, event.stem, mime, detail
    art.etag = head.get("ETag", "").strip('"')
    art.bytes = int(head.get("ContentLength", path.stat().st_size))
    art.last_modified = head.get("LastModified")
    return art




def _stage_key(event_id: int, stage: str) -> str:
    """The S3 key of one media stage of an event -- also the key of that stage's IndexProblem (retry state)."""
    return {"thumbnail": f"admin_cache/thumbs/{event_id}.jpg",
            "filmstrip": f"admin_cache/filmstrips/{event_id}.jpg",
            "rendition": f"admin_cache/renditions/{event_id}.mp4"}[stage]


def _problem_key(event_id: int, kind: str = "thumb") -> str:
    return _stage_key(event_id, "rendition" if kind == "rendition" else "thumbnail")


def _record_problem(session: Session, key: str, clip_etag: Optional[str], error: str, transient: bool = False,
                    now: Optional[datetime] = None) -> None:
    """Record a media failure about `clip_etag`: a transient one is retried after 2^attempts minutes (up to
    MAX_ATTEMPTS), a permanent one is not retried."""
    session.rollback()
    now = now or datetime.now(timezone.utc)
    row = session.get(IndexProblem, key)
    if row is None:
        row = IndexProblem(s3_key=key, attempts=0)
        session.add(row)
    attempts = (row.attempts or 0) if row.etag == clip_etag else 0
    attempts += 1
    row.reason = f"media: {error}"[:1000]
    row.seen_at = now
    row.etag = clip_etag
    row.attempts = attempts
    row.next_retry_at = (now + timedelta(minutes=2 ** attempts)
                         if transient and attempts < MAX_ATTEMPTS else None)
    session.commit()


def _clear_problem(session: Session, key: str) -> None:
    row = session.get(IndexProblem, key)
    if row is not None and row.reason.startswith("media:"):
        session.delete(row)


def clear_problems(session: Session, event_id: Optional[int] = None) -> int:
    """`manage media-retry`: forget media failures (all, or one event's) so the next media pass tries again."""
    stmt = delete(IndexProblem).where(IndexProblem.reason.like("media:%"))
    if event_id is not None:
        stmt = stmt.where(IndexProblem.s3_key.in_([_stage_key(event_id, stage) for stage in STAGES]))
    return session.execute(stmt.execution_options(synchronize_session=False)).rowcount or 0


def _waiting(session: Session, event_id: int, clip_etag: Optional[str], now: datetime) -> set:
    """The stages of the event whose last failure (about this clip revision) is not due for a retry yet, or
    never will be (permanent, or MAX_ATTEMPTS reached)."""
    keys = {_stage_key(event_id, stage): stage for stage in STAGES}
    rows = session.scalars(select(IndexProblem.s3_key).where(
        IndexProblem.s3_key.in_(list(keys)), IndexProblem.reason.like("media:%"), IndexProblem.etag == clip_etag,
        or_(IndexProblem.next_retry_at.is_(None), IndexProblem.next_retry_at > now)))
    return {keys[k] for k in rows}


def ensure_media(session: Session, s3: S3, event: Event, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                 workdir: Optional[Path] = None, now: Optional[datetime] = None) -> list:
    """Create whichever of thumbnail/filmstrip/rendition the event lacks (or has from a replaced clip).

    Every stage has its own retry state (an IndexProblem keyed by the stage's S3 key): a stage that failed is
    skipped until its retry is due, and a failure of one stage never stops the others. A failure before any stage
    can run (size cap, download, probe, length cap) is recorded for every stage that would have run. Each artifact
    is committed as soon as it is uploaded. Failures are recorded, never raised."""
    clip = _clip_for(session, event)
    if clip is None:
        return []
    now = now or datetime.now(timezone.utc)
    src_etag = clip.etag
    current = {}
    for a in session.scalars(select(Artifact).where(
            Artifact.event_id == event.id, Artifact.provenance == "cloud", Artifact.available.is_(True))):
        if (a.detail or {}).get("src_etag") == src_etag:
            current[a.role] = a
    waiting = _waiting(session, event.id, src_etag, now)
    thumb = current.get("thumbnail")
    # the stages that may run, as far as is known before the clip is probed (the same rule as _pending_ids)
    candidates = [stage for stage, missing in (
        ("thumbnail", thumb is None),
        ("filmstrip", "filmstrip" not in current),
        # whether a rendition is needed is known from the thumbnail (made from the probe); once a thumbnail is made,
        # a rendition is attempted in the same run when the probe says so
        ("rendition", "rendition" not in current and thumb is not None and thumb.detail.get("needs_rendition") is True),
    ) if missing and stage not in waiting]
    if not candidates:
        return []
    created: list = []

    def fail(stages, e: BaseException) -> None:
        transient = is_transient(e)
        for stage in stages:
            log.warning("media: event %s %s failed (%s): %s", event.id, stage,
                        "will retry" if transient else "quarantined", e)
            _record_problem(session, _stage_key(event.id, stage), src_etag, str(e) or type(e).__name__,
                            transient, now)

    def run(stage: str, make) -> None:
        if stage in waiting:
            return
        try:
            created.append(make())
            _clear_problem(session, _stage_key(event.id, stage))
            session.commit()
        except Exception as e:  # noqa: BLE001 -- recorded for this stage only
            fail([stage], e)

    with tempfile.TemporaryDirectory(prefix="hgmedia_", dir=workdir) as tmp:
        work = Path(tmp)
        src = work / "src.mp4"
        try:
            if clip.bytes is not None and clip.bytes > MAX_CLIP_BYTES:
                raise MediaError(f"too large: {clip.bytes // 1024 ** 2} MB (at most {MAX_CLIP_BYTES // 1024 ** 2} MB)")
            s3.download_file(clip.s3_key, src)
            info = probe(src, ffprobe)
            if info["duration"] > MAX_CLIP_SECONDS:
                raise MediaError(f"too large: {info['duration'] / 60:.0f} minutes "
                                 f"(at most {MAX_CLIP_SECONDS // 60})")
            need = needs_rendition(info, src)
        except Exception as e:  # noqa: BLE001 -- no stage could run: each candidate stage records it
            fail(candidates, e)
            return created
        if thumb is None or thumb.detail.get("needs_rendition") != need:
            def make_thumb():
                out = _make_thumb(work, info["duration"], ffmpeg)
                return _upsert(session, s3, event, "thumbnail", _stage_key(event.id, "thumbnail"), out,
                               "image/jpeg", {"src_etag": src_etag, "needs_rendition": need})
            run("thumbnail", make_thumb)
        if "filmstrip" not in current:
            def make_filmstrip():
                out, detail = _make_filmstrip(work, info["duration"], ffmpeg, ffprobe)
                detail["src_etag"] = src_etag
                return _upsert(session, s3, event, "filmstrip", _stage_key(event.id, "filmstrip"), out,
                               "image/jpeg", detail)
            run("filmstrip", make_filmstrip)
        rkey = _stage_key(event.id, "rendition")
        if not need:
            for a in session.scalars(select(Artifact).where(
                    Artifact.event_id == event.id, Artifact.role == "rendition", Artifact.provenance == "cloud",
                    Artifact.available.is_(True))):
                a.available = False
            _clear_problem(session, rkey)
            session.commit()
            if "rendition" in candidates and not any(a.role == "thumbnail" for a in created):
                # selected for a rendition the stored thumbnail asks for, but the probe says none is needed and
                # the thumbnail could not be refreshed: count it as an attempt, so the event cannot loop
                fail(["rendition"], MediaError("no rendition needed; the thumbnail could not be refreshed",
                                               transient=True))
        elif "rendition" not in current:
            def make_rendition():
                out = _make_rendition(work, ffmpeg)
                return _upsert(session, s3, event, "rendition", rkey, out, "video/mp4", {"src_etag": src_etag})
            run("rendition", make_rendition)
    return created


def _no_clip(event_id_col):
    """Condition: the event has no available original clip."""
    clip = aliased(Artifact)
    return ~exists().where(clip.event_id == event_id_col, clip.role == "original_video",
                           clip.available.is_(True))


DELETE_FAILED = "media: delete failed"
MAX_DELETE_BACKOFF_MINUTES = 24 * 60


def _delete_failed(session: Session, art: Artifact, code: str, now: datetime) -> int:
    """Record a failed S3 delete of a retired file: kept available and retried after 2^attempts minutes (at most
    a day apart; never given up -- the file must go). Returns the attempts so far."""
    row = session.get(IndexProblem, art.s3_key)
    if row is None:
        row = IndexProblem(s3_key=art.s3_key, attempts=0)
        session.add(row)
    attempts = (row.attempts or 0) + 1 if (row.reason or "").startswith(DELETE_FAILED) else 1
    row.reason = f"{DELETE_FAILED}: {code}"[:1000]
    row.seen_at, row.etag, row.attempts = now, art.etag, attempts
    row.next_retry_at = now + timedelta(minutes=min(2 ** attempts, MAX_DELETE_BACKOFF_MINUTES))
    return attempts


def retire_orphans(session: Session, s3: S3, limit: int = RETIRE_BATCH, now: Optional[datetime] = None) -> int:
    """Delete cloud-made media whose source is gone: the thumbnails, filmstrips and renditions of events with no
    available clip, and labeler opaque copies whose source artifact is unavailable or belongs to such an event.
    Objects are deleted first; only the keys S3 reports deleted are marked unavailable. A key whose delete failed
    (DeleteObjects' per-key Errors) stays available with a "delete failed" problem counting the attempts, and is
    retried once its backoff is due. Returns how many files were retired."""
    now = now or datetime.now(timezone.utc)
    backing_off = exists().where(IndexProblem.s3_key == Artifact.s3_key,
                                 IndexProblem.reason.like(DELETE_FAILED + "%"), IndexProblem.next_retry_at > now)
    derived = session.scalars(select(Artifact).where(
        Artifact.provenance == "cloud", Artifact.role.in_(DERIVED_ROLES), Artifact.available.is_(True),
        Artifact.event_id.is_not(None), _no_clip(Artifact.event_id), ~backing_off)
        .order_by(Artifact.id).limit(limit)).all()
    src = aliased(Artifact)
    source_id = cast(Artifact.detail["source_artifact_id"].as_string(), Integer)
    copies = session.scalars(select(Artifact).outerjoin(src, src.id == source_id).where(
        Artifact.role == "opaque_copy", Artifact.available.is_(True), ~backing_off,
        or_(src.id.is_(None), src.available.is_(False),
            and_(src.event_id.is_not(None), _no_clip(src.event_id)),
            src.id.in_([a.id for a in derived] or [-1])))
        .order_by(Artifact.id).limit(limit)).all()
    gone = list(derived) + list(copies)
    if not gone:
        return 0
    failed = s3.delete_keys([a.s3_key for a in gone])
    retired = []
    for a in gone:
        if a.s3_key in failed:
            attempts = _delete_failed(session, a, failed[a.s3_key], now)
            log.warning("media: could not delete retired file %s (%s), attempt %d; will retry", a.s3_key,
                        failed[a.s3_key], attempts)
            continue
        a.available = False
        retired.append(a.s3_key)
    if retired:
        session.execute(delete(IndexProblem).where(IndexProblem.s3_key.in_(retired),
                                                   IndexProblem.reason.like(DELETE_FAILED + "%"))
                        .execution_options(synchronize_session=False))
    session.commit()
    if retired:
        log.info("media: retired %d cloud file(s) whose source clip is gone", len(retired))
    return len(retired)


def _pending_ids(session: Session, limit: int, now: Optional[datetime] = None) -> list:
    """Events with a media stage to do now: a stage that is missing (for the preferred clip's revision) and whose
    own retry state lets it run (no failure about this revision, or its retry is due)."""
    now = now or datetime.now(timezone.utc)
    clip = aliased(Artifact)
    preferred = (select(Artifact.id)
                 .where(Artifact.event_id == Event.id, Artifact.role == "original_video",
                        Artifact.available.is_(True))
                 .order_by(case((Artifact.s3_key.startswith("production_", autoescape=True), 0), else_=1),
                           Artifact.s3_key)
                 .limit(1).correlate(Event).scalar_subquery())

    def cloud(role, *extra):
        a = aliased(Artifact)
        return select(a.id).where(a.event_id == Event.id, a.role == role, a.provenance == "cloud",
                                  a.available.is_(True), *[f(a) for f in extra])

    def fresh(a):
        return a.detail["src_etag"].as_string() == clip.etag

    def wants(a):
        return a.detail["needs_rendition"].as_boolean().is_(True)

    event_id = cast(Event.id, String)

    def free(stage):
        """No failure of this stage about this clip revision that is still waiting (or never retried)."""
        p = aliased(IndexProblem)
        key = {"thumbnail": "admin_cache/thumbs/" + event_id + ".jpg",
               "filmstrip": "admin_cache/filmstrips/" + event_id + ".jpg",
               "rendition": "admin_cache/renditions/" + event_id + ".mp4"}[stage]
        return ~select(p.s3_key).where(p.s3_key == key, p.reason.like("media:%"), p.etag == clip.etag,
                                       or_(p.next_retry_at.is_(None), p.next_retry_at > now)).exists()

    thumb_missing = ~cloud("thumbnail", fresh).exists()
    strip_missing = ~cloud("filmstrip", fresh).exists()
    rend_missing = and_(cloud("thumbnail", fresh, wants).exists(), ~cloud("rendition", fresh).exists())
    q = (select(Event.id).join(clip, clip.id == preferred)
         .where(or_(and_(thumb_missing, free("thumbnail")),
                    and_(strip_missing, free("filmstrip")),
                    and_(rend_missing, free("rendition"))))
         .order_by(Event.start_ts.desc(), Event.id.desc()).limit(limit))
    return list(session.scalars(q))


def process_pending(session: Session, s3: S3, limit: int = 50, ffmpeg: str = "ffmpeg",
                    ffprobe: str = "ffprobe", workdir: Optional[Path] = None,
                    now: Optional[datetime] = None) -> int:
    """Retire media whose source is gone, then attempt media for up to `limit` events (successes and failures
    both count); returns how many events gained artifacts. Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        retire_orphans(session, s3, now=now)
    except Exception:  # noqa: BLE001 -- retried on the next pass
        log.exception("media: could not retire media of deleted clips")
        session.rollback()
    try:
        event_ids = _pending_ids(session, limit, now)
    except Exception:  # noqa: BLE001
        log.exception("media: could not list pending events")
        session.rollback()
        return 0
    done = 0
    for event_id in event_ids:
        clip_etag = None
        try:
            event = session.get(Event, event_id)
            clip = _clip_for(session, event) if event is not None else None
            if clip is None:
                continue
            clip_etag = clip.etag
            made = ensure_media(session, s3, event, ffmpeg, ffprobe, workdir, now)
            session.commit()
            done += 1 if made else 0
        except Exception as e:  # noqa: BLE001 -- not a stage failure (ensure_media records those): every stage
            transient = is_transient(e)
            log.warning("media: event %s failed (%s): %s", event_id, "will retry" if transient else "quarantined", e)
            try:
                for stage in STAGES:
                    _record_problem(session, _stage_key(event_id, stage), clip_etag, str(e) or type(e).__name__,
                                    transient, now)
            except Exception:  # noqa: BLE001
                log.exception("media: could not record problem")
                session.rollback()
    return done
