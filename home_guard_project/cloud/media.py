"""Media worker: thumbnails, filmstrip sprites and browser/Qt-safe H.264 renditions (via ffmpeg).

Failures never propagate: they are recorded as IndexProblem rows ("media: <error>") about the clip's ETag
(thumbnail/filmstrip failures keyed by the thumbnail key, rendition failures by the rendition key). Every cloud
artifact carries detail.src_etag (the clip it was made from).

Failures come in two classes:
- transient (S3 or network errors, timeouts, ffmpeg missing, a full disk): retried with backoff -- attempt n waits
  2^n minutes (`next_retry_at`), and after MAX_ATTEMPTS the clip is left alone (`next_retry_at` NULL);
- permanent (a downloaded file ffmpeg cannot decode, a clip over the size or length caps): quarantined at once.
Either way the part is skipped until the clip changes or an operator runs `manage media-retry [--event ID]`.

Small-server caps: every ffmpeg run uses at most FFMPEG_THREADS threads, and clips over MAX_CLIP_BYTES or
MAX_CLIP_SECONDS are never downloaded or rendered (a permanent "too large").

Retention: media made in the cloud never outlives its source. When none of an event's clips is available any
more (the box's retention deleted them), its thumbnails, filmstrips and renditions -- and every labeler opaque
copy of the event's files -- are deleted from S3 and marked unavailable (`retire_orphans`, each media pass).
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


def _problem_key(event_id: int, kind: str = "thumb") -> str:
    if kind == "rendition":
        return f"admin_cache/renditions/{event_id}.mp4"
    return f"admin_cache/thumbs/{event_id}.jpg"


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
        stmt = stmt.where(IndexProblem.s3_key.in_([_problem_key(event_id), _problem_key(event_id, "rendition")]))
    return session.execute(stmt.execution_options(synchronize_session=False)).rowcount or 0


def ensure_media(session: Session, s3: S3, event: Event, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                 workdir: Optional[Path] = None, now: Optional[datetime] = None) -> list:
    """Create whichever of thumbnail/filmstrip/rendition the event lacks (or has from a replaced clip).

    Each artifact is committed as soon as it is uploaded. Thumbnail/filmstrip failures raise; a rendition
    failure is recorded as its own problem and leaves the other artifacts alone."""
    clip = _clip_for(session, event)
    if clip is None:
        return []
    src_etag = clip.etag
    if clip.bytes is not None and clip.bytes > MAX_CLIP_BYTES:
        raise MediaError(f"too large: {clip.bytes // 1024 ** 2} MB (at most {MAX_CLIP_BYTES // 1024 ** 2} MB)")
    current = {}
    for a in session.scalars(select(Artifact).where(
            Artifact.event_id == event.id, Artifact.provenance == "cloud", Artifact.available.is_(True))):
        if (a.detail or {}).get("src_etag") == src_etag:
            current[a.role] = a
    created: list = []
    with tempfile.TemporaryDirectory(prefix="hgmedia_", dir=workdir) as tmp:
        work = Path(tmp)
        src = work / "src.mp4"
        s3.download_file(clip.s3_key, src)
        info = probe(src, ffprobe)
        if info["duration"] > MAX_CLIP_SECONDS:
            raise MediaError(f"too large: {info['duration'] / 60:.0f} minutes (at most {MAX_CLIP_SECONDS // 60})")
        need = needs_rendition(info, src)
        if "thumbnail" not in current or current["thumbnail"].detail.get("needs_rendition") != need:
            out = _make_thumb(work, info["duration"], ffmpeg)
            created.append(_upsert(session, s3, event, "thumbnail", _problem_key(event.id), out, "image/jpeg",
                                   {"src_etag": src_etag, "needs_rendition": need}))
            session.commit()
        if "filmstrip" not in current:
            out, detail = _make_filmstrip(work, info["duration"], ffmpeg, ffprobe)
            detail["src_etag"] = src_etag
            created.append(_upsert(session, s3, event, "filmstrip", f"admin_cache/filmstrips/{event.id}.jpg",
                                   out, "image/jpeg", detail))
            session.commit()
        rkey = _problem_key(event.id, "rendition")
        if not need:
            for a in session.scalars(select(Artifact).where(
                    Artifact.event_id == event.id, Artifact.role == "rendition", Artifact.provenance == "cloud",
                    Artifact.available.is_(True))):
                a.available = False
            _clear_problem(session, rkey)
            session.commit()
        elif "rendition" not in current:
            try:
                out = _make_rendition(work, ffmpeg)
                created.append(_upsert(session, s3, event, "rendition", rkey, out, "video/mp4",
                                       {"src_etag": src_etag}))
                _clear_problem(session, rkey)
                session.commit()
            except Exception as e:  # noqa: BLE001
                log.warning("media: event %s rendition failed: %s", event.id, e)
                _record_problem(session, rkey, src_etag, str(e) or type(e).__name__, is_transient(e), now)
    return created


def _no_clip(event_id_col):
    """Condition: the event has no available original clip."""
    clip = aliased(Artifact)
    return ~exists().where(clip.event_id == event_id_col, clip.role == "original_video",
                           clip.available.is_(True))


def retire_orphans(session: Session, s3: S3, limit: int = RETIRE_BATCH) -> int:
    """Delete cloud-made media whose source is gone: the thumbnails, filmstrips and renditions of events with no
    available clip, and labeler opaque copies whose source artifact is unavailable or belongs to such an event.
    Objects are deleted first and only then marked unavailable, so a failed delete is retried next pass."""
    derived = session.scalars(select(Artifact).where(
        Artifact.provenance == "cloud", Artifact.role.in_(DERIVED_ROLES), Artifact.available.is_(True),
        Artifact.event_id.is_not(None), _no_clip(Artifact.event_id)).order_by(Artifact.id).limit(limit)).all()
    src = aliased(Artifact)
    source_id = cast(Artifact.detail["source_artifact_id"].as_string(), Integer)
    copies = session.scalars(select(Artifact).outerjoin(src, src.id == source_id).where(
        Artifact.role == "opaque_copy", Artifact.available.is_(True),
        or_(src.id.is_(None), src.available.is_(False),
            and_(src.event_id.is_not(None), _no_clip(src.event_id)),
            src.id.in_([a.id for a in derived] or [-1])))
        .order_by(Artifact.id).limit(limit)).all()
    gone = list(derived) + list(copies)
    if not gone:
        return 0
    s3.delete_keys([a.s3_key for a in gone])
    for a in gone:
        a.available = False
    session.commit()
    log.info("media: retired %d cloud file(s) whose source clip is gone", len(gone))
    return len(gone)


def _pending_ids(session: Session, limit: int, now: Optional[datetime] = None) -> list:
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

    def problem(key_expr):
        """A media problem about this clip revision that is not due for a retry yet (or never will be)."""
        p = aliased(IndexProblem)
        return select(p.s3_key).where(p.s3_key == key_expr, p.reason.like("media:%"), p.etag == clip.etag,
                                      or_(p.next_retry_at.is_(None), p.next_retry_at > now))

    event_id = cast(Event.id, String)
    thumb_key = "admin_cache/thumbs/" + event_id + ".jpg"
    rend_key = "admin_cache/renditions/" + event_id + ".mp4"
    base_missing = or_(~cloud("thumbnail", fresh).exists(), ~cloud("filmstrip", fresh).exists())
    rend_missing = and_(cloud("thumbnail", fresh, wants).exists(), ~cloud("rendition", fresh).exists())
    q = (select(Event.id).join(clip, clip.id == preferred)
         .where(or_(and_(base_missing, ~problem(thumb_key).exists()),
                    and_(rend_missing, ~problem(rend_key).exists())))
         .order_by(Event.start_ts.desc(), Event.id.desc()).limit(limit))
    return list(session.scalars(q))


def process_pending(session: Session, s3: S3, limit: int = 50, ffmpeg: str = "ffmpeg",
                    ffprobe: str = "ffprobe", workdir: Optional[Path] = None,
                    now: Optional[datetime] = None) -> int:
    """Retire media whose source is gone, then attempt media for up to `limit` events (successes and failures
    both count); returns how many events gained artifacts. Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        retire_orphans(session, s3)
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
            _clear_problem(session, _problem_key(event_id))
            session.commit()
            done += 1 if made else 0
        except Exception as e:  # noqa: BLE001
            transient = is_transient(e)
            log.warning("media: event %s failed (%s): %s", event_id, "will retry" if transient else "quarantined", e)
            try:
                _record_problem(session, _problem_key(event_id), clip_etag, str(e) or type(e).__name__,
                                transient, now)
            except Exception:  # noqa: BLE001
                log.exception("media: could not record problem")
                session.rollback()
    return done
