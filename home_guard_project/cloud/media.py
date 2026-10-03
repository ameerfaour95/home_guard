"""Media worker: thumbnails, filmstrip sprites and browser/Qt-safe H.264 renditions (via ffmpeg).

Failures never propagate: they are recorded as IndexProblem rows ("media: <error>") about the clip's ETag
(thumbnail/filmstrip failures keyed by the thumbnail key, rendition failures by the rendition key); that part is
skipped until the clip changes. Every cloud artifact carries detail.src_etag (the clip it was made from).
"""
from __future__ import annotations

import json
import logging
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import String, and_, case, cast, or_, select
from sqlalchemy.orm import Session, aliased

from .models import Artifact, Event, IndexProblem
from .s3 import S3

log = logging.getLogger(__name__)

TIMEOUT = 120
THUMB_W = 480
TILE_W = 160
MAX_TILES = 60
MAX_BOXES = 10000


class MediaError(Exception):
    pass


def _run(cmd: list, timeout: int = TIMEOUT, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    name = Path(str(cmd[0])).name
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{name} timed out") from e
    except OSError as e:
        raise MediaError(f"cannot run {name}: {e}") from e
    if r.returncode != 0:
        tail = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["failed"]
        raise MediaError(f"{name} exit {r.returncode}: {tail[0][:200]}")
    return r


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
        _run([ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", "src.mp4", "-frames:v", "1",
              "-vf", f"scale={THUMB_W}:-2", "-q:v", "3", "thumb.jpg"], cwd=work)
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
    _run([ffmpeg, "-v", "error", "-y", "-i", "src.mp4", "-vf", f"fps={fps:.6f},scale={TILE_W}:-2",
          "-frames:v", str(MAX_TILES), "-q:v", "4", "frames/f_%04d.jpg"], cwd=work)
    count = len(list((work / "frames").glob("f_*.jpg")))
    if count == 0:
        raise MediaError("no filmstrip frames")
    _run([ffmpeg, "-v", "error", "-y", "-framerate", "1", "-i", "frames/f_%04d.jpg",
          "-vf", f"tile={count}x1", "-frames:v", "1", "-q:v", "4", "strip.jpg"], cwd=work)
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError("filmstrip not written")
    height = _image_size(out, ffprobe)[1]
    return out, {"fps": 1 if fps == 1.0 else round(fps, 4), "tile_w": TILE_W, "tile_h": height, "count": count}


def _make_rendition(work: Path, ffmpeg: str) -> Path:
    _run([ffmpeg, "-v", "error", "-y", "-i", "src.mp4", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26",
          "-movflags", "+faststart", "-an", "rendition.mp4"], cwd=work)
    return work / "rendition.mp4"


def _clip_for(session: Session, event: Event) -> Optional[Artifact]:
    clips = session.scalars(select(Artifact).where(
        Artifact.event_id == event.id, Artifact.role == "original_video", Artifact.available.is_(True))).all()
    clips.sort(key=lambda a: (not a.s3_key.startswith("production_"), a.s3_key))
    return clips[0] if clips else None


def _upsert(session: Session, s3: S3, event: Event, role: str, key: str, path: Path, mime: str,
            detail: dict) -> Artifact:
    s3.upload_file(path, key, mime)
    head = s3.client.head_object(Bucket=s3.bucket, Key=key)
    art = session.scalars(select(Artifact).where(Artifact.s3_key == key)).first()
    if art is None:
        art = Artifact(s3_key=key, event_id=event.id, role=role)
        session.add(art)
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


def _record_problem(session: Session, key: str, clip_etag: Optional[str], error: str) -> None:
    session.rollback()
    row = session.get(IndexProblem, key)
    if row is None:
        row = IndexProblem(s3_key=key)
        session.add(row)
    row.reason = f"media: {error}"[:1000]
    row.seen_at = datetime.now(timezone.utc)
    row.etag = clip_etag
    session.commit()


def _clear_problem(session: Session, key: str) -> None:
    row = session.get(IndexProblem, key)
    if row is not None and row.reason.startswith("media:"):
        session.delete(row)


def ensure_media(session: Session, s3: S3, event: Event, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                 workdir: Optional[Path] = None) -> list:
    """Create whichever of thumbnail/filmstrip/rendition the event lacks (or has from a replaced clip).

    Each artifact is committed as soon as it is uploaded. Thumbnail/filmstrip failures raise; a rendition
    failure is recorded as its own problem and leaves the other artifacts alone."""
    clip = _clip_for(session, event)
    if clip is None:
        return []
    src_etag = clip.etag
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
                _record_problem(session, rkey, src_etag, str(e) or type(e).__name__)
    return created


def _pending_ids(session: Session, limit: int) -> list:
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
        p = aliased(IndexProblem)
        return select(p.s3_key).where(p.s3_key == key_expr, p.reason.like("media:%"), p.etag == clip.etag)

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
                    ffprobe: str = "ffprobe", workdir: Optional[Path] = None) -> int:
    """Attempt media for up to `limit` events (successes and failures both count); returns how many events
    gained artifacts. Never raises."""
    try:
        event_ids = _pending_ids(session, limit)
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
            made = ensure_media(session, s3, event, ffmpeg, ffprobe, workdir)
            _clear_problem(session, _problem_key(event_id))
            session.commit()
            done += 1 if made else 0
        except Exception as e:  # noqa: BLE001
            log.warning("media: event %s failed: %s", event_id, e)
            try:
                _record_problem(session, _problem_key(event_id), clip_etag, str(e) or type(e).__name__)
            except Exception:  # noqa: BLE001
                log.exception("media: could not record problem")
                session.rollback()
    return done
