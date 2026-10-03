"""Media worker: thumbnails, filmstrip sprites and browser/Qt-safe H.264 renditions (via ffmpeg).

Failures never propagate: they are recorded as IndexProblem rows ("media: <error>") keyed by the event's
thumbnail key, about the clip's ETag; the event is skipped until the clip changes.
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

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Artifact, Event, IndexProblem
from .s3 import S3

log = logging.getLogger(__name__)

TIMEOUT = 120
THUMB_W = 480
TILE_W = 160
MAX_TILES = 60


class MediaError(Exception):
    pass


def _run(cmd: list, timeout: int = TIMEOUT) -> subprocess.CompletedProcess:
    name = Path(str(cmd[0])).name
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
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
    with open(path, "rb") as f:
        pos = 0
        while True:
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


def needs_rendition(info: dict, path: Path) -> bool:
    return info["codec"] != "h264" or info["pix_fmt"] != "yuv420p" or not moov_first(path)


def _make_thumb(src: Path, out: Path, duration: float, ffmpeg: str) -> None:
    for t in (duration * 0.4, 0.0):
        _run([ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
              "-vf", f"scale={THUMB_W}:-2", "-q:v", "3", str(out)])
        if out.exists() and out.stat().st_size > 0:
            return
    raise MediaError("no thumbnail frame")


def _image_size(path: Path, ffprobe: str) -> tuple:
    r = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(path)])
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def _make_filmstrip(src: Path, out: Path, work: Path, duration: float, ffmpeg: str, ffprobe: str) -> dict:
    fps = 1.0 if duration <= MAX_TILES else MAX_TILES / duration
    frames = work / "frames"
    frames.mkdir()
    _run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-vf", f"fps={fps:.6f},scale={TILE_W}:-2",
          "-frames:v", str(MAX_TILES), "-q:v", "4", str(frames / "f_%04d.jpg")])
    count = len(list(frames.glob("f_*.jpg")))
    if count == 0:
        raise MediaError("no filmstrip frames")
    _run([ffmpeg, "-v", "error", "-y", "-framerate", "1", "-i", str(frames / "f_%04d.jpg"),
          "-vf", f"tile={count}x1", "-frames:v", "1", "-q:v", "4", str(out)])
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError("filmstrip not written")
    height = _image_size(out, ffprobe)[1]
    return {"fps": 1 if fps == 1.0 else round(fps, 4), "tile_w": TILE_W, "tile_h": height, "count": count}


def _make_rendition(src: Path, out: Path, ffmpeg: str) -> None:
    _run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26",
          "-movflags", "+faststart", "-an", str(out)])


def _clip_for(session: Session, event: Event) -> Optional[Artifact]:
    clips = session.scalars(select(Artifact).where(
        Artifact.event_id == event.id, Artifact.role == "original_video", Artifact.available.is_(True))).all()
    clips.sort(key=lambda a: (not a.s3_key.startswith("production_"), a.s3_key))
    return clips[0] if clips else None


def _upsert(session: Session, s3: S3, event: Event, role: str, key: str, path: Path, mime: str,
            detail: Optional[dict] = None) -> Artifact:
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


def ensure_media(session: Session, s3: S3, event: Event, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                 workdir: Optional[Path] = None) -> list:
    """Create whichever of thumbnail/filmstrip/rendition the event lacks; returns the new Artifacts."""
    clip = _clip_for(session, event)
    if clip is None:
        return []
    have = {a.role for a in session.scalars(select(Artifact).where(
        Artifact.event_id == event.id, Artifact.provenance == "cloud", Artifact.available.is_(True)))}
    created: list = []
    with tempfile.TemporaryDirectory(prefix="hgmedia_", dir=workdir) as tmp:
        work = Path(tmp)
        src = work / "src.mp4"
        s3.download_file(clip.s3_key, src)
        info = probe(src, ffprobe)
        if "thumbnail" not in have:
            out = work / "thumb.jpg"
            _make_thumb(src, out, info["duration"], ffmpeg)
            created.append(_upsert(session, s3, event, "thumbnail", f"admin_cache/thumbs/{event.id}.jpg",
                                   out, "image/jpeg"))
        if "filmstrip" not in have:
            out = work / "strip.jpg"
            detail = _make_filmstrip(src, out, work, info["duration"], ffmpeg, ffprobe)
            created.append(_upsert(session, s3, event, "filmstrip", f"admin_cache/filmstrips/{event.id}.jpg",
                                   out, "image/jpeg", detail))
        if "rendition" not in have and needs_rendition(info, src):
            out = work / "rendition.mp4"
            _make_rendition(src, out, ffmpeg)
            created.append(_upsert(session, s3, event, "rendition", f"admin_cache/renditions/{event.id}.mp4",
                                   out, "video/mp4"))
    return created


def _problem_key(event: Event) -> str:
    return f"admin_cache/thumbs/{event.id}.jpg"


def _record_problem(session: Session, event_id: int, clip_etag: Optional[str], error: str) -> None:
    session.rollback()
    key = f"admin_cache/thumbs/{event_id}.jpg"
    row = session.get(IndexProblem, key)
    if row is None:
        row = IndexProblem(s3_key=key)
        session.add(row)
    row.reason = f"media: {error}"[:1000]
    row.seen_at = datetime.now(timezone.utc)
    row.etag = clip_etag
    session.commit()


def process_pending(session: Session, s3: S3, limit: int = 50, ffmpeg: str = "ffmpeg",
                    ffprobe: str = "ffprobe", workdir: Optional[Path] = None) -> int:
    """Generate media for up to `limit` events; returns how many events gained artifacts. Never raises."""
    done = 0
    try:
        def has_cloud(role):
            return select(Artifact.id).where(Artifact.event_id == Event.id, Artifact.role == role,
                                             Artifact.provenance == "cloud", Artifact.available.is_(True)).exists()

        has_clip = select(Artifact.id).where(Artifact.event_id == Event.id, Artifact.role == "original_video",
                                             Artifact.available.is_(True)).exists()
        events = session.scalars(select(Event).where(has_clip, ~(has_cloud("thumbnail") & has_cloud("filmstrip")))
                                 .order_by(Event.start_ts.desc(), Event.id.desc())).all()
        event_ids = [e.id for e in events]
    except Exception:  # noqa: BLE001
        log.exception("media: could not list pending events")
        return 0
    for event_id in event_ids:
        if done >= limit:
            break
        clip_etag = None
        try:
            event = session.get(Event, event_id)
            clip = _clip_for(session, event) if event is not None else None
            if clip is None:
                continue
            clip_etag = clip.etag
            problem = session.get(IndexProblem, _problem_key(event))
            if problem is not None and problem.reason.startswith("media:") and problem.etag == clip_etag:
                continue
            made = ensure_media(session, s3, event, ffmpeg, ffprobe, workdir)
            stale = session.get(IndexProblem, _problem_key(event))
            if stale is not None and stale.reason.startswith("media:"):
                session.delete(stale)
            session.commit()
            done += 1 if made else 0
        except Exception as e:  # noqa: BLE001
            log.warning("media: event %s failed: %s", event_id, e)
            try:
                _record_problem(session, event_id, clip_etag, str(e) or type(e).__name__)
            except Exception:  # noqa: BLE001
                log.exception("media: could not record problem")
                session.rollback()
    return done
