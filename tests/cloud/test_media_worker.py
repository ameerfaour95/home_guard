"""Task 12: thumbnails, filmstrips and browser/Qt-safe renditions."""
import shutil
import subprocess

import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from . import builders as b
from .test_indexer import device, s3, s3client, session  # noqa: F401  (shared fixtures)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg not installed")


def _mp4v_clip(path, seconds=3, w=320, h=240):
    import cv2
    import numpy as np

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(seconds * 10):
        vw.write(np.full((h, w, 3), (i * 8) % 255, dtype=np.uint8))
    vw.release()
    return path.read_bytes()


def _h264_clip(path, faststart=True, seconds=2):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=10:duration={seconds}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    subprocess.run(cmd + [str(path)], check=True, timeout=120)
    return path.read_bytes()


def _setup(session, s3client, s3, body):
    b.seed_bucket(s3client, exclude=[b.TRAIN_CLIP])
    s3client.put_object(Bucket=b.BUCKET, Key=b.PROD_CLIP, Body=body)
    return b.index_fixture_bucket(session, s3)


def _media(session):
    return {a.role: a for a in session.scalars(select(m.Artifact).where(m.Artifact.provenance == "cloud"))}


def _good_events(session):
    return session.scalars(select(m.Event).where(m.Event.stem == b.STEM)).one()


def test_mp4v_clip_gets_thumb_filmstrip_and_rendition(session, s3, s3client, tmp_path):
    from home_guard_project.cloud.media import process_pending

    _setup(session, s3client, s3, _mp4v_clip(tmp_path / "c.mp4"))
    assert process_pending(session, s3) == 1
    arts = _media(session)
    ev = _good_events(session)
    assert set(arts) == {"thumbnail", "filmstrip", "rendition"}
    assert arts["thumbnail"].s3_key == f"admin_cache/thumbs/{ev.id}.jpg"
    assert arts["filmstrip"].s3_key == f"admin_cache/filmstrips/{ev.id}.jpg"
    assert arts["rendition"].s3_key == f"admin_cache/renditions/{ev.id}.mp4"
    for a in arts.values():
        assert a.camera == ev.camera and a.stem == ev.stem and a.event_id == ev.id and a.available
    d = arts["filmstrip"].detail
    assert d["tile_w"] == 160 and d["tile_h"] == 120 and d["fps"] == 1 and d["count"] == 3
    assert s3client.head_object(Bucket=b.BUCKET, Key=arts["thumbnail"].s3_key)["ContentType"] == "image/jpeg"
    assert s3client.head_object(Bucket=b.BUCKET, Key=arts["rendition"].s3_key)["ContentType"] == "video/mp4"
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(s3.get_bytes(arts["thumbnail"].s3_key), np.uint8), 1)
    assert img.shape[1] == 480
    strip = cv2.imdecode(np.frombuffer(s3.get_bytes(arts["filmstrip"].s3_key), np.uint8), 1)
    assert strip.shape[1] == 160 * 3 and strip.shape[0] == 120
    assert s3.get_bytes(b.PROD_CLIP)[:4] == _mp4v_clip(tmp_path / "c2.mp4")[:4]  # original untouched
    assert process_pending(session, s3) == 0  # second run creates nothing


def test_faststart_h264_gets_no_rendition(session, s3, s3client, tmp_path):
    from home_guard_project.cloud.media import process_pending

    _setup(session, s3client, s3, _h264_clip(tmp_path / "f.mp4", faststart=True))
    assert process_pending(session, s3) == 1
    assert set(_media(session)) == {"thumbnail", "filmstrip"}


def test_h264_with_trailing_moov_gets_rendition(session, s3, s3client, tmp_path):
    from home_guard_project.cloud.media import process_pending

    _setup(session, s3client, s3, _h264_clip(tmp_path / "n.mp4", faststart=False))
    process_pending(session, s3)
    assert "rendition" in _media(session)


def test_long_clip_filmstrip_capped_at_60_tiles(session, s3, s3client, tmp_path):
    from home_guard_project.cloud.media import process_pending

    _setup(session, s3client, s3, _h264_clip(tmp_path / "l.mp4", seconds=120))
    process_pending(session, s3)
    d = _media(session)["filmstrip"].detail
    assert 58 <= d["count"] <= 60
    assert abs(d["fps"] - 0.5) < 0.02


def test_corrupt_clip_records_problem_and_is_not_retried(session, s3, s3client):
    from home_guard_project.cloud.media import process_pending

    b.seed_bucket(s3client, exclude=[b.TRAIN_CLIP])  # FAKE_MP4 bodies are not videos
    b.index_fixture_bucket(session, s3)
    assert process_pending(session, s3) == 0
    assert not _media(session)
    q = select(m.IndexProblem).where(m.IndexProblem.reason.like("media:%"))
    probs = session.scalars(q).all()
    assert probs
    stamps = {p.s3_key: p.seen_at for p in probs}
    assert process_pending(session, s3) == 0
    session.expire_all()
    assert {p.s3_key: p.seen_at for p in session.scalars(q).all()} == stamps  # skipped, not re-run


# ---- fix round ----
def test_rendition_failure_keeps_thumbnail_and_filmstrip(session, s3, s3client, tmp_path, monkeypatch):
    from home_guard_project.cloud import media

    _setup(session, s3client, s3, _mp4v_clip(tmp_path / "c.mp4"))
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise media.MediaError("ffmpeg exit 1: encoder exploded")

    monkeypatch.setattr(media, "_make_rendition", boom)
    assert media.process_pending(session, s3) == 1
    session.expire_all()
    assert set(_media(session)) == {"thumbnail", "filmstrip"}
    ev = _good_events(session)
    prob = session.get(m.IndexProblem, f"admin_cache/renditions/{ev.id}.mp4")
    clip = session.scalars(select(m.Artifact).where(m.Artifact.event_id == ev.id,
                                                    m.Artifact.role == "original_video")).first()
    assert prob is not None and prob.reason.startswith("media:") and prob.etag == clip.etag
    assert media.process_pending(session, s3) == 0
    assert len(calls) == 1  # not retried for the same clip etag


def test_replaced_clip_regenerates_and_retires_stale_rendition(session, s3, s3client, tmp_path):
    from home_guard_project.cloud.media import process_pending

    _setup(session, s3client, s3, _mp4v_clip(tmp_path / "c.mp4"))
    process_pending(session, s3)
    old = _media(session)
    assert set(old) == {"thumbnail", "filmstrip", "rendition"}
    old_src = old["thumbnail"].detail["src_etag"]
    assert old["thumbnail"].detail["needs_rendition"] is True
    s3client.put_object(Bucket=b.BUCKET, Key=b.PROD_CLIP, Body=_h264_clip(tmp_path / "f.mp4", faststart=True))
    from home_guard_project.cloud.indexer import index_device

    dev = session.scalars(select(m.Device)).one()
    index_device(session, s3, dev, full_scan=True)
    assert process_pending(session, s3) == 1
    session.expire_all()
    arts = {a.role: a for a in session.scalars(select(m.Artifact).where(m.Artifact.provenance == "cloud"))}
    assert arts["thumbnail"].detail["src_etag"] != old_src
    assert arts["filmstrip"].detail["src_etag"] == arts["thumbnail"].detail["src_etag"]
    assert arts["thumbnail"].detail["needs_rendition"] is False
    assert arts["rendition"].available is False
    assert process_pending(session, s3) == 0


def test_limit_bounds_attempts_including_failures(session, s3, s3client, monkeypatch):
    from home_guard_project.cloud import media

    b.seed_bucket(s3client, exclude=[b.TRAIN_CLIP])
    b.index_fixture_bucket(session, s3)
    attempts = []
    real = media.ensure_media
    monkeypatch.setattr(media, "ensure_media", lambda *a, **k: (attempts.append(1), real(*a, **k))[1])
    media.process_pending(session, s3, limit=1)
    assert len(attempts) <= 1


def test_moov_first_survives_truncated_and_huge_boxes(tmp_path):
    import struct

    from home_guard_project.cloud.media import moov_first

    trunc = tmp_path / "t.mp4"
    trunc.write_bytes(struct.pack(">I4s", 24, b"ftyp") + b"isom")  # box claims 24 bytes, file ends
    assert moov_first(trunc) is False
    huge = tmp_path / "h.mp4"
    huge.write_bytes(struct.pack(">I4sQ", 1, b"free", 2 ** 63 + 5) + b"\0" * 8)
    assert moov_first(huge) is False
    loop = tmp_path / "l.mp4"
    loop.write_bytes(struct.pack(">I4s", 8, b"free") * 20000 + struct.pack(">I4s", 8, b"moov"))
    assert moov_first(loop) is False  # box loop capped
    assert moov_first(tmp_path / "missing.mp4") is False


def test_upload_file_rejects_keys_outside_writable_prefixes(s3, tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        s3.upload_file(f, "dataset_house/clips/x.jpg", "image/jpeg")
