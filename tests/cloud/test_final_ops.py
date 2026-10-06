"""Final fix round, recovery and operations (fix-final-decisions.md, fix-final.md R1-R4): media failure classes and
retries (I5), CLI commands under the loops' locks (R1), concurrent indexers (R2), logging (R3), bounded audit
de-duplication, id/length bounds (M1), resource caps (M2) and partial customer PATCH."""
import logging
import re
import shutil
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event as sa_event, func, select, text
from sqlalchemy.orm import sessionmaker

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, _event_id, _index, s3client  # noqa: F401  (fixture)

T0 = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg not installed")


@pytest.fixture()
def sm(db_engine):
    return sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture()
def s3(s3client):
    from home_guard_project.cloud.s3 import S3

    return S3(s3client, b.BUCKET)


def _one_clip(sm, s3client, s3):
    """The fixture bucket with only the production clip of the main event, indexed."""
    keep = {b.PROD_CLIP}
    clips = [k for k in b.seed_objects() if "/clips/" in k and k not in keep]
    b.seed_bucket(s3client, exclude=clips)
    with sm() as s:
        b.index_fixture_bucket(s, s3, now=NOW)
        return s.scalar(select(m.Event.id).where(m.Event.stem == b.STEM))


def _problem(s, event_id):
    return s.get(m.IndexProblem, f"admin_cache/thumbs/{event_id}.jpg")


# ---------------------------------------------------------------- I5: transient failures retry with backoff

def test_transient_media_failure_is_retried_with_backoff(sm, s3client, s3, monkeypatch):
    from botocore.exceptions import ReadTimeoutError

    from home_guard_project.cloud import media

    eid = _one_clip(sm, s3client, s3)
    with sm() as s:
        assert media._pending_ids(s, 50, now=T0) == [eid]

    def timeout(key, path):
        raise ReadTimeoutError(endpoint_url="https://s3")

    monkeypatch.setattr(s3, "download_file", timeout)
    with sm() as s:
        media.process_pending(s, s3, now=T0)
    with sm() as s:
        row = _problem(s, eid)
        assert row is not None and row.attempts == 1 and row.next_retry_at == T0 + timedelta(minutes=2)
        assert media._pending_ids(s, 50, now=T0 + timedelta(minutes=1)) == []
        assert media._pending_ids(s, 50, now=T0 + timedelta(minutes=3)) == [eid]
    with sm() as s:
        media.process_pending(s, s3, now=T0 + timedelta(minutes=3))
    with sm() as s:
        row = _problem(s, eid)
        assert row.attempts == 2 and row.next_retry_at == T0 + timedelta(minutes=3 + 4)


def test_transient_failures_stop_after_eight_attempts(sm, s3client, s3, monkeypatch):
    from home_guard_project.cloud import media

    eid = _one_clip(sm, s3client, s3)

    def missing_ffmpeg(*a, **k):
        raise media.MediaError("cannot run ffprobe: not found", transient=True)

    monkeypatch.setattr(media, "probe", missing_ffmpeg)
    now = T0
    for attempt in range(1, 9):
        with sm() as s:
            assert media._pending_ids(s, 50, now=now) == [eid], attempt
            media.process_pending(s, s3, now=now)
        with sm() as s:
            row = _problem(s, eid)
            assert row.attempts == attempt
            if attempt < 8:
                assert row.next_retry_at == now + timedelta(minutes=2 ** attempt)
                now = row.next_retry_at
            else:
                assert row.next_retry_at is None  # given up: only an operator retry runs it again
    with sm() as s:
        assert media._pending_ids(s, 50, now=now + timedelta(days=30)) == []


@needs_ffmpeg
def test_corrupt_clip_is_permanent_until_an_operator_retry(sm, s3client, s3, db_engine, monkeypatch, capsys):
    from home_guard_project.cloud import manage, media

    eid = _one_clip(sm, s3client, s3)  # FAKE_MP4 is not a video: a decode error on a downloaded file
    with sm() as s:
        media.process_pending(s, s3, now=T0)
    with sm() as s:
        row = _problem(s, eid)
        assert row is not None and row.next_retry_at is None and row.attempts == 1
        assert media._pending_ids(s, 50, now=T0 + timedelta(days=30)) == []
    monkeypatch.setenv("HG_CLOUD_DB_URL", db_engine.url.render_as_string(hide_password=False))
    assert manage.main(["media-retry", "--event", str(eid + 1000)]) == 0
    with sm() as s:
        assert _problem(s, eid) is not None
    assert manage.main(["media-retry", "--event", str(eid)]) == 0
    with sm() as s:
        assert _problem(s, eid) is None
        assert media._pending_ids(s, 50, now=T0) == [eid]
    assert "2 media problem" in capsys.readouterr().out  # the thumbnail's and the filmstrip's (one per stage)


# ---------------------------------------------------------------- M2: resource caps

def test_huge_clip_is_permanently_too_large_without_download(sm, s3client, s3, monkeypatch):
    from home_guard_project.cloud import media

    eid = _one_clip(sm, s3client, s3)
    with sm.begin() as s:
        s.scalars(select(m.Artifact).where(m.Artifact.s3_key == b.PROD_CLIP)).one().bytes = 600 * 1024 ** 2
    monkeypatch.setattr(s3, "download_file", lambda *a: pytest.fail("downloaded a clip over the size cap"))
    with sm() as s:
        media.process_pending(s, s3, now=T0)
    with sm() as s:
        row = _problem(s, eid)
        assert "too large" in row.reason and row.next_retry_at is None


def test_long_clip_is_permanently_too_large(sm, s3client, s3, monkeypatch):
    from home_guard_project.cloud import media

    eid = _one_clip(sm, s3client, s3)
    monkeypatch.setattr(media, "probe", lambda path, ffprobe: {"codec": "h264", "pix_fmt": "yuv420p",
                                                               "duration": 16 * 60.0, "width": 640, "height": 360})
    monkeypatch.setattr(media, "_make_thumb", lambda *a: pytest.fail("rendered a clip over the length cap"))
    with sm() as s:
        media.process_pending(s, s3, now=T0)
    with sm() as s:
        row = _problem(s, eid)
        assert "too large" in row.reason and row.next_retry_at is None


def test_every_ffmpeg_run_is_limited_to_two_threads(tmp_path, monkeypatch):
    from home_guard_project.cloud import media

    seen = []

    def capture(cmd, timeout=media.TIMEOUT, cwd=None):
        seen.append(cmd)
        raise media.MediaError("stop")

    monkeypatch.setattr(media, "_run", capture)
    for make in (lambda: media._make_thumb(tmp_path, 3.0, "ffmpeg"),
                 lambda: media._make_filmstrip(tmp_path, 3.0, "ffmpeg", "ffprobe"),
                 lambda: media._make_rendition(tmp_path, "ffmpeg")):
        with pytest.raises(media.MediaError):
            make()
    assert len(seen) == 3
    for cmd in seen:
        assert cmd[0] == "ffmpeg" and "-threads" in cmd and cmd[cmd.index("-threads") + 1] == "2", cmd


def test_export_selection_is_capped(client, staff_factory, s3client, monkeypatch):
    from home_guard_project.cloud import studio

    _index(client, s3client, consent_training=True)
    _, _, _, adm = staff_factory("admin")
    ids = [_event_id(client, s) for s in (b.STEM, b.PAUSED_STEM, b.COLLECT_STEM)]
    r = client.post("/v1/studio/collections", headers=adm, json={"name": "many"})
    cid = r.json()["id"]
    client.post(f"/v1/studio/collections/{cid}/items", headers=adm, json={"event_ids": ids})
    monkeypatch.setattr(studio, "MAX_EXPORT_EVENTS", 2)
    req = {"collection_id": cid, "name": "many", "formats": ["clips"]}
    for url in ("/v1/studio/exports/preview", "/v1/studio/exports"):
        r = client.post(url, headers=adm, json=req)
        assert r.status_code == 400 and "at most 2 events" in r.json()["detail"], r.text
    assert studio.MAX_EXPORT_EVENTS == 2 and studio.select_export_items  # the cap is the module constant
    monkeypatch.setattr(studio, "MAX_EXPORT_EVENTS", 20_000)
    assert client.post("/v1/studio/exports/preview", headers=adm, json=req).status_code == 200


def test_default_export_cap_is_twenty_thousand():
    from home_guard_project.cloud import studio

    assert studio.MAX_EXPORT_EVENTS == 20_000


# ---------------------------------------------------------------- R1: CLI commands share the loops' locks

@pytest.mark.parametrize("command,lock,target", [("index-once", "INDEXER_LOCK", "indexer.index_all"),
                                                 ("media-once", "MEDIA_LOCK", "media.process_pending")])
def test_cli_waits_for_the_server_loop(db_engine, monkeypatch, capsys, command, lock, target):
    from home_guard_project.cloud import indexer, loops, manage, media

    calls = []
    module, name = target.split(".")
    monkeypatch.setattr({"indexer": indexer, "media": media}[module], name,
                        lambda *a, **k: calls.append(1) or ({} if module == "indexer" else 0))
    monkeypatch.setattr(manage, "_s3", lambda: object())
    monkeypatch.setenv("HG_CLOUD_DB_URL", db_engine.url.render_as_string(hide_password=False))
    holder = db_engine.connect()
    try:
        holder.execute(text("SELECT pg_advisory_lock(:k)"), {"k": getattr(loops, lock)})
        holder.commit()
        assert manage.main([command]) == 0
        assert calls == []
        assert "already" in capsys.readouterr().out
        holder.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": getattr(loops, lock)})
        holder.commit()
    finally:
        holder.close()
    assert manage.main([command]) == 0
    assert calls == [1]


# ---------------------------------------------------------------- R2: two indexers at once

class _SyncedS3:
    """The real (moto) S3, but each indexer waits before listing until the other has loaded its state too, so both
    insert the same new keys at the same time."""

    def __init__(self, s3, barrier):
        self.s3, self.barrier = s3, barrier

    def list(self, prefix):
        if prefix.startswith("production_"):  # the first root listed: both passes have loaded, none has written
            self.barrier.wait()
        yield from self.s3.list(prefix)

    def __getattr__(self, name):
        return getattr(self.s3, name)


def test_two_indexers_on_one_database_both_succeed(sm, s3client, s3):
    from home_guard_project.cloud.indexer import index_device

    b.seed_bucket(s3client)
    with sm() as s:
        device_pk = b.enroll(s).id
        s.commit()
    barrier = threading.Barrier(2, timeout=60)
    errors, stats = [], []

    def run():
        try:
            with sm() as s:
                stats.append(index_device(s, _SyncedS3(s3, barrier), s.get(m.Device, device_pk), full_scan=True,
                                          now=NOW))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
            barrier.abort()

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert errors == []
    with sm() as s:
        keys = s.scalars(select(m.Artifact.s3_key)).all()
        assert len(keys) == len(set(keys))
        events = s.execute(select(m.Event.camera, m.Event.stem)).all()
        assert len(events) == len(set(events)) == 6
        runs = s.execute(select(m.AiRun.event_id, func.count()).group_by(m.AiRun.event_id)).all()
        assert all(n == 1 for _, n in runs)
        fb = s.scalars(select(m.Feedback.s3_key)).all()
        assert len(fb) == len(set(fb))
    assert sum(st.new_events for st in stats) == 6


# ---------------------------------------------------------------- R3: logging

def test_server_logging_is_configured_with_timestamps(db_engine, monkeypatch):
    from home_guard_project.cloud import app as app_module

    monkeypatch.setenv("HG_CLOUD_JWT_SECRET", "x" * 40)
    monkeypatch.setenv("HG_CLOUD_DB_URL", db_engine.url.render_as_string(hide_password=False))
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    try:
        api = app_module.create_app_from_env()
        api.state.engine.dispose()
        added = [h for h in root.handlers if h not in before]
        assert len(added) == 1 and root.level == logging.INFO
        record = logging.LogRecord("home_guard_project.cloud.loops", logging.ERROR, __file__, 1,
                                   "loop %s iteration failed", ("indexer",), None)
        line = added[0].format(record)
        assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} ERROR home_guard_project\.cloud\.loops "
                            r"loop indexer iteration failed", line), line
        app_module.create_app_from_env().state.engine.dispose()  # a second app adds no second handler
        assert [h for h in root.handlers if h not in before] == added
    finally:
        for h in [h for h in root.handlers if h not in before]:
            root.removeHandler(h)
        root.setLevel(level)


# ---------------------------------------------------------------- audit de-duplication is bounded

def test_view_audit_lookups_are_bounded_by_the_window(client, staff_factory, s3client, db_engine):
    _index(client, s3client)
    _, _, _, adm = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    with session_scope(client.app.state.engine) as s:
        c = s.scalars(select(m.Customer)).one()
        c.consent_recordings = True
        s.add(m.Artifact(event_id=eid, role="thumbnail", s3_key="admin_cache/thumbs/x.jpg", provenance="cloud"))
    statements = []

    def capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    engine = client.app.state.engine
    sa_event.listen(engine, "before_cursor_execute", capture)
    try:
        assert client.get(f"/v1/events/{eid}", headers=adm).status_code == 200
        assert client.get(f"/v1/events/{eid}/thumbnail", headers=adm, follow_redirects=False).status_code == 307
    finally:
        sa_event.remove(engine, "before_cursor_execute", capture)
    lookups = [st for st in statements if st.lstrip().upper().startswith("SELECT") and "FROM audit_log" in st]
    assert len(lookups) == 2, lookups
    for st in lookups:
        assert "audit_log.ts >" in st, st
    locks = [st for st in statements if "pg_advisory_xact_lock" in st]
    assert len(locks) >= 2  # event view and thumbnail view both serialise their check-then-insert
    with db_engine.connect() as conn:
        cols = conn.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname = "
                                 "'ix_audit_log_staff_action_target_ts'")).scalar()
    assert cols is not None and "(staff_id, action, target, ts)" in cols


# ---------------------------------------------------------------- M1: ids and lengths at the boundary

BIG = (2 ** 31, 10 ** 30)


def test_out_of_range_ids_are_404_not_500(client, staff_factory, s3client):
    _index(client, s3client, consent_training=True)
    _, _, _, adm = staff_factory("admin")
    for big in BIG:
        for method, url, body in (
                ("GET", f"/v1/events/{big}", None), ("GET", f"/v1/events/{big}/thumbnail", None),
                ("GET", f"/v1/events/{big}/detections", None),
                ("PATCH", f"/v1/events/{big}/review", {"reviewed": True}),
                ("POST", f"/v1/artifacts/{big}/access", {"purpose": "review"}),
                ("GET", f"/v1/customers/{big}", None), ("PATCH", f"/v1/customers/{big}", {"name": "x"}),
                ("GET", f"/v1/studio/collections/{big}/items", None),
                ("POST", f"/v1/studio/collections/{big}/items", {"event_ids": [1]}),
                ("DELETE", f"/v1/studio/collections/{big}/items", {"event_ids": [1]}),
                ("GET", f"/v1/studio/exports/{big}", None),
                ("POST", "/v1/devices/enroll", {"customer_id": big, "site": "zz"}),
                ("POST", "/v1/studio/exports/preview", {"collection_id": big, "name": "x", "formats": ["clips"]}),
                ("POST", "/v1/studio/exports", {"collection_id": big, "name": "x", "formats": ["clips"]})):
            r = client.request(method, url, headers=adm, json=body, follow_redirects=False)
            assert r.status_code == 404, (method, url, r.status_code, r.text)
        for url, params in (("/v1/events", {"customer_id": big}), ("/v1/events", {"collection_id": big}),
                            ("/v1/events/density", {"customer_id": big, "from_utc": "2026-10-02T00:00:00Z",
                                                    "to_utc": "2026-10-03T00:00:00Z"}),
                            ("/v1/audit", {"customer_id": big})):
            r = client.get(url, headers=adm, params=params)
            assert r.status_code == 200, (url, r.status_code, r.text)
            body = r.json()
            assert body.get("items", body.get("rows")) == [], (url, body)


def test_too_long_strings_are_422_not_500(client, staff_factory):
    _, _, _, adm = staff_factory("admin")
    cid = client.post("/v1/customers", headers=adm, json={"name": "Ok"}).json()["id"]
    cases = [("POST", "/v1/customers", {"name": "n" * 121}),
             ("POST", "/v1/customers", {"name": "n" * 256}),
             ("POST", "/v1/customers", {"name": "ok", "notes": "x" * 4001}),
             ("POST", "/v1/customers", {"name": "ok", "timezone": "z" * 201}),
             ("PATCH", f"/v1/customers/{cid}", {"name": "n" * 121}),
             ("POST", "/v1/devices/enroll", {"customer_id": cid, "site": "s" * 201}),
             ("POST", "/v1/devices/enroll", {"customer_id": cid, "site": "ok", "tailscale_host": "h" * 201}),
             ("POST", "/v1/studio/collections", {"name": "c" * 121}),
             ("POST", "/v1/studio/collections", {"name": "ok", "description": "d" * 4001})]
    for method, url, body in cases:
        r = client.request(method, url, headers=adm, json=body)
        assert r.status_code == 422, (method, url, r.status_code, r.text[:200])
    for params in ({"site": "s" * 201}, {"camera": "c" * 201}, {"q": "q" * 201}):
        assert client.get("/v1/events", headers=adm, params=params).status_code == 422, params
    assert client.post("/v1/customers", headers=adm, json={"name": "n" * 120, "notes": "x" * 4000}).status_code == 200


# ---------------------------------------------------------------- PATCH /customers is partial

def test_customer_patch_updates_only_the_fields_sent(client, staff_factory):
    _, _, _, adm = staff_factory("admin")
    full = {"name": "Acme", "timezone": "Europe/London", "consent_live": True, "consent_recordings": True,
            "consent_training": True, "notes": "gate code is private"}
    cid = client.post("/v1/customers", headers=adm, json=full).json()["id"]
    r = client.patch(f"/v1/customers/{cid}", headers=adm, json={"name": "Acme Ltd"})
    assert r.status_code == 200, r.text
    assert r.json() | {"devices": []} == {**full, "name": "Acme Ltd", "id": cid, "devices": [], "consent_proposed": None,
                                          "consent_source": "contract"}
    r = client.patch(f"/v1/customers/{cid}", headers=adm, json={"name": "Acme Ltd", "consent_training": False})
    assert r.json()["consent_training"] is False and r.json()["consent_live"] is True
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "customer_update")
                         .order_by(m.AuditLog.id)).all()
    assert [row.detail for row in rows] == [{"changed": ["name"]}, {"changed": ["consent_training"]}]
    # a no-op PATCH writes no audit row; a missing customer is a 404
    assert client.patch(f"/v1/customers/{cid}", headers=adm, json={"name": "Acme Ltd"}).status_code == 200
    with session_scope(client.app.state.engine) as s:
        assert s.scalar(select(func.count()).select_from(m.AuditLog)
                        .where(m.AuditLog.action == "customer_update")) == 2
    assert client.patch("/v1/customers/999999", headers=adm, json={"name": "x"}).status_code == 404
