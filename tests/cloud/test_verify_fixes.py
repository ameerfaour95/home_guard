"""Verification fix round (final-verify-codex.md): legacy identity history (C1), per-stage media retries (R5),
private collections on every collection filter (N1), per-key S3 delete results (N2), export name bounds (D12)."""
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .conftest import _db_url
from .test_event_routes import NOW, _event_id, _index, s3client  # noqa: F401  (fixture)

T0 = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
HIDDEN = ("zircona", "quillmere", "morvane", "brindlewood", "tessaly", "oakvale")


def _labeler_text(client, h, event_id):
    detail = client.get(f"/v1/events/{event_id}", headers=h)
    listed = client.get("/v1/events", headers=h)
    assert detail.status_code == listed.status_code == 200, (detail.text, listed.text)
    return (detail.text + listed.text).lower()


def _staff(engine, settings, role):
    import pyotp

    from home_guard_project.cloud import auth

    with session_scope(engine) as s:
        staff = m.Staff(email=f"{role}-{uuid.uuid4().hex[:6]}@example.com", name=role.title(), role=role,
                        password_hash=auth.hash_password("pw-" + uuid.uuid4().hex), totp_secret=pyotp.random_base32())
        s.add(staff)
        s.flush()
        return {"Authorization": f"Bearer {auth.make_access_token(staff, settings)}"}


# ---------------------------------------------------------------- C1: names that exist only in pre-0007 history

@pytest.fixture()
def fresh_url(pg_url, pg_template):
    name = "mig_" + uuid.uuid4().hex[:10]
    with pg_template.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    yield _db_url(pg_url, name)
    with pg_template.connect() as c:
        c.execute(text(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)"))


SUMMARY = ("A courier at Zircona's gate by the Quillmere Porch, seen by morvane-laptop and Morvane; "
           "Brindlewood Gate light on, Tessaly cam, oakvale entrance")


def _populate_0006(conn):
    stem = "oak_ch3_1791020177_alert"
    conn.execute(text("INSERT INTO customers (id, name, consent_training) VALUES (1, 'Felvoria', true)"))
    conn.execute(text("INSERT INTO devices (id, device_id, site, tailscale_host, customer_id) "
                      "VALUES (1, :d, 'oak', 'oak-box', 1)"), {"d": str(uuid.uuid4())})
    conn.execute(text("INSERT INTO events (id, device_pk, site, camera, stem, start_ts, summary, kind) "
                      "VALUES (1, 1, 'oak', 'oak_ch3', :stem, 1791020177, :summary, 'alert')"),
                 {"stem": stem, "summary": SUMMARY})
    # the customer's old name exists only in the audit history of a rename made before 0007
    conn.execute(text("INSERT INTO audit_log (ts, staff_id, action, customer_id, target, detail) "
                      "VALUES (now(), 1, 'customer_update', 1, 'Felvoria', CAST(:d AS jsonb))"),
                 {"d": json.dumps({"changed": {"name": ["Zircona", "Felvoria"], "notes": ["<redacted>"] * 2}})})
    revisions = {
        f"production_oak/meta/oak_ch3/2026-10-03/{stem}.meta.json":
            {"camera_name": "oak_ch3", "prompt_camera_name": "Quillmere Porch", "site": "oakvale"},
        "production_oak/_status/heartbeat.json":
            {"host": "morvane-laptop", "site": "oak", "cameras": {"tessaly_cam": {}}},
        f"production_oak/feedback/oak_ch3/2026-10-03/{stem}_1791020300000.feedback.json":
            {"host": "morvane-laptop", "alert": {"camera": "oak_ch3"}, "verdict": "true_alert"},
    }
    for i, (key, body) in enumerate(revisions.items()):
        conn.execute(text("INSERT INTO raw_revisions (s3_key, etag, fetched_at, body) "
                          "VALUES (:k, :e, now(), CAST(:b AS jsonb))"), {"k": key, "e": f"e{i}", "b": json.dumps(body)})
    conn.execute(text("CREATE TABLE camera_aliases (id serial PRIMARY KEY, device_pk integer, alias text)"))
    conn.execute(text("INSERT INTO camera_aliases (device_pk, alias) VALUES (1, 'Brindlewood Gate')"))


def test_legacy_history_is_backfilled_and_never_reaches_labelers(fresh_url):
    from alembic import command
    from fastapi.testclient import TestClient

    from home_guard_project.cloud.app import create_app
    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.settings import Settings

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0006")
    engine = create_engine(fresh_url.replace("postgresql://", "postgresql+psycopg://", 1))
    try:
        with engine.begin() as conn:
            _populate_0006(conn)
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            aliases = set(conn.execute(text("SELECT kind, value FROM identity_aliases WHERE device_pk = 1")).all())
        assert {("customer", "Zircona"), ("customer", "Felvoria"), ("display_name", "Quillmere Porch"),
                ("host", "morvane-laptop"), ("site", "oakvale"), ("camera", "tessaly_cam"),
                ("camera", "Brindlewood Gate")} <= aliases
    finally:
        engine.dispose()
    app = create_app(Settings.for_tests(fresh_url), s3=None, init_db=False)
    try:
        with TestClient(app) as client:
            lab = _staff(app.state.engine, app.state.settings, "labeler")
            text_seen = _labeler_text(client, lab, 1)
            assert "customer-" in text_seen
            for word in HIDDEN:
                assert word not in text_seen, word
    finally:
        app.state.engine.dispose()


def test_identity_reads_legacy_history_lazily_when_aliases_are_missing(client, staff_factory, s3client):
    device_pk, customer_id = _index(client, s3client, consent_training=True, customer_name="Felvoria")
    _, _, _, lab = staff_factory("labeler")
    eid = _event_id(client, b.STEM)
    with session_scope(client.app.state.engine) as s:
        s.execute(text("DELETE FROM identity_aliases"))  # as if no backfill had ever run
        s.add(m.AuditLog(ts=NOW, staff_id=1, action="customer_update", customer_id=customer_id, target="Felvoria",
                         detail={"changed": {"name": ["Zircona", "Felvoria"]}}))
        s.add(m.RawRevision(s3_key=b.PROD_META, etag="old-revision", fetched_at=NOW,
                            body={"camera_name": "front_side", "prompt_camera_name": "Quillmere Porch",
                                  "host": "morvane-laptop"}))
        ev = s.get(m.Event, eid)
        ev.summary, ev.summary_redacted = SUMMARY, None
    seen = _labeler_text(client, lab, eid)
    for word in ("zircona", "quillmere", "morvane"):
        assert word not in seen, word
    with session_scope(client.app.state.engine) as s:
        kinds = {(a.kind, a.value) for a in s.scalars(select(m.IdentityAlias))}
    assert {("customer", "Zircona"), ("display_name", "Quillmere Porch"), ("host", "morvane-laptop")} <= kinds


# ---------------------------------------------------------------- R5: every media stage has its own retry state

@pytest.fixture()
def sm(db_engine):
    return sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture()
def s3(s3client):
    from home_guard_project.cloud.s3 import S3

    return S3(s3client, b.BUCKET)


def _one_clip(sm, s3client, s3):
    clips = [k for k in b.seed_objects() if "/clips/" in k and k != b.PROD_CLIP]
    b.seed_bucket(s3client, exclude=clips)
    with sm() as s:
        b.index_fixture_bucket(s, s3, now=NOW)
        return s.scalar(select(m.Event.id).where(m.Event.stem == b.STEM))


def _stage_problems(s, eid):
    keys = {"thumbnail": f"admin_cache/thumbs/{eid}.jpg", "filmstrip": f"admin_cache/filmstrips/{eid}.jpg",
            "rendition": f"admin_cache/renditions/{eid}.mp4"}
    return {stage: s.get(m.IndexProblem, key) for stage, key in keys.items()}


def test_partial_media_respects_each_stage_backoff_and_cap(sm, s3client, s3, monkeypatch):
    """The nine-pass reproduction: a fresh thumbnail that needs a rendition, no filmstrip and no rendition, and
    both remaining stages fail transiently. Nine passes at the same time try each stage once; later passes follow
    each stage's backoff and stop at eight attempts."""
    from home_guard_project.cloud import media

    eid = _one_clip(sm, s3client, s3)
    with sm.begin() as s:
        clip = s.scalars(select(m.Artifact).where(m.Artifact.s3_key == b.PROD_CLIP)).one()
        s.add(m.Artifact(event_id=eid, role="thumbnail", s3_key=f"admin_cache/thumbs/{eid}.jpg", provenance="cloud",
                         detail={"src_etag": clip.etag, "needs_rendition": True}))
    runs = {"filmstrip": 0, "rendition": 0, "thumbnail": 0}

    def fail(stage):
        def run(*a, **k):
            runs[stage] += 1
            raise media.MediaError(f"{stage} timed out", transient=True)
        return run

    monkeypatch.setattr(s3, "download_file", lambda key, path: path.write_bytes(b.FAKE_MP4))
    monkeypatch.setattr(media, "probe", lambda path, ffprobe: {"codec": "mpeg4", "pix_fmt": "yuv420p",
                                                               "duration": 3.0, "width": 320, "height": 240})
    monkeypatch.setattr(media, "_make_thumb", fail("thumbnail"))
    monkeypatch.setattr(media, "_make_filmstrip", fail("filmstrip"))
    monkeypatch.setattr(media, "_make_rendition", fail("rendition"))
    for _ in range(9):
        with sm() as s:
            media.process_pending(s, s3, now=T0)
    with sm() as s:
        probs = _stage_problems(s, eid)
        assert probs["filmstrip"] is not None and probs["filmstrip"].attempts == 1
        assert probs["rendition"] is not None and probs["rendition"].attempts == 1
        assert probs["filmstrip"].next_retry_at == T0 + timedelta(minutes=2)
        assert media._pending_ids(s, 50, now=T0) == []
    assert runs == {"filmstrip": 1, "rendition": 1, "thumbnail": 0}
    now = T0
    for attempt in range(2, 12):
        now += timedelta(days=1)
        with sm() as s:
            media.process_pending(s, s3, now=now)
            media.process_pending(s, s3, now=now)  # a second pass at the same time does nothing
    with sm() as s:
        probs = _stage_problems(s, eid)
        for stage in ("filmstrip", "rendition"):
            assert probs[stage].attempts == media.MAX_ATTEMPTS and probs[stage].next_retry_at is None, stage
        assert media._pending_ids(s, 50, now=now + timedelta(days=30)) == []
    assert runs == {"filmstrip": media.MAX_ATTEMPTS, "rendition": media.MAX_ATTEMPTS, "thumbnail": 0}


def test_a_failed_stage_does_not_block_the_others(sm, s3client, s3, monkeypatch):
    from home_guard_project.cloud import media

    eid = _one_clip(sm, s3client, s3)
    monkeypatch.setattr(s3, "download_file", lambda key, path: path.write_bytes(b.FAKE_MP4))
    monkeypatch.setattr(media, "probe", lambda path, ffprobe: {"codec": "h264", "pix_fmt": "yuv420p",
                                                               "duration": 3.0, "width": 320, "height": 240})
    monkeypatch.setattr(media, "needs_rendition", lambda info, path: False)

    def thumb_fails(*a, **k):
        raise media.MediaError("no thumbnail frame")

    def strip(work, duration, ffmpeg, ffprobe):
        out = work / "strip.jpg"
        out.write_bytes(b.FAKE_JPG)
        return out, {"fps": 1, "tile_w": 160, "tile_h": 120, "count": 3}

    monkeypatch.setattr(media, "_make_thumb", thumb_fails)
    monkeypatch.setattr(media, "_make_filmstrip", strip)
    with sm() as s:
        media.process_pending(s, s3, now=T0)
    with sm() as s:
        probs = _stage_problems(s, eid)
        roles = set(s.scalars(select(m.Artifact.role).where(m.Artifact.event_id == eid,
                                                             m.Artifact.provenance == "cloud")))
        assert roles == {"filmstrip"}
        assert probs["thumbnail"] is not None and probs["thumbnail"].next_retry_at is None  # permanent
        assert probs["filmstrip"] is None
        assert media._pending_ids(s, 50, now=T0 + timedelta(days=30)) == []


# ---------------------------------------------------------------- N1: private collections everywhere

def test_private_collection_cannot_be_used_as_an_events_filter(client, staff_factory, s3client):
    _index(client, s3client, consent_training=True)
    _, _, _, lab_a = staff_factory("labeler")
    _, _, _, lab_b = staff_factory("labeler")
    eid = _event_id(client, b.STEM)
    mine = client.post("/v1/studio/collections", headers=lab_a, json={"name": "mine"}).json()["id"]
    assert client.post(f"/v1/studio/collections/{mine}/items", headers=lab_a,
                       json={"event_ids": [eid]}).status_code == 200
    own = client.get("/v1/events", headers=lab_a, params={"collection_id": mine, "with_total": True}).json()
    assert [it["id"] for it in own["items"]] == [eid]
    hidden = client.get("/v1/events", headers=lab_b, params={"collection_id": mine, "with_total": True})
    absent = client.get("/v1/events", headers=lab_b, params={"collection_id": 999999, "with_total": True})
    assert hidden.status_code == absent.status_code == 200
    assert hidden.json() == absent.json() and hidden.json()["items"] == [] and hidden.json()["total"] == 0


def test_role_change_never_exposes_a_private_collection(client, staff_factory, s3client):
    _index(client, s3client, consent_training=True)
    lab_a_row, _, _, lab_a = staff_factory("labeler")
    _, _, _, lab_b = staff_factory("labeler")
    _, _, _, adm = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    mine = client.post("/v1/studio/collections", headers=lab_a, json={"name": "Zircona household"}).json()["id"]
    client.post(f"/v1/studio/collections/{mine}/items", headers=lab_a, json={"event_ids": [eid]})
    with session_scope(client.app.state.engine) as s:
        assert s.get(m.Collection, mine).private_to_staff_id == lab_a_row.id
        s.get(m.Staff, lab_a_row.id).role = "admin"  # the creator is promoted (or the row is edited)
    listed = client.get("/v1/studio/collections", headers=lab_b).json()
    assert mine not in {c["id"] for c in listed} and "Zircona household" not in json.dumps(listed)
    for method, url, body in (("GET", f"/v1/studio/collections/{mine}/items", None),
                              ("POST", "/v1/studio/exports/preview",
                               {"collection_id": mine, "name": "x", "formats": ["clips"]}),
                              ("POST", "/v1/studio/exports",
                               {"collection_id": mine, "name": "x", "formats": ["clips"]})):
        r = client.request(method, url, headers=lab_b, json=body)
        assert r.status_code == 404 and r.json()["detail"] == "Collection not found", (method, url, r.text)
    assert client.get("/v1/events", headers=lab_b, params={"collection_id": mine}).json()["items"] == []
    # admins still see it; an admin-made collection stays visible to labelers whatever happens to its creator
    assert mine in {c["id"] for c in client.get("/v1/studio/collections", headers=adm).json()}
    shared = client.post("/v1/studio/collections", headers=adm, json={"name": "shared"}).json()["id"]
    with session_scope(client.app.state.engine) as s:
        assert s.get(m.Collection, shared).private_to_staff_id is None
    assert shared in {c["id"] for c in client.get("/v1/studio/collections", headers=lab_b).json()}


# ---------------------------------------------------------------- N2: per-key DeleteObjects results

def _orphaned_media(client, s3client):
    from .test_final_evidence import _reindex

    _index(client, s3client, consent_training=True)
    eid = _event_id(client, b.STEM)
    keys = [f"admin_cache/thumbs/{eid}.jpg", f"admin_cache/filmstrips/{eid}.jpg", f"admin_cache/renditions/{eid}.mp4"]
    with session_scope(client.app.state.engine) as s:
        for role, key in zip(("thumbnail", "filmstrip", "rendition"), keys):
            b.put(s3client, key, b.FAKE_JPG)
            s.add(m.Artifact(event_id=eid, role=role, s3_key=key, provenance="cloud", etag="x",
                             detail={"src_etag": "x"}))
    for key in (b.PROD_CLIP, b.TRAIN_CLIP):
        s3client.delete_object(Bucket=b.BUCKET, Key=key)
    _reindex(client, NOW)
    return keys


def _exists(s3client, key):
    return s3client.list_objects_v2(Bucket=b.BUCKET, Prefix=key).get("KeyCount", 0) > 0


def test_failed_per_key_deletes_stay_pending_and_are_retried(client, s3client, monkeypatch):
    from home_guard_project.cloud import media

    keys = _orphaned_media(client, s3client)
    s3 = client.app.state.s3
    denied = keys[1]
    real = s3.client.delete_objects

    def partial(Bucket, Delete):
        objs = [o for o in Delete["Objects"] if o["Key"] != denied]
        resp = real(Bucket=Bucket, Delete={**Delete, "Objects": objs}) if objs else {}
        return {**resp, "Errors": [{"Key": denied, "Code": "AccessDenied", "Message": "Access Denied"}]}

    monkeypatch.setattr(s3.client, "delete_objects", partial)
    with session_scope(client.app.state.engine) as s:
        assert media.retire_orphans(s, s3, now=T0) == 2
    with session_scope(client.app.state.engine) as s:
        avail = {a.s3_key: a.available for a in s.scalars(select(m.Artifact).where(m.Artifact.s3_key.in_(keys)))}
        problem = s.get(m.IndexProblem, denied)
    assert avail == {keys[0]: False, denied: True, keys[2]: False}
    assert _exists(s3client, denied) and not _exists(s3client, keys[0])
    assert problem is not None and "AccessDenied" in problem.reason and problem.attempts == 1
    with session_scope(client.app.state.engine) as s:  # a second failure counts another attempt
        assert media.retire_orphans(s, s3, now=T0 + timedelta(hours=1)) == 0
        assert s.get(m.IndexProblem, denied).attempts == 2
    monkeypatch.setattr(s3.client, "delete_objects", real)
    with session_scope(client.app.state.engine) as s:
        assert media.retire_orphans(s, s3, now=T0 + timedelta(days=1)) == 1
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(m.Artifact.available).where(m.Artifact.s3_key == denied)).one() is False
        assert s.get(m.IndexProblem, denied) is None
    assert not _exists(s3client, denied)


def test_delete_keys_reports_the_keys_that_failed(s3client):
    from home_guard_project.cloud.s3 import S3

    s3 = S3(s3client, b.BUCKET)
    b.seed_bucket(s3client)
    for k in ("admin_cache/a.jpg", "admin_cache/b.jpg"):
        b.put(s3client, k, b.FAKE_JPG)
    real = s3.client.delete_objects

    def partial(Bucket, Delete):
        real(Bucket=Bucket, Delete={**Delete, "Objects": [{"Key": "admin_cache/a.jpg"}]})
        return {"Deleted": [{"Key": "admin_cache/a.jpg"}],
                "Errors": [{"Key": "admin_cache/b.jpg", "Code": "InternalError", "Message": "x"}]}

    s3.client.delete_objects = partial
    assert s3.delete_keys(["admin_cache/a.jpg", "admin_cache/b.jpg"]) == {"admin_cache/b.jpg": "InternalError"}


# ---------------------------------------------------------------- D12: export names follow the name bound

def test_export_name_bound_is_120_and_422(client, staff_factory, s3client):
    _index(client, s3client, consent_training=True)
    _, _, _, adm = staff_factory("admin")
    cid = client.post("/v1/studio/collections", headers=adm, json={"name": "c"}).json()["id"]
    long_name = "a" * 121
    for url in ("/v1/studio/exports/preview", "/v1/studio/exports"):
        r = client.post(url, headers=adm, json={"collection_id": cid, "name": long_name, "formats": ["clips"]})
        assert r.status_code == 422, (url, r.status_code, r.text)
    ok = client.post("/v1/studio/exports/preview", headers=adm,
                     json={"collection_id": cid, "name": "a" * 120, "formats": ["clips"]})
    assert ok.status_code == 200, ok.text
    bad = client.post("/v1/studio/exports/preview", headers=adm,
                      json={"collection_id": cid, "name": "Not A Slug", "formats": ["clips"]})
    assert bad.status_code == 422
