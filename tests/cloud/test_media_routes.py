"""Task 11: audited media access and batched owner notices."""
import json
import threading
from datetime import datetime, timedelta

from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, s3client  # noqa: F401  (fixture)


def _setup(client, s3client, consent_recordings=True, consent_training=False):
    from home_guard_project.cloud.s3 import S3

    s3 = S3(s3client, b.BUCKET)
    b.seed_bucket(s3client)
    client.app.state.s3 = s3
    client.clock = NOW
    client.app.state.clock = lambda: client.clock
    with session_scope(client.app.state.engine) as s:
        dev = b.index_fixture_bucket(s, s3, consent_training=consent_training, now=NOW)
        cust = s.get(m.Customer, dev.customer_id)
        cust.consent_recordings = consent_recordings
        s.add(m.Camera(device_pk=dev.id, name="front_side", display_name="Front side"))
        return dev.id, dev.device_id


def _art(client, key):
    with session_scope(client.app.state.engine) as s:
        return s.scalar(select(m.Artifact.id).where(m.Artifact.s3_key == key))


def _notice_keys(s3client, device_id):
    resp = s3client.list_objects_v2(Bucket=b.BUCKET, Prefix=f"fleet/{device_id}/notices/")
    return [o["Key"] for o in resp.get("Contents", [])]


def _body(s3client, key):
    return json.loads(s3client.get_object(Bucket=b.BUCKET, Key=key)["Body"].read())


def _set_camera(client, art_id, camera):
    with session_scope(client.app.state.engine) as s:
        s.get(m.Artifact, art_id).camera = camera


def test_access_returns_presigned_url_and_audits_and_notifies(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    r = client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert b.PROD_CLIP in body["url"] and body["mime"] == "video/mp4"
    assert datetime.fromisoformat(body["expires_utc"].replace("Z", "+00:00")) == NOW + timedelta(seconds=300)
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).all()
        assert len(rows) == 1 and rows[0].target == b.PROD_CLIP and rows[0].reason == "support"
        assert rows[0].device_id == device_id
        assert "X-Amz" not in json.dumps(rows[0].detail or {}) and "X-Amz" not in rows[0].target
    keys = _notice_keys(s3client, device_id)
    assert len(keys) == 1
    notice = _body(s3client, keys[0])
    assert notice["schema_version"] == 1 and notice["kind"] == "recording"
    assert notice["cameras"] == ["Front side"]
    assert notice["message"].startswith("Home Guard support viewed recordings from Front side (")


def test_second_camera_within_window_updates_same_notice(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    a1, a2 = _art(client, b.PROD_CLIP), _art(client, b.FP_CLIP)
    _set_camera(client, a2, "driveway")
    assert client.post(f"/v1/artifacts/{a1}/access", json={"purpose": "support"}, headers=h).status_code == 200
    client.clock = NOW + timedelta(minutes=18)
    assert client.post(f"/v1/artifacts/{a2}/access", json={"purpose": "support"}, headers=h).status_code == 200
    keys = _notice_keys(s3client, device_id)
    assert len(keys) == 1
    notice = _body(s3client, keys[0])
    assert notice["cameras"] == ["Front side", "Driveway"]
    assert "Front side and Driveway" in notice["message"]
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.OwnerNotice)).all()
        assert len(rows) == 1 and rows[0].last_ts - rows[0].first_ts == 18 * 60
        assert not rows[0].pending_upload


def test_view_after_window_creates_second_notice(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h)
    client.clock = NOW + timedelta(minutes=31)
    client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h)
    assert len(_notice_keys(s3client, device_id)) == 2


def test_different_staff_get_separate_notices(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    aid = _art(client, b.PROD_CLIP)
    for _ in range(2):
        _, _, _, h = staff_factory("support")
        client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h)
    assert len(_notice_keys(s3client, device_id)) == 2


def test_labeler_needs_training_consent_and_creates_no_notice(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("labeler")
    aid = _art(client, b.PROD_CLIP)
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "training"}, headers=h).status_code == 403
    with session_scope(client.app.state.engine) as s:
        for c in s.scalars(select(m.Customer)):
            c.consent_training = True
    r = client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "training"}, headers=h)
    assert r.status_code == 200
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h).status_code == 403
    assert _notice_keys(s3client, device_id) == []
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(m.OwnerNotice)).all() == []
        assert len(s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).all()) == 1


def test_support_without_recordings_consent_403_for_everyone(client, staff_factory, s3client):
    _setup(client, s3client, consent_recordings=False)
    aid = _art(client, b.PROD_CLIP)
    for role in ("support", "admin"):
        _, _, _, h = staff_factory(role)
        for purpose in ("support", "review"):
            assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": purpose}, headers=h).status_code == 403


def test_training_purpose_not_for_support(client, staff_factory, s3client):
    _setup(client, s3client, consent_training=True)
    aid = _art(client, b.PROD_CLIP)
    _, _, _, h = staff_factory("support")
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "training"}, headers=h).status_code == 403
    _, _, _, ha = staff_factory("admin")
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "training"}, headers=ha).status_code == 200


def test_unknown_404_unavailable_410_and_auth(client, staff_factory, s3client):
    _setup(client, s3client)
    _, _, _, h = staff_factory("admin")
    assert client.post("/v1/artifacts/999999/access", json={"purpose": "support"}, headers=h).status_code == 404
    aid = _art(client, b.PROD_CLIP)
    with session_scope(client.app.state.engine) as s:
        s.get(m.Artifact, aid).available = False
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h).status_code == 410
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}).status_code == 401


def test_s3_failure_still_succeeds_audits_and_marks_pending(client, staff_factory, s3client, monkeypatch):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    from home_guard_project.cloud.s3 import S3

    def boom(self, key, body):
        raise RuntimeError("s3 down")

    with monkeypatch.context() as mp:
        mp.setattr(S3, "put_json", boom)
        r = client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h)
    assert r.status_code == 200
    with session_scope(client.app.state.engine) as s:
        notice = s.scalars(select(m.OwnerNotice)).one()
        assert notice.pending_upload is True
        assert len(s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).all()) == 1
    assert _notice_keys(s3client, device_id) == []
    from home_guard_project.cloud import audit

    with session_scope(client.app.state.engine) as s:
        assert audit.retry_pending_notices(s, client.app.state.s3, client.app.state.clock()) == 1
    assert len(_notice_keys(s3client, device_id)) == 1
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(m.OwnerNotice)).one().pending_upload is False


def test_concurrent_views_create_one_notice(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    staff, _, _, _ = staff_factory("support")
    from home_guard_project.cloud import audit

    errors = []

    def view():
        try:
            with session_scope(client.app.state.engine) as s:
                dev = s.scalar(select(m.Device))
                audit.owner_notice(s, client.app.state.s3, dev, s.get(m.Staff, staff.id), "recording",
                                   ["Front side"], NOW)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=view) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    with session_scope(client.app.state.engine) as s:
        assert len(s.scalars(select(m.OwnerNotice)).all()) == 1
    assert len(_notice_keys(s3client, device_id)) == 1


def test_thumbnail_redirects_audits_batched_no_notice(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    with session_scope(client.app.state.engine) as s:
        eid = s.scalar(select(m.Event.id).where(m.Event.stem == b.STEM))
        s.add(m.Artifact(event_id=eid, role="thumbnail", s3_key="admin_cache/test/thumb.jpg", provenance="cloud"))
    r = client.get(f"/v1/events/{eid}/thumbnail", headers=h, follow_redirects=False)
    assert r.status_code == 307 and "admin_cache/test/thumb.jpg" in r.headers["location"]
    client.get(f"/v1/events/{eid}/thumbnail", headers=h, follow_redirects=False)
    with session_scope(client.app.state.engine) as s:
        assert len(s.scalars(select(m.AuditLog).where(m.AuditLog.action == "thumbnail_view")).all()) == 1
        assert s.scalars(select(m.OwnerNotice)).all() == []
    assert client.get("/v1/events/999999/thumbnail", headers=h, follow_redirects=False).status_code == 404


def test_thumbnail_labeler_without_consent_404(client, staff_factory, s3client):
    _setup(client, s3client)
    _, _, _, h = staff_factory("labeler")
    with session_scope(client.app.state.engine) as s:
        eid = s.scalar(select(m.Event.id).where(m.Event.stem == b.STEM))
        s.add(m.Artifact(event_id=eid, role="thumbnail", s3_key="admin_cache/test/thumb.jpg", provenance="cloud"))
    assert client.get(f"/v1/events/{eid}/thumbnail", headers=h, follow_redirects=False).status_code == 404


# ---------------------------------------------------------------- fix round 11

def test_failed_commit_is_a_500_and_never_returns_a_presigned_url(client, staff_factory, s3client, monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session

    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    real = Session.commit
    state = {"fail": True}

    def boom(self):
        if state["fail"]:
            raise RuntimeError("commit failed")
        return real(self)

    monkeypatch.setattr(Session, "commit", boom)
    quiet = TestClient(client.app, raise_server_exceptions=False)
    r = quiet.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h)
    state["fail"] = False
    assert r.status_code == 500
    assert "X-Amz" not in r.text and b.PROD_CLIP not in r.text
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).all() == []


def test_audit_row_is_visible_to_a_new_session_when_the_response_arrives(client, staff_factory, s3client):
    _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h).status_code == 200
    with session_scope(client.app.state.engine) as s:
        assert len(s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).all()) == 1


def test_no_route_uses_the_request_scoped_session():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "home_guard_project" / "cloud"
    for p in list(root.glob("routes/*.py")) + [root / "deps.py"]:
        for line in p.read_text(encoding="utf-8").splitlines():
            if "Depends(get_session" in line and not line.lstrip().startswith("#"):
                assert 'scope="function"' in line, f"{p.name}: {line.strip()}"


def test_notice_without_camera_has_no_camera_list_and_no_fake_label(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    _set_camera(client, aid, None)
    with session_scope(client.app.state.engine) as s:
        s.get(m.Artifact, aid).event_id = None  # camera comes from the artifact alone
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h).status_code == 200
    notice = _body(s3client, _notice_keys(s3client, device_id)[0])
    assert notice["cameras"] == []
    assert " from " not in notice["message"]
    assert notice["message"].startswith("Home Guard support viewed recordings (")
    # first == last: a single time, no range
    assert "–" not in notice["message"]


def test_denied_access_is_audited_as_media_denied(client, staff_factory, s3client):
    _setup(client, s3client, consent_recordings=False)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h).status_code == 403
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_denied")).all()
        assert len(rows) == 1 and rows[0].target == b.PROD_CLIP
        assert rows[0].detail["purpose"] == "support" and rows[0].detail["reason"]
        assert "X-Amz" not in json.dumps(rows[0].detail)
        assert s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).all() == []


def test_unattached_artifact_resolves_device_by_key_prefix(client, staff_factory, s3client):
    _, device_id = _setup(client, s3client)
    _, _, _, h = staff_factory("support")
    aid = _art(client, b.PROD_CLIP)
    with session_scope(client.app.state.engine) as s:
        s.get(m.Artifact, aid).event_id = None
    assert client.post(f"/v1/artifacts/{aid}/access", json={"purpose": "support"}, headers=h).status_code == 200
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).one().device_id == device_id
