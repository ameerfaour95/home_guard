"""Task 10: event history, detail, sampled YOLO boxes, review state, labeler pseudonyms, density, review counts."""
import base64
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from home_guard_project.cloud import models as m
from home_guard_project.cloud import pseudonym
from home_guard_project.cloud.db import session_scope

from . import builders as b

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def s3client(monkeypatch):
    import boto3
    from moto import mock_aws

    for name, value in (("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing"),
                        ("AWS_SESSION_TOKEN", "testing"), ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")


def _index(client, s3client, consent_training=False, site=b.SITE, customer_name="Acme", seed=True):
    from home_guard_project.cloud.s3 import S3

    s3 = S3(s3client, b.BUCKET)
    if seed:
        b.seed_bucket(s3client)
    client.app.state.s3 = s3
    client.app.state.clock = lambda: NOW
    with session_scope(client.app.state.engine) as s:
        dev = b.index_fixture_bucket(s, s3, site=site, customer_name=customer_name,
                                     consent_training=consent_training, now=NOW)
        return dev.id, dev.customer_id


@pytest.fixture()
def indexed(client, s3client):
    return _index(client, s3client)


@pytest.fixture()
def indexed_consent(client, s3client):
    return _index(client, s3client, consent_training=True)


def _event_id(client, stem):
    with session_scope(client.app.state.engine) as s:
        return s.scalar(select(m.Event.id).where(m.Event.stem == stem))


def _all_ids(client):
    with session_scope(client.app.state.engine) as s:
        return set(s.scalars(select(m.Event.id)))


def _page_all(client, h, **params):
    ids, cursor, pages = [], None, 0
    while True:
        q = dict(params, limit=1)
        if cursor:
            q["cursor"] = cursor
        r = client.get("/v1/events", params=q, headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        ids += [it["id"] for it in body["items"]]
        pages += 1
        cursor = body["next_cursor"]
        if not cursor:
            return ids, pages
        assert pages < 50


# ---------------------------------------------------------------- list

def test_cursor_pages_cover_every_event_once_in_order(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    ids, pages = _page_all(client, h)
    assert len(ids) == len(set(ids)) == 6  # two events share a start_ts: the id tie-break keeps both
    assert set(ids) == _all_ids(client)
    full = client.get("/v1/events", headers=h).json()
    assert [it["id"] for it in full["items"]] == ids and full["next_cursor"] is None
    starts = [it["start_utc"] for it in full["items"]]
    assert starts == sorted(starts, reverse=True)


def test_summary_fields(client, staff_factory, indexed):
    _, _, _, h = staff_factory("support")
    items = client.get("/v1/events", headers=h).json()["items"]
    front = next(it for it in items if it["id"] == _event_id(client, b.STEM))
    assert front["customer_name"] == "Acme" and front["site"] == "test" and front["camera"] == "front_side"
    assert front["kind"] == "alert" and front["alert_command"] == "[send_message]"
    assert front["timezone"] == "Asia/Jerusalem"
    assert front["completeness"]["copies"] == ["production", "training"]
    assert front["completeness"]["ai"] == "real"
    assert front["reviewed"] is False and front["flagged"] is False and front["thumbnail_url"] is None
    assert front["start_utc"].startswith("2026-10-03T09:36:14")
    assert front["owner_verdicts"] == ["true_alert"] and front["detected"] == ["person"]


def test_thumbnail_url_when_thumbnail_artifact_exists(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    with session_scope(client.app.state.engine) as s:
        s.add(m.Artifact(event_id=eid, role="thumbnail", s3_key="admin_cache/test/thumb.jpg", provenance="cloud"))
    items = client.get("/v1/events", headers=h).json()["items"]
    by_id = {it["id"]: it for it in items}
    assert by_id[eid]["thumbnail_url"] == f"/v1/events/{eid}/thumbnail"
    assert sum(1 for it in items if it["thumbnail_url"]) == 1


def test_filters(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")

    def stems(**params):
        r = client.get("/v1/events", params=params, headers=h)
        assert r.status_code == 200, r.text
        ids = [it["id"] for it in r.json()["items"]]
        with session_scope(client.app.state.engine) as s:
            names = dict(s.execute(select(m.Event.id, m.Event.stem)).all())
        return sorted(names[i] for i in ids)

    assert stems(kind="false_positive") == sorted([b.FP_STEM, b.FALLBACK_STEM])
    assert b.STEM in stems(q="walking") and b.FP_STEM not in stems(q="walking")
    assert stems(q="walking -nothing") == stems(q="walking")
    assert stems(camera="left_side_1") == [b.PAUSED_STEM]
    assert stems(ai="fallback") == [b.FALLBACK_STEM]
    assert stems(verdict="true_alert") == [b.STEM]
    assert stems(site="test") == stems() and stems(site="other") == []
    assert stems(filter="paused") == [b.PAUSED_STEM]
    assert stems(filter="ai_failed") == [b.FALLBACK_STEM]
    assert stems(filter="ai_dismissed_person") == sorted([b.FP_STEM, b.FALLBACK_STEM])
    assert stems(filter="false_alarm") == [] and stems(filter="real_but_wrong") == []
    assert stems(filter="low_conf") == []
    with session_scope(client.app.state.engine) as s:
        ev = s.scalars(select(m.Event).where(m.Event.stem == b.FP_STEM)).one()
        ev.class_max_conf = {"person": 0.31, "car": 0.2}
        ev.owner_verdicts = ["false_alarm"]
    assert stems(filter="low_conf") == [b.FP_STEM]
    assert stems(filter="false_alarm") == [b.FP_STEM] and stems(verdict="false_alarm") == [b.FP_STEM]
    assert stems(from_utc="2026-10-03T00:00:00Z") == sorted(set(stems()) - {b.COLLECT_STEM})
    assert stems(to_utc="2026-10-03T00:00:00Z") == [b.COLLECT_STEM]
    cid = client.get("/v1/events", headers=h).json()["items"][0]["customer_id"]
    assert len(stems(customer_id=cid)) == 6 and stems(customer_id=cid + 100) == []


def test_bad_params(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    assert client.get("/v1/events", params={"filter": "nope"}, headers=h).status_code == 400
    for bad in ("garbage!", base64.urlsafe_b64encode(b"x:y").decode(), base64.urlsafe_b64encode(b"nan:1").decode()):
        assert client.get("/v1/events", params={"cursor": bad}, headers=h).status_code == 400
    assert client.get("/v1/events", params={"limit": 0}, headers=h).status_code == 422
    assert client.get("/v1/events", params={"limit": 501}, headers=h).status_code == 422
    assert len(client.get("/v1/events", params={"limit": 500}, headers=h).json()["items"]) == 6
    assert client.get("/v1/events").status_code == 401


def test_default_limit_is_100(client, staff_factory):
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        dev = b.enroll(s, "bulk", "Bulk")
        for i in range(120):
            s.add(m.Event(device_pk=dev.id, site="bulk", camera="c", stem=f"c_{i}_trigger", kind="trigger",
                          start_ts=1791000000.0 + i))
    body = client.get("/v1/events", headers=h).json()
    assert len(body["items"]) == 100 and body["next_cursor"]
    assert body["items"][0]["completeness"] == {"video": False, "boxes": "none", "ai": "none",
                                                "owner_feedback": False, "expired": False, "copies": []}
    rest = client.get("/v1/events", params={"cursor": body["next_cursor"]}, headers=h).json()
    assert len(rest["items"]) == 20 and rest["next_cursor"] is None


# ---------------------------------------------------------------- detail

def test_detail_admin(client, staff_factory, indexed):
    staff, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    r = client.get(f"/v1/events/{eid}", headers=h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert [run["status"] for run in d["ai_runs"]] == ["real"]
    run = d["ai_runs"][0]
    assert run["purpose"] == "guard" and run["model"] == "gpt-4o" and run["prompt"].startswith("You are the eyes")
    assert run["raw_text_artifact_id"] is not None and len(run["input_frame_artifact_ids"]) == 3
    assert d["dispatch"]["channel"] == "telegram" and d["dispatch"]["sent"] is True
    assert d["dispatch"]["detail"]["telegram"]["telegram"]["results"][0]["ok"] is True
    assert [(f["verdict"], f["raw_text"]) for f in d["feedback"]] == [("true_alert", "test message")]
    roles = {a["role"] for a in d["artifacts"]}
    assert {"original_video", "meta", "raw_answer", "teacher_frame", "feedback"} <= roles
    assert all(a["detail"] is None or "copy" in a["detail"] for a in d["artifacts"])
    assert any(a["s3_key"] == b.TRAIN_CLIP for a in d["artifacts"])
    assert "teacher" in d["raw_meta"] and "dispatch" in d["raw_meta"]["alert"]  # training copy, unredacted
    assert d["fps"] == 4.4 and d["frame_size"] == [704, 576] and d["clip_start_local"] == "2026-10-03 12:36:14"
    assert d["duration_sec"] == pytest.approx(9.7956, abs=1e-3) and d["alert_reason"] == ""
    assert client.get("/v1/events/999999", headers=h).status_code == 404


def test_detail_without_production_copy_has_no_dispatch(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    paused = client.get(f"/v1/events/{_event_id(client, b.PAUSED_STEM)}", headers=h).json()
    assert paused["dispatch"] is None and paused["raw_meta"]["kind"] == "paused"
    assert paused["ai_runs"] == [] or paused["ai_runs"][0]["status"] == "none"
    prod_only = client.get(f"/v1/events/{_event_id(client, b.TRAVERSAL_STEM)}", headers=h).json()
    assert prod_only["raw_meta"]["clip_path"].startswith("..")  # the production copy's newest revision
    assert prod_only["dispatch"]["sent"] is True


def test_event_view_audit_is_batched_per_ten_minutes(client, staff_factory, indexed):
    staff, _, _, h = staff_factory("admin")
    other, _, _, h2 = staff_factory("support")
    eid = _event_id(client, b.STEM)
    other_eid = _event_id(client, b.FP_STEM)

    def views():
        with session_scope(client.app.state.engine) as s:
            return s.execute(select(m.AuditLog.staff_id, m.AuditLog.target).where(
                m.AuditLog.action == "event_view").order_by(m.AuditLog.id)).all()

    client.app.state.clock = lambda: datetime.now(timezone.utc)
    for _ in range(3):
        assert client.get(f"/v1/events/{eid}", headers=h).status_code == 200
    assert len(views()) == 1
    client.get(f"/v1/events/{other_eid}", headers=h)
    client.get(f"/v1/events/{eid}", headers=h2)
    assert len(views()) == 3
    client.app.state.clock = lambda: datetime.now(timezone.utc) + timedelta(minutes=11)
    client.get(f"/v1/events/{eid}", headers=h)
    rows = views()
    assert len(rows) == 4 and rows[-1][0] == staff.id
    with session_scope(client.app.state.engine) as s:
        row = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "event_view")).first()
        assert row.customer_id is not None and row.device_id is not None


# ---------------------------------------------------------------- labeler

def test_labeler_sees_pseudonyms_and_no_dispatch(client, staff_factory, indexed_consent):
    _, cust_id = indexed_consent
    _, _, _, h = staff_factory("labeler")
    secret = client.app.state.settings.jwt_secret
    items = client.get("/v1/events", headers=h).json()["items"]
    assert len(items) == 6
    cust_p = pseudonym.customer(secret, cust_id)
    cam_p = pseudonym.camera(secret, "test", "front_side")
    assert cust_p.startswith("customer-") and len(cust_p) == len("customer-") + 6 and cam_p.startswith("cam-")
    for it in items:
        assert it["customer_name"] == cust_p and it["site"] == cust_p and it["camera"].startswith("cam-")
    assert {it["camera"] for it in items} >= {cam_p}
    eid = _event_id(client, b.STEM)
    r = client.get(f"/v1/events/{eid}", headers=h)
    assert r.status_code == 200
    d = r.json()
    text = r.text
    assert d["camera"] == cam_p and d["dispatch"] is None
    assert "dispatch" not in json.dumps(d["raw_meta"])
    assert d["raw_meta"]["teacher"]["prompt"].startswith("You are the eyes")
    assert all(f["raw_text"] == "" and f["note"] == "" for f in d["feedback"]) and d["feedback"]
    assert "Acme" not in text and "-100100" not in text and "test message" not in text
    assert all(not a["s3_key"].startswith(("dataset_test/", "production_test/")) for a in d["artifacts"])
    assert all("front_side" not in a["s3_key"] for a in d["artifacts"])
    # filters take the pseudonyms the labeler sees
    by_cam = client.get("/v1/events", params={"camera": cam_p}, headers=h).json()["items"]
    assert {it["id"] for it in by_cam} == {eid, _event_id(client, b.TRAVERSAL_STEM)}
    assert len(client.get("/v1/events", params={"site": cust_p}, headers=h).json()["items"]) == 6
    assert client.get("/v1/events", params={"site": "test"}, headers=h).json()["items"] == []
    assert client.get("/v1/events", params={"camera": "front_side"}, headers=h).json()["items"] == []
    # stable across requests
    assert client.get("/v1/events", headers=h).json()["items"] == items


def test_labeler_without_training_consent_sees_nothing(client, staff_factory, indexed):
    _, _, _, h = staff_factory("labeler")
    eid = _event_id(client, b.STEM)
    assert client.get("/v1/events", headers=h).json() == {"items": [], "next_cursor": None, "total": None, "total_capped": False}
    assert client.get(f"/v1/events/{eid}", headers=h).status_code == 404
    assert client.get(f"/v1/events/{eid}/detections", headers=h).status_code == 404
    assert client.patch(f"/v1/events/{eid}/review", json={"reviewed": True}, headers=h).status_code == 404
    assert client.get("/v1/events/review-count", headers=h).json() == {"unreviewed_24h": 0, "flagged_open": 0}
    d = client.get("/v1/events/density", params={"from_utc": "2026-10-03T00:00:00Z",
                                                 "to_utc": "2026-10-04T00:00:00Z"}, headers=h).json()
    assert d["rows"] == []


# ---------------------------------------------------------------- detections

def test_detections_for_the_collection_clip(client, staff_factory, s3client):
    from home_guard_project.cloud.routes import events as events_routes

    events_routes.LABEL_CACHE.clear()
    b.seed_bucket(s3client)
    b.put(s3client, b.YOLO_LABELS[1], "")  # second sampled frame: YOLO ran, found nothing
    _index(client, s3client, seed=False)
    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.COLLECT_STEM)
    r = client.get(f"/v1/events/{eid}/detections", headers=h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provenance"] == "sampled" and d["model"] == "yolo (collection export)"
    frames = {f["frame_index"]: f for f in d["frames"]}
    assert len(frames) == 32 and [f["frame_index"] for f in d["frames"]] == sorted(frames)
    f0 = frames[0]
    assert f0["status"] == "ran" and f0["t_sec"] == 0.0
    box = f0["boxes"][0]
    assert box["label"] == "person" and box["cls"] == 0 and box["conf"] is None
    assert all(0 <= v <= 1 for v in box["xyxy"])
    assert box["xyxy"] == pytest.approx([0.4, 0.3, 0.6, 0.7])
    assert frames[2]["status"] == "ran_empty" and frames[2]["boxes"] == []
    assert frames[2]["t_sec"] == pytest.approx(2 / 7.0)
    assert frames[4]["status"] == "not_run"  # label file never uploaded
    # cached per (key, etag): a second call reads nothing from S3
    calls = []
    real = client.app.state.s3.get_text
    client.app.state.s3.get_text = lambda *a, **k: calls.append(a) or real(*a, **k)
    assert client.get(f"/v1/events/{eid}/detections", headers=h).json() == d
    assert calls == []
    # a new revision of the label (new ETag) is read again
    b.put(s3client, b.YOLO_LABELS[1], "2 0.5 0.5 1.0 1.0\n")
    with session_scope(client.app.state.engine) as s:
        from home_guard_project.cloud.indexer import index_device

        index_device(s, client.app.state.s3, s.get(m.Device, s.scalar(select(m.Device.id))), now=NOW)
    again = client.get(f"/v1/events/{eid}/detections", headers=h).json()
    f2 = next(f for f in again["frames"] if f["frame_index"] == 2)
    assert f2["status"] == "ran" and f2["boxes"][0]["label"] == "car" and f2["boxes"][0]["xyxy"] == [0, 0, 1, 1]


def test_detections_none_when_not_sampled(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    d = client.get(f"/v1/events/{_event_id(client, b.STEM)}/detections", headers=h).json()
    assert d == {"provenance": "none", "model": None, "frames": []}
    assert client.get("/v1/events/999999/detections", headers=h).status_code == 404


def test_labeler_detections_with_consent(client, staff_factory, indexed_consent):
    _, _, _, h = staff_factory("labeler")
    d = client.get(f"/v1/events/{_event_id(client, b.COLLECT_STEM)}/detections", headers=h).json()
    assert d["provenance"] == "sampled" and d["frames"][0]["boxes"][0]["label"] == "person"


# ---------------------------------------------------------------- review

def test_review_patch_round_trips(client, staff_factory, indexed):
    staff, _, _, h = staff_factory("support")
    eid = _event_id(client, b.STEM)
    r = client.patch(f"/v1/events/{eid}/review", json={"reviewed": True}, headers=h)
    assert r.status_code == 200, r.text
    assert (r.json()["id"], r.json()["reviewed"], r.json()["flagged"]) == (eid, True, False)
    r = client.patch(f"/v1/events/{eid}/review", json={"flagged": True}, headers=h)
    assert (r.json()["reviewed"], r.json()["flagged"]) == (True, True)
    assert [it["id"] for it in client.get("/v1/events", params={"reviewed": True}, headers=h).json()["items"]] == [eid]
    assert eid not in [it["id"] for it in client.get("/v1/events", params={"reviewed": False}, headers=h).json()["items"]]
    assert [it["id"] for it in client.get("/v1/events", params={"flagged": True}, headers=h).json()["items"]] == [eid]
    assert client.get(f"/v1/events/{eid}", headers=h).json()["reviewed"] is True
    with session_scope(client.app.state.engine) as s:
        rs = s.get(m.ReviewState, eid)
        assert rs.by == staff.id and rs.at is not None
        audits = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "review")).all()
    assert len(audits) == 2 and audits[0].detail == {"reviewed": True}
    assert client.patch("/v1/events/999999/review", json={"reviewed": True}, headers=h).status_code == 404


def test_labeler_review_returns_pseudonyms(client, staff_factory, indexed_consent):
    _, _, _, h = staff_factory("labeler")
    r = client.patch(f"/v1/events/{_event_id(client, b.STEM)}/review", json={"flagged": True}, headers=h)
    assert r.status_code == 200 and r.json()["customer_name"].startswith("customer-")
    assert r.json()["camera"].startswith("cam-")


# ---------------------------------------------------------------- aggregates

def test_review_count(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    assert client.get("/v1/events/review-count", headers=h).json() == {"unreviewed_24h": 6, "flagged_open": 0}
    eid, fid = _event_id(client, b.STEM), _event_id(client, b.FP_STEM)
    client.patch(f"/v1/events/{eid}/review", json={"reviewed": True}, headers=h)
    client.patch(f"/v1/events/{fid}/review", json={"flagged": True}, headers=h)
    assert client.get("/v1/events/review-count", headers=h).json() == {"unreviewed_24h": 5, "flagged_open": 1}
    client.patch(f"/v1/events/{fid}/review", json={"reviewed": True}, headers=h)
    assert client.get("/v1/events/review-count", headers=h).json() == {"unreviewed_24h": 4, "flagged_open": 0}
    client.app.state.clock = lambda: NOW + timedelta(hours=1)  # collect clip (10-02 12:30Z) is now > 24 h old
    assert client.get("/v1/events/review-count", headers=h).json()["unreviewed_24h"] == 3


def test_density_hourly(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    r = client.get("/v1/events/density", params={"from_utc": "2026-10-03T00:30:00Z", "to_utc": "2026-10-03T11:15:00Z",
                                                 "site": "test"}, headers=h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["bucket"] == "hour" and d["timezone"] == "Asia/Jerusalem"
    starts = d["starts_utc"]
    assert starts[0].startswith("2026-10-03T00:00:00") and starts[-1].startswith("2026-10-03T11:00:00")
    assert len(starts) == 12
    rows = {row["camera"]: row for row in d["rows"]}
    heartbeat_cams = set(b.fixture_json("heartbeat.json")["cameras"])
    assert heartbeat_cams | {"back_door", "front_side", "left_side_1"} == set(rows)
    assert list(rows) == sorted(rows)
    front = rows["front_side"]
    assert len(front["events"]) == 12 and front["events"][9] == 2 and sum(front["events"]) == 2
    assert front["alerts"][9] == 2 and sum(front["false_alarms"]) == 0
    assert sum(rows["back_door"]["events"]) == 2  # the collect clip of 10-02 is outside the window
    assert sum(rows["bian_ch2"]["events"]) == 0
    assert sum(rows["left_side_1"]["events"]) == 1 and rows["left_side_1"]["alerts"] == [0] * 12
    kind = client.get("/v1/events/density", params={"from_utc": "2026-10-03T00:00:00Z",
                                                    "to_utc": "2026-10-03T12:00:00Z", "kind": "false_positive"},
                      headers=h).json()
    assert sum(sum(row["events"]) for row in kind["rows"]) == 2
    cam = client.get("/v1/events/density", params={"from_utc": "2026-10-03T00:00:00Z",
                                                   "to_utc": "2026-10-03T12:00:00Z", "camera": "front_side"},
                     headers=h).json()
    assert [row["camera"] for row in cam["rows"]] == ["front_side"]


def test_density_daily_false_alarms_and_bounds(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        ev = s.scalars(select(m.Event).where(m.Event.stem == b.FP_STEM)).one()
        ev.owner_verdicts = ["false_alarm"]
    d = client.get("/v1/events/density", params={"from_utc": "2026-10-02T05:00:00Z", "to_utc": "2026-10-03T13:00:00Z",
                                                 "bucket": "day"}, headers=h).json()
    assert [s[:10] for s in d["starts_utc"]] == ["2026-10-02", "2026-10-03"]
    back = next(row for row in d["rows"] if row["camera"] == "back_door")
    assert back["events"] == [1, 2] and back["false_alarms"] == [0, 1]
    assert client.get("/v1/events/density", params={"from_utc": "2026-10-03T00:00:00Z",
                                                    "to_utc": "2026-10-02T00:00:00Z"}, headers=h).status_code == 400
    assert client.get("/v1/events/density", params={"from_utc": "2020-01-01T00:00:00Z",
                                                    "to_utc": "2026-10-02T00:00:00Z"}, headers=h).status_code == 400


def test_density_labeler_pseudonyms_and_multi_site_utc(client, staff_factory, indexed_consent):
    _, cust_id = indexed_consent
    _, _, _, lab = staff_factory("labeler")
    secret = client.app.state.settings.jwt_secret
    d = client.get("/v1/events/density", params={"from_utc": "2026-10-03T00:00:00Z",
                                                 "to_utc": "2026-10-03T12:00:00Z"}, headers=lab).json()
    cams = {row["camera"] for row in d["rows"]}
    assert pseudonym.camera(secret, "test", "front_side") in cams and all(c.startswith("cam-") for c in cams)
    with session_scope(client.app.state.engine) as s:
        b.enroll(s, "other", "Other Co")
    _, _, _, adm = staff_factory("admin")
    d = client.get("/v1/events/density", params={"from_utc": "2026-10-03T00:00:00Z",
                                                 "to_utc": "2026-10-03T12:00:00Z"}, headers=adm).json()
    assert d["timezone"] == "UTC"


def test_fleet_activity(client, staff_factory, indexed):
    _, _, _, h = staff_factory("support")
    _, _, _, lab = staff_factory("labeler")
    r = client.get("/v1/fleet/activity", params={"hours": 24}, headers=h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["bucket"] == "hour" and d["timezone"] == "UTC" and len(d["starts_utc"]) == 24
    # the last bucket is the hour that contains now; the first starts 23 h before it
    assert d["starts_utc"][-1].startswith("2026-10-03T12:00:00") and d["starts_utc"][0].startswith("2026-10-02T13:00")
    assert [row["camera"] for row in d["rows"]] == ["fleet"]
    row = d["rows"][0]
    assert len(row["events"]) == 24 and sum(row["events"]) == 5 and sum(row["alerts"]) == 2  # collect clip too old
    assert client.get("/v1/fleet/activity", params={"hours": 4}, headers=h).json()["rows"][0]["events"] == [2, 0, 0, 0]
    assert client.get("/v1/fleet/activity", headers=lab).status_code == 403
    assert client.get("/v1/fleet/activity", params={"hours": 169}, headers=h).status_code == 422


# ---------------------------------------------------------------- fix round (Codex review of Task 10)

def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("ascii")).decode("ascii")


def test_invalid_cursors_are_400_before_any_query(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    bad = ["", "=", _b64("1791020174.1329982:999999999999999999999999999999"), _b64(f"1791020174.1:{2 ** 63}"),
           _b64("1791020174.1:0"), _b64("1791020174.1:-1"), _b64("1.0:2:3"), _b64("inf:1"), _b64("-inf:1"),
           _b64("1e999:1"), _b64(" 1.0:2"), _b64("1.0: 2"), _b64("1.0:"), _b64(":1"), _b64("1.0:+2"), _b64("1_0:2")]
    for cursor in bad:
        r = client.get("/v1/events", params={"cursor": cursor}, headers=h)
        assert r.status_code == 400, (cursor, r.status_code, r.text)
    ok = client.get("/v1/events", params={"cursor": _b64(f"1791020174.1:{2 ** 63 - 1}")}, headers=h)
    assert ok.status_code == 200 and ok.json()["items"]
    assert len(client.get("/v1/events", headers=h).json()["items"]) == 6  # absent cursor: first page
    with session_scope(client.app.state.engine) as s:
        col = m.Collection(name="c", created_at=NOW)
        s.add(col)
        s.flush()
        cid = col.id
    for cursor in ("", _b64(f"1.0:{2 ** 63}")):
        r = client.get(f"/v1/studio/collections/{cid}/items", params={"cursor": cursor}, headers=h)
        assert r.status_code == 400, (cursor, r.text)


def test_density_validates_the_requested_interval_not_the_rounded_one(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")

    def status(f, t, bucket="hour"):
        return client.get("/v1/events/density", params={"from_utc": f, "to_utc": t, "bucket": bucket},
                          headers=h).status_code

    for bucket in ("hour", "day"):
        assert status("2026-10-03T10:50:00Z", "2026-10-03T10:10:00Z", bucket) == 400  # reversed, one bucket
        assert status("2026-10-03T10:30:00Z", "2026-10-03T10:30:00Z", bucket) == 400  # empty, one bucket
        assert status("2026-10-03T10:10:00Z", "2026-10-03T10:50:00Z", bucket) == 200
    # naive endpoints are UTC; an offset endpoint is compared after conversion to UTC
    assert status("2026-10-03T10:10:00", "2026-10-03T10:50:00") == 200
    assert status("2026-10-03T10:10:00", "2026-10-03T10:50:00+03:00") == 400  # 07:50Z is before 10:10Z
    d = client.get("/v1/events/density", params={"from_utc": "2026-10-03T10:10:00Z",
                                                 "to_utc": "2026-10-03T10:50:00Z"}, headers=h).json()
    assert len(d["starts_utc"]) == 1 and d["starts_utc"][0].startswith("2026-10-03T10:00:00")


def test_event_view_audit_is_race_free(client, staff_factory, indexed, monkeypatch):
    import threading
    import time
    from types import SimpleNamespace

    from home_guard_project.cloud import audit
    from home_guard_project.cloud.routes import events as events_routes

    staff, _, _, _ = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    real_record = audit.record

    def slow_record(*args, **kwargs):  # widen the window between the check and the insert
        time.sleep(0.3)
        return real_record(*args, **kwargs)

    monkeypatch.setattr(audit, "record", slow_record)
    request = SimpleNamespace(app=client.app)
    n = 6
    barrier = threading.Barrier(n)
    errors = []

    def view():
        try:
            with session_scope(client.app.state.engine) as s:
                ev = s.get(m.Event, eid)
                cid = s.scalar(select(m.Device.customer_id).where(m.Device.id == ev.device_pk))
                who = s.get(m.Staff, staff.id)
                barrier.wait(timeout=20)
                events_routes._audit_view(s, request, who, ev, cid)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=view) for _ in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "event_view")).all()
    assert len(rows) == 1 and rows[0].ts == NOW  # the batching clock is also the audit row's clock
