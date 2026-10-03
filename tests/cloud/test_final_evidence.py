"""Final fix round, evidence truthfulness (fix-final-decisions.md): expiry means "no video left" (Claude#2), derived
media follows its source clip's retention (Claude#3), frozen feedback in export snapshots (I3) and one strict
YOLO label validator for viewing and training (I4)."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, _event_id, _index, s3client  # noqa: F401  (fixture)
from .test_studio import _get, _keys, _make_collection, _seed

LATER = datetime.fromtimestamp(1791020177, timezone.utc) + timedelta(days=20)


def _reindex(client, now):
    from home_guard_project.cloud.indexer import index_device

    with session_scope(client.app.state.engine) as s:
        for dev in s.scalars(select(m.Device)).all():
            index_device(s, client.app.state.s3, dev, full_scan=True, now=now)


# ---------------------------------------------------------------- Claude#2: expiry

def test_training_copy_stays_exportable_after_production_retention(client, staff_factory, s3client):
    _index(client, s3client, consent_training=True)
    client.app.state.export_runner = lambda job: job()
    _, _, _, adm = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    cid = _make_collection(client, adm, [eid])
    s3client.delete_object(Bucket=b.BUCKET, Key=b.PROD_CLIP)  # the box deleted its 14-day production copy
    _reindex(client, LATER)
    with session_scope(client.app.state.engine) as s:
        comp = s.get(m.Event, eid).completeness
    assert comp["video"] is True and comp["expired"] is False
    req = {"collection_id": cid, "name": "late", "formats": ["clips"]}
    preview = client.post("/v1/studio/exports/preview", headers=adm, json=req).json()
    assert preview["included_ids"] == [eid] and preview["excluded"] == []

    s3client.delete_object(Bucket=b.BUCKET, Key=b.TRAIN_CLIP)  # no copy of the video left at all
    _reindex(client, LATER)
    with session_scope(client.app.state.engine) as s:
        comp = s.get(m.Event, eid).completeness
    assert comp["video"] is False and comp["expired"] is True
    preview = client.post("/v1/studio/exports/preview", headers=adm, json=req).json()
    assert preview["excluded"] == [{"event_id": eid, "reason": "expired"}]


# ---------------------------------------------------------------- Claude#3: derived media retention

def test_derived_media_is_deleted_when_no_source_clip_is_left(client, staff_factory, s3client):
    from home_guard_project.cloud import media

    _index(client, s3client, consent_training=True)
    s3 = client.app.state.s3
    gone, kept = _event_id(client, b.STEM), _event_id(client, b.PAUSED_STEM)
    made = {}
    with session_scope(client.app.state.engine) as s:
        s.scalars(select(m.Customer)).one().consent_recordings = True
        for eid in (gone, kept):
            for role, key in (("thumbnail", f"admin_cache/thumbs/{eid}.jpg"),
                              ("filmstrip", f"admin_cache/filmstrips/{eid}.jpg"),
                              ("rendition", f"admin_cache/renditions/{eid}.mp4")):
                b.put(s3client, key, b.FAKE_JPG)
                s.add(m.Artifact(event_id=eid, role=role, s3_key=key, provenance="cloud", etag="x",
                                 detail={"src_etag": "x"}))
            clip = s.scalars(select(m.Artifact).where(m.Artifact.event_id == eid,
                                                      m.Artifact.role == "original_video")).first()
            opaque = f"admin_cache/opaque/{eid:040d}.mp4"
            b.put(s3client, opaque, b.FAKE_MP4)
            s.add(m.Artifact(role="opaque_copy", s3_key=opaque, provenance="cloud",
                             detail={"source_artifact_id": clip.id}))
            made[eid] = [f"admin_cache/thumbs/{eid}.jpg", f"admin_cache/filmstrips/{eid}.jpg",
                         f"admin_cache/renditions/{eid}.mp4", opaque]
    for key in (b.PROD_CLIP, b.TRAIN_CLIP):
        s3client.delete_object(Bucket=b.BUCKET, Key=key)
    _reindex(client, NOW)
    with session_scope(client.app.state.engine) as s:
        media.process_pending(s, s3, limit=0)  # the media loop's pass (no clip needs rendering here)
    with session_scope(client.app.state.engine) as s:
        avail = {a.s3_key: a.available for a in s.scalars(select(m.Artifact).where(m.Artifact.provenance == "cloud"))}
    for key in made[gone]:
        assert avail[key] is False, key
        assert not _keys(s3client, key), key
    for key in made[kept]:
        assert avail[key] is True, key
        assert _keys(s3client, key), key
    _, _, _, adm = staff_factory("admin")
    r = client.get(f"/v1/events/{gone}/thumbnail", headers=adm, follow_redirects=False)
    assert r.status_code == 404
    assert client.get(f"/v1/events/{kept}/thumbnail", headers=adm, follow_redirects=False).status_code == 307
    listed = {it["id"]: it["thumbnail_url"] for it in client.get("/v1/events", headers=adm).json()["items"]}
    assert listed[gone] is None and listed[kept] is not None


# ---------------------------------------------------------------- I3: feedback frozen in the export snapshot

def test_feedback_changed_after_the_request_is_reported_not_exported(client, staff_factory, s3client):
    from home_guard_project.cloud import studio

    _seed(client, s3client)
    _, _, _, adm = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    cid = _make_collection(client, adm, [eid])
    client.app.state.export_runner = lambda job: None  # queued, not built yet
    r = client.post("/v1/studio/exports", headers=adm,
                    json={"collection_id": cid, "name": "fb", "formats": ["clips"]})
    export_id = r.json()["id"]
    b.put(s3client, b.FEEDBACK, b.feedback_body(verdict="false_alarm"))  # the owner changes his answer
    _reindex(client, NOW)
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK)).one().verdict == "false_alarm"
        export = studio.build_export(s, client.app.state.s3, export_id, secret=client.app.state.settings.jwt_secret)
        state, prefix = export.state, export.s3_prefix
    manifest = json.loads(_get(s3client, prefix + "manifest.json"))
    item = next(i for i in manifest["items"] if i["event_id"] == eid)
    assert item["owner_verdicts"] == ["true_alert"]
    assert all(f["verdict"] == "true_alert" for f in item["owner_feedback"])
    assert {"event_id": eid, "reason": "changed_since_request", "note": "owner feedback"} in manifest["missing"]
    assert state == "partial"


def test_unchanged_feedback_is_exported_from_the_snapshot(client, staff_factory, s3client):
    _seed(client, s3client)
    client.app.state.export_runner = lambda job: job()
    _, _, _, adm = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    cid = _make_collection(client, adm, [eid])
    r = client.post("/v1/studio/exports", headers=adm, json={"collection_id": cid, "name": "fb", "formats": ["clips"]})
    out = r.json()
    manifest = json.loads(_get(s3client, out["s3_prefix"] + "manifest.json"))
    item = manifest["items"][0]
    assert [f["verdict"] for f in item["owner_feedback"]] == ["true_alert"]
    assert item["owner_feedback"][0]["received_utc"] == "2026-10-03T09:40:00Z"
    assert manifest["missing"] == [] and out["state"] == "ready"


# ---------------------------------------------------------------- I4: one strict label validator

def test_parse_label_is_strict():
    from home_guard_project.fleet_contract.yolo_labels import parse_label

    rows, problem = parse_label("0 0.5 0.5 0.2 0.4\n\n16 0.3 0.3 0.1 0.1\n")
    assert problem is None and [r.cls for r in rows] == [0, 16]
    assert parse_label("") == ([], None)
    assert parse_label("\n  \n") == ([], None)
    for bad in ("garbage here", "0 0.5 0.5 0.2", "0 0.5 0.5 0.2 0.4 0.9", "0.5 0.5 0.5 0.2 0.2",
                "0 0.5 0.5 -0.2 0.4", "0 0.5 0.5 0 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0.2 inf",
                "-1 0.5 0.5 0.2 0.2", "0 1.5 0.5 0.2 0.2", "x 0.5 0.5 0.2 0.2"):
        assert parse_label("0 0.5 0.5 0.2 0.4\n" + bad + "\n") == ([], "invalid_row"), bad


def test_exporter_and_viewer_share_the_validator():
    from home_guard_project.cloud import studio
    from home_guard_project.fleet_contract.yolo_labels import parse_label

    for text in ("0 0.5 0.5 0.2 0.4\n", "0.5 0.5 0.5 0.2 0.2\n", "0 0.5 0.5 -0.2 0.4\n", "junk\n", ""):
        assert (studio.remap_label(text)[2] == "invalid_row") == (parse_label(text)[1] == "invalid_row"), text


def _collect_frames(client, h):
    eid = _event_id(client, b.COLLECT_STEM)
    r = client.get(f"/v1/events/{eid}/detections", headers=h)
    assert r.status_code == 200, r.text
    return {f["frame_index"]: f for f in r.json()["frames"]}


@pytest.mark.parametrize("text", ["garbage here\n", "0.5 0.5 0.5 0.2 0.2\n", "0 0.5 0.5 -0.2 0.4\n"])
def test_malformed_label_is_not_run_and_a_problem(client, staff_factory, s3client, text):
    from home_guard_project.cloud.routes.events import LABEL_CACHE

    LABEL_CACHE.clear()
    b.seed_bucket(s3client)
    b.put(s3client, b.YOLO_LABELS[0], text)
    _index(client, s3client, seed=False)
    _, _, _, adm = staff_factory("admin")
    frames = _collect_frames(client, adm)
    assert frames[0]["status"] == "not_run" and frames[0]["boxes"] == []
    assert frames[2]["status"] == "ran"
    with session_scope(client.app.state.engine) as s:
        problem = s.get(m.IndexProblem, b.YOLO_LABELS[0])
    assert problem is not None and "label" in problem.reason


def test_empty_label_is_ran_empty(client, staff_factory, s3client):
    from home_guard_project.cloud.routes.events import LABEL_CACHE

    LABEL_CACHE.clear()
    b.seed_bucket(s3client)
    b.put(s3client, b.YOLO_LABELS[0], "")
    _index(client, s3client, seed=False)
    _, _, _, adm = staff_factory("admin")
    assert _collect_frames(client, adm)[0]["status"] == "ran_empty"


def test_detections_read_only_the_indexed_revision(client, staff_factory, s3client):
    from home_guard_project.cloud.routes.events import LABEL_CACHE

    LABEL_CACHE.clear()
    _index(client, s3client)
    _, _, _, adm = staff_factory("admin")
    b.put(s3client, b.YOLO_LABELS[0], "0 0.1 0.1 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")  # replaced, not indexed yet
    frame = _collect_frames(client, adm)[0]
    assert frame["status"] == "not_run"  # never content of a revision the index has not seen
    _reindex(client, NOW)
    LABEL_CACHE.clear()
    frame = _collect_frames(client, adm)[0]
    assert frame["status"] == "ran" and len(frame["boxes"]) == 2
