"""/v1/tagging: the studio over the database (indexed customer events + their answers) and the local dataset; tags as
append-only tag_events; media grants with consent; exports to local files; admin only."""
import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.tagstudio.config import StudioPaths
from home_guard_project.cloud.tagstudio.service import TagStudio

from .tagstudio_fixtures import VIDEO_BYTES, alert_meta, dataset_row, eval_results, feedback_record, make_dataset, \
    put_owner_clip

STEM = "house2_ch2_1791091199_alert"
NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _household(s, name, site, consent):
    customer = m.Customer(name=name, consent_live=consent, consent_recordings=consent, consent_training=consent)
    s.add(customer); s.flush()
    device = m.Device(device_id=str(uuid.uuid4()), site=site, customer_id=customer.id)
    s.add(device); s.flush()
    return customer, device


def _event(s, device, stem, label="suspicious", summary="Two people walk along the wall.", verdict=None,
           owner_label="", videos=True):
    ev = m.Event(device_pk=device.id, site=device.site, camera=stem.rsplit("_", 2)[0], stem=stem, kind="alert",
                 start_ts=1791091195.0, end_ts=1791091205.0, day="2026-10-04", summary=summary, label=label,
                 clip_start_local="2026-10-04 08:19:55", duration_sec=9.7, fps=7.0)
    s.add(ev); s.flush()
    s.add(m.AiRun(event_id=ev.id, status="real", model="gpt-4o", prompt_version="v1",
                  parsed={"summary": summary, "label": label, "raw_label": label}))
    if videos:
        for role, folder in (("original_video", "clips"), ("crop_video", "vlm_crops")):
            s.add(m.Artifact(event_id=ev.id, role=role, s3_key=f"dataset_{device.site}/{folder}/{stem}.mp4"))
    if verdict:
        key = f"production_{device.site}/feedback/{stem}_1.feedback.json"
        s.add(m.Feedback(device_pk=device.id, event_id=ev.id, alert_stem=stem, verdict=verdict, s3_key=key,
                         received_at=NOW))
        s.add(m.RawRevision(s3_key=key, etag="e1", fetched_at=NOW, body=feedback_record(
            stem, ev.camera, "2026-10-04T05:40:00Z", verdict, owner_label)))
    s.flush()
    return ev.id


@pytest.fixture()
def studio(client, tmp_path):
    ds = make_dataset(str(tmp_path / "ds"), [
        dataset_row("front_side_1771696865_trigger", "A man walks past.", alert=False),
        dataset_row("front_side_1771696897_trigger", "Climbs the gate.", alert=True)])
    # a local copy of the consenting household's clip, and of the other household's
    put_owner_clip(ds, "production_house2", alert_meta("house2_ch2", STEM, label="suspicious"), STEM)
    put_owner_clip(ds, "production_other", alert_meta("other_ch1", "other_ch1_1791091300_alert"),
                   "other_ch1_1791091300_alert")
    eval_results(str(tmp_path / "eval" / "results"), "p__m", "m", {"front_side_1771696865_trigger": "suspicious"},
                 0.9, 0.1)
    s = TagStudio(StudioPaths.resolve(env={}, dataset=ds, eval_dir=str(tmp_path / "eval"),
                                      exports=str(tmp_path / "exports")))
    client.app.state.tagstudio = s
    client.app.state.clock = lambda: NOW
    with session_scope(client.app.state.engine) as session:
        _, house2 = _household(session, "House two", "house2", True)
        _, other = _household(session, "Other", "other", False)
        ids = {"consenting": _event(session, house2, STEM, verdict="false_alarm", owner_label="empty"),
               "refusing": _event(session, other, "other_ch1_1791091300_alert", label="normal")}
    return s, ids


def test_state_and_queue_join_every_source(client, staff_factory, studio):
    _, ids = studio
    _, _, _, h = staff_factory("admin")
    state = client.get("/v1/tagging/state", headers=h).json()
    assert state["total"] == 4 and [g["id"] for g in state["taxonomy"]["groups"]] == ["N", "S", "E"]
    queue = client.get("/v1/tagging/queue?tier=all", headers=h).json()
    keys = [i["key"] for i in queue["items"]]
    # the local owner-feedback copies joined their indexed events: one clip each
    assert set(keys) == {f"ev:{ids['consenting']}", f"ev:{ids['refusing']}", "ds:front_side_1771696865_trigger",
                         "ds:front_side_1771696897_trigger"}
    first = queue["items"][0]
    assert first["key"] == f"ev:{ids['consenting']}" and first["tier_name"] == "contradiction"
    assert first["labels"]["owner"]["label"] == "empty" and first["labels"]["ai"]["label"] == "suspicious"
    opened = client.get("/v1/tagging/queue", headers=h).json()["items"]
    assert "ds:front_side_1771696897_trigger" not in [i["key"] for i in opened]      # a migrated tag counts as done


def test_save_is_append_only_and_validated(client, staff_factory, studio):
    _, ids = studio
    _, _, _, h = staff_factory("admin")
    key = f"ev:{ids['consenting']}"
    r = client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {"category": "Q1"}})
    assert r.status_code == 422 and "category" in r.json()["detail"]
    r = client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {
        "category": "N6", "zone": "yard", "description": "Two people walk along the wall to the car."}})
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["tag"]["raw_label"] == "normal" and saved["assessment"]["tier_name"] == "done"
    assert saved["next_key"] == "ds:front_side_1771696865_trigger"
    client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {"needs_check": True}})
    clip = client.get("/v1/tagging/clip", params={"key": key}, headers=h).json()
    assert clip["prefilled_from"] == "studio" and clip["form"]["zone"] == "yard" and clip["form"]["needs_check"]
    assert len(clip["history"]) == 2 and clip["assessment"]["tier_name"] == "check"
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.TagEvent).order_by(m.TagEvent.id)).all()
        assert [r.fields for r in rows][1] == {"needs_check": True} and rows[0].clip_id == STEM
    assert client.post("/v1/tagging/tag", headers=h, json={"key": "ds:nope", "fields": {}}).status_code == 404


def test_media_follows_consent_and_is_served_with_range(client, staff_factory, studio):
    _, ids = studio
    _, _, _, h = staff_factory("admin")
    r = client.post("/v1/tagging/media", headers=h, json={"key": f"ev:{ids['refusing']}", "kind": "clip"})
    assert r.status_code == 403 and r.json()["detail"] == "This customer has not agreed to training use"
    r = client.post("/v1/tagging/media", headers=h, json={"key": f"ev:{ids['consenting']}", "kind": "clip"})
    assert r.status_code == 200, r.text      # the local copy, so no storage is needed
    url = r.json()["url"]
    assert "/v1/tagging/file/" in url
    path = url.split("://", 1)[1].split("/", 1)[1]
    got = client.get("/" + path, headers={"Range": "bytes=100-199"})
    assert got.status_code == 206 and got.content == VIDEO_BYTES[100:200]
    assert client.get("/" + path[:-3] + "xyz").status_code == 404
    r = client.post("/v1/tagging/media", headers=h, json={"key": "ds:front_side_1771696865_trigger", "kind": "crop"})
    assert r.status_code == 200
    with session_scope(client.app.state.engine) as s:
        actions = s.scalars(select(m.AuditLog.action)).all()
        assert actions.count("media_view") == 1      # the customer clip was audited; our own dataset clip is not


def test_export_writes_local_files(client, staff_factory, studio, tmp_path):
    _, ids = studio
    _, _, _, h = staff_factory("admin")
    client.post("/v1/tagging/tag", headers=h, json={"key": f"ev:{ids['consenting']}", "fields": {
        "category": "S6", "description": "Two people walk along the side of the house."}})
    out = client.post("/v1/tagging/export", headers=h, json={}).json()
    assert os.path.isfile(out["training_path"]) and str(tmp_path / "exports") in out["training_path"]
    assert out["counts"]["studio"] == 1 and out["counts"]["migrated"] == 1 and out["counts"]["contradicted"] == 1
    lines = open(out["training_path"], encoding="utf-8").read().splitlines()
    assert any('"category": "S6"' in line and f"dataset_house2/clips/{STEM}.mp4" in line for line in lines)


@pytest.mark.parametrize("role", ["support", "labeler"])
def test_admin_only(client, staff_factory, studio, role):
    _, _, _, h = staff_factory(role)
    assert client.get("/v1/tagging/state", headers=h).status_code == 403
    assert client.post("/v1/tagging/tag", headers=h, json={"key": "ds:x", "fields": {}}).status_code == 403
    assert client.get("/v1/tagging/queue").status_code == 401
