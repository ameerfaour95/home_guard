"""/v1/tagging: the studio over the database (indexed customer events + their answers) and the local dataset; tags as
append-only tag_events; media grants with consent; exports to local files; admin only."""
import json
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
    # a dataset clip no prompt answered is tagged with the category form
    r = client.post("/v1/tagging/tag", headers=h, json={"key": "ds:front_side_1771696865_trigger",
                                                        "fields": {"raw_label": "suspicious"}})
    assert r.status_code == 422 and "Choose a category" in r.json()["detail"]
    r = client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {
        "category": "N6", "zone": "yard", "description": "Two people walk along the wall to the car."}})
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["tag"]["raw_label"] == "normal" and saved["assessment"]["tier_name"] == "done"
    assert saved["next_key"] == "ds:front_side_1771696865_trigger"
    client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {"needs_check": True}})
    clip = client.get("/v1/tagging/clip", params={"key": key}, headers=h).json()
    assert clip["prefilled_from"] == "studio" and clip["form"]["zone"] == "yard" and clip["form"]["needs_check"]
    old = client.get("/v1/tagging/clip", params={"key": "ds:front_side_1771696897_trigger"}, headers=h).json()
    assert old["prefilled_from"] == "old" and old["form"]["category"] == "" and old["form"]["raw_label"] == "suspicious"
    assert len(clip["history"]) == 2 and clip["assessment"]["tier_name"] == "check"
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.TagEvent).order_by(m.TagEvent.id)).all()
        assert [r.fields for r in rows][1] == {"needs_check": True} and rows[0].clip_id == STEM
        assert rows[0].fields["prompt_version"] == "2026-10-03.test"     # the clip's prompt version, stored
    assert client.post("/v1/tagging/tag", headers=h, json={"key": "ds:nope", "fields": {}}).status_code == 404


def test_media_follows_consent_and_is_served_with_range(client, staff_factory, studio):
    _, ids = studio
    _, _, _, h = staff_factory("admin")
    r = client.post("/v1/tagging/media", headers=h, json={"key": f"ev:{ids['refusing']}", "kind": "clip"})
    assert r.status_code == 403 and r.json()["detail"] == "This customer withdrew consent for training use"
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


def test_missing_video_says_why_and_playable_clips_come_first(client, staff_factory, studio):
    _, ids = studio
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        house = s.scalar(select(m.Device).where(m.Device.site == "house2"))
        gone = _event(s, house, "house2_ch5_1791091500_alert", verdict="false_alarm", owner_label="empty", videos=False)
        s.add(m.Artifact(event_id=gone, role="original_video", s3_key="dataset_house2/clips/gone.mp4", available=False))
    clip = client.get("/v1/tagging/clip", params={"key": f"ev:{gone}"}, headers=h).json()
    assert clip["media"] == {"clip": False, "crop": False}
    assert clip["media_reasons"]["clip"].startswith("removed from S3")
    assert clip["media_reasons"]["crop"] == "never uploaded to S3"
    refusing = client.get("/v1/tagging/clip", params={"key": f"ev:{ids['refusing']}"}, headers=h).json()
    assert refusing["media_reasons"]["clip"] == "no training consent from this customer"
    dataset = client.get("/v1/tagging/clip", params={"key": "ds:front_side_1771696897_trigger"}, headers=h).json()
    assert dataset["media"] == {"clip": True, "crop": True} and dataset["media_reasons"] == {"clip": "", "crop": ""}
    # both are the same kind of contradiction (owner empty vs AI suspicious): the one with a video comes first
    queue = client.get("/v1/tagging/queue", headers=h).json()["items"]
    order = [i["key"] for i in queue]
    assert order.index(f"ev:{ids['consenting']}") < order.index(f"ev:{gone}")
    assert next(i for i in queue if i["key"] == f"ev:{gone}")["has_media"] is False


# ---------------------------------------------------------------- delete, YOLO boxes of dataset clips, "Suggest tag"

def test_delete_takes_a_clip_out_of_the_queue_and_the_export(client, staff_factory, studio):
    _, _, _, h = staff_factory("admin")
    key = "ds:front_side_1771696865_trigger"
    assert key in [i["key"] for i in client.get("/v1/tagging/queue", headers=h).json()["items"]]
    r = client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {"delete": True}})
    assert r.status_code == 200, r.text
    assert key not in [i["key"] for i in client.get("/v1/tagging/queue", headers=h).json()["items"]]
    out = client.post("/v1/tagging/export", headers=h, json={}).json()
    assert out["counts"]["delete"] == 1
    for path in (out["training_path"], out["eval_path"]):
        assert "front_side_1771696865_trigger" not in open(path, encoding="utf-8").read()


def _yolo(ds, folder, clip_id, frames):
    """Label files of the unified dataset: {frame: [(class, cx, cy, w, h)]}, contiguous ids (0 person, 2 car)."""
    root = os.path.join(ds, "yolo", "labels", folder)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(ds, "yolo", "classes.txt"), "w", encoding="utf-8") as f:
        f.write("0 person\n1 bicycle\n2 car\n3 motorcycle\n4 bus\n5 truck\n6 bird\n7 cat\n8 dog\n")
    for frame, boxes in frames.items():
        with open(os.path.join(root, f"{clip_id}_f{frame:04d}.txt"), "w", encoding="utf-8") as f:
            f.write("".join(f"{c} {cx} {cy} {w} {h}\n" for c, cx, cy, w, h in boxes))


def test_dataset_clip_opens_with_its_yolo_boxes_editable(client, staff_factory, studio):
    s, _ = studio
    _, _, _, h = staff_factory("admin")
    clip = "front_side_1771696865_trigger"
    walk = {f: [(0, 0.2 + f * 0.01, 0.5, 0.1, 0.3), (2, 0.7, 0.7, 0.2, 0.2)] for f in range(0, 40, 4)}
    _yolo(str(s.paths.dataset), "front_side", clip, walk)
    r = client.get("/v1/tagging/boxes", params={"key": f"ds:{clip}"}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 0 and body["status"] == "new" and body["suggestions_used"] is True
    assert sorted(t["label"] for t in body["tracks"]) == ["car", "person"]
    assert all(t["source"] == "yolo" for t in body["tracks"]) and body["frame_count"] >= 37
    person = next(t for t in body["tracks"] if t["label"] == "person")
    assert person["keyframes"][0]["xyxy"] == pytest.approx([0.15, 0.35, 0.25, 0.65])
    # the tagger deletes the car and fixes the person: saved as a version of its own, the car gone for good
    person["source"] = "human"
    r = client.put("/v1/tagging/boxes", headers=h, json={"key": f"ds:{clip}", "base_version": 0, "tracks": [person],
                                                          "description": "A man walks past.", "status": "edited"})
    assert r.status_code == 200, r.text
    again = client.get("/v1/tagging/boxes", params={"key": f"ds:{clip}"}, headers=h).json()
    assert again["version"] == 1 and [t["label"] for t in again["tracks"]] == ["person"]
    assert again["tracks"][0]["source"] == "human" and again["tracks"][0]["track_id"].startswith("t-")
    stale = client.put("/v1/tagging/boxes", headers=h, json={"key": f"ds:{clip}", "base_version": 0, "tracks": [],
                                                              "description": "", "status": "edited"})
    assert stale.status_code == 409
    # a clip without label files opens empty (the Label view says "No YOLO boxes for this clip")
    empty = client.get("/v1/tagging/boxes", params={"key": "ds:front_side_1771696897_trigger"}, headers=h).json()
    assert empty["tracks"] == [] and empty["suggestions_used"] is False


class FakeModel:
    model = "test/qwen-fake"

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.chat = self.completions = self

    def create(self, **kwargs):
        import json
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.answer)))])


def _frames(video):
    import numpy as np

    return [np.zeros((12, 16, 3), np.uint8)] * 5, [0, 17, 34, 51, 67], 7.0


def test_suggest_fills_the_whole_tag_once_per_clip(client, staff_factory, studio):
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    # a clip the Eye answered: the suggestion follows the Eye's category form
    put_owner_clip(str(s.paths.dataset), "production_house2", alert_meta(
        "house2_ch2", STEM, label="suspicious", teacher={"model": "m", "prompt_version": "2026-10-06.eye-v3"}), STEM)
    fake = FakeModel({"summary": "A man walks to the gate, tries the latch and walks away.", "category": "s1",
                      "other_text": "", "zone": "gate", "movement": "approaching", "flags": ["touching_handle", "x"],
                      "people": 1, "vehicles": 0, "vehicle_moving": False, "animals": 0, "visibility": "clear",
                      "appearance": ["dark jacket", "grey cap"], "evidence_frame": 3, "raw_label": "suspicious"})
    s.suggest_client, s.suggest_frames = fake, _frames
    key = f"ev:{ids['consenting']}"
    r = client.post("/v1/tagging/suggest", headers=h, json={"key": key})
    assert r.status_code == 200, r.text
    got = r.json()
    f = got["fields"]
    assert got["model"] == "test/qwen-fake" and got["cached"] is False
    assert f["category"] == "S1" and f["raw_label"] == "suspicious" and f["zone"] == "gate"
    assert f["flags"] == ["touching_handle"] and f["appearance"] == ["dark jacket", "grey cap"]
    assert f["description"].startswith("A man walks") and f["evidence_frame"] == 34
    assert f["evidence_sec"] == pytest.approx(34 / 7, abs=1e-3)
    sent = fake.calls[0]
    assert sent["response_format"]["json_schema"]["strict"] is True and sent["temperature"] == 0
    assert sum(1 for part in sent["messages"][0]["content"] if part["type"] == "image_url") == 5
    assert "No special activity." in sent["messages"][0]["content"][0]["text"]
    again = client.post("/v1/tagging/suggest", headers=h, json={"key": key}).json()
    assert again["cached"] is True and len(fake.calls) == 1 and again["fields"] == f
    # the tagger saves it as suggested: the tag says so, and the export carries it
    tag = dict(f, suggested_by=got["model"], suggestion_use="accepted")
    r = client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": tag})
    assert r.status_code == 200, r.text
    saved = r.json()["tag"]["fields"]
    assert saved["suggestion_use"] == "accepted" and saved["appearance"] == f["appearance"]
    bad = client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {"suggestion_use": "maybe"}})
    assert bad.status_code == 422
    out = client.post("/v1/tagging/export", headers=h, json={}).json()
    line = next(x for x in open(out["training_path"], encoding="utf-8") if STEM in x)
    assert '"suggestion_use": "accepted"' in line and '"suggested_by": "test/qwen-fake"' in line
    with session_scope(client.app.state.engine) as session:
        assert session.scalars(select(m.AuditLog.action)).all().count("tag_suggested") == 1
    # a customer who withdrew consent: the model never sees the clip
    refused = client.post("/v1/tagging/suggest", headers=h, json={"key": f"ev:{ids['refusing']}"})
    assert refused.status_code == 403 and len(fake.calls) == 1


def test_suggest_follows_a_legacy_clip_schema_and_sends_the_model_input(client, staff_factory, studio):
    from home_guard_project.fleet_contract import prompt_schemas as ps
    from .test_model_view import video
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    ds = str(s.paths.dataset)
    meta = alert_meta("house2_ch2", STEM, label="suspicious", vlm_input="crop",
                      vlm_crop={"fps": 5.0, "vlm_crop_path": f"vlm_crops/house2_ch2/2026-10-04/{STEM}.mp4"})
    put_owner_clip(ds, "production_house2", meta, STEM)
    video(os.path.join(ds, "owner_feedback", "production_house2", "vlm_crops", "house2_ch2", "2026-10-04",
                       f"{STEM}.mp4"), 11)
    fake = FakeModel({"summary": "A man looks into the car.", "label": "suspicious", "raw_label": "suspicious",
                      "applied_fact_id": "", "serious_behaviour": True, "people": 1, "vehicle_moving": False,
                      "animals": 0, "why": "He looks into a car.", "summary_owner": ""})
    s.suggest_client = fake
    r = client.post("/v1/tagging/suggest", headers=h, json={"key": f"ev:{ids['consenting']}"})
    assert r.status_code == 200, r.text
    got = r.json()
    sent = fake.calls[0]
    assert tuple(sent["response_format"]["json_schema"]["schema"]["properties"]) == ps.field_order(ps.PROMPT_VERSION)
    # the frames the AI saw (rendered like the box from the saved crop: 3 at 1 fps), not 5 picked from the clip
    assert sum(1 for part in sent["messages"][0]["content"] if part["type"] == "image_url") == 3
    assert got["fields"]["why"] == "He looks into a car." and got["fields"]["serious_behaviour"] is True
    assert got["prompt_version"].startswith("2026-10-09.studio-suggest-legacy")


def test_suggest_says_why_it_cannot(client, staff_factory, studio, monkeypatch):
    s, _ = studio
    _, _, _, h = staff_factory("admin")
    from home_guard_project.cloud.tagstudio import suggest

    monkeypatch.setattr(suggest.SuggestConfig, "resolve", classmethod(lambda cls, **kw: cls(api_key="")))
    s.suggest_frames = _frames
    r = client.post("/v1/tagging/suggest", headers=h, json={"key": "ds:front_side_1771696865_trigger"})
    assert r.status_code == 409 and "OPENROUTER_API_KEY" in r.json()["detail"]
    with pytest.raises(suggest.SuggestError, match="Gemini"):
        suggest.Suggester(suggest.SuggestConfig(model="google/gemini-2.5-pro", api_key="k"), "x.jsonl")


def test_event_annotation_preloads_the_dataset_yolo_boxes_first(client, staff_factory, studio):
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    _yolo(str(s.paths.dataset), "house2_ch2", STEM, {f: [(0, 0.5, 0.5, 0.2, 0.4)] for f in range(0, 30, 5)})
    body = client.get(f"/v1/events/{ids['consenting']}/annotation", headers=h).json()
    assert [(t["label"], t["source"]) for t in body["tracks"]] == [("person", "yolo")]


def test_captions_use_the_boxs_camera_names_with_the_channel_rule_for_an_old_id(client, staff_factory, studio):
    """The house was renamed: the box's heartbeat lists newsite_ch2 as "חניה"; the clip still carries house2_ch2."""
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as session:
        dev = session.scalar(select(m.Device).where(m.Device.site == "house2"))
        dev.last_heartbeat = {"camera_list": [{"id": "newsite_ch2", "name": "חניה", "channel": 2, "enabled": True}]}
        fb = session.scalar(select(m.Feedback).where(m.Feedback.event_id == ids["consenting"]))
        fb.owner_label = "empty"
    key = f"ev:{ids['consenting']}"
    clip = client.get("/v1/tagging/clip", headers=h, params={"key": key}).json()
    assert clip["item"]["camera"] == "house2_ch2" and clip["item"]["camera_display"] == "חניה"
    row = next(r for r in client.get("/v1/tagging/queue?tier=all", headers=h).json()["items"] if r["key"] == key)
    assert row["camera_display"] == "חניה"
    assert [i["camera_name"] for i in client.get("/v1/inbox", headers=h).json()] == ["חניה"]
    client.post("/v1/tagging/tag", headers=h, json={"key": key, "fields": {"raw_label": "normal", "description": "x"}})
    out = client.post("/v1/tagging/export", headers=h, json={}).json()
    line = next(json.loads(x) for x in open(out["training_path"], encoding="utf-8") if STEM in x)
    assert line["camera_name"] == "house2_ch2" and line["camera_display"] == "חניה"   # raw id kept for traceability
    # a dataset clip with no house: the box's channel rule, never the raw id
    ds = client.get("/v1/tagging/clip", headers=h, params={"key": "ds:front_side_1771696865_trigger"}).json()
    assert ds["item"]["camera_display"] == "Front side"


# ---------------------------------------------------------------- Suggest regression (2026-10-09): the real request path

class GoneThenAnswers:
    """An OpenAI-shaped client with no fixed model (so the studio's configured model is asked first): the first model
    asked answers 404 like a deprecated OpenRouter model, the next one answers."""

    def __init__(self, answer, gone=1):
        self.answer, self.gone, self.calls = answer, gone, []
        self.chat = self.completions = self

    def create(self, **kwargs):
        import json
        from types import SimpleNamespace

        self.calls.append(kwargs)
        if len(self.calls) <= self.gone:
            err = RuntimeError("Error code: 404 - Qwen3 VL 32B Instruct was deprecated on Oct 9, 2026.")
            err.status_code = 404
            raise err
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.answer)))])


LEGACY_ANSWER = {"summary": "A man stands near a car.", "label": "normal", "raw_label": "normal", "applied_fact_id": "",
                 "serious_behaviour": False, "people": 1, "vehicle_moving": False, "animals": 0,
                 "why": "Ordinary activity.", "summary_owner": ""}


def _image_sizes(call):
    import base64

    import cv2
    import numpy as np
    out = []
    for part in call["messages"][0]["content"]:
        if part["type"] == "image_url":
            data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            out.append((img.shape[1], img.shape[0]))
    return out


def test_suggest_on_a_legacy_assumed_clip_through_the_route(client, staff_factory, studio):
    from home_guard_project.cloud.tagstudio import suggest as sg
    from home_guard_project.fleet_contract import prompt_schemas as ps
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    # the box's AI answered (alert + model_response) but recorded no prompt version: legacy assumed
    put_owner_clip(str(s.paths.dataset), "production_house2", alert_meta("house2_ch2", STEM, teacher=None), STEM)
    s.suggest_client, s.suggest_frames = GoneThenAnswers(LEGACY_ANSWER), _frames
    r = client.post("/v1/tagging/suggest", headers=h, json={"key": f"ev:{ids['consenting']}", "refresh": True})
    assert r.status_code == 200, r.text
    got = r.json()
    calls = s.suggest_client.calls
    assert calls[0]["model"] == sg.SuggestConfig.resolve().model and calls[1]["model"] == sg.FALLBACK_MODELS[0]
    assert got["model"] == sg.FALLBACK_MODELS[0]                         # the answer says who gave it
    assert tuple(calls[1]["response_format"]["json_schema"]["schema"]["properties"]) == ps.field_order(ps.PROMPT_VERSION)
    assert got["fields"]["why"] == "Ordinary activity." and got["prompt_version"].startswith("2026-10-09.studio-suggest-legacy")


def test_suggest_on_a_model_input_2_clip_sends_the_capped_frames(client, staff_factory, studio):
    from .test_model_view import video
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    ds = str(s.paths.dataset)
    meta = alert_meta("house2_ch2", STEM, vlm_input="crop",
                      vlm_crop={"fps": 5.0, "vlm_crop_path": f"vlm_crops/house2_ch2/2026-10-04/{STEM}.mp4"},
                      model_input={"version": "model-input-2", "vlm_input": "crop", "frame_indices": [0, 5],
                                   "times": [0.0, 1.0], "fps": 5.0, "sample_fps": 1.0, "step": 5,
                                   "size": [1024, 768], "max_side": 1024})
    put_owner_clip(ds, "production_house2", meta, STEM)
    video(os.path.join(ds, "owner_feedback", "production_house2", "vlm_crops", "house2_ch2", "2026-10-04",
                       f"{STEM}.mp4"), 6, size=(1600, 1200))
    s.suggest_client = GoneThenAnswers(LEGACY_ANSWER, gone=0)
    r = client.post("/v1/tagging/suggest", headers=h, json={"key": f"ev:{ids['consenting']}", "refresh": True})
    assert r.status_code == 200, r.text
    assert _image_sizes(s.suggest_client.calls[0]) == [(1024, 768), (1024, 768)]   # the box's own frames, capped


def test_suggest_says_plainly_when_no_model_is_offered(client, staff_factory, studio):
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    s.suggest_client, s.suggest_frames = GoneThenAnswers(LEGACY_ANSWER, gone=99), _frames
    r = client.post("/v1/tagging/suggest", headers=h, json={"key": f"ev:{ids['consenting']}", "refresh": True})
    assert r.status_code == 409
    assert "no longer offered" in r.json()["detail"] and "HG_SUGGEST_MODEL" in r.json()["detail"]
