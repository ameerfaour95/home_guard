"""Task 13: built-in filters, collections and versioned training exports."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud import studio
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, _event_id, _index, s3client  # noqa: F401

LABEL_0 = "0 0.5 0.5 0.2 0.4\n16 0.3 0.3 0.1 0.1\n4 0.1 0.1 0.1 0.1\n"
IDENTITY = ("front_side", "back_door", "left_side", "Acme", "dataset_", "production_", b.STEM, b.COLLECT_STEM)


COLLECT_FRAMES = [f["frame_index"] for f in b.fixture_json("collect.meta.json")["yolo_export"]["exported_frames"]]


def _frame_keys(frame):
    base = f"dataset_test/yolo/%s/back_door/2026-10-02/{b.COLLECT_STEM}_f{frame:04d}"
    return base % "images" + ".jpg", base % "labels" + ".txt"


def _seed(client, s3client, consent_training=True, all_frames=True, overrides=None):
    """The fixture bucket, indexed. With `all_frames`, every sampled frame the collection meta lists has its image
    and label (the fixtures carry only frames 0 and 2). `overrides`: key -> body, or None to delete the key."""
    b.seed_bucket(s3client)
    b.put(s3client, b.YOLO_LABELS[0], LABEL_0)
    if all_frames:
        for frame in COLLECT_FRAMES:
            image, label = _frame_keys(frame)
            if frame not in (0, 2):
                b.put(s3client, image, b.FAKE_JPG)
                b.put(s3client, label, b.LABEL_TEXT)
    for key, body in (overrides or {}).items():
        if body is None:
            s3client.delete_object(Bucket=b.BUCKET, Key=key)
        else:
            b.put(s3client, key, body)
    client.app.state.export_runner = lambda job: job()  # synchronous for tests
    return _index(client, s3client, consent_training=consent_training, seed=False)


@pytest.fixture()
def seeded(client, s3client):
    return _seed(client, s3client)


@pytest.fixture()
def seeded_no_consent(client, s3client):
    return _seed(client, s3client, consent_training=False)


def _ids(client):
    return [_event_id(client, s) for s in (b.STEM, b.PAUSED_STEM, b.COLLECT_STEM)]


def _make_collection(client, h, ids, name="night people"):
    r = client.post("/v1/studio/collections", headers=h, json={"name": name, "description": "d"})
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    r = client.post(f"/v1/studio/collections/{cid}/items", headers=h, json={"event_ids": ids})
    assert r.status_code == 200, r.text
    return cid


def _export(client, h, cid, name="people", formats=("yolo", "vlm_jsonl", "clips"), **kw):
    r = client.post("/v1/studio/exports", headers=h,
                    json={"collection_id": cid, "name": name, "formats": list(formats), **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _get(s3client, key):
    return s3client.get_object(Bucket=b.BUCKET, Key=key)["Body"].read()


def _keys(s3client, prefix):
    out = []
    for page in s3client.get_paginator("list_objects_v2").paginate(Bucket=b.BUCKET, Prefix=prefix):
        out += [o["Key"] for o in page.get("Contents", [])]
    return out


# ---------------------------------------------------------------- filters

def test_builtin_filters_listed_for_every_role(client, staff_factory):
    for role in ("admin", "support", "labeler"):
        _, _, _, h = staff_factory(role)
        r = client.get("/v1/studio/filters", headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert [f["key"] for f in body] == ["false_alarm", "ai_dismissed_person", "ai_failed", "real_but_wrong",
                                            "low_conf", "paused", "needs_labeling", "to_review", "rejected"]
        assert [f["title"] for f in body] == [
            "Owner said false alarm", "AI dismissed, YOLO saw a person", "AI call failed or fell back",
            "Owner said real but wrong", "Low detector confidence", "Paused-camera footage", "Needs labeling",
            "Labels to review", "Labels rejected"]
        assert all(f["builtin"] and f["query"] == {"filter": f["key"]} and f["description"] for f in body)
    assert client.get("/v1/studio/filters").status_code == 401


def test_builtin_filters_drive_events_route(client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    for f in studio.BUILTIN_FILTERS:
        assert client.get("/v1/events", params=f.query, headers=h).status_code == 200
    paused = client.get("/v1/events", params={"filter": "paused"}, headers=h).json()["items"]
    assert [i["id"] for i in paused] == [_event_id(client, b.PAUSED_STEM)]
    assert client.get("/v1/events", params={"filter": "nope"}, headers=h).status_code == 400


# ---------------------------------------------------------------- collections

def test_collections_create_list_add_remove_and_audit(client, staff_factory, seeded):
    staff, _, _, h = staff_factory("labeler")
    ids = _ids(client)
    cid = _make_collection(client, h, ids)
    cols = client.get("/v1/studio/collections", headers=h).json()
    assert [(c["id"], c["event_count"], c["created_by"]) for c in cols] == [(cid, 3, staff.name)]
    # adding again is idempotent
    r = client.post(f"/v1/studio/collections/{cid}/items", headers=h, json={"event_ids": ids[:1]})
    assert r.json()["event_count"] == 3
    r = client.request("DELETE", f"/v1/studio/collections/{cid}/items", headers=h, json={"event_ids": ids[:1]})
    assert r.status_code == 200 and r.json()["event_count"] == 2
    missing = client.post(f"/v1/studio/collections/{cid}/items", headers=h, json={"event_ids": [ids[0], 999999]})
    assert missing.status_code == 404
    assert client.get("/v1/studio/collections", headers=h).json()[0]["event_count"] == 2  # nothing half-added
    assert client.post("/v1/studio/collections/9999/items", headers=h,
                       json={"event_ids": ids}).status_code == 404
    with session_scope(client.app.state.engine) as s:
        actions = [a for a in s.scalars(select(m.AuditLog.action).where(m.AuditLog.staff_id == staff.id))]
    assert "collection_add" in actions and "collection_remove" in actions


def test_collections_forbidden_for_support(client, staff_factory, seeded):
    _, _, _, sup = staff_factory("support")
    assert client.get("/v1/studio/collections", headers=sup).status_code == 403
    assert client.post("/v1/studio/collections", headers=sup, json={"name": "x"}).status_code == 403


def test_labeler_cannot_add_events_they_cannot_see(client, staff_factory, seeded_no_consent):
    _, _, _, lab = staff_factory("labeler")
    _, _, _, adm = staff_factory("admin")
    cid = client.post("/v1/studio/collections", headers=lab, json={"name": "x"}).json()["id"]
    hidden = client.post(f"/v1/studio/collections/{cid}/items", headers=lab, json={"event_ids": _ids(client)[:1]})
    absent = client.post(f"/v1/studio/collections/{cid}/items", headers=lab, json={"event_ids": [999999]})
    assert hidden.status_code == absent.status_code == 404 and hidden.json() == absent.json()
    # an admin may collect them (the export still leaves out events without training consent)
    assert client.post(f"/v1/studio/collections/{cid}/items", headers=adm,
                       json={"event_ids": _ids(client)}).json()["event_count"] == 3
    assert client.get("/v1/studio/collections", headers=lab).json()[0]["event_count"] == 0
    gone = client.request("DELETE", f"/v1/studio/collections/{cid}/items", headers=lab,
                          json={"event_ids": _ids(client)[:1]})
    assert gone.status_code == 404 and gone.json() == absent.json()


# ---------------------------------------------------------------- exports

def test_export_all_formats(client, s3client, staff_factory, seeded):
    staff, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    cid = _make_collection(client, h, [alert, paused, collect])
    out = _export(client, h, cid)
    assert out["state"] == "ready", out
    assert out["version"] == 1 and out["s3_prefix"] == "training_exports/people/v1/" and out["item_count"] == 3
    assert out["created_by"] == staff.name and out["error"] is None
    prefix = out["s3_prefix"]

    manifest = json.loads(_get(s3client, prefix + "manifest.json"))
    assert manifest["schema_version"] == 2 and manifest["name"] == "people" and manifest["version"] == 1
    assert manifest["collection_id"] == cid and manifest["formats"] == ["yolo", "vlm_jsonl", "clips"]
    assert manifest["class_map"] == {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "bus",
                                     "5": "truck", "6": "bird", "7": "cat", "8": "dog"}
    items = {i["event_id"]: i for i in manifest["items"]}
    assert set(items) == {alert, paused, collect} and manifest["missing"] == []
    assert items[alert]["split"] == items[paused]["split"]  # same site + day
    assert items[alert]["group"] == items[paused]["group"] != items[collect]["group"]
    for item in items.values():
        assert item["customer"].startswith("customer-") and item["camera"].startswith("cam-")
        assert all(src.startswith("artifact-") for src in item["sources"]) and item["sources"]
        assert "site" not in item and "stem" not in item
    assert items[alert]["ai_status"] == "real" and items[paused]["ai_status"] == "none"
    assert items[alert]["clip_sha256"] == hashlib.sha256(b.FAKE_MP4).hexdigest()
    assert manifest["dropped_boxes"] == 1
    text = json.dumps(manifest)
    assert not [t for t in IDENTITY if t in text]

    # YOLO: images and remapped labels under the event id, data.yaml in contiguous order
    split = items[collect]["split"]
    label = _get(s3client, f"{prefix}yolo/labels/{split}/{collect}_f0000.txt").decode()
    assert [line.split()[0] for line in label.splitlines()] == ["0", "8"]
    assert label.splitlines()[1].split()[1:] == ["0.3", "0.3", "0.1", "0.1"]
    assert _get(s3client, f"{prefix}yolo/images/{split}/{collect}_f0000.jpg") == b.FAKE_JPG
    assert _get(s3client, f"{prefix}yolo/labels/{split}/{collect}_f0002.txt").decode() == b.LABEL_TEXT
    yaml = _get(s3client, prefix + "yolo/data.yaml").decode()
    assert "path: .\n" in yaml and "train: images/train\n" in yaml
    for sp in ("val", "test"):  # a YOLO split without images points at train (and the manifest warns)
        assert f"{sp}: images/{sp if sp == split else 'train'}\n" in yaml
    names = [line.split(":", 1)[1].strip() for line in yaml.split("names:", 1)[1].splitlines() if line.strip()]
    assert names == ["person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog"]

    # clips: one per event, by event id
    for ev, item in items.items():
        assert _get(s3client, f"{prefix}clips/{item['split']}/{ev}.mp4") == b.FAKE_MP4

    # vlm/<split>.jsonl: only the real AI answer, identity redacted from the prompt
    whole = "".join(_get(s3client, f"{prefix}vlm/{sp}.jsonl").decode() for sp in ("train", "val", "test"))
    lines = [json.loads(x) for x in whole.splitlines()]
    assert [x["event_id"] for x in lines] == [alert]
    line = lines[0]
    assert line["ai_status"] == "real" and line["videos"] == [f"clips/{items[alert]['split']}/{alert}.mp4"]
    user, assistant = line["messages"]
    assert user["role"] == "user" and user["content"].startswith("<video>")
    assert "front_side" not in user["content"] and "You are the eyes" in user["content"]
    assert line["prompt_redacted"] is True
    assert assistant["role"] == "assistant" and json.loads(assistant["content"])["summary"].startswith("A person")
    assert not [t for t in IDENTITY if t in whole]

    # the private mapping keeps provenance for admins
    private = json.loads(_get(s3client, prefix + "_private/mapping.json"))
    assert private[str(collect)]["stem"] == b.COLLECT_STEM and private[str(collect)]["clip_key"] == b.COLLECT_CLIP
    assert private[str(collect)]["clip_source"] == "original"

    got = client.get(f"/v1/studio/exports/{out['id']}", headers=h).json()
    assert got["state"] == "ready" and "manifest.json" in got["manifest_url"] and "_private" not in got["manifest_url"]
    assert [e["id"] for e in client.get("/v1/studio/exports", headers=h).json()] == [out["id"]]
    with session_scope(client.app.state.engine) as s:
        assert s.scalar(select(m.AuditLog).where(m.AuditLog.action == "export_create")) is not None


def test_export_missing_clip_is_partial(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    cid = _make_collection(client, h, [alert, paused, collect])
    s3client.delete_object(Bucket=b.BUCKET, Key=b.COLLECT_CLIP)
    out = _export(client, h, cid)
    assert out["state"] == "partial"
    manifest = json.loads(_get(s3client, out["s3_prefix"] + "manifest.json"))
    assert {"event_id": collect, "reason": "clip_missing"} in manifest["missing"]
    assert len(manifest["items"]) == 3


def test_export_versions_never_overwrite(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("labeler")
    cid = _make_collection(client, h, _ids(client))
    first = _export(client, h, cid, formats=["clips"])
    before = {k: s3client.head_object(Bucket=b.BUCKET, Key=k)["ETag"] for k in _keys(s3client, first["s3_prefix"])}
    second = _export(client, h, cid, formats=["clips"])
    assert (first["version"], second["version"]) == (1, 2)
    assert second["s3_prefix"] == "training_exports/people/v2/" and second["state"] == "ready"
    after = {k: s3client.head_object(Bucket=b.BUCKET, Key=k)["ETag"] for k in _keys(s3client, first["s3_prefix"])}
    assert before == after
    # a version prefix already used in S3 (but unknown to the database) is skipped too
    b.put(s3client, "training_exports/other/v1/manifest.json", "{}")
    assert _export(client, h, cid, name="other", formats=["clips"])["version"] == 2


def test_exports_forbidden_for_support(client, staff_factory, seeded):
    _, _, _, adm = staff_factory("admin")
    _, _, _, sup = staff_factory("support")
    cid = _make_collection(client, adm, _ids(client))
    assert client.post("/v1/studio/exports", headers=sup,
                       json={"collection_id": cid, "name": "x", "formats": ["yolo"]}).status_code == 403
    assert client.get("/v1/studio/exports", headers=sup).status_code == 403
    assert client.post("/v1/studio/exports", headers=adm,
                       json={"collection_id": 9999, "name": "x", "formats": ["yolo"]}).status_code == 404
    assert client.get("/v1/studio/exports/9999", headers=adm).status_code == 404


def test_fallback_ai_only_leaves_vlm(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    with session_scope(client.app.state.engine) as s:
        s.scalar(select(m.AiRun).where(m.AiRun.event_id == alert)).status = "fallback"
    cid = _make_collection(client, h, [alert, collect])
    out = _export(client, h, cid)
    assert out["item_count"] == 2
    assert all(_get(s3client, f"{out['s3_prefix']}vlm/{sp}.jsonl") == b"" for sp in ("train", "val", "test"))
    assert _keys(s3client, out["s3_prefix"] + "clips/")  # its clip is still exported
    out = _export(client, h, cid, include_fallback_ai=True)
    whole = "".join(_get(s3client, f"{out['s3_prefix']}vlm/{sp}.jsonl").decode() for sp in ("train", "val", "test"))
    lines = [json.loads(x) for x in whole.splitlines()]
    assert [(x["event_id"], x["ai_status"]) for x in lines] == [(alert, "fallback")]


def test_labeler_export_leaves_out_unseen_events(client, s3client, staff_factory, seeded_no_consent):
    _, _, _, adm = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    cid = _make_collection(client, adm, _ids(client))
    out = _export(client, lab, cid, formats=["clips"])
    manifest = json.loads(_get(s3client, out["s3_prefix"] + "manifest.json"))
    assert manifest["items"] == [] and manifest["excluded"] == [] and out["item_count"] == 0
    out = _export(client, adm, cid, formats=["clips"])
    manifest = json.loads(_get(s3client, out["s3_prefix"] + "manifest.json"))
    # households without consent are only counted in the manifest; the event ids stay in _private/
    assert manifest["excluded"] == [] and manifest["excluded_counts"] == {"no_training_consent": 3}


def test_export_matches_preview(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("labeler")
    cid = _make_collection(client, h, _ids(client))
    req = {"collection_id": cid, "name": "same", "formats": ["clips"]}
    preview = client.post("/v1/studio/exports/preview", headers=h, json=req).json()
    out = _export(client, h, cid, name="same", formats=["clips"])
    manifest = json.loads(_get(s3client, out["s3_prefix"] + "manifest.json"))
    counts = {}
    for item in manifest["items"]:
        counts[item["split"]] = counts.get(item["split"], 0) + 1
    assert sorted(i["event_id"] for i in manifest["items"]) == preview["included_ids"]
    assert {k: v for k, v in preview["split_counts"].items() if v} == counts


def test_crash_marks_failed_without_paths(client, s3client, staff_factory, seeded, monkeypatch):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))

    def boom(*a, **k):
        raise RuntimeError(r"disk broke at C:\Users\someone\secret\file.py and /home/x/y.py")

    monkeypatch.setattr(studio, "_export_clip", boom)
    out = _export(client, h, cid, formats=["clips"])
    assert out["state"] == "failed" and out["manifest_url"] is None
    assert "RuntimeError" in out["error"] and "Users" not in out["error"] and "/home" not in out["error"]


def test_build_export_runs_synchronously_and_once(client, s3client, staff_factory, seeded):
    staff, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    client.app.state.export_runner = lambda job: None  # never started by the route
    out = _export(client, h, cid, formats=["clips"])
    assert out["state"] == "queued"
    with session_scope(client.app.state.engine) as s:
        export = studio.build_export(s, client.app.state.s3, out["id"], secret=client.app.state.settings.jwt_secret)
        assert export.state == "ready" and export.item_count == 3
        again = studio.build_export(s, client.app.state.s3, out["id"], secret=client.app.state.settings.jwt_secret)
        assert again.state == "ready"  # a finished export is never rebuilt


def test_sweep_marks_stale_exports_failed(client, staff_factory, seeded):
    staff, _, _, _ = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        for name, state, age in (("old", "running", 2), ("young", "running", 0), ("oldq", "queued", 2),
                                 ("done", "ready", 5)):
            s.add(m.Export(name=name, version=1, state=state, created_by=staff.id,
                           created_at=NOW - timedelta(hours=age, minutes=1), s3_prefix=f"training_exports/{name}/v1/"))
    with session_scope(client.app.state.engine) as s:
        assert studio.sweep_stale_exports(s, NOW) == 2
        states = dict(s.execute(select(m.Export.name, m.Export.state)).all())
        assert states == {"old": "failed", "young": "running", "oldq": "failed", "done": "ready"}
        assert s.scalar(select(m.Export.error).where(m.Export.name == "old"))


def test_default_runner_builds_on_a_thread(client, s3client, staff_factory, seeded):
    import time

    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    client.app.state.export_runner = None  # the app's bounded executor
    out = _export(client, h, cid, formats=["clips"])
    assert out["state"] in ("queued", "running", "ready")
    deadline = time.monotonic() + 30
    while client.get(f"/v1/studio/exports/{out['id']}", headers=h).json()["state"] in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert client.get(f"/v1/studio/exports/{out['id']}", headers=h).json()["state"] == "ready"


def test_clip_rendition_used_only_when_original_is_not_h264(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, _, collect = _ids(client)
    rendered = b"\x00\x00\x00\x18ftypisom-rendition" + b"\x01" * 32
    with session_scope(client.app.state.engine) as s:
        for ev in (alert, collect):
            key = f"admin_cache/renditions/{ev}.mp4"
            b.put(s3client, key, rendered)
            original = s.scalar(select(m.Artifact.etag).where(  # made from the very original that is exported
                m.Artifact.event_id == ev, m.Artifact.role == "original_video",
                m.Artifact.s3_key.startswith("dataset_")))
            s.add(m.Artifact(event_id=ev, role="rendition", s3_key=key, available=True, provenance="cloud",
                             etag=s3client.head_object(Bucket=b.BUCKET, Key=key)["ETag"].strip('"'),
                             detail={"src_etag": original}))
    cid = _make_collection(client, h, [alert, collect])
    out = _export(client, h, cid, formats=["clips"])
    manifest = json.loads(_get(s3client, out["s3_prefix"] + "manifest.json"))
    split = {i["event_id"]: i["split"] for i in manifest["items"]}
    # the collection clip's meta names no codec (old recorder, mp4v): the H.264 rendition is exported
    assert _get(s3client, f"{out['s3_prefix']}clips/{split[collect]}/{collect}.mp4") == rendered
    # the alert clip is H.264 already: the original is exported
    assert _get(s3client, f"{out['s3_prefix']}clips/{split[alert]}/{alert}.mp4") == b.FAKE_MP4
    private = json.loads(_get(s3client, out["s3_prefix"] + "_private/mapping.json"))
    assert private[str(collect)]["clip_source"] == "rendition" and private[str(alert)]["clip_source"] == "original"
    assert "rendition" not in json.dumps(manifest)


def test_app_startup_sweeps_stale_exports(client, staff_factory):
    from fastapi.testclient import TestClient

    staff, _, _, _ = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        s.add(m.Export(name="stuck", version=1, state="running", created_by=staff.id,
                       created_at=datetime.now(timezone.utc) - timedelta(hours=3), s3_prefix="training_exports/stuck/v1/"))
    assert client.app.state.export_runner is None  # the bounded executor runs real jobs
    with TestClient(client.app):  # a restart
        pass
    with session_scope(client.app.state.engine) as s:
        assert s.scalar(select(m.Export.state).where(m.Export.name == "stuck")) == "failed"


def test_vlm_prompt_falls_back_when_redaction_leaves_identity():
    leaky = SimpleNamespace(text=lambda v: v, mentions=lambda v: True)
    assert studio.vlm_prompt(leaky, "Camera front_side at night") == (studio.PLACEHOLDER_PROMPT, True)
    clean = SimpleNamespace(text=lambda v: v.replace("front_side", "cam-1"), mentions=lambda v: "front_side" in v)
    assert studio.vlm_prompt(clean, "Camera front_side") == ("<video>Camera cam-1", True)
    assert studio.vlm_prompt(clean, "Camera") == ("<video>Camera", False)
    assert studio.vlm_prompt(clean, None) == (studio.PLACEHOLDER_PROMPT, False)
    assert studio.PLACEHOLDER_PROMPT == "<video>Describe what happens in this security camera clip."


def test_remap_label():
    text, dropped, problem = studio.remap_label("0 0.5 0.5 0.2 0.4\n16 0.1 0.1 0.1 0.1\n4 0.1 0.1 0.1 0.1\n\n"
                                                "2.0 1 1 1 1\n")
    assert text == "0 0.5 0.5 0.2 0.4\n8 0.1 0.1 0.1 0.1\n2 1 1 1 1\n" and dropped == 1 and problem is None
