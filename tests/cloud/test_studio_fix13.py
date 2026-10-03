"""Task 13 fix round (Codex review): privacy, trainable bundles, honest provenance and crash recovery."""
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, update

from home_guard_project.cloud import models as m
from home_guard_project.cloud import studio
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, s3client  # noqa: F401
from .test_studio import (COLLECT_FRAMES, IDENTITY, _export, _frame_keys, _get, _ids, _keys, _make_collection,
                          _seed)


@pytest.fixture()
def seeded(client, s3client):
    return _seed(client, s3client)


def _manifest(s3client, out):
    return json.loads(_get(s3client, out["s3_prefix"] + "manifest.json"))


def _set_consent(client, value):
    with session_scope(client.app.state.engine) as s:
        for c in s.scalars(select(m.Customer)):
            c.consent_training = value


def _secret(client):
    return client.app.state.settings.jwt_secret


def _queued(client, h, cid, **kw):
    """Create an export without running it."""
    client.app.state.export_runner = lambda job: None
    out = _export(client, h, cid, **kw)
    client.app.state.export_runner = lambda job: job()
    assert out["state"] == "queued"
    return out


def _build(client, export_id):
    with session_scope(client.app.state.engine) as s:
        export = studio.build_export(s, client.app.state.s3, export_id, secret=_secret(client))
        return export.state


# ---------------------------------------------------------------- C1 / M1: names and splits

def test_export_and_collection_names_must_not_identify_a_household(client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:  # another customer: names of every household count
        b.enroll(s, site="bian_house", customer_name="Daniel Levi")
    cid = _make_collection(client, h, _ids(client))
    for name in ("acme-night", "test-people", "bian-v1", "levi_set", "front_side-people"):
        req = {"collection_id": cid, "name": name, "formats": ["clips"]}
        pre = client.post("/v1/studio/exports/preview", headers=h, json=req)
        made = client.post("/v1/studio/exports", headers=h, json=req)
        assert pre.status_code == made.status_code == 400, (name, pre.text, made.text)
        assert pre.json()["detail"] == made.json()["detail"] == "Name must not identify a household"
    for body in ({"name": "Acme people"}, {"name": "people", "description": "night shots at Bian House"}):
        r = client.post("/v1/studio/collections", headers=h, json=body)
        assert r.status_code == 400 and r.json()["detail"] == "Name must not identify a household"
    with session_scope(client.app.state.engine) as s:
        assert s.scalar(select(m.Export.id)) is None


@pytest.mark.parametrize("split", [
    {"train": 0.8, "holdout": 0.2},
    {"train": 0.0, "val": 0.0, "test": 0.0},
    {"train": 0.5, "val": 0.2},
    {"val": 1.0},
    {"train": 1.2, "val": -0.2},
])
def test_split_validated_identically_by_preview_and_export(client, staff_factory, seeded, split):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    req = {"collection_id": cid, "name": "people", "formats": ["clips"], "split": split}
    pre = client.post("/v1/studio/exports/preview", headers=h, json=req)
    made = client.post("/v1/studio/exports", headers=h, json=req)
    assert pre.status_code == made.status_code == 400, (pre.text, made.text)
    assert pre.json() == made.json()


def test_valid_split_accepted(client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    req = {"collection_id": cid, "name": "people", "formats": ["clips"], "split": {"train": 0.7, "val": 0.3}}
    assert client.post("/v1/studio/exports/preview", headers=h, json=req).status_code == 200


# ---------------------------------------------------------------- I4: who sees an export

def test_labelers_see_only_their_own_exports_of_visible_events(client, s3client, staff_factory, seeded):
    _, _, _, adm = staff_factory("admin")
    _, _, _, lab_a = staff_factory("labeler")
    _, _, _, lab_b = staff_factory("labeler")
    cid = _make_collection(client, adm, _ids(client))
    admin_export = _export(client, adm, cid, name="adm", formats=["clips"])
    mine = _export(client, lab_a, cid, name="mine", formats=["clips"])
    assert [e["id"] for e in client.get("/v1/studio/exports", headers=lab_a).json()] == [mine["id"]]
    assert client.get("/v1/studio/exports", headers=lab_b).json() == []
    absent = client.get("/v1/studio/exports/999999", headers=lab_b)
    for export_id in (mine["id"], admin_export["id"]):
        hidden = client.get(f"/v1/studio/exports/{export_id}", headers=lab_b)
        assert hidden.status_code == absent.status_code == 404 and hidden.json() == absent.json()
    assert client.get(f"/v1/studio/exports/{admin_export['id']}", headers=lab_a).status_code == 404
    assert client.get(f"/v1/studio/exports/{mine['id']}", headers=lab_a).json()["manifest_url"]

    _set_consent(client, False)  # the household withdraws training consent
    assert client.get("/v1/studio/exports", headers=lab_a).json() == []
    gone = client.get(f"/v1/studio/exports/{mine['id']}", headers=lab_a)
    assert gone.status_code == 404 and gone.json() == absent.json()
    listed = {e["id"]: e for e in client.get("/v1/studio/exports", headers=adm).json()}
    assert set(listed) == {mine["id"], admin_export["id"]}
    for e in listed.values():
        assert e["error"].startswith("warning: ") and "consent" in e["error"]
    _set_consent(client, True)
    assert client.get(f"/v1/studio/exports/{mine['id']}", headers=adm).json()["error"] is None


def test_manifest_never_lists_exclusions_of_non_consenting_customers(client, s3client, staff_factory):
    _seed(client, s3client, consent_training=False)
    _, _, _, adm = staff_factory("admin")
    ids = _ids(client)
    cid = _make_collection(client, adm, ids)
    out = _export(client, adm, cid, formats=["clips"])
    manifest = _manifest(s3client, out)
    assert manifest["excluded"] == [] and manifest["excluded_counts"] == {"no_training_consent": 3}
    assert not [i for i in ids if f'"event_id": {i}' in json.dumps(manifest)]
    private = json.loads(_get(s3client, out["s3_prefix"] + "_private/excluded.json"))
    assert sorted(e["event_id"] for e in private) == sorted(ids)


# ---------------------------------------------------------------- I1: VLM lines always have their clips

def test_vlm_implies_clips_in_preview_and_export(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, _, collect = _ids(client)
    cid = _make_collection(client, h, [alert, collect])
    pre = client.post("/v1/studio/exports/preview", headers=h,
                      json={"collection_id": cid, "name": "vlm", "formats": ["vlm_jsonl"]}).json()
    assert any("clips" in w for w in pre["warnings"])
    out = _export(client, h, cid, name="vlm", formats=["vlm_jsonl"])
    assert out["state"] == "ready", out
    manifest = _manifest(s3client, out)
    assert "clips" in manifest["formats"] and manifest["requested_formats"] == ["vlm_jsonl"]
    lines = [json.loads(x) for split in ("train", "val", "test")
             for x in _get(s3client, f"{out['s3_prefix']}vlm/{split}.jsonl").decode().splitlines()]
    assert [x["event_id"] for x in lines] == [alert]
    for line in lines:
        assert _get(s3client, out["s3_prefix"] + line["videos"][0]) == b.FAKE_MP4


def test_vlm_line_dropped_when_its_clip_is_not_in_the_export(client, s3client, staff_factory, seeded,
                                                            monkeypatch):
    _, _, _, h = staff_factory("admin")
    alert, _, _ = _ids(client)
    cid = _make_collection(client, h, [alert])
    monkeypatch.setattr(studio, "_export_clip", lambda s3, src, dest, etag: None)  # "copied", but nothing landed
    out = _export(client, h, cid, name="vlm", formats=["vlm_jsonl"])
    manifest = _manifest(s3client, out)
    assert out["state"] == "partial"
    assert {"event_id": alert, "reason": "clip_not_in_export"} in manifest["missing"]
    assert all(not _get(s3client, f"{out['s3_prefix']}vlm/{sp}.jsonl") for sp in ("train", "val", "test"))


# ---------------------------------------------------------------- I2: an absolute YOLO root after download

def test_export_download_makes_data_yaml_absolute(client, s3client, staff_factory, seeded, monkeypatch, tmp_path,
                                                  capsys):
    from home_guard_project.cloud import manage
    from home_guard_project.cloud.s3 import S3

    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    out = _export(client, h, cid)
    readme = _get(s3client, out["s3_prefix"] + "README.txt").decode()
    assert "yolo detect train data=" in readme and "llamafactory-cli train" in readme
    assert "export-download" in readme and "media_dir" in readme
    monkeypatch.setenv("HG_CLOUD_DB_URL", client.app.state.engine.url.render_as_string(hide_password=False))
    monkeypatch.setattr(manage, "_s3", lambda: S3(s3client, b.BUCKET))
    dest = tmp_path / "people_v1"
    assert manage.main(["export-download", str(out["id"]), "--dest", str(dest)]) == 0
    yaml = (dest / "yolo" / "data.yaml").read_text()
    root = [line for line in yaml.splitlines() if line.startswith("path:")][0].split(":", 1)[1].strip()
    assert Path(root).is_absolute() and Path(root) == (dest / "yolo").resolve()
    assert (dest / "manifest.json").exists() and (dest / "vlm" / "dataset_info.json").exists()
    assert not (dest / "_private").exists()
    assert str(dest.resolve()) in (dest / "README.txt").read_text()
    assert manage.main(["export-download", "999999", "--dest", str(tmp_path / "x")]) == 1


# ---------------------------------------------------------------- I3: label validation

def test_remap_label_validates_rows():
    ok = studio.remap_label("0 0.5 0.5 0.2 0.4\n16 0.3 0.3 0.1 0.1\n4 0.1 0.1 0.1 0.1\n")
    assert ok == ("0 0.5 0.5 0.2 0.4\n8 0.3 0.3 0.1 0.1\n", 1, None)
    assert studio.remap_label("") == ("", 0, None)  # an empty source file is a real negative
    assert studio.remap_label("0 1.0000001 0.5 0.2 0.2\n")[0] == "0 1 0.5 0.2 0.2\n"  # tiny overshoot clipped
    for bad in ("0 1.2 0.5 -0.2 0.4", "0 0.5 0.5 0.2", "0 0.5 0.5 0.2 0.4 0.9", "0 nan 0.5 0.2 0.2",
                "0 0.5 0.5 0 0.2", "0 0.5 0.5 0.2 inf", "x 0.5 0.5 0.2 0.2", "0.5 0.5 0.5 0.2 0.2", "bad"):
        text, _, problem = studio.remap_label("0 0.5 0.5 0.2 0.4\n" + bad + "\n")
        assert text == "" and problem == "invalid_row", bad
    text, dropped, problem = studio.remap_label("4 0.1 0.1 0.1 0.1\n")
    assert text == "" and dropped == 1 and problem == "only_unmapped_classes"


def test_invalid_label_quarantines_the_sample(client, s3client, staff_factory):
    bad_label = _frame_keys(4)[1]
    _seed(client, s3client, overrides={bad_label: "0 1.2 0.5 -0.2 0.4\n"})
    _, _, _, h = staff_factory("admin")
    _, _, collect = _ids(client)
    cid = _make_collection(client, h, [collect])
    out = _export(client, h, cid, formats=["yolo"])
    manifest = _manifest(s3client, out)
    assert manifest["quarantined_samples"]["count"] == 1
    assert manifest["quarantined_samples"]["samples"] == [{"event_id": collect, "frame": 4, "reason": "invalid_row"}]
    assert not [k for k in _keys(s3client, out["s3_prefix"]) if k.endswith(f"{collect}_f0004.txt")
                or k.endswith(f"{collect}_f0004.jpg")]
    assert manifest["dropped_boxes"] == 1  # the unmapped class 4 row of frame 0


# ---------------------------------------------------------------- I5: renditions only of the selected original

def _add_rendition(client, s3client, event_id, src_etag):
    key = f"admin_cache/renditions/{event_id}.mp4"
    body = b"\x00\x00\x00\x18ftypisom-rendition" + b"\x01" * 32
    b.put(s3client, key, body)
    etag = s3client.head_object(Bucket=b.BUCKET, Key=key)["ETag"].strip('"')
    with session_scope(client.app.state.engine) as s:
        s.add(m.Artifact(event_id=event_id, role="rendition", s3_key=key, etag=etag, available=True,
                         provenance="cloud", detail={"src_etag": src_etag}))
    return body


def _original_etag(client, event_id):
    with session_scope(client.app.state.engine) as s:
        return s.scalar(select(m.Artifact.etag).where(m.Artifact.event_id == event_id,
                                                      m.Artifact.role == "original_video",
                                                      m.Artifact.s3_key.startswith("dataset_")))


def test_stale_rendition_is_never_exported(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    _, paused, collect = _ids(client)
    _add_rendition(client, s3client, collect, "an-older-clip")
    cid = _make_collection(client, h, [collect])
    out = _export(client, h, cid, formats=["clips"])
    split = _manifest(s3client, out)["items"][0]["split"]
    assert _get(s3client, f"{out['s3_prefix']}clips/{split}/{collect}.mp4") == b.FAKE_MP4
    private = json.loads(_get(s3client, out["s3_prefix"] + "_private/mapping.json"))
    assert private[str(collect)]["clip_source"] == "original"


# ---------------------------------------------------------------- I6: every expected sampled frame accounted for

def test_missing_sampled_frames_make_the_export_partial(client, s3client, staff_factory):
    _seed(client, s3client, all_frames=False)  # the meta lists 32 frames; only 0 and 2 were uploaded
    _, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    cid = _make_collection(client, h, [alert, paused, collect])
    out = _export(client, h, cid, formats=["yolo"])
    assert out["state"] == "partial"
    manifest = _manifest(s3client, out)
    missing = {(x["event_id"], x["frame"], x["reason"]) for x in manifest["missing"]}
    expected = {(collect, f, r) for f in COLLECT_FRAMES if f not in (0, 2)
                for r in ("yolo_image_missing", "yolo_label_missing")}
    assert missing == expected
    assert manifest["no_weak_labels"] == sorted([alert, paused])


def test_unavailable_images_are_missing_not_silently_skipped(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    _, _, collect = _ids(client)
    with session_scope(client.app.state.engine) as s:
        s.execute(update(m.Artifact).where(m.Artifact.role == "yolo_image").values(available=False))
    cid = _make_collection(client, h, [collect])
    out = _export(client, h, cid, formats=["yolo"])
    manifest = _manifest(s3client, out)
    assert out["state"] == "partial"
    assert {x["frame"] for x in manifest["missing"] if x["reason"] == "yolo_image_missing"} == set(COLLECT_FRAMES)


# ---------------------------------------------------------------- I7: the export is the snapshot taken at creation

def test_export_builds_the_snapshot_taken_at_creation(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    cid = _make_collection(client, h, [alert, collect])
    out = _queued(client, h, cid, formats=["clips"])
    # after the request: the collection grows and a source clip is replaced
    client.post(f"/v1/studio/collections/{cid}/items", headers=h, json={"event_ids": [paused]})
    b.put(s3client, b.COLLECT_CLIP, b.FAKE_MP4 + b"changed")
    assert _build(client, out["id"]) == "partial"
    manifest = _manifest(s3client, out)
    assert sorted(i["event_id"] for i in manifest["items"]) == sorted([alert, collect])
    assert {"event_id": collect, "reason": "changed_since_request"} in manifest["missing"]
    assert not [k for k in _keys(s3client, out["s3_prefix"]) if k.endswith(f"/{collect}.mp4")]


def test_consent_withdrawn_before_build_drops_the_events(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    out = _queued(client, h, cid, formats=["clips"])
    _set_consent(client, False)
    _build(client, out["id"])
    manifest = _manifest(s3client, out)
    assert manifest["items"] == [] and manifest["excluded_counts"] == {"no_training_consent": 3}
    assert not _keys(s3client, out["s3_prefix"] + "clips/")


def test_consent_withdrawn_during_build_fails_and_removes_files(client, s3client, staff_factory, seeded,
                                                                monkeypatch):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    real = studio._export_clip

    def withdraw_then_copy(s3, src, dest, etag):
        _set_consent(client, False)
        return real(s3, src, dest, etag)

    monkeypatch.setattr(studio, "_export_clip", withdraw_then_copy)
    out = _export(client, h, cid, formats=["clips"])
    assert out["state"] == "failed" and out["error"] == studio.CONSENT_WITHDRAWN_ERROR
    assert _keys(s3client, out["s3_prefix"]) == []


# ---------------------------------------------------------------- I8 / I9: leases, heartbeats, sweeps

def test_claim_is_atomic_and_exclusive(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    out = _queued(client, h, cid, formats=["clips"])
    with session_scope(client.app.state.engine) as s:
        assert studio.claim_export(s, out["id"], "worker-a") is True
    with session_scope(client.app.state.engine) as s:
        assert studio.claim_export(s, out["id"], "worker-b") is False
        export = studio.build_export(s, client.app.state.s3, out["id"], secret=_secret(client))
        assert (export.state, export.worker_id) == ("running", "worker-a")
        assert export.heartbeat_at is not None
    assert _keys(s3client, out["s3_prefix"]) == []  # the second worker wrote nothing


def test_build_records_its_worker_and_heartbeat(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    out = _export(client, h, cid, formats=["clips"])
    with session_scope(client.app.state.engine) as s:
        export = s.get(m.Export, out["id"])
        assert export.state == "ready" and export.worker_id and export.heartbeat_at is not None


def test_a_build_that_lost_its_lease_never_publishes(client, s3client, staff_factory, seeded, monkeypatch):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    real = studio._export_clip
    calls = []

    def swept_meanwhile(s3, src, dest, etag):
        calls.append(dest)
        with session_scope(client.app.state.engine) as s:
            s.execute(update(m.Export).values(state="failed", error=studio.STALE_ERROR))
        return real(s3, src, dest, etag)

    monkeypatch.setattr(studio, "_export_clip", swept_meanwhile)
    out = _export(client, h, cid, formats=["clips"])
    assert calls and out["state"] == "failed" and out["error"] == studio.STALE_ERROR
    assert out["s3_prefix"] + "manifest.json" not in _keys(s3client, out["s3_prefix"])


def test_sweep_uses_heartbeats_not_creation_time(client, staff_factory, seeded):
    staff, _, _, _ = staff_factory("admin")
    rows = (  # name, state, created minutes ago, heartbeat minutes ago
        ("long_healthy", "running", 180, 1),
        ("dead_worker", "running", 20, 6),
        ("never_beat", "running", 10, None),
        ("fresh_start", "running", 2, None),
        ("old_queue", "queued", 31, None),
        ("young_queue", "queued", 10, None),
        ("done", "ready", 600, 300),
    )
    with session_scope(client.app.state.engine) as s:
        for name, state, created, beat in rows:
            s.add(m.Export(name=name, version=1, state=state, created_by=staff.id,
                           created_at=NOW - timedelta(minutes=created),
                           heartbeat_at=None if beat is None else NOW - timedelta(minutes=beat),
                           s3_prefix=f"training_exports/{name}/v1/"))
    with session_scope(client.app.state.engine) as s:
        assert studio.sweep_stale_exports(s, NOW) == 3
        states = dict(s.execute(select(m.Export.name, m.Export.state)).all())
    assert states == {"long_healthy": "running", "dead_worker": "failed", "never_beat": "failed",
                      "fresh_start": "running", "old_queue": "failed", "young_queue": "queued", "done": "ready"}


# ---------------------------------------------------------------- I10 / I11: per-split VLM files, counts, warnings

def test_vlm_files_per_split_with_llamafactory_registration(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    cid = _make_collection(client, h, [alert, paused, collect])
    out = _export(client, h, cid)
    prefix = out["s3_prefix"]
    assert prefix + "vlm.jsonl" not in _keys(s3client, prefix)
    manifest = _manifest(s3client, out)
    split = {i["event_id"]: i["split"] for i in manifest["items"]}
    for sp in ("train", "val", "test"):
        for line in _get(s3client, f"{prefix}vlm/{sp}.jsonl").decode().splitlines():
            assert json.loads(line)["split"] == sp
    lines = _get(s3client, f"{prefix}vlm/{split[alert]}.jsonl").decode().splitlines()
    assert [json.loads(x)["event_id"] for x in lines] == [alert]
    info = json.loads(_get(s3client, prefix + "vlm/dataset_info.json"))
    entry = info[f"homeguard_{split[alert]}"]
    assert entry == {"file_name": f"{split[alert]}.jsonl", "formatting": "sharegpt",
                     "columns": {"messages": "messages", "videos": "videos"},
                     "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                              "assistant_tag": "assistant"}}
    assert manifest["vlm"] == {"jsonl_dir": "vlm", "media_dir": ".", "datasets": sorted(info)}


def test_counts_per_format_and_split_and_small_set_warnings(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    cid = _make_collection(client, h, [alert, paused, collect])
    out = _export(client, h, cid)
    manifest = _manifest(s3client, out)
    split = {i["event_id"]: i["split"] for i in manifest["items"]}
    counts = manifest["counts"]
    assert counts["yolo"][split[collect]] == {"images": len(COLLECT_FRAMES), "boxes": len(COLLECT_FRAMES) + 1}
    assert counts["vlm"][split[alert]] == 1 and sum(counts["vlm"].values()) == 1
    assert sum(counts["clips"].values()) == 3
    assert any("groups" in w for w in manifest["warnings"])  # two site+day groups
    yaml = _get(s3client, out["s3_prefix"] + "yolo/data.yaml").decode()
    assert "path: .\n" in yaml
    if split[collect] == "train":
        assert "val: images/train\n" in yaml
        assert any("validation reuses training data" in w for w in manifest["warnings"])


def test_preview_warns_about_small_sets(client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    cid = _make_collection(client, h, _ids(client))
    pre = client.post("/v1/studio/exports/preview", headers=h,
                      json={"collection_id": cid, "name": "people", "formats": ["yolo"]}).json()
    assert any("groups" in w for w in pre["warnings"])


# ---------------------------------------------------------------- I12 / M2 / M5 / <video>: honest records

def test_item_provenance(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, paused, collect = _ids(client)
    cid = _make_collection(client, h, [alert, paused, collect])
    out = _export(client, h, cid)
    manifest = _manifest(s3client, out)
    items = {i["event_id"]: i for i in manifest["items"]}
    assert items[collect]["yolo"] == {"source": "detector_weak_label", "model": "unknown",
                                      "class_schema": "coco→contiguous-9"}
    ai = items[alert]["ai"]
    assert ai["run_id"].startswith("ai-run-") and ai["status"] == "real" and ai["model"] == "gpt-4o"
    assert ai["prompt_version"] and ai["input"] == "sampled_frames"
    assert items[paused]["ai"] is None
    fb = items[alert]["owner_feedback"]
    assert len(fb) == 1 and fb[0]["feedback_id"].startswith("feedback-") and fb[0]["verdict"] == "true_alert"
    assert fb[0]["received_utc"]
    text = json.dumps(manifest)
    assert not [t for t in IDENTITY if t in text]


def test_vlm_prompt_redaction_flag():
    leaky = type("I", (), {"text": staticmethod(lambda v: v), "mentions": staticmethod(lambda v: True)})
    assert studio.vlm_prompt(leaky, "Camera front_side") == (studio.PLACEHOLDER_PROMPT, True)
    assert studio.vlm_prompt(leaky, None) == (studio.PLACEHOLDER_PROMPT, False)


def test_media_tokens_normalised_to_one_video(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, _, _ = _ids(client)
    with session_scope(client.app.state.engine) as s:
        run = s.scalar(select(m.AiRun).where(m.AiRun.event_id == alert))
        run.prompt = "<video>Look at <image> this <video> clip."
        run.parsed = {**run.parsed, "summary": "A person <video> walks <image> by."}
    cid = _make_collection(client, h, [alert])
    out = _export(client, h, cid, formats=["vlm_jsonl"])
    lines = [json.loads(x) for sp in ("train", "val", "test")
             for x in _get(s3client, f"{out['s3_prefix']}vlm/{sp}.jsonl").decode().splitlines()]
    assert len(lines) == 1
    user, assistant = lines[0]["messages"]
    whole = user["content"] + assistant["content"]
    assert whole.count("<video>") == 1 and "<image>" not in whole and user["content"].startswith("<video>")


def test_clip_hash_streamed_or_marked_not_exported(client, s3client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    alert, _, collect = _ids(client)
    cid = _make_collection(client, h, [alert, collect])
    items = _manifest(s3client, _export(client, h, cid, formats=["clips"]))["items"]
    assert {(i["clip_sha256"], i["clip_hash"]) for i in items} == {(hashlib.sha256(b.FAKE_MP4).hexdigest(),
                                                                    "sha256")}
    items = _manifest(s3client, _export(client, h, cid, name="labels", formats=["yolo"]))["items"]
    assert {(i["clip_sha256"], i["clip_hash"]) for i in items} == {(None, "not_exported")}


# ---------------------------------------------------------------- M3 / M4: bounded workers, size guard

def test_exports_run_on_a_bounded_executor(client):
    executor = client.app.state.export_executor
    assert executor._max_workers == 2


def test_objects_too_large_for_server_side_copy_are_missing(client, s3client, staff_factory, seeded, monkeypatch):
    _, _, _, h = staff_factory("admin")
    alert, _, _ = _ids(client)
    monkeypatch.setattr(studio, "MAX_COPY_BYTES", 10)
    cid = _make_collection(client, h, [alert])
    out = _export(client, h, cid, formats=["clips"])
    manifest = _manifest(s3client, out)
    assert out["state"] == "partial"
    assert {"event_id": alert, "reason": "too_large", "note": "too large for server-side copy"} in manifest["missing"]
