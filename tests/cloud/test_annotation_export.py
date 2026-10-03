"""Training exports with human annotations: human labels win over weak labels, EVERY frame of the clip is extracted
(ffmpeg) with its labels from box_at (empty file = a human negative), the VLM answer is the human description
(provenance human_corrected), and drop_clip excludes the clip (dropped_by_labeler)."""
import json
import shutil
import subprocess

import pytest

from home_guard_project.cloud import pseudonym

from . import builders as b
from .test_event_routes import _event_id, s3client  # noqa: F401
from .test_studio import _export, _get, _keys, _make_collection, _seed

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg/ffprobe not installed")
FRAMES = 14  # the generated clip: 2 s at 7 fps


def make_clip(tmp_path, frames=FRAMES, rate=7) -> bytes:
    out = tmp_path / f"clip_{frames}_{rate}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={rate}",
                    "-frames:v", str(frames), "-pix_fmt", "yuv420p", "-c:v", "libx264", str(out)], check=True)
    return out.read_bytes()


PERSON = {"track_id": "p1", "label": "person", "source": "human", "keyframes": [
    {"frame": 0, "t_sec": 0.0, "xyxy": [0.1, 0.1, 0.3, 0.5], "enabled": True},
    {"frame": 7, "t_sec": 1.0, "xyxy": [0.5, 0.1, 0.7, 0.5], "enabled": True},
    {"frame": 9, "t_sec": 9 / 7, "xyxy": [0.5, 0.1, 0.7, 0.5], "enabled": False}]}


def _annotate(client, h, eid, tracks=(), description="", status="submitted", **kw):
    r = client.put(f"/v1/events/{eid}/annotation", headers=h, json={
        "base_version": client.get(f"/v1/events/{eid}/annotation", headers=h).json()["version"],
        "tracks": list(tracks), "description": description, "status": status, **kw})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def labeled(client, s3client, staff_factory, tmp_path):
    clip = make_clip(tmp_path)
    _seed(client, s3client, overrides={b.COLLECT_CLIP: clip, b.TRAIN_CLIP: clip, b.PAUSED_CLIP: clip})
    labeler, _, _, lab = staff_factory("labeler")
    _, _, _, adm = staff_factory("admin")
    collect, alert, paused = (_event_id(client, s) for s in (b.COLLECT_STEM, b.STEM, b.PAUSED_STEM))
    _annotate(client, lab, collect, tracks=[PERSON], description="One person crosses the yard.")
    _annotate(client, lab, alert, description="A courier leaves a parcel.")
    assert client.post(f"/v1/events/{alert}/annotation/review", headers=adm,
                       json={"decision": "accept"}).status_code == 200
    _annotate(client, lab, paused, description="nothing here", drop_clip=True)
    cid = _make_collection(client, adm, [collect, alert, paused])
    return {"adm": adm, "cid": cid, "collect": collect, "alert": alert, "paused": paused, "labeler": labeler}


def _manifest(s3client, export):
    return json.loads(_get(s3client, export["s3_prefix"] + "manifest.json"))


@needs_ffmpeg
def test_human_labels_every_frame_and_provenance(client, s3client, labeled):
    adm, collect = labeled["adm"], labeled["collect"]
    export = _export(client, adm, labeled["cid"], formats=("yolo", "vlm_jsonl", "clips"))
    assert export["state"] in ("ready", "partial"), export
    man = _manifest(s3client, export)
    items = {i["event_id"]: i for i in man["items"]}
    item = items[collect]
    secret = client.app.state.settings.jwt_secret
    who = pseudonym.staff(secret, labeled["labeler"].id)
    assert item["yolo_frames"] == FRAMES
    assert item["yolo"] == {"source": "human", "annotation_version": 1, "labeled_by": who,
                            "class_schema": "coco→contiguous-9"}
    assert item["annotation"] == {"version": 1, "labeled_by": who, "status": "submitted"}
    split = item["split"]
    prefix = export["s3_prefix"]
    labels = sorted(k for k in _keys(s3client, f"{prefix}yolo/labels/{split}/") if f"/{collect}_f" in k)
    images = sorted(k for k in _keys(s3client, f"{prefix}yolo/images/{split}/") if f"/{collect}_f" in k)
    assert [k.rsplit("/", 1)[1] for k in labels] == [f"{collect}_f{i:04d}.txt" for i in range(FRAMES)]
    assert len(images) == FRAMES and all(_get(s3client, k)[:2] == b"\xff\xd8" for k in images)
    text = lambda i: _get(s3client, f"{prefix}yolo/labels/{split}/{collect}_f{i:04d}.txt").decode()  # noqa: E731
    assert text(0) == "0 0.200000 0.300000 0.200000 0.400000\n"
    assert text(3) == "0 0.371429 0.300000 0.200000 0.400000\n"  # moved by time, 3/7 s of the way
    assert text(8) == "0 0.600000 0.300000 0.200000 0.400000\n"
    assert text(9) == "" and text(13) == ""  # hidden from the disabled keyframe on: human negatives
    # the reviewed clip with no box at all: every frame an empty label
    alert = items[labeled["alert"]]
    assert alert["yolo_frames"] == FRAMES and alert["annotation"]["status"] == "reviewed"
    assert man["yolo"]["labels"] == "human"


@needs_ffmpeg
def test_vlm_answer_is_the_human_description(client, s3client, labeled):
    export = _export(client, labeled["adm"], labeled["cid"], formats=("vlm_jsonl",))
    prefix = export["s3_prefix"]
    lines = [json.loads(line) for key in _keys(s3client, prefix + "vlm/") if key.endswith(".jsonl")
             for line in _get(s3client, key).decode().splitlines() if line]
    by_event = {line["event_id"]: line for line in lines}
    alert = by_event[labeled["alert"]]
    answer = json.loads(alert["messages"][1]["content"])
    assert answer["summary"] == "A courier leaves a parcel."
    assert answer["label"] == "normal" and answer["people"] == 1  # the AI's other fields kept
    assert alert["label_source"] == "human_corrected" and alert["annotation_version"] == 1
    collect = by_event[labeled["collect"]]  # no AI answer: the description itself
    assert collect["messages"][1]["content"] == "One person crosses the yard."
    assert collect["label_source"] == "human_corrected"


@needs_ffmpeg
def test_dropped_clip_is_excluded_from_preview_and_export(client, s3client, labeled):
    adm, cid, paused = labeled["adm"], labeled["cid"], labeled["paused"]
    preview = client.post("/v1/studio/exports/preview", headers=adm,
                          json={"collection_id": cid, "name": "people", "formats": ["yolo"]}).json()
    assert {"event_id": paused, "reason": "dropped_by_labeler"} in preview["excluded"]
    assert paused not in preview["included_ids"]
    export = _export(client, adm, cid, formats=("yolo",))
    man = _manifest(s3client, export)
    assert paused not in {i["event_id"] for i in man["items"]}
    assert {"event_id": paused, "reason": "dropped_by_labeler"} in man["excluded"]


@needs_ffmpeg
def test_a_newer_unsubmitted_version_keeps_weak_labels_out_and_uses_the_submitted_one(client, s3client, labeled):
    """An edit after submitting makes the clip unlabeled again: the export falls back to the weak labels."""
    adm, collect = labeled["adm"], labeled["collect"]
    _annotate(client, adm, collect, tracks=[], description="draft", status="edited")
    export = _export(client, adm, labeled["cid"], formats=("yolo",))
    item = {i["event_id"]: i for i in _manifest(s3client, export)["items"]}[collect]
    assert item["annotation"] is None and item["yolo"]["source"] == "detector_weak_label"
