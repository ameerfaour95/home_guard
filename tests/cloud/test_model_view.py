"""What the AI saw of a clip (tagstudio/model_view.py): the saved frames the box sent, else the meta's model-input
recipe over the saved crop clip, else the crop rendered by the box's own render_model_input, else whole frames; and
POST /v1/tagging/model_input with its consent rule and audit."""
import base64
import json
import os

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.tagstudio import model_view as mv
from home_guard_project.fleet_contract.model_input import encode_jpeg

from .tagstudio_fixtures import alert_meta, put_owner_clip
from .test_tagging_routes import STEM, studio  # noqa: F401 (fixture)


def video(path, n, size=(96, 64), fps=5.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), (i * 20) % 255, np.uint8)
        writer.write(frame)
    writer.release()
    return path


def jpeg(value, size=(48, 40)):
    return encode_jpeg(np.full((size[1], size[0], 3), value, np.uint8))


def test_sent_frames_come_first(tmp_path):
    sent = [jpeg(10), jpeg(200)]
    view = mv.build({"vlm_input": "crop"}, sent, video(str(tmp_path / "c.mp4"), 10))
    assert view.source == mv.SENT and view.frames == sent
    assert view.label == "What the AI sees: crop · 1 fps · 2 frames · 48×40"


def test_the_recipe_picks_the_box_frames_from_the_saved_crop(tmp_path):
    crop = video(str(tmp_path / "c.mp4"), 12)
    meta = {"vlm_input": "crop", "vlm_crop": {"fps": 5.0},
            "model_input": {"version": "model-input-1", "vlm_input": "crop", "frame_indices": [1, 6, 11],
                            "times": [0.2, 1.2, 2.2], "union_crop": None, "fps": 5.0, "sample_fps": 1.0, "step": 5,
                            "size": [384, 384]}}
    view = mv.build(meta, (), crop)
    assert view.source == mv.RECIPE and len(view.frames) == 3 and view.times == [0.2, 1.2, 2.2]
    assert view.label == "What the AI sees: crop · 1 fps · 3 frames · 384×384"
    # the teacher's copy of the recipe counts too (older metas carry only that one)
    assert mv.build({"teacher": {"model_input": meta["model_input"]}}, (), crop).source == mv.RECIPE


def test_rendered_like_the_box_without_saved_frames_or_recipe(tmp_path):
    crop = video(str(tmp_path / "c.mp4"), 11)
    view = mv.build({"vlm_input": "crop", "vlm_crop": {"fps": 5.0}}, (), crop)
    # every round(5 / 1)-th frame from the first: 0, 5, 10
    assert view.source == mv.RENDERED and len(view.frames) == 3 and view.record["frame_indices"] == [0, 5, 10]
    assert view.vlm_input == "crop" and list(view.size) == [96, 64]


def test_whole_frames_when_no_crop_was_saved(tmp_path):
    clip = video(str(tmp_path / "w.mp4"), 15, fps=7.0)
    view = mv.build({"fps_estimated": 7.0}, (), None, clip)
    assert view.source == mv.WHOLE and view.record["frame_indices"] == [0, 7, 14]
    assert view.label.startswith("What the AI sees: whole frame · 1 fps · 3 frames")
    assert mv.build({}, (), None, None) is None


def test_local_sent_frames_follow_the_copy_layout(tmp_path):
    root = tmp_path / "production_x"
    meta_path = root / "meta" / "cam" / "2026-10-04" / "cam_1_alert.meta.json"
    meta_path.parent.mkdir(parents=True)
    rels = [f"vlm_crops\\cam\\2026-10-04\\cam_1_alert_f{i}.jpg" for i in range(2)]
    for i, rel in enumerate(rels):
        path = root / rel.replace("\\", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg(50 * i))
    meta = {"teacher": {"input_frames": rels}}
    assert mv.local_sent_frames(meta, str(meta_path)) == [jpeg(0), jpeg(50)]
    meta["teacher"]["input_frames"].append("vlm_crops\\cam\\2026-10-04\\missing_f9.jpg")
    assert mv.local_sent_frames(meta, str(meta_path)) == []          # never a partial set


def test_route_shows_the_model_input_with_consent_and_audit(client, staff_factory, studio, tmp_path):  # noqa: F811
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    ds = str(s.paths.dataset)
    meta = alert_meta("house2_ch2", STEM, label="suspicious", vlm_input="crop",
                      vlm_crop={"fps": 5.0, "vlm_crop_path": f"vlm_crops\\house2_ch2\\2026-10-04\\{STEM}.mp4"})
    put_owner_clip(ds, "production_house2", meta, STEM)
    video(os.path.join(ds, "owner_feedback", "production_house2", "vlm_crops", "house2_ch2", "2026-10-04",
                       f"{STEM}.mp4"), 11)
    r = client.post("/v1/tagging/model_input", headers=h, json={"key": f"ev:{ids['refusing']}"})
    assert r.status_code == 403
    r = client.post("/v1/tagging/model_input", headers=h, json={"key": f"ev:{ids['consenting']}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "rendered" and len(body["frames"]) == 3 and body["prompt_version"] == "2026-10-03.test"
    assert body["label"] == "What the AI sees: crop · 1 fps · 3 frames · 96×64"
    assert base64.b64decode(body["frames"][0])[:2] == b"\xff\xd8"
    with session_scope(client.app.state.engine) as session:
        audit = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "media_view").order_by(m.AuditLog.id.desc()))
        assert audit.detail == {"via": "tagging", "kind": "model_input", "source": "rendered"}
    assert client.post("/v1/tagging/model_input", headers=h, json={"key": "ds:nope"}).status_code == 404
