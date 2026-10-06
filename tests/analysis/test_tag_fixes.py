"""Tag fixes for YOLO training: LS keyframe clock, newest annotation only, vehicle classes, frame review, dataset build."""
import json

import cv2
import numpy as np
import yaml

from home_guard_project.analysis import detections as dt
from home_guard_project.analysis.analyze import parse_export, write_yolo_labels
from home_guard_project.analysis.build_yolo_dataset import build
from home_guard_project.analysis.config import AnalysisConfig
from home_guard_project.analysis.review_frames import review_task
from home_guard_project.analysis.utils.ls_convert import box_at
from home_guard_project.analysis.vehicle_classes import fix_tasks

CFG = AnalysisConfig(s3_bucket="b", s3_prefix="p", s3_region="r",
                     coco_labels={0: "person", 2: "car", 7: "truck"}, output_dir="")
META = "meta/cam/2026-01-01/clip1.meta.json"


def kf(frame, x, y=10.0, w=10.0, h=20.0, enabled=True, fps=7.0):
    # Label Studio's own export writes time = frame / fps
    return {"frame": frame, "x": x, "y": y, "width": w, "height": h, "enabled": enabled,
            "rotation": 0, "time": frame / fps}


def rect(rid, label, seq, frames_count=7):
    return {"id": rid, "type": "videorectangle", "from_name": "bbox",
            "value": {"labels": [label], "sequence": seq, "framesCount": frames_count}}


def task(results, tid=1, desc="a person walks by", updated="2026-03-01T10:00"):
    return {"id": tid, "data": {"meta_path": META, "fps": 7.0},
            "annotations": [{"updated_at": updated, "result": results + [
                {"type": "textarea", "from_name": "vlm_description", "value": {"text": [desc]}}]}]}


def cache(frames, native_fps=7.0, ls_fps=7.0):
    return {"clip": "clip1", "native_fps": native_fps, "ls_fps": ls_fps, "n_frames": 7, "frames": frames}


# --- Label Studio clock --------------------------------------------------------

def test_ls_time_equals_frame_over_fps_box_shows_on_its_own_frame():
    seq = [kf(1, 10), kf(4, 40)]
    assert box_at(seq, 1, 7.0)["x"] == 10        # was None: the query ran one frame early
    assert abs(box_at(seq, 2, 7.0)["x"] - 20) < 1e-6
    assert box_at(seq, 4, 7.0)["x"] == 40


def test_zero_based_times_still_supported():
    seq = [dict(kf(1, 10), time=0.0), dict(kf(4, 40), time=3 / 7.0)]
    assert box_at(seq, 1, 7.0)["x"] == 10
    assert abs(box_at(seq, 2, 7.0)["x"] - 20) < 1e-6


# --- analyzer --------------------------------------------------------------------

def test_only_newest_annotation_counts(tmp_path):
    old = {"updated_at": "2026-03-01T09:00", "result": [rect("a", "person", [kf(1, 10)])]}
    new = {"updated_at": "2026-03-01T20:00", "result": [rect("a", "person", [kf(1, 10)]),
           {"type": "textarea", "from_name": "vlm_description", "value": {"text": ["two men talk"]}}]}
    exp = tmp_path / "e.json"
    exp.write_text(json.dumps([{"id": 1, "data": {"meta_path": META, "fps": 7}, "annotations": [old, new]}]))
    tasks = parse_export(str(exp), CFG)
    assert len(tasks[0].tracks) == 1
    write_yolo_labels(tasks, CFG, str(tmp_path / "out"))
    lines = (tmp_path / "out/yolo/labels/cam/2026-01-01/clip1_f0000.txt").read_text().split("\n")
    assert len([x for x in lines if x]) == 1


def test_sliver_boxes_are_dropped(tmp_path):
    exp = tmp_path / "e.json"
    exp.write_text(json.dumps([task([rect("a", "person", [kf(1, 10, w=0.1)]), rect("b", "car", [kf(1, 50)])])]))
    write_yolo_labels(parse_export(str(exp), CFG), CFG, str(tmp_path / "out"))
    body = (tmp_path / "out/yolo/labels/cam/2026-01-01/clip1_f0000.txt").read_text().split()
    assert body[0] == "1" and len(body) == 5     # only the car (class 1 in 0..N-1)


# --- vehicle classes ---------------------------------------------------------------

def test_car_track_becomes_truck_when_detector_agrees():
    t = task([rect("v", "car", [kf(1, 10), kf(7, 10)]), rect("w", "car", [kf(1, 60), kf(7, 60)])])
    frames = {n: [(7, 0.9, 10, 10, 10, 20), (2, 0.9, 60, 10, 10, 20)] for n in range(7)}
    rows = fix_tasks([t], {"1": cache(frames)})
    labels = {r["value"]["labels"][0] for r in t["annotations"][0]["result"] if r["type"] == "videorectangle"}
    assert labels == {"truck", "car"}
    assert {r["track_id"]: r["decision"] for r in rows} == {"v": "changed", "w": "kept"}


def test_unsure_detector_keeps_car_and_flags_review():
    t = task([rect("v", "car", [kf(1, 10), kf(7, 10)])])
    rows = fix_tasks([t], {"1": cache({0: [(7, 0.9, 10, 10, 10, 20)]})})     # one frame < min_frames
    assert t["annotations"][0]["result"][0]["value"]["labels"] == ["car"]
    assert rows[0]["decision"].startswith("review")


def test_deleted_tasks_are_skipped():
    t = task([rect("v", "car", [kf(1, 10)])], desc="[delete]")
    assert fix_tasks([t], {"1": cache({0: [(7, 0.9, 10, 10, 10, 20)]})}) == []


# --- frame review -----------------------------------------------------------------

def _review(results, big, deployed=None):
    t = task(results)
    return review_task(t, dt.latest_annotation(t), cache(big), cache(deployed) if deployed else None)


def test_untagged_confident_object_excludes_the_frame():
    big = {0: [(0, 0.9, 50, 10, 10, 20)], 1: [(0, 0.55, 50, 10, 10, 20)], 2: []}
    excluded, hard, _ = _review([], big)
    assert excluded == {0: "missed:0"}             # 0.55 is below the bar (shadows, plants)
    assert hard == []


def test_held_person_box_with_nothing_under_it_is_excluded_but_parked_car_is_not():
    results = [rect("p", "person", [kf(1, 10)]), rect("c", "car", [kf(1, 60)])]
    big = {0: [(0, 0.9, 10, 10, 10, 20)], 3: []}
    excluded, _, _ = _review(results, big)
    assert excluded == {3: "held_unsupported:person"}


def test_deployed_false_person_is_hard_negative_unless_big_model_sees_one():
    deployed = {0: [(0, 0.6, 50, 10, 10, 20)], 1: [(0, 0.6, 50, 10, 10, 20)], 2: [(0, 0.3, 50, 10, 10, 20)]}
    big = {0: [], 1: [(0, 0.5, 50, 10, 10, 20)], 2: []}
    excluded, hard, _ = _review([], big, deployed)
    assert hard == [0]
    assert excluded == {1: "missed:0"}


def test_tagged_person_is_not_a_hard_negative():
    deployed = {0: [(0, 0.8, 10, 10, 10, 20)]}
    excluded, hard, _ = _review([rect("p", "person", [kf(1, 10), kf(7, 10)])], {0: [(0, 0.9, 10, 10, 10, 20)]}, deployed)
    assert hard == [] and excluded == {}


# --- dataset build ----------------------------------------------------------------

def test_build_skips_excluded_and_repeats_hard_negatives(tmp_path):
    clips = tmp_path / "ds" / "clips" / "cam" / "2026-01-01"
    clips.mkdir(parents=True)
    vw = cv2.VideoWriter(str(clips / "clip1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (64, 48))
    for i in range(12):
        vw.write(np.full((48, 64, 3), i * 20, np.uint8))
    vw.release()
    lab = tmp_path / "an" / "yolo" / "labels" / "cam" / "2026-01-01"
    lab.mkdir(parents=True)
    for n in range(12):
        (lab / f"clip1_f{n:04d}.txt").write_text("0 0.5 0.5 0.2 0.2\n" if n < 6 else "")
    (tmp_path / "an" / "yolo" / "data.yaml").write_text(yaml.safe_dump({"names": {0: "person"}}))
    out = tmp_path / "out"
    # 6 fps / 3 fps -> every 2nd frame; frame 2 excluded, frame 7 a hard negative
    build(str(out), [(str(tmp_path / "an"), str(tmp_path / "ds"))], {"clip1": {7}}, {"clip1": {2}},
          fps=3.0, val_share=0.0, hn_repeat=3)
    imgs = sorted(p.name for p in (out / "images").rglob("*.jpg"))
    assert imgs == [f"clip1_f{n:04d}.jpg" for n in (0, 4, 6, 7, 8, 10)]
    lines = (out / "train.txt").read_text().split() + (out / "val.txt").read_text().split()
    # the only clip lands in val (each source keeps >= 1 val clip), where nothing is repeated
    assert sum(x.endswith("clip1_f0007.jpg") for x in lines) == 1
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == {0: "person"}
