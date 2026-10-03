"""write_yolo_labels: native-frame naming, interpolation parity with the Admin Center, negatives."""
import json
import os

import yaml

from home_guard_project.analysis.analyze import parse_export, write_yolo_labels
from home_guard_project.analysis.config import AnalysisConfig
from home_guard_project.analysis.utils.ls_convert import box_at

CFG = AnalysisConfig(s3_bucket="b", s3_prefix="p", s3_region="r",
                     coco_labels={0: "person", 2: "car"}, output_dir="")
META = "meta/cam/2026-01-01/clip1.meta.json"


def kf(frame, x, enabled=True, time=None):
    d = {"frame": frame, "x": x, "y": 10.0, "width": 10.0, "height": 20.0, "enabled": enabled, "rotation": 0}
    if time is not None:
        d["time"] = time
    return d


def run(tmp_path, data, seq, frames_count, label="person", annotated=True):
    result = []
    if seq is not None:
        result.append({"type": "videorectangle",
                       "value": {"labels": [label], "sequence": seq, "framesCount": frames_count, "duration": 1}})
    task = {"id": 1, "data": dict(meta_path=META, **data),
            "annotations": [{"result": result}] if annotated else []}
    exp = tmp_path / "export.json"
    exp.write_text(json.dumps([task]))
    out = tmp_path / "out"
    tasks = parse_export(str(exp), CFG)
    write_yolo_labels(tasks, CFG, str(out))
    d = out / "yolo" / "labels" / "cam" / "2026-01-01"
    return {p.name: p.read_text() for p in d.iterdir()}, out


def xc(txt):
    return float(txt.split()[1])


def test_legacy_7fps_label_for_image_f0000_is_ls_frame_1(tmp_path):
    files, _ = run(tmp_path, {"fps": 7}, [kf(1, 10), kf(7, 40)], 7)
    assert sorted(files) == [f"clip1_f{i:04d}.txt" for i in range(7)]
    assert abs(xc(files["clip1_f0000.txt"]) - 0.15) < 1e-6      # LS frame 1 (x=10, w=10)
    assert abs(xc(files["clip1_f0006.txt"]) - 0.45) < 1e-6      # LS frame 7 (x=40)


def test_uca_30fps_maps_ls_10fps_to_native(tmp_path):
    meta_dir = tmp_path / "ds" / "meta" / "cam" / "2026-01-01"
    meta_dir.mkdir(parents=True)
    (meta_dir / "clip1.meta.json").write_text(json.dumps({"fps_estimated": 30.0}))
    result = [{"type": "videorectangle", "value": {"labels": ["person"], "framesCount": 100,
               "sequence": [kf(1, 0), kf(100, 99)]}}]
    exp = tmp_path / "e.json"
    exp.write_text(json.dumps([{"id": 1, "data": {"meta_path": META, "fps": 10},
                                "annotations": [{"result": result}]}]))
    tasks = parse_export(str(exp), CFG)
    write_yolo_labels(tasks, CFG, str(tmp_path / "out"), dataset_dir=str(tmp_path / "ds"))
    d = tmp_path / "out" / "yolo" / "labels" / "cam" / "2026-01-01"
    names = sorted(p.name for p in d.iterdir())
    assert names[0] == "clip1_f0000.txt" and "clip1_f0010.txt" in names
    # image f0010 -> LS frame round(10*10/30)+1 = 4
    assert abs(xc((d / "clip1_f0010.txt").read_text()) - (3 + 5) / 100) < 1e-6
    # every native frame has a file, not only multiples of 10
    assert len(names) >= 290


def test_native_frame_space_round_trip_adjacent_keyframes_30fps(tmp_path):
    seq = [kf(1, 10, time=0.0), kf(2, 20, time=1 / 30), kf(4, 20, enabled=False, time=3 / 30)]
    files, _ = run(tmp_path, {"fps": 30, "frame_space": "native"}, seq, 6)
    assert sorted(files) == [f"clip1_f{i:04d}.txt" for i in range(6)]
    assert abs(xc(files["clip1_f0000.txt"]) - 0.15) < 1e-6      # native 0 = LS frame 1
    assert abs(xc(files["clip1_f0001.txt"]) - 0.25) < 1e-6      # native 1 = LS frame 2 (adjacent keyframe)
    assert abs(xc(files["clip1_f0002.txt"]) - 0.25) < 1e-6
    assert files["clip1_f0003.txt"] == "" and files["clip1_f0005.txt"] == ""


def test_enabled_to_disabled_moves_toward_disabled_position():
    seq = [kf(1, 10), kf(11, 50, enabled=False)]
    b = box_at(seq, 6)
    assert abs(b["x"] - 30) < 1e-6
    assert box_at(seq, 11) is None and box_at(seq, 12) is None


def test_nonuniform_times_follow_time_not_frame():
    # frame 6 is midway by frame number, but time says 0.5s of a 2s segment
    seq = [kf(1, 0, time=0.0), kf(11, 100, time=2.0)]
    assert abs(box_at(seq, 6, fps=10)["x"] - 25) < 1e-6         # t=0.5 of 2.0
    assert abs(box_at(seq, 6)["x"] - 50) < 1e-6                 # no fps -> by frame
    assert box_at([kf(5, 1, time=0.4)], 1, fps=10) is None      # before first keyframe


def test_hidden_until_next_keyframe_and_held_after_last():
    seq = [kf(1, 10), kf(3, 10, enabled=False), kf(6, 20)]
    assert box_at(seq, 4) is None and box_at(seq, 2) is not None
    assert box_at(seq, 9)["x"] == 20


def test_empty_negatives_and_yaml(tmp_path):
    files, out = run(tmp_path, {"fps": 7}, None, 0, annotated=True)
    assert files == {} or all(v == "" for v in files.values())
    files, out = run(tmp_path, {"fps": 7, "framesCount": 5}, [kf(1, 10), kf(2, 10, enabled=False)], 5)
    assert sorted(files) == [f"clip1_f{i:04d}.txt" for i in range(5)]
    assert files["clip1_f0000.txt"] != "" and files["clip1_f0003.txt"] == ""
    y = yaml.safe_load((out / "yolo" / "data.yaml").read_text())
    assert y["train"] == "labels"
    os.makedirs(out / "yolo" / "images")
    exp = tmp_path / "export.json"
    write_yolo_labels(parse_export(str(exp), CFG), CFG, str(out))
    y = yaml.safe_load((out / "yolo" / "data.yaml").read_text())
    assert y["path"] == "." and y["train"] == "images" and y["val"] == "images" and y["nc"] == 2


def test_fps_capped_at_10_with_no_meta_skips_task(tmp_path, caplog):
    """When fps=10 (capped) and no meta.json, task should be skipped."""
    result = [{"type": "videorectangle",
               "value": {"labels": ["person"], "sequence": [kf(1, 10), kf(100, 40)], "framesCount": 100, "duration": 1}}]
    task = {"id": 1, "data": dict(meta_path=META, fps=10),
            "annotations": [{"result": result}]}
    exp = tmp_path / "export.json"
    exp.write_text(json.dumps([task]))
    out = tmp_path / "out"
    tasks = parse_export(str(exp), CFG)
    write_yolo_labels(tasks, CFG, str(out), dataset_dir=None)
    d = out / "yolo" / "labels" / "cam" / "2026-01-01"
    # Directory should not exist or be empty because task was skipped
    assert not d.exists() or len(list(d.iterdir())) == 0
    assert "unknown native fps" in caplog.text.lower()


def test_fps_capped_at_10_with_meta_recovers(tmp_path):
    """When fps=10 (capped) but meta.json has native fps, task should process."""
    meta_dir = tmp_path / "ds" / "meta" / "cam" / "2026-01-01"
    meta_dir.mkdir(parents=True)
    (meta_dir / "clip1.meta.json").write_text(json.dumps({"fps_estimated": 30.0}))

    result = [{"type": "videorectangle",
               "value": {"labels": ["person"], "sequence": [kf(1, 10), kf(100, 40)], "framesCount": 100, "duration": 1}}]
    task = {"id": 1, "data": dict(meta_path=META, fps=10),
            "annotations": [{"result": result}]}
    exp = tmp_path / "export.json"
    exp.write_text(json.dumps([task]))
    out = tmp_path / "out"
    tasks = parse_export(str(exp), CFG)
    # Pass dataset_dir so it can read the meta.json
    write_yolo_labels(tasks, CFG, str(out), dataset_dir=str(tmp_path / "ds"))
    d = out / "yolo" / "labels" / "cam" / "2026-01-01"
    files = {p.name: p.read_text() for p in d.iterdir()}
    # Should have labels for the clip
    assert len(files) > 0
    assert any(f != "" for f in files.values())
