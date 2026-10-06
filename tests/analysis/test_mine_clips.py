"""Clip mining: rarity weights, clip kinds, scoring, finding untagged clips."""
import os

from home_guard_project.analysis import mine_clips as mc


def test_rare_classes_weigh_more():
    w = mc.class_weights({0: 13549, 2: 12249, 8: 74})
    assert w[0] < 0.1 and w[2] < 0.1
    assert 2.0 < w[8] < 2.3
    assert w[4] == 3.0  # bus: none at all, capped


def test_clip_kind_is_the_part_after_the_timestamp():
    assert mc.clip_kind("back_door_1790944263_trigger") == "trigger"
    assert mc.clip_kind("ameer_tes2_ch3_1790944263_false_positive") == "false_positive"
    assert mc.clip_kind("odd") == ""


def test_score_counts_rare_classes_night_far_people_and_false_alarms():
    w = mc.class_weights({0: 13549, 2: 12249, 8: 74})
    s, why = mc.score_clip({"classes": {"0": 0.9, "8": 0.7, "4": 0.3}, "night": 0.8, "small_person": 2,
                            "kind": "false_positive"}, w)
    assert why == ["dog", "night", "far_person", "false_positive"]
    assert abs(s - (w[8] + 2.0 + 1.0 + 1.5)) < 1e-3
    assert mc.score_clip({"classes": {"0": 0.99}, "night": 0.0, "small_person": 0, "kind": "trigger"}, w) == (0.0, [])


def test_find_clips_needs_meta_and_skips_tagged(tmp_path):
    for cid, meta in (("cam_1790000001_trigger", True), ("cam_1790000002_trigger", False), ("cam_1790000003_alert", True)):
        v = tmp_path / "clips" / "cam" / "2026-10-02" / f"{cid}.mp4"
        v.parent.mkdir(parents=True, exist_ok=True)
        v.write_bytes(b"x")
        if meta:
            m = tmp_path / "meta" / "cam" / "2026-10-02" / f"{cid}.meta.json"
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text("{}")
    found = mc.find_clips([str(tmp_path)], skip={"cam_1790000003_alert"})
    assert [c["id"] for c in found] == ["cam_1790000001_trigger"]
    out = tmp_path / "out"
    mc.copy_clip(found[0], str(out))
    assert os.path.exists(out / "clips" / "cam" / "2026-10-02" / "cam_1790000001_trigger.mp4")
    assert os.path.exists(out / "meta" / "cam" / "2026-10-02" / "cam_1790000001_trigger.meta.json")


def test_a_clip_in_two_sources_counts_once(tmp_path):
    for src in ("a", "b"):
        for sub, name in (("clips", "cam_1790000001_fp.mp4"), ("meta", "cam_1790000001_fp.meta.json")):
            f = tmp_path / src / sub / "cam" / "2026-10-02" / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")
    assert len(mc.find_clips([str(tmp_path / "a"), str(tmp_path / "b")], set())) == 1
    assert mc.clip_kind("cam_1790000001_fp") in mc.HARD_NEGATIVE_KINDS


def test_pick_caps_each_camera():
    rows = [{"camera": "ch3", "score": 3.0}] * 5 + [{"camera": "ch6", "score": 2.0}] * 2 + [{"camera": "x", "score": 0.0}]
    got = mc.pick(rows, top=10, per_camera=3)
    assert [r["camera"] for r in got] == ["ch3"] * 3 + ["ch6"] * 2
