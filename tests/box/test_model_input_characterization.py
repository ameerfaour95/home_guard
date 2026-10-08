"""What the vision model sees, pinned byte for byte: the characterization behind data_collection/model_input.py.

The expected hashes in fixtures/model_input_characterization.json were recorded from the code BEFORE the
model input moved into one function (2026-10-08): inference._prepare_alert (crop or whole sub-stream frames,
sampled at vlm.sample_fps) and inference._jpegs (the JPEG bytes the backend sends). Synthetic cameras and a
fake detector keep every run identical. Regenerate only on purpose: HG_WRITE_CHARACTERIZATION=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.data_collection import vlm_crop

EXPECTED = Path(__file__).with_name("fixtures") / "model_input_characterization.json"
WRITE = os.environ.get("HG_WRITE_CHARACTERIZATION") == "1"


class _Cls:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Xyxy:
    def __init__(self, box):
        self.box = box

    def tolist(self):
        return list(self.box)


class _Box:
    def __init__(self, cls_id, box):
        self.cls = _Cls(cls_id)
        self.xyxy = [_Xyxy(box)]


def detector_for(boxes_at):
    """A YOLO stand-in: the sub frame's index is in pixel (0, 0, 0); *boxes_at(i)* gives [(class, xyxy)]."""
    def detect(frame, verbose=False, conf=0.0, imgsz=640, **_):
        i = int(frame[0, 0, 0])
        return [SimpleNamespace(boxes=[_Box(c, b) for c, b in boxes_at(i)])]
    return detect


def noise(n, w, h, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        f[0, 0, 0] = i
        frames.append(f)
    return frames


def walk(i):
    return [] if i < 12 else [(0, (20 + i * 2.5, 90 + i * 0.4, 50 + i * 2.5, 190 + i * 0.4)), (14, (10, 10, 20, 20))]


# name -> sub (n, w, h), main (n, w, h) or None, main start offset, fps (sub, main), sample fps, detections
SCENARIOS = {
    "crop_walk": dict(sub=(100, 352, 288), main=(50, 704, 576), sub_fps=10., main_fps=5., sample=1., boxes=walk),
    "crop_walk_golden_size": dict(sub=(100, 352, 288), main=(50, 1408, 1152), sub_fps=10., main_fps=5., sample=1.,
                                  boxes=walk),
    "whole_no_main_stream": dict(sub=(100, 352, 288), main=None, sub_fps=10., main_fps=5., sample=1., boxes=walk),
    "edge_of_frame_crop": dict(sub=(100, 352, 288), main=(50, 704, 576), sub_fps=10., main_fps=5., sample=1.,
                               boxes=lambda i: [(0, (300 + (i % 7), 200, 352, 288))]),
    "top_left_edge_crop": dict(sub=(60, 352, 288), main=(30, 704, 576), sub_fps=10., main_fps=5., sample=1.,
                               boxes=lambda i: [(0, (0, 0, 25, 40 + (i % 5)))]),
    "small_person_below_min_size": dict(sub=(100, 352, 288), main=(50, 1280, 720), sub_fps=10., main_fps=5.,
                                        sample=1., boxes=lambda i: [(0, (150 + i * 0.3, 140, 156 + i * 0.3, 152))]),
    "big_car_and_two_people": dict(sub=(80, 352, 288), main=(40, 704, 576), sub_fps=10., main_fps=5., sample=1.,
                                   boxes=lambda i: [(2, (20, 100, 260, 250)), (0, (270, 60 + i % 3, 300, 160)),
                                                    (0, (5, 30, 30, 120))]),
    "short_clip_odd_rates": dict(sub=(37, 352, 288), main=(23, 640, 480), sub_fps=10., main_fps=7.5, sample=2.,
                                 boxes=lambda i: [(0, (100 + i, 80, 140 + i, 200))]),
    "long_clip": dict(sub=(150, 352, 288), main=(75, 640, 480), sub_fps=10., main_fps=5., sample=1., boxes=walk),
    "gap_then_detections": dict(sub=(100, 352, 288), main=(50, 704, 576), sub_fps=10., main_fps=5., sample=1.,
                                boxes=lambda i: [(0, (40 + i, 50, 90 + i, 200))] if i in range(20, 31) or i > 70
                                else []),
    "main_starts_later": dict(sub=(100, 352, 288), main=(30, 704, 576), main_start=1004., sub_fps=10., main_fps=5.,
                              sample=1., boxes=walk),
    "no_trigger_detection": dict(sub=(100, 352, 288), main=(50, 704, 576), sub_fps=10., main_fps=5., sample=1.,
                                 boxes=lambda i: [(14, (10, 10, 20, 20))]),
    "too_few_main_frames": dict(sub=(100, 352, 288), main=(1, 704, 576), sub_fps=10., main_fps=5., sample=1.,
                                boxes=walk),
    "whole_short_sub_sample_2": dict(sub=(23, 320, 240), main=None, sub_fps=7.5, main_fps=5., sample=2., boxes=walk),
    "snapshot_only": dict(sub=(0, 352, 288), main=(50, 704, 576), sub_fps=10., main_fps=5., sample=1., boxes=walk),
}

# (frames, fps, sample fps): the sampling step, banker's rounding and all
SAMPLING = [(50, 5., 1.), (23, 7.5, 1.), (23, 7.5, 2.), (40, 12.5, 1.), (10, 0., 1.), (10, 5., 0.), (7, 2.5, 1.),
            (100, 10., 1.), (100, 10., 3.), (1, 5., 1.), (0, 5., 1.), (30, 15., 4.)]


def frame_hash(frame) -> str:
    h = hashlib.sha256(str(frame.shape).encode())
    h.update(np.ascontiguousarray(frame).tobytes())
    return h.hexdigest()


def cfg(sample_fps):
    return SimpleNamespace(MAIN_STREAM_ENABLED=True, STORE_FPS=10.0, YOLO_TRIGGER_CONF=0.35, YOLO_IMGSZ=640,
                           TRIGGER_CLASS_IDS=[0, 2, 5, 7], CROP_PADDING=0.3, CROP_MIN_SIZE=384, CROP_EMA_ALPHA=0.3,
                           CLIP_SECONDS=10., VLM_SAMPLE_FPS=sample_fps)


def run_scenario(spec):
    n, w, h = spec["sub"]
    sub = noise(n, w, h, seed=n * 31 + w)
    main_cap = None
    if spec["main"] is not None:
        mn, mw, mh = spec["main"]
        main = noise(mn, mw, mh, seed=mn * 17 + mw)
        main_cap = SimpleNamespace(get_clip_frames=mock.Mock(
            return_value=(main, spec.get("main_start", 1000.), 1010., spec["main_fps"])))
    sub_cap = SimpleNamespace(get_clip_last_seconds=mock.Mock(return_value=(sub, 1000., 1010., spec["sub_fps"])))
    job = inf.AlertJob("cam", "cam_1004_alert", 1004., labels=["person"])
    if not sub:
        job.snapshot = noise(1, w, h, seed=99)[0]
    detector = mock.Mock(side_effect=detector_for(spec["boxes"]))
    frames, clip = inf._prepare_alert(job, cfg(spec["sample"]), detector, sub_cap, main_cap, {})
    crop = None
    if job.crop is not None:
        c = job.crop
        crop = {"first_crop": list(c.first_crop), "size": [c.width, c.height], "source": list(c.source),
                "frames": [frame_hash(f) for f in c.frames], "fps": job.crop_fps,
                "meta": vlm_crop.crop_meta(c, job.crop_settings, "rel.mp4", job.crop_fps)}
    return {
        "input_meta": job.input_meta,
        "detector_calls": detector.call_count,
        "sent": [frame_hash(f) for f in frames],
        "jpegs": [hashlib.sha256(j).hexdigest() for j in inf._jpegs(frames)],
        "jpeg_direct": [hashlib.sha256(inf.frame_to_jpeg_bytes(f)).hexdigest() for f in frames],
        "crop": crop,
        "clip": [frame_hash(f) for _, f in clip],
    }


def sampling():
    out = []
    for n, fps, sample_fps in SAMPLING:
        frames = [np.full((2, 2, 3), i, np.uint8) for i in range(n)]
        picked = vlm_crop.sample_for_vlm(frames, fps=fps, sample_fps=sample_fps)
        out.append({"case": [n, fps, sample_fps], "picked": [int(f[0, 0, 0]) for f in picked]})
    return out


def observed():
    return {"scenarios": {name: run_scenario(spec) for name, spec in SCENARIOS.items()}, "sampling": sampling()}


class ModelInputCharacterizationTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.got = json.loads(json.dumps(observed()))
        if WRITE:
            EXPECTED.write_text(json.dumps(cls.got, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        cls.expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

    def test_every_scenario_sends_the_recorded_bytes(self):
        self.assertEqual(sorted(self.got["scenarios"]), sorted(self.expected["scenarios"]))
        for name in SCENARIOS:
            with self.subTest(name):
                self.assertEqual(self.got["scenarios"][name], self.expected["scenarios"][name])

    def test_sampling_picks_the_recorded_frames(self):
        self.assertEqual(self.got["sampling"], self.expected["sampling"])

    def test_the_scenarios_cover_both_inputs_and_the_fallbacks(self):
        kinds = {s["input_meta"].get("vlm_fallback_reason", s["input_meta"]["vlm_input"])
                 for s in self.expected["scenarios"].values()}
        self.assertEqual(kinds, {"crop", "no_main_stream", "no_trigger_class_detection_or_usable_crop",
                                 "too_few_main_frames", "too_few_sub_frames"})
        self.assertTrue(all(s["sent"] and s["jpegs"] == s["jpeg_direct"] for s in self.expected["scenarios"].values()))


if __name__ == "__main__":
    unittest.main()
