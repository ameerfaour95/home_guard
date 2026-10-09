"""box.yaml vlm_max_side: the long side of the frames sent to the vision model, applied in one place
(model_input.render_model_input, shared by the box and the labeling studio). Off (0, the default) changes nothing:
the same bytes, record and version."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.data_collection import model_input as mi
import test_model_input_characterization as ch


def prepared(name, **kw):
    spec = ch.SCENARIOS[name]
    n, w, h = spec["sub"]
    sub = ch.noise(n, w, h, seed=n * 31 + w)
    main = None
    if spec["main"] is not None:
        mn, mw, mh = spec["main"]
        main = ch.noise(mn, mw, mh, seed=mn * 17 + mw)
    sub_cap = SimpleNamespace(get_clip_last_seconds=mock.Mock(return_value=(sub, 1000., 1010., spec["sub_fps"])))
    main_cap = None if main is None else SimpleNamespace(
        get_clip_frames=mock.Mock(return_value=(main, 1000., 1010., spec["main_fps"])))
    job = inf.AlertJob("cam", "cam_1004_alert", 1004., labels=["person"])
    detector = mock.Mock(side_effect=ch.detector_for(spec["boxes"]))
    frames, _ = inf._prepare_alert(job, ch.cfg(spec["sample"]), detector, sub_cap, main_cap, {}, **kw)
    return job, frames


def big(n=4, w=2240, h=1520):
    rng = np.random.default_rng(3)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


class OffIsUnchangedTest(unittest.TestCase):
    def test_off_sends_the_same_bytes_and_record_as_before(self):
        for name in ch.SCENARIOS:
            with self.subTest(name):
                job0, sent0 = prepared(name)
                job1, sent1 = prepared(name, max_side=0)
                self.assertEqual([ch.frame_hash(f) for f in sent1], [ch.frame_hash(f) for f in sent0])
                self.assertEqual(inf._jpegs(sent1), inf._jpegs(sent0))
                self.assertEqual(job1.model_input.record(), job0.model_input.record())
                self.assertEqual(job1.model_input.version, "model-input-1")
                self.assertNotIn("max_side", job1.model_input.record())

    def test_a_cap_above_the_frames_changes_no_pixel(self):
        frames = big(2, 700, 500)
        off = mi.render_model_input(frames, {"fps": 1.0}, mi.ModelInputConfig(1.0))
        on = mi.render_model_input(frames, {"fps": 1.0}, mi.ModelInputConfig(1.0, max_side=768))
        self.assertEqual(on.jpegs(), off.jpegs())
        self.assertEqual(on.record()["max_side"], 768)        # the recipe says the cap was on
        self.assertEqual(on.version, mi.MODEL_INPUT_VERSION_MAX_SIDE)


class CapTest(unittest.TestCase):
    def test_the_long_side_is_capped_aspect_kept(self):
        got = mi.render_model_input(big(), {"fps": 1.0}, mi.ModelInputConfig(1.0, max_side=768))
        self.assertEqual({f.shape[:2] for f in got.frames}, {(521, 768)})
        self.assertEqual(got.size, (768, 521))
        record = got.record()
        self.assertEqual((record["max_side"], record["size"], record["version"]), (768, [768, 521], "model-input-2"))

    def test_the_box_crop_path_is_capped_too(self):
        job, sent = prepared("crop_walk_golden_size", max_side=64)
        self.assertTrue(all(max(f.shape[:2]) <= 64 for f in sent))
        self.assertEqual(job.model_input.record()["max_side"], 64)
        self.assertEqual(inf._model_input_record(job)["max_side"], 64)

    def test_box_yaml_setting(self):
        self.assertEqual(inf.AlertSettings.from_box_settings({}).vlm_max_side, 0)
        self.assertEqual(inf.AlertSettings.from_box_settings({"vlm_max_side": 768}).vlm_max_side, 768)
        for bad in (100, "big", -5, 99999):
            with self.assertLogs("box.inference", "WARNING"):
                self.assertEqual(inf.AlertSettings.from_box_settings({"vlm_max_side": bad}).vlm_max_side, 0)

    def test_the_guard_passes_the_setting_to_the_render(self):
        job = inf.AlertJob("cam", "cam_1_alert", 0.0, labels=["person"])
        settings = inf.AlertSettings(vlm_max_side=768)
        with mock.patch.object(inf, "_prepare_alert", return_value=([], [])) as prep, \
                mock.patch.object(inf, "_attach_tracker"), mock.patch.object(inf, "_attach_track_boxes"), \
                mock.patch.object(inf.threading, "Thread"):
            inf._start_due_alerts([job], 100.0, object(), None, {"cam": SimpleNamespace(sub_cap=None)},
                                  {"cam": None}, {}, None, {}, {}, settings, None, None, "", "")
        self.assertEqual(prep.call_args.kwargs["max_side"], 768)


if __name__ == "__main__":
    unittest.main()
