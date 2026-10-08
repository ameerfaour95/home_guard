"""The clip's meta says how the model's input was made: ``model_input`` beside ``vlm_input``, and in the teacher."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import alert_clips, inference as inf
from home_guard_project.data_collection import model_input as mi
import test_inference_crop_parity as parity
import test_model_input_characterization as ch

KEYS = {"version", "vlm_input", "frame_indices", "times", "union_crop", "fps", "sample_fps", "step", "size"}


def saved_metas(name):
    """Run a characterization scenario through prepare, the worker and the clip writer; the metas written."""
    spec = ch.SCENARIOS[name]
    n, w, h = spec["sub"]
    sub = ch.noise(n, w, h, seed=n * 31 + w)
    sub_cap = SimpleNamespace(get_clip_last_seconds=mock.Mock(return_value=(sub, 1000., 1010., spec["sub_fps"])))
    main_cap = None
    if spec["main"] is not None:
        mn, mw, mh = spec["main"]
        main_cap = SimpleNamespace(get_clip_frames=mock.Mock(
            return_value=(ch.noise(mn, mw, mh, seed=mn * 17 + mw), 1000., 1010., spec["main_fps"])))
    job = inf.AlertJob("cam", "cam_1004_alert", 1004., labels=["person"])
    frames, clip = inf._prepare_alert(job, ch.cfg(spec["sample"]), mock.Mock(side_effect=ch.detector_for(spec["boxes"])),
                                      sub_cap, main_cap, {})
    parity.run_worker(job, frames)
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(alert_clips, "_to_h264", return_value=False):
        production, training = Path(tmp) / "production", Path(tmp) / "training"
        inf._save_clip(job, clip, str(production), str(training))
        metas = [json.loads(next(root.rglob("*.meta.json")).read_text()) for root in (production, training)]
    return job, metas


class ModelInputMetaTest(unittest.TestCase):
    def test_a_crop_clip_records_its_recipe(self):
        job, metas = saved_metas("crop_walk")
        for meta in metas:
            got = meta["model_input"]
            self.assertEqual(set(got), KEYS)
            self.assertEqual(got, {k: v for k, v in json.loads(json.dumps(job.model_input.record())).items()
                                   if k != "crops"})
            self.assertEqual((got["version"], got["vlm_input"], got["fps"], got["sample_fps"], got["step"]),
                             (mi.MODEL_INPUT_VERSION, "crop", 5., 1., 5))
            self.assertEqual(got["frame_indices"], list(range(0, 50, 5)))
            self.assertEqual(got["times"], [float(i) for i in range(10)])
            self.assertEqual(got["size"], [job.crop.width, job.crop.height])
            self.assertEqual(got["union_crop"], list(job.model_input.union_crop))
            self.assertLess(len(json.dumps(got)), 400)
            # Additive: the fields the readers already take are all still there, unchanged.
            self.assertEqual(meta["vlm_input"], "crop")
            self.assertIn("vlm_crop", meta)
            self.assertEqual(meta["teacher"]["model_input"], got)
            self.assertTrue(meta["teacher"]["prompt_version"])

    def test_a_whole_frame_fallback_clip_records_its_recipe(self):
        job, metas = saved_metas("whole_no_main_stream")
        for meta in metas:
            got = meta["model_input"]
            self.assertEqual(set(got), KEYS)
            self.assertEqual((got["vlm_input"], got["fps"], got["step"], got["union_crop"], got["size"]),
                             ("whole_frame_fallback", 10., 10, None, [352, 288]))
            self.assertEqual(got["frame_indices"], list(range(0, 100, 10)))
            self.assertEqual((meta["vlm_input"], meta["vlm_fallback_reason"]), ("whole_frame_fallback", "no_main_stream"))
            self.assertEqual(meta["teacher"]["model_input"], got)

    def test_the_in_memory_input_fields_are_unchanged(self):
        job, _ = saved_metas("crop_walk")
        self.assertNotIn("model_input", job.input_meta)

    def test_no_record_before_the_job_is_prepared(self):
        job = inf.AlertJob("cam", "cam_1004_alert", 1004.)
        self.assertIsNone(inf._model_input_record(job))
        self.assertEqual(inf._clip_extra(job), {})


if __name__ == "__main__":
    unittest.main()
