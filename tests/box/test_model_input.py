"""render_model_input: the record it keeps, the saved-clip path the labeling studio takes, and its vendorability."""
from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.data_collection import model_input as mi
import test_model_input_characterization as ch


def prepared(name):
    """Run a characterization scenario through the box and return the job and the frames it sent."""
    spec = ch.SCENARIOS[name]
    n, w, h = spec["sub"]
    mn, mw, mh = spec["main"]
    sub = ch.noise(n, w, h, seed=n * 31 + w)
    main = ch.noise(mn, mw, mh, seed=mn * 17 + mw)
    sub_cap = SimpleNamespace(get_clip_last_seconds=mock.Mock(return_value=(sub, 1000., 1010., spec["sub_fps"])))
    main_cap = SimpleNamespace(get_clip_frames=mock.Mock(return_value=(main, 1000., 1010., spec["main_fps"])))
    job = inf.AlertJob("cam", "cam_1004_alert", 1004., labels=["person"])
    detector = mock.Mock(side_effect=ch.detector_for(spec["boxes"]))
    frames, _ = inf._prepare_alert(job, ch.cfg(spec["sample"]), detector, sub_cap, main_cap, {})
    return job, frames, main


def hashes(frames):
    return [ch.frame_hash(f) for f in frames]


class SavedCropClipTest(unittest.TestCase):
    def test_the_saved_crop_clip_renders_to_the_frames_the_box_sent(self):
        for name in ("crop_walk", "edge_of_frame_crop", "short_clip_odd_rates", "top_left_edge_crop"):
            with self.subTest(name):
                job, sent, _ = prepared(name)
                studio = mi.render_model_input(job.crop.frames, {"vlm_input": "crop", "fps": job.crop_fps},
                                               mi.config_from(ch.cfg(ch.SCENARIOS[name]["sample"])))
                self.assertEqual(hashes(studio.frames), hashes(sent))
                self.assertEqual(studio.jpegs(), inf._jpegs(sent))
                self.assertEqual(studio.frame_indices, job.model_input.frame_indices)
                self.assertEqual(studio.vlm_input, "crop")


class RecordTest(unittest.TestCase):
    def test_the_box_keeps_how_its_frames_were_made(self):
        job, sent, main = prepared("crop_walk")
        got = job.model_input
        self.assertIs(got.frames, sent)
        self.assertEqual((got.vlm_input, got.fps, got.sample_fps, got.step), ("crop", 5., 1., 5))
        self.assertEqual(got.frame_indices, list(range(0, 50, 5)))
        self.assertEqual(got.times, [float(i) for i in range(10)])
        self.assertEqual(got.crops, [job.crop.crops[i] for i in got.frame_indices])
        self.assertEqual(got.size, (job.crop.width, job.crop.height))
        self.assertEqual(got.union_crop, (min(c[0] for c in got.crops), min(c[1] for c in got.crops),
                                          max(c[2] for c in got.crops), max(c[3] for c in got.crops)))
        self.assertEqual(got.version, mi.MODEL_INPUT_VERSION)
        record = json.loads(json.dumps(got.record()))
        self.assertEqual(record["frame_indices"], got.frame_indices)
        self.assertEqual(record["size"], list(got.size))
        self.assertEqual(set(record), {"version", "vlm_input", "frame_indices", "times", "crops", "union_crop", "fps",
                                       "sample_fps", "step", "size"})

    def test_whole_frames_are_sent_as_they_are(self):
        frames = ch.noise(23, 64, 48, seed=3)
        got = mi.render_model_input(frames, {"fps": 7.5}, mi.ModelInputConfig(sample_fps=2.))
        self.assertEqual(got.vlm_input, mi.VLM_INPUT_WHOLE)
        self.assertEqual(got.frame_indices, [0, 4, 8, 12, 16, 20])
        self.assertTrue(all(a is frames[i] for a, i in zip(got.frames, got.frame_indices)))
        self.assertEqual((got.crops, got.union_crop, got.size), ([None] * 6, None, (64, 48)))

    def test_frames_without_a_usable_box_are_dropped_before_sampling(self):
        frames = ch.noise(6, 64, 48, seed=4)
        crops = [None, (0, 0, 20, 20), (10, 10, 30, 30), (60, 40, 60, 48), (0, 0, 22, 18), (5, 5, 25, 25)]
        got = mi.render_model_input(frames, {"fps": 2., "crops": crops}, mi.ModelInputConfig(sample_fps=1.))
        self.assertEqual(got.frame_indices, [1, 4])     # usable: 1, 2, 4, 5 (3 is empty); every 2nd
        self.assertEqual(got.size, (20, 20))
        self.assertEqual([f.shape for f in got.frames], [(20, 20, 3), (20, 20, 3)])
        np.testing.assert_array_equal(got.frames[0], frames[1][0:20, 0:20])

    def test_nothing_to_send(self):
        got = mi.render_model_input([], {"fps": 5.}, mi.ModelInputConfig())
        self.assertEqual((got.frames, got.frame_indices, got.size, got.jpegs()), ([], [], None, []))


class ContractTest(unittest.TestCase):
    def test_overlay_is_reserved(self):
        frames = ch.noise(10, 64, 48, seed=5)
        plain = mi.render_model_input(frames, {"fps": 5.}, mi.ModelInputConfig())
        explicit = mi.render_model_input(frames, {"fps": 5.}, mi.ModelInputConfig(), overlay=None)
        self.assertEqual(hashes(plain.frames), hashes(explicit.frames))
        self.assertEqual(plain.record(), explicit.record())
        with self.assertRaises(NotImplementedError):
            mi.render_model_input(frames, {"fps": 5.}, mi.ModelInputConfig(), overlay={"P1": (0, 0, 10, 10)})

    def test_one_box_per_frame(self):
        with self.assertRaises(ValueError):
            mi.render_model_input(ch.noise(3, 64, 48, seed=6), {"fps": 5., "crops": [(0, 0, 9, 9)]},
                                  mi.ModelInputConfig())

    def test_the_input_frames_are_not_changed(self):
        frames = ch.noise(10, 64, 48, seed=7)
        before = hashes(frames)
        mi.render_model_input(frames, {"fps": 5., "crops": [(1, 2, 40, 41)] * 9 + [(0, 0, 30, 30)]},
                              mi.ModelInputConfig(sample_fps=5.))
        self.assertEqual(hashes(frames), before)

    def test_jpeg_is_the_backends_encoding(self):
        frame = ch.noise(1, 64, 48, seed=8)[0]
        self.assertEqual(mi.encode_jpeg(frame), inf.frame_to_jpeg_bytes(frame))
        self.assertEqual(hashlib.sha256(mi.encode_jpeg(frame)).hexdigest(),
                         hashlib.sha256(inf._jpegs([frame])[0]).hexdigest())

    def test_vendorable_imports_only_numpy_and_cv2(self):
        tree = ast.parse(Path(mi.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "no relative imports: the file is copied verbatim")
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported - {"__future__", "dataclasses", "typing"}, {"cv2", "numpy"})


if __name__ == "__main__":
    unittest.main()
