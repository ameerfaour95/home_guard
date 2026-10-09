"""The stage-3 tag benchmark's plumbing: human tracks land on the box's sent frames, arm A is the box's input."""
import json
import os
import tempfile
import unittest

import numpy as np

from home_guard_project.analysis import tag_bench as tb
from home_guard_project.box import inference as inf
from home_guard_project.data_collection import model_input as mi


def export(tracks, fps=10.0, frames=100):
    """A one-task Label Studio export: *tracks* = [(label, [(frame, x, y, w, h, enabled)])], LS percent coords."""
    result = []
    for label, kfs in tracks:
        seq = [{"frame": f, "x": x, "y": y, "width": w, "height": h, "enabled": en, "time": f / fps}
               for f, x, y, w, h, en in kfs]
        result.append({"type": "videorectangle", "value": {"labels": [label], "sequence": seq,
                                                            "framesCount": frames, "duration": frames / fps}})
    return [{"id": 7, "data": {"meta_path": "meta/cam/2026-01-01/cam_1_trigger.meta.json", "fps": fps,
                               "duration_sec": frames / fps},
             "annotations": [{"result": result, "completed_by": 1}]}]


def parsed(tracks, **kw):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export(tracks, **kw), f)
        return tb.parse_export(path, tb._cfg())[0]


class TracksOnFramesTests(unittest.TestCase):
    def test_a_30fps_clip_maps_to_the_10fps_label_studio_frames(self):
        # UCF-style: the video is 30 fps, Label Studio ran at 10; a box held from LS frame 1 to 100.
        pt = parsed([("person", [(1, 10, 20, 30, 40, True), (100, 10, 20, 30, 40, True)])])
        ents = tb.entities_for(pt, 30.0, (320, 240))
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0].kind, "person")
        np.testing.assert_allclose(ents[0].box_at(0), (32.0, 48.0, 128.0, 144.0))
        np.testing.assert_allclose(ents[0].box_at(150), (32.0, 48.0, 128.0, 144.0))
        self.assertIsNone(ents[0].box_at(400))           # past the labelled frames

    def test_a_hidden_keyframe_hides_the_entity(self):
        pt = parsed([("car", [(1, 0, 0, 10, 10, True), (5, 0, 0, 10, 10, False), (20, 0, 0, 10, 10, True)])],
                    fps=7.0, frames=30)
        ent = tb.entities_for(pt, 7.0, (700, 500))[0]
        self.assertIsNotNone(ent.box_at(0))
        self.assertIsNone(ent.box_at(10))
        self.assertIsNotNone(ent.box_at(25))

    def test_animals_are_not_tagged(self):
        pt = parsed([("dog", [(1, 0, 0, 10, 10, True)]), ("truck", [(1, 0, 0, 10, 10, True)])])
        self.assertEqual([e.kind for e in tb.entities_for(pt, 10.0, (100, 100))], ["truck"])


class ArmsTests(unittest.TestCase):
    def test_arm_a_is_the_box_input_byte_for_byte(self):
        rng = np.random.default_rng(3)
        frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(30)]
        sent = mi.render_model_input(frames, {"fps": 10.0}, mi.ModelInputConfig(1.0))
        pt = parsed([("person", [(1, 10, 20, 30, 40, True), (30, 50, 20, 30, 40, True)]),
                     ("person", [(1, 60, 20, 10, 40, True), (30, 60, 20, 10, 40, True)])], frames=30)
        ents = tb.entities_for(pt, 10.0, (320, 240))
        tagged = tb.to.tag_model_input(sent, ents)
        r = tb.Rendered("c", sent, tagged, None, ("P1", "P2"), {"P1": "person", "P2": "person"}, {})
        box = mi.render_model_input(frames, {"fps": 10.0}, mi.ModelInputConfig(1.0))
        self.assertEqual([f.tobytes() for f in tb.frames_of("A", r)], [f.tobytes() for f in box.frames])
        self.assertEqual(tagged.ids, {"t0": "P1", "t1": "P2"})
        self.assertTrue(all(not np.array_equal(a, b) for a, b in zip(tb.frames_of("B", r), box.frames)))

    def test_arm_a_prompt_is_the_legacy_prompt_and_b_only_adds(self):
        clip = tb.Clip({"camera": "front_side", "source": "house"}, None, {"clip_start_local": "2026-02-21 20:01:27"})
        a, fmt_a = tb.prompt_of("A", clip, ["P1", "CAR1"])
        self.assertEqual(a, inf.build_prompt("front_side", 0, "20:01:27", 0, 0, owner_language="en"))
        self.assertIs(fmt_a, inf.VLM_RESPONSE_FORMAT)
        b, fmt_b = tb.prompt_of("B", clip, ["CAR1", "P1"])
        self.assertTrue(b.startswith(a + "\n\n"))
        self.assertIn("Tags in these frames: P1, CAR1.", b)
        self.assertIn("per_entity", fmt_b["json_schema"]["schema"]["required"])

    def test_crime_category_cameras_are_never_named(self):
        self.assertEqual(tb.camera_of({"camera": "Abuse", "source": "uca"}), tb.GENERIC_CAMERA)

    def test_lexical_swap_sees_actions_follow_the_tags(self):
        b = {"P1": "opens the car door", "P2": "stands by the gate smoking"}
        self.assertEqual(tb.lexical_swap(("P1", "P2"), b, {"P2": "opens the car door", "P1": "smoking at the gate"}),
                         "followed")
        self.assertEqual(tb.lexical_swap(("P1", "P2"), b, dict(b)), "ignored")
        self.assertEqual(tb.lexical_swap(("P1", "P2"), {"P1": "walks", "P2": "walks"}, {"P1": "walks", "P2": "walks"}),
                         "indistinct")
        self.assertEqual(tb.lexical_swap(("P1", "P2"), b, {"P1": "walks"}), "indistinct")

    def test_swap_pair_prefers_people(self):
        self.assertEqual(tb.swap_pair(["P1", "P2", "CAR1", "CAR2"]), ("P1", "P2"))
        self.assertEqual(tb.swap_pair(["P1", "CAR1", "CAR2"]), ("CAR1", "CAR2"))
        self.assertIsNone(tb.swap_pair(["P1", "CAR1"]))


if __name__ == "__main__":
    unittest.main()
