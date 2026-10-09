"""The stage-3 benchmark's Set-of-Mark overlay (analysis/tag_overlay.py) and that the box's input is untouched."""
import unittest

import numpy as np

from home_guard_project.analysis import tag_overlay as to
from home_guard_project.data_collection import model_input as mi


def walker(x0, dx, y=100, w=40, h=120, start=0, stop=10 ** 6):
    """A box moving right by *dx* pixels a frame, visible on clip frames [start, stop)."""
    return lambda i: (x0 + dx * i, y, x0 + dx * i + w, y + h) if start <= i < stop else None


def clip(n=30, w=320, h=240, value=60):
    return [np.full((h, w, 3), value, np.uint8) for _ in range(n)]


class IdsTests(unittest.TestCase):
    def setUp(self):
        frames = clip()
        self.sent = mi.render_model_input(frames, {"fps": 5.0}, mi.ModelInputConfig(1.0))   # clip frames 0,5,..,25
        # "late" enters at clip frame 10 on the left; "early" walks from the start; the car is parked.
        self.entities = [to.Entity("late", "person", walker(10, 0, start=10)),
                         to.Entity("early", "person", walker(150, 2)),
                         to.Entity("car", "car", lambda i: (200, 150, 300, 230))]

    def test_ids_follow_first_appearance_and_stay_on_their_entity(self):
        tagged = to.tag_model_input(self.sent, self.entities)
        self.assertEqual(tagged.ids, {"early": "P1", "car": "CAR1", "late": "P2"})
        for k, marks in enumerate(tagged.marks):
            by_id = {m.id: m for m in marks}
            self.assertEqual(by_id["P1"].box[0], 150 + 2 * self.sent.frame_indices[k])
            self.assertEqual(by_id["CAR1"].box, (200, 150, 300, 230))
            if self.sent.frame_indices[k] >= 10:
                self.assertEqual(by_id["P2"].box[0], 10)
            else:
                self.assertNotIn("P2", by_id)
        colors = {m.id: m.color for marks in tagged.marks for m in marks}
        for marks in tagged.marks:
            for m in marks:
                self.assertEqual(m.color, colors[m.id])     # one colour per id in every frame
        self.assertEqual(len(set(colors.values())), 3)

    def test_ties_go_left_to_right(self):
        ents = [to.Entity("b", "person", walker(200, 0)), to.Entity("a", "person", walker(20, 0))]
        self.assertEqual(to.tag_model_input(self.sent, ents).ids, {"a": "P1", "b": "P2"})

    def test_the_swap_changes_only_the_text(self):
        plain = to.tag_model_input(self.sent, self.entities)
        swapped = to.tag_model_input(self.sent, self.entities, relabel={"P1": "P2", "P2": "P1"})
        self.assertEqual(swapped.ids, plain.ids)
        last_plain = {m.id: m for m in plain.marks[-1]}
        last_swap = {m.id: m for m in swapped.marks[-1]}
        self.assertEqual(last_swap["P1"].label, "P2")
        self.assertEqual(last_swap["P2"].label, "P1")
        self.assertEqual(last_swap["P1"].box, last_plain["P1"].box)
        self.assertEqual(last_swap["P1"].color, last_plain["P1"].color)
        self.assertFalse(np.array_equal(swapped.frames[-1], plain.frames[-1]))

    def test_outline_only_no_fill(self):
        tagged = to.tag_model_input(self.sent, [to.Entity("p", "person", lambda i: (100, 60, 220, 200))],
                                    stamp=False)
        img = tagged.frames[0]
        self.assertTrue((img[130, 160] == 60).all())          # the inside of the box is untouched
        self.assertFalse((img[130, 100] == 60).all())         # the outline is drawn


class CropCoordsTests(unittest.TestCase):
    def test_boxes_are_drawn_in_crop_pixels(self):
        frames = clip(10, 640, 480)
        crops = [(100, 50, 420, 290)] * 10                      # 320 x 240 cut, resized to 160 x 120
        sent = mi.render_model_input(frames, {"fps": 5.0, "crops": crops, "crop_size": (160, 120)},
                                     mi.ModelInputConfig(1.0))
        self.assertEqual(sent.frames[0].shape[:2], (120, 160))
        tagged = to.tag_model_input(sent, [to.Entity("p", "person", lambda i: (200, 90, 300, 250))], stamp=False)
        self.assertEqual(tagged.marks[0][0].box, (50, 20, 100, 100))    # (x - 100) / 2, (y - 50) / 2

    def test_a_box_outside_the_crop_is_not_drawn(self):
        frames = clip(5, 640, 480)
        sent = mi.render_model_input(frames, {"fps": 5.0, "crops": [(0, 0, 200, 200)] * 5},
                                     mi.ModelInputConfig(1.0))
        tagged = to.tag_model_input(sent, [to.Entity("p", "person", lambda i: (400, 300, 500, 450))])
        self.assertEqual(tagged.ids, {})
        self.assertEqual(tagged.marks, [[]])

    def test_to_frame_coords_clips_to_the_frame(self):
        self.assertEqual(to.to_frame_coords((-10, -10, 50, 50), None, (100, 100)), (0, 0, 50, 50))
        self.assertIsNone(to.to_frame_coords((120, 10, 150, 50), None, (100, 100)))


class NothingToDrawTests(unittest.TestCase):
    def test_no_tracks_draws_nothing_but_the_stamp(self):
        frames = clip(10)
        sent = mi.render_model_input(frames, {"fps": 5.0}, mi.ModelInputConfig(1.0))
        bare = to.tag_model_input(sent, [], stamp=False)
        self.assertEqual(bare.ids, {})
        for a, b in zip(bare.frames, sent.frames):
            self.assertTrue(np.array_equal(a, b))
        stamped = to.tag_model_input(sent, [])
        self.assertFalse(np.array_equal(stamped.frames[0], sent.frames[0]))
        self.assertTrue((stamped.frames[0][:120] == 60).all())   # the stamp stays in the bottom corner

    def test_drawing_never_touches_the_box_frames(self):
        frames = clip(10)
        sent = mi.render_model_input(frames, {"fps": 5.0}, mi.ModelInputConfig(1.0))
        before = [f.copy() for f in sent.frames]
        to.tag_model_input(sent, [to.Entity("p", "person", walker(10, 3))])
        for a, b in zip(before, sent.frames):
            self.assertTrue(np.array_equal(a, b))


class BoxInputUnchangedTests(unittest.TestCase):
    def test_the_box_function_still_refuses_an_overlay(self):
        with self.assertRaises(NotImplementedError):
            mi.render_model_input(clip(5), {"fps": 5.0}, mi.ModelInputConfig(1.0), overlay=object())
        self.assertEqual(mi.MODEL_INPUT_VERSION, "model-input-1")


if __name__ == "__main__":
    unittest.main()
