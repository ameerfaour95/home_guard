from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from home_guard_project.box import scene_interview as si
from home_guard_project.box import scene_map as sm
from home_guard_project.data_collection import zones as z


def picture(h: int = 120, w: int = 160) -> np.ndarray:
    img = np.full((h, w, 3), 128, np.uint8)
    img[60:, :80] = (40, 160, 40)          # a lawn, bottom left
    img[70:110, 100:150] = (200, 200, 200)  # a car, right
    return img


def rect_mask(h: int, w: int, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    m[y1:y2, x1:x2] = True
    return m


def two_objects(image):
    h, w = image.shape[:2]
    return [rect_mask(h, w, 70, 110, 100, 150), rect_mask(h, w, 60, 120, 0, 80),
            rect_mask(h, w, 0, 2, 0, 2),                       # a speck: too small to ask about
            rect_mask(h, w, 61, 120, 0, 80)]                   # the lawn again: a duplicate


class RegionsTest(unittest.TestCase):
    def test_masks_become_numbered_polygons_largest_first(self) -> None:
        regions, method = si.propose_regions(picture(), segmenter=two_objects)
        self.assertEqual(method, "sam")
        self.assertEqual([r.number for r in regions], [1, 2])
        lawn, car = regions
        self.assertAlmostEqual(lawn.area, 0.25, places=2)
        self.assertTrue(sm.inside((0.25, 0.75), lawn.points))
        self.assertTrue(sm.inside((0.78, 0.75), car.points))
        self.assertLessEqual(len(lawn.points), z.MAX_POINTS)

    def test_without_a_segmenter_it_falls_back_to_a_grid_and_says_so(self) -> None:
        def broken(image):
            raise RuntimeError("no weights offline")

        with self.assertLogs(si.log, level="WARNING"):
            regions, method = si.propose_regions(picture(), segmenter=broken)
        self.assertEqual(method, "grid")
        self.assertEqual(len(regions), si.GRID_ROWS * si.GRID_COLS)
        self.assertEqual(regions[0].number, 1)

    def test_blacked_out_parts_are_never_numbered(self) -> None:
        img = picture()
        img[:, 80:] = 0                                         # outside today's zone, blacked out at frame read
        regions, _ = si.propose_regions(img, segmenter=two_objects)
        self.assertEqual(len(regions), 1)                       # the car is in the black part
        grid, _ = si.propose_regions(img, segmenter=si.grid_segmenter)
        self.assertTrue(all(r.points[0][0] < 0.5 for r in grid))

    def test_numbered_picture(self) -> None:
        img = picture()
        regions, _ = si.propose_regions(img, segmenter=two_objects)
        out = si.numbered_overlay(img, regions)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))
        self.assertTrue(np.array_equal(img, picture()))           # the caller's picture is untouched

    def test_ownership_picture_colours_by_ground(self) -> None:
        img = np.full((40, 40, 3), 128, np.uint8)
        scene = sm.SceneMap("cam", areas=(
            sm.Area("yard", sm.MINE, "yard", ((0, 0), (0.5, 0), (0.5, 1), (0, 1))),
            sm.Area("car", sm.WATCH, "car", ((0.5, 0), (1, 0), (1, 1), (0.5, 1)), owner="neighbour")),
            lines=(sm.Line("railing", (0.5, 0.0), (0.5, 1.0), "right"),))
        out = si.ownership_overlay(img, scene)
        left, right = out[20, 8].astype(int), out[20, 32].astype(int)
        self.assertGreater(left[0], left[2])                    # blue (BGR) on the owner's side
        self.assertGreater(right[2], right[0])                  # orange on the neighbour's

    def test_questions_are_short_and_capped(self) -> None:
        regions, _ = si.propose_regions(picture(), segmenter=si.grid_segmenter)
        qs = si.questions(regions)
        self.assertEqual(len(qs), si.MAX_QUESTIONS)
        self.assertTrue(all(q["options"] == list(si.OPTIONS) for q in qs))
        self.assertEqual(qs[0]["number"], 1)


class AnswersTest(unittest.TestCase):
    def test_english_answers(self) -> None:
        answers = si.parse_answers("3,5 are mine; 7 is the neighbour's car. 2 is the street\n4 hide it; 6 skip")
        got = {(a.number, a.kind, a.owner, a.zone) for a in answers}
        self.assertEqual(got, {(3, "mine", "", "other"), (5, "mine", "", "other"),
                               (7, "watch_no_alert", "neighbour", "car"), (2, "watch_no_alert", "public", "street"),
                               (4, "black", "", "other")})
        self.assertEqual(next(a for a in answers if a.number == 7).name, "neighbour’s car")
        self.assertEqual(next(a for a in answers if a.number == 4).name, "")         # "hide it" names nothing

    def test_an_apostrophe_is_not_a_quoted_name(self) -> None:
        answers = si.parse_answers("7 the neighbour's car, 8 the neighbour's gate")
        self.assertEqual([(a.number, a.zone) for a in answers], [(7, "car"), (8, "gate")])

    def test_a_comma_list_is_one_answer_per_number(self) -> None:
        answers = si.parse_answers("1 שלי, 2 של השכן, 3 רחוב, המעקה בין 2 ל-1")
        self.assertEqual([(a.number, a.kind, a.owner) for a in answers],
                         [(1, "mine", ""), (2, "watch_no_alert", "neighbour"), (3, "watch_no_alert", "public")])
        self.assertEqual(si.parse_lines("1 שלי, 2 של השכן, 3 רחוב, המעקה בין 2 ל-1"), [("המעקה", 2, 1)])
        self.assertEqual(si.parse_lines("the railing between 2 and 1; 3 ו-4 שלי"), [("railing", 2, 1)])
        self.assertEqual([a.number for a in si.parse_answers("3 ו-4 שלי")], [3, 4])

    def test_hebrew_answers(self) -> None:
        answers = si.parse_answers("1 שלי; 2 הרכב של השכן; 3 כביש; 4 להסתיר; 5 לא יודע; 6 'בית הכלב' שלי")
        got = {(a.number, a.kind, a.owner) for a in answers}
        self.assertEqual(got, {(1, "mine", ""), (2, "watch_no_alert", "neighbour"), (3, "watch_no_alert", "public"),
                               (4, "black", ""), (6, "mine", "")})
        dog = next(a for a in answers if a.number == 6)
        self.assertEqual(dog.name, "בית הכלב")
        self.assertEqual(next(a for a in answers if a.number == 2).zone, "car")

    def test_the_neighbours_driveway_is_parking_not_mine(self) -> None:
        (a,) = si.parse_answers("8 is my neighbor's driveway")
        self.assertEqual((a.kind, a.owner, a.zone), ("watch_no_alert", "neighbour", "parking"))

    def test_a_clause_without_a_kind_is_not_guessed(self) -> None:
        self.assertEqual(si.parse_answers("3 is a tree"), [])


class ApplyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.zones = os.path.join(self.tmp.name, "zones.yaml")
        self.regions, _ = si.propose_regions(picture(), segmenter=two_objects)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_answers_become_the_scene_map(self) -> None:
        scene, notes = si.apply_answers(sm.SceneMap("front"), self.regions,
                                        si.parse_answers("1 is our lawn, mine; 2 is the neighbour's car; 9 mine"))
        self.assertEqual([(a.name, a.kind, a.ground, a.zone) for a in scene.areas],
                         [("our lawn", "mine", "mine", "yard"),
                          ("neighbour’s car", "watch_no_alert", "neighbour", "car")])
        self.assertEqual(notes, ["there is no 9 in the picture"])

    def test_answering_the_same_number_again_replaces_it(self) -> None:
        scene, _ = si.apply_answers(sm.SceneMap("front"), self.regions, si.parse_answers("2 mine"))
        scene, _ = si.apply_answers(scene, self.regions, si.parse_answers("2 is the neighbour's car"))
        self.assertEqual([a.ground for a in scene.areas], ["neighbour"])

    def test_answer_makes_a_draft_and_only_confirm_saves(self) -> None:
        out = os.path.join(self.tmp.name, "interview")
        proposal = si.propose("front", picture(), out, segmenter=two_objects)
        self.assertTrue(os.path.isfile(proposal["image"]))
        self.assertEqual(proposal["method"], "sam")
        self.assertEqual([q["number"] for q in proposal["questions"]], [1, 2])
        result = si.answer("front", "1 mine; 2 hide it", proposal["regions_path"], picture(), out,
                           zones_path=self.zones)
        self.assertTrue(os.path.isfile(result["image"]))
        self.assertEqual(json.loads(json.dumps(result["map"]))["areas"][1]["kind"], "black")
        self.assertEqual(sm.load_scene_map("front", self.zones).areas, ())          # nothing saved yet
        done = si.confirm("front", out, zones_path=self.zones)
        self.assertTrue(done["restart_needed"])                   # a new black area: the readers must reload
        saved = sm.load_scene_map("front", self.zones)
        self.assertEqual(saved.areas[0].kind, "mine")
        self.assertTrue(saved.confirmed)
        self.assertEqual(len(z.load_black(z.scene_maps_path_for(self.zones))["front"]), 1)
        self.assertFalse(os.path.exists(si.draft_path("front", out)))
        with self.assertRaises(OSError):
            si.confirm("front", out, zones_path=self.zones)       # no draft left to confirm

    def test_no_mask_change_needs_no_restart(self) -> None:
        out = os.path.join(self.tmp.name, "interview")
        proposal = si.propose("front", picture(), out, segmenter=two_objects)
        result = si.answer("front", "1 mine", proposal["regions_path"], None, out, zones_path=self.zones)
        self.assertIsNone(result["image"])
        self.assertFalse(si.confirm("front", out, zones_path=self.zones)["restart_needed"])

    def test_todays_black_outside_is_previewed_as_the_neighbours_and_confirmed(self) -> None:
        z.save_zone("front", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], self.zones)
        z.save_zone("ameer_test_front", [(0, 0), (1, 0), (1, 1)], self.zones)   # no camera of this box
        out = os.path.join(self.tmp.name, "interview")
        proposal = si.propose("front", picture(), out, segmenter=two_objects,
                              watched=sm.load_scene_map("front", self.zones).watched)
        result = si.answer("front", "1 mine", proposal["regions_path"], picture(), out, zones_path=self.zones)
        self.assertEqual(result["map"]["outside"], "watch_no_alert")
        self.assertEqual(result["map"]["rest_owner"], "neighbour")
        done = si.confirm("front", out, zones_path=self.zones, known_cameras=["front"])
        self.assertTrue(done["restart_needed"])                   # the zone's black outside is gone
        self.assertEqual(z.load_zones(self.zones), {"ameer_test_front": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]})

    def test_stale_keys_share_the_channel(self) -> None:
        z.save_zone("ameer_week_0_1_ch2", [(0, 0), (1, 0), (1, 1)], self.zones)
        z.save_zone("ameer_test_ch2", [(0, 0), (1, 0), (1, 1)], self.zones)
        z.save_zone("ameer_test_ch5", [(0, 0), (1, 0), (1, 1)], self.zones)
        self.assertEqual(si.stale_keys("ameer_week_0_1_ch2", ["ameer_week_0_1_ch2", "ameer_week_0_1_ch5"],
                                       self.zones), ["ameer_test_ch2"])

    def test_a_line_between_two_regions_that_touch(self) -> None:
        a = si.Region(1, ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)), 0.5)
        b = si.Region(2, ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)), 0.5)
        line = si.line_between("railing", a, b, ours=a)
        self.assertAlmostEqual(line.a[0], 0.5, delta=0.03)
        self.assertAlmostEqual(line.b[0], 0.5, delta=0.03)
        self.assertEqual(line.crossing((0.8, 0.5), (0.2, 0.5)), "in")
        far = si.Region(3, ((0.9, 0.9), (1.0, 0.9), (1.0, 1.0)), 0.01)
        with self.assertRaises(ValueError):
            si.line_between("x", a, far, ours=a)
        with self.assertRaises(ValueError):
            si.line_between("x", a, b, ours=None)
        scene, notes = si.apply_answers(sm.SceneMap("front"), [a, b], si.parse_answers("1 שלי, 2 של השכן"),
                                        si.parse_lines("המעקה בין 2 ל-1"))
        self.assertEqual([ln.name for ln in scene.lines], ["המעקה"])
        self.assertEqual(notes, [])
        _, notes = si.apply_answers(sm.SceneMap("front"), [a, b], [], si.parse_lines("המעקה בין 2 ל-1"))
        self.assertIn("which side", notes[0])



class CommandLineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cameras = os.path.join(self.tmp.name, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            f.write("cameras:\n  front: rtsp://x\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_main(self, *argv):
        import contextlib
        import io
        from unittest import mock

        from home_guard_project.box import find_cameras

        out = io.StringIO()
        with mock.patch.object(find_cameras, "CAMERAS_PATH", self.cameras), \
                mock.patch.object(sm, "load_scene_map", return_value=sm.SceneMap("front")), \
                contextlib.redirect_stdout(out):
            code = si.main(list(argv))
        return code, json.loads(out.getvalue())

    def test_show_prints_the_map(self) -> None:
        code, data = self.run_main("show", "--camera", "front", "--json")
        self.assertEqual(code, 0)
        self.assertEqual((data["camera"], data["outside"], data["areas"]), ("front", "unmapped", []))

    def test_an_unknown_camera_is_a_plain_error(self) -> None:
        code, data = self.run_main("show", "--camera", "nope", "--json")
        self.assertEqual(code, 1)
        self.assertIn("unknown camera", data["error"])


if __name__ == "__main__":
    unittest.main()
