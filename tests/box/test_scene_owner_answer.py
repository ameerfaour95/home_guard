"""The owner's real camera-1 answer of 2026-10-08 22:55 (the regions SAM numbered on the box; the picture stays there).

Bugs it showed: "4 המעקה ביני לבין השכן" became a neighbour's AREA (it is the railing between us: ours, and a
boundary inward to us); "7 גם" (the same) was dropped; and nothing told the owner that 7 was not understood.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from home_guard_project.box import scene_interview as si
from home_guard_project.box import scene_map as sm

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
REGIONS = os.path.join(FIXTURES, "scene_ch1_regions.json")
CAM = "ameer_week_0_1_ch1"
ANSWER = "2 שלי 1 שלי 3 שלי 4 המעקה ביני לבין השכן 7 גם 6 שלי 9 של השכן 5 שלי 11 שלי"


class OwnerAnswerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.zones = os.path.join(self.tmp, "zones.yaml")
        self.regions_path = os.path.join(self.tmp, "regions.json")
        shutil.copy(REGIONS, self.regions_path)

    def test_the_railing_and_its_same_are_boundaries_not_the_neighbours_ground(self):
        answers = {a.number: a for a in si.parse_answers(ANSWER)}
        self.assertEqual(sorted(answers), [1, 2, 3, 4, 5, 6, 7, 9, 11])
        self.assertEqual((answers[4].kind, answers[4].name), (si.BOUNDARY, "המעקה"))
        self.assertEqual((answers[7].kind, answers[7].name), (si.BOUNDARY, "המעקה"))       # "7 גם"
        self.assertEqual((answers[9].kind, answers[9].owner), (sm.WATCH, sm.NEIGHBOUR))
        for n in (1, 2, 3, 5, 6, 11):
            self.assertEqual(answers[n].kind, sm.MINE, n)

    def test_the_draft_keeps_the_railing_ours_with_a_line_inward_to_us(self):
        result = si.answer(CAM, ANSWER, self.regions_path, None, self.tmp, self.zones)
        scene = sm.SceneMap.from_dict(CAM, result["map"])
        regions = {r.number: r for r in si.load_regions(self.regions_path, CAM)}
        rail = next(a for a in scene.areas if a.points == regions[4].points)
        self.assertEqual((rail.kind, rail.zone), (sm.MINE, "fence"))            # never a neighbour's area
        self.assertFalse(any(a.kind == sm.WATCH and a.zone == "fence" for a in scene.areas))
        lines = {ln.name: ln for ln in scene.lines}
        self.assertIn("המעקה 4", lines)
        line = lines["המעקה 4"]
        # From the neighbour's house (9) onto our garden bed (2) is inward; the other way is outward.
        nine, two = si._inside_point(regions[9].points), si._inside_point(regions[2].points)
        self.assertEqual(line.crossing(nine, two), sm.IN)
        self.assertEqual(result["not_understood"], [])
        self.assertTrue(all(n in {r for r in regions} for n in (4, 7)))
        seven = [ln for name, ln in lines.items() if name.endswith("7")]
        if seven:                                       # 7 is a line too, or the owner is asked which side
            line7 = seven[0]                            # short: its sides, not a crossing, show where ours is
            self.assertEqual(sm._side(line7.a, line7.b, two), line7.inward)
            self.assertNotEqual(sm._side(line7.a, line7.b, nine), line7.inward)
        else:
            self.assertEqual([s["number"] for s in result["sides_needed"]], [7])

    def test_a_number_that_made_nothing_is_said(self):
        result = si.answer(CAM, "2 שלי 1 שלי 7 בבקשה", self.regions_path, None, self.tmp, self.zones)
        self.assertEqual(result["not_understood"], [7])
        self.assertIn("I did not understand 7", result["notes"])

    def test_an_unclear_side_is_asked_never_guessed(self):
        # Only the railing, nothing else: nobody can tell which side is ours.
        result = si.answer(CAM, "4 המעקה ביני לבין השכן", self.regions_path, None, self.tmp, self.zones)
        self.assertEqual([s["number"] for s in result["sides_needed"]], [4])
        self.assertEqual(sm.SceneMap.from_dict(CAM, result["map"]).lines, ())
        again = si.answer(CAM, "4 המעקה ביני לבין השכן", self.regions_path, None, self.tmp, self.zones,
                          sides={4: "left"})
        self.assertEqual(again["sides_needed"], [])
        self.assertEqual([ln.inward for ln in sm.SceneMap.from_dict(CAM, again["map"]).lines], ["left"])

    def test_the_stage_2c_draft_of_that_answer_still_loads(self):
        with open(os.path.join(FIXTURES, "scene_ch1_draft_2c.json"), encoding="utf-8") as f:
            data = json.load(f)
        scene = sm.SceneMap.from_dict(CAM, data)
        self.assertIn("המעקה ביני לבין השכן", [a.name for a in scene.areas])     # the bug, as it was saved

    def test_side_words_follow_the_picture(self):
        self.assertEqual(si.side_words((0.5, 0.0), (0.5, 1.0)), {"left": "הצד הימני", "right": "הצד השמאלי"})
        self.assertEqual(si.side_words((0.0, 0.5), (1.0, 0.5)), {"left": "הצד העליון", "right": "הצד התחתון"})


if __name__ == "__main__":
    unittest.main()
