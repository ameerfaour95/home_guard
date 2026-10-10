"""A foot point in a GAP between the owner's areas (live bug 2026-10-10, ch2 17:06: a person on the neighbour's
stairs fell between two SAM polygons, the ground was unknown, and the alert went out)."""
from __future__ import annotations

import unittest

from home_guard_project.box import ground as gr
from home_guard_project.box import scene_map as sm
from home_guard_project.box.tracker import CameraTracker

# Shaped like ch2: the boundary runs up to the right, ours (the side passage) below it, the neighbour's stairs
# above it, and the owner's polygons leave thin gaps along the line.
LINE = sm.Line("the boundary 2", (0.557, 1.0), (0.990, 0.521), inward="right")
OURS = sm.Area("side passage", sm.MINE, "yard", ((0.68, 1.0), (0.99, 0.66), (0.99, 1.0)))
STAIRS = sm.Area("their stairs", sm.WATCH, "other", ((0.35, 0.80), (0.75, 0.35), (0.85, 0.45), (0.50, 0.92)),
                 owner="neighbour")
OUR_CORNER = sm.Area("our corner", sm.MINE, "yard", ((0.05, 0.05), (0.20, 0.05), (0.20, 0.20), (0.05, 0.20)))
THEIR_SIDE = sm.plain_name(sm.SIDE_NAMES["neighbour"])          # "the neighbour’s side"
CH2 = sm.SceneMap("ameer_v2_ch2", areas=(OURS, STAIRS, OUR_CORNER), lines=(LINE,))


class GroundAtTest(unittest.TestCase):
    def test_inside_an_area_is_unchanged(self):
        self.assertEqual(CH2.ground_at((0.9, 0.95))[:2], ("mine", "area"))
        self.assertEqual(CH2.ground_at((0.6, 0.6))[:2], ("neighbour", "area"))
        self.assertEqual(CH2.place_at((0.6, 0.6)), STAIRS)

    def test_a_gap_point_next_to_the_neighbours_area_is_the_neighbours(self):
        ground, how, area = CH2.ground_at((0.86, 0.47))           # a sliver from the stairs, far side of the line
        self.assertEqual((ground, how, area), ("neighbour", "near_area", STAIRS))

    def test_a_gap_point_by_a_line_counts_when_a_nearby_area_agrees(self):
        ground, how, area = CH2.ground_at((0.84, 0.56))           # 0.06 from the stairs, on their side of the line
        self.assertEqual((ground, how), ("neighbour", "line"))
        self.assertEqual((area.name, area.ground, area.implicit), (THEIR_SIDE, "neighbour", True))
        self.assertEqual(CH2.ground_at((0.90, 0.70))[:2], ("mine", "line"))      # our side, near our passage

    def test_our_yard_on_the_far_side_of_the_line_is_not_the_neighbours(self):
        # Beyond the line's far side geometrically, but our own corner is the area close by: they disagree.
        self.assertEqual(CH2.ground_at((0.26, 0.12))[:2], ("", "unknown"))

    def test_a_boundary_wall_near_the_point_neither_agrees_nor_disagrees(self):
        # The real 17:06 case: the last foot points came close to the wall the owner called the railing between
        # them (ours, zone fence) while their stairs were a little further away, on the same side of the line.
        wall = sm.Area("fence", sm.MINE, "fence", ((0.63, 0.93), (0.66, 0.90), (0.68, 0.92), (0.65, 0.95)))
        scene = sm.SceneMap("ameer_v2_ch2", areas=(OURS, STAIRS, wall), lines=(LINE,))
        p = (0.58, 0.95)
        self.assertLess(sm.NEAR_AREA, sm.polygon_distance(p, wall.points))
        self.assertLess(sm.polygon_distance(p, wall.points), sm.polygon_distance(p, STAIRS.points))
        self.assertEqual(CH2.ground_at(p)[:2], ("neighbour", "line"))       # without the wall: the same
        self.assertEqual(scene.ground_at(p)[:2], ("neighbour", "line"))

    def test_far_from_every_area_stays_unknown(self):
        self.assertEqual(CH2.ground_at((0.10, 0.90))[:2], ("", "unknown"))

    def test_without_lines_as_before_plus_the_near_area(self):
        no_lines = sm.SceneMap("cam", areas=(OURS, STAIRS))
        self.assertEqual(no_lines.ground_at((0.84, 0.56))[:2], ("", "unknown"))       # no line: no side
        self.assertEqual(no_lines.ground_at((0.86, 0.47))[:2], ("neighbour", "near_area"))
        rest = sm.SceneMap("cam", areas=(OURS,), rest=sm.WATCH, rest_owner="neighbour")
        self.assertEqual(rest.ground_at((0.10, 0.10))[:2], ("neighbour", "rest"))
        self.assertEqual(rest.ground_at((0.66, 0.99))[:2], ("mine", "near_area"))     # a sliver of ours wins

    def test_outside_todays_drawn_zone_is_black(self):
        zoned = sm.SceneMap("cam", areas=(STAIRS,), watched=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
        self.assertEqual(zoned.ground_at((0.9, 0.9))[:2], ("", "unknown"))


REST_CH2 = sm.SceneMap("ameer_v2_ch2", areas=(OURS, STAIRS, OUR_CORNER), lines=(LINE,), rest=sm.WATCH,
                       rest_owner="neighbour")


class RestStepTest(unittest.TestCase):
    """Step 4 (home-guard-15's acceptance condition): with the rest set, a point beyond the nearest line counts as the
    rest's ground when no area within AGREE_DISTANCE says otherwise."""

    def test_beyond_the_line_and_far_from_every_area_is_the_rest(self):
        self.assertEqual(CH2.ground_at((0.10, 0.90))[:2], ("", "unknown"))           # without the rest: unknown
        self.assertEqual(REST_CH2.ground_at((0.10, 0.90))[:2], ("neighbour", "rest"))

    def test_the_risk_case_stays_unknown_even_with_the_rest_set(self):
        # A gap in our yard, beyond the nearest line, next to an area of ours: never the neighbour's.
        self.assertEqual(REST_CH2.ground_at((0.26, 0.12))[:2], ("", "unknown"))

    def test_on_our_side_of_the_line_the_rest_never_applies(self):
        p = (0.95, 0.85)                                       # on our side of the line, far from every area
        self.assertEqual(sm._side(LINE.a, LINE.b, p), LINE.inward)
        far_from_all = sm.SceneMap("cam", areas=(OUR_CORNER,), lines=(LINE,), rest=sm.WATCH, rest_owner="neighbour")
        self.assertEqual(far_from_all.ground_at(p)[:2], ("", "unknown"))

    def test_an_agreeing_area_still_names_the_line_step(self):
        self.assertEqual(REST_CH2.ground_at((0.84, 0.56))[:2], ("neighbour", "line"))
        self.assertEqual(REST_CH2.ground_at((0.90, 0.70))[:2], ("mine", "line"))

    def test_a_public_rest_makes_the_far_side_public(self):
        public = sm.SceneMap("cam", areas=(OURS,), lines=(LINE,), rest=sm.WATCH, rest_owner="public")
        self.assertEqual(public.ground_at((0.10, 0.90))[:2], ("public", "rest"))

    def test_no_lines_and_the_rest_set_is_the_rest_as_before(self):
        rest = sm.SceneMap("cam", areas=(OURS,), rest=sm.WATCH, rest_owner="neighbour")
        self.assertEqual(rest.ground_at((0.10, 0.10))[:2], ("neighbour", "rest"))

    def test_unmapped_is_unchanged(self):
        self.assertEqual(sm.SceneMap("cam").ground_at((0.5, 0.5))[:2], ("", "unknown"))


class GapTrackTest(unittest.TestCase):
    def walk_in_the_gap(self):
        return [sm.Track("person", [(i * 0.5, 0.86 + 0.002 * (i % 2), 0.47) for i in range(10)])]

    def test_a_person_standing_in_the_gap_by_their_stairs_is_off_our_ground(self):
        g = gr.ground_of(self.walk_in_the_gap(), CH2)
        self.assertEqual((g.on, g.off_our_ground), ("neighbour", True))
        self.assertEqual(g.record()["placed"], {"near_area": 10})

    def test_the_tracker_names_where_they_stood_even_in_a_gap(self):
        tracker = CameraTracker("ameer_v2_ch2")
        for i in range(12):
            x = 0.84 + 0.002 * (i % 2)
            tracker.update(i * 0.5, [(0, 0.9, x - 0.04, 0.36, x + 0.04, 0.56)], scene_map=CH2)
        facts = tracker.facts(0.0, 6.0, scene_map=CH2)
        person = facts.people[0]
        self.assertEqual(person.path, (THEIR_SIDE,))
        self.assertTrue(dict(person.seconds_per_area).get(THEIR_SIDE, 0) >= 4)
        self.assertTrue(dict(person.seconds_per_ground).get("neighbour", 0) >= 4)      # seconds per ground
        self.assertIn("seconds_per_ground", person.record())

    def test_crossing_inward_is_unchanged(self):
        walk = [sm.Track("person", [(i * 0.5, 0.60 + 0.03 * i, 0.60 + 0.035 * i) for i in range(12)])]
        self.assertTrue(gr.ground_of(walk, CH2).crossed_inward)


if __name__ == "__main__":
    unittest.main()
