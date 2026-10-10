from __future__ import annotations

import unittest

from home_guard_project.box import ground as gr
from home_guard_project.box import scene_map as sm

LEFT = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
RIGHT = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
# The railing down the middle; ours (the yard) is on the left as seen on the picture.
RAILING = sm.Line("railing", (0.5, 0.0), (0.5, 1.0), inward="right")


def yard_and_neighbour(lines=(RAILING,)) -> sm.SceneMap:
    return sm.SceneMap("cam", areas=(sm.Area("yard", sm.MINE, "yard", LEFT),
                                     sm.Area("their driveway", sm.WATCH, "parking", RIGHT, owner="neighbour")),
                       lines=tuple(lines))


def walk(xs, y=0.6, t0=0.0, step=0.5, kind="person"):
    return sm.Track(kind, [(t0 + i * step, x, y) for i, x in enumerate(xs)])


class GroundOfTest(unittest.TestCase):
    def test_no_map_or_no_people_says_nothing(self) -> None:
        self.assertEqual(gr.ground_of([walk([0.8] * 6)], None), gr.UNKNOWN)
        self.assertEqual(gr.ground_of([walk([0.8] * 6)], sm.SceneMap("cam", watched=LEFT)), gr.UNKNOWN)
        self.assertEqual(gr.ground_of([], yard_and_neighbour()), gr.UNKNOWN)
        self.assertEqual(gr.ground_of([walk([0.8] * 6, kind="vehicle")], yard_and_neighbour()), gr.UNKNOWN)
        self.assertEqual(gr.UNKNOWN.on, "")
        self.assertFalse(gr.UNKNOWN.entered)

    def test_a_person_who_stays_on_the_neighbours_ground(self) -> None:
        g = gr.ground_of([walk([0.8, 0.82, 0.85, 0.8, 0.78, 0.8])], yard_and_neighbour())
        self.assertEqual((g.on, g.crossed_inward, g.entered_from, g.line), ("neighbour", False, "", ""))
        self.assertTrue(g.off_our_ground)

    def test_crossing_the_railing_inward(self) -> None:
        g = gr.ground_of([walk([0.8, 0.75, 0.7, 0.6, 0.55, 0.45, 0.4, 0.35, 0.3])], yard_and_neighbour())
        self.assertEqual(g.on, "mine")
        self.assertTrue(g.crossed_inward)
        self.assertEqual(g.line, "railing")
        self.assertEqual(g.entered_from, "their driveway")
        self.assertTrue(g.entered)
        self.assertFalse(g.off_our_ground)

    def test_entering_our_area_from_the_neighbours_without_a_line(self) -> None:
        g = gr.ground_of([walk([0.8, 0.75, 0.7, 0.6, 0.55, 0.45, 0.4, 0.35, 0.3])], yard_and_neighbour(lines=()))
        self.assertFalse(g.crossed_inward)
        self.assertEqual(g.entered_from, "their driveway")
        self.assertTrue(g.entered)

    def test_walking_out_is_not_entering(self) -> None:
        g = gr.ground_of([walk([0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.7, 0.75, 0.8])], yard_and_neighbour())
        self.assertFalse(g.entered)
        self.assertEqual(g.on, "mine")                  # was on our ground: never "off our ground"

    def test_a_wobble_on_the_line_is_not_a_crossing(self) -> None:
        g = gr.ground_of([walk([0.52, 0.49, 0.52, 0.53, 0.52, 0.54])], yard_and_neighbour())
        self.assertFalse(g.entered)

    def test_one_person_the_map_cannot_place_keeps_it_unknown(self) -> None:
        scene = sm.SceneMap("cam", areas=(sm.Area("road", sm.WATCH, "street", RIGHT),))
        g = gr.ground_of([walk([0.8] * 6), walk([0.2] * 6)], scene)
        self.assertEqual(g.on, "")
        self.assertFalse(g.off_our_ground)

    def test_mostly_off_our_ground_counts_as_off(self) -> None:
        scene = sm.SceneMap("cam", areas=(sm.Area("road", sm.WATCH, "street", RIGHT),))
        g = gr.ground_of([walk([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2])], scene)
        self.assertEqual(g.on, "public")
        g = gr.ground_of([walk([0.8, 0.8, 0.8, 0.2, 0.2, 0.2])], scene)
        self.assertEqual(g.on, "")

    def test_the_rest_of_a_confirmed_map_is_the_neighbours(self) -> None:
        scene = sm.SceneMap("cam", areas=(sm.Area("yard", sm.MINE, "yard", LEFT),), rest=sm.WATCH,
                            rest_owner="neighbour", confirmed=1.0)
        g = gr.ground_of([walk([0.8, 0.75, 0.7, 0.6, 0.55, 0.45, 0.4, 0.35, 0.3])], scene)
        self.assertEqual(g.entered_from, sm.REST_NAME)
        self.assertEqual(g.on, "mine")

    def test_the_record(self) -> None:
        g = gr.ground_of([walk([0.8] * 6)], yard_and_neighbour())
        self.assertEqual(g.record(), {"on": "neighbour", "crossed_inward": False, "line": "", "entered_from": "",
                                      "from_ground": "", "people": 1, "entered": False, "off_our_ground": True,
                                      "placed": {"area": 6}})

    def test_a_line_crossed_from_a_place_the_map_does_not_name(self) -> None:
        scene = sm.SceneMap("cam", areas=(sm.Area("yard", sm.MINE, "yard", LEFT),), lines=(RAILING,))
        g = gr.ground_of([walk([0.8, 0.75, 0.7, 0.6, 0.55, 0.45, 0.4, 0.35, 0.3])], scene)
        self.assertTrue(g.crossed_inward)
        self.assertEqual((g.entered_from, g.from_ground), ("", ""))
        self.assertEqual(gr.entered_text(g, "he"), "נכנס אל השטח שלנו דרך railing")


class ActionTest(unittest.TestCase):
    def test_actions_are_what_hands_and_feet_do_not_where_or_when(self) -> None:
        for text in ("tries the car door handle", "climbs the fence", "peers into the window", "מנסה לפתוח את הדלת",
                     "takes a bag from the car", "crouching behind the car"):
            self.assertTrue(gr.is_action(text), text)
        for text in ("a man walks at night", "a person stands near the entrance", "loitering by the car",
                     "אדם הולך בלילה", "wears a mask", ""):
            self.assertFalse(gr.is_action(text), text)


class TextTest(unittest.TestCase):
    def test_the_owner_reads_where_they_came_from(self) -> None:
        g = gr.Ground(on="mine", crossed_inward=True, line="המעקה", entered_from="their driveway",
                      from_ground="neighbour", people=1)
        self.assertEqual(gr.entered_text(g, "he"), "נכנס מהצד של השכן אל השטח שלנו (דרך המעקה)")
        self.assertEqual(gr.entered_text(g, "en"), "came in from the neighbour's side onto our ground (across המעקה)")
        public = gr.Ground(on="mine", entered_from="road", from_ground="public", people=1)
        self.assertEqual(gr.entered_text(public, "he"), "נכנס מהרחוב אל השטח שלנו")
        self.assertEqual(gr.entered_text(gr.UNKNOWN, "he"), "")


if __name__ == "__main__":
    unittest.main()
