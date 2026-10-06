from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from home_guard_project.box import scene_map as sm
from home_guard_project.data_collection import zones as z

LEFT = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
RIGHT = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
TOP_RIGHT = [(0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (0.75, 0.25)]
# A vertical railing down the middle; the yard (mine) is on the left as seen on the picture.
RAILING = sm.Line("railing", (0.5, 0.0), (0.5, 1.0), inward="right")


def yard_and_neighbour() -> sm.SceneMap:
    return sm.SceneMap("front", areas=(
        sm.Area("yard", sm.MINE, "yard", tuple(LEFT)),
        sm.Area("neighbour driveway", sm.WATCH, "parking", tuple(RIGHT), owner="neighbour"),
    ), lines=(RAILING,))


class AreaTest(unittest.TestCase):
    def test_kinds_and_grounds(self) -> None:
        self.assertEqual(sm.KINDS, ("mine", "watch_no_alert", "black"))
        self.assertEqual(sm.Area("y", sm.MINE, "yard", tuple(LEFT)).ground, "mine")
        self.assertEqual(sm.Area("d", sm.WATCH, "parking", tuple(RIGHT), owner="neighbour").ground, "neighbour")
        self.assertEqual(sm.Area("s", sm.WATCH, "street", tuple(RIGHT)).ground, "public")
        self.assertEqual(sm.Area("w", sm.BLACK, "window", tuple(RIGHT)).ground, "")

    def test_bad_values_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            sm.Area("y", "maybe", "yard", tuple(LEFT))
        with self.assertRaises(ValueError):
            sm.Area("y", sm.MINE, "garden", tuple(LEFT))       # not a taxonomy zone
        with self.assertRaises(ValueError):
            sm.Area("y", sm.MINE, "yard", ((0, 0), (1, 1)))
        with self.assertRaises(ValueError):
            sm.Area("y", sm.WATCH, "yard", tuple(LEFT), owner="cousin")
        with self.assertRaises(ValueError):
            sm.Line("gate", (0, 0), (0, 0), inward="right")      # not a line
        with self.assertRaises(ValueError):
            sm.Line("gate", (0, 0), (1, 1), inward="up")

    def test_owner_words_become_a_plain_short_name(self) -> None:
        area = sm.Area('the "dog" house; ok\n' + "x" * 80, sm.MINE, "yard", tuple(LEFT))
        self.assertNotIn('"', area.name)
        self.assertNotIn(";", area.name)
        self.assertLessEqual(len(area.name), sm.NAME_LIMIT)


class WhereTest(unittest.TestCase):
    def test_area_at_a_foot_point(self) -> None:
        m = yard_and_neighbour()
        self.assertEqual(m.area_at((0.2, 0.9)).name, "yard")
        self.assertEqual(m.area_at((0.8, 0.9)).name, "neighbour driveway")

    def test_the_innermost_area_wins(self) -> None:
        m = sm.SceneMap("front", areas=(
            sm.Area("street", sm.WATCH, "street", tuple(RIGHT)),
            sm.Area("neighbour car", sm.WATCH, "car", tuple(TOP_RIGHT), owner="neighbour"),
        ))
        self.assertEqual(m.area_at((0.9, 0.1)).name, "neighbour car")
        self.assertEqual(m.area_at((0.9, 0.9)).name, "street")
        self.assertIsNone(m.area_at((0.1, 0.5)))                 # unmapped

    def test_crossing_a_line_and_its_direction(self) -> None:
        self.assertEqual(RAILING.crossing((0.7, 0.5), (0.3, 0.5)), "in")     # into the yard (left on the picture)
        self.assertEqual(RAILING.crossing((0.3, 0.5), (0.7, 0.5)), "out")
        self.assertIsNone(RAILING.crossing((0.3, 0.5), (0.4, 0.5)))
        self.assertIsNone(RAILING.crossing((0.3, 0.5), (0.5, 0.5)))           # touching is not crossing

    def test_the_inward_side_is_as_seen_on_the_picture(self) -> None:
        # Drawn left to right along the bottom; inward "left" of travel is up the picture.
        gate = sm.Line("gate", (0.0, 0.8), (1.0, 0.8), inward="left")
        self.assertEqual(gate.crossing((0.5, 0.95), (0.5, 0.6)), "in")
        self.assertEqual(sm.Line.toward("gate", (0.0, 0.8), (1.0, 0.8), (0.5, 0.2)).inward, "left")
        self.assertEqual(sm.Line.toward("gate", (0.0, 0.8), (1.0, 0.8), (0.5, 0.9)).inward, "right")

    def test_inward_side_is_inferred_from_the_mine_areas(self) -> None:
        line = sm.inward_from_areas("railing", (0.5, 0.0), (0.5, 1.0), [sm.Area("yard", sm.MINE, "yard", tuple(RIGHT))])
        self.assertEqual(line.crossing((0.3, 0.5), (0.7, 0.5)), "in")


class StorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.zones = os.path.join(self.tmp.name, "zones.yaml")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_camera_with_no_zone_and_no_map_is_unmapped(self) -> None:
        m = sm.load_scene_map("front", self.zones)
        self.assertEqual(m.outside, "unmapped")
        self.assertEqual(m.all_areas(), ())
        self.assertFalse(m.informative)

    def test_todays_drawn_zone_migrates_to_mine_with_black_outside(self) -> None:
        z.save_zone("front", LEFT, self.zones)
        m = sm.load_scene_map("front", self.zones)
        self.assertEqual(m.outside, "black")
        self.assertTrue(m.migrated)
        (area,) = m.all_areas()
        self.assertEqual((area.kind, area.ground, area.implicit), ("mine", "mine", True))
        self.assertEqual(m.area_at((0.2, 0.5)), area)
        self.assertIsNone(m.area_at((0.8, 0.5)))                 # outside the zone: black, nobody is seen there
        self.assertFalse(m.informative)                          # nothing the code can tell the Eye
        self.assertFalse(os.path.exists(z.scene_maps_path_for(self.zones)))   # migration writes nothing

    def test_save_and_load_round_trip(self) -> None:
        z.save_zone("front", [(0, 0), (1, 0), (1, 1), (0, 1)], self.zones)
        saved = sm.save_scene_map(yard_and_neighbour(), self.zones)
        m = sm.load_scene_map("front", self.zones)
        self.assertEqual(m.areas, saved.areas)
        self.assertEqual(m.lines, saved.lines)
        self.assertTrue(m.informative)
        self.assertFalse(m.migrated)
        self.assertEqual([a.name for a in m.all_areas()][-1], sm.WATCHED_NAME)   # today's zone stays, as mine

    def test_black_areas_reach_the_frame_mask(self) -> None:
        sm.save_scene_map(sm.SceneMap("front", areas=(sm.Area("window", sm.BLACK, "window", tuple(TOP_RIGHT)),)),
                          self.zones)
        self.assertEqual(z.load_black(z.scene_maps_path_for(self.zones)), {"front": [TOP_RIGHT]})

    def test_saving_one_camera_keeps_the_others_and_clear_removes_one(self) -> None:
        sm.save_scene_map(yard_and_neighbour(), self.zones)
        sm.save_scene_map(sm.SceneMap("back", areas=(sm.Area("roof", sm.MINE, "roof", tuple(LEFT)),)), self.zones)
        self.assertEqual(set(z.read_scene_maps(z.scene_maps_path_for(self.zones))), {"front", "back"})
        self.assertTrue(sm.clear_scene_map("front", self.zones))
        self.assertFalse(sm.clear_scene_map("front", self.zones))
        self.assertEqual(set(z.read_scene_maps(z.scene_maps_path_for(self.zones))), {"back"})

    def test_a_damaged_entry_loads_as_no_map(self) -> None:
        z.write_scene_maps({"front": {"areas": [{"name": "y", "kind": "mine", "zone": "yard", "points": [[0, 0]]}],
                                      "lines": "nope"}}, z.scene_maps_path_for(self.zones))
        with self.assertLogs(sm.log, level="WARNING"):
            m = sm.load_scene_map("front", self.zones)
        self.assertEqual(m.areas, ())
        self.assertEqual(m.lines, ())

    def test_role_and_zones_from_the_map(self) -> None:
        m = yard_and_neighbour()
        self.assertEqual(m.camera_role(), "private")              # the largest of the owner's own areas
        self.assertEqual(m.zones(), ("yard", "parking"))
        self.assertEqual(sm.SceneMap("x").camera_role(), "")
        gate = sm.SceneMap("x", areas=(sm.Area("gate", sm.MINE, "gate", tuple(LEFT)),
                                       sm.Area("s", sm.WATCH, "street", tuple(RIGHT))))
        self.assertEqual(gate.camera_role(), "entrance")


def det(cls: int, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9):
    return (cls, conf, x1, y1, x2, y2)


class TracksTest(unittest.TestCase):
    def test_foot_point_is_the_bottom_middle_of_the_box(self) -> None:
        self.assertEqual(sm.foot_point((0.2, 0.1, 0.4, 0.9)), (0.3, 0.9))

    def test_detections_become_one_track_per_person(self) -> None:
        looks = [(0.0, [det(0, 0.70, 0.2, 0.80, 0.6), det(0, 0.10, 0.2, 0.20, 0.6)]),
                 (1.0, [det(0, 0.62, 0.2, 0.72, 0.6), det(0, 0.11, 0.2, 0.21, 0.6)]),
                 (2.0, [det(0, 0.40, 0.2, 0.50, 0.6)])]
        tracks = sm.tracks_from_detections(looks)
        self.assertEqual([t.kind for t in tracks], ["person", "person"])
        moving = max(tracks, key=lambda t: len(t.points))
        self.assertEqual([round(p[1], 2) for p in moving.points], [0.75, 0.67, 0.45])

    def test_parked_vehicles_are_left_out_and_moving_ones_kept(self) -> None:
        looks = [(0.0, [det(2, 0.6, 0.5, 0.9, 0.8), det(2, 0.0, 0.5, 0.2, 0.8)]),
                 (1.0, [det(2, 0.6, 0.5, 0.9, 0.8), det(2, 0.1, 0.5, 0.3, 0.8)]),
                 (2.0, [det(2, 0.6, 0.5, 0.9, 0.8), det(2, 0.2, 0.5, 0.4, 0.8)])]
        tracks = sm.tracks_from_detections(looks)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].kind, "vehicle")

    def test_detections_from_a_yolo_result_are_normalised(self) -> None:
        class Box:                                   # one box the way ultralytics iterates them
            def __init__(self, cls, xyxy, conf):
                self.cls, self.xyxy, self.conf = np.array([cls]), np.array([xyxy]), np.array([conf])

        class Result:
            boxes = [Box(0.0, [64.0, 0.0, 128.0, 96.0], 0.8), Box(14.0, [0, 0, 10, 10], 0.9)]

        self.assertEqual(sm.detections_from_result(Result(), 256, 96), [(0, 0.8, 0.25, 0.0, 0.5, 1.0)])
        self.assertEqual(sm.detections_from_result(None, 256, 96), [])


class FactsTest(unittest.TestCase):
    def test_a_person_who_crosses_into_the_yard(self) -> None:
        looks = [(10.0, [det(0, 0.80, 0.2, 0.90, 0.6)]), (14.0, [det(0, 0.60, 0.2, 0.70, 0.6)]),
                 (18.0, [det(0, 0.40, 0.2, 0.50, 0.6)]), (22.0, [det(0, 0.25, 0.2, 0.35, 0.6)])]
        facts = sm.scene_facts(yard_and_neighbour(), sm.tracks_from_detections(looks))
        self.assertEqual(facts.ground, "mine")
        self.assertTrue(facts.crossed_in)
        self.assertEqual(facts.zone, "yard")
        self.assertEqual(facts.line, "ZONE FACTS (from code): person 1 entered 'yard' (mine) from "
                                     "'neighbour driveway' (neighbour's), crossed 'railing' inward, 4s in 'yard'")

    def test_a_person_who_stays_on_the_neighbours_ground(self) -> None:
        looks = [(0.0, [det(0, 0.70, 0.2, 0.80, 0.6)]), (35.0, [det(0, 0.72, 0.2, 0.82, 0.6)])]
        facts = sm.scene_facts(yard_and_neighbour(), sm.tracks_from_detections(looks))
        self.assertEqual((facts.ground, facts.crossed_in, facts.zone), ("neighbour", False, "parking"))
        self.assertEqual(facts.line, "ZONE FACTS (from code): person 1 is in 'neighbour driveway' (neighbour's) "
                                     "for 35s, did not cross 'railing'")

    def test_mine_wins_when_two_people_are_on_different_ground(self) -> None:
        looks = [(0.0, [det(0, 0.70, 0.2, 0.80, 0.6), det(0, 0.10, 0.2, 0.20, 0.6)]),
                 (5.0, [det(0, 0.71, 0.2, 0.81, 0.6), det(0, 0.11, 0.2, 0.21, 0.6)])]
        facts = sm.scene_facts(yard_and_neighbour(), sm.tracks_from_detections(looks))
        self.assertEqual(facts.ground, "mine")
        self.assertIn("person 1", facts.line)
        self.assertIn("person 2", facts.line)

    def test_people_decide_the_ground_and_an_unplaced_person_keeps_it_unknown(self) -> None:
        m = sm.SceneMap("front", areas=(sm.Area("road", sm.WATCH, "street", tuple(RIGHT)),
                                        sm.Area("yard", sm.MINE, "yard", tuple(TOP_RIGHT))))
        # A person the map cannot place (left half) and a car driving along the road.
        looks = [(0.0, [det(0, 0.10, 0.2, 0.20, 0.6), det(2, 0.55, 0.5, 0.65, 0.8)]),
                 (1.0, [det(0, 0.11, 0.2, 0.21, 0.6), det(2, 0.70, 0.5, 0.80, 0.8)])]
        facts = sm.scene_facts(m, sm.tracks_from_detections(looks))
        self.assertEqual((facts.ground, facts.zone), ("", ""))
        self.assertIn("vehicle 1", facts.line)
        # Two people: one on the road, one unplaced: still unknown.
        looks = [(0.0, [det(0, 0.10, 0.2, 0.20, 0.6), det(0, 0.70, 0.2, 0.80, 0.6)]),
                 (1.0, [det(0, 0.11, 0.2, 0.21, 0.6), det(0, 0.71, 0.2, 0.81, 0.6)])]
        self.assertEqual(sm.scene_facts(m, sm.tracks_from_detections(looks)).ground, "")
        # Only a car on the road: the vehicle decides.
        looks = [(0.0, [det(2, 0.55, 0.5, 0.65, 0.8)]), (1.0, [det(2, 0.70, 0.5, 0.80, 0.8)])]
        self.assertEqual(sm.scene_facts(m, sm.tracks_from_detections(looks)).ground, "public")

    def test_no_facts_without_an_informative_map_or_without_tracks(self) -> None:
        looks = [(0.0, [det(0, 0.70, 0.2, 0.80, 0.6)]), (5.0, [det(0, 0.72, 0.2, 0.82, 0.6)])]
        migrated = sm.SceneMap("front", watched=tuple(LEFT))
        self.assertEqual(sm.scene_facts(migrated, sm.tracks_from_detections(looks)), sm.NO_FACTS)
        self.assertEqual(sm.scene_facts(yard_and_neighbour(), []), sm.NO_FACTS)
        self.assertEqual(sm.NO_FACTS.line, "")
        self.assertEqual(sm.NO_FACTS.ground, "")

    def test_outside_every_area_is_said_plainly(self) -> None:
        m = sm.SceneMap("front", areas=(sm.Area("yard", sm.MINE, "yard", tuple(TOP_RIGHT)),))
        looks = [(0.0, [det(0, 0.10, 0.2, 0.20, 0.9)]), (3.0, [det(0, 0.12, 0.2, 0.22, 0.9)])]
        facts = sm.scene_facts(m, sm.tracks_from_detections(looks))
        self.assertEqual(facts.ground, "")
        self.assertEqual(facts.line, "ZONE FACTS (from code): person 1 is outside the mapped areas")

    def test_the_line_stays_short_with_many_tracks(self) -> None:
        looks = [(float(t), [det(0, 0.05 + 0.1 * i, 0.2, 0.1 + 0.1 * i, 0.6) for i in range(8)]) for t in range(3)]
        facts = sm.scene_facts(yard_and_neighbour(), sm.tracks_from_detections(looks))
        self.assertLessEqual(facts.line.count("person "), sm.MAX_TRACKS_PER_KIND)
        self.assertLessEqual(len(facts.line), sm.FACTS_LIMIT)

    def test_the_record_for_training(self) -> None:
        looks = [(0.0, [det(0, 0.70, 0.2, 0.80, 0.6)]), (35.0, [det(0, 0.72, 0.2, 0.82, 0.6)])]
        facts = sm.scene_facts(yard_and_neighbour(), sm.tracks_from_detections(looks))
        self.assertEqual(facts.record(), {"zone_facts": facts.line, "ground": "neighbour", "crossed_in": False,
                                          "zone": "parking"})


if __name__ == "__main__":
    unittest.main()
