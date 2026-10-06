"""The person tracker (box/tracker.py): synthetic looks for match, loss, return, areas and line crossings."""
from __future__ import annotations

import threading
import unittest

from home_guard_project.box import scene_map as sm
from home_guard_project.box import tracker as tr

T0 = 1_000_000.0
STREET = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.3), (0.0, 0.3))
PARKING = ((0.0, 0.3), (1.0, 0.3), (1.0, 0.6), (0.0, 0.6))
ENTRANCE = ((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))
# The gate: a horizontal line across the picture between the parking and the entrance; the entrance is mine.
GATE = sm.Line("gate", (0.0, 0.6), (1.0, 0.6), inward="right")


def house_map() -> sm.SceneMap:
    return sm.SceneMap("front", areas=(
        sm.Area("street", sm.WATCH, "street", STREET),
        sm.Area("parking", sm.MINE, "parking", PARKING),
        sm.Area("entrance", sm.MINE, "entrance", ENTRANCE),
    ), lines=(GATE,))


def person(fx: float, fy: float, w: float = 0.06, h: float = 0.2, conf: float = 0.8) -> sm.Detection:
    """A person box whose feet stand at (fx, fy)."""
    return (0, conf, round(fx - w / 2, 4), round(fy - h, 4), round(fx + w / 2, 4), round(fy, 4))


def car(fx: float, fy: float, w: float = 0.2, h: float = 0.12) -> sm.Detection:
    return (2, 0.9, round(fx - w / 2, 4), round(fy - h, 4), round(fx + w / 2, 4), round(fy, 4))


def walk(tracker: tr.CameraTracker, start: float, points, step: float = 0.5, make=person, scene=None) -> float:
    """Feed one detection per look along *points*; returns the time of the last look."""
    ts = start
    for x, y in points:
        tracker.update(ts, [make(x, y)], scene_map=scene)
        ts += step
    return ts - step


def line_points(a, b, n):
    return [(a[0] + (b[0] - a[0]) * i / (n - 1), a[1] + (b[1] - a[1]) * i / (n - 1)) for i in range(n)]


class GateTest(unittest.TestCase):
    def test_the_distance_gate_grows_with_the_gap_and_is_capped(self) -> None:
        self.assertAlmostEqual(tr.distance_gate(1.0), 0.12)
        self.assertAlmostEqual(tr.distance_gate(10.0), 0.25)
        self.assertAlmostEqual(tr.distance_gate(0.05), tr.MIN_GATE)     # back-to-back looks still match
        self.assertAlmostEqual(tr.distance_gate(-1.0), tr.MIN_GATE)


class MatchTest(unittest.TestCase):
    def test_a_person_walking_keeps_one_track(self) -> None:
        t = tr.CameraTracker("front")
        walk(t, T0, line_points((0.2, 0.9), (0.6, 0.9), 20))
        tracks = t.tracks_between(T0, T0 + 20)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].kind, "person")
        self.assertEqual(len(tracks[0].points), 20)

    def test_overlapping_boxes_match_by_iou_before_distance(self) -> None:
        # Two people side by side; each box overlaps its own previous box most.
        t = tr.CameraTracker("front")
        for i in range(6):
            ts = T0 + i * 0.5
            t.update(ts, [person(0.30 + 0.005 * i, 0.9, w=0.1), person(0.42 - 0.005 * i, 0.9, w=0.1)])
        tracks = sorted(t.tracks_between(T0, T0 + 10), key=lambda k: k.points[0][1])
        self.assertEqual(len(tracks), 2)
        self.assertLess(tracks[0].points[-1][1], tracks[1].points[-1][1])   # they did not swap
        self.assertTrue(all(len(k.points) == 6 for k in tracks))

    def test_a_far_jump_starts_a_new_track(self) -> None:
        t = tr.CameraTracker("front")
        t.update(T0, [person(0.1, 0.9)])
        t.update(T0 + 0.5, [person(0.1, 0.9)])
        t.update(T0 + 1.0, [person(0.9, 0.9)])         # 0.8 picture widths in half a second: someone else
        t.update(T0 + 1.5, [person(0.9, 0.9)])
        self.assertEqual(len(t.tracks_between(T0, T0 + 5)), 2)

    def test_a_longer_gap_allows_a_longer_step(self) -> None:
        t = tr.CameraTracker("front")
        t.update(T0, [person(0.1, 0.9)])
        t.update(T0 + 0.5, [person(0.1, 0.9)])
        t.update(T0 + 2.5, [person(0.3, 0.9)])          # 0.2 in 2 s: within 0.24
        self.assertEqual(len(t.tracks_between(T0, T0 + 5)), 1)

    def test_people_and_vehicles_never_share_a_track(self) -> None:
        t = tr.CameraTracker("front")
        t.update(T0, [person(0.5, 0.5)])
        t.update(T0 + 0.5, [car(0.5, 0.5)])
        t.update(T0 + 1.0, [person(0.5, 0.5), car(0.6, 0.5)])
        kinds = sorted(k.kind for k in t.tracks_between(T0, T0 + 5))
        self.assertEqual(kinds, ["person", "vehicle"])

    def test_other_classes_are_ignored(self) -> None:
        t = tr.CameraTracker("front")
        for i in range(4):
            t.update(T0 + i, [(16, 0.9, 0.4, 0.4, 0.5, 0.5)])     # a dog
        self.assertEqual(t.tracks_between(T0, T0 + 10), [])

    def test_one_hit_is_not_a_track(self) -> None:
        t = tr.CameraTracker("front")
        t.update(T0, [person(0.5, 0.9)])
        self.assertEqual(t.tracks_between(T0, T0 + 5), [])
        self.assertTrue(t.facts(T0, T0 + 5).empty)


class LossTest(unittest.TestCase):
    def test_a_person_is_lost_after_four_seconds_unseen(self) -> None:
        t = tr.CameraTracker("front")
        walk(t, T0, [(0.5, 0.9)] * 3)
        t.update(T0 + 1.0 + 3.9, [person(0.5, 0.9)])    # still the same track
        t.update(T0 + 4.9 + 4.1, [person(0.5, 0.9)])    # lost, a new one starts
        t.update(T0 + 9.5, [person(0.5, 0.9)])
        self.assertEqual(len(t.tracks_between(T0, T0 + 20)), 2)

    def test_a_vehicle_is_kept_six_seconds(self) -> None:
        t = tr.CameraTracker("front")
        walk(t, T0, line_points((0.2, 0.5), (0.4, 0.5), 3), make=car)
        t.update(T0 + 1.0 + 5.5, [car(0.45, 0.5)])
        self.assertEqual(len(t.tracks_between(T0, T0 + 20)), 1)

    def test_history_is_kept_ten_minutes(self) -> None:
        t = tr.CameraTracker("front")
        walk(t, T0, [(0.5, 0.9)] * 3)
        t.update(T0 + 500, [])
        self.assertEqual(len(t.tracks_between(T0, T0 + 5)), 1)
        t.update(T0 + 1 + tr.HISTORY_SEC + 1, [])
        self.assertEqual(t.tracks_between(T0, T0 + 5), [])

    def test_points_are_thinned_and_tracks_capped(self) -> None:
        t = tr.CameraTracker("front")
        walk(t, T0, [(0.5, 0.9)] * 700, step=0.2)
        (track,) = t.tracks_between(T0, T0 + 1000)
        self.assertLessEqual(len(track.points), tr.MAX_POINTS)
        self.assertEqual(track.points[0][0], T0)                       # first and last look survive
        self.assertAlmostEqual(track.points[-1][0], T0 + 699 * 0.2)
        crowd = tr.CameraTracker("crowd")
        for i in range(10):
            crowd.update(T0 + i * 0.1, [person(0.01 + 0.0125 * k, 0.9, w=0.004) for k in range(tr.MAX_ACTIVE + 20)])
        self.assertLessEqual(crowd.active_count(), tr.MAX_ACTIVE)

    def test_a_clock_that_jumps_back_starts_over(self) -> None:
        t = tr.CameraTracker("front")
        walk(t, T0, [(0.5, 0.9)] * 3)
        t.update(T0 - 3600, [person(0.5, 0.9)])
        self.assertEqual(t.tracks_between(T0 - 10, T0 + 10), [])


class ReturnTest(unittest.TestCase):
    def test_coming_back_to_the_same_spot_unmapped_is_a_return(self) -> None:
        t = tr.CameraTracker("front")
        end = walk(t, T0, [(0.5, 0.9)] * 4)
        t.update(end + 60, [])                            # gone a minute
        end = walk(t, end + 61, [(0.55, 0.9)] * 4)        # back, 0.05 from where they were last
        t.update(end + 60, [])
        end = walk(t, end + 61, [(0.45, 0.9)] * 4)
        facts = t.facts(end - 2, end)
        self.assertEqual(facts.people[0].returns, 2)
        self.assertIsNotNone(facts.people[0].prev_id)
        self.assertIn("came back 2 times in 10 min", facts.line())

    def test_somewhere_else_is_not_a_return_unmapped(self) -> None:
        t = tr.CameraTracker("front")
        end = walk(t, T0, [(0.1, 0.9)] * 4)
        end = walk(t, end + 30, [(0.8, 0.9)] * 4)
        self.assertEqual(t.facts(end - 2, end).people[0].returns, 0)

    def test_after_ten_minutes_it_is_a_new_visit(self) -> None:
        t = tr.CameraTracker("front")
        end = walk(t, T0, [(0.5, 0.9)] * 4)
        end = walk(t, end + tr.RETURN_SEC + 5, [(0.5, 0.9)] * 4)
        self.assertEqual(t.facts(end - 2, end).people[0].returns, 0)

    def test_mapped_the_same_area_counts_even_far_away(self) -> None:
        t = tr.CameraTracker("front")
        scene = house_map()
        end = walk(t, T0, [(0.1, 0.9)] * 4, scene=scene)
        end = walk(t, end + 30, [(0.9, 0.9)] * 4, scene=scene)     # far, but still in the entrance
        self.assertEqual(t.facts(end - 2, end, scene).people[0].returns, 1)

    def test_mapped_another_area_is_not_a_return_even_close(self) -> None:
        t = tr.CameraTracker("front")
        scene = house_map()
        end = walk(t, T0, [(0.5, 0.62)] * 4, scene=scene)           # entrance
        end = walk(t, end + 30, [(0.5, 0.58)] * 4, scene=scene)     # parking, 0.04 away
        self.assertEqual(t.facts(end - 2, end, scene).people[0].returns, 0)

    def test_the_map_getter_is_used_when_no_map_is_passed(self) -> None:
        scene = house_map()
        t = tr.CameraTracker("front", scene_map=lambda: scene)
        end = walk(t, T0, [(0.1, 0.9)] * 4)
        end = walk(t, end + 30, [(0.9, 0.9)] * 4)
        self.assertEqual(t.facts(end - 2, end).people[0].returns, 1)


class FactsTest(unittest.TestCase):
    def visit(self):
        """street (0.2) -> parking (0.45) -> through the gate -> entrance (0.8), 0.5 s per look."""
        t = tr.CameraTracker("front")
        scene = house_map()
        pts = (line_points((0.5, 0.15), (0.5, 0.45), 10) + line_points((0.5, 0.47), (0.5, 0.8), 12)
               + [(0.5, 0.8)] * 50)
        end = walk(t, T0, pts, scene=scene)
        return t, scene, end

    def test_time_in_view_areas_path_and_crossing(self) -> None:
        t, scene, end = self.visit()
        facts = t.facts(end - 10, end, scene)
        (p,) = facts.people
        self.assertEqual(p.time_in_view_s, round(end - T0))
        self.assertEqual(p.path, ("street", "parking", "entrance"))
        self.assertEqual(p.zone_path, ("street", "parking", "entrance"))
        self.assertEqual(dict(p.seconds_per_area)["entrance"], 28)         # from the 16th look on
        self.assertEqual(p.crossings, (("gate", "in"),))
        line = facts.line()
        self.assertTrue(line.startswith("TRACKER FACTS (from code): person 1 in view 36s, 28s in 'entrance'"), line)
        self.assertIn("path 'street' > 'parking' > 'entrance'", line)
        self.assertIn("crossed 'gate' inward", line)
        self.assertTrue(line.endswith("."))

    def test_the_line_has_no_counts_classes_confidences_or_coordinates(self) -> None:
        t, scene, end = self.visit()
        end = walk(t, end + 0.5, line_points((0.1, 0.5), (0.9, 0.5), 20), make=car, scene=scene)
        line = t.facts(end - 10, end, scene).line()
        self.assertIn("person 1", line)
        for word in ("car", "vehicle", "0.8", "conf", "people", "2 "):
            self.assertNotIn(word, line.replace("TRACKER FACTS", ""))

    def test_nothing_useful_is_an_empty_line(self) -> None:
        t = tr.CameraTracker("front")
        end = walk(t, T0, [(0.5, 0.9)] * 6)                 # 2.5 s, unmapped, first visit
        facts = t.facts(T0, end)
        self.assertFalse(facts.empty)
        self.assertEqual(facts.line(), "")
        self.assertEqual(tr.TrackerFacts().line(), "")

    def test_a_long_visit_unmapped_is_worth_saying(self) -> None:
        t = tr.CameraTracker("front")
        end = walk(t, T0, [(0.5, 0.9)] * 80)                # 39.5 s
        self.assertEqual(t.facts(end - 10, end).line(), "TRACKER FACTS (from code): person 1 in view 40s; stationary 40s.")

    def test_the_line_is_capped(self) -> None:
        t = tr.CameraTracker("front")
        names = [f"a very long owner name for area {i}" for i in range(6)]
        areas = tuple(sm.Area(n, sm.MINE, "yard", ((i / 6, 0.0), ((i + 1) / 6, 0.0), ((i + 1) / 6, 1.0), (i / 6, 1.0)))
                      for i, n in enumerate(names))
        scene = sm.SceneMap("front", areas=areas)
        xs = line_points((0.02, 0.0), (0.97, 0.0), 60)
        for i, (x, _) in enumerate(xs):
            t.update(T0 + i * 0.5, [person(x, 0.3), person(x, 0.6), person(x, 0.9)], scene_map=scene)
        line = t.facts(T0, T0 + 40, scene).line()
        self.assertLessEqual(len(line), tr.LINE_LIMIT)
        self.assertTrue(line.endswith("."))

    def test_parked_vehicles_never_enter_the_facts(self) -> None:
        t = tr.CameraTracker("front")
        for i in range(10):
            t.update(T0 + i * 0.5, [car(0.5, 0.5), car(0.1 + 0.08 * i, 0.25)])
        facts = t.facts(T0, T0 + 10)
        self.assertEqual(len(facts.vehicles), 1)
        self.assertEqual(facts.case_memory_dict()["vehicles"], 1)
        self.assertEqual(len(t.tracks_between(T0, T0 + 10)), 1)

    def test_case_memory_dict_matches_the_signature_contract(self) -> None:
        from home_guard_project.box.case_memory.signature import build_signature

        t, scene, end = self.visit()
        d = t.facts(end - 10, end, scene).case_memory_dict()
        self.assertEqual(d, {"time_in_view_s": 36.0, "path": ["street", "parking", "entrance"],
                             "entry_edge": "street", "exit_edge": "entrance", "people": 1, "vehicles": 0})
        sig = build_signature("front", end, {"label": "normal"}, d)
        self.assertEqual(sig.path, ("street", "parking", "entrance"))
        self.assertEqual(sig.dwell_s, 36.0)
        self.assertEqual((sig.people, sig.vehicles), (1, 0))

    def test_case_memory_dict_unmapped_uses_picture_edges(self) -> None:
        t = tr.CameraTracker("front")
        end = walk(t, T0, line_points((0.02, 0.9), (0.6, 0.9), 20))
        d = t.facts(T0, end).case_memory_dict()
        self.assertEqual(d["path"], [])
        self.assertEqual((d["entry_edge"], d["exit_edge"]), ("left", ""))

    def test_people_is_the_most_seen_together(self) -> None:
        t = tr.CameraTracker("front")
        for i in range(6):
            ts = T0 + i * 0.5
            t.update(ts, [person(0.2, 0.9), person(0.8, 0.9)] if i < 3 else [person(0.2, 0.9)])
        self.assertEqual(t.facts(T0, T0 + 5).case_memory_dict()["people"], 2)

    def test_empty_facts_give_no_case_memory_dict(self) -> None:
        self.assertEqual(tr.CameraTracker("front").facts(T0, T0 + 10).case_memory_dict(), {})

    def test_record_is_json_ready(self) -> None:
        import json

        t, scene, end = self.visit()
        rec = t.facts(end - 10, end, scene).record()
        json.dumps(rec)
        self.assertEqual(rec["version"], tr.TRACKER_FACTS_VERSION)
        self.assertEqual(rec["people"][0]["path"], ["street", "parking", "entrance"])
        self.assertEqual(rec["line"], t.facts(end - 10, end, scene).line())
        self.assertEqual(rec["case_memory"]["people"], 1)

    def test_tracks_between_feeds_the_zone_facts(self) -> None:
        t, scene, end = self.visit()
        facts = sm.scene_facts(scene, t.tracks_between(end - 10, end))
        self.assertEqual(facts.ground, "mine")
        self.assertIn("person 1 is in 'entrance' (mine)", facts.line)
        whole = sm.scene_facts(scene, t.tracks_between(T0, end))
        self.assertTrue(whole.crossed_in)


class InertiaTest(unittest.TestCase):
    """Area entry needs three looks in a row for a person (one for a vehicle); a stay survives two seconds outside."""

    def facts(self, points, make=person, step=0.5):
        t = tr.CameraTracker("front")
        scene = house_map()
        end = walk(t, T0, points, step=step, make=make, scene=scene)
        facts = t.facts(T0, end, scene)
        return (facts.people or facts.vehicles)[0]

    def test_two_looks_in_an_area_are_not_an_entry_for_a_person(self) -> None:
        p = self.facts([(0.5, 0.57)] * 5 + [(0.5, 0.62)] * 2 + [(0.5, 0.57)] * 5)
        self.assertEqual(p.path, ("parking",))
        self.assertEqual(p.crossings, ())                   # back across the gate before three looks

    def test_three_looks_are(self) -> None:
        p = self.facts([(0.5, 0.57)] * 5 + [(0.5, 0.62)] * 3)
        self.assertEqual(p.path, ("parking", "entrance"))
        self.assertEqual(p.crossings, (("gate", "in"),))

    def test_one_look_is_enough_for_a_vehicle(self) -> None:
        v = self.facts(line_points((0.5, 0.45), (0.5, 0.58), 6) + [(0.5, 0.62)], make=car)
        self.assertEqual(v.path, ("parking", "entrance"))
        self.assertEqual(v.crossings, (("gate", "in"),))

    def test_a_missed_look_does_not_split_a_stay(self) -> None:
        # In the entrance, one look wobbles into the parking, back within the grace: one stay of 9.5 s.
        p = self.facts([(0.5, 0.62)] * 10 + [(0.5, 0.58)] + [(0.5, 0.62)] * 9)
        self.assertEqual(p.path, ("entrance",))
        self.assertEqual(dict(p.seconds_per_area), {"entrance": 10})

    def test_after_the_grace_it_is_a_new_stay(self) -> None:
        t = tr.CameraTracker("front")
        scene = house_map()
        pts = [(0.5, 0.62)] * 4 + [(0.5, 0.58)] * 8 + [(0.5, 0.62)] * 4     # 4 s in the parking
        end = walk(t, T0, pts, scene=scene)
        p = t.facts(T0, end, scene).people[0]
        self.assertEqual(p.path, ("entrance", "parking", "entrance"))
        self.assertEqual(dict(p.seconds_per_area)["entrance"], 3)        # 1.5 + 1.5, the gap not counted

    def test_stationary_time_tells_standing_from_walking(self) -> None:
        t = tr.CameraTracker("front")
        jitter = [(0.5 + (0.008 if i % 2 else 0.0), 0.9) for i in range(40)]      # 19.5 s standing
        end = walk(t, T0, line_points((0.1, 0.9), (0.49, 0.9), 20) + jitter)
        p = t.facts(T0, end).people[0]
        self.assertEqual(p.stationary_s, 20)
        self.assertIn("stationary 20s", t.facts(T0, end).line())
        walker = tr.CameraTracker("front")
        end = walk(walker, T0, line_points((0.05, 0.9), (0.95, 0.9), 60))
        self.assertEqual(walker.facts(T0, end).people[0].stationary_s, 0)


class RegistryTest(unittest.TestCase):
    def test_one_tracker_per_camera_and_cached_maps(self) -> None:
        loads = []

        def load(camera):
            loads.append(camera)
            return house_map()

        clock = [T0]
        reg = tr.TrackerRegistry(scene_map_loader=load, refresh_sec=30.0, clock=lambda: clock[0])
        self.assertIs(reg.get("a"), reg.get("a"))
        self.assertIsNot(reg.get("a"), reg.get("b"))
        for i in range(5):
            reg.update("a", T0 + i * 0.5, [person(0.5, 0.9)])
        self.assertEqual(loads, ["a"])
        clock[0] += 31
        reg.update("a", T0 + 31, [])
        self.assertEqual(loads, ["a", "a"])
        self.assertFalse(reg.facts("a", T0, T0 + 5).empty)

    def test_a_map_that_fails_to_load_is_no_map(self) -> None:
        reg = tr.TrackerRegistry(scene_map_loader=lambda c: (_ for _ in ()).throw(OSError("disk")))
        self.assertIsNone(reg.scene_map("a"))
        reg.update("a", T0, [person(0.5, 0.9)])

    def test_an_uninformative_map_is_no_map(self) -> None:
        reg = tr.TrackerRegistry(scene_map_loader=lambda c: sm.SceneMap(c, watched=STREET))
        self.assertIsNone(reg.scene_map("a"))

    def test_writer_and_readers_on_threads(self) -> None:
        t = tr.CameraTracker("front")
        errors = []

        def write():
            for i in range(400):
                t.update(T0 + i * 0.1, [person(0.3 + 0.001 * (i % 50), 0.9), car(0.1 + 0.002 * (i % 100), 0.4)])

        def read():
            try:
                for _ in range(200):
                    t.facts(T0, T0 + 100).record()
                    t.tracks_between(T0, T0 + 100)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=write)] + [threading.Thread(target=read) for _ in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
