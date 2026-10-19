"""'It stayed' (stayed_vehicles.py): a car the plate reader confirmed keeps the owner's name on its parking spot.

Owner, 2026-10-19: how would the box know "my Tesla" when the plate can't be read (night, far, angle, rear)? Once a
plate read matches a car he named on a camera, every later alert there - day or night - that has a vehicle on that
spot names it, until the spot is empty for a while, the car leaves from it, or another plate is read there.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

from home_guard_project.box import car_plates as cp
from home_guard_project.box import known_vehicles as kv
from home_guard_project.box import stayed_vehicles as sv

CAM = "ameer_v2_ch6"
SPOT = (0.20, 0.40, 0.45, 0.75)
TESLA = {"id": "V1", "label": "הרכב שלך", "plate": "17279804"}
DAY = dt.datetime(2026, 10, 19, 14, 30).timestamp()
NIGHT = dt.datetime(2026, 10, 19, 23, 10).timestamp()


def shifted(box, dx: float = 0.0, dy: float = 0.0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def det(box, cls: int = 2) -> tuple:
    return (cls, 0.8, *box)


def track(tid: int, boxes: List[Any], t0: float = 100.0, step: float = 1.0) -> Dict[str, Any]:
    return {"id": tid, "kind": "vehicle", "first_seen": t0, "last_seen": t0 + step * (len(boxes) - 1), "shown": False,
            "boxes": [{"ts": t0 + i * step, "box": list(b)} for i, b in enumerate(boxes)]}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="stayed_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.clock = {"now": DAY}
        self.store = self.make()

    def make(self) -> sv.StayedVehicles:
        return sv.StayedVehicles(os.path.join(self.dir, sv.FILE_NAME), empty_sec=600.0, now=lambda: self.clock["now"])


class StayedTests(Base):
    def test_a_plate_confirmed_car_is_named_on_its_spot_later_without_a_plate(self) -> None:
        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        names = self.store.names_for(CAM, [track(9, [shifted(SPOT, 0.01)] * 5)], NIGHT)
        self.assertEqual(list(names.values()), [{"label": "הרכב שלך", "vehicle_id": "V1", "since": "14:30",
                                                 "left": False}])
        self.assertEqual(self.store.names_for("ameer_v2_ch3", [track(9, [SPOT] * 5)], NIGHT), {})   # another camera
        self.assertEqual(self.store.names_for(CAM, [track(9, [shifted(SPOT, 0.2)] * 5)], NIGHT), {})  # another spot

    def test_an_unnamed_plate_starts_nothing(self) -> None:
        self.assertIsNone(self.store.plate_read(CAM, SPOT, "11111111", None, DAY))
        self.assertEqual(self.store.active(), [])

    def test_looks_keep_it_and_a_sustained_empty_spot_ends_it(self) -> None:
        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        ts = DAY
        for i in range(40):                        # IR night looks with a jittery box: still there
            ts += 30
            self.store.look(CAM, ts, [det(shifted(SPOT, 0.03, -0.02))])
        self.assertEqual(len(self.store.active(CAM)), 1)
        for i in range(10):                        # empty 10 minutes but only 10 looks: the camera hardly looked
            ts += 60
            self.store.look(CAM, ts, [])
        self.assertEqual(len(self.store.active(CAM)), 1)
        for i in range(30):
            ts += 5
            self.store.look(CAM, ts, [det((0.7, 0.1, 0.8, 0.2))])    # another car elsewhere does not keep it
        self.assertEqual(self.store.active(CAM), [])
        with open(self.store.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[-1]["end"], "the spot is empty")

    def test_a_person_in_front_for_a_moment_does_not_end_it(self) -> None:
        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        self.store.look(CAM, DAY + 5, [det((0.3, 0.3, 0.36, 0.8), cls=0)])
        self.store.look(CAM, DAY + 10, [det(SPOT)])
        self.assertEqual(self.store.active(CAM)[0]["empty_looks"], 0)

    def test_a_car_that_leaves_from_the_spot_is_named_once_and_ends_it(self) -> None:
        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        leaving = track(4, [SPOT, shifted(SPOT, 0.1), shifted(SPOT, 0.3), shifted(SPOT, 0.5), shifted(SPOT, 0.5)])
        names = self.store.names_for(CAM, [leaving], DAY + 600)
        self.assertTrue(next(iter(names.values()))["left"])
        self.assertEqual(self.store.active(CAM), [])
        self.assertEqual(self.store.names_for(CAM, [track(5, [SPOT] * 3)], DAY + 900), {})

    def test_another_plate_on_the_spot_ends_it_and_one_digit_off_is_the_same_car(self) -> None:
        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        self.store.plate_read(CAM, shifted(SPOT, 0.01), "17279894", None, DAY + 60)     # a misread by one digit
        self.assertEqual(len(self.store.active(CAM)), 1)
        self.store.plate_read(CAM, SPOT, "55512345", None, DAY + 120)
        self.assertEqual(self.store.active(CAM), [])

    def test_it_survives_a_restart_but_not_a_long_one(self) -> None:
        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        self.clock["now"] = DAY + 600
        self.assertEqual(len(self.make().active(CAM)), 1)
        self.clock["now"] = DAY + sv.RESTART_GAP_SEC + 10
        again = self.make()
        self.assertEqual(again.active(CAM), [])

    def test_two_cars_side_by_side_keep_their_own_names(self) -> None:
        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        other = shifted(SPOT, 0.3)
        self.store.plate_read(CAM, other, "1609875", {"id": "V2", "label": "הרכב של החשמלאי מנסור"}, DAY)
        names = self.store.names_for(CAM, [track(1, [SPOT] * 3), track(2, [other] * 3)], NIGHT)
        self.assertEqual(sorted(v["label"] for v in names.values()), ["הרכב של החשמלאי מנסור", "הרכב שלך"])


class ServiceTests(Base):
    def test_an_alerts_plate_match_starts_the_identity_with_the_vehicles_box(self) -> None:
        import numpy as np

        from home_guard_project.box import plates as pl

        known = kv.store_in(self.dir)
        tesla = known.add("17279804", "הרכב שלך", "לבן", CAM, now=1.0)

        class Reader:
            def available(self):
                return True

            def read(self, crops):
                return [pl.PlateRead("17279804", 0.99, 70.0) for _ in crops]

        svc = cp.PlateService(cp.PlateSettings(True, (CAM,), "CPU", 5), reader=Reader(), store=known,
                              stayed=self.store, start=False)
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
        shot = cp.Shot("7@1.000", [pl.VehicleCrop(img, (10, 10, 490, 290))] * 3, [(0, 0, 500, 300)] * 3, [None] * 3,
                       (2592, 1520), box=SPOT)
        out = svc.run_once(cp.Job("a1", CAM, DAY, shots=[shot]))
        self.assertEqual(out["vehicles"]["7@1.000"]["box"], list(SPOT))
        active = self.store.active(CAM)
        self.assertEqual((active[0]["vehicle_id"], active[0]["label"]), (tesla["id"], "הרכב שלך"))
        self.assertNotIn("17279804", json.dumps(cp.meta_record(out)))


class InferenceTests(Base):
    def job(self, ts: float, tracks: List[Any]):
        from home_guard_project.box import inference as inf

        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(ts)}_alert", ts=ts)
        job.track_boxes = {"tracks": tracks}
        return job

    def test_at_night_the_alert_names_the_car_that_stayed_and_the_meta_says_since_when(self) -> None:
        from home_guard_project.box import inference as inf

        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        svc = cp.PlateService(cp.PlateSettings(True, (CAM,), "CPU", 5), reader=object(), stayed=self.store,
                              start=False)
        tracks = [{"id": 3, "kind": "person", "first_seen": 99.0, "shown": True, "boxes": []},
                  track(12, [SPOT] * 4)]
        job = self.job(NIGHT, tracks)
        lines = [{"id": "P1", "appearance": "אדם", "action": "עומד ליד הרכב"},
                 {"id": "CAR1", "appearance": "רכב לבן", "action": "חונה"}]
        with mock.patch.object(inf, "PLATES", svc), mock.patch.object(inf, "EVENTS", None):
            inf._plates_to_event(job, job.stem)
            named = inf._named_vehicles(job, job.stem, tracks, {}, lines, ["P1", "CAR1"])
            cars = inf._car_entities(job, job.stem)
        self.assertEqual(named[1], {"id": "CAR1", "appearance": "הרכב שלך", "action": "חונה"})
        self.assertEqual(job.input_meta["stayed"], {"12@100.000": {"vehicle_id": "V1", "since": "14:30",
                                                                    "left": False}})
        self.assertNotIn("plates", job.input_meta)                  # no plate read at night
        self.assertEqual(cars, [{"id": "CAR1", "appearance": "הרכב שלך", "action": ""}])
        self.assertNotIn("17279804", json.dumps(job.input_meta))

    def test_the_event_vehicle_gets_the_name(self) -> None:
        from home_guard_project.box import inference as inf
        from home_guard_project.box.events import EventBook, Session

        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        svc = cp.PlateService(cp.PlateSettings(True, (CAM,), "CPU", 5), reader=object(), stayed=self.store,
                              start=False)
        book = EventBook(os.path.join(self.dir, "events"))
        leaving = track(12, [SPOT, shifted(SPOT, 0.2), shifted(SPOT, 0.5)])
        job = self.job(NIGHT, [leaving])
        s = Session(id="s1", camera=CAM, opened=NIGHT, last_active=NIGHT)
        s.entities = [{"id": "CAR1", "kind": "vehicle", "tracks": ["12@100.000"], "owner_label": ""}]
        book._open[CAM] = s
        book._by_alert[job.stem] = "s1"
        with mock.patch.object(inf, "PLATES", svc), mock.patch.object(inf, "EVENTS", book):
            inf._plates_to_event(job, job.stem)
        self.assertEqual(s.entities[0]["owner_label"], "הרכב שלך")
        self.assertEqual(s.entities[0]["named_by"], "stayed since 14:30")
        self.assertTrue(job.input_meta["stayed"]["12@100.000"]["left"])
        self.assertEqual(self.store.active(CAM), [])                 # it left: the spot holds no name now

    def test_the_guard_loops_looks_reach_the_store(self) -> None:
        from home_guard_project.box import inference as inf

        self.store.plate_read(CAM, SPOT, "17279804", TESLA, DAY)
        svc = cp.PlateService(cp.PlateSettings(True, (CAM,), "CPU", 5), reader=object(), stayed=self.store,
                              start=False)
        with mock.patch.object(inf, "PLATES", svc):
            inf._stayed_look(CAM, DAY + 50, [det(SPOT)])
        self.assertEqual(self.store.active(CAM)[0]["last_seen"], DAY + 50)


if __name__ == "__main__":
    unittest.main()
