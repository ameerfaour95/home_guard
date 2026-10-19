# -*- coding: utf-8 -*-
"""A live look names the cars the owner named (assistant_v3/skills.Toolbox._named_cars, vehicles.known_cars).

Replay of 2026-10-19 15:53: "מה יש בחוץ" got "... ובכניסה הראשית שני רכבים חונים" while his own Tesla (known car
V6f3978d9, "הרכב שלך", "טסלה לבנה") stood there. Now each live photo of a plate camera, by day, goes through the
plate reader on its main-stream frame (the photo's own detector boxes, a 3 s budget), and a car he named reaches the
writer as a code-made evidence line plus a note to call it by his words; the plate never appears. An unknown plate
names nothing; at night a spot a plate confirmed earlier (stayed) still names it.
"""
import datetime as dt
import os
import unittest
from typing import Any, Dict, Optional

from home_guard_project.box import car_plates, eval_chat as ec
from home_guard_project.box import known_vehicles as kv
from home_guard_project.box import stayed_vehicles as sv

from test_assistant_v3 import FakeModel, act, acts_for

DAY = "2026-10-19"
HOUSE = {"cameras": ["ameer_v2_ch2", "ameer_v2_ch3", "ameer_v2_ch5", "ameer_v2_ch6", "ameer_v2_ch8"],
         "aliases": {"ameer_v2_ch3": ["פרגולה"], "ameer_v2_ch6": ["כניסה ראשית"]}}
ASKED = "מה יש בחוץ"
TESLA = "17279804"


class FakePlates:
    def __init__(self, plates_by_camera: Dict[str, str], stayed: Any = None) -> None:
        self.plates_by_camera, self.asked, self.stayed = plates_by_camera, [], stayed
        self.settings = car_plates.PlateSettings(True, ("ameer_v2_ch3", "ameer_v2_ch6"), "CPU", 5)
        self.store: Optional[kv.KnownVehicles] = None

    def read_media(self, path: str, camera: str, wait: float = 0.0, boxes: Any = None) -> Dict[str, Any]:
        self.asked.append((camera, round(wait, 1) <= 3.0, bool(boxes)))
        plate = self.plates_by_camera.get(camera)
        if not plate:
            return {"key": path, "vehicles": {}}
        known = self.store.match(plate) if self.store is not None else None
        v = {"plate": plate, "reads": 5}
        if known:
            v.update(vehicle_id=known["id"], label=known["label"])
        return {"key": path, "vehicles": {"v1": v}}


class LiveLookNamesKnownCars(unittest.TestCase):
    def run_case(self, clock: str, plates: FakePlates, live: Dict[str, Any]):
        understand = acts_for({ASKED: {"emotion": "neutral", "acts": [act("question_live", quote=ASKED,
                                                                            camera="house")]}})
        writer = FakeModel(lambda user, kw: "בחוץ שקט.")
        ec.V3.clear()
        ec.V3.update(write=writer, understand=understand, critic=None, escalate=None)
        case = {"id": "t", "source": "test", "date": DAY, "house": HOUSE, "time": clock, "alerts": [], "live": live,
                "message": {"text": ASKED}, "expect": {"intent": "x", "ideal": "x", "max_questions": 0}}
        world = ec.World(case, None, None)
        ec._local.world = world
        self.addCleanup(self._cleanup)
        world.seed()
        store = kv.store_in(world.events.directory)
        store.add(TESLA, "הרכב שלך", "טסלה לבנה", "ameer_v2_ch6", now=1.0)
        plates.store = store
        car_plates._SERVICE = plates
        reply = world.agent.handle(ASKED, ec.CHAT, ec.OWNER, None, False)
        return reply, writer, world

    def _cleanup(self) -> None:
        ec._local.world = None
        ec.V3.clear()
        car_plates._SERVICE = None

    def test_the_1553_exchange_names_his_tesla_from_its_plate(self) -> None:
        plates = FakePlates({"ameer_v2_ch6": TESLA, "ameer_v2_ch3": "55512345"})   # ch3: a stranger's car
        live = {"ameer_v2_ch6": {"description": "Two cars parked on the paved driveway.", "vehicles": 2},
                "ameer_v2_ch3": {"description": "A parked car near construction materials.", "vehicles": 1},
                "ameer_v2_ch2": {"description": "A silver car and building materials.", "vehicles": 1}}
        reply, writer, world = self.run_case("15:53:23", plates, live)
        prompt = writer.calls[0]["user"]
        self.assertIn("רכבים מוכרים בכניסה ראשית (קורא לוחיות הרישוי של הקופסה): הרכב שלך (טסלה לבנה)", prompt)
        self.assertIn("קרא לרכב כזה בשם הזה", prompt)                    # the note: name it in the answer
        self.assertNotIn("פרגולה (קורא", prompt)                          # an unknown plate names nothing
        self.assertNotIn(TESLA, prompt)
        self.assertNotIn("55512345", prompt)
        self.assertEqual(sorted(c for c, *_ in plates.asked), ["ameer_v2_ch3", "ameer_v2_ch6"])   # plate cams only
        self.assertTrue(all(fast and boxes for _, fast, boxes in plates.asked))   # within 3 s, the look's own boxes
        self.assertEqual(world.vision_calls and [c for c in world.vision_calls if "plate" in str(c).lower()], [])

    def test_at_night_the_spot_his_plate_confirmed_still_names_it(self) -> None:
        import tempfile

        tmp = tempfile.mkdtemp(prefix="stayed_live_")
        stayed = sv.StayedVehicles(os.path.join(tmp, sv.FILE_NAME))
        plates = FakePlates({}, stayed=stayed)
        live = {"ameer_v2_ch6": {"description": "A dark driveway with a parked car.", "vehicles": 1}}
        # the World's detector puts its first car at (0.1, 0.5, 0.25, 0.8); his Tesla was read there by day
        night = dt.datetime(2026, 10, 19, 22, 40).timestamp()
        stayed.plate_read("ameer_v2_ch6", (0.1, 0.5, 0.25, 0.8), TESLA, {"id": "V?", "label": "הרכב שלך"},
                          night - 600)
        reply, writer, world = self.run_case("22:40:00", plates, live)
        prompt = writer.calls[0]["user"]
        self.assertIn("רכבים מוכרים בכניסה ראשית", prompt)
        self.assertIn("הרכב שלך", prompt)
        self.assertEqual(plates.asked, [])                  # no plate read at night


class ReceiptWordingTests(unittest.TestCase):
    def test_any_description_reads_naturally(self) -> None:
        self.assertEqual(kv.masked_line({"plate": TESLA, "label": "הרכב שלך", "description": "טסלה לבנה"}),
                         "הרכב (טסלה לבנה) •••••804 = הרכב שלך")
        self.assertEqual(kv.masked_line({"plate": "1609875", "label": "הרכב של מנסור", "description": "לבן"}),
                         "הרכב הלבן ••••875 = הרכב של מנסור")
        self.assertEqual(kv.masked_line({"plate": "1609875", "label": "x", "description": ""}), "הרכב ••••875 = x")


if __name__ == "__main__":
    unittest.main()
