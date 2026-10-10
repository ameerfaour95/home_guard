"""2026-10-10 17:03, camera 2: the live look told a person the detector did not find as a fact, and "של מי הרכב"
did not use the owner's fresh scene map. The grounded look (brain/grounded_look.py): detector facts and the map go
to the vision model, code checks its answer, and the map tools answer whose ground it is."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

from home_guard_project.box import scene_map as sm
from home_guard_project.box.brain import grounded_look as gl
from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import Services, ToolContext, check_camera, look_around, map_info, where_is
from home_guard_project.box.brain.vision import Vision, look_prompt, look_schema

NOW = 1_791_641_027.0
CH2 = "ameer_v2_ch2"
TRUCK = {"label": "truck", "conf": 0.84, "box": [0.40, 0.50, 0.68, 0.77]}
PERSON = {"label": "person", "conf": 0.81, "box": [0.80, 0.60, 0.85, 0.90]}
# The 17:03 answer, as the vision model / the chat said it.
SAID_HE = "במצלמה 2 יש אדם בלבוש כהה הולך ליד צד הבית, בזמן שטנדר חונה בחניה ליד חומרי בנייה. התמונה ברורה."
SAID_EN = "A person in dark clothing walks by the side of the house, while a pickup is parked near building materials."


def neighbour_map(camera: str = CH2) -> sm.SceneMap:
    """Left half the neighbour's driveway, right half ours, and the owner's wall (a fence area) across the middle."""
    return sm.SceneMap(camera, areas=(
        sm.Area("driveway", sm.WATCH, "parking", [(0.0, 0.3), (0.7, 0.3), (0.7, 1.0), (0.0, 1.0)], owner=sm.NEIGHBOUR),
        sm.Area("yard", sm.MINE, "yard", [(0.7, 0.3), (1.0, 0.3), (1.0, 1.0), (0.7, 1.0)]),
        sm.Area("wall", sm.MINE, "fence", [(0.3, 0.7), (0.9, 0.7), (0.9, 0.8), (0.3, 0.8)]),
    ))


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.zones = os.path.join(self.root, "zones.yaml")
        self.photo = os.path.join(self.root, "photo.jpg")
        with open(self.photo, "wb") as f:
            f.write(b"\xff\xd8jpeg")

    def save(self, scene: sm.SceneMap) -> None:
        sm.save_scene_map(scene, self.zones)


class EnforceTest(Base):
    def facts(self, objects: List[Dict[str, Any]], mapped: bool = True) -> gl.Facts:
        source, seen = gl.detect(CH2, self.photo, NOW, detector=lambda _p: objects)
        scene = neighbour_map() if mapped else None
        return gl.Facts(CH2, source, [gl.place(scene, s) for s in seen], mapped, gl.area_names(scene))

    def test_no_person_detected_but_the_model_states_one_is_corrected(self) -> None:
        look = {"ok": True, "description": SAID_HE, "quality": "clear", "people": 1}
        out = gl.enforce(look, self.facts([TRUCK]), "מצלמה 2")
        self.assertEqual(out["people"], 0)
        self.assertIn("לא רואה אנשים במצלמה 2 עכשיו", out["description"])
        self.assertIn("ייתכן שיש אדם, לא בטוח", out["description"])
        self.assertNotIn("יש אדם בלבוש כהה", out["description"])           # never as a fact
        self.assertIn("טנדר חונה בחניה", out["description"])                # the rest of the answer stays
        self.assertEqual(out["corrected"], ["people"])
        self.assertGreaterEqual(out["unsure_people"], 1)

    def test_english_answer_too(self) -> None:
        look = {"ok": True, "description": SAID_EN, "quality": "clear", "people": 1}
        out = gl.enforce(look, self.facts([TRUCK]))
        self.assertTrue(out["description"].startswith("No people seen now"))
        self.assertNotIn("person in dark clothing", out["description"])
        self.assertIn("pickup is parked", out["description"])
        self.assertIn("There may be a person, not certain.", out["description"])

    def test_an_answer_with_no_person_is_left_alone(self) -> None:
        look = {"ok": True, "description": "A pickup is parked. No people are visible.", "quality": "clear",
                "people": 0}
        out = gl.enforce(look, self.facts([TRUCK]))
        self.assertEqual(out["description"], look["description"])
        self.assertNotIn("corrected", out)

    def test_a_person_the_model_missed_is_added_from_the_detector(self) -> None:
        look = {"ok": True, "description": "A quiet yard.", "quality": "clear", "people": 0}
        out = gl.enforce(look, self.facts([PERSON]))
        self.assertEqual(out["people"], 1)
        self.assertIn("detector found 1 person", out["description"])

    def test_a_vehicle_the_detector_did_not_find_is_only_maybe(self) -> None:
        look = {"ok": True, "description": "A white car is parked by the gate.", "quality": "clear", "people": 0}
        out = gl.enforce(look, self.facts([]))
        self.assertNotIn("white car is parked", out["description"])
        self.assertIn("There may be a vehicle, not certain.", out["description"])

    def test_no_detector_no_check(self) -> None:
        look = {"ok": True, "description": SAID_EN, "quality": "clear", "people": 1}
        self.assertEqual(gl.enforce(look, gl.Facts(CH2)), look)


class PlaceTest(Base):
    def test_a_car_on_the_neighbours_driveway_is_the_neighbours_even_across_our_wall(self) -> None:
        seen = gl.place(neighbour_map(), gl.Seen("truck", 0.84, tuple(TRUCK["box"])))
        self.assertEqual((seen.ground, seen.area), (sm.NEIGHBOUR, "driveway"))
        self.assertIn("בשטח של השכן", gl.whose_text(seen))
        self.assertIn("לפי המפה שציירת", gl.whose_text(seen))

    def test_a_person_by_the_feet(self) -> None:
        seen = gl.place(neighbour_map(), gl.Seen("person", 0.8, tuple(PERSON["box"])))
        self.assertEqual(seen.ground, sm.MINE)

    def test_a_spot_outside_every_area_is_not_mapped(self) -> None:
        seen = gl.place(neighbour_map(), gl.Seen("person", 0.8, (0.4, 0.0, 0.45, 0.2)))
        self.assertEqual(seen.ground, "")
        self.assertEqual(gl.whose_text(seen), "")

    def test_map_by_channel_after_a_rename(self) -> None:
        self.save(neighbour_map("ameer_week_0_1_ch2"))
        scene = gl.load_map(CH2, self.zones)
        self.assertIsNotNone(scene)
        self.assertEqual(scene.camera, "ameer_week_0_1_ch2")
        self.assertIsNone(gl.load_map("ameer_v2_ch3", self.zones))

    def test_the_live_detector_counts_only_when_fresh(self) -> None:
        status = os.path.join(self.root, "ai_status.json")
        cams = {CH2: {"checked_ts": NOW - 1, "ts": NOW - 30, "objects": [PERSON]}}
        with open(status, "w", encoding="utf-8") as f:
            json.dump({"cameras": cams}, f)
        self.assertEqual(gl.detect(CH2, self.photo, NOW, None, status), ("live", []))   # the newest look: nothing
        self.assertEqual(gl.detect(CH2, self.photo, NOW + 5, None, status), ("", []))   # too old: unknown

    def test_facts_text_names_the_detector_and_the_map(self) -> None:
        self.save(neighbour_map())
        facts = gl.grounded_facts(CH2, self.photo, NOW, detector=lambda _p: [TRUCK], zones_path=self.zones)
        text = gl.facts_text(facts)
        self.assertIn("- people: 0", text)
        self.assertIn("truck/pickup (0.84) on the neighbour's ground", text)
        self.assertIn("never as a fact", text)
        self.assertIn(text, look_prompt(CH2, False, facts=text))
        self.assertIn("unsure_people", look_schema(False, grounded=True)["properties"])
        self.assertNotIn("unsure_people", look_schema(False)["properties"])


class FakeVision:
    """The 17:03 model: a person and a pickup, whatever the facts say."""

    def __init__(self) -> None:
        self.facts: List[str] = []

    def look(self, camera, images, guard=False, question="", what="a live photo", facts=""):
        self.facts.append(facts)
        return {"ok": True, "description": SAID_HE, "quality": "clear", "people": 1, "label": "normal", "why": ""}


class Deliver:
    def __init__(self) -> None:
        self.photos: List[Any] = []

    def photo(self, chat_id, path, caption=""):
        self.photos.append((path, caption))
        return {"ok": True, "message_id": len(self.photos)}


class ToolsTest(Base):
    def ctx(self, text: str, vision: Any = None, detect: Any = None) -> ToolContext:
        self.deliver = Deliver()
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=self.root, mute=None, deliver=self.deliver,
                            now=lambda: NOW, grab_photo=lambda cam: {"image": self.photo}, vision=vision,
                            detect=detect, zones_path=self.zones)
        snap = HouseSnapshot(now=NOW, mode="guard", mode_ends=NOW + 3600, mode_started=NOW - 3600, start_hour=0,
                             end_hour=0, cameras=(CameraState(CH2, True, (), live=True),))
        return ToolContext(turn_id="-5:1", chat_id="-5", speaker={"user_id": 1, "name": "Ameer"}, text=text,
                           lang="he", mode="guard", snapshot=snap, state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.root, ".r"), now=lambda: NOW))

    def test_check_camera_gives_the_facts_and_corrects_the_person(self) -> None:
        self.save(neighbour_map())
        vision = FakeVision()
        ctx = self.ctx("מה קורה במצלמה 2", vision, detect=lambda _p: [TRUCK])
        out = check_camera(ctx, {"camera": CH2})
        self.assertIn("- people: 0", vision.facts[0])
        self.assertEqual(out["people"], 0)
        self.assertIn("ייתכן שיש אדם, לא בטוח", out["description"])
        self.assertNotIn("יש אדם בלבוש כהה", out["description"])
        self.assertEqual(out["detector"]["people"], 0)
        self.assertIn("בשטח של השכן", out["whose_ground"])
        self.assertIn("never say there is one", out["note"])
        # "של מי הרכב" right after: the same look's truck, on the map.
        ctx2 = ToolContext(**{**ctx.__dict__, "text": "של מי הרכב", "receipts": [], "shown": []})
        ans = where_is(ctx2, {"camera": CH2, "what": "הרכב"})
        self.assertTrue(ans["ok"])
        self.assertIn("הטנדר בשטח של השכן (לפי המפה שציירת)", ans["say"])

    def test_look_around_caption_never_states_the_person(self) -> None:
        self.save(neighbour_map())
        out = look_around(self.ctx("יש מישהו בחוץ?", FakeVision(), detect=lambda _p: [TRUCK]), {})
        row = out["cameras"][0]
        self.assertEqual(row["people"], 0)
        self.assertEqual(row["unsure_people"], 1)
        self.assertEqual(len(self.deliver.photos), 1)                  # a possible person is shown, not hidden
        self.assertIn("ייתכן שיש אדם, לא בטוח", self.deliver.photos[0][1])
        self.assertNotIn("יש אדם בלבוש כהה", self.deliver.photos[0][1])

    def test_where_is_without_a_map_is_honest(self) -> None:
        ans = where_is(self.ctx("של מי הרכב", detect=lambda _p: [TRUCK]), {"camera": CH2, "what": "הרכב"})
        self.assertEqual(ans["map"], "none")
        self.assertIn("אין עדיין מפה", ans["say"])
        self.assertIn("באפליקציה", ans["say"])

    def test_where_is_takes_a_new_detector_look_when_none_is_fresh(self) -> None:
        self.save(neighbour_map())
        ans = where_is(self.ctx("זה אצלי או אצל השכן?", detect=lambda _p: [PERSON]), {"camera": CH2, "what": "האיש"})
        self.assertIn("בשטח שלך", ans["say"])
        self.assertEqual(self.deliver.photos, [])                        # nothing sent

    def test_map_info(self) -> None:
        self.save(neighbour_map())
        info = map_info(self.ctx("איפה הגבול עם השכן?"), {"camera": CH2})
        self.assertEqual(info["map"], "yes")
        self.assertEqual(info["neighbour"], ["driveway"])
        self.assertEqual(info["ours"], ["yard", "wall"])
        self.assertNotIn("ameer_", json.dumps(info, ensure_ascii=False))

    def test_map_info_without_a_map(self) -> None:
        info = map_info(self.ctx("מה המצלמה הזו רואה?"), {"camera": CH2})
        self.assertEqual(info["map"], "none")
        self.assertIn("אין עדיין מפה", info["say"])


class VisionFactsTest(unittest.TestCase):
    def test_the_facts_reach_the_model_and_unsure_people_comes_back(self) -> None:
        seen: Dict[str, Any] = {}

        def complete(prompt, images, schema):
            seen.update(prompt=prompt, schema=schema)
            return json.dumps({"description": "A pickup.", "quality": "clear", "people": 0, "unsure_people": 1})

        out = Vision(complete).look(CH2, [b"x"], guard=False, facts="DETECTOR FACTS: people 0")
        self.assertIn("DETECTOR FACTS: people 0", seen["prompt"])
        self.assertIn("unsure_people", seen["schema"]["properties"])
        self.assertEqual(out["unsure_people"], 1)


if __name__ == "__main__":
    unittest.main()
