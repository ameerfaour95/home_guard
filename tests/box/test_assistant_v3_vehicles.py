# -*- coding: utf-8 -*-
"""Assistant v3: a car named by its plate (assistant_v3/vehicles.py), replayed on the owner's real words.

2026-10-15 09:18 the owner asked "תראה לי", got a 10 s live video of the pergola, and answered "זה הרכב של החשמלאי
מנסור זה לא של הפרגולה תזכור את המספר שלו". The understanding made a camera fact of it and the writer said "אני
אזכור" with no number read. Now the plate of the car in that video is bound to his words, in code, with a receipt and
Undo; when no plate can be read he is told so and asked to type it, and nothing is claimed or saved.
"""
import datetime as dt
import json
import os
import unittest
from typing import Any, Dict, Optional

from home_guard_project.box import car_plates, eval_chat as ec
from home_guard_project.box import known_vehicles as kv
from home_guard_project.box.assistant_v3 import vehicles as v

from test_assistant_v3 import FakeModel, act, acts_for

DAY = "2026-10-15"
HOUSE = {"cameras": ["ameer_v2_ch3", "ameer_v2_ch6"], "aliases": {"ameer_v2_ch3": ["פרגולה"],
                                                                    "ameer_v2_ch6": ["כניסה ראשית"]}}
SAID = "זה הרכב של החשמלאי מנסור זה לא של הפרגולה תזכור את המספר שלו"
VIDEO = "ameer_v2_ch3_1792045079_949825b2_live.mp4"


def at(clock: str) -> float:
    return dt.datetime.strptime(f"{DAY} {clock}", "%Y-%m-%d %H:%M:%S").timestamp()


class FakePlates:
    """car_plates.service() as the assistant sees it: reads of a media file or of an alert, by key."""

    def __init__(self, by_key: Dict[str, Dict[str, Any]]) -> None:
        self.by_key, self.asked = by_key, []

    def read_media(self, path: str, camera: str, wait: float = 0.0) -> Optional[Dict[str, Any]]:
        self.asked.append(("media", os.path.basename(path), camera))
        return self.by_key.get(os.path.basename(path), {"key": path, "vehicles": {}})

    def result(self, key: str, wait: float = 0.0) -> Optional[Dict[str, Any]]:
        self.asked.append(("alert", key))
        return self.by_key.get(key)


class VehicleTurns(unittest.TestCase):
    def setUp(self) -> None:
        self.understand = acts_for({SAID: {"emotion": "neutral", "acts": [   # what the box's understanding said
            act("camera_fact", quote=SAID, subject="הרכב של החשמלאי מנסור", camera="cam6")]}})
        self.writer = FakeModel(lambda user, kw: "אני אזכור לזהות אותו כרכב של החשמלאי בכניסה הראשית.")
        ec.V3.clear()
        ec.V3.update(write=self.writer, understand=self.understand, critic=None, escalate=None)
        case = {"id": "t", "source": "test", "date": DAY, "house": HOUSE, "time": "09:19:06", "alerts": [],
                "message": {"text": SAID}, "expect": {"intent": "x", "ideal": "x", "max_questions": 0}}
        self.world = ec.World(case, None, None)
        ec._local.world = self.world
        self.world.seed()
        self.addCleanup(self._cleanup)
        # "תראה לי" -> the live video of the pergola, as the box's chat state kept it (E105 on the box)
        state = self.world.agent.memory.load(ec.CHAT)
        h = state.add_handle("clip", os.path.join(self.world.root, ".live", VIDEO), "ameer_v2_ch3", at("09:17:59"))
        state.add_turn("1", "תראה לי", "הנה 10 שניות חיות מהפרגולה.", [h], [], at("09:18:25"))
        self.world.agent.memory.save(ec.CHAT, state)

    def _cleanup(self) -> None:
        ec._local.world = None
        ec.V3.clear()
        car_plates._SERVICE = None

    def say(self, text: str, plates: Optional[FakePlates], alert: Optional[Dict[str, Any]] = None):
        car_plates._SERVICE = plates
        return self.world.agent.handle(text, ec.CHAT, ec.OWNER, alert, bool(alert))

    def store(self) -> kv.KnownVehicles:
        return kv.store_in(self.world.events.directory)

    def test_the_plate_in_the_video_he_answered_is_bound_to_his_words(self) -> None:
        plates = FakePlates({VIDEO: {"key": VIDEO, "vehicles": {"v1": {"plate": "1609875", "reads": 6}}}})
        reply = self.say(SAID, plates)
        self.assertEqual(reply.text, "רשמתי: הרכב ••••875 = הרכב של החשמלאי מנסור.")
        self.assertEqual(self.writer.calls, [])                       # code wrote it, not the writer
        self.assertEqual(self.world.vision_calls, [])                 # the plate reader read it, never a VLM
        self.assertEqual(plates.asked, [("media", VIDEO, "ameer_v2_ch3")])
        cars = self.store().all()
        self.assertEqual([(c["plate"], c["label"], c["camera"]) for c in cars],
                         [("1609875", "הרכב של החשמלאי מנסור", "ameer_v2_ch3")])
        self.assertEqual([r.tool for r in reply.receipts], ["known_vehicle"])
        self.assertNotIn("1609875", json.dumps([r.detail for r in reply.receipts], ensure_ascii=False))
        self.assertTrue(reply.undo_token)
        from home_guard_project.box.camera_profiles import PROFILES_NAME

        self.assertFalse(os.path.exists(os.path.join(self.world.events.directory, PROFILES_NAME))
                         and "החשמלאי" in open(os.path.join(self.world.events.directory, PROFILES_NAME),
                                               encoding="utf-8").read())   # no camera fact of the sentence
        undone = self.world.agent.undo_turn(ec.CHAT, reply.undo_token, ec.OWNER)
        self.assertIn("מחקתי", undone.text)
        self.assertNotIn("1609875", undone.text)
        self.assertEqual(self.store().all(), [])

    def test_no_plate_readable_is_said_plainly_and_nothing_is_claimed(self) -> None:
        reply = self.say(SAID, FakePlates({}))
        self.assertIn("המספר לא קריא", reply.text)
        self.assertIn("הרכב של החשמלאי מנסור", reply.text)
        self.assertNotIn("אזכור", reply.text)
        self.assertNotIn("רשמתי", reply.text)
        self.assertEqual(self.store().all(), [])
        self.assertEqual([r.tool for r in reply.receipts], [])
        # ... then he types it: bound to the words he gave a minute ago
        self.world.clock["now"] += 60
        typed = self.say("16-098-75", FakePlates({}))
        self.assertEqual(typed.text, "רשמתי: הרכב ••••875 = הרכב של החשמלאי מנסור.")
        self.assertEqual(self.store().all()[0]["source"], "typed")

    def test_no_reader_on_the_box_is_said_too(self) -> None:
        reply = self.say(SAID, None)
        self.assertIn("לא יכול לקרוא מספרי רכב", reply.text)
        self.assertEqual(self.store().all(), [])

    def test_a_typed_plate_with_a_name(self) -> None:
        reply = self.say("המספר 56-542-30 זה של החשמלאי", None)
        self.assertEqual(reply.text, "רשמתי: הרכב ••••230 = הרכב של החשמלאי.")
        self.assertEqual(self.store().all()[0]["plate"], "5654230")

    def test_two_different_plates_ask_him_to_type_it(self) -> None:
        plates = FakePlates({VIDEO: {"key": VIDEO, "vehicles": {"v1": {"plate": "1609875", "reads": 4},
                                                                 "v2": {"plate": "5654230", "reads": 3}}}})
        reply = self.say(SAID, plates)
        self.assertIn("יותר מרכב אחד", reply.text)
        self.assertEqual(self.store().all(), [])

    def test_a_reply_to_an_alert_uses_that_alerts_plate(self) -> None:
        alert = {"alert_id": "ameer_v2_ch3_1792402338_alert", "camera": "ameer_v2_ch3", "ts": at("09:18:30"),
                 "label": "suspicious", "summary": "A person stands still next to a white sedan."}
        plates = FakePlates({alert["alert_id"]: {"key": alert["alert_id"], "vehicles": {
            "12@5.000": {"plate": "1609875", "reads": 3}}}})
        reply = self.say("זה הרכב של החשמלאי מנסור", plates, alert)
        self.assertEqual(reply.text, "רשמתי: הרכב הלבן ••••875 = הרכב של החשמלאי מנסור.")
        self.assertEqual(plates.asked, [("alert", alert["alert_id"])])

    def test_a_question_about_a_car_is_not_a_naming(self) -> None:
        self.assertFalse(v.is_question(SAID))
        self.assertTrue(v.is_question("של מי הרכב הזה?"))
        self.assertTrue(v.is_question("מה המספר של הרכב של מנסור"))

    def test_his_words(self) -> None:
        self.assertEqual(v.label_of(SAID), "הרכב של החשמלאי מנסור")
        self.assertEqual(v.label_of("זה האוטו שלי"), "האוטו שלך")
        self.assertEqual(v.label_of("המספר 56-542-30 זה של החשמלאי"), "הרכב של החשמלאי")
        self.assertEqual(v.typed_plate("המספר 56-542-30 זה של החשמלאי"), "5654230")
        self.assertEqual(v.typed_plate("עד 18:30 היום"), "")
        self.assertTrue(v.asks_number("תזכור את המספר שלו"))
        self.assertFalse(v.asks_number("תזכור שהעובדים פה עד 6"))
        self.assertEqual(v.colour_of("", "A person near a parked white car"), "לבן")
        self.assertFalse(v.names_car("זה לא הרכב של החשמלאי"))
        self.assertFalse(v.names_car("הרכב של השכן חוסם לי את היציאה"))
        self.assertTrue(v.names_car("הרכב הלבן הוא של השכן"))
        self.assertTrue(v.names_car("זה האוטו שלי"))
        self.assertEqual(v.label_of("זה לא הרכב של השכן"), "")

    def test_what_do_you_remember_lists_the_car_masked(self) -> None:
        from home_guard_project.box.assistant_v3.memory_view import MemoryView

        self.store().add("1609875", "הרכב של החשמלאי מנסור", "לבן", "ameer_v2_ch3", now=at("09:19:00"))
        state = self.world.agent.memory.load(ec.CHAT)
        view = MemoryView(self.world.services, state, self.world.registry.snapshot(), "he", at("09:20:00"))
        lines = [r.line(self.world.registry.snapshot(), "he", at("09:20:00")) for r in view.records()
                 if r.type == "vehicle"]
        self.assertEqual(len(lines), 1)
        self.assertIn("הרכב הלבן ••••875 = הרכב של החשמלאי מנסור", lines[0])
        self.assertNotIn("1609875", lines[0])
        self.assertIn("vehicle", [r.type for r in view.relevant("מה עם הרכב של החשמלאי", [])])


if __name__ == "__main__":
    unittest.main()
