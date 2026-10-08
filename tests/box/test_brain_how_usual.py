"""Task 2.9: the assistant's side of the per-camera memory - how_usual ("זה רגיל?", "כמה פעמים זה קורה?", "מה רגיל
בחצר האחורית בלילה?") answers from the box's own counts, and camera_fact keeps what the owner teaches about a camera
or the house, from the owner's own words, with a code-written receipt and Undo. Names only, never camera ids."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest

from test_brain_agent import FakeRegistry, Scripted, call, reply

from home_guard_project.box import baseline as bl
from home_guard_project.box.brain import profiles
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.memory import ChatMemory, ChatState
from home_guard_project.box.brain.receipts import DONE, ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.render import receipt_line
from home_guard_project.box.brain.tools import TOOLS, Services, ToolContext, camera_fact, how_usual
from home_guard_project.box.camera_profiles import CameraProfiles
from home_guard_project.box.events import EventBook

MAIN, BACK = "ameer_week_0_1_ch6", "ameer_week_0_1_ch1"
FIRST = dt.datetime(2026, 9, 1)
NOW = (FIRST + dt.timedelta(days=20, hours=10)).timestamp()


def at(day: int, hour: int, minute: int = 0) -> float:
    return (FIRST + dt.timedelta(days=day, hours=hour, minutes=minute)).timestamp()


def house() -> HouseSnapshot:
    return HouseSnapshot(now=NOW, mode="assistant", mode_ends=NOW + 3600, mode_started=NOW - 3600, start_hour=0,
                         end_hour=0, cameras=(CameraState(MAIN, True, ("הכניסה הראשית",), live=True),
                                              CameraState(BACK, True, ("הדלת האחורית",), live=True)))


def record(event_id, camera, start, summary):
    return {"event_id": event_id, "camera": camera, "start": start, "end": start + 240, "people_max": 1,
            "outcome": "left", "observations": [{"ts": start, "people": 1, "summary": summary}]}


class HowUsualTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.dir = os.path.join(self.root, "events")
        self.events = EventBook(self.dir)

    def build(self, days: int = 20) -> None:
        rows = []
        for d in range(days):
            rows.append(record(f"m{d}", MAIN, at(d, 8, 10), "A courier knocks on the door and leaves a package"))
            rows.append(record(f"b{d}", BACK, at(d, 12, 5), "A man walks across the yard"))
        bl.Baseline(os.path.join(self.dir, bl.BASELINE_NAME)).build_from_archive(rows, [MAIN, BACK],
                                                                                  now=at(days, 3, 30))

    def ctx(self, text: str, lang: str = "he") -> ToolContext:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=None, now=lambda: NOW, events=self.events)
        return ToolContext(turn_id="-5:1", chat_id="-5", speaker={"user_id": 1, "name": "Ameer"}, text=text,
                           lang=lang, mode="assistant", snapshot=house(), state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.root, ".r"), now=lambda: NOW))

    # -- registration ----------------------------------------------------------------------------------------------
    def test_registered_with_schemas_and_one_prompt_line(self) -> None:
        self.assertIn("how_usual", TOOLS)
        self.assertIn("camera_fact", TOOLS)
        for mode in ("guard", "assistant"):
            self.assertIn("how_usual", profiles.tool_names(mode, "big"))
            self.assertIn("camera_fact", profiles.tool_names(mode, "big"))
            self.assertIn("how_usual", profiles.tool_names(mode, "fast"))
            self.assertNotIn("camera_fact", profiles.tool_names(mode, "fast"))     # the fast model never writes
        schemas = {s["function"]["name"]: s["function"] for s in profiles.tools_for("assistant")}
        self.assertEqual(schemas["camera_fact"]["parameters"]["required"], ["owner_words"])
        self.assertEqual(schemas["how_usual"]["parameters"]["required"], [])
        prompt = profiles.system_prompt("assistant", 14)
        self.assertIn("how_usual", prompt)
        self.assertIn("camera_fact", prompt)

    # -- how_usual -------------------------------------------------------------------------------------------------
    def test_the_knock_at_the_main_entrance_and_at_the_back_door(self) -> None:
        self.build()
        out = how_usual(self.ctx("דופקים בדלת האחורית, זה רגיל?"), {"camera": "הדלת האחורית",
                                                                     "activity": "דפיקה בדלת"})
        self.assertTrue(out["ok"])
        self.assertEqual(out["answer"].split("; ")[0], "דפיקה בדלת האחורית: אף פעם ב-20 הימים האחרונים")
        self.assertIn("בכניסה הראשית: בערך פעם ביום", out["answer"])
        text = json.dumps(out, ensure_ascii=False)
        self.assertNotIn("ameer_week", text)
        self.assertNotIn("_ch", text)

    def test_what_is_usual_at_night(self) -> None:
        self.build()
        out = how_usual(self.ctx("מה רגיל בדלת האחורית בלילה?"), {"camera": "הדלת האחורית", "time": "בלילה"})
        self.assertEqual(out["asked_part_of_day"]["days_with_people"], 0)
        self.assertTrue(out["answer"].startswith("הדלת האחורית בלילה (21-24): אנשים אף פעם"))
        general = how_usual(self.ctx("what is usual at the main entrance?", "en"), {"camera": "הכניסה הראשית"})
        self.assertIn("busiest 08:00-09:00", general["answer"])
        self.assertIn("a knock at the door - about once a day", general["answer"])

    def test_is_this_usual_about_the_alert_being_discussed(self) -> None:
        self.build()
        ctx = self.ctx("זה רגיל?")
        ctx.alert_handle = ctx.state.add_handle("event", f"{BACK}_1_alert", BACK, at(20, 3, 10),
                                                "A man knocks on the back door")
        out = how_usual(ctx, {})
        self.assertEqual(out["camera"], "הדלת האחורית")
        self.assertEqual(out["at_that_time"]["rarity"], "rare")
        self.assertIn("לא רגיל למצלמה הזו", out["answer"])
        self.assertIn("בדרך כלל רק בכניסה הראשית", out["answer"])

    def test_short_history_says_so_and_taught_facts_are_listed(self) -> None:
        self.build(3)
        CameraProfiles(os.path.join(self.dir, "camera_profiles.json")).add_fact(BACK, "בדלת האחורית משתמשים רק אנחנו")
        out = how_usual(self.ctx("מה רגיל בדלת האחורית?"), {"camera": "הדלת האחורית"})
        self.assertFalse(out["enough_data"])
        self.assertIn("יש לי רק 3 ימים של היסטוריה", out["answer"])
        self.assertIn("מה שלימדת: בדלת האחורית משתמשים רק אנחנו", out["answer"])

    def test_without_the_event_book(self) -> None:
        ctx = self.ctx("x")
        ctx.services.events = None
        self.assertFalse(how_usual(ctx, {"camera": "הדלת האחורית"})["ok"])
        self.assertFalse(camera_fact(ctx, {"owner_words": "x"})["ok"])

    # -- camera_fact -----------------------------------------------------------------------------------------------
    def test_a_fact_from_the_owners_words_with_a_code_receipt(self) -> None:
        text = "תזכור: בדלת האחורית משתמשים רק אנחנו"
        ctx = self.ctx(text)
        out = camera_fact(ctx, {"camera": "הדלת האחורית", "owner_words": "בדלת האחורית משתמשים רק אנחנו"})
        self.assertTrue(out["ok"])
        (receipt,) = ctx.receipts
        self.assertEqual((receipt.tool, receipt.status), ("camera_fact", DONE))
        self.assertEqual(receipt.detail["rule"], "family_only")
        line = receipt_line(receipt, "he", snapshot=house())
        self.assertEqual(line, "שמרתי על הדלת האחורית: \"בדלת האחורית משתמשים רק אנחנו\". זה נשאר עד שתבקש למחוק.")
        store = CameraProfiles(os.path.join(self.dir, "camera_profiles.json"))
        self.assertEqual([f["text"] for f in store.facts(BACK)], ["בדלת האחורית משתמשים רק אנחנו"])

    def test_words_not_in_the_message_are_refused(self) -> None:
        ctx = self.ctx("מה רגיל בדלת האחורית?")
        out = camera_fact(ctx, {"camera": "הדלת האחורית", "owner_words": "אף אחד לא מגיע לדלת האחורית"})
        self.assertFalse(out["ok"])
        self.assertEqual(ctx.receipts, [])

    def test_house_facts_roles_and_removal(self) -> None:
        ctx = self.ctx("יש לנו כלב")
        self.assertTrue(camera_fact(ctx, {"owner_words": "יש לנו כלב", "whole_house": True})["ok"])
        self.assertEqual(receipt_line(ctx.receipts[0], "he", snapshot=house()),
                         "שמרתי על הבית: \"יש לנו כלב\". זה נשאר עד שתבקש למחוק.")
        ctx = self.ctx("זו הכניסה הראשית")
        self.assertTrue(camera_fact(ctx, {"camera": "הכניסה הראשית", "owner_words": "זו הכניסה הראשית",
                                          "role": "entrance"})["ok"])
        self.assertEqual(receipt_line(ctx.receipts[0], "he", snapshot=house()), "שמרתי: הכניסה הראשית - כניסה.")
        ctx = self.ctx("תשכח את מה שאמרתי על הכלב")
        out = camera_fact(ctx, {"owner_words": "תשכח את מה שאמרתי על הכלב", "whole_house": True, "remove": True,
                                "fact": "כלב"})
        self.assertTrue(out["ok"])
        self.assertEqual(receipt_line(ctx.receipts[0], "he", snapshot=house()),
                         "מחקתי ממה שאני יודע על הבית: \"יש לנו כלב\".")
        self.assertEqual(CameraProfiles(os.path.join(self.dir, "camera_profiles.json")).house_facts(), [])


class CameraFactUndoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.events = EventBook(os.path.join(self.root, "events"))
        self.store = CameraProfiles(os.path.join(self.root, "events", "camera_profiles.json"))

    def agent(self, big) -> OwnerAgentV2:
        now = dt.datetime(2026, 10, 3, 23, 0).timestamp()
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=self.root, mute=None, deliver=None, now=lambda: now,
                            read_settings=lambda: {"owner_language": "he"}, events=self.events)
        return OwnerAgentV2(big, FakeRegistry(), ChatMemory(os.path.join(self.root, "c")),
                            ReceiptBook(os.path.join(self.root, "r"), now=lambda: now), services, now=lambda: now)

    def test_undo_a_saved_fact_and_a_removal(self) -> None:
        text = "בדלת האחורית משתמשים רק אנחנו"
        agent = self.agent(Scripted([call("camera_fact", camera="back_door", owner_words=text), reply("")]))
        first = agent.handle(text, "-5", {"user_id": 1})
        self.assertIn("שמרתי על", first.text)
        self.assertNotIn("back_door", first.text)
        self.assertTrue(first.undo_token)
        self.assertEqual(len(self.store.facts("back_door")), 1)
        undone = agent.undo_turn("-5", first.undo_token, {"user_id": 1})
        self.assertEqual(self.store.facts("back_door"), [])
        self.assertIn("מחקתי", undone.text)

        self.store.add_fact("back_door", text)
        forget = "תשכח את מה שאמרתי על הדלת האחורית"
        agent = self.agent(Scripted([call("camera_fact", camera="back_door", owner_words=forget, remove=True,
                                          fact="רק אנחנו"), reply("")]))
        second = agent.handle(forget, "-5", {"user_id": 1})
        self.assertEqual(self.store.facts("back_door"), [])
        agent.undo_turn("-5", second.undo_token, {"user_id": 1})
        self.assertEqual([f["text"] for f in self.store.facts("back_door")], [text])


if __name__ == "__main__":
    unittest.main()
