# tests/box/test_brain_house_run.py
"""House commands run through house_state's writer, are read back, and get a receipt in the owner's language."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import unittest

from test_brain_tools_read import snapshot

from home_guard_project.box.brain import house
from home_guard_project.box.brain.house import parse_command, run_command
from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import DONE, FAILED, ReceiptBook
from home_guard_project.box.brain.render import receipt_line
from home_guard_project.box.brain.tools import Services, ToolContext
from home_guard_project.box.feedback import Feedback, MuteState
from home_guard_project.box.house_state import HouseStateStore


def at(*args) -> float:
    return dt.datetime(*args).timestamp()


EVENING = at(2026, 10, 5, 22, 30)
MORNING = at(2026, 10, 6, 8, 0)


class HouseRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.clock = [EVENING]
        self.mute_path = os.path.join(self.root, "alert_mute.json")
        self.store = HouseStateStore(os.path.join(self.root, "house_state.jsonl"), mute_path=self.mute_path,
                                     now=lambda: self.clock[0])

    def ctx(self, lang: str = "he") -> ToolContext:
        services = Services(roots=lambda: [self.root], desc_dir=self.root, feedback_dir=self.root,
                            work_dir=self.root, mute=None, deliver=None, house=self.store,
                            now=lambda: self.clock[0])
        return ToolContext(turn_id=f"-5:{int(self.clock[0] * 1000)}", chat_id="-5", speaker={"user_id": 1, "name": "Ameer"},
                           text="", lang=lang, mode="guard", snapshot=snapshot("guard"), state=ChatState(),
                           services=services, book=ReceiptBook(os.path.join(self.root, ".receipts"),
                                                               now=lambda: self.clock[0]))

    def run_text(self, text: str, lang: str = "he"):
        ctx = self.ctx(lang)
        out = run_command(ctx, parse_command(text, self.clock[0]))
        return ctx, out

    def test_going_to_sleep_goes_through_the_writer_and_names_the_end(self) -> None:
        ctx, out = self.run_text("הולכים לישון")
        self.assertTrue(out.handled)
        receipt = ctx.receipts[0]
        self.assertEqual((receipt.tool, receipt.status, receipt.detail["state"]), ("house_state", DONE, "home_asleep"))
        now = self.store.current()
        self.assertEqual((now.state, now.source, now.by), ("home_asleep", "owner", "Ameer"))
        self.assertEqual(receipt_line(receipt, "he"), "✓ מצב הבית: בבית, ישנים, עד מחר 06:00")
        self.assertEqual(receipt_line(receipt, "en"), "✓ House: home, asleep, until tomorrow 06:00")

    def test_away_vacation_and_back(self) -> None:
        self.clock[0] = MORNING
        ctx, _ = self.run_text("we left", "en")
        self.assertEqual(receipt_line(ctx.receipts[0], "en"), "✓ House: away, until you tell me otherwise")
        ctx, _ = self.run_text("on vacation until 20.10", "en")
        self.assertEqual(receipt_line(ctx.receipts[0], "en"), "✓ House: on vacation, until 20.10 23:59")
        self.assertEqual(self.store.current().state, "vacation")
        ctx, _ = self.run_text("חזרנו")
        self.assertEqual(receipt_line(ctx.receipts[0], "he"), "✓ מצב הבית: בבית, ערים, עד 00:00")
        self.clock[0] = at(2026, 10, 7, 1, 0)                     # back after midnight: home and asleep
        ctx, _ = self.run_text("חזרנו")
        self.assertEqual((self.store.current().state, receipt_line(ctx.receipts[0], "he")),
                         ("home_asleep", "✓ מצב הבית: בבית, ישנים, עד 06:00"))

    def test_expecting_a_package_and_a_plumber(self) -> None:
        self.clock[0] = MORNING
        ctx, _ = self.run_text("מצפים לחבילה היום")
        self.assertEqual(receipt_line(ctx.receipts[0], "he"), "✓ מצפים ל: חבילה, עד 23:59")
        ctx, _ = self.run_text("the plumber is coming at 10", "en")
        self.assertEqual(receipt_line(ctx.receipts[0], "en"), "✓ Expecting: the plumber at 10:00, until 13:00")
        self.assertEqual([e["text"] for e in self.store.current().expecting], ["חבילה", "the plumber at 10:00"])

    def test_no_receipt_claims_what_the_writer_did_not_confirm(self) -> None:
        def refuse(*args, **kwargs):
            raise ValueError("until must be later than now and than since")

        self.store.set_house_state = refuse
        ctx, out = self.run_text("going to sleep", "en")
        self.assertEqual((ctx.receipts[0].status, out.handled), (FAILED, True))
        self.assertTrue(receipt_line(ctx.receipts[0], "en").startswith("✗ Changing the house state could not be done"))
        store = HouseStateStore(self.store.path, now=lambda: self.clock[0])
        store.set_house_state = lambda *a, **k: {"id": "H9", "status": "applied", "state": "home_asleep"}
        self.store = store                                        # "applied", but nothing in the file
        ctx, _ = self.run_text("going to sleep", "en")
        self.assertEqual((ctx.receipts[0].status, ctx.receipts[0].reason), (FAILED, "not_confirmed"))

    def test_cancel_a_vacation_a_note_or_nothing(self) -> None:
        self.clock[0] = MORNING
        self.run_text("בחופשה עד 20.10")
        ctx, _ = self.run_text("בטל את החופשה")
        self.assertEqual(receipt_line(ctx.receipts[0], "he"), "✓ בוטל: בחופשה. מצב הבית עכשיו: בבית, ערים, עד 00:00")
        self.assertEqual(self.store.current().source, "schedule")
        ctx, _ = self.run_text("cancel the vacation", "en")
        self.assertEqual((ctx.receipts[0].status, ctx.receipts[0].reason), (FAILED, "nothing_to_cancel"))
        self.run_text("מצפים לחבילה היום")
        ctx, _ = self.run_text("בטל את החבילה")
        self.assertEqual(receipt_line(ctx.receipts[0], "he"), "✓ כבר לא מצפים ל: חבילה")
        self.assertEqual(self.store.current().expecting, [])
        ctx, _ = self.run_text("cancel the plumber", "en")
        self.assertEqual(ctx.receipts[0].reason, "no_such_note")

    def test_a_bare_cancel_with_no_recent_house_command_is_not_ours(self) -> None:
        ctx = self.ctx()
        self.assertFalse(run_command(ctx, parse_command("בטל", self.clock[0])).handled)
        ctx.state.house_last = {"token": "123", "ts": self.clock[0] - 60}
        self.assertEqual(run_command(ctx, parse_command("בטל", self.clock[0])).undo, "123")
        ctx.state.house_last = {"token": "123", "ts": self.clock[0] - 3600}
        self.assertFalse(run_command(ctx, parse_command("בטל", self.clock[0])).handled)

    def test_the_status_line(self) -> None:
        self.clock[0] = MORNING
        self.run_text("מצפים לחבילה היום")
        MuteState(self.mute_path).apply(Feedback(action="mute", mute_until=MORNING + 3600, camera="front_side"),
                                        MORNING)
        ctx, out = self.run_text("מה המצב?")
        self.assertEqual(ctx.receipts, [])
        lines = out.text.split("\n")
        self.assertEqual(lines[0], "🏠 מצב הבית: בבית, ערים (לפי לוח הלילה 00:00–06:00), עד 00:00")
        self.assertEqual(lines[1], "📦 מצפים ל: חבילה עד 23:59")
        self.assertEqual(lines[2], "🔕 השתקות: front_side עד 09:00")
        self.assertEqual(lines[3], "📷 צופות: main_entrance, front_side (2 מתוך 3) · כבויות: back_door")
        self.assertEqual(out.rows, ())
        self.run_text("we left", "en")
        ctx, out = self.run_text("status", "en")
        self.assertTrue(out.text.startswith("🏠 House: away (Ameer said so), until you tell me otherwise"))
        self.assertIn("📦 Expecting: חבילה until 23:59", out.text)

    def test_a_proposal_gets_yes_and_no_buttons_and_the_answer_goes_through_the_writer(self) -> None:
        self.run_text("הולכים לישון")
        self.clock[0] = at(2026, 10, 6, 2, 0)
        proposal = self.store.set_house_state("home_awake", "system")          # loosening from the box itself
        self.assertEqual(proposal["status"], "pending")
        ctx, out = self.run_text("status", "en")
        self.assertIn("⏳ Waiting for your yes or no: set the house to home, awake (asked by the box, expires 14:00)",
                      out.text)
        pid = proposal["id"]
        self.assertEqual(out.rows, (((f"✓ Yes", f"hs:{pid}:y"), ("✗ No", f"hs:{pid}:n")),))
        ctx = self.ctx("en")
        house.answer_proposal(ctx, pid, True)
        self.assertEqual(receipt_line(ctx.receipts[0], "en"), "✓ Approved: set the house to home, awake")
        self.assertEqual(self.store.current().state, "home_awake")
        ctx = self.ctx("en")
        house.answer_proposal(ctx, pid, False)                                 # already answered
        self.assertEqual(ctx.receipts[0].reason, "proposal_gone")

    def test_context_line_for_the_model(self) -> None:
        self.clock[0] = MORNING
        self.run_text("מצפים לחבילה היום")
        line = house.context_line(self.store, MORNING)
        self.assertEqual(line, "[HOUSE STATE] home_awake (schedule) until 00:00 · expecting: חבילה until 23:59")


if __name__ == "__main__":
    unittest.main()
