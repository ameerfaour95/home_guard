# tests/box/test_brain_house_agent.py
"""House state through the assistant: commands in code (any mode, no model), Undo, the status line, the model's
own tools for what the code does not parse, and the owner's yes or no to proposals."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import unittest

from test_brain_agent import FakeRegistry, Scripted, call, reply

from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.house_state import HouseStateStore


def at(*args) -> float:
    return dt.datetime(*args).timestamp()


ME = {"user_id": 1, "name": "Ameer"}


class HouseAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.clock = [at(2026, 10, 5, 23, 0)]
        self.store = HouseStateStore(os.path.join(self.root, "house_state.jsonl"), now=lambda: self.clock[0])

    def agent(self, big=None, fast=None, mode="guard") -> OwnerAgentV2:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=None, house=self.store, now=lambda: self.clock[0],
                            read_settings=lambda: {"alert_start_hour": 22, "alert_end_hour": 6})
        return OwnerAgentV2(big or Scripted([], "big"), FakeRegistry(mode),
                            ChatMemory(os.path.join(self.root, ".conversations")),
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: self.clock[0]), services,
                            fast_model=fast or Scripted([], "fast"), now=lambda: self.clock[0])

    def say(self, agent, text, when):
        self.clock[0] = when
        return agent.handle(text, "-5", ME)

    def test_a_full_day_in_both_modes_with_no_model_at_all(self) -> None:
        for mode in ("guard", "assistant"):
            with self.subTest(mode=mode):
                self.store = HouseStateStore(os.path.join(self.root, f"{mode}.jsonl"), now=lambda: self.clock[0])
                big, fast = Scripted([], "big"), Scripted([], "fast")
                agent = self.agent(big, fast, mode)
                out = self.say(agent, "הולכים לישון", at(2026, 10, 5, 23, 0))
                self.assertEqual(out.text, "✓ מצב הבית: בבית, ישנים, עד מחר 06:00")
                self.assertTrue(out.undo_token)
                out = self.say(agent, "מה המצב?", at(2026, 10, 6, 7, 0))         # the morning default
                self.assertTrue(out.text.startswith("🏠 מצב הבית: בבית, ערים (לפי לוח הלילה 00:00–06:00), עד 00:00"))
                out = self.say(agent, "יצאנו", at(2026, 10, 6, 8, 30))
                self.assertEqual(out.text, "✓ מצב הבית: לא בבית, עד שתגידו אחרת")
                out = self.say(agent, "status", at(2026, 10, 6, 13, 0))
                self.assertTrue(out.text.startswith("🏠 House: away (Ameer said so), until you tell me otherwise"))
                out = self.say(agent, "חזרנו", at(2026, 10, 6, 18, 0))
                self.assertEqual(out.text, "✓ מצב הבית: בבית, ערים, עד 00:00")
                out = self.say(agent, "מה המצב?", at(2026, 10, 7, 1, 0))
                self.assertTrue(out.text.startswith("🏠 מצב הבית: בבית, ישנים (לפי לוח הלילה 00:00–06:00), עד 06:00"))
                self.assertEqual((big.seen, fast.seen), ([], []))               # never a model

    def test_undo_puts_back_what_was_there(self) -> None:
        agent = self.agent()
        self.say(agent, "going to sleep", at(2026, 10, 5, 23, 0))
        left = self.say(agent, "we left", at(2026, 10, 5, 23, 10))
        out = agent.undo_turn("-5", left.undo_token, ME)
        self.assertEqual(out.text, "✓ House back to: home, asleep, until tomorrow 06:00")
        self.assertEqual((self.store.current().state, self.store.current().source), ("home_asleep", "owner"))
        self.assertEqual(agent.undo_turn("-5", left.undo_token, ME).text, t("nothing_to_undo", "en"))

    def test_a_bare_cancel_right_after_a_house_command_undoes_it(self) -> None:
        agent = self.agent()
        self.say(agent, "מצפים לחבילה היום", at(2026, 10, 6, 9, 0))
        out = self.say(agent, "בטל", at(2026, 10, 6, 9, 5))
        self.assertEqual(out.text, "✓ כבר לא מצפים ל: חבילה")
        self.assertEqual(self.store.current().expecting, [])
        late = self.say(agent, "בטל", at(2026, 10, 6, 11, 0))                   # long after: the model's turn
        self.assertEqual(late.text, t("unavailable", "he"))                     # (the scripted model is empty)

    def test_a_command_with_more_in_it_also_goes_to_the_model(self) -> None:
        big = Scripted([reply("אין תמונה כרגע.")], "big")
        fast = Scripted([call("hand_off", reason="x")], "fast")
        out = self.agent(big, fast).handle("יצאנו, ותבדוק שהשער סגור", "-5", ME)
        self.assertEqual(out.text, "אין תמונה כרגע.\n✓ מצב הבית: לא בבית, עד שתגידו אחרת")
        self.assertIn("[ALREADY DONE THIS TURN]", big.seen[0][0][-1])

    def test_the_model_sees_the_house_state_and_cannot_claim_a_change_it_did_not_make(self) -> None:
        big = Scripted([reply("I've set the house to vacation mode."), reply("I set the house to vacation.")], "big")
        out = self.agent(big).handle("we're going on vacation soon", "-5", ME)
        self.assertIn("[HOUSE STATE] home_awake (schedule) until 00:00", big.seen[0][0][-1])
        self.assertEqual(out.text, t("nothing_done", "en"))
        self.assertEqual(self.store.current().source, "schedule")

    def test_the_model_can_change_the_state_with_the_owners_words(self) -> None:
        self.clock[0] = at(2026, 10, 6, 9, 0)
        big = Scripted([call("house_state", state="vacation", until="2026-10-09", owner_words="לחופשה באילת"),
                        reply("חופשה נעימה!")], "big")
        out = self.agent(big).handle("אנחנו נוסעים לחופשה באילת עד סוף השבוע", "-5", ME)
        self.assertEqual(out.text, "חופשה נעימה!\n✓ מצב הבית: בחופשה, עד 09.10 23:59")
        self.assertEqual(self.store.current().state, "vacation")
        refused = Scripted([call("house_state", state="back", owner_words="we are back"), reply("ok")], "big")
        self.agent(refused).handle("is the family back?", "-5", ME)          # not asked: no change
        self.assertEqual(self.store.current().state, "vacation")

    def test_a_pending_proposal_is_answered_with_a_button(self) -> None:
        agent = self.agent()
        self.say(agent, "הולכים לישון", at(2026, 10, 5, 23, 0))
        self.clock[0] = at(2026, 10, 6, 2, 0)
        pid = self.store.set_house_state("home_awake", "system")["id"]
        out = agent.handle("status", "-5", ME)
        self.assertEqual(out.rows, ((("✓ Yes", f"hs:{pid}:y"), ("✗ No", f"hs:{pid}:n")),))
        answer = agent.answer_proposal("-5", pid, False, ME)
        self.assertEqual(answer.text, "✓ Declined: set the house to home, awake")
        self.assertEqual(self.store.current().state, "home_asleep")
        self.assertEqual(agent.answer_proposal("-5", pid, True, ME).text,
                         "✗ Answering the request could not be done: that request was already answered or has "
                         "expired")


if __name__ == "__main__":
    unittest.main()
