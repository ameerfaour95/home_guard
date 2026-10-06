# tests/box/test_brain_followup.py
"""Chatting about an alert (§10): the history keeps the vision agent's observation as text, never the pictures;
a visual detail the observation does not give is asked of the clip (ask_vision), never made up."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from test_brain_agent import NOW, FakeRegistry, Scripted, call, reply
from test_brain_events import meta
from test_brain_tools_read import FakeVision

from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.tools import Services

ALERT_ID = "main_entrance_1_alert"


class FollowUpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.vision = FakeVision(ask_result={"ok": True, "answer": "The car is white.", "frame": 2, "seen": True})

    def agent(self, big: Scripted, summary: str) -> OwnerAgentV2:
        meta(self.root, "main_entrance", ALERT_ID, NOW - 120, summary=summary, label="normal", people=0)
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=None, vision=self.vision,
                            clip_frames_at=lambda path, count, start=None, end=None: [(0.0, b"a"), (2.0, b"b")],
                            read_settings=lambda: {"alert_start_hour": 22, "alert_end_hour": 6}, now=lambda: NOW)
        agent = OwnerAgentV2(big, FakeRegistry("guard"), ChatMemory(os.path.join(self.root, ".conversations")),
                             ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW), services,
                             now=lambda: NOW)
        agent.note_alert("-5", {"alert_id": ALERT_ID, "camera": "main_entrance", "ts": NOW - 120, "summary": summary,
                                "label": "normal", "visibility": "the plate is too far to read"})
        return agent

    def test_follow_ups_read_the_observation_as_text_and_no_pictures_are_resent(self) -> None:
        big = Scripted([reply("A man walked to the door.")])
        self.agent(big, "A man walks to the door.").handle("מה הוא עשה?", "-5", {"user_id": 1})
        contents = big.seen[0][0]
        self.assertTrue(all(isinstance(c, str) for c in contents))           # text only, never an image
        alert_line = next(c for c in contents if c.startswith("[ALERT E1"))
        self.assertIn("main_entrance", alert_line)
        self.assertIn("observation: A man walks to the door.", alert_line)
        self.assertIn("not visible: the plate is too far to read", alert_line)
        self.assertIn("[EVENT BEING DISCUSSED] E1", contents[-1])

    def test_a_colour_the_observation_does_not_give_calls_the_tool(self) -> None:
        big = Scripted([reply("The car is red.")])                             # made up
        out = self.agent(big, "A car parks at the gate.").handle("what color is the car?", "-5", {"user_id": 1})
        self.assertIn("ask_vision", out.tools_called)
        self.assertEqual(self.vision.asked, [("main_entrance", "what color is the car?", "English", 2)])
        self.assertTrue(out.text.startswith(t("checking_clip", "en")))
        self.assertIn("The car is white.", out.text)
        self.assertNotIn("red", out.text)

    def test_a_question_the_observation_answers_does_not_call_the_tool(self) -> None:
        big = Scripted([reply("The car is white.")])
        out = self.agent(big, "A white car parks at the gate.").handle("what color is the car?", "-5",
                                                                        {"user_id": 1})
        self.assertNotIn("ask_vision", out.tools_called)
        self.assertEqual((out.text, self.vision.asked), ("The car is white.", []))

    def test_a_vision_answer_is_kept_in_the_history_for_the_next_question(self) -> None:
        self.vision.ask_result = {"ok": True, "answer": "A phone.", "frame": 2, "seen": True}
        big = Scripted([call("ask_vision", question="what was in his hand?"), reply("He was holding a phone."),
                        reply("Yes, a phone.")])
        agent = self.agent(big, "A man walks to the door.")
        first = agent.handle("what was in his hand?", "-5", {"user_id": 1})
        self.assertEqual(first.text, "He was holding a phone.")              # backed by the tool's answer
        agent.handle("a phone, you said?", "-5", {"user_id": 1})
        history = "\n".join(big.seen[-1][0])
        self.assertIn('asked "what was in his hand?": A phone.', history)
        self.assertEqual(len(self.vision.asked), 1)

    def test_a_live_look_ends_the_talk_about_the_alert_and_its_description_counts(self) -> None:
        shot = os.path.join(self.root, "live.jpg")
        with open(shot, "wb") as f:
            f.write(b"jpg")
        big = Scripted([call("check_camera", camera="entrance"), reply("A man in a red shirt is at the door."),
                        reply("His shirt is red.")])
        agent = self.agent(big, "A car parks at the gate.")
        agent.services.grab_photo = lambda camera: {"camera": camera, "image": shot}
        agent.services.deliver = type("D", (), {"photo": lambda self, chat, path, caption="": {"ok": True}})()
        self.vision.result = {"ok": True, "description": "A man in a red shirt stands at the door.",
                              "quality": "clear", "people": 1, "label": "normal", "why": ""}
        first = agent.handle("who is at the door now?", "-5", {"user_id": 1})
        self.assertNotIn("ask_vision", first.tools_called)                  # the live description backs it
        self.assertIsNone(agent.memory.load("-5").topic_event(NOW))
        second = agent.handle("what colour is his shirt?", "-5", {"user_id": 1})
        self.assertEqual((second.text, self.vision.asked), ("His shirt is red.", []))

    def test_the_same_alert_twice_is_one_history_entry(self) -> None:
        big = Scripted([reply("ok")])
        agent = self.agent(big, "A man walks to the door.")
        agent.note_alert("-5", {"alert_id": ALERT_ID, "camera": "main_entrance", "ts": NOW - 120,
                                "summary": "A man walks to the door."})      # the escalation reminder
        self.assertEqual(sum(1 for turn in agent.memory.load("-5").turns if turn.get("kind") == "alert"), 1)
        self.assertIsNone(agent.note_alert("-5", {"camera": "x"}))           # no id: nothing to note


if __name__ == "__main__":
    unittest.main()
