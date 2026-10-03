# tests/box/test_brain_memory.py
from __future__ import annotations

import json
import os
import tempfile
import unittest

from home_guard_project.box.brain.memory import KEEP_HANDLES, MAX_HISTORY_TURNS, ChatMemory, ChatState

TS = 1_790_000_000.0


class ChatStateTest(unittest.TestCase):
    def test_handles_are_stable_and_reused(self) -> None:
        s = ChatState()
        a = s.add_handle("event", "main_entrance_1_alert", "main_entrance", TS, "a man at the door")
        b = s.add_handle("event", "back_door_2_alert", "back_door", TS)
        self.assertEqual((a, b), ("E1", "E2"))
        self.assertEqual(s.add_handle("event", "main_entrance_1_alert", "main_entrance", TS), "E1")
        self.assertEqual(s.resolve(" e1 ")["ref"], "main_entrance_1_alert")
        self.assertIsNone(s.resolve("E9"))

    def test_handles_are_capped(self) -> None:
        s = ChatState()
        for i in range(KEEP_HANDLES + 5):
            s.add_handle("event", f"a{i}")
        self.assertEqual(len(s.handles), KEEP_HANDLES)
        self.assertIsNone(s.resolve("E1"))
        self.assertIsNotNone(s.resolve(f"E{KEEP_HANDLES + 5}"))

    def test_language_follows_the_speaker_and_short_replies_inherit_it(self) -> None:
        s = ChatState()
        self.assertEqual(s.language_for("u1", "תכבה את המצלמה"), "he")
        self.assertEqual(s.language_for("u1", "8"), "he")
        self.assertEqual(s.language_for("u2", "what happened"), "en")
        self.assertEqual(s.language_for("u1", "answer in English please"), "en")
        self.assertEqual(s.language_for("u1", "מה קורה בכניסה"), "en")   # the explicit choice sticks
        self.assertEqual(s.language_for("u3", "👍"), "en")
        self.assertEqual(s.language_for("u4", "👍", default="he"), "he")       # the box's configured language
        self.assertEqual(s.language_for("u5", "مرحبا، ماذا يحدث؟", default="he"), "he")   # not spoken yet

    def test_history_carries_handles_and_receipts(self) -> None:
        s = ChatState()
        h = s.add_handle("event", "main_entrance_1_alert", "main_entrance", TS)
        s.add_turn("u1", "send me the video", "Here it is.", [h], ["R1 send_media E1 done"], TS)
        msgs = s.history_messages(now=TS + 60)
        self.assertEqual(msgs[0], {"role": "user", "content": "send me the video"})
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertIn("Here it is.", msgs[1]["content"])
        self.assertIn("E1=event main_entrance", msgs[1]["content"])
        self.assertIn("R1 send_media E1 done", msgs[1]["content"])
        self.assertEqual(s.last_turn_handles(), ["E1"])

    def test_history_is_the_last_24_hours_capped_at_60_turns(self) -> None:
        s = ChatState()
        s.add_turn("u1", "yesterday morning", "ok", [], [], TS - 30 * 3600)
        for i in range(70):
            s.add_turn("u1", f"q{i}", "a", [], [], TS - 3600 + i)
        msgs = s.history_messages(now=TS)
        self.assertEqual(len(msgs), 2 * MAX_HISTORY_TURNS)
        self.assertNotIn("yesterday morning", [m["content"] for m in msgs])
        self.assertEqual(msgs[-2]["content"], "q69")


class ChatMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.memory = ChatMemory(self.dir)

    def test_round_trip(self) -> None:
        s = self.memory.load("-5")
        s.add_handle("event", "x")
        s.pending = {"question": "Which camera?", "choices": ["a", "b"], "ts": TS}
        s.add_turn("u1", "hi", "hello", ["E1"], [], TS)
        self.memory.save("-5", s)
        back = self.memory.load("-5")
        self.assertEqual(back.to_dict(), s.to_dict())

    def test_a_v1_file_is_upgraded(self) -> None:
        with open(os.path.join(self.dir, "-5.json"), "w", encoding="utf-8") as f:
            json.dump({"messages": [{"role": "user", "content": "q1", "ts": TS},
                                    {"role": "assistant", "content": "a1", "ts": TS}]}, f)
        s = self.memory.load("-5")
        self.assertEqual([(t["text"], t["reply"]) for t in s.turns], [("q1", "a1")])

    def test_damaged_file_is_a_fresh_chat(self) -> None:
        with open(os.path.join(self.dir, "-5.json"), "w", encoding="utf-8") as f:
            f.write("{oops")
        self.assertEqual(self.memory.load("-5").turns, [])


if __name__ == "__main__":
    unittest.main()
