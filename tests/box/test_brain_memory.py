# tests/box/test_brain_memory.py
from __future__ import annotations

import json
import os
import tempfile
import unittest

from home_guard_project.box.brain.memory import KEEP_HANDLES, MAX_HISTORY_TURNS, TOPIC_SECONDS, ChatMemory, ChatState

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

    def test_topic_camera_and_event_are_kept_for_an_hour(self) -> None:
        s = ChatState()
        self.assertIsNone(s.topic_camera(TS))
        s.set_topic_camera("camera_3", "פרגולה", TS)
        self.assertEqual(s.topic_camera(TS + 60), ("camera_3", "פרגולה"))
        self.assertIsNone(s.topic_camera(TS + TOPIC_SECONDS + 1))
        h = s.add_handle("event", "camera_3_1_alert", "camera_3", TS, "A man walks to the gate.")
        s.set_topic_event(h, TS)
        self.assertEqual(s.topic_event(TS + 60), "E1")
        self.assertIsNone(s.topic_event(TS + TOPIC_SECONDS + 1))
        s.set_topic_event("E9", TS)                      # an unknown handle is no topic
        self.assertIsNone(s.topic_event(TS))

    def test_an_alert_goes_into_the_history_as_text_without_images(self) -> None:
        s = ChatState()
        h = s.add_handle("event", "camera_3_1_alert", "camera_3", TS, "A man in a grey hoodie walks to the gate.")
        s.note_observation(h, "A man in a grey hoodie walks to the gate.", "face hidden by the hood", "suspicious")
        s.add_event_turn(h, TS)
        s.add_answer(h, "what was in his hand?", "A phone.", 3, "14:02:05")
        s.add_turn("u1", "מה היה לו ביד?", "טלפון.", [h], [], TS + 30)
        msgs = s.history_messages(now=TS + 60)
        self.assertEqual(msgs[0]["role"], "assistant")
        text = msgs[0]["content"]
        self.assertIn("[ALERT E1", text)
        self.assertIn("camera_3", text)
        self.assertIn("A man in a grey hoodie walks to the gate.", text)
        self.assertIn("not visible: face hidden by the hood", text)
        self.assertLess(len(text), 900)                  # about 150 tokens, never a picture
        self.assertEqual(msgs[1], {"role": "user", "content": "מה היה לו ביד?"})
        self.assertIn('asked "what was in his hand?": A phone. (frame 3, 14:02:05)', s.event_text(h))

    def test_topics_and_observations_survive_a_round_trip_and_bad_files(self) -> None:
        s = ChatState()
        h = s.add_handle("event", "x", "camera_3", TS, "a car")
        s.note_observation(h, "a car", "", "")
        s.set_topic_camera("camera_3", "", TS)
        s.set_topic_event(h, TS)
        back = ChatState.from_dict(s.to_dict())
        self.assertEqual(back.to_dict(), s.to_dict())
        bad = ChatState.from_dict({"version": 2, "topic": "x", "topic_event_ref": {"handle": 5, "ts": "y"}})
        self.assertIsNone(bad.topic_camera(TS))
        self.assertIsNone(bad.topic_event(TS))


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

    def test_a_hand_edited_v2_file_never_raises_and_handles_never_collide(self) -> None:
        with open(os.path.join(self.dir, "-5.json"), "w", encoding="utf-8") as f:
            json.dump({"version": 2,
                       "turns": ["x", {"text": "q", "reply": "a", "ts": "soon"}],
                       "handles": {"E1": "bad", "Z9": {}, "E3": {"kind": "event", "ref": "r", "ts": "x"}},
                       "next_handle": "two",
                       "overrides": {"u1": "fr"},
                       "languages": {"u2": 5}}, f)
        s = self.memory.load("-5")
        s.history_messages(TS)
        self.assertEqual(s.add_handle("event", "new"), "E4")
        self.assertIn("E3", s.handles)
        self.assertEqual(s.language_for("u1", "8"), "en")

    def test_a_v1_file_with_bad_messages_is_a_fresh_chat(self) -> None:
        with open(os.path.join(self.dir, "-5.json"), "w", encoding="utf-8") as f:
            json.dump({"messages": "oops"}, f)
        self.assertEqual(self.memory.load("-5").to_dict(), ChatState().to_dict())

    def test_save_of_unserializable_state_does_not_raise_or_leave_tmp(self) -> None:
        s = ChatState()
        s.pending = {"question": "q", "choices": [object()]}
        self.memory.save("-5", s)
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])


if __name__ == "__main__":
    unittest.main()
