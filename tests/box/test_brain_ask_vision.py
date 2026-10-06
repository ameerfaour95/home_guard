# tests/box/test_brain_ask_vision.py
"""ask_vision: a follow-up question about a saved clip, answered from its frames with the frame that shows it."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import unittest

from test_brain_events import meta
from test_brain_tools_read import NOW, HOUR, FakeVision, snapshot

from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.tools import MAX_DESCRIBE_PER_TURN, Services, ToolContext, ask_vision, find_events


class AskVisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.ts = NOW - 5 * HOUR                     # the clip runs ts-10 .. ts, the detector fired at ts-6
        meta(self.root, "main_entrance", "main_entrance_1_alert", self.ts,
             summary="A man walks to the door.", label="normal", people=1)
        meta(self.root, "back_door", "back_door_2_alert", self.ts, summary="A woman.", with_clip=False)
        self.vision = FakeVision()
        self.frames = []

    def clip_frames_at(self, path, count, start=None, end=None):
        self.frames.append((count, start, end))
        return [(0.0, b"a"), (2.5, b"b"), (5.0, b"c")]

    def ctx(self, vision="default") -> ToolContext:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=None, vision=self.vision if vision == "default" else vision,
                            clip_frames_at=self.clip_frames_at, now=lambda: NOW)
        return ToolContext(turn_id="t1", chat_id="-5", speaker={}, text="מה היה לו ביד?", lang="he",
                           mode="guard", snapshot=snapshot("guard"), state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW))

    def handle(self, ctx: ToolContext, camera: str = "main_entrance") -> str:
        return next(e["handle"] for e in find_events(ctx, {"last_hours": 24})["events"] if e["camera"] == camera)

    def test_answers_with_the_frame_and_its_time_and_keeps_the_answer(self) -> None:
        ctx = self.ctx()
        handle = self.handle(ctx)
        out = ask_vision(ctx, {"event_id": handle, "question": "what was in his hand?"})
        at = dt.datetime.fromtimestamp(self.ts - 10 + 2.5).strftime("%H:%M:%S")
        self.assertEqual((out["ok"], out["answer"], out["frame"], out["time"]), (True, "A phone.", 2, at))
        self.assertEqual(self.vision.asked, [("main_entrance", "what was in his hand?", "Hebrew", 3)])
        self.assertEqual(self.frames, [(8, None, None)])                 # more frames than the alert's look
        self.assertIn(f'asked "what was in his hand?": A phone. (frame 2, {at})', ctx.state.event_text(handle))
        self.assertEqual(ctx.vision_notes, [f'{handle} asked "what was in his hand?": A phone. (frame 2, {at})'])
        self.assertEqual(ctx.state.topic_event(NOW), handle)

    def test_the_same_question_is_answered_from_memory(self) -> None:
        ctx = self.ctx()
        handle = self.handle(ctx)
        ask_vision(ctx, {"event_id": handle, "question": "what was in his hand?"})
        again = ask_vision(ctx, {"event_id": handle, "question": "What was in his hand?"})
        self.assertEqual((again["answer"], again["cached"]), ("A phone.", True))
        self.assertEqual(len(self.vision.asked), 1)

    def test_no_event_named_uses_the_event_being_discussed(self) -> None:
        ctx = self.ctx()
        handle = self.handle(ctx)
        ctx.state.set_topic_event(handle, NOW)
        self.assertTrue(ask_vision(ctx, {"question": "what colour was his shirt?"})["ok"])
        fresh = self.ctx()
        self.assertIn("which event", ask_vision(fresh, {"question": "what colour?"})["error"])

    def test_a_time_range_is_seconds_from_the_detector_trigger(self) -> None:
        ctx = self.ctx()
        ask_vision(ctx, {"event_id": self.handle(ctx), "question": "q", "time_range": {"from_sec": -2, "to_sec": 2}})
        self.assertEqual(self.frames, [(8, 2.0, 6.0)])                    # trigger is 4 s into the clip
        self.assertFalse(ask_vision(ctx, {"event_id": "E1", "question": "q", "time_range": "the start"})["ok"])

    def test_failures_are_explicit(self) -> None:
        ctx = self.ctx()
        self.assertIn("no longer on the box", ask_vision(ctx, {"event_id": self.handle(ctx, "back_door"),
                                                               "question": "q"})["error"])
        self.assertIn("not available", ask_vision(self.ctx(vision=None), {"event_id": "E1", "question": "q"})["error"])
        refused = self.ctx(vision=FakeVision(ask_result={"ok": False, "refused": True, "error": "refused"}))
        out = ask_vision(refused, {"event_id": self.handle(refused), "question": "q"})
        self.assertEqual((out["ok"], out["error"]), (False, "refused"))
        capped = self.ctx()
        handle = self.handle(capped)
        for i in range(MAX_DESCRIBE_PER_TURN):
            self.assertTrue(ask_vision(capped, {"event_id": handle, "question": f"q{i}"})["ok"])
        self.assertFalse(ask_vision(capped, {"event_id": handle, "question": "one more"})["ok"])


if __name__ == "__main__":
    unittest.main()
