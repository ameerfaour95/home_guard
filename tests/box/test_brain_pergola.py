# tests/box/test_brain_pergola.py
"""The pergola conversation of 2026-10-05, replayed through the real agent, registry and alias file.

The owner wrote: "remember that if I talk to you about the pergola, it's camera 3" - the assistant said it would
remember and saved nothing; "are there people working near the pergola?" went to the fast model, which searched
the history; "give me a picture" had no camera and the model guessed one. The models are scripted; what is
checked is what the code guarantees around them.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import unittest
from typing import Any, List

import yaml

from test_brain_agent import Scripted, call, reply
from test_brain_tools_act import FakeDeliver

from home_guard_project.box.brain import aliases
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import HouseRegistry
from home_guard_project.box.brain.tools import Services

NOW = dt.datetime(2026, 10, 5, 15, 0).timestamp()

REMEMBER = "תזכור שאם אני מדבר איתך על הפרגולה זה מצלמה 3"
WORKERS = "יש אנשים שעובדים ליד הפרגולה?"
PICTURE = "תביא לי תמונה"


class NoMute:
    def muted_until(self, now: float, camera: str = "") -> Any:
        return None


class PergolaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.cameras = os.path.join(self.root, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": {f"camera_{i}": {"url": "rtsp://x"} for i in range(1, 6)}}, f)
        self.aliases = os.path.join(self.root, "camera_aliases.yaml")
        aliases.add_alias("camera_1", "כניסה", [f"camera_{i}" for i in range(1, 6)], self.aliases)
        self.photo = os.path.join(self.root, "live.jpg")
        with open(self.photo, "wb") as f:
            f.write(b"jpg")
        self.grabbed: List[str] = []
        self.deliver = FakeDeliver()

    def grab(self, camera: str) -> dict:
        self.grabbed.append(camera)
        return {"camera": camera, "image": self.photo}

    def agent(self, big: Scripted, fast: Scripted) -> OwnerAgentV2:
        services = Services(
            roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"), feedback_dir=self.root,
            work_dir=os.path.join(self.root, ".live"), mute=NoMute(), deliver=self.deliver, grab_photo=self.grab,
            add_alias=lambda cam, alias, cams: aliases.add_alias(cam, alias, cams, self.aliases),
            remove_alias=lambda cam, alias: aliases.remove_alias(cam, alias, self.aliases),
            read_settings=lambda: {"alert_start_hour": 22, "alert_end_hour": 6, "owner_language": "he"},
            now=lambda: NOW)
        registry = HouseRegistry(NoMute(), lambda: (22, 6), self.cameras, self.aliases,
                                 os.path.join(self.root, "status.json"), now=lambda: NOW)
        return OwnerAgentV2(big, registry, ChatMemory(os.path.join(self.root, ".conversations")),
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW), services,
                            fast_model=fast, now=lambda: NOW)

    def test_the_owners_conversation_ends_with_a_photo_from_camera_3(self) -> None:
        fast = Scripted([call("check_camera", camera="פרגולה"), reply("לא רואים אנשים ליד הפרגולה."),
                         call("check_camera"), reply("זו המצלמה של הפרגולה.")], "fast")
        big = Scripted([call("set_alias", camera="מצלמה 3", alias="פרגולה"), reply("אזכור.")], "big")
        agent = self.agent(big, fast)

        first = agent.handle(REMEMBER, "-5", {"user_id": 1})
        self.assertEqual(first.tier, "big")                       # "תזכור" is a big-model word now
        self.assertEqual(fast.seen, [])
        self.assertIn("מצלמה 3 = camera_3", big.seen[0][0][-1])  # resolved in code before the model
        self.assertEqual(aliases.load_aliases(self.aliases)["camera_3"], ["פרגולה"])
        self.assertIn('✓ "פרגולה" מעכשיו זה camera 3', first.text)
        self.assertTrue(first.undo_token)                         # the receipt has an Undo button
        self.assertEqual(self.grabbed, ["camera_3"])              # ... and a photo of the camera itself

        second = agent.handle(WORKERS, "-5", {"user_id": 1})
        self.assertEqual(second.tier, "fast")
        block = fast.seen[0][0][-1]
        self.assertIn("פרגולה = camera_3", block)
        self.assertIn("[CAMERA BEING DISCUSSED] camera_3 (פרגולה)", block)
        self.assertIn("[RIGHT NOW]", block)                       # a live look, not a history search
        self.assertEqual(self.grabbed[-1], "camera_3")

        third = agent.handle(PICTURE, "-5", {"user_id": 1})
        self.assertEqual(third.tools_called, ("check_camera",))
        self.assertEqual(self.grabbed[-1], "camera_3")            # the photo comes from camera 3
        self.assertIn("(פרגולה)", third.text)                  # and the reply names it by the family's name
        self.assertNotIn("camera_3", third.text)

    def test_a_remember_promise_without_a_save_becomes_not_saved_yet(self) -> None:
        big = Scripted([reply("אזכור שהפרגולה זה מצלמה 3."), reply("בסדר, אני זוכר.")], "big")
        out = self.agent(big, Scripted([], "fast")).handle(REMEMBER, "-5", {"user_id": 1})
        self.assertEqual(out.text, t("not_saved_yet", "he"))
        self.assertNotIn("camera_3", aliases.load_aliases(self.aliases))
        self.assertEqual(out.guard_hits, 2)

    def test_the_rewrite_may_still_save_the_name(self) -> None:
        big = Scripted([reply("אזכור."), call("set_alias", camera="camera_3", alias="פרגולה"), reply("בסדר.")],
                       "big")
        out = self.agent(big, Scripted([], "fast")).handle(REMEMBER, "-5", {"user_id": 1})
        self.assertEqual(aliases.load_aliases(self.aliases)["camera_3"], ["פרגולה"])
        self.assertIn('✓ "פרגולה" מעכשיו זה camera 3', out.text)

    def test_a_picture_with_no_camera_being_discussed_asks_with_camera_buttons(self) -> None:
        fast = Scripted([call("check_camera")], "fast")
        out = self.agent(Scripted([], "big"), fast).handle(PICTURE, "-5", {"user_id": 1})
        self.assertEqual(out.text, "על איזו מצלמה?")
        self.assertEqual(out.buttons, tuple(f"camera_{i}" for i in range(1, 6)))
        self.assertEqual(self.grabbed, [])                        # no guess

    def test_undo_takes_the_name_back(self) -> None:
        big = Scripted([call("set_alias", camera="camera_3", alias="פרגולה"), reply("בסדר.")], "big")
        agent = self.agent(big, Scripted([], "fast"))
        first = agent.handle(REMEMBER, "-5", {"user_id": 1})
        out = agent.undo_turn("-5", first.undo_token, {"user_id": 1})
        self.assertEqual(out.text, '✓ "פרגולה" כבר לא camera 3')
        self.assertNotIn("camera_3", aliases.load_aliases(self.aliases))
        self.assertEqual(agent.undo_turn("-5", first.undo_token, {"user_id": 1}).text, t("nothing_to_undo", "he"))


if __name__ == "__main__":
    unittest.main()
