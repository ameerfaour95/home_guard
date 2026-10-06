# tests/box/test_brain_stale_cameras.py
"""2026-10-06 on the box: the site and every camera were renamed (ameer_tes2_ch6 -> ameer_week_0_1_ch6), and an
alert from before the rename made the chat talk about ameer_tes2_ch6 - a camera that no longer exists - and
"save" the owner's name for it. Only cameras in the current registry may become the topic or get a name; an old
alert's camera is mapped through the rename aliases, or dropped."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import unittest
from typing import Any

import yaml

from test_brain_agent import Scripted, call, reply

from home_guard_project.box.brain import aliases
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import HouseRegistry, current_camera
from home_guard_project.box.brain.tools import Services

NOW = dt.datetime(2026, 10, 6, 23, 52).timestamp()
NEW = [f"ameer_week_0_1_ch{i}" for i in (1, 2, 6)]
OLD_ALERT = {"alert_id": "ameer_tes2_ch6_1791317682_alert", "camera": "ameer_tes2_ch6", "ts": NOW - 300,
             "summary": "A man walks out of the front door.", "label": "normal"}


class NoMute:
    def muted_until(self, now: float, camera: str = "") -> Any:
        return None


class StaleCameraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.cameras = os.path.join(self.root, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": {name: "rtsp://x" for name in NEW}}, f)
        self.aliases = os.path.join(self.root, "camera_aliases.yaml")

    def keep_old_names(self) -> None:
        aliases.add_alias("ameer_week_0_1_ch6", "ameer_tes2_ch6", NEW, self.aliases)    # what 5679f36 writes

    def agent(self, big: Scripted) -> OwnerAgentV2:
        services = Services(
            roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"), feedback_dir=self.root,
            work_dir=os.path.join(self.root, ".live"), mute=NoMute(), deliver=None,
            add_alias=lambda cam, alias, cams: aliases.add_alias(cam, alias, cams, self.aliases),
            read_settings=lambda: {"alert_start_hour": 22, "alert_end_hour": 6}, now=lambda: NOW)
        registry = HouseRegistry(NoMute(), lambda: (22, 6), self.cameras, self.aliases,
                                 os.path.join(self.root, "status.json"), now=lambda: NOW)
        return OwnerAgentV2(big, registry, ChatMemory(os.path.join(self.root, ".conversations")),
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW), services,
                            now=lambda: NOW)

    def test_current_camera_maps_old_names_or_drops_them(self) -> None:
        registry = HouseRegistry(NoMute(), lambda: (22, 6), self.cameras, self.aliases, "", now=lambda: NOW)
        self.assertIsNone(current_camera(registry.snapshot(), "ameer_tes2_ch6"))
        self.keep_old_names()
        snap = registry.snapshot()
        self.assertEqual(current_camera(snap, "ameer_tes2_ch6"), "ameer_week_0_1_ch6")
        self.assertEqual(current_camera(snap, "ameer_week_0_1_ch2"), "ameer_week_0_1_ch2")
        self.assertIsNone(current_camera(snap, ""))

    def test_an_old_alert_maps_to_the_renamed_camera(self) -> None:
        self.keep_old_names()
        big = Scripted([reply("ok")])
        self.agent(big).handle("מה זה היה?", "-5", {"user_id": 1}, alert=dict(OLD_ALERT), threaded=True)
        block = big.seen[0][0][-1]
        self.assertIn("[CAMERA BEING DISCUSSED] ameer_week_0_1_ch6", block)
        self.assertIn("ameer_tes2_ch6 (now ameer_week_0_1_ch6)", block)

    def test_an_old_alert_of_a_camera_that_is_gone_is_no_topic(self) -> None:
        big = Scripted([reply("ok")])
        agent = self.agent(big)
        agent.handle("מה זה היה?", "-5", {"user_id": 1}, alert=dict(OLD_ALERT), threaded=True)
        block = big.seen[0][0][-1]
        self.assertIn("[CAMERA BEING DISCUSSED] none", block)
        self.assertIn("ameer_tes2_ch6 (no longer a camera of this house)", block)
        self.assertIsNone(agent.memory.load("-5").topic_camera(NOW))
        agent.note_alert("-6", dict(OLD_ALERT))                                  # the alert's own history line
        self.assertIsNone(agent.memory.load("-6").topic_camera(NOW))

    def test_a_saved_topic_for_a_camera_that_is_gone_is_ignored(self) -> None:
        agent = self.agent(Scripted([reply("ok")]))
        state = agent.memory.load("-5")
        state.set_topic_camera("ameer_tes2_ch6", "כניסה ראשית", NOW - 60)
        agent.memory.save("-5", state)
        agent.handle("hi", "-5", {"user_id": 1})
        self.assertIsNone(agent.memory.load("-5").topic_camera(NOW))

    def test_naming_a_camera_that_is_gone_fails_with_no_receipt_and_no_saved_claim(self) -> None:
        big = Scripted([call("set_alias", camera="ameer_tes2_ch6", alias="כניסה ראשית"),
                        reply("שמרתי: כניסה ראשית זה ameer_tes2_ch6."), reply("שמרתי.")])
        out = self.agent(big).handle("תקרא לה כניסה ראשית", "-5", {"user_id": 1}, alert=dict(OLD_ALERT),
                                     threaded=True)
        self.assertEqual(out.receipts, ())
        self.assertEqual(out.text, t("not_saved_yet", "he"))
        self.assertFalse(os.path.exists(self.aliases))                        # nothing was written


if __name__ == "__main__":
    unittest.main()
