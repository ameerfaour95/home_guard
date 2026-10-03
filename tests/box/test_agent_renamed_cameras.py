"""2026-10-03: the setup re-ran discovery and renamed every camera in cameras.yaml.

The alert saved at 20:21:47 kept its old camera name ("main_entrance"), so the
assistant - which only knew the new names - searched ameer_test_ch2, ch3, ... one
by one, found nothing, and told the owner no such video existed.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

from test_agent import RecordingModel, call, say
from test_archive import make_alert

from home_guard_project.box.agent import AgentContext, OwnerAgent
from home_guard_project.box.feedback import MuteState

NOW = dt.datetime(2026, 10, 3, 21, 40).timestamp()
SAVED_AT = dt.datetime(2026, 10, 3, 20, 21, 47).timestamp()
WOMAN = "A woman holding a phone walks towards the entrance"
CURRENT = ["ameer_test_ch2", "ameer_test_ch3", "ameer_test_ch5", "ameer_test_ch6", "ameer_test_ch8"]


class RenamedCamerasTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for target in ("make_embedder", "make_look_now"):
            patcher = mock.patch(f"home_guard_project.box.agent.{target}", return_value=None)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.live = os.path.join(tmp.name, "production_multi")
        make_alert(self.live, "main_entrance", "main_entrance_1_alert", SAVED_AT, WOMAN, date="2026-10-03")
        make_alert(self.live, "front_side", "front_side_1_alert", NOW - 5 * 3600, "a cat on the path",
                   date="2026-10-03")
        # Older than the retention window: its camera name is not offered as an earlier name.
        make_alert(self.live, "old_garage", "old_garage_1_alert", NOW - 30 * 86400, "a van", date="2026-09-03")
        self.mute = MuteState(os.path.join(tmp.name, "alert_mute.json"))

    def _ctx(self) -> AgentContext:
        return AgentContext(camera_names=list(CURRENT), mute_state=self.mute, feedback_dir=self.live,
                            roots=lambda: [self.live], now=lambda: NOW)

    def _run(self, *responses: Any) -> tuple:
        model = RecordingModel(list(responses))
        reply = OwnerAgent(model, self._ctx()).handle("send me the video of the woman with the two children",
                                                      "-1001", {}, None)
        return model, reply

    @staticmethod
    def _tool_results(model: RecordingModel) -> List[Dict[str, Any]]:
        return [json.loads(m["content"]) for m in model.seen[-1] if m.get("role") == "tool"]

    def test_a_search_on_a_new_camera_name_still_finds_the_video_saved_under_the_old_one(self) -> None:
        model, reply = self._run(
            call("find_alerts", day="2026-10-03", time_from="20:20", time_to="20:22", camera="ameer_test_ch2",
                 what="woman with two children"),
            call("send_clip", alert_id="main_entrance_1_alert"),
            say("Here is the video."),
        )
        (found, sent) = self._tool_results(model)
        self.assertEqual(found["count"], 1)
        self.assertEqual(found["alerts"][0]["id"], "main_entrance_1_alert")
        self.assertEqual(found["alerts"][0]["camera"], "main_entrance")
        self.assertEqual(found["searched_camera"], "ameer_test_ch2")
        self.assertEqual(found["camera_note"], "nothing was saved on ameer_test_ch2 in that time; "
                                               "these were saved on other cameras")
        self.assertTrue(sent["ok"])
        self.assertEqual(len(reply.clips), 1)
        self.assertTrue(reply.clips[0].endswith("main_entrance_1_alert.mp4"))

    def test_a_camera_search_that_finds_something_has_no_note(self) -> None:
        model, _ = self._run(call("find_alerts", day="today", camera="front_side"), say("A cat."))
        (found,) = self._tool_results(model)
        self.assertEqual([a["id"] for a in found["alerts"]], ["front_side_1_alert"])
        self.assertNotIn("camera_note", found)

    def test_the_context_line_names_the_earlier_camera_names(self) -> None:
        model, _ = self._run(say("ok"))
        context = model.seen[0][-1]["content"]
        self.assertIn("Cameras: " + ", ".join(CURRENT), context)
        self.assertIn("Earlier camera names in saved alerts: front_side, main_entrance.", context)
        self.assertNotIn("old_garage", context)                   # outside the retention window

    def test_no_earlier_names_line_when_every_saved_camera_is_current(self) -> None:
        ctx = self._ctx()
        ctx.camera_names = CURRENT + ["main_entrance", "front_side"]
        model = RecordingModel([say("ok")])
        OwnerAgent(model, ctx).handle("hi", "-1001", {}, None)
        self.assertNotIn("Earlier camera names", model.seen[0][-1]["content"])

    def test_the_records_are_read_once_for_the_context_line_and_the_tools(self) -> None:
        from home_guard_project.box import agent as agent_mod

        with mock.patch.object(agent_mod, "load_records", wraps=agent_mod.load_records) as loads:
            self._run(call("find_alerts", day="today", camera="main_entrance"),
                      call("summarize_activity", day="today", camera="main_entrance"), say("ok"))
        self.assertEqual(loads.call_count, 1)

    def test_an_old_camera_name_from_the_saved_alerts_can_be_searched(self) -> None:
        model, _ = self._run(call("find_alerts", day="2026-10-03", camera="Main_Entrance"), say("ok"))
        (found,) = self._tool_results(model)
        self.assertNotIn("error", found)
        self.assertEqual(found["searched"]["camera"], "main_entrance")
        self.assertEqual([a["id"] for a in found["alerts"]], ["main_entrance_1_alert"])
        self.assertNotIn("camera_note", found)

    def test_a_name_neither_current_nor_saved_is_still_refused(self) -> None:
        model, _ = self._run(call("find_alerts", day="today", camera="garden"), say("which camera?"))
        (found,) = self._tool_results(model)
        self.assertIn("unknown camera", found["error"])
        self.assertIn("main_entrance", found["error"])           # the model is told the old names too

    def test_summarize_on_a_camera_with_nothing_falls_back_to_every_camera(self) -> None:
        model, _ = self._run(call("summarize_activity", day="today", camera="ameer_test_ch3"), say("ok"))
        (summary,) = self._tool_results(model)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["by_camera"], {"front_side": 1, "main_entrance": 1})
        self.assertEqual(summary["searched_camera"], "ameer_test_ch3")
        self.assertIn("nothing was saved on ameer_test_ch3", summary["camera_note"])

    def test_summarize_on_an_old_camera_name_is_accepted(self) -> None:
        model, _ = self._run(call("summarize_activity", day="today", camera="main_entrance"), say("ok"))
        (summary,) = self._tool_results(model)
        self.assertEqual((summary["total"], summary["by_camera"]), (1, {"main_entrance": 1}))
        self.assertNotIn("camera_note", summary)

    def test_pausing_an_old_camera_name_is_still_refused(self) -> None:
        model, _ = self._run(call("pause_alerts", owner_words="send me the video", camera="main_entrance",
                                  minutes=60), say("which camera?"))
        (paused,) = self._tool_results(model)
        self.assertIn("unknown camera", paused["error"])
        self.assertFalse(self.mute.is_muted(NOW + 60, "main_entrance"))


if __name__ == "__main__":
    unittest.main()
