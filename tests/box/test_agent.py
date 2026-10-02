from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
from typing import Any, List

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from test_archive import make_alert

from home_guard_project.box.agent import UNAVAILABLE_REPLY, AgentContext, OwnerAgent, _reply_language
from home_guard_project.box.feedback import MuteState

NOW = dt.datetime(2026, 10, 2, 15, 0).timestamp()
HOUR = 3600.0
ALERT = {"alert_id": "front_door_3_alert", "camera": "front_door", "summary": "a person at the door", "ts": NOW - 60}


class ScriptedModel(FakeMessagesListChatModel):
    """Plays back prepared model answers, so the tool loop runs without a network."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


class BrokenModel(ScriptedModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("no internet")


def call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


class OwnerAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.live = os.path.join(tmp.name, "production_multi")
        self.site = os.path.join(tmp.name, "production_archive", "house2")
        make_alert(self.site, "back_yard", "back_yard_2_alert", NOW - 3 * HOUR, "a car is parked in the yard")
        make_alert(self.live, "front_door", "front_door_3_alert", NOW - 60, "a person at the door")
        self.mute = MuteState(os.path.join(tmp.name, "alert_mute.json"))

    def _agent(self, responses: List[AIMessage], model_cls: type = ScriptedModel) -> OwnerAgent:
        ctx = AgentContext(
            camera_names=["front_door", "back_yard"],
            mute_state=self.mute,
            feedback_dir=self.live,
            roots=lambda: [self.live, self.site],
            now=lambda: NOW,
        )
        return OwnerAgent(model_cls(responses=responses), ctx)

    def _saved(self) -> List[dict]:
        found = []
        for dirpath, _, names in os.walk(os.path.join(self.live, "feedback")):
            for name in sorted(names):
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    found.append(json.load(f))
        return found

    def test_a_false_alarm_is_recorded_with_the_owners_own_words(self) -> None:
        agent = self._agent([call("record_verdict", verdict="false_alarm", note="nothing was there"),
                             AIMessage(content="Thanks, I marked it as a false alarm.")])
        reply = agent.handle("no there was nothing", "-1001", {"user_id": 42, "name": "Dana"}, ALERT)

        self.assertEqual(reply.text, "Thanks, I marked it as a false alarm.")
        (saved,) = self._saved()
        self.assertEqual((saved["verdict"], saved["raw_text"]), ("false_alarm", "no there was nothing"))
        self.assertEqual(saved["alert"]["alert_id"], "front_door_3_alert")
        self.assertEqual(saved["from"], {"user_id": 42, "name": "Dana"})

    def test_pause_until_a_time_then_continue(self) -> None:
        agent = self._agent([
            call("pause_alerts", owner_words="stop until six", until="18:00"), AIMessage(content="Paused until 18:00."),
            call("resume_alerts"), AIMessage(content="Alerts are back on."),
        ])
        agent.handle("it's me in the garden, stop until six", "-1001", {}, ALERT)
        self.assertTrue(self.mute.is_muted(NOW + HOUR, "front_door"))
        self.assertFalse(self.mute.is_muted(NOW + 4 * HOUR, "front_door"))

        agent.handle("continue", "-1001", {}, None)
        self.assertFalse(self.mute.is_muted(NOW + HOUR, "front_door"))
        self.assertEqual([s["action"] for s in self._saved()], ["mute", "resume"])

    def test_a_pause_the_model_asks_for_cannot_exceed_the_cap(self) -> None:
        agent = self._agent([call("pause_alerts", owner_words="Stop  FOREVER", minutes=999999), AIMessage(content="ok")])
        agent.handle("stop forever", "-1001", {}, None)
        self.assertTrue(self.mute.is_muted(NOW + HOUR, "front_door"))
        self.assertFalse(self.mute.is_muted(NOW + 25 * HOUR, "front_door"))

    def test_no_pause_unless_the_owner_asked_for_it_in_this_message(self) -> None:
        agent = self._agent([
            call("record_verdict", verdict="false_alarm"),
            call("pause_alerts", owner_words="please pause the alerts", minutes=60),   # words the owner never wrote
            AIMessage(content="Marked as a false alarm."),
        ])
        agent.handle("no there was nothing", "-1001", {}, ALERT)
        self.assertFalse(self.mute.is_muted(NOW + 60, "front_door"))
        self.assertEqual([s["action"] for s in self._saved()], ["none"])

    def test_the_owner_asks_for_a_video_and_gets_the_clip(self) -> None:
        agent = self._agent([
            call("find_alerts", last_hours=6, what="car"),
            call("send_clip", alert_id="back_yard_2_alert"),
            AIMessage(content="A car was parked in the yard at noon. Here is the video."),
        ])
        reply = agent.handle("what was that car today? send me the video", "-1001", {}, None)

        self.assertEqual(len(reply.clips), 1)
        self.assertTrue(reply.clips[0].endswith("back_yard_2_alert.mp4"))
        self.assertTrue(os.path.isfile(reply.clips[0]))

    def test_the_model_is_told_which_language_to_answer_in(self) -> None:
        self.assertEqual(_reply_language("זה אני בגינה, תפסיק עד 22:00"), "Hebrew")
        self.assertEqual(_reply_language("ابعتلي الفيديو تبع السيارة"), "Arabic")
        self.assertIn("English", _reply_language("no there was nothing"))
        self.assertIn("English", _reply_language("ok 👍"))

    def test_a_video_that_was_never_saved_is_not_sent(self) -> None:
        agent = self._agent([call("send_clip", alert_id="made_up_alert"), AIMessage(content="I have no such video.")])
        self.assertEqual(agent.handle("send the video", "-1001", {}, None).clips, ())

    def test_a_made_up_verdict_is_not_recorded_as_one(self) -> None:
        agent = self._agent([call("record_verdict", verdict="delete everything"), AIMessage(content="ok")])
        agent.handle("delete everything", "-1001", {}, ALERT)
        (saved,) = self._saved()
        self.assertEqual(saved["verdict"], "none")

    def test_a_message_that_needs_no_tool_is_saved_anyway(self) -> None:
        agent = self._agent([AIMessage(content="You're welcome.")])
        agent.handle("thanks", "-1001", {}, ALERT)
        (saved,) = self._saved()
        self.assertEqual((saved["verdict"], saved["action"], saved["raw_text"]), ("none", "none", "thanks"))

    def test_without_the_model_the_message_is_saved_and_the_owner_is_told(self) -> None:
        agent = self._agent([], model_cls=BrokenModel)
        reply = agent.handle("no there was nothing", "-1001", {}, ALERT)
        self.assertEqual(reply.text, UNAVAILABLE_REPLY)
        (saved,) = self._saved()
        self.assertEqual(saved["raw_text"], "no there was nothing")


if __name__ == "__main__":
    unittest.main()
