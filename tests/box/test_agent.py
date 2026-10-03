from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import types
import unittest
from typing import Any, List
from unittest import mock

from test_archive import make_alert

from home_guard_project.box.agent import (
    UNAVAILABLE_REPLY,
    AgentContext,
    ModelMessage,
    OwnerAgent,
    ToolCall,
    _OpenAIChat,
    _reply_language,
)
from home_guard_project.box.feedback import MuteState

NOW = dt.datetime(2026, 10, 2, 15, 0).timestamp()
HOUR = 3600.0
ALERT = {"alert_id": "front_door_3_alert", "camera": "front_door", "summary": "a person at the door", "ts": NOW - 60}


class ScriptedModel:
    """Plays back prepared model answers, so the tool loop runs without a network."""

    def __init__(self, responses: List[ModelMessage]) -> None:
        self._responses = list(responses)
        self._i = 0

    def chat(self, messages: Any, tools: Any, tool_choice: Any = None) -> ModelMessage:
        msg = self._responses[self._i]
        self._i += 1
        return msg


class BrokenModel(ScriptedModel):
    def chat(self, messages: Any, tools: Any, tool_choice: Any = None) -> ModelMessage:
        raise ConnectionError("no internet")


class DropsAfterModel(ScriptedModel):
    """Plays its scripted responses, then raises - a model/network drop partway through a turn."""

    def chat(self, messages: Any, tools: Any, tool_choice: Any = None) -> ModelMessage:
        if self._i < len(self._responses):
            msg = self._responses[self._i]
            self._i += 1
            return msg
        raise ConnectionError("dropped mid-turn")


class RecordingModel(ScriptedModel):
    """Remembers the messages it was handed, to check what context the agent built."""

    def __init__(self, responses: List[ModelMessage]) -> None:
        super().__init__(responses)
        self.seen: List[List[dict]] = []

    def chat(self, messages: Any, tools: Any, tool_choice: Any = None) -> ModelMessage:
        self.seen.append(list(messages))
        return super().chat(messages, tools, tool_choice)


def call(name: str, **args: Any) -> ModelMessage:
    return ModelMessage(tool_calls=(ToolCall(id=f"call_{name}", name=name, arguments=args),))


def say(text: str) -> ModelMessage:
    return ModelMessage(content=text)


class OwnerAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Never embed or build a live-view caller (so no network) in the unit tests, whatever the env holds.
        patcher = mock.patch("home_guard_project.box.agent.make_embedder", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        lv = mock.patch("home_guard_project.box.agent.make_look_now", return_value=None)
        lv.start()
        self.addCleanup(lv.stop)
        self.live = os.path.join(tmp.name, "production_multi")
        self.site = os.path.join(tmp.name, "production_archive", "house2")
        make_alert(self.site, "back_yard", "back_yard_2_alert", NOW - 3 * HOUR, "a car is parked in the yard")
        make_alert(self.live, "front_door", "front_door_3_alert", NOW - 60, "a person at the door")
        self.mute = MuteState(os.path.join(tmp.name, "alert_mute.json"))

    def _ctx(self) -> AgentContext:
        return AgentContext(
            camera_names=["front_door", "back_yard"],
            mute_state=self.mute,
            feedback_dir=self.live,
            roots=lambda: [self.live, self.site],
            now=lambda: NOW,
        )

    def _agent(self, responses: List[ModelMessage], model_cls: type = ScriptedModel) -> OwnerAgent:
        return OwnerAgent(model_cls(responses=responses), self._ctx())

    def _saved(self) -> List[dict]:
        found = []
        for dirpath, _, names in os.walk(os.path.join(self.live, "feedback")):
            for name in sorted(names):
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    found.append(json.load(f))
        return found

    def test_a_false_alarm_is_recorded_with_the_owners_own_words(self) -> None:
        agent = self._agent([call("record_verdict", verdict="false_alarm", note="nothing was there"),
                             say("Thanks, I marked it as a false alarm.")])
        reply = agent.handle("no there was nothing", "-1001", {"user_id": 42, "name": "Dana"}, ALERT)

        self.assertEqual(reply.text, "Thanks, I marked it as a false alarm.")
        (saved,) = self._saved()
        self.assertEqual((saved["verdict"], saved["raw_text"]), ("false_alarm", "no there was nothing"))
        self.assertEqual(saved["alert"]["alert_id"], "front_door_3_alert")
        self.assertEqual(saved["from"], {"user_id": 42, "name": "Dana"})

    def test_pause_until_a_time_then_continue(self) -> None:
        agent = self._agent([
            call("pause_alerts", owner_words="stop until six", until="18:00"), say("Paused until 18:00."),
            call("resume_alerts"), say("Alerts are back on."),
        ])
        agent.handle("it's me in the garden, stop until six", "-1001", {}, ALERT)
        self.assertTrue(self.mute.is_muted(NOW + HOUR, "front_door"))
        self.assertFalse(self.mute.is_muted(NOW + 4 * HOUR, "front_door"))

        agent.handle("continue", "-1001", {}, None)
        self.assertFalse(self.mute.is_muted(NOW + HOUR, "front_door"))
        self.assertEqual([s["action"] for s in self._saved()], ["mute", "resume"])

    def test_a_pause_the_model_asks_for_cannot_exceed_the_cap(self) -> None:
        agent = self._agent([call("pause_alerts", owner_words="Stop  FOREVER", minutes=999999), say("ok")])
        agent.handle("stop forever", "-1001", {}, None)
        self.assertTrue(self.mute.is_muted(NOW + HOUR, "front_door"))
        self.assertFalse(self.mute.is_muted(NOW + 25 * HOUR, "front_door"))

    def test_no_pause_unless_the_owner_asked_for_it_in_this_message(self) -> None:
        agent = self._agent([
            call("record_verdict", verdict="false_alarm"),
            call("pause_alerts", owner_words="please pause the alerts", minutes=60),   # words the owner never wrote
            say("Marked as a false alarm."),
        ])
        agent.handle("no there was nothing", "-1001", {}, ALERT)
        self.assertFalse(self.mute.is_muted(NOW + 60, "front_door"))
        self.assertEqual([s["action"] for s in self._saved()], ["none"])

    def test_the_owner_asks_for_a_video_and_gets_the_clip(self) -> None:
        agent = self._agent([
            call("find_alerts", last_hours=6, what="car"),
            call("send_clip", alert_id="back_yard_2_alert"),
            say("A car was parked in the yard at noon. Here is the video."),
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
        agent = self._agent([call("send_clip", alert_id="made_up_alert"), say("I have no such video.")])
        self.assertEqual(agent.handle("send the video", "-1001", {}, None).clips, ())

    def test_a_made_up_verdict_is_not_recorded_as_one(self) -> None:
        agent = self._agent([call("record_verdict", verdict="delete everything"), say("ok")])
        agent.handle("delete everything", "-1001", {}, ALERT)
        (saved,) = self._saved()
        self.assertEqual(saved["verdict"], "none")

    def test_a_message_that_needs_no_tool_is_saved_anyway(self) -> None:
        agent = self._agent([say("You're welcome.")])
        agent.handle("thanks", "-1001", {}, ALERT)
        (saved,) = self._saved()
        self.assertEqual((saved["verdict"], saved["action"], saved["raw_text"]), ("none", "none", "thanks"))

    def test_the_conversation_history_is_carried_into_later_messages(self) -> None:
        model = RecordingModel([say("Hello."), say("It was quiet.")])
        agent = OwnerAgent(model, self._ctx())
        agent.handle("hello there", "-1001", {}, None)
        agent.handle("anything happen today?", "-1001", {}, None)
        second_context = [m.get("content") for m in model.seen[1]]
        self.assertIn("hello there", second_context)   # the owner's earlier message
        self.assertIn("Hello.", second_context)        # and the assistant's earlier reply

    def test_summarize_activity_gathers_the_periods_events_for_the_model(self) -> None:
        model = RecordingModel([call("summarize_activity", day="today"), say("A car and a person today.")])
        agent = OwnerAgent(model, self._ctx())
        reply = agent.handle("what happened today?", "-1001", {}, None)
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        result = json.loads(tool_msgs[0]["content"])
        self.assertEqual(result["total"], 2)                       # both saved events, not capped like find_alerts
        summaries = " ".join(e["summary"] for e in result["events"])
        self.assertIn("car", summaries)
        self.assertIn("person", summaries)
        self.assertEqual(reply.text, "A car and a person today.")

    def test_history_is_reloaded_by_a_fresh_agent_after_a_restart(self) -> None:
        OwnerAgent(ScriptedModel([say("noted")]), self._ctx()).handle("remember this", "-1001", {}, None)
        model = RecordingModel([say("ok")])
        OwnerAgent(model, self._ctx()).handle("and now this", "-1001", {}, None)  # a new process = a restart
        first_context = [m.get("content") for m in model.seen[0]]
        self.assertIn("remember this", first_context)
        self.assertIn("noted", first_context)

    def test_without_the_model_the_message_is_saved_and_the_owner_is_told(self) -> None:
        agent = self._agent([], model_cls=BrokenModel)
        reply = agent.handle("no there was nothing", "-1001", {}, ALERT)
        self.assertEqual(reply.text, UNAVAILABLE_REPLY)
        (saved,) = self._saved()
        self.assertEqual(saved["raw_text"], "no there was nothing")

    def test_a_single_word_quote_does_not_authorize_a_pause(self) -> None:
        agent = self._agent([call("pause_alerts", owner_words="nothing", minutes=60), say("ok")])
        agent.handle("no there was nothing", "-1001", {}, ALERT)
        self.assertFalse(self.mute.is_muted(NOW + 60, "front_door"))   # one word is not enough to pause

    def test_pause_with_an_unknown_camera_is_refused_and_pauses_nothing(self) -> None:
        agent = self._agent([call("pause_alerts", owner_words="quiet on garden", camera="garden", minutes=60),
                             say("which camera?")])
        agent.handle("be quiet on garden cam", "-1001", {}, ALERT)
        self.assertFalse(self.mute.is_muted(NOW + 60, "front_door"))   # NOT silently widened to all cameras
        self.assertFalse(self.mute.is_muted(NOW + 60, "back_yard"))

    def test_mute_this_camera_never_pauses_the_whole_house(self) -> None:
        """2026-10-03: "mute this camera" (about back_door) paused every camera for a day."""
        model = RecordingModel([call("pause_alerts", owner_words="תשתיק את המצלמה הזאת"),
                                call("pause_alerts", owner_words="תשתיק את המצלמה הזאת", camera="back_yard"),
                                say("Paused back_yard.")])
        OwnerAgent(model, self._ctx()).handle("תשתיק את המצלמה הזאת", "-1001", {}, None)
        refused = [m for m in model.seen[1] if m.get("role") == "tool"][0]["content"]
        self.assertIn("ONE camera", refused)
        self.assertFalse(self.mute.is_muted(NOW + 60, "front_door"))      # the rest of the house still alerts
        self.assertTrue(self.mute.is_muted(NOW + 60, "back_yard"))

    def test_asking_for_all_cameras_still_pauses_all(self) -> None:
        from home_guard_project.box.agent import asks_for_one_camera

        for words in ("mute this camera", "תשתיק את המצלמה הזאת", "תכבה את המצלמה", "اسكت هذه الكاميرا"):
            self.assertTrue(asks_for_one_camera(words), words)
        for words in ("stop alerts for an hour", "mute all cameras", "תשתיק את כל המצלמות", "תשתיק הכל לשעה",
                      "اسكت كل الكاميرات"):
            self.assertFalse(asks_for_one_camera(words), words)
        agent = self._agent([call("pause_alerts", owner_words="stop alerts for an hour", minutes=60), say("ok")])
        agent.handle("stop alerts for an hour", "-1001", {}, None)
        self.assertTrue(self.mute.is_muted(NOW + 60, "front_door") and self.mute.is_muted(NOW + 60, "back_yard"))

    def test_turning_a_camera_off_restarts_only_after_the_reply(self) -> None:
        calls = []
        ctx = self._ctx()
        ctx.set_camera = lambda camera, active: calls.append((camera, active)) or {"ok": True}
        reply = OwnerAgent(ScriptedModel([call("set_camera_active", camera="back_yard", active=False),
                                          say("back_yard is off.")]), ctx).handle("turn off back yard", "-1001", {}, None)
        self.assertEqual(calls, [("back_yard", False)])
        self.assertTrue(reply.restart)

    def test_find_with_an_unknown_camera_is_refused(self) -> None:
        model = RecordingModel([call("find_alerts", camera="garden", what="car"), say("which camera?")])
        OwnerAgent(model, self._ctx()).handle("the car on garden", "-1001", {}, None)
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertIn("unknown camera", tool_msgs[0]["content"])

    def test_a_mid_turn_failure_after_a_tool_acted_reports_what_happened(self) -> None:
        agent = OwnerAgent(DropsAfterModel([call("pause_alerts", owner_words="stop until six", until="18:00")]),
                           self._ctx())
        reply = agent.handle("stop until six please", "-1001", {}, ALERT)
        self.assertNotEqual(reply.text, UNAVAILABLE_REPLY)             # reports the pause, not "unavailable"
        self.assertIn("paused", reply.text.lower())
        self.assertTrue(self.mute.is_muted(NOW + HOUR, "front_door"))  # and the pause really applied

    def test_handle_does_not_raise_when_saving_the_message_fails(self) -> None:
        agent = self._agent([say("you're welcome")])
        with mock.patch.object(agent, "_save", side_effect=OSError("disk full")):
            reply = agent.handle("thanks", "-1001", {}, ALERT)        # must not raise into the caller
        self.assertEqual(reply.text, "you're welcome")

    def test_check_camera_takes_a_live_look_and_attaches_the_photo(self) -> None:
        seen = []
        ctx = self._ctx()
        ctx.look_now = lambda cam: (seen.append(cam) or
                                    {"camera": cam, "description": "A person is at the front door.",
                                     "image": "/tmp/front_door.jpg"})
        model = RecordingModel([call("check_camera", camera="front_door"), say("Someone's at the front door.")])
        reply = OwnerAgent(model, ctx).handle("who's at the door now?", "-1001", {}, None)
        self.assertEqual(seen, ["front_door"])
        self.assertEqual(reply.photos, ("/tmp/front_door.jpg",))
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertIn("A person is at the front door", tool_msgs[0]["content"])
        self.assertEqual(reply.text, "Someone's at the front door.")

    def test_check_camera_refuses_an_unknown_camera_without_looking(self) -> None:
        seen = []
        ctx = self._ctx()
        ctx.look_now = lambda cam: (seen.append(cam) or {"camera": cam, "description": "x", "image": "/tmp/x.jpg"})
        model = RecordingModel([call("check_camera", camera="garage"), say("which camera?")])
        OwnerAgent(model, ctx).handle("check the garage", "-1001", {}, None)
        self.assertEqual(seen, [])                                   # never attempted the live look
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertIn("unknown camera", tool_msgs[0]["content"])

    def test_check_camera_reports_a_failure_without_a_photo(self) -> None:
        ctx = self._ctx()
        ctx.look_now = lambda cam: {"error": "could not get a picture from front_door right now"}
        model = RecordingModel([call("check_camera", camera="front_door"), say("I couldn't get a look just now.")])
        reply = OwnerAgent(model, ctx).handle("check the front door", "-1001", {}, None)
        self.assertEqual(reply.photos, ())
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertIn("could not get a picture", tool_msgs[0]["content"])

    def test_set_camera_active_turns_a_camera_off(self) -> None:
        calls = []
        ctx = self._ctx()
        ctx.set_camera = lambda cam, active: (calls.append((cam, active)) or {"ok": True})
        model = RecordingModel([call("set_camera_active", camera="front_door", active=False),
                                say("Done — the front camera is off.")])
        OwnerAgent(model, ctx).handle("disable the front camera", "-1001", {}, None)
        self.assertEqual(calls, [("front_door", False)])
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertIn("turned off", tool_msgs[0]["content"])

    def test_set_camera_active_relays_an_error(self) -> None:
        ctx = self._ctx()
        ctx.set_camera = lambda cam, active: {"error": "unknown camera 'garage'"}
        model = RecordingModel([call("set_camera_active", camera="garage", active=False),
                                say("I couldn't find that camera.")])
        OwnerAgent(model, ctx).handle("turn off the garage camera", "-1001", {}, None)
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertIn("unknown camera", tool_msgs[0]["content"])

    def test_invalid_json_arguments_return_an_error_without_running_the_tool(self) -> None:
        bad = ModelMessage(tool_calls=(ToolCall(id="c1", name="record_verdict", arguments={},
                                                raw_arguments="{bad json", valid=False),))
        model = RecordingModel([bad, say("could you rephrase?")])
        OwnerAgent(model, self._ctx()).handle("mark it real", "-1001", {}, ALERT)
        tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
        self.assertIn("not valid JSON", tool_msgs[0]["content"])
        (saved,) = self._saved()
        self.assertEqual(saved["verdict"], "none")                    # the verdict tool never ran


class OpenAIChatWrapperTest(unittest.TestCase):
    """_OpenAIChat is the only part that talks to the SDK; the scripted-model tests bypass it."""

    @staticmethod
    def _tc(name: str, arguments: str) -> Any:
        return types.SimpleNamespace(id="c1", type="function",
                                     function=types.SimpleNamespace(name=name, arguments=arguments))

    def _client(self, message: Any, finish_reason: str = "stop") -> Any:
        captured: dict = {}

        def create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason=finish_reason, message=message)])

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        client._captured = captured
        return client

    def test_valid_tool_call_parses_with_raw_args(self) -> None:
        msg = types.SimpleNamespace(content=None, tool_calls=[self._tc("record_verdict", '{"verdict":"false_alarm"}')])
        out = _OpenAIChat(self._client(msg), "gpt-4o-mini").chat([], [{"type": "function"}])
        call = out.tool_calls[0]
        self.assertEqual((call.arguments, call.valid, call.raw_arguments),
                         ({"verdict": "false_alarm"}, True, '{"verdict":"false_alarm"}'))

    def test_invalid_json_is_flagged_not_run(self) -> None:
        msg = types.SimpleNamespace(content=None, tool_calls=[self._tc("record_verdict", "{bad")])
        call = _OpenAIChat(self._client(msg), "m").chat([], [{"x": 1}]).tool_calls[0]
        self.assertFalse(call.valid)
        self.assertEqual(call.raw_arguments, "{bad")

    def test_truncated_response_marks_args_invalid(self) -> None:
        msg = types.SimpleNamespace(content=None, tool_calls=[self._tc("find_alerts", '{"what":"c')])
        call = _OpenAIChat(self._client(msg, finish_reason="length"), "m").chat([], [{"x": 1}]).tool_calls[0]
        self.assertFalse(call.valid)

    def test_tool_choice_is_passed_through(self) -> None:
        client = self._client(types.SimpleNamespace(content="hi", tool_calls=[]))
        _OpenAIChat(client, "m").chat([], [{"type": "function"}], tool_choice="none")
        self.assertEqual(client._captured.get("tool_choice"), "none")

    def test_no_tools_sends_no_tool_params(self) -> None:
        client = self._client(types.SimpleNamespace(content="hi", tool_calls=[]))
        _OpenAIChat(client, "m").chat([], [])
        self.assertNotIn("tools", client._captured)
        self.assertNotIn("tool_choice", client._captured)


if __name__ == "__main__":
    unittest.main()
