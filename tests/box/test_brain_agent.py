# tests/box/test_brain_agent.py
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, List
from unittest.mock import Mock, patch

from test_brain_events import meta
from test_brain_tools_read import snapshot

from home_guard_project.box.brain.agent import OwnerAgentV2, build_owner_agent, context_block, follow_up_camera_receipts
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.models import ModelMessage, ToolCall
from home_guard_project.box.brain.receipts import DONE, FAILED, REQUESTED, Receipt, ReceiptBook
from home_guard_project.box.brain.tools import Services

NOW = dt.datetime(2026, 10, 3, 23, 0).timestamp()


def call(name: str, **args: Any) -> ModelMessage:
    return ModelMessage(tool_calls=(ToolCall(id=f"c_{name}", name=name, arguments=args),), usage=(10, 2))


def reply(answer: str) -> ModelMessage:
    return call("reply", answer=answer)


class Scripted:
    def __init__(self, responses: List[ModelMessage], name: str = "big") -> None:
        self.responses, self.model_name, self.seen = list(responses), name, []

    def chat(self, messages, tools, tool_choice=None):
        self.seen.append(([m.get("content") for m in messages], [t["function"]["name"] for t in tools]))
        if not self.responses:
            raise ConnectionError("script ended")
        return self.responses.pop(0)


class FakeRegistry:
    def __init__(self, mode="guard"):
        self.mode = mode

    def snapshot(self):
        return snapshot(self.mode)


class AgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        meta(self.root, "main_entrance", "main_entrance_1_alert", NOW - 3600,
             summary="A man stands at the door.", label="suspicious", people=1)
        self.calls = []

    def agent(self, big, fast=None, mode="guard", run_tool=None) -> OwnerAgentV2:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=None, read_settings=lambda: {"alert_start_hour": 22, "alert_end_hour": 6},
                            now=lambda: NOW)
        return OwnerAgentV2(big, FakeRegistry(mode), ChatMemory(os.path.join(self.root, ".conversations")),
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW), services,
                            fast_model=fast, run_tool=run_tool, now=lambda: NOW)

    def fake_send(self, ctx, name, args):
        """A stand-in tool runner: send_media succeeds with a receipt; everything else runs for real."""
        self.calls.append(name)
        if name == "send_media":
            from home_guard_project.box.brain.tools import _issue, _result  # noqa: PLC0415

            return _result(_issue(ctx, "send_media", DONE, args.get("handle", ""), {"kind": "video", "bounds": "b"}))
        from home_guard_project.box.brain.tools import TOOLS  # noqa: PLC0415

        return TOOLS[name](ctx, args)

    def test_reply_tool_ends_the_turn_and_receipts_are_rendered(self) -> None:
        big = Scripted([call("find_events", last_hours=24), call("send_media", handle="E1"), reply("Here is 01:22.")])
        out = self.agent(big, run_tool=self.fake_send).handle("send the last video", "-5", {"user_id": 1})
        self.assertEqual(out.text, "Here is 01:22.\n✓ Video sent (b)")
        self.assertEqual(out.tools_called, ("find_events", "send_media"))
        self.assertEqual(self.calls.count("send_media"), 1)

    def test_a_false_claim_gets_one_rewrite_then_falls_back(self) -> None:
        big = Scripted([reply("I sent you the video."), reply("Here is the video.")])
        out = self.agent(big).handle("send the video", "-5", {"user_id": 1})
        self.assertEqual(out.guard_hits, 2)
        self.assertEqual(out.text, "I did not do anything yet - please tell me again what you need.")
        good = Scripted([reply("I sent you the video."), reply("There was one event at 22:00.")])
        self.assertEqual(self.agent(good).handle("send the video", "-5", {"user_id": 1}).text,
                         "There was one event at 22:00.")

    def test_clarification_sends_buttons_and_the_answer_comes_back(self) -> None:
        big = Scripted([call("ask_clarification", question="Which camera?", choices=["main_entrance", "front_side"])])
        agent = self.agent(big)
        first = agent.handle("turn off the camera", "-5", {"user_id": 1})
        self.assertEqual((first.text, first.buttons), ("Which camera?", ("main_entrance", "front_side")))
        self.assertTrue(first.question_token)
        self.assertIsNone(agent.handle_choice("-5", "stale", 1, {"user_id": 1}))   # an old button
        big.responses = [reply("ok")]
        agent.handle_choice("-5", first.question_token, 1, {"user_id": 1})
        last_user = big.seen[-1][0][-1]
        self.assertIn('You asked: "Which camera?"', last_user)
        self.assertIn('answered: "front_side"', last_user)

    def test_nothing_runs_after_a_question_in_the_same_response(self) -> None:
        msg = ModelMessage(tool_calls=(
            ToolCall(id="q", name="ask_clarification", arguments={"question": "Which?", "choices": ["a", "b"]}),
            ToolCall(id="p", name="send_media", arguments={"handle": "E1"})))
        out = self.agent(Scripted([msg]), run_tool=self.fake_send).handle("send it", "-5", {"user_id": 1})
        self.assertEqual((out.text, self.calls), ("Which?", ["ask_clarification"]))

    def test_wrong_mode_tool_is_refused(self) -> None:
        big = Scripted([call("describe_event", handle="E1"), reply("Done.")])
        self.agent(big, mode="guard").handle("describe it", "-5", {"user_id": 1})
        self.assertNotIn("describe_event", big.seen[0][1])

    def test_the_same_acting_call_runs_once_per_turn(self) -> None:
        big = Scripted([call("find_events", last_hours=24), call("send_media", handle="E1"),
                        call("send_media", handle="e1", owner_words="again please"), reply("Sent twice?")])
        out = self.agent(big, run_tool=self.fake_send).handle("send it", "-5", {"user_id": 1})
        self.assertEqual(self.calls.count("send_media"), 1)
        self.assertEqual(len(out.receipts), 1)

    def test_model_failure_answers_in_the_owners_language_and_saves_the_message(self) -> None:
        out = self.agent(Scripted([])).handle("מה קורה בכניסה", "-5", {"user_id": 1})
        self.assertEqual(out.text, "לא הצלחתי לטפל בזה כרגע, אבל ההודעה שלך נשמרה.")
        self.assertTrue(glob.glob(os.path.join(self.root, "feedback", "**", "*.feedback.json"), recursive=True))

    def test_plain_text_without_reply_is_accepted(self) -> None:
        out = self.agent(Scripted([ModelMessage(content="Two events today.")])).handle("anything?", "-5", {})
        self.assertEqual(out.text, "Two events today.")

    def test_fast_model_answers_simple_turns_and_hands_off_hard_ones(self) -> None:
        fast = Scripted([reply("Hello.")], "fast")
        big = Scripted([], "big")
        out = self.agent(big, fast=fast).handle("hi", "-5", {})
        self.assertEqual((out.text, out.tier, out.escalated), ("Hello.", "fast", False))
        self.assertIn("hand_off", fast.seen[0][1])
        fast = Scripted([call("hand_off", reason="judgement")], "fast")
        big = Scripted([reply("🟡 suspicious: a man at the door.")], "big")
        out = self.agent(big, fast=fast).handle("tell me more about that guy", "-5", {})
        self.assertEqual((out.tier, out.escalated), ("big", True))
        self.assertNotIn("hand_off", big.seen[0][1])

    def test_a_false_claim_from_the_fast_model_goes_to_the_big_model(self) -> None:
        fast = Scripted([reply("I sent the video.")], "fast")
        big = Scripted([reply("Nothing was recorded at the gate today.")], "big")
        out = self.agent(big, fast=fast).handle("anything at the gate today?", "-5", {})
        self.assertEqual((out.tier, out.text), ("big", "Nothing was recorded at the gate today."))

    def test_state_changes_replies_to_alerts_and_answers_to_questions_skip_the_fast_model(self) -> None:
        fast = Scripted([reply("should not run")], "fast")
        big = Scripted([reply("Which camera?"), reply("ok"), reply("ok")], "big")
        agent = self.agent(big, fast=fast)
        self.assertEqual(agent.handle("turn off the front camera", "-5", {}).tier, "big")
        alert = {"alert_id": "main_entrance_1_alert", "camera": "main_entrance", "ts": NOW - 60, "summary": "x"}
        self.assertEqual(agent.handle("ok", "-5", {}, alert=alert, threaded=True).tier, "big")
        self.assertEqual(fast.seen, [])

    def test_context_block(self) -> None:
        block = context_block(snapshot("guard"), "SETTINGS: x", NOW, "he", "E3",
                              {"camera": "main_entrance", "ts": NOW - 60, "summary": "a man"}, None, "hi")
        self.assertIn("main_entrance  aka: entrance, front door  live  alerts on", block)
        self.assertIn("SETTINGS: x", block)
        self.assertIn("[ALERT THIS MESSAGE ANSWERS] E3 main_entrance", block)
        self.assertIn("[ANSWER IN] Hebrew", block)
        self.assertIn("[BOX LANGUAGE] English", block)
        self.assertTrue(block.endswith("[MESSAGE]\nhi"))

    def test_fast_error_escalates_even_with_content(self) -> None:
        fast = Scripted([ModelMessage(content="Must not use this.", error="offline")])
        out = self.agent(Scripted([reply("Hello.")]), fast).handle("hi", "-5")
        self.assertEqual((out.text, out.tier, out.escalated), ("Hello.", "big", True))

    def test_big_error_is_unavailable_in_the_persons_language(self) -> None:
        for rounds in (0, 5):
            with self.subTest(rounds=rounds):
                agent = self.agent(Scripted([ModelMessage(content="Ignore this.", error="offline")]))
                agent.max_rounds = rounds
                out = agent.handle("מה קורה", "-5", {"user_id": 1})
                self.assertEqual(out.text, t("unavailable", "he"))
                self.assertEqual(agent.memory.load("-5").turns[-1]["text"], "מה קורה")

    def test_big_failure_keeps_receipts_and_says_unavailable(self) -> None:
        big = Scripted([call("send_media", handle="E1"), ModelMessage(error="offline")])
        out = self.agent(big, run_tool=self.fake_send).handle("send the video", "-5")
        self.assertEqual(out.text, t("unavailable", "en") + "\n✓ Video sent (b)")
        self.assertEqual(len(out.receipts), 1)

    def test_bad_metadata_entry_preserves_speaker_language_and_alert_identity(self) -> None:
        agent = self.agent(Scripted([reply("Hello.")]))
        agent.services.read_settings = lambda: {"owner_language": "he", "unused": float("nan")}
        agent.handle("8", "-5", {"user_id": 1, "extra": object()},
                     {"alert_id": "original", "camera": "main_entrance", "ts": float("inf")})
        state = agent.memory.load("-5")
        self.assertEqual(state.turns[-1]["speaker"], "1")
        self.assertEqual(state.handles["E1"]["ref"], "original")
        self.assertIn("[ANSWER IN] Hebrew", agent.model.seen[0][0][-1])

    def test_final_call_refusal_and_usage_are_accounted_for(self) -> None:
        agent = self.agent(Scripted([ModelMessage(content="No.", refused=True, usage=(4, 2))]))
        agent.max_rounds = 0
        out = agent.handle("hi", "-5")
        self.assertEqual(out.text, t("unavailable", "en"))
        self.assertEqual(out.usage, {"big": (4, 2)})

    def test_malformed_turn_inputs_and_settings_are_contained(self) -> None:
        for settings, who, alert in (([], [], []), ({}, {"extra": object()}, None),
                                     ({}, {}, {"alert_id": "a", "ts": "bad"}),
                                     ({}, {}, {"alert_id": "a", "ts": float("nan")}),
                                     ({}, {}, {"alert_id": "a", "ts": float("inf")})):
            with self.subTest(settings=settings, who=who, alert=alert):
                agent = self.agent(Scripted([reply("Hello.")]))
                agent.services.read_settings = lambda: settings
                out = agent.handle("hi", "-5", who, alert)
                self.assertTrue(out.text)
                self.assertEqual(agent.memory.load("-5").turns[-1]["text"], "hi")

    def test_malformed_saved_question_and_invalid_utf8_are_contained(self) -> None:
        agent = self.agent(Scripted([reply("Hello."), reply("Hello.")]))
        agent.handle("hi", "-5")
        path = glob.glob(os.path.join(self.root, ".conversations", "*.json"))[0]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "pending": {"question": "Which?", "choices": [None, {}]}}, f)
        self.assertTrue(agent.handle("hi", "-5").text)
        with open(path, "wb") as f:
            f.write(b"\xff")
        self.assertIsNone(agent.handle_choice("-5", "old", 0))

    def test_bad_choice_indexes_and_choices_do_not_consume_the_question(self) -> None:
        agent = self.agent(Scripted([call("ask_clarification", question="Which?", choices=["a", "b"])]))
        first = agent.handle("hi", "-5")
        for index in ("1", None, [], float("nan"), float("inf"), True):
            with self.subTest(index=index):
                self.assertIsNone(agent.handle_choice("-5", first.question_token, index))
        state = agent.memory.load("-5")
        state.pending["choices"] = [None]
        agent.memory.save("-5", state)
        self.assertIsNone(agent.handle_choice("-5", first.question_token, 0))

    def test_failing_memory_does_not_escape_handle(self) -> None:
        agent = self.agent(Scripted([]))
        agent.memory.load = Mock(side_effect=ValueError("damaged"))
        self.assertEqual(agent.handle("מה קורה", "-5").text, t("unavailable", "he"))
        self.assertTrue(glob.glob(os.path.join(self.root, "feedback", "**", "*.feedback.json"), recursive=True))
        self.assertIsNone(agent.handle_choice("-5", "old", 0))

    def test_fast_actions_are_not_repeated_by_big_after_handoff(self) -> None:
        fast = Scripted([call("find_events", last_hours=24), call("send_media", handle="E1"), call("hand_off")])
        big = Scripted([call("send_media", handle="e1"), reply("One event.")])
        out = self.agent(big, fast, run_tool=self.fake_send).handle("send the video", "-5")
        self.assertEqual(self.calls.count("send_media"), 1)
        self.assertEqual(len(out.receipts), 1)
        self.assertIn("[ALREADY DONE THIS TURN]", big.seen[0][0][-1])

    def test_malformed_tool_calls_do_not_run(self) -> None:
        for args in ([], {"handle": object()}, {"seconds": float("nan")}, {"camera": object()}):
            with self.subTest(args=args):
                self.calls.clear()
                big = Scripted([ModelMessage(tool_calls=(ToolCall("bad", "send_media", args),)), reply("Hello.")])
                out = self.agent(big, run_tool=self.fake_send).handle("hi", "-5")
                self.assertEqual(self.calls, [])
                self.assertEqual(out.text, "Hello.")

    def test_followups_close_only_after_delivery_and_expire_after_ten_minutes(self) -> None:
        agent = self.agent(Scripted([]))
        book = agent.book
        book.issue("a", "set_camera_active", REQUESTED, "back_door", {"active": False, "chat_id": "-5"})
        deliverer = Mock()
        deliverer.text.return_value = {"ok": False}
        self.assertEqual(follow_up_camera_receipts(book, agent.registry, deliverer, NOW), 0)
        self.assertEqual(len(book.open_receipts("set_camera_active")), 1)
        deliverer.text.return_value = {"ok": True}
        self.assertEqual(follow_up_camera_receipts(book, agent.registry, deliverer, NOW), 1)
        self.assertEqual(book.open_receipts("set_camera_active"), [])
        self.assertEqual(follow_up_camera_receipts(book, agent.registry, deliverer, NOW), 0)
        receipt = book.issue("b", "set_camera_active", REQUESTED, "front_side", {"active": False, "chat_id": "-5"})
        with patch.object(book, "update", wraps=book.update) as update:
            self.assertEqual(follow_up_camera_receipts(book, agent.registry, deliverer, NOW + 600), 0)
            self.assertEqual(follow_up_camera_receipts(book, agent.registry, deliverer, NOW + 601), 1)
            self.assertEqual(update.call_args.args[1], FAILED)

    def test_followups_skip_malformed_receipts_and_continue(self) -> None:
        good = Receipt("R2", "b", "set_camera_active", REQUESTED, "back_door",
                       {"active": False, "chat_id": "-5"}, ts=NOW)
        for detail, ts in (([], NOW), ({"active": "false"}, NOW),
                           ({"active": False}, "bad"), ({"active": False}, float("nan")),
                           ({"active": False}, float("inf"))):
            with self.subTest(detail=detail, ts=ts):
                bad = Receipt("R1", "a", "set_camera_active", REQUESTED, "back_door", detail, ts=ts)
                book = Mock()
                book.open_receipts.return_value = [bad, good]
                deliverer = Mock()
                deliverer.text.return_value = {"ok": True}
                self.assertEqual(follow_up_camera_receipts(book, FakeRegistry(), deliverer, NOW), 1)
                book.update.assert_called_once_with(good, DONE, reason=None)

    def test_followup_delivery_exception_and_bad_result_leave_receipt_open(self) -> None:
        agent = self.agent(Scripted([]))
        agent.book.issue("a", "set_camera_active", REQUESTED, "back_door", {"active": False, "chat_id": "-5"})
        for result in ([], {"ok": "false"}, ConnectionError("offline")):
            deliverer = Mock()
            if isinstance(result, Exception):
                deliverer.text.side_effect = result
            else:
                deliverer.text.return_value = result
            self.assertEqual(follow_up_camera_receipts(agent.book, agent.registry, deliverer, NOW), 0)
            self.assertEqual(len(agent.book.open_receipts("set_camera_active")), 1)

    def test_builder_handles_bad_settings_and_vision_budget_without_network(self) -> None:
        for settings in ([], {"vision_daily_budget": "bad"}, {"vision_daily_budget": float("inf")}):
            with self.subTest(settings=settings), \
                    patch("home_guard_project.box.brain.models.make_model", return_value=Scripted([])), \
                    patch("home_guard_project.box.brain.vision.make_vision", return_value=None), \
                    patch("home_guard_project.box.embeddings.make_embedder", return_value=None), \
                    patch("home_guard_project.box.brain.agent._event_book", return_value=None):
                agent, deliverer = build_owner_agent(settings, {}, None, Mock(), self.root, self.root, self.root)
                self.assertIsNotNone(agent)
                self.assertIsNotNone(deliverer)


    # -- Fix 1 ---------------------------------------------------------------------------
    def counting(self, ctx, name, args):
        self.calls.append(name)
        from home_guard_project.box.brain.tools import TOOLS, _issue, _result  # noqa: PLC0415

        if name == "pause_alerts":
            return _result(_issue(ctx, name, DONE, "front_side", {}))
        return TOOLS[name](ctx, args)

    def test_nulls_and_empty_optionals_do_not_make_a_new_operation(self) -> None:
        fast = Scripted([call("find_events", last_hours=24), call("send_media", handle="E1"),
                         call("hand_off", reason="x")], "fast")
        big = Scripted([call("send_media", handle="E1", from_sec=None, seconds=None), reply("Here it is.")], "big")
        out = self.agent(big, fast=fast, run_tool=self.fake_send).handle("send the last video", "-5", {"user_id": 1})
        self.assertEqual(self.calls.count("send_media"), 1)
        self.assertEqual(len(out.receipts), 1)

    def test_camera_and_cameras_are_one_operation(self) -> None:
        big = Scripted([call("record_clip", camera="front"), call("record_clip", camera="front_side", seconds=10),
                        reply("Recording.")])
        self.agent(big, run_tool=self.counting).handle("record the front camera", "-5", {"user_id": 1})
        self.assertEqual(self.calls.count("record_clip"), 1)
        self.calls.clear()
        words = "pause alerts for now"
        big = Scripted([call("pause_alerts", camera="front", owner_words=words),
                        call("pause_alerts", cameras=["front_side"], owner_words=words), reply("Paused.")])
        self.agent(big, run_tool=self.counting).handle(words, "-6", {"user_id": 1})
        self.assertEqual(self.calls.count("pause_alerts"), 1)

    def test_verdict_handle_defaults_to_the_alert_and_clip_seconds_to_ten(self) -> None:
        from home_guard_project.box.brain.agent import _effective  # noqa: PLC0415
        from home_guard_project.box.brain.tools import ToolContext  # noqa: PLC0415

        ctx = ToolContext(turn_id="t", chat_id="1", speaker={}, text="", lang="en", mode="guard",
                          snapshot=snapshot("guard"), state=None, services=None, book=None, threaded=False)
        ctx.alert_handle = "E3"
        self.assertEqual(_effective(ctx, "record_verdict", {"verdict": "ok"}),
                         _effective(ctx, "record_verdict", {"verdict": "ok", "handle": "e3"}))
        self.assertEqual(_effective(ctx, "record_clip", {"camera": "front"}),
                         _effective(ctx, "record_clip", {"camera": "front", "seconds": 10}))

    def question_with_send(self):
        return ModelMessage(tool_calls=(
            ToolCall(id="p", name="send_media", arguments={"handle": "E1"}),
            ToolCall(id="q", name="ask_clarification", arguments={"question": "Which?", "choices": ["a", "b"]})))

    def test_a_question_still_reports_what_was_done(self) -> None:
        big = Scripted([call("find_events", last_hours=24), self.question_with_send()])
        out = self.agent(big, run_tool=self.fake_send).handle("send it", "-5", {"user_id": 1})
        self.assertIn("✓ Video sent", out.text)
        self.assertTrue(out.text.endswith("\n\nWhich?"))
        self.assertEqual(out.buttons, ("a", "b"))

    def test_a_failed_receipt_before_a_question_is_shown(self) -> None:
        def failing(ctx, name, args):
            if name == "send_media":
                from home_guard_project.box.brain.tools import _issue, _result  # noqa: PLC0415

                return _result(_issue(ctx, "send_media", FAILED, "E1", {"kind": "video"}, "telegram"))
            return self.counting(ctx, name, args)

        big = Scripted([call("find_events", last_hours=24), self.question_with_send()])
        out = self.agent(big, run_tool=failing).handle("send it", "-5", {"user_id": 1})
        self.assertIn("✗", out.text)
        self.assertTrue(out.text.endswith("Which?"))

    def test_a_used_question_token_does_nothing_the_second_time(self) -> None:
        big = Scripted([call("ask_clarification", question="Which?", choices=["a", "b"]), reply("ok")])
        agent = self.agent(big)
        first = agent.handle("hi", "-5")
        self.assertIsNotNone(agent.handle_choice("-5", first.question_token, 0))
        seen = len(big.seen)
        self.assertIsNone(agent.handle_choice("-5", first.question_token, 0))
        self.assertEqual(len(big.seen), seen)

    def test_the_token_is_checked_inside_the_turn_lock(self) -> None:
        big = Scripted([call("ask_clarification", question="Which?", choices=["a", "b"]), reply("ok")])
        agent = self.agent(big)
        first = agent.handle("hi", "-5")
        real, loads = agent.memory.load, {"n": 0}

        def load(chat_id):                 # another tap wins between the early check and the turn
            loads["n"] += 1
            state = real(chat_id)
            if loads["n"] > 1:
                state.pending = None
            return state

        agent.memory.load = load
        seen = len(big.seen)
        self.assertIsNone(agent.handle_choice("-5", first.question_token, 0))
        self.assertEqual(len(big.seen), seen)

    def test_the_rewrite_request_contains_the_bad_answer(self) -> None:
        big = Scripted([ModelMessage(content="I sent you the video."), reply("There was one event.")])
        out = self.agent(big).handle("send the video", "-5", {"user_id": 1})
        self.assertEqual(out.text, "There was one event.")
        sent = big.seen[1][0]
        self.assertEqual(sent[-2], "I sent you the video.")
        self.assertIn("[BOX] Your answer describes actions", sent[-1])

    def test_an_empty_fast_answer_is_a_handoff(self) -> None:
        bad = ToolCall("x", "reply", {"answer": object()}, valid=False)
        for fast_msg in (ModelMessage(), ModelMessage(tool_calls=(bad,))):
            with self.subTest(fast=fast_msg):
                out = self.agent(Scripted([reply("Hello.")]), Scripted([fast_msg], "fast")).handle("hi", "-5")
                self.assertEqual((out.text, out.tier, out.escalated), ("Hello.", "big", True))

    def test_a_failure_after_tools_ran_keeps_receipts_and_after_actions(self) -> None:
        big = Scripted([call("set_camera_active", camera="front_side", active=False,
                             owner_words="turn off the front"), reply("Done.")])
        agent = self.agent(big)
        agent.services.set_camera = lambda camera, active: {"ok": True}
        agent.services.request_restart = lambda: None
        agent.memory.save = Mock(side_effect=OSError("disk"))
        out = agent.handle("turn off the front camera", "-5", {"user_id": 1})
        self.assertTrue(out.receipts)
        self.assertTrue(out.after)
        self.assertEqual(len(glob.glob(os.path.join(self.root, "feedback", "**", "*.feedback.json"),
                                       recursive=True)), 1)

    def test_builder_wires_services(self) -> None:
        from home_guard_project.box import find_cameras  # noqa: PLC0415
        from home_guard_project.box.brain.vision import BudgetedVision  # noqa: PLC0415

        with patch("home_guard_project.box.brain.models.make_model", return_value=Scripted([])), \
                patch("home_guard_project.box.brain.vision.make_vision", return_value=Mock()), \
                patch("home_guard_project.box.embeddings.make_embedder", return_value=None), \
                patch("home_guard_project.box.brain.agent._event_book", return_value=None):
            with patch.object(find_cameras, "apply_changes") as apply:
                agent, _ = build_owner_agent({}, {"OPENAI_API_KEY": "k"}, None, Mock(), self.root, self.root,
                                             self.root)
                agent.services.set_camera("front_side", False)
        services = agent.services
        self.assertIs(services.request_restart, find_cameras._restart_running_mode)
        self.assertEqual(services.feedback_dir, self.root)
        self.assertIsInstance(services.vision, BudgetedVision)
        self.assertTrue(os.path.abspath(services.vision.path).startswith(
            os.path.abspath(os.path.join(self.root, ".registry"))))
        self.assertIs(apply.call_args.kwargs["restart"], False)


if __name__ == "__main__":
    unittest.main()
