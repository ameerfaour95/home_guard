# tests/box/test_brain_models.py
from __future__ import annotations

import json
import logging
import types
import unittest
from unittest import mock

from home_guard_project.box.brain.models import AnthropicChat, OpenAIChat, make_model, to_anthropic_messages

TOOLS = [{"type": "function", "function": {"name": "check_camera", "description": "look now",
                                           "parameters": {"type": "object", "properties": {"camera": {"type": "string"}},
                                                          "required": ["camera"]}}}]
NS = types.SimpleNamespace


class FakeOpenAI:
    def __init__(self, message, finish="stop"):
        self.kwargs = None
        self._message, self._finish = message, finish
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return NS(choices=[NS(message=self._message, finish_reason=self._finish)],
                  usage=NS(prompt_tokens=100, completion_tokens=20))


class FakeAnthropic:
    def __init__(self, content, stop="end_turn"):
        self.kwargs = None
        self._content, self._stop = content, stop
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return NS(content=self._content, stop_reason=self._stop, usage=NS(input_tokens=90, output_tokens=15))


class OpenAIChatTest(unittest.TestCase):
    def test_tool_calls_usage_and_private_keys(self) -> None:
        call = NS(id="c1", function=NS(name="check_camera", arguments='{"camera": "gate"}'))
        client = FakeOpenAI(NS(content=None, tool_calls=[call]))
        msg = OpenAIChat(client, "gpt-4o").chat(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "_raw": [1]}], TOOLS)
        self.assertEqual((msg.tool_calls[0].name, msg.tool_calls[0].arguments), ("check_camera", {"camera": "gate"}))
        self.assertEqual(msg.usage, (100, 20))
        self.assertNotIn("_raw", client.kwargs["messages"][1])
        self.assertEqual(client.kwargs["tool_choice"], "auto")

    def test_truncated_or_bad_json_is_invalid(self) -> None:
        call = NS(id="c1", function=NS(name="check_camera", arguments='{"camera": "ga'))
        msg = OpenAIChat(FakeOpenAI(NS(content=None, tool_calls=[call]), finish="length"), "m").chat([], TOOLS)
        self.assertFalse(msg.tool_calls[0].valid)

    def test_no_temperature_for_reasoning_models(self) -> None:
        client = FakeOpenAI(NS(content="ok", tool_calls=None))
        OpenAIChat(client, "gpt-6-sol", temperature=None).chat([], TOOLS)
        self.assertNotIn("temperature", client.kwargs)


class AnthropicChatTest(unittest.TestCase):
    def test_conversion_groups_tool_results_and_replays_raw_blocks(self) -> None:
        raw = [NS(type="thinking"), NS(type="tool_use", id="t1", name="check_camera", input={"camera": "gate"})]
        system, msgs = to_anthropic_messages([
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "look"},
            {"role": "assistant", "content": None, "_raw": raw,
             "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "check_camera", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": '{"ok": true}'},
            {"role": "tool", "tool_call_id": "t2", "content": '{"ok": false}'},
        ])
        self.assertEqual(system, "rules")
        self.assertIs(msgs[1]["content"], raw)
        self.assertEqual([b["tool_use_id"] for b in msgs[2]["content"]], ["t1", "t2"])

    def test_assistant_without_raw_is_rebuilt(self) -> None:
        _, msgs = to_anthropic_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "x", "arguments": '{"k": 1}'}}]},
        ])
        self.assertEqual(msgs[1]["content"][1], {"type": "tool_use", "id": "t1", "name": "x", "input": {"k": 1}})

    def test_chat_reads_text_tools_usage_and_refusal(self) -> None:
        client = FakeAnthropic([NS(type="text", text="Looking."),
                                NS(type="tool_use", id="t1", name="check_camera", input={"camera": "gate"})],
                               stop="tool_use")
        chat = AnthropicChat(client, "claude-sonnet-5-5", effort="low")
        msg = chat.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], TOOLS)
        self.assertEqual((msg.content, msg.tool_calls[0].arguments, msg.usage), ("Looking.", {"camera": "gate"},
                                                                                 (90, 15)))
        self.assertEqual(client.kwargs["tool_choice"], {"type": "auto"})
        self.assertEqual(client.kwargs["output_config"], {"effort": "low"})
        self.assertEqual(client.kwargs["tools"][0]["input_schema"]["required"], ["camera"])
        chat.chat([{"role": "user", "content": "q"}], TOOLS, tool_choice="none")
        self.assertEqual(client.kwargs["tool_choice"], {"type": "none"})
        refused = AnthropicChat(FakeAnthropic([], stop="refusal"), "m").chat([{"role": "user", "content": "q"}], TOOLS)
        self.assertTrue(refused.refused)


class MakeModelTest(unittest.TestCase):
    def test_no_key_no_model_and_unknown_provider(self) -> None:
        self.assertIsNone(make_model("openai:gpt-4o", {}))
        self.assertIsNone(make_model("anthropic:claude-sonnet-5-5", {}))
        self.assertIsNone(make_model("mistral:x", {"MISTRAL_API_KEY": "k"}))
        self.assertIsNone(make_model("", {"OPENAI_API_KEY": "k"}))

    def test_providers(self) -> None:
        self.assertEqual(make_model("openai:gpt-4o", {"OPENAI_API_KEY": "k"}).model_name, "gpt-4o")
        self.assertEqual(make_model("gpt-4o-mini", {"OPENAI_API_KEY": "k"}).model_name, "gpt-4o-mini")
        self.assertIsInstance(make_model("anthropic:claude-haiku-4-5", {"ANTHROPIC_API_KEY": "k"}), AnthropicChat)
        gem = make_model("gemini:gemini-flash", {"GEMINI_API_KEY": "k"})
        self.assertIn("generativelanguage", str(gem._client.base_url))


class RobustnessTest(unittest.TestCase):
    def test_malformed_content_and_call_identifiers_cannot_escape(self) -> None:
        self.assertIsNone(OpenAIChat(FakeOpenAI(NS(content=[], tool_calls=[])), "m").chat([], TOOLS).content)
        for call_id, name in (([], "x"), ("c", None), ("", "x")):
            with self.subTest(call_id=call_id, name=name):
                oa = FakeOpenAI(NS(content=None, tool_calls=[
                    NS(id=call_id, function=NS(name=name, arguments="{}"))]))
                ant = FakeAnthropic([NS(type="tool_use", id=call_id, name=name, input={})])
                for chat in (OpenAIChat(oa, "m"), AnthropicChat(ant, "m")):
                    self.assertEqual(chat.chat([], TOOLS).tool_calls, ())

    def test_refusal_truncation_no_tools_and_raw_identity(self) -> None:
        oa = FakeOpenAI(NS(content=None, tool_calls=None, refusal="declined"))
        self.assertTrue(OpenAIChat(oa, "m").chat([], []).refused)
        self.assertNotIn("tool_choice", oa.kwargs)
        raw = [NS(type="tool_use", id="c", name="x", input={})]
        ant = FakeAnthropic(raw, stop="max_tokens")
        msg = AnthropicChat(ant, "m").chat([], [])
        self.assertFalse(msg.tool_calls[0].valid)
        self.assertIs(msg.raw, raw)
        self.assertNotIn("tools", ant.kwargs)

    def test_factory_settings_with_injected_sdk_constructors(self) -> None:
        with mock.patch("openai.OpenAI") as oa, mock.patch("anthropic.Anthropic") as ant, \
                mock.patch("home_guard_project.box.brain.models._http_client"):
            model = make_model("openai:gpt-6-sol", {"OPENAI_API_KEY": "k"})
            model.chat([], TOOLS)
            self.assertNotIn("temperature", oa.return_value.chat.completions.create.call_args.kwargs)
            make_model("gemini:flash", {"GEMINI_API_KEY": "g"})
            self.assertEqual(oa.call_args.kwargs["base_url"],
                             "https://generativelanguage.googleapis.com/v1beta/openai/")
            haiku = make_model("anthropic:claude-haiku-4-5", {"ANTHROPIC_API_KEY": "k"})
            haiku.chat([], TOOLS)
            self.assertNotIn("output_config", ant.return_value.messages.create.call_args.kwargs)
            sonnet = make_model("anthropic:claude-sonnet-5-5", {"ANTHROPIC_API_KEY": "k"})
            sonnet.chat([], TOOLS)
            self.assertEqual(ant.return_value.messages.create.call_args.kwargs["output_config"], {"effort": "low"})

    def test_openai_malformed_requests_responses_and_offline_client(self) -> None:
        client = FakeOpenAI(NS(content="ok", tool_calls=None))
        chat = OpenAIChat(client, "m")
        self.assertIsNone(chat.chat([None], TOOLS).content)
        client._message = NS(content="ok", tool_calls=[NS(function=[])])
        self.assertEqual(chat.chat([], TOOLS).tool_calls, ())
        client.chat.completions.create = mock.Mock(side_effect=ConnectionError("offline"))
        self.assertIsNone(chat.chat([], TOOLS).content)

    def test_bad_usage_is_zero_without_losing_text(self) -> None:
        for bad in ("bad", [], float("nan"), float("inf")):
            with self.subTest(bad=bad):
                oa = FakeOpenAI(None)
                oa.chat.completions.create = mock.Mock(return_value=NS(
                    choices=[NS(message=NS(content="ok", tool_calls=None))],
                    usage=NS(prompt_tokens=bad, completion_tokens=bad)))
                ant = FakeAnthropic([])
                ant.messages.create = mock.Mock(return_value=NS(
                    content=[NS(type="text", text="ok")],
                    usage=NS(input_tokens=bad, output_tokens=bad)))
                for chat in (OpenAIChat(oa, "m"), AnthropicChat(ant, "m")):
                    msg = chat.chat([], TOOLS)
                    self.assertEqual((msg.content, msg.usage), ("ok", (0, 0)))

    def test_bad_arguments_are_invalid_on_both_providers(self) -> None:
        for args in ([], {"x": float("nan")}, {"x": float("inf")}, {"x": object()}):
            with self.subTest(args=args):
                ant = FakeAnthropic([NS(type="tool_use", id="c", name="x", input=args)])
                msg = AnthropicChat(ant, "m").chat([], TOOLS)
                self.assertFalse(msg.tool_calls[0].valid)
                self.assertEqual(msg.tool_calls[0].arguments, {})
        for raw in ('[]', '{"x": NaN}', '{"x": 1e999}', b'\xff', [], {}, 0, None):
            with self.subTest(raw=raw):
                call = NS(id="c", function=NS(name="x", arguments=raw))
                msg = OpenAIChat(FakeOpenAI(NS(content=None, tool_calls=[call])), "m").chat([], TOOLS)
                self.assertFalse(msg.tool_calls[0].valid)

    def test_anthropic_malformed_requests_responses_and_offline_client(self) -> None:
        client = FakeAnthropic([NS(type="text", text=[])])
        chat = AnthropicChat(client, "m")
        self.assertIsNone(chat.chat([], TOOLS).content)
        client._content = []
        self.assertIsNone(chat.chat([], [{"function": []}, None]).content)
        client.messages.create = mock.Mock(side_effect=ConnectionError("offline"))
        self.assertIsNone(chat.chat([], TOOLS).content)

    def test_converter_skips_malformed_history_and_preserves_raw(self) -> None:
        raw = [NS(type="thinking", signature="unchanged")]
        system, msgs = to_anthropic_messages([
            None, [], {"role": "system", "content": "rules"},
            {"role": "assistant", "content": "ok", "tool_calls": [None, {"function": []}]},
            {"role": "assistant", "_raw": raw},
        ])
        self.assertEqual(system, "rules")
        self.assertEqual(msgs[0]["content"], [{"type": "text", "text": "ok"}])
        self.assertIs(msgs[1]["content"], raw)
        self.assertEqual(to_anthropic_messages(None), ("", []))

    def test_factory_malformed_config_and_initialization_failure(self) -> None:
        for spec, env in (([], {}), ("openai:m", []), ("openai:m", {"OPENAI_API_KEY": []}),
                          ("openai:", {"OPENAI_API_KEY": "k"})):
            with self.subTest(spec=spec, env=env):
                self.assertIsNone(make_model(spec, env))
        with mock.patch("openai.OpenAI", side_effect=ValueError("bad client")), \
                mock.patch("home_guard_project.box.brain.models._http_client") as http:
            self.assertIsNone(make_model("openai:m", {"OPENAI_API_KEY": "k"}))
            http.return_value.close.assert_called_once()

    def test_forced_choices_are_never_sent(self) -> None:
        oa = FakeOpenAI(NS(content="ok", tool_calls=None))
        ant = FakeAnthropic([])
        for choice in ("required", {"type": "function", "function": {"name": "x"}}):
            OpenAIChat(oa, "m").chat([], TOOLS, choice)
            AnthropicChat(ant, "m").chat([], TOOLS, choice)
            self.assertEqual(oa.kwargs["tool_choice"], "auto")
            self.assertEqual(ant.kwargs["tool_choice"], {"type": "auto"})


class FailureVisibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        from home_guard_project.box.brain import models
        models._last_warned.clear()

    def test_failure_is_reported_and_normal_reply_has_no_error(self) -> None:
        client = FakeOpenAI(None)
        client.chat.completions.create = mock.Mock(side_effect=ConnectionError("offline"))
        with self.assertLogs("box.brain.models", "WARNING"):
            msg = OpenAIChat(client, "m").chat([], TOOLS)
        self.assertEqual(msg.error, "ConnectionError: offline")
        self.assertEqual(msg.tool_calls, ())
        ant = FakeAnthropic([])
        ant.messages.create = mock.Mock(side_effect=RuntimeError("x" * 500))
        with self.assertLogs("box.brain.models", "WARNING"):
            msg = AnthropicChat(ant, "m").chat([], TOOLS)
        self.assertTrue(msg.error.startswith("RuntimeError: "))
        self.assertLessEqual(len(msg.error), 300)
        ok = OpenAIChat(FakeOpenAI(NS(content="hi", tool_calls=None)), "m").chat([], TOOLS)
        self.assertEqual(ok.error, "")

    def test_failures_log_once_per_window(self) -> None:
        client = FakeOpenAI(None)
        client.chat.completions.create = mock.Mock(side_effect=ConnectionError("offline"))
        chat = OpenAIChat(client, "m")
        with mock.patch("home_guard_project.box.brain.models.time.monotonic", return_value=1000.0):
            with self.assertLogs("box.brain.models", "WARNING") as cm:
                chat.chat([], TOOLS)
                chat.chat([], TOOLS)
            self.assertEqual(len(cm.records), 1)
            self.assertIsNotNone(cm.records[0].exc_info)
        with mock.patch("home_guard_project.box.brain.models.time.monotonic", return_value=1301.0):
            with self.assertLogs("box.brain.models", "WARNING") as cm:
                chat.chat([], TOOLS)
            self.assertEqual(len(cm.records), 1)

    def test_empty_raw_is_rebuilt_or_dropped(self) -> None:
        _, msgs = to_anthropic_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "_raw": []},
            {"role": "assistant", "content": None, "_raw": []},
        ])
        self.assertEqual(msgs, [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": [{"type": "text", "text": "a"}]}])

    def test_empty_user_text_is_skipped(self) -> None:
        _, msgs = to_anthropic_messages([{"role": "user", "content": ""}, {"role": "user", "content": None},
                                         {"role": "user", "content": "q"}])
        self.assertEqual(msgs, [{"role": "user", "content": "q"}])

    def test_model_name_is_stripped(self) -> None:
        model = make_model("openai: gpt-6-sol", {"OPENAI_API_KEY": "k"})
        self.assertEqual(model.model_name, "gpt-6-sol")
        self.assertIsNone(model._temperature)

    def test_missing_sdk_is_named_in_the_log(self) -> None:
        with mock.patch.dict("sys.modules", {"anthropic": None}):
            with self.assertLogs("box.brain.models", "WARNING") as cm:
                self.assertIsNone(make_model("anthropic:claude-sonnet-5-5", {"ANTHROPIC_API_KEY": "k"}))
        self.assertIn("anthropic is not installed", " ".join(r.getMessage() for r in cm.records))


if __name__ == "__main__":
    unittest.main()


class OpenRouterModelTest(unittest.TestCase):
    """2026-10-08: the box's OpenAI credit ran out; the assistant can run on OpenRouter like the Eye."""

    def test_openrouter_model(self) -> None:
        from home_guard_project.box.brain.models import OpenAIChat, make_model

        model = make_model("openrouter:openai/gpt-4o", {"OPENROUTER_API_KEY": "k"})
        self.assertIsInstance(model, OpenAIChat)
        self.assertEqual(model.model_name, "openai/gpt-4o")
        self.assertIsNone(make_model("openrouter:openai/gpt-4o", {"OPENAI_API_KEY": "k"}))
