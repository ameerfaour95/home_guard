# tests/box/test_brain_vision_providers.py
"""The assistant's vision model follows box.yaml's vlm_provider / vlm_model (and the fallback), like the Eye.

2026-10-06: the box switched to vlm_provider: openrouter, vlm_model: qwen/qwen3.5-9b, and every assistant look
(describe_event, assess_event, ask_vision, live photos) sent "qwen/qwen3.5-9b" to OpenAI and failed.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from home_guard_project.box.brain.vision import make_vision

ANSWER = json.dumps({"description": "A man at the door.", "quality": "clear", "people": 1})


def client_answering(*answers):
    """A fake OpenAI client whose create() returns (or raises) each of *answers* in turn."""
    client = mock.Mock()

    def create(**kwargs):
        item = answers[min(len(client.chat.completions.create.call_args_list) - 1, len(answers) - 1)]
        if isinstance(item, Exception):
            raise item
        return mock.Mock(choices=[mock.Mock(message=mock.Mock(refusal=None, content=item))])

    client.chat.completions.create.side_effect = create
    return client


class VisionProviderTest(unittest.TestCase):
    def test_openrouter_takes_its_url_key_and_reasoning_off_body_and_the_model_as_given(self) -> None:
        client = client_answering(ANSWER)
        with mock.patch("openai.OpenAI", return_value=client) as ctor, mock.patch("httpx.Client"):
            vision = make_vision({"OPENROUTER_API_KEY": "or-key", "OPENAI_API_KEY": "sk"}, "qwen/qwen3.5-9b",
                                 provider="openrouter")
            self.assertTrue(vision.look("gate", [b"jpg"], guard=False)["ok"])
        made = ctor.call_args.kwargs
        self.assertEqual((made["api_key"], made["base_url"]), ("or-key", "https://openrouter.ai/api/v1"))
        self.assertEqual((made["timeout"], made["max_retries"]), (30.0, 1))
        sent = client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["model"], "qwen/qwen3.5-9b")
        self.assertEqual(sent["extra_body"], {"reasoning": {"enabled": False}})
        self.assertEqual(vision.model_name, "qwen/qwen3.5-9b")

    def test_a_missing_key_or_an_unknown_provider_is_no_vision_and_a_warning(self) -> None:
        for env, provider in (({"OPENAI_API_KEY": "sk"}, "openrouter"), ({"OPENROUTER_API_KEY": "k"}, "nope")):
            with self.subTest(provider=provider), mock.patch("openai.OpenAI") as ctor, mock.patch("httpx.Client"), \
                    self.assertLogs("box.brain.vision", level="WARNING") as logs:
                self.assertIsNone(make_vision(env, "qwen/qwen3.5-9b", provider=provider))
                ctor.assert_not_called()
            self.assertIn("qwen/qwen3.5-9b", logs.output[0])

    def test_no_provider_is_openai_exactly_as_before(self) -> None:
        client = client_answering(ANSWER)
        with mock.patch("openai.OpenAI", return_value=client) as ctor, mock.patch("httpx.Client"):
            for vision in (make_vision({"OPENAI_API_KEY": "sk"}),
                           make_vision({"OPENAI_API_KEY": "sk"}, provider=""),
                           make_vision({"OPENAI_API_KEY": "sk"}, "gpt-4o", provider="openai")):
                self.assertTrue(vision.look("gate", [b"jpg"], guard=False)["ok"])
        self.assertTrue(all("base_url" not in c.kwargs and c.kwargs["api_key"] == "sk" for c in ctor.call_args_list))
        sent = client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["model"], "gpt-4o")
        self.assertNotIn("extra_body", sent)
        self.assertIsNone(make_vision({}, provider="openai"))                 # no key: no vision, as before

    def test_the_fallback_answers_once_when_the_main_model_fails(self) -> None:
        main, spare = client_answering(RuntimeError("502 upstream")), client_answering(ANSWER)
        env = {"OPENROUTER_API_KEY": "or-key", "DASHSCOPE_API_KEY": "ds-key"}
        with mock.patch("openai.OpenAI", side_effect=[main, spare]) as ctor, mock.patch("httpx.Client"), \
                self.assertLogs("box.brain.vision", level="WARNING") as logs:
            vision = make_vision(env, "qwen/qwen3.5-9b", provider="openrouter",
                                 fallback_provider="dashscope-intl", fallback_model="qwen3-vl-8b-instruct")
            self.assertTrue(vision.look("gate", [b"jpg"], guard=False)["ok"])
        self.assertEqual([c.kwargs["api_key"] for c in ctor.call_args_list], ["or-key", "ds-key"])
        sent = spare.chat.completions.create.call_args.kwargs
        self.assertEqual((sent["model"], sent["extra_body"]), ("qwen3-vl-8b-instruct", {"enable_thinking": False}))
        self.assertIn("fallback", " ".join(logs.output))

    def test_the_fallback_also_answers_a_reply_that_is_not_json(self) -> None:
        main, spare = client_answering("I think it is fine"), client_answering(ANSWER)
        with mock.patch("openai.OpenAI", side_effect=[main, spare]), mock.patch("httpx.Client"):
            vision = make_vision({"OPENROUTER_API_KEY": "k"}, "qwen/qwen3.5-9b", provider="openrouter",
                                 fallback_model="qwen/qwen3-vl-8b-instruct")       # same provider by default
            self.assertTrue(vision.look("gate", [b"jpg"], guard=False)["ok"])
        self.assertEqual(spare.chat.completions.create.call_args.kwargs["model"], "qwen/qwen3-vl-8b-instruct")

    def test_the_fallback_alone_when_the_main_model_cannot_be_built(self) -> None:
        spare = client_answering(ANSWER)
        with mock.patch("openai.OpenAI", return_value=spare) as ctor, mock.patch("httpx.Client"), \
                self.assertLogs("box.brain.vision", level="WARNING"):
            vision = make_vision({"OPENAI_API_KEY": "sk"}, "qwen/qwen3.5-9b", provider="openrouter",
                                 fallback_provider="openai", fallback_model="gpt-4o")
            self.assertTrue(vision.look("gate", [b"jpg"], guard=False)["ok"])
        self.assertEqual(ctor.call_count, 1)
        self.assertEqual(spare.chat.completions.create.call_args.kwargs["model"], "gpt-4o")
        self.assertEqual(vision.model_name, "gpt-4o")

    def test_the_same_model_as_fallback_is_no_fallback(self) -> None:
        with mock.patch("openai.OpenAI", return_value=client_answering(ANSWER)) as ctor, mock.patch("httpx.Client"):
            make_vision({"OPENROUTER_API_KEY": "k"}, "qwen/qwen3.5-9b", provider="openrouter",
                        fallback_provider="openrouter", fallback_model="qwen/qwen3.5-9b")
        self.assertEqual(ctor.call_count, 1)

    def test_a_model_that_refuses_the_schema_gets_plain_json_from_then_on(self) -> None:
        client = client_answering(ValueError("response_format json_schema is not supported"), ANSWER, ANSWER)
        with mock.patch("openai.OpenAI", return_value=client), mock.patch("httpx.Client"):
            vision = make_vision({"OPENROUTER_API_KEY": "k"}, "qwen/qwen3.5-9b", provider="openrouter")
            self.assertTrue(vision.look("gate", [b"jpg"], guard=False)["ok"])
            self.assertTrue(vision.look("gate", [b"jpg"], guard=False)["ok"])
        formats = [c.kwargs["response_format"]["type"] for c in client.chat.completions.create.call_args_list]
        self.assertEqual(formats, ["json_schema", "json_object", "json_object"])


class AgentWiringTest(unittest.TestCase):
    def test_the_assistant_builds_its_vision_from_the_box_settings(self) -> None:
        import tempfile

        from home_guard_project.box.brain.agent import build_owner_agent

        settings = {"vlm_provider": "OpenRouter", "vlm_model": "qwen/qwen3.5-9b",
                    "vlm_fallback_provider": "dashscope-intl", "vlm_fallback_model": "qwen3-vl-8b-instruct"}
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("home_guard_project.box.brain.models.make_model", return_value=mock.Mock()), \
                mock.patch("home_guard_project.box.brain.vision.make_vision", return_value=None) as made, \
                mock.patch("home_guard_project.box.embeddings.make_embedder", return_value=None), \
                mock.patch("home_guard_project.box.brain.agent._event_book", return_value=None):
            build_owner_agent(settings, {"OPENAI_API_KEY": "k"}, None, mock.Mock(), root, root, root)
            build_owner_agent({}, {"OPENAI_API_KEY": "k"}, None, mock.Mock(), root, root, root)
        first, second = made.call_args_list
        self.assertEqual(first.args[1:], ("qwen/qwen3.5-9b",))
        self.assertEqual(first.kwargs, {"provider": "openrouter", "fallback_provider": "dashscope-intl",
                                        "fallback_model": "qwen3-vl-8b-instruct"})
        self.assertEqual((second.args[1:], second.kwargs),
                         (("gpt-4o",), {"provider": "openai", "fallback_provider": "", "fallback_model": ""}))


if __name__ == "__main__":
    unittest.main()
