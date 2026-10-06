"""Tests for the vision model's provider options and the Qwen fallback (box/inference.py)."""
from __future__ import annotations

import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import httpx  # noqa: F401  (imported before the TLS setup is patched)
import openai  # noqa: F401

from home_guard_project.box import inference as inf
from home_guard_project.box.inference import AlertSettings, FallbackBackend, GptBackend, NullBackend

ANSWER = {"summary": "a person walks", "label": "normal", "people": 1,
          "vehicle_moving": False, "animals": 0, "why": "", "summary_owner": ""}


def openai_client():
    """Patch the OpenAI client and the TLS setup GptBackend builds (ssl.create_default_context
    aborts the process on a machine whose antivirus sets SSLKEYLOGFILE)."""
    stack = ExitStack()
    stack.enter_context(mock.patch("ssl.create_default_context"))
    stack.enter_context(mock.patch("httpx.Client"))
    client = stack.enter_context(mock.patch("openai.OpenAI"))
    return stack, client


class patched_client:
    def __enter__(self):
        self.stack, client = openai_client()
        return client

    def __exit__(self, *exc):
        self.stack.close()


def response(content: str, usage=(1200, 60)):
    u = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=u)


class Stub:
    """A backend that answers (raw, parsed) or raises."""

    def __init__(self, name: str, answer=None, error: Exception = None) -> None:
        self.model_name = name
        self.answer, self.error, self.calls = answer, error, 0
        self.last_prompt = f"prompt of {name}"
        self.last_frame_jpegs = [name.encode()]
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1}

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, **kw):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


class GptBackendOptionsTest(unittest.TestCase):
    def test_base_url_timeout_and_key_reach_the_client(self) -> None:
        with patched_client() as client:
            GptBackend("k", "Qwen/Qwen3-VL-4B-Instruct", base_url="http://pod:8000/v1")
        kw = client.call_args.kwargs
        self.assertEqual((kw["api_key"], kw["base_url"], kw["timeout"]), ("k", "http://pod:8000/v1", 30.0))

    def test_no_base_url_for_openai(self) -> None:
        with patched_client() as client:
            GptBackend("sk-1", "gpt-4o")
        self.assertNotIn("base_url", client.call_args.kwargs)

    def test_extra_body_sent_and_usage_recorded(self) -> None:
        with patched_client():
            b = GptBackend("or-1", "qwen/qwen3.5-9b", extra_body={"reasoning": {"enabled": False}})
        b._client = mock.Mock()
        b._client.chat.completions.create.return_value = response(json.dumps(ANSWER), usage=(2100, 75))
        raw, parsed = b.analyze([], "cam", 0, 0, 0)
        self.assertEqual(parsed["summary"], "a person walks")
        self.assertEqual(b._client.chat.completions.create.call_args.kwargs["extra_body"],
                         {"reasoning": {"enabled": False}})
        self.assertEqual(b.last_usage, {"prompt_tokens": 2100, "completion_tokens": 75})

    def test_backend_built_without_init_still_works(self) -> None:
        """Older tests build GptBackend with __new__ and none of the new attributes."""
        b = GptBackend.__new__(GptBackend)
        b._model = b.model_name = "gpt-stub"
        b._response_format = inf.VLM_RESPONSE_FORMAT
        b._client = mock.Mock()
        b._client.chat.completions.create.return_value = response(json.dumps(ANSWER), usage=None)
        b.analyze([], "cam", 0, 0, 0)
        self.assertNotIn("extra_body", b._client.chat.completions.create.call_args.kwargs)
        self.assertEqual(b.last_usage, {"prompt_tokens": 0, "completion_tokens": 0})


class FallbackBackendTest(unittest.TestCase):
    def test_primary_fine_fallback_never_called(self) -> None:
        p, f = Stub("qwen3-vl", answer=("{}", dict(ANSWER))), Stub("qwen3.5", answer=("{}", {"summary": "f"}))
        fb = FallbackBackend(p, f)
        self.assertEqual(fb.analyze([], "cam", 0, 0, 0, owner_language="he")[1]["summary"], "a person walks")
        self.assertEqual((p.calls, f.calls, fb.model_name, fb.last_prompt), (1, 0, "qwen3-vl", "prompt of qwen3-vl"))

    def test_primary_raises_fallback_answers(self) -> None:
        p, f = Stub("qwen3-vl", error=TimeoutError("slow")), Stub("qwen3.5", answer=("{}", {"summary": "f"}))
        fb = FallbackBackend(p, f)
        with self.assertLogs("box.inference", "WARNING") as logs:
            raw, parsed = fb.analyze([], "cam", 0, 0, 0)
        self.assertEqual(parsed["summary"], "f")
        self.assertEqual((fb.model_name, fb.last_frame_jpegs), ("qwen3.5", [b"qwen3.5"]))
        self.assertIn("VLM fallback", logs.output[0])

    def test_primary_junk_answer_goes_to_fallback(self) -> None:
        p, f = Stub("qwen3-vl", answer=("not json", None)), Stub("qwen3.5", answer=("{}", {"summary": "f"}))
        self.assertEqual(FallbackBackend(p, f).analyze([], "cam", 0, 0, 0)[1]["summary"], "f")

    def test_both_fail_raises(self) -> None:
        p, f = Stub("qwen3-vl", error=RuntimeError("a")), Stub("qwen3.5", error=RuntimeError("b"))
        with self.assertRaises(RuntimeError):
            FallbackBackend(p, f).analyze([], "cam", 0, 0, 0)


class MakeBackendTest(unittest.TestCase):
    PAIR = dict(vlm_provider="vllm", vlm_model="Qwen/Qwen3-VL-4B-Instruct",
                vlm_fallback_provider="vllm", vlm_fallback_model="Qwen/Qwen3.5-4B")

    def test_defaults_are_todays_single_gpt4o(self) -> None:
        with patched_client():
            b = inf.make_backend(AlertSettings(), {"OPENAI_API_KEY": "sk"})
        self.assertIsInstance(b, GptBackend)
        self.assertEqual(b.model_name, "gpt-4o")

    def test_qwen_pair_wraps_with_fallback(self) -> None:
        with patched_client() as client:
            b = inf.make_backend(AlertSettings(**self.PAIR), {"VLLM_BASE_URL": "http://pod:8000/v1"})
        self.assertIsInstance(b, FallbackBackend)
        self.assertEqual((b.primary.model_name, b.fallback.model_name),
                         ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3.5-4B"))
        self.assertEqual(client.call_args.kwargs["base_url"], "http://pod:8000/v1")

    def test_missing_primary_uses_fallback_alone(self) -> None:
        s = AlertSettings(vlm_provider="openrouter", vlm_model="qwen/qwen3-vl-8b-instruct",
                          vlm_fallback_provider="ollama", vlm_fallback_model="qwen3.5:4b-bf16")
        with patched_client(), self.assertLogs("box.inference", "WARNING"):
            b = inf.make_backend(s, {})
        self.assertIsInstance(b, GptBackend)
        self.assertEqual(b.model_name, "qwen3.5:4b-bf16")

    def test_no_fallback_by_default(self) -> None:
        with patched_client():
            b = inf.make_backend(AlertSettings(vlm_provider="ollama", vlm_model="qwen3-vl:4b-instruct-bf16"), {})
        self.assertIsInstance(b, GptBackend)

    def test_same_model_as_fallback_is_not_wrapped(self) -> None:
        s = AlertSettings(vlm_provider="ollama", vlm_model="m", vlm_fallback_provider="ollama", vlm_fallback_model="m")
        with patched_client():
            self.assertIsInstance(inf.make_backend(s, {}), GptBackend)

    def test_nothing_buildable_is_null(self) -> None:
        with self.assertLogs("box.inference", "WARNING"):
            self.assertIsInstance(inf.make_backend(AlertSettings(**self.PAIR), {}), NullBackend)

    def test_settings_read_from_box_yaml(self) -> None:
        s = AlertSettings.from_box_settings(dict(self.PAIR))
        self.assertEqual((s.vlm_provider, s.vlm_model, s.vlm_fallback_provider, s.vlm_fallback_model),
                         ("vllm", "Qwen/Qwen3-VL-4B-Instruct", "vllm", "Qwen/Qwen3.5-4B"))
        d = AlertSettings.from_box_settings({})
        self.assertEqual((d.vlm_provider, d.vlm_model, d.vlm_fallback_provider, d.vlm_fallback_model),
                         ("openai", "gpt-4o", "", ""))


if __name__ == "__main__":
    unittest.main()
