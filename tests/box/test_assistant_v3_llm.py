"""The v3 model client against endpoints that refuse "reasoning off" (2026-10-20: the escalation model,
anthropic/claude-sonnet-5.5, answered every call with 400 "Reasoning is mandatory for this endpoint and cannot be
disabled", so each angry repair fell back to the small writer)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from home_guard_project.box.assistant_v3.llm import V3Chat


def _response(text: str) -> Any:
    msg = SimpleNamespace(content=text, tool_calls=None, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None))


class FakeClient:
    def __init__(self, refuse_off: bool) -> None:
        self.refuse_off = refuse_off
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        effort = ((kwargs.get("extra_body") or {}).get("reasoning") or {}).get("effort")
        if self.refuse_off and effort in (None, "none"):
            raise RuntimeError("Error code: 400 - {'error': {'message': 'Reasoning is mandatory for this endpoint "
                               "and cannot be disabled.'}}")
        return _response("שלום")


class ReasoningFloorTest(unittest.TestCase):
    def test_sonnet_is_asked_with_low_effort_and_room_for_its_thinking(self) -> None:
        client = FakeClient(refuse_off=True)
        msg = V3Chat(client, "anthropic/claude-sonnet-5.5").chat([{"role": "user", "content": "x"}], max_tokens=400)
        self.assertEqual(msg.error, "")
        self.assertEqual(msg.content, "שלום")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["extra_body"]["reasoning"]["effort"], "low")
        self.assertGreaterEqual(client.calls[0]["max_tokens"], 2000)

    def test_an_unknown_model_that_refuses_is_asked_again_with_low_effort(self) -> None:
        client = FakeClient(refuse_off=True)
        msg = V3Chat(client, "vendor/new-thinker").chat([{"role": "user", "content": "x"}], max_tokens=300)
        self.assertEqual(msg.content, "שלום")
        self.assertEqual([c["extra_body"]["reasoning"]["effort"] for c in client.calls], ["none", "low"])

    def test_the_writer_keeps_reasoning_off_and_its_budget(self) -> None:
        client = FakeClient(refuse_off=False)
        V3Chat(client, "openai/gpt-6-luna").chat([{"role": "user", "content": "x"}], max_tokens=300)
        self.assertEqual(client.calls[0]["extra_body"]["reasoning"]["effort"], "none")
        self.assertEqual(client.calls[0]["max_tokens"], 300)

    def test_another_error_is_not_retried(self) -> None:
        client = FakeClient(refuse_off=False)

        def boom(**kwargs: Any) -> Any:
            client.calls.append(kwargs)
            raise RuntimeError("timeout")

        client.chat.completions.create = boom
        msg = V3Chat(client, "openai/gpt-6-luna").chat([{"role": "user", "content": "x"}])
        self.assertIn("timeout", msg.error)
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
