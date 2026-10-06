"""Tests for box/providers.py: where the vision model is asked, and what it costs."""
from __future__ import annotations

import unittest

from home_guard_project.box import providers as pv


class ProvidersTest(unittest.TestCase):
    def test_openai_uses_sdk_default_url(self) -> None:
        self.assertEqual(pv.resolve("openai", {"OPENAI_API_KEY": " sk-1 "}), ("sk-1", None, None))

    def test_openrouter_url_key_and_reasoning_off(self) -> None:
        key, url, extra = pv.resolve("openrouter", {"OPENROUTER_API_KEY": "or-1"})
        self.assertEqual((key, url), ("or-1", "https://openrouter.ai/api/v1"))
        self.assertEqual(extra, {"reasoning": {"enabled": False}})

    def test_thinking_models_keep_their_reasoning(self) -> None:
        env = {"OPENROUTER_API_KEY": "or-1"}
        self.assertIsNone(pv.resolve("openrouter", env, "qwen/qwen3-vl-8b-thinking")[2])
        self.assertEqual(pv.resolve("openrouter", env, "qwen/qwen3-vl-8b-instruct")[2], {"reasoning": {"enabled": False}})

    def test_ollama_needs_no_key_and_url_can_be_overridden(self) -> None:
        self.assertEqual(pv.resolve("ollama", {})[:2], ("ollama", "http://localhost:11434/v1"))
        self.assertEqual(pv.resolve("ollama", {"OLLAMA_BASE_URL": "http://gpu:11434/v1"})[1], "http://gpu:11434/v1")
        self.assertEqual(pv.resolve("ollama", {}, "qwen3.5:4b-bf16")[2], {"reasoning_effort": "none"})
        self.assertIsNone(pv.resolve("ollama", {}, "qwen3-vl:4b-thinking-bf16")[2])

    def test_vllm_needs_a_url_key_optional(self) -> None:
        with self.assertRaises(pv.ProviderError) as cm:
            pv.resolve("vllm", {})
        self.assertIn("VLLM_BASE_URL", str(cm.exception))
        self.assertEqual(pv.resolve("vllm", {"VLLM_BASE_URL": "http://pod:8000/v1"})[:2], ("none", "http://pod:8000/v1"))
        self.assertEqual(pv.resolve("vllm", {"VLLM_BASE_URL": "u", "VLLM_API_KEY": "k"})[0], "k")

    def test_dashscope_thinking_off(self) -> None:
        _, url, extra = pv.resolve("dashscope-intl", {"DASHSCOPE_API_KEY": "d"})
        self.assertEqual(url, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(extra, {"enable_thinking": False})

    def test_missing_key_names_the_variable(self) -> None:
        with self.assertRaises(pv.ProviderError) as cm:
            pv.resolve("openrouter", {"OPENROUTER_API_KEY": "  "})
        self.assertIn("OPENROUTER_API_KEY", str(cm.exception))

    def test_unknown_provider_lists_known_ones(self) -> None:
        with self.assertRaises(pv.ProviderError) as cm:
            pv.get("together")
        self.assertIn("ollama", str(cm.exception))

    def test_model_key_round_trip(self) -> None:
        self.assertEqual(pv.model_key("openai", "gpt-4o"), "gpt-4o")
        self.assertEqual(pv.model_key("ollama", "qwen3-vl:4b-instruct-bf16"), "ollama:qwen3-vl:4b-instruct-bf16")
        self.assertEqual(pv.bare_model("ollama:qwen3-vl:4b-instruct-bf16"), "qwen3-vl:4b-instruct-bf16")
        self.assertEqual(pv.bare_model("gpt-4o"), "gpt-4o")
        self.assertEqual(pv.bare_model("openai/gpt-6-luna:batch"), "openai/gpt-6-luna:batch")

    def test_cost(self) -> None:
        self.assertEqual(pv.price_of("openrouter:qwen/qwen3.5-9b"), (0.10, 0.15))
        self.assertAlmostEqual(pv.cost_usd("gpt-4o", 1_000_000, 100_000), 2.50 + 1.00)
        self.assertIsNone(pv.cost_usd("ollama:qwen3.5:4b-bf16", 10, 10))   # local: not billed


if __name__ == "__main__":
    unittest.main()
