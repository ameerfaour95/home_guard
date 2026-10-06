# Qwen Vision Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure Qwen vision models (Qwen3-VL instruct/thinking, Qwen2.5-VL, Qwen3.5) on our 220 tagged clips with real tokens, latency and cost, and give the box a Qwen primary + Qwen fallback backend (default Qwen3-VL-4B-Instruct → Qwen3.5-4B).

**Architecture:** A small provider table (`box/providers.py`: openai, openrouter, ollama, vllm, dashscope-intl) gives `inference.GptBackend` a `base_url`, key and request extras; a `FallbackBackend` wraps primary + fallback. `eval_prompt.py` gains `--provider`, per-answer tokens/cost/latency, a day/night split and a `compare` command that applies the owner's choice rule. The 4B/3B/7B models run on the laptop GPU through Ollama; larger ones through OpenRouter.

**Tech Stack:** Python 3.12, `openai` SDK (OpenAI-compatible endpoints), Ollama (laptop, RTX 5070 Ti 12 GB), OpenRouter, unittest-style tests run with pytest.

Spec: `docs/superpowers/specs/2026-10-06-qwen-vision-swap-design.md`.

## Global Constraints

- Defaults keep today's behaviour exactly: `vlm_provider: openai`, `vlm_model: gpt-4o`, `vlm_fallback_provider: ""`, `vlm_fallback_model: ""` (no fallback). This plan does not switch the box; hosting the 4B models is decided after the eval.
- gpt-4o is never the fallback. The fallback is the other 4B Qwen.
- Default primary: Qwen3-VL-4B-Instruct, fallback Qwen3.5-4B. Qwen3.5-4B becomes primary (and Qwen3-VL-4B the fallback) only if it beats Qwen3-VL-4B (spec §6).
- An alert is never dropped because of the swap: primary failure → fallback once.
- Secrets only in `api_key.env` at the repo root (gitignored); never printed, never committed.
- Existing tests build `GptBackend` with `__new__` and set only `_model`, `model_name`, `_response_format`, `_client`/`_complete`. New attributes must be read with `getattr(self, name, default)`.
- Old results files have no token columns; every reader must accept their absence.
- Test command (pytest is not in the venv; uv needs system certs on this laptop):
  `uv run --system-certs --with pytest python -m pytest <files> -q`
  Baseline before Task 1: `tests/box/test_inference.py tests/box/test_eval_prompt.py tests/box/test_house_fact_labels.py tests/box/test_inference_crop_parity.py` → 176 passed.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `*.md` is gitignored; add docs with `git add -f`.

## File structure

| File | Responsibility |
|---|---|
| `home_guard_project/box/providers.py` (new) | Provider table, key/URL lookup, price table, cost |
| `home_guard_project/box/inference.py` (modify) | `GptBackend` base_url/extra_body/timeout/usage; `FallbackBackend`; settings fields; `build_gpt`; `make_backend` |
| `home_guard_project/box/eval_prompt.py` (modify) | `--provider`, usage columns, day/night + cost summary, `compare` + choice rule |
| `tests/box/test_providers.py` (new) | providers tests |
| `tests/box/test_vlm_fallback.py` (new) | GptBackend options, FallbackBackend, make_backend |
| `tests/box/test_eval_prompt.py` (modify) | new eval behaviour |
| `home_guard_project/box/README.md` (modify) | how to run the model eval and the box settings |

---

### Task 1: Provider table

**Files:**
- Create: `home_guard_project/box/providers.py`
- Test: `tests/box/test_providers.py`

**Interfaces:**
- Produces:
  - `Provider(name: str, base_url: Optional[str], key_env: Optional[str], extra_body: Optional[Dict[str, Any]] = None, base_url_env: Optional[str] = None, key_required: bool = True)` (frozen dataclass)
  - `PROVIDERS: Dict[str, Provider]`, keys `openai`, `openrouter`, `ollama`, `vllm`, `dashscope-intl`
  - `ProviderError(Exception)`
  - `get(name: str) -> Provider`
  - `resolve(name: str, env: Mapping[str, str]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]`, returning `(api_key, base_url, extra_body)`
  - `model_key(provider: str, model: str) -> str`: bare for openai, else `provider:model`
  - `bare_model(key: str) -> str`: the inverse of `model_key`
  - `price_of(model: str) -> Optional[Tuple[float, float]]` ($/M in, $/M out; bare or keyed ids)
  - `cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]`

- [ ] **Step 1: Write the failing test**

Create `tests/box/test_providers.py`:

```python
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

    def test_ollama_needs_no_key_and_url_can_be_overridden(self) -> None:
        self.assertEqual(pv.resolve("ollama", {})[:2], ("ollama", "http://localhost:11434/v1"))
        self.assertEqual(pv.resolve("ollama", {"OLLAMA_BASE_URL": "http://gpu:11434/v1"})[1], "http://gpu:11434/v1")

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_providers.py -q`
Expected: FAIL with `ImportError: cannot import name 'providers'`.

- [ ] **Step 3: Write the implementation**

Create `home_guard_project/box/providers.py`:

```python
"""Where the vision model is asked and what it costs.

Every provider here speaks the OpenAI ``chat.completions`` API, so one client
(``inference.GptBackend``) reaches all of them with a different base URL and key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: Optional[str]                      # None: the OpenAI SDK default (or base_url_env, then required)
    key_env: Optional[str]                       # the api_key.env variable holding the key
    extra_body: Optional[Dict[str, Any]] = None  # sent with every request
    base_url_env: Optional[str] = None           # overrides base_url when set
    key_required: bool = True


PROVIDERS: Dict[str, Provider] = {
    "openai": Provider("openai", None, "OPENAI_API_KEY"),
    # Hybrid Qwen models think by default; the task does not need it and it costs money and seconds.
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                           {"reasoning": {"enabled": False}}),
    # The laptop GPU (research runs of the 4B/3B/7B models).
    "ollama": Provider("ollama", "http://localhost:11434/v1", None, base_url_env="OLLAMA_BASE_URL",
                       key_required=False),
    # Our own GPU server (stage B): the URL is required, the key optional.
    "vllm": Provider("vllm", None, "VLLM_API_KEY", base_url_env="VLLM_BASE_URL", key_required=False),
    "dashscope-intl": Provider("dashscope-intl", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                               "DASHSCOPE_API_KEY", {"enable_thinking": False}),
}

# $ per million tokens (input, output). OpenRouter list prices on 2026-10-06; OpenAI's own for gpt-4o.
# Only for reports: a model missing here (every local model) is shown with tokens and no dollars.
PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "openai/gpt-4o": (2.50, 10.00),
    "qwen/qwen3.5-9b": (0.10, 0.15),
    "qwen/qwen3-vl-8b-instruct": (0.117, 0.455),
    "qwen/qwen3-vl-8b-thinking": (0.18, 2.10),
    "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
    "qwen/qwen2.5-vl-72b-instruct": (0.80, 1.00),
}


class ProviderError(Exception):
    """Unknown provider, or its key or URL is not set."""


def get(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ProviderError(f"unknown provider {name!r}; known: {', '.join(sorted(PROVIDERS))}") from None


def resolve(name: str, env: Mapping[str, str]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """``(api_key, base_url, extra_body)`` for *name*; raises naming the missing variable."""
    p = get(name)
    url = (str(env.get(p.base_url_env) or "").strip() if p.base_url_env else "") or p.base_url
    if url is None and p.base_url_env:
        raise ProviderError(f"{p.base_url_env} is not set (the server's address, e.g. http://host:8000/v1)")
    key = str(env.get(p.key_env) or "").strip() if p.key_env else ""
    if not key:
        if p.key_required:
            raise ProviderError(f"{p.key_env} is not set (put it in api_key.env at the repo root)")
        key = "ollama" if p.name == "ollama" else "none"   # the SDK wants some key; these servers ignore it
    return key, url, dict(p.extra_body) if p.extra_body else None


def model_key(provider: str, model: str) -> str:
    """How results name a model: bare for openai (older results use that), else ``provider:model``."""
    return model if provider == "openai" else f"{provider}:{model}"


def bare_model(key: str) -> str:
    prefix, sep, rest = key.partition(":")
    return rest if sep and prefix in PROVIDERS else key


def price_of(model: str) -> Optional[Tuple[float, float]]:
    return PRICES.get(bare_model(model))


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    price = price_of(model)
    if price is None:
        return None
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_providers.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add home_guard_project/box/providers.py tests/box/test_providers.py
git commit -m "Box: provider table for the vision model (OpenAI, OpenRouter, Ollama, vLLM, Alibaba) with prices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: GptBackend options, FallbackBackend, settings, make_backend

**Files:**
- Modify: `home_guard_project/box/inference.py`: `AlertSettings` (~lines 418-462), `GptBackend` (~510-568), `make_backend` (~570-587), the worker's teacher record (~1346), the startup log (~1575)
- Test: `tests/box/test_vlm_fallback.py`

**Interfaces:**
- Consumes: `providers.resolve(name, env) -> (api_key, base_url, extra_body)`
- Produces:
  - `GptBackend(api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None, extra_body: Optional[Dict[str, Any]] = None, timeout: float = 30.0)`; attribute `last_usage: Dict[str, int]` with `prompt_tokens`, `completion_tokens`
  - `usage_of(resp: Any) -> Dict[str, int]`
  - `FallbackBackend(primary, fallback)`: `analyze(...)` has the same signature as `GptBackend.analyze`; read-only `model_name`, `last_prompt`, `last_frame_jpegs`, `last_usage` of whichever backend answered last; attributes `primary`, `fallback`
  - `AlertSettings` new fields `vlm_provider: str = "openai"`, `vlm_fallback_provider: str = ""`, `vlm_fallback_model: str = ""`, with box.yaml keys of the same names
  - `build_gpt(provider: str, model: str, env: Mapping[str, str], timeout: float = 30.0) -> GptBackend`

- [ ] **Step 1: Write the failing tests**

First check the logger name: `grep -n 'getLogger' home_guard_project/box/inference.py`. The tests below use `"box.inference"`; if the real name differs, use it in every `assertLogs`.

Create `tests/box/test_vlm_fallback.py`:

```python
"""Tests for the vision model's provider options and the Qwen fallback (box/inference.py)."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import inference as inf
from home_guard_project.box.inference import AlertSettings, FallbackBackend, GptBackend, NullBackend

ANSWER = {"summary": "a person walks", "label": "normal", "people": 1,
          "vehicle_moving": False, "animals": 0, "why": "", "summary_owner": ""}


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
        with mock.patch("openai.OpenAI") as client:
            GptBackend("k", "Qwen/Qwen3-VL-4B-Instruct", base_url="http://pod:8000/v1")
        kw = client.call_args.kwargs
        self.assertEqual((kw["api_key"], kw["base_url"], kw["timeout"]), ("k", "http://pod:8000/v1", 30.0))

    def test_no_base_url_for_openai(self) -> None:
        with mock.patch("openai.OpenAI") as client:
            GptBackend("sk-1", "gpt-4o")
        self.assertNotIn("base_url", client.call_args.kwargs)

    def test_extra_body_sent_and_usage_recorded(self) -> None:
        with mock.patch("openai.OpenAI"):
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
        with mock.patch("openai.OpenAI"):
            b = inf.make_backend(AlertSettings(), {"OPENAI_API_KEY": "sk"})
        self.assertIsInstance(b, GptBackend)
        self.assertEqual(b.model_name, "gpt-4o")

    def test_qwen_pair_wraps_with_fallback(self) -> None:
        with mock.patch("openai.OpenAI") as client:
            b = inf.make_backend(AlertSettings(**self.PAIR), {"VLLM_BASE_URL": "http://pod:8000/v1"})
        self.assertIsInstance(b, FallbackBackend)
        self.assertEqual((b.primary.model_name, b.fallback.model_name),
                         ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3.5-4B"))
        self.assertEqual(client.call_args.kwargs["base_url"], "http://pod:8000/v1")

    def test_missing_primary_uses_fallback_alone(self) -> None:
        s = AlertSettings(vlm_provider="openrouter", vlm_model="qwen/qwen3-vl-8b-instruct",
                          vlm_fallback_provider="ollama", vlm_fallback_model="qwen3.5:4b-bf16")
        with mock.patch("openai.OpenAI"), self.assertLogs("box.inference", "WARNING"):
            b = inf.make_backend(s, {})
        self.assertIsInstance(b, GptBackend)
        self.assertEqual(b.model_name, "qwen3.5:4b-bf16")

    def test_no_fallback_by_default(self) -> None:
        with mock.patch("openai.OpenAI"):
            b = inf.make_backend(AlertSettings(vlm_provider="ollama", vlm_model="qwen3-vl:4b-instruct-bf16"), {})
        self.assertIsInstance(b, GptBackend)

    def test_same_model_as_fallback_is_not_wrapped(self) -> None:
        s = AlertSettings(vlm_provider="ollama", vlm_model="m", vlm_fallback_provider="ollama", vlm_fallback_model="m")
        with mock.patch("openai.OpenAI"):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_vlm_fallback.py -q`
Expected: FAIL with `ImportError: cannot import name 'FallbackBackend'`.

- [ ] **Step 3: Implement in `inference.py`**

3a. Add `from . import providers` next to the other local imports. Add `Mapping` to the `typing` import if it is missing.

3b. `AlertSettings`: after `vlm_model: str = "gpt-4o"` add

```python
    vlm_provider: str = "openai"        # providers.PROVIDERS: openai | openrouter | ollama | vllm | dashscope-intl
    vlm_fallback_provider: str = ""     # asked once when the main model fails (the other 4B Qwen)
    vlm_fallback_model: str = ""        # empty: no fallback
```

and in `from_box_settings`, after `vlm_model=...`:

```python
            vlm_provider=str(g("vlm_provider", "openai") or "openai").strip().lower(),
            vlm_fallback_provider=str(g("vlm_fallback_provider", "") or "").strip().lower(),
            vlm_fallback_model=str(g("vlm_fallback_model", "") or "").strip(),
```

3c. Above `class GptBackend`, add:

```python
def usage_of(resp: Any) -> Dict[str, int]:
    """Tokens the provider billed for one call; zeros when it did not say."""
    u = getattr(resp, "usage", None)
    return {"prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0)}
```

3d. `GptBackend.__init__`: change the signature to

```python
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None,
                 extra_body: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> None:
```

keep the import and the trust-store `http_client` block unchanged, and replace the line
`self._client = OpenAI(api_key=api_key, http_client=http_client) if http_client else OpenAI(api_key=api_key)` with:

```python
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        if http_client:
            kwargs["http_client"] = http_client
        self._client = OpenAI(**kwargs)
```

Then after `self.model_name = model` add:

```python
        self._extra_body = dict(extra_body) if extra_body else None
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
```

3e. In `GptBackend.analyze`, right after `raw = resp.choices[0].message.content or ""`, add `self.last_usage = usage_of(resp)`.

3f. Replace `GptBackend._complete` with:

```python
    def _complete(self, content: List[Dict[str, Any]], response_format: Dict[str, Any]) -> Any:
        kwargs: Dict[str, Any] = dict(model=self._model, messages=[{"role": "user", "content": content}],
                                      temperature=0, response_format=response_format)
        extra = getattr(self, "_extra_body", None)
        if extra:
            kwargs["extra_body"] = extra
        return self._client.chat.completions.create(**kwargs)
```

3g. After `GptBackend`, add:

```python
class FallbackBackend:
    """Asks *primary*; on an error or an answer that is not a JSON object, asks *fallback*
    once with the same frames. Exposes the answering backend's record fields."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        self.primary, self.fallback = primary, fallback
        self._last = primary

    @property
    def model_name(self) -> str:
        return str(getattr(self._last, "model_name", ""))

    @property
    def last_prompt(self) -> str:
        return str(getattr(self._last, "last_prompt", ""))

    @property
    def last_frame_jpegs(self) -> List[bytes]:
        return list(getattr(self._last, "last_frame_jpegs", None) or [])

    @property
    def last_usage(self) -> Dict[str, int]:
        return dict(getattr(self._last, "last_usage", None) or {"prompt_tokens": 0, "completion_tokens": 0})

    def analyze(self, frames_bgr: List[Any], camera_name: str, t_sec: int, start_hour: int, end_hour: int,
                **kwargs: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        self._last = self.primary
        try:
            raw, parsed = self.primary.analyze(frames_bgr, camera_name, t_sec, start_hour, end_hour, **kwargs)
            if isinstance(parsed, dict):
                return raw, parsed
            reason = "the answer was not a JSON object"
        except Exception as exc:  # noqa: BLE001 - any failure goes to the fallback
            reason = f"{type(exc).__name__}: {exc}"
        log.warning("[%s] VLM fallback to %s: %s", camera_name, getattr(self.fallback, "model_name", "?"), reason)
        self._last = self.fallback
        return self.fallback.analyze(frames_bgr, camera_name, t_sec, start_hour, end_hour, **kwargs)
```

3h. Replace `make_backend` with:

```python
def build_gpt(provider: str, model: str, env: Mapping[str, str], timeout: float = 30.0) -> GptBackend:
    key, base_url, extra_body = providers.resolve(provider, env)
    return GptBackend(key, model, base_url=base_url, extra_body=extra_body, timeout=timeout)


def make_backend(settings: AlertSettings, env: Dict[str, str]):
    """The vision model from settings: the main model, wrapped with the fallback when one is set
    and differs; the fallback alone if the main one cannot be built; NullBackend when neither can
    (missing keys, dry-run, unknown backend name)."""
    if settings.dry_run:
        log.info("dry_run on: using NullBackend (no VLM calls).")
        return NullBackend()
    if settings.vlm_backend != "gpt":
        log.warning("Unknown vlm_backend '%s'; using NullBackend.", settings.vlm_backend)
        return NullBackend()
    primary = fallback = None
    try:
        primary = build_gpt(settings.vlm_provider, settings.vlm_model, env)
    except Exception as exc:  # noqa: BLE001
        log.warning("Vision model %s (%s) cannot be used: %s", settings.vlm_model, settings.vlm_provider, exc)
    wants_fallback = bool(settings.vlm_fallback_model) and (
        (settings.vlm_fallback_provider, settings.vlm_fallback_model) != (settings.vlm_provider, settings.vlm_model))
    if wants_fallback:
        try:
            fallback = build_gpt(settings.vlm_fallback_provider or settings.vlm_provider,
                                 settings.vlm_fallback_model, env)
        except Exception as exc:  # noqa: BLE001
            log.warning("Fallback vision model %s (%s) cannot be used: %s",
                        settings.vlm_fallback_model, settings.vlm_fallback_provider, exc)
    if primary is not None and fallback is not None:
        return FallbackBackend(primary, fallback)
    if primary is not None:
        return primary
    if fallback is not None:
        log.warning("Using the fallback %s alone.", settings.vlm_fallback_model)
        return fallback
    log.warning("No vision model could be built; using NullBackend.")
    return NullBackend()
```

3i. In the worker's teacher record (~line 1346), replace

```python
                "frames": list(backend.last_frame_jpegs) if isinstance(backend, GptBackend) else _jpegs(frames),
```

with

```python
                "frames": list(getattr(backend, "last_frame_jpegs", None) or []) or _jpegs(frames),
```

3j. In the startup log (~line 1575), change the `backend=%s` argument from `settings.vlm_backend` to
`f"{settings.vlm_provider}:{settings.vlm_model}" + (f" -> {settings.vlm_fallback_model}" if settings.vlm_fallback_model else "")`.

- [ ] **Step 4: Run new and existing tests**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_vlm_fallback.py tests/box/test_inference.py tests/box/test_eval_prompt.py tests/box/test_house_fact_labels.py tests/box/test_inference_crop_parity.py tests/box/test_quiet_log.py tests/box/test_brain_telegram.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add home_guard_project/box/inference.py tests/box/test_vlm_fallback.py
git commit -m "Box: vision model can come from OpenRouter/Ollama/vLLM, with a second Qwen asked once when it fails; tokens recorded per call

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Eval provider, tokens, cost, latency, day/night

**Files:**
- Modify: `home_guard_project/box/eval_prompt.py`: constants (~lines 75-80), `FakeBackend`, `GptAsker`, `summarize`, `format_summary`, `_answer_row`, `score_rows`, `run_eval`, `_make_gpt`, `main`
- Test: `tests/box/test_eval_prompt.py`

**Interfaces:**
- Consumes: `providers.model_key`, `providers.cost_usd`, `providers.ProviderError`, `inference.build_gpt`, the backend's `last_usage`
- Produces:
  - `USAGE_COLUMNS = ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s")`; `ANSWER_COLUMNS` and `RESULT_COLUMNS` end with it, and `RESULT_COLUMNS` then ends with `"local_time"`
  - `is_night(local_time: Optional[str]) -> Optional[bool]` (19:00–05:59 is night; None when unknown)
  - `summarize(rows)` adds keys `missed_alerts: List[str]`, `false_alarms: List[str]`, `tokens_in_mean`, `tokens_out_mean`, `cost_per_call`, `latency_mean`, `per_month_150`, `per_month_300` (all `Optional[float]`), `day` and `night` (each `{"alerts_caught", "alerts_total", "normal_flagged", "normal_total", "errors"}`), and `unknown_time: int`
  - `_make_gpt(provider: str, model: str) -> GptAsker` (timeout 120 s, because thinking models and a model's first load into the GPU are slow)
  - CLI: `run --provider {dashscope-intl,ollama,openai,openrouter,vllm}` (default `openai`); results `model` = `providers.model_key(provider, model)`

- [ ] **Step 1: Write the failing tests**

Find the fixture class whose tests call `ev.run_eval(self.out, ...)` (`grep -n "self.out =" tests/box/test_eval_prompt.py`). Move its `setUp`/`tearDown` into a mixin `EvalDirMixin` (no test methods) that the existing class and the new `ProviderRunTest` both inherit, so the existing tests are not run twice. Add `from unittest import mock` to the imports if it is missing. `read_jsonl` is the helper already defined in the test file.

Append to `tests/box/test_eval_prompt.py`:

```python
class CostAndTimeTest(unittest.TestCase):
    def rows(self):
        def r(cid, ours, ai, t, pin=2000, pout=80, cost=0.0002, lat=1.5, error=""):
            return {"clip_id": cid, "ours_label": ours, "ai_label": ai, "ai_summary": "x", "ours_text": "y",
                    "local_time": t, "prompt_tokens": pin, "completion_tokens": pout, "cost_usd": cost,
                    "latency_s": lat, "error": error}
        return [r("a1", "alert", "suspicious", "02:10:00"), r("a2", "alert", "normal", "14:00:00"),
                r("n1", "normal", "suspicious", "21:30:00"), r("n2", "normal", "normal", "09:00:00"),
                r("n3", "normal", "normal", None, cost=None, pin=None, pout=None, lat=None),
                r("e1", "normal", "", "10:00:00", error="boom")]

    def test_is_night(self) -> None:
        self.assertTrue(ev.is_night("02:10:00"))
        self.assertTrue(ev.is_night("19:00:00"))
        self.assertFalse(ev.is_night("06:00:00"))
        self.assertIsNone(ev.is_night(None))

    def test_summary_lists_misses_and_false_alarms(self) -> None:
        s = ev.summarize(self.rows())
        self.assertEqual(s["missed_alerts"], ["a2"])
        self.assertEqual(s["false_alarms"], ["n1"])

    def test_day_night_split(self) -> None:
        s = ev.summarize(self.rows())
        self.assertEqual((s["night"]["alerts_caught"], s["night"]["alerts_total"]), (1, 1))
        self.assertEqual((s["night"]["normal_flagged"], s["night"]["normal_total"]), (1, 1))
        self.assertEqual((s["day"]["alerts_caught"], s["day"]["alerts_total"]), (0, 1))
        self.assertEqual(s["day"]["errors"], 1)
        self.assertEqual(s["unknown_time"], 1)

    def test_cost_includes_errored_calls_and_projects_a_month(self) -> None:
        s = ev.summarize(self.rows())
        self.assertAlmostEqual(s["cost_per_call"], 0.0002)
        self.assertAlmostEqual(s["per_month_150"], 0.0002 * 150 * 30)
        self.assertAlmostEqual(s["tokens_in_mean"], 2000)
        self.assertIn("per box per month", ev.format_summary(s))

    def test_old_rows_without_tokens(self) -> None:
        s = ev.summarize([{"clip_id": "a", "ours_label": "normal", "ai_label": "normal", "error": ""}])
        self.assertIsNone(s["cost_per_call"])
        self.assertIn("cost              unknown", ev.format_summary(s))


class ProviderRunTest(EvalDirMixin, unittest.TestCase):
    def test_usage_cost_and_latency_recorded(self) -> None:
        backend = ev.FakeBackend()
        backend.model_name = "openrouter:qwen/qwen3.5-9b"   # a priced model, so cost is filled in
        ev.run_eval(self.out, backend, model="openrouter:qwen/qwen3.5-9b", tag="q", limit=2)
        rows = read_jsonl(os.path.join(self.out, "results", "q.jsonl"))
        self.assertEqual((rows[0]["prompt_tokens"], rows[0]["completion_tokens"]), (1000, 50))
        self.assertAlmostEqual(rows[0]["cost_usd"], (1000 * 0.10 + 50 * 0.15) / 1e6)
        self.assertGreaterEqual(rows[0]["latency_s"], 0)

    def test_local_model_has_tokens_but_no_cost(self) -> None:
        ev.run_eval(self.out, ev.FakeBackend(), model="ollama:qwen3-vl:4b-instruct-bf16", tag="l", limit=1)
        rows = read_jsonl(os.path.join(self.out, "results", "l.jsonl"))
        self.assertEqual(rows[0]["prompt_tokens"], 1000)
        self.assertIsNone(rows[0]["cost_usd"])

    def test_main_provider_needs_its_key(self) -> None:
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
                mock.patch("dotenv.load_dotenv", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                ev.main(["run", "--dir", self.out, "--provider", "openrouter", "--model", "qwen/qwen3.5-9b"])
        self.assertIn("OPENROUTER_API_KEY", str(cm.exception))

    def test_main_provider_tag_names_provider(self) -> None:
        with mock.patch.object(ev, "_make_gpt", return_value=ev.FakeBackend()) as make:
            code = ev.main(["run", "--dir", self.out, "--provider", "ollama", "--model", "qwen3-vl:4b-instruct-bf16",
                            "--limit", "1", "--yes"])
        self.assertEqual(code, 0)
        make.assert_called_once_with("ollama", "qwen3-vl:4b-instruct-bf16")
        tag = ev.default_tag(None, "ollama:qwen3-vl:4b-instruct-bf16")
        rows = read_jsonl(os.path.join(self.out, "results", f"{tag}.jsonl"))
        self.assertEqual(rows[0]["model"], "ollama:qwen3-vl:4b-instruct-bf16")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_eval_prompt.py -q -k "CostAndTime or ProviderRun"`
Expected: FAIL with `AttributeError: module ... has no attribute 'is_night'`.

- [ ] **Step 3: Implement**

3a. Change the import to `from . import inference, providers`.

3b. Replace the two column constants and add the new ones:

```python
USAGE_COLUMNS = ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s")
ANSWER_COLUMNS = ("clip_id", "ai_label", "ai_summary", "ai_people", "ai_animals", "ai_vehicle_moving",
                  "raw", "error", "prompt_id", "prompt_sha12", "model", "input_sha12") + USAGE_COLUMNS
RESULT_COLUMNS = ("clip_id", "camera", "ours_label", "ours_text", "ai_label", "ai_summary", "ai_people", "ai_animals",
                  "ai_vehicle_moving", "raw", "error", "prompt_id", "prompt_sha12", "model", "input_sha12"
                  ) + USAGE_COLUMNS + ("local_time",)
NIGHT_FROM, NIGHT_UNTIL = 19, 6          # local hours: 19:00-05:59 is night
CALLS_PER_DAY = (150, 300)               # a typical and a busy house (plan page §2)
```

3c. `FakeBackend.__init__` gets `self.last_usage = {"prompt_tokens": 1000, "completion_tokens": 50}`.

3d. `GptAsker` gets:

```python
    @property
    def last_usage(self) -> Dict[str, int]:
        return dict(getattr(self.backend, "last_usage", None) or {})
```

3e. Above `summarize`, add:

```python
def is_night(local_time: Optional[str]) -> Optional[bool]:
    if not local_time:
        return None
    try:
        hour = int(str(local_time).split(":")[0])
    except ValueError:
        return None
    return hour >= NIGHT_FROM or hour < NIGHT_UNTIL


def _counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    ok = [r for r in rows if not r.get("error")]
    alerts = [r for r in ok if r.get("ours_label") == "alert"]
    normals = [r for r in ok if r.get("ours_label") == "normal"]
    return {"alerts_caught": sum(r.get("ai_label") in CAUGHT_LABELS for r in alerts), "alerts_total": len(alerts),
            "normal_flagged": sum(r.get("ai_label") != "normal" for r in normals), "normal_total": len(normals),
            "errors": len(rows) - len(ok)}


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None
```

3f. In `summarize`, assign the existing returned dict to `base`, then compute and return `{**base, **extra}`:

```python
    cost = _mean(r.get("cost_usd") for r in rows)          # errored calls were paid for too
    extra = {
        "missed_alerts": sorted(str(r.get("clip_id")) for r in alerts if r not in caught),
        "false_alarms": sorted(str(r.get("clip_id")) for r in flagged),
        "tokens_in_mean": _mean(r.get("prompt_tokens") for r in rows),
        "tokens_out_mean": _mean(r.get("completion_tokens") for r in rows),
        "cost_per_call": cost,
        "latency_mean": _mean(r.get("latency_s") for r in rows),
        "per_month_150": None if cost is None else cost * CALLS_PER_DAY[0] * 30,
        "per_month_300": None if cost is None else cost * CALLS_PER_DAY[1] * 30,
        "day": _counts([r for r in rows if is_night(r.get("local_time")) is False]),
        "night": _counts([r for r in rows if is_night(r.get("local_time")) is True]),
        "unknown_time": sum(is_night(r.get("local_time")) is None for r in rows),
    }
```

3g. `format_summary`: build the existing lines into a list `lines` (head + the current lines + the outdated line), insert these after the `errors` line, and `return "\n".join(line for line in lines if line)`:

```python
def _split_line(name: str, c: Optional[Dict[str, int]]) -> str:
    if not c:
        return ""
    return (f"{name:<18}alerts {c['alerts_caught']}/{c['alerts_total']}, "
            f"normal flagged {c['normal_flagged']}/{c['normal_total']}, errors {c['errors']}")
```

```python
        _split_line("day", s.get("day")),
        _split_line("night", s.get("night")),
        f"missed alerts     {', '.join(s['missed_alerts']) or '-'}" if "missed_alerts" in s else "",
        f"false alarms      {', '.join(s['false_alarms']) or '-'}" if "false_alarms" in s else "",
        (f"tokens per call   in {_num(s.get('tokens_in_mean'))} out {_num(s.get('tokens_out_mean'))}   "
         f"latency {_num(s.get('latency_mean'))} s"),
        ("cost              unknown (local or unpriced model)" if s.get("cost_per_call") is None else
         f"cost              ${s['cost_per_call']:.5f} per call; per box per month "
         f"${s['per_month_150']:.2f} at 150 calls/day, ${s['per_month_300']:.2f} at 300"),
```

3h. `_answer_row` gets two keyword parameters and fills the usage columns:

```python
def _answer_row(clip_id: str, raw: Any, parsed: Any, error: str, prompt_id: str, model: str,
                prompt_sha12: str, input_sha12: str, usage: Optional[Dict[str, int]] = None,
                latency_s: Optional[float] = None) -> Dict[str, Any]:
```

Before the `return`:

```python
    pin = (usage or {}).get("prompt_tokens")
    pout = (usage or {}).get("completion_tokens")
```

and add to the returned dict:

```python
        "prompt_tokens": pin, "completion_tokens": pout,
        "cost_usd": providers.cost_usd(model, pin or 0, pout or 0) if usage else None,
        "latency_s": None if latency_s is None else round(latency_s, 3),
```

3i. In `score_rows`, add `"local_time": _clip_time(m)` to each scored dict.

3j. In the `run_eval` loop, time the call and pass the usage:

```python
                try:
                    frames = _load_frames(out_dir, m["frames"])
                    started = time.monotonic()
                    raw, parsed = backend.ask(m, frames)
                    took = time.monotonic() - started
                    result = _answer_row(clip_id, raw, parsed, "", prompt_id, model, sha12, fingerprints[clip_id],
                                         usage=getattr(backend, "last_usage", None), latency_s=took)
                except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
                    result = _answer_row(clip_id, raw, None, f"{type(exc).__name__}: {exc}", prompt_id, model,
                                         sha12, fingerprints[clip_id])
```

3k. Replace `_make_gpt`:

```python
def _make_gpt(provider: str, model: str) -> Any:
    try:
        import truststore  # noqa: PLC0415

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        from .boxconfig import PROJECT_ROOT  # noqa: PLC0415

        load_dotenv(os.path.join(PROJECT_ROOT, "api_key.env"))
    except Exception:  # noqa: BLE001
        pass
    try:
        backend = inference.build_gpt(provider, model, os.environ, timeout=120.0)
    except providers.ProviderError as exc:
        raise SystemExit(f"{exc}, or use --fake.") from None
    backend.model_name = providers.model_key(provider, model)
    return GptAsker(backend)
```

3l. In `main`, add to `runp`:

```python
    runp.add_argument("--provider", default="openai", choices=sorted(providers.PROVIDERS),
                      help="Where the model is asked: openai, openrouter (OPENROUTER_API_KEY), ollama (the laptop "
                           "GPU, no key), vllm (VLLM_BASE_URL), dashscope-intl (DASHSCOPE_API_KEY).")
```

and change the `run` branch:

```python
        model_id = providers.model_key(args.provider, args.model)
        backend = FakeBackend() if args.fake else _make_gpt(args.provider, args.model)
        model = None if args.fake else model_id
        tag = args.tag or default_tag(prompt_text, "fake" if args.fake else model_id, fake=args.fake)
```

Change the `run` parser's help from "Box: ask the model..." to "Ask the model about every prepared clip and print the score (laptop or box, wherever the provider is reachable)."

- [ ] **Step 4: Run all eval and inference tests**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_eval_prompt.py tests/box/test_vlm_fallback.py tests/box/test_inference.py tests/box/test_providers.py -q`
Expected: all pass. The existing column test (`set(rows[0]) == set(ev.ANSWER_COLUMNS)`) passes because `_answer_row` now writes every column.

- [ ] **Step 5: Commit**

```bash
git add home_guard_project/box/eval_prompt.py tests/box/test_eval_prompt.py
git commit -m "Box eval: any provider, tokens/cost/latency per answer, day/night split, missed alerts and false alarms by name

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Eval `compare` and the choice rule

**Files:**
- Modify: `home_guard_project/box/eval_prompt.py`: a new section before "Command line", and `main`
- Modify: `home_guard_project/box/README.md`: the section "Scoring the AI's prompt against our tags"
- Test: `tests/box/test_eval_prompt.py`

**Interfaces:**
- Consumes: `summarize`, `score_rows`, `current_answers`, `_fingerprints`, `read_jsonl`, `_results_paths`, `_results_lock`, `_lock_path`, `_one_value`, `_num`, `MANIFEST`, `RESULTS_DIR`
- Produces:
  - `SERIOUS_RE` (words for forced entry / climbing in our alert text)
  - `serious_caught(rows) -> set`
  - `beats(cand_rows, base_rows) -> Tuple[bool, str]`
  - `choose_pair(default: Tuple[str, List[Dict]], challenger: Tuple[str, List[Dict]]) -> Tuple[str, str, str]`, returning `(primary, fallback, reason)`
  - `load_scored(out_dir: str, tag: str) -> List[Dict[str, Any]]`
  - `format_compare(default, challenger, others: Dict[str, List[Dict]], reference: Optional[List[Dict]] = None) -> str`
  - CLI: `compare --dir D --default TAG --challenger TAG [--others TAG ...] [--reference TAG]`

The default and the challenger are passed as results files (tags), not hard-coded model ids, because the same model has different ids on Ollama (`qwen3-vl:4b-instruct-bf16`), vLLM (`Qwen/Qwen3-VL-4B-Instruct`) and elsewhere.

- [ ] **Step 1: Write the failing tests**

Append to `tests/box/test_eval_prompt.py`:

```python
def scored(model, caught=(), missed=(), flagged=(), clean=(), errors=0, cost=None, serious=()):
    """Scored rows for one model: alert clips caught/missed, normal clips flagged/clean."""
    def alert(cid, ai):
        text = "A man forces the door. [alert]" if cid in serious else "Loitering. [alert]"
        return {"clip_id": cid, "ours_label": "alert", "ai_label": ai, "model": model, "ours_text": text,
                "cost_usd": cost, "error": ""}
    rows = [alert(c, "suspicious") for c in caught] + [alert(c, "normal") for c in missed]
    rows += [{"clip_id": c, "ours_label": "normal", "ai_label": "suspicious", "model": model, "ours_text": "",
              "cost_usd": cost, "error": ""} for c in flagged]
    rows += [{"clip_id": c, "ours_label": "normal", "ai_label": "normal", "model": model, "ours_text": "",
              "cost_usd": cost, "error": ""} for c in clean]
    rows += [{"clip_id": f"err{i}", "ours_label": "normal", "ai_label": "", "model": model, "ours_text": "",
              "cost_usd": cost, "error": "boom"} for i in range(errors)]
    return rows


VL, Q35 = "ollama:qwen3-vl:4b-instruct-bf16", "ollama:qwen3.5:4b-bf16"


class ChoosePairTest(unittest.TestCase):
    def test_default_kept_when_challenger_is_not_better(self) -> None:
        p, f, _ = ev.choose_pair((VL, scored(VL, caught="ab", missed="c", flagged="x")),
                                 (Q35, scored(Q35, caught="ab", missed="c", flagged="xy")))
        self.assertEqual((p, f), (VL, Q35))

    def test_challenger_wins_on_alerts(self) -> None:
        p, f, why = ev.choose_pair((VL, scored(VL, caught="ab", missed="c", flagged="x")),
                                   (Q35, scored(Q35, caught="abc", flagged="xy")))
        self.assertEqual((p, f), (Q35, VL))
        self.assertIn("more alerts", why)

    def test_challenger_wins_on_false_alarms_at_equal_alerts(self) -> None:
        p, _, _ = ev.choose_pair((VL, scored(VL, caught="ab", flagged="xy")),
                                 (Q35, scored(Q35, caught="ab", flagged="x")))
        self.assertEqual(p, Q35)

    def test_challenger_loses_by_missing_a_forced_entry_the_default_caught(self) -> None:
        d = scored(VL, caught="ab", missed="c", serious="a")
        c = scored(Q35, caught="bcd", missed="a", serious="a")
        p, _, why = ev.choose_pair((VL, d), (Q35, c))
        self.assertEqual(p, VL)
        self.assertIn("forced", why)

    def test_challenger_loses_with_more_errors(self) -> None:
        p, _, _ = ev.choose_pair((VL, scored(VL, caught="ab", missed="c")),
                                 (Q35, scored(Q35, caught="abc", errors=3)))
        self.assertEqual(p, VL)

    def test_compare_names_pair_flags_reference_and_strong_others(self) -> None:
        big = "openrouter:qwen/qwen3-vl-32b-instruct"
        text = ev.format_compare((VL, scored(VL, caught="a", missed="b", flagged="xy")),
                                 (Q35, scored(Q35, caught="a", missed="b", flagged="xyz")),
                                 {big: scored(big, caught="ab", flagged="x", cost=0.0003)},
                                 reference=scored("gpt-4o", caught="ab", flagged="x"))
        self.assertIn("WORSE than gpt-4o", text.splitlines()[0])
        self.assertIn(f"primary: {VL}", text)
        self.assertIn(f"fallback: {Q35}", text)
        self.assertIn(f"{big} beats both 4B models", text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_eval_prompt.py -q -k "ChoosePair"`
Expected: FAIL with `AttributeError: ... 'choose_pair'`.

- [ ] **Step 3: Implement**

Add before `# Command line` in `eval_prompt.py`:

```python
# ----------------------------------------------------------------------------
# Choosing the box's vision model (spec 2026-10-06 §6)
# ----------------------------------------------------------------------------
# Our alert text for a break-in: the one kind of alert a new primary may never newly miss.
SERIOUS_RE = re.compile(r"\b(forc\w*|break\w*|broke|climb\w*|pry\w*|jimm\w*|smash\w*)\b", re.IGNORECASE)


def serious_caught(rows: Sequence[Dict[str, Any]]) -> set:
    return {str(r.get("clip_id")) for r in rows
            if not r.get("error") and r.get("ours_label") == "alert" and r.get("ai_label") in CAUGHT_LABELS
            and SERIOUS_RE.search(str(r.get("ours_text") or ""))}


def beats(cand_rows: Sequence[Dict[str, Any]], base_rows: Sequence[Dict[str, Any]]) -> Tuple[bool, str]:
    """Whether *cand* beats *base*: more alerts caught, or as many with fewer false alarms; never
    by newly missing a forced-entry/climbing alert, never with more errors."""
    c, b = summarize(cand_rows), summarize(base_rows)
    if c["errors"] > b["errors"]:
        return False, f"more errors ({c['errors']} vs {b['errors']})"
    lost = sorted(serious_caught(base_rows) - serious_caught(cand_rows))
    if lost:
        return False, f"misses forced-entry/climbing clip(s) the other caught: {', '.join(lost)}"
    if c["alerts_caught"] > b["alerts_caught"]:
        return True, f"catches more alerts ({c['alerts_caught']} vs {b['alerts_caught']})"
    if c["alerts_caught"] == b["alerts_caught"] and c["normal_flagged"] < b["normal_flagged"]:
        return True, f"same alerts, fewer false alarms ({c['normal_flagged']} vs {b['normal_flagged']})"
    return False, (f"not better (alerts {c['alerts_caught']} vs {b['alerts_caught']}, "
                   f"false alarms {c['normal_flagged']} vs {b['normal_flagged']})")


def choose_pair(default: Tuple[str, List[Dict[str, Any]]],
                challenger: Tuple[str, List[Dict[str, Any]]]) -> Tuple[str, str, str]:
    """``(primary, fallback, reason)``: the default stays primary unless the challenger beats it;
    the other one is the fallback."""
    ok, why = beats(challenger[1], default[1])
    if ok:
        return challenger[0], default[0], f"{challenger[0]} {why}"
    return default[0], challenger[0], f"{challenger[0]} {why}"


def _money(v: Optional[float], fmt: str) -> str:
    return "-" if v is None else fmt.format(v)


def format_compare(default: Tuple[str, List[Dict[str, Any]]], challenger: Tuple[str, List[Dict[str, Any]]],
                   others: Dict[str, List[Dict[str, Any]]],
                   reference: Optional[List[Dict[str, Any]]] = None) -> str:
    primary, fallback, why = choose_pair(default, challenger)
    rows_of = {default[0]: default[1], challenger[0]: challenger[1], **others}
    c = summarize(rows_of[primary])
    lines: List[str] = []
    if reference is not None:
        r = summarize(reference)
        if c["alerts_caught"] < r["alerts_caught"] or c["normal_flagged"] > r["normal_flagged"]:
            lines.append(f"NOTE: {primary} is WORSE than gpt-4o on this set (alerts {c['alerts_caught']} vs "
                         f"{r['alerts_caught']}, false alarms {c['normal_flagged']} vs {r['normal_flagged']}); "
                         f"misses: {', '.join(c['missed_alerts']) or '-'}; false alarms: "
                         f"{', '.join(c['false_alarms']) or '-'}")
        else:
            lines.append(f"{primary} is no worse than gpt-4o on this set.")
    lines.append(f"{'model':44} {'alerts':>7} {'false':>7} {'err':>4} {'night alerts':>12} {'tok in':>7} "
                 f"{'sec':>5} {'$/call':>9} {'$/mo@150':>9}")
    table = dict(rows_of)
    if reference is not None:
        table["gpt-4o (reference)"] = reference
    for m, rows in table.items():
        s = summarize(rows)
        night = f"{s['night']['alerts_caught']}/{s['night']['alerts_total']}"
        lines.append(f"{m:44} {s['alerts_caught']:>3}/{s['alerts_total']:<3} "
                     f"{s['normal_flagged']:>3}/{s['normal_total']:<3} {s['errors']:>4} {night:>12} "
                     f"{_num(s['tokens_in_mean']):>7} {_num(s['latency_mean']):>5} "
                     f"{_money(s['cost_per_call'], '${:.5f}'):>9} {_money(s['per_month_150'], '${:.2f}'):>9}")
    lines += [f"primary: {primary}", f"fallback: {fallback}", f"  {why}"]
    for m, rows in others.items():
        if beats(rows, default[1])[0] and beats(rows, challenger[1])[0]:
            lines.append(f"  recommendation: {m} beats both 4B models (owner's call; not chosen automatically)")
    lines.append("  (18 alerts is a small set: a one-clip difference is noise)")
    return "\n".join(lines)


def load_scored(out_dir: str, tag: str) -> List[Dict[str, Any]]:
    """Scored rows of results file *tag*: the latest answer per clip next to the current manifest."""
    jsonl_path, _, _ = _results_paths(out_dir, tag)
    with _results_lock(_lock_path(out_dir, tag)):
        manifest = read_jsonl(os.path.join(out_dir, MANIFEST))
        fingerprints = _fingerprints(out_dir, manifest)
        latest = current_answers(read_jsonl(jsonl_path), fingerprints)
        scored, _ = score_rows(manifest, latest, fingerprints)
    return scored
```

In `main`, after the `summ` parser:

```python
    comp = sub.add_parser("compare", help="One table across results files, and the box's primary and fallback model.")
    comp.add_argument("--dir", required=True)
    comp.add_argument("--default", required=True, help="Results file of Qwen3-VL-4B-Instruct (the default primary).")
    comp.add_argument("--challenger", required=True, help="Results file of Qwen3.5-4B.")
    comp.add_argument("--others", nargs="*", default=[], help="Results files of the other models, reported only.")
    comp.add_argument("--reference", default=None, help="Results file of gpt-4o on the same prompt.")
```

and before the existing `summary` handling (`if not os.path.isfile(_results_paths(args.dir, args.tag)[0])`):

```python
    if args.command == "compare":
        tags = [args.default, args.challenger, *args.others] + ([args.reference] if args.reference else [])
        for tag in tags:
            if not os.path.isfile(_results_paths(args.dir, tag)[0]):
                print(f"Error: no results named {tag} in {os.path.join(args.dir, RESULTS_DIR)}", file=sys.stderr)
                return 1

        def named(tag: str) -> Tuple[str, List[Dict[str, Any]]]:
            rows = load_scored(args.dir, tag)
            return (_one_value(rows, "model") or tag), rows

        others = dict(named(t) for t in args.others)
        reference = load_scored(args.dir, args.reference) if args.reference else None
        print(format_compare(named(args.default), named(args.challenger), others, reference))
        return 0
```

README: add to the section "Scoring the AI's prompt against our tags":

```markdown
### Comparing vision models

Local models run on the laptop GPU through Ollama (no key; `ollama pull <model>` first); hosted ones through OpenRouter (`OPENROUTER_API_KEY` in `api_key.env`).

    .venv\Scripts\python.exe -m home_guard_project.box.eval_prompt run --dir <eval_set> --provider ollama --model qwen3-vl:4b-instruct-bf16 --yes
    .venv\Scripts\python.exe -m home_guard_project.box.eval_prompt run --dir <eval_set> --provider openrouter --model qwen/qwen3-vl-32b-instruct --yes
    .venv\Scripts\python.exe -m home_guard_project.box.eval_prompt compare --dir <eval_set> --default <Qwen3-VL-4B tag> --challenger <Qwen3.5-4B tag> --others <tags...> --reference <gpt-4o tag>

Each answer stores tokens, latency and, for hosted models, cost. The summary shows day/night, missed alerts and false alarms by clip, and $ per box per month. `compare` names the box's primary and fallback: Qwen3-VL-4B-Instruct primary and Qwen3.5-4B fallback, swapped only if Qwen3.5-4B beats it.

### The box's vision model settings (box.yaml)

    vlm_provider: vllm                     # openai | openrouter | ollama | vllm | dashscope-intl
    vlm_model: Qwen/Qwen3-VL-4B-Instruct
    vlm_fallback_provider: vllm            # empty: no fallback
    vlm_fallback_model: Qwen/Qwen3.5-4B

`vllm` needs `VLLM_BASE_URL` (and `VLLM_API_KEY` if the server has one) in `api_key.env`. If the main model fails, the same alert goes to the fallback once (`VLM fallback` in the log). Without these settings the box keeps gpt-4o.
```

- [ ] **Step 4: Run tests**

Run: `uv run --system-certs --with pytest python -m pytest tests/box/test_eval_prompt.py tests/box/test_vlm_fallback.py tests/box/test_providers.py tests/box/test_inference.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add home_guard_project/box/eval_prompt.py tests/box/test_eval_prompt.py
git add -f home_guard_project/box/README.md
git commit -m "Box eval: compare models and name the box's primary and fallback (Qwen3-VL-4B, Qwen3.5-4B)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Run the eval

Operational, with no new code. Do the steps in order and stop at the first failure.

- [ ] **Step 1: Keep the eval set somewhere durable**

```bash
mkdir -p /c/Users/ameer/Ameer/home_guard_eval
cp -r "/c/Users/ameer/AppData/Local/Temp/claude/C--Users-ameer-Ameer-home-guard/7a43598a-6e5e-48cb-909b-165aa43912a1/scratchpad/eval_set" /c/Users/ameer/Ameer/home_guard_eval/
wc -l /c/Users/ameer/Ameer/home_guard_eval/eval_set/manifest.jsonl
```
Expected: `220`.

- [ ] **Step 2: Fake run end to end**

```bash
cd /c/Users/ameer/Ameer/home_guard
.venv/Scripts/python.exe -m home_guard_project.box.eval_prompt run --dir /c/Users/ameer/Ameer/home_guard_eval/eval_set --fake --limit 10
```
Expected: a summary with `tokens per call   in 1000.0 out 50.0` and `cost              unknown`.

- [ ] **Step 3: Install Ollama and pull the local models**

```bash
winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
ollama --version
for m in qwen3-vl:4b-instruct-bf16 qwen3.5:4b-bf16 qwen3-vl:4b-thinking-bf16 qwen2.5vl:3b-fp16 qwen2.5vl:7b-q8_0 qwen3-vl:8b-instruct-q8_0; do ollama pull $m; done
ollama list
```
Expected: six models listed (about 45 GB; the disk has 1.6 TB free). If `ollama` is not on PATH after the install, use `"$LOCALAPPDATA/Programs/Ollama/ollama.exe"`.

- [ ] **Step 4: Smoke each local model on 3 clips**

For each model above:

```bash
.venv/Scripts/python.exe -m home_guard_project.box.eval_prompt run --dir /c/Users/ameer/Ameer/home_guard_eval/eval_set --provider ollama --model <model> --limit 3 --yes
```
Expected: 0 errors, `tokens per call in` > 0, latency shown. Check that `qwen3.5:4b-bf16` gives short JSON with no long thinking (latency close to Qwen3-VL-4B's, `completion_tokens` under ~200). If it thinks, give the `ollama` provider `extra_body={"think": False}` (Ollama's own switch, accepted as an extra body field) in `providers.py` with a test, commit, and repeat this step. A model that errors on all 3 clips is dropped and noted.

- [ ] **Step 5: Full local runs**

Run the same command without `--limit`, the default and the challenger first (`qwen3-vl:4b-instruct-bf16`, `qwen3.5:4b-bf16`), then the other four, one after another in the background. Ollama keeps one model loaded at a time.

- [ ] **Step 6: Hosted runs (need OpenRouter credit)**

Check the credit without printing the key:

```bash
K=$(grep '^OPENROUTER_API_KEY=' api_key.env | cut -d= -f2-); curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $K"
```
If `total_credits` is 0, skip this step and say so in the report. Otherwise smoke (`--limit 3`), then run in full: `qwen/qwen3-vl-8b-instruct`, `qwen/qwen3-vl-8b-thinking`, `qwen/qwen3-vl-32b-instruct`, `qwen/qwen3.5-9b`, `qwen/qwen2.5-vl-72b-instruct`, and `openai/gpt-4o` as the reference on today's prompt. Expected spend: about $4 (the thinking model's output tokens can add a little).

- [ ] **Step 7: Compare**

```bash
ls /c/Users/ameer/Ameer/home_guard_eval/eval_set/results/*.jsonl
.venv/Scripts/python.exe -m home_guard_project.box.eval_prompt compare --dir /c/Users/ameer/Ameer/home_guard_eval/eval_set \
  --default <tag ending _ollama_qwen3-vl_4b-instruct-bf16> --challenger <tag ending _ollama_qwen3.5_4b-bf16> \
  --others <every other model tag> --reference <tag ending _openrouter_openai_gpt-4o, if it was run>
```
Save the output to the scratchpad. Read the chosen primary's missed alerts and false alarms (the `.csv` of its tag) and note patterns (night, one camera, clothing words).

- [ ] **Step 8: Report and record**

- Report to the owner:
  - the table, and the primary + fallback
  - the 4B models against gpt-4o
  - Qwen3-VL instruct vs thinking, and Qwen2.5-VL vs Qwen3-VL
  - the 8B quantization check (Ollama q8_0 vs OpenRouter)
  - the hosting options for the box from the spec (rented GPU with vLLM, Featherless, the laptop over Tailscale), with the measured tokens and latency
- Update memory (`project_eval_prompt_tool.md`, `project_vlm_choice_qwen35_and_agents.md`) and the plan page §2/§13 (week 0 item 1: measured; the box switch waits for hosting).
- This plan does not switch the box.
