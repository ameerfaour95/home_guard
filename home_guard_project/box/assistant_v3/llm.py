"""The v3 assistant's model client: one OpenAI-compatible endpoint (OpenRouter by default), JSON-schema output,
reasoning effort, prompt caching, and the usage ledger.

Same ``chat(messages, tools, tool_choice=None, **opts) -> ModelMessage`` shape as ``brain/models.py`` so the eval's
rate gate and spend counter (eval_chat.CountingModel) wrap it unchanged. The extra options:

- ``json_schema``: a JSON schema the answer must follow (``response_format`` json_schema, strict);
- ``max_tokens``; ``reasoning``: "none" | "low" | "medium" (OpenRouter's unified reasoning effort);
- ``cache``: True marks the first system message for prompt caching. OpenAI, Gemini and DeepSeek cache by
  themselves; Anthropic and Qwen need an explicit ``cache_control`` breakpoint (OpenRouter docs, prompt caching), which
  is added only for those model families.

Never raises: a failed call is a ``ModelMessage`` with ``error`` set.
"""
from __future__ import annotations

import json
import logging
import ssl
import time
from typing import Any, Dict, List, Optional

from ..brain.models import ModelMessage, ToolCall, _arguments, _call_identity, _public, _tokens

log = logging.getLogger("box.assistant_v3.llm")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# $ per million tokens (input, output, cached input), OpenRouter list prices read 2026-10-10.
PRICES: Dict[str, tuple] = {
    "anthropic/claude-haiku-5.5": (0.10, 0.50, 0.01),
    "openai/gpt-6-luna": (0.10, 0.50, 0.01),
    "qwen/qwen3.7-flash": (0.03, 0.13, 0.006),
    "openai/gpt-4o": (2.50, 10.00, 1.25),
    "openai/gpt-4o-mini": (0.15, 0.60, 0.075),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00, 0.10),
    "openai/gpt-6-sol": (2.00, 10.00, 0.20),
    "openai/gpt-4.1": (2.00, 8.00, 0.50),
    "qwen/qwen3.8-omni-flash": (0.15, 0.47, 0.016),
}
_NEEDS_BREAKPOINT = ("anthropic/", "qwen/")
_NO_TEMPERATURE = ("anthropic/claude-haiku-5", "anthropic/claude-sonnet-5", "openai/gpt-6", "openai/o")


def usd(model: str, prompt: int, completion: int, cached: int = 0) -> float:
    a, b, c = PRICES.get(model.split(":", 1)[-1], (2.5, 10.0, 1.25))
    return (max(0, prompt - cached) * a + cached * c + completion * b) / 1e6


class V3Chat:
    """One model on an OpenAI-compatible endpoint."""

    def __init__(self, client: Any, model_name: str, usage_agent: str = "brain_v3", reasoning: str = "none",
                 temperature: Optional[float] = 0.2, timeout: float = 30.0) -> None:
        self._client = client
        self.model_name = model_name
        self.usage_agent = usage_agent
        self.reasoning = reasoning
        self.temperature = None if model_name.startswith(_NO_TEMPERATURE) else temperature
        self.timeout = timeout
        self.last_cached = 0

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None,
             max_tokens: int = 600, reasoning: Optional[str] = None, cache: bool = True) -> ModelMessage:
        try:
            return self._chat(messages, tools, tool_choice, json_schema, max_tokens, reasoning, cache)
        except Exception as exc:  # noqa: BLE001 - never into the Telegram loop
            log.warning("%s failed: %s", self.model_name, exc)
            return ModelMessage(error=f"{type(exc).__name__}: {exc}"[:300])

    def _chat(self, messages, tools, tool_choice, json_schema, max_tokens, reasoning, cache) -> ModelMessage:
        msgs = [_public(m) for m in messages]
        if cache and self.model_name.startswith(_NEEDS_BREAKPOINT) and msgs and msgs[0].get("role") == "system":
            text = msgs[0].get("content")
            if isinstance(text, str) and len(text) > 1500:     # below the provider's minimum it is ignored anyway
                msgs[0] = {"role": "system", "content": [{"type": "text", "text": text,
                                                           "cache_control": {"type": "ephemeral"}}]}
        kwargs: Dict[str, Any] = {"model": self.model_name, "messages": msgs, "max_tokens": max_tokens,
                                  "timeout": self.timeout}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "none" if tool_choice == "none" else "auto"
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "out", "strict": True, "schema": json_schema}}
        effort = reasoning or self.reasoning
        extra: Dict[str, Any] = {"usage": {"include": True}}
        if effort:
            extra["reasoning"] = {"effort": effort, "exclude": True}
        kwargs["extra_body"] = extra
        from .. import usage_ledger  # noqa: PLC0415

        started = time.monotonic()
        resp = usage_ledger.call(self.usage_agent, lambda: self._client.chat.completions.create(**kwargs),
                                 client=self._client, model=self.model_name)
        choice = resp.choices[0]
        msg = choice.message
        truncated = getattr(choice, "finish_reason", None) == "length"
        calls: List[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            if not _call_identity(tc.id, tc.function.name):
                continue
            raw = tc.function.arguments if isinstance(tc.function.arguments, str) else ""
            try:
                args = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                args = {}
            args, ok = _arguments(args)
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw,
                                  valid=ok and not truncated))
        u = getattr(resp, "usage", None)
        details = getattr(u, "prompt_tokens_details", None)
        self.last_cached = _tokens(getattr(details, "cached_tokens", 0)) if details is not None else 0
        content = getattr(msg, "content", None)
        log.debug("%s %.1fs in=%s out=%s cached=%s", self.model_name, time.monotonic() - started,
                  getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0), self.last_cached)
        return ModelMessage(content=content if isinstance(content, str) else None, tool_calls=tuple(calls),
                            usage=(_tokens(getattr(u, "prompt_tokens", 0)), _tokens(getattr(u, "completion_tokens", 0))),
                            refused=bool(getattr(msg, "refusal", None)),
                            error="truncated" if truncated and json_schema is not None else "")


def make_v3_model(spec: str, env: Dict[str, str], usage_agent: str = "brain_v3", reasoning: str = "none",
                  temperature: Optional[float] = 0.2) -> Optional[V3Chat]:
    """``"openrouter:anthropic/claude-haiku-5.5"`` (a bare ``vendor/model`` means OpenRouter) -> a V3Chat, or None
    without the provider's key."""
    spec = str(spec or "").strip()
    if not spec:
        return None
    provider, sep, name = spec.partition(":")
    if not sep:
        provider, name = "openrouter", spec
    from ..providers import PROVIDERS  # noqa: PLC0415

    known = PROVIDERS.get(provider.lower())
    if known is None:
        log.warning("v3: unknown provider %s", provider)
        return None
    key = env.get(known.key_env or "", "") if known.key_env else ""
    if known.key_required and not str(key or "").strip():
        return None
    base_url = (env.get(known.base_url_env) if known.base_url_env else None) or known.base_url or OPENROUTER_URL
    import httpx  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=key or "none", base_url=base_url,
                    http_client=httpx.Client(verify=ssl.create_default_context(), timeout=45.0), max_retries=1)
    return V3Chat(client, name.strip(), usage_agent=usage_agent, reasoning=reasoning, temperature=temperature)


def parse_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The first JSON object in *text* (models sometimes wrap it in prose or a code fence)."""
    raw = str(text or "")
    start, end = raw.find("{"), raw.rfind("}")
    if not 0 <= start < end:
        return None
    try:
        value = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
