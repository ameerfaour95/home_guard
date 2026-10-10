# home_guard_project/box/brain/models.py
"""The chat model behind the assistant, behind one interface: ``chat(messages, tools, tool_choice) -> ModelMessage``.

Messages and tools use the OpenAI chat format inside the agent. ``OpenAIChat``
serves OpenAI and Gemini (Gemini's OpenAI-compatible endpoint); ``AnthropicChat``
converts to Anthropic's Messages format with the official SDK. Which model runs
is one setting, ``"<provider>:<model>"``, so the eval can compare providers on
the same cases. Forced tool choice is never used (Claude Opus 5.5 and Sonnet 5.5
reject it): the agent asks for ``reply`` in the prompt and accepts plain text.
"""

from __future__ import annotations

import json
import logging
import ssl
import time
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("box.brain.models")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")    # reject a non-default temperature


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""
    valid: bool = True


@dataclass(frozen=True)
class ModelMessage:
    content: Optional[str] = None
    tool_calls: Tuple[ToolCall, ...] = ()
    raw: Any = None                      # the provider's own assistant content, replayed unchanged
    usage: Tuple[int, int] = (0, 0)      # (input tokens, output tokens)
    refused: bool = False
    error: str = ""                     # set when the call itself failed (not when the model said nothing)


@lru_cache(maxsize=32)
def _warn_once(message: str) -> None:
    log.warning(message)


_WARN_EVERY = 300.0
_last_warned: Dict[Tuple[str, str], float] = {}


def _log_failure(where: str, exc: BaseException) -> None:
    """Log with the traceback, at most once per (function, exception type) per 300 s."""
    key = (where, type(exc).__name__)
    now = time.monotonic()
    last = _last_warned.get(key)
    if last is not None and now - last < _WARN_EVERY:
        return
    _last_warned[key] = now
    log.warning("%s failed (%s: %s); ignoring this model result", where, type(exc).__name__, exc, exc_info=True)


def _safe(fallback):
    """Contain malformed data and SDK failures at the poll-loop boundary."""
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - never escape into the Telegram loop
                _log_failure(func.__qualname__, exc)
                if fallback is ModelMessage:
                    return ModelMessage(error=f"{type(exc).__name__}: {exc}"[:300])
                return fallback()
        return wrapped
    return decorate


def _tokens(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        _warn_once("Invalid model usage; using zero")
        return 0


def _arguments(value: Any) -> Tuple[Dict[str, Any], bool]:
    try:
        if not isinstance(value, dict):
            raise ValueError("Arguments must be an object")
        # Reject NaN/Infinity (including exponent overflow), cycles and objects.
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        return dict(value), True
    except (TypeError, ValueError, OverflowError, RecursionError):
        _warn_once("Invalid tool arguments; disabling the tool call")
        return {}, False


def _call_identity(call_id: Any, name: Any) -> bool:
    if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
        return True
    _warn_once("Invalid tool identity; skipping the tool call")
    return False


def _public(message: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in message.items() if not k.startswith("_")}


class OpenAIChat:
    def __init__(self, client: Any, model_name: str, temperature: Optional[float] = 0.0,
                 extra_body: Optional[Dict[str, Any]] = None) -> None:
        self._client = client
        self.model_name = model_name
        self._temperature = temperature
        self._extra_body = dict(extra_body) if extra_body else None
        self.usage_agent = "brain"           # the usage ledger's agent (brain/agent.py: brain_fast for the fast one)

    @_safe(ModelMessage)
    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
             tool_choice: Optional[str] = None) -> ModelMessage:
        # Capped: without it OpenRouter reserves the model's whole output limit against the balance (402 on low credit).
        kwargs: Dict[str, Any] = {"model": self.model_name, "messages": [_public(m) for m in messages],
                                  "max_tokens": 2000}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "none" if tool_choice == "none" else "auto"
        if getattr(self, "_extra_body", None):
            kwargs["extra_body"] = self._extra_body
        from .. import usage_ledger  # noqa: PLC0415

        resp = usage_ledger.call(getattr(self, "usage_agent", "brain"),
                                 lambda: self._client.chat.completions.create(**kwargs), client=self._client,
                                 model=self.model_name, images=usage_ledger.images_in(kwargs["messages"]))
        choice = resp.choices[0]
        truncated = getattr(choice, "finish_reason", None) == "length"
        msg = choice.message
        calls: List[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            if not _call_identity(tc.id, tc.function.name):
                continue
            raw = tc.function.arguments
            valid = not truncated and isinstance(raw, str)
            if not isinstance(raw, str):
                _warn_once("Invalid tool arguments; disabling the tool call")
                raw = ""
            try:
                args = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                _warn_once("Invalid tool arguments; disabling the tool call")
                args, valid = {}, False
            args, args_valid = _arguments(args)
            valid = valid and args_valid
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw, valid=valid))
        usage = getattr(resp, "usage", None)
        content = getattr(msg, "content", None)
        if content is not None and not isinstance(content, str):
            _warn_once("Invalid model text; ignoring it")
            content = None
        return ModelMessage(
            content=content, tool_calls=tuple(calls),
            usage=(_tokens(getattr(usage, "prompt_tokens", 0)), _tokens(getattr(usage, "completion_tokens", 0))),
            refused=bool(getattr(msg, "refusal", None)),
        )


def _anthropic_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool.get("function") or {}
    return {"name": fn.get("name", ""), "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}}


@_safe(lambda: ("", []))
def to_anthropic_messages(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """``(system text, messages)`` in Anthropic's format; consecutive tool results share one user message."""
    clean = []
    for message in messages:
        if isinstance(message, dict):
            clean.append(message)
        else:
            _warn_once("Invalid history entry; skipping it")
    messages = clean
    system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            content = m.get("content")
            if content:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            if m.get("_raw") is not None and m.get("_raw") != []:
                out.append({"role": "assistant", "content": m["_raw"]})
                continue
            blocks: List[Dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for call in m.get("tool_calls") or []:
                if (not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                        or not isinstance(call.get("id"), str)
                        or not isinstance(call["function"].get("name"), str)):
                    _warn_once("Invalid history tool call; skipping it")
                    continue
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except (ValueError, TypeError):
                    _warn_once("Invalid history tool arguments; using an empty object")
                    args = {}
                args, _ = _arguments(args)
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                               "input": args if isinstance(args, dict) else {}})
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": m.get("content") or ""}
            last = out[-1] if out else None
            if (last and last["role"] == "user" and isinstance(last["content"], list) and last["content"]
                    and isinstance(last["content"][0], dict)
                    and last["content"][0].get("type") == "tool_result"):
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


class AnthropicChat:
    def __init__(self, client: Any, model_name: str, max_tokens: int = 4096, effort: Optional[str] = None) -> None:
        self._client = client
        self.model_name = model_name
        self._max_tokens = max_tokens
        self._effort = effort
        self.usage_agent = "brain"

    @_safe(ModelMessage)
    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
             tool_choice: Optional[str] = None) -> ModelMessage:
        system, converted = to_anthropic_messages(messages)
        kwargs: Dict[str, Any] = {"model": self.model_name, "max_tokens": self._max_tokens, "messages": converted}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_anthropic_tool(t) for t in tools]
            kwargs["tool_choice"] = {"type": "none"} if tool_choice == "none" else {"type": "auto"}
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        from .. import usage_ledger  # noqa: PLC0415

        resp = usage_ledger.call(getattr(self, "usage_agent", "brain"), lambda: self._client.messages.create(**kwargs),
                                 provider="anthropic", model=self.model_name,
                                 images=usage_ledger.images_in(converted))
        usage = getattr(resp, "usage", None)
        tokens = (_tokens(getattr(usage, "input_tokens", 0)), _tokens(getattr(usage, "output_tokens", 0)))
        if getattr(resp, "stop_reason", None) == "refusal":
            return ModelMessage(raw=resp.content, usage=tokens, refused=True)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        cut = getattr(resp, "stop_reason", None) == "max_tokens"
        calls = []
        for b in resp.content:
            if getattr(b, "type", "") == "tool_use":
                if not _call_identity(b.id, b.name):
                    continue
                args, valid = _arguments(getattr(b, "input", None))
                calls.append(ToolCall(id=b.id, name=b.name, arguments=args,
                                      raw_arguments=json.dumps(args, ensure_ascii=False, allow_nan=False),
                                      valid=valid and not cut))
        return ModelMessage(content=text or None, tool_calls=tuple(calls), raw=resp.content, usage=tokens)


def _http_client() -> Any:
    import httpx  # noqa: PLC0415

    return httpx.Client(verify=ssl.create_default_context())     # OS trust store: works behind TLS interception


@_safe(lambda: None)
def make_model(spec: str, env: Dict[str, str]) -> Optional[Any]:
    """The chat model for ``"<provider>:<model>"`` (a bare name means OpenAI), or None without its key."""
    if not isinstance(spec, str) or not isinstance(env, dict):
        _warn_once("Invalid model configuration; disabling the model")
        return None
    spec = spec.strip()
    if not spec:
        return None
    provider, separator, name = spec.partition(":")
    if not separator:
        provider, name = "openai", provider
    if not name.strip():
        _warn_once("Missing model name; disabling the model")
        return None
    provider = provider.lower()
    name = name.strip()
    temperature = None if name.startswith(_REASONING_PREFIXES) else 0.0
    if provider in ("openai", "gemini"):
        key = env.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY", "")
        if not isinstance(key, str) or not key.strip():
            return None
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            log.warning("pip/uv: openai is not installed; the %s model is disabled", provider)
            return None

        extra = {"base_url": GEMINI_BASE_URL} if provider == "gemini" else {}
        http_client = _http_client()
        try:
            client = OpenAI(api_key=key, http_client=http_client, **extra)
        except Exception:
            http_client.close()
            raise
        return OpenAIChat(client, name, temperature)
    if provider == "anthropic":
        key = env.get("ANTHROPIC_API_KEY", "")
        if not isinstance(key, str) or not key.strip():
            return None
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            log.warning("pip/uv: anthropic is not installed; the anthropic model is disabled")
            return None

        # The box injects the OS trust store at start-up (truststore), which the SDK's client uses.
        effort = None if name.startswith("claude-haiku") else env.get("ANTHROPIC_EFFORT", "low")
        return AnthropicChat(anthropic.Anthropic(api_key=key), name, effort=effort)
    # Any other OpenAI-compatible provider of providers.PROVIDERS ("openrouter:openai/gpt-4o"). 2026-10-08: the box's
    # OpenAI account ran out of credit while OpenRouter (already paying for the Eye) had it; the assistant can follow.
    from ..providers import PROVIDERS  # noqa: PLC0415

    known = PROVIDERS.get(provider)
    if known is not None and provider != "openai":
        key = env.get(known.key_env or "", "") if known.key_env else ""
        if known.key_required and (not isinstance(key, str) or not key.strip()):
            return None
        base_url = (env.get(known.base_url_env) if known.base_url_env else None) or known.base_url
        if not base_url:
            _warn_once("Missing base URL for the %s model; disabling it" % provider)
            return None
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            log.warning("pip/uv: openai is not installed; the %s model is disabled", provider)
            return None
        http_client = _http_client()
        try:
            client = OpenAI(api_key=key or "none", base_url=base_url, http_client=http_client)
        except Exception:
            http_client.close()
            raise
        # OpenRouter says what each call cost when asked (the usage ledger's usd_source "provider").
        usage_body = {"usage": (known.extra_body or {}).get("usage")} if (known.extra_body or {}).get("usage") else None
        return OpenAIChat(client, name, temperature, extra_body=usage_body)
    _warn_once("Unknown model provider; disabling the model")
    return None
