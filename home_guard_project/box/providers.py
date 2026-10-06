"""Where the vision model is asked and what it costs.

Every provider here speaks the OpenAI ``chat.completions`` API, so one client
(``inference.GptBackend``) reaches all of them with a different base URL and key.
"""
from __future__ import annotations

import os
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
    # The laptop GPU (research runs of the 4B/3B/7B models). Ollama ignores ``think: false`` on its
    # OpenAI endpoint; ``reasoning_effort: none`` turns Qwen3.5's thinking off (1,000+ tokens -> ~100).
    "ollama": Provider("ollama", "http://localhost:11434/v1", None, {"reasoning_effort": "none"},
                       base_url_env="OLLAMA_BASE_URL", key_required=False),
    # Our own GPU server (stage B): the URL is required, the key optional.
    "vllm": Provider("vllm", None, "VLLM_API_KEY", base_url_env="VLLM_BASE_URL", key_required=False),
    "dashscope-intl": Provider("dashscope-intl", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                               "DASHSCOPE_API_KEY", {"enable_thinking": False}),
    # Google AI Studio's OpenAI-compatible endpoint (the translator, without OpenRouter in between).
    "google": Provider("google", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    # Our API gateway (home_guard_project/gateway): the provider keys stay on our server, the box
    # holds only its own token, and the model is an alias (e.g. ``eye``) the server maps to a provider.
    "gateway": Provider("gateway", None, "HOMEGUARD_BOX_TOKEN", base_url_env="HOMEGUARD_GATEWAY_URL"),
}

OPENAI_URL = "https://api.openai.com/v1"

# $ per million tokens (input, output). OpenRouter list prices on 2026-10-06; OpenAI's own for gpt-4o,
# gpt-4o-mini and the embeddings model. The eval reports and the gateway's per-box metering both read
# this table: a model missing here (every local model) is shown with tokens and no dollars, and the
# gateway refuses to route to it unless its config names a price.
PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "openai/gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "qwen/qwen3.5-9b": (0.10, 0.15),
    "qwen/qwen3-vl-8b-instruct": (0.117, 0.455),
    "qwen/qwen3-vl-8b-thinking": (0.18, 2.10),
    "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
    "qwen/qwen2.5-vl-72b-instruct": (0.80, 1.00),
    # The owner's translator (messenger.py) and the model it is compared with.
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "openai/gpt-6-luna": (0.10, 0.50),
}


class ProviderError(Exception):
    """Unknown provider, or its key or URL is not set."""


def get(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ProviderError(f"unknown provider {name!r}; known: {', '.join(sorted(PROVIDERS))}") from None


def resolve(name: str, env: Mapping[str, str],
            model: str = "") -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """``(api_key, base_url, extra_body)`` for *name*; raises naming the missing variable.
    A ``*thinking*`` model gets no extras: its provider refuses "thinking off"."""
    p = get(name)
    url = (str(env.get(p.base_url_env) or "").strip() if p.base_url_env else "") or p.base_url
    if url is None and p.base_url_env:
        raise ProviderError(f"{p.base_url_env} is not set (the server's address, e.g. http://host:8000/v1)")
    key = str(env.get(p.key_env) or "").strip() if p.key_env else ""
    if not key:
        if p.key_required:
            raise ProviderError(f"{p.key_env} is not set (put it in api_key.env at the repo root)")
        key = "ollama" if p.name == "ollama" else "none"   # the SDK wants some key; these servers ignore it
    extra = dict(p.extra_body) if p.extra_body and "thinking" not in model.lower() else None
    return key, url, extra


def openai_url(path: str, env: Optional[Mapping[str, str]] = None) -> str:
    """The OpenAI endpoint *path* (e.g. ``/embeddings``) for callers that post to it directly:
    ``OPENAI_BASE_URL`` when set (the gateway, like the OpenAI SDK itself does), else OpenAI."""
    base = str((os.environ if env is None else env).get("OPENAI_BASE_URL") or "").strip() or OPENAI_URL
    return base.rstrip("/") + "/" + path.lstrip("/")


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
