"""The gateway's YAML config: where it listens, its limits, and the model aliases.

An alias is what a box asks for (``model: eye``); the config maps it to an ordered
list of upstreams, each a provider from ``box/providers.py`` and a model there. The
first upstream answers; the next one is asked only when it fails. Every upstream
needs a price (``providers.PRICES``, or ``price_per_m`` here), so the daily caps
can never be bypassed by a model the meter cannot count.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from home_guard_project.box import providers

KINDS = ("chat", "embeddings")
ENDPOINTS = {"chat": "/chat/completions", "embeddings": "/embeddings"}


class ConfigError(Exception):
    """The config file is missing a value or has a wrong one."""


@dataclass(frozen=True)
class Upstream:
    provider: str
    model: str
    key_env: Optional[str] = None                      # overrides the provider's key variable
    price_per_m: Optional[Tuple[float, float]] = None  # overrides providers.PRICES ($ per million in/out)

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def price(self) -> Tuple[float, float]:
        price = self.price_per_m or providers.price_of(self.model)
        assert price is not None    # load_config refuses an upstream without a price
        return price

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        p_in, p_out = self.price()
        return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000

    def resolve(self, env: Mapping[str, str]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """``(api_key, base_url, extra_body)``; raises providers.ProviderError when the key is not set."""
        p = providers.get(self.provider)
        if self.key_env and p.key_env:
            env = {**env, p.key_env: env.get(self.key_env, "")}
        key, url, extra = providers.resolve(self.provider, env, self.model)
        return key, (url or providers.OPENAI_URL).rstrip("/"), extra


@dataclass(frozen=True)
class Alias:
    name: str
    kind: str
    upstreams: Tuple[Upstream, ...]


@dataclass(frozen=True)
class Config:
    aliases: Dict[str, Alias]
    db_path: str = "gateway.sqlite3"
    host: str = "0.0.0.0"
    port: int = 8080
    box_daily_usd: Optional[float] = 1.0          # None: no cap; a box's own cap (add-box --cap) wins
    fleet_daily_usd: Optional[float] = None       # None: no cap
    box_rpm: int = 60                             # requests a minute per box; a loop on a box cannot run up a bill
    max_body_bytes: int = 8_000_000               # six 1080p JPEGs in base64 are ~3 MB
    upstream_timeout_sec: float = 20.0            # one attempt
    deadline_sec: float = 27.0                    # the whole request; below the box's 30 s client timeout
    max_concurrent: int = 256
    admin_token_sha256: str = ""                  # empty: the admin endpoints are off
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)


def _number(raw: Mapping[str, Any], key: str, default: Any, kind: type = float) -> Any:
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be a number")
    try:
        value = kind(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{key} must be a number") from None
    if value < 0:
        raise ConfigError(f"{key} must not be negative")
    return value


def _upstream(alias: str, raw: Any) -> Upstream:
    if not isinstance(raw, dict) or not isinstance(raw.get("provider"), str) or not isinstance(raw.get("model"), str):
        raise ConfigError(f"alias {alias!r}: each upstream needs a provider and a model")
    provider, model = raw["provider"].strip().lower(), raw["model"].strip()
    try:
        providers.get(provider)
    except providers.ProviderError as exc:
        raise ConfigError(f"alias {alias!r}: {exc}") from None
    if provider == "gateway":
        raise ConfigError(f"alias {alias!r}: the gateway cannot be its own upstream")
    price = raw.get("price_per_m")
    if price is not None:
        if (not isinstance(price, (list, tuple)) or len(price) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 for x in price)):
            raise ConfigError(f"alias {alias!r}: price_per_m must be [input, output] dollars per million tokens")
        price = (float(price[0]), float(price[1]))
    elif providers.price_of(model) is None:
        raise ConfigError(f"alias {alias!r}: no price for {model!r}; add it to box/providers.py PRICES "
                          f"or give the upstream a price_per_m (e.g. [0, 0] for our own GPU)")
    key_env = raw.get("key_env")
    if key_env is not None and not isinstance(key_env, str):
        raise ConfigError(f"alias {alias!r}: key_env must be a variable name")
    return Upstream(provider, model, key_env or None, price)


def _alias(name: str, raw: Any) -> Alias:
    if isinstance(raw, list):
        raw = {"upstreams": raw}
    if not isinstance(raw, dict):
        raise ConfigError(f"alias {name!r} must be a list of upstreams or a mapping with 'upstreams'")
    kind = str(raw.get("kind", "chat")).strip().lower()
    if kind not in KINDS:
        raise ConfigError(f"alias {name!r}: kind must be one of {', '.join(KINDS)}")
    ups = raw.get("upstreams")
    if not isinstance(ups, list) or not ups:
        raise ConfigError(f"alias {name!r} needs at least one upstream")
    return Alias(name, kind, tuple(_upstream(name, u) for u in ups))


def parse_config(raw: Any, env: Optional[Mapping[str, str]] = None) -> Config:
    env = os.environ if env is None else env
    if not isinstance(raw, dict):
        raise ConfigError("the config must be a mapping")
    aliases_raw = raw.get("aliases")
    if not isinstance(aliases_raw, dict) or not aliases_raw:
        raise ConfigError("the config needs at least one alias under 'aliases'")
    aliases = {str(name): _alias(str(name), value) for name, value in aliases_raw.items()}
    listen = raw.get("listen") or {}
    if not isinstance(listen, dict):
        raise ConfigError("listen must be a mapping with host and port")
    caps = raw.get("caps") or {}
    if not isinstance(caps, dict):
        raise ConfigError("caps must be a mapping")
    admin = str(env.get("HOMEGUARD_ADMIN_TOKEN_SHA256") or raw.get("admin_token_sha256") or "").strip().lower()
    if admin and (len(admin) != 64 or any(c not in "0123456789abcdef" for c in admin)):
        raise ConfigError("admin_token_sha256 must be the 64-character sha256 of the admin token")
    cfg = Config(
        aliases=aliases,
        db_path=str(env.get("HOMEGUARD_GATEWAY_DB") or raw.get("db_path") or "gateway.sqlite3"),
        host=str(listen.get("host", "0.0.0.0")),
        port=_number(listen, "port", 8080, int),
        box_daily_usd=_number(caps, "box_daily_usd", 1.0),
        fleet_daily_usd=_number(caps, "fleet_daily_usd", None),
        box_rpm=_number(raw, "box_rpm", 60, int),
        max_body_bytes=_number(raw, "max_body_bytes", 8_000_000, int),
        upstream_timeout_sec=_number(raw, "upstream_timeout_sec", 20.0),
        deadline_sec=_number(raw, "deadline_sec", 27.0),
        max_concurrent=_number(raw, "max_concurrent", 256, int),
        admin_token_sha256=admin,
        env=env,
    )
    if not cfg.upstream_timeout_sec or not cfg.deadline_sec:
        raise ConfigError("upstream_timeout_sec and deadline_sec must be above zero")
    return cfg


def load_config(path: str, env: Optional[Mapping[str, str]] = None) -> Config:
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    return parse_config(raw, env)
