"""Ask an alias's upstreams in order: retry once on 429/5xx/timeout, then the next upstream.

A 401/403/404 from an upstream (our key, or a model it no longer offers) moves to the
next upstream without a retry. Any other 4xx is the request's own fault and goes back
to the box as the provider sent it, so the box's SDK sees the provider's message (the
box's "response_format" fallback depends on it). Prompts and keys are never logged:
an error message is kept only cut short, with data URLs and key-like strings removed.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import httpx

from home_guard_project.box import providers

from .settings import ENDPOINTS, Alias, Upstream

log = logging.getLogger("gateway.router")

RETRYABLE = frozenset({408, 429}) | frozenset(range(500, 600))
NEXT_UPSTREAM = frozenset({401, 403, 404})
RETRY_PAUSE_SEC = 0.5

_DATA_URL = re.compile(r"data:[^\s\"']{16,}")
_KEYLIKE = re.compile(r"\b(sk|hgb|or|key)[-_][A-Za-z0-9_\-]{8,}")


def scrub(text: str, limit: int = 200) -> str:
    """An upstream's error message, safe to store: no images, nothing that looks like a key."""
    text = _DATA_URL.sub("data:<omitted>", str(text or ""))
    return _KEYLIKE.sub("<key>", text)[:limit]


@dataclass
class Attempt:
    upstream: Upstream
    status: int                      # 0: no answer (timeout, connection, not configured)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""


@dataclass
class Outcome:
    status: int
    body: bytes
    attempts: List[Attempt] = field(default_factory=list)
    upstream: Optional[Upstream] = None     # the one that answered (status 200)
    cost_usd: float = 0.0


def error_body(message: str, code: str, kind: str = "gateway_error") -> bytes:
    return json.dumps({"error": {"message": message, "type": kind, "code": code}}).encode("utf-8")


def _usage(body: Dict[str, Any]) -> tuple:
    u = body.get("usage") if isinstance(body, dict) else None
    u = u if isinstance(u, dict) else {}
    try:
        prompt = int(u.get("prompt_tokens") or 0)
        completion = int(u.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return 0, 0
    return max(prompt, 0), max(completion, 0)


def _message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        return str(err.get("message") if isinstance(err, dict) else err)
    except Exception:  # noqa: BLE001 - not JSON
        return resp.text[:200]


def forward(alias: Alias, payload: Dict[str, Any], client: httpx.Client, env: Mapping[str, str],
            timeout: float, deadline: float, clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep) -> Outcome:
    """Send *payload* (the box's request body) to *alias*'s upstreams. Never raises.

    *deadline* is the clock time by which the whole request must finish; an attempt
    gets ``min(timeout, time left)``.
    """
    path = ENDPOINTS[alias.kind]
    attempts: List[Attempt] = []
    for upstream in alias.upstreams:
        try:
            key, base_url, extra = upstream.resolve(env)
        except providers.ProviderError as exc:
            log.warning("upstream %s is not configured: %s", upstream.name, exc)
            attempts.append(Attempt(upstream, 0, error=f"not configured: {exc}"))
            continue
        body = dict(payload, model=upstream.model)
        for k, v in (extra or {}).items():
            body.setdefault(k, v)
        data = json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        for retry in (False, True):
            left = deadline - clock()
            if left <= 0.5:
                attempts.append(Attempt(upstream, 0, error="no time left before the deadline"))
                return _failed(attempts)
            t0 = clock()
            try:
                resp = client.post(base_url + path, content=data, headers=headers, timeout=min(timeout, left))
            except httpx.TimeoutException:
                attempts.append(Attempt(upstream, 0, int((clock() - t0) * 1000), error="timeout"))
                status = 504
            except httpx.HTTPError as exc:
                attempts.append(Attempt(upstream, 0, int((clock() - t0) * 1000),
                                        error=scrub(f"{type(exc).__name__}: {exc}")))
                status = 502
            else:
                ms = int((clock() - t0) * 1000)
                status = resp.status_code
                if status == 200:
                    try:
                        parsed = resp.json()
                        if not isinstance(parsed, dict):
                            raise ValueError("not an object")
                    except ValueError:
                        attempts.append(Attempt(upstream, 200, ms, error="the answer was not JSON"))
                        status = 502
                    else:
                        p_in, p_out = _usage(parsed)
                        cost = upstream.cost_usd(p_in, p_out)
                        attempts.append(Attempt(upstream, 200, ms, p_in, p_out, cost))
                        return Outcome(200, resp.content, attempts, upstream, cost)
                else:
                    attempts.append(Attempt(upstream, status, ms, error=scrub(_message(resp))))
                    if status not in RETRYABLE and status not in NEXT_UPSTREAM:
                        return Outcome(status, resp.content, attempts)     # the request's own fault
            if status in NEXT_UPSTREAM or retry:
                break
            sleep(RETRY_PAUSE_SEC)
    return _failed(attempts)


def _failed(attempts: List[Attempt]) -> Outcome:
    def what(a: Attempt) -> str:
        return str(a.status) if a.status else ("timeout" if a.error == "timeout" else "no answer")

    tried = ", ".join(f"{a.upstream.name}={what(a)}" for a in attempts) or "none"
    return Outcome(502, error_body(f"every upstream failed ({tried})", "upstreams_failed"), attempts)
