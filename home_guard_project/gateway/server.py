"""The gateway's HTTP server: OpenAI-compatible for the boxes, plus a small admin API.

    POST /v1/chat/completions    box token; ``model`` is an alias (``eye``, ``gpt-4o-mini`` ...)
    POST /v1/embeddings          box token; an alias of kind ``embeddings``
    GET  /v1/models              box token; the aliases on offer
    GET  /healthz                no token; ``{"ok": true}``
    GET  /admin/costs?days=7&box=<id>   admin token; per box and UTC day: calls, tokens, dollars
    GET  /admin/boxes                   admin token; every box, its cap and today's spend

Every error the gateway itself sends carries ``x-should-retry: false``: it already
retried and fell back, so the box's OpenAI SDK must not repeat the call (it would
otherwise retry a 5xx twice). A box over its daily cap gets 402 ``box_daily_cap``
(``fleet_daily_cap`` when the whole fleet is); the box treats that like any model
failure and still alerts from the detector. Plain ``http.server`` with a thread per
request: thousands of homes at one call per ten minutes is a few requests a second,
each waiting seconds on a provider.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import httpx

from . import router
from .config import Config
from .store import Call, Store, utc_day

log = logging.getLogger("gateway.server")

Reply = Tuple[int, bytes, Dict[str, str]]
_NO_RETRY = {"x-should-retry": "false"}
_KIND_OF_PATH = {"/v1/chat/completions": "chat", "/v1/embeddings": "embeddings"}


def _error(status: int, message: str, code: str, kind: str = "gateway_error",
           headers: Optional[Dict[str, str]] = None) -> Reply:
    return status, router.error_body(message, code, kind), dict(headers or _NO_RETRY)


def _json(payload: Any, status: int = 200) -> Reply:
    return status, json.dumps(payload).encode("utf-8"), {}


def _bearer(headers: Any) -> str:
    value = str(headers.get("Authorization") or "")
    return value[7:].strip() if value[:7].lower() == "bearer " else ""


class Gateway:
    """What the server does, without the HTTP plumbing (the tests call it directly too)."""

    def __init__(self, cfg: Config, store: Store, client: Optional[httpx.Client] = None,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        self.cfg, self.store = cfg, store
        self.client = client or httpx.Client(limits=httpx.Limits(max_connections=cfg.max_concurrent,
                                                                 max_keepalive_connections=64))
        self._slots = threading.BoundedSemaphore(max(1, cfg.max_concurrent))
        self._recent: Dict[str, Deque[float]] = {}
        self._recent_lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    # --- admission ----------------------------------------------------------------------------
    def _rate_ok(self, box_id: str) -> bool:
        if not self.cfg.box_rpm:
            return True
        now = self._clock()
        with self._recent_lock:
            q = self._recent.setdefault(box_id, deque())
            while q and now - q[0] >= 60.0:
                q.popleft()
            if len(q) >= self.cfg.box_rpm:
                return False
            q.append(now)
            return True

    def _over_cap(self, box_id: str, own_cap: Optional[float]) -> Optional[str]:
        """``box_daily_cap`` / ``fleet_daily_cap`` when that cap is reached today (UTC), else None."""
        day = utc_day()
        cap = own_cap if own_cap is not None else self.cfg.box_daily_usd
        if cap is not None and self.store.spent(box_id, day) >= cap:
            return "box_daily_cap"
        fleet = self.cfg.fleet_daily_usd
        if fleet is not None and self.store.fleet_spent(day) >= fleet:
            return "fleet_daily_cap"
        return None

    def _is_admin(self, headers: Any) -> bool:
        want = self.cfg.admin_token_sha256
        got = _bearer(headers)
        return bool(want and got) and hmac.compare_digest(hashlib.sha256(got.encode("utf-8")).hexdigest(), want)

    # --- requests -----------------------------------------------------------------------------
    def get(self, path: str, query: Dict[str, Any], headers: Any) -> Reply:
        if path == "/healthz":
            return _json({"ok": True})
        if path == "/v1/models":
            if self.store.box_for_token(_bearer(headers)) is None:
                return _error(401, "unknown or revoked box token", "invalid_token", "authentication_error")
            return _json({"object": "list", "data": [{"id": a.name, "object": "model", "owned_by": "homeguard",
                                                      "kind": a.kind} for a in self.cfg.aliases.values()]})
        if path.startswith("/admin/"):
            if not self._is_admin(headers):
                return _error(401, "admin token required", "invalid_token", "authentication_error")
            if path == "/admin/costs":
                try:
                    days = min(max(int((query.get("days") or ["7"])[0]), 1), 366)
                except ValueError:
                    return _error(400, "days must be a whole number", "bad_request")
                box = (query.get("box") or [None])[0]
                return _json({"days": days, "rows": self.store.costs(days, box)})
            if path == "/admin/boxes":
                return _json({"day": utc_day(), "boxes": self.store.list_boxes()})
        return _error(404, f"no such endpoint: {path}", "not_found")

    def post(self, path: str, headers: Any, body: bytes) -> Reply:
        kind = _KIND_OF_PATH.get(path)
        if kind is None:
            return _error(404, f"no such endpoint: {path}", "not_found")
        box = self.store.box_for_token(_bearer(headers))
        if box is None:
            return _error(401, "unknown or revoked box token", "invalid_token", "authentication_error")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return _error(400, "the body is not JSON", "bad_request", "invalid_request_error")
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            return _error(400, "the body needs a model", "bad_request", "invalid_request_error")
        if payload.get("stream"):
            return _error(400, "streaming is not offered by this gateway", "stream_unsupported",
                          "invalid_request_error")
        alias = self.cfg.aliases.get(payload["model"])
        if alias is None or alias.kind != kind:
            offered = ", ".join(sorted(a.name for a in self.cfg.aliases.values() if a.kind == kind))
            return _error(400, f"model {payload['model']!r} is not offered here; ask for one of: {offered}",
                          "model_not_found", "invalid_request_error")
        if not self._rate_ok(box.box_id):
            return _error(429, f"more than {self.cfg.box_rpm} requests a minute from this box", "box_rate_limit",
                          "rate_limit_error")
        cap = self._over_cap(box.box_id, box.daily_cap_usd)
        if cap is not None:
            self.store.record(Call(box.box_id, kind, alias.name, "", "", 402, error=cap))
            log.info("box=%s alias=%s refused: %s", box.box_id, alias.name, cap)
            who = "this box" if cap == "box_daily_cap" else "the service"
            return _error(402, f"{who} reached its daily AI budget; it resets at 00:00 UTC", cap,
                          "insufficient_quota")
        if not self._slots.acquire(blocking=False):
            return _error(503, "the gateway is busy; try again in a moment", "busy",
                          headers={"x-should-retry": "true", "retry-after": "1"})
        try:
            start = self._clock()
            out = router.forward(alias, payload, self.client, self.cfg.env, self.cfg.upstream_timeout_sec,
                                 start + self.cfg.deadline_sec, clock=self._clock, sleep=self._sleep)
        finally:
            self._slots.release()
        for a in out.attempts:
            self.store.record(Call(box.box_id, kind, alias.name, a.upstream.provider, a.upstream.model, a.status,
                                   a.prompt_tokens, a.completion_tokens, a.cost_usd, a.latency_ms, a.error))
        last = out.attempts[-1] if out.attempts else None
        log.info("box=%s alias=%s -> %s status=%d attempts=%d in=%d out=%d $%.6f %dms", box.box_id, alias.name,
                 out.upstream.name if out.upstream else (last.upstream.name if last else "-"), out.status,
                 len(out.attempts), last.prompt_tokens if last else 0, last.completion_tokens if last else 0,
                 out.cost_usd, int((self._clock() - start) * 1000))
        if out.status == 200 and out.upstream is not None:
            return 200, out.body, {"x-homeguard-upstream": out.upstream.name,
                                   "x-homeguard-cost-usd": f"{out.cost_usd:.6f}"}
        return out.status, out.body, dict(_NO_RETRY)


class Handler(BaseHTTPRequestHandler):
    gateway: Gateway
    timeout = 30                      # seconds to read a request; a stalled client does not hold a thread
    server_version = "HomeGuardGateway/1"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:   # the gateway logs its own line per request
        pass

    def _send(self, reply: Reply) -> None:
        status, body, headers = reply
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        url = urlsplit(self.path)
        self._send(self.gateway.get(url.path, parse_qs(url.query), self.headers))

    def do_POST(self) -> None:  # noqa: N802
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            self._send(_error(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required", "length_required"))
            return
        try:
            length = int(raw_len)
        except ValueError:
            length = -1
        if length < 0 or length > self.gateway.cfg.max_body_bytes:
            self.close_connection = True
            if 0 < length <= 4 * self.gateway.cfg.max_body_bytes:
                # Read it away first: closing on an unread body resets the connection and the
                # client sees a reset instead of this answer. Anything larger is just cut off.
                while length > 0:
                    chunk = self.rfile.read(min(length, 65536))
                    if not chunk:
                        break
                    length -= len(chunk)
            self._send(_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                              f"the request is larger than {self.gateway.cfg.max_body_bytes} bytes", "too_large"))
            return
        body = self.rfile.read(length)
        try:
            reply = self.gateway.post(urlsplit(self.path).path, self.headers, body)
        except Exception:  # noqa: BLE001 - a bug here must answer, not hang the box until its timeout
            log.exception("request failed")
            reply = _error(500, "the gateway failed on this request", "internal_error")
        self._send(reply)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def make_server(gateway: Gateway, host: Optional[str] = None, port: Optional[int] = None) -> Server:
    handler = type("BoundHandler", (Handler,), {"gateway": gateway})
    return Server((gateway.cfg.host if host is None else host, gateway.cfg.port if port is None else port), handler)
