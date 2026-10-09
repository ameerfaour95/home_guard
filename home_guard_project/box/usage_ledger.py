"""What every AI call on the box cost: one JSON line per call, for the Admin Center's cost monitor.

Owner request (2026-10-09): "In admin, in monitoring, I want to monitor the cost, for each agent, for each model.
I want to see the cost every time, so we can keep track." The contract the Admin reads is
``docs/contracts/usage_ledger.md``; ``tests/box/test_usage_ledger.py`` pins it.

Where: ``<state_dir>/usage/<YYYY-MM-DD>.jsonl`` (the box's local date of the call), one object per line:

    {"ts": 1791583412.31, "box_id": "9f...", "site": "ameer_tes2", "agent": "eye", "provider": "openrouter",
     "model": "qwen/qwen3.5-9b", "route": "direct", "prompt_tokens": 9123, "completion_tokens": 180,
     "cached_tokens": 0, "images": 8, "usd": 0.000939, "usd_source": "provider", "seconds": 3.42, "ok": true,
     "error_kind": "", "camera": "ameer_tes2_ch6", "alert_id": "ameer_tes2_ch6_1791583405_alert"}

How: each call site wraps the request itself (:func:`call`, or :func:`record` after it), inside whatever thread makes
the HTTP call, so a call its caller stopped waiting for is still counted when it finishes. Failed and timed-out calls
are recorded too (``ok`` false, ``usd`` 0 unless the provider billed something). Lines go to a small in-memory queue
that a daemon thread appends to the day's file (one write per batch, under a lock file: the inference process and the
assistant may write the same day). Nothing here ever raises into a caller.

Set ``HOMEGUARD_USAGE_LEDGER=off`` to write nothing (the test suite does).
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import contextvars
import datetime as dt
import functools
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

USAGE_DIR = "usage"          # under paths.state_dir(); in the site outbox: usage/<date>.jsonl
VERSION = 1

# Who made the call. A line with anything else says "other".
AGENTS: Tuple[str, ...] = (
    "eye",                # inference.GptBackend.analyze: the alert's vision model
    "eye_fallback",       # the same call on the fallback vision model (inference.FallbackBackend)
    "eye_rescue",         # no model answered: the main model again on 768 px frames (inference.vlm_rescue)
    "second_look",        # the yes/no check before a red (inference.second_look -> GptBackend.verify)
    "activity_look",      # the owner's-activity context look on a red (activity_memory.red_look)
    "describer",          # the alert's per-person description (describer.Describer.ask)
    "translator",         # English -> the owner's language (messenger.Messenger, its main model)
    "translator_fast",    # the translator's hedge model, raced after a slow or broken answer
    "brain",              # the owner's assistant, its main model (brain/models.py; agent.py v1)
    "brain_fast",         # the assistant's fast model (box.yaml agent_fast_model)
    "brain_tool_vision",  # the assistant's camera look (brain/vision.py: look_around / ask_vision)
    "embeddings",         # alert search and case memory vectors (embeddings.py)
    "investigator",       # reserved: the lingering investigator is tracker-only today (no model call)
    "transcription",      # the owner's voice messages (voice.py)
    "case_judge",         # the case-memory judge (case_memory/judge.py)
    "other",              # anything else (live_view, eval tools)
)

FIELDS: Tuple[str, ...] = (
    "ts", "box_id", "site", "agent", "provider", "model", "route", "prompt_tokens", "completion_tokens",
    "cached_tokens", "images", "usd", "usd_source", "seconds", "ok", "error_kind", "camera", "alert_id",
)
ROUTES = ("direct", "gateway", "fallback-direct")
USD_SOURCES = ("provider", "price_table", "unknown")
ERROR_KINDS = ("", "timeout", "429", "5xx", "other")

# --- the call's context: agent override, camera, alert ------------------------------------------------------------
_SCOPE: "contextvars.ContextVar[Dict[str, Any]]" = contextvars.ContextVar("usage_ledger_scope", default={})
_NOTE = threading.local()          # set by the HTTP layer during a call (note_route), read by record()


@contextlib.contextmanager
def scope(**fields: Any) -> Iterator[None]:
    """Calls made inside (in this thread, or a thread started with :func:`carry`) get these fields: ``camera``,
    ``alert_id``, and ``agent``, which OVERRIDES the call site's own agent (the 768 px rescue runs the Eye's code
    as ``eye_rescue``; the activity look runs the second look's as ``activity_look``)."""
    token = _SCOPE.set({**_SCOPE.get(), **{k: v for k, v in fields.items() if v not in (None, "")}})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def carry(fn: Callable[..., Any]) -> Callable[..., Any]:
    """*fn* bound to the current scope, for a ``threading.Thread`` target (threads do not inherit it)."""
    return functools.partial(contextvars.copy_context().run, fn)


def bound(fn: Callable[..., Any], **fields: Any) -> Callable[..., Any]:
    """*fn* run inside ``scope(**fields)``: a thread target that starts a new alert's scope."""
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        with scope(**fields):
            return fn(*args, **kwargs)
    return run


def note_route(route: str, usd: Optional[float] = None) -> None:
    """For the HTTP layer (ai_route's RoutingTransport): how the call now running in this thread went out
    (``gateway`` / ``fallback-direct`` / ``direct``) and, when known, what the gateway billed. Read by the next
    :func:`record` in this thread; :func:`call` clears it first."""
    try:
        _NOTE.route = route if route in ROUTES else None
        _NOTE.usd = float(usd) if usd is not None else None
    except (TypeError, ValueError):
        _NOTE.usd = None


def _clear_note() -> None:
    _NOTE.route = None
    _NOTE.usd = None


# --- reading a response --------------------------------------------------------------------------------------------
def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)      # pydantic: fields the SDK does not declare (OpenRouter's cost)
        if isinstance(extra, Mapping):
            value = extra.get(name)
    return value


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) and out >= 0 else None


def usage_of(response: Any) -> Dict[str, Any]:
    """``{prompt_tokens, completion_tokens, cached_tokens, cost}`` from any response the box gets: an OpenAI-SDK
    object, a JSON dict (embeddings), Anthropic's (``input_tokens``/``output_tokens``/cache reads). *cost* is the
    provider's own dollar figure (OpenRouter ``usage.cost``), else None."""
    u = _get(response, "usage")
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost": None}
    prompt = _get(u, "prompt_tokens")
    completion = _get(u, "completion_tokens")
    cached = _int(_get(_get(u, "prompt_tokens_details"), "cached_tokens"))
    if prompt is None and _get(u, "input_tokens") is not None:          # Anthropic, audio transcription
        cache_read = _int(_get(u, "cache_read_input_tokens"))
        prompt = _int(_get(u, "input_tokens")) + cache_read + _int(_get(u, "cache_creation_input_tokens"))
        completion = _get(u, "output_tokens")
        cached = cached or cache_read
    if prompt is None and _get(u, "total_tokens") is not None:          # embeddings: total == prompt
        prompt = _get(u, "total_tokens")
    return {"prompt_tokens": _int(prompt), "completion_tokens": _int(completion), "cached_tokens": cached,
            "cost": _float(_get(u, "cost"))}


def provider_of(client: Any) -> str:
    """The provider a client (or base URL) talks to, by host: providers.PROVIDERS names, else the host."""
    url = client if isinstance(client, str) else str(getattr(client, "base_url", "") or "")
    url = url.lower()
    if not url:
        return ""
    gateway = str(os.environ.get("HOMEGUARD_GATEWAY_URL") or "").strip().lower().rstrip("/")
    if gateway and url.startswith(gateway):
        return "gateway"
    for needle, name in (("openrouter.ai", "openrouter"), ("api.openai.com", "openai"),
                         ("generativelanguage.googleapis.com", "google"), ("dashscope", "dashscope-intl"),
                         ("api.anthropic.com", "anthropic"), (":11434", "ollama")):
        if needle in url:
            return name
    host = url.split("://", 1)[-1].split("/", 1)[0]
    return host


def _split_model(answered: str, provider: str) -> Tuple[str, str, bool]:
    """``(provider, model, via_gateway)``: the gateway names its upstream ``provider:model`` (providers.model_key)."""
    from . import providers  # noqa: PLC0415

    prefix, sep, rest = answered.partition(":")
    if sep and prefix in providers.PROVIDERS and rest:
        return prefix, rest, True
    return provider, answered, provider == "gateway"


def _configured_route(provider: str, model: str) -> str:
    """The route box.yaml asks for this call (ai_route, when that module is there): what a call without a gateway
    answer is recorded as."""
    try:
        from . import ai_route  # noqa: PLC0415  (the gateway branch; absent before it)

        route = ai_route.current()
        if not getattr(route, "routed", False):
            return "direct"
        routes = getattr(ai_route, "ROUTES", {}) or {}
        if not any(key[1:] == (provider, model) for key in routes):
            return "direct"                          # no alias: that call always goes directly
        return "fallback-direct" if getattr(route, "falls_back", False) else "gateway"
    except Exception:  # noqa: BLE001 - no ai_route module, or no box.yaml: direct
        return "direct"


def error_kind(error: Any) -> str:
    """``timeout`` / ``429`` / ``5xx`` / ``other`` for an exception; "" for none."""
    if error is None:
        return ""
    if isinstance(error, TimeoutError):
        return "timeout"
    name = type(error).__name__.lower()
    if "timeout" in name or "deadline" in name:
        return "timeout"
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    if status == 429 or "ratelimit" in name:
        return "429"
    if status is not None and status >= 500:
        return "5xx"
    if "internalserver" in name or "serviceunavailable" in name:
        return "5xx"
    return "other"


def images_in(messages: Any) -> int:
    """How many pictures a chat request carries (``image_url`` parts)."""
    n = 0
    try:
        for m in messages or ():
            content = m.get("content") if isinstance(m, Mapping) else None
            if isinstance(content, list):
                n += sum(1 for part in content if isinstance(part, Mapping) and part.get("type") in ("image_url", "image"))
    except Exception:  # noqa: BLE001
        return n
    return n


# --- the line ------------------------------------------------------------------------------------------------------
def build_line(agent: str, response: Any = None, started: Optional[float] = None, *, provider: str = "",
               model: str = "", error: Any = None, images: int = 0, camera: Optional[str] = None,
               alert_id: Optional[str] = None, route: Optional[str] = None, seconds: Optional[float] = None,
               client: Any = None, ok: Optional[bool] = None, now: Optional[float] = None) -> Dict[str, Any]:
    """The ledger line for one call (see the module doc). Pure apart from reading the box's identity."""
    from . import providers  # noqa: PLC0415

    ctx = _SCOPE.get()
    agent = str(ctx.get("agent") or agent or "other")
    if agent not in AGENTS:
        agent = "other"
    provider = str(provider or provider_of(client) or "")
    answered = _get(response, "model")
    answered = answered.strip() if isinstance(answered, str) and answered.strip() else ""
    provider, model_name, via_gateway = _split_model(answered or str(model or ""), provider)
    used = usage_of(response)
    note_r, note_usd = getattr(_NOTE, "route", None), getattr(_NOTE, "usd", None)
    if route not in ROUTES:
        route = note_r or ("gateway" if via_gateway else _configured_route(provider, model_name))
    cost = used["cost"] if used["cost"] is not None else note_usd
    if cost is not None:
        usd, source = cost, "provider"
    else:
        priced = providers.cost_usd(model_name, used["prompt_tokens"], used["completion_tokens"])
        if priced is None and model and model != model_name:
            priced = providers.cost_usd(str(model), used["prompt_tokens"], used["completion_tokens"])
        usd, source = (priced, "price_table") if priced is not None else (0.0, "unknown")
    if seconds is None and started is not None:
        seconds = time.monotonic() - started
    ts = time.time() if now is None else now
    return {
        "ts": round(ts, 3), "box_id": _identity()[0], "site": _identity()[1], "agent": agent,
        "provider": provider, "model": model_name, "route": route,
        "prompt_tokens": used["prompt_tokens"], "completion_tokens": used["completion_tokens"],
        "cached_tokens": used["cached_tokens"], "images": _int(images), "usd": round(float(usd), 8),
        "usd_source": source, "seconds": round(float(seconds or 0.0), 3),
        "ok": bool(error is None if ok is None else ok), "error_kind": error_kind(error),
        "camera": str(camera or ctx.get("camera") or ""), "alert_id": str(alert_id or ctx.get("alert_id") or ""),
    }


def record(agent: str, response: Any = None, started: Optional[float] = None, **fields: Any) -> Optional[Dict[str, Any]]:
    """Record one AI call. *agent*: one of :data:`AGENTS` (a :func:`scope` agent overrides it); *response*: what
    the API returned (None when it failed); *started*: ``time.monotonic()`` when the call began. Keyword fields:
    ``provider`` / ``client`` (to name the provider), ``model`` (the one asked; the answer's own wins), ``error``
    (the exception), ``images``, ``camera``, ``alert_id``, ``route``, ``seconds``. Returns the line, or None when
    the ledger is off. Never raises."""
    try:
        if not LEDGER.enabled:
            _clear_note()
            return None
        line = build_line(agent, response, started, **fields)
        _clear_note()
        LEDGER.put(line)
        return line
    except Exception as exc:  # noqa: BLE001 - the ledger must never cost an alert
        log.debug("usage ledger: line not recorded: %s", exc)
        return None


def call(agent: str, fn: Callable[[], Any], **fields: Any) -> Any:
    """``fn()`` (the API request), recorded either way; its answer is returned and its exception re-raised."""
    _clear_note()
    started = time.monotonic()
    try:
        response = fn()
    except BaseException as exc:
        record(agent, None, started, error=exc, **fields)
        raise
    record(agent, response, started, **fields)
    return response


# --- identity --------------------------------------------------------------------------------------------------------
_ID_CACHE: Dict[str, Any] = {}


def _identity() -> Tuple[str, str]:
    """``(box_id, site)``, read once per process (configure() sets them)."""
    if "id" not in _ID_CACHE:
        try:
            from .box_identity import box_id  # noqa: PLC0415

            _ID_CACHE["id"] = box_id() or ""
        except Exception:  # noqa: BLE001
            _ID_CACHE["id"] = ""
    if "site" not in _ID_CACHE:
        try:
            from .boxconfig import load_box_settings  # noqa: PLC0415

            _ID_CACHE["site"] = str(load_box_settings().get("site") or "")
        except Exception:  # noqa: BLE001
            _ID_CACHE["site"] = ""
    return _ID_CACHE["id"], _ID_CACHE["site"]


# --- the file --------------------------------------------------------------------------------------------------------
def default_dir() -> str:
    from . import paths  # noqa: PLC0415

    return os.path.join(paths.state_dir(), USAGE_DIR)


def day_of(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


@contextlib.contextmanager
def _file_lock(path: str) -> Iterator[None]:
    """A cross-process lock on *path* (best effort: on any failure the write goes ahead unlocked)."""
    fd = None
    locked = False
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            for _ in range(50):
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    time.sleep(0.02)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
    except Exception:  # noqa: BLE001
        pass
    try:
        yield
    finally:
        if fd is not None:
            try:
                if locked and os.name == "nt":
                    import msvcrt  # noqa: PLC0415

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except Exception:  # noqa: BLE001
                pass
            os.close(fd)


def append_lines(directory: str, lines: Sequence[Mapping[str, Any]]) -> int:
    """Append *lines* to their days' files under *directory*, one write per day under the lock. Returns lines
    written."""
    by_day: Dict[str, List[str]] = {}
    for line in lines:
        by_day.setdefault(day_of(float(line.get("ts") or time.time())), []).append(
            json.dumps(line, ensure_ascii=False, separators=(",", ":")))
    os.makedirs(directory, exist_ok=True)
    written = 0
    with _file_lock(os.path.join(directory, ".lock")):
        for day, texts in sorted(by_day.items()):
            data = "".join(t + "\n" for t in texts).encode("utf-8")
            fd = os.open(os.path.join(directory, f"{day}.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND
                         | getattr(os, "O_BINARY", 0), 0o644)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            written += len(texts)
    return written


class Ledger:
    """The process's writer: a bounded queue and a daemon thread that empties it every *interval* seconds."""

    MAX_QUEUE = 5000

    def __init__(self, directory: Optional[str] = None, interval: float = 2.0) -> None:
        self._dir = directory
        self.interval = interval
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(self.MAX_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._enabled: Optional[bool] = None
        self.dropped = 0

    @property
    def enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        return str(os.environ.get("HOMEGUARD_USAGE_LEDGER") or "").strip().lower() not in ("0", "off", "false", "no")

    @property
    def directory(self) -> str:
        return self._dir or default_dir()

    def configure(self, directory: Optional[str] = None, enabled: Optional[bool] = None) -> None:
        self.flush()
        self._dir = directory
        self._enabled = enabled

    def put(self, line: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(line)
        except queue.Full:
            self.dropped += 1
            return
        self._start()

    def _start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="usage-ledger", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            time.sleep(self.interval)
            self.flush()

    def flush(self) -> int:
        """Write everything queued now. Never raises; lines that cannot be written are dropped with a warning."""
        lines: List[Dict[str, Any]] = []
        while True:
            try:
                lines.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not lines:
            return 0
        try:
            return append_lines(self.directory, lines)
        except Exception as exc:  # noqa: BLE001
            log.warning("usage ledger: %d line(s) not written: %s", len(lines), exc)
            return 0


LEDGER = Ledger()
atexit.register(LEDGER.flush)


def configure(directory: Optional[str] = None, enabled: Optional[bool] = None, site: Optional[str] = None,
              box_id: Optional[str] = None) -> None:
    """Point the ledger elsewhere, switch it on or off, or set the identity (tests; None: the defaults)."""
    LEDGER.configure(directory, enabled)
    _ID_CACHE.clear()
    if site is not None:
        _ID_CACHE["site"] = site
    if box_id is not None:
        _ID_CACHE["id"] = box_id


def flush() -> int:
    return LEDGER.flush()


# --- reading it back -------------------------------------------------------------------------------------------------
def read_day(day: str, directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """The lines of *day* (YYYY-MM-DD); damaged lines are skipped."""
    path = os.path.join(directory or LEDGER.directory, f"{day}.jsonl")
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for text in f:
                try:
                    line = json.loads(text)
                except ValueError:
                    continue
                if isinstance(line, dict):
                    out.append(line)
    except OSError:
        pass
    return out


def totals(lines: Iterable[Mapping[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """Per *key* (``agent`` or ``model``...): calls, failed, prompt/completion tokens, images, usd."""
    out: Dict[str, Dict[str, Any]] = {}
    for line in lines:
        name = str(line.get(key) or "?")
        t = out.setdefault(name, {"calls": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                  "images": 0, "usd": 0.0})
        t["calls"] += 1
        t["failed"] += 0 if line.get("ok") else 1
        t["prompt_tokens"] += _int(line.get("prompt_tokens"))
        t["completion_tokens"] += _int(line.get("completion_tokens"))
        t["images"] += _int(line.get("images"))
        t["usd"] += _float(line.get("usd")) or 0.0
    return out


def usage_today(directory: Optional[str] = None, now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """The heartbeat's ``usage_today``: ``{agent: {calls, usd}, ..., "total": {calls, usd}}`` for the box's local
    date. Never raises ({"total": zeros} when there is nothing)."""
    try:
        lines = read_day(day_of(time.time() if now is None else now), directory)
        per = totals(lines, "agent")
    except Exception:  # noqa: BLE001
        per = {}
    out = {agent: {"calls": t["calls"], "usd": round(t["usd"], 6)} for agent, t in sorted(per.items())}
    out["total"] = {"calls": sum(t["calls"] for t in per.values()), "usd": round(sum(t["usd"] for t in per.values()), 6)}
    return out


def _table(title: str, rows: Dict[str, Dict[str, Any]]) -> List[str]:
    out = [title, f"  {'':<36}{'calls':>7}{'failed':>8}{'in tok':>12}{'out tok':>10}{'images':>8}{'$':>12}"]
    for name, t in sorted(rows.items(), key=lambda kv: -kv[1]["usd"]):
        out.append(f"  {name[:35]:<36}{t['calls']:>7}{t['failed']:>8}{t['prompt_tokens']:>12}"
                   f"{t['completion_tokens']:>10}{t['images']:>8}{t['usd']:>12.6f}")
    calls = sum(t["calls"] for t in rows.values())
    usd = sum(t["usd"] for t in rows.values())
    out.append(f"  {'total':<36}{calls:>7}{'':>8}{'':>12}{'':>10}{'':>8}{usd:>12.6f}")
    return out


def summary(day: str, days: int = 1, directory: Optional[str] = None) -> str:
    """The ``summary`` command's text: per agent and per model for *day*, or for the *days* ending on it with a
    per-day trend."""
    end = dt.datetime.strptime(day, "%Y-%m-%d").date()
    span = [(end - dt.timedelta(days=i)).isoformat() for i in range(max(1, days) - 1, -1, -1)]
    per_day = {d: read_day(d, directory) for d in span}
    lines = [line for d in span for line in per_day[d]]
    label = day if len(span) == 1 else f"{span[0]} .. {span[-1]}"
    out = [f"AI usage {label} ({directory or LEDGER.directory})"]
    if not lines:
        out.append("  no calls recorded")
        return "\n".join(out)
    out += _table("per agent:", totals(lines, "agent"))
    out += _table("per model:", totals(lines, "model"))
    sources = totals(lines, "usd_source")
    out.append("dollars from: " + ", ".join(f"{k} {v['calls']} calls" for k, v in sorted(sources.items())))
    if len(span) > 1:
        out.append("per day:")
        for d in span:
            t = totals(per_day[d], "site").values()
            calls, usd = sum(x["calls"] for x in t), sum(x["usd"] for x in t)
            out.append(f"  {d}  {calls:>6} calls  ${usd:.6f}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m home_guard_project.box.usage_ledger",
                                description="What the box's AI calls cost (the usage ledger).")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary", help="calls, tokens and dollars per agent and per model")
    s.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD (default today, local)")
    s.add_argument("--days", type=int, default=1, help="the N days ending on --date, with a per-day trend")
    s.add_argument("--dir", default=None, help="the ledger folder (default <state_dir>/usage)")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    print(summary(args.date, args.days, args.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
