"""Warn before OpenRouter retires a model the box uses.

OpenRouter lists each model's ``expiration_date`` (GET /api/v1/models). 2026-10-09: qwen/qwen3.5-9b (the Eye, the
second look, the describer) was set to go on 2026-10-21 and google/gemini-2.5-flash-lite (the translator's racer) on
2026-10-20, and qwen3-vl-32b went that day with no date at all, which broke the Admin's Suggest. So at inference
start (and daily after), every OpenRouter model box.yaml names (or its defaults) is checked:

- an ``expiration_date`` within WARN_DAYS days (or already past) -> a WARNING naming the role, model and date;
- a model missing from the list -> a WARNING that it is not listed (retired, or the name is wrong).

The list is fetched at most once a day (cached in the state folder); offline, the last cached list is used, and with
none nothing is checked. The check never raises and runs on its own thread, so it never delays or stops the box.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

log = logging.getLogger("box.model_expiry")

MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_NAME = "openrouter_models.json"
CACHE_MAX_AGE_SEC = 24 * 3600
WARN_DAYS = 14
FETCH_TIMEOUT_SEC = 20.0


def configured_models(box_settings: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``(role, model)`` for every OpenRouter model the box would call with these settings (defaults included)."""
    g = box_settings.get
    found: List[Tuple[str, str]] = []

    def add(role: str, provider: Any, model: Any) -> None:
        model = str(model or "").strip()
        if str(provider or "").strip().lower() == "openrouter" and model and (role, model) not in found:
            found.append((role, model))

    eye_provider = str(g("vlm_provider") or "openai").strip().lower()
    add("Eye (vlm_model)", eye_provider, g("vlm_model"))
    add("Eye fallback (vlm_fallback_model)", g("vlm_fallback_provider") or eye_provider, g("vlm_fallback_model"))
    try:
        from . import describer  # noqa: PLC0415

        on, provider, model, _ = describer.settings_of(box_settings)
        if on:
            add("describer (describer_model)", provider, model)
    except Exception as exc:  # noqa: BLE001 - one role unknown is no reason to skip the others
        log.debug("describer settings unread: %s", exc)
    try:
        from . import messenger  # noqa: PLC0415

        provider, model, _ = messenger.settings_of(box_settings)
        add("translator (messenger_model)", provider, model)
        add("translator racer (messenger_hedge_model)", provider,
            g("messenger_hedge_model") or messenger.HEDGE_MODEL)
    except Exception as exc:  # noqa: BLE001
        log.debug("translator settings unread: %s", exc)
    for key in ("agent_model", "agent_fast_model"):           # "provider:model" specs
        provider, sep, model = str(g(key) or "").partition(":")
        if sep:
            add(f"assistant ({key})", provider, model)
    return found


def _fetch() -> List[Dict[str, Any]]:
    import urllib.request  # noqa: PLC0415

    with urllib.request.urlopen(MODELS_URL, timeout=FETCH_TIMEOUT_SEC) as resp:  # noqa: S310 - fixed https URL
        data = json.load(resp).get("data")
    if not isinstance(data, list):
        raise ValueError("the models list has no data")
    return [{"id": m.get("id"), "expiration_date": m.get("expiration_date")}
            for m in data if isinstance(m, dict) and m.get("id")]


def load_models(cache_path: str, now: Optional[float] = None,
                fetch: Callable[[], List[Dict[str, Any]]] = _fetch) -> Optional[List[Dict[str, Any]]]:
    """OpenRouter's model list: the cache when it is under a day old, else fetched (and cached); on a failed fetch
    the cache however old; None when there is neither."""
    now = time.time() if now is None else now
    cached: Optional[Dict[str, Any]] = None
    try:
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if not isinstance(cached, dict) or not isinstance(cached.get("models"), list):
            cached = None
    except (OSError, ValueError):
        cached = None
    if cached is not None and now - float(cached.get("fetched_at") or 0) < CACHE_MAX_AGE_SEC:
        return cached["models"]
    try:
        models = fetch()
    except Exception as exc:  # noqa: BLE001 - offline is normal; the check just has less to go on
        log.info("OpenRouter model list not fetched (%s: %s)%s.", type(exc).__name__, exc,
                 "; using the cached one" if cached else "; model expiry not checked")
        return cached["models"] if cached else None
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": now, "models": models}, f)
        os.replace(tmp, cache_path)
    except OSError as exc:
        log.info("OpenRouter model list not cached: %s", exc)
    return models


def _as_date(value: Any) -> Optional[date]:
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def findings(models: List[Dict[str, Any]], configured: List[Tuple[str, str]], today: date,
             days: int = WARN_DAYS) -> List[Dict[str, Any]]:
    """Each configured model that expires within *days* (``expires``, ``days_left``) or is not listed (``gone``)."""
    by_id = {str(m.get("id")): m for m in models}
    out: List[Dict[str, Any]] = []
    for role, model in configured:
        listed = by_id.get(model)
        if listed is None:
            out.append({"role": role, "model": model, "gone": True})
            continue
        expires = _as_date(listed.get("expiration_date"))
        if expires is not None and (expires - today).days <= days:
            out.append({"role": role, "model": model, "expires": expires.isoformat(),
                        "days_left": (expires - today).days})
    return out


def check(box_settings: Mapping[str, Any], cache_path: Optional[str] = None, now: Optional[float] = None,
          fetch: Callable[[], List[Dict[str, Any]]] = _fetch, days: int = WARN_DAYS) -> List[Dict[str, Any]]:
    """Log a WARNING per finding and return them; [] when nothing is due or nothing could be checked. Never raises."""
    try:
        configured = configured_models(box_settings)
        if not configured:
            return []
        if cache_path is None:
            from . import paths  # noqa: PLC0415

            cache_path = os.path.join(paths.state_dir(), CACHE_NAME)
        models = load_models(cache_path, now=now, fetch=fetch)
        if models is None:
            return []
        today = datetime.fromtimestamp(time.time() if now is None else now).date()
        found = findings(models, configured, today, days)
        for f in found:
            if f.get("gone"):
                log.warning("Model %s (%s) is not in OpenRouter's model list: retired or misspelled. Calls to it "
                            "will fail; set another model in box.yaml.", f["model"], f["role"])
            else:
                log.warning("Model %s (%s) is retired by OpenRouter on %s (%s). Set another model in box.yaml "
                            "before then.", f["model"], f["role"], f["expires"],
                            f"in {f['days_left']} days" if f["days_left"] >= 0 else "already past")
        return found
    except Exception as exc:  # noqa: BLE001 - a guard must never stop the box
        log.info("Model expiry check failed: %s: %s", type(exc).__name__, exc)
        return []


def start_check(box_settings: Mapping[str, Any], every_sec: float = CACHE_MAX_AGE_SEC) -> threading.Thread:
    """:func:`check` now and then once a day, on a daemon thread (the fetch may take seconds offline)."""
    settings = dict(box_settings)

    def loop() -> None:
        while True:
            check(settings)
            time.sleep(max(60.0, float(every_sec)))

    t = threading.Thread(target=loop, name="model-expiry", daemon=True)
    t.start()
    return t
