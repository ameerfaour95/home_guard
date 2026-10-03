# home_guard_project/box/brain/mode.py
"""Which mode the owner's assistant is in, and the one status line shown everywhere.

Guard inside the owner's alert hours, Assistant outside them - decided in code
from the same hours the guard loop reads, never by the model. A start hour
equal to the end hour means guarding all day (as ``inference.in_alert_window``).
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence, Tuple

from ..inference import in_alert_window
from .i18n import t

GUARD = "guard"
ASSISTANT = "assistant"


def resolve_mode(now_ts: float, start_hour: int, end_hour: int) -> str:
    hour = dt.datetime.fromtimestamp(now_ts).hour
    return GUARD if in_alert_window(hour, start_hour, end_hour) else ASSISTANT


def _next_hour(now_ts: float, hour: int) -> float:
    moment = dt.datetime.fromtimestamp(now_ts).replace(hour=hour, minute=0, second=0, microsecond=0)
    if moment.timestamp() <= now_ts:
        moment += dt.timedelta(days=1)
    return moment.timestamp()


def _last_hour(now_ts: float, hour: int) -> float:
    moment = dt.datetime.fromtimestamp(now_ts).replace(hour=hour, minute=0, second=0, microsecond=0)
    if moment.timestamp() > now_ts:
        moment -= dt.timedelta(days=1)
    return moment.timestamp()


def mode_ends_at(now_ts: float, start_hour: int, end_hour: int) -> Optional[float]:
    """When the current mode ends: the end hour while guarding, the start hour otherwise. None when guarding all day."""
    if start_hour == end_hour:
        return None
    if resolve_mode(now_ts, start_hour, end_hour) == GUARD:
        return _next_hour(now_ts, end_hour)
    return _next_hour(now_ts, start_hour)


def mode_started_at(now_ts: float, start_hour: int, end_hour: int) -> Optional[float]:
    """When the current mode began. None when guarding all day."""
    if start_hour == end_hour:
        return None
    if resolve_mode(now_ts, start_hour, end_hour) == GUARD:
        return _last_hour(now_ts, start_hour)
    return _last_hour(now_ts, end_hour)


def hhmm(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def status_line(mode: str, now_ts: float, start_hour: int, end_hour: int,
                paused: Sequence[Tuple[str, float]] = (), offline: Sequence[str] = (), quiet_log: bool = True) -> str:
    """The line the app and Telegram show. *paused* is ``[(camera, until_ts)]``; *offline* camera names."""
    ends = mode_ends_at(now_ts, start_hour, end_hour)
    if mode == GUARD:
        head = f"🛡️ Guarding until {hhmm(ends)}" if ends else "🛡️ Guarding all day"
    else:
        head = ("💬 Assistant · quiet logging" if quiet_log else "💬 Assistant · not recording") + (
            f" · guarding from {hhmm(ends)}" if ends else "")
    parts = [head]
    parts += [f"{name} alerts paused until {hhmm(until)}" for name, until in paused]
    parts += [f"⚠️ {name} offline" for name in offline]
    return " · ".join(parts)


def switch_announcement(mode: str, now_ts: float, start_hour: int, end_hour: int,
                        live: int, total: int, lang: str, quiet_log: bool = True) -> str:
    """The one Telegram message sent when the mode changes."""
    ends = mode_ends_at(now_ts, start_hour, end_hour)
    if mode == GUARD:
        until = t("guard_until", lang, time=hhmm(ends)) if ends else ""
        return t("guard_started", lang, until=until, live=live, total=total)
    nxt = t("guard_next", lang, time=hhmm(ends)) if ends else ""
    return t("guard_ended" if quiet_log else "guard_ended_no_log", lang, next=nxt)
