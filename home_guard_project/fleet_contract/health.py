"""Deterministic health verdicts from the legacy heartbeat snapshot."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from .legacy import Heartbeat


def _utc(value: datetime) -> datetime:
    # Legacy UTC fields and callers may omit the timezone marker.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def camera_stale(newest: Optional[datetime], now) -> bool:
    return newest is None or _utc(now) - _utc(newest) >= timedelta(hours=24)


def verdict(hb: Optional[Heartbeat], now: datetime, alert_hours: Optional[tuple[int, int]] = None) -> tuple[str, list[dict]]:
    """Return the worst applicable condition and all contributing reasons.

    ``alert_hours`` is reserved: the legacy contract specifies no schedule-based
    exceptions to these thresholds. Missing optional values cannot establish a
    fault, except a camera with no newest clip, which is explicitly stale.
    """
    if hb is None:
        return "unknown", [{"code": "no_heartbeat", "message": "No heartbeat received", "severity": "unknown"}]

    reasons: list[dict] = []

    def add(code: str, message: str, severity: str):
        reasons.append({"code": code, "message": message, "severity": severity})

    if hb.time_utc is not None:
        age = _utc(now) - _utc(hb.time_utc)
        if age > timedelta(minutes=90):
            minutes = int(age.total_seconds() // 60)
            elapsed = f"{minutes // 60} h" if minutes >= 120 else f"{minutes} min"
            add("heartbeat_old", f"Last heard {elapsed} ago", "offline")
    if hb.stopped:
        add("stopped_by_owner", "Engine stopped by the owner", "warning")
    elif hb.collector_running is False:
        add("engine_down", "Collector is not running", "critical")
    if hb.disk_free_gb is not None:
        if hb.disk_free_gb < 20:
            add("disk_low", f"Only {hb.disk_free_gb:g} GB of disk space free", "critical")
        elif hb.disk_free_gb < 50:
            add("disk_low", f"Only {hb.disk_free_gb:g} GB of disk space free", "warning")
    quiet = sorted(camera for camera, newest in hb.cameras.items() if camera_stale(newest, now))
    if quiet:
        add("camera_quiet", "No clip for at least 24 h or no clip recorded: " + ", ".join(quiet), "warning")
    if hb.clips_outbox > 500:
        add("upload_backlog", f"{hb.clips_outbox} clips in the upload outbox", "warning")

    severity_order = {"healthy": 0, "unknown": 1, "warning": 2, "critical": 3, "offline": 4}
    worst = max((reason["severity"] for reason in reasons), key=severity_order.__getitem__, default="healthy")
    return worst, reasons
