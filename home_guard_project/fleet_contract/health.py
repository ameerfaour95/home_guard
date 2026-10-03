"""Deterministic health verdicts from the legacy heartbeat snapshot."""

from datetime import datetime, timedelta
from typing import Optional

from .legacy import Heartbeat
from ._time import normalize_utc


def camera_stale(newest: Optional[datetime], now) -> bool:
    return newest is None or normalize_utc(now) - normalize_utc(newest) >= timedelta(hours=24)


def verdict(hb: Optional[Heartbeat], now: datetime, alert_hours: Optional[tuple[int, int]] = None) -> tuple[str, list[dict]]:
    """Return the worst applicable condition and all contributing reasons.

    ``alert_hours`` is reserved: the legacy contract specifies no schedule-based
    exceptions to these thresholds. A missing heartbeat time is offline; a
    camera with no newest clip is stale. Other missing values imply no fault.
    """
    if hb is None:
        return "unknown", [{"code": "no_heartbeat", "message": "No heartbeat received", "severity": "unknown"}]

    reasons: list[dict] = []

    def add(code: str, message: str, severity: str):
        reasons.append({"code": code, "message": message, "severity": severity})

    if hb.time_utc is None:
        add("heartbeat_old", "Heartbeat has no valid time", "offline")
    else:
        age = normalize_utc(now) - normalize_utc(hb.time_utc)
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
    quiet = []
    never = []
    quietest_age = timedelta(0)
    for camera, newest in hb.cameras.items():
        name = camera.replace("_", " ")
        name = name[:1].upper() + name[1:]
        if newest is None:
            never.append(name)
        elif camera_stale(newest, now):
            quiet.append(name)
            quietest_age = max(quietest_age, normalize_utc(now) - normalize_utc(newest))
    if quiet:
        hours = int(quietest_age.total_seconds() // 3600)
        add("camera_quiet", f"No clip for {hours} h: " + ", ".join(quiet), "warning")
    if never:
        add("camera_never", "No clip recorded yet: " + ", ".join(never), "warning")
    if hb.clips_outbox > 500:
        add("upload_backlog", f"{hb.clips_outbox} clips in the upload outbox", "warning")

    severity_order = {"healthy": 0, "unknown": 1, "warning": 2, "critical": 3, "offline": 4}
    worst = max((reason["severity"] for reason in reasons), key=severity_order.__getitem__, default="healthy")
    return worst, reasons
