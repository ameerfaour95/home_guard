"""Deterministic health verdicts from the legacy heartbeat snapshot.

Camera warnings cover only the cameras the house has now. The box's heartbeat lists every camera folder still in its
14-day archive, so a site rename (``ameer_tes2_ch6`` -> ``ameer_week_0_1_ch6``) or a removed camera would otherwise
warn for two weeks about a camera that no longer exists. ``split_cameras`` decides current vs retired.
"""

from datetime import datetime, timedelta
from typing import Iterable, Optional

from .camera_names import channel_of, display_name, family_names
from .legacy import Heartbeat
from ._time import normalize_utc


def camera_label(camera: str, names: Optional[dict] = None) -> str:
    """The camera as staff read it: the owner's name when known, else "Camera N" from the channel by the box's rule
    (camera_names.display_name), else the id in words. Never the raw id.

    ``names`` is ONE house's id -> name (its camera_list). An id the box no longer has (a site rename:
    ameer_tes2_ch6 after ameer_week_0_1_ch6) takes the name of the one current camera on its channel, the box's own
    rule; never pass names from several houses, or one house's ch6 would borrow another's name."""
    aliases = {cam: [name] for cam, name in (names or {}).items() if name}
    if family_names(camera, aliases) or channel_of(camera):
        return display_name(camera, "en", aliases)
    text = camera.replace("_", " ")
    return text[:1].upper() + text[1:]


CURRENT_WINDOW = timedelta(hours=48)


def camera_list(body) -> Optional[list[dict]]:
    """The box's configured cameras from the heartbeat's ``camera_list: [{"id", "name", "channel", "enabled"}]``
    (box heartbeat.camera_list: cameras.yaml with the family's names via camera_names.display_name), each as
    ``{"id", "name", "enabled"}``; None from a box that does not send it (older boxes: the 48 h rule decides).
    ``cameras`` stays the per-camera clip times."""
    value = body.get("camera_list") if isinstance(body, dict) else None
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
            name = item.get("name")
            out.append({"id": item["id"], "name": name.strip() if isinstance(name, str) else "",
                        "enabled": item.get("enabled") is not False})
    return out


def listed_cameras(body) -> Optional[dict[str, str]]:
    """``{id: owner name}`` of the cameras the box has switched on now; None without a ``camera_list``."""
    cams = camera_list(body)
    return None if cams is None else {c["id"]: c["name"] for c in cams if c["enabled"]}


def split_cameras(cameras: dict[str, Optional[datetime]], now: datetime, site: str = "",
                  recent: Iterable[str] = (), listed: Optional[Iterable[str]] = None) -> tuple[list[str], list[str]]:
    """(current, retired) camera ids, each sorted.

    ``cameras`` maps every known id to its newest clip (None: never); ``recent`` names cameras with a clip in the last
    48 h from another source (the event index); ``listed`` is the box's own current list when it sends one, and then
    it alone decides. Otherwise a camera is current when it had a clip in the last 48 h. An older one is retired when
    a newer id took its channel (a site rename) or when its id carries another site's name; a quiet camera of this
    site stays current, so a dead camera keeps its warning.
    """
    ids = set(cameras) | set(recent)
    if listed is not None:
        listed = set(listed)
        return sorted(listed), sorted(ids - listed)
    now = normalize_utc(now)
    fresh = set(recent) | {c for c, newest in cameras.items()
                           if newest is not None and now - normalize_utc(newest) < CURRENT_WINDOW}
    fresh_channels = {channel_of(c) for c in fresh} - {None}
    current = set(fresh)
    for camera in ids - fresh:
        channel = channel_of(camera)
        if channel is None:
            current.add(camera)          # no channel to compare: never guess it away
        elif channel in fresh_channels:
            continue                     # a newer id has this channel now
        elif not site or camera.startswith(site + "_"):
            current.add(camera)
    return sorted(current), sorted(ids - current)


def _names(names: list[str]) -> str:
    return ", ".join(names)


def owner_name(camera: str, names: Optional[dict] = None) -> str:
    """The family's name for *camera* from one house's ``names`` (exact id, else its channel), or ""."""
    found = family_names(camera, {cam: [name] for cam, name in (names or {}).items() if name})
    return found[-1].strip() if found else ""


def camera_stale(newest: Optional[datetime], now) -> bool:
    return newest is None or normalize_utc(now) - normalize_utc(newest) >= timedelta(hours=24)


def verdict(hb: Optional[Heartbeat], now: datetime, alert_hours: Optional[tuple[int, int]] = None,
            cameras: Optional[Iterable[str]] = None, names: Optional[dict] = None) -> tuple[str, list[dict]]:
    """Return the worst applicable condition and all contributing reasons.

    ``alert_hours`` is reserved: the legacy contract specifies no schedule-based
    exceptions to these thresholds. A missing heartbeat time is offline; a
    camera with no newest clip is stale. Other missing values imply no fault.
    ``cameras`` limits the camera warnings to those ids (the current cameras; default every heartbeat camera) and
    ``names`` gives owner names (id -> name). Repeats are grouped: one warning lists every quiet camera.
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
    judged = hb.cameras if cameras is None else {c: hb.cameras.get(c) for c in cameras}
    for camera, newest in judged.items():
        name = camera_label(camera, names)
        if newest is None:
            never.append(name)
        elif camera_stale(newest, now):
            quiet.append((name, int((normalize_utc(now) - normalize_utc(newest)).total_seconds() // 3600)))
    if quiet:
        hours = sorted({h for _, h in quiet})
        since = f"{hours[0]} h" if len(hours) == 1 else f"{hours[0]}-{hours[-1]} h"
        add("camera_quiet", f"No clip for {since} from {_names([n for n, _ in quiet])} — check "
            + ("it has power and network" if len(quiet) == 1 else "they have power and network"), "warning")
    if never:
        add("camera_never", f"No clip recorded yet from {_names(never)} — check "
            + ("its login and stream on the box" if len(never) == 1 else "their login and stream on the box"), "warning")
    if hb.clips_outbox > 500:
        add("upload_backlog", f"{hb.clips_outbox} clips in the upload outbox", "warning")

    severity_order = {"healthy": 0, "unknown": 1, "warning": 2, "critical": 3, "offline": 4}
    worst = max((reason["severity"] for reason in reasons), key=severity_order.__getitem__, default="healthy")
    return worst, reasons
