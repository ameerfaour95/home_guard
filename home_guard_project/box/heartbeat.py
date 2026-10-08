"""Status report for a collector box, written to S3 so it can be checked remotely.

The payload holds counts and timestamps only — no URLs and no log text, so
camera credentials cannot leak through it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import socket
import time
from typing import Any, Dict, List, Optional

from .outbox import META_SUFFIX

STATUS_KEY = "_status/heartbeat.json"


def _iso_utc(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collector_alive(
    alive_path: str,
    now: Optional[float] = None,
    max_age_sec: float = 180.0,
) -> bool:
    """True if run_collector.sh touched *alive_path* within *max_age_sec*."""
    now = time.time() if now is None else now
    try:
        return now - os.path.getmtime(alive_path) <= max_age_sec
    except OSError:
        return False


def _clip_times(dataset_dir: str) -> Dict[str, list[float]]:
    """Map camera name -> meta mtimes for every clip under *dataset_dir*."""
    meta_root = os.path.join(dataset_dir, "meta")
    times: Dict[str, list[float]] = {}
    for dirpath, _, filenames in os.walk(meta_root):
        for name in filenames:
            if not name.endswith(META_SUFFIX):
                continue
            camera = os.path.relpath(dirpath, meta_root).replace("\\", "/").split("/")[0]
            times.setdefault(camera, []).append(os.path.getmtime(os.path.join(dirpath, name)))
    return times


def _local_ip() -> Optional[str]:
    """This machine's address on the house network (tells which network the box is on)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # UDP connect sends nothing; it only picks the route
            return s.getsockname()[0]
    except OSError:
        return None


def _disk_free_gb(path: str) -> float:
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    return round(shutil.disk_usage(path).free / 1e9, 1)


def camera_list(cameras_path: str, lang: str = "en", aliases_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The cameras in *cameras_path* (cameras.yaml) as ``{"id", "name", "channel", "enabled"}``, active ones first:
    *name* is what the owner reads (camera_names.display_name in *lang*, from the family's names in *aliases_path*,
    default brain/aliases' file), *channel* the ``_chN`` number as text or None. Never the stream URL (it holds the
    camera's login). [] when the file is missing or cannot be read."""
    from .camera_names import channel_of, display_name  # noqa: PLC0415

    try:
        import yaml  # noqa: PLC0415

        with open(cameras_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        active = list((raw.get("cameras") or {}).keys())
        disabled = [c for c in (raw.get("disabled") or {}).keys() if c not in active]
        aliases = None
        if aliases_path is not None:
            from .brain.aliases import load_aliases  # noqa: PLC0415

            aliases = load_aliases(aliases_path)
        return [{"id": str(cam), "name": display_name(str(cam), lang, aliases), "channel": channel_of(str(cam)),
                 "enabled": on}
                for cams, on in ((active, True), (disabled, False)) for cam in cams]
    except Exception:  # noqa: BLE001 - the heartbeat goes out without the list
        return []


def build_heartbeat(
    site: str,
    live_dir: str,
    outbox_dir: str,
    alive_path: str,
    now: Optional[float] = None,
    mode: Optional[str] = None,
    cameras_path: Optional[str] = None,
    lang: str = "en",
    aliases_path: Optional[str] = None,
) -> Dict[str, Any]:
    """The status payload. With *cameras_path* it also carries ``camera_list`` (see camera_list): ``cameras`` was
    already the per-camera clip counts, so the configured cameras with their owner-facing names go beside it."""
    now = time.time() if now is None else now
    live = _clip_times(live_dir)
    # The outbox has one dataset folder per site.
    outbox: Dict[str, list[float]] = {}
    if os.path.isdir(outbox_dir):
        for site_folder in os.listdir(outbox_dir):
            for camera, times in _clip_times(os.path.join(outbox_dir, site_folder)).items():
                outbox.setdefault(camera, []).extend(times)

    cameras: Dict[str, Dict[str, Any]] = {}
    for camera in sorted(set(live) | set(outbox)):
        times = live.get(camera, []) + outbox.get(camera, [])
        cameras[camera] = {"clips_waiting": len(times), "newest_clip_utc": _iso_utc(max(times))}

    all_times = [t for per_cam in (*live.values(), *outbox.values()) for t in per_cam]
    return {
        "site": site,
        "mode": mode,
        "host": socket.gethostname(),
        "local_ip": _local_ip(),
        "time_utc": _iso_utc(now),
        "collector_running": collector_alive(alive_path, now),
        "disk_free_gb": _disk_free_gb(live_dir),
        "clips_live": sum(len(t) for t in live.values()),
        "clips_outbox": sum(len(t) for t in outbox.values()),
        "newest_clip_utc": _iso_utc(max(all_times)) if all_times else None,
        "cameras": cameras,
        **({"camera_list": camera_list(cameras_path, lang, aliases_path)} if cameras_path else {}),
    }


def put_heartbeat(payload: Dict[str, Any], bucket: str, prefix: str, s3_client: Any = None) -> str:
    """Write *payload* to ``<prefix>/_status/heartbeat.json``. Returns the key."""
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    key = f"{prefix}/{STATUS_KEY}"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return key
