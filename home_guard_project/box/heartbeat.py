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
from typing import Any, Dict, Optional

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


def build_heartbeat(
    site: str,
    live_dir: str,
    outbox_dir: str,
    alive_path: str,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    now = time.time() if now is None else now
    live = _clip_times(live_dir)
    outbox = _clip_times(outbox_dir)

    cameras: Dict[str, Dict[str, Any]] = {}
    for camera in sorted(set(live) | set(outbox)):
        times = live.get(camera, []) + outbox.get(camera, [])
        cameras[camera] = {"clips_waiting": len(times), "newest_clip_utc": _iso_utc(max(times))}

    all_times = [t for per_cam in (*live.values(), *outbox.values()) for t in per_cam]
    return {
        "site": site,
        "host": socket.gethostname(),
        "local_ip": _local_ip(),
        "time_utc": _iso_utc(now),
        "collector_running": collector_alive(alive_path, now),
        "disk_free_gb": _disk_free_gb(live_dir),
        "clips_live": sum(len(t) for t in live.values()),
        "clips_outbox": sum(len(t) for t in outbox.values()),
        "newest_clip_utc": _iso_utc(max(all_times)) if all_times else None,
        "cameras": cameras,
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
