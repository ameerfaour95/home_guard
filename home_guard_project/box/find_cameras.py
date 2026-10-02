"""
Find cameras without prompts, for a box with no screen (works the same over
Ethernet or Wi-Fi — the box only needs to be on the cameras' network).

Usage:
    python -m home_guard_project.box.find_cameras scan
    python -m home_guard_project.box.find_cameras probe --host 192.168.1.50 --user admin --prefix house2
    python -m home_guard_project.box.find_cameras probe --host 192.168.1.50 --user admin --prefix house2 --write

``scan`` needs no password: it lists devices that answer on the camera port.
``probe`` logs in to one recorder or camera and lists its channels; with
``--write`` it saves them to data_collection/cameras.yaml. The password comes
from the HG_CAMERA_PASSWORD environment variable (preferred) or --password.

This reuses the probing code of data_collection/discover.py, which stays the
interactive way to do the same thing.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Any, Dict, List

log = logging.getLogger("box.cameras")

PASSWORD_ENV = "HG_CAMERA_PASSWORD"
RTSP_PORTS = (554, 8554)


def camera_names(found: List[Dict[str, Any]], prefix: str) -> Dict[str, str]:
    """Map ``<prefix>_ch<N>`` to each found stream URL, so names never collide between sites."""
    return {f"{prefix}_ch{f['channel']}": f["url"] for f in found}


def redact(url: str) -> str:
    """Hide the user and password of an RTSP URL for printing."""
    return re.sub(r"(rtsp://)[^/@\s]+@", r"\1<user>:<password>@", url)


def _scan() -> None:
    from home_guard_project.data_collection import discover

    local_ip = discover._get_local_ip()
    print(f"This machine: {local_ip}")

    for port in RTSP_PORTS:
        hosts = sorted(discover.subnet_scan(port=port))
        print(f"Devices answering on port {port}: {hosts if hosts else 'none'}")

    devices = discover.onvif_discover()
    if devices:
        for d in devices:
            print(f"ONVIF device: {d.get('ip')}:{d.get('port')}")
    else:
        print("ONVIF devices: none answered")


def _probe(args: argparse.Namespace) -> None:
    from home_guard_project.data_collection import discover

    password = args.password or os.environ.get(PASSWORD_ENV)
    if not password:
        log.error("No password: set %s or pass --password.", PASSWORD_ENV)
        sys.exit(2)

    cfg = discover._load_discovery_config()
    found = discover.probe_rtsp_channels(
        args.host,
        args.port,
        args.user,
        password,
        max_channels=cfg["max_channels"],
        stream=cfg["stream"],
        timeout=cfg["probe_timeout_sec"],
        consecutive_fail_stop=cfg["consecutive_fail_stop"],
    )
    if not found:
        print("No working stream found. Check the address, the login, and that RTSP is enabled on the device.")
        sys.exit(1)

    cameras = camera_names(found, args.prefix)
    for name, f in zip(cameras, found):
        print(f"  {name:<20} {f['w']}x{f['h']}  {redact(f['url'])}")

    if args.write:
        discover.write_cameras_yaml(cameras)
        print(f"Saved {len(cameras)} camera(s). The collector picks them up within a minute.")
    else:
        print("Nothing saved. Add --write to save these to cameras.yaml.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find cameras on this network without prompts.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="List devices that answer on the camera port (no password needed).")

    probe = sub.add_parser("probe", help="Log in to one recorder/camera and list its channels.")
    probe.add_argument("--host", required=True, help="Address of the recorder or camera.")
    probe.add_argument("--port", type=int, default=554)
    probe.add_argument("--user", default="admin")
    probe.add_argument("--password", default=None, help=f"Prefer the {PASSWORD_ENV} environment variable.")
    probe.add_argument("--prefix", required=True, help="Site name put in front of camera names, e.g. house2.")
    probe.add_argument("--write", action="store_true", help="Save the result to cameras.yaml.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

    if args.command == "scan":
        _scan()
    else:
        _probe(args)


if __name__ == "__main__":
    main()
