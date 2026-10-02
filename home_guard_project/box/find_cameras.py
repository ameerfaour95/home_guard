"""
Find cameras without prompts, for a box with no screen (works the same over
Ethernet or Wi-Fi — the box only needs to be on the cameras' network).

Usage:
    python -m home_guard_project.box.find_cameras scan
    python -m home_guard_project.box.find_cameras auto  --user admin --prefix house2 --write
    python -m home_guard_project.box.find_cameras probe --host 192.168.1.50 --user admin --prefix house2 --write

``scan``  needs no password: it lists devices that answer on the camera port.
``auto``  scans, then logs in to every device found with one username and
          password and lists all their channels. This is the one-step command.
``probe`` does the same for a single address.

With ``--write`` the result is saved to data_collection/cameras.yaml. With
``--json`` the result is printed as one JSON object on stdout (progress goes to
stderr), so a setup program can call this and read the answer. The password
comes from the HG_CAMERA_PASSWORD environment variable, from --password-file
(base64, what the setup program uses), or from --password.

This reuses the probing code of data_collection/discover.py, which stays the
interactive way to do the same thing.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List

import yaml

log = logging.getLogger("box.cameras")

PASSWORD_ENV = "HG_CAMERA_PASSWORD"
RTSP_PORTS = (554, 8554)

# host -> streams found on it, as returned by discover.probe_rtsp_channels
Found = Dict[str, List[Dict[str, Any]]]


def _named(found: Found, prefix: str) -> List[tuple[str, str, Dict[str, Any]]]:
    """(name, host, stream) for every stream. Names carry the site so they never collide between houses."""
    hosts = [h for h in sorted(found) if found[h]]
    rows = []
    for host in hosts:
        # With several devices, channel numbers repeat: add the last number of the address.
        tag = f"_{host.rsplit('.', 1)[-1]}" if len(hosts) > 1 else ""
        for stream in found[host]:
            rows.append((f"{prefix}{tag}_ch{stream['channel']}", host, stream))
    return rows


def camera_names(found: Found, prefix: str) -> Dict[str, str]:
    """Camera name -> RTSP URL, the mapping cameras.yaml holds."""
    return {name: stream["url"] for name, _, stream in _named(found, prefix)}


def describe(found: Found, prefix: str) -> List[Dict[str, Any]]:
    """What was found, without URLs or credentials — safe to print or send to a setup program."""
    return [
        {"name": name, "host": host, "channel": s["channel"], "width": s["w"], "height": s["h"]}
        for name, host, s in _named(found, prefix)
    ]


def redact(url: str) -> str:
    """Hide the user and password of an RTSP URL for printing."""
    return re.sub(r"(rtsp://)[^/@\s]+@", r"\1<user>:<password>@", url)


# ── Camera confirmation (snapshots + apply), for the setup UI ────────────────
_NAME_RE = re.compile(r"^[a-z0-9_]+$")
CAMERAS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection", "cameras.yaml"))
_YAML_HEADER = (
    "# ──────────────────────────────────────────────────────────────────────────────\n"
    "#  Camera RTSP streams — DO NOT COMMIT (contains credentials)\n"
    "# ──────────────────────────────────────────────────────────────────────────────\n\n"
)


def _read_cameras_raw(path: str = CAMERAS_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_cameras(active: Dict[str, str], disabled: Dict[str, str], path: str = CAMERAS_PATH) -> None:
    data: Dict[str, Any] = {"cameras": active}
    if disabled:
        data["disabled"] = disabled
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_YAML_HEADER)
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    os.replace(tmp, path)


def apply_changes(changes: Dict[str, Any], path: str = CAMERAS_PATH) -> Dict[str, Any]:
    """Rewrite cameras.yaml from {"cameras":[{"name","new_name","enabled"}]}.

    A disabled camera is kept (recoverable) in a 'disabled:' section the loader
    ignores. Names are validated, duplicates refused, and the file is written
    atomically. On success the running program is asked to restart so it picks
    the cameras up. Returns the new {"active", "disabled"} name lists.
    """
    raw = _read_cameras_raw(path)
    active = dict(raw.get("cameras") or {})
    disabled = dict(raw.get("disabled") or {})
    known = {**disabled, **active}  # name -> url

    new_active: Dict[str, str] = {}
    new_disabled: Dict[str, str] = {}
    final_names: set = set()
    handled: set = set()

    for ch in changes.get("cameras", []):
        old = str(ch.get("name", ""))
        new = str(ch.get("new_name") or old)
        enabled = bool(ch.get("enabled", True))
        if old not in known:
            raise ValueError(f"unknown camera: {old!r}")
        if not _NAME_RE.match(new):
            raise ValueError(f"invalid camera name {new!r}: use lowercase letters, digits and underscores")
        if new in final_names:
            raise ValueError(f"duplicate camera name: {new!r}")
        final_names.add(new)
        handled.add(old)
        (new_active if enabled else new_disabled)[new] = known[old]

    # Cameras not mentioned keep their current bucket (unless a rename took the name).
    for name, url in active.items():
        if name not in handled and name not in final_names:
            new_active[name] = url
            final_names.add(name)
    for name, url in disabled.items():
        if name not in handled and name not in final_names:
            new_disabled[name] = url
            final_names.add(name)

    _write_cameras(new_active, new_disabled, path)
    try:
        from . import control  # noqa: PLC0415

        control.request_restart()
    except Exception:  # noqa: BLE001
        pass
    return {"active": sorted(new_active), "disabled": sorted(new_disabled)}


def _grab_snapshot(url: str, out_path: str, width: int = 960) -> bool:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        frame = None
        for _ in range(6):  # a few reads to get past the stream's buffer
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                break
        if frame is None:
            return False
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, max(1, int(h * width / w))))
        return bool(cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]))
    finally:
        cap.release()


def take_snapshots(out_dir: str, path: str = CAMERAS_PATH) -> Dict[str, Any]:
    """Save one <camera>.jpg (main stream, downscaled) per camera in cameras.yaml."""
    cameras = dict(_read_cameras_raw(path).get("cameras") or {})
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for name, url in cameras.items():
        out = os.path.join(out_dir, f"{name}.jpg")
        ok = False
        try:
            ok = _grab_snapshot(url, out)
        except Exception as exc:  # noqa: BLE001
            log.warning("snapshot failed for %s: %s", name, exc)
        results.append({"name": name, "file": out if ok else "", "ok": ok})
    return {"snapshots": results}


def _scan() -> Dict[str, Any]:
    from home_guard_project.data_collection import discover

    return {
        "local_ip": discover._get_local_ip(),
        "rtsp_hosts": {str(port): sorted(discover.subnet_scan(port=port)) for port in RTSP_PORTS},
        "onvif": [{"ip": d.get("ip"), "port": d.get("port")} for d in discover.onvif_discover()],
    }


def _probe_host(host: str, port: int, user: str, password: str) -> List[Dict[str, Any]]:
    from home_guard_project.data_collection import discover

    cfg = discover._load_discovery_config()
    return discover.probe_rtsp_channels(
        host,
        port,
        user,
        password,
        max_channels=cfg["max_channels"],
        stream=cfg["stream"],
        timeout=cfg["probe_timeout_sec"],
        consecutive_fail_stop=cfg["consecutive_fail_stop"],
    )


def decode_password_file(text: str) -> str:
    """The password from a --password-file: one line of base64 (UTF-8), so any character survives the trip."""
    return base64.b64decode(text.strip()).decode("utf-8")


def _password(args: argparse.Namespace) -> str:
    password = args.password or os.environ.get(PASSWORD_ENV)
    if args.password_file:
        with open(args.password_file, encoding="utf-8-sig") as f:
            password = decode_password_file(f.read())
    if not password:
        log.error("No password: set %s, or pass --password-file or --password.", PASSWORD_ENV)
        sys.exit(2)
    return password


def _finish(found: Found, args: argparse.Namespace, extra: Dict[str, Any]) -> None:
    """Save and report the cameras in *found*. Exit code 1 when none were found."""
    cameras = camera_names(found, args.prefix)
    saved = False
    if cameras and args.write:
        from home_guard_project.data_collection import discover

        discover.write_cameras_yaml(cameras)
        saved = True

    rows = describe(found, args.prefix)
    if args.json:
        print(json.dumps({**extra, "cameras": rows, "saved": saved}, indent=2))
    elif not rows:
        print("No working stream found. Check the login, and that RTSP is enabled on the device.")
    else:
        for r in rows:
            print(f"  {r['name']:<22} {r['host']:<16} {r['width']}x{r['height']}")
        if saved:
            print(f"Saved {len(rows)} camera(s). The collector picks them up within a minute.")
        else:
            print("Nothing saved. Add --write to save these to cameras.yaml.")

    if not rows:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find cameras on this network without prompts.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON on stdout.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="List devices that answer on the camera port (no password needed).")

    def add_login(p: argparse.ArgumentParser) -> None:
        p.add_argument("--user", default="admin")
        p.add_argument("--password", default=None, help=f"Prefer the {PASSWORD_ENV} environment variable.")
        p.add_argument("--password-file", default=None,
                       help="File holding the password as base64 (used by the setup program; it deletes the file).")
        p.add_argument("--prefix", required=True, help="Site name put in front of camera names, e.g. house2.")
        p.add_argument("--write", action="store_true", help="Save the result to cameras.yaml.")

    auto = sub.add_parser("auto", help="Scan, then log in to every device found and list all channels.")
    add_login(auto)

    probe = sub.add_parser("probe", help="Log in to one recorder/camera and list its channels.")
    probe.add_argument("--host", required=True, help="Address of the recorder or camera.")
    probe.add_argument("--port", type=int, default=554)
    add_login(probe)

    snaps = sub.add_parser("snapshots", help="Save one JPEG per camera (for the setup UI to show).")
    snaps.add_argument("--out", required=True, help="Directory to write <camera>.jpg files into.")

    applyp = sub.add_parser("apply", help="Rewrite cameras.yaml from a changes JSON (rename / enable / disable).")
    applyp.add_argument("--changes", required=True,
                        help='JSON file: {"cameras":[{"name","new_name","enabled"}]}.')

    args = parser.parse_args()
    # Progress goes to stderr so --json output on stdout stays parseable.
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    )

    if args.command == "scan":
        result = _scan()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"This machine: {result['local_ip']}")
            for port, hosts in result["rtsp_hosts"].items():
                print(f"Devices answering on port {port}: {hosts if hosts else 'none'}")
            onvif = [f"{d['ip']}:{d['port']}" for d in result["onvif"]]
            print(f"ONVIF devices: {onvif if onvif else 'none answered'}")
        return

    if args.command == "snapshots":
        result = take_snapshots(args.out)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for s in result["snapshots"]:
                print(f"  {s['name']:<22} {'ok' if s['ok'] else 'FAILED'}  {s['file']}")
        if not any(s["ok"] for s in result["snapshots"]):
            sys.exit(1)
        return

    if args.command == "apply":
        with open(args.changes, encoding="utf-8") as f:
            changes = json.load(f)
        try:
            result = apply_changes(changes)
        except ValueError as exc:
            if args.json:
                print(json.dumps({"error": str(exc)}))
            else:
                print(f"Error: {exc}")
            sys.exit(1)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Active cameras:  {result['active']}")
            print(f"Disabled:        {result['disabled']}")
        return

    password = _password(args)

    if args.command == "probe":
        found = {args.host: _probe_host(args.host, args.port, args.user, password)}
        _finish(found, args, {"hosts_tried": [args.host]})
        return

    scan = _scan()
    targets = [(host, int(port)) for port, hosts in scan["rtsp_hosts"].items() for host in hosts]
    if not targets:
        log.warning("No device answers on the camera port. Is the box on the cameras' network?")
    found = {host: _probe_host(host, port, args.user, password) for host, port in targets}
    _finish(found, args, {"local_ip": scan["local_ip"], "hosts_tried": [h for h, _ in targets]})


if __name__ == "__main__":
    main()
