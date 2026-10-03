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
import hashlib
import json
import logging
import os
import re
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote as urlquote

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
ZONES_PATH = os.path.join(os.path.dirname(CAMERAS_PATH), "zones.yaml")
CAMERA_ALERTS_PATH = os.path.join(os.path.dirname(CAMERAS_PATH), "camera_alerts.yaml")
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


def apply_changes(changes: Dict[str, Any], path: str = CAMERAS_PATH, zones_path: str = ZONES_PATH,
                  restart: bool = True, alerts_path: Optional[str] = None,
                  aliases_path: Optional[str] = None) -> Dict[str, Any]:
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
    renames: Dict[str, str] = {}

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
        if new != old:
            renames[old] = new

    # Cameras not mentioned keep their current bucket (unless a rename took the name).
    for name, url in active.items():
        if name not in handled and name not in final_names:
            new_active[name] = url
            final_names.add(name)
    for name, url in disabled.items():
        if name not in handled and name not in final_names:
            new_disabled[name] = url
            final_names.add(name)

    if renames:
        from ..data_collection.zones import remap_zones  # noqa: PLC0415

        # Each watch zone follows its camera, swaps and chains included. The zone
        # file covers old and new names while cameras.yaml is written, so a
        # failure part-way never leaves a camera unmasked.
        remap_zones(renames, zones_path, between=lambda: _write_cameras(new_active, new_disabled, path))
        try:
            from .camera_alerts import remap_camera_alerts  # noqa: PLC0415

            # Each camera's alert choice follows it too. A failure here only puts the
            # renamed camera on the house default - never a camera left unwatched.
            remap_camera_alerts(renames, alerts_path or CAMERA_ALERTS_PATH)
        except Exception as exc:  # noqa: BLE001
            log.warning("Alert choices not carried over the rename: %s", exc)
    else:
        _write_cameras(new_active, new_disabled, path)
    if renames:
        try:
            from .brain.aliases import ALIASES_PATH, remap_aliases  # noqa: PLC0415

            remap_aliases(renames, aliases_path or ALIASES_PATH)
        except Exception as exc:  # noqa: BLE001 - a name file must never block a camera change
            log.warning("Camera aliases not carried over the rename: %s", exc)
    if restart:
        _restart_running_mode()
    return {"active": sorted(new_active), "disabled": sorted(new_disabled)}


def _restart_running_mode() -> None:
    try:
        from . import control  # noqa: PLC0415

        control.request_restart()
    except Exception:  # noqa: BLE001
        pass


def _known_camera(name: str, cameras_path: str) -> str:
    raw = _read_cameras_raw(cameras_path)
    known = {**dict(raw.get("disabled") or {}), **dict(raw.get("cameras") or {})}
    if name not in known:
        raise ValueError(f"unknown camera: {name!r}")
    return name


def zone_rows(cameras_path: str = CAMERAS_PATH, zones_path: str = ZONES_PATH) -> Dict[str, Any]:
    """Every camera with its watch zone: ``{"cameras": [{"name", "points"}]}``; ``points`` is [] for the whole picture."""
    from ..data_collection.zones import load_zones  # noqa: PLC0415

    raw = _read_cameras_raw(cameras_path)
    zones = load_zones(zones_path)
    names = list(dict(raw.get("cameras") or {})) + list(dict(raw.get("disabled") or {}))
    return {"cameras": [{"name": n, "points": [[x, y] for x, y in zones.get(n, [])]} for n in names]}


def camera_alert_rows(cameras_path: str = CAMERAS_PATH, alerts_path: str = CAMERA_ALERTS_PATH,
                      box_path: Optional[str] = None) -> Dict[str, Any]:
    """The house default and every camera's own choice (``None`` = the house default).

    ``{"house": ["person"], "cameras": [{"name", "alert_on": [...] | None}]}``
    """
    from .boxconfig import BOX_YAML, BoxConfigError, load_box_settings  # noqa: PLC0415
    from .camera_alerts import DEFAULT, load_camera_alerts, load_camera_sensitivity, parse_types  # noqa: PLC0415
    from .inference import AlertSettings  # noqa: PLC0415

    try:
        box = load_box_settings(box_path or BOX_YAML)
    except (BoxConfigError, OSError):
        box = {}
    try:
        house = parse_types(box.get("alert_on") or ",".join(DEFAULT))
    except ValueError:
        house = DEFAULT
    try:
        house_sensitivity = AlertSettings.from_box_settings(box).thresholds()
    except (TypeError, ValueError):
        house_sensitivity = AlertSettings().thresholds()
    raw = _read_cameras_raw(cameras_path)
    own = load_camera_alerts(alerts_path)
    own_sensitivity = load_camera_sensitivity(alerts_path)
    names = list(dict(raw.get("cameras") or {})) + list(dict(raw.get("disabled") or {}))
    return {"house": list(house), "house_sensitivity": house_sensitivity,
            "cameras": [{"name": n, "alert_on": list(own[n]) if n in own else None,
                         "sensitivity": dict(own_sensitivity[n]) if n in own_sensitivity else None}
                        for n in names]}


def set_camera_alerts_command(camera: str, types: Optional[str], cameras_path: str = CAMERAS_PATH,
                              alerts_path: str = CAMERA_ALERTS_PATH) -> Dict[str, Any]:
    """Give a camera its own alert types, or put it back on the house default (*types* None).

    No restart: inference mode re-reads the choices within seconds.
    """
    from .camera_alerts import clear_camera_alerts, set_camera_alerts  # noqa: PLC0415

    name = _known_camera(str(camera), cameras_path)
    if types is None:
        clear_camera_alerts(name, alerts_path)
        return {"camera": name, "alert_on": None}
    return {"camera": name, "alert_on": list(set_camera_alerts(name, types, alerts_path))}


def set_camera_sensitivity_command(camera: str, values: Optional[str], cameras_path: str = CAMERAS_PATH,
                                   alerts_path: str = CAMERA_ALERTS_PATH) -> Dict[str, Any]:
    """Give a camera its own detector certainty per type (``person=0.5,vehicle=0.8``), or the house values (None).

    No restart: inference mode re-reads it within seconds.
    """
    from .camera_alerts import clear_camera_sensitivity, set_camera_sensitivity  # noqa: PLC0415

    name = _known_camera(str(camera), cameras_path)
    if values is None:
        clear_camera_sensitivity(name, alerts_path)
        return {"camera": name, "sensitivity": None}
    return {"camera": name, "sensitivity": set_camera_sensitivity(name, values, alerts_path)}


def set_zone(camera: str, points_text: str, cameras_path: str = CAMERAS_PATH,
             zones_path: str = ZONES_PATH, restart: bool = True) -> Dict[str, Any]:
    """Store a camera's watch zone from ``x,y;x,y;...`` and ask the running mode to restart."""
    from ..data_collection.zones import parse_points, save_zone  # noqa: PLC0415

    name = _known_camera(str(camera), cameras_path)
    corners = save_zone(name, parse_points(points_text), zones_path)
    if restart:
        _restart_running_mode()
    return {"camera": name, "points": [[x, y] for x, y in corners]}


def clear_zone_command(camera: str, cameras_path: str = CAMERAS_PATH,
                       zones_path: str = ZONES_PATH, restart: bool = True) -> Dict[str, Any]:
    """Watch the whole picture again for this camera."""
    from ..data_collection.zones import clear_zone  # noqa: PLC0415

    name = _known_camera(str(camera), cameras_path)
    clear_zone(name, zones_path)
    if restart:
        _restart_running_mode()
    return {"camera": name, "points": []}


def looks_blank(frame: Any) -> bool:
    """True for the flat grey picture a decoder shows before the stream's first full frame.

    An H.265 stream opened mid-way has nothing to build its first pictures
    from, and they come out as an even mid-grey with a few specks.
    """
    import numpy as np  # noqa: PLC0415

    small = np.asarray(frame)[::8, ::8]
    return float(small.std()) < 8.0 and 100.0 < float(small.mean()) < 156.0


def _grab_snapshot(url: str, out_path: str, width: int = 960, wait_sec: float = 6.0) -> bool:
    import time  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        frame = None
        deadline = time.time() + wait_sec
        # Read on until a real picture arrives: the first frames can be blank grey.
        while time.time() < deadline:
            ok, f = cap.read()
            if not ok or f is None:
                continue
            frame = f
            if not looks_blank(f):
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


def digest_response(user: str, realm: str, password: str, method: str, uri: str, nonce: str) -> str:
    """The ``response`` value of HTTP/RTSP Digest authentication (RFC 2069, no qop)."""
    md5 = lambda text: hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: E731, S324 - the protocol's own hash
    return md5(f"{md5(f'{user}:{realm}:{password}')}:{nonce}:{md5(f'{method}:{uri}')}")


def _rtsp_describe(host: str, port: int, uri: str, authorization: str = "", timeout: float = 5.0) -> str:
    request = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n"
    if authorization:
        request += f"Authorization: {authorization}\r\n"
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        conn.sendall((request + "\r\n").encode("utf-8"))
        return conn.recv(2048).decode("latin-1", "replace")


def rtsp_login_check(host: str, port: int, user: str, password: str, timeout: float = 3.0,
                     paths: Sequence[str] = ()) -> str:
    """Does the device accept this login? ``accepted``, ``refused``, ``silent`` (no answer) or
    ``unknown`` (it answers, but asks for a kind of login this does not speak).

    A few quick requests, before the slow search for the device's stream
    address. Most cameras check the login before they look at the address, so
    a made-up address is enough: 401 with our login means the login is wrong.
    Some recorders answer 401 to any address they do not serve, even with the
    right login (the owner's own recorder does); a refusal therefore counts
    only once real stream addresses (*paths*) are refused as well.
    """
    try:
        state = _login_state(host, port, user, password, f"rtsp://{host}:{port}/", timeout)
        if state != "refused":
            return state
        for path in paths:
            if _login_state(host, port, user, password, f"rtsp://{host}:{port}{path}", timeout) == "accepted":
                return "accepted"
        return "refused"
    except (OSError, IndexError):
        return "silent"


def _login_state(host: str, port: int, user: str, password: str, uri: str, timeout: float) -> str:
    """One DESCRIBE of *uri*, answering the device's login challenge: accepted, refused or unknown."""
    reply = _rtsp_describe(host, port, uri, timeout=timeout)
    if " 401 " not in reply.splitlines()[0]:
        return "accepted"                      # it asks for no login at all, or serves the address
    authorization = _authorization(reply, user, password, uri)
    if not authorization:
        return "unknown"
    reply = _rtsp_describe(host, port, uri, authorization, timeout=timeout)
    return "refused" if " 401 " in reply.splitlines()[0] else "accepted"


def first_stream_paths() -> List[str]:
    """One real stream address per known recorder family (channel 1, main stream), for the login check."""
    from home_guard_project.data_collection import discover

    return [pattern["tpl"].format(ch=1, stream=0, stream_0=0) for pattern in discover._RTSP_PATTERNS]


def _authorization(challenge: str, user: str, password: str, uri: str) -> str:
    """The Authorization header that answers a device's 401 reply (Digest preferred, else Basic), or ""."""
    digest = re.search(r'WWW-Authenticate:\s*Digest\s+(.*)', challenge, re.IGNORECASE)
    if digest:
        fields = dict(re.findall(r'(\w+)="([^"]*)"', digest.group(1)))
        realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
        answer = digest_response(user, realm, password, "DESCRIBE", uri, nonce)
        return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{answer}"'
    if re.search(r"WWW-Authenticate:\s*Basic", challenge, re.IGNORECASE):
        return "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return ""


class DeviceSilent(Exception):
    """The device stopped answering (or never did)."""


def rtsp_stream_state(
    host: str, port: int, user: str, password: str, path: str, timeout: float = 3.0,
) -> Optional[bool]:
    """Ask the device to describe the stream at *path*, logging in if it must.

    True: it answers 200. False: it answers something else. None: it does not answer.
    """
    uri = f"rtsp://{host}:{port}{path}"
    try:
        reply = _rtsp_describe(host, port, uri, timeout=timeout)
        if " 401 " in reply.splitlines()[0]:
            authorization = _authorization(reply, user, password, uri)
            if not authorization:
                return False
            reply = _rtsp_describe(host, port, uri, authorization, timeout=timeout)
        return " 200 " in reply.splitlines()[0]
    except (OSError, IndexError):
        return None


def rtsp_stream_exists(host: str, port: int, user: str, password: str, path: str, timeout: float = 5.0) -> bool:
    """True if the device answers 200 when asked to describe the stream at *path*."""
    return bool(rtsp_stream_state(host, port, user, password, path, timeout))


def rtsp_channels(
    host: str,
    port: int,
    user: str,
    password: str,
    patterns: List[Dict[str, str]],
    max_channels: int,
    stream: int,
    timeout: float = 5.0,
) -> Tuple[Optional[Dict[str, str]], List[Tuple[int, str]]]:
    """Which channels of a recorder have a stream: ``(pattern, [(channel, path)])`` for the
    first address pattern that any channel answers to.

    A describe request takes milliseconds, so every channel can be asked. Opening
    each stream to see whether it works takes seconds, and the older search gave
    up when channel 1 was empty, which is normal on a recorder.
    """
    silent = 0
    for pattern in patterns:
        hits = []
        for channel in range(1, max_channels + 1):
            path = pattern["tpl"].format(ch=channel, stream=stream, stream_0=stream - 1)
            state = rtsp_stream_state(host, port, user, password, path, timeout)
            if state is None:
                # No answer at all. Every further question would wait out the same
                # timeout: fifty of them is four minutes spent on a dead device.
                silent += 1
                if silent >= 2:
                    raise DeviceSilent(host)
                continue
            silent = 0
            if state:
                hits.append((channel, path))
        if hits:
            return pattern, hits
    return None, []


def _probe_host(host: str, port: int, user: str, password: str) -> List[Dict[str, Any]]:
    from home_guard_project.data_collection import discover

    cfg = discover._load_discovery_config()
    try:
        pattern, hits = rtsp_channels(host, port, user, password, discover._RTSP_PATTERNS,
                                      cfg["max_channels"], cfg["stream"])
    except DeviceSilent:
        log.warning("  %s does not answer; skipped.", host)
        return []
    if hits:
        log.info("  %s: %d channel(s) answer (%s): %s", host, len(hits), pattern["name"],
                 ", ".join(str(channel) for channel, _ in hits))
        login = f"{urlquote(user, safe='')}:{urlquote(password, safe='')}@"
        urls = [f"rtsp://{login}{host}:{port}{path}" for _, path in hits]
        # Opening a stream to read its size takes a second or more: open a few at a time.
        with ThreadPoolExecutor(max_workers=min(4, len(urls))) as pool:
            opened = list(pool.map(
                lambda url: discover.validate_stream(url, timeout=max(cfg["probe_timeout_sec"], 10.0)), urls))
        found = []
        for (channel, _), url, (ok, width, height) in zip(hits, urls, opened):
            if ok:
                log.info("  Channel %d: OK (%dx%d)", channel, width, height)
                found.append({"channel": channel, "url": url, "pattern": pattern["name"], "w": width, "h": height})
            else:
                log.info("  Channel %d: answers, but no picture arrived", channel)
        return found
    # A device that does not answer describe requests this way: the slower search that opens each stream.
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
        refused = extra.get("login_refused") or []
        if refused:
            print(f"{len(refused)} device(s) answered but refused this login: {', '.join(refused)}. "
                  "Use the cameras' own user name and password.")
        else:
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

    sub.add_parser("zones", help="List every camera with its watch zone (empty = whole picture).")
    setz = sub.add_parser("set-zone", help="Store a camera's watch zone; everything outside it is blacked out.")
    setz.add_argument("--camera", required=True)
    setz.add_argument("--points", required=True,
                      help="Corners as x,y;x,y;x,y in 0-1 picture fractions, 3 to 32 of them, no spaces (quote the value in PowerShell).")
    clrz = sub.add_parser("clear-zone", help="Watch the whole picture again for a camera.")
    clrz.add_argument("--camera", required=True)
    sub.add_parser("camera-alerts", help="List the house's alert types and each camera's own choice.")
    seta = sub.add_parser("set-camera-alerts", help="Choose what one camera alerts about (person/vehicle/animal).")
    seta.add_argument("--camera", required=True)
    which = seta.add_mutually_exclusive_group(required=True)
    which.add_argument("--on", help="person, vehicle, animal - one or more, comma-separated, no spaces.")
    which.add_argument("--default", action="store_true", help="Use the house default (alert_on in box.yaml).")
    sens = sub.add_parser("set-camera-sensitivity",
                          help="How sure the detector must be on one camera, per type (person/vehicle/animal).")
    sens.add_argument("--camera", required=True)
    pick = sens.add_mutually_exclusive_group(required=True)
    pick.add_argument("--values", help="type=certainty pairs from 0.05 to 0.95, e.g. person=0.5,vehicle=0.8 (no spaces).")
    pick.add_argument("--default", action="store_true", help="Use the house values (conf_* / inference_conf in box.yaml).")

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

    if args.command in ("camera-alerts", "set-camera-alerts", "set-camera-sensitivity"):
        try:
            if args.command == "camera-alerts":
                result = camera_alert_rows(CAMERAS_PATH, CAMERA_ALERTS_PATH)
            elif args.command == "set-camera-sensitivity":
                result = set_camera_sensitivity_command(args.camera, None if args.default else args.values,
                                                        CAMERAS_PATH, CAMERA_ALERTS_PATH)
            else:
                result = set_camera_alerts_command(args.camera, None if args.default else args.on,
                                                   CAMERAS_PATH, CAMERA_ALERTS_PATH)
        except Exception as exc:  # noqa: BLE001 - the app needs a plain error, never a traceback
            message = str(exc) if isinstance(exc, ValueError) else f"could not change the alert choice: {exc}"
            print(json.dumps({"error": message}) if args.json else f"Error: {message}")
            sys.exit(1)
        if args.json:
            print(json.dumps(result, indent=2))
        elif "sensitivity" in result and "cameras" not in result:
            own = result["sensitivity"]
            print(f"  {result['camera']:<22} "
                  f"{', '.join(f'{k} {v}' for k, v in own.items()) if own else 'house sensitivity'}")
        else:
            if "house" in result:
                print(f"  house default          {'+'.join(result['house'])}")
            for row in result.get("cameras", [{"name": result.get("camera"), "alert_on": result.get("alert_on")}]):
                print(f"  {row['name']:<22} {'+'.join(row['alert_on']) if row['alert_on'] else 'house default'}")
        return

    if args.command in ("zones", "set-zone", "clear-zone"):
        try:
            if args.command == "zones":
                result = zone_rows()
            elif args.command == "set-zone":
                result = set_zone(args.camera, args.points)
            else:
                result = clear_zone_command(args.camera)
        except ValueError as exc:
            if args.json:
                print(json.dumps({"error": str(exc)}))
            else:
                print(f"Error: {exc}")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001 - the app needs a plain error, never a traceback
            log.exception("zone command failed")
            message = f"could not change the zone: {exc}"
            if args.json:
                print(json.dumps({"error": message}))
            else:
                print(f"Error: {message}")
            sys.exit(1)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for row in result.get("cameras", [result]):
                corners = row.get("points") or []
                print(f"  {row['camera' if 'camera' in row else 'name']:<22} "
                      f"{'whole picture' if not corners else str(len(corners)) + ' corners'}")
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

    found, extra = search(args.user, password)
    _finish(found, args, extra)


def lan_addresses() -> List[str]:
    """This machine's addresses on local networks, where cameras can be.

    Not the VPN (Tailscale's 100.64.0.0/10), not link-local, not loopback. The
    default-route trick (``discover._get_local_ip``) picks the wrong interface
    for a moment after the box changes network, which made a search right
    after the Wi-Fi step scan nothing.
    """
    found: List[str] = []
    try:
        candidates = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        candidates = []
    for ip in candidates:
        first, second = (int(x) for x in ip.split(".")[:2])
        private = first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168)
        if private and ip not in found:
            found.append(ip)
    return found


SETTLE_SCANS = 3          # scans of an empty network before giving up ...
SETTLE_WAIT_SEC = 4.0     # ... this long apart: ARP and a fresh Wi-Fi link need a few seconds


def rtsp_hosts(addresses: Optional[Sequence[str]] = None, sleep: Any = None) -> Dict[int, List[str]]:
    """Devices that answer on each camera port, on every local network this machine is on.

    A network that shows nothing is scanned again a few seconds later: right
    after the box joins a Wi-Fi, the first scan comes back empty.
    """
    import time

    from home_guard_project.data_collection import discover

    sleep = sleep or time.sleep
    addresses = list(addresses) if addresses is not None else (lan_addresses() or [discover._get_local_ip() or ""])
    subnets = [discover._subnet_from_ip(ip) for ip in addresses if ip]
    jobs = [(subnet, port) for subnet in subnets for port in RTSP_PORTS]
    hosts: Dict[int, List[str]] = {port: [] for port in RTSP_PORTS}
    for attempt in range(SETTLE_SCANS):
        if attempt:
            log.info("No device answered yet; the network may still be settling. Scanning again...")
            sleep(SETTLE_WAIT_SEC)
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
            results = list(pool.map(lambda job: discover.subnet_scan(subnet=job[0], port=job[1]), jobs))
        for (subnet, port), answering in zip(jobs, results):
            hosts[port] = sorted(set(hosts[port]) | set(answering), key=lambda ip: list(map(int, ip.split("."))))
        if any(hosts.values()):
            break
    return hosts


def search(user: str, password: str) -> Tuple[Found, Dict[str, Any]]:
    """Find every camera this login opens: ``(found, facts for the report)``.

    Kept short on purpose, because an installer is waiting on it: the ports are
    scanned together, the login is checked on all devices at once, and a
    device that refuses the login or does not answer is not asked again.
    """
    from home_guard_project.data_collection import discover

    addresses = lan_addresses()
    targets = [(host, port) for port, hosts in rtsp_hosts(addresses or None).items() for host in hosts]
    local_ip = addresses[0] if addresses else discover._get_local_ip()
    if not targets:
        log.warning("No device answers on the camera port (scanned %s from %s). Is the box on the cameras' network?",
                    ", ".join(discover._subnet_from_ip(ip) for ip in addresses) or "no local network", local_ip)
        return {}, {"local_ip": local_ip, "hosts_tried": [], "devices_found": 0,
                    "login_refused": [], "no_answer": []}

    paths = first_stream_paths()
    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
        states = dict(zip(
            (host for host, _ in targets),
            pool.map(lambda target: rtsp_login_check(target[0], target[1], user, password, paths=paths), targets),
        ))
    refused = [host for host, _ in targets if states[host] == "refused"]
    silent = [host for host, _ in targets if states[host] == "silent"]
    if refused:
        log.warning("%d of %d device(s) refused this login: %s", len(refused), len(targets), ", ".join(refused))
    if silent:
        log.warning("%d of %d device(s) did not answer: %s", len(silent), len(targets), ", ".join(silent))

    found = {host: _probe_host(host, port, user, password)
             for host, port in targets if states[host] not in ("refused", "silent")}
    return found, {
        "local_ip": local_ip,
        "hosts_tried": [host for host, _ in targets],
        "devices_found": len(targets),
        "login_refused": refused,
        "no_answer": silent,
    }


if __name__ == "__main__":
    main()
