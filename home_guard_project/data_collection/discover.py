"""
Camera discovery module — find RTSP cameras on the local network.

Strategies (tried in order):
  1. ONVIF WS-Discovery  →  ONVIF Media service  →  RTSP URIs
  2. Subnet port scan    →  credentials + channel probing
  3. Manual entry

Runnable standalone:  py home_guard_project/data_collection/discover.py
"""

from __future__ import annotations

import logging
import os
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as urlquote

import cv2
import yaml

log = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
_CAMERAS_YAML = os.path.join(_DIR, "cameras.yaml")

# Common RTSP path templates per manufacturer.
# {ch} = channel number, {stream} = stream index (1=main, 2=sub)
_RTSP_PATTERNS: List[Dict[str, str]] = [
    {"name": "Hikvision/ISAPI",   "tpl": "/Streaming/Channels/{ch}01"},
    {"name": "Hikvision/unicast", "tpl": "/unicast/c{ch}/s{stream}/live"},
    {"name": "Dahua",             "tpl": "/cam/realmonitor?channel={ch}&subtype={stream_0}"},
    {"name": "ONVIF generic",     "tpl": "/onvif/profile{stream}/media.smp"},
    {"name": "Generic /live",     "tpl": "/live/ch{ch:02d}_{stream_0}"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_ip() -> Optional[str]:
    """Best-effort detection of this machine's LAN IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _subnet_from_ip(ip: str) -> str:
    parts = ip.rsplit(".", 1)
    return f"{parts[0]}.0/24"


def _tcp_open(ip: str, port: int, timeout: float = 1.0) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.close()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: ONVIF WS-Discovery
# ─────────────────────────────────────────────────────────────────────────────

def onvif_discover(timeout: float = 5.0) -> List[Dict[str, Any]]:
    """
    Send a WS-Discovery multicast probe and return found ONVIF devices.

    Each entry: {"ip": str, "port": int, "xaddrs": [str], "scopes": [str]}
    Returns empty list if the WSDiscovery library is not installed.
    """
    try:
        from WSDiscovery import WSDiscovery  # type: ignore[import-untyped]
    except ImportError:
        log.warning("WSDiscovery not installed — skipping ONVIF discovery")
        return []

    devices: List[Dict[str, Any]] = []
    wsd = WSDiscovery()
    wsd.start()
    try:
        services = wsd.searchServices(timeout=timeout)
        for svc in services:
            xaddrs = svc.getXAddrs()
            scopes = [str(s) for s in (svc.getScopes() or [])]
            is_onvif = any("onvif" in s.lower() for s in scopes) or any(
                "onvif" in x.lower() for x in xaddrs
            )
            if not is_onvif:
                continue
            for xaddr in xaddrs:
                m = re.search(r"https?://([^:/]+)(?::(\d+))?", xaddr)
                if m:
                    devices.append({
                        "ip": m.group(1),
                        "port": int(m.group(2)) if m.group(2) else 80,
                        "xaddrs": xaddrs,
                        "scopes": scopes,
                    })
    finally:
        wsd.stop()

    seen = set()
    unique: List[Dict[str, Any]] = []
    for d in devices:
        if d["ip"] not in seen:
            seen.add(d["ip"])
            unique.append(d)
    return unique


def onvif_get_rtsp_uris(
    ip: str, port: int, user: str, password: str,
) -> List[Dict[str, Any]]:
    """
    Connect to a device via ONVIF and retrieve RTSP stream URIs for all
    media profiles.

    Returns list of {"name": str, "uri": str, "resolution": (w, h)}.
    """
    try:
        from onvif import ONVIFCamera  # type: ignore[import-untyped]
    except ImportError:
        log.warning("onvif-zeep-async not installed — cannot query ONVIF media")
        return []

    results: List[Dict[str, Any]] = []
    try:
        cam = ONVIFCamera(ip, port, user, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        for prof in profiles:
            token = prof.token
            name = prof.Name or token
            stream_setup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            uri_resp = media.GetStreamUri({
                "StreamSetup": stream_setup,
                "ProfileToken": token,
            })
            uri = str(uri_resp.Uri)
            # Embed credentials into the URI
            uri = re.sub(
                r"rtsp://",
                f"rtsp://{urlquote(user, safe='')}:{urlquote(password, safe='')}@",
                uri, count=1,
            )
            w = h = 0
            try:
                enc = prof.VideoEncoderConfiguration
                if enc and enc.Resolution:
                    w, h = enc.Resolution.Width, enc.Resolution.Height
            except Exception:
                pass
            results.append({"name": name, "uri": uri, "resolution": (w, h)})
    except Exception as exc:
        log.warning("ONVIF media query failed for %s:%d — %s", ip, port, exc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Subnet port scan
# ─────────────────────────────────────────────────────────────────────────────

def subnet_scan(
    subnet: Optional[str] = None, port: int = 554, timeout: float = 0.8,
    workers: int = 64,
) -> List[str]:
    """
    Scan a /24 subnet for hosts with *port* open.
    Returns list of IP addresses.
    """
    if subnet is None:
        local_ip = _get_local_ip()
        if local_ip is None:
            log.error("Cannot determine local IP for subnet scan")
            return []
        subnet = _subnet_from_ip(local_ip)

    base = subnet.split("/")[0].rsplit(".", 1)[0]
    targets = [f"{base}.{i}" for i in range(1, 255)]

    found: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_tcp_open, ip, port, timeout): ip for ip in targets}
        for fut in as_completed(futs):
            if fut.result():
                found.append(futs[fut])

    found.sort(key=lambda ip: list(map(int, ip.split("."))))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Channel probing
# ─────────────────────────────────────────────────────────────────────────────

def validate_stream(url: str, timeout: float = 5.0) -> Tuple[bool, int, int]:
    """
    Try to open an RTSP URL and read one frame.
    Returns (success, width, height).
    """
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000",
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
    except Exception:
        pass
    ok = False
    w = h = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            ok = True
            break
        time.sleep(0.1)
    cap.release()
    return ok, w, h


def probe_rtsp_channels(
    ip: str,
    port: int,
    user: str,
    password: str,
    max_channels: int = 16,
    stream: int = 1,
    timeout: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Probe common RTSP URL patterns on *ip* for channels 1..*max_channels*.

    Returns list of {"channel": int, "url": str, "pattern": str, "w": int, "h": int}
    for every working stream.
    """
    cred = f"{urlquote(user, safe='')}:{urlquote(password, safe='')}@"
    found: List[Dict[str, Any]] = []

    # First, detect which pattern works by testing channel 1 against all patterns
    working_pattern = None
    for pat in _RTSP_PATTERNS:
        path = pat["tpl"].format(
            ch=1, stream=stream, stream_0=stream - 1,
        )
        url = f"rtsp://{cred}{ip}:{port}{path}"
        log.info("  Trying %s pattern: %s", pat["name"], path)
        ok, w, h = validate_stream(url, timeout=timeout)
        if ok:
            log.info("  >> %s works!  (%dx%d)", pat["name"], w, h)
            working_pattern = pat
            found.append({
                "channel": 1, "url": url,
                "pattern": pat["name"], "w": w, "h": h,
            })
            break

    if working_pattern is None:
        log.warning("No working RTSP pattern found for %s:%d", ip, port)
        return []

    # Probe remaining channels with the working pattern
    for ch in range(2, max_channels + 1):
        path = working_pattern["tpl"].format(
            ch=ch, stream=stream, stream_0=stream - 1,
        )
        url = f"rtsp://{cred}{ip}:{port}{path}"
        ok, w, h = validate_stream(url, timeout=timeout)
        if ok:
            log.info("  Channel %d: OK (%dx%d)", ch, w, h)
            found.append({
                "channel": ch, "url": url,
                "pattern": working_pattern["name"], "w": w, "h": h,
            })
        else:
            log.info("  Channel %d: no stream", ch)

    return found


# ─────────────────────────────────────────────────────────────────────────────
# cameras.yaml writer
# ─────────────────────────────────────────────────────────────────────────────

def write_cameras_yaml(
    cameras: Dict[str, str], path: str = _CAMERAS_YAML,
) -> None:
    data = {
        "cameras": cameras,
    }
    header = (
        "# ──────────────────────────────────────────────────────────────────────────────\n"
        "#  Camera RTSP streams — DO NOT COMMIT (contains credentials)\n"
        "# ──────────────────────────────────────────────────────────────────────────────\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    log.info("Wrote %d cameras to %s", len(cameras), path)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return val or default


def _prompt_yn(msg: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    val = _prompt(f"{msg} [{hint}]")
    if not val:
        return default
    return val.lower().startswith("y")


def interactive_discover() -> Dict[str, str]:
    """Run the full interactive camera discovery flow. Returns name→url map."""
    cameras: Dict[str, str] = {}

    print("\n══════════════════════════════════════════════")
    print("  Camera Discovery")
    print("══════════════════════════════════════════════\n")

    print("  [1] Auto-discover cameras (ONVIF + network scan)")
    print("  [2] Enter camera details manually")
    choice = _prompt("\nChoose", "1")

    if choice == "1":
        cameras = _auto_discover_flow()
    else:
        cameras = _manual_flow()

    return cameras


def _auto_discover_flow() -> Dict[str, str]:
    cameras: Dict[str, str] = {}

    # Phase 1: ONVIF
    print("\n── ONVIF Discovery ──")
    print("Sending WS-Discovery probe (5 seconds)...")
    devices = onvif_discover(timeout=5.0)

    if devices:
        print(f"\nFound {len(devices)} ONVIF device(s):")
        for i, d in enumerate(devices, 1):
            print(f"  [{i}] {d['ip']}:{d['port']}")

        user = _prompt("\nONVIF username", "admin")
        password = _prompt("ONVIF password")

        for d in devices:
            print(f"\nQuerying ONVIF media on {d['ip']}...")
            uris = onvif_get_rtsp_uris(d["ip"], d["port"], user, password)
            if uris:
                print(f"  Found {len(uris)} stream(s):")
                for j, u in enumerate(uris, 1):
                    res = f"{u['resolution'][0]}x{u['resolution'][1]}" if u["resolution"][0] else "?"
                    print(f"    [{j}] {u['name']}  ({res})  {u['uri'][:80]}...")

                for j, u in enumerate(uris, 1):
                    if _prompt_yn(f"  Include stream [{j}] {u['name']}?"):
                        name = _prompt(f"    Friendly name for [{j}]", u["name"])
                        name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
                        cameras[name] = u["uri"]
            else:
                print("  No streams returned. Will try port scanning...")

    if not cameras:
        # Phase 2: Subnet scan
        print("\n── Network Scan ──")
        local_ip = _get_local_ip()
        if local_ip:
            subnet = _subnet_from_ip(local_ip)
            print(f"Local IP: {local_ip}  →  scanning {subnet} for RTSP (port 554)...")
        else:
            subnet = _prompt("Enter subnet to scan (e.g. 192.168.1.0/24)")

        hosts = subnet_scan(subnet=subnet, port=554)
        if not hosts:
            print("No hosts with port 554 found.")
            print("Falling back to manual entry.\n")
            return _manual_flow()

        print(f"\nFound {len(hosts)} host(s) with port 554 open:")
        for i, ip in enumerate(hosts, 1):
            print(f"  [{i}] {ip}")

        sel = _prompt("Select host number (or 'all')", "1")
        if sel.lower() == "all":
            selected = hosts
        else:
            try:
                selected = [hosts[int(sel) - 1]]
            except (IndexError, ValueError):
                selected = [hosts[0]]

        user = _prompt("RTSP username", "admin")
        password = _prompt("RTSP password")

        for ip in selected:
            print(f"\nProbing RTSP channels on {ip}...")
            streams = probe_rtsp_channels(ip, 554, user, password)
            if not streams:
                print(f"  No working channels found on {ip}")
                continue
            print(f"  Found {len(streams)} channel(s):")
            for s in streams:
                print(f"    Channel {s['channel']}: {s['w']}x{s['h']}  ({s['pattern']})")

            for s in streams:
                if _prompt_yn(f"  Include channel {s['channel']}?"):
                    name = _prompt(
                        f"    Friendly name for channel {s['channel']}",
                        f"camera_{s['channel']}",
                    )
                    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
                    cameras[name] = s["url"]

    return cameras


def _manual_flow() -> Dict[str, str]:
    cameras: Dict[str, str] = {}
    print("\n── Manual Camera Entry ──")
    print("Enter full RTSP URLs, or provide NVR details to auto-probe channels.\n")

    mode = _prompt("[1] Enter NVR IP + probe channels  [2] Paste full RTSP URLs", "1")

    if mode == "1":
        ip = _prompt("NVR IP address", "192.168.68.106")
        port = int(_prompt("RTSP port", "554"))
        user = _prompt("Username", "admin")
        password = _prompt("Password")

        print(f"\nProbing channels on {ip}:{port}...")
        streams = probe_rtsp_channels(ip, port, user, password)
        if streams:
            print(f"Found {len(streams)} channel(s):")
            for s in streams:
                print(f"  Channel {s['channel']}: {s['w']}x{s['h']}  ({s['pattern']})")
            for s in streams:
                if _prompt_yn(f"  Include channel {s['channel']}?"):
                    name = _prompt(
                        f"    Friendly name", f"camera_{s['channel']}",
                    )
                    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
                    cameras[name] = s["url"]
        else:
            print("No channels found. Enter URLs manually.")
            mode = "2"

    if mode == "2":
        print("Enter cameras one per line.  Empty name to finish.\n")
        while True:
            name = _prompt("Camera name (empty to stop)")
            if not name:
                break
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
            url = _prompt(f"  RTSP URL for '{name}'")
            if url:
                cameras[name] = url

    return cameras


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Check for existing cameras.yaml
    if os.path.isfile(_CAMERAS_YAML):
        with open(_CAMERAS_YAML, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        cams = existing.get("cameras", {})
        if cams:
            print(f"\nExisting cameras.yaml found with {len(cams)} camera(s):")
            for name, url in cams.items():
                # Mask credentials in display
                display = re.sub(r"://[^@]+@", "://<credentials>@", str(url))
                print(f"  {name}: {display}")
            if _prompt_yn("\nUse these cameras?"):
                print("Keeping existing cameras.yaml.")
                return

    cameras = interactive_discover()

    if not cameras:
        print("\nNo cameras configured. Exiting.")
        sys.exit(1)

    print(f"\n── Summary: {len(cameras)} camera(s) ──")
    for name, url in cameras.items():
        display = re.sub(r"://[^@]+@", "://<credentials>@", url)
        print(f"  {name}: {display}")

    if _prompt_yn("\nWrite to cameras.yaml?"):
        write_cameras_yaml(cameras)
        print("Done! cameras.yaml written.")
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
