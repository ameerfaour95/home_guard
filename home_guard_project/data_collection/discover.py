"""
Camera discovery module — find RTSP cameras on the local network.

Strategies (tried in order):
  1. ONVIF WS-Discovery  →  ONVIF Media service  →  RTSP URIs
  2. ARP table scan       →  port 554 check on known hosts
  3. Subnet port scan     →  credentials + channel probing
  4. Manual entry

Runnable standalone:  py home_guard_project/data_collection/discover.py
"""

from __future__ import annotations

import logging
import os
import platform
import re
import socket
import subprocess
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


def _os_open(path: str) -> None:
    """Open a file with the default OS application."""
    system = platform.system().lower()
    try:
        if system == "windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


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
# Strategy 1.5: ARP table scan
# ─────────────────────────────────────────────────────────────────────────────

def _ping_sweep(subnet_base: str, workers: int = 128) -> None:
    """Fire-and-forget pings to populate the ARP table.

    Sends a single ping to every host in the /24 subnet. We don't care about
    the result — the OS ARP table gets populated as a side effect.
    """
    is_win = platform.system().lower() == "windows"
    flag = "-n" if is_win else "-c"
    timeout_flag = "-w" if is_win else "-W"
    timeout_val = "200" if is_win else "1"

    def _ping_one(ip: str) -> None:
        try:
            subprocess.run(
                ["ping", flag, "1", timeout_flag, timeout_val, ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except Exception:
            pass

    targets = [f"{subnet_base}.{i}" for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_ping_one, targets))


def arp_scan(local_ip: Optional[str] = None) -> List[str]:
    """Read the OS ARP table and return IPs on the same /24 subnet.

    On Windows: ``arp -a``
    On Linux/macOS: ``arp -a`` or ``ip neigh``

    Returns a sorted list of IPs (excluding the local machine and broadcast).
    """
    if local_ip is None:
        local_ip = _get_local_ip()
    if local_ip is None:
        return []

    subnet_base = local_ip.rsplit(".", 1)[0]

    # Ping sweep to populate the ARP table with fresh entries
    log.info("Ping sweep on %s.0/24 to populate ARP table...", subnet_base)
    _ping_sweep(subnet_base)

    # Read ARP table
    try:
        out = subprocess.check_output(
            ["arp", "-a"], text=True, timeout=10,
        )
    except Exception:
        log.warning("arp -a failed")
        return []

    # Parse IPs — works for both Windows and Unix output formats
    # Windows: "  192.168.68.106     xx-xx-xx  dynamic"
    # Unix:    "? (192.168.68.106) at xx:xx:xx [ether] on eth0"
    ips: List[str] = []
    for m in re.finditer(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", out):
        ip = m.group(1)
        if ip.startswith(subnet_base + ".") and ip != local_ip and not ip.endswith(".255"):
            ips.append(ip)

    # Deduplicate and sort
    ips = sorted(set(ips), key=lambda ip: list(map(int, ip.split("."))))
    return ips


def arp_rtsp_scan(local_ip: Optional[str] = None, port: int = 554) -> List[str]:
    """Discover RTSP hosts by combining ARP table with port 554 check.

    Much faster than a blind subnet scan because it only tests hosts that
    actually exist on the network (according to the ARP table).
    """
    candidates = arp_scan(local_ip)
    if not candidates:
        return []

    log.info("ARP found %d host(s), checking port %d...", len(candidates), port)
    found: List[str] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = {pool.submit(_tcp_open, ip, port, 1.5): ip for ip in candidates}
        for fut in as_completed(futs):
            if fut.result():
                found.append(futs[fut])

    found.sort(key=lambda ip: list(map(int, ip.split("."))))
    return found


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
        from wsdiscovery import WSDiscovery  # type: ignore[import-untyped]
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
    stream: int = 0,
    timeout: float = 5.0,
    consecutive_fail_stop: int = 2,
) -> List[Dict[str, Any]]:
    """
    Probe common RTSP URL patterns on *ip* for channels 1..*max_channels*.

    Stops early after *consecutive_fail_stop* consecutive failures to avoid
    wasting time on non-existent channels.

    Returns list of {"channel": int, "url": str, "pattern": str, "w": int, "h": int}
    for every working stream.
    """
    cred = f"{urlquote(user, safe='')}:{urlquote(password, safe='')}@"
    found: List[Dict[str, Any]] = []

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

    fails = 0
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
            fails = 0
        else:
            log.info("  Channel %d: no stream", ch)
            fails += 1
            if fails >= consecutive_fail_stop:
                log.info("  %d consecutive failures — stopping probe.", fails)
                break

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Live preview
# ─────────────────────────────────────────────────────────────────────────────

def _has_gui() -> bool:
    """Check if OpenCV was built with GUI (highgui) support."""
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


_GUI_AVAILABLE: Optional[bool] = None


def _preview_stream(url: str, title: str = "Preview", timeout: float = 8.0) -> None:
    """Show a live preview of an RTSP stream.

    Tries the OpenCV GUI first. If the build lacks highgui support, falls
    back to saving a snapshot and opening it with the OS image viewer.
    """
    global _GUI_AVAILABLE
    if _GUI_AVAILABLE is None:
        _GUI_AVAILABLE = _has_gui()

    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000",
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
    except Exception:
        pass

    if _GUI_AVAILABLE:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        deadline = time.time() + timeout
        shown = False
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                if max(frame.shape[:2]) > 720:
                    scale = 720.0 / max(frame.shape[:2])
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                cv2.imshow(title, frame)
                shown = True
                deadline = time.time() + 30
            key = cv2.waitKey(30) & 0xFF
            if key != 255 and shown:
                break
        cap.release()
        cv2.destroyWindow(title)
        cv2.waitKey(1)
    else:
        # Fallback: grab one frame, save as temp image, open with OS viewer
        frame = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
                break
            time.sleep(0.1)
        cap.release()

        if frame is None:
            print("    (could not grab a preview frame)")
            return

        import tempfile
        fd, img_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(img_path, frame)
        print(f"    Snapshot saved: {img_path}")

        _os_open(img_path)
        _prompt("    Press Enter here when done viewing...")


def _select_streams(streams: List[Dict[str, Any]]) -> Dict[str, str]:
    """Walk through discovered streams, preview each one, and let the user
    name the ones they want to keep."""
    cameras: Dict[str, str] = {}
    for s in streams:
        ch = s["channel"]
        url = s["url"]
        print(f"\n  Previewing channel {ch} ({s['w']}x{s['h']})...")
        print("  >> Press any key in the preview window to continue.")
        _preview_stream(url, title=f"Channel {ch} — press any key")

        if _prompt_yn(f"  Include channel {ch}?"):
            name = _prompt(
                f"    Friendly name for channel {ch}",
                f"camera_{ch}",
            )
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
            cameras[name] = url
    return cameras


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


def _load_discovery_config() -> Dict[str, Any]:
    """Load discovery-related settings from config.yaml (best-effort)."""
    config_path = os.path.join(_DIR, "config.yaml")
    defaults = {
        "max_channels": 16,
        "consecutive_fail_stop": 2,
        "probe_timeout_sec": 5.0,
        "stream": 0,
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        disc = data.get("discovery", {})
        if isinstance(disc, dict):
            st = str(disc.get("stream_type", "main")).strip().lower()
            return {
                "max_channels": int(disc.get("max_channels", defaults["max_channels"])),
                "consecutive_fail_stop": int(disc.get("consecutive_fail_stop", defaults["consecutive_fail_stop"])),
                "probe_timeout_sec": float(disc.get("probe_timeout_sec", defaults["probe_timeout_sec"])),
                "stream": 0 if st == "main" else 1,
            }
    except Exception:
        pass
    return defaults


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


_TIER_LABEL = {0: "high", 1: "medium", 2: "low"}
_LABEL_TIER = {v: k for k, v in _TIER_LABEL.items()}


def _parse_channel_and_tier(uri: str) -> Optional[Tuple[str, int]]:
    """Extract (channel_id, tier) from a Hikvision-style unicast URI.

    Example: rtsp://.../unicast/c3/s0/live -> ("c3", 0)
    Returns None if the URI doesn't match (e.g. NVR mosaic/oddball streams).
    """
    m = re.search(r"/unicast/c(\d+)/s([0-2])/", uri)
    if not m:
        return None
    return f"c{m.group(1)}", int(m.group(2))


def _group_streams_by_channel(
    uris: List[Dict[str, Any]],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Group ONVIF stream URIs by physical channel and quality tier.

    Returns {channel_id: {tier: stream_dict}}. Streams that don't match the
    unicast/c<N>/s<T> pattern are skipped (e.g. NVR mosaic on c0).
    """
    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for u in uris:
        parsed = _parse_channel_and_tier(u["uri"])
        if parsed is None:
            continue
        ch, tier = parsed
        if ch == "c0":
            continue
        grouped.setdefault(ch, {})[tier] = u
    return grouped


def _pick_stream_for_channel(
    tiers: Dict[int, Dict[str, Any]], requested_tier: int,
) -> Dict[str, Any]:
    """Pick one stream for a channel given the user's requested tier.

    Prefers the highest-quality tier that is <= requested (smaller tier number
    = higher quality). If none qualify, falls back upward to the closest
    available tier above the request so no camera is dropped.
    """
    candidates_at_or_below = [t for t in tiers if t >= requested_tier]
    if candidates_at_or_below:
        return tiers[min(candidates_at_or_below)]
    return tiers[max(tiers)]


def _ask_quality_tier(
    grouped: Dict[str, Dict[int, Dict[str, Any]]],
) -> int:
    """Show which tiers are available across all channels and ask the user."""
    available: Dict[int, int] = {}
    for ch_tiers in grouped.values():
        for tier in ch_tiers:
            available[tier] = available.get(tier, 0) + 1

    if not available:
        return 0

    print("\n  Available quality tiers:")
    tier_order = sorted(available.keys())
    for tier in tier_order:
        label = _TIER_LABEL.get(tier, f"s{tier}")
        print(f"    {label:<6} (s{tier}) — on {available[tier]} channel(s)")

    default_label = _TIER_LABEL.get(tier_order[0], f"s{tier_order[0]}")
    choice = _prompt(
        f"  Choose quality ({'/'.join(_TIER_LABEL[t] for t in tier_order)})",
        default_label,
    ).strip().lower()
    return _LABEL_TIER.get(choice, tier_order[0])


def _auto_discover_flow() -> Dict[str, str]:
    cameras: Dict[str, str] = {}
    disc = _load_discovery_config()

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
            if not uris:
                print("  No streams returned. Will try port scanning...")
                continue

            grouped = _group_streams_by_channel(uris)
            if not grouped:
                print(
                    "  No streams matched the unicast/c<N>/s<T> pattern. "
                    "Falling back to per-stream prompt."
                )
                for j, u in enumerate(uris, 1):
                    if _prompt_yn(f"  Include stream [{j}] {u['name']}?"):
                        name = _prompt(f"    Friendly name for [{j}]", u["name"])
                        name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
                        cameras[name] = u["uri"]
                continue

            print(
                f"  Found {sum(len(t) for t in grouped.values())} stream(s) "
                f"across {len(grouped)} physical camera(s)."
            )
            requested_tier = _ask_quality_tier(grouped)

            for ch in sorted(grouped.keys(), key=lambda c: int(c[1:])):
                stream = _pick_stream_for_channel(grouped[ch], requested_tier)
                w, h = stream["resolution"]
                actual_tier = _parse_channel_and_tier(stream["uri"])[1]
                tier_note = ""
                if actual_tier != requested_tier:
                    tier_note = (
                        f"  [only {_TIER_LABEL.get(actual_tier, f's{actual_tier}')} "
                        f"available]"
                    )
                res = f"{w}x{h}" if w else "?"
                print(f"\n  {ch}: picked s{actual_tier} ({res}){tier_note}")
                # Preview low-quality (medium/low tier) for naming, even if the
                # user picked high — high-bitrate streams take longer to open
                # and the preview is just for identification.
                preview_uri = stream["uri"]
                preview_tier = grouped[ch].get(2) or grouped[ch].get(1) or stream
                preview_uri = preview_tier["uri"]
                print(f"    Opening preview for {ch}...")
                print("    >> Press any key in the preview window to continue.")
                _preview_stream(preview_uri, title=f"{ch} — press any key")
                default_name = f"camera_{ch}"
                name = _prompt(
                    f"    Friendly name for {ch}", default_name,
                )
                name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
                cameras[name] = stream["uri"]

    if not cameras:
        # Phase 2: ARP table scan (fast — only tests known devices)
        print("\n── ARP Network Scan ──")
        local_ip = _get_local_ip()
        if local_ip:
            print(f"Local IP: {local_ip}")
            print("Pinging subnet + reading ARP table to find devices...")
            hosts = arp_rtsp_scan(local_ip, port=554)
        else:
            hosts = []

        # Phase 3: Fall back to full subnet TCP scan
        if not hosts:
            print("ARP scan found no RTSP hosts. Trying full subnet scan...")
            if local_ip:
                subnet = _subnet_from_ip(local_ip)
                print(f"Scanning {subnet} for port 554 (this may take a moment)...")
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
            streams = probe_rtsp_channels(
                ip, 554, user, password,
                max_channels=disc["max_channels"],
                stream=disc["stream"],
                timeout=disc["probe_timeout_sec"],
                consecutive_fail_stop=disc["consecutive_fail_stop"],
            )
            if not streams:
                print(f"  No working channels found on {ip}")
                continue
            print(f"  Found {len(streams)} channel(s):")
            for s in streams:
                print(f"    Channel {s['channel']}: {s['w']}x{s['h']}  ({s['pattern']})")

            cameras.update(_select_streams(streams))

    return cameras


def _manual_flow() -> Dict[str, str]:
    cameras: Dict[str, str] = {}
    disc = _load_discovery_config()
    print("\n── Manual Camera Entry ──")
    print("Enter full RTSP URLs, or provide NVR details to auto-probe channels.\n")

    mode = _prompt("[1] Enter NVR IP + probe channels  [2] Paste full RTSP URLs", "1")

    if mode == "1":
        ip = _prompt("NVR IP address", "192.168.68.106")
        port = int(_prompt("RTSP port", "554"))
        user = _prompt("Username", "admin")
        password = _prompt("Password")

        print(f"\nProbing channels on {ip}:{port}...")
        streams = probe_rtsp_channels(
            ip, port, user, password,
            max_channels=disc["max_channels"],
            stream=disc["stream"],
            timeout=disc["probe_timeout_sec"],
            consecutive_fail_stop=disc["consecutive_fail_stop"],
        )
        if streams:
            print(f"Found {len(streams)} channel(s):")
            for s in streams:
                print(f"  Channel {s['channel']}: {s['w']}x{s['h']}  ({s['pattern']})")
            cameras.update(_select_streams(streams))
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
