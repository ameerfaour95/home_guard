"""Find a recorder (or camera) that came back on a new network address, and point the box at it.

2026-10-19: a power cut also restarted the house's Wi-Fi mesh, which gave the recorder a new address (.106 -> .108).
cameras.yaml still named the old one, so the box saw no camera from Oct 18 06:58 until it was fixed by hand the next
morning, and the owner's "who is outside" got "no live video". It had happened before (.107 -> .109).

Every camera URL names its device's address. When every camera of one address has sent no picture for SILENT_SEC,
one background search runs (again every RETRY_SEC while they stay silent):

1. The old address still serves the first camera's stream: the device is there (booting, or a passing network
   problem), nothing to move.
2. Otherwise the local network is scanned on the same port, and every device that is not already in cameras.yaml is
   asked for EVERY stream path of that address with the same login. A device that serves them all is the same
   recorder. Exactly one such device: cameras.yaml is rewritten (the address in every URL, active and disabled; a
   copy of the old file is kept beside it) and the runner restarts the program, which then opens the new address.
   None, or more than one (two identical cameras with the same login could each match): nothing is changed.

Each step is logged in one ASCII line starting "Recorder search:". box.yaml ``find_moved_recorder: on | off``
(default on).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

log = logging.getLogger("box.recorder_finder")

SILENT_SEC = 120.0      # every camera of an address silent this long: look for it (a restarting recorder takes ~1 min)
RETRY_SEC = 300.0       # while they stay silent, look again this often (a recorder can take minutes to boot)
CHECK_EVERY_SEC = 10.0  # how often the guard loop's beat is looked at

Device = Tuple[str, int]   # (host, port)


def enabled(box_settings: Mapping[str, Any]) -> bool:
    """box.yaml ``find_moved_recorder`` (default on). YAML reads on/off as booleans; an unknown value is on."""
    value = (box_settings or {}).get("find_moved_recorder", True)
    if isinstance(value, bool):
        return value
    return str("" if value is None else value).strip().lower() not in ("off", "false", "no", "0")


def device_of(url: str) -> Optional[Tuple[Device, str, str, str]]:
    """``((host, port), user, password, path?query)`` of an RTSP URL; None when it is not one."""
    try:
        parts = urlsplit(str(url).strip())
        port = parts.port or 554
    except ValueError:
        return None
    if parts.scheme.lower() != "rtsp" or not parts.hostname:
        return None
    path = (parts.path or "/") + ("?" + parts.query if parts.query else "")
    return (parts.hostname, port), unquote(parts.username or ""), unquote(parts.password or ""), path


def group_by_device(cameras: Mapping[str, str]) -> Dict[Device, Dict[str, Any]]:
    """``{device: {"cameras": [names], "user", "password", "paths": [paths]}}`` for the watched cameras."""
    groups: Dict[Device, Dict[str, Any]] = {}
    for name, url in cameras.items():
        parsed = device_of(url)
        if parsed is None:
            continue
        device, user, password, path = parsed
        group = groups.setdefault(device, {"cameras": [], "user": user, "password": password, "paths": []})
        group["cameras"].append(name)
        if path not in group["paths"]:
            group["paths"].append(path)
    return groups


def file_hosts(text: str) -> List[str]:
    """Every device address named by an RTSP URL in the cameras file (active and disabled)."""
    return sorted({m.group(1).lower() for m in re.finditer(r"rtsp://(?:[^/@\s'\"]*@)?([^:/\s'\"]+)", text)})


def move_host(text: str, old: str, new: str) -> Tuple[str, int]:
    """*text* with the address *old* replaced by *new* in every RTSP URL, and how many URLs changed."""
    pattern = re.compile(r"(rtsp://(?:[^/@\s'\"]*@)?)" + re.escape(old) + r"(?=[:/\s'\"]|$)", re.IGNORECASE)
    return pattern.subn(lambda m: m.group(1) + new, text)


def rewrite_cameras_file(path: str, old: str, new: str, now: Optional[float] = None) -> int:
    """Point every URL of *old* in the cameras file at *new*; a copy of the old file is kept beside it.

    Written through a synced temp file (2026-10-10: a power cut left replaced-but-unsynced files as zero bytes).
    Returns the number of URLs changed (0: the file is left as it was)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    moved, count = move_host(text, old, new)
    if not count:
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if now is None else now))
    shutil.copy2(path, f"{path}.bak-{stamp}")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(moved)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return count


def find_device(group: Mapping[str, Any], old: Device, known_hosts: Sequence[str],
                scan: Callable[[int], List[str]], serves: Callable[[str, int, str, str, str], Optional[bool]]
                ) -> Dict[str, Any]:
    """Where the device that was at *old* is now: ``{"outcome", "host", "candidates"}``.

    outcome: ``still_there`` (the old address serves its stream), ``found`` (*host*), ``none`` or ``ambiguous``.
    *scan(port)* lists the addresses that answer on a port; *serves(host, port, user, password, path)* is
    find_cameras.rtsp_stream_state (True: 200, False: refused, None: no answer)."""
    host, port = old
    user, password, paths = group["user"], group["password"], list(group["paths"])
    if serves(host, port, user, password, paths[0]):
        return {"outcome": "still_there", "host": host, "candidates": []}
    known = {h.lower() for h in known_hosts}
    candidates = [h for h in scan(port) if h.lower() not in known]
    matches = [h for h in candidates if all(serves(h, port, user, password, p) for p in paths)]
    if len(matches) == 1:
        return {"outcome": "found", "host": matches[0], "candidates": candidates}
    return {"outcome": "ambiguous" if matches else "none", "host": "", "candidates": candidates, "matches": matches}


def _scan(port: int) -> List[str]:
    from .find_cameras import rtsp_hosts  # noqa: PLC0415

    return list(rtsp_hosts().get(port) or [])


def _serves(host: str, port: int, user: str, password: str, path: str) -> Optional[bool]:
    from .find_cameras import rtsp_stream_state  # noqa: PLC0415

    return rtsp_stream_state(host, port, user, password, path)


def _restart() -> None:
    from . import control  # noqa: PLC0415

    control.request_restart()


class RecorderFinder:
    """Watches the cameras' last pictures (from the guard loop) and searches for a device whose cameras all went
    silent. *moved(old, new, count)* is told before the restart (the system notices' "found it" note)."""

    def __init__(self, cameras: Mapping[str, str], cameras_path: str, started: float,
                 moved: Optional[Callable[[str, str, int], None]] = None,
                 scan: Callable[[int], List[str]] = _scan, serves: Callable[..., Optional[bool]] = _serves,
                 restart: Callable[[], None] = _restart, background: bool = True) -> None:
        self.groups = group_by_device(cameras)
        self.cameras_path, self.started = cameras_path, started
        self.moved, self.scan, self.serves, self.restart = moved, scan, serves, restart
        self.background = background
        self._last_search: Dict[Device, float] = {}
        self._running: Optional[threading.Thread] = None
        self._next_check = 0.0
        self.done = False           # the file was rewritten: the restart is on its way, nothing more to do

    def check(self, now: float, last_frames: Mapping[str, float]) -> None:
        """The guard loop's beat: *last_frames* is each camera's last picture time (0: none yet). Never raises."""
        if self.done or now < self._next_check:
            return
        self._next_check = now + CHECK_EVERY_SEC
        if self._running is not None and self._running.is_alive():
            return
        for device, group in self.groups.items():
            latest = max([float(last_frames.get(name) or 0.0) for name in group["cameras"]] + [self.started])
            if now - latest < SILENT_SEC or now - self._last_search.get(device, 0.0) < RETRY_SEC:
                continue
            self._last_search[device] = now
            if self.background:
                self._running = threading.Thread(target=self._search, args=(device, group, latest), daemon=True,
                                                 name="recorder-search")
                self._running.start()
            else:
                self._search(device, group, latest)
            return                  # one search at a time

    def _search(self, device: Device, group: Mapping[str, Any], latest: float) -> None:
        host, port = device
        names = ", ".join(group["cameras"])
        try:
            log.info("Recorder search: no picture from %s at %s:%d for %.0f s; looking for it", names, host, port,
                     time.time() - latest)
            try:
                with open(self.cameras_path, encoding="utf-8") as f:
                    known = file_hosts(f.read())
            except OSError:
                known = [d[0] for d in self.groups]
            result = find_device(group, device, known, self.scan, self.serves)
            outcome = result["outcome"]
            if outcome == "still_there":
                log.info("Recorder search: %s:%d still serves its stream; not moving anything", host, port)
                return
            if outcome == "none":
                log.info("Recorder search: not found on the network (%d other device(s) on port %d: %s)",
                         len(result["candidates"]), port, ", ".join(result["candidates"]) or "none")
                return
            if outcome == "ambiguous":
                log.warning("Recorder search: %d devices serve the same streams (%s); not choosing one",
                            len(result["matches"]), ", ".join(result["matches"]))
                return
            new = result["host"]
            count = rewrite_cameras_file(self.cameras_path, host, new)
            if not count:
                log.warning("Recorder search: found at %s but %s names no URL at %s; nothing changed", new,
                            self.cameras_path, host)
                return
            log.info("Recorder search: moved %s -> %s (%d camera URL(s) in %s); restarting to open it", host, new,
                     count, os.path.basename(self.cameras_path))
            self.done = True
            if self.moved is not None:
                try:
                    self.moved(host, new, count)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Recorder search: move not noted for the notices: %s", exc)
            self.restart()
        except Exception as exc:  # noqa: BLE001 - a failed search must never stop the alerts
            log.warning("Recorder search failed: %s", exc)


def start(box_settings: Mapping[str, Any], cameras: Mapping[str, str], started: float,
          moved: Optional[Callable[[str, str, int], None]] = None) -> Optional[RecorderFinder]:
    """At inference start, with the watched cameras' URLs. None when switched off or no URL names a device."""
    from . import paths  # noqa: PLC0415

    if not enabled(box_settings):
        log.info("Recorder search off (find_moved_recorder)")
        return None
    finder = RecorderFinder(cameras, paths.cameras_yaml(), started, moved=moved)
    if not finder.groups:
        return None
    log.info("Recorder search on: %d device(s) (searched after %.0f s with no picture, again every %.0f s)",
             len(finder.groups), SILENT_SEC, RETRY_SEC)
    return finder
