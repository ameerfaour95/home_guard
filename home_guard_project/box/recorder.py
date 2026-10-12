"""The house recorder's own recording: a past moment the box kept nothing of.

2026-10-12 13:37-13:39 the owner asked when the pickup now standing at the pergola had come, and when someone got out
of it. The pergola alerts only on people, so the box kept no clip and no journal line of the truck (its log only said
"this camera alerts only on person; no alert" at 12:56); the assistant sent 10 live seconds and then set two watches he
never asked for. The recorder (a Uniview NVR) records every camera all day, and plays any moment back over RTSP:
``rtsp://<user>:<pass>@<host>:554/c<channel>/b<start>/e<end>/replay/``, with start and end in unix seconds of the
recorder's own clock. That clock is set by hand and ran 5 min ahead of the box (ONVIF GetSystemDateAndTime, which
needs no login, says by how much).

- ``arrival``: when the vehicles standing at a camera now came: the box's own detector (no AI, no cost) on single
  frames of the recording, stepping back 1, 2, 5, 10 ... minutes until the vehicle's place is empty, then narrowing
  that gap to ``ARRIVAL_PRECISION_SEC``. Each look reads two frames a second apart (a decode-smeared frame can hide a
  car), and any of the two showing a vehicle on the place counts as there.
- ``clip``: a few seconds of the recording around a moment, masked to the camera's watch zone like every picture the
  box keeps (fail closed), at ``CLIP_FPS`` and ``CLIP_WIDTH``.

A camera whose address is not a Uniview recorder channel has no recording here (``available`` False): nothing is
guessed. Nothing here raises.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("box.recorder")

_LIVE = re.compile(r"^(rtsp://[^/]+)/unicast/c(\d+)/s\d+/live/?$", re.IGNORECASE)
_HOST = re.compile(r"^rtsp://(?:[^@/]*@)?([^:/]+)")
STEPS_BACK_SEC = (60, 120, 300, 600, 1200, 2400, 3600, 7200, 14400, 28800, 43200)
ARRIVAL_PRECISION_SEC = 15.0
WORKERS = 3                     # parallel playbacks: the recorder serves a few at once
OFFSET_TTL_SEC = 3600.0
CLIP_FPS = 5.0
CLIP_WIDTH = 1280
READ_FRAMES = 30                # a look reads this many frames and keeps two, a second apart (25 fps)


def replay_url(live_url: str, start: float, end: float) -> Optional[str]:
    """The recorder's playback of *live_url*'s channel from *start* to *end* (recorder clock); None for an address
    that is not a Uniview recorder channel."""
    m = _LIVE.match(str(live_url or "").strip())
    if not m:
        return None
    return f"{m.group(1)}/c{m.group(2)}/b{int(start)}/e{int(end)}/replay/"


def onvif_utc(xml: str) -> Optional[float]:
    """The UTC time in an ONVIF GetSystemDateAndTime answer, as unix seconds; None when it has none."""
    utc = re.search(r"UTCDateTime>(.*?)</[\w:]*UTCDateTime", str(xml or ""), re.S)
    if not utc:
        return None

    def part(name: str) -> int:
        m = re.search(rf"<[\w:]*{name}>\s*(\d+)\s*<", utc.group(1))
        if not m:
            raise ValueError(name)
        return int(m.group(1))

    try:
        when = dt.datetime(part("Year"), part("Month"), part("Day"), part("Hour"), part("Minute"), part("Second"),
                           tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return when.timestamp()


_ONVIF_TIME = ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
               '<s:Body><GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/></s:Body></s:Envelope>')


def _post_onvif(host: str) -> str:
    import requests  # noqa: PLC0415

    r = requests.post(f"http://{host}/onvif/device_service", data=_ONVIF_TIME, timeout=5,
                      headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    return r.text if r.status_code == 200 else ""


def _present(boxes: Sequence[Sequence[float]], place: Sequence[float]) -> bool:
    from .watches import overlaps  # noqa: PLC0415

    return any(overlaps(place, b) for b in boxes)


class Recorder:
    """Playback of the recorder's channels. *open_capture(url)* -> a cv2-like capture; *post(host)* -> the ONVIF
    time answer; *detect(image path)* (given per call) -> the detector's objects."""

    def __init__(self, cameras_path: str, work_dir: str, zones_path: Optional[str] = None,
                 open_capture: Optional[Callable[[str], Any]] = None, post: Callable[[str], str] = _post_onvif,
                 now: Callable[[], float] = time.time, url_of: Optional[Callable[[str], Optional[str]]] = None) -> None:
        self.cameras_path, self.work_dir, self.zones_path = cameras_path, work_dir, zones_path
        self._open, self._post, self.now = open_capture, post, now
        self._url_of = url_of
        self._offset: Dict[str, Tuple[float, float]] = {}      # host -> (offset, when measured)
        self._lock = threading.Lock()
        self._detect_lock = threading.Lock()     # one detector, shared by the parallel playbacks

    # -- addresses and the clock ----------------------------------------------------------------------------------
    def _live(self, camera: str) -> Optional[str]:
        if self._url_of is not None:
            return self._url_of(camera)
        from .live_view import _camera_url  # noqa: PLC0415

        return _camera_url(camera, self.cameras_path)

    def available(self, camera: str) -> bool:
        try:
            return replay_url(self._live(camera) or "", 0, 1) is not None
        except Exception:  # noqa: BLE001
            return False

    def offset(self, camera: str) -> float:
        """Recorder clock minus box clock, in seconds (0 when the recorder does not say)."""
        host_m = _HOST.match(self._live(camera) or "")
        if not host_m:
            return 0.0
        host = host_m.group(1)
        with self._lock:
            cached = self._offset.get(host)
            if cached and self.now() - cached[1] < OFFSET_TTL_SEC:
                return cached[0]
        before = self.now()
        try:
            theirs = onvif_utc(self._post(host))
        except Exception as exc:  # noqa: BLE001
            log.info("Recorder clock not read (%s): %s", host, exc)
            theirs = None
        after = self.now()
        value = round(theirs - (before + after) / 2.0) if theirs is not None else 0.0
        if abs(value) > 6 * 3600:
            log.warning("Recorder clock %+.0f s from the box: not trusted", value)
            value = 0.0
        with self._lock:
            self._offset[host] = (float(value), after)
        if value:
            log.info("Recorder %s clock is %+.0f s from the box", host, value)
        return float(value)

    def _capture(self, url: str) -> Any:
        if self._open is not None:
            return self._open(url)
        import cv2  # noqa: PLC0415

        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    def _mask(self, camera: str) -> Tuple[Any, bool]:
        from ..data_collection.zones import ZoneMask  # noqa: PLC0415
        from .live_view import strict_black, strict_zone  # noqa: PLC0415

        polygon, readable = strict_zone(camera, self.zones_path)
        black, black_readable = strict_black(camera, self.zones_path)
        if not (readable and black_readable):
            return None, False
        return (ZoneMask(polygon, black) if (polygon or black) else None), True

    # -- looks ----------------------------------------------------------------------------------------------------
    def frames(self, camera: str, ts: float) -> List[Any]:
        """Two masked frames of *camera* about a second apart from *ts* (box clock); [] when there are none."""
        live = self._live(camera)
        url = replay_url(live or "", ts + self.offset(camera), ts + self.offset(camera) + 10) if live else None
        if not url:
            return []
        mask, ok = self._mask(camera)
        if not ok:
            return []                                  # a zone we cannot read: no picture at all
        cap = self._capture(url)
        kept: List[Any] = []
        try:
            for i in range(READ_FRAMES):
                got, frame = cap.read()
                if not got:
                    break
                if i in (4, READ_FRAMES - 1):
                    kept.append(mask.apply(frame) if mask is not None else frame)
        except Exception as exc:  # noqa: BLE001
            log.info("Recorder playback of %s at %s failed: %s", camera, ts, exc)
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        return kept

    def vehicles_at(self, camera: str, ts: float, detect: Callable[[str], Any]) -> Optional[List[List[float]]]:
        """The vehicle boxes (0..1) at *camera* at *ts*: the union of the two frames' finds; None with no frame."""
        import cv2  # noqa: PLC0415

        from .watches import vehicles  # noqa: PLC0415

        frames = self.frames(camera, ts)
        if not frames:
            return None
        os.makedirs(self.work_dir, exist_ok=True)
        found: List[List[float]] = []
        for frame in frames:
            path = os.path.join(self.work_dir, f"{camera}_{int(ts)}_{uuid.uuid4().hex[:6]}_rec.jpg")
            try:
                if not cv2.imwrite(path, frame):
                    continue
                with self._detect_lock:
                    objects = detect(path)
                found += vehicles(objects) if objects is not None else []
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return found

    def arrival(self, camera: str, detect: Callable[[str], Any], now: Optional[float] = None,
                limit: int = 3) -> Dict[str, Any]:
        """When the vehicles standing at *camera* now came. ``{"ok", "vehicles": [{"box", "arrived", "before",
        "since_before"}], "looks"}``: *arrived* is the first look (box clock) with the vehicle on its place, *before*
        the last look without it (to ``ARRIVAL_PRECISION_SEC``); *since_before* (no *arrived*) when it stood there
        on every look back to ``STEPS_BACK_SEC[-1]``. ``{"ok": False, "error"}`` without a recording."""
        now = float(self.now() if now is None else now)
        if not self.available(camera):
            return {"ok": False, "error": "no_recorder"}
        latest = now - 20.0                            # the recorder writes a little behind the live picture
        looks = [0]

        def at(ts: float) -> Optional[List[List[float]]]:
            looks[0] += 1
            return self.vehicles_at(camera, ts, detect)

        current = at(latest)
        if current is None:
            return {"ok": False, "error": "no_playback"}
        if not current:
            return {"ok": True, "vehicles": [], "looks": looks[0]}
        out = []
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for place in current[:limit]:
                out.append(self._arrival_of(place, latest, pool, at))
        return {"ok": True, "vehicles": out, "looks": looks[0]}

    def _arrival_of(self, place: Sequence[float], latest: float, pool: cf.ThreadPoolExecutor,
                    at: Callable[[float], Optional[List[List[float]]]]) -> Dict[str, Any]:
        times = [latest - s for s in STEPS_BACK_SEC]
        seen = dict(zip(times, pool.map(at, times)))
        there, gone = latest, None
        for ts in times:                                 # newest first: the first empty look ends the stay
            boxes = seen[ts]
            if boxes is None:
                continue                                 # no frame: neither there nor gone
            if _present(boxes, place):
                there = ts
            else:
                gone = ts
                break
        if gone is None:
            return {"box": list(place), "since_before": there}
        while there - gone > ARRIVAL_PRECISION_SEC:
            step = (there - gone) / (WORKERS + 1)
            probes = [gone + step * (i + 1) for i in range(WORKERS)]
            results = list(pool.map(at, probes))
            new_gone, new_there = gone, there
            for ts, boxes in zip(probes, results):
                if boxes is None:
                    continue
                if _present(boxes, place):
                    new_there = min(new_there, ts)
                else:
                    new_gone = max(new_gone, ts) if ts < new_there else new_gone
            if (new_gone, new_there) == (gone, there):
                break                                    # no frame in the gap: as close as the recording allows
            gone, there = new_gone, new_there
        return {"box": list(place), "arrived": there, "before": gone}

    # -- a clip ---------------------------------------------------------------------------------------------------
    def clip(self, camera: str, start: float, seconds: float) -> Dict[str, Any]:
        """*seconds* of the recording of *camera* from *start* (box clock), masked: ``{"ok", "path", "start",
        "end"}`` or ``{"ok": False, "error"}``. Never raises."""
        import cv2  # noqa: PLC0415

        path = None
        writer = None
        written = 0
        try:
            live = self._live(camera)
            off = self.offset(camera)
            url = replay_url(live or "", start + off, start + off + seconds + 2) if live else None
            if not url:
                return {"ok": False, "error": "no_recorder"}
            mask, ok = self._mask(camera)
            if not ok:
                return {"ok": False, "error": "zone_unreadable"}
            os.makedirs(self.work_dir, exist_ok=True)
            path = os.path.join(self.work_dir, f"{camera}_{int(start)}_{uuid.uuid4().hex[:8]}_rec.mp4")
            cap = self._capture(url)
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) if hasattr(cap, "get") else 0.0
                fps = fps if 1.0 <= fps <= 60.0 else 25.0
                every = max(1, int(round(fps / CLIP_FPS)))
                total = int(seconds * fps)
                deadline = time.monotonic() + seconds * 2 + 20
                for i in range(total):
                    if time.monotonic() > deadline:
                        break
                    got, frame = cap.read()
                    if not got:
                        break
                    if i % every:
                        continue
                    if mask is not None:
                        frame = mask.apply(frame)        # raises: nothing unmasked is kept
                    h, w = frame.shape[:2]
                    if w > CLIP_WIDTH:
                        frame = cv2.resize(frame, (CLIP_WIDTH, int(round(h * CLIP_WIDTH / w))))
                    if writer is None:
                        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS,
                                                 (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise OSError("could not open the recording writer")
                    writer.write(frame)
                    written += 1
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
            if not written:
                raise OSError("no frame in the recording")
            from .alert_clips import _to_h264  # noqa: PLC0415

            _to_h264(path)
            return {"ok": True, "path": path, "start": start, "end": start + written / CLIP_FPS}
        except Exception as exc:  # noqa: BLE001
            log.warning("Recorder clip of %s at %s failed: %s", camera, start, exc)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return {"ok": False, "error": "error"}
