"""Local, demand-driven preview transport. Never opens a camera."""

from pathlib import Path
import hashlib
import json
import math
import os
import time
import threading


def camera_key(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _fresh_timestamp(timestamp, now, max_age):
    # Windows file mtimes and time.time() can round the same FILETIME to
    # adjacent floats. A just-written file may therefore be one ULP ahead.
    # Accept only that representation error, not a clock-skew time window.
    return timestamp <= math.nextafter(now, math.inf) and now - timestamp <= max_age


class PreviewWriter:
    def __init__(self, directory, enabled=False, fps=2.0, clock=time.time):
        self.directory = Path(directory)
        self.enabled = enabled
        self.interval = 1 / max(0.1, fps)
        self.hero = None
        self.viewer_checked = float("-inf")
        self.clock = clock
        self.last = {}
        self.sources = {}
        self.camera_names = []
        self.manifest_pending = False
        self.preference = {}
        self._pending = {}
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = None

    def offer(self, camera, frame):
        """Latest masked capture only; never encode or wait for disk on detection."""
        if not self.enabled:
            return
        with self._pending_lock:
            self._pending[camera] = frame
            if self._worker is None:
                self._worker = threading.Thread(target=self._drain, name="preview-publisher", daemon=True)
                self._worker.start()

    def _drain(self):
        while not self._stop.wait(.008):
            with self._pending_lock:
                frames = dict(self._pending)
            for camera, frame in frames.items():
                self.publish(camera, frame, source=frame)

    def close(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)

    def set_cameras(self, names):
        self.camera_names = list(names)
        if not self.enabled:
            return False
        temp = self.directory / ("cameras." + str(os.getpid()) + ".tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temp.write_text(json.dumps(list(names)), encoding="utf-8")
            os.replace(temp, self.directory / "cameras.json")
            self.manifest_pending = False
            return True
        except OSError:
            self.manifest_pending = True
            return False
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def wanted(self, camera, source=None):
        if not self.enabled or (
            source is not None and self.sources.get(camera) is source
        ):
            return False
        try:
            timestamp = (self.directory / "viewer.alive").stat().st_mtime
            now = self.clock()
            if now-self.viewer_checked>=.25:
                self.viewer_checked=now
                try:
                    preference=json.loads((self.directory / "viewer.alive").read_text(encoding="utf-8"))
                    self.preference=preference if isinstance(preference,dict) else {}
                    self.hero=preference.get("hero") if isinstance(preference,dict) else None
                except (OSError,ValueError): self.hero=None;self.preference={}
            live = self.preference.get("version") == 2
            if live and (not self.preference.get("visible") or camera not in self.preference.get("cameras", [])):
                return False
            interval=(1/18 if camera==self.hero else 1/6) if live else (1/6 if camera==self.hero else self.interval)
            return (
                _fresh_timestamp(timestamp, now, 3 if live else 15)
                and now - self.last.get(camera, float("-inf")) >= interval
            )
        except OSError:
            return False

    def publish(self, camera, frame, source=None):
        if not self.wanted(camera, source):
            return False
        if self.manifest_pending:
            self.set_cameras(self.camera_names)
        self.last[camera] = self.clock()  # throttle failed writes too
        import cv2

        key = camera_key(camera)
        target = self.directory / (key + ".jpg")
        temp = self.directory / (key + "." + str(os.getpid()) + ".tmp")
        try:
            if max(frame.shape[:2])>1280:
                scale=1280/max(frame.shape[:2])
                frame=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            quality = 75 if self.preference.get("version") == 2 else 88
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return False
            temp.write_bytes(encoded.tobytes())
            os.replace(temp, target)
            if source is not None:
                self.sources[camera] = source
            return True
        except (OSError, cv2.error):
            return False
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


class PreviewReader:
    def __init__(self, directory, stale_seconds=12, clock=time.time):
        self.directory = Path(directory)
        self.stale_seconds = stale_seconds
        self.clock = clock

    def names(self):
        try:
            raw = json.loads(
                (self.directory / "cameras.json").read_text(encoding="utf-8")
            )
            return (
                [name for name in raw if isinstance(name, str)]
                if isinstance(raw, list)
                else []
            )
        except (OSError, ValueError):
            return []

    def touch(self, hero=None, *, visible=None, cameras=()):
        self.directory.mkdir(parents=True, exist_ok=True)
        marker=self.directory / "viewer.alive"
        if hero is None and visible is None:
            marker.touch()
        else:
            temporary=self.directory / ("viewer."+str(os.getpid())+".tmp")
            try:
                demand={"hero":hero}
                if visible is not None:
                    demand.update(version=2,visible=bool(visible),cameras=list(cameras))
                temporary.write_text(json.dumps(demand),encoding="utf-8")
                os.replace(temporary,marker)
            except OSError:
                if visible is None: marker.touch()
            finally:
                temporary.unlink(missing_ok=True)

    def read(self, camera):
        path = self.directory / (camera_key(camera) + ".jpg")
        try:
            timestamp = path.stat().st_mtime
            if not _fresh_timestamp(timestamp, self.clock(), self.stale_seconds):
                return None
            return path.read_bytes()  # closes before Qt decodes, important on Windows
        except OSError:
            return None
