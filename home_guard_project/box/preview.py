"""Local, demand-driven preview transport. Never opens a camera."""

from pathlib import Path
import hashlib
import json
import math
import os
import time


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
        self.clock = clock
        self.last = {}
        self.sources = {}
        self.camera_names = []
        self.manifest_pending = False

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
            return (
                _fresh_timestamp(timestamp, now, 15)
                and now - self.last.get(camera, float("-inf")) >= self.interval
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
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
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

    def touch(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "viewer.alive").touch()

    def read(self, camera):
        path = self.directory / (camera_key(camera) + ".jpg")
        try:
            timestamp = path.stat().st_mtime
            if not _fresh_timestamp(timestamp, self.clock(), self.stale_seconds):
                return None
            return path.read_bytes()  # closes before Qt decodes, important on Windows
        except OSError:
            return None
