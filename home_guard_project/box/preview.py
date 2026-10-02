"""Local, demand-driven preview transport. Never opens a camera."""

from pathlib import Path
import hashlib
import json
import os
import time


def camera_key(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


class PreviewWriter:
    def __init__(self, directory, enabled=False, fps=2.0, clock=time.time):
        self.directory = Path(directory)
        self.enabled = enabled
        self.interval = 1 / max(0.1, fps)
        self.clock = clock
        self.last = {}
        self.sources = {}

    def set_cameras(self, names):
        if not self.enabled:
            return False
        temp = self.directory / ("cameras." + str(os.getpid()) + ".tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temp.write_text(json.dumps(list(names)), encoding="utf-8")
            os.replace(temp, self.directory / "cameras.json")
            return True
        except OSError:
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
        now = self.clock()
        try:
            age = now - (self.directory / "viewer.alive").stat().st_mtime
            return (
                0 <= age <= 15
                and now - self.last.get(camera, float("-inf")) >= self.interval
            )
        except OSError:
            return False

    def publish(self, camera, frame, source=None):
        if not self.wanted(camera, source):
            return False
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
            age = self.clock() - path.stat().st_mtime
            if not 0 <= age <= self.stale_seconds:
                return None
            return path.read_bytes()  # closes before Qt decodes, important on Windows
        except OSError:
            return None
