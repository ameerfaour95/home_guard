"""Local, demand-driven preview transport. Never opens a camera."""

from pathlib import Path
import hashlib
import json
import math
import os
import time
import threading


def preview_settings():
    """The existing display settings, with the same local overlay precedence."""
    import yaml
    settings = {}
    base = Path(__file__).parents[1] / "data_collection" / "config.yaml"
    for path in (base, os.environ.get("HOME_GUARD_CONFIG_OVERLAY")):
        if not path:
            continue
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            display = data.get("display", {}) if isinstance(data, dict) else {}
            if isinstance(display, dict):
                settings.update(display)
        except (OSError, ValueError, yaml.YAMLError):
            pass
    return settings


class PreviewRateController:
    """Aggregate cost across visible cameras; quick backoff, cautious recovery."""
    RATES = ((16, 5), (12, 3), (8, 2), (6, 2))

    def __init__(self, cap_percent=10, clock=time.monotonic):
        try:
            cap = float(cap_percent)
        except (TypeError, ValueError):
            cap = 10
        self.cap = (cap if math.isfinite(cap) and 0 < cap <= 100 else 10) / 100
        self.clock = clock
        self.level = 0
        self.costs = {}
        self.overhead = .01
        self.last_change = clock()

    @property
    def rates(self):
        return self.RATES[self.level]

    def estimate(self, cameras, hero, level=None):
        rates = self.RATES[self.level if level is None else level]
        # Reserve one percentage point for handoff, lease checks and telemetry.
        return self.overhead + sum(self.costs.get(name, .004) * rates[0 if name == hero else 1]
                         for name in set(cameras))

    def observe(self, camera, seconds, cameras, hero):
        previous = self.costs.get(camera, seconds)
        self.costs[camera] = previous * .8 + max(0, seconds) * .2
        now = self.clock()
        if now - self.last_change < 2:
            return
        if self.estimate(cameras, hero) > self.cap and self.level < len(self.RATES)-1:
            self.level += 1
            self.last_change = now
        elif (self.level > 0 and now - self.last_change >= 5
              and self.estimate(cameras, hero, self.level-1) < self.cap * .8):
            self.level -= 1
            self.last_change = now


def camera_key(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _fresh_timestamp(timestamp, now, max_age):
    # Windows file mtimes and time.time() can round the same FILETIME to
    # adjacent floats. A just-written file may therefore be one ULP ahead.
    # Accept only that representation error, not a clock-skew time window.
    return timestamp <= math.nextafter(now, math.inf) and now - timestamp <= max_age


class PreviewWriter:
    def __init__(self, directory, enabled=False, fps=2.0, clock=time.time, cpu_cap_percent=None):
        self.directory = Path(directory)
        self.enabled = enabled
        self.interval = 1 / max(0.1, fps)
        self.hero = None
        self.viewer_checked = float("-inf")
        self.viewer_timestamp = float("-inf")
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
        if cpu_cap_percent is None:
            cpu_cap_percent = preview_settings().get("preview_cpu_cap_percent", 10)
        self.rate_controller = PreviewRateController(cpu_cap_percent)
        self.metrics_started = time.time()
        self.worker_cpu_seconds = 0.0
        self.offer_cpu_seconds = 0.0
        self.encode_cpu_seconds = 0.0
        self._overhead_sample = (time.monotonic(), 0.0, 0.0)
        self.published = {}
        self.published_roles = {"hero": 0, "thumbnail": 0}
        self.detector_loops = 0
        self.detector_instrumented = False
        self.metrics_written = float("-inf")

    def offer(self, camera, frame):
        """Latest masked capture only; never encode or wait for disk on detection."""
        if not self.enabled:
            return
        began = time.thread_time()
        with self._pending_lock:
            self._pending[camera] = frame
            if self._worker is None:
                self._worker = threading.Thread(target=self._drain, name="preview-publisher", daemon=True)
                self._worker.start()
        self.offer_cpu_seconds += time.thread_time() - began

    def _drain(self):
        cpu_started = time.thread_time()
        while not self._stop.wait(.008):
            with self._pending_lock:
                frames = dict(self._pending)
            for camera, frame in frames.items():
                self.publish(camera, frame, source=frame)
            # Include wait/lock bookkeeping too, not just each loop body.
            self.worker_cpu_seconds = time.thread_time() - cpu_started
            self.write_metrics()
        self.worker_cpu_seconds = time.thread_time() - cpu_started

    def close(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)

    def write_metrics(self):
        """One atomic local snapshot/second, including when the app is closed."""
        now = time.monotonic()
        if now - self.metrics_written < 1:
            return
        self.metrics_written = now
        controller = self.rate_controller
        previous_time, previous_cpu, previous_encode = self._overhead_sample
        cpu = self.worker_cpu_seconds + self.offer_cpu_seconds
        if now - previous_time >= 1:
            overhead = max(0, (cpu-previous_cpu-self.encode_cpu_seconds+previous_encode)/(now-previous_time))
            controller.overhead = max(.01, controller.overhead*.8 + overhead*.2)
            self._overhead_sample = (now, cpu, self.encode_cpu_seconds)
        live = self.preference.get("version") == 2
        visible = bool(self.preference.get("visible")) if live else self.hero is not None
        try:
            age = time.time() - (self.directory / "viewer.alive").stat().st_mtime
            visible = visible and 0 <= age <= (3 if live else 15)
        except OSError:
            visible = False
        cameras = self.preference.get("cameras", []) if visible else []
        data = {"pid": os.getpid(), "started": self.metrics_started, "updated": time.time(),
                "monotonic": now, "publisher_cpu_seconds": self.worker_cpu_seconds + self.offer_cpu_seconds,
                "published": dict(self.published), "published_roles": dict(self.published_roles),
                "detector_loops": self.detector_loops, "detector_instrumented": self.detector_instrumented,
                "visible": visible, "hero": self.hero,
                "rates": controller.rates, "cap_percent": controller.cap * 100,
                "overhead_one_core_percent": controller.overhead * 100,
                "encode_publish_ema_ms": {k: v*1000 for k, v in controller.costs.items()},
                "estimated_one_core_percent": controller.estimate(cameras, self.hero)*100 if visible else 0,
                "floor_over_budget": visible and controller.level == len(controller.RATES)-1
                    and controller.estimate(cameras, self.hero) > controller.cap}
        temp = self.directory / ("publisher_metrics." + str(os.getpid()) + ".tmp")
        try:
            temp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(temp, self.directory / "publisher_metrics.json")
        except OSError:
            pass
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

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
            now = self.clock()
            if now-self.viewer_checked>=.25:
                self.viewer_checked=now
                # All cameras share this lease. One stat/read per quarter second.
                self.viewer_timestamp = float("-inf")
                self.viewer_timestamp = (self.directory / "viewer.alive").stat().st_mtime
                try:
                    preference=json.loads((self.directory / "viewer.alive").read_text(encoding="utf-8"))
                    self.preference=preference if isinstance(preference,dict) else {}
                    self.hero=preference.get("hero") if isinstance(preference,dict) else None
                except (OSError,ValueError): self.hero=None;self.preference={}
            live = self.preference.get("version") == 2
            if live and (not self.preference.get("visible") or camera not in self.preference.get("cameras", [])):
                return False
            hero_rate, thumbnail_rate = self.rate_controller.rates
            interval=(1/hero_rate if camera==self.hero else 1/thumbnail_rate) if live else (1/6 if camera==self.hero else self.interval)
            return (
                _fresh_timestamp(self.viewer_timestamp, now, 3 if live else 15)
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
        began = time.perf_counter()
        cpu_began = time.thread_time()
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
            self.published[camera] = self.published.get(camera, 0) + 1
            self.published_roles["hero" if camera == self.hero else "thumbnail"] += 1
            return True
        except (OSError, cv2.error):
            return False
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            self.encode_cpu_seconds += time.thread_time() - cpu_began
            if self.preference.get("version") == 2:
                self.rate_controller.observe(camera, time.perf_counter()-began,
                                             self.preference.get("cameras", []), self.hero)
            if threading.current_thread() is not self._worker:
                self.worker_cpu_seconds += time.thread_time() - cpu_began


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
