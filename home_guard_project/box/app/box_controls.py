"""Local control adapter and an isolated, mutable demo box."""
from dataclasses import dataclass
from pathlib import Path
import time
from decimal import Decimal, ROUND_HALF_UP
from .. import control, boxconfig


@dataclass(frozen=True)
class Settings:
    mode: str = "data_collection"
    alert_start_hour: int = 0
    alert_end_hour: int = 0
    alert_cooldown_sec: int = 120
    show_cameras: bool = False

    def __post_init__(self):
        if self.mode not in boxconfig.MODES or type(self.show_cameras) is not bool:
            raise ValueError("Invalid settings")
        for key, limits in boxconfig.NUMBER_OPTIONS.items():
            value = getattr(self, key)
            if type(value) is not int or not limits[0] <= value <= limits[1]:
                raise ValueError("Invalid settings")

    def options(self):
        return {key: getattr(self, key) for key in ("mode", "alert_start_hour", "alert_end_hour", "alert_cooldown_sec", "show_cameras")}

    @classmethod
    def from_options(cls, options):
        defaults = cls()
        return cls(**{key: options.get(key, value) if options.get(key) is not None else value for key, value in defaults.options().items()})


def minutes_to_seconds(minutes):
    value = Decimal(str(minutes))
    if not value.is_finite() or not Decimal(10)/60 <= value <= 1440:
        raise ValueError("Invalid interval")
    return int((value*60).to_integral_value(rounding=ROUND_HALF_UP))


class BoxControls:
    def __init__(self, demo=False, stopped=False, clock=time.time, settings=None):
        self.demo = demo
        self._stopped = stopped
        self.clock = clock
        self._settings = settings or Settings(show_cameras=True)
        self.pending_at = None

    def is_stopped(self):
        return self._stopped if self.demo else control.is_stopped()

    def stop(self):
        if self.demo:
            self._stopped = True
        else:
            control.stop()

    def start(self):
        if self.demo:
            self._stopped = False
        else:
            control.start()

    def load_settings(self):
        if self.demo:
            return self._settings
        return Settings.from_options({key: boxconfig.get_option(key) for key in Settings().options()})

    def save_settings(self, settings):
        previous = self.load_settings()
        changed = {key: value for key, value in settings.options().items() if previous.options()[key] != value}
        if not changed:
            return False
        requested = self.clock()
        restart = False
        try:
            if self.demo:
                self._settings = settings
                restart = any(key in boxconfig.RESTART_OPTIONS for key in changed)
            else:
                for key, value in changed.items():
                    boxconfig.set_option(key, str(value).lower() if isinstance(value, bool) else str(value))
                    restart = restart or key in boxconfig.RESTART_OPTIONS
        finally:
            if restart:
                if not self.demo:
                    control.request_restart()
                if not self.is_stopped():
                    self.pending_at = requested
        return restart

    def phase(self, running=True):
        if self.is_stopped():
            self.pending_at = None
            return "stopped"
        if self.pending_at is not None:
            if self.demo:
                applied = self.clock() - self.pending_at >= 2
            else:
                root = Path(boxconfig.LOG_DIR)
                try:
                    applied = not (root/control.RESTART_FLAG).exists() and (root/'collector.winpid').stat().st_mtime >= self.pending_at and Path(boxconfig.ALIVE_FILE).stat().st_mtime >= self.pending_at
                except OSError:
                    applied = False
            if not applied:
                return "restarting"
            self.pending_at = None
        return "running" if running else "starting"
