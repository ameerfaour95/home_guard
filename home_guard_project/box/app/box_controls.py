"""Local control adapter and an isolated, mutable demo box."""
from dataclasses import dataclass, replace
from pathlib import Path
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from .. import control, boxconfig


@dataclass(frozen=True)
class Settings:
    mode: str = "data_collection"
    alert_start_hour: int = 0
    alert_end_hour: int = 0
    alert_cooldown_sec: int = 120
    show_cameras: bool = False
    inference_conf: float = 0.4

    alert_on: str = "person"
    conf_person: Optional[float] = None
    conf_vehicle: Optional[float] = None
    conf_animal: Optional[float] = None

    def __post_init__(self):
        from .alert_types import ordered_types
        object.__setattr__(self, "alert_on", ",".join(ordered_types(self.alert_on)))
        if self.mode not in boxconfig.MODES or type(self.show_cameras) is not bool:
            raise ValueError("Invalid settings")
        for key, limits in boxconfig.NUMBER_OPTIONS.items():
            value = getattr(self, key)
            if type(value) is not int or not limits[0] <= value <= limits[1]:
                raise ValueError("Invalid settings")
        for key,(low,high) in boxconfig.DECIMAL_OPTIONS.items():
            value=getattr(self,key)
            if type(value) not in (int,float) or not low<=value<=high: raise ValueError('Invalid settings')
        for key, (low, high) in boxconfig.TYPE_CONF_OPTIONS.items():
            value = getattr(self, key)
            if value is not None and (type(value) not in (int, float) or not low <= value <= high):
                raise ValueError('Invalid sensitivity')

    def effective_sensitivity(self):
        return {key.removeprefix('conf_'): self.inference_conf if getattr(self, key) is None else getattr(self, key)
                for key in boxconfig.TYPE_CONF_OPTIONS}

    def options(self):
        return {key: getattr(self, key) for key in ("mode", "alert_start_hour", "alert_end_hour", "alert_cooldown_sec", "show_cameras", "inference_conf", "alert_on", *boxconfig.TYPE_CONF_OPTIONS)}

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
        self.camera_alert_on = {}
        self.camera_sensitivity = {}

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

    def reported_status(self):
        if self.demo:
            s=self._settings
            return {'updated':self.clock(),'settings':dict(conf=s.inference_conf,alert_start_hour=s.alert_start_hour,alert_end_hour=s.alert_end_hour,cooldown_sec=s.alert_cooldown_sec,alert_on=s.alert_on.split(","),camera_alert_on={k:list(v) for k,v in self.camera_alert_on.items()},sensitivity=s.effective_sensitivity(),camera_sensitivity={k:dict(v) for k,v in self.camera_sensitivity.items()})}
        if hasattr(self,'live_status'): return self.live_status
        from ..ai_status import read_status
        return read_status(Path(boxconfig.LOG_DIR)/'ai_status.json')

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

class RemoteSettingsBackend:
    """Settings transport for a laptop connection, with injectable process/status readers."""
    def __init__(self, target, settings, runner, status_reader, key=None, clock=time.time):
        from .remote_cameras import target_user
        target_user(target)
        self.target, self._settings = target, settings
        self.runner, self.status_reader, self.clock = runner, status_reader, clock
        self.key = Path(key) if key else Path.home()/'.ssh'/'homeguard_box'

    def load_settings(self): return self._settings

    def reported_status(self): return self.status_reader()

    def save_settings(self, settings):
        changed = {key:value for key,value in settings.options().items() if self._settings.options()[key]!=value}
        for key,value in changed.items():
            value = str(value).lower() if isinstance(value,bool) else str(value)
            command = r'cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box set-option '+key+'='+value
            result = self.runner.run(['ssh.exe','-i',str(self.key),'-o','LogLevel=ERROR',self.target,command])
            if result.returncode: raise RuntimeError('Settings command failed')
            self._settings = replace(self._settings, **{key:getattr(settings,key)})
        # set-option itself requests a restart for keys in the runtime table.
        return any(key in boxconfig.RESTART_OPTIONS for key in changed)
