from pathlib import Path
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace

from .. import boxconfig
from .alert_types import CameraAlertsBackend

@dataclass(frozen=True)
class Camera:
    name: str
    enabled: bool = True
    file: str = ""
    ok: bool = False

def changes_payload(changes):
    rows = []
    names = set()
    originals = set()
    for old, new, enabled in changes:
        if not old or old in originals or not re.fullmatch(r"[a-z0-9_]+", new) or new in names or type(enabled) is not bool:
            raise ValueError("Invalid camera changes")
        names.add(new)
        originals.add(old)
        rows.append({"name": old, "new_name": new, "enabled": enabled})
    return {"cameras": rows}

def zone_points(points):
    result = [[round(float(x), 4), round(float(y), 4)] for x, y in points]
    if len(result) not in (0, *range(3, 33)) or any(not math.isfinite(v) or not 0 <= v <= 1 for point in result for v in point):
        raise ValueError("Invalid zone points")
    return result

def zone_operation(name, points=None):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Invalid camera name")
    if points is None:
        return ["clear-zone", "--camera", name]
    values = zone_points(points)
    if not values:
        raise ValueError("A zone needs at least three corners")
    return ["set-zone", "--camera", name, "--points", ";".join(f"{x:.4f},{y:.4f}" for x, y in values)]

class CameraControls(CameraAlertsBackend):
    def __init__(self, box, names=(), runner=None):
        self.box = box
        self.runner = runner or subprocess.run
        self.records = [Camera(name.lower().replace(" ", "_"), True, ok=True) for name in names]
        self.out = Path(boxconfig.LOG_DIR) / "app_snapshots"
        self._zones = {}

    def zones(self):
        if self.box.demo:
            return {c.name: [p[:] for p in self._zones.get(c.name, [])] for c in self.records}
        code, data = self.command("zones")
        if code or not isinstance(data.get("cameras"), list):
            raise ValueError("Invalid zone result")
        return {row["name"]: zone_points(row["points"]) for row in data["cameras"]}

    def set_zone(self, name, points):
        args = zone_operation(name, points)
        if self.box.demo:
            self._zones[name] = zone_points(points)
            return [p[:] for p in self._zones[name]]
        return self._write_zone(args)

    def clear_zone(self, name):
        args = zone_operation(name)
        if self.box.demo:
            self._zones.pop(name, None)
            return []
        return self._write_zone(args)

    def _write_zone(self, args):
        code, data = self.command(*args)
        if code or not isinstance(data.get("points"), list):
            raise ValueError("Invalid zone result")
        return zone_points(data["points"])

    def command(self, *args):
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        result = self.runner([sys.executable, "-m", "home_guard_project.box.find_cameras", "--json", *map(str, args)], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", timeout=480, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        data = json.loads(result.stdout)
        if "error" in data:
            raise ValueError(str(data["error"]) if args and args[0] in ("camera-alerts", "set-camera-alerts") else "Camera command failed")
        return result.returncode, data

    def load(self):
        if self.box.demo or getattr(self,"toggle_pending",False):
            return list(self.records)
        from ..find_cameras import _read_cameras_raw
        raw = _read_cameras_raw()
        previous={camera.name:camera for camera in self.records}
        self.records = [replace(previous[name],enabled=enabled) if name in previous else Camera(name,enabled) for group, enabled in (("cameras", True), ("disabled", False)) for name in raw.get(group, {})]
        order={name:i for i,name in enumerate(previous)}
        self.records.sort(key=lambda c:order.get(c.name,len(order)))
        return list(self.records)

    def search(self, user, password):
        """Search only after an explicit user action; passwords stay off argv."""
        if self.box.demo:
            self.records = [Camera(name, True, ok=True) for name in ('front_door','garden','driveway')]
            return list(self.records)
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None)
        env['HG_CAMERA_PASSWORD']=password
        site=boxconfig.load_box_settings().get('site','home')
        result=self.runner([sys.executable,'-m','home_guard_project.box.find_cameras','--json','auto','--user',user,'--prefix',site,'--write'],env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',timeout=480,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        data=json.loads(result.stdout)
        if result.returncode not in (0,1) or not isinstance(data.get('cameras'),list):
            raise ValueError('Camera search failed')
        if not data['cameras']: return []
        self.load()
        if not self.box.is_stopped():
            from .. import control
            control.request_restart()
        return self.snapshots()

    def snapshots(self):
        if self.box.demo:
            return list(self.records)
        self.out.mkdir(parents=True, exist_ok=True)
        code, data = self.command("snapshots", "--out", self.out)
        if code not in (0, 1) or not isinstance(data.get("snapshots"), list):
            raise ValueError("Invalid snapshot result")
        shots = {row["name"]: row for row in data["snapshots"]}
        updated = []
        for camera in self.records:
            shot = shots.get(camera.name, {})
            file = str(shot.get("file") or "")
            ok = bool(shot.get("ok"))
            if ok and (not file or not Path(file).resolve().is_relative_to(self.out.resolve())):
                raise ValueError("Invalid snapshot path")
            updated.append(replace(camera, file=file, ok=ok))
        self.records = updated
        return list(updated)

    def save(self, changes):
        payload = changes_payload(changes)
        if {row["name"] for row in payload["cameras"]} != {c.name for c in self.records}:
            raise ValueError("Camera list changed")
        requested = self.box.clock()
        if not self.box.demo:
            self.out.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.out) as directory:
                path = Path(directory) / "changes.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                code, data = self.command("apply", "--changes", path)
                if code != 0 or not isinstance(data.get("active"), list) or not isinstance(data.get("disabled"), list):
                    raise ValueError("Invalid apply result")
        old = {c.name: c for c in self.records}
        self._zones = {row["new_name"]: self._zones.get(row["name"], []) for row in payload["cameras"]}
        self.records = [replace(old[row["name"]], name=row["new_name"], enabled=row["enabled"]) for row in payload["cameras"]]
        if not self.box.is_stopped():
            self.box.pending_at = requested
        return list(self.records)
