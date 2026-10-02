from pathlib import Path
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace

from .. import boxconfig

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

class CameraControls:
    def __init__(self, box, names=(), runner=None):
        self.box = box
        self.runner = runner or subprocess.run
        self.records = [Camera(name.lower().replace(" ", "_"), True, ok=True) for name in names]
        self.out = Path(boxconfig.LOG_DIR) / "app_snapshots"

    def command(self, *args):
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        result = self.runner([sys.executable, "-m", "home_guard_project.box.find_cameras", "--json", *map(str, args)], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", timeout=480, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        data = json.loads(result.stdout)
        if "error" in data:
            raise ValueError("Camera command failed")
        return result.returncode, data

    def load(self):
        if self.box.demo:
            return list(self.records)
        from ..find_cameras import _read_cameras_raw
        raw = _read_cameras_raw()
        self.records = [Camera(name, enabled) for group, enabled in (("cameras", True), ("disabled", False)) for name in raw.get(group, {})]
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
        self.records = [replace(old[row["name"]], name=row["new_name"], enabled=row["enabled"]) for row in payload["cameras"]]
        if not self.box.is_stopped():
            self.box.pending_at = requested
        return list(self.records)
