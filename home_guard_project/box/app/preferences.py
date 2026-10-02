"""Remember only the last validated box address; never persist form secrets."""
from pathlib import Path
import json
import os
import tempfile
from dataclasses import dataclass,asdict
from .remote_cameras import target_user

@dataclass(frozen=True)
class ViewerSettings:
    detections: bool = False
    detection_labels: str = 'confidence'
    def __post_init__(self):
        if type(self.detections) is not bool or self.detection_labels not in ('confidence','name','none'):
            raise ValueError('Invalid viewer settings')

class ViewerPreference:
    def __init__(self,path=None):
        self.path=Path(path) if path else Path.home()/'.homeguard'/'viewer.json'
    def load(self):
        try: return ViewerSettings(**json.loads(self.path.read_text(encoding='utf-8')))
        except (OSError,ValueError,TypeError): return ViewerSettings()
    def save(self,settings):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,name=tempfile.mkstemp(prefix='viewer-',suffix='.tmp',dir=self.path.parent)
        temporary=Path(name)
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream: json.dump(asdict(settings),stream)
            os.replace(temporary,self.path)
        finally: temporary.unlink(missing_ok=True)

class AddressPreference:
    def __init__(self,path=None):
        self.path=Path(path) if path else Path.home()/'.homeguard'/'last_box.json'
    def load(self):
        try:
            value=json.loads(self.path.read_text(encoding='utf-8'))['target']
            target_user(value)
            return value
        except (OSError,ValueError,KeyError,TypeError): return ''
    def load_user(self):
        try:
            value=json.loads(self.path.read_text(encoding="utf-8")).get("successful_user", "")
            target_user(value+"@box.example")
            return value
        except (OSError,ValueError,TypeError,AttributeError): return ""
    def save_success(self,target):
        self.save(target, successful_user=target_user(target))
    def save(self,target,successful_user=None):
        target_user(target)
        remembered = successful_user or self.load_user()
        data = {"target": target}
        if remembered: data["successful_user"] = remembered
        self.path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,name=tempfile.mkstemp(prefix='last-box-',suffix='.tmp',dir=self.path.parent)
        temporary=Path(name)
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump(data,stream)
            os.replace(temporary,self.path)
        finally: temporary.unlink(missing_ok=True)
