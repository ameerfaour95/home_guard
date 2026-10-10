"""Local viewer and installer preferences; never persist customer details or secrets."""
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
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump(asdict(settings),stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,self.path)
        finally: temporary.unlink(missing_ok=True)

class LanguagePreference:
    """The installer's language for setup and the laptop's viewer ("en" or "he"), remembered on this laptop."""
    def __init__(self,path=None):
        self.path=Path(path) if path else Path.home()/'.homeguard'/'language.json'
    def load(self,default='en'):
        try:
            value=json.loads(self.path.read_text(encoding='utf-8'))['language']
            return value if value in ('en','he') else default
        except (OSError,ValueError,KeyError,TypeError): return default
    def save(self,language):
        if language not in ('en','he'): raise ValueError('Invalid language')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,name=tempfile.mkstemp(prefix='language-',suffix='.tmp',dir=self.path.parent)
        temporary=Path(name)
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump({'language':language},stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,self.path)
        finally: temporary.unlink(missing_ok=True)


def app_language(args):
    """The language the app opens in: --lang, else on the box its owner's (box.yaml owner_language), else (setup
    and the viewer on a laptop) the installer's remembered choice; English when nothing says."""
    if getattr(args,'lang',None) in ('en','he'): return args.lang
    if getattr(args,'demo',False): return 'en'
    if getattr(args,'setup',False) or getattr(args,'remote_box',None): return LanguagePreference().load()
    try:
        from .. import boxconfig
        value=str(boxconfig.load_box_settings().get('owner_language') or 'en')
        return value if value in ('en','he') else 'en'
    except Exception: return 'en'


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
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,self.path)
        finally: temporary.unlink(missing_ok=True)


class InstallerPreference:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path.home()/'.homeguard'/'installer.json'

    @staticmethod
    def validate(value):
        if not isinstance(value, str) or len(value.strip()) > 120 or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError('Invalid installer name')
        return value.strip()

    def load(self):
        try: return self.validate(json.loads(self.path.read_text(encoding='utf-8'))['installer'])
        except (OSError, ValueError, KeyError, TypeError): return ''

    def save(self, name):
        name = self.validate(name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name_tmp = tempfile.mkstemp(prefix='installer-', suffix='.tmp', dir=self.path.parent)
        temporary = Path(name_tmp)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump({'installer': name}, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally: temporary.unlink(missing_ok=True)
