"""Remember only the last validated box address; never persist form secrets."""
from pathlib import Path
import json
import os
import tempfile
from .remote_cameras import target_user

class AddressPreference:
    def __init__(self,path=None):
        self.path=Path(path) if path else Path.home()/'.homeguard'/'last_box.json'
    def load(self):
        try:
            value=json.loads(self.path.read_text(encoding='utf-8'))['target']
            target_user(value)
            return value
        except (OSError,ValueError,KeyError,TypeError): return ''
    def save(self,target):
        target_user(target)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,name=tempfile.mkstemp(prefix='last-box-',suffix='.tmp',dir=self.path.parent)
        temporary=Path(name)
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump({'target':target},stream)
            os.replace(temporary,self.path)
        finally: temporary.unlink(missing_ok=True)
