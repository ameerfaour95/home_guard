"""Persist display preferences and last email; never passwords, codes or tokens."""
import json
import os
from pathlib import Path


class Preferences:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(os.environ.get('APPDATA', str(Path.home()))) / 'HomeGuardAdmin' / 'prefs.json'

    def email(self):
        return self.get('email', '')

    def get(self, key, default=''):
        try:
            value = json.loads(self.path.read_text(encoding='utf-8')).get(key, default)
            return value if isinstance(value, str) else default
        except (OSError, ValueError, AttributeError):
            return default

    def save_email(self, email):
        self.save(email=email)

    def save(self, **changes):
        try:
            values = {key:self.get(key) for key in ('email', 'server', 'theme')}
            values.update({key:value for key,value in changes.items() if key in values})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({k:v for k,v in values.items() if v}), encoding='utf-8')
            temp.replace(self.path)
        except OSError:
            pass  # A read-only profile must not prevent sign-in.
