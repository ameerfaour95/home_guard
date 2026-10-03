"""Only the last email is persisted, never passwords, codes or tokens."""
import json
import os
from pathlib import Path


class Preferences:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(os.environ.get('APPDATA', str(Path.home()))) / 'HomeGuardAdmin' / 'prefs.json'

    def email(self):
        try:
            value = json.loads(self.path.read_text(encoding='utf-8')).get('email', '')
            return value if isinstance(value, str) else ''
        except (OSError, ValueError, AttributeError):
            return ''

    def save_email(self, email):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({'email': email}), encoding='utf-8')
            temp.replace(self.path)
        except OSError:
            pass  # A read-only profile must not prevent sign-in.
