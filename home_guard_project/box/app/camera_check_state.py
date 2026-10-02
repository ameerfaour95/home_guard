"""Photo failures cannot undo a successful setup."""
from dataclasses import dataclass
from .engine_backend import OutputParser
from .strings import tr

@dataclass
class CameraCheckState:
    setup_finished: bool = False
    house: str = ''
    cameras: tuple = ()
    photo_failed: bool = False
    details: str = ''
    def heading(self):
        if not self.setup_finished: return tr('camera_check_only_hint')
        return tr('camera_setup_finished',house=self.house,count=len(self.cameras),names=', '.join(self.cameras) or tr('camera_none_found'))
    def failure(self, reason, output='', target=''):
        self.photo_failed=True
        parser=OutputParser((target,))
        self.details='\n'.join(parser.safe(line) for line in (output or str(reason)).splitlines() if line.strip())
    def clear_failure(self):
        self.photo_failed=False
        self.details=''
