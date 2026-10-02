"""Camera search progress advances on meaningful messages, not decoder noise."""
from dataclasses import dataclass
from .setup_details import readable_line
from .strings import tr

@dataclass
class CameraSearch:
    started_at: float = 0
    running: bool = False
    sentence: str = ''
    role: str = 'muted'
    def start(self, now):
        self.started_at=now
        self.running=True
        self.sentence=tr('camera_search_wait')
        self.role='muted'
    def feed(self, line):
        if not self.running: return
        message=readable_line(line)
        if message and message[1] != line.strip():
            self.role,self.sentence=message
    def elapsed(self, now):
        return max(0,int(now-self.started_at))
    def finish(self): self.running=False
