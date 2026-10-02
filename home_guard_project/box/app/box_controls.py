"""Local control adapter and an isolated, mutable demo box."""
from dataclasses import dataclass
from pathlib import Path
import time
from .. import control, boxconfig


class BoxControls:
    def __init__(self, demo=False, stopped=False, clock=time.time):
        self.demo = demo
        self._stopped = stopped
        self.clock = clock

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
