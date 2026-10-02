import time
from pathlib import Path
from .. import boxconfig
from ..feedback import MuteState, Feedback

class AlertPause:
    def __init__(self,demo=False,path=None,clock=time.time):
        self.demo,self.clock=demo,clock
        self.path=Path(path) if path else Path(boxconfig.LOG_DIR)/'alert_mute.json'
        self.demo_until=0
    def status(self,cameras):
        now=self.clock()
        if self.demo: return (self.demo_until,False) if self.demo_until>now else (None,False)
        mute=MuteState(str(self.path))
        global_until=mute.muted_until(now,'')
        if global_until: return global_until,False
        deadlines=[mute.muted_until(now,name) for name in cameras]
        return max((value for value in deadlines if value),default=None),True
    def resume(self):
        if self.demo: self.demo_until=0
        else: MuteState(str(self.path)).apply(Feedback(action='resume'),self.clock())
