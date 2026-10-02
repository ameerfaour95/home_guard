"""Slider values and confirmation from the program's reported live settings."""
import math
from ..boxconfig import LIVE_OPTIONS
KEYS={'inference_conf':'conf','alert_start_hour':'alert_start_hour','alert_end_hour':'alert_end_hour','alert_cooldown_sec':'cooldown_sec'}
def slider_conf(value): return int(value)/20
def conf_slider(value): return max(1,min(19,int(math.floor(float(value)*20+.5))))

class AppliedState:
    def __init__(self): self.expected={};self.requested=0
    def request(self,before,after,now):
        self.expected.update({KEYS[key]:value for key,value in after.options().items() if key in LIVE_OPTIONS and before.options()[key]!=value})
        self.requested=now
    def status(self,data,now):
        if not self.expected: return ''
        settings=data.get('settings',{}) if isinstance(data,dict) else {}
        if not isinstance(settings,dict): settings={}
        try:
            matches=all(type(settings.get(key)) in (int,float) and math.isfinite(settings[key]) and abs(settings[key]-value)<1e-6 for key,value in self.expected.items())
            fresh=0<=now-float(data.get('updated',0))<=15
        except (TypeError,ValueError): matches=False;fresh=False
        if matches and fresh: self.expected={};return 'applied'
        return 'live_not_picked_up' if now-self.requested>=10 else 'applying'
