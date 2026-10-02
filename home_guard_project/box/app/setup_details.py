"""Readable setup history, separate from its already-redacted technical log."""
from dataclasses import dataclass, field
import re
from .strings import tr
from .engine_backend import ENGINE_STEPS

@dataclass
class StepDetails:
    status: str = 'pending'
    result: str = ''
    messages: list = field(default_factory=list)

def readable_line(line):
    text=line.strip()
    if not text: return None
    low=text.lower()
    match=re.search(r'(\d+) of (\d+) device\(s\) refused this login',low)
    if match: return ('error',tr('detail_camera_refused_count',count=match[1],total=match[2]))
    match=re.search(r'(\d+) channel\(s\) answer',low)
    if match: return ('muted',tr('detail_recorder_answers',count=match[1]))
    match=re.search(r'channel (\d+): ok \((\d+x\d+)\)',low)
    if match: return ('ok',tr('detail_channel_ok',channel=match[1],size=match[2]))
    match=re.search(r'channel (\d+): answers, but no picture arrived',low)
    if match: return ('warning',tr('detail_channel_no_picture',channel=match[1]))
    if 'does not answer; skipped' in low: return ('warning',tr('detail_device_skipped'))
    if '401' in low and ('unauthorized' in low or 'describe' in low or 'login' in low): return ('error',tr('detail_login_refused'))
    if 'timed out' in low or 'timeout' in low: return ('warning',tr('detail_timeout'))
    if 'no working' in low and ('rtsp' in low or 'stream' in low): return ('warning',tr('detail_no_stream'))
    if re.search(r'\[\w+ @ (?:0x)?[0-9a-f]+\]',low) or 'cap_ffmpeg_impl' in low or '_opencv_' in low: return None
    if 'trying' in low and ('pattern' in low or 'stream' in low or 'rtsp' in low): return ('muted',tr('detail_try_stream'))
    if 'locating cameras' in low or 'searching for cameras' in low: return ('muted',tr('detail_find_cameras'))
    if re.search(r'updat(?:e|ing)|git pull|fetching.*(?:software|repository)|already up.to.date',low): return ('muted',tr('detail_update'))
    if re.search(r'copying|copied|\bscp\b|transferred.*file',low): return ('muted',tr('detail_copy'))
    if re.search(r'\bssh\b|running.*(?:command|on.*box)|executing.*(?:command|on.*box)',low): return ('muted',tr('detail_command'))
    return ('error' if '[fail]' in low or 'error' in low else 'warning' if '[warn]' in low or 'warning' in low else 'ok' if '[ok]' in low else 'muted',text)

class DetailsModel:
    def __init__(self):
        self.groups={step:StepDetails() for step in ENGINE_STEPS}
        self.current='connect';self.failed=None;self.raw=[]
    def feed(self,event):
        if event.kind=='step':
            self.current=event.step
            group=self.groups[event.step];group.status=event.status;group.result=event.text
            self.raw.append('@@step '+event.step+' '+event.status+(' '+event.text if event.text else ''))
            if event.text:
                message=('error' if event.status=='fail' else 'warning' if event.status=='warn' else 'muted',event.text)
                if message not in group.messages: group.messages.append(message)
            if event.status=='fail': self.failed=event.step
        elif event.kind=='detail':
            if event.text.strip(): self.raw.append(event.text)
            message=readable_line(event.text)
            if message and message not in self.groups[self.current].messages: self.groups[self.current].messages.append(message)
        elif event.kind=='check':
            self.raw.append('@@check '+event.status+' '+event.text)
            message=('error' if event.status=='FAIL' else 'warning' if event.status=='WARN' else 'ok',event.text)
            if message not in self.groups['readiness'].messages: self.groups['readiness'].messages.append(message)
        elif event.kind=='camera':
            self.raw.append('@@camera '+event.name+' '+event.text)
            message=('ok',tr('detail_camera_found',name=event.name,size=event.text))
            if message not in self.groups['cameras'].messages: self.groups['cameras'].messages.append(message)
        elif event.kind=='done': self.raw.append('@@done '+event.status)
        # Rescue credentials are intentionally absent from the support log.
    def technical_log(self): return '\n'.join(self.raw)
