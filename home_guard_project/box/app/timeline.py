"""Bounded, chronological timeline merged from the two local contracts."""
from dataclasses import dataclass
from pathlib import Path
import re
from .ai_view import decisions
from .detector_view import timestamp,fresh
from .engine_backend import OutputParser

@dataclass(frozen=True)
class Item:
    ts: float
    who: str
    kind: str
    text: str
    name: str = ''
    camera: str = ''
    image: str = ''
    delivered: bool = True
    error: str = ''
    urgent: bool = False
    muted: bool = False

def image_path(directory,name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.jpe?g',name,re.I): return None
    return Path(directory)/name

def merge_timeline(data,feed):
    result=[];parser=OutputParser()
    for entry in feed if isinstance(feed,list) else []:
        if not isinstance(entry,dict) or entry.get('who') not in ('box','owner','assistant'): continue
        result.append(Item(timestamp(entry.get('ts')),entry['who'],str(entry.get('kind','message')),str(entry.get('text','')),str(entry.get('name','')),str(entry.get('camera','')),str(entry.get('image','')),entry.get('delivered') is True,parser.safe(str(entry.get('error',''))),entry.get('command')=='[call_owner]'))
    for d in decisions(data,50):
        if d.false_positive:
            result.append(Item(d.ts,'assistant','quiet',d.summary,camera=d.camera))
        elif d.muted:
            result.append(Item(d.ts,'assistant','quiet',d.summary,camera=d.camera,muted=True))
        elif not any(row.who=='box' and row.camera==d.camera and abs(row.ts-d.ts)<5 for row in result):
            result.append(Item(d.ts,'box','alert',d.summary,camera=d.camera,delivered=d.sent,error=d.error,urgent=d.command=='[call_owner]'))
        elif d.command=='[call_owner]':
            from dataclasses import replace
            result=[replace(row,urgent=True) if row.who=='box' and row.camera==d.camera and abs(row.ts-d.ts)<5 else row for row in result]
    return sorted(result,key=lambda item:item.ts)[-200:]

def thinking_camera(data,now):
    thinking=data.get('thinking')
    if not isinstance(thinking,dict) or not fresh(thinking.get('ts'),now,60) or now-timestamp(thinking.get('ts'))>=60: return ''
    return str(thinking.get('camera') or '').replace('_',' ').title()

@dataclass(frozen=True)
class QuietGroup:
    records: tuple
    @property
    def ts(self): return self.records[-1].ts
    @property
    def camera(self): return self.records[-1].camera

def group_quiet(rows):
    """Only adjacent quiet events from the same camera and day form a group."""
    import time
    result=[]
    for row in rows:
        if row.kind=='quiet':
            previous=result[-1] if result else None
            if isinstance(previous,QuietGroup) and previous.camera==row.camera and time.localtime(previous.ts)[:3]==time.localtime(row.ts)[:3]:
                result[-1]=QuietGroup(previous.records+(row,))
            else: result.append(QuietGroup((row,)))
        else: result.append(row)
    return result

def text_direction(text):
    import unicodedata
    for char in text:
        direction=unicodedata.bidirectional(char)
        if direction in ('R','AL'): return 'rtl'
        if direction=='L': return 'ltr'
    return 'ltr'
