"""Present recorded AI decisions without contacting the detector or Telegram."""
from dataclasses import dataclass
from .detector_view import fresh,timestamp,object_name
from .engine_backend import OutputParser
from .strings import tr

@dataclass(frozen=True)
class Decision:
    ts: float
    camera: str
    labels: tuple
    summary: str
    command: str
    sent: bool
    false_positive: bool
    muted: bool
    error: str
    label: str = ""
    def outcome(self):
        if self.sent: return ('ok',tr('ai_urgent_sent' if self.command=='[call_owner]' else 'ai_sent'))
        if self.false_positive: return ('muted',tr('ai_training'))
        if self.muted: return ('warning',tr('ai_muted'))
        return ('error',tr('ai_not_sent'))
    def seen(self): return ', '.join(object_name(label) for label in self.labels)

def decisions(data,limit=20):
    raw=data.get('decisions',[]) if isinstance(data,dict) else []
    result=[]
    parser=OutputParser()
    for entry in raw if isinstance(raw,list) else []:
        if not isinstance(entry,dict): continue
        labels=entry.get('labels',[])
        result.append(Decision(timestamp(entry.get('ts')),str(entry.get('camera') or ''),tuple(str(v) for v in labels if isinstance(v,str)) if isinstance(labels,list) else (),str(entry.get('summary') or ''),str(entry.get('command') or ''),entry.get('sent') is True,entry.get('false_positive') is True,entry.get('muted') is True,parser.safe(str(entry.get('error') or '')),str(entry.get('label') or '')))
    return sorted(reversed(result),key=lambda d:d.ts,reverse=True)[:limit]

def status_note(data,now,stopped):
    if stopped: return tr('ai_stopped')
    if not fresh(data.get('updated') if isinstance(data,dict) else None,now,15): return tr('ai_not_updating')
    return ''

def undelivered_alert(data):
    for decision in decisions(data,limit=50):
        if decision.false_positive or decision.muted: continue
        return decision if not decision.sent else None
    return None
