"""Time and geometry rules for the detector display; no camera access."""
from collections import Counter
from dataclasses import dataclass
import math
from .strings import tr
from .theme import DETECTOR_PERSON, DETECTOR_VEHICLE, MUTED

VEHICLES={'car','truck','bus','motorcycle','bicycle','boat','train'}

def timestamp(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else 0
    except (TypeError,ValueError): return 0

def fresh(value,now,seconds):
    age=now-timestamp(value)
    return 0 <= age <= seconds and timestamp(value)>0

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple
    @property
    def color(self): return DETECTOR_PERSON if self.label=='person' else DETECTOR_VEHICLE if self.label in VEHICLES else MUTED
    @property
    def width(self): return 2 if self.label=='person' or self.label in VEHICLES else 1
    def caption(self): return tr('detection_label',label=object_name(self.label),percent=round(self.confidence*100))

def object_name(name,count=1):
    from .strings import TEXT
    key='object_'+name.replace(' ','_')+('_plural' if count!=1 else '')
    return tr(key) if key in TEXT else name

def camera_view(data,name,now,stopped=False):
    if stopped: return (),tr('stopped')
    cameras=data.get('cameras',{}) if isinstance(data,dict) else {}
    entry=cameras.get(name,{}) if isinstance(cameras,dict) else {}
    if not isinstance(entry,dict): entry={}
    found=[]
    if fresh(data.get('updated'),now,15) and fresh(entry.get('ts'),now,3):
        objects=entry.get('objects',[])
        for obj in objects if isinstance(objects,list) else []:
            if not isinstance(obj,dict): continue
            box=obj.get('box')
            try:
                coords=tuple(float(v) for v in box)
                conf=float(obj.get('conf',0));name_=obj.get('label')
                if len(coords)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in coords) or coords[2]<=coords[0] or coords[3]<=coords[1] or not math.isfinite(conf) or not 0<=conf<=1 or not isinstance(name_,str): continue
                found.append(Detection(name_,conf,coords))
            except (TypeError,ValueError): continue
    if found:
        counts=Counter(d.label for d in found)
        return tuple(found),tr('detector_sees',objects=', '.join(tr('object_count',count=n,label=object_name(label,n)) for label,n in counts.items()))
    return (),tr('detector_nothing') if fresh(entry.get('checked_ts'),now,10) else tr('detector_not_looking')

def box_rect(box,picture):
    x,y,width,height=picture
    x1,y1,x2,y2=box
    return x+x1*width,y+y1*height,(x2-x1)*width,(y2-y1)*height
