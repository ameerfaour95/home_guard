"""Bounded viewer-only optical flow between real detector observations.

Flow runs on small grayscale frames in the decode worker. It never extends
visibility: camera_view / entry_opacity own expiry and fading.
"""
from dataclasses import dataclass, replace
import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from .detector_view import camera_view, timestamp
from .motion import HOVER_MS


@dataclass
class TrackState:
    stamp: float
    interval: float
    gray: object
    anchors: tuple
    objects: tuple
    arrived: float
    ease_from: tuple = ()


def bounded_box(anchor, box):
    """Translation is at most half the last real box width, in any direction."""
    dx,dy=box[0]-anchor[0],box[1]-anchor[1]
    distance=math.hypot(dx,dy)
    limit=(anchor[2]-anchor[0])/2
    if distance>limit:
        dx*=limit/distance;dy*=limit/distance
    dx=max(-anchor[0],min(1-anchor[2],dx))
    dy=max(-anchor[1],min(1-anchor[3],dy))
    return anchor[0]+dx,anchor[1]+dy,anchor[2]+dx,anchor[3]+dy


class OverlayTracker:
    def __init__(self): self.states={}

    @staticmethod
    def flow(previous,gray,objects):
        import cv2
        import numpy as np
        height,width=gray.shape
        moved=[]
        for obj in objects:
            x1,y1,x2,y2=obj.box
            mask=np.zeros_like(gray)
            mask[int(y1*height):int(y2*height),int(x1*width):int(x2*width)]=255
            points=cv2.goodFeaturesToTrack(previous,12,.03,4,mask=mask)
            box=obj.box
            if points is not None and len(points)>=3:
                next_points,valid,error=cv2.calcOpticalFlowPyrLK(previous,gray,points,None,
                    winSize=(15,15),maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,.03))
                if next_points is not None:
                    good=(valid.reshape(-1)>0)&(error.reshape(-1)<30)
                    if good.sum()>=3:
                        shift=np.median((next_points-points).reshape(-1,2)[good],axis=0)
                        dx,dy=float(shift[0])/width,float(shift[1])/height
                        if abs(dx)<=.15 and abs(dy)<=.15:
                            box=(x1+dx,y1+dy,x2+dx,y2+dy)
            moved.append(replace(obj,box=box))
        return tuple(moved)

    def update(self,name,image,data,now):
        import numpy as np
        observations,_=camera_view(data,name,now)
        if not observations:
            self.states.pop(name,None)
            return ()
        small=image.scaled(320,180,Qt.AspectRatioMode.KeepAspectRatio).convertToFormat(QImage.Format.Format_Grayscale8)
        gray=np.frombuffer(small.constBits(),np.uint8).reshape(small.height(),small.bytesPerLine())[:,:small.width()].copy()
        stamp=timestamp(data['cameras'][name].get('ts'))
        old=self.states.get(name)
        if old is None or old.gray.shape!=gray.shape:
            # Until two samples establish cadence, use a conservative one second.
            state=TrackState(stamp,1.,gray,observations[:12],observations[:12],now)
            self.states[name]=state
            return state.objects
        if stamp>old.stamp:
            remaining=list(old.objects)
            starts=[]
            for obj in observations[:12]:
                candidates=[o for o in remaining if o.label==obj.label]
                match=min(candidates,key=lambda o:sum((a-b)**2 for a,b in zip(o.box,obj.box))) if candidates else obj
                starts.append(replace(obj,box=match.box))
                if match in remaining: remaining.remove(match)
            old=TrackState(stamp,stamp-old.stamp,gray,observations[:12],tuple(starts),now,tuple(starts))
            self.states[name]=old
        if old.ease_from:
            progress=max(0.,min(1.,(now-old.arrived)/(HOVER_MS/1000)))
            eased=1-(1-progress)**3
            old.objects=tuple(replace(target,box=tuple(a+(b-a)*eased for a,b in zip(start.box,target.box)))
                              for start,target in zip(old.ease_from,old.anchors))
            if progress>=1: old.ease_from=()
        elif 0<=now-old.stamp<=old.interval:
            predicted=self.flow(old.gray,gray,old.objects)
            old.objects=tuple(replace(obj,box=bounded_box(anchor.box,obj.box))
                              for anchor,obj in zip(old.anchors,predicted))
        # Beyond one interval no extrapolation; thus also frozen after 1.5.
        # Keep the last bounded estimate until detector_view expires it.
        old.gray=gray
        return old.objects
