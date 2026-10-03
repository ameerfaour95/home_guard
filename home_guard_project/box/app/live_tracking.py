"""Small viewer-only motion bridge between timestamped detector observations.

Sparse optical flow runs on 320-pixel grayscale frames in the decode worker. It
does not run detection, change detector output, or extend its three-second TTL.
Lost tracks disappear; the next detector observation seeds them again.
"""
from dataclasses import replace
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from .detector_view import camera_view


class OverlayTracker:
    def __init__(self): self.states={}

    def update(self,name,image,data,now):
        import cv2
        import numpy as np
        entries=data.get('cameras',{})
        entry=entries.get(name,{}) if isinstance(entries,dict) else {}
        observations,_=camera_view(data,name,now)
        if not observations:
            self.states.pop(name,None)
            return ()
        small=image.scaled(320,180,Qt.AspectRatioMode.KeepAspectRatio).convertToFormat(QImage.Format.Format_Grayscale8)
        gray=np.frombuffer(small.constBits(),np.uint8).reshape(small.height(),small.bytesPerLine())[:,:small.width()].copy()
        height,width=gray.shape
        stamp=entry.get('ts')
        old=self.states.get(name)
        if old is None or old[0]!=stamp or old[1].shape!=gray.shape:
            objects=observations[:12]
        else:
            _,previous,objects=old
            moved=[]
            for obj in objects:
                x1,y1,x2,y2=obj.box
                mask=np.zeros_like(gray)
                mask[int(y1*height):int(y2*height),int(x1*width):int(x2*width)]=255
                points=cv2.goodFeaturesToTrack(previous,12,.03,4,mask=mask)
                if points is None or len(points)<3: continue
                next_points,valid,error=cv2.calcOpticalFlowPyrLK(previous,gray,points,None,
                    winSize=(15,15),maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,.03))
                if next_points is None: continue
                good=(valid.reshape(-1)>0)&(error.reshape(-1)<30)
                if good.sum()<3: continue
                shift=np.median((next_points-points).reshape(-1,2)[good],axis=0)
                dx,dy=float(shift[0])/width,float(shift[1])/height
                if abs(dx)>.15 or abs(dy)>.15: continue
                box=(max(0,x1+dx),max(0,y1+dy),min(1,x2+dx),min(1,y2+dy))
                if box[2]>box[0] and box[3]>box[1]: moved.append(replace(obj,box=box))
            objects=tuple(moved)
        self.states[name]=(stamp,gray,objects)
        return objects
