"""Picture placement and names read only from local files."""
from PySide6.QtCore import Qt,QRectF
from PySide6.QtGui import QPixmap,QPainter
from PySide6.QtWidgets import QGraphicsScene,QGraphicsPixmapItem,QGraphicsBlurEffect

def image_rect(area,source,hero=True,dpr=1):
    x,y,w,h=area;sw,sh=source
    scale=min(w/sw,h/sh,1/max(1,dpr)) if hero else max(w/sw,h/sh)
    width,height=sw*scale,sh*scale
    return x+(w-width)/2,y+(h-height)/2,width,height

def ambient_picture(pix):
    small=pix.scaled(256,256,Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation)
    scene=QGraphicsScene();item=QGraphicsPixmapItem(small);effect=QGraphicsBlurEffect();effect.setBlurRadius(12);item.setGraphicsEffect(effect);scene.addItem(item)
    result=QPixmap(small.size());result.fill(Qt.GlobalColor.transparent);p=QPainter(result);scene.render(p,QRectF(result.rect()),QRectF(small.rect()));p.end();return result

def active_names(preview,fallback):
    from ..find_cameras import _read_cameras_raw
    try:
        raw=_read_cameras_raw()
        if not isinstance(raw,dict) or not isinstance(raw.get('cameras'),dict): return preview or fallback
        active=list(raw['cameras']);disabled=set(raw.get('disabled',{}))
        return list(dict.fromkeys([name for name in (preview or active) if name in active and name not in disabled]+active))
    except (OSError,ValueError,TypeError): return preview or fallback
