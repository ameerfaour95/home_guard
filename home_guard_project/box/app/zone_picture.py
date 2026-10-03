"""Saved watch areas on camera cards; the pixmap stays raw for the editor."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor

from .ai_activity_ui import AlertPicture
from .zone_editor import alpha, animate, colors, polygon_path
from . import motion


class ZonePicture(AlertPicture):
    def __init__(self, path, points=()):
        super().__init__(path)
        self.points = list(points)
        self.previous = []
        self.fade = 1.

    def set_zone(self, points):
        self.previous, self.points = self.points, list(points)
        if getattr(self, 'animation', None): self.animation.stop()
        self.animation = animate(self, 0., 1., motion.PANE_MS, self.advance)

    def advance(self, value):
        self.fade = float(value); self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pix.isNull(): return
        t = colors(self)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()); clip = QPainterPath(); clip.addRoundedRect(rect, 10, 10); p.setClipPath(clip)
        scale = max(rect.width()/self.pix.width(), rect.height()/self.pix.height())
        w, h = self.pix.width()*scale, self.pix.height()*scale
        picture = QRectF((rect.width()-w)/2, (rect.height()-h)/2, w, h)
        for points, opacity in ((self.previous, 1-self.fade), (self.points, self.fade)):
            if len(points) < 3 or opacity <= 0: continue
            shape = polygon_path(points, picture)
            outside = QPainterPath(); outside.addRect(picture)
            p.fillPath(outside.subtracted(shape), alpha(t['bg'], .35*opacity))
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(alpha(t['action'], opacity), 1.5)); p.drawPath(shape)
