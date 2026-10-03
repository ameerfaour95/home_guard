"""The shared People, Vehicles and Animals outlines."""
from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainterPath


def draw_type_icon(p, kind):
    if kind == 'person':
        p.drawEllipse(QRectF(10, 0, 12, 12))
        p.drawRoundedRect(QRectF(5, 17, 22, 17), 7, 7)
    elif kind == 'vehicle':
        p.drawRoundedRect(QRectF(1, 13, 32, 16), 4, 4)
        p.drawLine(5, 13, 10, 4); p.drawLine(10, 4, 25, 4); p.drawLine(25, 4, 30, 13)
        p.drawLine(7, 19, 10, 19); p.drawLine(24, 19, 27, 19)
        p.drawLine(7, 29, 7, 33); p.drawLine(27, 29, 27, 33)
    else:
        for x, y in ((1, 10), (9, 1), (21, 1), (29, 10)):
            p.drawEllipse(QRectF(x, y, 6, 9))
        paw = QPainterPath(); paw.moveTo(8, 29); paw.cubicTo(7, 22, 14, 16, 18, 17); paw.cubicTo(23, 16, 30, 25, 27, 30); paw.cubicTo(24, 35, 20, 30, 18, 31); paw.cubicTo(13, 34, 8, 34, 8, 29)
        p.drawPath(paw)
