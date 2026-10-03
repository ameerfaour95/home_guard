"""Small native line icons, shared by rail, tabs and completeness markers."""
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor
from ..theme import PALETTES


def draw_icon(p, name, rect, color):
    p.save()
    p.translate(rect.x(), rect.y())
    p.scale(rect.width()/24, rect.height()/24)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(color), 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    if name in ('Fleet', 'Timeline'):
        for y in (5, 12, 19):
            p.drawLine(4, y, 20, y)
            p.drawEllipse(QPointF(8 if y == 12 else 15, y), 2, 2)
    elif name in ('Review', 'reviewed'):
        p.drawRoundedRect(QRectF(5, 3, 14, 18), 2, 2)
        p.drawLine(8, 12, 11, 15); p.drawLine(11, 15, 17, 8)
    elif name in ('Studio', 'boxes'):
        for x, y in ((4, 4), (14, 4), (4, 14), (14, 14)):
            p.drawRect(QRectF(x, y, 6, 6))
    elif name in ('Event', 'video'):
        p.drawRoundedRect(QRectF(3, 5, 18, 14), 2, 2)
        p.drawLine(10, 9, 15, 12); p.drawLine(15, 12, 10, 15); p.drawLine(10, 15, 10, 9)
    elif name in ('Conversation', 'feedback'):
        p.drawRoundedRect(QRectF(3, 4, 18, 14), 3, 3)
        p.drawLine(7, 18, 7, 21); p.drawLine(7, 21, 11, 18)
    elif name in ('Access', 'expired'):
        p.drawRoundedRect(QRectF(5, 10, 14, 11), 2, 2)
        p.drawArc(QRectF(8, 2, 8, 13), 0, 180*16)
    elif name == 'flagged':
        p.drawLine(6, 3, 6, 22); p.drawRect(QRectF(6, 4, 13, 9))
    elif name == 'ai':
        p.drawEllipse(QRectF(5, 5, 14, 14))
        p.drawLine(12, 1, 12, 5); p.drawLine(12, 19, 12, 23)
        p.drawLine(1, 12, 5, 12); p.drawLine(19, 12, 23, 12)
    else:
        p.drawRoundedRect(QRectF(5, 3, 14, 18), 2, 2)
        for y in (8, 12, 16):
            p.drawLine(8, y, 16, y)
    p.restore()


def icon(name, theme='dark'):
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    draw_icon(painter, name, QRectF(0, 0, 24, 24), PALETTES[theme]['muted'])
    painter.end()
    return QIcon(pix)
