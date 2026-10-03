from datetime import timedelta
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget, QToolTip
from ..event_logic import density
from ..formatting import local_time
from ..theme import PALETTES
from datetime import timedelta
from zoneinfo import ZoneInfo


def bar_metrics(events, alerts, peak, height=28):
    total = height * max(0, events) / max(1, peak)
    return total, total * min(max(0, alerts), events) / max(1, events)


def activity_tooltip(hour, counts, zone):
    start = hour.astimezone(ZoneInfo(zone)).strftime('%H:%M')
    end = (hour+timedelta(hours=1)).astimezone(ZoneInfo(zone)).strftime('%H:%M')
    events, alerts, false = counts
    return f'{start}–{end} · {events} events · {alerts} alerts · {false} false alarm' + ('' if false == 1 else 's')


class DensityStrip(QWidget):
    selected = Signal(str, object)

    def __init__(self, theme='dark', fleet=False):
        super().__init__()
        self.tokens, self.fleet = PALETTES[theme], fleet
        self.hours, self.rows, self.zone = [], {}, 'UTC'
        self.message = 'Loading activity…'
        self.setMouseTracking(True)
        self.setMinimumHeight(66)

    def set_density(self, result):
        self.zone, self.hours = result.timezone, result.starts_utc
        self.rows = {row.camera: list(zip(row.events, row.alerts, row.false_alarms)) for row in result.rows}
        if self.fleet:
            self.rows = {'Fleet activity': [tuple(sum(row[i][j] for row in self.rows.values()) for j in range(3))
                                           for i in range(len(self.hours))]}
        self.setFixedHeight(78 if self.fleet else 24+22*max(1, len(self.rows)))
        self.update()

    def set_error(self):
        self.hours,self.rows = [],{}
        self.message = 'Activity unavailable · Refresh to retry'
        self.update()

    def set_events(self, events, start, end, zone='UTC'):
        self.zone = zone
        self.hours, self.rows = density(events, start, end)
        if self.fleet:
            self.rows = {'Fleet activity': [list(map(sum, zip(*(row[i] for row in self.rows.values()))))
                                          for i in range(len(self.hours))]} if self.rows else {}
        self.setFixedHeight(62 if self.fleet else 24+22*max(1, len(self.rows)))
        self.update()

    def cell_rect(self, row, col):
        width = (self.width()-150)/max(1, len(self.hours))
        if self.fleet:
            return QRectF(150+col*width, 8, max(1, width-3), 34)
        return QRectF(150+col*width, row*22+22, max(1, width-3), 18)

    def paintEvent(self, event):
        p = QPainter(self)
        t = self.tokens
        p.setPen(QColor(t['muted']))
        p.drawText(0, 14, '24 h activity' if self.fleet else 'CAMERA / HOUR')
        if not self.hours:
            p.drawText(150, 36, self.message)
            return
        if self.fleet:
            p.drawText(0, 36, 'Events / alerts')
            p.setPen(QColor(t['border'])); p.drawLine(150, 42, self.width(), 42)
            cells = self.rows.get('Fleet activity', [])
            peak = max(5, max((v[0] for v in cells), default=0))
            for i, (count, alerts, false) in enumerate(cells):
                cell = self.cell_rect(0, i)
                height, segment = bar_metrics(count, alerts, peak)
                width = min(14, cell.width()*.45)
                rect = QRectF(cell.center().x()-width/2, 42-height, width, height)
                p.fillRect(rect, QColor(t['action']))
                p.fillRect(QRectF(rect.x(), rect.y(), width, segment), QColor(t['error']))
                if false:
                    p.setPen(QPen(QColor(t['warning']), 2))
                    p.drawLine(int(rect.center().x()-2), 47, int(rect.center().x()+2), 47)
                if i % 3 == 0:
                    p.setPen(QColor(t['muted']))
                    p.drawText(QRectF(cell.center().x()-30, 52, 60, 20), Qt.AlignmentFlag.AlignCenter,
                               self.hours[i].astimezone(ZoneInfo(self.zone)).strftime('%H:%M'))
            return
        for i in range(0, len(self.hours), max(1, len(self.hours)//8)):
            p.drawText(int(self.cell_rect(0, i).x()), 14, local_time(self.hours[i], self.zone)[13:18])
        peak = max((v[0] for row in self.rows.values() for v in row), default=1) or 1
        for r, (camera, cells) in enumerate(self.rows.items()):
            p.setPen(QColor(t['muted']))
            from ..formatting import camera_name
            name = camera_name(camera)
            p.drawText(0, 36+r*22, p.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, 142))
            for c, (count, alerts, false) in enumerate(cells):
                rect = self.cell_rect(r, c)
                color = QColor(t['error'] if self.fleet and alerts else t['action'])
                color.setAlpha(int(35+130*count/peak) if count else 16)
                if self.fleet:
                    center = rect.center().x()
                    rect.setLeft(center-5); rect.setRight(center+5)
                    rect.setTop(rect.bottom()-max(2, 22*count/peak))
                p.fillRect(rect, color)
                if alerts and not self.fleet:
                    p.setPen(QPen(QColor(t['error']), 1)); p.drawRect(rect)
                if false:
                    p.setPen(QPen(QColor(t['warning']), 2)); p.drawLine(rect.bottomLeft(), rect.bottomRight())
        if not self.rows:
            p.setPen(QColor(t['muted'])); p.drawText(150, 38, 'No activity in this range')

    def hit(self, position):
        for r, (camera, cells) in enumerate(self.rows.items()):
            for c, counts in enumerate(cells):
                if self.cell_rect(r, c).contains(position):
                    return camera, self.hours[c], counts

    def mouseMoveEvent(self, event):
        hit = self.hit(event.position())
        if hit:
            camera, hour, (count, alerts, false) = hit
            QToolTip.showText(event.globalPosition().toPoint(),
                activity_tooltip(hour, (count, alerts, false), self.zone) if self.fleet else
                f'{camera} · {local_time(hour, self.zone)}\n{count} events · {alerts} alerts · {false} false alarms', self)

    def mousePressEvent(self, event):
        hit = self.hit(event.position())
        if hit and not self.fleet:
            self.selected.emit(hit[0], hit[1])
