"""Virtualised fleet rows with stable identity and explicit severity ordering."""
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor, QPainter, QFont
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from .formatting import age, mode_name, site_name, local_time
from .theme import PALETTES, verdict_color

# Server order; stable sort preserves ties exactly as received.
SEVERITY = {'offline': 0, 'critical': 1, 'warning': 2, 'unknown': 3, 'healthy': 4}
HEADERS = ['Customer / site', 'Health / top reason', 'Last seen', 'Cameras¹', 'Newest clip', '24 h activity', '7 d false', 'Mode']


class FleetModel(QAbstractTableModel):
    def __init__(self, now, theme='dark'):
        super().__init__()
        self.now, self.theme = now, theme
        self.devices, self.rows = [], []
        self.verdict, self.query = 'all', ''
        self.timezones = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return HEADERS[section]
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignRight if section in (3, 5, 6) else Qt.AlignmentFlag.AlignLeft)

    def replace(self, devices):
        self.devices = devices
        self.apply_filter()

    def apply_filter(self, verdict=None, query=None):
        if verdict is not None:
            self.verdict = verdict
        if query is not None:
            self.query = query.casefold().strip()
        self.beginResetModel()
        self.rows = sorted([d for d in self.devices if
            (self.verdict == 'all' or d.verdict == self.verdict) and
            self.query in ' '.join([d.customer_name, d.site, site_name(d.site), d.device_id,
                                    *[r.message for r in d.reasons]]).casefold()],
            key=lambda d: SEVERITY[d.verdict])
        self.endResetModel()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        d, col = self.rows[index.row()], index.column()
        if role == Qt.ItemDataRole.UserRole:
            return d
        if role == Qt.ItemDataRole.ToolTipRole:
            if col in (2, 4):
                return local_time(d.last_seen_utc if col == 2 else d.newest_clip_utc,
                                  self.timezones.get(d.customer_id, 'UTC'))
            return '\n'.join([d.customer_name, site_name(d.site), d.device_id, *[r.message for r in d.reasons]])
        if role == Qt.ItemDataRole.DisplayRole:
            values = [f'{d.customer_name}\n{site_name(d.site)}',
                      f'{d.verdict.title()}\n{d.reasons[0].message if d.reasons else "No health details reported"}',
                      age(d.last_seen_utc, self.now()),
                      f'{d.cameras_total-d.cameras_stale}/{d.cameras_total}' if d.verdict != 'offline' else f'—/{d.cameras_total}',
                      age(d.newest_clip_utc, self.now()),
                      f'{d.events_24h} events\n{d.alerts_24h} alerts',
                      str(d.false_alarms_7d), mode_name(d.mode)]
            return values[col]


class FleetDelegate(QStyledItemDelegate):
    def __init__(self, theme='dark', parent=None):
        super().__init__(parent)
        self.theme, self.tokens = theme, PALETTES[theme]

    def paint(self, painter: QPainter, option, index):
        painter.save()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.fillRect(option.rect, QColor(self.tokens['raised' if selected or hover else 'surface']))
        painter.setPen(QColor(self.tokens['border']))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        rect = option.rect.adjusted(12, 0, -8, 0)
        lines = str(index.data() or '').split('\n')
        device = index.data(Qt.ItemDataRole.UserRole)
        font = QFont('Segoe UI', 10)
        if index.column() in (0, 1):
            font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        color = verdict_color(device.verdict, self.theme) if index.column() == 1 else self.tokens['text']
        painter.setPen(QColor(color))
        if index.column() == 1:
            painter.setBrush(QColor(color))
            painter.setPen(Qt.PenStyle.NoPen)
            dot_y = int(rect.center().y()-3) if rect.width()-14 >= 420 else rect.y()+21
            painter.drawEllipse(rect.x(), dot_y, 6, 6)
            rect.adjust(14, 0, 0, 0)
            painter.setPen(QColor(color))
            if rect.width() >= 420:
                painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter, lines[0])
                painter.setFont(QFont('Segoe UI', 9))
                painter.setPen(QColor(self.tokens['muted']))
                reason_rect = rect.adjusted(82, 0, 0, 0)
                painter.drawText(reason_rect, Qt.AlignmentFlag.AlignVCenter,
                                 painter.fontMetrics().elidedText(lines[1], Qt.TextElideMode.ElideRight, reason_rect.width()))
                painter.restore()
                return
        text = painter.fontMetrics().elidedText(lines[0], Qt.TextElideMode.ElideRight, rect.width())
        alignment = Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignRight if index.column() in (3, 5, 6) else Qt.AlignmentFlag.AlignLeft)
        if len(lines) == 1:
            painter.drawText(rect, alignment, text)
        else:
            painter.drawText(rect.adjusted(0, 14, 0, -36), alignment, text)
            painter.setFont(QFont('Segoe UI', 9))
            painter.setPen(QColor(self.tokens['muted']))
            text = painter.fontMetrics().elidedText(lines[1], Qt.TextElideMode.ElideRight, rect.width())
            painter.drawText(rect.adjusted(0, 36, 0, -12), alignment, text)
        if selected and index.column() == 0:
            painter.fillRect(option.rect.x(), option.rect.y(), 3, option.rect.height(), QColor(self.tokens['action']))
        painter.restore()
