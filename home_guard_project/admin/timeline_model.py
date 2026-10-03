from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QPixmap
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from .event_logic import KINDS, VERDICTS, decision, ai_status, provenance
from .formatting import local_time, age
from .theme import PALETTES
from .widgets.icons import draw_icon

HEADERS = ['Recording', 'Time / camera', 'Kind', 'AI decision', 'Owner verdict', 'Record', 'Review']


class TimelineModel(QAbstractTableModel):
    def __init__(self, now, zone='UTC'):
        super().__init__()
        self.rows, self.images = [], {}
        self.now, self.zone = now, zone

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return HEADERS[section]

    def set_page(self, items, append=False, images=None):
        if not append:
            self.beginResetModel(); self.rows = items; self.images = {}; self.endResetModel()
        else:
            known = {e.id for e in self.rows}
            items = [e for e in items if e.id not in known]
            if items:
                start = len(self.rows)
                self.beginInsertRows(QModelIndex(), start, start+len(items)-1)
                self.rows.extend(items); self.endInsertRows()
        for url, data in (images or {}).items():
            pix = QPixmap(); pix.loadFromData(data); self.images[url] = pix

    def update_review(self, event):
        for i, row in enumerate(self.rows):
            if row.id == event.id:
                self.rows[i] = event
                self.dataChanged.emit(self.index(i, 0), self.index(i, 6))

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        e, col = self.rows[index.row()], index.column()
        if role == Qt.ItemDataRole.UserRole:
            return e
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == 6:
                return ('Reviewed' if e.reviewed else 'Unreviewed')+' · '+('Flagged' if e.flagged else 'Not flagged')
            if col == 5:
                c = e.completeness
                return '\n'.join(['Video saved' if c.video else 'No video saved', provenance(c.boxes), ai_status(c.ai),
                                   'Expired' if c.expired else 'Not expired'])
            return f'{local_time(e.start_utc, self.zone)}\n{e.summary}'
        if role == Qt.ItemDataRole.DisplayRole:
            return ['', local_time(e.start_utc, self.zone)[13:18]+'  ·  '+age(e.start_utc, self.now())+'\n'+e.camera+'  ·  '+e.summary,
                    KINDS[e.kind], decision(e.alert_command), ', '.join(VERDICTS.get(v, v.replace('_', ' ')) for v in e.owner_verdicts) or 'No feedback',
                    '', ''][col]


class TimelineDelegate(QStyledItemDelegate):
    def __init__(self, theme='dark', parent=None):
        super().__init__(parent)
        self.t = PALETTES[theme]

    def paint(self, p: QPainter, option, index):
        p.save()
        t, e = self.t, index.data(Qt.ItemDataRole.UserRole)
        rect = option.rect.adjusted(10, 8, -10, -8)
        selected = option.state & QStyle.StateFlag.State_Selected
        p.fillRect(option.rect, QColor(t['raised' if selected else 'surface']))
        p.setPen(QColor(t['border'])); p.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = index.column()
        if col == 0:
            pix = index.model().images.get(e.thumbnail_url)
            area = QRectF(rect.x(), rect.y()+2, 88, 50)
            p.fillRect(area, QColor(t['raised']))
            if pix and not pix.isNull():
                p.drawPixmap(area.toRect(), pix)
            else:
                draw_icon(p, 'video', QRectF(area.center().x()-10, area.center().y()-10, 20, 20), t['muted'])
        elif col in (5, 6):
            if col == 5:
                c = e.completeness
                badges = [('video', 'action' if c.video else 'border'), ('boxes', 'action' if c.boxes != 'none' else 'border'),
                          ('ai', 'action' if c.ai == 'real' else 'warning' if c.ai in ('failed', 'fallback') else 'border')]
                if c.expired:
                    badges.append(('expired', 'error'))
            else:
                badges = [('reviewed', 'action' if e.reviewed else 'border'), ('flagged', 'warning' if e.flagged else 'border')]
            for i, (name, token) in enumerate(badges):
                draw_icon(p, name, QRectF(rect.x()+i*24, rect.center().y()-9, 18, 18), t[token])
        else:
            lines = str(index.data()).split('\n')
            p.setFont(QFont('Segoe UI', 9))
            color = t['error'] if col == 2 and e.kind == 'alert' else t['secondary']
            if col in (2, 4) and (col == 2 or e.owner_verdicts):
                text = p.fontMetrics().elidedText(lines[0], Qt.TextElideMode.ElideRight, rect.width()-14)
                chip = QRectF(rect.x(), rect.center().y()-13, min(rect.width(), p.fontMetrics().horizontalAdvance(text)+14), 26)
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['raised'])); p.drawRoundedRect(chip, 4, 4)
                p.setPen(QColor(color)); p.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
            else:
                p.setPen(QColor(t['text'] if col == 1 else color))
                text = p.fontMetrics().elidedText(lines[0], Qt.TextElideMode.ElideRight, rect.width())
                p.drawText(rect.adjusted(0, 0, 0, -24) if len(lines) > 1 else rect, Qt.AlignmentFlag.AlignVCenter, text)
                if len(lines) > 1:
                    p.setPen(QColor(t['muted']))
                    p.drawText(rect.adjusted(0, 26, 0, 0), Qt.AlignmentFlag.AlignVCenter,
                               p.fontMetrics().elidedText(lines[1], Qt.TextElideMode.ElideRight, rect.width()))
        if selected and col == 0:
            p.fillRect(option.rect.x(), option.rect.y(), 2, option.rect.height(), QColor(t['action']))
        p.restore()
