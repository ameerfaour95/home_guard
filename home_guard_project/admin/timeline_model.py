from .formatting import camera_name
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPixmap
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from .event_logic import VERDICTS, decision, ai_status, provenance, kind_label, two_lines
from .formatting import local_time, age
from .theme import PALETTES
from .widgets.icons import draw_icon

MEMBER = '      ·  '  # a clip of an open event: indented under it (no arrow glyph: not every UI font has one)
HEADERS = ['Recording', 'Time / camera', 'Kind', 'AI decision', 'Owner verdict', 'Record', 'Review']


class TimelineModel(QAbstractTableModel):
    def __init__(self, now, zone='UTC'):
        super().__init__()
        self.rows, self.images = [], {}
        self.now, self.zone = now, zone
        # Events, not only clips: the clips of one box session (one ongoing activity at one camera) collapse into
        # their newest clip's row, which reads "Pergola 09:42-17:05 · 23 clips · 1 message sent".
        self.grouping = True
        self.members = {}    # (site, session_id) -> loaded rows, newest first
        self.sessions = {}   # (site, session_id) -> EventSession from the server (all clips, not only loaded)
        self.expanded = set()

    def regroup(self):
        self.members = {}
        for i, e in enumerate(self.rows):
            if e.session_id:
                self.members.setdefault((e.site, e.session_id), []).append(i)

    def key(self, row):
        e = self.rows[row]
        return (e.site, e.session_id) if e.session_id else None

    def size(self, key):
        known = self.sessions.get(key)
        return max(len(self.members.get(key, ())), known.clips if known else 0)

    def lead(self, row):
        """The row standing for its whole event (the newest loaded clip of a session with more than one clip)."""
        key = self.key(row)
        return bool(self.grouping and key and self.members[key][0] == row and self.size(key) > 1)

    def member(self, row):
        key = self.key(row)
        return bool(self.grouping and key and self.members[key][0] != row)

    def hidden(self, row):
        return self.member(row) and self.key(row) not in self.expanded

    def session_line(self, row):
        key, e = self.key(row), self.rows[row]
        known, clips = self.sessions.get(key), [self.rows[i] for i in self.members[key]]
        first = known.first_utc if known else min(c.start_utc for c in clips)
        last = known.last_utc if known else max(c.start_utc for c in clips)
        sent = known.sent if known else sum(c.outcome_code == 'sent' for c in clips)
        sign = '−' if key in self.expanded else '+'  # open / closed (glyphs every UI font has)
        return (f'{sign} {self.size(key)} clips  ·  {camera_name(e)}  ·  '
                f'{local_time(first, e.timezone)[13:18]}–{local_time(last, e.timezone)[13:18]}'
                f'  ·  {sent} message{"" if sent == 1 else "s"} sent')

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return HEADERS[section]

    def set_page(self, items, append=False, images=None):
        if not append:
            self.beginResetModel(); self.rows = items; self.images = {}; self.expanded = set(); self.regroup()
            self.endResetModel()
        else:
            known = {e.id for e in self.rows}
            items = [e for e in items if e.id not in known]
            if items:
                start = len(self.rows)
                self.beginInsertRows(QModelIndex(), start, start+len(items)-1)
                self.rows.extend(items); self.regroup(); self.endInsertRows()
        for url, data in (images or {}).items():
            pix = QPixmap(); pix.loadFromData(data); self.images[url] = pix

    def update_review(self, event):
        for i, row in enumerate(self.rows):
            if row.id == event.id:
                self.rows[i] = event
                self.dataChanged.emit(self.index(i, 0), self.index(i, 6))

    def remove_event(self, event_id):
        for i,row in enumerate(self.rows):
            if row.id == event_id:
                self.beginRemoveRows(QModelIndex(),i,i); self.rows.pop(i); self.regroup(); self.endRemoveRows(); return

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
                                   'No video copy remains' if c.expired else 'Video retention active'])
            if col == 3:
                return e.outcome or decision(e.alert_command)  # the full outcome, also when the cell wraps it
            return (f'{camera_name(e)}\n{local_time(e.start_utc, e.timezone)}\n{e.summary}'
                    + (f'\n{e.outcome}' if e.outcome else ''))
        if role == Qt.ItemDataRole.DisplayRole:
            when = local_time(e.start_utc, e.timezone)[13:18]+'  ·  '+age(e.start_utc, self.now())
            first = (self.session_line(index.row()) if self.lead(index.row()) else
                     MEMBER+when if self.member(index.row()) else when)
            return ['', first+'\n'+camera_name(e)+'  ·  '+e.summary,
                    'Event' if self.lead(index.row()) else kind_label(e), e.outcome or decision(e.alert_command), ', '.join(VERDICTS.get(v, v.replace('_', ' ')) for v in e.owner_verdicts) or 'No feedback',
                    '', ''][col]


def status_chip_width(text):
    """The annotation chip fits its word, so the evidence icons can sit beside it."""
    return QFontMetrics(QFont('Segoe UI', 8)).horizontalAdvance(text) + 16


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
                p.setPen(QColor(t['muted'])); p.setFont(QFont('Segoe UI', 8))
                p.drawText(area, Qt.AlignmentFlag.AlignCenter, 'Preview not\navailable' if not e.thumbnail_url else 'Loading…')
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
            color = t['error'] if col == 2 and lines[0] == 'Alert' else t['secondary']
            if col == 3:  # the outcome is the point of the row: two lines, never cut to "Kept in the event, no..."
                p.setPen(QColor(color))
                shown = two_lines(p.fontMetrics(), lines[0], rect.width())
                height = p.fontMetrics().height()
                top = rect.center().y() - height*len(shown)/2
                for i, text in enumerate(shown):
                    p.drawText(QRectF(rect.x(), top+i*height, rect.width(), height), Qt.AlignmentFlag.AlignVCenter, text)
            elif col in (2, 4) and (col == 2 or e.owner_verdicts):
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
        if col == 0:
            text = (e.annotation_status or 'new').title()
            chip = QRectF(rect.x(), rect.bottom()-18, status_chip_width(text), 19)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['bubble'])); p.drawRoundedRect(chip, 3, 3)
            p.setPen(QColor(t['error'] if e.annotation_status == 'rejected' else t['action']))
            p.setFont(QFont('Segoe UI', 8)); p.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
        p.restore()
