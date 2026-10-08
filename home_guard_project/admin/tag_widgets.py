"""Small widgets of the tagging workspace: queue rows, category buttons, chips, opinion cards, a flow layout."""
from PySide6.QtCore import Qt, QRect, QRectF, QSize, QPoint, QAbstractListModel, QModelIndex, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (QAbstractButton, QStyledItemDelegate, QStyle, QLayout, QWidget, QFrame,
                               QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel)
from .theme import PALETTES
from .widgets.common import label

TIER_TOKENS = {'contradiction': 'error', 'check': 'warning', 'untagged': 'action', 'done': 'ok'}
TIER_TITLES = {'contradiction': 'Contradiction', 'check': 'To check', 'untagged': 'Untagged', 'done': 'Done'}
LABEL_TOKENS = {'normal': 'ok', 'empty': 'muted', 'suspicious': 'warning', 'escalation': 'error', 'alert': 'error'}
GROUP_TOKENS = {'N': 'ok', 'S': 'warning', 'E': 'error', '': 'action'}
WHO_TITLES = {'old': 'Old tag', 'owner': "Customer's answer", 'ai': 'AI label', 'teacher': 'Teacher suggestion',
              'studio': 'Our tag'}
ORIGIN_TITLES = {'dataset': 'Old tags', 'customer': 'Customer', 'owner_feedback': 'Customer (local)'}


def soft(color, alpha=46):
    c = QColor(color); c.setAlpha(alpha); return c


def short_label(entry):
    """What a queue row shows for one party: the category, else the label word."""
    if entry.get('category'):
        return entry['category']
    if entry.get('disputes_ai'):
        return 'AI wrong'
    return {'suspicious': 'susp', 'escalation': 'esc'}.get(entry.get('label', ''), entry.get('label', '')) or '—'


class QueueModel(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.rows = []

    def set_rows(self, rows):
        self.beginResetModel(); self.rows = list(rows); self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return row
        if role == Qt.ItemDataRole.DisplayRole:
            return row['clip_id']
        if role == Qt.ItemDataRole.ToolTipRole:
            return '\n'.join([row['clip_id'], *row['reasons']])
        return None

    def row_of(self, key):
        return next((i for i, r in enumerate(self.rows) if r['key'] == key), -1)


class QueueDelegate(QStyledItemDelegate):
    HEIGHT = 70

    def __init__(self, theme='dark', parent=None):
        super().__init__(parent)
        self.t = PALETTES[theme]

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), self.HEIGHT)

    def paint(self, p, option, index):
        row = index.data(Qt.ItemDataRole.UserRole)
        if not row:
            return
        t = self.t
        r = option.rect
        p.save(); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = option.state & QStyle.StateFlag.State_Selected
        p.fillRect(r, QColor(t['raised'] if selected else t['surface']))
        p.setPen(QPen(QColor(t['border']), 1)); p.drawLine(r.bottomLeft(), r.bottomRight())
        tier = QColor(t[TIER_TOKENS.get(row['tier_name'], 'muted')])
        p.fillRect(QRect(r.x(), r.y(), 3, r.height()), tier)
        x, w = r.x() + 14, r.width() - 24
        font = QFont(option.font); font.setPixelSize(13); font.setWeight(QFont.Weight.DemiBold); p.setFont(font)
        origin = ORIGIN_TITLES.get(row['origin'], row['origin'])
        small = QFont(option.font); small.setPixelSize(11)
        ow = QFontMetrics(small).horizontalAdvance(origin)
        tx = x
        if row.get('alert'):
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['error'])); p.drawEllipse(QRectF(x, r.y() + 13, 7, 7))
            tx = x + 12
        p.setPen(QColor(t['text']))
        p.drawText(QRect(tx, r.y() + 8, w - ow - 10 - (tx - x), 18), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetrics(font).elidedText(row['clip_id'], Qt.TextElideMode.ElideMiddle, w - ow - 10 - (tx - x)))
        p.setFont(small); p.setPen(QColor(t['muted']))
        p.drawText(QRect(x + w - ow, r.y() + 8, ow, 18), Qt.AlignmentFlag.AlignVCenter, origin)
        if not row.get('has_media', True):
            p.drawText(QRect(x + w - 60, r.y() + 47, 60, 16), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       'no video')
        p.setPen(QColor(t['secondary']))
        reason = '; '.join(row['reasons'])
        p.drawText(QRect(x, r.y() + 27, w, 16), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetrics(small).elidedText(reason, Qt.TextElideMode.ElideRight, w))
        cx = x
        for who in ('studio', 'old', 'owner', 'ai', 'teacher'):
            entry = row['labels'].get(who)
            if not entry:
                continue
            text = f"{WHO_TITLES[who].split()[0][0] if who != 'studio' else '✓'} {short_label(entry)}"
            tw = QFontMetrics(small).horizontalAdvance(text) + 12
            if cx + tw > x + w - (64 if not row.get('has_media', True) else 0):
                break
            color = QColor(t[LABEL_TOKENS.get(entry.get('label', ''), 'muted')])
            chip = QRectF(cx, r.y() + 47, tw, 16)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(soft(color, 60)); p.drawRoundedRect(chip, 4, 4)
            p.setPen(color); p.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
            cx += tw + 5
        p.restore()


class CategoryButton(QAbstractButton):
    """One taxonomy category: id and English name (two lines when narrow), the Hebrew name under it (right to left)."""
    ONE_LINE, TWO_LINES = 46, 62

    def __init__(self, category, theme='dark', parent=None):
        super().__init__(parent)
        self.category, self.t = category, PALETTES[theme]
        self.hint = False                      # the teacher suggests this category
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{category['id']}  {category['name']}\n{category['definition']}\n{category['he']}")
        self.setAccessibleName(f"{category['id']} {category['name']}")
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def _fonts(self):
        bold = QFont(self.font()); bold.setPixelSize(12); bold.setWeight(QFont.Weight.DemiBold)
        normal = QFont(self.font()); normal.setPixelSize(12)
        return bold, normal

    def _wraps(self, width):
        bold, normal = self._fonts()
        room = width - 22 - QFontMetrics(bold).horizontalAdvance(self.category['id']) - 6
        return QFontMetrics(normal).horizontalAdvance(self.category['name']) > room

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.TWO_LINES if self._wraps(width) else self.ONE_LINE

    def sizeHint(self):
        return QSize(120, self.heightForWidth(self.width() if self.width() > 0 else 120))

    def minimumSizeHint(self):
        return QSize(64, self.ONE_LINE)

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self.t
        accent = QColor(t[GROUP_TOKENS.get(self.category.get('group', ''), 'action')])
        rect = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        if self.isChecked():
            p.setBrush(soft(accent, 70)); p.setPen(QPen(accent, 1.5))
        else:
            p.setBrush(QColor(t['raised'] if self.underMouse() else t['surface']))
            p.setPen(QPen(QColor(t['action'] if self.hasFocus() else t['border']), 1))
        p.drawRoundedRect(rect, 6, 6)
        if self.hint and not self.isChecked():
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['action']))
            p.drawEllipse(QRectF(rect.right() - 10, rect.top() + 6, 5, 5))
        bold, normal = self._fonts()
        idw = QFontMetrics(bold).horizontalAdvance(self.category['id']) + 6
        p.setFont(bold); p.setPen(accent if self.category['id'] != 'other' else QColor(t['text']))
        p.drawText(QRect(9, 5, idw, 18), Qt.AlignmentFlag.AlignVCenter, self.category['id'])
        p.setFont(normal); p.setPen(QColor(t['text']))
        wraps = self._wraps(self.width())
        name_rect = QRect(9 + idw, 5, self.width() - 22 - idw, 34 if wraps else 18)
        if wraps:      # two lines; a name still too long for them is cut at the end of the second
            fm = QFontMetrics(normal)
            words, first = self.category['name'].split(), ''
            while words and fm.horizontalAdvance((first + ' ' + words[0]).strip()) <= name_rect.width():
                first = (first + ' ' + words.pop(0)).strip()
            second = fm.elidedText(' '.join(words), Qt.TextElideMode.ElideRight, name_rect.width())
            p.drawText(QRect(name_rect.x(), 5, name_rect.width(), 17), Qt.AlignmentFlag.AlignVCenter, first)
            p.drawText(QRect(name_rect.x(), 22, name_rect.width(), 17), Qt.AlignmentFlag.AlignVCenter, second)
        else:
            p.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter, self.category['name'])
        he = QFont(self.font()); he.setPixelSize(11); p.setFont(he); p.setPen(QColor(t['muted']))
        p.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        bottom = QRect(9, self.height() - 22, self.width() - 18, 16)
        p.drawText(bottom, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   QFontMetrics(he).elidedText(self.category['he'], Qt.TextElideMode.ElideRight, bottom.width()))

    def enterEvent(self, event):
        self.update(); super().enterEvent(event)

    def leaveEvent(self, event):
        self.update(); super().leaveEvent(event)


class FlowLayout(QLayout):
    """Wraps its items like words in a line."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self.items, self._spacing = [], spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, i):
        return self.items[i] if 0 <= i < len(self.items) else None

    def takeAt(self, i):
        return self.items.pop(i) if 0 <= i < len(self.items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect); self._place(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _place(self, rect, measure):
        x, y, line = rect.x(), rect.y(), 0
        for item in self.items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and line > 0:
                x, y, line = rect.x(), y + line + self._spacing, 0
            if not measure:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line = max(line, hint.height())
        return y + line - rect.y()


class ChipGroup(QWidget):
    """Checkable chips for one field: one choice (click again to clear), or several with `multi`."""
    changed = Signal()

    def __init__(self, values, multi=False, titles=None):
        super().__init__()
        self.multi, self.buttons = multi, {}
        flow = FlowLayout(self)
        for value in values:
            chip = QPushButton((titles or {}).get(value, value.replace('_', ' ')))
            chip.setObjectName('chip'); chip.setCheckable(True); chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda checked, v=value: self._clicked(v, checked))
            flow.addWidget(chip); self.buttons[value] = chip
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def _clicked(self, value, checked):
        if not self.multi:
            for v, b in self.buttons.items():
                if v != value:
                    b.setChecked(False)
        self.changed.emit()

    def value(self):
        chosen = [v for v, b in self.buttons.items() if b.isChecked()]
        return chosen if self.multi else (chosen[0] if chosen else '')

    def set_value(self, value):
        wanted = set(value or []) if self.multi else {value}
        for v, b in self.buttons.items():
            b.setChecked(v in wanted)


class Pill(QLabel):
    def __init__(self, theme='dark'):
        super().__init__()
        self.t = PALETTES[theme]
        self.setTextFormat(Qt.TextFormat.PlainText)

    def show_label(self, text, token):
        color = self.t.get(token, self.t['muted'])
        self.setText(text)
        self.setStyleSheet(f'color: {color}; background: {soft(color, 60).name(QColor.NameFormat.HexArgb)};'
                           ' border-radius: 4px; padding: 2px 8px; font-size: 8pt; font-weight: 600;')
        self.setVisible(bool(text))


# Where a label came from (design principle 4): one chip vocabulary on every screen.
PROVENANCE = {'owner': ('Owner · Telegram', 'warning'), 'admin': ('Admin · {who}', 'ok'),
              'model': ('Model · {who}', 'action'), 'yolo': ('YOLO weak', 'muted'), 'tracker': ('Tracker', 'muted'),
              'dataset': ('Dataset labels', 'muted')}
# The short caption of a machine box nobody checked yet, by what it was preloaded from (AnnotationOut.preload_source).
MACHINE_NAMES = {'tracker': 'Tracker', 'dataset': 'Dataset', 'yolo': 'YOLO'}


def machine_name(source):
    return MACHINE_NAMES.get(source or 'yolo', 'YOLO')


def provenance_text(kind, who=''):
    """The chip text of a label's source: 'Owner · Telegram', 'Admin · Dana', 'Model · gpt-4o', 'YOLO weak',
    'Tracker', 'Dataset labels'; '' for an unknown kind."""
    text, _ = PROVENANCE.get(kind, ('', ''))
    return text.format(who=who or 'unknown') if text else ''


class ProvenanceChip(Pill):
    """A small chip naming who made a label (see PROVENANCE)."""

    def __init__(self, theme='dark', kind='', who=''):
        super().__init__(theme)
        self.kind = ''
        self.setAccessibleName('Label source')
        self.show_source(kind, who)

    def show_source(self, kind, who=''):
        self.kind = kind if kind in PROVENANCE else ''
        self.show_label(provenance_text(kind, who), PROVENANCE.get(kind, ('', 'muted'))[1])


class OpinionCard(QFrame):
    """What one party said: label, category, words, and where it came from."""

    def __init__(self, who, theme='dark'):
        super().__init__()
        self.who, self.t = who, PALETTES[theme]
        self.setObjectName('opinion')
        layout = QVBoxLayout(self); layout.setContentsMargins(12, 10, 12, 10); layout.setSpacing(4)
        head = QHBoxLayout(); head.setSpacing(6)
        head.addWidget(label(WHO_TITLES[who].upper(), 'eyebrowMuted'))
        self.source = ProvenanceChip(theme); head.addWidget(self.source)
        head.addStretch()
        self.label_pill, self.category_pill = Pill(theme), Pill(theme)
        head.addWidget(self.category_pill); head.addWidget(self.label_pill)
        layout.addLayout(head)
        self.text = label('', '', True); self.text.setObjectName('opinionText')
        self.text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.text, 1)
        self.detail = label('', 'muted', True); self.detail.setObjectName('opinionDetail'); layout.addWidget(self.detail)
        self.setMinimumHeight(96); self.setMaximumHeight(150)

    def show_opinion(self, op, conflict=False, categories=None):
        self.setProperty('conflict', 'yes' if conflict else 'no'); self.style().unpolish(self); self.style().polish(self)
        if not op:
            self.label_pill.hide(); self.category_pill.hide(); self.source.show_source('')
            self.text.setText({'old': 'No old tag for this clip.', 'owner': 'The customer has not answered.',
                               'ai': 'No AI answer recorded.', 'teacher': 'No teacher answer yet.'}[self.who])
            self.text.setProperty('empty', 'yes'); self.detail.setText('')
        else:
            self.text.setProperty('empty', 'no')
            effective = op.get('effective_label') or ''
            self.label_pill.show_label('says the AI was wrong' if op.get('disputes_ai') and not effective else effective,
                                       LABEL_TOKENS.get(effective, 'action'))
            cat = op.get('category') or ''
            name = (categories or {}).get(cat, '')
            self.category_pill.show_label(f'{cat} {name}'.strip(), 'action')
            self.text.setText(op.get('text') or '(no words)')
            d = op.get('detail') or {}
            self.source.show_source(*{'owner': ('owner',), 'ai': ('model', d.get('model')),
                                      'teacher': ('model', d.get('model'))}.get(self.who, ('',)))
            bits = {'owner': [d.get('verdict', '').replace('_', ' '), d.get('owner_label') and 'tag: ' + d['owner_label'],
                              d.get('from'), (op.get('at') or '')[:16].replace('T', ' ')],
                    'ai': [d.get('model'), d.get('prompt_version'), d.get('final_label') and d['final_label'] != effective
                           and 'acted on ' + d['final_label'], d.get('ai_status') not in (None, 'real') and f"AI call: {d['ai_status']}"],
                    'teacher': [d.get('model'), d.get('rank') and f"rank {d['rank']}", d.get('zone'), d.get('movement')],
                    'old': [d.get('by'), d.get('batch'), d.get('delete') and 'marked [delete]']}.get(self.who, [])
            self.detail.setText('  ·  '.join(str(b) for b in bits if b))
        self.text.style().unpolish(self.text); self.text.style().polish(self.text)
