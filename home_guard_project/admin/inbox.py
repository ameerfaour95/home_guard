"""Studio · Inbox: what house owners answered from Telegram, waiting for an admin.

Every owner answer (a tag button, their own words, a voice answer) appears once with its chip "Owner · Telegram", the
clip and what the model said. An answer is a candidate, not a label (the Frigate+ pattern): Accept as tag opens
Tag · AI with the owner's answer as the starting point (the admin checks it, picks the category and saves: only that
saved tag is training data); Fix opens Tag · AI to tag it from scratch; Not a label records a complaint or a question.
Every decision is audited by the cloud.

Keys: A accept as tag · F fix · N not a label · J / K next / previous answer · R refresh.
"""
import re
from datetime import datetime, time, timedelta, timezone

from PySide6.QtCore import Qt, QDate, Signal
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QComboBox, QDateEdit, QTableWidget,
                               QTableWidgetItem, QAbstractItemView, QHeaderView, QFrame, QLineEdit, QSizePolicy)
from .backend import AuthError
from .tag_widgets import BidiElideDelegate, Pill, ProvenanceChip, LABEL_TOKENS
from .workers import TaskRunner
from .widgets.common import label, button

# One name per owner tag, everywhere on this screen: "nothing there" is the owner's tag, "no tag" is its absence.
OWNER_TAG_TITLES = {'': 'no tag', 'empty': 'nothing there', 'other': 'other', 'rule_mismatch': 'rule mismatch',
                    'normal': 'normal', 'suspicious': 'suspicious', 'escalation': 'escalation'}
OWNER_LABELS = [('', 'All owner tags'), ('normal', 'normal'), ('suspicious', 'suspicious'),
                ('escalation', 'escalation'), ('empty', 'nothing there'), ('other', 'other (own words)'),
                ('rule_mismatch', 'rule mismatch'), ('-', 'no tag (words only)')]
HANDLED = [('unhandled', 'Waiting'), ('handled', 'Handled'), ('all', 'All answers')]
DECISION_TITLES = {'accepted': 'Accepted as tag', 'fixed': 'Fixed in Tag · AI', 'not_label': 'Not a label'}
COLUMNS = ('When', 'House', 'Camera', 'Source', 'Owner tag', 'Owner said', 'Model said', 'Status')


def camera_title(item):
    """The owner's name for the camera, else "Camera 6" from a ..._ch6 id; never a bare id when a name exists."""
    if item.camera_name:
        return item.camera_name
    m = re.search(r'ch(\d+)$', item.camera or '')
    return f'Camera {m.group(1)}' if m else (item.camera or '').replace('_', ' ')


def owner_tag(item):
    return OWNER_TAG_TITLES.get(item.owner_label, item.owner_label)


def owner_words(item):
    """What the owner said in words: their text, else a voice answer's transcript, else a typed message."""
    return item.owner_text or item.transcript or item.raw_text or ''


def owner_prefill(item):
    """The Tag · AI form fields an owner's answer starts a tag with (a draft: the admin checks it and picks the
    category). The owner's words go to the notes, which never train."""
    label = item.owner_label
    fields = {'normal': {'raw_label': 'normal'}, 'rule_mismatch': {'raw_label': 'normal'},
              'suspicious': {'raw_label': 'suspicious'}, 'escalation': {'raw_label': 'escalation'},
              'empty': {'category': 'N10', 'raw_label': 'normal'},
              'other': {'category': 'other', 'other_text': owner_words(item)[:200]}}.get(label, {})
    words = owner_words(item)
    said = f'Owner · Telegram{" (" + label + ")" if label else ""}: {words}' if words else f'Owner · Telegram: {label}'
    return dict(fields, notes=said)


class InboxScreen(QWidget):
    session_expired = Signal()
    tag_requested = Signal(str, object)   # a clip key, and the owner's answer as Tag · AI fields (None: from scratch)

    def __init__(self, backend, role='admin', theme='dark'):
        super().__init__()
        self.backend, self.role, self.theme = backend, role, theme
        self.items, self.customers, self.cameras = [], {}, {}
        self.loader, self.writer = TaskRunner(self), TaskRunner(self)
        self.loader.finished.connect(self.loaded); self.writer.finished.connect(self.decided)
        self.pending = None
        root = QVBoxLayout(self); root.setContentsMargins(24, 16, 24, 16); root.setSpacing(12)
        head = QHBoxLayout(); titles = QVBoxLayout(); titles.setSpacing(2)
        titles.addWidget(label('STUDIO', 'eyebrow')); titles.addWidget(label('Inbox', 'clipTitle'))
        titles.addWidget(label('What owners answered from Telegram. Candidates until you accept them: only a tag you '
                               'save in Tag · AI trains the model.', 'muted', True))
        head.addLayout(titles, 1)
        self.count = label('', 'statNumber'); self.count.setAlignment(Qt.AlignmentFlag.AlignRight); head.addWidget(self.count)
        head.addWidget(button('Refresh  R', self.load, 'compact'))
        root.addLayout(head)
        filters = QHBoxLayout(); filters.setSpacing(8)
        today = QDate.currentDate()
        self.date_from, self.date_to = QDateEdit(today.addDays(-14)), QDateEdit(today)
        for edit, name in ((self.date_from, 'From day'), (self.date_to, 'To day')):
            edit.setCalendarPopup(True); edit.setDisplayFormat('yyyy-MM-dd'); edit.setAccessibleName(name)
            edit.dateChanged.connect(lambda *_: self.load())
        filters.addWidget(label('From', 'muted')); filters.addWidget(self.date_from)
        filters.addWidget(label('to', 'muted')); filters.addWidget(self.date_to)
        self.customer, self.camera, self.owner_label, self.handled = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        self.customer.addItem('All houses', None); self.camera.addItem('All cameras', '')
        for value, text in OWNER_LABELS: self.owner_label.addItem(text, value)
        for value, text in HANDLED: self.handled.addItem(text, value)
        for combo, name in ((self.customer, 'House'), (self.camera, 'Camera'), (self.owner_label, 'Owner tag'),
                            (self.handled, 'Handled')):
            combo.setAccessibleName(name); combo.activated.connect(lambda *_: self.load()); filters.addWidget(combo)
        filters.addStretch(); root.addLayout(filters)
        self.error = label('', 'error', True); self.error.hide(); root.addWidget(self.error)
        split = QSplitter(Qt.Orientation.Horizontal); split.setHandleWidth(12); split.setChildrenCollapsible(False)
        split.setStyleSheet('QSplitter::handle { background: transparent; }')
        root.addWidget(split, 1)
        self.table = QTableWidget(0, len(COLUMNS)); self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide(); self.table.setWordWrap(False); self.table.setAccessibleName('Owner answers')
        self.table.setItemDelegate(BidiElideDelegate(self.table))   # Hebrew cells elide at their own end
        # the one-letter keys belong to the screen: a focused table would take them for its type-to-search
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus); self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        header = self.table.horizontalHeader()
        for c in range(len(COLUMNS)):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c in (5, 6) else QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(3, ProvenanceChip(self.theme, 'owner').sizeHint().width() + 16)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed); header.resizeSection(7, 140)
        header.setMinimumSectionSize(60)
        self.table.itemSelectionChanged.connect(self.show_selected)
        split.addWidget(self.table)
        split.addWidget(self._detail_panel())
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 1); split.setSizes([900, 360])
        for key, slot in {'A': self.accept, 'F': self.fix, 'N': self.not_label, 'J': lambda: self.move(1),
                          'K': lambda: self.move(-1), 'R': self.load}.items():
            sc = QShortcut(QKeySequence(key), self); sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        self.show_selected()

    def _detail_panel(self):
        panel = QFrame(); panel.setObjectName('panel'); panel.setMinimumWidth(340)
        col = QVBoxLayout(panel); col.setContentsMargins(16, 14, 16, 14); col.setSpacing(8)
        chips = QHBoxLayout(); chips.setSpacing(6)
        self.source_chip = ProvenanceChip(self.theme, 'owner'); chips.addWidget(self.source_chip)
        self.decision_chip = ProvenanceChip(self.theme); chips.addWidget(self.decision_chip)
        self.decision_text = label('', 'muted'); chips.addWidget(self.decision_text); chips.addStretch()
        col.addLayout(chips)
        self.clip = label('Select an answer', 'section', True); col.addWidget(self.clip)
        self.clip_meta = label('', 'muted', True); self.clip_meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        col.addWidget(self.clip_meta)
        owner = QHBoxLayout(); owner.addWidget(label('OWNER SAID', 'eyebrow'))
        self.owner_pill = Pill(self.theme); owner.addWidget(self.owner_pill); owner.addStretch(); col.addLayout(owner)
        self.owner_said = label('', '', True); self.owner_said.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.owner_said.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        col.addWidget(self.owner_said)
        self.owner_how = label('', 'muted', True); col.addWidget(self.owner_how)
        model = QHBoxLayout(); model.addWidget(label('MODEL SAID', 'eyebrow'))
        self.model_pill = Pill(self.theme); model.addWidget(self.model_pill)
        self.model_chip = ProvenanceChip(self.theme); model.addWidget(self.model_chip); model.addStretch()
        col.addLayout(model)
        self.model_said = label('', '', True); col.addWidget(self.model_said)
        self.consent = label('', 'error', True); self.consent.hide(); col.addWidget(self.consent)
        col.addStretch()
        self.note = QLineEdit(); self.note.setPlaceholderText('Note (why it is not a label, what you fixed)')
        col.addWidget(self.note)
        row = QHBoxLayout(); row.setSpacing(8)
        self.accept_button = button('Accept as tag  A', self.accept, 'primary')
        self.accept_button.setToolTip("Open Tag · AI with the owner's answer filled in as a draft: check it, pick the "
                                      'category and save (A)')
        self.fix_button = button('Fix  F', self.fix, 'compact')
        self.fix_button.setToolTip('The owner is not right or not precise: open Tag · AI and tag it yourself (F)')
        self.not_label_button = button('Not a label  N', self.not_label, 'compact')
        self.not_label_button.setToolTip('A complaint or a question, not about what the clip shows (N)')
        self.reopen_button = button('Back to waiting', self.reopen, 'link')
        for b in (self.accept_button, self.fix_button, self.not_label_button):
            row.addWidget(b)
        row.addStretch(); row.addWidget(self.reopen_button)
        col.addLayout(row)
        return panel

    # ------------------------------------------------------------------ loading
    def filters(self):
        start = datetime.combine(self.date_from.date().toPython(), time(), timezone.utc)
        end = datetime.combine(self.date_to.date().toPython(), time(), timezone.utc) + timedelta(days=1)
        owner = self.owner_label.currentData()
        return dict(from_utc=start.isoformat(), to_utc=end.isoformat(), customer_id=self.customer.currentData(),
                    camera=self.camera.currentData() or None,
                    owner_label='' if owner == '-' else (owner or None), handled=self.handled.currentData())

    def open(self):
        self.load()

    def load(self):
        request = self.filters()
        if not self.loader.start(lambda: self.backend.inbox(**request)):
            self.pending = True

    def loaded(self, items, error):
        if self.pending:
            self.pending = None; self.load(); return
        if self.handle_error(error):
            return
        self.error.hide()
        selected = self.current().feedback_id if self.current() else None
        self.items = items
        for i in items:
            self.customers.setdefault(i.customer_id, i.customer)
            self.cameras.setdefault(i.camera, camera_title(i))
        self.fill_combo(self.customer, sorted(self.customers.items(), key=lambda kv: kv[1]), 'All houses', None)
        self.fill_combo(self.camera, sorted(self.cameras.items(), key=lambda kv: kv[1]), 'All cameras', '')
        self.count.setText(str(len(items)))
        self.count.setToolTip(f'{len(items)} answer(s) in this view')
        self.table.setRowCount(len(items))
        for r, i in enumerate(items):
            when = i.received_utc.astimezone().strftime('%d.%m %H:%M') if i.received_utc else '—'
            status = DECISION_TITLES.get(i.decision) or ('Waiting · not a label?' if i.probably_not_label
                                                         else 'Waiting')
            values = (when, i.customer, camera_title(i), '', owner_tag(i), owner_words(i) or '—',
                      i.model_label or '—', status)
            for c, value in enumerate(values):
                cell = QTableWidgetItem(value); cell.setToolTip(value)
                self.table.setItem(r, c, cell)
            self.table.setCellWidget(r, 3, ProvenanceChip(self.theme, 'owner'))
        row = next((r for r, i in enumerate(items) if i.feedback_id == selected), 0 if items else -1)
        if row >= 0:
            self.table.selectRow(row)
        self.show_selected()

    def fill_combo(self, combo, entries, all_text, all_value):
        current = combo.currentData()
        combo.blockSignals(True); combo.clear(); combo.addItem(all_text, all_value)
        for value, text in entries:
            combo.addItem(text, value)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0); combo.blockSignals(False)

    def handle_error(self, error):
        if not error:
            return False
        if isinstance(error, AuthError):
            self.session_expired.emit()
        else:
            self.error.setText(str(error) or 'The inbox could not load'); self.error.show()
        return True

    # ------------------------------------------------------------------ the selected answer
    def current(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.items[rows[0].row()] if rows and rows[0].row() < len(self.items) else None

    def show_selected(self):
        i = self.current()
        for b in (self.accept_button, self.fix_button, self.not_label_button):
            b.setEnabled(i is not None and not self.writer.busy)
        self.reopen_button.setVisible(bool(i and i.decision))
        if i is None:
            self.clip.setText('No answer selected' if self.items else 'Nothing waiting: every owner answer is handled')
            for w in (self.clip_meta, self.owner_said, self.owner_how, self.model_said, self.decision_text):
                w.setText('')
            self.owner_pill.hide(); self.model_pill.hide(); self.model_chip.show_source(''); self.decision_chip.show_source('')
            self.consent.hide()
            return
        when = i.received_utc.astimezone().strftime('%Y-%m-%d %H:%M') if i.received_utc else ''
        self.clip.setText(f'{i.customer}  ·  {camera_title(i)}')
        self.clip_meta.setText(f'Answered {when} by {i.tagged_by or "the owner"}  ·  clip {i.clip_key}')
        self.owner_pill.show_label(owner_tag(i), LABEL_TOKENS.get(i.owner_label, 'muted'))
        words = owner_words(i)
        self.owner_said.setText(words or ('(tag button, no words)' if i.owner_label else '(no words)'))
        how = []
        if i.transcript and words == i.transcript:
            how.append('voice answer, transcribed')
        if i.verdict and i.verdict != 'none':
            how.append('verdict: ' + i.verdict.replace('_', ' '))
        if i.probably_not_label and not i.decision:
            how.append("probably not a label: a question, a complaint or a command (the box's own rule)")
        self.owner_how.setText('  ·  '.join(how))
        self.model_pill.show_label(i.model_label or 'no label', LABEL_TOKENS.get(i.model_label or '', 'action'))
        self.model_chip.show_source('model', i.model)
        self.model_chip.setToolTip(f'Prompt version: {i.prompt_version}' if i.prompt_version else '')
        self.model_said.setText(i.model_summary or '(no summary)')
        self.decision_chip.show_source('admin' if i.decision else '', i.decided_by)
        self.decision_text.setText(DECISION_TITLES.get(i.decision, '') + (f': {i.decision_note}' if i.decision_note else ''))
        self.consent.setVisible(not i.consent_training)
        self.consent.setText(f'{i.customer} withdrew consent to training use: this answer cannot become a training '
                             'label.')
        self.accept_button.setEnabled(i.consent_training and not self.writer.busy)

    def move(self, delta):
        if self.items:
            row = self.table.currentRow()
            self.table.selectRow(max(0, min(len(self.items) - 1, (row if row >= 0 else -delta) + delta)))

    # ------------------------------------------------------------------ decisions
    def decide(self, decision, then=None):
        i = self.current()
        if i is None or self.writer.busy:
            return
        note = self.note.text().strip()
        self.writer.start(lambda: (decision, then, self.backend.inbox_decide(i.feedback_id, decision, note)))
        self.show_selected()

    def accept(self):
        i = self.current()
        if i is not None and i.consent_training:
            self.decide('accepted', (i.clip_key, owner_prefill(i)))

    def fix(self):
        i = self.current()
        if i is not None:
            self.decide('fixed', (i.clip_key, None))

    def not_label(self):
        self.decide('not_label')

    def reopen(self):
        i = self.current()
        if i is not None and not self.writer.busy:
            self.writer.start(lambda: (None, None, self.backend.inbox_reopen(i.feedback_id)))

    def decided(self, result, error):
        if self.handle_error(error):
            self.show_selected(); return
        _, then, _ = result
        self.note.clear()
        if then:
            self.tag_requested.emit(*then)
        self.load()
