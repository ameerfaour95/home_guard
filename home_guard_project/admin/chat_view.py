"""The customer page's Chat tab: the owner's Telegram conversation with the box, read-only.

Loaded only when staff open the tab (the cloud audits every view and tells the owner), one day at a time or as a
search across the uploaded days. Owner messages sit on the right, the box and the assistant on the left; an alert
is a card with its picture, a button press a short chip, and a message Telegram refused says so in red.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QLineEdit, QScrollArea,
    QLabel, QSizePolicy)
from .backend import AuthError
from .formatting import local_time
from .workers import TaskRunner
from .widgets.common import label, button

WHO = {'box': 'Home Guard', 'assistant': 'Assistant', 'owner': 'Owner'}


class ChatView(QWidget):
    session_expired = Signal()

    def __init__(self, backend, theme='dark'):
        super().__init__()
        self.backend, self.theme = backend, theme
        self.customer_id, self.zone, self.loaded_for = None, 'UTC', None
        self.result, self.bubbles = None, []
        self.runner = TaskRunner(self); self.runner.finished.connect(self.loaded)
        self.image_runner = TaskRunner(self); self.image_runner.finished.connect(self.image_loaded)
        self.pending_images, self.image_labels, self.current_image = [], {}, None
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 12, 0, 0); layout.setSpacing(8)
        bar = QHBoxLayout(); bar.setSpacing(8)
        bar.addWidget(label('Day', 'muted'))
        self.day = QComboBox(); self.day.setAccessibleName('Chat day'); self.day.setMinimumWidth(150)
        self.day.activated.connect(lambda _: self.load(day=self.day.currentData()))
        bar.addWidget(self.day)
        self.search = QLineEdit(); self.search.setPlaceholderText('Search the last 14 days…  Enter')
        self.search.returnPressed.connect(lambda: self.load(q=self.search.text().strip() or None))
        bar.addWidget(self.search, 1)
        bar.addWidget(button('Search', lambda: self.load(q=self.search.text().strip() or None)))
        self.clear_search = button('Back to the day', lambda: (self.search.clear(), self.load(day=self.day.currentData())),
                                   'link')
        self.clear_search.hide(); bar.addWidget(self.clear_search)
        layout.addLayout(bar)
        self.note = label('Read-only. Opening the chat is recorded in the audit log and the owner is told.', 'muted', True)
        layout.addWidget(self.note)
        self.status = label('', 'muted', True); layout.addWidget(self.status)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setMinimumHeight(360)
        self.list = QWidget(); self.rows = QVBoxLayout(self.list); self.rows.setContentsMargins(0, 0, 8, 0); self.rows.setSpacing(8)
        self.rows.addStretch(); self.scroll.setWidget(self.list)
        layout.addWidget(self.scroll, 1)

    def open(self, customer_id, zone):
        """Remember the customer; nothing is fetched until the tab is shown."""
        self.customer_id, self.zone, self.loaded_for = customer_id, zone, None
        self.day.clear(); self.search.clear(); self.clear_search.hide(); self.show_lines([], 'Open the tab to load the chat.')

    def ensure_loaded(self):
        if self.customer_id is not None and self.loaded_for != self.customer_id:
            self.load()

    def load(self, day=None, q=None):
        if self.customer_id is None or not self.runner.start(
                lambda cid=self.customer_id: (cid, q, self.backend.chat(cid, day=day, q=q))):
            return
        self.loaded_for = self.customer_id
        self.status.setText('Searching…' if q else 'Loading the chat…')

    def loaded(self, result, error):
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit()
            self.show_lines([], f'The chat could not be loaded: {error}'); return
        cid, q, page = result
        if cid != self.customer_id:
            return
        self.result = page
        self.day.blockSignals(True); self.day.clear()
        for d in page.days:
            self.day.addItem(d, d)
        if page.day:
            self.day.setCurrentIndex(max(0, self.day.findData(page.day)))
        self.day.blockSignals(False)
        self.clear_search.setVisible(bool(q))
        if q:
            text = f'{len(page.messages)} messages match “{q}” in the last {len(page.days)} days, newest first.'
        elif not page.days:
            text = 'No chat uploaded yet: the box sends it with its clips every 15 minutes (14 days are kept).'
        else:
            text = f'{len(page.messages)} messages on {page.day}.'
        self.show_lines(page.messages, text)

    def show_lines(self, lines, status):
        self.status.setText(status)
        while self.rows.count() > 1:
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().hide(); item.widget().deleteLater()
        self.bubbles, self.pending_images, self.image_labels = [], [], {}
        for line in lines:
            row = self.bubble(line)
            self.rows.insertWidget(self.rows.count()-1, row); self.bubbles.append(row)
        self.next_image()

    def bubble(self, line):
        owner = line.who == 'owner'
        row = QWidget(); outer = QHBoxLayout(row); outer.setContentsMargins(0, 0, 0, 0)
        card = QFrame(); card.setObjectName('opinion'); card.setMaximumWidth(560); card.setMinimumWidth(360)
        card.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        box = QVBoxLayout(card); box.setContentsMargins(12, 8, 12, 8); box.setSpacing(4)
        who = WHO.get(line.who, line.who.title()) + (f' · {line.name}' if line.name else '')
        about = (f'   ·   about {line.camera_name}' if line.camera_name and line.kind not in ('alert', 'photo', 'video')
                 else '')
        head = label(f'{who}   {local_time(line.ts, self.zone)}{about}', 'muted', True); box.addWidget(head)
        if line.kind == 'button':
            text = label(f'Pressed: {line.text}', 'badge', True)
        elif line.kind in ('photo', 'video') and line.who == 'assistant':
            text = label(f'Sent a {line.kind}' + (f' from {line.camera_name}' if line.camera_name else '')
                         + (f': {line.text}' if line.text else ''), 'muted', True)
        elif line.kind == 'video':  # the box's clip that follows an alert
            text = label(line.text or 'Video of the alert', 'muted', True)
        elif line.kind == 'voice':
            text = label(f'Voice message, transcribed: {line.text}', '', True)
        else:
            text = label(line.text or '(no text)', 'opinionText' if line.kind == 'alert' else '', True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(text)
        if line.kind == 'alert':
            meta = 'ALERT' + (f' · {line.camera_name}' if line.camera_name else '')
            box.insertWidget(0, label(meta, 'eyebrow'))
            if line.image:
                picture = QLabel('Loading picture…'); picture.setObjectName('muted'); picture.setFixedSize(320, 180)
                picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
                box.addWidget(picture)
                self.pending_images.append((line.site, line.image)); self.image_labels.setdefault((line.site, line.image), []).append(picture)
        if not line.delivered:
            box.addWidget(label(f'Not delivered to the owner: {line.error or "reason not recorded"}', 'error', True))
        if owner:
            outer.addStretch(); outer.addWidget(card)
        else:
            outer.addWidget(card); outer.addStretch()
        row.line = line
        return row

    def next_image(self):
        while self.pending_images and not self.image_runner.busy:
            site, image = self.pending_images.pop(0)
            cid = self.customer_id; self.current_image = (site, image)
            self.image_runner.start(lambda: (site, image, self.backend.media_bytes(self.backend.chat_image(cid, site, image).url)))

    def image_loaded(self, result, error):
        if not error:
            site, image, data = result
            pix = QPixmap(); pix.loadFromData(data)
            for picture in self.image_labels.get((site, image), []):
                if pix.isNull():
                    picture.setText('Picture unavailable')
                else:
                    picture.setPixmap(pix.scaled(picture.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))
        elif isinstance(error, AuthError):
            self.session_expired.emit(); return
        else:
            for picture in self.image_labels.get(self.current_image, []):
                picture.setText('Picture unavailable')
        self.next_image()
