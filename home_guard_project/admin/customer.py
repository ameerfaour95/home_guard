from .formatting import camera_name
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QTabWidget, QDialog
from .backend import OfflineError, AuthError
from .fleet_model import SEVERITY
from .formatting import site_name, age, utcnow
from .workers import TaskRunner
from .timeline import TimelineScreen
from .event_view import EventView
from .widgets.icons import icon
from .widgets.common import label, button, EmptyState, Skeleton


class CustomerScreen(QWidget):
    back = Signal()
    session_expired = Signal()

    def __init__(self, backend, theme='dark', role='admin', review=False):
        super().__init__()
        self.backend, self.role, self.review_mode = backend, role, review
        self.customer_id, self.device_id, self.zone = None, '', 'UTC'
        self.requested = None
        self.pending_navigation = None
        layout = QVBoxLayout(self); layout.setContentsMargins(32, 20, 32, 20); layout.setSpacing(10)
        if not review:
            back = button('←  Back to Fleet', self.back.emit, 'link')
            back.setStyleSheet('padding: 0; min-height: 20px;')
            layout.addWidget(back, alignment=Qt.AlignmentFlag.AlignLeft)
        self.stack = QStackedWidget(); layout.addWidget(self.stack, 1)
        self.loading = Skeleton(theme); self.stack.addWidget(self.loading)
        self.body = QWidget(); content = QVBoxLayout(self.body); content.setContentsMargins(0, 0, 0, 0); content.setSpacing(8)
        header = QHBoxLayout(); names = QVBoxLayout(); names.setSpacing(4)
        self.name = label('Review' if review else '', 'title'); names.addWidget(self.name)
        self.health = label('Recorded events across the fleet' if review else '', 'muted', True); names.addWidget(self.health)
        header.addLayout(names, 1)
        self.consent = label('Access follows customer consent' if review else '', 'muted', True)
        self.consent.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); header.addWidget(self.consent)
        self.consent_button = button('Consent…', self.review_consent, 'compact'); self.consent_button.hide()
        header.addWidget(self.consent_button, alignment=Qt.AlignmentFlag.AlignVCenter)
        content.addLayout(header)
        self.consent_note = label('', 'error', True); self.consent_note.hide(); content.addWidget(self.consent_note)
        self.customer = None
        self.consent_runner = TaskRunner(self); self.consent_runner.finished.connect(self.consent_saved)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True); content.addWidget(self.tabs, 1)
        self.tabs.tabBar().setDrawBase(False)
        self.timeline = TimelineScreen(backend, theme); self.event_view = EventView(backend, role, theme)
        self.tabs.addTab(self.timeline, icon('Timeline', theme), 'Timeline')
        self.tabs.addTab(self.event_view, icon('Event', theme), 'Event'); self.tabs.setTabEnabled(1, False)
        for title, description in [('Conversation', 'Owner messages and their context will appear here.'),
                                   ('Config', 'Box settings and change history will appear here.'),
                                   ('Access', 'Staff recording access and owner notices will appear here.')]:
            if role == 'labeler' and title in ('Conversation', 'Access'):
                continue
            self.tabs.addTab(EmptyState(f'{title} is coming soon in this release', description, eyebrow=title.upper()), icon(title, theme), title)
        self.timeline.event_requested.connect(self.open_event)
        self.timeline.session_expired.connect(self.session_expired)
        self.timeline.runner.finished.connect(self.page_arrived)
        self.event_view.navigate.connect(self.navigate_event)
        self.event_view.review_changed.connect(self.timeline.model.update_review)
        self.event_view.session_expired.connect(self.session_expired)
        self.stack.addWidget(self.body)
        self.failure = EmptyState('Customer could not be loaded', 'Please try again. Your place in the fleet is saved.', eyebrow='UNAVAILABLE')
        self.failure.action.show(); self.failure.action.clicked.connect(lambda: self.open(self.customer_id, self.device_id)); self.stack.addWidget(self.failure)
        self.offline = EmptyState("Can't reach Home Guard Cloud", 'Check your connection, then try loading this customer again.', eyebrow='OFFLINE')
        self.offline.action.show(); self.offline.action.clicked.connect(lambda: self.open(self.customer_id, self.device_id)); self.stack.addWidget(self.offline)
        self.runner = TaskRunner(self); self.runner.finished.connect(self.completed)
        if review:
            self.stack.setCurrentWidget(self.body)
            self.timeline.open()

    def open(self, customer_id, device_id=''):
        self.customer_id, self.device_id = customer_id, device_id
        self.event_view.player.player.stop()
        self.stack.setCurrentWidget(self.loading)
        if self.runner.busy:
            return
        self.requested = (customer_id, device_id)
        self.runner.start(lambda: self.backend.customer(customer_id))

    def completed(self, customer, error):
        if self.requested != (self.customer_id, self.device_id):
            self.open(self.customer_id, self.device_id); return
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit()
            else:
                self.stack.setCurrentWidget(self.offline if isinstance(error, OfflineError) else self.failure)
            return
        self.zone, self.customer = customer.timezone, customer
        self.name.setText(customer.name)
        proposed = getattr(customer, 'consent_proposed', None)
        missing = not (customer.consent_recordings and customer.consent_training)
        self.consent_button.setVisible(self.role == 'admin' and not self.review_mode)
        self.consent_button.setText('Confirm consent given at setup…' if proposed else 'Consent…')
        self.consent_button.setObjectName('primary' if proposed else 'compact')
        self.consent_button.style().unpolish(self.consent_button); self.consent_button.style().polish(self.consent_button)
        if proposed and missing and self.role == 'admin':
            self.consent_note.setText('The box’s setup recorded the owner’s consent, but nobody has confirmed it yet: '
                                      'until then this household’s recordings cannot be opened or used for training.')
            self.consent_note.show()
        else:
            self.consent_note.hide()
        devices = [d for d in customer.devices if d.device_id == self.device_id] or customer.devices
        if devices:
            device = min(devices, key=lambda d: SEVERITY[d.verdict])
            reason = device.reasons[0].message if device.reasons else 'No health details reported'
            now = getattr(self.backend, 'now', None) or utcnow()
            self.health.setText(f'{site_name(device.site)}  ·  {device.verdict.title()} — {reason}')
        else:
            self.health.setText('No boxes enrolled')
        self.consent.setText('Consent  ·  '+ '  /  '.join(f'{text}: {"yes" if allowed else "no"}' for text, allowed in
                            [('live', customer.consent_live), ('recordings', customer.consent_recordings), ('training', customer.consent_training)])
                            + (f'\nLast heard {age(device.last_seen_utc, now).lower()}' if devices else ''))
        self.consent.setMinimumWidth(410)
        self.tabs.setCurrentIndex(0); self.tabs.setTabEnabled(1, False)
        self.timeline.open(customer.id, customer.timezone)
        self.stack.setCurrentWidget(self.body)

    def review_consent(self):
        """Show what the box recorded at setup. An admin can only confirm that, never grant consent of their own."""
        customer = self.customer
        if customer is None:
            return None
        proposed = getattr(customer, 'consent_proposed', None) or {}
        dialog = QDialog(self); dialog.setWindowTitle('Customer consent'); dialog.setObjectName('commandPalette')
        layout = QVBoxLayout(dialog); layout.setContentsMargins(24, 20, 24, 20); layout.setSpacing(12)
        layout.addWidget(label(f'Consent for {customer.name}', 'section'))
        row = QHBoxLayout(); row.addStretch()
        if proposed:
            when = str(proposed.get('recorded_utc') or '')[:16].replace('T', ' ')
            by = proposed.get('installer') or 'the installer'
            layout.addWidget(label(f'At setup on {when} UTC ({by} installed the box), the customer answered:', 'muted', True))
            for key, text in (('live', 'Live view: staff may watch the cameras live'),
                              ('recordings', 'Recordings: staff may open saved clips'),
                              ('training', 'Training: clips may be tagged and used to train the AI')):
                layout.addWidget(label(f'{"Yes" if proposed.get(key) else "No"}   ·   {text}',
                                       '' if proposed.get(key) else 'muted'))
            layout.addWidget(label('Confirming applies exactly these answers and is recorded in the audit log with your '
                                   'name. Consent the customer did not give cannot be added here.', 'muted', True))
            row.addWidget(button('Cancel', dialog.reject))
            row.addWidget(button('Confirm the consent this customer gave at setup', dialog.accept, 'primary'))
            dialog.accepted.connect(lambda: self.confirm_consent(proposed.get('recorded_utc')))
        else:
            current = '  ·  '.join(f'{text}: {"yes" if getattr(customer, f"consent_{key}") else "no"}'
                                   for key, text in (('live', 'live view'), ('recordings', 'recordings'), ('training', 'training')))
            layout.addWidget(label(f'Now: {current}', 'muted', True))
            layout.addWidget(label("No consent was recorded at this customer's setup, so there is nothing to confirm. "
                                   "Consent can only come from the customer: they answer during the box's setup.", '', True))
            row.addWidget(button('Close', dialog.reject, 'primary'))
        layout.addLayout(row)
        dialog.setMinimumWidth(520)
        dialog.show()
        self.consent_dialog = dialog
        return dialog

    def confirm_consent(self, recorded_utc):
        self.consent_button.setEnabled(False)
        customer_id = self.customer.id
        self.consent_runner.start(lambda: self.backend.confirm_consent(customer_id, recorded_utc))

    def consent_saved(self, customer, error):
        self.consent_button.setEnabled(True)
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit(); return
            self.consent_note.setText(f'Consent was not saved: {error}'); self.consent_note.show(); return
        self.open(self.customer_id, self.device_id)

    def open_event(self, event_id):
        rows = self.timeline.model.rows
        for i, event in enumerate(rows):
            if event.id == event_id:
                self.timeline.table.setCurrentIndex(self.timeline.model.index(i, 0))
                self.event_view.prev.setEnabled(i > 0)
                self.event_view.next.setEnabled(i < len(rows)-1 or bool(self.timeline.cursor))
                if self.review_mode and self.role == 'labeler':
                    self.name.setText(event.customer_name)
                    self.health.setText(f'{event.site}  ·  {camera_name(event)}  ·  Times shown in {event.timezone}')
                break
        self.tabs.setTabEnabled(1, True); self.tabs.setCurrentIndex(1)
        self.event_view.open(event_id, self.zone)

    def navigate_event(self, direction):
        rows = self.timeline.model.rows
        index = next((i for i, e in enumerate(rows) if e.id == self.event_view.event_id), -1)
        target = index+direction
        if 0 <= target < len(rows):
            self.open_event(rows[target].id)
        elif target == len(rows) and self.timeline.cursor:
            self.pending_navigation = target; self.timeline.load_older()

    def page_arrived(self, result, error):
        if self.pending_navigation is not None and not error:
            target, self.pending_navigation = self.pending_navigation, None
            if target < len(self.timeline.model.rows):
                self.open_event(self.timeline.model.rows[target].id)
