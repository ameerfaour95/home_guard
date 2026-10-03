from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QTabWidget
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
        content.addLayout(header)
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
        self.zone = customer.timezone
        self.name.setText(customer.name)
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

    def open_event(self, event_id):
        rows = self.timeline.model.rows
        for i, event in enumerate(rows):
            if event.id == event_id:
                self.timeline.table.setCurrentIndex(self.timeline.model.index(i, 0))
                self.event_view.prev.setEnabled(i > 0)
                self.event_view.next.setEnabled(i < len(rows)-1 or bool(self.timeline.cursor))
                if self.review_mode and self.role == 'labeler':
                    self.name.setText(event.customer_name)
                    self.health.setText(f'{event.site}  ·  {event.camera}  ·  Times shown in UTC')
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
