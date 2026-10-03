from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QFrame
from .backend import OfflineError, AuthError
from .fleet_model import SEVERITY
from .formatting import site_name
from .workers import TaskRunner
from .widgets.common import label, button, EmptyState, Skeleton


class CustomerScreen(QWidget):
    back = Signal()
    session_expired = Signal()

    def __init__(self, backend, theme='dark'):
        super().__init__()
        self.backend = backend
        self.customer_id, self.device_id = None, ''
        self.requested = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.addWidget(button('←  Back to Fleet', self.back.emit, 'link'), alignment=Qt.AlignmentFlag.AlignLeft)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.loading = Skeleton(theme)
        self.stack.addWidget(self.loading)
        self.body = QWidget()
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(0, 16, 0, 0)
        self.content.setSpacing(24)
        self.stack.addWidget(self.body)
        self.failure = EmptyState('Customer could not be loaded', 'Please try again. Your place in the fleet is saved.', eyebrow='REQUEST FAILED')
        self.failure.action.show()
        self.failure.action.clicked.connect(lambda: self.open(self.customer_id, self.device_id))
        self.stack.addWidget(self.failure)
        self.offline = EmptyState("Can't reach Home Guard Cloud", 'Check your connection, then try loading this customer again.', eyebrow='OFFLINE')
        self.offline.action.show()
        self.offline.action.clicked.connect(lambda: self.open(self.customer_id, self.device_id))
        self.stack.addWidget(self.offline)
        self.runner = TaskRunner(self)
        self.runner.finished.connect(self.completed)

    def open(self, customer_id, device_id=''):
        self.customer_id, self.device_id = customer_id, device_id
        self.stack.setCurrentWidget(self.loading)
        if self.runner.busy:
            return
        self.requested = (customer_id, device_id)
        self.runner.start(lambda: self.backend.customer(customer_id))

    def completed(self, customer, error):
        if self.requested != (self.customer_id, self.device_id):
            self.open(self.customer_id, self.device_id)
            return
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit()
            else:
                self.stack.setCurrentWidget(self.offline if isinstance(error, OfflineError) else self.failure)
            return
        while self.content.count():
            item = self.content.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.content.addWidget(label(customer.name, 'title'))
        devices = [d for d in customer.devices if d.device_id == self.device_id] or customer.devices
        verdict = min(devices, key=lambda d: SEVERITY[d.verdict]).verdict.title() if devices else 'No boxes enrolled'
        self.content.addWidget(label('  /  '.join([verdict, ', '.join(site_name(d.site) for d in devices), customer.timezone]), 'muted', True))
        card = QFrame()
        card.setObjectName('card')
        row = QHBoxLayout(card)
        row.setContentsMargins(24, 24, 24, 24)
        for text, allowed in [('Live view', customer.consent_live), ('Recordings', customer.consent_recordings), ('Training', customer.consent_training)]:
            row.addWidget(label(f'{text} consent\n' + ('Granted' if allowed else 'Not granted')))
        self.content.addWidget(card)
        self.content.addWidget(EmptyState('The full customer view is coming next', 'Timeline, video playback and AI answers arrive in round 2.\nCustomer consent will govern access to recordings and live views.', eyebrow='CUSTOMER WORKSPACE'), 1)
        self.stack.setCurrentWidget(self.body)
