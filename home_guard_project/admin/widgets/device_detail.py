from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QScrollArea, QWidget
from .common import label, button
from ..formatting import local_time, mode_name, site_name
from ..theme import verdict_color


class DeviceDetail(QFrame):
    open_customer = Signal(int)
    dismissed = Signal()

    def __init__(self, theme='dark'):
        super().__init__()
        self.theme, self.device = theme, None
        self.setObjectName('detail')
        self.setFixedWidth(288)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        top = QHBoxLayout()
        top.addWidget(label('DEVICE DETAILS', 'eyebrow'))
        top.addStretch()
        close = button('×', self.dismissed.emit, 'link')
        close.setAccessibleName('Close device details')
        close.setFixedWidth(32)
        top.addWidget(close)
        outer.addLayout(top)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName('detailBody')
        self.content = QVBoxLayout(body)
        self.content.setContentsMargins(0, 8, 0, 8)
        self.content.setSpacing(12)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        outer.addWidget(button('Open customer  →', lambda: self.open_customer.emit(self.device.customer_id), 'primary'))

    def show_device(self, device, timezone='UTC'):
        self.device = device
        while self.content.count():
            item = self.content.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.content.addWidget(label(device.customer_name, 'section', True))
        self.content.addWidget(label(site_name(device.site), 'muted', True))
        verdict = label(device.verdict.title(), 'section')
        verdict.setStyleSheet(f'color: {verdict_color(device.verdict, self.theme)};')
        self.content.addWidget(verdict)
        for reason in device.reasons:
            self.content.addWidget(label('•  ' + reason.message, '', True))
        self.content.addSpacing(8)
        values = [('DEVICE', device.device_id), ('HOST', device.host or 'Not reported'),
                  ('DISK FREE', f'{device.disk_free_gb:g} GB' if device.disk_free_gb is not None else 'Not reported'),
                  ('COLLECTOR', 'Running' if device.collector_running else 'Stopped' if device.collector_running is False else 'Not reported'),
                  ('MODE', mode_name(device.mode)), ('LAST SEEN · CUSTOMER TIME', local_time(device.last_seen_utc, timezone)),
                  ('VERSIONS', 'Not reported by the current API')]
        for title, value in values:
            self.content.addWidget(label(title, 'muted'))
            self.content.addWidget(label(value, '', True))
        self.content.addStretch()
