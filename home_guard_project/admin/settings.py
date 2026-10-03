import os
from pathlib import Path
from urllib.parse import urlsplit
from PySide6.QtCore import QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox
from .widgets.common import label, button


def valid_server(value):
    try:
        url = urlsplit(value)
        return bool(url.hostname and not (url.username or url.password or url.query or url.fragment)
                    and (url.port is None or 1 <= url.port <= 65535)
                    and (url.scheme == 'https' or (url.scheme == 'http' and url.hostname in ('localhost','127.0.0.1'))))
    except ValueError:
        return False


class SettingsDialog(QDialog):
    saved = Signal(str, str)

    def __init__(self, server, theme, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('Settings'); self.setModal(True); self.setFixedWidth(560)
        layout = QVBoxLayout(self); layout.setContentsMargins(28,24,28,24); layout.setSpacing(14)
        layout.addWidget(label('Settings', 'title'))
        layout.addWidget(label('Home Guard Cloud server', 'section'))
        self.server = QLineEdit(server); self.server.setPlaceholderText('https://cloud.example.com'); layout.addWidget(self.server)
        layout.addWidget(label('Use the server address supplied by your administrator. Changing servers signs you out.', 'muted', True))
        layout.addWidget(label('Appearance', 'section'))
        self.theme = QComboBox(); self.theme.addItem('Dark','dark'); self.theme.addItem('Light','light')
        self.theme.setCurrentIndex(max(0,self.theme.findData(theme))); layout.addWidget(self.theme)
        layout.addWidget(button('Open log folder', self.open_logs))
        self.error = label('', 'error', True); layout.addWidget(self.error)
        actions = QHBoxLayout(); actions.addStretch(); actions.addWidget(button('Cancel',self.reject))
        actions.addWidget(button('Save settings', self.save, 'primary')); layout.addLayout(actions)

    def save(self):
        server = self.server.text().strip().rstrip('/')
        if not valid_server(server):
            self.error.setText('Enter an HTTPS server URL without credentials, query or fragment. HTTP is allowed for localhost or 127.0.0.1.'); return
        self.saved.emit(server, self.theme.currentData()); self.accept()

    def open_logs(self):
        path = Path(os.environ.get('LOCALAPPDATA', Path.home()))/'HomeGuardAdmin'/'logs'
        path.mkdir(parents=True, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.error.setText(f'Open this folder in File Explorer: {path}')
