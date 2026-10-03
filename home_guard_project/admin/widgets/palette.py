from difflib import SequenceMatcher
from PySide6.QtCore import Qt, QStringListModel, Signal, QEvent
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QListView
from .common import label
from ..formatting import site_name, camera_name


class CommandPalette(QDialog):
    jump = Signal(int, str)
    execute = Signal(str, object)

    def __init__(self, parent, devices, customers, role, filters=(), cameras=()):
        super().__init__(parent)
        self.setWindowTitle('Jump to…')
        self.setObjectName('commandPalette')
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setFixedSize(640, 432)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 16)
        layout.setSpacing(16)
        layout.addWidget(label('SEARCH & COMMANDS', 'eyebrow'))
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search names, cameras, filters, commands or #event ID…')
        self.search.setAccessibleName('Command palette search')
        layout.addWidget(self.search)
        self.entries = []
        if role != 'labeler':
            self.entries += [(f'{c.name}   ·   Customer', c.id, '') for c in customers]
            self.entries += [(f'{site_name(d.site)}   ·   {d.device_id}   ·   {d.customer_name}', d.customer_id, d.device_id) for d in devices]
        if role != 'labeler':
            self.entries += [(f'{camera_name(name)}   ·   Camera', cid, 'camera:'+name) for name,cid in cameras]
        else:
            self.search.setPlaceholderText('Find a command or saved filter…')
        self.entries += [('Filter: '+f.title, 0, 'filter:'+f.key) for f in filters]
        commands = ['Go to Studio', 'Go to Review']
        if role != 'labeler': commands += ['Go to Fleet']
        if role == 'admin': commands += ['Go to Audit']
        if role != 'support': commands += ['Export collection…']
        commands += ['Sign out']
        self.entries += [(name, 0, 'command:'+name) for name in commands]
        self.results = QListView()
        self.results.setSpacing(4)
        self.model = QStringListModel(self)
        self.results.setModel(self.model)
        self.results.clicked.connect(self.activate)
        self.results.activated.connect(self.activate)
        layout.addWidget(self.results, 1)
        self.hint = label('↑ ↓ Navigate     Enter Open     Esc Close', 'muted')
        layout.addWidget(self.hint)
        self.search.textChanged.connect(self.filter)
        self.search.installEventFilter(self)
        self.filter('')

    def filter(self, query):
        query = query.casefold().strip()
        commands_only = query.startswith('>')
        if commands_only: query = query[1:].strip()
        def score(entry):
            text = entry[0].casefold()
            if query and text == query: return 4
            if query and text.startswith(query): return 3
            if not query or query in text:
                return 2
            it = iter(text)
            if all(char in it for char in query):
                return 1
            return max(SequenceMatcher(None, query, word).ratio() for word in text.split())
        entries = [(f'Open event #{query[1:]}', int(query[1:]), 'event')] if query.startswith('#') and query[1:].isdigit() else self.entries
        if commands_only: entries = [e for e in entries if e[2].startswith('command:')]
        self.matches = sorted([(score(e), e) for e in entries], key=lambda item: -item[0])
        self.matches = [entry for score_value, entry in self.matches if score_value >= .6]
        self.model.setStringList([e[0] for e in self.matches])
        if self.matches:
            self.results.setCurrentIndex(self.model.index(0))
        self.hint.setText('↑ ↓ Navigate     Enter Open     Esc Close' if self.matches else 'No matches. Try a customer name or device ID.' if self.entries else 'No customers loaded. Close search and check the Fleet connection.')

    def activate(self, index):
        if index.isValid() and index.row() < len(self.matches):
            _, customer_id, device_id = self.matches[index.row()]
            self.accept()
            if device_id == 'event': self.execute.emit('event',customer_id)
            elif device_id.startswith(('camera:','filter:','command:')):
                kind,value = device_id.split(':',1); self.execute.emit(kind,(customer_id,value) if kind == 'camera' else value)
            else: self.jump.emit(customer_id,device_id)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                delta = 1 if event.key() == Qt.Key.Key_Down else -1
                row = max(0, min(len(self.matches)-1, self.results.currentIndex().row()+delta))
                self.results.setCurrentIndex(self.model.index(row))
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.activate(self.results.currentIndex())
                return True
        return super().eventFilter(watched, event)
