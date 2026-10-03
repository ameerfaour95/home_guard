from difflib import SequenceMatcher
from PySide6.QtCore import Qt, QStringListModel, Signal, QEvent
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QListView
from .common import label
from ..formatting import site_name


class CommandPalette(QDialog):
    jump = Signal(int, str)

    def __init__(self, parent, devices, customers, role):
        super().__init__(parent)
        self.setWindowTitle('Jump to…')
        self.setObjectName('commandPalette')
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setFixedSize(640, 432)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 16)
        layout.setSpacing(16)
        layout.addWidget(label('JUMP TO CUSTOMER OR DEVICE', 'eyebrow'))
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search by name, site or device…')
        self.search.setAccessibleName('Command palette search')
        layout.addWidget(self.search)
        self.entries = []
        if role != 'labeler':
            self.entries += [(f'{c.name}   ·   Customer', c.id, '') for c in customers]
            self.entries += [(f'{site_name(d.site)}   ·   {d.device_id}   ·   {d.customer_name}', d.customer_id, d.device_id) for d in devices]
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
        def score(entry):
            text = entry[0].casefold()
            if not query or query in text:
                return 2
            it = iter(text)
            if all(char in it for char in query):
                return 1
            return max(SequenceMatcher(None, query, word).ratio() for word in text.split())
        self.matches = sorted([(score(e), e) for e in self.entries], key=lambda item: -item[0])
        self.matches = [entry for score_value, entry in self.matches if score_value >= .6]
        self.model.setStringList([e[0] for e in self.matches])
        if self.matches:
            self.results.setCurrentIndex(self.model.index(0))
        self.hint.setText('↑ ↓ Navigate     Enter Open     Esc Close' if self.matches else 'No matches. Try a customer name or device ID.' if self.entries else 'No customers loaded. Close search and check the Fleet connection.')

    def activate(self, index):
        if index.isValid() and index.row() < len(self.matches):
            _, customer_id, device_id = self.matches[index.row()]
            self.jump.emit(customer_id, device_id)
            self.accept()

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
