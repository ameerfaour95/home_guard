from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from .workers import TaskRunner
from .backend import AuthError
from .formatting import local_time, humanise
from .widgets.common import label, button
from .widgets.data_table import RowsModel, data_table


class IndexProblems(QWidget):
    session_expired = Signal()

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        layout = QVBoxLayout(self); layout.setContentsMargins(32,24,32,20); layout.setSpacing(16)
        top = QHBoxLayout(); top.addWidget(label('Index problems', 'title'), 1)
        self.refresh_button = button('Refresh', self.refresh); top.addWidget(self.refresh_button); layout.addLayout(top)
        layout.addWidget(label('Evidence Cloud could not process. Share the object path with the service administrator.', 'muted', True))
        self.message = label('Loading index problems…', 'muted', True); layout.addWidget(self.message)
        self.model = RowsModel([('Last seen · UTC', lambda r:local_time(r.seen_utc,'UTC')),
                                ('Problem', lambda r:humanise(r.reason)), ('Object path', lambda r:r.s3_key)])
        self.table = data_table(self.model, 2); self.table.setColumnWidth(0,210); self.table.setColumnWidth(1,420)
        layout.addWidget(self.table,1)
        self.runner = TaskRunner(self); self.runner.finished.connect(self.loaded)

    def showEvent(self, event):
        super().showEvent(event); self.refresh()

    def refresh(self):
        if self.runner.busy: return
        self.refresh_button.setEnabled(False); self.message.setText('Loading index problems…')
        self.runner.start(self.backend.index_problems)

    def loaded(self, result, error):
        self.refresh_button.setEnabled(True)
        if error:
            self.message.setText(str(error)+' Select Refresh to retry.')
            if isinstance(error, AuthError): self.session_expired.emit()
            return
        self.model.replace(result)
        self.message.setText(f'{len(result)} problems · Most recent first' if result else 'All clear. No index problems were reported.')
