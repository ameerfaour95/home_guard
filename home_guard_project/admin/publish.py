"""Admin-only tagging batches, using the existing asynchronous backend pattern."""
import re
from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QApplication, QPlainTextEdit
from .workers import TaskRunner
from .backend import AuthError
from .widgets.common import label, button
from .widgets.data_table import RowsModel, data_table


class PublishDialog(QDialog):
    published = Signal(object)
    session_expired = Signal()

    def __init__(self, backend, collection, parent=None):
        super().__init__(parent)
        self.backend, self.collection = backend, collection
        self.setWindowTitle('Publish as tagging batch'); self.resize(620, 370)
        layout = QVBoxLayout(self); layout.setContentsMargins(24, 24, 24, 24); layout.setSpacing(16)
        layout.addWidget(label('Publish as tagging batch', 'section'))
        layout.addWidget(label(collection.name, 'muted'))
        self.summary = label('Checking labeled, not-labeled and dropped clips…', 'muted', True); layout.addWidget(self.summary)
        layout.addWidget(label('Batch name', 'eyebrow'))
        self.name = QLineEdit(); self.name.setPlaceholderText('site_batch_1'); self.name.setAccessibleName('Batch name'); layout.addWidget(self.name)
        self.error = label('', 'error', True); layout.addWidget(self.error)
        layout.addWidget(label('Saved exactly like your Label Studio batches: <batch>.json, dataset_multi/ and analysis_output/ (YOLO labels + images, vlm_training.jsonl).', 'muted', True))
        actions = QHBoxLayout(); actions.addStretch(); actions.addWidget(button('Cancel', self.reject))
        self.submit = button('Publish batch', self.publish, 'primary'); self.submit.setEnabled(False); actions.addWidget(self.submit); layout.addLayout(actions)
        self.loader, self.writer = TaskRunner(self), TaskRunner(self)
        self.loader.finished.connect(self.loaded); self.writer.finished.connect(self.saved)
        def fetch():
            events, cursor = [], None
            while True:
                page = backend.collection_events(collection.id, cursor=cursor)
                events.extend(page.items); cursor = page.next_cursor
                if not cursor: break
            return events, [backend.annotation(e.id) for e in events], backend.publishes()
        self.loader.start(fetch)

    def loaded(self, result, error):
        if error: self.failed(error); return
        events, annotations, batches = result
        labeled = sum(a.status in ('submitted', 'reviewed') for a in annotations)
        dropped = sum(a.drop_clip for a in annotations)
        self.summary.setText(f'{len(events)} clips   ·   {labeled} labeled   ·   {len(events)-labeled} not labeled   ·   {dropped} dropped')
        site = re.sub('[^a-z0-9_]', '_', events[0].site.lower() if events else 'site')
        names = {p.batch_name for p in batches}; n = 1
        while f'{site}_batch_{n}' in names: n += 1
        self.name.setText(f'{site}_batch_{n}'); self.submit.setEnabled(True)

    def failed(self, error):
        self.error.setText(str(error))
        if isinstance(error, AuthError): self.session_expired.emit()

    def publish(self):
        name = self.name.text().strip()
        if not re.fullmatch(r'[a-z0-9_]+', name):
            self.error.setText('Use lowercase letters, numbers and underscores only.'); return
        self.submit.setEnabled(False); self.name.setEnabled(False)
        self.writer.start(lambda: self.backend.publish_collection(self.collection.id, name))

    def saved(self, result, error):
        self.submit.setEnabled(True); self.name.setEnabled(True)
        if error: self.failed(error); return
        self.published.emit(result); self.accept()


class PublishList(QWidget):
    session_expired = Signal()

    def __init__(self, backend, theme='dark'):
        super().__init__(); self.backend = backend
        box = QVBoxLayout(self); box.setContentsMargins(0, 12, 0, 0); box.setSpacing(12)
        header = QHBoxLayout(); header.addWidget(label('Tagging batches', 'section'), 1); header.addWidget(button('Refresh', self.refresh)); box.addLayout(header)
        box.addWidget(label('Label Studio tasks, every labeled YOLO frame and corrected AI descriptions.', 'muted', True))
        self.model = RowsModel([('Batch', lambda p: p.batch_name), ('State', lambda p: p.state.title()),
            ('Tasks', lambda p: p.tasks), ('YOLO frames', lambda p: p.yolo_frames), ('VLM lines', lambda p: p.vlm_lines), ('Missing', lambda p: len(p.missing))])
        self.table = data_table(self.model); box.addWidget(self.table, 1)
        self.table.setColumnWidth(0, 280)
        from .studio import StateDelegate
        self.table.setItemDelegateForColumn(1, StateDelegate(theme, self.table))
        self.detail = QPlainTextEdit('Select a batch to see missing items and its S3 path.')
        self.detail.setReadOnly(True); self.detail.setMaximumHeight(150); self.detail.setAccessibleName('Batch path and missing items'); box.addWidget(self.detail)
        self.copy = button('Copy S3 path', self.copy_path); self.copy.setEnabled(False); box.addWidget(self.copy)
        self.table.selectionModel().currentRowChanged.connect(self.selected)
        self.runner = TaskRunner(self); self.runner.finished.connect(self.loaded)
        self.timer = QTimer(self); self.timer.setInterval(5000); self.timer.timeout.connect(self.refresh)

    def showEvent(self, event):
        self.refresh(); self.timer.start(); super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop(); super().hideEvent(event)

    def refresh(self):
        if not self.runner.busy: self.runner.start(self.backend.publishes)

    def current(self):
        index = self.table.currentIndex()
        return self.model.items[index.row()] if index.isValid() else None

    def loaded(self, result, error):
        if error:
            self.detail.setPlainText(str(error))
            if isinstance(error, AuthError): self.session_expired.emit()
            return
        old = self.current(); self.model.replace(result)
        if result:
            index = next((i for i, p in enumerate(result) if old and p.batch_name == old.batch_name), 0)
            self.table.setCurrentIndex(self.model.index(index, 0))
        else: self.detail.setPlainText('No tagging batches yet. Open a collection to publish one.')

    def selected(self, *_):
        item = self.current(); self.copy.setEnabled(bool(item))
        if item:
            missing = '\n'.join(f'#{m.event_id} · {m.reason}' for m in item.missing)
            self.detail.setPlainText(item.s3_prefix+'\n'+('Missing items:\n'+missing if missing else 'No missing items.'))

    def copy_path(self):
        item = self.current()
        if item: QApplication.clipboard().setText(item.s3_prefix)
