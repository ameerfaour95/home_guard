from PySide6.QtCore import Qt, Signal, QSize, QRectF
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QListView, QStyledItemDelegate, QAbstractItemView, QLineEdit, QStyle)
from .workers import TaskRunner, closing
from .backend import BackendError, AuthError
from .widgets.common import label, button
from .widgets.data_table import RowsModel
from .theme import PALETTES
from .formatting import local_time, camera_name
from .event_logic import provenance, ai_status


class CollectionPicker(QDialog):
    added = Signal(object)
    session_expired = Signal()

    def __init__(self, backend, event_id, last=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.backend, self.event_id, self.last = backend, event_id, last
        self.setWindowTitle('Add to collection'); self.setFixedWidth(440); self.setModal(True)
        box = QVBoxLayout(self); box.setContentsMargins(24, 24, 24, 24); box.setSpacing(16)
        box.addWidget(label('Add to collection', 'section'))
        box.addWidget(label(f'Event #{event_id} · Keep this moment for training', 'muted'))
        self.choices = QComboBox(); box.addWidget(self.choices)
        self.message = label('Loading collections…', 'muted', True); box.addWidget(self.message)
        self.submit = button('Add event', self.add, 'primary'); self.submit.setEnabled(False)
        actions = QHBoxLayout(); actions.addWidget(button('Cancel', self.reject)); actions.addWidget(self.submit); box.addLayout(actions)
        self.runner = TaskRunner(self); self.runner.finished.connect(self.loaded)
        self.writer = TaskRunner(self); self.writer.finished.connect(self.saved)
        self.runner.start(backend.collections)

    def loaded(self, collections, error):
        if error:
            if isinstance(error, AuthError): self.session_expired.emit()
            self.message.setText(str(error)); return
        for collection in collections:
            self.choices.addItem(f'{collection.name}  ·  {collection.event_count} events', collection.id)
        self.choices.setCurrentIndex(max(0, self.choices.findData(self.last)))
        self.message.setText('Choose a collection.' if collections else 'Create a collection in Studio first.')
        self.submit.setEnabled(bool(collections))

    def add(self):
        cid = self.choices.currentData()
        if cid is not None:
            self.submit.setEnabled(False); self.choices.setEnabled(False)
            self.writer.start(lambda: self.backend.add_collection_items(cid, [self.event_id]))

    def saved(self, collection, error):
        self.submit.setEnabled(True); self.choices.setEnabled(True)
        if error:
            if isinstance(error, AuthError): self.session_expired.emit()
            self.message.setText(str(error)); return
        self.added.emit(collection); self.accept()


class CreateCollection(QDialog):
    created = Signal(object)
    session_expired = Signal()

    def __init__(self, backend, parent=None):
        super().__init__(parent); self.backend = backend
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('New collection'); self.setFixedWidth(480); self.setModal(True)
        box = QVBoxLayout(self); box.setContentsMargins(24, 24, 24, 24); box.setSpacing(14)
        box.addWidget(label('New collection', 'section'))
        if getattr(parent, 'role', None) == 'labeler':
            box.addWidget(label('Visible only to you and administrators.', 'muted', True))
        self.name = QLineEdit(); self.name.setPlaceholderText('Collection name')
        self.description = QLineEdit(); self.description.setPlaceholderText('What are you collecting?')
        box.addWidget(self.name); box.addWidget(self.description)
        self.error = label('', 'error', True); box.addWidget(self.error)
        self.submit = button('Create collection', self.create, 'primary'); box.addWidget(self.submit)
        self.runner = TaskRunner(self); self.runner.finished.connect(self.completed)

    def create(self):
        name, description = self.name.text().strip(), self.description.text().strip()
        if not name:
            self.error.setText('Give this collection a name.'); return
        if len(name) > 120:
            self.error.setText('Use 120 characters or fewer for the collection name.'); return
        self.submit.setEnabled(False)
        self.runner.start(lambda: self.backend.create_collection(name, description))

    def completed(self, result, error):
        self.submit.setEnabled(True)
        if error:
            if isinstance(error, AuthError): self.session_expired.emit()
            self.error.setText(str(error)); return
        self.created.emit(result); self.accept()


class GridDelegate(QStyledItemDelegate):
    def __init__(self, images, theme, parent):
        super().__init__(parent); self.images, self.t = images, PALETTES[theme]

    def sizeHint(self, option, index):
        return QSize(264, 246)

    def paint(self, p, option, index):
        e, t = index.data(Qt.ItemDataRole.UserRole), self.t
        p.save(); p.setRenderHint(p.RenderHint.Antialiasing)
        r = option.rect.adjusted(6, 6, -6, -6)
        p.setPen(QColor(t['action' if option.state & QStyle.StateFlag.State_Selected else 'border']))
        p.setBrush(QColor(t['surface'])); p.drawRoundedRect(r, 8, 8)
        photo = QRectF(r.x()+8, r.y()+8, r.width()-16, 128)
        p.fillRect(photo, QColor(t['raised']))
        pix = self.images.get(e.thumbnail_url)
        if pix and not pix.isNull():
            p.drawPixmap(photo.toRect(), pix)
        else:
            p.setPen(QColor(t['muted'])); p.drawText(photo, Qt.AlignmentFlag.AlignCenter, 'Preview not available' if not e.thumbnail_url else 'Loading preview…')
        lines = [(camera_name(e), 'text'), (local_time(e.start_utc, e.timezone), 'muted'),
                 (provenance(e.completeness.boxes), 'action'),
                 ('No video copy remains' if e.completeness.expired else ai_status(e.completeness.ai), 'warning' if e.completeness.ai != 'real' else 'muted')]
        for i,(text, color) in enumerate(lines):
            p.setFont(QFont('Segoe UI', 9 if i == 0 else 8)); p.setPen(QColor(t[color]))
            p.drawText(QRectF(r.x()+12,r.y()+146+i*20,r.width()-24,20), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(text,Qt.TextElideMode.ElideRight,r.width()-24))
        p.restore()


class CollectionGrid(QWidget):
    changed = Signal()
    event_requested = Signal(int)
    session_expired = Signal()

    def __init__(self, backend, theme='dark'):
        super().__init__(); self.backend, self.collection = backend, None
        box = QVBoxLayout(self); box.setContentsMargins(0,0,0,0); box.setSpacing(12)
        top = QHBoxLayout(); self.title = label('', 'section'); top.addWidget(self.title,1)
        self.remove = button('Remove selected', self.remove_selected); self.remove.setEnabled(False); top.addWidget(self.remove)
        box.addLayout(top); self.message = label('', 'muted', True); box.addWidget(self.message)
        self.model = RowsModel([('Event',lambda e:e.camera+'\n'+e.summary)])
        self.images = {}; self.grid = QListView(); self.grid.setModel(self.model)
        self.grid.setViewMode(QListView.ViewMode.IconMode); self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static); self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.setUniformItemSizes(True); self.grid.setSpacing(4)
        self.grid.setItemDelegate(GridDelegate(self.images, theme, self.grid)); box.addWidget(self.grid,1)
        self.grid.selectionModel().selectionChanged.connect(lambda *_: self.remove.setEnabled(bool(self.grid.selectedIndexes())))
        self.grid.doubleClicked.connect(lambda index: self.event_requested.emit(index.data(Qt.ItemDataRole.UserRole).id))
        self.runner, self.writer = TaskRunner(self), TaskRunner(self)
        self.runner.finished.connect(self.loaded); self.writer.finished.connect(self.removed)
        self.generation = 0
        from .thumbnails import VisibleThumbnails
        self.thumbnail_cache = {}
        self.thumbnails = VisibleThumbnails(self.grid, backend, self.thumbnail_cache)
        self.thumbnails.loaded.connect(self.thumbnails_loaded)
        self.thumbnails.session_expired.connect(self.session_expired)

    def open(self, collection):
        self.thumbnails.schedule()
        self.collection = collection; self.generation += 1
        self.title.setText(collection.name); self.message.setText('Loading collection…'); self.model.replace([])
        self.request()

    def request(self):
        if self.runner.busy:
            return
        cid, self.pending = self.collection.id, self.generation
        def fetch():
            events, cursor = [], None
            while True:
                if closing.is_set(): return [], {}
                page = self.backend.collection_events(cid, cursor=cursor)
                events.extend(page.items)
                cursor = page.next_cursor
                if not cursor: break
            images = {}
            return events, images
        self.runner.start(fetch)

    def loaded(self, result, error):
        if self.pending != self.generation:
            self.request(); return
        if error:
            if isinstance(error, AuthError): self.session_expired.emit()
            self.message.setText(str(error)); return
        events, images = result
        self.images.clear()
        for url, data in images.items():
            pix = QPixmap(); pix.loadFromData(data); self.images[url] = pix
        self.model.replace(events)
        self.thumbnails.schedule()
        self.message.setText(f'{len(events)} events · {self.collection.description}' if events else 'This collection is empty. Add events from Review with c.')

    def thumbnails_loaded(self, images):
        for url, data in images.items():
            pix = QPixmap(); pix.loadFromData(data); self.images[url] = pix
        self.grid.viewport().update()

    def remove_selected(self):
        ids = [i.data(Qt.ItemDataRole.UserRole).id for i in self.grid.selectedIndexes()]
        if ids and not self.writer.busy:
            cid = self.collection.id
            self.remove.setEnabled(False)
            self.writer.start(lambda: self.backend.remove_collection_items(cid, ids))

    def removed(self, collection, error):
        if error:
            if isinstance(error, AuthError): self.session_expired.emit()
            self.message.setText(str(error)); self.remove.setEnabled(True); return
        if collection.id == self.collection.id:
            self.open(collection)
        self.changed.emit()
