"""Visible-only thumbnail access; offscreen rows never create access requests."""
from threading import Event
from PySide6.QtCore import QObject, QTimer, QEvent, Signal, Qt
from .workers import TaskRunner, closing
from .backend import BackendError, AuthError


class VisibleThumbnails(QObject):
    loaded = Signal(object)
    session_expired = Signal()

    def __init__(self, view, backend, cache):
        super().__init__(view)
        self.view, self.backend, self.cache = view, backend, cache
        self.cancel = Event()
        self.runner = TaskRunner(self)
        self.runner.finished.connect(self.completed)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.fetch)
        view.viewport().installEventFilter(self)
        view.verticalScrollBar().valueChanged.connect(self.schedule)
        view.horizontalScrollBar().valueChanged.connect(self.schedule)

    def schedule(self, *_):
        self.cancel.set()
        self.timer.start()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Hide, QEvent.Type.Show, QEvent.Type.Resize):
            self.schedule()
        return False

    def fetch(self):
        if self.runner.busy or not self.view.isVisible(): return
        self.cancel = Event()
        cancel, urls = self.cancel, set()
        model = self.view.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if self.view.visualRect(index).intersects(self.view.viewport().rect()):
                event = index.data(Qt.ItemDataRole.UserRole)
                if event.thumbnail_url and event.thumbnail_url not in self.cache:
                    urls.add(event.thumbnail_url)
        if not urls: return
        def fetch():
            images = {}
            for url in urls:
                if cancel.is_set() or closing.is_set(): break
                try:
                    images[url] = self.backend.media_bytes(url)
                except AuthError:
                    raise
                except BackendError:
                    images[url] = b''
            return images
        self.runner.start(fetch)

    def completed(self, images, error):
        if isinstance(error, AuthError):
            self.session_expired.emit()
            return
        if images:
            self.cache.update(images)
            self.loaded.emit(images)
        self.fetch()
