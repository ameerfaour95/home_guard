"""One owner per screen; Qt queued slots keep every widget update on the UI thread."""
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from .backend import BackendError, ServerError


class Signals(QObject):
    done = Signal(object, object)


class Job(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = Signals()

    def run(self):
        try:
            result, error = self.operation(), None
        except BackendError as exc:
            result, error = None, exc
        except Exception:
            result, error = None, ServerError()
        self.signals.done.emit(result, error)


class TaskRunner(QObject):
    finished = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job = None

    @property
    def busy(self):
        return self.job is not None

    def start(self, operation):
        if self.busy:
            return False
        self.job = Job(operation)
        self.job.signals.done.connect(self._done)
        QThreadPool.globalInstance().start(self.job)
        return True

    @Slot(object, object)
    def _done(self, result, error):
        self.job = None
        self.finished.emit(result, error)
