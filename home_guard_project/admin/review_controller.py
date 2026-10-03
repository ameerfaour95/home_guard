"""Serialize saves; retain the last desired value per field and event."""
from dataclasses import replace
from PySide6.QtCore import QObject, Signal
from .workers import TaskRunner
from .backend import AuthError


class ReviewController(QObject):
    finished = Signal(object, object)
    optimistic = Signal(object)

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.runner = TaskRunner(self)
        self.runner.finished.connect(self._done)
        self.queue = {}
        self.desired = {}
        self.active_before = None
        self.active_changes = {}

    @property
    def busy(self):
        return self.runner.busy or bool(self.queue)

    def submit(self, event, key, value=None):
        current = self.desired.get(event.id, event)
        value = not getattr(current, key) if value is None else value
        before, changes = self.queue.get(event.id, (replace(current), {}))
        changes[key] = value
        self.queue[event.id] = before, changes
        self.desired[event.id] = replace(current, **{key: value})
        self.optimistic.emit(self.desired[event.id])
        self._start()

    def _start(self):
        if self.runner.busy or not self.queue:
            return
        eid = next(iter(self.queue))
        self.active_before, self.active_changes = self.queue.pop(eid)
        changes = dict(self.active_changes)
        self.runner.start(lambda: self.backend.review(eid, **changes))

    def _done(self, result, error):
        before = self.active_before
        confirmed = before if error else result
        pending = self.queue.get(before.id)
        if pending:
            self.queue[before.id] = replace(confirmed), pending[1]
            displayed = replace(confirmed, **pending[1])
            self.desired[before.id] = displayed
        else:
            displayed = confirmed
            self.desired.pop(before.id, None)
        self.optimistic.emit(displayed)
        if isinstance(error, AuthError):
            self.queue.clear()
            self.desired.clear()
        self.finished.emit(confirmed, error)
        self._start()
