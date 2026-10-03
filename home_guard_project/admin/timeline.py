from datetime import timedelta, timezone
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QDateTime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QTableView,
    QHeaderView, QAbstractItemView, QStackedWidget, QDialog, QDateTimeEdit, QDialogButtonBox, QSizePolicy)
from .backend import AuthError, BackendError
from PySide6.QtGui import QPixmap
from .formatting import utcnow, local_time
from .event_logic import KINDS
from .timeline_model import TimelineModel, TimelineDelegate
from .workers import TaskRunner
from .review_controller import ReviewController
from .widgets.activity import DensityStrip
from .widgets.common import label, button, Skeleton, EmptyState


class TimelineScreen(QWidget):
    event_requested = Signal(int)
    session_expired = Signal()

    def __init__(self, backend, theme='dark'):
        super().__init__()
        self.backend = backend
        self.customer_id, self.zone, self.cursor = None, 'UTC', None
        self.start, self.end = self.now()-timedelta(hours=24), self.now()
        self.cell = None
        self.saved_filter = None
        self.generation = 0
        self.runner, self.density_runner = [TaskRunner(self) for _ in range(2)]
        self.review_runner = ReviewController(backend, self)
        self.runner.finished.connect(self.completed)
        self.density_runner.finished.connect(self.density_loaded)
        self.review_runner.finished.connect(self.review_done)
        self.thumbnail_runner = TaskRunner(self)
        self.thumbnail_runner.finished.connect(self.thumbnails_loaded)
        self.thumbnail_cache = {}
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(8)
        ranges = QHBoxLayout(); ranges.setSpacing(8)
        self.range_chips = {}
        for title, hours in [('6 h', 6), ('24 h', 24), ('7 d', 168), ('Custom', None)]:
            chip = button(title, lambda checked=False, h=hours: self.set_range(h))
            chip.setCheckable(True); chip.setChecked(hours == 24)
            ranges.addWidget(chip); self.range_chips[title] = chip
        self.range_text = label('', 'muted'); ranges.addWidget(self.range_text); ranges.addStretch()
        self.clear_cell = button('Clear hour filter', self.reset_cell, 'link'); self.clear_cell.hide(); ranges.addWidget(self.clear_cell)
        self.range_bar = QWidget(); self.range_bar.setLayout(ranges); layout.addWidget(self.range_bar)
        self.density = DensityStrip(theme); self.density.selected.connect(self.filter_cell); layout.addWidget(self.density)
        filters = QHBoxLayout(); filters.setSpacing(8)
        self.filters = {}
        choices = {'kind': [('All kinds', None)]+[(v, k) for k, v in KINDS.items()],
                   'camera': [('All cameras', None)],
                   'ai': [('All AI states', None)]+[(s.title(), s) for s in ('real', 'failed', 'fallback', 'none')],
                   'verdict': [('All owner verdicts', None), ('Confirmed', 'real'), ('False alarm', 'false_alarm'), ('Real, wrong decision', 'real_but_wrong')],
                   'reviewed': [('Any review', None), ('Unreviewed', False), ('Reviewed', True)],
                   'flagged': [('Any flag', None), ('Flagged', True), ('Unflagged', False)]}
        for key, values in choices.items():
            combo = QComboBox(); combo.setAccessibleName(key)
            for text, data in values:
                combo.addItem(text, data)
            combo.currentIndexChanged.connect(self.reload)
            self.filters[key] = combo; filters.addWidget(combo)
        self.search = QLineEdit(); self.search.setPlaceholderText('Search events…'); self.search.setMinimumWidth(120)
        self.debounce = QTimer(self); self.debounce.setSingleShot(True); self.debounce.setInterval(250); self.debounce.timeout.connect(self.reload)
        self.search.textChanged.connect(lambda: self.debounce.start())
        filters.addWidget(self.search, 1); self.filter_bar = QWidget(); self.filter_bar.setLayout(filters); layout.addWidget(self.filter_bar)
        self.banner = label('', 'error', True); self.banner.hide(); layout.addWidget(self.banner)
        self.stack = QStackedWidget(); layout.addWidget(self.stack, 1)
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.model = TimelineModel(self.now, self.zone)
        self.table = QTableView(); self.table.setModel(self.model); self.table.setItemDelegate(TimelineDelegate(theme, self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False); self.table.verticalHeader().hide(); self.table.verticalHeader().setDefaultSectionSize(64)
        for i, width in enumerate((108, 188, 140, 142, 164, 112, 72)):
            self.table.setColumnWidth(i, width)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.installEventFilter(self)
        self.table.doubleClicked.connect(lambda _: self.open_current())
        self.table.verticalScrollBar().valueChanged.connect(self.scroll_end)
        self.stack.addWidget(self.table)
        self.loading = Skeleton(theme); self.stack.addWidget(self.loading)
        self.empty = EmptyState('No events match this view', 'Choose another range or clear the filters.', eyebrow='TIMELINE')
        self.empty.action.setText('Clear filters'); self.empty.action.show(); self.empty.action.clicked.connect(self.clear_filters)
        self.stack.addWidget(self.empty)
        self.failure = EmptyState('Timeline could not be loaded', 'Check your connection and try again. Your filters are saved.', eyebrow='UNAVAILABLE')
        self.failure.action.show(); self.failure.action.clicked.connect(self.reload); self.stack.addWidget(self.failure)
        footer = QHBoxLayout(); self.count = label('Loading events…', 'muted'); footer.addWidget(self.count)
        footer.addStretch(); self.key_hint = label('j / k Select   ·   Enter Open   ·   r Review   ·   f Flag', 'muted'); footer.addWidget(self.key_hint)
        self.older = button('Load older', self.load_older); footer.addWidget(self.older); layout.addLayout(footer)

    def now(self):
        return getattr(self.backend, 'now', None) or utcnow()

    def open(self, customer_id=None, zone='UTC'):
        self.customer_id, self.zone, self.model.zone = customer_id, zone, zone
        self.cell = None
        self.clear_cell.hide()
        for combo in self.filters.values():
            combo.blockSignals(True); combo.setCurrentIndex(0); combo.blockSignals(False)
        self.search.blockSignals(True); self.search.clear(); self.search.blockSignals(False)
        self.set_range(24)

    def set_range(self, hours):
        if hours is None:
            dialog = QDialog(self); dialog.setWindowTitle('Custom range · UTC')
            form = QVBoxLayout(dialog)
            form.addWidget(label('Enter start and end in UTC. The timeline displays customer time.', 'muted', True))
            edits = []
            for value in (self.start, self.end):
                edit = QDateTimeEdit(QDateTime(value)); edit.setDisplayFormat('yyyy-MM-dd HH:mm'); edit.setCalendarPopup(True)
                edits.append(edit); form.addWidget(edit)
            actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            actions.accepted.connect(dialog.accept); actions.rejected.connect(dialog.reject); form.addWidget(actions)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.range_chips['Custom'].setChecked(False); return
            start, end = [e.dateTime().toPython().replace(tzinfo=timezone.utc) for e in edits]
            if not start < end or end-start > timedelta(days=31):
                self.banner.setText('Choose a range between one minute and 31 days.'); self.banner.show(); return
            self.start, self.end = start, end
        else:
            self.end = self.now(); self.start = self.end-timedelta(hours=hours)
        for title, chip in self.range_chips.items():
            chip.setChecked(title == ({6:'6 h', 24:'24 h', 168:'7 d'}.get(hours, 'Custom')))
        self.range_text.setText(f'{local_time(self.start, self.zone)[:6]} {local_time(self.start, self.zone)[13:18]} — {local_time(self.end, self.zone)[:6]} {local_time(self.end, self.zone)[13:18]}  ·  {self.zone}')
        self.cell = None; self.clear_cell.hide(); self.reload(); self.load_density()

    def query(self):
        start, end = self.start, self.end
        filters = {k: v.currentData() for k, v in self.filters.items() if v.currentData() is not None}
        if self.cell:
            camera, hour = self.cell
            filters['camera'] = camera; start, end = max(start, hour), min(end, hour+timedelta(hours=1))
        return dict(filters, filter=self.saved_filter, customer_id=self.customer_id, from_utc=start.isoformat(), to_utc=end.isoformat(), q=self.search.text().strip() or None)

    def reload(self, *_):
        self.generation += 1
        self.cursor = None
        self.request_page(False)

    def request_page(self, append):
        if self.runner.busy:
            return
        generation, query, cursor = self.generation, self.query(), self.cursor if append else None
        if not append:
            self.stack.setCurrentWidget(self.loading)
        self.older.setEnabled(False)
        self.pending = generation, append
        self.runner.start(lambda: self.backend.events(**query, cursor=cursor, limit=25))

    def completed(self, result, error):
        generation, append = self.pending
        if generation != self.generation:
            self.request_page(False); return
        self.older.setEnabled(True)
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit()
            self.banner.setText(str(error)); self.banner.show()
            self.stack.setCurrentWidget(self.table if append else self.failure); return
        self.banner.hide()
        page = result
        self.model.set_page(page.items, append, self.thumbnail_cache)
        self.cursor = page.next_cursor
        self.older.setEnabled(bool(self.cursor))
        self.older.setText('Load older' if self.cursor else 'End of range')
        self.stack.setCurrentWidget(self.table if self.model.rows else self.empty)
        self.count.setText(f'{len(self.model.rows)} events loaded' + (' · More available' if self.cursor else ' · All in view'))
        if self.model.rows and not append:
            self.table.setCurrentIndex(self.model.index(0, 0))
        self.load_thumbnails()

    def load_thumbnails(self):
        urls = {e.thumbnail_url for e in self.model.rows if e.thumbnail_url} - self.thumbnail_cache.keys()
        if not urls or self.thumbnail_runner.busy:
            return
        def fetch():
            images = {}
            for url in urls:
                try:
                    images[url] = self.backend.media_bytes(url)
                except BackendError:
                    images[url] = b''  # Keep the recording tile on failure.
            return images
        self.thumbnail_runner.start(fetch)

    def thumbnails_loaded(self, images, error):
        if error:
            return
        self.thumbnail_cache.update(images)
        # Bound decoded thumbnails to the current list; retain a small byte cache.
        current = {e.thumbnail_url for e in self.model.rows}
        for url in current & images.keys():
            pix = QPixmap(); pix.loadFromData(images[url]); self.model.images[url] = pix
        self.table.viewport().update()
        if len(self.thumbnail_cache) > 256:
            self.thumbnail_cache = {url: data for url, data in self.thumbnail_cache.items() if url in current}
        self.load_thumbnails()

    def load_density(self):
        if self.density_runner.busy:
            return
        key = self.customer_id, self.start, self.end
        self.density_pending = key
        self.density_runner.start(lambda: self.backend.density(customer_id=key[0], from_utc=key[1].isoformat(), to_utc=key[2].isoformat()))

    def density_loaded(self, events, error):
        if self.density_pending != (self.customer_id, self.start, self.end):
            self.load_density(); return
        if error:
            self.density.set_error()
            self.density.setToolTip('Activity could not be loaded. Change range to retry.'); self.density.update(); return
        self.density.set_density(events)
        combo = self.filters['camera']; selected = combo.currentData(); combo.blockSignals(True)
        combo.clear(); combo.addItem('All cameras', None)
        for camera in sorted({e.camera for e in events.rows}):
            combo.addItem(camera, camera)
        combo.setCurrentIndex(max(0, combo.findData(selected))); combo.blockSignals(False)

    def filter_cell(self, camera, hour):
        self.cell = camera, hour; self.clear_cell.setText(f'{camera} · {local_time(hour, self.zone)[13:18]}  ×')
        self.clear_cell.show(); self.reload()

    def reset_cell(self):
        self.cell = None; self.clear_cell.hide(); self.reload()

    def clear_filters(self):
        for combo in self.filters.values():
            combo.blockSignals(True); combo.setCurrentIndex(0); combo.blockSignals(False)
        self.search.clear(); self.reset_cell()

    def load_older(self):
        if self.cursor:
            self.request_page(True)

    def scroll_end(self, value):
        bar = self.table.verticalScrollBar()
        if bar.maximum() > 0 and value >= bar.maximum()-1:
            self.load_older()

    def current(self):
        index = self.table.currentIndex()
        return self.model.rows[index.row()] if index.isValid() else None

    def open_current(self):
        if self.current():
            self.event_requested.emit(self.current().id)

    def review_current(self, key):
        event = self.current()
        if event:
            self.review_runner.submit(event, key)

    def review_done(self, event, error):
        if error:
            self.banner.setText(str(error)); self.banner.show()
            if isinstance(error, AuthError):
                self.session_expired.emit()
        else:
            self.model.update_review(event)
            if self.filters['reviewed'].currentData() is not None or self.filters['flagged'].currentData() is not None:
                self.reload()

    def eventFilter(self, watched, event):
        if watched is self.table and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_J, Qt.Key.Key_K):
                row = self.table.currentIndex().row()+(1 if key == Qt.Key.Key_J else -1)
                self.table.setCurrentIndex(self.model.index(max(0, min(len(self.model.rows)-1, row)), 0)); return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.open_current(); return True
            if key in (Qt.Key.Key_R, Qt.Key.Key_F):
                self.review_current('reviewed' if key == Qt.Key.Key_R else 'flagged'); return True
        return super().eventFilter(watched, event)
