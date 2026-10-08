from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QDate
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QTableView,
    QHeaderView, QAbstractItemView, QStackedWidget, QDateEdit, QSizePolicy, QScrollArea, QCheckBox)
from .backend import AuthError, BackendError
from PySide6.QtGui import QPixmap
from .formatting import utcnow, local_time, camera_name, remember_names
from .event_logic import KINDS, VERDICTS
from .timeline_model import TimelineModel, TimelineDelegate
from .workers import TaskRunner
from .review_controller import ReviewController
from .widgets.activity import DensityStrip
from .widgets.common import label, button, Skeleton, EmptyState


RANGES = {'6 h': 6, '24 h': 24, '7 d': 168, '30 d': 720}


class TimelineScreen(QWidget):
    event_requested = Signal(int)
    session_expired = Signal()

    def __init__(self, backend, theme='dark', role='admin'):
        super().__init__()
        self.backend, self.role = backend, role
        self.customer_id, self.zone, self.cursor = None, 'UTC', None
        self.start, self.end = self.now()-timedelta(hours=24), self.now()
        self.cell = None
        self.cameras = None  # the house's cameras (CameraOut) once known; None: every camera the index has seen
        self.density_result = None
        self.saved_filter = None
        self.generation = 0
        self.runner, self.density_runner = [TaskRunner(self) for _ in range(2)]
        self.sessions_runner = TaskRunner(self); self.sessions_runner.finished.connect(self.sessions_loaded)
        self.review_runner = ReviewController(backend, self)
        self.runner.finished.connect(self.completed)
        self.density_runner.finished.connect(self.density_loaded)
        self.review_runner.finished.connect(self.review_done)
        self.thumbnail_cache = {}
        self.density_rows = 8  # camera rows the strip shows before it scrolls
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(8)
        ranges = QHBoxLayout(); ranges.setSpacing(8)
        self.range_chips = {}
        for title, hours in RANGES.items():
            chip = button(title, lambda checked=False, h=hours: self.set_range(h))
            chip.setCheckable(True); chip.setChecked(hours == 24)
            ranges.addWidget(chip); self.range_chips[title] = chip
        # any whole days in the house's time zone, up to 31 days
        self.date_from, self.date_to = QDateEdit(), QDateEdit()
        for text, edit in (('From', self.date_from), ('to', self.date_to)):
            edit.setCalendarPopup(True); edit.setDisplayFormat('dd MMM yyyy'); edit.setAccessibleName(f'{text} date')
            edit.dateChanged.connect(lambda _: self.set_days())
            ranges.addSpacing(4 if text == 'to' else 12); ranges.addWidget(label(text, 'muted')); ranges.addWidget(edit)
        self.range_text = label('', 'muted'); ranges.addWidget(self.range_text, 1); ranges.addStretch()
        self.range_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)  # informational: may clip
        self.clear_cell = button('Clear hour filter', self.reset_cell, 'link'); self.clear_cell.hide(); ranges.addWidget(self.clear_cell)
        self.range_bar = QWidget(); self.range_bar.setLayout(ranges); layout.addWidget(self.range_bar)
        self.density = DensityStrip(theme); self.density.selected.connect(self.filter_cell)
        # A house with many cameras would push the event list off a 768-pixel screen: show a few rows, scroll the rest.
        self.density_scroll = QScrollArea(); self.density_scroll.setWidget(self.density); self.density_scroll.setWidgetResizable(True)
        self.density_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fit_density(); self.density_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(self.density_scroll)
        filters = QHBoxLayout(); filters.setSpacing(8)
        self.filters = {}
        choices = {'kind': [('All kinds', None)]+[(v, k) for k, v in KINDS.items()],
                   'camera': [('All cameras', None)],
                   'ai': [('All AI states', None)]+[(s.title(), s) for s in ('real', 'failed', 'fallback', 'none')],
                   'verdict': [('All owner answers', None)]+[(v, k) for k, v in VERDICTS.items()],
                   'reviewed': [('Any review', None), ('Unreviewed', False), ('Reviewed', True)],
                   'flagged': [('Any flag', None), ('Flagged', True), ('Unflagged', False)],
                   # the baseline in shadow mode records what it would do: these clips it would have raised
                   'would_raise': [('Any decision', None), ('Would raise: rare for this camera', True)]}
        for key, values in choices.items():
            combo = QComboBox(); combo.setAccessibleName(key)
            combo.setMinimumWidth(96)  # sized to its text, but may narrow so the row fits the 1200-pixel minimum window
            for text, data in values:
                combo.addItem(text, data)
            combo.currentIndexChanged.connect(self.reload)
            self.filters[key] = combo; filters.addWidget(combo)
        self.group_toggle = QCheckBox('Group by event'); self.group_toggle.setChecked(True)
        self.group_toggle.setToolTip("One row per event (the box's session at a camera); click it to see its clips")
        self.group_toggle.toggled.connect(lambda _: self.apply_groups())
        ranges.insertWidget(ranges.indexOf(self.range_text), self.group_toggle)  # beside the range: the filter row is full
        self.retired_toggle = QCheckBox('Retired cameras'); self.retired_toggle.hide()
        self.retired_toggle.setToolTip('Also list cameras this house no longer has (old ids after a rename, removed cameras)')
        self.retired_toggle.toggled.connect(lambda _: self.apply_density())
        ranges.insertWidget(ranges.indexOf(self.range_text), self.retired_toggle)
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
        self.table.clicked.connect(lambda index: index.column() in (0, 1) and self.toggle_group(index.row()))
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
        self.show_dates()

    def now(self):
        return getattr(self.backend, 'now', None) or utcnow()

    def open(self, customer_id=None, zone='UTC'):
        self.customer_id, self.zone, self.model.zone = customer_id, zone, zone
        self.cameras = None; self.retired_toggle.hide()
        self.retired_toggle.blockSignals(True); self.retired_toggle.setChecked(False); self.retired_toggle.blockSignals(False)
        self.cell = None
        self.clear_cell.hide()
        for combo in self.filters.values():
            combo.blockSignals(True); combo.setCurrentIndex(0); combo.blockSignals(False)
        self.search.blockSignals(True); self.search.clear(); self.search.blockSignals(False)
        self.set_range(24)

    def tz(self):
        try:
            return ZoneInfo(self.zone)
        except (ZoneInfoNotFoundError, ValueError):
            return timezone.utc

    def set_range(self, hours):
        self.end = self.now(); self.start = self.end-timedelta(hours=hours)
        self.range_changed(hours)

    def set_days(self, first=None, last=None):
        """Whole days in the house's zone, from the date pickers (or the dates given)."""
        first = first or self.date_from.date().toPython()
        last = last or self.date_to.date().toPython()
        if last < first:
            first, last = last, first
        if (last-first).days >= 31:
            self.banner.setText('Choose a range of at most 31 days.'); self.banner.show(); self.show_dates(); return
        self.banner.hide()
        self.start = datetime.combine(first, time(0), self.tz()).astimezone(timezone.utc)
        self.end = datetime.combine(last+timedelta(days=1), time(0), self.tz()).astimezone(timezone.utc)
        self.range_changed(None)

    def show_dates(self):
        for edit, value in ((self.date_from, self.start), (self.date_to, self.end-timedelta(microseconds=1))):
            edit.blockSignals(True); edit.setDate(QDate(value.astimezone(self.tz()).date())); edit.blockSignals(False)

    def range_changed(self, hours):
        for title, chip in self.range_chips.items():
            chip.setChecked(RANGES[title] == hours)
        self.show_dates()
        self.range_text.setText(f'{local_time(self.start, self.zone)[:6]} {local_time(self.start, self.zone)[13:18]} — {local_time(self.end, self.zone)[:6]} {local_time(self.end, self.zone)[13:18]}  ·  {self.zone}')
        self.cell = None; self.clear_cell.hide(); self.reload(); self.load_density()

    def query(self):
        start, end = self.start, self.end
        filters = {k: v.currentData() for k, v in self.filters.items() if v.currentData() is not None}
        if self.cell:
            camera, hour = self.cell
            filters['camera'] = camera; start, end = max(start, hour), min(end, hour+timedelta(hours=1))
        if self.role != 'labeler' and '/' in (filters.get('camera') or ''):
            filters['site'], filters['camera'] = filters['camera'].split('/', 1)
        query = dict(filters, filter=self.saved_filter, from_utc=start.isoformat(), to_utc=end.isoformat())
        if self.role != 'labeler':
            query.update(customer_id=self.customer_id, q=self.search.text().strip() or None)
        return query

    def reload(self, *_):
        if hasattr(self, 'thumbnails'): self.thumbnails.schedule()
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
        self.apply_groups()
        if self.model.rows and not append:
            self.table.setCurrentIndex(self.model.index(0, 0))
        self.load_thumbnails()
        self.load_sessions()

    # ------------------------------------------------------------ events (box sessions)

    def apply_groups(self):
        self.model.grouping = self.group_toggle.isChecked()
        header, n = self.table.verticalHeader(), len(self.model.rows)
        for visual in range(n):  # back to time order (model order), then an open event's clips right under it
            if header.visualIndex(visual) != visual:
                header.moveSection(header.visualIndex(visual), visual)
        for key in self.model.expanded if self.model.grouping else ():
            rows = self.model.members.get(key, [])
            target = header.visualIndex(rows[0]) + 1 if rows else 0
            for row in rows[1:]:
                header.moveSection(header.visualIndex(row), target if header.visualIndex(row) > target else target - 1)
                target = header.visualIndex(row) + 1
        for row in range(n):
            self.table.setRowHidden(row, self.model.hidden(row))
        if self.model.rows:
            self.model.dataChanged.emit(self.model.index(0, 1), self.model.index(len(self.model.rows)-1, 3))

    def toggle_group(self, row, expand=None):
        if not self.model.lead(row):
            return
        key = self.model.key(row)
        if expand is None:
            expand = key not in self.model.expanded
        self.model.expanded = (self.model.expanded | {key}) if expand else (self.model.expanded - {key})
        self.apply_groups()

    def reveal(self, row):
        """Open the event a clip row belongs to, so selecting it never lands on a hidden row."""
        if 0 <= row < len(self.model.rows) and self.model.hidden(row):
            self.toggle_group(self.model.members[self.model.key(row)][0], True)

    def next_row(self, row, step):
        """The next visible row on screen from *row* (step +1 / -1), or *row* itself at the end. Rows are moved on
        screen (an open event's clips sit under it), so this walks the view's order, not the model's."""
        header, n = self.table.verticalHeader(), len(self.model.rows)
        visual = header.visualIndex(row) + step if 0 <= row < n else (0 if step > 0 else n - 1)
        while 0 <= visual < n and self.table.isRowHidden(header.logicalIndex(visual)):
            visual += step
        return header.logicalIndex(visual) if 0 <= visual < n else row

    def load_sessions(self):
        wanted = sorted({e.session_id for e in self.model.rows if e.session_id}
                        - {sid for _, sid in self.model.sessions})
        if wanted and hasattr(self.backend, 'event_sessions'):
            self.sessions_runner.start(lambda: self.backend.event_sessions(wanted))

    def sessions_loaded(self, sessions, error):
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit()
            return  # an older server: groups count the loaded clips only
        self.model.sessions.update({(s.site, s.session_id): s for s in sessions})
        self.apply_groups()
        self.load_sessions()  # ids that arrived with a later page

    def load_thumbnails(self):
        if not hasattr(self, 'thumbnails'):
            from .thumbnails import VisibleThumbnails
            self.thumbnails = VisibleThumbnails(self.table, self.backend, self.thumbnail_cache)
            self.thumbnail_runner = self.thumbnails.runner
            self.thumbnails.loaded.connect(lambda images: self.thumbnails_loaded(images, None))
            self.thumbnails.session_expired.connect(self.session_expired)
        self.thumbnails.schedule()

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
            for url in list(self.thumbnail_cache):
                if url not in current: self.thumbnail_cache.pop(url)

    def load_density(self):
        if self.density_runner.busy:
            return
        key = self.customer_id, self.start, self.end
        self.density_pending = key
        query = dict(from_utc=key[1].isoformat(), to_utc=key[2].isoformat())
        if self.role != 'labeler': query['customer_id'] = key[0]
        self.density_runner.start(lambda: self.backend.density(**query))

    def density_loaded(self, events, error):
        if self.density_pending != (self.customer_id, self.start, self.end):
            self.load_density(); return
        if error:
            self.density.set_error()
            self.density.setToolTip('Activity could not be loaded. Change range to retry.'); self.density.update(); return
        self.density_result = events
        self.apply_density()

    def set_cameras(self, cameras):
        """The house's camera list: the strip and the camera filter then show current cameras; retired ones only
        behind the toggle."""
        self.cameras = cameras
        # owner names for current cameras; retired ids read 'Camera 6 (old)' so they never pass for the current one
        remember_names({c.camera: c.name if c.current else f'{c.name} (off by the owner)' if not c.enabled else f'{c.name} (old)'
                        for c in cameras or () if c.owner_named or not c.current})
        self.retired_toggle.setVisible(any(not c.current for c in cameras or ()))
        self.apply_density()

    def retired_ids(self):
        return {c.camera for c in self.cameras or () if not c.current}

    def shown(self, camera):
        # density rows are '<site>/<camera>' when the customer has more than one box
        return self.retired_toggle.isChecked() or camera.split('/', 1)[-1] not in self.retired_ids()

    def apply_density(self):
        events = self.density_result
        if events is None:
            return
        from dataclasses import replace
        self.density.set_density(replace(events, rows=[r for r in events.rows if self.shown(r.camera)]))
        self.fit_density()
        retired = self.retired_ids()
        combo = self.filters['camera']; selected = combo.currentData(); combo.blockSignals(True)
        combo.clear(); combo.addItem('All cameras', None)
        for camera in sorted({e.camera for e in events.rows if self.shown(e.camera)},
                             key=lambda c: (c.split('/', 1)[-1] in retired, camera_name(c))):
            combo.addItem(camera_name(camera), camera)
            combo.setItemData(combo.count()-1, camera_name(camera), Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(max(0, combo.findData(selected))); combo.blockSignals(False)
        if selected is not None and combo.currentData() != selected:
            self.reload()

    def fit_density(self):
        # the grid's own height up to density_rows camera rows; more scroll
        self.density_scroll.setFixedHeight(min(self.density.minimumHeight(), 24+22*self.density_rows) + 4)

    def filter_cell(self, camera, hour):
        self.cell = camera, hour; self.clear_cell.setText(f'{camera_name(camera)} · {local_time(hour, self.zone)[13:18]}  ×')
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
                row = self.next_row(self.table.currentIndex().row(), 1 if key == Qt.Key.Key_J else -1)
                self.table.setCurrentIndex(self.model.index(max(0, min(len(self.model.rows)-1, row)), 0)); return True
            if key in (Qt.Key.Key_Right, Qt.Key.Key_Left) and self.model.lead(self.table.currentIndex().row()):
                self.toggle_group(self.table.currentIndex().row(), key == Qt.Key.Key_Right); return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.open_current(); return True
            if key in (Qt.Key.Key_R, Qt.Key.Key_F):
                self.review_current('reviewed' if key == Qt.Key.Key_R else 'flagged'); return True
        return super().eventFilter(watched, event)
