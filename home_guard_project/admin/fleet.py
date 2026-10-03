from datetime import timedelta
import time
from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTableView, QHeaderView, QStackedWidget, QAbstractItemView
from .backend import OfflineError, AuthError, ForbiddenError
from .demo_backend import DemoBackend
from .formatting import utcnow
from .fleet_model import FleetModel, FleetDelegate
from .widgets.common import label, button, EmptyState, Skeleton
from .widgets.device_detail import DeviceDetail
from .workers import TaskRunner
from .widgets.activity import DensityStrip


class FleetScreen(QWidget):
    customer_requested = Signal(int)
    loaded = Signal(object, object)
    session_expired = Signal()

    def __init__(self, backend, theme='dark'):
        super().__init__()
        self.backend, self.theme = backend, theme
        self.snapshot = None
        self.received_at = None
        self.customers = []
        self.selected_id = None
        self.runner = TaskRunner(self)
        self.runner.finished.connect(self.completed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 24)
        layout.setSpacing(24)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(8)
        titles.addWidget(label('Fleet', 'title'))
        self.summary = label('Your customer boxes, in one place.', 'muted')
        titles.addWidget(self.summary)
        heading.addLayout(titles, 1)
        self.updated = label('Connecting…', 'muted')
        heading.addWidget(self.updated)
        self.refresh_button = button('Refresh', self.refresh)
        heading.addWidget(self.refresh_button)
        layout.addLayout(heading)
        self.activity = DensityStrip(theme, fleet=True)
        layout.addWidget(self.activity)
        self.activity_runner = TaskRunner(self)
        self.activity_runner.finished.connect(self.activity_loaded)
        filters = QHBoxLayout()
        filters.setSpacing(8)
        self.chips = {}
        for key, title in [('all', 'All boxes'), ('critical', 'Critical'), ('warning', 'Warning'), ('offline', 'Offline'), ('healthy', 'Healthy'), ('unknown', 'Unknown')]:
            chip = button(title, lambda checked=False, v=key: self.filter(verdict=v))
            chip.setCheckable(True)
            chip.setChecked(key == 'all')
            self.chips[key] = chip
            filters.addWidget(chip)
        filters.addStretch()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter fleet…    /')
        self.search.setAccessibleName('Filter fleet')
        self.search.setMinimumWidth(176)
        self.search.textChanged.connect(lambda text: self.filter(query=text))
        filters.addWidget(self.search)
        layout.addLayout(filters)
        self.banner = label('', 'error', True)
        self.banner.hide()
        layout.addWidget(self.banner)
        self.stack = QStackedWidget()
        self.content = QWidget()
        content_layout = QHBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)
        self.model = FleetModel(self.now, theme)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(FleetDelegate(theme, self.table))
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.setMouseTracking(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(72)
        self.table.horizontalHeader().setMinimumSectionSize(64)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.column_widths = [176, 248, 88, 76, 96, 100, 76, 128]
        for col, width in enumerate(self.column_widths):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setMinimumSectionSize(76)
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
        self.table.selectionModel().currentRowChanged.connect(self.row_changed)
        self.table.doubleClicked.connect(lambda index: self.open_current())
        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.setInterval(450)
        self.hover_row = -1
        self.table.entered.connect(self.hover_entered)
        self.hover_timer.timeout.connect(self.preview_hover)
        self.detail = DeviceDetail(theme)
        self.detail.hide()
        self.detail.open_customer.connect(self.customer_requested)
        self.detail.dismissed.connect(self.detail.hide)
        content_layout.addWidget(self.table, 1)
        content_layout.addWidget(self.detail)
        self.stack.addWidget(self.content)
        self.loading = Skeleton(theme)
        self.stack.addWidget(self.loading)
        self.empty = EmptyState('No boxes to show', 'Customer boxes will appear here when they are enrolled in Home Guard Cloud.')
        self.stack.addWidget(self.empty)
        self.no_results = EmptyState('No matching boxes', 'Try another customer, site or health filter.')
        self.no_results.action.setText('Clear filters')
        self.no_results.action.show()
        self.no_results.action.clicked.connect(self.clear_filters)
        self.stack.addWidget(self.no_results)
        self.failure = EmptyState("Can't reach Home Guard Cloud", 'Check your connection and try again. Customer boxes continue running independently.', eyebrow='CONNECTION UNAVAILABLE')
        self.failure.action.show()
        self.failure.action.clicked.connect(self.refresh)
        self.stack.addWidget(self.failure)
        self.server_failure = EmptyState('Fleet is unavailable', 'Home Guard Cloud could not load the fleet. Please try again.', eyebrow='REQUEST FAILED')
        self.server_failure.action.show()
        self.server_failure.action.clicked.connect(self.refresh)
        self.stack.addWidget(self.server_failure)
        layout.addWidget(self.stack, 1)
        footer = QHBoxLayout()
        self.count = label('Loading fleet…', 'muted')
        footer.addWidget(self.count)
        footer.addStretch()
        footer.addWidget(label('¹ Reporting / total     ·     ↑ ↓ Select     Enter Open customer', 'muted'))
        layout.addLayout(footer)
        shortcut = QShortcut(QKeySequence('/'), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.search.setFocus)
        self.timer = QTimer(self)
        self.timer.setInterval(30_000)
        self.timer.timeout.connect(self.refresh)
        self.forbidden = False
        self.age_timer = QTimer(self)
        self.age_timer.setInterval(1000)
        self.age_timer.timeout.connect(self.update_age)
        self.age_timer.start()
        self.refresh()

    def now(self):
        if isinstance(self.backend, DemoBackend) and self.snapshot:
            return self.snapshot.generated_utc + timedelta(seconds=time.monotonic() - self.received_at)
        return utcnow()

    def refresh(self):
        if self.runner.busy:
            return
        self.refresh_button.setEnabled(False)
        if self.snapshot is None:
            self.stack.setCurrentWidget(self.loading)
        def fetch():
            return self.backend.fleet(), self.backend.customers()
        self.runner.start(fetch)

    def completed(self, result, error):
        self.refresh_button.setEnabled(True)
        if error:
            self.show_error(error)
            return
        snapshot, customers = result
        self.customers = customers
        self.model.timezones = {c.id: c.timezone for c in customers}
        self.snapshot, self.received_at = snapshot, time.monotonic()
        self.banner.hide()
        scroll = self.table.verticalScrollBar().value()
        horizontal = self.table.horizontalScrollBar().value()
        selected = self.selected_id
        self.model.replace(snapshot.devices)
        self.restore_selection(selected)
        self.table.verticalScrollBar().setValue(scroll)
        self.table.horizontalScrollBar().setValue(horizontal)
        counts = {v: sum(d.verdict == v for d in snapshot.devices) for v in self.chips}
        self.summary.setText(f'{len(snapshot.devices)} boxes    /    {counts["healthy"]} healthy    /    {counts["critical"]+counts["warning"]} need attention    /    {counts["offline"]} offline' + (f'    /    {counts["unknown"]} unknown' if counts['unknown'] else ''))
        self.update_view()
        self.update_age()
        self.loaded.emit(snapshot, customers)
        end = snapshot.generated_utc
        self.activity_runner.start(lambda: self.backend.activity(hours=24))

    def activity_loaded(self, result, error):
        if error:
            self.activity.setToolTip('Activity could not be refreshed. Try Refresh.')
            if not self.activity.hours: self.activity.set_error()
            return
        self.activity.set_density(result)

    def show_error(self, error):
        if isinstance(error, ForbiddenError):
            self.forbidden = True
            self.timer.stop()
        if isinstance(error, AuthError):
            self.timer.stop()
            self.session_expired.emit()
            return
        self.updated.setText('Offline' if isinstance(error, OfflineError) else 'Update failed')
        if self.snapshot:
            self.banner.setText(str(error) + ' · Showing the last successful snapshot. Retrying every 30 seconds.')
            self.banner.show()
        else:
            self.banner.hide()
            self.summary.setText('Your customer boxes, in one place.')
            self.stack.setCurrentWidget(self.failure if isinstance(error, OfflineError) else self.server_failure)
            self.count.setText('No fleet data loaded')

    def showEvent(self, event):
        super().showEvent(event)
        if not self.forbidden:
            self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def update_age(self):
        if self.snapshot and self.received_at and not self.banner.isVisible():
            self.updated.setText(f'Updated {int(time.monotonic()-self.received_at)} s ago')

    def update_view(self):
        self.stack.setCurrentWidget(self.content if self.model.rows else self.no_results if self.model.devices else self.empty)
        self.count.setText(f'{len(self.model.rows)} of {len(self.model.devices)} boxes   ·   Worst first' + ('   ·   Demo snapshot' if isinstance(self.backend, DemoBackend) else ''))

    def filter(self, verdict=None, query=None):
        selected = self.selected_id
        self.model.apply_filter(verdict, query)
        for key, chip in self.chips.items():
            chip.setChecked(key == self.model.verdict)
        self.restore_selection(selected)
        if self.snapshot is not None:
            self.update_view()

    def clear_filters(self):
        self.search.clear()
        self.filter(verdict='all')

    def restore_selection(self, device_id):
        self.selected_id = None
        for row, device in enumerate(self.model.rows):
            if device.device_id == device_id:
                self.table.setCurrentIndex(self.model.index(row, 0))
                return
        self.detail.hide()

    def row_changed(self, current, previous):
        if current.isValid():
            device = self.model.rows[current.row()]
            self.selected_id = device.device_id
            self.show_detail(device)

    def show_detail(self, device):
        self.detail.show_device(device, self.model.timezones.get(device.customer_id, 'UTC'))
        self.detail.show()

    def hover_entered(self, index):
        self.hover_row = index.row()
        if not self.selected_id:
            self.hover_timer.start()

    def preview_hover(self):
        if self.table.underMouse() and not self.selected_id and 0 <= self.hover_row < len(self.model.rows):
            self.show_detail(self.model.rows[self.hover_row])

    def open_current(self):
        index = self.table.currentIndex()
        if index.isValid():
            self.customer_requested.emit(self.model.rows[index.row()].customer_id)

    def eventFilter(self, watched, event):
        if watched is self.table.viewport() and event.type() == QEvent.Type.Resize:
            # Preserve legible columns in compact layouts; the view scrolls
            # horizontally when the detail panel takes the remaining space.
            extra = max(0, self.table.viewport().width() - sum(self.column_widths))
            reason_extra = min(extra, 312)
            remainder = extra-reason_extra
            for col, base in enumerate(self.column_widths):
                width = base + (reason_extra if col == 1 else remainder//7)
                if col == 7:
                    width += remainder % 7
                if self.table.columnWidth(col) != width:
                    self.table.setColumnWidth(col, width)
        if watched is self.table and event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.open_current()
            return True
        return super().eventFilter(watched, event)
