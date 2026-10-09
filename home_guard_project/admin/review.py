from .formatting import camera_name
from dataclasses import replace
from PySide6.QtCore import Qt, Signal, QEvent, QTimer, QRectF, QSize, QModelIndex
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QComboBox,
    QApplication, QLineEdit, QPlainTextEdit, QAbstractSpinBox, QSizePolicy)
from .timeline import TimelineScreen
from .timeline_model import TimelineDelegate, status_chip_width, MEMBER
from .event_view import EventView
from .event_logic import VERDICTS, kind_label, two_lines
from .formatting import local_time
from .workers import TaskRunner
from .review_controller import ReviewController
from .backend import AuthError
from .widgets.common import label, button
from .widgets.icons import draw_icon


class ReviewDelegate(TimelineDelegate):
    """The timeline's thumbnail, typography and evidence palette in a compact row."""
    def paint(self, painter, option, index):
        if index.column() != 1:
            super().paint(painter, option, index)
            if index.column() == 0:
                e,t,p = index.data(Qt.ItemDataRole.UserRole),self.t,painter
                p.save()
                for i,(name,present) in enumerate([('video',e.completeness.video),('boxes',e.completeness.boxes != 'none'),('ai',e.completeness.ai == 'real')]):
                    # beside the annotation chip, which the timeline paints at the cell's bottom left
                    x = option.rect.x()+10+status_chip_width((e.annotation_status or 'new').title())+6+i*20
                    draw_icon(p,name,QRectF(x,option.rect.bottom()-25,15,15),t['action' if present else 'border'])
                p.restore()
            return
        from PySide6.QtWidgets import QStyle
        e, p, t = index.data(Qt.ItemDataRole.UserRole), painter, self.t
        p.save()
        p.fillRect(option.rect, QColor(t['raised' if option.state & QStyle.StateFlag.State_Selected else 'surface']))
        p.setPen(QColor(t['border'])); p.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        rect = option.rect.adjusted(8, 10, -10, -8)
        model, row = index.model(), index.row()
        head = (model.session_line(row) if model.lead(row) else
                (MEMBER if model.member(row) else '') + f'{local_time(e.start_utc, e.timezone)[13:18]}  ·  {camera_name(e)}')
        outcome = e.outcome or (f'{kind_label(e)} · '+{'real':'AI answer saved','failed':'AI failed','fallback':'Fallback AI','none':'No AI answer'}.get(e.completeness.ai, 'Unknown AI state'))
        p.setFont(QFont('Segoe UI', 8))
        wrapped = two_lines(p.fontMetrics(), outcome, rect.width())  # the outcome reads in full: up to two lines
        lines = [head, e.summary or 'No summary saved', *wrapped,
                 ('Reviewed' if e.reviewed else 'Unreviewed') + ('  ·  Flagged' if e.flagged else '') +
                 ('  ·  '+', '.join(VERDICTS.get(v,v.replace('_',' ').capitalize()) for v in e.owner_verdicts) if e.owner_verdicts else '')]
        last = len(lines) - 1
        for i, text in enumerate(lines):
            p.setFont(QFont('Segoe UI', 9 if i < 2 else 8))
            p.setPen(QColor(t['text' if i == 0 else 'action' if i == last and e.reviewed else 'muted']))
            p.drawText(QRectF(rect.x(), rect.y()+i*20, rect.width(), 20), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, rect.width()))
        p.restore()


class ReviewScreen(QWidget):
    session_expired = Signal()
    review_changed = Signal()

    def __init__(self, backend, theme='dark', role='admin'):
        super().__init__()
        self.backend, self.role = backend, role
        self.active_id = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.saved_filters, self.total, self.done, self.pending_next = [], 0, 0, None
        self.last_collection = None
        self.mutation = ReviewController(backend, self); self.mutation.finished.connect(self.mutation_done)
        self.metadata = TaskRunner(self); self.metadata.finished.connect(self.metadata_loaded)
        layout = QVBoxLayout(self); layout.setContentsMargins(24, 20, 24, 16); layout.setSpacing(12)
        top = QHBoxLayout(); top.addWidget(label('Activity', 'title')); top.addStretch()
        self.progress = label('Loading review count…', 'muted'); top.addWidget(self.progress)
        layout.addLayout(top)
        self.toast = label('', 'badge', True); self.toast.hide(); layout.addWidget(self.toast)
        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.setInterval(7000)
        self.toast_timer.timeout.connect(self.toast.hide)
        body = QHBoxLayout(); body.setSpacing(16); layout.addLayout(body, 1)
        self.sidebar = QWidget(); self.sidebar.setFixedWidth(184)
        filters = QVBoxLayout(self.sidebar); filters.setContentsMargins(0, 0, 0, 0); filters.setSpacing(6)
        filters.addWidget(label('SAVED VIEWS', 'eyebrow'))
        self.saved = QComboBox(); self.saved.addItem('All events', None); filters.addWidget(self.saved)
        self.saved.currentIndexChanged.connect(self.select_filter)
        filters.addSpacing(14); filters.addWidget(label('REFINE EVENTS', 'eyebrow'))
        self.timeline = TimelineScreen(backend, theme, role)
        self.mutation.optimistic.connect(self.timeline.model.update_review)
        self.mutation.optimistic.connect(self.sync_detail)
        # each combo's first item names it ('All kinds', 'All cameras', ...): no separate captions, so the whole
        # sidebar fits a 768-pixel screen
        for key, combo in self.timeline.filters.items():
            combo.setToolTip({'ai': 'AI state', 'verdict': 'Owner answer',
                              'decision': "The box's decision and its AI call"}.get(key, key.title()))
            if key == 'camera' and role == 'labeler':
                combo.hide(); continue  # a labeler's cameras are pseudonyms
            filters.addWidget(combo)
            if key == 'camera':
                filters.addWidget(self.timeline.retired_toggle)
        self.timeline.filters['reviewed'].blockSignals(True)
        self.timeline.filters['reviewed'].setCurrentIndex(1)
        self.timeline.filters['reviewed'].blockSignals(False)
        filters.addWidget(self.timeline.search)
        self.timeline.search.setVisible(role != 'labeler')
        filters.addWidget(button('Reset filters', self.clear_filters, 'link')); filters.addStretch()
        body.addWidget(self.sidebar)
        self.timeline.filter_bar.hide(); self.timeline.density.hide(); self.timeline.density_scroll.hide(); self.timeline.key_hint.hide()
        self.timeline.range_text.hide()
        layout.insertWidget(1, self.timeline.range_bar)  # date range across houses, above the list
        self.timeline.table.setItemDelegate(ReviewDelegate(theme, self.timeline.table))
        self.timeline.table.horizontalHeader().hide(); self.timeline.table.verticalHeader().setDefaultSectionSize(116)
        self.timeline.table.setColumnWidth(0, 102)
        for col in range(2, 7):
            self.timeline.table.hideColumn(col)
        self.timeline.table.setMinimumWidth(300)
        self.timeline.count.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.split = QSplitter(Qt.Orientation.Horizontal); self.split.setHandleWidth(16); body.addWidget(self.split, 1)
        self.split.setObjectName('reviewSplit'); self.split.setStyleSheet('QSplitter#reviewSplit::handle { background: transparent; }')
        self.split.addWidget(self.timeline)
        self.event_view = EventView(backend, role, theme); self.event_view.autoplay = True
        self.event_view.splitter.setOrientation(Qt.Orientation.Vertical)
        self.event_view.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.event_view.title.setMinimumWidth(0)
        self.event_view.review.setText('Reviewed'); self.event_view.flag.setText('Flag')
        self.event_view.review_handler = self.mutate
        for key in ('J','K','R','F'):
            self.event_view.shortcuts[key].setEnabled(False)
        self.split.addWidget(self.event_view); self.split.setSizes([400, 1000]); self.split.setChildrenCollapsible(False)
        self.event_view.navigate.connect(self.move)
        self.timeline.table.selectionModel().currentRowChanged.connect(self.selected)
        self.timeline.runner.finished.connect(self.page_loaded)
        self.timeline.session_expired.connect(self.session_expired)
        self.event_view.session_expired.connect(self.session_expired)
        self.event_view.runner.finished.connect(self.detail_loaded)
        self.timeline.event_requested.connect(self.open_event)
        layout.addWidget(label('j / k  Next / previous    ·    Space  Play / pause    ·    r  Reviewed + next    ·    f  Flag    ·    c  Add to collection', 'muted'))
        QApplication.instance().installEventFilter(self)
        self.metadata.start(lambda: (backend.saved_filters(), backend.review_count()))
        self.cameras_runner = TaskRunner(self); self.cameras_runner.finished.connect(self.cameras_loaded)
        self.started = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self.started:
            self.started = True
            self.timeline.reload(); self.timeline.load_density()  # the density rows fill the camera filter
            if self.role != 'labeler' and hasattr(self.backend, 'cameras'):
                self.cameras_runner.start(self.backend.cameras)

    def cameras_loaded(self, cameras, error):
        if error:
            if isinstance(error, AuthError): self.session_expired.emit()
            return  # an older server: every camera the index saw stays listed
        self.timeline.set_cameras(cameras)

    def metadata_loaded(self, result, error):
        if error:
            self.notify(str(error), True); self.progress.setText('Review count unavailable')
            if isinstance(error,AuthError): self.session_expired.emit()
            return
        self.saved_filters, counts = result
        self.total = counts.unreviewed_24h
        selected = self.timeline.saved_filter
        self.saved.blockSignals(True)
        self.saved.clear(); self.saved.addItem('All events', None)
        for item in self.saved_filters:
            self.saved.addItem(item.title, item.key)
        self.saved.setCurrentIndex(max(0, self.saved.findData(selected))); self.saved.blockSignals(False)
        self.update_progress()

    def update_progress(self):
        if self.timeline.saved_filter or any(self.timeline.filters[k].currentData() is not None for k in ('kind','camera','ai','verdict','flagged','decision')) or self.timeline.search.text():
            found = (f'{self.timeline.total} matching' if self.timeline.total is not None
                     else f'{len(self.timeline.model.rows)} matches loaded')
            self.progress.setText(f'{found} · {self.total} unreviewed in last 24 h')
        else:
            self.progress.setText(f'{self.done} of {self.total} unreviewed in last 24 h · reviewed this session')

    def select_filter(self):
        self.timeline.saved_filter = self.saved.currentData(); self.timeline.reload()

    def open_filter(self, key):
        self.timeline.customer_id = None
        self.timeline.saved_filter = key
        self.timeline.filters['reviewed'].blockSignals(True); self.timeline.filters['reviewed'].setCurrentIndex(0); self.timeline.filters['reviewed'].blockSignals(False)
        self.saved.blockSignals(True); self.saved.setCurrentIndex(max(0, self.saved.findData(key))); self.saved.blockSignals(False)
        self.timeline.reload()

    def clear_filters(self):
        self.timeline.customer_id = None
        self.saved.setCurrentIndex(0); self.timeline.clear_filters()

    def selected(self, current, previous):
        if current.isValid():
            self.open_event(self.timeline.model.rows[current.row()].id)

    def open_event(self, eid):
        self.active_id = eid
        row = next((i for i,e in enumerate(self.timeline.model.rows) if e.id == eid),None)
        if row is not None and self.timeline.table.currentIndex().row() != row:
            self.timeline.table.setCurrentIndex(self.timeline.model.index(row,0)); return
        if row is None:
            self.timeline.table.setCurrentIndex(QModelIndex())
        self.event_view.open(eid)

    def active_event(self):
        return next((e for e in self.timeline.model.rows if e.id == self.active_id),None) or (
            self.event_view.recording if self.event_view.recording and self.event_view.recording.id == self.active_id else None)

    def detail_loaded(self, result, error):
        view = self.event_view
        if not error and view.recording is result:
            view.title.setText(f'#{result.id} · {camera_name(result)}')
            view.title.setToolTip(f'{camera_name(result)} · {local_time(result.start_utc,result.timezone)}')
            if self.mutation.busy:
                current = next((e for e in self.timeline.model.rows if e.id == result.id),None)
                if current: self.sync_detail(current)

    def page_loaded(self, result, error):
        if not error and self.pending_next is not None:
            anchor, generation = self.pending_next
            self.pending_next = None
            if generation == self.timeline.generation and result.items:
                # The cursor follows anchor even when its optimistic removal
                # has shifted every row. Resolve the returned event identity.
                target = result.items[0].id
                self.open_event(target)
        if not error and not self.timeline.model.rows:
            self.event_view.player.player.stop(); self.event_view.recording = None
            self.event_view.tabs.hide(); self.event_view.title.setText('No events in this view')
        self.update_progress()

    def move(self, delta):
        current = self.timeline.table.currentIndex().row()
        row = self.timeline.next_row(current, delta)
        if row == current:
            row = current + delta  # past the end: load older below
        if 0 <= row < len(self.timeline.model.rows):
            self.timeline.table.setCurrentIndex(self.timeline.model.index(row, 0))
        elif delta > 0 and self.timeline.cursor:
            self.pending_next = self.active_id, self.timeline.generation
            self.timeline.load_older()

    def mutate(self, key):
        event = self.active_event()
        if not event:
            return
        self.mutation.submit(event, key, True if key == 'reviewed' else None)
        if key == 'reviewed':
            self.move(1)

    def sync_detail(self, event):
        view = self.event_view
        if view.recording and view.recording.id == event.id:
            view.recording.reviewed, view.recording.flagged = event.reviewed, event.flagged
            view.review.setChecked(event.reviewed); view.flag.setChecked(event.flagged)

    def mutation_done(self, result, error):
        before = self.mutation.active_before
        key = 'reviewed' if 'reviewed' in self.mutation.active_changes else 'flagged'
        if error:
            self.notify(f'Event #{before.id}: change not saved. {error}', True)
            if isinstance(error, AuthError):
                self.session_expired.emit()
        else:
            if key == 'reviewed' and not before.reviewed:
                self.done += 1
            self.notify('Marked reviewed' if key == 'reviewed' else 'Flag updated')
            self.review_changed.emit()
            selected_filter = self.timeline.filters[key].currentData()
            if result.id not in self.mutation.queue and selected_filter is not None and selected_filter != getattr(result,key):
                self.timeline.model.remove_event(result.id)
                if not self.timeline.model.rows:
                    if self.timeline.cursor: self.timeline.load_older()
                    else:
                        self.timeline.stack.setCurrentWidget(self.timeline.empty)
                        self.event_view.player.player.stop(); self.event_view.recording = None
                        self.event_view.tabs.hide(); self.event_view.title.setText('All caught up in this view')
        self.update_progress()

    def notify(self, text, error=False):
        self.toast.setObjectName('error' if error else 'badge'); self.toast.setText(text); self.toast.show()
        self.toast.style().unpolish(self.toast)
        self.toast.style().polish(self.toast)
        self.toast_timer.start()

    def add_to_collection(self):
        event = self.active_event()
        if event:
            from .collections import CollectionPicker
            self.picker = CollectionPicker(self.backend, event.id, self.last_collection, self)
            self.picker.session_expired.connect(self.session_expired)
            self.picker.added.connect(self.collection_added); self.picker.show()

    def collection_added(self, collection):
        self.last_collection = collection.id; self.notify(f'Added to {collection.name}')

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and self.isVisible() and isinstance(watched, QWidget) and self.isAncestorOf(watched):
            if watched.window() != self.window() or isinstance(QApplication.focusWidget(), (QLineEdit,QPlainTextEdit,QComboBox,QAbstractSpinBox)):
                return False
            if event.modifiers() != Qt.KeyboardModifier.NoModifier:
                return False
            action = {Qt.Key.Key_J: lambda: self.move(1), Qt.Key.Key_K: lambda: self.move(-1),
                      Qt.Key.Key_R: lambda: self.mutate('reviewed'), Qt.Key.Key_F: lambda: self.mutate('flagged'),
                      Qt.Key.Key_C: self.add_to_collection}.get(event.key())
            if action:
                action(); return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        compact = self.width() < 1400
        self.event_view.splitter.setOrientation(Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal)
        self.event_view.splitter.setSizes([340, 320] if compact else [650, 360])
        super().resizeEvent(event)

    def minimumSizeHint(self):
        return QSize(1050,600)
