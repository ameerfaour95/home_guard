"""Label Studio-style video annotation workspace with optimistic version saves."""
from copy import deepcopy
from time import monotonic
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QImage
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink, QMediaMetaData
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPlainTextEdit,
    QCheckBox, QComboBox, QScrollArea, QFrame, QDialog, QLineEdit, QApplication, QSizePolicy)
from home_guard_project.fleet_contract.tracks import frame_at
from .backend import AuthError, ConflictError
from .workers import TaskRunner, closing
from .widgets.common import label, button
from .label_document import LabelDocument, CLASSES
from .label_canvas import LabelCanvas, TrackTimeline
from .models import ReviewDecision

STALE_REVIEW = 'This clip changed since you opened it — reload'

HELP = [('← / →', 'One frame'), ('Shift + ← / →', 'Five frames'), ('Space', 'Play / pause in real time'),
        ('. / ,', 'Next / previous keyframe'), ('1–9', 'Class: person, bicycle, car, motorcycle, bus, truck, bird, cat, dog'),
        ('K', 'Add / remove keyframe'), ('H', 'Hide / keep segment'), ('C', 'Copy box to next frame'),
        ('Del / Shift + Del', 'Delete keyframe / track'), ('Ctrl + Z / Ctrl + Shift + Z', 'Undo / redo (including text)'),
        ('Ctrl + S', 'Save draft'), ('Ctrl + Enter', 'Submit and open next clip'), ('Esc', 'Deselect'), ('?', 'Keyboard help')]


class LabelView(QWidget):
    session_expired = Signal()
    annotation_saved = Signal(object)

    def __init__(self, backend, role='admin', theme='dark'):
        super().__init__()
        self.backend, self.role, self.doc = backend, role, None
        self.queue, self.queue_index, self.pending_open = [], -1, None
        self.conflicted, self.submit_pending, self.saved_at = False, False, None
        self.pending_frame, self.pending_copy = None, None
        self.loader, self.writer, self.reader, self.media = [TaskRunner(self) for _ in range(4)]
        self.loader.finished.connect(self.loaded); self.writer.finished.connect(self.saved)
        self.reader.finished.connect(self.read_done); self.media.finished.connect(self.media_loaded)
        self.autosave = QTimer(self); self.autosave.setSingleShot(True); self.autosave.setInterval(2000)
        self.autosave.timeout.connect(self.save)
        self.clock = QTimer(self); self.clock.setInterval(1000); self.clock.timeout.connect(self.update_save_state); self.clock.start()
        self.player = QMediaPlayer(self); self.sink = QVideoSink(self); self.player.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self.frame_decoded)
        self.player.errorOccurred.connect(lambda *_: self.media_error())
        self.player.playbackStateChanged.connect(lambda s: self.play.setText('Pause' if s == QMediaPlayer.PlaybackState.PlayingState else 'Play'))
        root = QVBoxLayout(self); root.setContentsMargins(24, 18, 24, 18); root.setSpacing(12)
        top = QHBoxLayout(); self.title = label('Label', 'section'); top.addWidget(self.title, 1)
        self.status = label('QUEUE', 'badge'); top.addWidget(self.status)
        self.version = button('Versions', self.versions); top.addWidget(self.version)
        self.save_state = label('Select a clip', 'muted'); top.addWidget(self.save_state)
        self.save_button = button('Save', self.save); top.addWidget(self.save_button)
        self.submit_button = button('Submit →', self.submit, 'primary'); top.addWidget(self.submit_button)
        root.addLayout(top)
        queue = QHBoxLayout(); self.queue_title = label('Needs labeling', 'eyebrow'); queue.addWidget(self.queue_title)
        self.clip_picker = QComboBox(); self.clip_picker.setMinimumWidth(250); self.clip_picker.activated.connect(self.choose_clip)
        queue.addWidget(self.clip_picker, 1)
        queue.addWidget(button('Previous clip', lambda: self.open_index(self.queue_index-1)))
        queue.addWidget(button('Next clip', lambda: self.open_index(self.queue_index+1)))
        queue.addWidget(button('?', self.help)); root.addLayout(queue)
        self.error = label('', 'error', True); self.error.hide(); root.addWidget(self.error)
        self.conflict_bar = QWidget(); conflict = QHBoxLayout(self.conflict_bar); conflict.setContentsMargins(0, 0, 0, 0)
        conflict.addWidget(label('Someone else saved a newer version', 'error'), 1)
        conflict.addWidget(button('Reload', lambda: self.resolve_conflict(False)))
        conflict.addWidget(button('Keep mine', lambda: self.resolve_conflict(True)))
        self.conflict_bar.hide(); root.addWidget(self.conflict_bar)
        self.splitter = QSplitter(Qt.Orientation.Horizontal); root.addWidget(self.splitter, 1)
        self.splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.splitter.setHandleWidth(14)
        self.splitter.setStyleSheet('QSplitter::handle { background: transparent; }')
        left = QWidget(); column = QVBoxLayout(left); column.setContentsMargins(0, 0, 0, 0); column.setSpacing(10)
        tools = QHBoxLayout(); self.classes = QComboBox()
        for i, name in enumerate(CLASSES): self.classes.addItem(f'{i+1}  {name}', name)
        self.classes.activated.connect(lambda: self.doc and self.doc.change_class(self.classes.currentData()))
        tools.addWidget(self.classes)
        self.suggestions = button('Accept all suggestions', lambda: self.doc and self.doc.accept_all())
        tools.addWidget(self.suggestions); tools.addStretch()
        tools.addWidget(label('Drag to draw · 8 resize handles', 'muted')); column.addLayout(tools)
        self.canvas = LabelCanvas(theme); self.canvas.selected.connect(self.selection_changed)
        self.canvas.interaction_started.connect(self.player.pause); column.addWidget(self.canvas, 1)
        transport = QHBoxLayout(); self.play = button('Play', self.toggle_play); transport.addWidget(self.play)
        transport.addWidget(button('‹', lambda: self.step(-1))); transport.addWidget(button('›', lambda: self.step(1)))
        self.position = label('Frame 1 · 0.000 s', 'muted'); transport.addWidget(self.position, 1)
        transport.addWidget(button('Keyframe K', lambda: self.doc and self.doc.toggle_keyframe()))
        transport.addWidget(button('Hide / keep H', lambda: self.doc and self.doc.set_enabled()))
        column.addLayout(transport)
        self.timeline = TrackTimeline(theme); self.timeline.seek_requested.connect(self.seek); self.timeline.selected.connect(self.selection_changed)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.timeline)
        scroll.setMinimumHeight(122); scroll.setMaximumHeight(200); scroll.setFrameShape(QFrame.Shape.NoFrame); column.addWidget(scroll)
        column.addWidget(label('● Keyframes    Solid: kept / interpolated    Dotted: hidden', 'muted'))
        self.splitter.addWidget(left)
        panel = QFrame(); panel.setObjectName('card'); right = QVBoxLayout(panel); right.setContentsMargins(18, 16, 18, 16); right.setSpacing(8)
        panel.setStyleSheet('QCheckBox { background: transparent; } QWidget#annotationReview { background: transparent; }')
        right.addWidget(label('DESCRIPTION', 'eyebrow')); right.addWidget(label('AI draft', 'section'))
        self.ai_meta = label('', 'muted', True); right.addWidget(self.ai_meta); self.ai_meta.setVisible(role == 'admin')
        self.ai_text = QPlainTextEdit(); self.ai_text.setReadOnly(True); self.ai_text.setMaximumHeight(150); self.ai_text.setAccessibleName('Original AI draft'); right.addWidget(self.ai_text, 1)
        self.ai_text.setMinimumHeight(55)
        row = QHBoxLayout(); row.addWidget(label('Your description', 'section'), 1); self.diff = label('', 'badge'); row.addWidget(self.diff); right.addLayout(row)
        self.description = QPlainTextEdit(); self.description.setPlaceholderText('Describe what happens in this clip…'); self.description.setAccessibleName('Your description')
        self.description.setMinimumHeight(95)
        self.description.setUndoRedoEnabled(False); self.description.textChanged.connect(self.text_changed); right.addWidget(self.description, 2)
        self.drop = QCheckBox('Drop this clip from training'); self.needs_review = QCheckBox('Needs review')
        self.drop.toggled.connect(self.text_changed); self.needs_review.toggled.connect(self.text_changed)
        right.addWidget(self.drop); right.addWidget(self.needs_review)
        self.review_panel = QWidget(); review = QVBoxLayout(self.review_panel); review.setContentsMargins(0, 10, 0, 0)
        self.review_panel.setObjectName('annotationReview')
        review.addWidget(label('Admin review', 'section'))
        self.review_note = QLineEdit(); self.review_note.setPlaceholderText('Review note'); review.addWidget(self.review_note)
        self.at_frame = QCheckBox('At this frame'); self.at_frame.setChecked(True); review.addWidget(self.at_frame)
        actions = QHBoxLayout(); actions.addWidget(button('Accept', lambda: self.review('accept'), 'primary'))
        actions.addWidget(button('Reject', lambda: self.review('reject'))); review.addLayout(actions); right.addWidget(self.review_panel)
        self.review_panel.hide()
        self.review_feedback = label('', 'error', True); right.addWidget(self.review_feedback)
        self.splitter.addWidget(panel); panel.setMinimumWidth(310)
        self.splitter.setStretchFactor(0, 3); self.splitter.setStretchFactor(1, 1); self.splitter.setSizes([920, 360])
        self.shortcuts = {}
        callbacks = {'Left': lambda: self.step(-1), 'Right': lambda: self.step(1),
            'Shift+Left': lambda: self.step(-5), 'Shift+Right': lambda: self.step(5), 'Space': self.toggle_play,
            '.': lambda: self.jump_keyframe(1), ',': lambda: self.jump_keyframe(-1),
            'K': lambda: self.doc.toggle_keyframe(), 'H': lambda: self.doc.set_enabled(), 'C': self.copy_next,
            'Del': lambda: self.doc.delete_keyframe(), 'Shift+Del': lambda: self.doc.delete_track(),
            'Ctrl+Z': lambda: self.doc.undo(), 'Ctrl+Shift+Z': lambda: self.doc.redo(),
            'Ctrl+S': self.save, 'Ctrl+Return': self.submit, 'Esc': self.deselect, '?': self.help}
        callbacks.update({str(i+1): lambda n=n: self.doc.change_class(n) for i, n in enumerate(CLASSES)})
        for key, fn in callbacks.items():
            shortcut = QShortcut(QKeySequence(key), self); shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda fn=fn, key=key: self.run_shortcut(key, fn)); self.shortcuts[key] = shortcut
        QApplication.instance().focusChanged.connect(self.focus_changed)
        self.set_ready(False)

    def focus_changed(self, old, new):
        typing = isinstance(new, (QPlainTextEdit, QLineEdit))
        for key, shortcut in self.shortcuts.items():
            shortcut.setEnabled(not typing or key.startswith('Ctrl+'))

    def run_shortcut(self, key, fn):
        if self.doc and (self.pending_frame is None or key in ('Left', 'Right', 'Shift+Left', 'Shift+Right', '?', 'Esc')):
            fn()

    def set_ready(self, ready):
        self.splitter.setEnabled(ready); self.submit_button.setEnabled(ready); self.save_button.setEnabled(ready); self.version.setEnabled(ready)

    def show_error(self, error):
        self.error.setText(str(error)); self.error.show()
        if isinstance(error, AuthError): self.session_expired.emit()

    def open_queue(self, key='needs_labeling', event_id=None):
        if self.doc and (self.doc.dirty or self.writer.busy):
            self.pending_open = ('queue', key, event_id); self.save(); return
        if self.loader.busy: return
        self.autosave.stop(); self.player.stop(); self.set_ready(False)
        self.queue_title.setText(key.replace('_', ' ').upper()); self.error.hide()
        self.loading_kind = 'queue'
        def fetch():
            items, cursor = [], None
            while not closing.is_set():
                page = self.backend.events(filter=key, cursor=cursor, limit=500)
                items.extend(page.items); cursor = page.next_cursor
                if not cursor: break
            if event_id is not None and not any(e.id == event_id for e in items): items.insert(0, self.backend.event(event_id))
            return items, event_id
        self.loader.start(fetch)

    def open_event(self, event_id):
        self.open_queue('needs_labeling', event_id)

    def choose_clip(self, index):
        self.clip_picker.setCurrentIndex(self.queue_index); self.open_index(index)

    def open_index(self, index):
        if not 0 <= index < len(self.queue) or self.loader.busy: return
        if self.doc and (self.doc.dirty or self.writer.busy):
            self.pending_open = ('index', index); self.save(); return
        self.autosave.stop(); self.player.stop(); self.player.setSource(QUrl()); self.set_ready(False)
        self.queue_index = index; self.clip_picker.setCurrentIndex(index)
        self.loading_kind = 'clip'; eid = self.queue[index].id
        self.loader.start(lambda: (self.backend.event(eid), self.backend.annotation(eid)))

    def loaded(self, result, error):
        if error: self.show_error(error); self.set_ready(bool(self.doc)); return
        if self.loading_kind == 'queue':
            self.queue, eid = result; self.clip_picker.clear()
            for i, e in enumerate(self.queue): self.clip_picker.addItem(f'{i+1} / {len(self.queue)}   ·   {e.camera}   ·   #{e.id}')
            if self.queue:
                self.open_index(next((i for i, e in enumerate(self.queue) if e.id == eid), 0))
            else:
                self.title.setText('Queue complete'); self.save_state.setText('No clips in this view')
                self.doc = None; self.canvas.doc = None; self.timeline.doc = None; self.canvas.image = QImage()
                self.canvas.message = 'No clips in this queue'; self.canvas.update(); self.timeline.update()
                self.ai_text.clear(); self.description.clear()
            return
        event, annotation = result
        self.recording = event; self.install(annotation)
        self.title.setText(f'Label  ·  {event.camera}  ·  #{event.id}')
        self.canvas.image = QImage(); self.canvas.message = 'Loading recording…'; self.canvas.frame_size = tuple(annotation.frame_size or [640, 360])
        self.request_media()

    def install(self, annotation):
        if self.doc: self.doc.deleteLater()
        self.doc = LabelDocument(annotation, self.recording.duration_sec, self)
        self.pending_frame, self.pending_copy = None, None
        self.canvas.doc = self.timeline.doc = self.doc
        self.doc.changed.connect(self.changed); self.doc.position_changed.connect(self.position_changed)
        self.conflicted = False; self.conflict_bar.hide(); self.error.hide(); self.saved_at = None
        self.ai_text.setPlainText(annotation.ai_description)
        self.ai_meta.setText(f'{annotation.ai_model or "Unknown model"}  ·  {annotation.ai_prompt_version or "No prompt version"}')
        self.review_note.setText(annotation.review_note)
        self.review_feedback.setText(annotation.review_note + (f' · Frame {annotation.review_frame+1}' if annotation.review_frame is not None else ''))
        self.set_ready(True); self.refresh(); self.position_changed()

    def request_media(self):
        if self.media.busy:
            self.media_dirty = True; return
        self.media_dirty = False; self.media_event = self.recording.id
        video = next((a for role in ('original_video', 'clip', 'rendition') for a in self.recording.artifacts if a.available and a.role == role), None)
        if not video: self.media_error(); return
        self.media.start(lambda: self.backend.artifact_access(video.id, 'training' if self.role == 'labeler' else 'review'))

    def media_loaded(self, access, error):
        if self.media_dirty or self.media_event != self.recording.id: self.request_media(); return
        if error: self.show_error(error); self.media_error(); return
        self.player.setSource(QUrl(access.url)); self.player.play(); self.player.pause()

    def media_error(self):
        self.canvas.message = 'Recording unavailable. Reopen this clip to retry.'; self.canvas.update()
        self.show_error('Recording unavailable. Text and existing annotations are still available.')

    def frame_decoded(self, frame):
        if not self.doc or not frame.isValid(): return
        image = frame.toImage()
        if image.isNull(): return
        fps = self.player.metaData().value(QMediaMetaData.Key.VideoFrameRate)
        if fps and float(fps) > 0: self.doc.fps = float(fps)
        self.doc.frame_size = [image.width(), image.height()]
        t = frame.startTime()/1_000_000 if frame.startTime() >= 0 else self.player.position()/1000
        native_frame = frame_at(t, self.doc.fps)
        if self.pending_frame is not None and native_frame != self.pending_frame:
            return  # A decoder can deliver an earlier seek's frame while scrubbing.
        self.canvas.image = image
        self.pending_frame = None; self.canvas.setEnabled(True)
        self.doc.seek(native_frame, t)
        if self.pending_copy:
            track_id, box = self.pending_copy; self.pending_copy = None
            track = next((tr for tr in self.doc.tracks if tr.track_id == track_id), None)
            if track: self.doc.put_box(box, track)

    def position_changed(self):
        if not self.doc: return
        self.position.setText(f'Frame {self.doc.frame+1} / {self.doc.frame_count}   ·   {self.doc.t_sec:.3f} s')
        self.canvas.update(); self.timeline.update()

    def seek(self, frame):
        if not self.doc: return
        self.player.pause()
        frame = max(0, min(self.doc.frame_count-1, frame))
        if self.player.source().isEmpty():
            self.doc.seek(frame); return
        if frame == self.doc.frame and self.pending_frame is None: return
        self.pending_frame = frame; self.canvas.setEnabled(False)
        self.position.setText(f'Seeking frame {frame+1}…')
        # Qt's paused decoder includes a frame's end timestamp in its seek interval.
        # Seek inside the requested frame, then use the decoded presentation time.
        start = self.doc.time_for(frame)
        span = self.doc.time_for(frame+1)-start if frame+1 < self.doc.frame_count else 1/self.doc.fps
        self.player.setPosition(round((start+span/2)*1000))

    def step(self, delta):
        if self.doc: self.seek((self.pending_frame if self.pending_frame is not None else self.doc.frame)+delta)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.player.pause()
        else: self.player.play()

    def copy_next(self):
        from home_guard_project.fleet_contract.tracks import box_at
        if not self.doc.track or self.doc.frame == self.doc.frame_count-1: return
        if self.player.source().isEmpty(): self.doc.copy_next(); return
        box = box_at(self.doc.track, self.doc.t_sec)
        if box:
            self.pending_copy = (self.doc.selected, box); self.seek(self.doc.frame+1)

    def jump_keyframe(self, direction):
        if not self.doc or not self.doc.track: return
        frames = [k.frame for k in self.doc.track.keyframes if (k.frame-self.doc.frame)*direction > 0]
        if frames: self.seek(min(frames) if direction > 0 else max(frames))

    def deselect(self):
        self.doc.selected = None; self.selection_changed()

    def selection_changed(self):
        if self.doc:
            name = self.doc.track.label if self.doc.track else self.doc.current_class
            self.classes.setCurrentIndex(CLASSES.index(name)); self.canvas.update(); self.timeline.update()

    def text_changed(self, *_):
        if self.doc: self.doc.set_text(self.description.toPlainText(), self.drop.isChecked(), self.needs_review.isChecked())

    def changed(self):
        self.refresh()
        if not self.conflicted and self.doc.dirty: self.autosave.start()
        else: self.autosave.stop()

    def refresh(self):
        if not self.doc: return
        for widget, value in ((self.description, self.doc.description), (self.drop, self.doc.drop_clip), (self.needs_review, self.doc.needs_review)):
            widget.blockSignals(True)
            if widget is self.description:
                if widget.toPlainText() != value: widget.setPlainText(value)
            else: widget.setChecked(value)
            widget.blockSignals(False)
        self.status.setText(self.doc.annotation.status.upper()); self.version.setText(f'v{self.doc.annotation.version} · Versions')
        self.diff.setText('Changed' if self.doc.description != self.doc.annotation.ai_description else 'AI draft')
        self.suggestions.setEnabled(any(t.source == 'suggestion' for t in self.doc.tracks))
        self.review_panel.setVisible(self.role == 'admin' and self.doc.annotation.status == 'submitted')
        self.timeline.refresh(); self.selection_changed(); self.update_save_state()

    def update_save_state(self):
        if not self.doc: return
        text = 'Saving…' if self.writer.busy else 'Unsaved changes' if self.doc.dirty else f'Saved {int(monotonic()-self.saved_at)} s ago' if self.saved_at else 'Saved' if self.doc.annotation.version else 'New annotation'
        self.save_state.setText(text)

    def save(self, status='edited'):
        if isinstance(status, bool): status = 'edited'  # QPushButton.clicked(bool)
        if not self.doc or self.conflicted: return
        if self.writer.busy:
            if status == 'submitted': self.submit_pending = True
            return
        if not self.doc.dirty and status == 'edited': self.finish_navigation(); return
        errors = self.doc.errors()
        if errors: self.show_error('\n'.join(errors)); return
        self.autosave.stop(); self.error.hide(); self.write_kind = 'save'; self.write_status = status
        self.sent_doc, self.sent_snapshot = self.doc, self.doc.snapshot()
        eid, request = self.doc.annotation.event_id, self.doc.request(status)
        if status == 'submitted': self.set_ready(False)
        self.writer.start(lambda: self.backend.save_annotation(eid, request)); self.update_save_state()

    def saved(self, annotation, error):
        self.set_ready(True)
        if error:
            self.submit_pending = False
            if isinstance(error, ConflictError) and self.write_kind == 'review':
                self.show_error(STALE_REVIEW)
            elif isinstance(error, ConflictError):
                self.conflicted = True; self.autosave.stop(); self.conflict_bar.show()
            else: self.show_error(error)
            self.update_save_state(); return
        self.doc.annotation = annotation; self.doc.saved_snapshot = self.sent_snapshot
        self.saved_at = monotonic(); self.refresh(); self.annotation_saved.emit(annotation)
        if self.write_kind == 'review':
            self.review_feedback.setText(annotation.review_note)
        if self.submit_pending:
            self.submit_pending = False; self.save('submitted'); return
        if self.doc.dirty: self.autosave.start(); return
        if self.pending_open: self.finish_navigation()
        elif self.write_status == 'submitted':
            if self.queue_index+1 < len(self.queue): self.open_index(self.queue_index+1)
            else: self.save_state.setText('Submitted · queue complete')

    def finish_navigation(self):
        pending, self.pending_open = self.pending_open, None
        if pending:
            if pending[0] == 'queue': self.open_queue(*pending[1:])
            else: self.open_index(pending[1])

    def submit(self):
        if self.doc: self.save('submitted')

    def resolve_conflict(self, keep):
        if self.reader.busy: return
        self.read_kind = 'keep' if keep else 'reload'
        eid = self.doc.annotation.event_id; self.read_event = eid
        self.reader.start(lambda: self.backend.annotation(eid))

    def versions(self):
        if not self.doc or self.reader.busy: return
        self.read_kind = 'history'; eid = self.doc.annotation.event_id
        self.read_event = eid
        self.reader.start(lambda: self.backend.annotation_history(eid))

    def read_done(self, result, error):
        if not self.doc or self.doc.annotation.event_id != self.read_event: return
        if error: self.show_error(error); return
        if self.read_kind == 'history':
            self.show_dialog('Versions', '\n\n'.join(f'v{v.version}  ·  {v.status.title()}  ·  {v.author or "Unknown author"}\n{v.created_utc:%Y-%m-%d %H:%M:%S} UTC  ·  {v.tracks_count} tracks'+('  ·  Text changed' if v.description_changed else '') for v in result) or 'No saved versions yet.')
        elif self.read_kind == 'reload':
            self.pending_open = None; self.install(result)
        else:
            self.doc.annotation.version = result.version
            self.conflicted = False; self.conflict_bar.hide(); self.save(self.write_status)

    def review(self, decision):
        if not self.doc or self.role != 'admin' or self.writer.busy: return
        if self.doc.dirty: self.show_error('Save and submit your changes before reviewing.'); return
        if decision == 'reject' and not self.review_note.text().strip(): self.show_error('Add a note explaining what needs to change.'); self.review_note.setFocus(); return
        eid = self.doc.annotation.event_id
        request = ReviewDecision(decision, self.review_note.text().strip(), self.doc.frame if self.at_frame.isChecked() else None, self.doc.annotation.version)
        self.sent_snapshot = self.doc.snapshot(); self.write_kind, self.write_status = 'review', 'reviewed'
        self.set_ready(False); self.writer.start(lambda: self.backend.review_annotation(eid, request))

    def show_dialog(self, title, text):
        self.popover = QDialog(self); self.popover.setWindowTitle(title); self.popover.resize(540, 430)
        layout = QVBoxLayout(self.popover); layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(label(title, 'section')); content = QPlainTextEdit(text); content.setReadOnly(True); layout.addWidget(content)
        layout.addWidget(button('Close', self.popover.accept)); self.popover.show()

    def help(self):
        self.show_dialog('Keyboard shortcuts', '\n\n'.join(f'{key}   —   {text}' for key, text in HELP))

    def hideEvent(self, event):
        self.player.pause(); self.save(); super().hideEvent(event)
