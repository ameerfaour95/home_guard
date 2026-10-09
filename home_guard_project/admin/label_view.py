"""Tag · YOLO: boxes and tracks on full frames, with optimistic version saves.

Boxes only (the owner, 2026-10-09: YOLO and AI are separate tabs, "don't make them together"): the right side lists
the clip's tracks with their provenance and the clip's training checks; the scene description is tagged in Tag · AI.
A clip's saved description is carried through every save unchanged."""
from copy import deepcopy
from time import monotonic
from PySide6.QtCore import Qt, QTimer, QUrl, Signal, QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QShortcut, QKeySequence, QImage
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink, QMediaMetaData
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPlainTextEdit,
    QCheckBox, QComboBox, QScrollArea, QFrame, QDialog, QLineEdit, QApplication, QSizePolicy, QListWidget,
    QListWidgetItem, QAbstractItemView)
from home_guard_project.fleet_contract.tracks import frame_at
from .backend import AuthError, ConflictError
from .workers import TaskRunner, closing
from .widgets.common import label, button
from .label_document import LabelDocument, CLASSES
from .label_canvas import LabelCanvas, TrackTimeline
from .player import SESSION
from .tag_widgets import ProvenanceChip, machine_name
from .models import ReviewDecision
from .formatting import camera_name


def clip_caption(key):
    """A dataset clip's title: its camera by the box's channel rule and its time, never the raw id ("ds:<clip id>")."""
    from home_guard_project.fleet_contract.health import camera_label
    from home_guard_project.fleet_contract.keys import stem_kind
    stem = key.split(':', 1)[-1].rsplit('/', 1)[-1]
    parts = stem.rsplit('_', 2)
    camera = parts[0] if len(parts) == 3 and parts[1].isdigit() else stem
    when = ''
    try:
        from datetime import datetime
        ts = stem_kind(stem)[1]
        when = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else ''
    except Exception:  # noqa: BLE001 - the time is a caption only
        pass
    return f'{camera_label(camera)}  ·  {when}' if when else camera_label(camera)

STALE_REVIEW = 'This clip changed since you opened it — reload'

HELP = [('← / →', 'One frame'), ('Shift + ← / →', 'Five frames'), ('Space', 'Play / pause in real time'),
        ('. / ,', 'Next / previous keyframe'), ('1–9', 'Class: person, bicycle, car, motorcycle, bus, truck, bird, cat, dog'),
        ('K', 'Add / remove keyframe'), ('O', 'Hide / keep segment (the object is out of view from here)'),
        ('B', 'Boxes on / off (all of them, also while playing)'), ('H', 'Hide / show the selected track (view only)'),
        ('Alt + M', 'Split the selected track at this frame (a new object from here)'),
        ('M', 'Merge the selected track with the nearest one it never overlaps (P1 stays P1)'),
        ('C', 'Copy box to next frame'),
        ('Del / Shift + Del', 'Delete the selected box (its whole track) / only this keyframe'), ('Ctrl + Z / Ctrl + Shift + Z', 'Undo / redo (including text)'),
        ('Wheel / pinch, + / −, 0', 'Zoom around the cursor, in / out, fit the picture'),
        ('Middle button or Space + drag', 'Pan the zoomed picture'),
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
        # every frame decoded once, in the background, when a clip opens (frame_cache.py): steps are lookups
        self.frames, self.frames_for, self.refine = None, None, None
        self.cacher = TaskRunner(self); self.cacher.finished.connect(self.frames_decoded)
        self.play_timer = QTimer(self); self.play_timer.timeout.connect(self.play_cached)
        self.media_buffer = None
        # the first decoded frame goes on screen as soon as it exists (the player's own first frame may be later)
        self.first_frame = QTimer(self); self.first_frame.setInterval(15); self.first_frame.timeout.connect(self.show_first)
        self.loader.finished.connect(self.loaded); self.writer.finished.connect(self.saved)
        self.reader.finished.connect(self.read_done); self.media.finished.connect(self.media_loaded)
        # a seek the decoder never answers exactly (another frame's timestamp, or none while paused) settles anyway
        self.seek_watchdog = QTimer(self); self.seek_watchdog.setSingleShot(True); self.seek_watchdog.setInterval(700)
        self.seek_watchdog.timeout.connect(self.seek_timed_out); self.seek_nudged = False
        self.autosave = QTimer(self); self.autosave.setSingleShot(True); self.autosave.setInterval(2000)
        self.autosave.timeout.connect(self.save)
        self.clock = QTimer(self); self.clock.setInterval(1000); self.clock.timeout.connect(self.update_save_state)   # runs while shown
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
        # The YOLO boxes open as normal, editable tracks: fix them in place; nothing to accept first.
        self.boxes_source = ProvenanceChip(theme); tools.addWidget(self.boxes_source)
        self.boxes_note = label('', 'muted'); self.boxes_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        tools.addWidget(self.boxes_note, 1)
        for text, slot, tip in (('−', lambda: self.canvas.zoom_out(), 'Zoom out  −'),
                                ('+', lambda: self.canvas.zoom_in(), 'Zoom in  +  (or the mouse wheel / a pinch)'),
                                ('Fit', lambda: self.canvas.fit(), 'The whole picture  0')):
            b = button(text, slot, 'compact'); b.setToolTip(tip); tools.addWidget(b)
        self.boxes_toggle = button('Boxes  B', self.toggle_boxes, 'compact'); self.boxes_toggle.setCheckable(True)
        self.boxes_toggle.setChecked(SESSION['boxes']); self.boxes_toggle.setToolTip('Show / hide every box, also while playing (B)')
        tools.addWidget(self.boxes_toggle)
        self.hide_track = button('Hide track  H', self.toggle_track_hidden, 'compact')
        self.hide_track.setToolTip('Hide / show the selected track on the picture (view only, nothing is deleted)')
        tools.addWidget(self.hide_track)
        hint = label('Drag to draw · 8 resize handles', 'muted'); hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); tools.addWidget(hint, 1)
        column.addLayout(tools)
        self.canvas = LabelCanvas(theme); self.canvas.selected.connect(self.selection_changed)
        self.canvas.zoom_changed.connect(self.zoomed)
        self.canvas.interaction_started.connect(self.player.pause); column.addWidget(self.canvas, 1)
        transport = QHBoxLayout(); self.play = button('Play', self.toggle_play); transport.addWidget(self.play)
        transport.addWidget(button('‹', lambda: self.step(-1))); transport.addWidget(button('›', lambda: self.step(1)))
        self.position = label('Frame 1 · 0.000 s', 'muted'); self.position.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        transport.addWidget(self.position, 1)
        transport.addWidget(button('Keyframe  K', lambda: self.doc and self.doc.toggle_keyframe(), 'compact'))
        transport.addWidget(button('Hide / keep  O', lambda: self.doc and self.doc.set_enabled(), 'compact'))
        self.delete_box = button('Delete  Del', self.delete_selected, 'compact')
        self.delete_box.setToolTip('Remove the selected box and its whole track (Ctrl+Z brings it back)')
        transport.addWidget(self.delete_box)
        self.split_button = button('Split  Alt+M', self.split_track, 'compact')
        self.split_button.setToolTip('The selected track ends here and a new one starts from this frame')
        transport.addWidget(self.split_button)
        self.merge_button = button('Merge  M', self.merge_track, 'compact')
        self.merge_button.setToolTip('Join the selected track with the nearest one of its kind that is never on screen '
                                     'at the same time: the first one keeps its name (P1 stays P1)')
        transport.addWidget(self.merge_button)
        column.addLayout(transport)
        self.timeline = TrackTimeline(theme); self.timeline.seek_requested.connect(self.seek); self.timeline.selected.connect(self.selection_changed)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.timeline)
        scroll.setMinimumHeight(122); scroll.setMaximumHeight(200); scroll.setFrameShape(QFrame.Shape.NoFrame); column.addWidget(scroll)
        column.addWidget(label('● Keyframes    Solid: kept / interpolated    Dotted: hidden', 'muted'))
        self.splitter.addWidget(left)
        panel = QFrame(); panel.setObjectName('card'); right = QVBoxLayout(panel); right.setContentsMargins(18, 16, 18, 16); right.setSpacing(8)
        panel.setStyleSheet('QCheckBox { background: transparent; } QWidget#annotationReview { background: transparent; }')
        right.addWidget(label('TRACKS', 'eyebrow'))
        self.track_list = QListWidget(); self.track_list.setAccessibleName('Tracks of this clip')
        self.track_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.track_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # its type-to-search would take the one-letter keys
        self.track_list.itemClicked.connect(self.track_clicked); right.addWidget(self.track_list, 2)
        right.addWidget(label('SELECTED TRACK', 'eyebrow'))
        head = QHBoxLayout(); self.track_name = label('No track selected', 'section'); head.addWidget(self.track_name, 1)
        self.track_source = ProvenanceChip(theme); head.addWidget(self.track_source); right.addLayout(head)
        self.track_detail = label('', 'muted', True); right.addWidget(self.track_detail)
        right.addWidget(label('THIS CLIP', 'eyebrow'))
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
            'K': lambda: self.doc.toggle_keyframe(), 'O': lambda: self.doc.set_enabled(), 'C': self.copy_next,
            'B': self.toggle_boxes, 'H': self.toggle_track_hidden, 'Alt+M': self.split_track, 'M': self.merge_track,
            '+': lambda: self.canvas.zoom_in(), '=': lambda: self.canvas.zoom_in(), '-': lambda: self.canvas.zoom_out(),
            '0': lambda: self.canvas.fit(),
            'Del': self.delete_selected, 'Shift+Del': lambda: self.doc.delete_keyframe(),
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
        if key in ('B', '+', '=', '-', '0'):
            fn(); return  # the view toggles work with or without a clip, also while seeking
        if key == 'Space' and self.canvas.zoom > 1.0 and self.canvas.underMouse():
            return  # zoomed in with the mouse on the picture: Space held is for panning
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

    def open_clip(self, key):
        """A clip of the unified dataset (key "ds:<clip id>"): its boxes and text, saved as versions of its own."""
        if self.doc and (self.doc.dirty or self.writer.busy):
            self.pending_open = ('clip', key); self.save(); return
        if self.loader.busy: return
        self.autosave.stop(); self.player.stop(); self.player.setSource(QUrl()); self.set_ready(False)
        self.error.hide(); self.loading_kind = 'dataset'
        self.queue, self.queue_index = [], -1; self.clip_picker.clear(); self.clip_picker.addItem(key)
        self.queue_title.setText('DATASET CLIP')
        self.loader.start(lambda: (key, self.backend.clip_boxes(key)))

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
            for i, e in enumerate(self.queue): self.clip_picker.addItem(f'{i+1} / {len(self.queue)}   ·   {camera_name(e)}   ·   #{e.id}')
            if self.queue:
                self.open_index(next((i for i, e in enumerate(self.queue) if e.id == eid), 0))
            else:
                self.title.setText('Queue complete'); self.save_state.setText('No clips in this view')
                self.doc = None; self.canvas.doc = None; self.timeline.doc = None; self.canvas.image = QImage()
                self.canvas.message = 'No clips in this queue'; self.canvas.update(); self.timeline.update()
                self.track_list.clear()
            return
        if self.loading_kind == 'dataset':
            from types import SimpleNamespace
            key, annotation = result
            self.recording = SimpleNamespace(id=key, clip_key=key, camera=key.split(':', 1)[-1], artifacts=[],
                                             duration_sec=(annotation.frame_count / annotation.fps)
                                             if annotation.frame_count and annotation.fps else None)
            self.install(annotation)
            self.title.setText(f'Label  ·  {clip_caption(key)}')
            self.canvas.image = QImage(); self.canvas.message = 'Loading recording…'
            self.canvas.frame_size = tuple(annotation.frame_size or [640, 360])
            self.request_media(); return
        event, annotation = result
        self.recording = event; self.install(annotation)
        self.title.setText(f'Label  ·  {camera_name(event)}  ·  #{event.id}')
        self.canvas.image = QImage(); self.canvas.message = 'Loading recording…'; self.canvas.frame_size = tuple(annotation.frame_size or [640, 360])
        self.request_media()

    def install(self, annotation):
        self.frames, self.frames_for, self.refine = None, None, None
        if hasattr(self, 'play_timer'): self.play_timer.stop(); self.play.setText('Play')
        if self.doc: self.doc.deleteLater()
        self.doc = LabelDocument(annotation, self.recording.duration_sec, self)
        self.pending_frame, self.pending_copy = None, None
        self.canvas.doc = self.timeline.doc = self.doc
        self.doc.changed.connect(self.changed); self.doc.position_changed.connect(self.position_changed)
        self.conflicted = False; self.conflict_bar.hide(); self.error.hide(); self.saved_at = None
        self.review_note.setText(annotation.review_note)
        self.review_feedback.setText(annotation.review_note + (f' · Frame {annotation.review_frame+1}' if annotation.review_frame is not None else ''))
        self.set_ready(True); self.refresh(); self.position_changed()

    def request_media(self):
        if self.media.busy:
            self.media_dirty = True; return
        self.media_dirty = False; self.media_event = self.recording.id
        if getattr(self.recording, 'clip_key', None):
            from types import SimpleNamespace
            key = self.recording.clip_key
            self.media.start(lambda: self.fetched(SimpleNamespace(**self.backend.tagging_media(key, 'clip')))); return
        video = next((a for role in ('original_video', 'clip', 'rendition') for a in self.recording.artifacts if a.available and a.role == role), None)
        if not video: self.media_error(); return
        self.media.start(lambda: self.fetched(self.backend.artifact_access(video.id, 'training' if self.role == 'labeler' else 'review')))

    def fetched(self, access):
        """Worker thread: the media grant plus the clip's bytes (one download for the player and the frame cache)."""
        from .frame_cache import fetch
        return access, fetch(self.backend, access.url)

    def media_loaded(self, result, error):
        if self.media_dirty or self.media_event != self.recording.id: self.request_media(); return
        if error: self.show_error(error); self.media_error(); return
        access, data = result
        if data:   # the downloaded bytes, from memory
            buffer = QBuffer(self); buffer.setData(QByteArray(data)); buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            self.player.setSourceDevice(buffer, QUrl('label.mp4'))
            if self.media_buffer is not None: self.media_buffer.deleteLater()
            self.media_buffer = buffer
        else:
            self.player.setSource(QUrl(access.url))
        self.player.play(); self.player.pause()
        self.decode_frames(data or access.url)

    # ------------------------------------------------------------------ the frame cache
    def decode_frames(self, url):
        from .frame_cache import decode
        self.frames, self.frames_for = None, self.recording.id
        token = self.recording.id

        def started(clip):      # worker thread: frames show up as they decode
            if self.frames_for == token: self.frames = clip
        if not self.cacher.start(lambda: (token, decode(url, lambda: self.frames_for != token or closing.is_set(), token=token,
                                                         started=started))):
            self.pending_decode = url
        self.first_frame.start()

    def show_first(self):
        if not self.doc or self.frames_for is None or not self.canvas.image.isNull():
            self.first_frame.stop(); return
        if self.pending_frame is None and self.cached(self.doc.frame) is not None:
            self.first_frame.stop(); self.show_cached(self.doc.frame)

    def frames_decoded(self, result, error):
        pending, self.pending_decode = getattr(self, 'pending_decode', None), None
        if pending is not None and self.recording is not None:
            self.decode_frames(pending); return
        if error or not result: return
        token, clip = result
        if clip is None or not self.doc or token != self.frames_for: return
        self.show_first()
        self.frames = clip
        self.position_changed()

    def cached(self, frame):
        """*frame* from the cache when it has decoded (and belongs to the open clip), else None."""
        clip = self.frames
        if clip is None or clip.token != self.frames_for or self.frames_for is None: return None
        return clip.get(frame)

    def zoomed(self):
        """Zoomed in on a cached (screen-sized) frame: fetch the full-size one so the detail is real."""
        if (self.doc and self.canvas.zoom > 1.0 and not self.play_timer.isActive()
                and self.canvas.image is self.cached(self.doc.frame)):
            self.refine_frame(self.doc.frame)

    def frames_ready(self):
        return self.frames is not None and self.frames.done and self.frames.token == self.frames_for

    def play_cached(self):
        """One playback tick from the cache (the player is not used for playback once frames are in)."""
        if not self.doc or self.doc.frame+1 >= self.doc.frame_count or self.cached(self.doc.frame+1) is None:
            self.stop_cached(); return
        self.show_cached(self.doc.frame+1)

    def stop_cached(self):
        if self.play_timer.isActive():
            self.play_timer.stop(); self.play.setText('Play')
            if self.doc: self.show_cached(self.doc.frame)       # the paused frame refines when zoomed

    def show_cached(self, frame):
        """A frame step from the cache: no decoder round trip; a zoomed view gets the full-size frame on top."""
        self.pending_frame = None; self.seek_watchdog.stop()
        self.canvas.image = self.cached(frame); self.canvas.setEnabled(True)
        self.doc.frame_size = list(self.frames.size)
        self.doc.seek(frame, self.doc.time_for(frame))
        self.apply_pending_copy()
        self.refine = None
        if self.canvas.zoom > 1.0 and not self.play_timer.isActive():
            self.refine_frame(frame)

    def refine_frame(self, frame):
        """Ask the decoder for *frame* at full size, shown when it arrives if the view is still on it."""
        if self.player.source().isEmpty() or self.frames is None or self.frames.scale >= 1.0: return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: return
        self.refine = frame
        start = self.doc.time_for(frame)
        span = self.doc.time_for(frame+1)-start if frame+1 < self.doc.frame_count else 1/self.doc.fps
        self.player.setPosition(round((start+span/2)*1000))

    def apply_pending_copy(self):
        if self.pending_copy:
            track_id, box = self.pending_copy; self.pending_copy = None
            track = next((tr for tr in self.doc.tracks if tr.track_id == track_id), None)
            if track: self.doc.put_box(box, track)

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
        if self.refine is not None and self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            if abs(native_frame - self.refine) <= 1 and self.doc.frame == self.refine:
                self.canvas.image = image; self.refine = None; self.canvas.update()
            return                                 # the full-size copy of the cached frame on screen
        if (self.pending_frame is None and self.cached(self.doc.frame) is not None
                and self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState):
            return                                 # a late decoder frame never moves a view the cache already serves
        if self.pending_frame is not None:
            if abs(native_frame - self.pending_frame) > 1:
                return  # A decoder can deliver an earlier seek's frame while scrubbing.
            self.settle_seek(t, image); return   # the asked frame, give or take the decoder's rounding
        self.canvas.image = image
        self.canvas.setEnabled(True)
        self.doc.seek(native_frame, t)

    def settle_seek(self, t_sec, image=None):
        """The pending seek is done: the clicked frame is shown (with the decoder's own time for it)."""
        frame, self.pending_frame = self.pending_frame, None
        self.seek_watchdog.stop()
        if frame is None or not self.doc: return
        if image is not None: self.canvas.image = image
        self.canvas.setEnabled(True)
        self.doc.seek(frame, t_sec)
        self.apply_pending_copy()

    def seek_timed_out(self):
        """No decoded frame answered the seek: nudge the paused decoder once (play, pause), then settle anyway."""
        if self.pending_frame is None or not self.doc: return
        if not self.seek_nudged:
            self.seek_nudged = True
            self.player.play(); self.player.pause(); self.seek_watchdog.start(); return
        self.settle_seek(self.doc.time_for(self.pending_frame))

    def position_changed(self):
        if not self.doc: return
        self.position.setText(f'Frame {self.doc.frame+1} / {self.doc.frame_count}   ·   {self.doc.t_sec:.3f} s')
        self.canvas.update(); self.timeline.update()

    def seek(self, frame):
        if not self.doc: return
        self.player.pause()
        frame = max(0, min(self.doc.frame_count-1, frame))
        if self.cached(frame) is not None:
            if self.play_timer.isActive(): self.play_timer.stop(); self.play.setText('Play')
            self.show_cached(frame); return
        if self.player.source().isEmpty():
            self.doc.seek(frame); return
        if frame == self.doc.frame and self.pending_frame is None: return
        self.pending_frame = frame; self.canvas.setEnabled(False)
        self.seek_nudged = False; self.seek_watchdog.start()
        self.position.setText(f'Seeking frame {frame+1}…')
        # Qt's paused decoder includes a frame's end timestamp in its seek interval.
        # Seek inside the requested frame, then use the decoded presentation time.
        start = self.doc.time_for(frame)
        span = self.doc.time_for(frame+1)-start if frame+1 < self.doc.frame_count else 1/self.doc.fps
        self.player.setPosition(round((start+span/2)*1000))

    def step(self, delta):
        if self.doc: self.seek((self.pending_frame if self.pending_frame is not None else self.doc.frame)+delta)

    def toggle_play(self):
        if self.play_timer.isActive(): self.stop_cached(); return
        if (self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState and self.doc
                and self.cached(self.doc.frame) is not None):
            if self.doc.frame+1 >= self.doc.frame_count or self.cached(self.doc.frame+1) is None:
                self.show_cached(0)                                            # at the end: from the start
            self.play_timer.start(max(10, round(1000/(self.doc.fps or 7)))); self.play.setText('Pause'); return
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

    def toggle_boxes(self):
        SESSION['boxes'] = not SESSION['boxes']
        self.boxes_toggle.setChecked(SESSION['boxes']); self.canvas.update()

    def toggle_track_hidden(self):
        if self.doc and self.doc.track:
            self.doc.hidden ^= {self.doc.selected}
            self.canvas.update(); self.timeline.update(); self.refresh_tracks(); self.selection_changed()

    def split_track(self):
        if self.doc and self.doc.split():
            self.save_state.setText('Split: a new track starts at this frame')

    def merge_track(self):
        if not self.doc or not self.doc.track:
            return
        if not self.doc.merge():
            self.show_error('Nothing to merge with: no other track of this kind that is never on screen at the same time.')

    def delete_selected(self):
        if self.doc and self.doc.delete_track():
            self.canvas.update(); self.timeline.update()

    def deselect(self):
        self.doc.selected = None; self.selection_changed()

    def selection_changed(self):
        if self.doc:
            name = self.doc.track.label if self.doc.track else self.doc.current_class
            self.classes.setCurrentIndex(CLASSES.index(name)); self.canvas.update(); self.timeline.update()
            self.delete_box.setEnabled(self.doc.track is not None)
            for b in (self.split_button, self.merge_button, self.hide_track):
                b.setEnabled(self.doc.track is not None)
            self.hide_track.setText('Show track  H' if self.doc.selected in self.doc.hidden else 'Hide track  H')
            self.boxes_toggle.setChecked(SESSION['boxes'])
            self.show_track()

    def track_source_kind(self, track):
        """The chip of a track: who drew it (the preload's source until a person edits it)."""
        if track.source not in ('yolo', 'suggestion'):
            return 'admin'
        return self.doc.preload_source if self.doc.preload_source in ('tracker', 'dataset') else 'yolo'

    def refresh_tracks(self):
        """The track list: one row per track, in order of first appearance, with who drew it."""
        names = self.doc.display_names()
        first = lambda t: min((k.t_sec for k in t.keyframes), default=float('inf'))  # noqa: E731
        self.track_list.blockSignals(True); self.track_list.clear()
        for tr in sorted(self.doc.tracks, key=lambda t: (first(t), t.track_id)):
            source = self.track_source_kind(tr)
            who = 'checked' if source == 'admin' else machine_name(source)
            hidden = '  ·  hidden' if tr.track_id in self.doc.hidden else ''
            item = QListWidgetItem(f'{names.get(tr.track_id, tr.label)}   ·   {who}   ·   {len(tr.keyframes)} keyframes{hidden}')
            item.setData(Qt.ItemDataRole.UserRole, tr.track_id); item.setToolTip(f'id {tr.track_id}')
            self.track_list.addItem(item)
            if tr.track_id == self.doc.selected:
                item.setSelected(True)
        self.track_list.blockSignals(False)

    def track_clicked(self, item):
        if self.doc:
            self.doc.selected = item.data(Qt.ItemDataRole.UserRole); self.selection_changed()

    def show_track(self):
        tr = self.doc.track if self.doc else None
        for row in range(self.track_list.count()):
            item = self.track_list.item(row)
            item.setSelected(bool(tr) and item.data(Qt.ItemDataRole.UserRole) == tr.track_id)
        if tr is None:
            self.track_name.setText('No track selected'); self.track_source.show_source(''); self.track_detail.setText('')
            return
        self.track_name.setText(self.doc.display_name(tr))
        source = self.track_source_kind(tr)
        self.track_source.show_source(source, 'you' if source == 'admin' else '')
        shown = [k.t_sec for k in tr.keyframes if k.enabled]
        span = f'{min(shown):.1f}–{max(k.t_sec for k in tr.keyframes):.1f} s' if shown else 'hidden'
        self.track_detail.setText(f'{tr.label}  ·  {len(tr.keyframes)} keyframes  ·  {span}'
                                  + ('  ·  hidden from view' if tr.track_id in self.doc.hidden else ''))

    def text_changed(self, *_):
        # the description belongs to Tag · AI: carried unchanged; only the clip's training checks change here
        if self.doc: self.doc.set_text(self.doc.description, self.drop.isChecked(), self.needs_review.isChecked())

    def changed(self):
        self.refresh()
        if not self.conflicted and self.doc.dirty: self.autosave.start()
        else: self.autosave.stop()

    def refresh(self):
        if not self.doc: return
        for widget, value in ((self.drop, self.doc.drop_clip), (self.needs_review, self.doc.needs_review)):
            widget.blockSignals(True); widget.setChecked(value); widget.blockSignals(False)
        self.status.setText(self.doc.annotation.status.upper()); self.version.setText(f'v{self.doc.annotation.version} · Versions')
        self.refresh_tracks()
        unchecked = len(self.doc.unchecked)
        source = self.doc.preload_source if self.doc.preload_source in ('tracker', 'dataset') else 'yolo'
        self.boxes_source.show_source(source if unchecked else '')
        machine = machine_name(source).lower() if source != 'yolo' else 'YOLO'
        self.boxes_note.setText('No YOLO boxes for this clip' if not self.doc.tracks and not self.doc.annotation.version
                                else f'{unchecked} {machine} box track{"s" if unchecked != 1 else ""} not checked yet'
                                if unchecked else '')
        self.delete_box.setEnabled(self.doc.track is not None)
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
        key = getattr(self.recording, 'clip_key', None)
        if key:
            self.writer.start(lambda: self.backend.save_clip_boxes(key, request))
        else:
            self.writer.start(lambda: self.backend.save_annotation(eid, request))
        self.update_save_state()

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
            elif pending[0] == 'clip': self.open_clip(pending[1])
            else: self.open_index(pending[1])

    def submit(self):
        if self.doc: self.save('submitted')

    def resolve_conflict(self, keep):
        if self.reader.busy: return
        self.read_kind = 'keep' if keep else 'reload'
        eid = self.doc.annotation.event_id; self.read_event = eid
        self.reader.start(lambda: self.backend.annotation(eid))

    def versions(self):
        if not self.doc or self.reader.busy or getattr(self.recording, 'clip_key', None): return
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

    def showEvent(self, event):
        self.clock.start(); super().showEvent(event)

    def hideEvent(self, event):
        self.player.pause(); self.save(); self.clock.stop(); self.first_frame.stop()
        if self.play_timer.isActive(): self.stop_cached()
        super().hideEvent(event)

    def closeEvent(self, event):
        """Closed for good: nothing of it keeps running (timers, the player, the frame decode)."""
        for timer in (self.clock, self.first_frame, self.play_timer, self.seek_watchdog):
            timer.stop()
        self.frames_for = None                                  # the background decode stops at its next frame
        self.player.stop(); self.player.setSource(QUrl())
        super().closeEvent(event)
