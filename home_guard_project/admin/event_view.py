from .formatting import camera_name
import json
from PySide6.QtCore import Qt, Signal, QTimer
from threading import Event
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget
from .backend import AuthError, BackendError, UnsupportedError
from .workers import TaskRunner, closing
from .review_controller import ReviewController
from .event_logic import KINDS
from .formatting import local_time
from .player import EventPlayer
from .ai_record import AiRecord, TextDisclosure
from .widgets.common import label, button


class EventView(QWidget):
    label_requested = Signal(int)
    navigate = Signal(int)
    review_changed = Signal(object)
    session_expired = Signal()

    def __init__(self, backend, role='admin', theme='dark'):
        super().__init__()
        self.backend, self.role, self.zone = backend, role, 'UTC'
        self.recording, self.event_id, self.generation = None, None, 0
        self.runner, self.evidence_runner = [TaskRunner(self) for _ in range(2)]
        self.review_runner = ReviewController(backend, self)
        self.review_runner.optimistic.connect(self.review_optimistic)
        self.cancelled = Event()
        self.open_timer = QTimer(self)
        self.open_timer.setSingleShot(True)
        self.open_timer.setInterval(150)
        self.open_timer.timeout.connect(self.request_event)
        self.asset_runner = TaskRunner(self)
        self.asset_runner.finished.connect(self.assets_loaded)
        self.loaded_assets = set()
        self.runner.finished.connect(self.loaded); self.evidence_runner.finished.connect(self.evidence_loaded); self.review_runner.finished.connect(self.review_done)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(12)
        top = QHBoxLayout(); self.title = label('Select an event', 'section'); top.addWidget(self.title, 1)
        self.label_button = button('Label', lambda: self.event_id is not None and self.label_requested.emit(self.event_id))
        self.label_button.setVisible(role in ('admin', 'labeler')); top.addWidget(self.label_button)
        self.review = button('Reviewed  R', lambda: self.toggle_review('reviewed')); self.review.setCheckable(True); top.addWidget(self.review)
        self.flag = button('Flag  F', lambda: self.toggle_review('flagged')); self.flag.setCheckable(True); top.addWidget(self.flag)
        self.prev = button('←', lambda: self.navigate.emit(-1)); self.prev.setToolTip('Previous event · k / ←')
        self.next = button('→', lambda: self.navigate.emit(1)); self.next.setToolTip('Next event · j / →')
        top.addWidget(self.prev); top.addWidget(self.next); layout.addLayout(top)
        self.banner = label('', 'error', True); self.banner.hide(); layout.addWidget(self.banner)
        self.retry = button('Retry event', lambda: self.open(self.event_id, self.zone)); self.retry.hide(); layout.addWidget(self.retry)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        self.tabs.tabBar().setDrawBase(False)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.player = EventPlayer(theme); self.record = AiRecord(role)
        self.record.assets_requested.connect(self.load_assets)
        self.player.player.errorOccurred.connect(self.refresh_media)
        self.media_retried = False
        self.media_runner = TaskRunner(self)
        self.media_runner.finished.connect(self.media_refreshed)
        self.splitter.addWidget(self.player); self.splitter.addWidget(self.record)
        self.splitter.setStretchFactor(0, 3); self.splitter.setStretchFactor(1, 2); self.splitter.setSizes([740, 380])
        self.tabs.addTab(self.splitter, 'Recording && AI')
        self.raw = None
        if role == 'admin':
            self.raw = TextDisclosure('Raw event metadata')
            self.raw.toggle.setChecked(True)
            self.tabs.addTab(self.raw, 'Raw')
        self.shortcuts = {}
        self.review_handler = None
        self.autoplay = False
        for key, callback in [('J', lambda: self.navigate.emit(1)), ('K', lambda: self.navigate.emit(-1)),
                              ('Left', lambda: self.navigate.emit(-1)), ('Right', lambda: self.navigate.emit(1)),
                              ('R', lambda: self.toggle_review('reviewed')), ('F', lambda: self.toggle_review('flagged')),
                              ('Space', self.player.toggle_play), ('D', self.player.toggle_boxes),
                              (',', lambda: self.player.step(-1)), ('.', lambda: self.player.step(1))]:
            shortcut = QShortcut(QKeySequence(key), self); shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut); shortcut.activated.connect(callback); self.shortcuts[key] = shortcut

    def open(self, event_id, zone='UTC'):
        self.cancelled.set()
        self.cancelled = Event()
        self.loaded_assets.clear()
        self.media_retried = False
        self.event_id, self.zone = event_id, zone
        self.generation += 1
        self.recording = None; self.player.player.stop()
        self.title.setText('Loading event…'); self.tabs.hide(); self.banner.hide(); self.retry.hide()
        self.review.setEnabled(False); self.flag.setEnabled(False)
        self.open_timer.start()

    def hideEvent(self, event):
        self.cancelled.set()
        self.open_timer.stop()
        self.player.player.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if (self.cancelled.is_set() or self.recording is None) and self.event_id is not None:
            self.open(self.event_id, self.zone)

    def request_event(self):
        if self.runner.busy or self.cancelled.is_set() or self.open_timer.isActive() or not self.isVisible():
            return
        eid = self.event_id
        self.pending = self.generation
        self.runner.start(lambda: self.backend.event(eid))

    def loaded(self, event, error):
        if self.pending != self.generation:
            self.request_event(); return
        if self.cancelled.is_set():
            return
        if error:
            self.title.setText('Event unavailable'); self.banner.setText(str(error)); self.banner.show(); self.retry.show()
            if isinstance(error, AuthError):
                self.session_expired.emit()
            return
        self.recording = event
        self.zone = event.timezone
        self.title.setToolTip(event.camera)
        self.title.setText(f'{camera_name(event)}  ·  {KINDS.get(event.kind, "Unknown")}  ·  {local_time(event.start_utc, self.zone)}')
        self.title.setToolTip(event.camera+' · '+local_time(event.start_utc, self.zone))
        self.review.setEnabled(True); self.flag.setEnabled(True)
        self.review.setChecked(event.reviewed); self.flag.setChecked(event.flagged)
        self.player.reset(event, self.zone); self.record.set_event(event, self.zone)
        if self.raw:
            self.raw.text.setPlainText(json.dumps(event.raw_meta, indent=2, ensure_ascii=False))
        self.tabs.show(); self.tabs.setCurrentIndex(0)
        self.load_evidence()

    def load_evidence(self):
        if self.evidence_runner.busy or self.recording is None or self.cancelled.is_set() or not self.isVisible():
            return
        event, purpose = self.recording, 'training' if self.role == 'labeler' else 'support' if self.role == 'support' else 'review'
        self.evidence_pending = self.generation
        cancel = self.cancelled
        def fetch():
            result = dict(detections=None, video=None, images={}, answers={}, filmstrip=None, errors=[], auth=False, unsupported=False)
            def attempt(operation, default=None):
                if cancel.is_set() or closing.is_set():
                    return default
                try:
                    return operation()
                except BackendError as exc:
                    result['errors'].append(str(exc)); result['auth'] |= isinstance(exc, AuthError)
                    result['unsupported'] |= isinstance(exc, UnsupportedError)
                    return default
            result['detections'] = attempt(lambda: self.backend.detections(event.id))
            available = [a for a in event.artifacts if a.available]
            video = next((a for role in ('rendition', 'original_video', 'clip') for a in available if a.role == role), None)
            if video:
                access = attempt(lambda: self.backend.artifact_access(video.id, purpose))
                result['video'] = access.url if access else None
            strip = next((a for a in available if a.role == 'filmstrip' and a.detail), None)
            if strip:
                access = attempt(lambda: self.backend.artifact_access(strip.id, purpose))
                data = attempt(lambda: self.backend.media_bytes(access.url)) if access else None
                if data:
                    result['filmstrip'] = data, strip.detail
            return result
        self.evidence_runner.start(fetch)

    def evidence_loaded(self, result, error):
        if self.evidence_pending != self.generation:
            self.load_evidence(); return
        if self.cancelled.is_set():
            return
        if error:
            self.banner.setText(str(error)); self.banner.show(); self.retry.show()
            self.player.media_failed('Recording access could not be loaded. Retry this event.'); return
        if result['auth']:
            self.session_expired.emit(); return
        if result['detections']:
            self.player.canvas.overlay.set_detections(result['detections'])
        else:
            self.player.canvas.overlay.message = 'Saved boxes could not be loaded'
            self.player.canvas.overlay.update()
        if result['video']:
            self.player.open_url(result['video'])
            if self.autoplay and self.isVisible():
                self.player.audio.setMuted(True); self.player.speed.setCurrentIndex(1)
                self.player.player.setPlaybackRate(1.0); self.player.player.play()
        else:
            self.player.media_failed('Recording access is not available yet on this server.' if result['unsupported'] else 'No copy of this video remains.' if self.recording.completeness.expired else 'No recording is available for this event.')
        if result['filmstrip']:
            self.player.scrubber.set_filmstrip(*result['filmstrip'])
        self.load_assets()
        if result['errors']:
            self.banner.setText('Some saved evidence is unavailable. '+result['errors'][0]); self.banner.show(); self.retry.show()
            if result['unsupported']:
                self.banner.setText('Not available yet: recording access on this server. Saved AI and detections are shown below.')
                self.retry.hide()

    def toggle_review(self, key):
        if self.review_handler:
            self.review_handler(key); return
        if self.recording:
            self.review_runner.submit(self.recording, key)
        self.review.setChecked(bool(self.recording and self.recording.reviewed)); self.flag.setChecked(bool(self.recording and self.recording.flagged))

    def review_done(self, summary, error):
        if error:
            self.banner.setText(str(error)); self.banner.show()
            if isinstance(error, AuthError):
                self.session_expired.emit()
            return
        if self.recording and self.recording.id == summary.id:
            self.recording.reviewed, self.recording.flagged = summary.reviewed, summary.flagged
            self.review.setChecked(summary.reviewed); self.flag.setChecked(summary.flagged)
        self.review_changed.emit(summary)

    def review_optimistic(self, summary):
        if self.recording and self.recording.id == summary.id:
            self.recording.reviewed, self.recording.flagged = summary.reviewed, summary.flagged
            self.review.setChecked(summary.reviewed)
            self.flag.setChecked(summary.flagged)

    def load_assets(self):
        if self.asset_runner.busy or not self.recording or self.cancelled.is_set() or self.evidence_runner.busy:
            return
        images, answers = self.record.visible_assets()
        images, answers = set(images)-self.loaded_assets, set(answers)-self.loaded_assets
        if not images and not answers:
            return
        self.loaded_assets.update(images | answers)
        cancel, self.asset_generation = self.cancelled, self.generation
        purpose = 'training' if self.role == 'labeler' else 'support' if self.role == 'support' else 'review'
        def fetch():
            result = ({}, {})
            for ids, output, text in ((images, result[0], False), (answers, result[1], True)):
                for aid in ids:
                    if cancel.is_set() or closing.is_set():
                        return result
                    access = self.backend.artifact_access(aid, purpose)
                    if cancel.is_set() or closing.is_set():
                        return result
                    data = self.backend.media_bytes(access.url)
                    output[aid] = data.decode('utf-8', errors='replace') if text else data
            return result
        self.asset_runner.start(fetch)

    def assets_loaded(self, result, error):
        if self.asset_generation != self.generation:
            self.load_assets()
            return
        if self.cancelled.is_set():
            return
        if error:
            self.banner.setText(str(error)); self.banner.show()
            self.record.set_assets({aid: b'' for aid in self.record.frames}, {aid: str(error) for aid in self.record.raw_answers})
            if isinstance(error, AuthError): self.session_expired.emit()
        else:
            self.record.set_assets(*result)
            self.load_assets()

    def refresh_media(self, *_):
        if self.media_retried or not self.recording or self.cancelled.is_set() or not self.isVisible():
            return
        video = next((a for role in ('rendition', 'original_video', 'clip') for a in self.recording.artifacts if a.available and a.role == role), None)
        if video is None:
            return
        self.media_retried = True
        self.media_generation, self.media_position = self.generation, self.player.player.position()
        cancel = self.cancelled
        purpose = 'training' if self.role == 'labeler' else 'support' if self.role == 'support' else 'review'
        self.media_runner.start(lambda: None if cancel.is_set() or closing.is_set() else self.backend.artifact_access(video.id, purpose))

    def media_refreshed(self, result, error):
        if self.media_generation != self.generation or self.cancelled.is_set():
            return
        if result and not error:
            self.player.error.hide()
            self.player.open_url(result.url)
            self.player.player.setPosition(self.media_position)
        elif isinstance(error, AuthError):
            self.session_expired.emit()
