import json
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget
from .backend import AuthError, BackendError
from .workers import TaskRunner
from .event_logic import KINDS
from .formatting import local_time
from .player import EventPlayer
from .ai_record import AiRecord, TextDisclosure
from .widgets.common import label, button


class EventView(QWidget):
    navigate = Signal(int)
    review_changed = Signal(object)
    session_expired = Signal()

    def __init__(self, backend, role='admin', theme='dark'):
        super().__init__()
        self.backend, self.role, self.zone = backend, role, 'UTC'
        self.recording, self.event_id, self.generation = None, None, 0
        self.runner, self.evidence_runner, self.review_runner = [TaskRunner(self) for _ in range(3)]
        self.runner.finished.connect(self.loaded); self.evidence_runner.finished.connect(self.evidence_loaded); self.review_runner.finished.connect(self.review_done)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(12)
        top = QHBoxLayout(); self.title = label('Select an event', 'section'); top.addWidget(self.title, 1)
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
        self.splitter.addWidget(self.player); self.splitter.addWidget(self.record)
        self.splitter.setStretchFactor(0, 3); self.splitter.setStretchFactor(1, 2); self.splitter.setSizes([740, 380])
        self.tabs.addTab(self.splitter, 'Recording && AI')
        self.raw = None
        if role == 'admin':
            self.raw = TextDisclosure('Raw event metadata')
            self.raw.toggle.setChecked(True)
            self.tabs.addTab(self.raw, 'Raw')
        for key, callback in [('J', lambda: self.navigate.emit(1)), ('K', lambda: self.navigate.emit(-1)),
                              ('Left', lambda: self.navigate.emit(-1)), ('Right', lambda: self.navigate.emit(1)),
                              ('R', lambda: self.toggle_review('reviewed')), ('F', lambda: self.toggle_review('flagged')),
                              ('Space', self.player.toggle_play), ('D', self.player.toggle_boxes),
                              (',', lambda: self.player.step(-1)), ('.', lambda: self.player.step(1))]:
            shortcut = QShortcut(QKeySequence(key), self); shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut); shortcut.activated.connect(callback)

    def open(self, event_id, zone='UTC'):
        self.event_id, self.zone = event_id, zone
        self.generation += 1
        self.recording = None; self.player.player.stop()
        self.title.setText('Loading event…'); self.tabs.hide(); self.banner.hide(); self.retry.hide()
        self.review.setEnabled(False); self.flag.setEnabled(False)
        self.request_event()

    def request_event(self):
        if self.runner.busy:
            return
        eid = self.event_id
        self.pending = self.generation
        self.runner.start(lambda: self.backend.event(eid))

    def loaded(self, event, error):
        if self.pending != self.generation:
            self.request_event(); return
        if error:
            self.title.setText('Event unavailable'); self.banner.setText(str(error)); self.banner.show(); self.retry.show()
            if isinstance(error, AuthError):
                self.session_expired.emit()
            return
        self.recording = event
        self.title.setText(f'{event.camera}  ·  {KINDS[event.kind]}  ·  {local_time(event.start_utc, self.zone)}')
        self.title.setToolTip(self.title.text())
        self.review.setEnabled(True); self.flag.setEnabled(True)
        self.review.setChecked(event.reviewed); self.flag.setChecked(event.flagged)
        self.player.reset(event, self.zone); self.record.set_event(event, self.zone)
        if self.raw:
            self.raw.text.setPlainText(json.dumps(event.raw_meta, indent=2, ensure_ascii=False))
        self.tabs.show(); self.tabs.setCurrentIndex(0)
        self.load_evidence()

    def load_evidence(self):
        if self.evidence_runner.busy or self.recording is None:
            return
        event, purpose = self.recording, 'training' if self.role == 'labeler' else 'support' if self.role == 'support' else 'review'
        self.evidence_pending = self.generation
        def fetch():
            result = dict(detections=None, video=None, images={}, answers={}, filmstrip=None, errors=[], auth=False)
            def attempt(operation, default=None):
                try:
                    return operation()
                except BackendError as exc:
                    result['errors'].append(str(exc)); result['auth'] |= isinstance(exc, AuthError)
                    return default
            result['detections'] = attempt(lambda: self.backend.detections(event.id))
            available = [a for a in event.artifacts if a.available]
            video = next((a for role in ('rendition', 'original_video', 'clip') for a in available if a.role == role), None)
            if video:
                access = attempt(lambda: self.backend.artifact_access(video.id, purpose))
                result['video'] = access.url if access else None
            for run in event.ai_runs:
                for aid in run.input_frame_artifact_ids:
                    def frame(aid=aid):
                        return self.backend.media_bytes(self.backend.artifact_access(aid, purpose).url)
                    result['images'][aid] = attempt(frame, b'')
                if run.raw_text_artifact_id:
                    def answer():
                        return self.backend.media_bytes(self.backend.artifact_access(run.raw_text_artifact_id, purpose).url).decode('utf-8', errors='replace')
                    result['answers'][run.raw_text_artifact_id] = attempt(answer, 'Saved answer unavailable.')
            strip = next((a for a in available if a.role == 'filmstrip' and a.detail), None)
            if strip:
                data = attempt(lambda: self.backend.media_bytes(self.backend.artifact_access(strip.id, purpose).url))
                if data:
                    result['filmstrip'] = data, strip.detail
            return result
        self.evidence_runner.start(fetch)

    def evidence_loaded(self, result, error):
        if self.evidence_pending != self.generation:
            self.load_evidence(); return
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
        else:
            self.player.media_failed('Recording expired or unavailable.' if self.recording.completeness.expired else 'No recording is available for this event.')
        if result['filmstrip']:
            self.player.scrubber.set_filmstrip(*result['filmstrip'])
        self.record.set_assets(result['images'], result['answers'])
        if result['errors']:
            self.banner.setText('Some saved evidence is unavailable. '+result['errors'][0]); self.banner.show(); self.retry.show()

    def toggle_review(self, key):
        if self.recording and not self.review_runner.busy:
            eid, value = self.recording.id, not getattr(self.recording, key)
            self.review_runner.start(lambda: self.backend.review(eid, **{key: value}))
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
