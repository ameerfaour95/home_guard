"""QMediaPlayer → QVideoSink → video canvas with an independent transparent overlay."""
from datetime import timedelta
from PySide6.QtCore import Qt, QRectF, QUrl
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QShortcut, QKeySequence, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink, QAudioOutput
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QComboBox, QLabel
from .event_logic import video_rect, map_box, nearest_frame, provenance
from .formatting import local_time
from .theme import PALETTES
from .widgets.common import label, button


class BoxOverlay(QWidget):
    def __init__(self, canvas, theme):
        super().__init__(canvas)
        self.canvas, self.t = canvas, PALETTES[theme]
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.frames, self.times, self.status = [], [], 'none'
        self.position_ms, self.offset_ms, self.enabled = 0, 0, True
        self.message = None

    def set_detections(self, detections):
        self.frames = sorted(detections.frames, key=lambda f: f.t_sec)
        self.times = [f.t_sec for f in self.frames]
        self.status = detections.provenance
        self.message = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self.message or provenance(self.status, self.frames)
        if not self.enabled:
            text += ' · Hidden'
        frame = nearest_frame(self.frames, self.position_ms, self.offset_ms, self.times)
        if self.enabled and frame and frame.status == 'not_run':
            text += ' · Detector did not run on this frame'
        if self.enabled and frame and frame.status == 'ran':
            for box in frame.boxes:
                token = 'action' if box.cls == 0 else 'warning' if box.cls in (1, 2, 3, 5, 7) else 'ok'
                color = QColor(self.t[token])
                rect = QRectF(*map_box(box.xyxy, self.canvas.display_rect()))
                p.setPen(QPen(color, 2)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(rect)
                caption = box.label+(f'  {box.conf:.0%}' if box.conf is not None else '')
                width = p.fontMetrics().horizontalAdvance(caption)+12
                tag = QRectF(rect.x(), max(0, rect.y()-24), width, 24)
                p.fillRect(tag, color); p.setPen(QColor(self.t['bg'])); p.drawText(tag, Qt.AlignmentFlag.AlignCenter, caption)
        bg = QRectF(12, 12, min(self.width()-24, p.fontMetrics().horizontalAdvance(text)+20), 28)
        p.fillRect(bg, QColor(12, 18, 24, 220)); p.setPen(QColor('#edf4f6'))
        p.drawText(bg.adjusted(10, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(bg.width()-14)))


class VideoCanvas(QWidget):
    def __init__(self, theme):
        super().__init__()
        self.image = QImage(); self.frame_size = (640, 360)
        self.message = 'Loading recording…'
        self.setMinimumSize(320, 180)
        self.overlay = BoxOverlay(self, theme)

    def display_rect(self):
        size = (self.image.width(), self.image.height()) if not self.image.isNull() else self.frame_size
        return video_rect(self.width(), self.height(), *size)

    def resizeEvent(self, event):
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor('#070c10'))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.image.isNull():
            p.drawImage(QRectF(*self.display_rect()), self.image)
        else:
            p.setPen(QColor('#a0adb8')); p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)

    def frame_changed(self, frame):
        if frame.isValid():
            image = frame.toImage()
            if not image.isNull():
                self.image = image
                if frame.startTime() >= 0:
                    self.overlay.position_ms = frame.startTime()/1000
                self.update(); self.overlay.update()


class FilmstripSlider(QSlider):
    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal)
        self.setMouseTracking(True)
        self.sprite, self.metadata = QPixmap(), {}
        self.preview = QLabel(None, Qt.WindowType.ToolTip)
        self.preview.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    def set_filmstrip(self, data=None, metadata=None):
        self.sprite = QPixmap()
        if data:
            self.sprite.loadFromData(data)
        self.metadata = metadata or {}
        self.preview.hide()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        m = self.metadata
        if self.sprite.isNull() or not all(m.get(k) for k in ('tile_w', 'tile_h', 'count', 'fps')):
            return
        seconds = max(0., min(1., event.position().x()/max(1, self.width())))*self.maximum()/1000
        idx = min(m['count']-1, int(seconds*m['fps']))
        columns = max(1, self.sprite.width()//m['tile_w'])
        pix = self.sprite.copy((idx % columns)*m['tile_w'], (idx//columns)*m['tile_h'], m['tile_w'], m['tile_h'])
        self.preview.setPixmap(pix); self.preview.adjustSize()
        point = event.globalPosition().toPoint(); point.setY(point.y()-pix.height()-24)
        self.preview.move(point); self.preview.show()

    def leaveEvent(self, event):
        self.preview.hide(); super().leaveEvent(event)

    def hideEvent(self, event):
        self.preview.hide(); super().hideEvent(event)


class EventPlayer(QWidget):
    def __init__(self, theme='dark'):
        super().__init__()
        self.recording, self.zone = None, 'UTC'
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self); self.audio.setVolume(.5); self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink(self); self.player.setVideoSink(self.sink)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(10)
        self.canvas = VideoCanvas(theme); layout.addWidget(self.canvas, 1)
        self.sink.videoFrameChanged.connect(self.canvas.frame_changed)
        self.scrubber = FilmstripSlider(); self.scrubber.setAccessibleName('Recording position')
        self.scrubber.sliderMoved.connect(self.player.setPosition); layout.addWidget(self.scrubber)
        transport = QHBoxLayout(); transport.setSpacing(8)
        self.play = button('Play', self.toggle_play); transport.addWidget(self.play)
        self.previous_frame = button('‹', lambda: self.step(-1)); self.previous_frame.setToolTip('Previous frame  ,')
        self.next_frame = button('›', lambda: self.step(1)); self.next_frame.setToolTip('Next frame  .')
        transport.addWidget(self.previous_frame); transport.addWidget(self.next_frame)
        self.timestamp = label('00:00 / 00:00', 'muted'); transport.addWidget(self.timestamp); transport.addStretch()
        self.speed = QComboBox()
        for text, speed in [('0.5×', .5), ('1×', 1.), ('2×', 2.)]:
            self.speed.addItem(text, speed)
        self.speed.setCurrentIndex(1); self.speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.speed.currentData()))
        transport.addWidget(self.speed)
        self.boxes = button('Boxes  D', self.toggle_boxes); self.boxes.setCheckable(True); self.boxes.setChecked(True); transport.addWidget(self.boxes)
        layout.addLayout(transport)
        alignment = QHBoxLayout(); alignment.addWidget(label('Box offset', 'muted'))
        self.offset = QSlider(Qt.Orientation.Horizontal); self.offset.setRange(-500, 500); self.offset.setSingleStep(10)
        self.offset.setMaximumWidth(220); self.offset.setAccessibleName('Box timing offset in milliseconds')
        self.offset.setToolTip('Positive offset selects later detections. Range −500 to +500 ms.')
        self.offset.valueChanged.connect(self.offset_changed); alignment.addWidget(self.offset)
        self.offset_text = label('0 ms', 'muted'); self.offset_text.setMinimumWidth(60); alignment.addWidget(self.offset_text)
        alignment.addStretch(); self.local_clock = label('', 'muted'); alignment.addWidget(self.local_clock)
        layout.addLayout(alignment)
        self.error = label('', 'error', True); self.error.hide(); layout.addWidget(self.error)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.scrubber.setMaximum)
        self.player.playbackStateChanged.connect(lambda state: self.play.setText('Pause' if state == QMediaPlayer.PlaybackState.PlayingState else 'Play'))
        self.player.errorOccurred.connect(lambda *_: self.media_failed())

    def reset(self, event, zone):
        self.player.stop(); self.player.setSource(QUrl())
        self.recording, self.zone = event, zone
        self.canvas.image = QImage(); self.canvas.frame_size = tuple(event.frame_size or [640, 360]); self.canvas.message = 'Loading recording…'
        self.canvas.overlay.frames = []; self.canvas.overlay.times = []; self.canvas.overlay.status = event.completeness.boxes
        self.canvas.overlay.message = 'Loading saved boxes…'; self.canvas.overlay.position_ms = 0
        self.canvas.update(); self.canvas.overlay.update()
        self.offset.setValue(0); self.scrubber.setValue(0); self.scrubber.set_filmstrip(); self.error.hide()
        self.set_transport(False); self.position_changed(0)

    def set_transport(self, enabled):
        for widget in (self.play, self.previous_frame, self.next_frame, self.scrubber, self.speed):
            widget.setEnabled(enabled)

    def open_url(self, url):
        self.player.setSource(QUrl(url)); self.set_transport(True)
        # Prime the decoder with an actual video frame, then remain paused.
        self.player.play(); self.player.pause()

    def media_failed(self, message='Recording could not be played. Reopen the event to request fresh access.'):
        self.canvas.message = message; self.canvas.update()
        self.error.setText(message); self.error.show(); self.set_transport(False)

    def position_changed(self, position):
        if not self.scrubber.isSliderDown():
            self.scrubber.setValue(position)
        duration = self.player.duration() or int((self.recording.duration_sec or 0)*1000) if self.recording else 0
        def stamp(ms):
            return f'{ms//60000:02}:{ms//1000 % 60:02}.{ms % 1000//100}'
        self.timestamp.setText(f'{stamp(position)} / {stamp(duration)}')
        if self.recording:
            self.local_clock.setText(local_time(self.recording.start_utc+timedelta(milliseconds=position), self.zone))

    def toggle_play(self):
        if not self.play.isEnabled():
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def step(self, direction):
        if self.recording and self.play.isEnabled():
            self.player.pause()
            fps = self.recording.fps or 25
            frame = round(self.player.position()*fps/1000)+direction
            self.player.setPosition(max(0, min(self.player.duration(), round(frame*1000/fps))))

    def toggle_boxes(self):
        self.canvas.overlay.enabled = not self.canvas.overlay.enabled
        self.boxes.setChecked(self.canvas.overlay.enabled); self.canvas.overlay.update()

    def offset_changed(self, value):
        self.offset_text.setText(f'{value:+d} ms' if value else '0 ms')
        self.canvas.overlay.offset_ms = value; self.canvas.overlay.update()

    def hideEvent(self, event):
        self.player.pause(); super().hideEvent(event)
