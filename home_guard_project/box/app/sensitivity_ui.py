"""Accessible per-type sliders and independent live camera sensitivity state."""
from concurrent.futures import ThreadPoolExecutor
import time
from PySide6.QtCore import Qt, Signal, QTimer, QRectF
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QPushButton
from .alert_types import TYPES
from .live_settings import AppliedState, slider_conf, conf_slider
from .strings import tr
from .type_icons import draw_type_icon
from .zone_editor import colors

RECOMMENDED = dict(person=.5, vehicle=.7, animal=.6)


class TypeIcon(QWidget):
    def __init__(self, kind):
        super().__init__(); self.kind = kind; self.setFixedSize(26, 28)

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(1, 2); p.scale(.65, .65)
        p.setPen(QPen(QColor(colors(self)['action']), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush); draw_type_icon(p, self.kind)


class SensitivitySlider(QSlider):
    committed = Signal()

    def __init__(self, kind):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(1, 19); self.setSingleStep(1); self.setPageStep(1)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(tr('alert_type_' + kind) + ' ' + tr('sensitivity'))
        self.setMinimumHeight(26)
        self.setStyleSheet('QSlider { background: transparent; } QSlider:focus { border: none; }')
        self.sliderReleased.connect(self.committed)

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        if not event.isAutoRepeat() and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Home, Qt.Key.Key_End, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self.committed.emit()

    def mouseReleaseEvent(self, event):
        dragging = self.isSliderDown()
        super().mouseReleaseEvent(event)
        if not dragging and event.button() == Qt.MouseButton.LeftButton:
            self.committed.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.hasFocus():
            p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(colors(self)['action']), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 5, 5)


class SensitivitySliders(QWidget):
    changed = Signal(str, float)

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(12)
        self.sliders = {}; self.readouts = {}
        for kind in TYPES:
            row = QVBoxLayout(); row.setSpacing(2)
            heading = QHBoxLayout(); heading.setSpacing(8)
            heading.addWidget(TypeIcon(kind)); heading.addWidget(QLabel(tr('alert_type_' + kind))); heading.addStretch()
            readout = QLabel(); readout.setObjectName('muted'); heading.addWidget(readout)
            row.addLayout(heading)
            slider = SensitivitySlider(kind); row.addWidget(slider)
            ends = QHBoxLayout()
            for index, key in enumerate(('catches_more', 'fewer_false_alarms')):
                if index: ends.addStretch()
                text = QLabel(tr(key)); text.setObjectName('muted'); ends.addWidget(text)
            row.addLayout(ends); root.addLayout(row)
            self.sliders[kind] = slider; self.readouts[kind] = readout
            slider.valueChanged.connect(lambda value, k=kind: self.update_readout(k, value))
            slider.committed.connect(lambda k=kind: self.changed.emit(k, slider_conf(self.sliders[k].value())))
        self.set_values(RECOMMENDED)

    def update_readout(self, kind, value):
        text = tr('certainty_required', percent=round(slider_conf(value)*100))
        self.readouts[kind].setText(text); self.sliders[kind].setAccessibleDescription(text)

    def set_values(self, values):
        for kind, slider in self.sliders.items():
            slider.setValue(conf_slider(values[kind]))
            self.update_readout(kind, slider.value())


class CameraSensitivityPanel(QWidget):
    saved = Signal(object)

    def __init__(self, controls, name):
        super().__init__()
        self.controls, self.name = controls, name
        self.house = None; self.own = None; self.pending = None
        self.ack = AppliedState(); self.future = None; self.operation = ''; self.last_status = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        root = QVBoxLayout(self); root.setContentsMargins(0, 8, 0, 0); root.setSpacing(10)
        title = QLabel(tr('sensitivity')); title.setObjectName('section'); root.addWidget(title)
        self.default = QRadioButton(tr('sensitivity_house')); self.custom = QRadioButton(tr('sensitivity_camera'))
        group = QButtonGroup(self); group.addButton(self.default); group.addButton(self.custom)
        choices = QHBoxLayout(); choices.addWidget(self.default, 1); choices.addWidget(self.custom, 1); root.addLayout(choices)
        self.house_note = QLabel(''); self.house_note.setWordWrap(True); self.house_note.setObjectName('muted'); root.addWidget(self.house_note)
        self.sliders = SensitivitySliders(); root.addWidget(self.sliders); self.sliders.hide()
        self.note = QLabel(''); self.note.setObjectName('muted'); self.note.setWordWrap(True); self.note.setTextFormat(Qt.TextFormat.PlainText); root.addWidget(self.note)
        self.retry = QPushButton(tr('retry')); self.retry.setObjectName('textAction'); root.addWidget(self.retry, 0, Qt.AlignmentFlag.AlignLeft); self.retry.hide()
        self.default.clicked.connect(lambda: self.apply(None))
        self.custom.clicked.connect(self.choose_custom)
        self.sliders.changed.connect(self.change_type)
        self.retry.clicked.connect(lambda: self.apply(self.pending))
        if name is None:
            self.default.hide(); self.custom.hide()
            self.reset = QPushButton(tr('reset_sensitivity')); self.reset.setObjectName('textAction')
            self.reset.clicked.connect(lambda: self.apply(RECOMMENDED)); root.addWidget(self.reset, 0, Qt.AlignmentFlag.AlignLeft)
            self.reset.setEnabled(False)
        self.set_controls_enabled(False)
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(150)

    def load(self, data):
        self.house = data.get('house_sensitivity')
        self.own = next((r.get('sensitivity') for r in data['cameras'] if r['name'] == self.name), None)
        if not self.house:
            self.house_note.setText(tr('sensitivity_unavailable')); return
        self.house_note.setText(' · '.join(f"{tr('alert_type_' + k)} {self.house[k]:.0%}" for k in TYPES))
        self.sync(); self.set_controls_enabled(True)

    def set_controls_enabled(self, enabled):
        self.default.setEnabled(enabled); self.custom.setEnabled(enabled); self.sliders.setEnabled(enabled)
        if hasattr(self, 'reset'): self.reset.setEnabled(enabled)

    def sync(self):
        self.default.setChecked(self.own is None); self.custom.setChecked(self.own is not None)
        self.sliders.set_values({**self.house, **(self.own or {})})
        self.sliders.setVisible(self.own is not None or self.name is None)

    def choose_custom(self):
        # Reveal first; only the sliders the owner changes become overrides.
        self.sliders.setVisible(True)
        self.note.setText(tr('sensitivity_choose_hint') if self.own is None else '')

    def change_type(self, kind, value):
        self.apply({**(self.own or {}), kind: value})

    def apply(self, value):
        if not self.house or (self.future is not None and self.operation == 'save'): return
        self.pending = value; self.requested = time.time()
        self.ack.expected.clear()
        self.note.setText(tr('applying')); self.note.setStyleSheet('')
        self.retry.hide(); self.set_controls_enabled(False)
        self.operation = 'save'
        self.future = (self.pool.submit(self.controls.set_house_sensitivity, value) if self.name is None else
                       self.pool.submit(self.controls.set_camera_sensitivity, self.name, value))

    def poll(self):
        if self.future is not None:
            if not self.future.done(): return
            future, self.future = self.future, None
            try:
                result = future.result()
                if self.operation == 'save':
                    if self.name is None:
                        self.house = result
                        self.house_note.setText(' · '.join(f"{tr('alert_type_' + k)} {self.house[k]:.0%}" for k in TYPES))
                        self.ack.expected = {('sensitivity', k): v for k, v in self.pending.items()}
                        self.ack.requested = self.requested
                    else:
                        self.own = result
                        self.ack.request_camera_sensitivity(self.name, result, self.requested)
                    self.sync()
                    self.saved.emit(result)
                else:
                    self.show_status(result)
            except Exception as exc:
                if self.operation == 'save':
                    self.sync(); self.note.setText(str(exc)); self.retry.show()
                    self.note.setStyleSheet('color: ' + colors(self)['error'])
                else: self.show_status({})
            self.set_controls_enabled(bool(self.house))
        if self.ack.expected and self.future is None and time.monotonic()-self.last_status > .75:
            self.last_status = time.monotonic(); self.operation = 'status'
            self.future = self.pool.submit(self.controls.alert_reported_status)

    def show_status(self, result):
        status = self.ack.status(result, time.time())
        if status: self.note.setText(tr('live_applied' if status == 'applied' else status))

    def saving(self):
        return self.future is not None and self.operation == 'save'

    def shutdown(self):
        self.timer.stop(); self.pool.shutdown(wait=False, cancel_futures=True)
