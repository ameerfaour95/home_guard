"""Shared, keyboard-accessible alert tiles and live camera choices."""
from concurrent.futures import ThreadPoolExecutor
import time
from PySide6.QtCore import Qt, QRectF, QSize, Signal, QTimer, QVariantAnimation
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath
from PySide6.QtWidgets import QWidget, QAbstractButton, QVBoxLayout, QHBoxLayout, QBoxLayout, QLabel, QDialog, QRadioButton, QPushButton, QSizePolicy
from .alert_types import TYPES, ordered_types
from .live_settings import AppliedState
from .strings import tr
from .zone_editor import colors, alpha, ZonePill
from . import motion


def type_names(value):
    return ', '.join(tr('alert_type_' + kind) for kind in ordered_types(value))


def alert_summary(data, saved):
    settings = data.get('settings', {}) if isinstance(data, dict) else {}
    if not isinstance(settings, dict): settings = {}
    try: names = type_names(settings.get('alert_on', saved))
    except ValueError: names = type_names(saved)
    own = settings.get('camera_alert_on', {})
    count = len(own) if isinstance(own, dict) else 0
    return names + (' · ' + tr('alert_custom_one' if count == 1 else 'alert_custom_many', count=count) if count else '')


class ElidedLabel(QLabel):
    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self.setWordWrap(False)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setPen(self.palette().windowText().color())
        p.drawText(self.contentsRect(), Qt.AlignmentFlag.AlignVCenter,
                   self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width()))

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)
        self.setAccessibleName(text)


class ResponsiveColumns(QWidget):
    def __init__(self):
        super().__init__()
        self.row = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(16)

    def resizeEvent(self, event):
        self.row.setDirection(QBoxLayout.Direction.TopToBottom if self.width() < 1000 else QBoxLayout.Direction.LeftToRight)
        super().resizeEvent(event)


class AlertTile(QAbstractButton):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.compact = False
        self.pulse = 0.
        self.setText(tr('alert_type_' + kind))
        self.setAccessibleDescription(tr('alert_hint_' + kind))
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setProperty('handlesMotion', True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(154)
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(motion.PANE_MS * 2)
        self.animation.setEasingCurve(motion.EASING)
        self.animation.setStartValue(1.)
        self.animation.setEndValue(0.)
        self.animation.valueChanged.connect(self.advance)

    def advance(self, value):
        self.pulse = value
        self.update()

    def refuse(self):
        self.animation.stop()
        self.animation.start()

    def sizeHint(self): return QSize(208, 90 if self.compact else 154)
    def minimumSizeHint(self): return QSize(160, self.sizeHint().height())

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click(); event.accept(); return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(1. if self.isEnabled() else .38)
        selected = self.isChecked()
        p.setBrush(alpha(t['action'], .12) if selected else QColor(t['surface']))
        p.setPen(QPen(QColor(t['warning'] if self.pulse > .1 else t['action'] if selected or self.hasFocus() else t['border']), 2 if selected or self.hasFocus() else 1))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(2, 2, -2, -2), 12, 12)
        if self.hasFocus():
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor(t['action']), 1, Qt.PenStyle.DashLine))
            p.drawRoundedRect(QRectF(self.rect()).adjusted(6, 6, -6, -6), 9, 9)
        p.save(); p.translate(18, 24 if self.compact else 18)
        p.setPen(QPen(QColor(t['action'] if selected else t['secondary']), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.kind == 'person':
            p.drawEllipse(QRectF(10, 0, 12, 12))
            p.drawRoundedRect(QRectF(5, 17, 22, 17), 7, 7)
        elif self.kind == 'vehicle':
            p.drawRoundedRect(QRectF(1, 13, 32, 16), 4, 4)
            p.drawLine(5, 13, 10, 4); p.drawLine(10, 4, 25, 4); p.drawLine(25, 4, 30, 13)
            p.drawLine(7, 19, 10, 19); p.drawLine(24, 19, 27, 19)
            p.drawLine(7, 29, 7, 33); p.drawLine(27, 29, 27, 33)
        else:
            for x, y in ((1, 10), (9, 1), (21, 1), (29, 10)):
                p.drawEllipse(QRectF(x, y, 6, 9))
            paw = QPainterPath(); paw.moveTo(8, 29); paw.cubicTo(7, 22, 14, 16, 18, 17); paw.cubicTo(23, 16, 30, 25, 27, 30); paw.cubicTo(24, 35, 20, 30, 18, 31); paw.cubicTo(13, 34, 8, 34, 8, 29)
            p.drawPath(paw)
        p.restore()
        x, y = (68, 16) if self.compact else (18, 65)
        font = QFont(self.font()); font.setPixelSize(17); font.setWeight(QFont.Weight.DemiBold); p.setFont(font)
        p.setPen(QColor(t['text'])); p.drawText(QRectF(x, y, self.width()-x-34, 24), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
        font.setPixelSize(13); font.setWeight(QFont.Weight.Normal); p.setFont(font); p.setPen(QColor(t['secondary']))
        p.drawText(QRectF(x, y+30, self.width()-x-16, 50), Qt.TextFlag.TextWordWrap, tr('alert_hint_' + self.kind))
        p.setPen(QPen(QColor(t['action'] if selected else t['border']), 1.5)); p.setBrush(QColor(t['action']) if selected else Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(self.width()-34, 17, 17, 17))
        if selected:
            p.setPen(QPen(QColor(t['bg']), 1.8)); p.drawLine(self.width()-30, 25, self.width()-26, 29); p.drawLine(self.width()-26, 29, self.width()-21, 22)


class AlertTiles(QWidget):
    changed = Signal(str)
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(8)
        self.row = QBoxLayout(QBoxLayout.Direction.LeftToRight); self.row.setSpacing(10); root.addLayout(self.row)
        self.tiles = {}
        for kind in TYPES:
            tile = AlertTile(kind); self.tiles[kind] = tile; self.row.addWidget(tile, 1)
            tile.clicked.connect(lambda checked, k=kind: self.toggle(k))
        self.hint = QLabel(''); self.hint.setWordWrap(True); self.hint.setObjectName('muted'); self.hint.hide(); root.addWidget(self.hint)
        self.set_value('person')

    def value(self): return ','.join(k for k in TYPES if self.tiles[k].isChecked())

    def set_value(self, value):
        selected = ordered_types(value)
        for kind, tile in self.tiles.items(): tile.setChecked(kind in selected)

    def toggle(self, kind):
        if not self.value():
            self.tiles[kind].setChecked(True); self.tiles[kind].refuse()
            self.hint.setText(tr('alert_keep_one')); self.hint.show(); return
        self.hint.hide(); self.changed.emit(self.value())

    def resizeEvent(self, event):
        compact = self.width() < 590
        self.row.setDirection(QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight)
        for tile in self.tiles.values():
            tile.compact = compact; tile.setFixedHeight(90 if compact else 154); tile.update()
        super().resizeEvent(event)


class CameraAlertButton(QAbstractButton):
    def __init__(self, house, own):
        super().__init__()
        self.setProperty('handlesMotion', True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(80); self.setFixedHeight(32)
        self.refresh(house, own)

    def refresh(self, house, own):
        self.custom = own is not None
        self.setText(tr('alert_camera_custom' if self.custom else 'alert_camera_default', types=type_names(own if self.custom else house)))
        self.setToolTip(self.text()); self.setAccessibleName(self.text()); self.update()

    def sizeHint(self): return QSize(240, 32)
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter): self.click(); event.accept(); return
        super().keyPressEvent(event)
    def paintEvent(self, event):
        t = colors(self); p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(1 if self.isEnabled() else .4)
        p.setBrush(QColor(t['raised'])); p.setPen(QPen(QColor(t['action'] if self.hasFocus() else t['border']), 1))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 8, 8)
        if self.custom:
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['action'])); p.drawEllipse(QRectF(10, 13, 6, 6))
        font = QFont(self.font()); font.setPixelSize(12); p.setFont(font); p.setPen(QColor(t['secondary']))
        left = 23 if self.custom else 10
        text = p.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width()-left-10)
        p.drawText(self.rect().adjusted(left, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter, text)


class CameraAlertsDialog(QDialog):
    saved = Signal(object)
    house_saved = Signal(object)
    def __init__(self, controls, name, house, own, parent=None):
        super().__init__(parent)
        self.controls, self.name = controls, name
        self.house, self.own = house, own
        self.ack = AppliedState(); self.future = None; self.operation = ''; self.loading = True
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.setWindowTitle(tr('alert_camera_title', camera=name.replace('_', ' ').title()) if name else tr('alert_house_title'))
        self.setModal(True); self.resize(760, 420)
        root = QVBoxLayout(self); root.setContentsMargins(28, 24, 28, 24); root.setSpacing(14)
        title = QLabel(self.windowTitle()); title.setObjectName('section'); root.addWidget(title)
        self.default = QRadioButton(tr('alert_use_default')); self.custom = QRadioButton(tr('alert_choose_camera'))
        self.house_note = QLabel(tr('alert_house_value', types=type_names(house))); self.house_note.setObjectName('muted'); self.house_note.setWordWrap(True)
        if name:
            root.addWidget(self.default); root.addWidget(self.house_note); root.addWidget(self.custom)
        else:
            self.default.hide(); self.custom.hide(); self.house_note.setText(tr('alert_house_caption')); root.addWidget(self.house_note)
        self.tiles = AlertTiles(); self.tiles.set_value(own or house); root.addWidget(self.tiles)
        self.default.setChecked(own is None); self.custom.setChecked(own is not None)
        self.tiles.setVisible(own is not None or name is None)
        self.note = QLabel(tr('alert_loading')); self.note.setWordWrap(True); self.note.setTextFormat(Qt.TextFormat.PlainText); self.note.setObjectName('muted'); root.addWidget(self.note)
        buttons = QHBoxLayout(); self.retry = ZonePill(tr('retry'), compact=True); self.retry.hide(); buttons.addWidget(self.retry); buttons.addStretch()
        self.done_button = ZonePill(tr('alert_done'), primary=True, compact=True); buttons.addWidget(self.done_button); root.addLayout(buttons)
        self.done_button.clicked.connect(self.accept); self.retry.clicked.connect(self.retry_action)
        self.default.clicked.connect(lambda: self.apply(None)); self.custom.clicked.connect(lambda: self.apply(self.tiles.value()))
        self.tiles.changed.connect(self.apply)
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(150)
        self.finished.connect(self.shutdown)
        self.submit('load', controls.camera_alerts)

    def submit(self, operation, task):
        self.operation = operation; self.future = self.pool.submit(task)
        self.default.setEnabled(False); self.custom.setEnabled(False); self.tiles.setEnabled(False)
        self.retry.hide(); self.done_button.setEnabled(operation != 'save')

    def retry_action(self):
        if self.loading: self.submit('load', self.controls.camera_alerts)
        else: self.apply(self.pending)

    def apply(self, value):
        if (self.future is not None and self.operation != 'status') or self.loading: return
        self.pending = value
        self.tiles.setVisible(value is not None or self.name is None)
        self.note.setText(tr('applying'))
        self.requested = time.time()
        self.submit('save', lambda: self.controls.set_house_alerts(value) if self.name is None else self.controls.set_camera_alerts(self.name, value))

    def poll(self):
        if self.future is not None:
            if not self.future.done(): return
            future, self.future = self.future, None
            self.done_button.setEnabled(True)
            try:
                result = future.result()
                if self.operation == 'load':
                    self.house = result['house']
                    self.own = next((r['alert_on'] for r in result['cameras'] if r['name'] == self.name), None)
                    self.house_note.setText(tr('alert_house_value', types=type_names(self.house)) if self.name else tr('alert_house_caption'))
                    self.tiles.set_value(self.own or self.house)
                    self.default.setChecked(self.own is None); self.custom.setChecked(self.own is not None)
                    self.tiles.setVisible(self.own is not None or self.name is None)
                    self.loading = False; self.note.setText('')
                elif self.operation == 'save':
                    if self.name is None:
                        self.ack.expected['alert_on'] = result; self.ack.requested = self.requested
                        self.house = result; self.house_saved.emit(result)
                    else:
                        self.own = result; self.ack.request_camera(self.name, result, self.requested); self.saved.emit(result)
                else:
                    status = self.ack.status(result, time.time())
                    if status: self.note.setText(tr('live_applied' if status == 'applied' else status))
            except Exception as exc:
                if self.operation == 'status':
                    status = self.ack.status({}, time.time()); self.note.setText(tr(status))
                else:
                    self.note.setText(str(exc)); self.retry.show()
                self.note.setStyleSheet('color: ' + colors(self)['error'])
            else:
                self.note.setStyleSheet('color: ' + colors(self)['secondary'])
            self.default.setEnabled(not self.loading); self.custom.setEnabled(not self.loading); self.tiles.setEnabled(not self.loading)
        if self.ack.expected and self.future is None and time.monotonic() - getattr(self, 'last_status', 0) > .75:
            self.last_status = time.monotonic()
            # Status SSH runs off the GUI thread, just like the write command.
            self.operation = 'status'; self.future = self.pool.submit(self.controls.alert_reported_status)

    def shutdown(self):
        self.timer.stop(); self.pool.shutdown(wait=False, cancel_futures=True)

    def done(self, result):
        if self.future is not None and self.operation == 'save': return
        super().done(result)
