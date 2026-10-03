"""Draw picture-relative watch areas without changing the raw camera snapshot."""
from concurrent.futures import ThreadPoolExecutor
import math

from PySide6.QtCore import Qt, QPoint, QPointF, QRectF, QSize, Signal, QTimer, QVariantAnimation, QPropertyAnimation, QParallelAnimationGroup
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QProgressBar, QSizePolicy

from . import motion
from .camera_presentation import ambient_picture
from .strings import tr
from .theme import PALETTES


def colors(widget):
    """Inherit the actual window theme, including standalone offscreen previews."""
    while widget is not None:
        theme = widget.property('theme')
        if theme in PALETTES:
            return PALETTES[theme]
        args = getattr(widget, 'args', None)
        if args is not None:
            return PALETTES[getattr(args, 'theme', 'dark')]
        widget = widget.parentWidget()
    return PALETTES['dark']


def alpha(color, opacity):
    result = QColor(color)
    result.setAlphaF(opacity)
    return result


def animate(owner, start, end, duration, update):
    animation = QVariantAnimation(owner)
    animation.setDuration(duration)
    animation.setEasingCurve(motion.EASING)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.valueChanged.connect(update)
    animation.start()
    return animation


def polygon_path(points, rect):
    path = QPainterPath()
    for i, (x, y) in enumerate(points):
        point = QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())
        path.moveTo(point) if i == 0 else path.lineTo(point)
    if len(points) >= 3:
        path.closeSubpath()
    return path


class ZonePill(QPushButton):
    """Quiet palette-only hover and a one-pixel press inset, with shared timings."""
    def __init__(self, text, parent=None, primary=False, compact=False):
        super().__init__(text, parent)
        self.primary, self.compact = primary, compact
        self.hover = self.press = 0.
        self.setProperty('handlesMotion', True)
        # The app's ordinary button padding otherwise overrides the 36px tile slot.
        self.setStyleSheet('padding: 0; min-height: 0; border: none; background: transparent;')
        self.setAccessibleName(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(34 if compact else 48)
        self.setAutoDefault(False)
        self.pressed.connect(lambda: self.target('press', 1., motion.HOVER_MS))
        self.released.connect(lambda: self.target('press', 0., motion.HOVER_MS))

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + (64 if self.primary else 28), self.height())

    def target(self, name, value, duration):
        previous = getattr(self, '_' + name, None)
        if previous:
            previous.stop()
        def update(level):
            setattr(self, name, float(level))
            self.update()
        setattr(self, '_' + name, animate(self, getattr(self, name), value, duration, update))

    def enterEvent(self, event):
        self.target('hover', 1., motion.TOGGLE_MS)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.target('hover', 0., motion.TOGGLE_MS)
        super().leaveEvent(event)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(1. if self.isEnabled() or self.property('working') else .4)
        rect = QRectF(self.rect()).adjusted(1 + self.press, 1 + self.press, -1 - self.press, -1 - self.press)
        base = QColor(t['action'] if self.primary else t['surface'])
        highlight = QColor(t['ok'] if self.primary else t['raised'])
        mix = QColor.fromRgbF(*[base.getRgbF()[i] * (1 - self.hover) + highlight.getRgbF()[i] * self.hover for i in range(3)])
        p.setBrush(mix)
        p.setPen(QPen(alpha(t['text'], .12 if not self.primary else 0), 1))
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        foreground = t['bg'] if self.primary else t['secondary']
        p.setPen(QColor(foreground))
        font = self.font(); font.setPixelSize(13 if self.compact else 15); font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        text_rect = QRectF(rect)
        if self.primary:
            rtl = self.layoutDirection() == Qt.LayoutDirection.RightToLeft
            circle = QRectF(rect.left() + 12 if rtl else rect.right() - 34, rect.center().y() - 11, 22, 22)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(alpha(t['bg'], .13)); p.drawEllipse(circle)
            p.setPen(QPen(QColor(foreground), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            check = QPainterPath(); check.moveTo(circle.center() + QPointF(-4, 0)); check.lineTo(circle.center() + QPointF(-1, 3)); check.lineTo(circle.center() + QPointF(5, -3)); p.drawPath(check)
            text_rect.adjust(28 if rtl else 0, 0, 0 if rtl else -28, 0)
        p.setPen(QColor(foreground)); p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.text())
        if self.hasFocus() and self.property('keyboardFocus'):
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor(t['action']), 2)); p.drawRoundedRect(rect.adjusted(2, 2, -2, -2), rect.height()/2, rect.height()/2)


class FadingLabel(QLabel):
    """Crossfade copy in its reserved space so status changes never move controls."""
    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self.level = 1.
        self.before = QPixmap()
        self.chip_color = None
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def change(self, text):
        if text == self.text():
            return
        previous = getattr(self, 'fade', None)
        if previous:
            previous.stop()
        self.before = self.grab() if self.isVisible() else QPixmap()
        self.setText(text)
        self.fade = animate(self, 0., 1., motion.PANE_MS, self.advance)

    def advance(self, value):
        self.level = float(value); self.update()

    def paintEvent(self, event):
        # Render text directly; nested graphics effects can erase child labels on Windows.
        p = QPainter(self)
        if self.level < 1 and not self.before.isNull():
            p.setOpacity(1 - self.level); p.drawPixmap(0, 0, self.before)
        p.setOpacity(self.level)
        if self.chip_color and self.text():
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setBrush(alpha(self.chip_color, .07)); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(self.rect()), 12, 12)
        p.setPen(self.palette().windowText().color())
        flags = self.alignment() | Qt.TextFlag.TextWordWrap
        p.drawText(self.contentsRect(), flags, self.text())


class ZoneStage(QWidget):
    changed = Signal()

    def __init__(self, pixmap, points=(), parent=None):
        super().__init__(parent)
        self.pix = QPixmap(pixmap)
        self.ambient = ambient_picture(self.pix) if not self.pix.isNull() else QPixmap()
        self.points = [[round(x, 4), round(y, 4)] for x, y in points]
        self.overlay = 1. if len(self.points) >= 3 else 0.
        self.hovered = -1
        self.dragging = -1
        self.pointer = None
        self.closed = len(points) >= 3
        self.hover_level = 0.
        self.pulse_level = 0.
        self.corner_level = 1.
        self.removed = []
        self.last_shape = list(self.points) if len(self.points) >= 3 else []
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(tr('camera_zone_title'))
        self.setAccessibleDescription(tr('camera_zone_help'))
        self.setMinimumSize(440, 400)
        self.pulse = QVariantAnimation(self)
        self.pulse.setDuration(motion.PANE_MS)
        self.pulse.setEasingCurve(motion.EASING)
        self.pulse.setStartValue(0.); self.pulse.setEndValue(1.); self.pulse.setLoopCount(-1)
        self.pulse.valueChanged.connect(self.pulse_changed)

    def core_rect(self):
        return QRectF(self.rect()).adjusted(7, 7, -7, -7)

    def picture_rect(self):
        core = self.core_rect()
        if self.pix.isNull():
            return QRectF()
        scale = min(core.width()/self.pix.width(), core.height()/self.pix.height())
        w, h = self.pix.width()*scale, self.pix.height()*scale
        return QRectF(core.center().x()-w/2, core.center().y()-h/2, w, h)

    def to_fraction(self, point):
        rect = self.picture_rect()
        if rect.isEmpty():
            return (0., 0.)
        return (round(max(0., min(1., (point.x()-rect.x())/rect.width())), 4),
                round(max(0., min(1., (point.y()-rect.y())/rect.height())), 4))

    def to_stage(self, point):
        rect = self.picture_rect()
        return QPointF(rect.x()+point[0]*rect.width(), rect.y()+point[1]*rect.height())

    def area(self):
        if len(self.points) < 3:
            return 0.
        return abs(sum(x * self.points[(i+1)%len(self.points)][1] - y * self.points[(i+1)%len(self.points)][0] for i, (x, y) in enumerate(self.points))) / 2

    def set_points(self, points):
        previous = list(self.points)
        self.points = [[round(float(x), 4), round(float(y), 4)] for x, y in points][:32]
        if len(previous) >= 3:
            self.last_shape = previous
        if len(self.points) < 3:
            self.closed = False
        self.removed = previous[len(self.points):]
        for name in ('overlay_animation', 'corner_animation'):
            if getattr(self, name, None): getattr(self, name).stop()
        self.overlay_animation = animate(self, self.overlay, 1. if len(self.points) >= 3 else 0., motion.PANE_MS, self.overlay_changed)
        self.corner_animation = animate(self, 0., 1., motion.HOVER_MS, self.corners_changed)
        self.update_hover(self.pointer)
        self.changed.emit(); self.update()

    def overlay_changed(self, value):
        self.overlay = float(value); self.update()

    def corners_changed(self, value):
        self.corner_level = float(value); self.update()

    def pulse_changed(self, value):
        self.pulse_level = float(value); self.update()

    def undo(self):
        self.closed = False
        self.set_points(self.points[:-1])

    def clear(self):
        self.set_points([])

    def corner_at(self, point):
        if point is None:
            return -1
        for i, corner in enumerate(self.points):
            if math.hypot(*(point-self.to_stage(corner)).toTuple()) <= 12:
                return i
        return -1

    def update_hover(self, point):
        index = self.corner_at(point)
        if index != self.hovered:
            self.hovered = index
            if getattr(self, 'hover_animation', None): self.hover_animation.stop()
            self.hover_animation = animate(self, 0., 1., motion.HOVER_MS, self.hover_changed)
        snap = index == 0 and len(self.points) >= 3 and not self.closed and self.dragging < 0
        if snap and self.pulse.state() != QVariantAnimation.State.Running:
            self.pulse.start()
        elif not snap:
            self.pulse.stop(); self.pulse_level = 0.
        self.setCursor(Qt.CursorShape.ClosedHandCursor if self.dragging >= 0 else Qt.CursorShape.PointingHandCursor if index >= 0 else Qt.CursorShape.CrossCursor)

    def hover_changed(self, value):
        self.hover_level = float(value); self.update()

    def mousePressEvent(self, event):
        if not self.isEnabled(): return
        self.setFocus()
        if event.button() == Qt.MouseButton.RightButton:
            self.undo(); return
        if event.button() != Qt.MouseButton.LeftButton: return
        point = event.position()
        index = self.corner_at(point)
        if index == 0 and len(self.points) >= 3 and not self.closed:
            self.closed = True; self.pointer = None; self.pulse.stop(); self.update(); return
        if index >= 0:
            self.dragging = index; self.update_hover(point); return
        if self.picture_rect().contains(point) and len(self.points) < 32:
            self.closed = False
            self.set_points([*self.points, self.to_fraction(point)])

    def mouseMoveEvent(self, event):
        self.pointer = event.position() if self.picture_rect().contains(event.position()) else None
        if self.dragging >= 0:
            self.points[self.dragging] = list(self.to_fraction(event.position()))
            self.changed.emit()
        self.update_hover(event.position()); self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = -1; self.update_hover(event.position())

    def leaveEvent(self, event):
        self.pointer = None; self.update_hover(None); self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self); p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        outer = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        p.setBrush(QColor(t['raised'])); p.setPen(QPen(alpha(t['text'], .08), 1)); p.drawRoundedRect(outer, 22, 22)
        core = self.core_rect(); clip = QPainterPath(); clip.addRoundedRect(core, 16, 16); p.setClipPath(clip)
        if not self.pix.isNull():
            p.drawPixmap(core, self.ambient, QRectF(self.ambient.rect()))
            p.fillRect(core, alpha(t['bg'], .65))
            rect = self.picture_rect(); p.drawPixmap(rect, self.pix, QRectF(self.pix.rect()))
            shape = polygon_path(self.points, rect)
            if self.overlay > 0:
                outside = QPainterPath(); outside.addRect(rect)
                outside = outside.subtracted(shape if len(self.points) >= 3 else polygon_path(self.last_shape, rect))
                p.fillPath(outside, alpha(t['bg'], .65*self.overlay))
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor(t['action']), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)); p.drawPath(shape)
            if self.points and self.pointer is not None and not self.closed and self.dragging < 0:
                p.setPen(QPen(alpha(t['action'], .8), 2, Qt.PenStyle.DashLine)); p.drawLine(self.to_stage(self.points[-1]), self.pointer)
            for i, point in enumerate(self.points):
                center = self.to_stage(point)
                radius = 6 + (2*self.hover_level if i == self.hovered else 0)
                p.setOpacity(self.corner_level if i == len(self.points)-1 and not self.removed else 1.)
                if i == self.hovered:
                    p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(alpha(t['action'], .3), 2)); p.drawEllipse(center, radius+4, radius+4)
                if i == 0 and self.pulse.state() == QVariantAnimation.State.Running:
                    p.setPen(QPen(alpha(t['action'], .4*(1-self.pulse_level)), 2)); p.drawEllipse(center, radius+4+5*self.pulse_level, radius+4+5*self.pulse_level)
                p.setBrush(QColor(t['bg'])); p.setPen(QPen(QColor(t['action']), 2)); p.drawEllipse(center, radius, radius)
            p.setOpacity(1-self.corner_level)
            for point in self.removed:
                p.setBrush(QColor(t['bg'])); p.setPen(QPen(QColor(t['action']), 2)); p.drawEllipse(self.to_stage(point), 6, 6)
            p.setOpacity(1.)
        else:
            p.setPen(QColor(t['muted'])); p.drawText(core, Qt.AlignmentFlag.AlignCenter, tr('camera_no_photo'))
        p.setClipping(False); p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(alpha(t['text'], .1), 1)); p.drawRoundedRect(core, 16, 16)


class ZoneEditorDialog(QDialog):
    zone_saved = Signal(list)

    def __init__(self, controls, name, pixmap, points=(), parent=None):
        super().__init__(parent)
        self.controls, self.name = controls, name
        self.saved_points = None
        self.saving = False
        self.future = None
        self._closing = False
        self._opened = False
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.setWindowTitle(tr('camera_zone_title'))
        self.setModal(True)
        t = colors(self)
        self.setStyleSheet(f'QDialog {{ background: {t["bg"]}; }} QLabel {{ background: transparent; color: {t["secondary"]}; border: none; font-size: 11.25pt; }}')
        available = self.screen().availableGeometry()
        self.setMinimumSize(min(1100, available.width()-32), min(700, available.height()-48))
        self.resize(min(1280, available.width()-32), min(800, available.height()-48))
        root = QHBoxLayout(self); root.setContentsMargins(24, 24, 28, 24); root.setSpacing(28)
        self.stage = ZoneStage(pixmap, points, self); root.addWidget(self.stage, 72)
        column = QWidget(self); column.setMinimumWidth(282)
        column.setStyleSheet('background: transparent;')
        root.addWidget(column, 28)
        side = QVBoxLayout(column); side.setContentsMargins(0, 12, 0, 8); side.setSpacing(16)
        self.eyebrow = QLabel(tr('camera_zone_eyebrow', camera=name.replace('_', ' ').upper()).upper())
        self.eyebrow.setTextFormat(Qt.TextFormat.PlainText); self.eyebrow.setWordWrap(True)
        self.eyebrow.setStyleSheet(f'color: {t["muted"]}; font-size: 8.25pt; letter-spacing: 1.6px; font-weight: 600;')
        side.addWidget(self.eyebrow)
        title = QLabel(tr('camera_zone_title')); title.setWordWrap(True)
        title.setStyleSheet(f'color: {t["text"]}; font-size: 15pt; font-weight: 600;')
        side.addWidget(title)
        help_text = QLabel(tr('camera_zone_help')); help_text.setWordWrap(True)
        help_text.setStyleSheet(f'color: {t["muted"]}; font-size: 11.25pt;')
        side.addWidget(help_text)
        side.addSpacing(18)
        self.status = FadingLabel(parent=column)
        self.status.setMinimumHeight(68); self.status.setContentsMargins(14, 12, 14, 12)
        self.status.setStyleSheet(f'color: {t["secondary"]}; font-size: 11.25pt;')
        status_shell = QWidget(); status_shell.setStyleSheet(f'background: {t["raised"]}; border-radius: 14px;')
        status_layout = QVBoxLayout(status_shell); status_layout.setContentsMargins(0, 0, 0, 0); status_layout.addWidget(self.status)
        side.addWidget(status_shell)
        self.warning = FadingLabel(parent=column); self.warning.setMinimumHeight(76)
        self.warning.chip_color = t['warning']
        self.warning.setContentsMargins(12, 8, 12, 8)
        self.warning.setStyleSheet(f'color: {t["warning"]}; font-size: 9.75pt;')
        side.addWidget(self.warning)
        side.addStretch(1)
        edits = QHBoxLayout(); edits.setSpacing(10)
        self.undo_button = ZonePill(tr('camera_zone_undo'), compact=True)
        self.clear_button = ZonePill(tr('camera_zone_clear'), compact=True)
        edits.addWidget(self.undo_button); edits.addWidget(self.clear_button); edits.addStretch()
        side.addLayout(edits)
        self.undo_button.clicked.connect(self.stage.undo); self.clear_button.clicked.connect(self.stage.clear)
        self.save_button = ZonePill(tr('camera_zone_save'), primary=True)
        self.cancel_button = ZonePill(tr('camera_zone_cancel'))
        buttons = QHBoxLayout(); buttons.setSpacing(10); buttons.addWidget(self.save_button, 3); buttons.addWidget(self.cancel_button, 2)
        side.addLayout(buttons)
        bar_slot = QWidget(); bar_slot.setFixedHeight(5)
        bar_layout = QHBoxLayout(bar_slot); bar_layout.setContentsMargins(0, 0, 0, 0); bar_layout.setSpacing(10)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.setTextVisible(False); self.progress.setFixedHeight(3)
        self.progress.setStyleSheet(f'QProgressBar {{ background: {t["raised"]}; border: none; border-radius: 1px; }} QProgressBar::chunk {{ background: {t["action"]}; }}')
        bar_layout.addWidget(self.progress, 3); bar_layout.addStretch(2); self.progress.hide(); side.addWidget(bar_slot)
        self.save_button.setProperty('busyIndicator', 'bar'); self.save_button._busy_bar = self.progress
        self.error = FadingLabel(parent=column); self.error.setMinimumHeight(44)
        self.error.setStyleSheet(f'color: {t["error"]}; font-size: 9.75pt;'); side.addWidget(self.error)
        self.save_button.clicked.connect(self.save); self.cancel_button.clicked.connect(self.reject)
        self.stage.changed.connect(self.update_state)
        self.timer = QTimer(self); self.timer.setInterval(motion.HOVER_MS); self.timer.timeout.connect(self.poll)
        self.finished.connect(lambda: self.pool.shutdown(wait=False, cancel_futures=True))
        self.update_state()

    def update_state(self):
        count = len(self.stage.points)
        key = 'empty' if count == 0 else 'drawing' if count < 3 else 'closed'
        self.status.change(tr('camera_zone_status_' + key))
        self.warning.change(tr('camera_zone_small') if count >= 3 and self.stage.area() < .05 else '')
        self.save_button.setEnabled(not self.saving and (count == 0 or count >= 3))
        self.undo_button.setEnabled(not self.saving and count > 0)
        self.clear_button.setEnabled(not self.saving and count > 0)

    def save(self):
        if self.saving or self._closing:
            return
        self.saving = True
        self.error.change('')
        self.save_button.setText(tr('camera_zone_saving'))
        motion.busy(self.save_button, True)
        self.stage.setEnabled(False); self.cancel_button.setEnabled(False)
        self.update_state()
        points = [[round(x, 4), round(y, 4)] for x, y in self.stage.points]
        self.future = self.pool.submit(self.controls.set_zone, self.name, points) if len(points) >= 3 else self.pool.submit(self.controls.clear_zone, self.name)
        self.timer.start()

    def poll(self):
        if self.future is None or not self.future.done():
            return
        self.timer.stop()
        future, self.future = self.future, None
        self.saving = False
        motion.busy(self.save_button, False)
        self.save_button.setText(tr('camera_zone_save'))
        self.stage.setEnabled(True); self.cancel_button.setEnabled(True)
        self.update_state()
        try:
            self.saved_points = future.result()
        except Exception:
            self.error.change(tr('camera_zone_save_failed'))
            return
        self.zone_saved.emit(self.saved_points)
        self.done(QDialog.DialogCode.Accepted)

    def keyPressEvent(self, event):
        if self.saving:
            event.accept(); return
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Z) and (event.key() == Qt.Key.Key_Backspace or event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.stage.undo(); event.accept(); return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.save_button.isEnabled(): self.save()
            event.accept(); return
        super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._opened:
            self._opened = True
            self.destination = self.pos()
            self.transition(True)
            self.stage.setFocus()

    def transition(self, opening, result=None):
        previous = getattr(self, 'transition_group', None)
        if previous: previous.stop()
        group = QParallelAnimationGroup(self); self.transition_group = group
        duration = motion.PANE_MS if opening else motion.TOGGLE_MS
        opacity = QPropertyAnimation(self, b'windowOpacity'); opacity.setStartValue(0. if opening else self.windowOpacity()); opacity.setEndValue(1. if opening else 0.)
        position = QPropertyAnimation(self, b'pos'); position.setStartValue(self.destination + QPoint(0, 8) if opening else self.pos()); position.setEndValue(self.destination if opening else self.pos() + QPoint(0, 8))
        if opening: self.setWindowOpacity(0.); self.move(self.destination + QPoint(0, 8))
        for animation in (opacity, position):
            animation.setDuration(duration); animation.setEasingCurve(motion.EASING); group.addAnimation(animation)
        if not opening: group.finished.connect(lambda: self.finish_close(result))
        group.start()

    def finish_close(self, result):
        parent = self.parentWidget()
        super().done(result)
        if result == QDialog.DialogCode.Accepted and parent:
            motion.toast(parent.window(), tr('camera_zone_saved'))

    def done(self, result):
        if self.saving or self._closing: return
        self._closing = True
        if not self.isVisible(): self.finish_close(result)
        else: self.transition(False, result)

    def reject(self):
        self.done(QDialog.DialogCode.Rejected)

    def closeEvent(self, event):
        event.ignore()
        self.reject()
