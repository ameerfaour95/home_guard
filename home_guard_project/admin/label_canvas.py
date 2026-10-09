"""Interactive frame rectangles and the temporal track editor.

The canvas zooms (wheel or trackpad pinch around the cursor, + / - and Fit) and pans (drag with the middle button, or
with Space held). Every box is kept in IMAGE coordinates: the zoom only changes where the picture is drawn
(display_rect), so drawing, moving and resizing give the same box at any zoom."""
from PySide6.QtCore import Qt, QRectF, QPointF, QEvent, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget, QMenu, QApplication
from home_guard_project.fleet_contract.tracks import box_at
from .player import VideoCanvas, SESSION
from .tag_widgets import machine_name
from .event_logic import map_box
from .theme import PALETTES
from .label_document import CLASSES


def class_color(name, theme):
    # All class accents derive from the active design palette.
    base = QColor(PALETTES[theme][('action', 'warning', 'ok')[CLASSES.index(name) % 3]])
    h, s, v, a = base.getHsvF()
    return QColor.fromHsvF((h + .065*(CLASSES.index(name)//3)) % 1, s, v, a)


def machine_tag(doc):
    """The caption suffix of a preloaded box nobody checked yet: who drew it."""
    return '  ·  ' + machine_name(doc.preload_source)


class LabelCanvas(VideoCanvas):
    selected = Signal()
    interaction_started = Signal()

    MIN_ZOOM, MAX_ZOOM, STEP = 1.0, 8.0, 1.25
    zoom_changed = Signal()

    def __init__(self, theme='dark'):
        super().__init__(theme)
        self.overlay.hide()
        self.theme, self.doc = theme, None
        self.drag, self.preview = None, None
        self.zoom, self.pan, self.panning, self.space_held = 1.0, QPointF(0., 0.), None, False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
        self.setAccessibleName('Video annotation canvas: drag to draw, move or resize a box; wheel to zoom')
        # Space held = pan (Space alone still plays / pauses): an app-wide filter, only while the canvas is shown, so a
        # hidden or closed canvas never sees (or slows down) the rest of the app's events

    # ------------------------------------------------------------------ zoom and pan (image coordinates kept)
    def fit_rect(self):
        return super().display_rect()

    def display_rect(self):
        x, y, w, h = self.fit_rect()
        if self.zoom == 1.0 and self.pan.isNull():
            return (x, y, w, h)
        zw, zh = w*self.zoom, h*self.zoom
        return (x - (zw-w)/2 + self.pan.x(), y - (zh-h)/2 + self.pan.y(), zw, zh)

    def image_point(self, point):
        """The point under *point* in normalised image coordinates, not clamped (it may lie outside the picture)."""
        x, y, w, h = self.display_rect()
        return QPointF((point.x()-x)/w, (point.y()-y)/h) if w and h else QPointF(0., 0.)

    def zoom_to(self, zoom, anchor=None):
        """Zoom to *zoom* (1 = fit) keeping the image point under *anchor* (default the centre) where it is."""
        zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        anchor = anchor if anchor is not None else QPointF(self.width()/2, self.height()/2)
        before = self.image_point(anchor)
        self.zoom = zoom
        if zoom == 1.0:
            self.pan = QPointF(0., 0.)
        else:
            x, y, w, h = self.display_rect()
            # where the anchor's image point lands now, moved back under the anchor
            self.pan += QPointF(anchor.x() - (x + before.x()*w), anchor.y() - (y + before.y()*h))
        self.update(); self.zoom_changed.emit()

    def fit(self):
        self.zoom_to(1.0)

    def zoom_in(self, anchor=None):
        self.zoom_to(self.zoom*self.STEP, anchor)

    def zoom_out(self, anchor=None):
        self.zoom_to(self.zoom/self.STEP, anchor)

    def wheelEvent(self, event):
        steps = event.angleDelta().y()/120 or event.pixelDelta().y()/60
        if steps:
            self.zoom_to(self.zoom*(self.STEP**steps), event.position()); event.accept()

    def event(self, event):
        if event.type() == QEvent.Type.NativeGesture and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom_to(self.zoom*(1+event.value()), event.position()); return True   # a trackpad pinch
        return super().event(event)

    def showEvent(self, event):
        QApplication.instance().installEventFilter(self); super().showEvent(event)

    def hideEvent(self, event):
        QApplication.instance().removeEventFilter(self); self.space_held = False; super().hideEvent(event)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and event.key() == Qt.Key.Key_Space \
                and not event.isAutoRepeat():
            self.space_held = event.type() == QEvent.Type.KeyPress
        return False

    def rect_for(self, xyxy):
        return QRectF(*map_box(xyxy, self.display_rect()))

    def normal(self, point):
        x, y, w, h = self.display_rect()
        return QPointF(max(0., min(1., (point.x()-x)/w)), max(0., min(1., (point.y()-y)/h)))

    def handles(self, r):
        return [r.topLeft(), QPointF(r.center().x(), r.top()), r.topRight(),
                QPointF(r.right(), r.center().y()), r.bottomRight(),
                QPointF(r.center().x(), r.bottom()), r.bottomLeft(), QPointF(r.left(), r.center().y())]

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.doc:
            return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Contract boxes_at supplies geometry; per-track box_at retains selection identity.
        visible = iter(self.doc.visible_boxes())
        names = self.doc.display_names()
        for tr in self.doc.tracks:
            if box_at(tr, self.doc.t_sec) is None:
                continue
            _, xyxy = next(visible)
            if not SESSION['boxes'] or tr.track_id in self.doc.hidden:
                continue  # Boxes off (B) or this track hidden (H): the video alone
            if self.drag and tr.track_id == self.doc.selected and self.drag[0] != 'draw':
                xyxy = self.preview or xyxy
            r, color = self.rect_for(xyxy), class_color(tr.label, self.theme)
            pen = QPen(color, 2.5 if tr.track_id == self.doc.selected else 2)
            if tr.source in ('yolo', 'suggestion'): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(r)
            caption = names.get(tr.track_id, tr.label) + (machine_tag(self.doc) if tr.source in ('yolo', 'suggestion') else '')
            tag = QRectF(r.x(), max(0., self.display_rect()[1], r.y()-25), p.fontMetrics().horizontalAdvance(caption)+16, 25)
            p.fillRect(tag, color); p.setPen(QColor('#07181b')); p.drawText(tag, Qt.AlignmentFlag.AlignCenter, caption)
            if tr.track_id == self.doc.selected:
                p.setPen(QPen(color, 1.5)); p.setBrush(QColor('#edf4f6'))
                for h in self.handles(r): p.drawRect(QRectF(h.x()-4, h.y()-4, 8, 8))
        if self.drag and self.drag[0] == 'draw' and self.preview:
            p.setPen(QPen(class_color(self.doc.current_class, self.theme), 2)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.rect_for(self.preview))
        if self.zoom > 1.0:
            text = f'{self.zoom:.1f}×  ·  Fit 0  ·  pan: middle button or Space + drag'
            bg = QRectF(self.width()-p.fontMetrics().horizontalAdvance(text)-32, 12, p.fontMetrics().horizontalAdvance(text)+20, 28)
            p.fillRect(bg, QColor(12, 18, 24, 220)); p.setPen(QColor('#edf4f6'))
            p.drawText(bg, Qt.AlignmentFlag.AlignCenter, text)
        if not SESSION['boxes'] or self.doc.hidden:
            text = 'Boxes hidden  ·  B shows them' if not SESSION['boxes'] else f'{len(self.doc.hidden)} track(s) hidden  ·  H on a track shows it'
            bg = QRectF(12, 12, p.fontMetrics().horizontalAdvance(text)+20, 28)
            p.fillRect(bg, QColor(12, 18, 24, 220)); p.setPen(QColor('#edf4f6'))
            p.drawText(bg, Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton
                                                              and self.space_held and self.zoom > 1.0):
            self.panning = event.position(); self.setCursor(Qt.CursorShape.ClosedHandCursor); return
        if not self.doc or event.button() != Qt.MouseButton.LeftButton or not QRectF(*self.display_rect()).contains(event.position()):
            return
        if not SESSION['boxes']:
            return  # nothing to edit on a picture without its boxes
        self.setFocus(); self.interaction_started.emit()
        point = event.position(); normal = self.normal(point)
        if self.doc.track:
            box = box_at(self.doc.track, self.doc.t_sec)
            if box:
                for i, handle in enumerate(self.handles(self.rect_for(box))):
                    if (handle-point).manhattanLength() <= 12:
                        self.drag = ('resize', normal, box, i); self.preview = box; return
        for tr in reversed(self.doc.tracks):
            box = box_at(tr, self.doc.t_sec) if tr.track_id not in self.doc.hidden else None
            if box and self.rect_for(box).contains(point):
                self.doc.selected = tr.track_id
                self.drag = ('move', normal, box, None); self.preview = box
                self.selected.emit(); self.update(); return
        self.doc.selected = None
        self.drag = ('draw', normal, None, None); self.preview = None
        self.selected.emit(); self.update()

    def mouseMoveEvent(self, event):
        if self.panning is not None:
            self.pan += event.position() - self.panning; self.panning = event.position(); self.update(); return
        if not self.drag:
            return
        mode, origin, box, handle = self.drag
        pos = self.normal(event.position())
        if mode == 'draw':
            self.preview = [min(origin.x(), pos.x()), min(origin.y(), pos.y()), max(origin.x(), pos.x()), max(origin.y(), pos.y())]
        elif mode == 'move':
            dx = max(-box[0], min(1-box[2], pos.x()-origin.x()))
            dy = max(-box[1], min(1-box[3], pos.y()-origin.y()))
            self.preview = [box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy]
        else:
            self.preview = list(box)
            mx, my = 4/self.doc.frame_size[0], 4/self.doc.frame_size[1]
            if handle in (0, 6, 7): self.preview[0] = min(pos.x(), box[2]-mx)
            if handle in (0, 1, 2): self.preview[1] = min(pos.y(), box[3]-my)
            if handle in (2, 3, 4): self.preview[2] = max(pos.x(), box[0]+mx)
            if handle in (4, 5, 6): self.preview[3] = max(pos.y(), box[1]+my)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.panning is not None:
            self.panning = None; self.unsetCursor(); return
        if self.drag and self.preview:
            self.doc.put_box(self.preview, None if self.drag[0] == 'draw' else self.doc.track)
        self.drag, self.preview = None, None
        self.selected.emit(); self.update()


class TrackTimeline(QWidget):
    seek_requested = Signal(int)
    selected = Signal()

    def __init__(self, theme='dark'):
        super().__init__()
        self.doc, self.theme, self.drag, self.scrubbing = None, theme, None, None
        self.left, self.row_height = 172, 38
        self.setMinimumHeight(110); self.setMouseTracking(True)
        self.setAccessibleName('Track timeline: drag keyframes; right-click segments to keep or hide')

    def x(self, frame):
        return self.left+(self.width()-self.left-20)*frame/max(1, self.doc.frame_count-1)

    def frame(self, x):
        return max(0, min(self.doc.frame_count-1, round((x-self.left)/max(1, self.width()-self.left-20)*(self.doc.frame_count-1))))

    def refresh(self):
        self.setMinimumHeight(44+self.row_height*len(self.doc.tracks) if self.doc else 110)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = PALETTES[self.theme]; p.fillRect(self.rect(), QColor(t['surface']))
        if not self.doc: return
        p.setPen(QColor(t['muted'])); p.drawText(14, 24, 'OBJECT / TRACK')
        for i in range(6):
            f = round((self.doc.frame_count-1)*i/5)
            p.drawText(QRectF(self.x(f)-28, 5, 56, 25), Qt.AlignmentFlag.AlignCenter, f'{self.doc.time_for(f):.1f}s')
        names = self.doc.display_names()
        for row, tr in enumerate(self.doc.tracks):
            y = 48+row*self.row_height
            if tr.track_id == self.doc.selected: p.fillRect(QRectF(0, y-16, self.width(), 34), QColor(t['raised']))
            color = class_color(tr.label, self.theme)
            unchecked = machine_tag(self.doc).replace('  ·  ', '  · ') if tr.source in ('yolo', 'suggestion') else ''
            hidden = '  · hidden' if tr.track_id in self.doc.hidden else ''
            p.setPen(QColor(t['muted']) if hidden else color)
            p.drawText(14, y+5, names.get(tr.track_id, tr.label) + unchecked + hidden)
            for i, k in enumerate(tr.keyframes):
                end = tr.keyframes[i+1].frame if i+1 < len(tr.keyframes) else self.doc.frame_count-1
                pen = QPen(color, 5 if k.enabled else 1.5)
                if not k.enabled: pen.setStyle(Qt.PenStyle.DotLine)
                p.setPen(pen); p.drawLine(QPointF(self.x(k.frame), y), QPointF(self.x(end), y))
            for k in tr.keyframes:
                p.setPen(QPen(color, 2)); p.setBrush(color if k.enabled else QColor(t['surface']))
                p.drawEllipse(QPointF(self.x(k.frame), y), 5, 5)
        p.setPen(QPen(QColor(t['text']), 1.5)); x = self.x(self.doc.frame)
        p.drawLine(QPointF(x, 31), QPointF(x, self.height()))
        p.setBrush(QColor(t['text'])); p.drawEllipse(QPointF(x, 33), 3, 3)

    def mousePressEvent(self, event):
        if not self.doc: return
        row = int((event.position().y()-31)//self.row_height)
        frame = self.frame(event.position().x())
        if 0 <= row < len(self.doc.tracks):
            tr = self.doc.tracks[row]; self.doc.selected = tr.track_id; self.selected.emit()
            if event.button() == Qt.MouseButton.RightButton:
                previous = next((k for k in reversed(tr.keyframes) if k.frame <= frame), None)
                if previous:
                    self.seek_requested.emit(previous.frame)
                    menu = QMenu(self)
                    def set_segment(enabled):
                        previous.enabled = enabled; tr.source = 'human'; self.doc.checkpoint()
                    menu.addAction('Keep segment', lambda: set_segment(True))
                    menu.addAction('Hide segment', lambda: set_segment(False))
                    menu.addSeparator()
                    names = self.doc.display_names()
                    for other in self.doc.merge_candidates(tr):
                        menu.addAction(f'Merge with {names.get(other.track_id, other.label)}',
                                       lambda other=other: self.doc.merge(other) and self.selected.emit())
                    menu.exec(event.globalPosition().toPoint())
                return
            dot = next((k for k in tr.keyframes if abs(self.x(k.frame)-event.position().x()) < 9), None)
            if dot and event.button() == Qt.MouseButton.LeftButton:
                # a keyframe: jump to ITS frame; it moves only when it is dragged further than a few pixels
                self.drag = (tr.track_id, dot.frame, event.position().x(), False)
                self.seek_requested.emit(dot.frame); self.update(); return
        if event.position().x() >= self.left and event.button() == Qt.MouseButton.LeftButton:
            self.scrubbing = frame                 # a click seeks; holding the button and moving scrubs
            self.seek_requested.emit(frame)
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag:
            track_id, old, x0, moved = self.drag
            if moved or abs(event.position().x() - x0) > 4:
                self.drag = (track_id, old, x0, True)
                self.seek_requested.emit(self.frame(event.position().x()))
        elif getattr(self, 'scrubbing', None) is not None and event.buttons() & Qt.MouseButton.LeftButton:
            frame = self.frame(event.position().x())
            if frame != self.scrubbing:
                self.scrubbing = frame; self.seek_requested.emit(frame)
        elif self.doc:
            row = int((event.position().y()-31)//self.row_height)
            tracks = self.doc.tracks
            self.setToolTip(f'{self.doc.display_name(tracks[row])}  ·  id {tracks[row].track_id}'
                            if 0 <= row < len(tracks) and event.position().x() < self.left else '')

    def mouseReleaseEvent(self, event):
        self.scrubbing = None
        if self.drag:
            track_id, old, _, moved = self.drag
            if moved:
                self.doc.move_keyframe(track_id, old, self.frame(event.position().x()))
            self.drag = None
