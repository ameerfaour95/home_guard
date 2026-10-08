"""The scene map editor: per camera, whose ground each part of the picture is.

One widget (``SceneMapEditor``) runs in the box app's camera dialog (``SceneMapDialog``, from the card's "Map"
button) and in setup's "the map of each camera" step. It grew out of the watch-zone dialog:

1. The box numbers what it sees (``propose --embed``, FastSAM on the box; a calm loading state meanwhile).
2. The picture shows the numbered places, drawn here from their polygons; per number four big choices (ours, the
   neighbour's, street, hide) plus skip / don't know. Clicking a place selects its row; colours follow live.
3. Areas drawn by hand: click corners, double-click to close, drag corners, delete; each gets a choice and a name.
4. Boundary lines: two points, then a click on our side (``Line.toward``), or our side from the areas of ours
   (``inward_from_areas``); an arrow points to our side.
5. Save: a summary first, then ``confirm --map-b64`` on the box; the saved map is shown as the box returned it.

Box work runs off the GUI thread; errors show in plain words with Try again.
"""
from concurrent.futures import ThreadPoolExecutor
import math

from PySide6.QtCore import Qt, QPointF, QRectF, QSize, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from . import motion
from .camera_presentation import ambient_picture
from .scene_backend import SceneError
from .scene_model import (BOUNDARY, CHOICES, COLOURS, SKIP, UNKNOWN, Boundary, HandArea, boundary_from_areas,
                          boundary_toward,
                          build_map, counts, from_current, inward_vector, label_point, number_colour,
                          polygon_area, rest_after, restart_expected, MAX_CORNERS)
from .scene_strings import language, st
from .zone_editor import ZonePill, alpha, colors, polygon_path

PANEL_WIDTH = 420
COMPACT_BELOW = 700             # px of editor height: below it the answer cards go compact
LOADING, EDIT, SUMMARY, SAVED, ERROR = range(5)
REGIONS, DRAW, LINES = range(3)
HANDLE = 12                      # px: how close a click must be to a corner or a line


def fallback_name(position=None, lang=None):
    """A camera the box did not name: "מצלמה 2 מתוך 5" by its place in the list (never its id)."""
    if position:
        return st('camera_fallback', lang, number=position[0], total=position[1])
    return st('camera_plain', lang)


def plain_detail(detail, camera, name):
    """What the box said, with the camera's id swapped for the name on the screen (the box's own words may
    carry the id: "could not get a picture from front_door")."""
    return str(detail or "").replace(str(camera), name) if camera else str(detail or "")


def rgba(colour, opacity):
    c = QColor(colour)
    return f'rgba({c.red()}, {c.green()}, {c.blue()}, {opacity})'


def picture_pixmap(data):
    pix = QPixmap()
    if data:
        pix.loadFromData(bytes(data))
    return pix


def is_rtl(lang):
    return (lang or language()) == "he"


class Jobs:
    """Box commands on a worker thread, each result handed back on the GUI thread by a poll timer."""
    def __init__(self, owner):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = []
        self.timer = QTimer(owner)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.poll)
        self.closed = False

    def submit(self, task, done, failed):
        if self.closed:
            return
        self.pending.append((self.pool.submit(task), done, failed))
        self.timer.start()

    def busy(self):
        return bool(self.pending)

    def poll(self):
        ready = [job for job in self.pending if job[0].done()]
        self.pending = [job for job in self.pending if not job[0].done()]
        if not self.pending:
            self.timer.stop()
        for future, done, failed in ready:
            if self.closed or future.cancelled():
                continue
            try:
                value = future.result()
            except Exception as exc:  # noqa: BLE001 - every failure becomes plain words on screen
                failed(exc)
            else:
                done(value)

    def close(self):
        self.closed = True
        self.timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)


# ----------------------------------------------------------------------------
# The picture
# ----------------------------------------------------------------------------
class MapStage(QWidget):
    """The camera picture on a blurred bed of itself, with the places, the hand-drawn areas and the lines drawn
    over it, coloured by whose they are. Modes: ``regions`` (click a number), ``draw`` (corners), ``line``
    (two points and a side), ``view`` (no clicks: the summary and the saved map)."""
    region_clicked = Signal(int)
    hand_clicked = Signal(int)
    line_clicked = Signal(int)
    draft_changed = Signal()
    draft_closed = Signal(list)
    hand_changed = Signal(int)
    line_changed = Signal()
    side_picked = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pix = QPixmap()
        self.ambient = QPixmap()
        self.regions = ()
        self.labels = {}
        self.answers = {}
        self.hands = []
        self.lines = []
        self.walls = []                    # (Boundary, sure) along the places answered "it is the boundary"
        self.rest = None
        self.muted = False                 # the card's old photo behind an error: not the box's own picture
        self.mode = "view"
        self.selected_region = 0
        self.selected_hand = -1
        self.selected_line = -1
        self.draft = None                  # corners of the area being drawn
        self.line_draft = None             # the line's points being placed (0 to 2)
        self.pointer = None
        self.dragging = None               # (hand index, corner index)
        self.loading = ("", "")
        self.waiting = False
        self.text_direction = Qt.LayoutDirection.LeftToRight
        self.veil = 0.
        self.spin = 0.
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.fade = QVariantAnimation(self)
        self.fade.setDuration(motion.PANE_MS); self.fade.setEasingCurve(motion.EASING)
        self.fade.valueChanged.connect(self._veil)
        self.fade.finished.connect(self._faded)
        self.spinner = QVariantAnimation(self)
        self.spinner.setDuration(1100); self.spinner.setStartValue(0.); self.spinner.setEndValue(360.)
        self.spinner.setLoopCount(-1)
        self.spinner.valueChanged.connect(self._spin)

    # --- state -------------------------------------------------------------
    def _veil(self, value):
        self.veil = float(value); self.update()

    def _spin(self, value):
        self.spin = float(value); self.update()

    def _faded(self):
        if self.veil == 0.:
            self.spinner.stop()

    def set_loading(self, title="", hint="", instant=False):
        self.waiting = bool(title)
        if title:
            self.loading = (title, hint)
            self.spinner.start()
        target = 1. if title else 0.
        self.fade.stop()
        if instant or not self.isVisible():
            self._veil(target); self._faded()
            return
        self.fade.setStartValue(self.veil); self.fade.setEndValue(target); self.fade.start()

    def is_loading(self):
        """The box is working: clicks wait (the veil may still be fading out after it answered)."""
        return self.waiting

    def set_picture(self, pix):
        self.pix = QPixmap(pix) if pix is not None else QPixmap()
        self.ambient = ambient_picture(self.pix) if not self.pix.isNull() else QPixmap()
        self.update()

    def set_regions(self, regions, labels):
        self.regions = tuple(regions); self.labels = dict(labels); self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.draft = None; self.line_draft = None; self.dragging = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def start_draft(self):
        self.draft = []; self.selected_hand = -1; self.update(); self.draft_changed.emit()

    def stop_draft(self):
        self.draft = None; self.update(); self.draft_changed.emit()

    def undo_corner(self):
        if self.draft:
            self.draft.pop(); self.update(); self.draft_changed.emit()

    def close_draft(self):
        if self.draft is not None and len(self.draft) >= 3:
            points, self.draft = self.draft, None
            self.update()
            self.draft_closed.emit(points)
            return True
        return False

    def start_line(self):
        self.line_draft = []; self.selected_line = -1; self.update(); self.line_changed.emit()

    def stop_line(self):
        self.line_draft = None; self.update(); self.line_changed.emit()

    # --- geometry ----------------------------------------------------------
    def core_rect(self):
        return QRectF(self.rect()).adjusted(7, 7, -7, -7)

    def picture_rect(self):
        core = self.core_rect()
        if self.pix.isNull():
            return core
        scale = min(core.width() / self.pix.width(), core.height() / self.pix.height())
        w, h = self.pix.width() * scale, self.pix.height() * scale
        return QRectF(core.center().x() - w / 2, core.center().y() - h / 2, w, h)

    def to_stage(self, point):
        rect = self.picture_rect()
        return QPointF(rect.x() + point[0] * rect.width(), rect.y() + point[1] * rect.height())

    def to_fraction(self, point):
        rect = self.picture_rect()
        if rect.isEmpty():
            return (0., 0.)
        return (round(max(0., min(1., (point.x() - rect.x()) / rect.width())), 4),
                round(max(0., min(1., (point.y() - rect.y()) / rect.height())), 4))

    def region_at(self, point):
        rect = self.picture_rect()
        if self.pix.isNull() or not rect.contains(point):
            return 0
        hits = [r for r in self.regions if polygon_path(r.points, rect).contains(point)]
        return min(hits, key=lambda r: polygon_area(r.points)).number if hits else 0

    def hand_at(self, point):
        rect = self.picture_rect()
        hits = [i for i, h in enumerate(self.hands) if h.closed and polygon_path(h.points, rect).contains(point)]
        return min(hits, key=lambda i: polygon_area(self.hands[i].points)) if hits else -1

    def corner_at(self, point):
        if not 0 <= self.selected_hand < len(self.hands):
            return -1
        for i, corner in enumerate(self.hands[self.selected_hand].points):
            if math.hypot(*(point - self.to_stage(corner)).toTuple()) <= HANDLE:
                return i
        return -1

    def line_at(self, point):
        best, found = HANDLE, -1
        for i, line in enumerate(self.lines):
            a, b = self.to_stage(line.a), self.to_stage(line.b)
            d = b - a
            length = d.x() ** 2 + d.y() ** 2
            t = 0. if not length else max(0., min(1., QPointF.dotProduct(point - a, d) / length))
            distance = math.hypot(*(point - (a + d * t)).toTuple())
            if distance <= best:
                best, found = distance, i
        return found

    # --- mouse -------------------------------------------------------------
    def mousePressEvent(self, event):
        if not self.isEnabled() or self.waiting or self.pix.isNull():
            return
        self.setFocus()
        point = event.position()
        inside = self.picture_rect().contains(point)
        if event.button() == Qt.MouseButton.RightButton:
            if self.mode == "draw" and self.draft is not None:
                self.undo_corner()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode == "regions":
            hit = self.region_at(point)
            if hit:
                self.region_clicked.emit(hit)
        elif self.mode == "draw":
            if self.draft is not None:
                if len(self.draft) >= 3 and math.hypot(*(point - self.to_stage(self.draft[0])).toTuple()) <= HANDLE:
                    self.close_draft()
                elif inside and len(self.draft) < MAX_CORNERS and not (
                        self.draft and math.hypot(*(point - self.to_stage(self.draft[-1])).toTuple()) <= 4):
                    self.draft.append(list(self.to_fraction(point))); self.update(); self.draft_changed.emit()
                return
            corner = self.corner_at(point)
            if corner >= 0:
                self.dragging = (self.selected_hand, corner)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            hit = self.hand_at(point)
            if hit >= 0:
                self.hand_clicked.emit(hit)
        elif self.mode == "line":
            if self.line_draft is not None:
                if not inside:
                    return
                fraction = self.to_fraction(point)
                if len(self.line_draft) < 2:
                    if not self.line_draft or tuple(self.line_draft[0]) != fraction:
                        self.line_draft.append(fraction); self.update(); self.line_changed.emit()
                else:
                    self.side_picked.emit(fraction)
                return
            hit = self.line_at(point)
            if hit >= 0:
                self.line_clicked.emit(hit)

    def mouseDoubleClickEvent(self, event):
        if self.mode == "draw" and self.draft is not None and event.button() == Qt.MouseButton.LeftButton:
            self.close_draft()            # its first press already added the last corner
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        point = event.position()
        self.pointer = point if self.picture_rect().contains(point) else None
        if self.dragging is not None:
            hand, corner = self.dragging
            if 0 <= hand < len(self.hands):
                self.hands[hand].points[corner] = list(self.to_fraction(point))
                self.hand_changed.emit(hand)
            self.update()
            return
        cursor = Qt.CursorShape.ArrowCursor
        if not self.waiting and not self.pix.isNull():
            if self.mode == "regions" and self.region_at(point):
                cursor = Qt.CursorShape.PointingHandCursor
            elif self.mode == "draw":
                if self.draft is not None:
                    cursor = Qt.CursorShape.CrossCursor
                elif self.corner_at(point) >= 0:
                    cursor = Qt.CursorShape.OpenHandCursor
                elif self.hand_at(point) >= 0:
                    cursor = Qt.CursorShape.PointingHandCursor
            elif self.mode == "line":
                cursor = Qt.CursorShape.CrossCursor if self.line_draft is not None else (
                    Qt.CursorShape.PointingHandCursor if self.line_at(point) >= 0 else Qt.CursorShape.ArrowCursor)
        self.setCursor(cursor)
        if self.draft is not None or self.line_draft is not None:
            self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging is not None:
            self.dragging = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def leaveEvent(self, event):
        self.pointer = None; self.update()
        super().leaveEvent(event)

    # --- painting ----------------------------------------------------------
    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        outer = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        p.setBrush(QColor(t['raised'])); p.setPen(QPen(alpha(t['text'], .08), 1)); p.drawRoundedRect(outer, 22, 22)
        core = self.core_rect()
        clip = QPainterPath(); clip.addRoundedRect(core, 15, 15); p.setClipPath(clip)
        if not self.pix.isNull():
            scale = max(core.width() / self.ambient.width(), core.height() / self.ambient.height())
            w, h = self.ambient.width() * scale, self.ambient.height() * scale
            p.drawPixmap(QRectF(core.center().x() - w / 2, core.center().y() - h / 2, w, h), self.ambient,
                         QRectF(self.ambient.rect()))
            p.fillRect(core, alpha(t['bg'], .62))
            rect = self.picture_rect()
            p.drawPixmap(rect, self.pix, QRectF(self.pix.rect()))
            if self.muted:
                p.fillRect(rect, alpha(t['bg'], .55))
            p.save(); p.setClipRect(rect, Qt.ClipOperation.IntersectClip)
            if self.rest in COLOURS:
                p.fillRect(rect, alpha(COLOURS[self.rest], .2))
            self.paint_regions(p, rect, t)
            self.paint_hands(p, rect, t)
            self.paint_draft(p, rect, t)
            self.paint_lines(p, rect, t)
            self.paint_walls(p, rect, t)
            self.paint_line_draft(p, rect, t)
            p.restore()
            self.paint_numbers(p, t)
        if self.veil > 0:
            self.paint_veil(p, core, t)
        p.setClipping(False); p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(alpha(t['text'], .1), 1))
        p.drawRoundedRect(core, 15, 15)

    def paint_regions(self, p, rect, t):
        active = self.mode == "regions"
        for region in self.regions:                    # largest first: a small place stays on top
            shape = polygon_path(region.points, rect)
            choice = self.answers.get(region.number)
            choice = "mine" if choice == BOUNDARY else choice
            p.setOpacity(1. if active or self.mode == "view" else .4)
            if choice in CHOICES:
                p.fillPath(shape, alpha(COLOURS[choice], .82 if choice == "hide" else .42))
                p.setPen(QPen(alpha('#ffffff', .55) if choice == "hide" else QColor(COLOURS[choice]), 1.6))
            else:
                p.fillPath(shape, alpha(number_colour(region.number), .1))
                p.setPen(QPen(alpha(number_colour(region.number), .9), 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(shape)
        p.setOpacity(1.)
        lit = next((r for r in self.regions if r.number == self.selected_region), None)
        if lit is not None and active:
            shape = polygon_path(lit.points, rect)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(alpha('#000000', .55), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(shape)
            p.setPen(QPen(QColor('#ffffff'), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(shape)

    def paint_hands(self, p, rect, t):
        active = self.mode == "draw"
        for i, hand in enumerate(self.hands):
            if not hand.closed:
                continue
            shape = polygon_path(hand.points, rect)
            p.setOpacity(1. if active or self.mode == "view" else .55)
            if hand.choice in CHOICES:
                p.fillPath(shape, alpha(COLOURS[hand.choice], .82 if hand.choice == "hide" else .42))
                pen = QPen(alpha('#ffffff', .6) if hand.choice == "hide" else QColor(COLOURS[hand.choice]), 2)
            else:
                p.fillPath(shape, alpha('#ffffff', .1))
                pen = QPen(QColor('#ffffff'), 2, Qt.PenStyle.DashLine)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(pen); p.drawPath(shape)
        p.setOpacity(1.)
        if active and 0 <= self.selected_hand < len(self.hands) and self.draft is None:
            hand = self.hands[self.selected_hand]
            shape = polygon_path(hand.points, rect)
            p.setPen(QPen(alpha('#000000', .55), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(shape)
            p.setPen(QPen(QColor('#ffffff'), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(shape)
            for corner in hand.points:
                c = self.to_stage(corner)
                p.setBrush(QColor(t['bg'])); p.setPen(QPen(QColor('#ffffff'), 2)); p.drawEllipse(c, 6, 6)

    def paint_draft(self, p, rect, t):
        if self.draft is None:
            return
        accent = QColor(t['action'])
        if self.draft:
            path = polygon_path(self.draft, rect) if len(self.draft) < 3 else QPainterPath()
            if len(self.draft) >= 3:
                path.moveTo(self.to_stage(self.draft[0]))
                for corner in self.draft[1:]:
                    path.lineTo(self.to_stage(corner))
                fill = QPainterPath(path); fill.closeSubpath()
                p.fillPath(fill, alpha(t['action'], .16))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(accent, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(path)
            if self.pointer is not None:
                p.setPen(QPen(alpha(t['action'], .8), 2, Qt.PenStyle.DashLine))
                p.drawLine(self.to_stage(self.draft[-1]), self.pointer)
            for i, corner in enumerate(self.draft):
                c = self.to_stage(corner)
                closable = i == 0 and len(self.draft) >= 3
                if closable:
                    p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(alpha(t['action'], .45), 2)); p.drawEllipse(c, 12, 12)
                p.setBrush(QColor(t['bg'])); p.setPen(QPen(accent, 2)); p.drawEllipse(c, 7 if closable else 6, 7 if closable else 6)

    def arrow(self, p, line, colour, rect):
        """The arrow from the line's middle towards our side."""
        a, b = self.to_stage(line.a), self.to_stage(line.b)
        mid = (a + b) / 2
        vx, vy = inward_vector(line)
        direction = QPointF(vx * rect.width(), vy * rect.height())
        norm = math.hypot(direction.x(), direction.y()) or 1.
        unit = direction / norm
        length = max(34., min(64., min(rect.width(), rect.height()) * .09))
        tip = mid + unit * length
        side = QPointF(-unit.y(), unit.x())
        head = QPolygonF([tip, tip - unit * 13 + side * 8, tip - unit * 13 - side * 8])
        for pen_colour, width in ((alpha('#000000', .55), 7), (colour, 3)):
            p.setPen(QPen(pen_colour, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(mid, tip - unit * 9)
        p.setPen(QPen(alpha('#000000', .55), 3)); p.setBrush(colour); p.drawPolygon(head)

    def paint_lines(self, p, rect, t):
        for i, line in enumerate(self.lines):
            selected = i == self.selected_line and self.mode == "line"
            colour = QColor(t['action'] if selected else '#ffffff')
            p.setOpacity(1. if self.mode in ("line", "view") else .7)
            a, b = self.to_stage(line.a), self.to_stage(line.b)
            for pen_colour, width in ((alpha('#000000', .55), 8), (colour, 3.5)):
                p.setPen(QPen(pen_colour, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)); p.drawLine(a, b)
            self.arrow(p, line, colour, rect)
            for end in (a, b):
                p.setBrush(QColor(t['bg'])); p.setPen(QPen(colour, 2.2)); p.drawEllipse(end, 6, 6)
            if line.name:
                p.setLayoutDirection(self.text_direction)
                font = QFont(self.font()); font.setPixelSize(12); font.setWeight(QFont.Weight.DemiBold); p.setFont(font)
                width = p.fontMetrics().horizontalAdvance(line.name) + 18
                vx, vy = inward_vector(line)
                away = QPointF(-vx * rect.width(), -vy * rect.height())
                away = away / (math.hypot(away.x(), away.y()) or 1.) * 20
                centre = a + (b - a) * .25 + away
                box = QRectF(centre.x() - width / 2, centre.y() - 11, width, 22)
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(alpha(t['bg'], .82)); p.drawRoundedRect(box, 11, 11)
                p.setPen(QColor(t['text'])); p.drawText(box, Qt.AlignmentFlag.AlignCenter, line.name)
        p.setOpacity(1.)

    def paint_walls(self, p, rect, t):
        """A boundary place's line: with its arrow when the side is known, dashed with a "?" while it is not."""
        for line, sure in self.walls:
            a, b = self.to_stage(line.a), self.to_stage(line.b)
            colour = QColor('#ffffff') if sure else QColor(t['warning'])
            style = Qt.PenStyle.SolidLine if sure else Qt.PenStyle.DashLine
            for pen_colour, width in ((alpha('#000000', .55), 8), (colour, 3.5)):
                p.setPen(QPen(pen_colour, width, style, Qt.PenCapStyle.RoundCap)); p.drawLine(a, b)
            if sure:
                self.arrow(p, line, colour, rect)
                continue
            centre = (a + b) / 2
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(colour); p.drawEllipse(centre, 12, 12)
            font = QFont(self.font()); font.setPixelSize(15); font.setWeight(QFont.Weight.Bold); p.setFont(font)
            p.setPen(QColor(t['bg'])); p.drawText(QRectF(centre.x() - 12, centre.y() - 12, 24, 24), Qt.AlignmentFlag.AlignCenter, '?')

    def paint_line_draft(self, p, rect, t):
        if self.line_draft is None:
            return
        accent = QColor(t['action'])
        points = [self.to_stage(pt) for pt in self.line_draft]
        if len(points) == 1 and self.pointer is not None:
            p.setPen(QPen(alpha(t['action'], .8), 2.5, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(points[0], self.pointer)
        if len(points) == 2:
            a, b = self.line_draft
            if self.pointer is not None:
                fraction = self.to_fraction(self.pointer)
                try:
                    preview = boundary_toward(a, b, fraction)
                except ValueError:
                    preview = None
                if preview is not None:
                    vx, vy = inward_vector(preview)
                    d = QPointF((b[0] - a[0]) * rect.width(), (b[1] - a[1]) * rect.height())
                    norm = math.hypot(d.x(), d.y()) or 1.
                    along = d / norm * (rect.width() + rect.height()) * 2
                    out = QPointF(vx * rect.width(), vy * rect.height())
                    out = out / (math.hypot(out.x(), out.y()) or 1.) * max(70., min(rect.width(), rect.height()) * .22)
                    half = QPolygonF([points[0] - along, points[1] + along, points[1] + along + out, points[0] - along + out])
                    p.setPen(Qt.PenStyle.NoPen); p.setBrush(alpha(t['action'], .28)); p.drawPolygon(half)
                    self.arrow(p, preview, accent, rect)
            for pen_colour, width in ((alpha('#000000', .55), 8), (accent, 3.5)):
                p.setPen(QPen(pen_colour, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(points[0], points[1])
        for c in points:
            p.setBrush(QColor(t['bg'])); p.setPen(QPen(accent, 2.2)); p.drawEllipse(c, 7, 7)

    def paint_numbers(self, p, t):
        if not self.regions:
            return
        font = QFont(self.font()); font.setPixelSize(13); font.setWeight(QFont.Weight.Bold)
        p.setFont(font)
        dim = self.mode not in ("regions", "view")
        for region in self.regions:
            x, y = self.labels.get(region.number, (.5, .5))
            centre = self.to_stage((x, y))
            choice = self.answers.get(region.number)
            selected = region.number == self.selected_region and self.mode == "regions"
            radius = 15 if selected else 12.5
            p.setOpacity(.45 if dim else 1.)
            if selected:
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(alpha(t['action'], .35)); p.drawEllipse(centre, radius + 6, radius + 6)
            choice = "mine" if choice == BOUNDARY else choice
            fill = QColor(COLOURS[choice]) if choice in CHOICES else QColor('#ffffff')
            if choice == "hide":
                fill = QColor('#16191c')
            p.setBrush(fill)
            p.setPen(QPen(QColor('#ffffff') if choice in CHOICES else QColor(number_colour(region.number)), 2.4))
            p.drawEllipse(centre, radius, radius)
            p.setPen(QColor('#ffffff') if choice in CHOICES else QColor('#0c1218'))
            p.drawText(QRectF(centre.x() - radius, centre.y() - radius, radius * 2, radius * 2),
                       Qt.AlignmentFlag.AlignCenter, str(region.number))
        p.setOpacity(1.)

    def paint_veil(self, p, core, t):
        p.setLayoutDirection(self.text_direction)
        p.fillRect(core, alpha(t['bg'], (.72 if not self.pix.isNull() else .2) * self.veil))
        p.setOpacity(self.veil)
        centre = core.center() - QPointF(0, 30)
        p.setPen(QPen(alpha(t['text'], .12), 3)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(centre, 20, 20)
        p.setPen(QPen(QColor(t['action']), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(QRectF(centre.x() - 20, centre.y() - 20, 40, 40), int(-self.spin * 16), 100 * 16)
        font = QFont(self.font()); font.setPixelSize(19); font.setWeight(QFont.Weight.DemiBold); p.setFont(font)
        p.setPen(QColor(t['text']))
        p.drawText(QRectF(core.left() + 24, centre.y() + 38, core.width() - 48, 28), Qt.AlignmentFlag.AlignCenter,
                   self.loading[0])
        font.setPixelSize(14); font.setWeight(QFont.Weight.Normal); p.setFont(font); p.setPen(QColor(t['secondary']))
        p.drawText(QRectF(core.left() + 60, centre.y() + 70, core.width() - 120, 48),
                   Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, self.loading[1])
        p.setOpacity(1.)


# ----------------------------------------------------------------------------
# The side panel's parts
# ----------------------------------------------------------------------------
def keyboard_focus(widget):
    return widget.hasFocus() and bool(widget.property('keyboardFocus'))


class ChoiceButton(QAbstractButton):
    """One of the four big answers: its map colour, its word and a short hint."""
    def __init__(self, choice, lang, parent=None):
        super().__init__(parent)
        self.choice = choice
        self.setText(st('choice_' + choice, lang))
        self.hint = st('hint_' + choice, lang)
        self.setAccessibleName(self.text()); self.setAccessibleDescription(self.hint)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty('handlesMotion', True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.compact = False
        self.setFixedHeight(52)

    def set_compact(self, compact):
        """One line, the hint as a tooltip: a short screen (setup at 1366x768) keeps the open card in view."""
        self.compact = compact
        self.setFixedHeight(40 if compact else 52)
        self.setToolTip(self.hint if compact else '')
        self.update()

    def sizeHint(self):
        return QSize(170, self.height())

    def minimumSizeHint(self):
        return QSize(80, self.height())

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click(); event.accept(); return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(1. if self.isEnabled() else .45)
        on = self.isChecked()
        swatch = QColor(COLOURS[self.choice])
        ring = QColor('#ffffff') if self.choice == "hide" else swatch
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(alpha(ring if self.choice != "hide" else t['text'], .14) if on else QColor(t['raised']))
        p.setPen(QPen(ring if on else QColor(t['action'] if keyboard_focus(self) else t['border']), 2 if on else 1))
        p.drawRoundedRect(rect, 14, 14)
        rtl = self.layoutDirection() == Qt.LayoutDirection.RightToLeft
        if self.compact:
            font = QFont(self.font()); font.setPixelSize(12 if self.width() < 100 else 13)
            font.setWeight(QFont.Weight.DemiBold); p.setFont(font)
            p.setPen(QColor(t['text']))
            p.drawText(rect.adjusted(4, 0, -4, -3), Qt.AlignmentFlag.AlignCenter,
                       p.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, int(rect.width() - 8)))
            bar = QRectF(rect.left() + 12, rect.bottom() - 6, rect.width() - 24, 3)
            p.setPen(QPen(alpha('#ffffff', .6), 1) if self.choice == "hide" else Qt.PenStyle.NoPen)
            p.setBrush(swatch); p.drawRoundedRect(bar, 1.5, 1.5)
            return
        dot = QRectF(rect.right() - 30 if rtl else rect.left() + 14, rect.center().y() - 8, 16, 16)
        p.setBrush(swatch); p.setPen(QPen(alpha('#ffffff', .7), 1.2) if self.choice == "hide" else Qt.PenStyle.NoPen)
        p.drawEllipse(dot)
        if on:
            p.setPen(QPen(QColor('#ffffff'), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            c = dot.center(); check = QPainterPath()
            check.moveTo(c + QPointF(-3.5, 0)); check.lineTo(c + QPointF(-1, 2.6)); check.lineTo(c + QPointF(3.8, -2.6))
            p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(check)
        text = rect.adjusted(14, 5, -40, -5) if rtl else rect.adjusted(40, 5, -14, -5)
        align = Qt.AlignmentFlag.AlignRight if rtl else Qt.AlignmentFlag.AlignLeft
        font = QFont(self.font()); font.setPixelSize(15); font.setWeight(QFont.Weight.DemiBold); p.setFont(font)
        p.setPen(QColor(t['text']))
        p.drawText(QRectF(text.left(), text.top(), text.width(), 22), align | Qt.AlignmentFlag.AlignVCenter, self.text())
        font.setPixelSize(12); font.setWeight(QFont.Weight.Normal); p.setFont(font)
        p.setPen(QColor(t['secondary'] if on else t['muted']))
        p.drawText(QRectF(text.left(), text.top() + 21, text.width(), 18), align | Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(self.hint, Qt.TextElideMode.ElideRight, int(text.width())))


class QuietToggle(QPushButton):
    """Skip / Don't know: small text buttons that read as chosen when checked."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggled.connect(self.restyle)
        self.restyle(False)

    def restyle(self, on):
        t = colors(self)
        self.setStyleSheet(f'QPushButton {{ background: {t["raised"] if on else "transparent"}; color: {t["text"] if on else t["muted"]}; '
                           f'border: 1px solid {t["border"] if on else "transparent"}; border-radius: 12px; padding: 3px 12px; '
                           f'min-height: 20px; font-size: 10pt; font-weight: 500; }}'
                           f'QPushButton:hover {{ color: {t["text"]}; }}')


class TextAction(QPushButton):
    """A quiet text button for a secondary action inside the panel (delete, undo, cancel)."""
    def __init__(self, text, tone='secondary', parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        t = colors(self)
        self.setStyleSheet(f'QPushButton {{ background: transparent; color: {t[tone]}; border: none; padding: 4px 6px; '
                           f'min-height: 20px; font-size: 10.5pt; font-weight: 600; }}'
                           f'QPushButton:hover {{ color: {t["text"]}; }} QPushButton:disabled {{ color: {t["border"]}; }}')


class Badge(QWidget):
    """A number in its colour (a region), or a colour dot (an area), or a line glyph (a boundary)."""
    def __init__(self, number=0, colour=None, line=False, parent=None):
        super().__init__(parent)
        self.number, self.colour, self.line = number, colour, line
        self.choice = None
        self.setFixedSize(28, 28)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(2, 2, 24, 24)
        if self.line:
            p.setPen(QPen(QColor(t['text']), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(5, 21), QPointF(23, 7))
            p.setPen(QPen(QColor(t['action']), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(14, 14), QPointF(19, 20))
            return
        choice = "mine" if self.choice == BOUNDARY else self.choice
        if self.number:
            fill = QColor(COLOURS[choice]) if choice in CHOICES else QColor('#ffffff')
            if choice == "hide":
                fill = QColor('#16191c')
            p.setBrush(fill)
            p.setPen(QPen(QColor('#ffffff') if choice in CHOICES else QColor(number_colour(self.number)), 2.2))
            p.drawEllipse(box)
            font = QFont(self.font()); font.setPixelSize(12); font.setWeight(QFont.Weight.Bold); p.setFont(font)
            p.setPen(QColor('#ffffff') if choice in CHOICES else QColor('#0c1218'))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, str(self.number))
            return
        p.setBrush(QColor(COLOURS[choice]) if choice in CHOICES else Qt.BrushStyle.NoBrush)
        if choice in (None, "hide"):
            p.setPen(QPen(QColor('#ffffff'), 1.6, Qt.PenStyle.DashLine if choice is None else Qt.PenStyle.SolidLine))
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(box.adjusted(4, 4, -4, -4), 4, 4)


class Tick(QWidget):
    """A filled circle with a check: saved."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['ok'])); p.drawEllipse(QRectF(1, 1, 28, 28))
        p.setPen(QPen(QColor(t['bg']), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        check = QPainterPath(); check.moveTo(9, 15.5); check.lineTo(13.2, 19.5); check.lineTo(21, 11)
        p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(check)


class ItemRow(QFrame):
    """One place, drawn area or line in the panel's list: a header that selects it, and its answers below while
    selected."""
    selected = Signal()

    def __init__(self, badge, title, parent=None):
        super().__init__(parent)
        self.setObjectName('sceneItem')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.lit = False
        self.root = QVBoxLayout(self); self.root.setContentsMargins(12, 10, 12, 10); self.root.setSpacing(10)
        head = QHBoxLayout(); head.setSpacing(10)
        self.badge = badge; head.addWidget(badge)
        self.title = QLabel(title); self.title.setObjectName('sceneItemTitle'); head.addWidget(self.title, 1)
        self.tag = QLabel(''); self.tag.setObjectName('sceneItemTag'); head.addWidget(self.tag)
        self.head = head
        self.lit_only = []                     # header widgets shown only while selected (in place of the tag)
        self.root.addLayout(head)
        self.body = QWidget(); self.body.setObjectName('sceneItemBody')
        self.body_layout = QVBoxLayout(self.body); self.body_layout.setContentsMargins(0, 0, 0, 0); self.body_layout.setSpacing(10)
        self.root.addWidget(self.body)
        self.body.hide()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.restyle()

    def mousePressEvent(self, event):
        if not self.lit and event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit()
        super().mousePressEvent(event)

    def set_lit(self, lit):
        self.lit = lit
        self.body.setVisible(lit)
        for widget in self.lit_only:
            widget.setVisible(lit)
        if self.lit_only:
            self.tag.setVisible(not lit)
        self.setCursor(Qt.CursorShape.ArrowCursor if lit else Qt.CursorShape.PointingHandCursor)
        self.restyle()

    def restyle(self):
        t = colors(self)
        self.setStyleSheet(
            f'QFrame#sceneItem {{ background: {t["raised"] if self.lit else t["surface"]}; border: 1px solid '
            f'{t["action"] if self.lit else t["border"]}; border-radius: 14px; }}'
            f'QWidget#sceneItemBody {{ background: transparent; }}'
            f'QLabel#sceneItemTitle {{ color: {t["text"]}; font-size: 11.5pt; font-weight: 600; }}'
            f'QLabel#sceneItemTag {{ color: {t["muted"]}; font-size: 10pt; }}'
            f'QLineEdit {{ padding: 7px 10px; min-height: 18px; font-size: 10.5pt; border-radius: 10px; }}')


class ChoiceGrid(QWidget):
    """The four big answers, two by two; on a short screen, four in a row."""
    picked = Signal(str)

    def __init__(self, lang, parent=None, compact=False):
        super().__init__(parent)
        self.grid = QGridLayout(self); self.grid.setContentsMargins(0, 0, 0, 0); self.grid.setSpacing(8)
        self.buttons = {}
        for choice in CHOICES:
            button = ChoiceButton(choice, lang)
            button.clicked.connect(lambda checked=False, c=choice: self.picked.emit(c))
            self.buttons[choice] = button
        self.set_compact(compact)

    def set_compact(self, compact):
        for i, button in enumerate(self.buttons.values()):
            self.grid.removeWidget(button)
            button.set_compact(compact)
            self.grid.addWidget(button, 0 if compact else i // 2, i if compact else i % 2)
        self.grid.setSpacing(6 if compact else 8)

    def show_choice(self, choice):
        for c, button in self.buttons.items():
            button.setChecked(c == choice)


class WideChoice(ChoiceButton):
    """"It is the boundary": the region's fifth answer, as wide as the grid."""
    def __init__(self, lang, parent=None):
        super().__init__("mine", lang, parent)
        self.choice = BOUNDARY
        self.setText(st('choice_boundary', lang)); self.hint = st('hint_boundary', lang)
        self.setAccessibleName(self.text()); self.setAccessibleDescription(self.hint)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(1. if self.isEnabled() else .45)
        on = self.isChecked()
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(alpha('#ffffff', .1) if on else QColor(t['raised']))
        p.setPen(QPen(QColor('#ffffff') if on else QColor(t['action'] if keyboard_focus(self) else t['border']), 2 if on else 1))
        p.drawRoundedRect(rect, 14, 14)
        rtl = self.layoutDirection() == Qt.LayoutDirection.RightToLeft
        x = rect.right() - 34 if rtl else rect.left() + 12
        y = rect.center().y()
        if self.compact:
            y = rect.center().y()
        for pen_colour, width in ((QColor(t['bg']), 6), (QColor('#ffffff'), 2.6)):
            p.setPen(QPen(pen_colour, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(x, y + 7), QPointF(x + 22, y - 7))
        text = rect.adjusted(14, 5, -44, -5) if rtl else rect.adjusted(44, 5, -14, -5)
        align = Qt.AlignmentFlag.AlignRight if rtl else Qt.AlignmentFlag.AlignLeft
        font = QFont(self.font()); font.setPixelSize(14); font.setWeight(QFont.Weight.DemiBold); p.setFont(font)
        p.setPen(QColor(t['text']))
        if self.compact:
            text = rect.adjusted(14, 0, -44, 0) if rtl else rect.adjusted(44, 0, -14, 0)
            p.drawText(text, align | Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, int(text.width())))
            return
        p.drawText(QRectF(text.left(), text.top(), text.width(), 22), align | Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, int(text.width())))
        font.setPixelSize(12); font.setWeight(QFont.Weight.Normal); p.setFont(font)
        p.setPen(QColor(t['secondary'] if on else t['muted']))
        p.drawText(QRectF(text.left(), text.top() + 21, text.width(), 18), align | Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(self.hint, Qt.TextElideMode.ElideRight, int(text.width())))


class SideQuestion(QFrame):
    """"Which side of the railing is ours?", with the picture's own words for the two sides."""
    picked = Signal(str)

    def __init__(self, lang, parent=None):
        super().__init__(parent)
        self.lang = lang
        self.setObjectName('sceneSide'); self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        t = colors(self)
        self.setStyleSheet(f'QFrame#sceneSide {{ background: {rgba(t["warning"], .08)}; border: 1px solid '
                           f'{rgba(t["warning"], .45)}; border-radius: 12px; }}'
                           f'QFrame#sceneSide QLabel {{ background: transparent; }}')
        self.lay = lay = QVBoxLayout(self); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(8)
        self.question = words('', 'body'); self.question.setStyleSheet(f'color: {t["text"]}; font-size: 11pt; font-weight: 600;')
        lay.addWidget(self.question)
        self.hint = words(st('side_hint', lang), 'muted'); lay.addWidget(self.hint)
        row = QHBoxLayout(); row.setSpacing(8)
        self.buttons = {}
        for side in ('left', 'right'):
            button = QPushButton(''); button.setCheckable(True); button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(f'QPushButton {{ background: {t["raised"]}; color: {t["text"]}; border: 1px solid {t["border"]}; '
                                 f'border-radius: 12px; padding: 0 10px; min-height: 0; font-size: 10.5pt; font-weight: 600; }}'
                                 f'QPushButton:checked {{ border: 2px solid {t["action"]}; background: {rgba(t["action"], .14)}; }}')
            button.clicked.connect(lambda checked=False, side=side: self.picked.emit(side))
            button.setFixedHeight(38)
            row.addWidget(button, 1); self.buttons[side] = button
        lay.addLayout(row)

    def set_compact(self, compact):
        self.hint.setVisible(not compact)
        self.lay.setContentsMargins(*((10, 6, 10, 8) if compact else (12, 10, 12, 12)))
        self.lay.setSpacing(6 if compact else 8)
        for button in self.buttons.values():
            button.setFixedHeight(32 if compact else 38)

    def ask(self, name, sides, chosen=''):
        self.question.setText(st('side_question', self.lang, name=name))
        for side, button in self.buttons.items():
            button.setText(sides[side]); button.setChecked(side == chosen)


class RegionRow(ItemRow):
    answered = Signal(int, str)
    renamed = Signal(int, str)
    side_picked = Signal(int, str)

    def set_compact(self, compact):
        self.choices.set_compact(compact)
        self.boundary.set_compact(compact)
        self.side.set_compact(compact)
        self.body_layout.setSpacing(6 if compact else 10)
        self.root.setContentsMargins(*((10, 8, 10, 8) if compact else (12, 10, 12, 10)))
        self.root.setSpacing(6 if compact else 10)
        self.name.setFixedHeight(32 if compact else self.name.sizeHint().height())

    def __init__(self, number, lang, parent=None):
        self.number, self.lang = number, lang
        super().__init__(Badge(number), st('region_title', lang, number=number), parent)
        self.name = QLineEdit(); self.name.setPlaceholderText(st('name_placeholder', lang)); self.name.setMaxLength(40)
        self.name.setAccessibleName(st('name_placeholder', lang))
        self.name.textEdited.connect(lambda text: (self.renamed.emit(self.number, text), self.refresh()))
        self.body_layout.addWidget(self.name)
        self.choices = ChoiceGrid(lang); self.body_layout.addWidget(self.choices)
        self.choices.picked.connect(lambda c: self.answered.emit(self.number, c))
        self.boundary = WideChoice(lang); self.body_layout.addWidget(self.boundary)
        self.boundary.clicked.connect(lambda: self.answered.emit(self.number, BOUNDARY))
        self.side = SideQuestion(lang); self.side.hide(); self.body_layout.addWidget(self.side)
        self.side.picked.connect(lambda side: self.side_picked.emit(self.number, side))
        self.skip = QuietToggle(st('choice_skip', lang)); self.unknown = QuietToggle(st('choice_unknown', lang))
        self.skip.clicked.connect(lambda: self.answered.emit(self.number, SKIP))
        self.unknown.clicked.connect(lambda: self.answered.emit(self.number, UNKNOWN))
        for toggle in (self.skip, self.unknown):
            self.head.addWidget(toggle); toggle.hide(); self.lit_only.append(toggle)
        self.choice = None
        self.refresh()

    def set_answer(self, choice, name=None):
        self.choice = choice
        if name is not None and name != self.name.text():
            self.name.setText(name)
        self.refresh()

    def ask_side(self, sides=None, chosen=''):
        """Show "which side is ours?" (*sides*: the picture's words for left / right), or hide it (None)."""
        if sides is None:
            self.side.hide()
            return
        name = self.name.text().strip() or st('boundary_name', self.lang, number=self.number)
        self.side.ask(name, sides, chosen); self.side.show()

    def refresh(self):
        self.choices.show_choice(self.choice)
        self.boundary.setChecked(self.choice == BOUNDARY)
        self.skip.setChecked(self.choice == SKIP); self.unknown.setChecked(self.choice == UNKNOWN)
        self.badge.choice = self.choice; self.badge.update()
        name = self.name.text().strip()
        self.title.setText(name or st('region_title', self.lang, number=self.number))
        self.tag.setText(st('choice_' + self.choice, self.lang) if self.choice in CHOICES else
                         st('tag_' + (self.choice or 'none'), self.lang))
        colour = COLOURS.get('mine' if self.choice == BOUNDARY else self.choice)
        self.tag.setStyleSheet(f'color: {colour if colour and self.choice != "hide" else colors(self)["muted"]}; '
                               f'font-size: 10pt; font-weight: {600 if self.choice in CHOICES else 400};')


class HandRow(ItemRow):
    answered = Signal(int, str)
    renamed = Signal(int, str)
    deleted = Signal(int)

    def set_compact(self, compact):
        self.choices.set_compact(compact)

    def __init__(self, index, hand, lang, parent=None):
        self.index, self.lang, self.hand = index, lang, hand
        super().__init__(Badge(), '', parent)
        self.name = QLineEdit(); self.name.setPlaceholderText(st('name_placeholder', lang)); self.name.setMaxLength(40)
        self.name.setAccessibleName(st('name_placeholder', lang))
        from ..scene_map import WATCHED_NAME
        self.name.setText('' if hand.name == WATCHED_NAME else hand.name)
        self.watched = hand.name == WATCHED_NAME
        self.name.textEdited.connect(self.edited)
        self.body_layout.addWidget(self.name)
        self.choices = ChoiceGrid(lang); self.body_layout.addWidget(self.choices)
        self.choices.picked.connect(lambda c: self.answered.emit(self.index, c))
        actions = QHBoxLayout(); actions.addStretch(1)
        self.delete = TextAction(st('delete', lang), 'error'); actions.addWidget(self.delete)
        self.delete.clicked.connect(lambda: self.deleted.emit(self.index))
        self.body_layout.addLayout(actions)
        self.refresh()

    def edited(self, text):
        self.watched = False
        self.renamed.emit(self.index, text)
        self.refresh()

    def refresh(self):
        self.choices.show_choice(self.hand.choice)
        self.badge.choice = self.hand.choice; self.badge.update()
        name = self.name.text().strip()
        self.title.setText(name or (st('watched_name', self.lang) if self.watched else
                                    st('hand_title', self.lang, number=self.index + 1)))
        self.tag.setText(st('choice_' + self.hand.choice, self.lang) if self.hand.choice in CHOICES else st('choose_kind', self.lang))
        colour = COLOURS.get(self.hand.choice)
        self.tag.setStyleSheet(f'color: {colour if colour and self.hand.choice != "hide" else colors(self)["warning" if not self.hand.choice else "muted"]}; '
                               f'font-size: 10pt; font-weight: 600;')


class LineRow(ItemRow):
    renamed = Signal(int, str)
    flipped = Signal(int)
    deleted = Signal(int)

    def __init__(self, index, line, lang, parent=None):
        self.index, self.lang, self.line = index, lang, line
        super().__init__(Badge(line=True), line.name or st('line_title', lang, number=index + 1), parent)
        self.name = QLineEdit(line.name); self.name.setPlaceholderText(st('line_name_placeholder', lang))
        self.name.setMaxLength(40); self.name.setAccessibleName(st('line_name_placeholder', lang))
        self.name.textEdited.connect(self.edited)
        self.body_layout.addWidget(self.name)
        actions = QHBoxLayout(); actions.setSpacing(4)
        self.flip = TextAction(st('line_flip', lang)); actions.addWidget(self.flip)
        actions.addStretch(1)
        self.delete = TextAction(st('delete', lang), 'error'); actions.addWidget(self.delete)
        self.flip.clicked.connect(lambda: self.flipped.emit(self.index))
        self.delete.clicked.connect(lambda: self.deleted.emit(self.index))
        self.body_layout.addLayout(actions)

    def edited(self, text):
        self.renamed.emit(self.index, text)
        self.title.setText(text.strip() or st('line_title', self.lang, number=self.index + 1))


class Tabs(QWidget):
    """A segmented control: numbered places | draw by hand | boundaries."""
    changed = Signal(int)

    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.setObjectName('sceneTabs')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(self); row.setContentsMargins(4, 4, 4, 4); row.setSpacing(4)
        self.buttons = []
        t = colors(self)
        for i, name in enumerate(names):
            button = QPushButton(name); button.setCheckable(True); button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, i=i: self.select(i, True))
            row.addWidget(button, 1); self.buttons.append(button)
        self.setStyleSheet(
            f'QWidget#sceneTabs {{ background: {t["surface"]}; border: 1px solid {t["border"]}; border-radius: 14px; }}'
            f'QWidget#sceneTabs QPushButton {{ background: transparent; color: {t["muted"]}; border: none; border-radius: 10px; '
            f'padding: 8px 6px; min-height: 20px; font-size: 10.5pt; font-weight: 600; }}'
            f'QWidget#sceneTabs QPushButton:hover {{ color: {t["text"]}; }}'
            f'QWidget#sceneTabs QPushButton:checked {{ background: {t["raised"]}; color: {t["text"]}; }}')
        self.index = 0
        self.buttons[0].setChecked(True)

    def select(self, index, emit=False):
        self.index = index
        for i, button in enumerate(self.buttons):
            button.setChecked(i == index)
        if emit:
            self.changed.emit(index)


def clear(widget):
    """A container that shows what is behind it (the setup card, the dialog)."""
    widget.setObjectName('sceneClear')
    return widget


def scrolling(page):
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(clear(page))
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.viewport().setObjectName('sceneClear')
    return scroll


def show_row(scroll, row):
    """Scroll *row* into view, its top first when it is taller than the view."""
    bar, height = scroll.verticalScrollBar(), scroll.viewport().height()
    top, bottom = row.y(), row.y() + row.height()
    if top < bar.value() or bottom - top > height:
        bar.setValue(top - 4)
    elif bottom > bar.value() + height:
        bar.setValue(bottom - height + 4)


def scroll_list():
    """A scrolling column for rows; returns (scroll area, column layout)."""
    holder = clear(QWidget())
    column = QVBoxLayout(holder); column.setContentsMargins(0, 0, 6, 0); column.setSpacing(8)
    column.addStretch(1)
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(holder)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.viewport().setObjectName('sceneClear')
    return scroll, column


def align_labels(root):
    """Every label under *root* starts where the layout's reading starts, whatever script its text is in (a
    camera named in English still sits on the right of a Hebrew screen)."""
    rtl = root.layoutDirection() == Qt.LayoutDirection.RightToLeft
    side = Qt.AlignmentFlag.AlignRight if rtl else Qt.AlignmentFlag.AlignLeft
    for label in root.findChildren(QLabel):
        if label.property('keepAlignment'):
            continue
        vertical = label.alignment() & Qt.AlignmentFlag.AlignVertical_Mask
        label.setAlignment(side | Qt.AlignmentFlag.AlignAbsolute | (vertical or Qt.AlignmentFlag.AlignVCenter))


def words(text, role='body', wrap=True):
    label = QLabel(text); label.setTextFormat(Qt.TextFormat.PlainText); label.setWordWrap(wrap)
    label.setObjectName('scene_' + role)
    return label


class CountTile(QFrame):
    def __init__(self, colour, caption, parent=None):
        super().__init__(parent)
        self.setObjectName('sceneTile'); self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        t = colors(self)
        self.setStyleSheet(f'QFrame#sceneTile {{ background: {t["surface"]}; border: 1px solid {t["border"]}; border-radius: 14px; }}')
        lay = QHBoxLayout(self); lay.setContentsMargins(12, 8, 14, 8); lay.setSpacing(10)
        self.badge = Badge(line=colour is None); self.badge.choice = colour; lay.addWidget(self.badge)
        caption_label = QLabel(caption); caption_label.setStyleSheet(f'color: {t["secondary"]}; font-size: 10.5pt;')
        lay.addWidget(caption_label, 1)
        self.number = QLabel('0'); self.number.setStyleSheet(f'color: {t["text"]}; font-size: 15pt; font-weight: 600;')
        self.number.setProperty('keepAlignment', True)
        lay.addWidget(self.number)

    def set_count(self, count):
        self.number.setText(str(count))
        self.setStyleSheet(self.styleSheet())


# ----------------------------------------------------------------------------
# The editor
# ----------------------------------------------------------------------------
class SceneMapEditor(QWidget):
    """One camera's map. ``finished`` says how it ended: ("saved", Saved) | ("skip", None) | ("back", None) |
    ("cancel", None). In setup (*setup* = (number, total)) the footer offers Back / Skip for now / Save & next.

    *camera* is the box's id, for commands only. What the owner reads is *name* (the box's name for it), then the
    name in the box's own answers; without one "מצלמה 2 מתוך 5" by *position* (number, total). ``name_changed``
    says when the box named it."""
    finished = Signal(str, object)
    name_changed = Signal(str)

    def __init__(self, backend, camera, lang=None, setup=None, placeholder=None, parent=None, name="",
                 position=None):
        super().__init__(parent)
        self.backend, self.camera = backend, camera
        self.lang = lang or language()
        self.setup = setup
        self.position = position or setup
        self.name = name or fallback_name(self.position, self.lang)
        self.named_by_box = bool(name)
        self.proposal = None
        self.current = {}
        self.regions = ()
        self.answers, self.names = {}, {}
        self.sides = {}                   # region -> left | right, the owner's answer to "which side is ours?"
        self.hands, self.lines = [], []
        self.previous = None              # scene_backend.Previous: the map the last save replaced
        self.saved = None
        self.pending_map = None
        self.retry = None
        self.page = LOADING
        self.jobs = Jobs(self)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.close_jobs)        # never keep the app waiting on a box command
        self.setObjectName('sceneEditor')
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(self.lang) else Qt.LayoutDirection.LeftToRight)
        t = colors(self)
        self.setStyleSheet(
            'QWidget#sceneEditor, QWidget#sceneClear { background: transparent; }'
            f'QLabel#scene_eyebrow {{ color: {t["muted"]}; font-size: 9pt; letter-spacing: 1.2px; font-weight: 600; }}'
            f'QLabel#scene_title {{ color: {t["text"]}; font-size: 17pt; font-weight: 600; }}'
            f'QLabel#scene_heading {{ color: {t["text"]}; font-size: 14pt; font-weight: 600; }}'
            f'QLabel#scene_body {{ color: {t["secondary"]}; font-size: 10.5pt; }}'
            f'QLabel#scene_muted {{ color: {t["muted"]}; font-size: 10pt; }}'
            f'QLabel#scene_note {{ color: {t["warning"]}; background: {rgba(t["warning"], .1)}; border-radius: 12px; '
            f'padding: 10px 12px; font-size: 10pt; }}'
            f'QLabel#scene_detail {{ color: {t["secondary"]}; background: {t["surface"]}; border: 1px solid {t["border"]}; '
            f'border-radius: 10px; padding: 10px 12px; font-family: Consolas; font-size: 9.5pt; }}'
            f'QLabel#scene_error {{ color: {t["error"]}; font-size: 11pt; }}'
            f'QLabel#scene_count {{ color: {t["secondary"]}; font-size: 10.5pt; }}')
        root = QHBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(28)
        self.stage = MapStage(self)
        self.stage.setAccessibleName(self.name)
        self.stage.setLayoutDirection(Qt.LayoutDirection.LeftToRight)      # the picture never mirrors
        self.stage.text_direction = self.layoutDirection()
        if placeholder is not None and not placeholder.isNull():
            self.stage.set_picture(placeholder)          # the card's photo, until the box's own picture comes
        root.addWidget(self.stage, 1)
        panel = clear(QWidget()); panel.setFixedWidth(PANEL_WIDTH); self.panel = panel
        root.addWidget(panel)
        side = QVBoxLayout(panel); side.setContentsMargins(0, 0, 0, 0); side.setSpacing(10 if setup else 14)
        eyebrow = st('setup_eyebrow', self.lang, number=setup[0], total=setup[1]) if setup else st('eyebrow', self.lang)
        self.eyebrow = words(eyebrow.upper() if self.lang == 'en' else eyebrow, 'eyebrow'); side.addWidget(self.eyebrow)
        self.eyebrow.setVisible(not setup)              # setup says "camera 2 of 5" above, for the whole step
        # The camera's name, and at the end of its line "restore the previous map" with when it was replaced.
        self.heading = clear(QWidget()); heading = QHBoxLayout(self.heading)
        heading.setContentsMargins(0, 0, 0, 0); heading.setSpacing(12)
        self.title = words(self.name, 'title'); heading.addWidget(self.title, 1)
        restore = QVBoxLayout(); restore.setSpacing(0); restore.setContentsMargins(0, 0, 0, 0)
        self.restore_button = TextAction(st('restore_button', self.lang))
        self.restore_button.clicked.connect(self.restore_previous)
        self.restore_when = words('', 'muted', wrap=False)
        restore.addWidget(self.restore_button); restore.addWidget(self.restore_when)
        self.restore_row = clear(QWidget()); self.restore_row.setLayout(restore)
        self.restore_row.setStyleSheet(f'QWidget#sceneClear {{ background: transparent; }}'
                                       f'QLabel {{ color: {t["muted"]}; font-size: 10pt; background: transparent; }}')
        heading.addWidget(self.restore_row, 0, Qt.AlignmentFlag.AlignVCenter)
        side.addWidget(self.heading)
        # In setup the step's own header says which camera this is (and holds the restore action): more room
        # for the open answer card.
        self.heading.setVisible(not setup)
        self.compact = False
        self.pages = QStackedWidget(); self.pages.setObjectName('sceneClear'); side.addWidget(self.pages, 1)
        # The pages take the room there is (setup is short at 1366x768); the quiet ones scroll when it is not enough.
        self.pages.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        self.pages.addWidget(scrolling(self.build_loading(t)))
        self.pages.addWidget(self.build_edit(t))
        self.pages.addWidget(scrolling(self.build_summary(t)))
        self.pages.addWidget(scrolling(self.build_saved(t)))
        self.pages.addWidget(scrolling(self.build_error(t)))
        self.count_line = words('', 'count'); side.addWidget(self.count_line)
        footer = QHBoxLayout(); footer.setSpacing(10)
        self.back_button = ZonePill(st('setup_back' if setup else 'cancel', self.lang))
        self.skip_button = ZonePill(st('setup_skip', self.lang))
        self.primary = ZonePill(st('save_next' if setup else 'save', self.lang), primary=True)
        footer.addWidget(self.back_button, 2)
        if setup:
            footer.addWidget(self.skip_button, 2)
        else:
            self.skip_button.hide()
        footer.addWidget(self.primary, 3)
        side.addLayout(footer)
        self.back_button.clicked.connect(self.back_clicked)
        self.skip_button.clicked.connect(lambda: self.leave("skip"))
        self.primary.clicked.connect(self.primary_clicked)
        self.stage.region_clicked.connect(self.select_region)
        self.stage.hand_clicked.connect(self.select_hand)
        self.stage.line_clicked.connect(self.select_line)
        self.stage.draft_changed.connect(self.draft_changed)
        self.stage.draft_closed.connect(self.draft_closed)
        self.stage.hand_changed.connect(lambda i: self.refresh_counts())
        self.stage.line_changed.connect(self.line_draft_changed)
        self.stage.side_picked.connect(self.side_picked)
        align_labels(self)
        self.show_page(LOADING)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.panel.setFixedWidth(PANEL_WIDTH + (60 if self.width() >= 1500 else 0))
        self.set_compact(self.height() < COMPACT_BELOW)

    def set_compact(self, compact):
        """A short screen (1366x768): answers on one line each, hints as tooltips, so the open card fits."""
        if compact == self.compact:
            return
        self.compact = compact
        self.regions_hint.setVisible(not compact)
        for row in list(getattr(self, 'region_rows', {}).values()) + list(getattr(self, 'hand_rows', [])):
            row.set_compact(compact)
        lit = next((row for row in getattr(self, 'region_rows', {}).values() if row.lit), None)
        if lit is not None:                     # the open card changed height: bring it back into view
            self.reveal(lit)

    # --- building the panel ------------------------------------------------
    def build_loading(self, t):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(12)
        card = QFrame(); card.setObjectName('sceneLegend'); card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(f'QFrame#sceneLegend {{ background: {t["surface"]}; border: 1px solid {t["border"]}; border-radius: 16px; }}')
        legend = QVBoxLayout(card); legend.setContentsMargins(16, 14, 16, 16); legend.setSpacing(12)
        legend.addWidget(words(st('legend_title', self.lang), 'heading'))
        for choice in CHOICES:
            row = QHBoxLayout(); row.setSpacing(12)
            badge = Badge(); badge.choice = choice; row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(words(st('legend_' + choice, self.lang), 'body'), 1)
            legend.addLayout(row)
        lay.addWidget(card); lay.addStretch(1)
        return page

    def build_edit(self, t):
        page = clear(QWidget()); lay = QVBoxLayout(page); lay.setContentsMargins(0, 4, 0, 0); lay.setSpacing(12)
        self.tabs = Tabs([st('tab_regions', self.lang), st('tab_draw', self.lang), st('tab_lines', self.lang)])
        self.tabs.changed.connect(self.tab_changed)
        lay.addWidget(self.tabs)
        self.tab_pages = clear(QStackedWidget()); lay.addWidget(self.tab_pages, 1)
        # Numbered places
        regions = clear(QWidget()); col = QVBoxLayout(regions); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(10)
        self.grid_note = words(st('grid_note', self.lang), 'note'); self.grid_note.hide(); col.addWidget(self.grid_note)
        self.regions_hint = words(st('regions_hint', self.lang), 'body'); col.addWidget(self.regions_hint)
        self.region_scroll, self.region_column = scroll_list(); col.addWidget(self.region_scroll, 1)
        self.tab_pages.addWidget(regions)
        # Draw by hand
        draw = clear(QWidget()); col = QVBoxLayout(draw); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(10)
        col.addWidget(words(st('draw_hint', self.lang), 'body'))
        tools = QHBoxLayout(); tools.setSpacing(8)
        self.new_area = ZonePill('+  ' + st('draw_new', self.lang), compact=True); tools.addWidget(self.new_area)
        self.undo_corner = TextAction(st('draw_undo', self.lang)); tools.addWidget(self.undo_corner)
        self.stop_drawing = TextAction(st('draw_stop', self.lang)); tools.addWidget(self.stop_drawing)
        tools.addStretch(1)
        col.addLayout(tools)
        self.draw_status = words('', 'muted'); col.addWidget(self.draw_status)
        self.hand_scroll, self.hand_column = scroll_list(); col.addWidget(self.hand_scroll, 1)
        self.new_area.clicked.connect(self.start_area)
        self.undo_corner.clicked.connect(self.stage.undo_corner)
        self.stop_drawing.clicked.connect(self.stage.stop_draft)
        self.tab_pages.addWidget(draw)
        # Boundaries
        lines = clear(QWidget()); col = QVBoxLayout(lines); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(10)
        col.addWidget(words(st('line_hint', self.lang), 'body'))
        tools = QHBoxLayout(); tools.setSpacing(8)
        self.new_line = ZonePill('+  ' + st('line_new', self.lang), compact=True); tools.addWidget(self.new_line)
        self.from_areas = ZonePill(st('line_from_areas', self.lang), compact=True)
        self.from_areas.setToolTip(st('line_from_areas_tip', self.lang)); tools.addWidget(self.from_areas)
        self.cancel_line = TextAction(st('line_cancel', self.lang)); tools.addWidget(self.cancel_line)
        tools.addStretch(1)
        col.addLayout(tools)
        self.line_status = words('', 'muted'); col.addWidget(self.line_status)
        self.line_scroll, self.line_column = scroll_list(); col.addWidget(self.line_scroll, 1)
        self.new_line.clicked.connect(self.stage.start_line)
        self.from_areas.clicked.connect(self.side_from_areas)
        self.cancel_line.clicked.connect(self.stage.stop_line)
        self.tab_pages.addWidget(lines)
        return page

    def build_summary(self, t):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(12)
        lay.addWidget(words(st('summary_title', self.lang), 'heading'))
        lay.addWidget(words(st('summary_hint', self.lang), 'body'))
        grid = QGridLayout(); grid.setSpacing(8)
        self.tiles = {}
        for i, key in enumerate((*CHOICES, 'lines')):
            tile = CountTile(key if key != 'lines' else None, st('tile_' + key, self.lang))
            grid.addWidget(tile, i // 2, i % 2); self.tiles[key] = tile
        lay.addLayout(grid)
        self.rest_label = words('', 'body'); lay.addWidget(self.rest_label)
        self.left_out_label = words('', 'note'); lay.addWidget(self.left_out_label)
        self.walls_label = words('', 'note'); lay.addWidget(self.walls_label)
        self.restart_label = words(st('restart_note', self.lang), 'note'); lay.addWidget(self.restart_label)
        lay.addStretch(1)
        return page

    def build_saved(self, t):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(12)
        head = QHBoxLayout(); head.setSpacing(10)
        self.saved_title = words(st('saved_title', self.lang), 'heading')
        head.addWidget(Tick()); head.addWidget(self.saved_title, 1)
        lay.addLayout(head)
        self.saved_hint = words(st('saved_hint', self.lang), 'body'); lay.addWidget(self.saved_hint)
        self.saved_counts = words('', 'body'); lay.addWidget(self.saved_counts)
        self.saved_rest = words('', 'body'); lay.addWidget(self.saved_rest)
        self.saved_restart = words(st('saved_restart', self.lang), 'note'); lay.addWidget(self.saved_restart)
        lay.addStretch(1)
        return page

    def build_error(self, t):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(12)
        lay.addWidget(words(st('error_title', self.lang), 'heading'))
        self.error_label = words('', 'error'); lay.addWidget(self.error_label)
        self.error_said = words('', 'muted'); lay.addWidget(self.error_said)
        self.error_detail = words('', 'detail'); self.error_detail.setProperty('keepAlignment', True)
        self.error_detail.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.error_detail.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignAbsolute)
        self.error_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.error_detail)
        lay.addStretch(1)
        return page

    # --- flow --------------------------------------------------------------
    def start(self):
        """Ask the box for the numbered picture (FastSAM can take seconds)."""
        self.show_page(LOADING)
        self.stage.set_loading(st('loading_title', self.lang), st('loading_hint', self.lang))
        self.retry = self.start

        def task():
            try:
                previous = self.backend.previous(self.camera)
            except Exception:  # noqa: BLE001 - an older box, or no answer: simply no restore offered
                previous = None
            proposal = self.backend.propose(self.camera)
            aspect = 16 / 9
            pix = picture_pixmap(proposal.picture)
            if not pix.isNull():
                aspect = pix.width() / max(1, pix.height())
            labels = {}
            regions = proposal.regions
            for i, region in enumerate(regions):
                covered = [r.points for r in regions[i + 1:]]
                labels[region.number] = label_point(region.points, covered, aspect, steps=18)
            return proposal, labels, previous
        self.jobs.submit(task, lambda result: self.loaded(*result), self.failed)

    def loaded(self, proposal, labels=None, previous=None):
        if labels is None:
            labels = {r.number: label_point(r.points, [o.points for o in proposal.regions[i + 1:]])
                      for i, r in enumerate(proposal.regions)}
        self.proposal = proposal
        self.set_name((proposal.names or {}).get(self.lang, ''))
        self.current = dict(proposal.current or {})       # the whole truth: after a save, today's zone is gone
        self.regions = tuple(proposal.regions)
        self.answers, self.names, self.sides = {}, {}, {}
        self.previous = previous
        self.hands, self.lines = from_current(self.current)
        self.stage.set_picture(picture_pixmap(proposal.picture)); self.stage.muted = False
        self.stage.set_regions(self.regions, labels)
        self.stage.answers = self.answers; self.stage.hands = self.hands; self.stage.lines = self.lines
        self.stage.set_loading('')
        self.grid_note.setVisible(proposal.grid)
        self.build_region_rows()
        self.build_hand_rows()
        self.build_line_rows()
        self.refresh_walls()
        self.show_page(EDIT)
        self.tabs.select(REGIONS if self.regions else DRAW)
        self.tab_changed(REGIONS if self.regions else DRAW)
        if self.regions:
            self.select_region(self.regions[0].number)

    def failed(self, exc):
        self.stage.set_loading('')
        if not isinstance(exc, SceneError):
            exc = SceneError('invalid' if isinstance(exc, ValueError) else 'box_refused', str(exc))
        kind = exc.kind if f'error_{exc.kind}' in TEXT_KEYS else 'box_refused'
        detail = plain_detail(exc.detail, self.camera, self.name)
        self.error_label.setText(st('error_' + kind, self.lang, detail=detail) if kind == 'invalid' else st('error_' + kind, self.lang))
        said = bool(detail) and kind != 'invalid'
        self.error_said.setText(st('error_detail', self.lang, detail='').strip()); self.error_said.setVisible(said)
        self.error_detail.setText(detail); self.error_detail.setVisible(said)
        self.stage.muted = self.proposal is None; self.stage.update()
        self.show_page(ERROR)

    def show_page(self, page):
        self.page = page
        self.pages.setCurrentIndex(page)
        setup = bool(self.setup)
        busy = self.jobs.busy()
        if page == LOADING:
            self.primary.setText(st('save_next' if setup else 'save', self.lang)); self.primary.setEnabled(False)
        elif page == EDIT:
            self.primary.setText(st('save_next' if setup else 'save', self.lang)); self.primary.setEnabled(True)
            self.back_button.setText(st('setup_back' if setup else 'cancel', self.lang))
        elif page == SUMMARY:
            self.primary.setText(st('summary_save', self.lang)); self.primary.setEnabled(not busy)
            self.back_button.setText(st('summary_back', self.lang))
        elif page == SAVED:
            last = setup and self.setup[0] >= self.setup[1]
            self.primary.setText(st('finish' if last else 'next_camera', self.lang) if setup else st('done', self.lang))
            self.primary.setEnabled(True)
        elif page == ERROR:
            self.primary.setText(st('try_again', self.lang)); self.primary.setEnabled(True)
            self.back_button.setText(st('setup_back' if setup else 'cancel', self.lang))
        self.back_button.setVisible(page != SAVED and (setup and self.setup[0] > 1 or not setup or page == SUMMARY))
        self.skip_button.setVisible(setup and page in (LOADING, EDIT, ERROR))
        self.show_previous()
        self.stage.set_mode({EDIT: ('regions', 'draw', 'line')[self.tabs.index]}.get(page, 'view'))
        self.refresh_counts()

    def primary_clicked(self):
        if self.page == EDIT:
            self.open_summary()
        elif self.page == SUMMARY:
            self.save()
        elif self.page == SAVED:
            self.leave("saved", self.saved)
        elif self.page == ERROR and self.retry:
            self.retry()

    def back_clicked(self):
        if self.jobs.busy() and self.page == SUMMARY:
            return
        if self.page == SUMMARY:
            self.stage.rest = None
            self.show_page(EDIT)
            return
        self.leave("back" if self.setup else "cancel")

    def leave(self, how, value=None):
        if self.jobs.busy() and how != "saved":
            if self.page == SUMMARY:
                return                         # the box is saving: let it finish
            self.backend.cancel()
        self.finished.emit(how, value)

    def busy(self):
        return self.jobs.busy() and self.page == SUMMARY

    def close_jobs(self):
        self.jobs.close()
        try:
            self.backend.cancel()
        except Exception:  # noqa: BLE001
            pass

    def set_name(self, name):
        """The box's name for the camera (its own answers carry it); an empty one keeps what is shown."""
        name = str(name or '').strip()
        if not name or name == self.name:
            return
        self.name, self.named_by_box = name, True
        self.title.setText(name); self.stage.setAccessibleName(name)
        align_labels(self)
        self.name_changed.emit(name)

    # --- the previous map --------------------------------------------------
    def show_previous(self):
        """"Restore the previous map", while editing; greyed out until a save has replaced a map."""
        has = bool(self.previous and self.previous.exists)
        self.restore_row.setVisible(self.page == EDIT)
        self.restore_button.setEnabled(has and not self.jobs.busy())
        self.restore_button.setToolTip('' if has else st('restore_none', self.lang))
        when = ''
        if has and self.previous.saved_at:
            from datetime import datetime
            when = st('restore_from', self.lang, when=datetime.fromtimestamp(self.previous.saved_at).strftime('%d.%m %H:%M'))
        self.restore_when.setText(when)

    def restore_previous(self):
        if self.jobs.busy() or not (self.previous and self.previous.exists):
            return
        self.restore_button.setText(st('restoring', self.lang)); self.restore_button.setEnabled(False)
        self.primary.setEnabled(False)
        self.retry = self.restore_previous

        def done(saved):
            self.restore_button.setText(st('restore_button', self.lang))
            self.show_saved(saved)

        def failed(exc):
            self.restore_button.setText(st('restore_button', self.lang))
            self.failed(exc)
        self.jobs.submit(lambda: self.backend.restore(self.camera), done, failed)

    # --- the map -----------------------------------------------------------
    def walls(self):
        """``[(region number, Boundary, sure, side words)]`` for the places answered "it is the boundary": the line
        along each, its inward side from the owner's answer or from the areas around it (the engine's own
        ``boundary_line``); not sure when nothing says which side is ours."""
        numbers = [r for r in self.regions if self.answers.get(r.number) == BOUNDARY]
        if not numbers:
            return []
        from .. import scene_interview as si
        from .. import scene_map as sm
        scene = build_map(self.camera, self.regions, self.answers, self.names, self.hands, ())
        areas = [sm.Area(a['name'], a['kind'], a['zone'], a['points'], owner=a.get('owner', '')) for a in scene['areas']]
        others = [r.points for r in numbers]
        out = []
        for region in numbers:
            name = self.names.get(region.number, '').strip() or st('boundary_name', self.lang, number=region.number)
            try:
                line, sure = si.boundary_line(name, si.Region(region.number, region.points, region.area), areas,
                                              self.sides.get(region.number, ''), others)
            except ValueError:
                continue
            words_for = si.side_words(line.a, line.b, self.lang)
            out.append((region.number, Boundary(line.a, line.b, line.inward, name), sure, words_for))
        return out

    def build_map(self):
        rest = self.current.get('rest') or ''
        walls = [line for _n, line, sure, _w in self.walls() if sure]
        return build_map(self.camera, self.regions, self.answers, self.names, self.hands, self.lines + walls,
                         rest=rest, rest_owner=self.current.get('rest_owner') or '')

    def refresh_walls(self):
        """The boundary places' lines on the picture, and "which side is ours?" in each row that needs it."""
        walls = self.walls()
        self.stage.walls = [(line, sure) for _n, line, sure, _w in walls]
        asking = {n: w for n, _line, sure, w in walls if not sure or n in self.sides}
        for number, row in getattr(self, 'region_rows', {}).items():
            row.ask_side(asking.get(number), self.sides.get(number, ''))
        self.stage.update()
        return walls

    def count_text(self, scene):
        c = counts(scene)
        parts = [st('count_' + key, self.lang, count=c[key]) for key in CHOICES if c[key]]
        if c['lines']:
            parts.append(st('count_line_one', self.lang) if c['lines'] == 1 else st('count_lines', self.lang, count=c['lines']))
        return '  ·  '.join(parts) if parts else st('nothing_marked', self.lang)

    def refresh_counts(self):
        if self.page == EDIT:
            self.count_line.setText(self.count_text(self.build_map()))
            self.count_line.show()
        else:
            self.count_line.hide()
        if self.regions:
            answered = sum(1 for r in self.regions if self.answers.get(r.number))
            self.tabs.buttons[REGIONS].setText(st('tab_regions', self.lang) + f'  ·  {answered}/{len(self.regions)}')
            self.tabs.buttons[REGIONS].setAccessibleDescription(st('answered', self.lang, count=answered, total=len(self.regions)))

    def open_summary(self):
        self.stage.stop_draft() if self.stage.draft is not None else None
        scene = self.build_map()
        c = counts(scene)
        for key, tile in self.tiles.items():
            tile.set_count(c[key])
        rest = rest_after(self.current, scene)
        self.rest_label.setText(st('rest_' + rest, self.lang))
        self.stage.rest = rest if rest in COLOURS else None
        left_out = sum(1 for h in self.hands if h.closed and h.choice not in CHOICES)
        self.left_out_label.setText(st('left_out', self.lang, count=left_out)); self.left_out_label.setVisible(bool(left_out))
        self.restart_label.setVisible(restart_expected(self.current, scene))
        unclear = [line.name for _n, line, sure, _w in self.walls() if not sure]
        self.walls_label.setText(st('walls_unclear', self.lang, names=', '.join(unclear)))
        self.walls_label.setVisible(bool(unclear))
        self.pending_map = scene
        self.show_page(SUMMARY)

    def save(self):
        if self.jobs.busy() or self.pending_map is None:
            return
        scene = self.pending_map
        self.primary.setText(st('saving', self.lang)); self.primary.setEnabled(False)
        motion.busy(self.primary, True)
        self.back_button.setEnabled(False)
        self.retry = self.save_again

        def done(saved):
            motion.busy(self.primary, False); self.back_button.setEnabled(True)
            self.show_saved(saved)

        def failed(exc):
            motion.busy(self.primary, False); self.back_button.setEnabled(True)
            self.failed(exc)
        self.jobs.submit(lambda: self.backend.confirm(self.camera, scene), done, failed)

    def save_again(self):
        self.show_page(SUMMARY)
        self.save()

    def show_saved(self, saved):
        self.saved = saved
        self.set_name((getattr(saved, 'names', None) or {}).get(self.lang, ''))
        self.current = dict(saved.map)
        restored = getattr(saved, 'restored', False)
        self.saved_title.setText(st('restored_title' if restored else 'saved_title', self.lang))
        self.saved_hint.setText(st('restored_receipt', self.lang, camera=self.name)
                                if restored else st('saved_hint', self.lang))
        self.stage.walls = []
        hands, lines = from_current(saved.map)     # a restored drawn zone shows as ours, its outside hidden
        self.stage.regions = (); self.stage.labels = {}
        self.stage.hands = hands; self.stage.lines = lines
        from .scene_backend import Previous
        import time
        self.previous = Previous(True, time.time())           # the box keeps what this replaced: it can come back
        rest = 'hide' if saved.map.get('watched') else rest_after({}, saved.map)
        self.stage.rest = rest if rest in COLOURS else None
        self.saved_counts.setText(self.count_text(saved.map))
        self.saved_rest.setText(st('rest_' + rest, self.lang))
        self.saved_restart.setVisible(saved.restart_needed)
        self.show_page(SAVED)

    # --- tabs and selection ------------------------------------------------
    def tab_changed(self, index):
        self.tab_pages.setCurrentIndex(index)
        self.stage.set_mode(('regions', 'draw', 'line')[index])
        self.draft_changed(); self.line_draft_changed()

    def clear_column(self, column):
        while column.count() > 1:
            item = column.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def build_region_rows(self):
        self.clear_column(self.region_column)
        self.region_rows = {}
        for region in self.regions:
            row = RegionRow(region.number, self.lang)
            row.set_compact(self.compact)
            row.selected.connect(lambda n=region.number: self.select_region(n))
            row.answered.connect(self.answer_region)
            row.renamed.connect(self.rename_region)
            row.side_picked.connect(self.pick_side)
            self.region_column.insertWidget(self.region_column.count() - 1, row)
            self.region_rows[region.number] = row
        align_labels(self.region_scroll.widget())
        self.regions_hint.setText(st('regions_hint' if self.regions else 'regions_empty', self.lang))
        self.regions_hint.setVisible(not self.compact or not self.regions)
        self.refresh_counts()

    def select_region(self, number):
        if self.page != EDIT:
            return
        if self.tabs.index != REGIONS:
            self.tabs.select(REGIONS); self.tab_changed(REGIONS)
        self.stage.selected_region = number; self.stage.update()
        for n, row in self.region_rows.items():
            row.set_lit(n == number)
        row = self.region_rows.get(number)
        if row is not None:
            self.reveal(row)

    def reveal(self, row):
        """Show the open card once the layout has settled (now, and again after a resize or a new answer)."""
        for delay in (0, motion.PANE_MS):
            QTimer.singleShot(delay, row, lambda: self.show_region_row(row))     # dropped if the row is gone

    def show_region_row(self, row):
        """The open card in view (its top first); "which side is ours?" with its two answers too."""
        row.layout().activate(); self.region_scroll.widget().layout().activate()     # heights after any change
        show_row(self.region_scroll, row)
        if row.side.isVisible():
            self.region_scroll.ensureWidgetVisible(row.side, 0, 4)

    def answer_region(self, number, choice):
        if self.answers.get(number) == choice:
            self.answers.pop(number, None)      # a second tap on the same answer takes it back
        else:
            self.answers[number] = choice
        if self.answers.get(number) != BOUNDARY:
            self.sides.pop(number, None)
        self.region_rows[number].set_answer(self.answers.get(number))
        walls = self.refresh_walls(); self.refresh_counts()
        if any(n == number and not sure for n, _line, sure, _w in walls):
            row = self.region_rows[number]       # stay: the row asks which side is ours
            self.reveal(row)
            return
        if number in self.answers:
            following = [r.number for r in self.regions if r.number > number and not self.answers.get(r.number)]
            if following:
                QTimer.singleShot(motion.TOGGLE_MS, self, lambda n=following[0]: self.select_region(n))

    def rename_region(self, number, text):
        self.names[number] = text
        if self.answers.get(number) == BOUNDARY:
            self.refresh_walls()
        self.refresh_counts()

    def pick_side(self, number, side):
        self.sides[number] = side
        self.refresh_walls(); self.refresh_counts()

    def build_hand_rows(self):
        self.clear_column(self.hand_column)
        self.hand_rows = []
        for i, hand in enumerate(self.hands):
            row = HandRow(i, hand, self.lang)
            row.set_compact(self.compact)
            row.selected.connect(lambda i=i: self.select_hand(i))
            row.answered.connect(self.answer_hand)
            row.renamed.connect(self.rename_hand)
            row.deleted.connect(self.delete_hand)
            self.hand_column.insertWidget(self.hand_column.count() - 1, row)
            self.hand_rows.append(row)
        if not self.hands:
            empty = words(st('draw_empty', self.lang), 'muted')
            self.hand_column.insertWidget(0, empty)
        align_labels(self.hand_scroll.widget())
        self.stage.hands = self.hands
        self.refresh_counts()

    def select_hand(self, index):
        if self.tabs.index != DRAW:
            self.tabs.select(DRAW); self.tab_changed(DRAW)
        self.stage.selected_hand = index; self.stage.update()
        for i, row in enumerate(self.hand_rows):
            row.set_lit(i == index)
        if 0 <= index < len(self.hand_rows):
            row = self.hand_rows[index]
            QTimer.singleShot(0, row, lambda: show_row(self.hand_scroll, row))

    def answer_hand(self, index, choice):
        self.hands[index].choice = choice
        self.hand_rows[index].refresh(); self.stage.update(); self.refresh_counts()

    def rename_hand(self, index, text):
        self.hands[index].name = text
        self.refresh_counts()

    def delete_hand(self, index):
        del self.hands[index]
        self.stage.selected_hand = -1
        self.build_hand_rows(); self.stage.update()

    def start_area(self):
        if self.tabs.index != DRAW:
            self.tabs.select(DRAW); self.tab_changed(DRAW)
        for row in self.hand_rows:
            row.set_lit(False)
        self.stage.start_draft()
        self.stage.setFocus()

    def draft_changed(self):
        drawing = self.stage.draft is not None
        count = len(self.stage.draft or ())
        self.undo_corner.setVisible(drawing); self.undo_corner.setEnabled(count > 0)
        self.stop_drawing.setVisible(drawing)
        self.new_area.setEnabled(not drawing)
        self.draw_status.setText(st('draw_corners', self.lang, count=count) if count else st('draw_first', self.lang) if drawing else '')
        self.draw_status.setVisible(drawing)

    def draft_closed(self, points):
        self.hands.append(HandArea([list(p) for p in points]))
        self.build_hand_rows()
        self.draft_changed()
        self.select_hand(len(self.hands) - 1)

    def build_line_rows(self):
        self.clear_column(self.line_column)
        self.line_rows = []
        for i, line in enumerate(self.lines):
            row = LineRow(i, line, self.lang)
            row.selected.connect(lambda i=i: self.select_line(i))
            row.renamed.connect(self.rename_line)
            row.flipped.connect(self.flip_line)
            row.deleted.connect(self.delete_line)
            self.line_column.insertWidget(self.line_column.count() - 1, row)
            self.line_rows.append(row)
        if not self.lines:
            self.line_column.insertWidget(0, words(st('line_empty', self.lang), 'muted'))
        align_labels(self.line_scroll.widget())
        self.stage.lines = self.lines
        self.refresh_counts()

    def select_line(self, index):
        if self.tabs.index != LINES:
            self.tabs.select(LINES); self.tab_changed(LINES)
        self.stage.selected_line = index; self.stage.update()
        for i, row in enumerate(self.line_rows):
            row.set_lit(i == index)

    def rename_line(self, index, text):
        self.lines[index].name = text
        self.stage.update()

    def flip_line(self, index):
        self.lines[index] = self.lines[index].flipped()
        self.stage.update()

    def delete_line(self, index):
        del self.lines[index]
        self.stage.selected_line = -1
        self.build_line_rows(); self.stage.update()

    def line_draft_changed(self, message=''):
        placing = self.stage.line_draft is not None
        count = len(self.stage.line_draft or ())
        self.new_line.setEnabled(not placing)
        self.cancel_line.setVisible(placing)
        self.from_areas.setVisible(placing and count == 2)
        self.from_areas.setEnabled(bool(self.mine_polygons()))
        step = ('line_step_a', 'line_step_b', 'line_step_side')[min(count, 2)]
        self.line_status.setText(message or (st(step, self.lang) if placing else ''))
        self.line_status.setVisible(placing or bool(message))

    def mine_polygons(self):
        found = [r.points for r in self.regions if self.answers.get(r.number) == 'mine']
        return found + [h.points for h in self.hands if h.closed and h.choice == 'mine']

    def add_line(self, line):
        self.lines.append(line)
        self.stage.line_draft = None
        self.build_line_rows()
        self.line_draft_changed()
        self.select_line(len(self.lines) - 1)

    def side_picked(self, point):
        a, b = self.stage.line_draft
        try:
            self.add_line(boundary_toward(a, b, point))
        except ValueError:
            self.line_draft_changed(st('line_on_line', self.lang))

    def side_from_areas(self):
        if self.stage.line_draft is None or len(self.stage.line_draft) != 2:
            return
        a, b = self.stage.line_draft
        try:
            self.add_line(boundary_from_areas(a, b, self.mine_polygons()))
        except ValueError:
            self.line_draft_changed(st('line_no_areas', self.lang))

    def keyPressEvent(self, event):
        if self.page == EDIT and self.stage.draft is not None:
            if event.key() == Qt.Key.Key_Escape:
                self.stage.stop_draft(); event.accept(); return
            if event.key() == Qt.Key.Key_Backspace:
                self.stage.undo_corner(); event.accept(); return
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.stage.close_draft(); event.accept(); return
        if self.page == EDIT and self.stage.line_draft is not None and event.key() == Qt.Key.Key_Escape:
            self.stage.stop_line(); event.accept(); return
        super().keyPressEvent(event)


from .scene_strings import TEXT as _TEXT  # noqa: E402
TEXT_KEYS = set(_TEXT['en'])


# ----------------------------------------------------------------------------
# The box app's dialog, from the camera card
# ----------------------------------------------------------------------------
class SceneMapDialog(QDialog):
    """One camera's map, from its card. ``saved`` carries the Saved result."""
    saved = Signal(object)

    @staticmethod
    def opening_size(parent_size, screen_size):
        # Never larger than the screen, or than the app window when that is larger (offscreen screenshots).
        room = QSize(max(screen_size.width(), parent_size.width()), max(screen_size.height(), parent_size.height()))
        width = min(room.width(), max(1000, round(parent_size.width() * .95)))
        height = min(room.height(), max(620, round(parent_size.height() * .93)))
        return QSize(width, height)

    def __init__(self, backend, camera, lang=None, parent=None, start=True, placeholder=None, name="", position=None):
        super().__init__(parent)
        lang = lang or language()
        self.setModal(True)
        self.setObjectName('sceneDialog')
        t = colors(self)
        self.setStyleSheet(f'QDialog#sceneDialog {{ background: {t["bg"]}; }} QLabel {{ background: transparent; border: none; }}')
        screen = self.screen().availableGeometry().size()
        host = parent.window().size() if parent is not None else screen
        self.setMinimumSize(1000, 620)
        self.resize(self.opening_size(host, screen))
        layout = QVBoxLayout(self); layout.setContentsMargins(28, 26, 28, 26)
        self.editor = SceneMapEditor(backend, camera, lang, placeholder=placeholder, parent=self, name=name,
                                     position=position)
        layout.addWidget(self.editor)
        self.setWindowTitle(st('window_title', lang, camera=self.editor.name))
        self.editor.name_changed.connect(lambda n: self.setWindowTitle(st('window_title', lang, camera=n)))
        self.editor.finished.connect(self.finish)
        if start:
            self.editor.start()

    def finish(self, how, value):
        if how == "saved":
            self.saved.emit(value)
            if self.parentWidget() is not None:
                motion.toast(self.parentWidget().window(), st('toast_saved', self.editor.lang))
            self.accept()
        else:
            self.reject()

    def done(self, result):
        if self.editor.busy() and result == QDialog.DialogCode.Rejected:
            return                            # the box is saving the map: let it finish
        self.editor.close_jobs()
        super().done(result)


def map_button(lang=None):
    """The camera card's "Map" button (it took the watch-zone button's place)."""
    button = ZonePill(st('map_button', lang), compact=True)
    button.setToolTip(st('map_button_tip', lang))
    button.setAccessibleName(st('map_button', lang))
    return button


def open_map_dialog(page, name, demo_state=None):
    """Open the map editor for camera *name* from a camera page (camera_ui.CameraPage)."""
    from .scene_backend import scene_backend_for
    photo = page.zone_widgets.get(name, (None, None, None))[2]
    ids = [c.name for c in getattr(page.controls, 'records', [])]
    position = (ids.index(name) + 1, len(ids)) if name in ids else None
    # The card knows only ids; the box names the camera in its first answer (propose), "מצלמה 2 מתוך 5" till then.
    dialog = SceneMapDialog(scene_backend_for(page.controls), name, parent=page.widget, start=demo_state is None,
                            placeholder=getattr(photo, 'pix', None), position=position)
    page.zone_dialog = dialog            # the camera page pauses its polling while a dialog is open
    dialog.saved.connect(lambda saved: page.zone_saved(name, []))     # confirm removes today's zone: whole picture
    if demo_state:
        drive(dialog.editor, demo_state)

    def finished(result):
        if page.zone_dialog is dialog:
            page.zone_dialog = None
        dialog.deleteLater()
    dialog.finished.connect(finished)
    dialog.open()
    return dialog


# ----------------------------------------------------------------------------
# Demo states, for screenshots and tests (``--scene STATE``)
# ----------------------------------------------------------------------------
DEMO_STATES = ('loading', 'regions', 'grid', 'wall', 'drawing', 'line', 'summary', 'saved', 'restored', 'error')
DEMO_ANSWERS = {1: ('mine', {'he': 'הדשא', 'en': 'the lawn'}), 2: ('neighbour', {'he': 'הבית של השכן', 'en': 'the house across'}),
                3: ('mine', {'he': 'השביל', 'en': 'the driveway'}), 4: ('mine', {'he': '', 'en': ''}),
                5: ('hide', {'he': '', 'en': ''})}


def point_at(stage, fraction):
    """The demo's pointer at *fraction* of the picture, once the stage has its final size."""
    def place():
        stage.pointer = stage.to_stage(fraction); stage.update()
    place()
    for delay in (100, 300, 600):
        QTimer.singleShot(delay, stage, place)


def drive(editor, state):
    """Put a demo editor straight into *state*, synchronously (the demo backend needs no box)."""
    backend = editor.backend
    if state == 'loading':
        editor.show_page(LOADING)
        editor.stage.set_loading(st('loading_title', editor.lang), st('loading_hint', editor.lang), instant=True)
        editor.stage.set_picture(picture_pixmap(backend.picture(editor.camera)))
        return
    if state == 'error':
        editor.retry = editor.start
        editor.failed(SceneError('box_refused', f'could not get a picture from {editor.camera} right now (is it online?)'))
        return
    if state == 'restored':
        from .scene_backend import Previous
        editor.loaded(backend.propose(editor.camera), previous=Previous(True, 1791480000.))
        backend.backups[editor.camera] = ({'camera': editor.camera, 'watched': [[0, .3], [1, .3], [1, 1], [0, 1]],
                                           'areas': [], 'lines': []}, 1791480000.)
        editor.show_saved(backend.restore(editor.camera))
        return
    from .scene_backend import Previous
    editor.loaded(backend.propose(editor.camera, grid=state == 'grid'), previous=Previous(True, 1791480000.))
    if state == 'grid':
        editor.select_region(6)
        return
    if state == 'wall':
        # The house front (2) is the wall between us and the neighbour: nothing around it says which side is ours.
        editor.names[6] = {'he': 'הגדר הימנית', 'en': 'the right hedge'}[editor.lang]
        editor.region_rows[6].set_answer(None, editor.names[6])
        editor.answer_region(6, BOUNDARY)
        editor.select_region(6)
        return
    for number, (choice, names) in DEMO_ANSWERS.items():
        if number in editor.region_rows:
            editor.names[number] = names[editor.lang]
            editor.answers[number] = choice
            editor.region_rows[number].set_answer(choice, names[editor.lang])
    editor.stage.update(); editor.refresh_counts()
    editor.select_region(6)
    if state == 'regions':
        return
    editor.hands.append(HandArea([[.80, .07], [.97, .07], [.97, .27], [.80, .27]], 'hide',
                                 {'he': 'החלון של השכן', 'en': 'the neighbour’s window'}[editor.lang]))
    editor.build_hand_rows()
    if state == 'drawing':
        editor.tabs.select(DRAW); editor.tab_changed(DRAW)
        editor.start_area()
        editor.stage.draft = [[.02, .74], [.20, .70], [.30, .96]]
        point_at(editor.stage, (.08, .97))
        editor.draft_changed()
        return
    editor.lines.append(boundary_toward((.16, .47), (.86, .47), (.5, .75),
                                        {'he': 'המעקה', 'en': 'the railing'}[editor.lang]))
    editor.build_line_rows()
    if state == 'line':
        editor.select_line(0)
        editor.stage.start_line()
        editor.stage.line_draft = [(.40, .63), (.12, .99)]
        point_at(editor.stage, (.12, .62))
        editor.line_draft_changed()
        return
    editor.open_summary()
    if state == 'summary':
        return
    editor.show_saved(backend.confirm(editor.camera, editor.pending_map))


def open_demo_dialog(page, state):
    """Once the demo camera page has its cards, open the first camera's map in *state* (``card``: none)."""
    def attempt():
        if not page.loaded or page.future is not None or not page.controls.records:
            QTimer.singleShot(40, attempt)
            return
        if state != 'card':
            open_map_dialog(page, page.controls.records[0].name, demo_state=state)
    QTimer.singleShot(0, attempt)


def grab_with_dialogs(window):
    """A screenshot of *window* with any open dialog drawn over it, centred on a dimmed window, as it looks."""
    from PySide6.QtWidgets import QApplication
    image = window.grab()
    dialogs = [w for w in QApplication.topLevelWidgets() if isinstance(w, QDialog) and w.isVisible()]
    if not dialogs:
        return image
    p = QPainter(image)
    p.fillRect(image.rect(), QColor(0, 0, 0, 150))
    for dialog in dialogs:
        shot = dialog.grab()
        x = max(0, (window.width() - shot.width()) // 2)
        y = max(0, (window.height() - shot.height()) // 2)
        p.drawPixmap(x, y, shot)
    p.end()
    return image
