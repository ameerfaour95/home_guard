"""Tag · YOLO zoom and pan keep boxes in image coordinates; the track strip seeks: a click moves the video to that
frame, dragging scrubs, a keyframe dot jumps to its own frame; a seek never stays stuck on a decoder's timing."""
import pytest
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.label_canvas import LabelCanvas, TrackTimeline
from home_guard_project.admin.label_document import LabelDocument
from home_guard_project.admin.label_view import LabelView
from home_guard_project.admin.models import Keyframe, Track


def canvas(widgets, tracks=()):
    a = DemoBackend().annotation(101); a.tracks = list(tracks)
    d = LabelDocument(a, 6); d.seek(0, 0.)
    c = LabelCanvas(); c.doc = d; c.resize(640, 360); c.image = QImage(640, 360, QImage.Format.Format_RGB32)
    c.show(); widgets.append(c)
    return c, d


def at(c, nx, ny):
    """The widget point of image point (nx, ny) at the canvas's current zoom."""
    x, y, w, h = c.display_rect()
    return QPoint(round(x + nx*w), round(y + ny*h))


def drag(c, a, b):
    QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(c, b); QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=b)


def test_drawing_at_1x_then_resizing_at_3x_equals_drawing_at_3x(app, widgets):
    c1, d1 = canvas(widgets)
    drag(c1, at(c1, .40, .30), at(c1, .55, .50))                     # draw at 1x
    c1.zoom_to(3.0, QPointF(320, 180))
    box = d1.track.keyframes[0].xyxy
    drag(c1, at(c1, box[2], box[3]), at(c1, .60, .58))              # resize the bottom-right handle at 3x
    c3, d3 = canvas(widgets)
    c3.zoom_to(3.0, QPointF(320, 180))
    drag(c3, at(c3, .40, .30), at(c3, .60, .58))                     # the same box drawn at 3x directly
    one, three = d1.track.keyframes[0].xyxy, d3.track.keyframes[0].xyxy
    assert one == pytest.approx(three, abs=1.5/640)                  # within a pixel at 1x
    assert one == pytest.approx([.40, .30, .60, .58], abs=1.5/640)


def test_zoom_keeps_the_point_under_the_cursor_and_fit_resets(app, widgets):
    c, _ = canvas(widgets)
    anchor = QPointF(500, 100)
    before = c.image_point(anchor)
    c.zoom_to(3.0, anchor)
    assert c.zoom == 3.0 and c.image_point(anchor).x() == pytest.approx(before.x()) \
        and c.image_point(anchor).y() == pytest.approx(before.y())
    c.zoom_to(100.0); assert c.zoom == LabelCanvas.MAX_ZOOM
    c.fit(); assert c.zoom == 1.0 and c.display_rect() == c.fit_rect()


def test_wheel_zoom_and_middle_button_pan(app, widgets):
    c, _ = canvas(widgets)
    QTest.qWait(1)
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtCore import QPoint as P
    ev = QWheelEvent(QPointF(320, 180), c.mapToGlobal(QPointF(320, 180)), P(0, 0), P(0, 240),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    c.wheelEvent(ev)
    assert c.zoom == pytest.approx(LabelCanvas.STEP**2)
    x0 = c.display_rect()[0]
    QTest.mousePress(c, Qt.MouseButton.MiddleButton, pos=QPoint(300, 180))
    QTest.mouseMove(c, QPoint(340, 180)); QTest.mouseRelease(c, Qt.MouseButton.MiddleButton, pos=QPoint(340, 180))
    assert c.display_rect()[0] == pytest.approx(x0 + 40)
    assert c.doc.tracks == []                                        # panning never draws a box


def test_boxes_follow_the_zoom(app, widgets):
    t = Track('t-1', 'person', [Keyframe(0, 0., [.4, .4, .6, .6])], 'yolo', 'P1')
    c, _ = canvas(widgets, [t])
    small = c.rect_for(t.keyframes[0].xyxy)
    c.zoom_to(3.0)
    big = c.rect_for(t.keyframes[0].xyxy)
    assert big.width() == pytest.approx(small.width()*3) and big.center().x() == pytest.approx(small.center().x())
    c.grab()


# ---------------------------------------------------------------- the track strip

def view(widgets, wait):
    b = DemoBackend()
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy and not v.canvas.image.isNull(), 10)
    v.autosave.setInterval(10**9)
    return v


def test_clicking_a_time_moves_the_video_there(widgets, wait):
    v = view(widgets, wait)
    tl = v.timeline
    QTest.mouseClick(tl, Qt.MouseButton.LeftButton, pos=QPoint(round(tl.x(40)), 20))     # on the time ruler
    wait(lambda: v.doc.frame == 40 and v.pending_frame is None, 5)
    QTest.mouseClick(tl, Qt.MouseButton.LeftButton, pos=QPoint(round(tl.x(12)), tl.height()-6))   # below the rows
    wait(lambda: v.doc.frame == 12 and v.pending_frame is None, 5)


def test_dragging_the_playhead_scrubs(widgets, wait):
    v = view(widgets, wait)
    tl, seen = v.timeline, []
    tl.seek_requested.connect(seen.append)
    QTest.mousePress(tl, Qt.MouseButton.LeftButton, pos=QPoint(round(tl.x(5)), 20))
    for f in (15, 25, 35):
        QTest.mouseMove(tl, QPoint(round(tl.x(f)), 20))
    QTest.mouseRelease(tl, Qt.MouseButton.LeftButton, pos=QPoint(round(tl.x(35)), 20))
    assert seen[:4] == [5, 15, 25, 35]
    wait(lambda: v.doc.frame == 35 and v.pending_frame is None, 5)


def test_clicking_a_keyframe_dot_jumps_to_its_frame_and_never_moves_it(widgets, wait):
    v = view(widgets, wait)
    tl, tr = v.timeline, v.doc.tracks[0]
    dot = tr.keyframes[-1]
    y = 48                                                           # the first track's row
    QTest.mouseClick(tl, Qt.MouseButton.LeftButton, pos=QPoint(round(tl.x(dot.frame)) + 4, y))   # a little off
    wait(lambda: v.doc.frame == dot.frame and v.pending_frame is None, 5)
    assert [k.frame for k in tr.keyframes][-1] == dot.frame and not v.doc.dirty


def test_a_seek_does_not_stay_stuck_when_the_decoder_answers_off_by_one(widgets, wait):
    v = view(widgets, wait)
    v.seek(30)
    assert v.pending_frame == 30
    v.settle_seek(t_sec=31 / v.doc.fps)                              # a decoded frame a frame later than asked
    assert v.pending_frame is None and v.doc.frame == 30 and v.canvas.isEnabled()
    v.seek(10); v.seek_watchdog.timeout.emit()                       # nothing decoded at all
    wait(lambda: v.pending_frame is None, 5)
    assert v.doc.frame == 10 and v.canvas.isEnabled()
