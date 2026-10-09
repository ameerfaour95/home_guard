"""K extends a track (the owner, 2026-10-09: "a track's YOLO box starts in the middle; I put the player at the start
and press K and it isn't added"): K before the first keyframe or past the end adds a keyframe with the nearest box,
the span up to the track is interpolated, the track becomes human; inside a hidden (O) segment it shows from there.
The saved request exports YOLO labels on the newly covered frames."""
from dataclasses import asdict

from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.label_document import LabelDocument
from home_guard_project.admin.label_view import LabelView
from home_guard_project.admin.models import Keyframe, Track
from home_guard_project.cloud import labeling
from home_guard_project.fleet_contract.tracks import box_at

A, B, C = [.10, .20, .20, .40], [.50, .20, .60, .40], [.30, .30, .40, .60]


def document(*keyframes, source='yolo'):
    a = DemoBackend().annotation(101)                                   # 12 fps, 72 frames
    a.tracks = [Track('t-1', 'person', [Keyframe(f, f/12, list(box), on) for f, box, on in keyframes], source)]
    d = LabelDocument(a, 6); d.selected = 't-1'
    return d


def shown(d, frame):
    return box_at(d.tracks[0], d.time_for(frame))


def exported(d, frame):
    """The YOLO label rows the export writes for *frame* from what this document would save."""
    tracks = labeling.to_tracks([asdict(t) for t in d.request().tracks])
    return labeling.yolo_rows(tracks, d.time_for(frame))


def test_k_before_the_first_keyframe_extends_the_track_to_that_frame(app):
    d = document((20, A, True), (30, B, True), (40, B, False))
    assert shown(d, 5) is None and exported(d, 5) == []
    d.seek(5); d.toggle_keyframe()
    tr = d.tracks[0]
    assert [k.frame for k in tr.keyframes] == [5, 20, 30, 40] and tr.keyframes[0].xyxy == A and tr.keyframes[0].enabled
    assert tr.source == 'human'
    for f in range(5, 20):
        assert shown(d, f) == A                                          # the span is interpolated (A to A)
    assert shown(d, 4) is None                                           # nothing before the new start
    assert all(exported(d, f) for f in range(5, 40)) and exported(d, 4) == []
    d.undo()                                                             # one undo takes it all back
    assert [k.frame for k in d.tracks[0].keyframes] == [20, 30, 40] and d.tracks[0].source == 'yolo'


def test_k_at_frame_0_on_a_track_that_starts_later(app):
    d = document((12, B, True), (24, C, True))
    d.seek(0); d.toggle_keyframe()
    assert d.tracks[0].keyframes[0].frame == 0 and d.tracks[0].keyframes[0].xyxy == B
    mid = shown(d, 6)
    assert mid == B
    assert [round(v, 6) for v in shown(d, 18)] == [round((b+c)/2, 6) for b, c in zip(B, C)]   # untouched after


def test_k_past_the_end_extends_the_track_and_it_still_ends_after_the_new_keyframe(app):
    d = document((20, A, True), (30, B, True), (40, B, False))          # YOLO's end marker at 40
    assert shown(d, 50) is None
    d.seek(50); d.toggle_keyframe()
    tr = d.tracks[0]
    assert [(k.frame, k.enabled) for k in tr.keyframes] == [(20, True), (30, True), (50, True), (51, False)]
    assert tr.keyframes[2].xyxy == B and tr.source == 'human'
    for f in range(30, 51):
        assert shown(d, f) == B                                          # 40..49 used to be gone: now covered
    assert shown(d, 51) is None and shown(d, 60) is None
    assert all(exported(d, f) for f in range(40, 51)) and exported(d, 51) == []
    d.seek(55); d.toggle_keyframe()                                      # K further on extends it again
    assert [(k.frame, k.enabled) for k in d.tracks[0].keyframes][-2:] == [(55, True), (56, False)]


def test_k_past_the_end_on_the_last_frame_leaves_no_marker_outside_the_clip(app):
    d = document((20, A, True), (30, A, False))
    d.seek(71); d.toggle_keyframe()
    assert [(k.frame, k.enabled) for k in d.tracks[0].keyframes] == [(20, True), (71, True)]
    assert all(exported(d, f) for f in range(20, 72))


def test_k_inside_a_hidden_segment_shows_the_track_from_that_frame(app):
    d = document((10, A, True), (20, A, False), (40, C, True), (50, C, False))   # O hid 20..39
    d.seek(30); d.toggle_keyframe()
    tr = d.tracks[0]
    assert [(k.frame, k.enabled) for k in tr.keyframes] == [(10, True), (20, False), (30, True), (40, True),
                                                             (50, False)]
    assert tr.keyframes[2].xyxy == C                                     # the nearest visible keyframe's box (40)
    assert all(shown(d, f) is None for f in range(20, 30))               # before it stays hidden
    assert all(shown(d, f) == C for f in range(30, 50))
    assert exported(d, 25) == [] and exported(d, 30) and exported(d, 35)
    assert tr.source == 'human'


def test_k_where_the_box_is_visible_still_adds_and_removes_a_keyframe(app):
    d = document((0, A, True), (24, B, True))
    d.seek(12); d.toggle_keyframe()
    assert [k.frame for k in d.tracks[0].keyframes] == [0, 12, 24]
    d.toggle_keyframe()
    assert [k.frame for k in d.tracks[0].keyframes] == [0, 24]


def test_k_extends_in_the_view_and_the_save_keeps_it(widgets, wait):
    class Backend(DemoBackend):
        def annotation(self, event_id):
            a = super().annotation(event_id)
            if a.version == 0:                                           # YOLO found the person from frame 30 on
                a.tracks = [Track('t-1', 'person', [Keyframe(30, 30/12, A), Keyframe(50, 50/12, B),
                                                    Keyframe(60, 60/12, B, False)], 'yolo')]
            return a
    b = Backend()
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy)
    tr = v.doc.tracks[0]
    first = tr.keyframes[0].frame
    v.doc.selected = tr.track_id
    v.seek(0); wait(lambda: v.pending_frame is None and v.doc.frame == 0)
    assert box_at(tr, v.doc.t_sec) is None
    v.activateWindow(); wait(v.isActiveWindow)
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.keyClick(v, Qt.Key.Key_K)
    tr = next(t for t in v.doc.tracks if t.track_id == tr.track_id)
    assert tr.keyframes[0].frame == 0 and tr.source == 'human' and box_at(tr, v.doc.t_sec) is not None
    v.save_button.click()
    wait(lambda: b.annotation(101).version == 1)
    saved = next(t for t in b.annotation(101).tracks if t.track_id == tr.track_id)
    assert saved.keyframes[0].frame == 0 and saved.source == 'human'
    rows = labeling.yolo_rows(labeling.to_tracks([asdict(t) for t in b.annotation(101).tracks]), v.doc.time_for(0))
    assert len(rows) == 1 and first == 30                                # frame 0 now has the person's label


# ---------------------------------------------------------------- "extended here by K": a mark where the span starts

def timeline(widgets, d):
    from home_guard_project.admin.label_canvas import TrackTimeline
    tl = TrackTimeline(); tl.doc = d; tl.resize(900, 120); tl.show(); widgets.append(tl)
    return tl


def amber_at(tl, frame, row=0):
    from PySide6.QtGui import QColor
    from home_guard_project.admin.theme import PALETTES
    image = tl.grab().toImage()
    want = QColor(PALETTES['dark']['warning'])
    y = 48 + row*tl.row_height
    def near(c):                                                         # antialiased: close to amber
        return abs(c.red()-want.red()) + abs(c.green()-want.green()) + abs(c.blue()-want.blue()) < 60
    return any(near(image.pixelColor(round(tl.x(frame)) + dx, y + 15)) for dx in (-3, -2, 2, 3))   # the flag


def hover(tl, frame, row=0):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    QTest.mouseMove(tl, QPoint(round(tl.x(frame)), 48 + row*tl.row_height))
    return tl.toolTip()


def test_the_mark_sits_where_an_extension_at_the_start_begins(app, widgets):
    d = document((20, A, True), (30, B, True), (40, B, False))
    tl = timeline(widgets, d)
    assert d.extension_marks(d.tracks[0]) == [] and not amber_at(tl, 5)
    d.seek(5); d.toggle_keyframe()
    assert d.extension_marks(d.tracks[0]) == [5] and d.tracks[0].source == 'human'
    assert amber_at(tl, 5) and not amber_at(tl, 30)
    assert hover(tl, 5) == 'extended here by K' and hover(tl, 30) == ''
    d.undo()                                                             # undone: the mark goes with it
    assert d.extension_marks(d.tracks[0]) == [] and not amber_at(tl, 5)


def test_the_mark_sits_at_the_old_end_when_the_track_is_extended_past_it(app, widgets):
    d = document((20, A, True), (30, B, True), (40, B, False))
    tl = timeline(widgets, d)
    d.seek(50); d.toggle_keyframe()
    assert d.extension_marks(d.tracks[0]) == [40]                         # the first newly covered frame
    assert amber_at(tl, 40) and hover(tl, 40) == 'extended here by K'


def test_the_mark_sits_where_k_uncovered_a_hidden_gap(app, widgets):
    d = document((10, A, True), (20, A, False), (40, C, True), (50, C, False))
    tl = timeline(widgets, d)
    d.seek(30); d.toggle_keyframe()
    assert d.extension_marks(d.tracks[0]) == [30]
    assert amber_at(tl, 30) and not amber_at(tl, 20) and hover(tl, 30) == 'extended here by K'
    d.seek(12); d.toggle_keyframe()                                      # a plain K where the box shows: no mark
    assert d.extension_marks(d.tracks[0]) == [30]
