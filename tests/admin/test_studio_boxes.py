"""Tag · YOLO and Tag · AI boxes: the session's Boxes toggle (B), hiding one track (H), P1 / CAR1 names, split
(Alt+M) and merge (M) so P1 stays P1, and the provenance chips."""
from copy import deepcopy

import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from home_guard_project.admin import player
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.label_canvas import LabelCanvas
from home_guard_project.admin.label_document import LabelDocument, track_names
from home_guard_project.admin.label_view import LabelView
from home_guard_project.admin.models import Keyframe, Track
from home_guard_project.admin.tag_view import TagView
from home_guard_project.admin.tag_widgets import ProvenanceChip, provenance_text
from home_guard_project.fleet_contract.tracks import box_at


@pytest.fixture(autouse=True)
def boxes_on():
    player.SESSION['boxes'] = True
    yield
    player.SESSION['boxes'] = True


def kf(frame, box, enabled=True, fps=10.):
    return Keyframe(frame, frame / fps, list(box), enabled)


def document(tracks):
    a = DemoBackend().annotation(101)
    a.tracks = tracks
    return LabelDocument(a, 6)


def test_names_follow_the_box_entities():
    tracks = [Track('a', 'truck', [kf(5, [.5, .5, .6, .6])]), Track('b', 'person', [kf(2, [.1, .1, .2, .2])]),
              Track('c', 'car', [kf(1, [.3, .3, .4, .4])]), Track('d', 'dog', [kf(0, [.7, .7, .8, .8])]),
              Track('e', 'person', [kf(9, [.1, .5, .2, .6])])]
    assert track_names(tracks) == {'d': 'dog A1', 'c': 'car CAR1', 'b': 'person P1', 'a': 'truck CAR2',
                                   'e': 'person P2'}
    # a saved entity wins, and the others never take its name
    tracks[4].entity = 'P1'
    assert track_names(tracks)['e'] == 'person P1' and track_names(tracks)['b'] == 'person P2'
    assert track_names([Track('x', 'bicycle', [kf(0, [.1, .1, .2, .2])])]) == {'x': 'bicycle #1'}


def test_split_at_the_current_frame_and_merge_back():
    p = Track('t-1', 'person', [kf(0, [.1, .1, .2, .3]), kf(20, [.5, .1, .6, .3])], 'yolo')
    d = document([p]); d.selected = 't-1'; d.seek(10, 1.0)
    p = d.tracks[0]                                         # the document edits its own copy
    new = d.split()
    assert new is not None and d.selected == new.track_id and len(d.tracks) == 2
    assert [(k.frame, k.enabled) for k in p.keyframes] == [(0, True), (10, False)] and p.source == 'human'
    assert [(k.frame, k.enabled) for k in new.keyframes] == [(10, True), (20, True)]
    assert box_at(p, 1.5) is None and box_at(new, 1.5) == pytest.approx([.4, .1, .5, .3])
    assert d.display_names() == {'t-1': 'person P1', new.track_id: 'person P2'}
    # merging from the later track: the first one keeps its id, so P1 stays P1 across the cut
    assert d.merge_candidates(new) == [p]
    assert d.merge()
    assert [t.track_id for t in d.tracks] == ['t-1'] and d.selected == 't-1'
    assert box_at(p, 1.5) == pytest.approx([.4, .1, .5, .3])
    d.undo(); assert len(d.tracks) == 2


def test_merge_is_refused_when_the_tracks_overlap_in_time():
    a = Track('t-1', 'person', [kf(0, [.1, .1, .2, .3]), kf(10, [.2, .1, .3, .3]), kf(11, [.2, .1, .3, .3], False)])
    b = Track('t-2', 'person', [kf(5, [.6, .1, .7, .3]), kf(15, [.7, .1, .8, .3])])
    c = Track('t-3', 'car', [kf(12, [.6, .5, .9, .9])])
    d = document([a, b, c]); d.selected = 't-2'
    a, b, c = d.tracks
    assert d.merge_candidates() == [] and not d.merge(a) and len(d.tracks) == 3
    # touching is not overlapping: hidden from frame 11, the next one may start at 11
    b.keyframes[0] = kf(11, [.6, .1, .7, .3])
    assert d.merge_candidates() == [a] and d.merge()
    assert [(k.frame, k.enabled) for k in a.keyframes] == [(0, True), (10, True), (11, True), (15, True)]


def test_boxes_toggle_and_hidden_track_on_the_canvas(app, widgets):
    t = Track('t-1', 'person', [kf(0, [.1, .1, .4, .6])], 'yolo')
    d = document([t]); d.seek(0, 0.)
    c = LabelCanvas(); c.doc = d; c.resize(640, 360); c.image = QImage(640, 360, QImage.Format.Format_RGB32)
    c.show(); widgets.append(c)
    inside = QPoint(150, 100)
    player.SESSION['boxes'] = False
    QTest.mouseClick(c, Qt.MouseButton.LeftButton, pos=inside)
    assert d.selected is None and not d.tracks[1:]          # boxes off: nothing selected, nothing drawn
    player.SESSION['boxes'] = True
    d.hidden.add('t-1')
    QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=inside); QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=inside)
    assert d.selected != 't-1'                              # a hidden track cannot be picked
    d.hidden.clear(); d.tracks = [t]
    QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=inside); QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=inside)
    assert d.selected == 't-1'
    c.grab()                                                # paints with a hidden track and boxes off without error
    player.SESSION['boxes'] = False; d.hidden.add('t-1'); c.grab()


def test_label_view_keys_b_h_m(widgets, wait):
    b = DemoBackend()
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy)
    assert v.boxes_source.text() == 'YOLO weak' and v.boxes_toggle.isChecked()
    v.setFocus(); QTest.keyClick(v, Qt.Key.Key_B)
    assert player.SESSION['boxes'] is False and not v.boxes_toggle.isChecked()
    QTest.keyClick(v, Qt.Key.Key_B); assert player.SESSION['boxes'] is True
    first = v.doc.tracks[0].track_id
    v.doc.selected = first; v.selection_changed()
    QTest.keyClick(v, Qt.Key.Key_H)
    assert first in v.doc.hidden and v.hide_track.text() == 'Show track  H' and not v.doc.dirty   # view only
    QTest.keyClick(v, Qt.Key.Key_H); assert first not in v.doc.hidden
    v.doc.seek(v.doc.tracks[0].keyframes[0].frame + 1)
    before = len(v.doc.tracks)
    QTest.keyClick(v, Qt.Key.Key_M, Qt.KeyboardModifier.AltModifier)
    assert len(v.doc.tracks) == before + 1 and v.doc.selected != first     # Alt+M: a new track from this frame
    QTest.keyClick(v, Qt.Key.Key_M)                                           # M: merged back, the first keeps its id
    assert len(v.doc.tracks) == before and v.doc.selected == first


@pytest.mark.parametrize('source, chip, note', [('tracker', 'Tracker', 'tracker box track'),
                                                ('dataset', 'Dataset labels', 'dataset box track'),
                                                (None, 'YOLO weak', 'YOLO box track')])
def test_preload_source_is_named_on_the_label_view(widgets, wait, source, chip, note):
    b = DemoBackend()
    real = b.annotation
    def annotation(eid):
        a = real(eid); a.preload_source = source; return a
    b.annotation = annotation
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy)
    assert v.boxes_source.text() == chip and note in v.boxes_note.text()


def test_tag_view_boxes_overlay_and_toggle(widgets, wait):
    b = DemoBackend()
    boxes = b.annotation(101)
    boxes.tracks = [Track('t-1', 'person', [kf(0, [.1, .1, .4, .6])], 'yolo'),
                    Track('t-2', 'car', [kf(0, [.5, .5, .9, .9])], 'yolo')]
    boxes.preload_source = 'tracker'
    b.clip_boxes = lambda key: deepcopy(boxes)
    v = TagView(b, 'admin'); widgets.append(v); v.resize(1366, 700); v.show(); v.open()
    wait(lambda: v.detail is not None and not v.clip_runner.busy, 10)
    dataset_key = next(r['key'] for r in v.model.rows if r['key'].startswith('ds:'))
    v.open_key(dataset_key); wait(lambda: v.key == dataset_key and not v.boxes_runner.busy and v.canvas.overlay.tracks, 10)
    overlay = v.canvas.overlay
    assert overlay.names == {'t-1': 'person P1', 't-2': 'car CAR1'} and overlay.source == 'tracker'
    assert len(overlay.visible(0.)) == 2 and v.boxes.isChecked()
    v.setFocus(); QTest.keyClick(v, Qt.Key.Key_B)
    assert player.SESSION['boxes'] is False and not v.boxes.isChecked()
    v.view = 'crop'; v.render_boxes(); assert overlay.message is None       # boxes off: no crop note either
    QTest.keyClick(v, Qt.Key.Key_B); assert overlay.message and 'full scene' in overlay.message
    v.view = 'clip'; v.render_boxes(); assert overlay.message is None
    v.canvas.grab()


def test_provenance_chips(app):
    assert provenance_text('owner') == 'Owner · Telegram'
    assert provenance_text('admin', 'Dana') == 'Admin · Dana' and provenance_text('model', 'gpt-4o') == 'Model · gpt-4o'
    assert provenance_text('yolo') == 'YOLO weak' and provenance_text('tracker') == 'Tracker'
    assert provenance_text('nope') == ''
    chip = ProvenanceChip(kind='owner')
    assert chip.text() == 'Owner · Telegram' and chip.kind == 'owner'
    chip.show_source(''); assert chip.isHidden()


def test_tag_yolo_has_no_description_panel_only_tracks_and_clip_checks(widgets, wait):
    from PySide6.QtWidgets import QPlainTextEdit
    b = DemoBackend()
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy)
    assert not [w for w in v.findChildren(QPlainTextEdit) if w.isVisible()]       # the description is Tag · AI's
    assert v.track_list.count() == len(v.doc.tracks) and 'YOLO' in v.track_list.item(0).text()
    v.track_list.itemClicked.emit(v.track_list.item(0))
    assert v.doc.selected == v.track_list.item(0).data(Qt.ItemDataRole.UserRole)
    assert v.track_name.text() == v.doc.display_name(v.doc.track) and v.track_source.text() == 'YOLO weak'
    before = b.annotation(101).description
    v.needs_review.setChecked(True); v.save(); wait(lambda: not v.writer.busy)
    saved = b.annotation(101)
    assert saved.needs_review and saved.description == before                  # carried through unchanged


def test_entities_stay_stable_through_split_merge_new_boxes_and_saves(widgets, wait):
    p = Track('t-1', 'person', [kf(0, [.1, .1, .2, .3]), kf(20, [.5, .1, .6, .3])], 'yolo', 'P1')
    c = Track('t-2', 'car', [kf(0, [.6, .6, .9, .9])], 'yolo')                 # an old save: no entity yet
    d = document([p, c])
    assert [t.entity for t in d.tracks] == ['P1', 'CAR1'] and not d.dirty     # named once, on load
    d.selected = 't-1'; d.seek(10, 1.0)
    new = d.split()
    assert new.entity == 'P2' and d.tracks[0].entity == 'P1'                  # a fresh name, never a copy
    assert d.display_names()[new.track_id] == 'person P2'
    d.merge(); assert [t.entity for t in d.tracks] == ['P1', 'CAR1']          # the first keeps its name
    d.selected = None; d.current_class = 'person'; d.seek(5, .5); d.put_box([.3, .3, .4, .6])
    assert d.track.entity == 'P2'                                             # a new box: the next free id
    d.change_class('dog'); assert d.track.entity == 'A1'                      # its kind changed: a name of that kind
    # through a real save and reload: the names come back as they were
    b = DemoBackend()
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy)
    names = {t.track_id: t.entity for t in v.doc.tracks}
    assert all(names.values())
    v.doc.accept_all(); v.save(); wait(lambda: not v.writer.busy)
    assert sorted(t.entity for t in b.annotation(101).tracks) == sorted(names.values())
