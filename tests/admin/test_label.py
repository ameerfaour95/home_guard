from copy import deepcopy
from dataclasses import asdict
import json
import threading
import httpx
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtGui import QImage
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.backend import ConflictError, ServerError, ForbiddenError
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.models import Track, Keyframe, AnnotationIn, ReviewDecision, PublishOut
from home_guard_project.admin.label_document import LabelDocument, CLASSES
from home_guard_project.admin.label_canvas import LabelCanvas
from home_guard_project.admin.label_view import LabelView
from home_guard_project.admin.publish import PublishDialog, PublishList
from home_guard_project.fleet_contract.tracks import box_at, boxes_at, frame_time


def document():
    a = DemoBackend().annotation(101)
    a.tracks = [Track('human1', 'person', [Keyframe(0, 0., [.1, .2, .2, .4]), Keyframe(4, 4/12, [.5, .2, .6, .4])])]
    return LabelDocument(a, 6)


def view(widgets, wait, backend=None):
    b = backend or DemoBackend()
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy)
    return v


def test_founder_time_interpolation_and_hidden_segment(app):
    d = document(); d.selected = 'human1'; d.seek(2)
    assert d.visible_boxes()[0][1] == pytest.approx([.3, .2, .4, .4])
    assert d.visible_boxes() == boxes_at(d.tracks, d.t_sec)
    d.seek(0); d.set_enabled(); d.seek(2)
    assert box_at(d.track, d.t_sec) is None and not d.visible_boxes()
    d.seek(0); d.set_enabled(); d.seek(2)
    assert d.visible_boxes()[0][1] == pytest.approx([.3, .2, .4, .4])
    d.track.keyframes[1].t_sec = .8
    d.seek(2, .2)
    assert d.visible_boxes()[0][1] == pytest.approx([.2, .2, .3, .4])


def test_draw_move_resize_real_time_and_bounds(app, widgets):
    d = document(); d.tracks = []; d.seek(7, .619)
    c = LabelCanvas(); c.doc = d; c.resize(640, 360); c.image = QImage(640, 360, QImage.Format.Format_RGB32); c.show(); widgets.append(c)
    def drag(a, b):
        QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=QPoint(*a))
        QTest.mouseMove(c, QPoint(*b)); QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=QPoint(*b))
    drag((100, 100), (200, 200))
    assert d.keyframe().frame == 7 and d.keyframe().t_sec == .619
    assert d.keyframe().xyxy == pytest.approx([100/640, 100/360, 200/640, 200/360])
    d.seek(8, .701); drag((150, 150), (170, 160))
    assert d.keyframe().frame == 8 and d.keyframe().t_sec == .701
    assert d.keyframe().xyxy[0] == pytest.approx(120/640)
    d.seek(9, .79); drag((220, 210), (260, 230))
    assert d.keyframe().t_sec == .79 and d.keyframe().xyxy[2] == pytest.approx(260/640)
    d.seek(10); drag((180, 160), (1000, 1000))
    assert max(d.keyframe().xyxy) <= 1
    assert not d.put_box([0, 0, .001, .001])


def test_accept_history_text_class_keyframes(app):
    d = document(); d.tracks[0].source = 'suggestion'; d.history = [d.snapshot()]
    d.accept_all(); assert d.tracks[0].source == 'human'
    d.undo(); assert d.tracks[0].source == 'suggestion'
    d.redo(); d.selected = 'human1'; d.seek(1); d.change_class('car')
    assert d.keyframe().t_sec == frame_time(1, 12)
    d.copy_next(); assert d.frame == 2 and d.keyframe()
    d.set_text('Corrected description', True, True); d.undo()
    assert not d.drop_clip and d.description != 'Corrected description'
    d.redo(); assert d.drop_clip and d.needs_review
    d.move_keyframe('human1', 2, 3); assert any(k.frame == 3 for k in d.track.keyframes)
    d.seek(3); d.delete_keyframe(); assert not d.keyframe()
    d.delete_track(); assert not d.tracks; d.undo(); assert len(d.tracks) == 1


def test_autosave_debounce_and_save_button(widgets, wait):
    v = view(widgets, wait)
    v.description.setPlainText('First edit')
    QTest.qWait(1100); v.description.setPlainText('Second edit')
    QTest.qWait(1100); assert v.backend.annotation(101).version == 0
    wait(lambda: v.backend.annotation(101).version == 1)
    assert v.backend.annotation(101).description == 'Second edit'
    v.description.setPlainText('Explicit save'); v.save_button.click()
    wait(lambda: v.backend.annotation(101).version == 2)
    assert v.backend.annotation(101).status == 'edited'


def test_native_decoder_seek_and_copy_use_presentation_time(widgets, wait):
    v = view(widgets, wait)
    wait(lambda: not v.canvas.image.isNull())
    original = v.canvas.image.copy()
    v.seek(36)
    assert v.pending_frame == 36 and not v.canvas.isEnabled()
    wait(lambda: v.pending_frame is None)
    assert v.doc.frame == 36 and v.doc.t_sec == pytest.approx(3.)
    assert v.canvas.image != original
    v.doc.selected = v.doc.tracks[0].track_id
    v.copy_next(); wait(lambda: v.pending_copy is None)
    assert v.doc.frame == 37 and v.doc.keyframe().t_sec == v.doc.t_sec
    v.seek(2); v.step(1); wait(lambda: v.pending_frame is None)
    assert v.doc.frame == 3


def test_invalid_tracks_stay_inline_and_do_not_save(widgets, wait):
    v = view(widgets, wait)
    v.doc.tracks[0].keyframes[0].xyxy = [-.2, 0., .5, .5]; v.doc.checkpoint(); v.save()
    assert 'within 0..1' in v.error.text() and not v.writer.busy
    assert v.backend.annotation(101).version == 0


def test_save_in_flight_preserves_later_edits(widgets, wait):
    gate = threading.Event()
    class Slow(DemoBackend):
        def save_annotation(self, eid, value):
            gate.wait(3); return super().save_annotation(eid, value)
    v = view(widgets, wait, Slow())
    v.description.setPlainText('sent'); v.save()
    v.description.setPlainText('newer'); gate.set(); wait(lambda: not v.writer.busy)
    assert v.doc.dirty and v.doc.description == 'newer'
    v.save(); wait(lambda: not v.writer.busy)
    assert v.backend.annotation(101).description == 'newer' and not v.doc.dirty


def test_conflict_keep_and_reload(widgets, wait):
    v = view(widgets, wait)
    remote = AnnotationIn(0, [], 'Remote edit'); v.backend.save_annotation(101, remote)
    v.description.setPlainText('Mine'); v.save(); wait(lambda: v.conflicted)
    assert v.conflict_bar.isVisible() and not v.autosave.isActive()
    v.resolve_conflict(True); wait(lambda: v.backend.annotation(101).version == 2)
    assert v.backend.annotation(101).description == 'Mine'
    v.backend.save_annotation(101, AnnotationIn(2, [], 'Remote again'))
    v.description.setPlainText('Discard this'); v.save(); wait(lambda: v.conflicted)
    v.resolve_conflict(False); wait(lambda: v.doc.annotation.version == 3)
    assert v.description.toPlainText() == 'Remote again' and not v.doc.dirty


def test_switch_save_failure_and_submit_next(widgets, wait):
    class Failing(DemoBackend):
        fail = True
        def save_annotation(self, eid, value):
            if self.fail: raise ServerError()
            return super().save_annotation(eid, value)
    v = view(widgets, wait, Failing()); v.description.setPlainText('Keep me')
    next_index = v.queue_index+1
    if next_index == len(v.queue): next_index = 0
    v.open_index(next_index); wait(lambda: not v.writer.busy)
    assert v.doc.annotation.event_id == 101 and v.doc.dirty
    v.backend.fail = False; v.save(); wait(lambda: v.doc.annotation.event_id != 101)
    assert v.backend.annotation(101).description == 'Keep me'
    v.open_index(0); wait(lambda: v.queue_index == 0 and not v.loader.busy)
    eid = v.doc.annotation.event_id; v.submit()
    wait(lambda: v.queue_index == 1 and not v.loader.busy)
    assert v.backend.annotation(eid).status == 'submitted'


def test_labeler_privacy_and_keyboard(widgets, wait):
    v = view(widgets, wait, DemoBackend(role='labeler'))
    assert not v.ai_meta.isVisible() and not v.review_panel.isVisible()
    e = v.backend.event(101)
    assert e.customer_id == 0 and not e.raw_meta and all(r.prompt is None and r.raw_text_artifact_id is None for r in e.ai_runs)
    assert v.doc.annotation.ai_prompt_version is None
    expected = {'Left', 'Right', 'Shift+Left', 'Shift+Right', 'Space', '.', ',', 'K', 'H', 'C', 'Del', 'Shift+Del', 'Ctrl+Z', 'Ctrl+Shift+Z', 'Ctrl+S', 'Ctrl+Return', 'Esc', '?', *map(str, range(1, 10))}
    assert set(v.shortcuts) == expected
    v.player.setSource(__import__('PySide6.QtCore', fromlist=['QUrl']).QUrl())
    v.doc.seek(0); v.canvas.setFocus(); QTest.keyClick(v.canvas, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert v.doc.frame == 5
    QTest.keyClick(v.canvas, Qt.Key.Key_9); assert v.doc.current_class == 'dog'
    v.description.setFocus(); QTest.keyClicks(v.description, 'h c k 1')
    assert 'h c k 1' in v.doc.description and v.doc.current_class == 'dog'


def test_review_history_and_publish(widgets, wait):
    b = DemoBackend(); v = view(widgets, wait, b)
    v.queue = [b.event(101)]; v.queue_index = 0
    v.submit(); wait(lambda: v.doc.annotation.status == 'submitted')
    v.review_note.setText('Check the entrance box'); v.doc.seek(4); v.review('reject')
    wait(lambda: v.doc.annotation.status == 'rejected')
    assert b.annotation(101).review_frame == 4 and len(b.annotation_history(101)) == 2
    assert b.events(filter='rejected').items[0].id == 101
    b.save_annotation(101, AnnotationIn(2, [], 'Final', status='submitted'))
    c = b.create_collection('Label test'); b.add_collection_items(c.id, [101, 102])
    dialog = PublishDialog(b, c); widgets.append(dialog); dialog.show(); wait(lambda: not dialog.loader.busy)
    assert '1 labeled' in dialog.summary.text() and '1 not labeled' in dialog.summary.text()
    dialog.name.setText('Invalid-Name'); dialog.publish(); assert 'lowercase' in dialog.error.text()
    dialog.name.setText('label_batch_1'); dialog.publish(); wait(lambda: not dialog.writer.busy)
    result = b.publishes()[0]
    assert result.tasks == 1 and result.yolo_frames == 72 and result.vlm_lines == 1
    assert result.missing[0].event_id == 102
    screen = PublishList(b); widgets.append(screen); screen.show(); wait(lambda: bool(screen.model.items))
    screen.copy_path(); assert screen.current().s3_prefix.endswith('/tagging/label_batch_1/')
    with pytest.raises(ForbiddenError): DemoBackend(role='labeler').publishes()


def test_http_annotation_contract_and_conflict():
    b = DemoBackend(); calls = []
    def encode(value):
        if isinstance(value, list): return [encode(v) for v in value]
        return json.loads(json.dumps(asdict(value), default=lambda v: v.isoformat()))
    def respond(request):
        calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        path = request.url.path
        if path.endswith('/history'): return httpx.Response(200, json=[])
        if path.endswith('/publishes'): return httpx.Response(200, json=[])
        if path.endswith('/publish'): return httpx.Response(200, json=encode(b.publish_collection(1, 'site_batch_1')))
        return httpx.Response(200, json=encode(b.annotation(101)))
    h = HttpBackend('http://localhost:8610', transport=httpx.MockTransport(respond))
    h.annotation(101); h.save_annotation(101, AnnotationIn(0, [], 'Text')); h.review_annotation(101, ReviewDecision('reject', 'note', 4)); h.annotation_history(101); h.publishes()
    result = h.publish_collection(1, 'site_batch_1')
    assert result.s3_prefix == 's3://security-camera-project-v1/tagging/site_batch_1/'
    assert calls[-1] == ('POST', '/v1/studio/collections/1/publish', {'batch_name': 'site_batch_1'})
    assert calls[1] == ('PUT', '/v1/events/101/annotation', {'base_version': 0, 'tracks': [], 'description': 'Text', 'drop_clip': False, 'needs_review': False, 'status': 'edited'})
    assert calls[2][2] == {'decision': 'reject', 'note': 'note', 'frame': 4}
    with pytest.raises(ConflictError): h._parse(httpx.Response(409), object)
