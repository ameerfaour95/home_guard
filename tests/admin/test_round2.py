import json
import threading
from dataclasses import replace, asdict
from datetime import timedelta

import httpx
import pytest
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from home_guard_project.admin.backend import all_events, OfflineError
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.event_logic import density, video_rect, map_box, nearest_frame, provenance, ai_status, decision
from home_guard_project.admin.models import FrameBoxes, TokenPair, decode
from home_guard_project.admin.timeline import TimelineScreen
from home_guard_project.admin.customer import CustomerScreen
from home_guard_project.admin.fleet_model import FleetModel
from home_guard_project.admin.ai_record import TextDisclosure
from home_guard_project.admin.shell import Shell


def test_density_hour_boundaries_range_and_owner_false_alarms():
    backend = DemoBackend(); end = backend.now; start = end-timedelta(hours=6)
    base = backend.events().items[0]
    events = [replace(base, camera='a', start_utc=start, kind='alert', owner_verdicts=['false_alarm']),
              replace(base, camera='a', start_utc=start+timedelta(minutes=59), kind='trigger', owner_verdicts=[]),
              replace(base, camera='b', start_utc=start+timedelta(hours=1), kind='paused', owner_verdicts=[]),
              replace(base, camera='a', start_utc=end), replace(base, camera='a', start_utc=start-timedelta(seconds=1))]
    hours, rows = density(events, start, end)
    assert len(hours) == 6 and hours[0] == start
    assert rows['a'][0] == [2, 1, 1] and rows['b'][1] == [1, 0, 0]
    assert sum(c[0] for row in rows.values() for c in row) == 3
    partial_hours, partial = density(events, start+timedelta(minutes=30), end)
    assert partial_hours[0] == start and partial['a'][0] == [1, 0, 0]


@pytest.mark.parametrize('viewport,frame,expected', [((1000, 1000), (1920, 1080), (0, 218.75, 1000, 562.5)),
    ((1600, 900), (640, 480), (200, 0, 1200, 900)), ((640, 360), (640, 360), (0, 0, 640, 360))])
def test_letterbox_and_normalised_box_mapping(viewport, frame, expected):
    rect = video_rect(*viewport, *frame)
    assert rect == pytest.approx(expected)
    assert map_box([0, 0, 1, 1], rect) == rect
    assert map_box([.25, .25, .75, .75], rect) == (rect[0]+rect[2]/4, rect[1]+rect[3]/4, rect[2]/2, rect[3]/2)
    assert map_box([-1, -1, 2, 2], rect) == rect
    assert video_rect(0, 1, 1, 1) == (0, 0, 0, 0)


def test_nearest_frame_offset_ties_empty_and_not_run():
    frames = [FrameBoxes(i, float(i), status, []) for i, status in [(0, 'ran'), (2, 'ran_empty'), (4, 'not_run')]]
    assert nearest_frame(frames, 1000) is frames[0]
    assert nearest_frame(frames, 1500, -500) is frames[0]
    assert nearest_frame(frames, 1500, 500) is frames[1]
    assert nearest_frame(frames, 99999) is None
    assert nearest_frame(frames, -999) is frames[0]
    assert nearest_frame([], 0) is None


@pytest.mark.parametrize('status,expected', [('captured', 'Boxes: captured'), ('sampled', 'Boxes: sampled every 2nd frame'),
    ('recomputed', 'Boxes: recomputed in cloud'), ('none', 'No boxes saved for this clip')])
def test_provenance_is_honest(status, expected):
    frames = [FrameBoxes(i, i/12, 'ran', []) for i in (0, 2, 4)]
    assert provenance(status, frames) == expected
    assert provenance('sampled', frames[:1]) == 'Boxes: sampled frames'


@pytest.mark.parametrize('status,text', [('real', 'Real answer from gpt-4o'), ('failed', 'AI call failed — no answer'),
    ('fallback', 'Fallback: no model ran'), ('none', 'No AI answer recorded')])
def test_ai_status_words(status, text):
    assert ai_status(status, 'gpt-4o') == text
    assert decision('[none]') == 'No alert'
    assert decision('[send_message]') == 'Message sent'
    assert decision('[call_owner]') == 'Call owner'
    assert decision(None) == 'No decision recorded'


def test_stable_fleet_sort_preserves_received_ties():
    backend = DemoBackend(); device = backend.fleet().devices[0]
    items = [replace(device, device_id=str(i), customer_name=name, verdict=v) for i, (name, v) in enumerate([
        ('Z', 'healthy'), ('Z', 'critical'), ('Z', 'offline'), ('B', 'warning'), ('A', 'warning'), ('Q', 'unknown')])]
    model = FleetModel(lambda: backend.now); model.replace(items)
    assert [d.device_id for d in model.rows] == ['2', '1', '3', '4', '5', '0']


def test_demo_cursor_filters_and_review_round_trip():
    backend = DemoBackend()
    items = all_events(backend)
    assert len(items) == len({e.id for e in items}) == 96
    page = backend.events(limit=3); second = backend.events(limit=3, cursor=page.next_cursor)
    assert [e.id for e in page.items+second.items] == [e.id for e in items[:6]]
    backend.review(101, reviewed=True, flagged=True)
    assert backend.event(101).flagged
    assert 101 in [e.id for e in backend.events(customer_id=1, kind='alert', ai='real', verdict='true_alert',
        q='parcel', reviewed=True, flagged=True, from_utc=(backend.now-timedelta(hours=1)).isoformat(), to_utc=backend.now.isoformat()).items]
    assert not backend.events(ai='fallback', kind='alert').items


class RecordingBackend(DemoBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.queries, self.reviews = [], []

    def events(self, **filters):
        self.queries.append(filters.copy()); return super().events(**filters)

    def review(self, id, **changes):
        self.reviews.append((id, changes)); return super().review(id, **changes)


def timeline(widgets, wait, backend=None):
    screen = TimelineScreen(backend or RecordingBackend()); widgets.append(screen)
    screen.resize(1120, 600); screen.show(); screen.open(1, 'Asia/Jerusalem')
    wait(lambda: bool(screen.model.rows) and not screen.density_runner.busy)
    return screen


def test_timeline_filters_use_exact_contract_query_names(widgets, wait):
    screen = timeline(widgets, wait)
    values = dict(kind='alert', camera='Front door', ai='real', verdict='true_alert', reviewed=False, flagged=True)
    for key, value in values.items():
        combo = screen.filters[key]; combo.setCurrentIndex(combo.findData(value))
    screen.search.setText('parcel'); screen.debounce.stop(); screen.reload()
    wait(lambda: not screen.runner.busy)
    query = screen.backend.queries[-1]
    assert all(query[k] == v for k, v in values.items())
    assert query['q'] == 'parcel' and query['customer_id'] == 1 and query['limit'] == 25
    assert 'from_utc' in query and 'to_utc' in query and query['cursor'] is None
    hour = screen.end-timedelta(hours=1)
    screen.filter_cell('Garden', hour); wait(lambda: not screen.runner.busy)
    assert screen.backend.queries[-1]['camera'] == 'Garden'
    assert screen.backend.queries[-1]['from_utc'] == hour.isoformat()


def test_load_older_and_auto_scroll_preserve_existing_rows(widgets, wait):
    screen = timeline(widgets, wait); first = [e.id for e in screen.model.rows]
    assert len(first) == 25 and screen.cursor
    screen.load_older(); wait(lambda: len(screen.model.rows) > 25)
    assert [e.id for e in screen.model.rows[:25]] == first
    while screen.cursor:
        screen.scroll_end(screen.table.verticalScrollBar().maximum())
        wait(lambda: not screen.runner.busy)
    expected = all_events(screen.backend, **screen.query())
    assert [e.id for e in screen.model.rows] == [e.id for e in expected]
    assert not screen.older.isEnabled()


def test_keyboard_review_and_open_call_backend(widgets, wait):
    screen = timeline(widgets, wait); screen.table.setFocus()
    QTest.keyClick(screen.table, Qt.Key.Key_J)
    eid = screen.current().id
    reviewed = screen.current().reviewed
    QTest.keyClick(screen.table, Qt.Key.Key_R); wait(lambda: not screen.review_runner.busy)
    assert screen.backend.reviews[-1] == (eid, {'reviewed': not reviewed})
    QTest.keyClick(screen.table, Qt.Key.Key_F); wait(lambda: not screen.review_runner.busy)
    assert screen.backend.reviews[-1] == (eid, {'flagged': True})
    opened = []; screen.event_requested.connect(opened.append)
    QTest.keyClick(screen.table, Qt.Key.Key_Return)
    assert opened == [eid]
    QTest.keyClick(screen.table, Qt.Key.Key_K); assert screen.current().id == 101


def test_new_query_discards_late_old_page(widgets, wait):
    release = threading.Event(); started = threading.Event()
    class Slow(RecordingBackend):
        def events(self, **filters):
            if filters.get('limit') == 25 and not started.is_set():
                started.set(); release.wait(3)
            return super().events(**filters)
    screen = TimelineScreen(Slow()); widgets.append(screen); screen.show(); screen.open(1)
    try:
        wait(started.is_set)
        screen.filters['ai'].setCurrentIndex(screen.filters['ai'].findData('fallback'))
    finally:
        release.set()
    wait(lambda: bool(screen.model.rows) and not screen.runner.busy)
    assert all(e.completeness.ai == 'fallback' for e in screen.model.rows)


@pytest.mark.parametrize('role', ['admin', 'support', 'labeler'])
def test_role_tabs_dispatch_raw_and_owner_text(role, widgets, wait):
    backend = DemoBackend(role=role)
    screen = CustomerScreen(backend, role=role, review=role == 'labeler'); widgets.append(screen)
    screen.resize(1182, 688); screen.show()
    if role != 'labeler':
        screen.open(1)
    wait(lambda: bool(screen.timeline.model.rows))
    screen.open_event(101); view = screen.event_view
    wait(lambda: view.recording is not None and not view.evidence_runner.busy)
    names = [screen.tabs.tabText(i) for i in range(screen.tabs.count())]
    disclosures = [w.toggle.text() for w in view.record.findChildren(TextDisclosure)]
    if role == 'labeler':
        assert 'Conversation' not in names and 'Access' not in names
        assert view.record.dispatch_label is None and view.raw is None
        assert not any('Owner raw text' in s for s in disclosures)
        assert screen.name.text().startswith('customer-') and view.recording.camera.startswith('cam-')
        assert view.record.frames and not view.record.raw_answers
    else:
        assert 'Conversation' in names and 'Access' in names
        assert view.record.dispatch_label is not None
        assert any('Owner raw text' in s for s in disclosures)
        assert (view.raw is not None) == (role == 'admin')


def test_labeler_camera_filter_uses_server_pseudonym():
    backend = DemoBackend(role='labeler'); camera = backend.event(101).camera
    assert all(e.camera == camera for e in backend.events(camera=camera).items)
    assert backend.events(camera=camera).items


def test_real_video_decode_seek_overlay_and_shortcuts(widgets, wait):
    backend = DemoBackend(); screen = CustomerScreen(backend); widgets.append(screen)
    screen.resize(1182, 688); screen.show(); screen.open(1)
    wait(lambda: bool(screen.timeline.model.rows)); screen.open_event(101)
    view = screen.event_view; player = view.player
    wait(lambda: not view.evidence_runner.busy and player.player.duration() == 6000)
    player.player.play(); player.player.setPosition(3000)
    wait(lambda: player.canvas.overlay.position_ms >= 3000 and not player.canvas.image.isNull())
    player.player.pause()
    assert player.canvas.image.size().width() == 640
    assert player.canvas.overlay.status == 'sampled'
    assert player.canvas.overlay.frames
    before = player.canvas.overlay.enabled; player.setFocus()
    QTest.keyClick(player, Qt.Key.Key_D); assert player.canvas.overlay.enabled != before
    player.offset.setValue(250); assert player.canvas.overlay.offset_ms == 250
    player.player.setPosition(3000); player.step(1)
    assert player.player.position() == 3083
    screen.navigate_event(1); wait(lambda: view.recording is not None and view.recording.id == 102)
    assert 'AI call failed — no answer' in [w.text() for w in view.record.findChildren(QLabel)]
    screen.navigate_event(1); wait(lambda: view.recording is not None and view.recording.id == 103)
    assert 'Fallback: no model ran' in [w.text() for w in view.record.findChildren(QLabel)]


def test_offline_timeline_retry_state(widgets, wait):
    class Offline(DemoBackend):
        def events(self, **filters):
            raise OfflineError()
    screen = TimelineScreen(Offline()); widgets.append(screen); screen.show(); screen.open(1)
    wait(lambda: not screen.runner.busy)
    assert screen.stack.currentWidget() is screen.failure
    assert "Can't reach Home Guard Cloud" in screen.banner.text()


def test_http_mutations_access_detections_refresh_and_no_bearer_to_media():
    backend_demo = DemoBackend(); calls = []
    token = dict(access_token='old', refresh_token='refresh', expires_in=900, staff=asdict(backend_demo.me()))
    def wire(value):
        return json.loads(json.dumps(asdict(value), default=lambda d: d.isoformat()))
    def handler(request):
        calls.append(request)
        path = request.url.path
        if request.url.host == 'media.example':
            assert 'authorization' not in request.headers
            return httpx.Response(200, content=b'image')
        if path.endswith('/refresh'):
            return httpx.Response(200, json=dict(token, access_token='fresh'))
        if request.headers.get('authorization') == 'Bearer old':
            return httpx.Response(401)
        if path.endswith('/review'):
            assert request.method == 'PATCH' and json.loads(request.content) == {'flagged': True}
            return httpx.Response(200, json=wire(backend_demo.review(101, flagged=True)))
        if path.endswith('/detections'):
            return httpx.Response(200, json=wire(backend_demo.detections(101)))
        if path.endswith('/access'):
            assert request.method == 'POST' and json.loads(request.content) == {'purpose': 'training'}
            return httpx.Response(200, json=dict(url='https://media.example/clip.mp4?signature=demo', expires_utc=backend_demo.now.isoformat(), mime='video/mp4'))
        return httpx.Response(307, headers={'location': 'https://media.example/thumb.jpg'})
    backend = HttpBackend('https://cloud.example', transport=httpx.MockTransport(handler)); backend.tokens = decode(TokenPair, token)
    assert backend.review(101, flagged=True).flagged
    assert backend.detections(101).frames
    assert backend.artifact_access(1010, 'training').mime == 'video/mp4'
    assert backend.media_bytes('/v1/events/101/thumbnail') == b'image'
    assert len([r for r in calls if r.url.path.endswith('/refresh')]) == 1
    backend.close()


def test_compact_window_and_review_badge(widgets, wait):
    backend = DemoBackend(); shell = Shell(backend, backend.me()); widgets.append(shell)
    shell.resize(1366, 768); shell.show(); shell.open_customer(1)
    wait(lambda: bool(shell.customer_page.timeline.model.rows) and not shell.badge_runner.busy)
    assert (shell.width(), shell.height()) == (1366, 768)
    expected = all_events(backend, reviewed=False, from_utc=(backend.now-timedelta(hours=24)).isoformat(), to_utc=backend.now.isoformat())
    assert shell.review_badge.text() == str(len(expected))


def test_expired_recording_retains_ai_record(widgets, wait):
    backend = DemoBackend(); screen = CustomerScreen(backend); widgets.append(screen)
    screen.show(); screen.open(1); wait(lambda: bool(screen.timeline.model.rows))
    screen.open_event(121); view = screen.event_view
    wait(lambda: view.recording is not None and not view.evidence_runner.busy)
    assert view.recording.completeness.expired
    assert not view.player.play.isEnabled()
    assert view.player.error.text() == 'No copy of this video remains.'
    assert 'Fallback: no model ran' in [w.text() for w in view.record.findChildren(QLabel)]


def test_filmstrip_hover_uses_sprite_metadata(widgets, wait):
    from PySide6.QtCore import QBuffer, QIODevice, QPoint
    from PySide6.QtGui import QColor
    from home_guard_project.admin.player import FilmstripSlider
    strip = FilmstripSlider(); widgets.append(strip); strip.resize(200, 30); strip.setMaximum(2000)
    image = QImage(20, 10, QImage.Format.Format_RGB32); image.fill(QColor('red'))
    for x in range(10, 20):
        for y in range(10):
            image.setPixelColor(x, y, QColor('blue'))
    data = QBuffer(); data.open(QIODevice.OpenModeFlag.WriteOnly); image.save(data, 'PNG')
    strip.set_filmstrip(bytes(data.data()), dict(tile_w=10, tile_h=10, count=2, fps=1))
    strip.show(); QTest.mouseMove(strip, QPoint(150, 15))
    wait(lambda: strip.preview.isVisible())
    assert strip.preview.pixmap().toImage().pixelColor(5, 5) == QColor('blue')
    strip.hide(); assert not strip.preview.isVisible()
