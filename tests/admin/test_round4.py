import json
import ssl
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest

from home_guard_project.admin import http_backend, models
from home_guard_project.admin.backend import AuthError
from home_guard_project.admin.demo_backend import DemoBackend
from test_backends import TOKEN, DATA


def client(handler):
    backend = http_backend.HttpBackend('https://cloud.example', transport=httpx.MockTransport(handler))
    backend.tokens = models.decode(models.TokenPair, TOKEN)
    return backend


def test_c1_windows_store_and_verification(monkeypatch):
    calls = []
    original = ssl.SSLContext.load_default_certs
    def load(ctx, *args):
        calls.append(ctx)
        return original(ctx, *args)
    monkeypatch.setattr(ssl.SSLContext, 'load_default_certs', load)
    monkeypatch.setattr(sys, 'platform', 'win32')
    ctx = http_backend.tls_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert calls == [ctx]
    system = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    original(system)
    assert set(system.get_ca_certs(binary_form=True)) <= set(ctx.get_ca_certs(binary_form=True))


def test_c1_invalid_environment_cert_is_designed(monkeypatch):
    from home_guard_project.admin.backend import TlsError
    monkeypatch.setenv('SSL_CERT_FILE', 'C:/missing-homeguard-cert.pem')
    with pytest.raises(TlsError):
        http_backend.HttpBackend('https://cloud.example')


@pytest.mark.parametrize('failure', ['transport', 'server'])
def test_i1_refresh_is_burned_after_ambiguous_failure(failure):
    calls = []
    def handler(request):
        if request.url.path.endswith('/refresh'):
            calls.append(json.loads(request.content)['refresh_token'])
            if failure == 'transport':
                raise httpx.ReadTimeout('lost response', request=request)
            return httpx.Response(503)
        return httpx.Response(401)
    backend = client(handler)
    for _ in range(2):
        with pytest.raises(AuthError, match='Your session needs a new sign-in'):
            backend.me()
    assert calls == ['refresh-1']
    assert backend.tokens is None


def test_m9_concurrent_401s_single_flight():
    barrier, refreshes = threading.Barrier(8), []
    def handler(request):
        if request.url.path.endswith('/refresh'):
            refreshes.append(request)
            return httpx.Response(200, json=dict(TOKEN, access_token='access-2', refresh_token='refresh-2'))
        if request.headers['Authorization'] == 'Bearer access-1':
            barrier.wait(5)
            return httpx.Response(401)
        return httpx.Response(200, json=TOKEN['staff'])
    backend = client(handler)
    with ThreadPoolExecutor(8) as pool:
        assert all(s.role == 'admin' for s in pool.map(lambda _: backend.me(), range(8)))
    assert len(refreshes) == 1


@pytest.mark.parametrize('url', ['http://cloud.example', 'http://127.0.0.2', 'ftp://localhost'])
def test_i2_refuse_insecure_server(url):
    from home_guard_project.admin.backend import ConfigurationError
    with pytest.raises(ConfigurationError):
        http_backend.HttpBackend(url)


def test_i8_unknown_enums_keep_page():
    wire = json.loads((DATA/'events.json').read_text())
    wire['items'][0]['kind'] = 'future_kind'
    wire['items'][0]['completeness']['ai'] = 'future_ai'
    page = models.decode(models.EventPage, wire)
    assert page.items[0].kind == page.items[0].completeness.ai == 'unknown'
    assert len(page.items) == len(wire['items'])


def test_i7_token_repr_and_redaction():
    import logging
    from home_guard_project.admin.logging_setup import RedactionFilter
    assert 'access-1' not in repr(models.decode(models.TokenPair, TOKEN))
    record = logging.LogRecord('test', 40, __file__, 1,
        'password=secret totp=123456 refresh_token=refresh-1 Authorization: Bearer access-1 https://s3/a?X-Amz-Signature=private', (), None)
    RedactionFilter().filter(record)
    for secret in ('secret', '123456', 'refresh-1', 'access-1', 'private'):
        assert secret not in record.getMessage()


def test_m4_overlay_outside_coverage():
    from home_guard_project.admin.event_logic import nearest_frame
    frames = [models.FrameBoxes(i, float(i), 'ran_empty', []) for i in range(3)]
    assert nearest_frame(frames, 4000) is None
    assert nearest_frame(frames, -2000) is None
    assert nearest_frame(frames, 2100) == frames[-1]


def test_contract_2c_http_routes():
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith('/preview'):
            return httpx.Response(200, json=dict(included_ids=[101], excluded=[], split_counts={'train': 1}, groups=1, warnings=[]))
        return httpx.Response(200, json=dict(items=[], next_cursor=None, total=10000, total_capped=True))
    backend = client(handler)
    assert backend.collection_events(1).items == []
    assert backend.export_preview(collection_id=1).included_ids == [101]
    assert backend.events(with_total=True).total_capped
    assert paths[:2] == ['/v1/studio/collections/1/items', '/v1/studio/exports/preview']


def test_i3_page_landing_uses_event_id(app, widgets):
    from home_guard_project.admin.review import ReviewScreen
    backend = DemoBackend()
    screen = ReviewScreen(backend); widgets.append(screen)
    rows = backend.events(limit=5).items
    screen.timeline.model.set_page(rows[:2], False, {})
    screen.timeline.table.setCurrentIndex(screen.timeline.model.index(1, 0))
    screen.timeline.cursor = 'next'
    screen.timeline.load_older = lambda: None
    screen.move(1)
    screen.timeline.model.remove_event(rows[1].id)
    page = models.EventPage(rows[2:], None)
    screen.timeline.model.set_page(page.items, True, {})
    screen.page_loaded(page, None)
    assert screen.active_id == rows[2].id


def test_i4_fast_review_actions_are_not_lost(app, wait, widgets):
    from home_guard_project.admin.review import ReviewScreen
    backend = DemoBackend()
    gate = threading.Event()
    calls = []
    review = backend.review
    def save(eid, **changes):
        gate.wait(2); calls.append(eid)
        return review(eid, **changes)
    backend.review = save
    screen = ReviewScreen(backend); widgets.append(screen)
    rows = backend.events(limit=5).items
    expected = [e.id for e in rows[:3]]
    screen.timeline.model.set_page(rows, False, {})
    screen.timeline.table.setCurrentIndex(screen.timeline.model.index(0, 0))
    try:
        for _ in range(3): screen.mutate('reviewed')
    finally:
        gate.set()
    wait(lambda: not screen.mutation.busy)
    assert calls == expected


def test_i5_stale_evidence_does_not_request_media(app, wait, widgets):
    from home_guard_project.admin.event_view import EventView
    backend = DemoBackend()
    entered, release = threading.Event(), threading.Event()
    accesses = []
    detections, access = backend.detections, backend.artifact_access
    def blocked(eid):
        entered.set(); release.wait(2)
        return detections(eid)
    backend.detections = blocked
    backend.artifact_access = lambda aid, purpose: (accesses.append(aid), access(aid, purpose))[1]
    view = EventView(backend); widgets.append(view); view.show(); view.open(101)
    wait(entered.is_set)
    view.open(102)
    release.set()
    wait(lambda: not view.evidence_runner.busy and view.recording and view.recording.id == 102)
    old_ids = {a.id for a in backend.event(101).artifacts}
    assert not old_ids.intersection(accesses)


def test_i6_shutdown_has_two_second_bound(monkeypatch):
    from home_guard_project.admin import workers
    calls = []
    class Pool:
        def clear(self): calls.append('clear')
        def waitForDone(self, timeout): calls.append(timeout); return False
    monkeypatch.setattr(workers.QThreadPool, 'globalInstance', lambda: Pool())
    try:
        assert workers.shutdown_workers() is False
        assert calls == ['clear', 2000]
        assert workers.closing.is_set()
    finally:
        workers.closing.clear()


def test_m2_fleet_hidden_and_forbidden_stops_poll(app, widgets):
    from home_guard_project.admin.fleet import FleetScreen
    from home_guard_project.admin.backend import ForbiddenError
    view = FleetScreen(DemoBackend()); widgets.append(view)
    view.show(); view.hide()
    assert not view.timer.isActive()
    view.show(); view.show_error(ForbiddenError())
    assert not view.timer.isActive()


def test_m5_dialog_and_tooltip_ownership(app, widgets):
    from PySide6.QtCore import Qt
    from home_guard_project.admin.collections import CreateCollection, CollectionPicker
    from home_guard_project.admin.export_wizard import ExportWizard
    from home_guard_project.admin.player import FilmstripSlider
    backend = DemoBackend()
    dialogs = [CreateCollection(backend), CollectionPicker(backend, 101), ExportWizard(backend, backend.collections())]
    for dialog in dialogs:
        widgets.append(dialog)
        assert dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    slider = FilmstripSlider(); widgets.append(slider)
    assert slider.preview.parent() is slider


def test_m3_badge_refresh_retained_while_busy(app, wait, widgets):
    from home_guard_project.admin.shell import Shell
    backend = DemoBackend()
    gate = threading.Event()
    calls = []
    original = backend.review_count
    def count():
        calls.append(1); gate.wait(2); return original()
    backend.review_count = count
    shell = Shell(backend, backend.me(), 'DEMO', 'dark'); widgets.append(shell)
    try:
        wait(lambda: shell.badge_runner.busy)
        shell.update_badge()
        before = len(calls)
    finally:
        gate.set()
    wait(lambda: not shell.badge_runner.busy)
    assert len(calls) > before


def test_m7_old_session_cannot_retry_as_new_user():
    entered, release = threading.Event(), threading.Event()
    requests = []
    def handler(request):
        requests.append(request.headers.get('Authorization'))
        entered.set(); release.wait(2)
        return httpx.Response(401)
    backend = client(handler)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(backend.me)
        assert entered.wait(2)
        backend.clear_session()
        backend.tokens = models.decode(models.TokenPair, dict(TOKEN, access_token='other-user'))
        release.set()
        with pytest.raises(AuthError): future.result()
    assert requests == ['Bearer access-1']
    assert backend.tokens.access_token == 'other-user'


def test_m1_collection_dialog_expires_session(app, widgets):
    from home_guard_project.admin.collections import CreateCollection
    dialog = CreateCollection(DemoBackend()); widgets.append(dialog)
    expired = []
    dialog.session_expired.connect(lambda: expired.append(True))
    dialog.completed(None, AuthError())
    assert expired


def test_contract_preview_uses_protocol_and_enables_create(app, wait, widgets):
    from home_guard_project.admin.export_wizard import ExportWizard
    class Backend:
        def export_preview(self, **request):
            return models.ExportPreview([101, 102], [models.ExportExclusion(103, 'expired')], {'train': 2, 'val': 0, 'test': 0}, 1, ['Small dataset'])
    wizard = ExportWizard(Backend(), DemoBackend().collections()); widgets.append(wizard)
    wizard.name.setText('test_dataset'); wizard.set_step(2)
    wait(lambda: not wizard.runner.busy)
    wizard.check.setChecked(True)
    assert wizard.next.isEnabled()
    assert 'Small dataset' in wizard.summary.text()
    assert 'train: 2' in wizard.summary.text()


def test_thumbnails_only_request_visible_rows(app, wait, widgets):
    from home_guard_project.admin.timeline import TimelineScreen
    backend = DemoBackend()
    accesses = []
    backend.media_bytes = lambda url: accesses.append(url) or b''
    screen = TimelineScreen(backend); widgets.append(screen)
    screen.resize(1200, 420); screen.show()
    rows = backend.events(limit=25).items
    # Unique URLs distinguish offscreen rows even when fixtures reuse clips.
    rows = [replace(e, thumbnail_url=f'row-{e.id}') for e in rows]
    screen.pending = screen.generation, False
    screen.completed(models.EventPage(rows, None), None)
    wait(lambda: bool(accesses))
    visible = {e.thumbnail_url for i, e in enumerate(rows) if screen.table.visualRect(screen.model.index(i, 0)).intersects(screen.table.viewport().rect())}
    assert set(accesses) <= visible
    assert len(accesses) < len(rows)


def test_log_redacts_quoted_secrets_and_arbitrary_exception_messages(tmp_path):
    import logging
    from home_guard_project.admin.logging_setup import RedactionFilter
    try:
        raise RuntimeError('arbitrary-sensitive-value')
    except RuntimeError:
        record = logging.LogRecord('test', 40, __file__, 1, 'password="two secret words" code=123456', (), sys.exc_info())
    RedactionFilter().filter(record)
    assert 'secret words' not in record.getMessage()
    assert 'arbitrary-sensitive-value' not in record.getMessage()
    assert 'test_log_redacts' in record.getMessage()


def test_m6_toast_restarts_timer_and_repolishes(app, widgets):
    from home_guard_project.admin.review import ReviewScreen
    from PySide6.QtTest import QTest
    screen = ReviewScreen(DemoBackend()); widgets.append(screen)
    screen.notify('first')
    screen.toast_timer.setInterval(200)
    screen.toast_timer.start()
    QTest.qWait(100)
    screen.notify('new error', True)
    assert screen.toast.objectName() == 'error'
    assert screen.toast_timer.remainingTime() > 150


def test_unknown_saved_filter_count_is_not_a_one_plus_estimate(app, wait, widgets):
    from home_guard_project.admin.studio import StudioScreen
    class Backend(DemoBackend):
        def events(self, **query):
            assert query.get('with_total') is True
            return models.EventPage([], None, 10000, True)
    view = StudioScreen(Backend()); widgets.append(view)
    view.count_requested = 'saved'
    view.request_count()
    wait(lambda: not view.count_runner.busy)
    assert view.counts['saved'] == '10,000+'


def test_worker_exception_is_logged_and_designed(app, wait, tmp_path, monkeypatch):
    import logging
    from home_guard_project.admin.workers import TaskRunner
    logger = logging.getLogger('homeguard.admin')
    for handler in logger.handlers[:]:
        handler.close(); logger.removeHandler(handler)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    runner, results = TaskRunner(), []
    runner.finished.connect(lambda result, error: results.append(error))
    def broken(): raise KeyError('do-not-log-arbitrary-payload')
    runner.start(broken)
    wait(lambda: bool(results))
    text = (tmp_path/'HomeGuardAdmin/logs/admin.log').read_text()
    assert 'KeyError' in text and 'broken' in text
    assert 'do-not-log-arbitrary-payload' not in text
    assert 'local admin log' in str(results[0])
    for handler in logger.handlers[:]:
        handler.close(); logger.removeHandler(handler)


def test_media_error_requests_access_only_once(app, wait, widgets):
    from home_guard_project.admin.event_view import EventView
    backend = DemoBackend()
    calls = []
    access = backend.artifact_access
    backend.artifact_access = lambda aid, purpose: (calls.append(aid), access(aid, purpose))[1]
    view = EventView(backend); widgets.append(view); view.show()
    view.recording = backend.event(101)
    view.refresh_media(); view.refresh_media()
    wait(lambda: not view.media_runner.busy)
    assert calls == [1010]


def test_application_shutdown_with_stalled_worker_is_bounded(tmp_path):
    import subprocess
    script = '''
import sys, threading, time
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from home_guard_project.admin.shell import AdminWindow
from home_guard_project.admin.workers import TaskRunner
from home_guard_project.admin.__main__ import main
show = AdminWindow.show
def stalled(self):
    show(self)
    self.stall = TaskRunner(self)
    self.stall.start(lambda: threading.Event().wait(30))
    def quit():
        Path(marker).write_text(str(time.monotonic()))
        QApplication.instance().quit()
    QTimer.singleShot(100, quit)
AdminWindow.show = stalled
marker = sys.argv[1]
sys.argv = [sys.argv[0], '--demo']
main()
'''
    marker = tmp_path/'quit-time'
    result = subprocess.run([sys.executable, '-c', script, str(marker)], timeout=8, capture_output=True)
    import time
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert time.monotonic()-float(marker.read_text()) < 2.8


def test_audit_detail_survives_wire_decode_and_drawer(app, widgets):
    from home_guard_project.admin.audit import AuditScreen
    wire = json.loads((DATA/'audit.json').read_text())['items'][0]
    wire['detail'] = {'event_id': 101, 'purpose': 'review'}
    entry = models.decode(models.AuditEntry, wire)
    screen = AuditScreen(DemoBackend()); widgets.append(screen)
    screen.model.replace([entry])
    screen.select(screen.model.index(0, 0), None)
    assert json.loads(screen.json.toPlainText())['detail'] == wire['detail']


def test_thumbnail_501_is_typed_unsupported():
    from home_guard_project.admin.backend import UnsupportedError
    backend = client(lambda request: httpx.Response(501))
    with pytest.raises(UnsupportedError):
        backend.media_bytes('/v1/events/101/thumbnail')


def test_hidden_event_view_never_requests_evidence(app, widgets):
    from PySide6.QtTest import QTest
    from home_guard_project.admin.event_view import EventView
    backend = DemoBackend()
    calls = []
    original = backend.event
    backend.event = lambda eid: calls.append(eid) or original(eid)
    view = EventView(backend); widgets.append(view)
    view.open(101)
    QTest.qWait(220)
    assert calls == []


def test_stale_detail_completion_does_not_bypass_open_debounce(app, widgets):
    from home_guard_project.admin.event_view import EventView
    backend = DemoBackend()
    view = EventView(backend); widgets.append(view); view.show()
    view.open(102)
    view.pending = 0
    view.loaded(backend.event(101), None)
    assert not view.runner.busy
