"""Opt-in tests against the unmodified Cloud HTTP service on loopback.

Start admin.dev_server first, then HG_ADMIN_IT=1 uv run --group cloud --group
admin pytest tests/admin/test_live_server.py -v. No route overrides are used.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get('HG_ADMIN_IT') != '1', reason='Set HG_ADMIN_IT=1 and start admin.dev_server')]
ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT/'docs/admin/screenshots'


@pytest.fixture(scope='module')
def credentials():
    return json.loads((ROOT/'build/dev-login.json').read_text())


def sign_in(window, login):
    import pyotp
    window.signin.email.setText(login['email'])
    window.signin.password.setText(login['password'])
    for digit, value in zip(window.signin.totp.digits, pyotp.TOTP(login['totp_secret']).now()):
        digit.setText(value)
    window.signin.authenticate()


@pytest.fixture(scope='module')
def live(app, credentials, tmp_path_factory):
    import time
    from home_guard_project.admin.http_backend import HttpBackend
    from home_guard_project.admin.shell import AdminWindow
    from home_guard_project.admin.prefs import Preferences
    def wait(predicate, timeout=20):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate(): return
            time.sleep(.01)
        assert predicate(), 'Live UI did not reach the expected state'
    backend = HttpBackend('http://127.0.0.1:8600')
    window = AdminWindow(backend, prefs=Preferences(tmp_path_factory.mktemp('live')/'prefs.json'))
    window.resize(1920, 1080); window.show()
    sign_in(window, credentials['admin'])
    wait(lambda: window.shell is not None)
    yield window, backend, wait
    window.close()
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(2000)
    backend.close()
    window.deleteLater(); app.processEvents()


def test_live_fleet_customer_timeline_density_review_and_2c(live, app):
    from home_guard_project.admin.backend import UnsupportedError
    window, backend, wait = live
    shell = window.shell
    wait(lambda: shell.fleet.snapshot is not None and bool(shell.fleet.activity.hours))
    assert len(shell.fleet.snapshot.devices) == 4
    SHOTS.mkdir(parents=True, exist_ok=True)
    app.processEvents(); assert window.grab().save(str(SHOTS/'r4-live-fleet.png'))
    shell.open_customer(1)
    customer = shell.customer_page
    wait(lambda: bool(customer.timeline.model.rows) and bool(customer.timeline.density.hours))
    wait(lambda: any(customer.timeline.thumbnail_cache.values()))
    assert 'Daniel' in customer.name.text()
    app.processEvents(); assert window.grab().save(str(SHOTS/'r4-live-timeline.png'))
    customer.open_event(101)
    view = customer.event_view
    wait(lambda: view.recording is not None and not view.evidence_runner.busy and not view.asset_runner.busy)
    assert view.recording.ai_runs
    assert any(frame.boxes for frame in view.player.canvas.overlay.frames)
    if 'Not available yet' not in view.banner.text():
        view.player.player.play()
        view.player.player.setPosition(3000)
        wait(lambda: not view.player.canvas.image.isNull() and view.player.canvas.overlay.position_ms >= 3000)
        view.player.player.pause()
    app.processEvents(); assert window.grab().save(str(SHOTS/'r4-live-event.png'))
    before = view.recording.reviewed
    view.toggle_review('reviewed')
    wait(lambda: not view.review_runner.busy)
    assert backend.event(101).reviewed is not before
    backend.review(101, reviewed=before)
    assert backend.events(with_total=True, limit=1).total == 96
    assert backend.collection_events(1).items
    preview = backend.export_preview(collection_id=1, name='live_check', formats=['clips', 'vlm_jsonl'],
        split={'train': .8, 'val': .1, 'test': .1}, include_fallback_ai=False)
    assert preview.included_ids and sum(preview.split_counts.values()) == len(preview.included_ids)
    unavailable = []
    routes = [(backend.saved_filters, 'GET /studio/filters'), (backend.collections, 'GET /studio/collections'),
              (backend.exports, 'GET /studio/exports'), (backend.audit, 'GET /audit'),
              (lambda: backend.artifact_access(1010, 'review'), 'POST /artifacts/1010/access'),
              (lambda: backend.create_collection('live_probe'), 'POST /studio/collections'),
              (lambda: backend.add_collection_items(1, [101]), 'POST /studio/collections/1/items'),
              (lambda: backend.remove_collection_items(1, [101]), 'DELETE /studio/collections/1/items'),
              (lambda: backend.export(1), 'GET /studio/exports/1'),
              (lambda: backend.create_export(collection_id=1, name='live_probe', formats=['clips'], split={'train': .8, 'val': .1, 'test': .1}), 'POST /studio/exports'),
              (lambda: backend.media_bytes('/v1/events/101/thumbnail'), 'GET /events/101/thumbnail')]
    for call, name in routes:
        try: call()
        except UnsupportedError: unavailable.append(name)
    (ROOT/'build/live-route-results.json').write_text(json.dumps({'unavailable_501': unavailable}, indent=2))
    shell.navigate('Studio'); wait(lambda: not shell.screens['Studio'].runner.busy)
    assert shell.screens['Studio'].content_stack.currentWidget() is shell.screens['Studio'].unsupported
    shell.navigate('Audit'); wait(lambda: not shell.screens['Audit'].runner.busy)
    assert shell.screens['Audit'].stack.currentWidget() is shell.screens['Audit'].unsupported


def test_live_media_access_and_first_decoded_frame(live):
    from home_guard_project.admin.backend import UnsupportedError
    window, backend, wait = live
    window.shell.open_customer(1)
    customer = window.shell.customer_page
    wait(lambda: not customer.runner.busy and not customer.timeline.runner.busy and bool(customer.timeline.model.rows))
    customer.open_event(101)
    view = customer.event_view
    wait(lambda: view.recording is not None and not view.evidence_runner.busy)
    try:
        access = backend.artifact_access(1010, 'review')
    except UnsupportedError:
        assert 'Not available yet' in view.banner.text()
        pytest.xfail('Cloud POST /artifacts/{id}/access is still 501; no real API media URL exists')
    view.player.open_url(access.url)
    view.player.player.play()
    wait(lambda: not view.player.canvas.image.isNull())
    assert any(frame.boxes for frame in view.player.canvas.overlay.frames)


def test_live_labeler_signin_pseudonyms(app, credentials, widgets, wait, tmp_path):
    from home_guard_project.admin.http_backend import HttpBackend
    from home_guard_project.admin.shell import AdminWindow
    from home_guard_project.admin.prefs import Preferences
    backend = HttpBackend('http://127.0.0.1:8600')
    window = AdminWindow(backend, prefs=Preferences(tmp_path/'labeler.json')); widgets.append(window); window.show()
    sign_in(window, credentials['labeler'])
    wait(lambda: window.shell is not None, timeout=15)
    assert set(window.shell.navigation) == {'Review', 'Studio'}
    events = backend.events().items
    assert events and all(e.customer_name.startswith('customer-') and e.camera.startswith('cam-') for e in events)
    assert 'Daniel' not in repr(events)


def test_moto_synthetic_clip_decodes_independently(live, widgets):
    """Storage/decoder check only; deliberately does not claim API media integration."""
    import boto3
    from botocore.config import Config
    from home_guard_project.admin.player import EventPlayer
    _, backend, wait = live
    os.environ.pop('SSLKEYLOGFILE', None)
    s3 = boto3.client('s3', endpoint_url='http://127.0.0.1:8601', region_name='us-east-1',
        aws_access_key_id='local-dev', aws_secret_access_key='local-dev', config=Config(signature_version='s3v4'))
    url = s3.generate_presigned_url('get_object', Params={'Bucket': 'homeguard-admin-local', 'Key': 'admin_cache/dev/101/person.mp4'}, ExpiresIn=60)
    player = EventPlayer(); widgets.append(player); player.show()
    player.reset(backend.event(101), 'UTC')
    player.canvas.overlay.set_detections(backend.detections(101))
    player.open_url(url); player.player.play()
    wait(lambda: not player.canvas.image.isNull())
    assert any(frame.boxes for frame in player.canvas.overlay.frames)
    player.player.pause()


def test_built_exe_against_real_server(credentials):
    import pyotp
    import time
    executable = ROOT/'dist/HomeGuardAdmin/HomeGuardAdmin.exe'
    if not executable.exists(): pytest.skip('Build the exe before the packaged integration check')
    login = credentials['support']
    path = ROOT/'build/exe-live-login.json'
    path.unlink(missing_ok=True)
    result_path = ROOT/'build/exe-live-result.json'
    if result_path.exists(): result_path.unlink()
    env = dict(os.environ, HG_ADMIN_IT_LOGIN_FILE=str(path), HG_ADMIN_IT_RESULT=str(result_path))
    try:
        try:
            process = subprocess.Popen([str(executable), '--server', 'http://127.0.0.1:8600', '--smoke-test'], env=env)
        except OSError as error:
            if getattr(error, 'winerror', None) == 4551:
                pytest.xfail('Windows Application Control blocks this unsigned exe (WinError 4551)')
            raise
        deadline = time.monotonic()+40
        # Supply the one-use code only after this exact process is ready. A
        # cold start or another process inspecting the file cannot consume it.
        while time.monotonic() < deadline and process.poll() is None:
            try: ready = json.loads(result_path.read_text())
            except (OSError, ValueError): ready = {}
            if ready.get('ready_pid') == process.pid:
                path.write_text(json.dumps(dict(pid=process.pid, email=login['email'], password=login['password'], totp=pyotp.TOTP(login['totp_secret']).now())))
                break
            time.sleep(.05)
        assert process.wait(timeout=40) == 0, result_path.read_text() if result_path.exists() else 'No packaged probe result'
        result = json.loads(result_path.read_text())
        assert result['fleet'] == 4 and result['timeline'] > 0 and result['detections'] > 0
        assert result['frame_decoded'] and result['review_saved']
    finally:
        path.unlink(missing_ok=True)
