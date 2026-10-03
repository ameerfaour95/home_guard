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
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-fleet.png'))
    shell.open_customer(1)
    customer = shell.customer_page
    wait(lambda: bool(customer.timeline.model.rows) and bool(customer.timeline.density.hours))
    wait(lambda: any(customer.timeline.thumbnail_cache.values()))
    assert 'Daniel' in customer.name.text()
    density_row = next(r for r in customer.timeline.density.rows if '/' in r)
    customer.timeline.filters['camera'].setCurrentIndex(customer.timeline.filters['camera'].findData(density_row))
    wait(lambda:not customer.timeline.runner.busy)
    assert customer.timeline.model.rows and all(e.camera == density_row.split('/',1)[1] for e in customer.timeline.model.rows)
    customer.timeline.clear_filters(); wait(lambda:not customer.timeline.runner.busy)
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-timeline.png'))
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
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-event.png'))
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
    requests = []
    backend.client.event_hooks['request'] = [lambda request: requests.append(request)]
    window = AdminWindow(backend, prefs=Preferences(tmp_path/'labeler.json')); widgets.append(window); window.show()
    window.resize(1920,1080)
    sign_in(window, credentials['labeler'])
    wait(lambda: window.shell is not None, timeout=15)
    assert set(window.shell.navigation) == {'Review', 'Studio'}
    events = backend.events().items
    assert events and all(e.customer_name.startswith('customer-') and e.camera.startswith('cam-') for e in events)
    assert 'Daniel' not in repr(events)
    assert all(e.customer_id == 0 for e in events)
    shell = window.shell
    assert shell.search.isHidden()
    shell.navigate('Review')
    review = shell.review_page
    wait(lambda: bool(review.timeline.model.rows), timeout=20)
    assert review.timeline.search.isHidden()
    review.clear_filters()
    wait(lambda: not review.timeline.runner.busy, timeout=20)
    hidden = next(e for e in events if e.thumbnail_url is None)
    review.open_event(hidden.id)
    wait(lambda: review.event_view.recording and review.event_view.recording.id == hidden.id, timeout=20)
    assert review.event_view.recording.camera in review.event_view.title.text()
    # A camera command must also never put the sentinel ID on the wire.
    shell.execute_command('camera', (0, hidden.camera))
    wait(lambda: not review.timeline.runner.busy, timeout=20)
    assert review.timeline.customer_id is None
    backend.events(customer_id=0, q='should not be sent')
    assert not any('customer_id' in r.url.params or r.url.params.get('q') for r in requests)
    review.clear_filters(); wait(lambda: not review.timeline.runner.busy, timeout=20)
    review.open_event(hidden.id)
    wait(lambda: review.event_view.recording and review.event_view.recording.id == hidden.id and not review.event_view.evidence_runner.busy, timeout=20)
    # Scroll the null thumbnail into view, so the unavailable tile is in the evidence screenshot.
    row = next(i for i,e in enumerate(review.timeline.model.rows) if e.id == hidden.id)
    review.timeline.table.scrollTo(review.timeline.model.index(row,0))
    wait(lambda: not review.event_view.player.canvas.image.isNull() and not review.event_view.asset_runner.busy, timeout=20)
    review.event_view.player.player.pause()
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-labeler-review.png'))
    shell.open_palette()
    assert any(e[2].startswith('filter:') for e in shell.palette_dialog.entries)
    assert not any(e[2].startswith('camera:') for e in shell.palette_dialog.entries)
    shell.palette_dialog.close()


def test_live_studio_exact_filters_collection_export_and_manifest(live, app):
    window, backend, wait = live
    shell = window.shell; shell.navigate('Studio'); studio = shell.screens['Studio']
    wait(lambda: studio.loaded_once)
    assert studio.content_stack.currentWidget() is studio.tabs
    sources = json.loads((ROOT/'home_guard_project/admin/demo_data/events.json').read_text())['items']
    expected = {
        'false_alarm': sum('false_alarm' in e['owner_verdicts'] for e in sources),
        'real_but_wrong': sum('real_but_wrong' in e['owner_verdicts'] for e in sources),
        'ai_failed': sum(e['completeness']['ai'] in ('failed','fallback') for e in sources),
        'ai_dismissed_person': sum(e['kind']=='false_positive' and 'person' in e['detected'] for e in sources),
        'paused': sum(e['kind']=='paused' for e in sources),
        'low_conf': sum(e['id'] % 5 == 0 for e in sources),
    }
    for row, saved in enumerate(studio.filters):
        studio.filter_table.setCurrentIndex(studio.filter_model.index(row,0))
        wait(lambda: saved.key in studio.counts)
        assert studio.counts[saved.key] == str(expected[saved.key])
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-studio.png'))
    studio.tabs.setCurrentIndex(1)
    row = next(i for i,c in enumerate(studio.collections) if c.id == 4)
    studio.collection_table.setCurrentIndex(studio.collection_model.index(row,0)); studio.open_collection()
    wait(lambda: bool(studio.grid.model.items))
    assert {e.id for e in studio.grid.model.items} == {101,102,104}
    studio.open_export(); wizard = studio.wizard
    wizard.name.setText('live_training'); wizard.formats['yolo'].setChecked(False); wizard.formats['clips'].setChecked(False)
    wizard.advance(); wizard.advance()
    wait(lambda: wizard.preview is not None)
    assert wizard.preview.included_ids and any('clip' in w.lower() for w in wizard.preview.warnings)
    included_count = len(wizard.preview.included_ids)
    created = []; wizard.exported.connect(created.append)
    wizard.check.setChecked(True); wizard.advance()
    wait(lambda: bool(created))
    export_id = created[0].id
    wait(lambda: not studio.runner.busy)
    # UI uses the real export history poll until the background builder finishes.
    def ready():
        current = next((e for e in studio.exports if e.id == export_id), None)
        if current and current.state == 'ready': return True
        if not studio.runner.busy: studio.refresh()
        return False
    wait(ready, timeout=40)
    row = next(i for i,e in enumerate(studio.exports) if e.id == export_id)
    studio.export_table.setCurrentIndex(studio.export_model.index(row,0)); studio.open_manifest()
    wait(lambda: studio.manifest_data is not None)
    manifest = studio.manifest_data
    assert manifest['schema_version'] == 2 and sum(manifest['counts']['clips'].values()) == included_count
    assert 'vlm' in manifest['counts'] and manifest['warnings']
    assert f'manage export-download {export_id} --dest DIR' in studio.export_detail.text()
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-export-ready.png'))


def test_live_audit_drawer_and_index_problems(live, app):
    window, backend, wait = live
    shell = window.shell; shell.navigate('Audit'); audit = shell.screens['Audit']
    wait(lambda: audit.loaded_once)
    assert audit.stack.currentWidget() is audit.content
    audit.filters['action'].setText('collection_create'); audit.reload()
    wait(lambda: not audit.runner.busy and bool(audit.model.items))
    audit.table.setCurrentIndex(audit.model.index(0,0))
    assert audit.drawer.isVisible() and json.loads(audit.json.toPlainText())['detail']
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-audit.png'))
    shell.navigate('Index problems'); problems = shell.screens['Index problems']
    wait(lambda: bool(problems.model.items))
    assert len(problems.model.items) == 2
    app.processEvents(); assert window.grab().save(str(SHOTS/'r5-live-index-problems.png'))


def test_live_name_refusal_and_consent_thumbnail(live):
    from dataclasses import replace
    window, backend, wait = live
    shell = window.shell; shell.navigate('Studio'); studio = shell.screens['Studio']
    studio.new_collection(); dialog = studio.create_dialog
    dialog.name.setText('Daniel Levi collection'); dialog.create()
    wait(lambda:not dialog.runner.busy)
    assert dialog.error.text() == 'Name must not identify a household'
    dialog.reject()
    original = backend.customer(1)
    try:
        changed = backend.update_customer(replace(original,consent_recordings=False))
        assert changed.name == original.name and changed.consent_training == original.consent_training
        page = backend.events(customer_id=1)
        assert page.items and all(e.thumbnail_url is None for e in page.items)
    finally:
        backend.update_customer(original)


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
