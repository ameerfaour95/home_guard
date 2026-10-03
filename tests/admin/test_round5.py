import json
from dataclasses import replace
import httpx
import pytest
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.models import DispatchOut
from home_guard_project.admin.formatting import camera_name, delivery_text
from home_guard_project.admin.settings import valid_server, SettingsDialog
from home_guard_project.admin.prefs import Preferences


def test_delivery_and_camera_names():
    assert camera_name('front_door') == 'Front door'
    assert camera_name('front_door', 'Main entrance') == 'Main entrance'
    assert camera_name('cam-abcdef', 'Private name') == 'cam-abcdef'
    assert delivery_text(None) == 'Delivery not recorded'
    assert delivery_text(DispatchOut('telegram', None, {})) == 'Delivery not recorded'
    assert delivery_text(DispatchOut('telegram', True, {})) == 'Sent to the owner on Telegram'
    assert delivery_text(DispatchOut('telegram', False, {'reason':'owner_paused'})) == 'Not delivered (Owner paused)'


@pytest.mark.parametrize('url,valid', [('https://cloud.example.com',True), ('http://127.0.0.1:8600',True),
    ('http://cloud.example.com',False), ('https://user:secret@cloud.example.com',False),
    ('https://cloud.example.com?token=abc',False), ('https://cloud.example.com:bad',False), ('',False)])
def test_server_validation(url,valid):
    assert valid_server(url) is valid


def test_settings_persist_email_and_apply_theme(app, widgets, tmp_path, wait):
    from home_guard_project.admin.shell import AdminWindow
    prefs = Preferences(tmp_path/'prefs.json'); prefs.save_email('staff@example.com')
    window = AdminWindow(DemoBackend(), demo=True, prefs=prefs); widgets.append(window); window.show()
    wait(lambda: window.shell.fleet.snapshot is not None)
    window.open_settings(); dialog = window.settings_dialog
    dialog.server.setText('http://invalid.example.com'); dialog.save()
    assert dialog.error.text() and dialog.isVisible()
    dialog.server.setText('http://127.0.0.1:8000'); dialog.theme.setCurrentIndex(1); dialog.save()
    assert window.theme == 'light' and window.shell is not None
    assert prefs.email() == 'staff@example.com' and prefs.get('theme') == 'light'
    window.save_settings('https://cloud.example.com','dark')
    assert window.shell is None and window.configured_backend.base_url == 'https://cloud.example.com'
    assert prefs.get('server') == 'https://cloud.example.com'


def test_name_validation_and_inline_household_error(widgets, wait):
    from home_guard_project.admin.collections import CreateCollection
    from home_guard_project.admin.export_wizard import ExportWizard
    from home_guard_project.admin.backend import ValidationError
    class Backend(DemoBackend):
        def create_collection(self, name, description=''):
            raise ValidationError('Name must not identify a household')
    backend = Backend(); dialog = CreateCollection(backend); widgets.append(dialog)
    dialog.name.setText('a'*121); dialog.create()
    assert '120' in dialog.error.text() and not dialog.runner.busy
    dialog.name.setText('household name'); dialog.create(); wait(lambda:not dialog.runner.busy)
    assert dialog.error.text() == 'Name must not identify a household'
    wizard = ExportWizard(backend,backend.collections()); widgets.append(wizard)
    wizard.name.setText('a'*121); wizard.advance()
    assert wizard.step == 0 and '120' in wizard.error.text()


def test_http_labeler_filters_full_customer_body_and_errors():
    from home_guard_project.admin.backend import ValidationError
    captured = []
    demo = DemoBackend()
    def handler(request):
        captured.append(request)
        if request.method == 'PATCH':
            from dataclasses import asdict
            return httpx.Response(200,json=json.loads(json.dumps(asdict(demo.customer(1)),default=str)))
        return httpx.Response(400,json={'detail':'Name must not identify a household'})
    backend = HttpBackend('https://cloud.example.com',transport=httpx.MockTransport(handler))
    backend.tokens = DemoBackend(role='labeler').login('','','')
    with pytest.raises(ValidationError,match='Name must not identify a household'):
        backend.events(customer_id=0,q='private')
    assert not captured[-1].url.params
    customer = replace(demo.customer(1),notes='Updated notes')
    backend.update_customer(customer)
    body = json.loads(captured[-1].content)
    assert set(body) == {'name','notes','timezone','consent_live','consent_recordings','consent_training'}
    assert body['name'] == customer.name and body['notes'] == 'Updated notes'
    backend.close()


def test_manifest_failure_stays_inline(widgets, wait):
    from home_guard_project.admin.studio import StudioScreen
    from home_guard_project.admin.backend import OfflineError
    class Backend(DemoBackend):
        def export(self,id): raise OfflineError()
    screen = StudioScreen(Backend()); widgets.append(screen); screen.show(); wait(lambda:screen.loaded_once)
    screen.export_model.replace([replace(screen.exports[0], manifest_url='https://example.com/manifest.json')])
    screen.export_table.setCurrentIndex(screen.export_model.index(0,0)); screen.open_manifest()
    wait(lambda:not screen.manifest_runner.busy)
    assert "Can't reach" in screen.manifest_detail.text()


def test_density_camera_scope_uses_site_and_raw_camera(widgets):
    from home_guard_project.admin.timeline import TimelineScreen
    screen = TimelineScreen(DemoBackend()); widgets.append(screen)
    screen.cell = ('cedar_house/front_door', screen.start)
    query = screen.query()
    assert query['site'] == 'cedar_house' and query['camera'] == 'front_door'
