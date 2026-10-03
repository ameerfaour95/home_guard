import httpx
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.prefs import Preferences
from home_guard_project.admin.shell import AdminWindow
from home_guard_project.admin.demo_backend import DemoBackend

TOKEN = dict(access_token='a1', refresh_token='r1', expires_in=900,
             staff=dict(id=1, email='local@example.com', name='Local', role='admin'))


def make(widgets, tmp_path, handler, local=True):
    backend = HttpBackend('http://127.0.0.1:8610', transport=httpx.MockTransport(handler))
    window = AdminWindow(backend, prefs=Preferences(tmp_path/'prefs.json'), local=local)
    widgets.append(window); window.show()
    return window


def test_local_success_skips_signin(widgets, wait, tmp_path):
    seen = []
    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=TOKEN) if request.url.path == '/v1/auth/local' else httpx.Response(500)
    window = make(widgets, tmp_path, handler)
    wait(lambda: window.shell is not None)
    assert window.session.currentWidget() is window.shell
    assert seen[0] == '/v1/auth/local'
    assert window.backend.tokens.access_token == 'a1'


def test_local_404_shows_plain_signin(widgets, wait, tmp_path):
    window = make(widgets, tmp_path, lambda request: httpx.Response(404))
    wait(lambda: not window.local_runner.busy and window.connecting is None)
    assert window.shell is None
    assert window.session.currentWidget() is window.signin
    assert window.signin.error.text() == ''


def test_not_local_never_calls_local_endpoint(widgets, tmp_path):
    calls = []
    window = make(widgets, tmp_path, lambda r: calls.append(r) or httpx.Response(404), local=False)
    assert calls == [] and window.session.currentWidget() is window.signin


def test_refresh_failure_in_local_mode_relogs_in_silently(widgets, wait, tmp_path):
    window = make(widgets, tmp_path, lambda r: httpx.Response(200, json=TOKEN))
    wait(lambda: window.shell is not None)
    first = window.shell
    window.expired()
    wait(lambda: window.shell is not None and window.shell is not first and not window.local_runner.busy)
    assert window.session.currentWidget() is window.shell
    assert window.signin.error.text() == ''


def test_relogin_failure_falls_back_to_signin(widgets, wait, tmp_path):
    state = {'ok': True}
    def handler(request):
        return httpx.Response(200, json=TOKEN) if state['ok'] else httpx.Response(404)
    window = make(widgets, tmp_path, handler)
    wait(lambda: window.shell is not None)
    state['ok'] = False
    window.expired()
    wait(lambda: window.shell is None)
    assert window.session.currentWidget() is window.signin
