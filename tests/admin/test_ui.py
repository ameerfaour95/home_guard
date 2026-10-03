import json
import threading
from dataclasses import replace
from PySide6.QtCore import Qt, QThread
from PySide6.QtTest import QTest
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.backend import AuthError, OfflineError, RateLimitError, ServerError
from home_guard_project.admin.fleet import FleetScreen
from home_guard_project.admin.shell import Shell, AdminWindow
from home_guard_project.admin.signin import SignIn
from home_guard_project.admin.prefs import Preferences
from home_guard_project.admin.widgets.palette import CommandPalette
import pytest


def fleet_screen(widgets, wait, backend=None):
    screen = FleetScreen(backend or DemoBackend())
    widgets.append(screen)
    screen.resize(1182, 688)
    screen.show()
    wait(lambda: screen.snapshot is not None)
    return screen


def test_worst_first_filters_and_empty(widgets, wait):
    screen = fleet_screen(widgets, wait)
    assert [d.verdict for d in screen.model.rows] == ['offline', 'critical', 'warning', 'healthy']
    QTest.mouseClick(screen.chips['critical'], Qt.MouseButton.LeftButton)
    assert [d.verdict for d in screen.model.rows] == ['critical']
    screen.search.setText('does not exist')
    assert screen.stack.currentWidget() is screen.no_results
    screen.clear_filters()
    screen.search.setText('olive')
    assert screen.model.rows[0].site == 'olive_house'


def test_keyboard_selection_enter_and_slash(widgets, wait):
    screen = fleet_screen(widgets, wait)
    opened = []
    screen.customer_requested.connect(opened.append)
    screen.table.setFocus()
    screen.table.setCurrentIndex(screen.model.index(0, 0))
    QTest.keyClick(screen.table, Qt.Key.Key_Down)
    assert screen.selected_id == 'hg-pine-01'
    QTest.keyClick(screen.table, Qt.Key.Key_Return)
    assert opened == [3]
    QTest.keyClick(screen.table, Qt.Key.Key_Slash)
    assert screen.search.hasFocus()


def test_refresh_preserves_identity_scroll_and_filter(widgets, wait):
    screen = fleet_screen(widgets, wait)
    fleet = screen.snapshot
    many = [replace(fleet.devices[i % 4], device_id=f'device-{i:03}', customer_name=f'Household {i:03}') for i in range(80)]
    response = replace(fleet, devices=many)
    screen.completed((response, screen.customers), None)
    screen.filter(verdict='warning')
    screen.table.setCurrentIndex(screen.model.index(15, 0))
    selected = screen.selected_id
    screen.table.verticalScrollBar().setValue(10)
    scroll = screen.table.verticalScrollBar().value()
    screen.completed((replace(response, devices=list(reversed(many))), screen.customers), None)
    assert screen.selected_id == selected
    assert screen.table.verticalScrollBar().value() == scroll
    assert screen.chips['warning'].isChecked()
    assert screen.detail.device.device_id == selected


@pytest.mark.parametrize('error', [AuthError(), RateLimitError(), OfflineError()])
def test_signin_errors_clear_secrets_and_remember_email(error, widgets, wait, tmp_path):
    class FailingBackend(DemoBackend):
        def login(self, *args):
            raise error
    prefs = Preferences(tmp_path/'prefs.json')
    screen = SignIn(FailingBackend(), prefs)
    widgets.append(screen)
    screen.show()
    screen.email.setText('maya@example.com')
    screen.password.setText('secret')
    screen.totp.set_code('123456')
    QTest.mouseClick(screen.submit, Qt.MouseButton.LeftButton)
    wait(lambda: screen.submit.isEnabled())
    assert screen.error.text() == str(error)
    assert screen.password.text() == '' and screen.totp.code() == ''
    assert json.loads(prefs.path.read_text()) == {'email': 'maya@example.com'}


def test_totp_auto_advance_and_paste(widgets, app, tmp_path):
    screen = SignIn(DemoBackend(), Preferences(tmp_path/'prefs.json'))
    widgets.append(screen)
    screen.show()
    app.processEvents()
    screen.totp.digits[0].setFocus()
    QTest.keyClicks(screen.totp.digits[0], '1')
    assert screen.totp.digits[1].hasFocus()
    app.clipboard().setText('654321')
    QTest.keyClick(screen.totp.digits[1], Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert screen.totp.code() == '654321'


@pytest.mark.parametrize('role,expected', [('admin', ['Fleet', 'Review', 'Studio', 'Audit']), ('support', ['Fleet', 'Review', 'Studio']), ('labeler', ['Review', 'Studio'])])
def test_role_navigation(role, expected, widgets, wait):
    class CountBackend(DemoBackend):
        calls = 0
        def fleet(self):
            self.calls += 1
            return super().fleet()
    backend = CountBackend(role=role)
    shell = Shell(backend, backend.me())
    widgets.append(shell)
    shell.show()
    assert list(shell.navigation) == expected
    if role == 'labeler':
        assert backend.calls == 0
        shell.open_customer(1)
        shell.open_palette()
        assert shell.customer_page is None and shell.palette_dialog is not None
        assert shell.pages.currentWidget() is shell.screens['Studio']
    else:
        wait(lambda: shell.fleet.snapshot is not None)


def test_palette_fuzzy_jump(widgets, wait):
    backend = DemoBackend()
    shell = Shell(backend, backend.me())
    widgets.append(shell)
    shell.show()
    wait(lambda: len(shell.customers) == 3)
    shell.open_palette()
    palette = shell.palette_dialog
    palette.search.setText('oliv')
    assert palette.matches[0][2] == 'hg-olive-01'
    QTest.keyClick(palette.search, Qt.Key.Key_Return)
    wait(lambda: shell.customer_page.stack.currentWidget() is shell.customer_page.body)
    assert shell.customer_page.customer_id == 2
    assert shell.customer_page.device_id == 'hg-olive-01'


def test_offline_initial_and_cached_states(widgets, wait):
    class OfflineBackend(DemoBackend):
        def fleet(self):
            raise OfflineError()
    screen = FleetScreen(OfflineBackend())
    widgets.append(screen)
    screen.show()
    wait(lambda: screen.stack.currentWidget() is screen.failure)
    cached = fleet_screen(widgets, wait)
    cached.show_error(OfflineError())
    assert cached.stack.currentWidget() is cached.content
    assert cached.banner.isVisible()
    assert len(cached.model.rows) == 4


def test_empty_and_server_failure(widgets, wait):
    screen = fleet_screen(widgets, wait)
    screen.completed((replace(screen.snapshot, devices=[]), []), None)
    assert screen.stack.currentWidget() is screen.empty
    screen.snapshot = None
    screen.show_error(ServerError())
    assert screen.stack.currentWidget() is screen.server_failure


def test_worker_never_blocks_ui_and_delivers_on_main_thread(widgets, wait, app):
    release = threading.Event()
    worker_threads = []
    class SlowBackend(DemoBackend):
        def fleet(self):
            worker_threads.append(QThread.currentThread())
            release.wait(2)
            return super().fleet()
    screen = FleetScreen(SlowBackend())
    widgets.append(screen)
    delivered = []
    screen.loaded.connect(lambda *args: delivered.append(QThread.currentThread()))
    screen.show()
    try:
        wait(lambda: bool(worker_threads))
        assert screen.stack.currentWidget() is screen.loading
        screen.search.setText('olive')
        assert screen.search.text() == 'olive'
        assert screen.stack.currentWidget() is screen.loading
        assert worker_threads[0] != app.thread()
    finally:
        release.set()
    wait(lambda: bool(delivered))
    assert delivered == [app.thread()]


def test_successful_signin_and_expired_session(widgets, wait, tmp_path):
    window = AdminWindow(DemoBackend(), prefs=Preferences(tmp_path/'prefs.json'))
    widgets.append(window)
    window.show()
    signin = window.signin
    signin.email.setText('maya@example.com')
    signin.password.setText('sample')
    signin.totp.set_code('123456')
    signin.authenticate()
    wait(lambda: window.shell is not None and window.shell.fleet.snapshot is not None)
    window.shell.fleet.show_error(AuthError())
    assert window.shell is None
    assert window.signin.error.text() == 'Your session expired. Sign in again.'
    assert window.signin.email.text() == 'maya@example.com'


def test_demo_signout_restores_configured_server(widgets, wait, tmp_path):
    from home_guard_project.admin.http_backend import HttpBackend
    configured = HttpBackend('https://cloud.example')
    window = AdminWindow(configured, prefs=Preferences(tmp_path/'prefs.json'))
    widgets.append(window)
    window.use_demo()
    wait(lambda: window.shell.fleet.snapshot is not None)
    window.show_signin()
    assert window.backend is configured
    assert window.signin.backend is configured
    assert window.environment() == 'CLOUD'
    configured.close()


def test_preferences_tolerate_corrupt_file(tmp_path):
    prefs = Preferences(tmp_path/'prefs.json')
    prefs.path.write_text('{broken')
    assert prefs.email() == ''
    prefs.save_email('maya@example.com')
    assert prefs.email() == 'maya@example.com'
