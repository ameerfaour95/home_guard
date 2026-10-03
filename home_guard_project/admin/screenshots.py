"""Deterministic, offscreen review set: python -m home_guard_project.admin.screenshots."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import time
from pathlib import Path
from dataclasses import replace
from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication
from .theme import apply_theme
from .demo_backend import DemoBackend
from .backend import AuthError, OfflineError
from .prefs import Preferences
from .shell import AdminWindow, Shell


def main():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    output = Path('docs/admin/screenshots')
    output.mkdir(parents=True, exist_ok=True)
    def settle(predicate=lambda: True):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                for _ in range(4):
                    app.processEvents()
                return
            time.sleep(.01)
        raise RuntimeError('Screenshot state did not load')
    def capture(widget, name):
        settle()
        assert widget.grab().save(str(output / f'r1-{name}.png'))
    backend = DemoBackend()
    window = AdminWindow(backend, prefs=Preferences(Path('tmp/admin-screenshot-prefs.json')))
    window.show()
    window.signin.email.setText('maya@homeguard.example')
    capture(window, 'signin')
    window.signin.completed(None, AuthError())
    capture(window, 'signin-error')
    window.enter(backend.me())
    settle(lambda: window.shell.fleet.snapshot is not None)
    fleet = window.shell.fleet
    fleet.table.clearFocus()
    capture(window, 'fleet-1366x768')
    window.resize(1920, 1080)
    capture(window, 'fleet-1920x1080')
    fleet.table.setCurrentIndex(fleet.model.index(1, 0))
    capture(window, 'fleet-detail')
    window.resize(1366, 768)
    capture(window, 'fleet-detail-1366x768')
    fleet.table.clearSelection()
    fleet.table.setCurrentIndex(fleet.model.index(-1, -1))
    fleet.selected_id = None
    fleet.detail.hide()
    window.resize(1366, 768)
    fleet.filter(verdict='critical')
    capture(window, 'fleet-critical')
    fleet.clear_filters()
    window.shell.open_palette()
    settle()
    # Modal dialogs are separate native windows: composite their actual Qt grab.
    pixmap = window.grab()
    painter = QPainter(pixmap)
    dialog = window.shell.palette_dialog
    painter.drawPixmap(window.mapFromGlobal(dialog.pos()), dialog.grab())
    painter.end()
    assert pixmap.save(str(output/'r1-command-palette.png'))
    dialog.close()
    fleet.show_error(OfflineError())
    capture(window, 'offline-cached')
    fleet.snapshot = None
    fleet.show_error(OfflineError())
    capture(window, 'offline')
    fleet.completed((replace(backend.fleet(), devices=[]), []), None)
    capture(window, 'fleet-empty')
    fleet.stack.setCurrentWidget(fleet.loading)
    fleet.summary.setText('Your customer boxes, in one place.')
    fleet.updated.setText('Connecting…')
    fleet.count.setText('Loading fleet…')
    capture(window, 'fleet-loading')
    window.shell.open_customer(3)
    settle(lambda: window.shell.customer_page.stack.currentWidget() is window.shell.customer_page.body)
    capture(window, 'customer')
    window.close()
    labeler = Shell(DemoBackend(role='labeler'), DemoBackend(role='labeler').me())
    labeler.resize(1366, 768)
    labeler.show()
    capture(labeler, 'labeler-navigation')
    labeler.close()
    apply_theme(app, 'light')
    light = AdminWindow(backend, demo=True, theme='light', prefs=Preferences(Path('tmp/admin-screenshot-prefs.json')))
    light.show()
    settle(lambda: light.shell.fleet.snapshot is not None)
    capture(light, 'fleet-light')
    light.close()
    QThreadPool.globalInstance().waitForDone()
    print(f'Wrote {len(list(output.glob("r1-*.png")))} screenshots to {output}')


if __name__ == '__main__':
    main()
