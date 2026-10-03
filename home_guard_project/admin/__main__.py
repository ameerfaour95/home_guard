"""python -m home_guard_project.admin [--demo] [--server URL]."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark')
    parser.add_argument('--smoke-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke_test:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtCore import QTimer, QThreadPool
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from home_guard_project.admin.theme import apply_theme
    from home_guard_project.admin.http_backend import HttpBackend
    from home_guard_project.admin.demo_backend import DemoBackend
    from home_guard_project.admin.shell import AdminWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName('HomeGuardAdmin')
    apply_theme(app, args.theme)
    backend = DemoBackend() if args.demo else HttpBackend(args.server)
    window = AdminWindow(backend, demo=args.demo, theme=args.theme)
    icon = Path(__file__).parent.parent / 'box' / 'assets' / 'logo.ico'
    window.setWindowIcon(QIcon(str(icon)))
    window.show()
    if args.smoke_test:
        def verify():
            ok = window.signin is not None and not window.windowIcon().isNull()
            if args.demo:
                ok = ok and window.shell and window.shell.fleet.snapshot and len(window.shell.fleet.model.rows) == 4
            if args.demo and ok:
                window.shell.fleet.table.setCurrentIndex(window.shell.fleet.model.index(1, 0))
                ok = window.shell.fleet.detail.device.device_id == 'hg-pine-01' and not window.windowIcon().isNull()
            app.exit(0 if ok else 2)
        QTimer.singleShot(1500, verify)
    result = app.exec()
    QThreadPool.globalInstance().waitForDone()
    if hasattr(backend, 'close'):
        backend.close()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
