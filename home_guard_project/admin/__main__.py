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
    from home_guard_project.admin.backend import BackendError, UnavailableBackend
    from home_guard_project.admin.logging_setup import setup_logging, install_exception_hook
    from home_guard_project.admin.workers import shutdown_workers
    import home_guard_project.admin as admin_package
    app = QApplication(sys.argv[:1])
    app.setApplicationName('HomeGuardAdmin')
    apply_theme(app, args.theme)
    setup_logging()
    startup_error = None
    try:
        backend = DemoBackend() if args.demo else HttpBackend(args.server)
    except BackendError as error:
        startup_error = error
        backend = UnavailableBackend(args.server, error)
    window = AdminWindow(backend, demo=args.demo, theme=args.theme)
    if startup_error:
        window.signin.error.setText(str(startup_error))
    install_exception_hook(lambda message: window.statusBar().showMessage(message))
    # PyInstaller places the entry script at bundle root. Package __file__ keeps
    # the package-relative resource path in both source and frozen builds.
    icon = Path(admin_package.__file__).parent.parent / 'box' / 'assets' / 'logo.ico'
    window.setWindowIcon(QIcon(str(icon)))
    window.show()
    if args.smoke_test:
        import time
        started = time.monotonic()
        stage = [0]
        def verify():
            if time.monotonic()-started > 30:
                app.exit(2)
                return
            ok = window.signin is not None and not window.windowIcon().isNull()
            if not args.demo:
                app.exit(0 if ok else 2)
                return
            shell = window.shell
            if not ok or not shell or not shell.fleet.snapshot:
                return
            customer = shell.customer_page
            if stage[0] == 0:
                if len(shell.fleet.model.rows) != 4:
                    app.exit(2); return
                shell.open_customer(1); stage[0] = 1
            elif stage[0] == 1 and customer.timeline.model.rows:
                customer.open_event(101); stage[0] = 2
            elif stage[0] == 2:
                player = customer.event_view.player
                if player.player.duration() > 0:
                    player.player.play(); player.player.setPosition(3000); stage[0] = 3
            elif stage[0] == 3:
                player = customer.event_view.player
                if not player.canvas.image.isNull() and player.canvas.overlay.position_ms >= 3000:
                    player.player.pause()
                    if not player.canvas.overlay.frames:
                        app.exit(2); return
                    shell.navigate('Review'); stage[0] = 4
            elif stage[0] == 4 and shell.review_page.event_view.recording:
                shell.navigate('Studio'); stage[0] = 5
            elif stage[0] == 5 and shell.screens['Studio'].loaded_once:
                studio = shell.screens['Studio']; studio.open_export()
                studio.wizard.name.setText('smoke_dataset'); studio.wizard.advance(); studio.wizard.advance(); stage[0] = 6
            elif stage[0] == 6 and shell.screens['Studio'].wizard.preview is not None:
                wizard = shell.screens['Studio'].wizard
                if not wizard.preview.included_ids:
                    app.exit(2); return
                wizard.reject(); shell.navigate('Audit'); stage[0] = 7
            elif stage[0] == 7 and shell.screens['Audit'].loaded_once:
                app.exit(0 if shell.screens['Audit'].model.items else 2)
        smoke_timer = QTimer(window)
        smoke_timer.setInterval(100); smoke_timer.timeout.connect(verify); smoke_timer.start()
    result = app.exec()
    drained = shutdown_workers()
    if drained and hasattr(backend, 'close'):
        backend.close()
    if not drained:
        # Qt destroys the global pool with an unbounded join. All jobs have
        # cancellation set; enforce the process deadline if a socket is stuck.
        os._exit(result)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
