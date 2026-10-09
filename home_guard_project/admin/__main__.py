"""python -m home_guard_project.admin [--demo] [--server URL]."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--server')
    parser.add_argument('--local', action='store_true', help='sign in without credentials to a local service')
    parser.add_argument('--theme', choices=['dark', 'light'])
    parser.add_argument('--smoke-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    from home_guard_project.admin.prefs import Preferences
    prefs = Preferences()
    args.server = args.server or prefs.get('server') or 'http://127.0.0.1:8610'
    args.theme = args.theme or prefs.get('theme') or 'dark'
    from urllib.parse import urlsplit
    local = not args.demo and (args.local or urlsplit(args.server).hostname in ('127.0.0.1', 'localhost'))
    if args.theme not in ('dark', 'light'): args.theme = 'dark'
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
    window = AdminWindow(backend, demo=args.demo, theme=args.theme, prefs=prefs, local=local)
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
        import json
        started = time.monotonic()
        stage = [0]
        live = os.environ.get('HG_ADMIN_IT') == '1' and not args.demo
        live_result = {}
        if live:
            setup_logging().info('Packaged integration starting pid=%s', os.getpid())
            stage[0] = -1
            Path(os.environ['HG_ADMIN_IT_RESULT']).write_text(json.dumps(dict(ready_pid=os.getpid())))
        def verify():
            if time.monotonic()-started > 30:
                if live:
                    Path(os.environ['HG_ADMIN_IT_RESULT']).write_text(json.dumps(dict(live_result, stage=stage[0], signin_error=window.signin.error.text())))
                app.exit(2)
                return
            ok = window.signin is not None and not window.windowIcon().isNull()
            if live:
                if stage[0] == -1:
                    path = Path(os.environ['HG_ADMIN_IT_LOGIN_FILE'])
                    if not path.exists(): return
                    login = json.loads(path.read_text())
                    if login.get('pid') != os.getpid(): return
                    path.unlink()
                    window.signin.email.setText(login['email'])
                    window.signin.password.setText(login['password'])
                    for digit, value in zip(window.signin.totp.digits, login['totp']): digit.setText(value)
                    window.signin.authenticate()
                    stage[0] = 0
                    return
                shell = window.shell
                if not shell or not shell.fleet.snapshot: return
                customer = shell.customer_page
                if stage[0] == 0:
                    live_result['fleet'] = len(shell.fleet.snapshot.devices)
                    shell.open_customer(1); stage[0] = 1
                elif stage[0] == 1 and customer.timeline.model.rows and customer.timeline.density.hours:
                    live_result['timeline'] = len(customer.timeline.model.rows)
                    customer.open_event(101); stage[0] = 2
                elif stage[0] == 2 and customer.event_view.recording and not customer.event_view.evidence_runner.busy:
                    view = customer.event_view
                    live_result['detections'] = len(view.player.canvas.overlay.frames)
                    live_result['media_unavailable'] = 'Not available yet' in view.banner.text()
                    if not live_result['media_unavailable'] and view.player.canvas.image.isNull():
                        view.player.player.play()
                        return
                    live_result['frame_decoded'] = not view.player.canvas.image.isNull()
                    view.review_runner.finished.connect(lambda result, error: live_result.update(review_saved=error is None))
                    view.toggle_review('flagged'); stage[0] = 3
                elif stage[0] == 3 and not customer.event_view.review_runner.busy:
                    Path(os.environ['HG_ADMIN_IT_RESULT']).write_text(json.dumps(live_result))
                    app.exit(0 if live_result['detections'] and live_result['review_saved'] else 2)
                return
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
                if not shell.screens['Audit'].model.items:
                    app.exit(2); return
                shell.open_label(101); stage[0] = 8
            elif stage[0] == 8 and shell.label_page.doc and not shell.label_page.canvas.image.isNull():
                view = shell.label_page
                view.doc.accept_all(); view.doc.selected = view.doc.tracks[0].track_id
                view.needs_review.setChecked(True)   # an edit in the packaged Label editor (boxes and clip checks)
                view.seek(36); stage[0] = 9
            elif stage[0] == 9 and shell.label_page.pending_frame is None:
                view = shell.label_page
                if view.doc.frame != 36:
                    app.exit(2); return
                view.doc.set_enabled(); view.save(); stage[0] = 10
            elif stage[0] == 10 and shell.label_page.doc.annotation.version == 1:
                view = shell.label_page
                view.queue = [view.recording]; view.queue_index = 0
                view.submit(); stage[0] = 11
            elif stage[0] == 11 and shell.label_page.doc.annotation.status == 'submitted':
                view = shell.label_page
                view.review_note.setText('Packaged annotation workflow verified.'); view.review('accept'); stage[0] = 12
            elif stage[0] == 12 and shell.label_page.doc.annotation.status == 'reviewed':
                app.exit(0)
        smoke_timer = QTimer(window)
        smoke_timer.setInterval(100); smoke_timer.timeout.connect(verify); smoke_timer.start()
    result = app.exec()
    drained = shutdown_workers()
    if drained:
        for client in {backend, window.backend, window.configured_backend, *window.retired_backends}:
            if hasattr(client, 'close'): client.close()
    if not drained:
        # Qt destroys the global pool with an unbounded join. All jobs have
        # cancellation set; enforce the process deadline if a socket is stuck.
        os._exit(result)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
