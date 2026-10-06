"""Offscreen screenshots of the tagging studio, from the demo data or a running local service.

    python -m home_guard_project.admin.screenshots_tag                      # demo -> docs/admin/screenshots
    python -m home_guard_project.admin.screenshots_tag --server http://127.0.0.1:8610 --out DIR
"""
import argparse
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from time import monotonic, sleep
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication
from .theme import apply_theme
from .demo_backend import DemoBackend
from .shell import Shell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server')
    parser.add_argument('--out', default='docs/admin/screenshots')
    parser.add_argument('--theme', default='dark')
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    apply_theme(app, args.theme)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prefix = 'tag-real' if args.server else 'tag'

    def pump(seconds):
        until = monotonic() + seconds
        while monotonic() < until:
            app.processEvents(); sleep(.01)

    def wait(predicate, timeout=30):
        until = monotonic() + timeout
        while monotonic() < until:
            app.processEvents()
            if predicate():
                return True
            sleep(.01)
        raise RuntimeError('Screenshot state timed out')

    for width, height in [(1366, 768), (1920, 1080)]:
        if args.server:
            from .http_backend import HttpBackend
            backend = HttpBackend(args.server); backend.local_login()
        else:
            backend = DemoBackend()
        shell = Shell(backend, backend.me(), environment='LOCAL' if args.server else 'DEMO', theme=args.theme)
        shell.resize(width, height); shell.show(); shell.navigate('Tag')
        view = shell.tag_page
        wait(lambda: view.detail is not None and not view.media_runner.busy and not view.clip_runner.busy)
        wait(lambda: not view.canvas.image.isNull() or view.canvas.message != 'Loading video…', 15)
        pump(1.5)

        def capture(name):
            app.processEvents()
            assert shell.grab().save(str(out / f'{prefix}-{name}-{width}.png'))

        capture('queue')
        view.set_category('N3'); view.chips['zone'].set_value('entrance'); view.chip_changed('zone')
        view.chips['movement'].set_value('leaving'); view.chip_changed('movement')
        view.mark_evidence(); pump(.3)
        capture('tagging')
        view.save()
        wait(lambda: not view.save_runner.busy and not view.clip_runner.busy and not view.queue_runner.busy, 20)
        pump(1.5)
        capture('after-save')
        shell.close(); QThreadPool.globalInstance().waitForDone(); app.processEvents()


if __name__ == '__main__':
    main()
