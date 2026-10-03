"""Round 2 review captures, including a real QMediaPlayer-decoded frame."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import time
from pathlib import Path
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication, QFrame
from .theme import apply_theme
from .demo_backend import DemoBackend
from .shell import Shell


def main():
    app = QApplication.instance() or QApplication([]); apply_theme(app)
    output = Path('docs/admin/screenshots'); output.mkdir(parents=True, exist_ok=True)

    def settle(predicate=lambda: True, timeout=12):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            app.processEvents()
            if predicate():
                for _ in range(10):
                    app.processEvents()
                return
            time.sleep(.01)
        raise RuntimeError('Screenshot state did not load')

    def capture(window, name):
        settle()
        assert window.grab().save(str(output/f'r2-{name}.png'))

    backend = DemoBackend(); shell = Shell(backend, backend.me()); shell.resize(1920, 1080); shell.show()
    settle(lambda: shell.fleet.snapshot is not None and bool(shell.fleet.activity.hours))
    capture(shell, 'fleet-activity')
    capture(shell.findChild(QFrame, 'rail'), 'nav-icons')
    shell.open_customer(1)
    customer = shell.customer_page; timeline = customer.timeline
    settle(lambda: bool(timeline.model.rows) and bool(timeline.density.rows) and not timeline.thumbnail_runner.busy)
    capture(shell, 'customer-timeline-1920')
    shell.resize(1366, 768); settle()
    assert (shell.width(), shell.height()) == (1366, 768)
    capture(shell, 'customer-timeline-1366')
    timeline.filter_cell(timeline.model.rows[0].camera, timeline.model.rows[0].start_utc.replace(minute=0, second=0, microsecond=0))
    settle(lambda: not timeline.runner.busy and timeline.stack.currentWidget() is timeline.table)
    capture(shell, 'timeline-filtered')
    timeline.reset_cell(); settle(lambda: not timeline.runner.busy)

    def open_mid(page, eid):
        page.open_event(eid)
        view = page.event_view; player = view.player
        if not player.canvas.overlay.enabled:
            player.toggle_boxes()
        settle(lambda: view.recording is not None and not view.evidence_runner.busy and player.player.duration() > 0)
        player.player.play()
        player.player.setPosition(3000)
        settle(lambda: not player.canvas.image.isNull() and player.canvas.overlay.position_ms >= 3000)
        player.player.pause()
        player.player.setPosition(3000)
        settle(lambda: 2900 <= player.canvas.overlay.position_ms <= 3100)
        return view

    shell.resize(1920, 1080)
    view = open_mid(customer, 101); capture(shell, 'event-boxes-on')
    shell.resize(1366, 768); capture(shell, 'event-1366')
    view.player.toggle_boxes(); capture(shell, 'event-boxes-off')
    view = open_mid(customer, 102); capture(shell, 'event-ai-failed')
    view = open_mid(customer, 103); capture(shell, 'event-ai-fallback')
    shell.close()
    labeler_backend = DemoBackend(role='labeler')
    labeler = Shell(labeler_backend, labeler_backend.me()); labeler.resize(1366, 768); labeler.show(); labeler.navigate('Review')
    settle(lambda: bool(labeler.review_page.timeline.model.rows))
    open_mid(labeler.review_page, 101); capture(labeler, 'labeler-event')
    labeler.close()
    apply_theme(app, 'light')
    light = Shell(backend, backend.me(), theme='light'); light.resize(1366, 768); light.show(); light.open_customer(1)
    settle(lambda: bool(light.customer_page.timeline.model.rows) and not light.customer_page.timeline.thumbnail_runner.busy)
    capture(light, 'timeline-light'); light.close()
    QThreadPool.globalInstance().waitForDone()
    print(f'Wrote {len(list(output.glob("r2-*.png")))} Round 2 screenshots')


if __name__ == '__main__':
    main()
