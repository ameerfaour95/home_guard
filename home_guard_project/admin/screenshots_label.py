"""Reproducible offscreen Label screenshots using decoded synthetic demo video."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
from time import monotonic, sleep
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThreadPool
from .theme import apply_theme
from .demo_backend import DemoBackend
from .shell import Shell
from .models import Track, Keyframe


def main():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    out = Path('docs/admin/screenshots'); out.mkdir(parents=True, exist_ok=True)

    def wait(predicate, timeout=10):
        until = monotonic()+timeout
        while monotonic() < until:
            app.processEvents()
            if predicate(): return
            sleep(.01)
        raise RuntimeError('Screenshot state timed out')

    for width, height in [(1920, 1080), (1366, 768)]:
        b = DemoBackend(); shell = Shell(b, b.me()); shell.resize(width, height); shell.show(); shell.open_label(101)
        v = shell.label_page
        wait(lambda: v.doc and not v.canvas.image.isNull() and not v.media.busy)

        def capture(name):
            v.autosave.stop(); app.processEvents()
            assert shell.size().width() == width and shell.size().height() == height, shell.size()
            assert shell.grab().save(str(out/f'label-{name}-{width}.png'))

        capture('suggestions')
        v.doc.accept_all(); v.doc.selected = v.doc.tracks[0].track_id
        # An explicit temporal gap between 2 and 4 seconds, then the track returns.
        tr = v.doc.track
        start, end = tr.keyframes[0].xyxy[:], tr.keyframes[-1].xyxy[:]
        tr.keyframes = [Keyframe(0, 0., start), Keyframe(24, 2., [.325, .5, .369, .722], False),
                        Keyframe(48, 4., [.494, .5, .538, .722]), Keyframe(70, 70/12, end)]
        v.doc.checkpoint(); v.seek(36)
        wait(lambda: v.doc.frame == 36 and abs(v.doc.t_sec-3.) < .02)
        capture('hidden-segment')
        tr.keyframes = [Keyframe(0, 0., start), Keyframe(70, 70/12, end)]
        v.doc.checkpoint(); v.seek(35); wait(lambda: v.doc.frame == 35)
        capture('interpolation')
        v.drop.setChecked(False)
        v.needs_review.setChecked(True); capture('text-edited')
        v.queue = [b.event(101)]; v.queue_index = 0; v.submit()
        wait(lambda: not v.writer.busy and v.doc.annotation.status == 'submitted')
        v.review_note.setText('Tighten the entrance box.'); v.review_note.setCursorPosition(0)
        capture('admin-review')
        shell.close(); QThreadPool.globalInstance().waitForDone(); app.processEvents(); shell.deleteLater(); app.processEvents()
        b = DemoBackend(role='labeler'); shell = Shell(b, b.me()); shell.resize(width, height); shell.show()
        wait(lambda: shell.label_page.doc and not shell.label_page.loader.busy)
        shell.open_label(101); v = shell.label_page
        wait(lambda: v.doc and v.doc.annotation.event_id == 101 and not v.canvas.image.isNull() and not v.media.busy)
        capture('labeler')
        shell.close(); QThreadPool.globalInstance().waitForDone(); app.processEvents(); shell.deleteLater(); app.processEvents()


if __name__ == '__main__': main()
