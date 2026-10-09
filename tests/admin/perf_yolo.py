"""Tag · YOLO speed harness: time per action on a real clip (opening it, a frame step, play, select, draw, resize,
keyframe, hide, split / merge, save) with time.perf_counter around the handlers, including the paint they cause.

    python tests/admin/perf_yolo.py <clip.mp4> [--tracks 6] [--window]
    python tests/admin/perf_yolo.py --live <event id> [--base http://127.0.0.1:8610] [--window]

It runs the real LabelView over a demo backend that serves *clip* as event 101's video, with six tracks of boxes.
--live opens a real event from the local service instead (its own clip and tracks over the network; edits stay
local and nothing is saved). Offscreen by default (logic and decoding are the same); --window opens a real window (the numbers the owner feels).
Not collected by pytest (no test_ prefix); tests/admin/test_yolo_speed.py checks the targets on a small clip.
"""
import argparse
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def build(clip, tracks, window, live=None):
    if not window:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from home_guard_project.admin.theme import apply_theme
    apply_theme(app)
    from home_guard_project.admin.demo_backend import DemoBackend
    from home_guard_project.admin.label_view import LabelView
    from home_guard_project.admin.models import Keyframe, MediaAccess, Track
    if live:
        from home_guard_project.admin.http_backend import HttpBackend
        backend = HttpBackend(live); backend.local_login()
        return app, backend, LabelView

    class ClipBackend(DemoBackend):
        def artifact_access(self, id, purpose='review'):
            return MediaAccess(Path(clip).resolve().as_uri(), datetime.now(timezone.utc) + timedelta(minutes=10),
                               'video/mp4')

        def annotation(self, event_id):
            a = super().annotation(event_id)
            if a.version == 0:
                a.fps, a.frame_count = 7.0, 61
                a.tracks = [Track(f't-{n}', 'person' if n % 2 else 'car',
                                  [Keyframe(f, f / 7.0, [.05 + .12*n + .002*f, .2, .14 + .12*n + .002*f, .6])
                                   for f in range(0, 61, 5)], 'yolo')
                            for n in range(tracks)]
            return a
    return app, ClipBackend(), LabelView


def run(clip, tracks=6, window=False, repeat=8, live=None, event_id=101):
    app, backend, LabelView = build(clip, tracks, window, live)
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest

    def spin_until(pred, timeout=10.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            app.processEvents()
            if pred():
                return True
            time.sleep(.001)
        return False

    out = {}
    t0 = time.perf_counter()
    v = LabelView(backend, 'admin'); v.resize(1366, 768); v.show()
    v.autosave.setInterval(10**9)
    v.open_event(event_id)
    spin_until(lambda: v.doc is not None and not v.canvas.image.isNull() and not v.media.busy, 20)
    out['open clip (to first frame)'] = [(time.perf_counter() - t0) * 1000]
    if hasattr(v, 'frames_ready'):
        spin_until(lambda: v.frames_ready(), 20)
        out['open clip (all frames ready)'] = [(time.perf_counter() - t0) * 1000]

    def timed(name, action, done=lambda: True):
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            action()
            spin_until(done, 5)
            v.canvas.repaint(); v.timeline.repaint()       # the paint the action causes
            samples.append((time.perf_counter() - start) * 1000)
        out[name] = samples

    v.seek(5); spin_until(lambda: v.pending_frame is None)
    timed('repaint canvas + timeline (one)', lambda: None)
    def moves():
        c = v.canvas
        for i in range(10):
            c.preview = None; c.update(); app.processEvents()
    timed('10 canvas updates (a drag)', moves)
    timed('next frame (→)', lambda: v.step(1), lambda: v.pending_frame is None)
    timed('previous frame (←)', lambda: v.step(-1), lambda: v.pending_frame is None)
    timed('jump 20 frames (timeline click)', lambda: v.seek((v.doc.frame + 20) % 60), lambda: v.pending_frame is None)
    # play: frames shown per second over 2 s
    shown = []
    v.doc.position_changed.connect(lambda: shown.append(time.perf_counter()))
    v.seek(0); spin_until(lambda: v.pending_frame is None)
    shown.clear(); v.toggle_play(); start = time.perf_counter()
    spin_until(lambda: time.perf_counter() - start > 2.0, 3)
    v.toggle_play()
    out['play (frames shown per s)'] = [len(shown) / 2.0]
    c = v.canvas
    x, y, w, h = c.display_rect()
    box = lambda t: c.rect_for(__import__('home_guard_project.fleet_contract.tracks', fromlist=['box_at']).box_at(t, v.doc.t_sec))  # noqa: E731
    if not v.doc.tracks:                       # a clip with no boxes: draw one to work on
        a = QPoint(int(x + w*.3), int(y + h*.3)); b = QPoint(int(x + w*.45), int(y + h*.7))
        QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=a); QTest.mouseMove(c, b)
        QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=b)
    target = v.doc.tracks[min(1, len(v.doc.tracks)-1)]
    v.seek(next(k.frame for k in target.keyframes)); spin_until(lambda: v.pending_frame is None)

    def select():
        r = box(target)
        QTest.mouseClick(c, Qt.MouseButton.LeftButton, pos=r.center().toPoint())
    timed('select a box (click)', select)

    def draw():
        a = QPoint(int(x + w*.80), int(y + h*.70)); b = QPoint(int(x + w*.92), int(y + h*.92))
        QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=a)
        for i in range(1, 11):
            QTest.mouseMove(c, QPoint(a.x() + (b.x()-a.x())*i//10, a.y() + (b.y()-a.y())*i//10))
        QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=b)
        v.doc.undo()
    timed('draw a box (10 moves)', draw)

    def resize():
        v.doc.selected = target.track_id
        r = box(target)
        a = r.bottomRight().toPoint()
        QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=a)
        for i in range(1, 11):
            QTest.mouseMove(c, QPoint(a.x() + i, a.y() + i))
        QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=QPoint(a.x() + 10, a.y() + 10))
        v.doc.undo()
    timed('resize a box (10 moves)', resize)
    v.doc.selected = target.track_id
    timed('keyframe K', lambda: v.doc.toggle_keyframe())
    timed('hide / keep O', lambda: v.doc.set_enabled())
    timed('hide track H', v.toggle_track_hidden)

    def split_merge():
        v.doc.selected = target.track_id
        if v.doc.split():
            v.doc.merge()
    timed('split + merge', split_merge)
    if live:                                    # never write to the live service
        v.autosave.stop(); v.doc.saved_snapshot = v.doc.snapshot(); return out
    timed('save', lambda: v.save(), lambda: not v.writer.busy)
    v.close()
    return out


def report(out):
    rows = []
    for name, s in out.items():
        if 'per s' in name:
            rows.append(f'{name:36s} {s[0]:8.1f}')
        else:
            rows.append(f'{name:36s} median {statistics.median(s):8.1f} ms   max {max(s):8.1f} ms')
    return '\n'.join(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('clip', nargs='?'); p.add_argument('--tracks', type=int, default=6)
    p.add_argument('--window', action='store_true'); p.add_argument('--live', type=int)
    p.add_argument('--base', default='http://127.0.0.1:8610')
    args = p.parse_args()
    print(report(run(args.clip, args.tracks, args.window, live=args.base if args.live else None, event_id=args.live or 101)))
    sys.stdout.flush(); os._exit(0)
