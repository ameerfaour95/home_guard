"""Tag · YOLO speed (2026-10-09, the owner: "every command responds with a delay"): every frame is decoded once when a
clip opens, scaled for the screen within a memory cap, so a frame step, a timeline click and playback are lookups; over
the network the clip is downloaded once into memory for the player and the cache. Zoom fetches the full-size frame.
tests/admin/perf_yolo.py is the per-action harness behind the before/after table."""
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest

from home_guard_project.admin import frame_cache
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.label_view import LabelView

DEMO_CLIP = Path(__file__).resolve().parents[2]/'home_guard_project'/'admin'/'demo_data'/'media'


def demo_clip():
    return sorted(DEMO_CLIP.glob('*.mp4'))[0]


def view(widgets, wait, backend=None):
    b = backend or DemoBackend()
    v = LabelView(b, b.role); widgets.append(v); v.resize(1366, 768); v.show(); v.open_event(101)
    wait(lambda: v.doc is not None and not v.media.busy)
    wait(v.frames_ready, 10)
    return v


def test_the_memory_cap_scales_big_clips_and_leaves_small_ones_alone():
    big = frame_scale = frame_cache.frame_scale(2592, 1520, 61)
    assert 2592*big*1520*big*3*61 <= frame_cache.BUDGET_BYTES*1.01      # 61 frames of the main stream fit the budget
    assert round(2592*big) <= frame_cache.MAX_SIDE
    assert frame_cache.frame_scale(704, 576, 61) == 1.0                 # the sub-stream is kept at its own size
    long = frame_cache.frame_scale(704, 576, 1500)                       # a long clip shrinks to stay inside
    assert 704*long*576*long*3*1500 <= frame_cache.BUDGET_BYTES*1.01 and long < 1.0
    assert frame_scale > 0


def test_decoding_from_memory_equals_decoding_the_file():
    clip = demo_clip()
    from_file = frame_cache.decode(clip.as_uri())
    from_bytes = frame_cache.decode(clip.read_bytes())
    assert len(from_file) == len(from_bytes) > 30 and from_file.size == from_bytes.size == (640, 360)
    assert from_file.done and from_bytes.done
    small = frame_cache.decode(clip.read_bytes(), max_side=320)
    assert small.frames[0].width() == 320 and small.size == (640, 360) and small.scale == pytest.approx(.5)


def test_fetch_downloads_http_media_once_and_leaves_local_files_alone():
    class Backend:
        calls = []
        def media_bytes(self, url):
            self.calls.append(url); return b'clip-bytes'
    b = Backend()
    assert frame_cache.fetch(b, 'https://bucket.example/clip.mp4?sig=1') == b'clip-bytes'
    assert frame_cache.fetch(b, demo_clip().as_uri()) is None and len(b.calls) == 1

    class Broken:
        def media_bytes(self, url): raise OSError('offline')
    assert frame_cache.fetch(Broken(), 'https://bucket.example/clip.mp4') is None   # then the URL streams as before

    class Huge:
        def media_bytes(self, url): return b'x' * (frame_cache.MAX_FETCH_BYTES + 1)
    assert frame_cache.fetch(Huge(), 'https://bucket.example/clip.mp4') is None


def test_a_frame_step_is_a_lookup_not_a_decoder_seek(widgets, wait):
    v = view(widgets, wait)
    start = time.perf_counter()
    for _ in range(10):
        v.step(1)
        assert v.pending_frame is None and v.canvas.isEnabled()         # shown at once, nothing to wait for
        v.canvas.repaint(); v.timeline.repaint()
    per_step = (time.perf_counter() - start) * 1000 / 10
    assert per_step < 50, f'{per_step:.1f} ms per frame step'
    assert v.canvas.image is v.cached(v.doc.frame) and v.doc.frame == 10
    v.seek(40)                                                          # a timeline click
    assert v.doc.frame == 40 and v.pending_frame is None and v.canvas.image is v.cached(40)
    assert v.doc.t_sec == pytest.approx(v.doc.time_for(40))
    v.step(-1); assert v.doc.frame == 39


def test_playback_runs_from_the_cache_and_stops_at_the_end(widgets, wait):
    v = view(widgets, wait)
    v.seek(v.doc.frame_count - 6)
    v.toggle_play()
    assert v.play_timer.isActive() and v.play.text() == 'Pause'
    assert v.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState   # the decoder is not streaming
    wait(lambda: not v.play_timer.isActive(), 5)
    assert v.doc.frame == min(len(v.frames), v.doc.frame_count) - 1 and v.play.text() == 'Play'
    v.toggle_play()                                                     # at the end: plays again from the start
    assert v.doc.frame == 0 and v.play_timer.isActive()
    v.toggle_play(); assert not v.play_timer.isActive()
    v.seek(5); assert v.doc.frame == 5 and not v.play_timer.isActive()


def test_zooming_in_fetches_the_full_size_frame_and_boxes_stay_in_image_coordinates(widgets, wait, monkeypatch):
    monkeypatch.setattr(frame_cache, 'MAX_SIDE', 320)                   # cached at half size
    v = view(widgets, wait)
    v.seek(12)
    assert v.canvas.image.width() == 320 and v.doc.frame_size == [640, 360]
    c = v.canvas
    x, y, w, h = c.display_rect()
    before = c.image_point(QPointF(x + w*.4, y + h*.3))
    c.zoom_to(3.0)
    assert v.refine == 12                                               # the full-size frame is asked for
    wait(lambda: v.refine is None and v.canvas.image.width() == 640, 5)
    assert v.doc.frame == 12
    c.fit()
    x, y, w, h = c.display_rect()
    after = c.image_point(QPointF(x + w*.4, y + h*.3))
    assert (after.x(), after.y()) == pytest.approx((before.x(), before.y()), abs=1e-3)
    c.zoom_to(3.0); v.step(1)                                           # a step while zoomed shows the cached frame
    assert v.doc.frame == 13 and v.refine == 13                          # at once, then refines
    wait(lambda: v.refine is None and v.canvas.image.width() == 640, 5)


def test_downloaded_bytes_feed_the_player_and_the_cache(widgets, wait, monkeypatch):
    clip = demo_clip()
    calls = []

    class Backend(DemoBackend):
        def artifact_access(self, id, purpose='review'):
            access = super().artifact_access(id, purpose)
            access.url = 'https://bucket.example/' + clip.name + '?sig=1'
            return access

        def media_bytes(self, url):
            calls.append(url); return clip.read_bytes()
    v = view(widgets, wait, Backend())
    assert calls == ['https://bucket.example/' + clip.name + '?sig=1']   # one download
    assert v.player.sourceDevice() is v.media_buffer and v.media_buffer is not None
    assert len(v.frames) > 30
    v.step(1); assert v.pending_frame is None


def test_a_late_decoder_frame_does_not_move_a_view_the_cache_serves(widgets, wait):
    v = view(widgets, wait)
    v.seek(30)
    shown = v.canvas.image
    v.player.setPosition(0)                                             # the decoder's own first frame, arriving late
    v.player.play(); v.player.pause()
    QTest.qWait(400)
    assert v.doc.frame == 30 and v.canvas.image is shown


def test_a_new_clip_drops_the_old_frames(widgets, wait):
    v = view(widgets, wait)
    old = v.frames
    v.install(v.doc.annotation)
    assert v.frames is None and v.cached(0) is None and not v.play_timer.isActive()
    assert old is not None
