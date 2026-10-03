"""Quiet logging uses the collector's masked JPEG buffer, without cloud clients."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import alert_clips, inference as inf, boxconfig
from home_guard_project.box.alert_clips import quiet_stem, trim_quiet
from home_guard_project.box.inference import QuietEvent, QuietSaver, QuietTracker, count_people


def result(*labels):
    return SimpleNamespace(names=dict(enumerate(labels)),
                           boxes=[SimpleNamespace(cls=[i]) for i in range(len(labels))])


class QuietTrackerTest(unittest.TestCase):
    def test_one_visit_is_one_event(self):
        tr = QuietTracker("gate")
        self.assertIsNone(tr.look(100, True, ["person"], 1))
        self.assertIsNone(tr.look(105, True, ["person", "car"], 2))
        self.assertIsNone(tr.look(112, False, [], 0))
        ev = tr.look(116, False, [], 0)
        self.assertEqual((ev.start, ev.last_seen, ev.people, ev.labels), (100, 105, 2, {"person", "car"}))
        self.assertIsNone(tr.look(117, False, [], 0))

    def test_a_long_visit_is_cut_every_minute(self):
        tr = QuietTracker("gate")
        events = [e for s in range(130) if (e := tr.look(100 + s, True, ["car"], 0))]
        self.assertEqual([e.start for e in events], [100, 160])
        self.assertEqual(tr.flush().start, 220)
        self.assertIsNone(tr.flush())

    def test_gap_closes_before_new_trigger(self):
        tr = QuietTracker("gate")
        tr.look(100, True, ["person"], 1)
        self.assertEqual(tr.look(111, True, ["car"], 0).labels, {"person"})
        self.assertEqual(tr.flush().start, 111)

    def test_counts_are_peak_simultaneous_counts(self):
        tr = QuietTracker("gate")
        tr.look(100, True, ["person", "person", "car"], 2)
        tr.look(101, True, ["person", "car", "car"], 1)
        self.assertEqual(tr.flush().class_counts, {"person": 2, "car": 2})


class QuietSaverTest(unittest.TestCase):
    def test_marker_changes_are_coalesced_without_displacing_pending_clips(self):
        gate, started, saved = threading.Event(), threading.Event(), []
        def save(event, frames):
            started.set()
            gate.wait(5)
            saved.append(event)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, ".registry", "quiet_since.json")
            saver = QuietSaver(save)
            try:
                saver.submit("active", [])
                self.assertTrue(started.wait(2))
                saver.submit("pending", [])
                for stamp in range(100):
                    saver.marker(path, stamp)
            finally:
                gate.set()
                saver.stop()
            self.assertEqual(saved, ["active", "pending"])
            self.assertEqual(json.loads(Path(path).read_text()), {"since": 99})

    def test_blocked_marker_write_does_not_block_submission_and_failure_is_contained(self):
        gate, started, saved = threading.Event(), threading.Event(), []
        def blocked(*args, **kwargs):
            started.set()
            gate.wait(5)
            raise OSError("slow disk failed")
        with mock.patch.object(inf.os, "makedirs", side_effect=blocked):
            saver = QuietSaver(lambda event, frames: saved.append(event))
            try:
                saver.marker("unused/quiet_since.json", 100)
                self.assertTrue(started.wait(2))
                self.assertTrue(saver.submit("event", []))
                self.assertFalse(gate.is_set())
            finally:
                gate.set()
                saver.stop()
        self.assertEqual(saved, ["event"])

    def test_bounded_queue_drops_oldest_pending(self):
        gate, started, saved = threading.Event(), threading.Event(), []
        def save(event, frames):
            started.set()
            gate.wait(5)
            saved.append(event)
        saver = QuietSaver(save, max_pending=2)
        try:
            self.assertTrue(saver.submit("active", []))
            self.assertTrue(started.wait(2))
            self.assertTrue(saver.submit("old", []))
            self.assertTrue(saver.submit("new", []))
            self.assertFalse(saver.submit("newest", []))
        finally:
            gate.set()
            saver.stop()
        self.assertEqual(saved, ["active", "new", "newest"])

    def test_save_failure_does_not_kill_worker(self):
        calls = []
        def save(event, frames):
            calls.append(event)
            if event == "bad":
                raise OSError("disk full")
        saver = QuietSaver(save)
        saver.submit("bad", [])
        saver.submit("good", [])
        saver.stop()
        self.assertEqual(calls, ["bad", "good"])
        saver.stop()
        self.assertFalse(saver.submit("stopped", []))


class HelpersTest(unittest.TestCase):
    def test_count_people(self):
        self.assertEqual(count_people(result("person", "person", "car")), 2)
        self.assertEqual(count_people(SimpleNamespace(boxes=None)), 0)
        self.assertEqual(count_people(None), 0)

    def test_prime_resets_reference_and_pending_movement(self):
        mem = inf.VehicleMemory()
        a, b = [(0.1, 0.1, 0.2, 0.2)], [(0.5, 0.5, 0.6, 0.6)]
        self.assertFalse(mem.prime(a))
        self.assertFalse(mem.look(a, now=100))
        self.assertFalse(mem.look(b, now=101))
        self.assertTrue(mem.look(b, now=102))
        mem.look(a, now=103)
        self.assertFalse(mem.prime(b))
        self.assertFalse(mem.look(a, now=200))
        self.assertTrue(mem.look(a, now=201))

    def test_setting_defaults_off_and_reloads(self):
        settings = inf.AlertSettings.from_box_settings({})
        self.assertFalse(settings.quiet_log)
        self.assertIn("quiet_log", inf.apply_live_settings(settings, {"quiet_log": True}))
        self.assertTrue(settings.quiet_log)
        self.assertTrue(settings.live_values()["quiet_log"])
        self.assertIn("quiet_log", boxconfig.BOOLEAN_OPTIONS)
        self.assertIn("quiet_log", boxconfig.LIVE_OPTIONS)
        self.assertEqual(boxconfig.NUMBER_OPTIONS["quiet_max_gb"], (1, 500))

    def test_quiet_uses_masked_sub_buffer_with_exact_timestamps(self):
        sub = SimpleNamespace(buf_lock=threading.Lock(),
                              buf=deque([(95, b"before"), (96, b"masked"), (110, b"end"), (111, b"after")]))
        event = QuietEvent("gate", 100, 104, {"person"}, 1)
        self.assertEqual(inf._quiet_frames(event, sub, 120), [(96, b"masked"), (110, b"end")])

    def test_saved_meta_has_detector_counts_and_no_description(self):
        import numpy as np
        event = QuietEvent("gate", 100, 104, {"person", "car"}, 2,
                           class_counts={"person": 2, "car": 3})
        frames = [(96, np.zeros((24, 32, 3), np.uint8)), (110, np.zeros((24, 32, 3), np.uint8))]
        with tempfile.TemporaryDirectory() as root, mock.patch.object(alert_clips, "_to_h264", return_value=False), \
                mock.patch.object(inf, "dispatch_alert") as dispatch:
            inf._save_quiet(event, frames, root)
            meta = json.loads(next(Path(root).rglob("*.meta.json")).read_text())
            self.assertEqual(meta["kind"], "quiet")
            self.assertEqual(meta["mode"], "assistant")
            self.assertFalse(meta["described"])
            self.assertEqual(meta["alert"]["people"], 2)
            self.assertEqual(meta["alert"]["alert_command"], "[none]")
            self.assertEqual(meta["yolo"]["class_counts"], {"person": 2, "car": 3})
            self.assertEqual((meta["clip_start_ts"], meta["clip_end_ts"]), (96, 110))
            self.assertNotIn("teacher", meta)
            self.assertNotIn("vlm_input", meta)
            dispatch.assert_not_called()

    def test_failed_save_is_contained(self):
        with mock.patch.object(alert_clips, "write_alert_clip", side_effect=OSError("full")):
            inf._save_quiet(QuietEvent("gate", 100, 101), [], "unused")

    def test_missing_or_invalid_worker_label_is_empty_and_stays_loud(self):
        for parsed in (None, {}, {"label": "unknown"}, {"label": "normal"}):
            with self.subTest(parsed=parsed), mock.patch.object(inf, "owner_language", return_value="en"), \
                    mock.patch.object(inf, "dispatch_alert", return_value={"sent": True}) as dispatch:
                backend = SimpleNamespace(analyze=lambda *a, **kw: ("", parsed))
                job = inf.AlertJob("gate", "gate_100_alert", 100)
                inf._worker(backend, {}, {}, inf.AlertSettings(), "gate", [], job=job)
                valid = parsed == {"label": "normal"}
                self.assertEqual(job.alert["label"], "normal" if valid else "")
                self.assertEqual(dispatch.call_args.kwargs["alert"]["label"], job.alert["label"])
                self.assertEqual(dispatch.call_args.kwargs["silent"], valid)


class RunTest(unittest.TestCase):
    def run_loop(self, enabled=True, change=None, broken=False, positions=None, visible=None, duration=14,
                 snapshot_error=False, start_at=None, script=None, alert_on=None, cooldown=120):
        from home_guard_project.box import ai_status, camera_alerts, telegram_agent
        from home_guard_project.data_collection import config
        import numpy as np
        cfg = config.Config()
        cfg.CAMERAS = {"gate": "synthetic", "yard": "synthetic"}
        start = (start_at or datetime(2026, 10, 3, 12)).timestamp()
        self.start = start
        clock, looks, saved = [start], [], []
        status = mock.Mock()
        status.offline.return_value = []
        sub = SimpleNamespace(buf_lock=threading.Lock(), buf=deque(), keep_seconds=18.)
        streams = {c: SimpleNamespace(sub_cap=sub, read=lambda: (
                                      np.zeros((24, 32, 3), np.uint8)
                                      if visible is None or visible(clock[0] - start) else None),
                                      last_ts=start + 100.) for c in cfg.CAMERAS}
        detector, backend = mock.Mock(), mock.Mock()
        def predict(*a, **kw):
            looks.append(clock[0])
            if script:
                labels, delay = script(len(looks) - 1, clock[0] - start)
                clock[0] += delay
                return [result(*labels)]
            if broken and clock[0] < start + 1:
                raise ValueError("bad detector frame")
            if positions:
                return [result("car") if positions(clock[0] - start) else result()]
            return [result("person", "person", "car") if clock[0] < start + 2 else result()]
        detector.predict.side_effect = predict
        class StopLoop(BaseException):
            pass
        def sleep(_):
            clock[0] += .25
            sub.buf.append((clock[0], b"already masked"))
            if clock[0] >= start + duration:
                raise StopLoop
        settings = inf.AlertSettings(alert_start_hour=22, alert_end_hour=6, quiet_log=enabled)
        object.__setattr__(settings, "cooldown_sec", cooldown)
        if positions:
            object.__setattr__(settings, "alert_on", ("vehicle",))
        if alert_on:
            object.__setattr__(settings, "alert_on", alert_on)
        self.memories = []
        vehicle_memory = inf.VehicleMemory
        def make_memory():
            memory = vehicle_memory()
            memory.prime = mock.Mock(wraps=memory.prime)
            self.memories.append(memory)
            return memory
        self.admissions = []
        def drain(pending, *args):
            self.admissions.extend((j.camera, round(j.ts - start, 3), j.labels) for j in pending)
            pending.clear()
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            patches = [
                mock.patch("dotenv.load_dotenv"),
                mock.patch.object(boxconfig, "load_box_settings", return_value={}),
                mock.patch.object(boxconfig, "PRODUCTION_LIVE_DIR", root),
                mock.patch.object(config, "load_config", return_value=cfg),
                mock.patch.object(inf.AlertSettings, "from_box_settings", return_value=settings),
                mock.patch.object(inf, "make_backend", return_value=backend),
                mock.patch.object(inf, "load_detector", return_value=(detector, None)),
                mock.patch.object(inf, "_camera_streams", return_value=(streams, dict.fromkeys(streams))),
                mock.patch.object(inf, "LiveSettings", return_value=SimpleNamespace(check=lambda now: [])),
                mock.patch.object(inf, "filter_by_thresholds", side_effect=lambda r, *a: r),
                mock.patch.object(inf, "vehicle_boxes", side_effect=lambda r: positions(clock[0] - start) if positions else []),
                mock.patch.object(inf, "VehicleMemory", side_effect=make_memory),
                mock.patch.object(ai_status, "objects_from_result", return_value=[]),
                mock.patch.object(ai_status, "AiStatus", return_value=status),
                mock.patch.object(camera_alerts, "LiveCameraAlerts", return_value=SimpleNamespace(
                    check=lambda now: False, overrides={}, sensitivity={},
                    thresholds_for=lambda name, defaults: defaults, for_camera=lambda name, defaults: defaults)),
                mock.patch.object(telegram_agent, "start", return_value=None),
                mock.patch.object(inf, "dispatch_alert"),
                mock.patch.object(inf, "_start_due_alerts", side_effect=drain),
                mock.patch.object(inf, "_save_quiet", side_effect=lambda ev, frames, root: saved.append((ev, frames))),
                mock.patch.object(inf.time, "time", side_effect=lambda: clock[0]),
                mock.patch.object(inf.time, "sleep", side_effect=sleep),
                mock.patch.object(inf, "datetime"),
            ]
            for patch in patches:
                stack.enter_context(patch)
            if snapshot_error:
                stack.enter_context(mock.patch.object(inf, "_quiet_frames", side_effect=OSError("buffer unavailable")))
            inf.datetime.now.side_effect = lambda: datetime.fromtimestamp(clock[0])
            # Live mutation here is deliberate: run() shares this frozen settings instance.
            if change:
                def reload(now):
                    value = change(now - start)
                    old = settings.quiet_log
                    object.__setattr__(settings, "quiet_log", value)
                    return ["quiet_log"] if value != old else []
                inf.LiveSettings.return_value.check = reload
            with self.assertRaises(StopLoop):
                try:
                    inf.run()
                except StopLoop:
                    frame = sys.exc_info()[2]
                    while frame and frame.tb_frame.f_code is not inf.run.__code__:
                        frame = frame.tb_next
                    self.loop_state = dict(frame.tb_frame.f_locals)
                    drain(self.loop_state["pending"])
                    raise
            backend.analyze.assert_not_called()
            inf.dispatch_alert.assert_not_called()
            marker = Path(root, ".registry", "quiet_since.json").exists()
        return looks, saved, status, marker, sub

    def test_off_by_default_skips_detector_and_marks_not_recording(self):
        looks, saved, status, marker, sub = self.run_loop(enabled=False)
        self.assertEqual((looks, saved, marker), ([], [], False))
        self.assertIn("not recording", status.mode.call_args.args[1])
        self.assertEqual(sub.keep_seconds, 18.)

    def test_once_a_second_each_camera_saves_without_ai_or_message(self):
        looks, saved, status, marker, sub = self.run_loop()
        self.assertEqual(len(looks), 28)
        self.assertEqual(len(saved), 2)
        self.assertEqual({ev.camera for ev, _ in saved}, {"gate", "yard"})
        self.assertEqual(saved[0][0].class_counts, {"person": 2, "car": 1})
        self.assertTrue(saved[0][1])
        self.assertTrue(marker)
        self.assertIn("quiet logging", status.mode.call_args.args[1])
        self.assertGreaterEqual(sub.keep_seconds, 79)

    def test_live_disable_flushes_event_and_removes_marker(self):
        looks, saved, _, marker, _ = self.run_loop(change=lambda elapsed: elapsed < 2)
        self.assertEqual(len(looks), 4)
        self.assertEqual(len(saved), 2)
        self.assertFalse(marker)

    def test_detector_failure_does_not_end_loop(self):
        looks, saved, _, _, _ = self.run_loop(broken=True)
        self.assertEqual(len(looks), 28)
        self.assertEqual(len(saved), 2)

    def test_reconnect_gap_primes_instead_of_reporting_parked_cars(self):
        a, b = [(0.1, 0.1, 0.2, 0.2)], [(0.5, 0.5, 0.6, 0.6)]
        _, saved, _, _, _ = self.run_loop(positions=lambda t: a if t < 2 else b,
                                          visible=lambda t: t < 2 or t >= 65, duration=70)
        # Startup is an arrival; reconnect after a real gap must add no event.
        self.assertEqual([ev.start for ev, _ in saved], [self.start, self.start])
        self.assertEqual(len(self.memories), 4)
        for memory in self.memories[:2]:
            memory.prime.assert_not_called()
        for memory in self.memories[2:]:
            self.assertEqual(memory.prime.call_args_list, [mock.call(b)])

    def test_moving_vehicle_is_saved_after_confirmation(self):
        b = [(0.5, 0.5, 0.6, 0.6)]
        _, saved, _, _, _ = self.run_loop(positions=lambda t: [] if t < 2 else b, duration=16)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0][0].labels, {"car"})
        self.assertEqual(saved[0][0].people, 0)
        self.assertEqual(saved[0][0].start, datetime(2026, 10, 3, 12).timestamp() + 3)

    def test_offline_camera_still_closes_its_event(self):
        _, saved, _, _, _ = self.run_loop(visible=lambda t: t < 2)
        self.assertEqual(len(saved), 2)

    def test_snapshot_failure_does_not_end_detection(self):
        looks, saved, _, _, _ = self.run_loop(snapshot_error=True)
        self.assertEqual(len(looks), 28)
        self.assertEqual(saved, [])


class GuardRegressionTest(unittest.TestCase):
    run_loop = RunTest.run_loop

    def test_startup_vehicle_alerts_without_priming(self):
        self.run_loop(enabled=False, start_at=datetime(2026, 10, 3, 23), duration=2,
                      positions=lambda t: [(0.1, 0.1, 0.2, 0.2)])
        self.assertEqual(self.admissions, [("gate", 0, ["car"])])
        for memory in self.memories:
            memory.prime.assert_not_called()

    def test_hours_are_checked_for_each_camera_after_slow_inference(self):
        self.run_loop(enabled=False, start_at=datetime(2026, 10, 3, 5, 59, 59), duration=2,
                      script=lambda i, t: ([], 1.3) if i == 0 else (["person"], 0))
        self.assertEqual(self.admissions, [], "yard's first look is after 06:00")

    def test_quiet_to_guard_transition_is_handled_before_next_camera(self):
        _, saved, _, marker, _ = self.run_loop(
            start_at=datetime(2026, 10, 3, 21, 59, 59), duration=2,
            script=lambda i, t: (["person"], 1.3 if i == 0 else 0))
        self.assertEqual(self.admissions, [("yard", 1.3, ["person"]), ("gate", 1.55, ["person"])])
        self.assertEqual([ev.camera for ev, _ in saved], ["gate"])
        self.assertFalse(marker)

    def test_guard_to_quiet_transition_is_handled_before_next_camera(self):
        _, saved, _, marker, _ = self.run_loop(
            start_at=datetime(2026, 10, 3, 5, 59, 59), duration=2,
            script=lambda i, t: ([], 1.3) if i == 0 else (["person"], 0))
        self.assertEqual(self.admissions, [])
        self.assertEqual({ev.camera for ev, _ in saved}, {"gate", "yard"})
        self.assertTrue(all(ev.start >= self.start + 1.3 for ev, _ in saved))
        self.assertTrue(marker)

    def test_guard_never_primes_even_with_quiet_enabled_or_a_long_camera_gap(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.run_loop(enabled=enabled, start_at=datetime(2026, 10, 3, 23), duration=70,
                              positions=lambda t: [(t / 100, 0.1, t / 100 + .1, .2)],
                              visible=lambda t: t < 2 or t >= 65)
                for memory in self.memories:
                    memory.prime.assert_not_called()
                self.assertIsNone(self.loop_state["last_vehicle_look"])

    def test_quiet_vehicle_looks_do_not_consume_first_guard_arrival(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.run_loop(enabled=enabled, start_at=datetime(2026, 10, 3, 21, 59, 59, 500000), duration=3,
                              positions=lambda t: [(0.1, 0.1, 0.2, 0.2)])
                # 520aec5 first sees this car when the guard hours begin.
                self.assertEqual(self.admissions, [("gate", .5, ["car"])])

    def test_exactly_sixty_seconds_between_quiet_looks_does_not_prime(self):
        self.run_loop(duration=62, visible=lambda t: t == 0 or t >= 60,
                      positions=lambda t: [(0.1, 0.1, 0.2, 0.2)])
        for memory in self.memories:
            memory.prime.assert_not_called()

    def test_disabling_flushes_only_once_and_releases_quiet_state(self):
        flush = inf.QuietTracker.flush
        with mock.patch.object(inf.QuietTracker, "flush", autospec=True, side_effect=flush) as calls:
            self.run_loop(change=lambda t: t < 2, duration=5)
        self.assertEqual(calls.call_count, 2)
        for key in ("quiet", "quiet_vehicles", "quiet_look_ts", "last_vehicle_look", "default_retention"):
            self.assertIsNone(self.loop_state[key])

    def test_guard_admissions_match_520aec5_hand_written_script(self):
        # Baseline: gate's startup car reserves the only pending slot. Yard's
        # startup car is still remembered while waiting; it never alerts parked.
        # Yard's person then alerts. Gate's animal alerts when its cooldown ends.
        # Continued people alert only once each camera's 2-second cooldown ends.
        script = [(["car"], 0), (["car"], 0), (["person"], 0),
                  (["car"], 0), (["person"], 0), (["dog"], 0),
                  (["person"], 0), (["person"], 0)]
        self.run_loop(enabled=False, start_at=datetime(2026, 10, 3, 23), duration=3,
                      positions=lambda t: [(0.1, 0.1, 0.2, 0.2)],
                      script=lambda i, t: script[i] if i < len(script) else ([], 0),
                      alert_on=("person", "vehicle", "animal"), cooldown=2)
        self.assertEqual(self.admissions, [("gate", 0, ["car"]),
                                          ("yard", .25, ["person"]),
                                          ("gate", 2, ["dog"]),
                                          ("yard", 2.25, ["person"])])

    def test_disabled_quiet_state_is_not_allocated_or_flushed(self):
        with mock.patch.object(inf, "QuietTracker", wraps=inf.QuietTracker) as tracker, \
                mock.patch.object(inf.QuietTracker, "flush") as flush:
            self.run_loop(enabled=False, start_at=datetime(2026, 10, 3, 23), duration=2)
        tracker.assert_not_called()
        flush.assert_not_called()
        for key in ("quiet", "quiet_vehicles", "quiet_look_ts", "last_vehicle_look", "default_retention"):
            self.assertIsNone(self.loop_state.get(key), key)

    def marker_io(self, enabled, change=None):
        import builtins
        calls = []
        def spy(real):
            def record(path, *args, **kwargs):
                if ".registry" in str(path):
                    calls.append((real.__name__, threading.current_thread().name))
                return real(path, *args, **kwargs)
            return record
        with ExitStack() as stack:
            for obj, attr in ((os.path, "exists"), (os, "makedirs"), (os, "remove"), (builtins, "open")):
                stack.enter_context(mock.patch.object(obj, attr, side_effect=spy(getattr(obj, attr))))
            self.run_loop(enabled=enabled, change=change, duration=3)
        return calls

    def test_default_startup_does_not_even_check_the_marker(self):
        self.assertEqual(self.marker_io(False), [])

    def test_marker_io_runs_on_background_saver(self):
        calls = self.marker_io(True, change=lambda t: t < 2)
        self.assertTrue(calls)
        self.assertTrue(all(thread == "quiet-saver" for _, thread in calls), calls)

    def test_animals_are_guard_alerts_but_never_quiet_events(self):
        _, saved, _, _, _ = self.run_loop(script=lambda i, t: (["dog"], 0),
                                          alert_on=("animal",), duration=3)
        self.assertEqual(saved, [])
        self.run_loop(enabled=False, start_at=datetime(2026, 10, 3, 23),
                      script=lambda i, t: (["dog"], 0), alert_on=("animal",), duration=2)
        self.assertEqual(self.admissions, [("gate", 0, ["dog"]), ("yard", .25, ["dog"])])


class TrimQuietTest(unittest.TestCase):
    def clips(self, root):
        paths = []
        for i, stem in enumerate([quiet_stem("gate", 100), quiet_stem("gate", 200), "gate_150_alert"]):
            clip = Path(root, "clips", "gate", "2026-10-03", stem + ".mp4")
            meta = Path(root, "meta", "gate", "2026-10-03", stem + ".meta.json")
            for p in (clip, meta):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x" * (1000 if p == clip else 10))
                os.utime(p, (1000 + i, 1000 + i))
            paths.append((clip, meta))
        return paths

    def test_oldest_quiet_first_alerts_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.clips(root)
            self.assertEqual(quiet_stem("gate", 100.7), "gate_100_quiet")
            self.assertEqual(trim_quiet([root], 1500), 1)
            self.assertFalse(paths[0][0].exists() or paths[0][1].exists())
            self.assertTrue(all(p.exists() for pair in paths[1:] for p in pair))

    def test_failed_deletion_is_not_counted_as_freed_space(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.clips(root)
            remove = os.remove
            def fail_old(path):
                if str(path) == str(paths[0][0]):
                    raise PermissionError("busy")
                remove(path)
            with mock.patch.object(alert_clips.os, "remove", side_effect=fail_old):
                self.assertEqual(trim_quiet([root], 1000), 1)
            self.assertTrue(paths[0][0].exists())
            self.assertFalse(paths[1][0].exists())
            self.assertTrue(paths[2][0].exists())

    def test_disappearing_old_clip_is_already_freed(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.clips(root)
            remove = os.remove
            def disappear(path):
                if str(path) == str(paths[0][0]):
                    remove(path)  # another writer removed it after the scan
                    raise FileNotFoundError(path)
                remove(path)
            with mock.patch.object(alert_clips.os, "remove", side_effect=disappear):
                trim_quiet([root], 1000)
            self.assertTrue(paths[1][0].exists(), "the newer clip must survive")
            self.assertTrue(paths[2][0].exists())

    def test_cap_is_shared_across_live_and_archive(self):
        with tempfile.TemporaryDirectory() as root:
            live = Path(root, "live")
            archive = Path(root, "archive", "site")
            first, second = self.clips(live), self.clips(archive)
            self.assertEqual(trim_quiet([str(live), str(archive)], 2000), 2)
            self.assertFalse(first[0][0].exists() or second[0][0].exists())
            self.assertTrue(first[1][0].exists() and second[1][0].exists())

    def test_upload_expires_then_trims_before_any_network_setup(self):
        from home_guard_project.box import __main__ as main
        calls = []
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(main.sys, "argv", ["box", "upload"]), \
                mock.patch.object(main, "load_box_config", return_value=SimpleNamespace()), \
                mock.patch.object(main, "PRODUCTION_ARCHIVE_DIR", root), \
                mock.patch.object(main, "PRODUCTION_LIVE_DIR", "live"), \
                mock.patch.object(main, "expire_old_files", side_effect=lambda *a: calls.append("expire") or 0), \
                mock.patch.object(boxconfig, "load_box_settings", return_value={}), \
                mock.patch.object(alert_clips, "trim_quiet", side_effect=lambda *a: calls.append(a) or 0), \
                mock.patch("home_guard_project.s3_upload.config.load_config", side_effect=RuntimeError("offline")):
            Path(root, "site").mkdir()
            with self.assertRaisesRegex(RuntimeError, "offline"):
                main.main()
            self.assertEqual(calls, ["expire", (["live", os.path.join(root, "site")], 20_000_000_000)])


if __name__ == "__main__":
    unittest.main()
