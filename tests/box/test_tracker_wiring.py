"""The tracker in the guard loop (box/inference.py): every look feeds it, an alert job takes its facts, ZONE FACTS
prefer its tracks, and nothing it does can stop an alert."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np

from home_guard_project.box import alert_clips
from home_guard_project.box import inference as inf
from home_guard_project.box import scene_map as sm
from home_guard_project.box import tracker as tr
from home_guard_project.data_collection import config
import test_vlm_crop_parity as golden

T = 1004.0


def person(fx, fy):
    return (0, 0.8, fx - 0.03, fy - 0.2, fx + 0.03, fy)


def fed_registry(points, start=T - 30, step=0.5):
    reg = tr.TrackerRegistry(scene_map_loader=lambda c: None)
    ts = start
    for x, y in points:
        reg.update("cam", ts, [person(x, y)])
        ts += step
    return reg


class AttachTest(unittest.TestCase):
    def test_the_job_takes_the_facts_of_its_window(self) -> None:
        reg = fed_registry([(0.5, 0.9)] * 80)                       # in view since T - 30
        job = inf.AlertJob("cam", "cam_1004_alert", T, labels=["person"], input_meta={"vlm_input": "crop"})
        inf._attach_tracker(job, reg, T + alert_clips.POST_SECONDS)
        self.assertEqual(job.tracker_facts["people"], 1)
        self.assertGreaterEqual(job.tracker_facts["time_in_view_s"], 30)
        self.assertTrue(job.tracker_line.startswith("TRACKER FACTS (from code): person 1 in view"))
        self.assertEqual(job.tracker["window"], [T - alert_clips.PRE_SECONDS, T + alert_clips.POST_SECONDS])
        self.assertEqual(job.input_meta["tracker"], job.tracker)
        self.assertEqual(job.input_meta["vlm_input"], "crop")
        self.assertEqual(len(job.tracker_tracks), 1)
        self.assertTrue(all(T - alert_clips.PRE_SECONDS <= p[0] for p in job.tracker_tracks[0].points))

    def test_nothing_tracked_keeps_the_record_but_no_case_memory_dict(self) -> None:
        job = inf.AlertJob("cam", "s", T)
        inf._attach_tracker(job, tr.TrackerRegistry(scene_map_loader=lambda c: None), T + 6)
        self.assertIsNone(job.tracker_facts)
        self.assertEqual(job.tracker_line, "")
        self.assertEqual(job.input_meta["tracker"]["people"], [])

    def test_no_tracker_or_a_broken_one_leaves_the_job_alone(self) -> None:
        job = inf.AlertJob("cam", "s", T)
        inf._attach_tracker(job, None, T + 6)
        self.assertEqual((job.tracker_facts, job.tracker, job.input_meta), (None, {}, {}))
        broken = mock.Mock()
        broken.facts.side_effect = RuntimeError("boom")
        with self.assertLogs("box.inference", level="WARNING"):
            inf._attach_tracker(job, broken, T + 6)
        self.assertEqual((job.tracker_facts, job.tracker, job.input_meta), (None, {}, {}))


class StartDueAlertsTest(unittest.TestCase):
    def test_facts_are_attached_before_the_worker_and_the_clip_writer_start(self) -> None:
        cfg = config.Config()
        sub = SimpleNamespace(get_clip_last_seconds=mock.Mock(
            return_value=([np.zeros((48, 64, 3), np.uint8)] * 20, 1000., 1010., 2.)))
        job = inf.AlertJob("cam", "cam_1004_alert", T, labels=["person"])
        seen = {}

        class InlineThread:
            def __init__(self, target, args, daemon):
                self.target, self.args = target, args

            def start(self):
                task = self.args[7] if self.target is inf._worker else self.args[0]
                seen[self.target.__name__] = "tracker" in task.input_meta

        reg = fed_registry([(0.5, 0.9)] * 80)
        with mock.patch.object(inf.threading, "Thread", InlineThread):
            inf._start_due_alerts([job], T + 6, cfg, mock.Mock(return_value=[golden._Result([])]),
                                  {"cam": SimpleNamespace(sub_cap=sub)}, {"cam": None}, {}, mock.Mock(), {}, {},
                                  inf.AlertSettings(), None, None, "prod", "train", trackers=reg)
        self.assertEqual(seen, {"_worker": True, "_save_clip": True})
        self.assertEqual(job.tracker_facts["people"], 1)


class SceneTest(unittest.TestCase):
    def test_live_tracks_win_over_the_crop_looks(self) -> None:
        scene = sm.SceneMap("cam", areas=(
            sm.Area("yard", sm.MINE, "yard", ((0, 0), (0.5, 0), (0.5, 1), (0, 1))),
            sm.Area("road", sm.WATCH, "street", ((0.5, 0), (1, 0), (1, 1), (0.5, 1)))))
        looks = [(T, [person(0.7, 0.6)]), (T + 1, [person(0.72, 0.6)])]
        tracks = [sm.Track("person", [(T - 4, 0.2, 0.6), (T + 1, 0.25, 0.6)])]
        with mock.patch.object(sm, "load_scene_map", return_value=scene):
            _, from_looks = inf._scene("cam", looks)
            _, from_tracks = inf._scene("cam", looks, tracks)
            _, empty_tracks = inf._scene("cam", looks, [])
        self.assertEqual(from_looks.ground, "public")
        self.assertEqual(from_tracks.ground, "mine")
        self.assertEqual(empty_tracks.ground, "public")


class TeacherRecordTest(unittest.TestCase):
    def test_the_tracker_record_reaches_the_meta_twice(self) -> None:
        record = {"version": "tf1", "line": "", "people": []}
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = alert_clips.write_alert_clip(
                tmp, "cam", "cam_1004_alert", [(1000.0 + i, np.zeros((48, 64, 3), np.uint8)) for i in range(3)],
                {"labels": ["person"]}, h264=False,
                teacher={"model": "m", "prompt_version": "v", "prompt": "p", "frames": [], "raw": "{}",
                         "parsed": {}, "tracker": record},
                extra={"tracker": record})
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        self.assertEqual(meta["tracker"], record)
        self.assertEqual(meta["teacher"]["tracker"], record)


class RunLoopTest(unittest.TestCase):
    """inference.run() with synthetic cameras: every look feeds the camera's tracker, and a tracker that raises
    never stops the loop."""

    def run_loop(self, update_side_effect=None):
        from home_guard_project.box import ai_status, boxconfig, camera_alerts, telegram_agent
        from home_guard_project.box.brain import mode

        cfg = config.Config()
        cfg.CAMERAS = {"cam": "synthetic-sub"}
        sub = SimpleNamespace(get_clip_last_seconds=mock.Mock(return_value=([], 0., 0., 2.)))
        clock = [1000.]
        adapters = {"cam": SimpleNamespace(read=lambda: np.zeros((48, 64, 3), np.uint8),
                                           last_ts=1000., sub_cap=sub)}
        adapters["cam"].last_ts = 10_000.0      # never frozen
        detector = mock.Mock()
        detector.predict.return_value = [golden._Result([golden._Box(0, (16, 8, 32, 40))])]
        registry = mock.Mock()
        if update_side_effect is not None:
            registry.update.side_effect = update_side_effect

        class StopLoop(Exception):
            pass

        def sleep(_):
            clock[0] += .5
            if clock[0] > 1003:
                raise StopLoop

        with ExitStack() as stack:
            for patch in [
                mock.patch("dotenv.load_dotenv"),
                mock.patch.object(boxconfig, "load_box_settings", return_value={}),
                mock.patch.object(config, "load_config", return_value=cfg),
                mock.patch.object(inf, "make_backend", return_value=mock.Mock()),
                mock.patch.object(inf, "load_detector", return_value=(detector, None)),
                mock.patch.object(inf, "_camera_streams", return_value=(adapters, {"cam": None})),
                mock.patch.object(inf, "LiveSettings"),
                mock.patch.object(inf, "filter_by_thresholds", side_effect=lambda r, *a: r),
                mock.patch.object(inf, "detect_trigger", return_value=(False, False, [])),
                mock.patch.object(inf, "vehicle_boxes", return_value=[]),
                mock.patch.object(inf, "start_case_memory", return_value=False),
                mock.patch.object(ai_status, "AiStatus", return_value=mock.Mock()),
                mock.patch.object(ai_status, "objects_from_result", return_value=[]),
                mock.patch.object(mode, "ModeWatch", return_value=mock.Mock(due=lambda now: False)),
                mock.patch.object(camera_alerts, "LiveCameraAlerts", return_value=SimpleNamespace(
                    check=lambda now: False, overrides={}, sensitivity={},
                    thresholds_for=lambda name, defaults: defaults, for_camera=lambda name, defaults: defaults)),
                mock.patch.object(telegram_agent, "start", return_value=None),
                mock.patch.object(tr, "TrackerRegistry", return_value=registry),
                mock.patch.object(inf.time, "time", side_effect=lambda: clock[0]),
                mock.patch.object(inf.time, "sleep", side_effect=sleep),
            ]:
                stack.enter_context(patch)
            with self.assertRaises(StopLoop):
                inf.run()
        return registry, detector

    def test_every_look_feeds_the_tracker(self) -> None:
        registry, detector = self.run_loop()
        self.assertEqual(registry.update.call_count, detector.predict.call_count)
        self.assertGreaterEqual(registry.update.call_count, 5)
        camera, ts, dets = registry.update.call_args.args
        self.assertEqual(camera, "cam")
        self.assertEqual(dets, [(0, 0.0, 0.25, 0.1667, 0.5, 0.8333)])

    def test_a_tracker_that_raises_never_stops_the_looks(self) -> None:
        registry, detector = self.run_loop(update_side_effect=RuntimeError("boom"))
        self.assertGreaterEqual(detector.predict.call_count, 5)
        self.assertEqual(registry.update.call_count, detector.predict.call_count)


if __name__ == "__main__":
    unittest.main()
