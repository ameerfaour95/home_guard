"""Collector/inference parity, with synthetic cameras and a local GPT transport stub."""
from __future__ import annotations

import ast
import base64
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from home_guard_project.box import alert_clips, inference as inf, make_bundle
from home_guard_project.data_collection import config, streams, vlm_crop
import test_vlm_crop_parity as golden


ANSWER = {"summary": "a person walks", "label": "normal", "people": 1,
          "vehicle_moving": False, "animals": 0, "why": "", "summary_owner": ""}


def backend():
    """Exercise the real analyze/encoding path without constructing an OpenAI client."""
    obj = inf.GptBackend.__new__(inf.GptBackend)
    obj._model = obj.model_name = "gpt-local-test"
    obj._response_format = inf.VLM_RESPONSE_FORMAT
    obj._complete = mock.Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(ANSWER)))]))
    return obj


def readers(sub=None, main=None):
    if sub is None:
        sub = [np.full((48, 64, 3), i, np.uint8) for i in range(100)]
    if main is None:
        main = [np.full((96, 128, 3), i, np.uint8) for i in range(50)]
    return (SimpleNamespace(get_clip_last_seconds=mock.Mock(return_value=(sub, 1000., 1010., 10.))),
            SimpleNamespace(get_clip_frames=mock.Mock(return_value=(main, 1000., 1010., 5.))))


def job():
    return inf.AlertJob("cam", "cam_1004_alert", 1004., labels=["person"])


def run_worker(task, frames, model=None):
    model = model or backend()
    with mock.patch.object(inf, "dispatch_alert", return_value={"sent": True}) as dispatch, \
            mock.patch.object(inf, "owner_language", return_value="en"):
        inf._worker(model, {}, {}, inf.AlertSettings(), "cam", frames, job=task)
    return model, dispatch


class CropParityTest(unittest.TestCase):
    def test_sent_pixels_jpegs_and_both_saved_crops_equal_collector(self):
        reference, fps, reference_meta = golden.collector_crop_frames()
        cfg = golden.cfg("")
        cfg.CLIP_SECONDS, cfg.VLM_SAMPLE_FPS = 10., 1.
        sub, main = readers(golden.sub_frames(), golden.main_frames())
        task = job()
        detector = mock.Mock(side_effect=lambda frame, **kw: golden.fake_detector(
            frame, **{k: v for k, v in kw.items() if k != "device"}))
        frames, clip = inf._prepare_alert(task, cfg, detector, sub, main, {"device": "intel:gpu"})
        expected = vlm_crop.sample_for_vlm(reference, fps=fps, sample_fps=cfg.VLM_SAMPLE_FPS)
        self.assertEqual(golden.digest(frames), golden.digest(expected))
        self.assertEqual(golden.digest(task.crop.frames), golden.GOLDEN_SHA256)
        self.assertEqual(detector.call_count, 20)
        # The crop's own YOLO looks are kept, normalised, for the scene map (no extra detector call).
        self.assertEqual(len(task.scene_looks), 20)
        self.assertTrue(all(1000. <= ts <= 1010. for ts, _ in task.scene_looks))
        people = [d for _, dets in task.scene_looks for d in dets]
        self.assertTrue(people and all(d[0] == 0 and 0 <= d[2] <= d[4] <= 1 for d in people))
        for call in detector.call_args_list:
            self.assertEqual(call.kwargs, dict(verbose=False, conf=cfg.YOLO_TRIGGER_CONF,
                                               imgsz=cfg.YOLO_IMGSZ, device="intel:gpu"))
        sub.get_clip_last_seconds.assert_called_once_with(cfg.CLIP_SECONDS)
        main.get_clip_frames.assert_called_once_with(cfg.CLIP_SECONDS)
        self.assertEqual(golden.digest([f for _, f in clip]), golden.digest(golden.sub_frames()))

        model, dispatch = run_worker(task, frames)
        content = model._complete.call_args.args[0]
        sent = [base64.b64decode(p["image_url"]["url"].split(",", 1)[1])
                for p in content if p["type"] == "image_url"]
        self.assertEqual(sent, inf._jpegs(expected))
        self.assertEqual(task.teacher["frames"], sent)
        self.assertEqual(dispatch.call_args.kwargs["image"], inf.frame_to_jpeg_bytes(clip[-1][1]))

        written = {}
        real_writer = cv2.VideoWriter

        class Writer:
            def __init__(self, path, *args):
                self.path = path
                self.writer = real_writer(path, *args)
                written[path] = []

            def isOpened(self):
                return self.writer.isOpened()

            def write(self, frame):
                written[self.path].append(frame.copy())
                self.writer.write(frame)

            def release(self):
                self.writer.release()

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(alert_clips, "_to_h264", return_value=False), \
                mock.patch("cv2.VideoWriter", Writer):
            production, training = Path(tmp) / "production", Path(tmp) / "training"
            inf._save_clip(task, clip, str(production), str(training))
            for root in (production, training):
                meta = json.loads(next(root.rglob("*.meta.json")).read_text())
                self.assertEqual(meta["vlm_input"], "crop")
                crop_meta = meta["vlm_crop"]
                self.assertEqual({k: v for k, v in crop_meta.items() if k != "vlm_crop_path"},
                                 {k: v for k, v in reference_meta.items() if k != "vlm_crop_path"})
                crop_path = root / crop_meta["vlm_crop_path"]
                self.assertTrue(crop_path.is_file())
                self.assertEqual(golden.digest(written[str(crop_path)]), golden.GOLDEN_SHA256)
                clip_path = root / meta["clip_path"].replace("\\", "/")
                self.assertEqual(golden.digest(written[str(clip_path)]), golden.digest(golden.sub_frames()))
                cap = cv2.VideoCapture(str(crop_path))
                try:
                    self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), golden.N_MAIN)
                    self.assertEqual(cap.get(cv2.CAP_PROP_FPS), fps)
                finally:
                    cap.release()
            kept = json.loads(next(training.rglob("*.meta.json")).read_text())
            self.assertEqual([(training / p.replace("\\", "/")).read_bytes()
                              for p in kept["teacher"]["input_frames"]], sent)

    def test_window_constants_match_collector_config(self):
        self.assertEqual(alert_clips.PRE_SECONDS, 4)
        self.assertEqual(alert_clips.POST_SECONDS, 6)
        self.assertEqual(alert_clips.PRE_SECONDS + alert_clips.POST_SECONDS, config.Config().CLIP_SECONDS)
        self.assertEqual(config.Config().CLIP_SECONDS, 10)

    def test_retired_box_keys_are_ignored_even_if_unparseable(self):
        self.assertEqual(inf.AlertSettings.from_box_settings(
            {"alert_clip_frames": "retired", "alert_frame_interval_sec": None}), inf.AlertSettings())


class TimingAndFallbackTest(unittest.TestCase):
    def test_run_reserves_one_slot_across_cameras_during_post_roll(self):
        from home_guard_project.box import ai_status, boxconfig, camera_alerts, telegram_agent
        from home_guard_project.box.brain import mode

        cfg, model = config.Config(), backend()
        cfg.CAMERAS = {"cam": "synthetic-sub", "yard": "synthetic-yard"}
        sub, _ = readers()
        adapters = {name: SimpleNamespace(read=lambda: np.zeros((48, 64, 3), np.uint8),
                                          last_ts=1011., sub_cap=sub) for name in cfg.CAMERAS}
        clock = [1004.]
        calls, thoughts = [], []
        status = mock.Mock()
        status.thinking.side_effect = lambda *a, **kw: thoughts.append((a[0], clock[0]))
        original_complete = model._complete
        model._complete = lambda *a: (calls.append(clock[0]), original_complete(*a))[1]
        detector = mock.Mock(return_value=[golden._Result([])])
        detector.predict.return_value = [golden._Result([])]

        class StopLoop(Exception):
            pass

        class InlineThread:
            def __init__(self, target, args, daemon):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

            def is_alive(self):
                return False

        def sleep(_):
            clock[0] += .25
            if clock[0] > 1010.5:
                raise StopLoop

        with ExitStack() as stack:
            patches = [
                mock.patch("dotenv.load_dotenv"),
                mock.patch.object(boxconfig, "load_box_settings", return_value={}),
                mock.patch.object(config, "load_config", return_value=cfg),
                mock.patch.object(inf, "make_backend", return_value=model),
                mock.patch.object(inf, "load_detector", return_value=(detector, None)),
                mock.patch.object(inf, "_camera_streams", return_value=(adapters, dict.fromkeys(cfg.CAMERAS))),
                mock.patch.object(inf, "LiveSettings"),
                mock.patch.object(inf, "filter_by_thresholds", side_effect=lambda r, *a: r),
                mock.patch.object(inf, "detect_trigger", return_value=(True, False, ["person"])),
                mock.patch.object(inf, "vehicle_boxes", return_value=[]),
                mock.patch.object(ai_status, "objects_from_result", return_value=[]),
                mock.patch.object(ai_status, "AiStatus", return_value=status),
                mock.patch.object(mode, "ModeWatch", return_value=mock.Mock(due=lambda now: False)),
                mock.patch.object(camera_alerts, "LiveCameraAlerts", return_value=SimpleNamespace(
                    check=lambda now: False, overrides={}, sensitivity={},
                    thresholds_for=lambda name, defaults: defaults, for_camera=lambda name, defaults: defaults)),
                mock.patch.object(telegram_agent, "start", return_value=None),
                mock.patch.object(inf, "dispatch_alert", return_value={"sent": True}),
                mock.patch.object(inf, "_save_clip"),
                mock.patch.object(inf.threading, "Thread", InlineThread),
                mock.patch.object(inf.time, "time", side_effect=lambda: clock[0]),
                mock.patch.object(inf.time, "sleep", side_effect=sleep),
            ]
            for patch in patches:
                stack.enter_context(patch)
            with self.assertRaises(StopLoop):
                inf.run()
        self.assertEqual(calls, [1010.])
        self.assertEqual(thoughts, [("cam", 1004.), ("yard", 1010.)])

    def test_vlm_called_once_only_after_post_roll(self):
        cfg, task, model = config.Config(), job(), backend()
        sub, main = readers()
        pending = [task]
        saved = []

        class InlineThread:
            def __init__(self, target, args, daemon):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

        def tick(now):
            return inf._start_due_alerts(pending, now, cfg, mock.Mock(return_value=[golden._Result([])]),
                                         {"cam": SimpleNamespace(sub_cap=sub)}, {"cam": main}, {},
                                         model, {}, {}, inf.AlertSettings(), None, None, "prod", "train")

        with mock.patch.object(inf.threading, "Thread", InlineThread), \
                mock.patch.object(inf, "_save_clip", side_effect=lambda *args: saved.append(args)), \
                mock.patch.object(inf, "dispatch_alert", return_value={"sent": True}), \
                mock.patch.object(inf, "owner_language", return_value="en"):
            self.assertIsNone(tick(task.ts))
            self.assertIsNone(tick(task.ts + alert_clips.POST_SECONDS - .001))
            model._complete.assert_not_called()
            sub.get_clip_last_seconds.assert_not_called()
            self.assertEqual(pending, [task])
            self.assertIsNotNone(tick(task.ts + alert_clips.POST_SECONDS))
            self.assertIsNone(tick(task.ts + alert_clips.POST_SECONDS + 1))
        model._complete.assert_called_once()
        self.assertEqual(len(saved), 1)
        self.assertEqual(pending, [])
        self.assertTrue(task.ready.is_set())

    def test_no_detection_uses_whole_window_and_dispatches_and_saves_reason(self):
        cfg, task = config.Config(), job()
        sub, main = readers()
        with self.assertLogs(inf.log, level="WARNING") as logs:
            frames, clip = inf._prepare_alert(task, cfg, mock.Mock(return_value=[golden._Result([])]),
                                              sub, main, {})
        self.assertIn("whole_frame_fallback", " ".join(logs.output))
        expected = vlm_crop.sample_for_vlm(sub.get_clip_last_seconds.return_value[0], 10, cfg.VLM_SAMPLE_FPS)
        self.assertEqual(golden.digest(frames), golden.digest(expected))
        self.assertEqual(task.input_meta["vlm_input"], "whole_frame_fallback")
        self.assertIn("no_trigger_class_detection", task.input_meta["vlm_fallback_reason"])
        _, dispatch = run_worker(task, frames)
        dispatch.assert_called_once()
        self.assertEqual(task.alert["dispatch"], {"sent": True})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(alert_clips, "_to_h264", return_value=False):
            inf._save_clip(task, clip, str(Path(tmp) / "prod"), str(Path(tmp) / "train"))
            metas = list(Path(tmp).rglob("*.meta.json"))
            self.assertEqual(len(metas), 2)
            for path in metas:
                meta = json.loads(path.read_text())
                self.assertEqual(meta["vlm_input"], "whole_frame_fallback")
                self.assertEqual(meta["vlm_fallback_reason"], task.input_meta["vlm_fallback_reason"])
                self.assertNotIn("vlm_crop", meta)

    def test_missing_short_and_failed_main_streams_keep_alert(self):
        for mode in ("absent", "short", "read_error", "detector_error"):
            with self.subTest(mode=mode):
                task, cfg = job(), config.Config()
                sub, main = readers(main=[])
                detector = mock.Mock(side_effect=RuntimeError("detector unavailable"))
                if mode == "absent":
                    main = None
                elif mode == "read_error":
                    main.get_clip_frames.side_effect = RuntimeError("decode failed")
                elif mode == "detector_error":
                    sub, main = readers()
                frames, _ = inf._prepare_alert(task, cfg, detector, sub, main, {})
                self.assertEqual(task.input_meta["vlm_input"], "whole_frame_fallback")
                self.assertTrue(frames)
                _, dispatch = run_worker(task, frames)
                dispatch.assert_called_once()

    def test_empty_sub_ring_retains_trigger_snapshot(self):
        task, cfg = job(), config.Config()
        task.snapshot = np.full((48, 64, 3), 77, np.uint8)
        sub, main = readers(sub=[])
        frames, clip = inf._prepare_alert(task, cfg, mock.Mock(), sub, main, {})
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(clip), 1)
        self.assertEqual(task.input_meta["vlm_fallback_reason"], "too_few_sub_frames")
        _, dispatch = run_worker(task, frames)
        dispatch.assert_called_once()

    def test_trigger_before_window_closes_survives_six_second_wait(self):
        from datetime import datetime
        task = job()
        task.ts = datetime(2026, 10, 3, 5, 59, 58).timestamp()
        task.input_meta = {"vlm_input": "whole_frame_fallback"}
        model = backend()
        with mock.patch.object(inf, "datetime", wraps=datetime) as clock, \
                mock.patch.object(inf, "dispatch_alert", return_value={"sent": True}) as dispatch, \
                mock.patch.object(inf, "owner_language", return_value="en"):
            clock.now.return_value = datetime(2026, 10, 3, 6, 0, 4)
            inf._worker(model, {}, {}, inf.AlertSettings(alert_start_hour=22, alert_end_hour=6),
                        "cam", [], job=task)
        model._complete.assert_called_once()
        dispatch.assert_called_once()


class SharedReadersTest(unittest.TestCase):
    def test_scene_map_black_areas_mask_both_readers(self):
        cfg = config.Config()
        cfg.CAMERAS = {"cam": "synthetic-sub"}
        cfg.CAMERAS_MAIN = {"cam": "synthetic-main"}
        cfg.ROI_BLACK = {"cam": [[(.5, 0.), (1., 0.), (1., 1.), (.5, 1.)]]}
        with mock.patch("cv2.VideoCapture"), mock.patch.object(streams.threading, "Thread"):
            subs, mains = inf._camera_streams(cfg)
        frame = np.full((48, 64, 3), 255, np.uint8)
        for reader in (subs["cam"].sub_cap, mains["cam"]):
            masked = reader.mask.apply(frame)
            self.assertEqual(masked[:, 40:].max(), 0)
            self.assertEqual(masked[:, :20].min(), 255)

    def test_two_connections_per_camera_with_same_cfg_and_masks(self):
        cfg = config.Config()
        cfg.CAMERAS = {"cam": "synthetic-sub", "yard": "synthetic-yard-sub"}
        cfg.CAMERAS_MAIN = {"cam": "synthetic-main", "yard": "synthetic-yard-main"}
        cfg.ROI_ZONES = {"cam": [(0., 0.), (.5, 0.), (.5, 1.), (0., 1.)]}
        with mock.patch("cv2.VideoCapture") as capture, mock.patch.object(streams.threading, "Thread"):
            subs, mains = inf._camera_streams(cfg)
            self.assertEqual(capture.call_count, 4)
            self.assertEqual([c.args[0] for c in capture.call_args_list],
                             ["synthetic-sub", "synthetic-main", "synthetic-yard-sub", "synthetic-yard-main"])
            for name in cfg.CAMERAS:
                self.assertIsInstance(subs[name].sub_cap, streams.SubStreamThread)
                self.assertIsInstance(mains[name], streams.MainStreamThread)
                self.assertIs(subs[name].sub_cap.cfg, cfg)
                self.assertIs(mains[name].cfg, cfg)
                self.assertEqual(subs[name].sub_cap.store_interval, 1 / cfg.STORE_FPS)
                self.assertEqual(mains[name]._jpeg_params[-1], cfg.MAIN_JPEG_QUALITY)
                self.assertEqual(subs[name].last_ts, 0.)
            frame = np.full((48, 64, 3), 255, np.uint8)
            for reader in (subs["cam"].sub_cap, mains["cam"]):
                masked = reader.mask.apply(frame)
                self.assertEqual(masked[:, 40:].max(), 0)
                self.assertEqual(masked[:, :20].min(), 255)
            sub = subs["cam"].sub_cap
            sub.latest_frame = frame
            sub.buf.append((123., b"jpeg"))
            np.testing.assert_array_equal(subs["cam"].read(), frame)
            self.assertEqual(subs["cam"].last_ts, 123.)
            # Reconnect resets the collector's freeze timer but must not hide an outage.
            sub.last_frame_ts = 999.
            self.assertEqual(subs["cam"].last_ts, 123.)

    def test_disabled_main_stream_opens_only_sub(self):
        cfg = config.Config()
        cfg.CAMERAS, cfg.CAMERAS_MAIN = {"cam": "synthetic-sub"}, {"cam": "synthetic-main"}
        cfg.MAIN_STREAM_ENABLED = False
        with mock.patch("cv2.VideoCapture") as capture, mock.patch.object(streams.threading, "Thread"):
            _, mains = inf._camera_streams(cfg)
        capture.assert_called_once()
        self.assertIsNone(mains["cam"])


class BundleImportsTest(unittest.TestCase):
    def test_all_collector_imports_ship_in_bundle_including_transitive_modules(self):
        root = Path(__file__).resolve().parents[2]
        dc = root / "home_guard_project" / "data_collection"
        todo = [dc / "data_collection.py", root / "home_guard_project/box/inference.py"]
        visited = set()
        while todo:
            path = todo.pop()
            if path in visited:
                continue
            visited.add(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        names.append(node.module.split(".")[-1])
                    else:
                        names.extend(a.name for a in node.names)
                elif isinstance(node, ast.Import):
                    names.extend(a.name.split(".")[-1] for a in node.names)
                for name in names:
                    candidate = dc / (name + ".py")
                    if candidate.is_file():
                        rel = candidate.relative_to(root).as_posix()
                        self.assertIn(rel, make_bundle.INCLUDE_FILES, f"{path.name} imports {rel}")
                        todo.append(candidate)
        self.assertIn(dc / "streams.py", visited)
        self.assertIn(dc / "vlm_crop.py", visited)


if __name__ == "__main__":
    unittest.main()
