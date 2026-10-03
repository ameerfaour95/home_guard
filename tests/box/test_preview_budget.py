import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from home_guard_project.box.preview import PreviewRateController, PreviewReader, PreviewWriter
from home_guard_project.box.inference_preview import adapt_stream, preview_enabled


class BudgetTests(unittest.TestCase):
    def test_backoff_recovery_and_floor(self):
        now = [0]
        controller = PreviewRateController(clock=lambda: now[0])
        names = ["hero", "thumb"]
        for expected in ((12, 3), (8, 2), (6, 2), (6, 2)):
            now[0] += 2.1
            controller.observe("hero", .04, names, "hero")
            self.assertEqual(controller.rates, expected)
        self.assertGreater(controller.estimate(names, "hero"), controller.cap)
        # Low costs alone cannot cause immediate oscillation.
        for _ in range(40):
            controller.observe("hero", .0001, names, "hero")
            controller.observe("thumb", .0001, names, "hero")
        for expected in ((8, 2), (12, 3), (16, 5), (16, 5)):
            now[0] += 5.1
            controller.observe("hero", .0001, names, "hero")
            self.assertEqual(controller.rates, expected)

    def test_budget_is_aggregate_and_checks_next_rate_before_recovery(self):
        now = [0]
        controller = PreviewRateController(clock=lambda: now[0])
        names = ["hero"] + [str(i) for i in range(5)]
        controller.costs = dict.fromkeys(names, .003)
        now[0] = 3
        controller.observe("hero", .003, names, "hero")
        self.assertEqual(controller.rates, (12, 3))
        now[0] = 30
        controller.observe("hero", .003, names, "hero")
        self.assertEqual(controller.rates, (12, 3))
        self.assertAlmostEqual(controller.estimate(names, "hero"), .091)

    def test_worker_overhead_is_included_in_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PreviewWriter(directory, enabled=True, cpu_cap_percent=10)
            writer._overhead_sample = (0, 0, 0)
            writer.worker_cpu_seconds = .4
            writer.offer_cpu_seconds = .02
            writer.encode_cpu_seconds = .1
            with patch("home_guard_project.box.preview.time.monotonic", return_value=2):
                writer.write_metrics()
            self.assertAlmostEqual(writer.rate_controller.overhead, .04)
            controller = writer.rate_controller
            controller.clock = lambda: 3
            controller.last_change = 0
            controller.observe("hero", .004, ["hero"], "hero")
            self.assertEqual(controller.rates, (12, 3))

    def test_setting_uses_existing_display_overlay_and_validates_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory)/"config.yaml"
            overlay.write_text("display:\n  preview_enabled: true\n  preview_cpu_cap_percent: 7\n")
            with patch.dict(os.environ, {"HOME_GUARD_CONFIG_OVERLAY": str(overlay)}):
                self.assertTrue(preview_enabled())
                writer = PreviewWriter(directory)
                self.assertEqual(writer.rate_controller.cap, .07)
        for invalid in (None, "bad", float("nan"), -1, 0, 101):
            self.assertEqual(PreviewRateController(invalid).cap, .1)

    def test_encode_samples_control_rates_and_hidden_still_publishes_nothing(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            reader = PreviewReader(directory)
            reader.touch("hero", visible=True, cameras=["hero", "thumb"])
            writer = PreviewWriter(directory, enabled=True)
            writer.rate_controller.last_change -= 3
            frame = np.zeros((20, 20, 3), np.uint8)
            with patch("home_guard_project.box.preview.time.perf_counter", side_effect=[0, .04]):
                self.assertTrue(writer.publish("hero", frame))
            self.assertEqual(writer.rate_controller.rates, (12, 3))
            self.assertAlmostEqual(writer.rate_controller.costs["hero"], .04)
            for level in range(4):
                writer.rate_controller.level = level
                reader.touch("hero", visible=False, cameras=[])
                writer.viewer_checked = float("-inf")
                with patch("cv2.imencode") as encode:
                    self.assertFalse(writer.publish("hero", frame))
                    self.assertFalse(writer.publish("thumb", frame))
                    encode.assert_not_called()

    def test_telemetry_counts_successes_and_exact_outer_detector_loops(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            writer = PreviewWriter(directory, enabled=True)
            class Stream:
                def __init__(self, name, url): self.name = name; self._running = True
                def read(self): return None
            adapter = adapt_stream(Stream, writer)
            hero, thumb = adapter("hero", "unused"), adapter("thumb", "unused")
            for _ in range(7): hero.read(); thumb.read()
            PreviewReader(directory).touch("hero", visible=True, cameras=["hero", "thumb"])
            frame = np.zeros((20, 20, 3), np.uint8)
            writer.publish("hero", frame)
            writer.publish("thumb", frame)
            writer.metrics_written = float("-inf")
            writer.write_metrics()
            data = json.loads((Path(directory)/"publisher_metrics.json").read_text())
            self.assertEqual(data["detector_loops"], 7)
            self.assertTrue(data["detector_instrumented"])
            self.assertEqual(data["published_roles"], {"hero": 1, "thumbnail": 1})
            self.assertGreaterEqual(data["publisher_cpu_seconds"], 0)
