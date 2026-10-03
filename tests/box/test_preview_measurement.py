import copy
import importlib.util
from pathlib import Path
import tempfile
import json
import time
import unittest

spec = importlib.util.spec_from_file_location("measure_on_box", Path(__file__).resolve().parents[2]/"docs/ui/measure_on_box.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class MeasurementTests(unittest.TestCase):
    def test_rates_use_cumulative_deltas_and_actual_duration(self):
        first = {"pid": 3, "started": 1, "monotonic": 10, "publisher_cpu_seconds": 2,
                 "detector_loops": 100, "published_roles": {"hero": 10, "thumbnail": 20},
                 "published": {"hero": 10, "garden": 20}, "rates": [12, 3], "cap_percent": 10}
        last = copy.deepcopy(first)
        last.update(monotonic=70, publisher_cpu_seconds=6.8, detector_loops=220,
                    published_roles={"hero": 730, "thumbnail": 200}, published={"hero": 730, "garden": 200})
        result = probe.summarize(first, last)
        self.assertEqual(result["publisher_one_core_percent"], 8)
        self.assertEqual(result["detector_loop_hz"], 2)
        self.assertEqual(result["hero_published_fps"], 12)
        self.assertEqual(result["per_camera_published_fps"]["garden"], 3)
        last["started"] = 11
        with self.assertRaisesRegex(RuntimeError, "restarted"): probe.summarize(first, last)

    def test_missing_instrumentation_and_stale_data_fail_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metrics.json"
            path.write_text(json.dumps({"updated": time.time(), "detector_instrumented": False}))
            with self.assertRaisesRegex(RuntimeError, "unavailable"): probe.read_metrics(path)
            path.write_text(json.dumps({"updated": time.time()-10, "detector_instrumented": True}))
            with self.assertRaisesRegex(RuntimeError, "stale"): probe.read_metrics(path)
