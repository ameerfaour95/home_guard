import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from home_guard_project.data_collection import detector as det


class FakeModel:
    def __init__(self, path="m.pt", task=None):
        self.path, self.calls, self.busy, self.overlap = path, [], False, False

    def __call__(self, source, **kwargs):
        if self.busy:
            self.overlap = True
        self.busy = True
        time.sleep(0.01)
        self.calls.append(kwargs)
        self.busy = False
        return ["result"]

    def export(self, **kwargs):
        raise AssertionError("no export expected")


class DetectorTest(unittest.TestCase):
    def test_the_graphics_chip_device_is_passed_on_every_call(self):
        model = FakeModel()
        d = det.Detector(model, "intel:gpu")
        self.assertEqual(d("frame", conf=0.5), ["result"])
        self.assertEqual(model.calls, [{"conf": 0.5, "device": "intel:gpu"}])
        self.assertTrue(d.on_graphics_chip)

    def test_on_the_cpu_no_device_is_forced(self):
        model = FakeModel()
        d = det.Detector(model)
        d.predict("frame", verbose=False)
        self.assertEqual(model.calls, [{"verbose": False}])
        self.assertFalse(d.on_graphics_chip)

    def test_two_threads_never_run_the_model_at_once(self):
        # The collector's crop thread shares the detector with the main loop.
        model = FakeModel()
        d = det.Detector(model, "intel:gpu")
        threads = [threading.Thread(target=lambda: [d("f") for _ in range(5)]) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(model.calls), 20)
        self.assertFalse(model.overlap)


class LoadDetectorTest(unittest.TestCase):
    def test_no_graphics_chip_stays_on_the_cpu(self):
        openvino = SimpleNamespace(Core=lambda: SimpleNamespace(available_devices=["CPU"]))
        with mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeModel), "openvino": openvino}):
            d = det.load_detector("yolo11s.pt")
        self.assertFalse(d.on_graphics_chip)
        self.assertEqual(d.model.path, "yolo11s.pt")

    def test_cpu_setting_never_tries_the_graphics_chip(self):
        core = mock.Mock()
        with mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeModel),
                                           "openvino": SimpleNamespace(Core=core)}):
            d = det.load_detector("yolo11s.pt", "cpu")
        core.assert_not_called()
        self.assertFalse(d.on_graphics_chip)


if __name__ == "__main__":
    unittest.main()
