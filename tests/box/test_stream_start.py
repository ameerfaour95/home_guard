"""A camera that does not answer never holds up the box's start (live 2026-10-11 16:34: after a power cut the recorder
pinged but served nothing; every reader waited 30 s to connect inside its constructor, 5 cameras x 2 streams, and the
owner's Telegram assistant, started after the readers, did not answer for over ten minutes)."""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from home_guard_project.data_collection import config, streams


class _HangingCapture:
    """cv2.VideoCapture of a recorder that is up but serves nothing: the open blocks until released."""

    def __init__(self, gate: threading.Event) -> None:
        gate.wait(5)

    def set(self, *_a) -> bool:
        return True

    def isOpened(self) -> bool:
        return False

    def read(self):
        return False, None

    def release(self) -> None:
        pass


class StreamStartTest(unittest.TestCase):
    def test_readers_are_made_at_once_and_connect_on_their_own_thread(self) -> None:
        cfg = config.Config()
        gate = threading.Event()
        opened = []

        def capture(src, *_a):
            opened.append(threading.current_thread() is not threading.main_thread())
            return _HangingCapture(gate)

        readers = []
        with mock.patch.object(streams.cv2, "VideoCapture", capture):
            t0 = time.monotonic()
            for cls in (streams.SubStreamThread, streams.MainStreamThread):
                for name in ("ch1", "ch2", "ch3"):
                    readers.append(cls(cfg, f"rtsp://10.0.0.9/{name}", name=name))
            made = time.monotonic() - t0
            try:
                self.assertLess(made, 1.0)                       # 6 readers, none waited for its camera
                deadline = time.monotonic() + 3
                while len(opened) < len(readers) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(len(opened), len(readers))      # each is connecting...
                self.assertTrue(all(opened))                     # ...on its own thread
                self.assertFalse(any(r.is_opened() for r in readers if hasattr(r, "is_opened")))
            finally:
                gate.set()
                for r in readers:
                    r.release()


class _Counting:
    """A main stream that always has a frame: counts decodes (grab) and conversions (retrieve)."""

    def __init__(self, src, api=None, params=()) -> None:
        self.params, self.grabs, self.retrieves = list(params), 0, 0

    def set(self, *_a) -> bool:
        return True

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        self.grabs += 1
        time.sleep(0.002)
        return True

    def retrieve(self):
        import numpy as np
        self.retrieves += 1
        return True, np.zeros((8, 8, 3), np.uint8)

    def release(self) -> None:
        pass


class MainStreamLoadTest(unittest.TestCase):
    """11.10: five 2592x1520 main streams decoded and converted in full held the box at 100% CPU, and the Eye's
    crops came out smeared (ghost people). Every frame is decoded, only the stored ones are converted, on the
    graphics chip when there is one."""

    def test_only_stored_frames_are_converted_and_the_chip_is_asked_for(self) -> None:
        cfg = config.Config()
        cfg.MAIN_STORE_FPS = 5.0
        made = []

        def capture(*a):
            made.append(_Counting(*a))
            return made[-1]

        with mock.patch.object(streams.cv2, "VideoCapture", capture):
            reader = streams.MainStreamThread(cfg, "rtsp://10.0.0.9/main", name="ch3")
            time.sleep(0.6)
            reader.release()
        cap = made[0]
        self.assertGreater(cap.grabs, 50)
        self.assertLessEqual(cap.retrieves, 5)                 # ~5 a second, not every decoded frame
        self.assertGreaterEqual(cap.retrieves, 2)
        self.assertEqual(len(reader.buf), cap.retrieves)
        pairs = dict(zip(cap.params[::2], cap.params[1::2]))
        self.assertEqual(pairs.get(streams.cv2.CAP_PROP_HW_ACCELERATION), streams.cv2.VIDEO_ACCELERATION_ANY)
        self.assertEqual(pairs.get(streams.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), int(cfg.OPEN_TIMEOUT_MSEC))


if __name__ == "__main__":
    unittest.main()
