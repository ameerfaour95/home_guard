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


if __name__ == "__main__":
    unittest.main()
