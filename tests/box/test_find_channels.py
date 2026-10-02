from __future__ import annotations

import unittest
from unittest import mock

from home_guard_project.box import find_cameras

PATTERNS = [{"name": "A", "tpl": "/a/{ch}0{stream}"}, {"name": "Unicast", "tpl": "/unicast/c{ch}/s{stream}/live"}]


class ChannelSearchTest(unittest.TestCase):
    """A recorder whose first channel is empty still gives up its other channels."""

    def _channels(self, live_paths):
        asked = []

        def exists(host, port, user, password, path, timeout=5.0):
            asked.append(path)
            return path in live_paths

        with mock.patch.object(find_cameras, "rtsp_stream_state", side_effect=exists):
            return find_cameras.rtsp_channels("192.168.0.139", 554, "admin", "x", PATTERNS, 6, 0), asked

    def test_channel_one_may_be_empty(self) -> None:
        (pattern, hits), _ = self._channels({"/unicast/c2/s0/live", "/unicast/c3/s0/live", "/unicast/c6/s0/live"})
        self.assertEqual(pattern["name"], "Unicast")
        self.assertEqual(hits, [(2, "/unicast/c2/s0/live"), (3, "/unicast/c3/s0/live"), (6, "/unicast/c6/s0/live")])

    def test_gaps_do_not_stop_the_search(self) -> None:
        (_, hits), asked = self._channels({"/unicast/c1/s0/live", "/unicast/c6/s0/live"})
        self.assertEqual([channel for channel, _ in hits], [1, 6])
        self.assertIn("/unicast/c4/s0/live", asked)

    def test_nothing_anywhere(self) -> None:
        (pattern, hits), asked = self._channels(set())
        self.assertEqual((pattern, hits), (None, []))
        self.assertEqual(len(asked), 12)          # every channel of every pattern was asked once

    def test_a_device_that_does_not_answer_is_given_up_after_two_tries(self) -> None:
        asked = []

        def silent(host, port, user, password, path, timeout=5.0):
            asked.append(path)
            return None

        with mock.patch.object(find_cameras, "rtsp_stream_state", side_effect=silent):
            with self.assertRaises(find_cameras.DeviceSilent):
                find_cameras.rtsp_channels("192.168.0.109", 554, "admin", "x", PATTERNS, 6, 0)
        self.assertEqual(len(asked), 2)            # not 12: each would wait out a timeout

    def test_one_lost_answer_does_not_end_the_search(self) -> None:
        answers = {"/a/100": None, "/a/200": False, "/a/300": True}

        with mock.patch.object(find_cameras, "rtsp_stream_state",
                               side_effect=lambda h, p, u, pw, path, timeout=5.0: answers.get(path, False)):
            pattern, hits = find_cameras.rtsp_channels("h", 554, "admin", "x", PATTERNS, 3, 0)
        self.assertEqual((pattern["name"], hits), ("A", [(3, "/a/300")]))

    def test_stream_exists_logs_in_when_asked_to(self) -> None:
        replies = iter(['RTSP/1.0 401 Unauthorized\r\nWWW-Authenticate: Digest realm="r", nonce="n"\r\n\r\n',
                        "RTSP/1.0 200 OK\r\n\r\n"])
        with mock.patch.object(find_cameras, "_rtsp_describe", side_effect=lambda *a, **k: next(replies)):
            self.assertTrue(find_cameras.rtsp_stream_exists("h", 554, "admin", "x", "/unicast/c2/s0/live"))

    def test_an_empty_channel_does_not_exist(self) -> None:
        with mock.patch.object(find_cameras, "_rtsp_describe", return_value="RTSP/1.0 500 ServerInternal\r\n\r\n"):
            self.assertFalse(find_cameras.rtsp_stream_exists("h", 554, "admin", "x", "/unicast/c1/s0/live"))
        with mock.patch.object(find_cameras, "_rtsp_describe", side_effect=OSError("timed out")):
            self.assertFalse(find_cameras.rtsp_stream_exists("h", 554, "admin", "x", "/unicast/c1/s0/live"))


if __name__ == "__main__":
    unittest.main()
