"""2026-10-03: re-running the camera search renamed every camera.

The owner's names (main_entrance, front_side, ...) were replaced by
ameer_test_ch2.. because a search under a new site prefix rewrote cameras.yaml
from scratch. A stream the box already knows keeps its name; only a new stream
gets a new one.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import yaml

from home_guard_project.box import find_cameras
from home_guard_project.box.find_cameras import stream_key, write_found

OLD_1 = "rtsp://admin:old%40pw@192.168.1.50:554/unicast/c1/s0/live"
OLD_3 = "rtsp://admin:old%40pw@192.168.1.50:554/unicast/c3/s0/live"
NEW_1 = "rtsp://admin:n3w@192.168.1.50:554/unicast/c1/s0/live"     # same stream, new password
NEW_2 = "rtsp://admin:n3w@192.168.1.50:554/unicast/c2/s0/live"     # a stream never seen before
NEW_3 = "rtsp://admin:n3w@192.168.1.50:554/unicast/c3/s0/live"


def stream(channel: int, url: str) -> dict:
    return {"channel": channel, "url": url, "pattern": "x", "w": 1920, "h": 1080}


FOUND = {"192.168.1.50": [stream(1, NEW_1), stream(2, NEW_2), stream(3, NEW_3)]}


class KeepKnownNamesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "cameras.yaml")
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": {"main_entrance": OLD_1}, "disabled": {"right_side": OLD_3}}, f)

    def _saved(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_the_stream_identity_ignores_the_login(self) -> None:
        self.assertEqual(stream_key(OLD_1), stream_key(NEW_1))
        self.assertEqual(stream_key(OLD_1), "192.168.1.50:554/unicast/c1/s0/live")
        self.assertNotEqual(stream_key(NEW_1), stream_key(NEW_2))
        self.assertEqual(stream_key("rtsp://192.168.1.50/live"), stream_key("RTSP://u:p@192.168.1.50:554/live"))

    def test_a_known_stream_keeps_its_name_and_a_new_one_gets_a_new_name(self) -> None:
        result = write_found(FOUND, "ameer_test", self.path)
        saved = self._saved()
        self.assertEqual(saved["cameras"], {"main_entrance": NEW_1, "ameer_test_ch2": NEW_2})  # new login kept
        self.assertEqual(saved["disabled"], {"right_side": NEW_3})          # still off, still its name
        self.assertEqual(result, {"active": ["ameer_test_ch2", "main_entrance"], "disabled": ["right_side"]})

    def test_without_a_cameras_file_every_stream_is_named_as_before(self) -> None:
        os.remove(self.path)
        write_found(FOUND, "house2", self.path)
        self.assertEqual(self._saved(), {"cameras": {"house2_ch1": NEW_1, "house2_ch2": NEW_2,
                                                     "house2_ch3": NEW_3}})

    def test_a_new_stream_never_takes_a_kept_name(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:   # the owner once named channel 1 "ameer_test_ch2"
            yaml.safe_dump({"cameras": {"ameer_test_ch2": OLD_1}}, f)
        write_found(FOUND, "ameer_test", self.path)
        cameras = self._saved()["cameras"]
        self.assertEqual(cameras["ameer_test_ch2"], NEW_1)
        self.assertEqual(cameras["ameer_test_ch2_2"], NEW_2)
        self.assertEqual(cameras["ameer_test_ch3"], NEW_3)

    def test_the_search_command_saves_and_reports_the_kept_names(self) -> None:
        args = argparse.Namespace(prefix="ameer_test", write=True, json=True)
        out = io.StringIO()
        with mock.patch.object(find_cameras, "CAMERAS_PATH", self.path), contextlib.redirect_stdout(out):
            find_cameras._finish(FOUND, args, {})
        reported = json.loads(out.getvalue())
        self.assertTrue(reported["saved"])
        self.assertEqual([r["name"] for r in reported["cameras"]], ["main_entrance", "ameer_test_ch2", "right_side"])
        self.assertNotIn("n3w", out.getvalue())
        self.assertEqual(self._saved()["cameras"]["main_entrance"], NEW_1)


if __name__ == "__main__":
    unittest.main()
