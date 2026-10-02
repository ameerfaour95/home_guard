from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
from typing import Any, Dict

from helpers import make_clip

from home_guard_project.box.heartbeat import build_heartbeat, collector_alive, put_heartbeat

NOW = 1_800_000_000.0


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeS3:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class HeartbeatTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.live = os.path.join(tmp.name, "dataset_multi")
        self.outbox = os.path.join(tmp.name, "dataset_outbox")
        os.makedirs(self.live)
        os.makedirs(self.outbox)
        self.alive = os.path.join(tmp.name, "collector.alive")

    def _touch_alive(self, mtime: float) -> None:
        with open(self.alive, "w", encoding="utf-8"):
            pass
        os.utime(self.alive, (mtime, mtime))

    def test_collector_alive(self) -> None:
        self.assertFalse(collector_alive(self.alive, now=NOW))
        self._touch_alive(NOW - 30)
        self.assertTrue(collector_alive(self.alive, now=NOW))
        self._touch_alive(NOW - 600)
        self.assertFalse(collector_alive(self.alive, now=NOW))

    def test_counts_and_newest_clip(self) -> None:
        make_clip(self.live, "front", "front_1_trigger", NOW - 500)
        make_clip(self.live, "front", "front_2_trigger", NOW - 100)
        make_clip(os.path.join(self.outbox, "house2"), "yard", "yard_1_trigger", NOW - 9000)
        self._touch_alive(NOW - 10)

        hb = build_heartbeat("house2", self.live, self.outbox, self.alive, now=NOW, mode="inference")

        self.assertEqual(hb["site"], "house2")
        self.assertEqual(hb["mode"], "inference")
        self.assertEqual(hb["time_utc"], iso(NOW))
        self.assertTrue(hb["collector_running"])
        self.assertEqual(hb["clips_live"], 2)
        self.assertEqual(hb["clips_outbox"], 1)
        self.assertEqual(hb["newest_clip_utc"], iso(NOW - 100))
        self.assertEqual(
            hb["cameras"],
            {
                "front": {"clips_waiting": 2, "newest_clip_utc": iso(NOW - 100)},
                "yard": {"clips_waiting": 1, "newest_clip_utc": iso(NOW - 9000)},
            },
        )
        self.assertIsInstance(hb["host"], str)
        self.assertIn("local_ip", hb)
        self.assertTrue(hb["local_ip"] is None or hb["local_ip"].count(".") == 3)
        self.assertGreater(hb["disk_free_gb"], 0)

    def test_no_clips(self) -> None:
        hb = build_heartbeat("house2", self.live, self.outbox, self.alive, now=NOW)
        self.assertIsNone(hb["newest_clip_utc"])
        self.assertEqual(hb["cameras"], {})
        self.assertFalse(hb["collector_running"])

    def test_put_heartbeat(self) -> None:
        hb = build_heartbeat("house2", self.live, self.outbox, self.alive, now=NOW)
        s3 = FakeS3()
        key = put_heartbeat(hb, "my-bucket", "dataset_house2", s3_client=s3)
        self.assertEqual(key, "dataset_house2/_status/heartbeat.json")
        (call,) = s3.calls
        self.assertEqual(call["Bucket"], "my-bucket")
        self.assertEqual(call["Key"], key)
        self.assertEqual(call["ContentType"], "application/json")
        self.assertEqual(json.loads(call["Body"]), hb)
        self.assertNotIn(b"rtsp", call["Body"])


if __name__ == "__main__":
    unittest.main()
