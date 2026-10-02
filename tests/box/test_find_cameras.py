from __future__ import annotations

import json
import unittest

from home_guard_project.box.find_cameras import camera_names, decode_password_file, describe, redact

URL_1 = "rtsp://admin:s3cret@192.168.1.50:554/unicast/c1/s0/live"
URL_3 = "rtsp://admin:s3cret@192.168.1.50:554/unicast/c3/s0/live"
URL_B = "rtsp://admin:s3cret@192.168.1.77:554/Streaming/Channels/101"

RECORDER = [
    {"channel": 1, "url": URL_1, "pattern": "x", "w": 1920, "h": 1080},
    {"channel": 3, "url": URL_3, "pattern": "x", "w": 2560, "h": 1440},
]
SINGLE_CAMERA = [{"channel": 1, "url": URL_B, "pattern": "y", "w": 1280, "h": 720}]


class FindCamerasTest(unittest.TestCase):
    def test_one_host_names_use_site_prefix_and_channel(self) -> None:
        self.assertEqual(
            camera_names({"192.168.1.50": RECORDER}, "house2"),
            {"house2_ch1": URL_1, "house2_ch3": URL_3},
        )

    def test_several_hosts_get_the_last_address_number_in_the_name(self) -> None:
        self.assertEqual(
            camera_names({"192.168.1.50": RECORDER, "192.168.1.77": SINGLE_CAMERA}, "house2"),
            {"house2_50_ch1": URL_1, "house2_50_ch3": URL_3, "house2_77_ch1": URL_B},
        )

    def test_hosts_without_streams_are_ignored(self) -> None:
        self.assertEqual(
            camera_names({"192.168.1.50": RECORDER, "192.168.1.99": []}, "house2"),
            {"house2_ch1": URL_1, "house2_ch3": URL_3},
        )
        self.assertEqual(camera_names({}, "house2"), {})

    def test_describe_lists_cameras_without_credentials(self) -> None:
        rows = describe({"192.168.1.50": RECORDER}, "house2")
        self.assertEqual(
            rows,
            [
                {"name": "house2_ch1", "host": "192.168.1.50", "channel": 1, "width": 1920, "height": 1080},
                {"name": "house2_ch3", "host": "192.168.1.50", "channel": 3, "width": 2560, "height": 1440},
            ],
        )
        self.assertNotIn("s3cret", json.dumps(rows))

    def test_password_file_is_base64_utf8(self) -> None:
        # "p@ss wörd\"&" survives quoting-hostile characters and a trailing newline or BOM-less file.
        self.assertEqual(decode_password_file("cEBzcyB3w7ZyZCIm\n"), 'p@ss wörd"&')

    def test_redact_hides_credentials(self) -> None:
        self.assertEqual(redact(URL_1), "rtsp://<user>:<password>@192.168.1.50:554/unicast/c1/s0/live")
        self.assertEqual(redact("rtsp://192.168.1.50:554/live"), "rtsp://192.168.1.50:554/live")


if __name__ == "__main__":
    unittest.main()
