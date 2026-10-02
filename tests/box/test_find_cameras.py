from __future__ import annotations

import unittest

from home_guard_project.box.find_cameras import camera_names, redact

FOUND = [
    {"channel": 1, "url": "rtsp://admin:s3cret@192.168.1.50:554/unicast/c1/s0/live", "pattern": "x", "w": 1920, "h": 1080},
    {"channel": 3, "url": "rtsp://admin:s3cret@192.168.1.50:554/unicast/c3/s0/live", "pattern": "x", "w": 2560, "h": 1440},
]


class FindCamerasTest(unittest.TestCase):
    def test_camera_names_use_site_prefix_and_channel(self) -> None:
        self.assertEqual(
            camera_names(FOUND, "house2"),
            {
                "house2_ch1": FOUND[0]["url"],
                "house2_ch3": FOUND[1]["url"],
            },
        )

    def test_camera_names_empty(self) -> None:
        self.assertEqual(camera_names([], "house2"), {})

    def test_redact_hides_credentials(self) -> None:
        self.assertEqual(
            redact(FOUND[0]["url"]),
            "rtsp://<user>:<password>@192.168.1.50:554/unicast/c1/s0/live",
        )
        self.assertNotIn("s3cret", redact(FOUND[0]["url"]))

    def test_redact_leaves_url_without_credentials(self) -> None:
        self.assertEqual(redact("rtsp://192.168.1.50:554/live"), "rtsp://192.168.1.50:554/live")


if __name__ == "__main__":
    unittest.main()
