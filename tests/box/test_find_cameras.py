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


class LoginCheckTest(unittest.TestCase):
    """The quick check that tells a wrong camera login apart from a wrong stream address."""

    UNAUTHORIZED = ('RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\n'
                    'WWW-Authenticate: Digest realm="c4790592a19c", nonce="abc123"\r\n\r\n')

    def _check(self, replies):
        from unittest import mock

        from home_guard_project.box import find_cameras

        sent = []

        def fake(host, port, uri, authorization="", timeout=5.0):
            sent.append(authorization)
            return replies[len(sent) - 1]

        with mock.patch.object(find_cameras, "_rtsp_describe", side_effect=fake):
            return find_cameras.rtsp_login_check("192.168.0.108", 554, "admin", "secret"), sent

    def test_a_recorder_that_refuses_unknown_addresses_is_accepted_on_a_real_one(self) -> None:
        """The owner's recorder answers 401 to every address it does not serve, right login or not."""
        from unittest import mock

        from home_guard_project.box import find_cameras

        asked = []

        def fake(host, port, uri, authorization="", timeout=5.0):
            asked.append(uri)
            if uri.endswith("/unicast/c1/s0/live") and authorization:
                return "RTSP/1.0 200 OK\r\n\r\n"
            return self.UNAUTHORIZED

        paths = ["/Streaming/Channels/101", "/unicast/c1/s0/live", "/cam/realmonitor?channel=1&subtype=0"]
        with mock.patch.object(find_cameras, "_rtsp_describe", side_effect=fake):
            self.assertEqual(find_cameras.rtsp_login_check("10.0.0.9", 554, "admin", "secret", paths=paths), "accepted")
        self.assertTrue(asked[-1].endswith("/unicast/c1/s0/live"))   # it stopped at the first address that worked

        with mock.patch.object(find_cameras, "_rtsp_describe", return_value=self.UNAUTHORIZED):
            self.assertEqual(find_cameras.rtsp_login_check("10.0.0.9", 554, "admin", "wrong", paths=paths), "refused")

    def test_the_real_addresses_tried_cover_every_known_recorder_family(self) -> None:
        from home_guard_project.box import find_cameras

        paths = find_cameras.first_stream_paths()
        self.assertIn("/unicast/c1/s0/live", paths)
        self.assertIn("/Streaming/Channels/101", paths)
        self.assertTrue(all(path.startswith("/") and "{" not in path for path in paths))

    def test_digest_response_matches_the_rfc_2069_example(self) -> None:
        from home_guard_project.box.find_cameras import digest_response

        self.assertEqual(
            digest_response("Mufasa", "testrealm@host.com", "CircleOfLife", "GET", "/dir/index.html",
                            "dcd98b7102dd2f0e8b11d0f600bfb0c093"),
            "1949323746fe6a43ef61f9606e7febea",
        )

    def test_a_login_the_camera_rejects_twice_is_refused(self) -> None:
        result, sent = self._check([self.UNAUTHORIZED, self.UNAUTHORIZED])
        self.assertEqual(result, "refused")
        self.assertEqual(sent[0], "")
        self.assertIn('Digest username="admin", realm="c4790592a19c", nonce="abc123"', sent[1])
        self.assertNotIn("secret", sent[1])          # the password itself is never sent

    def test_any_other_answer_after_logging_in_means_accepted(self) -> None:
        for second in ("RTSP/1.0 404 Not Found\r\n\r\n", "RTSP/1.0 200 OK\r\n\r\n"):
            self.assertEqual(self._check([self.UNAUTHORIZED, second])[0], "accepted")

    def test_a_device_that_asks_for_no_login_is_accepted(self) -> None:
        result, sent = self._check(["RTSP/1.0 404 Not Found\r\n\r\n"])
        self.assertEqual((result, len(sent)), ("accepted", 1))

    def test_no_answer_is_silent_not_refused(self) -> None:
        from unittest import mock

        from home_guard_project.box import find_cameras

        with mock.patch.object(find_cameras, "_rtsp_describe", side_effect=OSError("timed out")):
            self.assertEqual(find_cameras.rtsp_login_check("192.168.0.9", 554, "admin", "x"), "silent")
