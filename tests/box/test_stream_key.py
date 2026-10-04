from __future__ import annotations

import unittest

from home_guard_project.box.find_cameras import stream_key


class StreamKeyTest(unittest.TestCase):
    def test_an_at_sign_in_the_path_keeps_two_channels_apart(self) -> None:
        a = stream_key("rtsp://u:p@192.0.2.1/channel/1@live")
        b = stream_key("rtsp://u:p@192.0.2.1/channel/2@live")
        self.assertNotEqual(a, b)
        self.assertEqual(a, "192.0.2.1:554/channel/1@live")

    def test_the_login_is_not_part_of_the_identity(self) -> None:
        self.assertEqual(stream_key("rtsp://admin:one@192.0.2.1:554/unicast/c1/s0/live"),
                         stream_key("rtsp://admin:two@192.0.2.1:554/unicast/c1/s0/live"))

    def test_a_password_with_an_at_sign(self) -> None:
        self.assertEqual(stream_key("rtsp://admin:p@ss@192.0.2.1/ch1"), "192.0.2.1:554/ch1")

    def test_the_query_is_part_of_the_identity(self) -> None:
        self.assertNotEqual(stream_key("rtsp://192.0.2.1/cam?channel=1&subtype=0"),
                            stream_key("rtsp://192.0.2.1/cam?channel=2&subtype=0"))

    def test_host_case_and_default_port(self) -> None:
        self.assertEqual(stream_key("RTSP://Cam.Local/ch1"), stream_key("rtsp://cam.local:554/ch1"))

    def test_an_unparseable_url_never_returns_its_text(self) -> None:
        key = stream_key("not a url with secret:pa55")
        self.assertNotIn("pa55", key)
        self.assertNotEqual(key, stream_key("another bad one"))
