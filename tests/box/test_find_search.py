from __future__ import annotations

import unittest
from unittest import mock

from home_guard_project.box import find_cameras

STREAM = {"channel": 2, "url": "rtsp://x", "pattern": "p", "w": 1920, "h": 1080}


class SearchTest(unittest.TestCase):
    """The whole search: which devices are asked for their cameras, and which are left alone."""

    def _search(self, hosts, logins):
        probed = []

        def probe(host, port, user, password):
            probed.append(host)
            return [STREAM]

        with mock.patch.object(find_cameras, "rtsp_hosts", return_value=hosts), \
                mock.patch.object(find_cameras, "rtsp_login_check",
                                  side_effect=lambda host, port, user, password, paths=(): logins[host]), \
                mock.patch.object(find_cameras, "_probe_host", side_effect=probe):
            found, extra = find_cameras.search("admin", "x")
        return found, extra, probed

    def test_only_devices_that_accept_the_login_are_searched(self) -> None:
        hosts = {554: ["10.0.0.8", "10.0.0.9", "10.0.0.39", "10.0.0.41"], 8554: []}
        logins = {"10.0.0.8": "refused", "10.0.0.9": "silent", "10.0.0.39": "accepted", "10.0.0.41": "refused"}
        found, extra, probed = self._search(hosts, logins)

        self.assertEqual(probed, ["10.0.0.39"])
        self.assertEqual(found, {"10.0.0.39": [STREAM]})
        self.assertEqual(extra["devices_found"], 4)
        self.assertEqual(extra["login_refused"], ["10.0.0.8", "10.0.0.41"])
        self.assertEqual(extra["no_answer"], ["10.0.0.9"])
        self.assertEqual(extra["hosts_tried"], ["10.0.0.8", "10.0.0.9", "10.0.0.39", "10.0.0.41"])

    def test_a_device_with_an_unfamiliar_login_scheme_is_still_searched(self) -> None:
        _, _, probed = self._search({554: ["10.0.0.5"], 8554: []}, {"10.0.0.5": "unknown"})
        self.assertEqual(probed, ["10.0.0.5"])

    def test_devices_on_the_second_camera_port_are_included(self) -> None:
        found, extra, _ = self._search({554: [], 8554: ["10.0.0.7"]}, {"10.0.0.7": "accepted"})
        self.assertEqual(list(found), ["10.0.0.7"])
        self.assertEqual(extra["devices_found"], 1)

    def test_an_empty_network(self) -> None:
        found, extra, probed = self._search({554: [], 8554: []}, {})
        self.assertEqual((found, probed), ({}, []))
        self.assertEqual((extra["devices_found"], extra["login_refused"], extra["no_answer"]), (0, [], []))


if __name__ == "__main__":
    unittest.main()
