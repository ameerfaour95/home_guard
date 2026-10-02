from __future__ import annotations

import unittest
from unittest import mock

from home_guard_project.box import find_cameras

STREAM = {"channel": 2, "url": "rtsp://x", "pattern": "p", "w": 1920, "h": 1080}


class HostScanTest(unittest.TestCase):
    """Where the search looks for devices, and how it copes with a network that is still settling."""

    def test_only_local_networks_are_candidates_not_the_vpn(self) -> None:
        with mock.patch("socket.gethostbyname_ex",
                        return_value=("box", [], ["100.121.29.9", "192.168.68.120", "169.254.3.4", "10.0.5.6"])):
            self.assertEqual(find_cameras.lan_addresses(), ["192.168.68.120", "10.0.5.6"])

    def test_an_empty_network_is_scanned_again_after_a_pause(self) -> None:
        from home_guard_project.data_collection import discover

        calls = {554: 0, 8554: 0}
        waited = []

        def scan(subnet, port):                      # the recorder appears on the second look
            calls[port] += 1
            return ["192.168.68.109"] if port == 554 and calls[port] == 2 else []

        with mock.patch.object(discover, "subnet_scan", side_effect=scan):
            hosts = find_cameras.rtsp_hosts(["192.168.68.120"], sleep=waited.append)
        self.assertEqual(hosts, {554: ["192.168.68.109"], 8554: []})
        self.assertEqual(waited, [find_cameras.SETTLE_WAIT_SEC])        # one pause, then it found the recorder

    def test_a_network_with_a_device_is_scanned_once(self) -> None:
        from home_guard_project.data_collection import discover

        waited = []
        with mock.patch.object(discover, "subnet_scan",
                               side_effect=lambda subnet, port: ["10.0.0.5"] if port == 554 else []) as scan:
            hosts = find_cameras.rtsp_hosts(["10.0.0.9"], sleep=waited.append)
        self.assertEqual((hosts[554], waited, scan.call_count), (["10.0.0.5"], [], 2))

    def test_a_network_that_stays_empty_gives_up_after_the_last_scan(self) -> None:
        from home_guard_project.data_collection import discover

        waited = []
        with mock.patch.object(discover, "subnet_scan", return_value=[]):
            hosts = find_cameras.rtsp_hosts(["10.0.0.9"], sleep=waited.append)
        self.assertEqual(hosts, {554: [], 8554: []})
        self.assertEqual(len(waited), find_cameras.SETTLE_SCANS - 1)

    def test_two_local_networks_are_both_scanned(self) -> None:
        from home_guard_project.data_collection import discover

        with mock.patch.object(discover, "subnet_scan",
                               side_effect=lambda subnet, port: [subnet.replace(".0/24", ".7")] if port == 554 else []):
            hosts = find_cameras.rtsp_hosts(["192.168.1.2", "10.0.0.2"], sleep=lambda s: None)
        self.assertEqual(hosts[554], ["10.0.0.7", "192.168.1.7"])


class SearchTest(unittest.TestCase):
    """The whole search: which devices are asked for their cameras, and which are left alone."""

    def _search(self, hosts, logins):
        probed = []

        def probe(host, port, user, password):
            probed.append(host)
            return [STREAM]

        with mock.patch.object(find_cameras, "rtsp_hosts", return_value=hosts), \
                mock.patch.object(find_cameras, "lan_addresses", return_value=["192.168.1.20"]), \
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
