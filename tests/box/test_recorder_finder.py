"""A recorder that came back on a new network address is found and the cameras are pointed at it (2026-10-19: a power
cut, the mesh gave the recorder .108 instead of .106, and the box saw no camera for a day and a half;
box/recorder_finder.py)."""
import os
import shutil
import tempfile
import unittest

from home_guard_project.box import recorder_finder as rf

USER, PW = "admin", "p@ss:word"
OLD, NEW = "192.168.68.106", "192.168.68.108"
PATHS = [f"/unicast/c{ch}/s1/live" for ch in (2, 3, 5)]
URL = "rtsp://admin:p%40ss%3Aword@{host}:554/unicast/c{ch}/s1/live"
CAMERAS = {f"ameer_v2_ch{ch}": URL.format(host=OLD, ch=ch) for ch in (2, 3, 5)}
FILE = ("# ──────\n#  Camera RTSP streams\n\n"
        "cameras:\n"
        "  ameer_v2_ch2: rtsp://admin:p%40ss%3Aword@192.168.68.106:554/unicast/c2/s0/live\n"
        "  ameer_v2_ch3: rtsp://admin:p%40ss%3Aword@192.168.68.106:554/unicast/c3/s0/live\n"
        "  ameer_v2_ch5: rtsp://admin:p%40ss%3Aword@192.168.68.106:554/unicast/c5/s0/live\n"
        "  doorbell: rtsp://192.168.68.1061/live\n"          # another device whose address starts the same
        "disabled:\n"
        "  ameer_v2_ch1: rtsp://admin:p%40ss%3Aword@192.168.68.106/unicast/c1/s0/live\n")


class Network:
    """A fake LAN: *devices* maps host -> the paths it serves to our login (None: the host does not answer)."""

    def __init__(self, devices):
        self.devices = devices
        self.asked = []

    def scan(self, port):
        return sorted(h for h, served in self.devices.items() if served is not None)

    def serves(self, host, port, user, password, path):
        self.asked.append((host, path))
        assert (user, password) == (USER, PW)
        served = self.devices.get(host)
        if served is None:
            return None
        return any(path.replace("/s1/", "/s0/") == p.replace("/s1/", "/s0/") for p in served)


class ParseTest(unittest.TestCase):
    def test_groups_cameras_by_device_with_the_login_decoded(self):
        groups = rf.group_by_device({**CAMERAS, "other": "rtsp://10.0.0.5:8554/live?ch=1", "bad": "http://x/"})
        self.assertEqual(sorted(groups), [("10.0.0.5", 8554), (OLD, 554)])
        group = groups[(OLD, 554)]
        self.assertEqual(group["cameras"], ["ameer_v2_ch2", "ameer_v2_ch3", "ameer_v2_ch5"])
        self.assertEqual((group["user"], group["password"]), (USER, PW))
        self.assertEqual(group["paths"], PATHS)
        self.assertEqual(groups[("10.0.0.5", 8554)]["paths"], ["/live?ch=1"])

    def test_move_host_touches_only_that_address(self):
        moved, count = rf.move_host(FILE, OLD, NEW)
        self.assertEqual(count, 4)                            # three active and the disabled one (no port)
        self.assertIn("rtsp://192.168.68.1061/live", moved)   # not a prefix match
        self.assertIn("@192.168.68.108/unicast/c1/s0/live", moved)
        self.assertNotIn("@192.168.68.106", moved)
        self.assertTrue(moved.startswith("# ──────\n#  Camera RTSP streams"))   # the rest of the file is kept

    def test_file_hosts(self):
        self.assertEqual(rf.file_hosts(FILE), ["192.168.68.106", "192.168.68.1061"])

    def test_the_switch(self):
        self.assertTrue(rf.enabled({}))
        self.assertTrue(rf.enabled({"find_moved_recorder": "on"}))
        for off in (False, "off", "no", "0"):
            self.assertFalse(rf.enabled({"find_moved_recorder": off}), off)


class FindDeviceTest(unittest.TestCase):
    group = rf.group_by_device(CAMERAS)[(OLD, 554)]

    def find(self, devices, known=(OLD,)):
        net = Network(devices)
        return rf.find_device(self.group, (OLD, 554), list(known), net.scan, net.serves), net

    def test_the_old_address_still_serving_moves_nothing(self):
        result, net = self.find({OLD: PATHS, NEW: PATHS})
        self.assertEqual(result["outcome"], "still_there")
        self.assertEqual(net.asked, [(OLD, PATHS[0])])        # no scan of the network

    def test_the_one_device_serving_every_path_is_the_recorder(self):
        result, _ = self.find({OLD: [], "192.168.68.1": [], NEW: PATHS})
        self.assertEqual((result["outcome"], result["host"]), ("found", NEW))

    def test_a_device_serving_only_some_paths_is_not_it(self):
        result, _ = self.find({OLD: None, NEW: PATHS[:1]})
        self.assertEqual(result["outcome"], "none")

    def test_two_devices_serving_the_same_streams_are_not_guessed_between(self):
        result, _ = self.find({OLD: None, NEW: PATHS, "192.168.68.120": PATHS})
        self.assertEqual(result["outcome"], "ambiguous")
        self.assertEqual(result["matches"], [NEW, "192.168.68.120"])

    def test_a_device_already_in_the_cameras_file_is_not_taken(self):
        result, _ = self.find({OLD: None, NEW: PATHS}, known=(OLD, NEW))
        self.assertEqual(result["outcome"], "none")

    def test_nothing_on_the_network(self):
        result, _ = self.find({OLD: None})
        self.assertEqual((result["outcome"], result["candidates"]), ("none", []))


class FinderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, "cameras.yaml")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(FILE)
        self.moves, self.restarts = [], []

    def finder(self, devices, started=1000.0):
        self.net = Network(devices)
        return rf.RecorderFinder(CAMERAS, self.path, started, moved=lambda *a: self.moves.append(a),
                                 scan=self.net.scan, serves=self.net.serves,
                                 restart=lambda: self.restarts.append(True), background=False)

    def read(self):
        with open(self.path, encoding="utf-8") as f:
            return f.read()

    def test_the_power_cut_morning(self):
        finder = self.finder({OLD: None, "192.168.68.1": [], NEW: PATHS})
        silent = {name: 0.0 for name in CAMERAS}             # no picture since the start
        finder.check(1000.0 + rf.SILENT_SEC - rf.CHECK_EVERY_SEC, silent)
        self.assertEqual(self.net.asked, [])                  # not yet two minutes
        with self.assertLogs("box.recorder_finder", "INFO") as logs:
            finder.check(1000.0 + rf.SILENT_SEC, silent)
        self.assertEqual(self.moves, [(OLD, NEW, 4)])
        self.assertEqual(self.restarts, [True])
        self.assertEqual(self.read(), rf.move_host(FILE, OLD, NEW)[0])
        backups = [n for n in os.listdir(self.dir) if n.startswith("cameras.yaml.bak-")]
        self.assertEqual(len(backups), 1)
        with open(os.path.join(self.dir, backups[0]), encoding="utf-8") as f:
            self.assertEqual(f.read(), FILE)
        self.assertTrue(any("Recorder search: moved 192.168.68.106 -> 192.168.68.108 (4 camera URL(s)" in line
                            for line in logs.output), logs.output)
        self.assertTrue(all(line.isascii() for line in logs.output), logs.output)
        asked = len(self.net.asked)
        finder.check(1000.0 + 10_000, silent)                 # done: the restart is on its way
        self.assertEqual(len(self.net.asked), asked)

    def test_one_camera_sending_pictures_means_the_recorder_is_there(self):
        finder = self.finder({OLD: None, NEW: PATHS})
        finder.check(2000.0, {"ameer_v2_ch2": 0.0, "ameer_v2_ch3": 1990.0, "ameer_v2_ch5": 0.0})
        self.assertEqual(self.net.asked, [])

    def test_not_found_is_looked_for_again_every_five_minutes(self):
        finder = self.finder({OLD: None})
        silent = {name: 0.0 for name in CAMERAS}
        t = 1000.0 + rf.SILENT_SEC
        finder.check(t, silent)
        self.assertEqual(len(self.net.asked), 1)
        finder.check(t + rf.RETRY_SEC - rf.CHECK_EVERY_SEC, silent)
        self.assertEqual(len(self.net.asked), 1)
        self.net.devices[NEW] = PATHS                         # the recorder finished booting on its new address
        finder.check(t + rf.RETRY_SEC, silent)
        self.assertEqual(self.moves, [(OLD, NEW, 4)])
        self.assertEqual(self.read(), rf.move_host(FILE, OLD, NEW)[0])

    def test_the_old_address_answering_changes_nothing(self):
        finder = self.finder({OLD: PATHS, NEW: PATHS})
        finder.check(1000.0 + rf.SILENT_SEC, {name: 0.0 for name in CAMERAS})
        self.assertEqual((self.moves, self.restarts, self.read()), ([], [], FILE))

    def test_a_failing_search_never_raises(self):
        finder = self.finder({OLD: None})

        def boom(port):
            raise OSError("network down")

        finder.scan = boom
        with self.assertLogs("box.recorder_finder", "WARNING"):
            finder.check(1000.0 + rf.SILENT_SEC, {name: 0.0 for name in CAMERAS})
        self.assertEqual(self.restarts, [])


if __name__ == "__main__":
    unittest.main()
