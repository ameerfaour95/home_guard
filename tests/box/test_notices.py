"""Access notices on the box: the cloud pushes them over ssh (notices add), the app lists and reads them."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.box import notices


def body(nid=7, kind="recording", cameras=("Front door", "Garden"), frm="2026-10-09T10:15:00Z",
         to="2026-10-09T10:40:00Z", **extra):
    data = {"schema_version": 1, "id": nid, "kind": kind, "staff_name": "Dana", "cameras": list(cameras),
            "from_utc": frm, "to_utc": to, "message": "Home Guard support viewed recordings (13:15)"}
    data.update(extra)
    return data


def b64(data):
    raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def cloud_command(data):
    """The exact line the cloud sends over ssh (home-guard-32's NOTICE_CLI), cmd.exe syntax."""
    return (r"cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box "
            + " ".join(notices.NOTICE_CLI).replace("<BASE64>", b64(data)))


class Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = os.path.join(tmp.name, "state", "notices")

    def cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = notices.main(list(argv), folder=self.folder)
        line = out.getvalue()
        self.assertEqual(line.count("\n"), 1, line)
        self.assertTrue(line.isascii(), line)
        return code, json.loads(line)

    def push(self, data):
        return self.cli("add", "--b64", b64(data), "--json")

    def files(self):
        return sorted(os.listdir(self.folder)) if os.path.isdir(self.folder) else []


class AddTest(Case):
    def test_a_new_notice_is_stored_under_its_time_and_id(self):
        self.assertEqual(self.push(body()), (0, {"result": "added", "id": 7}))
        self.assertEqual(self.files(), ["20261009T101500_7.json"])
        stored = json.load(open(os.path.join(self.folder, "20261009T101500_7.json"), encoding="utf-8"))
        self.assertEqual(stored["cameras"], ["Front door", "Garden"])
        self.assertEqual(stored["kind"], "recording")

    def test_the_same_body_again_is_unchanged_and_stays_read(self):
        self.push(body())
        self.cli("read-all", "--json")
        with mock.patch.object(notices, "_write_json") as write:
            self.assertEqual(self.push(body()), (0, {"result": "unchanged", "id": 7}))
        write.assert_not_called()
        self.assertEqual(notices.listing(self.folder)["unread"], 0)

    def test_a_rewritten_notice_replaces_the_old_one_and_is_unread_again(self):
        self.push(body())
        self.cli("read-all", "--json")
        changed = body(cameras=("Front door", "Garden", "Driveway"), to="2026-10-09T10:55:00Z")
        self.assertEqual(self.push(changed), (0, {"result": "updated", "id": 7}))
        self.assertEqual(self.files(), ["20261009T101500_7.json", "read.json"])
        row = notices.listing(self.folder, "en")["notices"][0]
        self.assertEqual((row["cameras"][-1], row["to_utc"], row["read"]), ("Driveway", "2026-10-09T10:55:00Z", False))

    def test_the_read_flags_live_beside_the_cloud_s_files(self):
        self.push(body())
        self.cli("read-all", "--json")
        stored = json.load(open(os.path.join(self.folder, "20261009T101500_7.json"), encoding="utf-8"))
        self.assertNotIn("read", stored)
        self.assertEqual(list(json.load(open(os.path.join(self.folder, "read.json"), encoding="utf-8"))), ["7"])

    def test_version_extra_fields_and_an_unknown_kind_are_accepted(self):
        data = body(nid=8, kind="live", message="Support watched live video", device_site="house2")
        data["version"] = data.pop("schema_version")
        self.assertEqual(self.push(data), (0, {"result": "added", "id": 8}))
        stored = json.load(open(os.path.join(self.folder, self.files()[0]), encoding="utf-8"))
        self.assertNotIn("device_site", stored)
        row = notices.listing(self.folder)["notices"][0]
        self.assertEqual((row["kind"], row["message"]), ("other", "Support watched live video"))
        self.push(body(nid=9, schema_version=2))                     # a newer schema: kept, shown as other
        self.assertEqual(notices.listing(self.folder)["notices"][0]["kind"], "other")

    def test_bad_input_is_refused_with_exit_1_and_nothing_stored(self):
        no_version = body(); no_version.pop("schema_version")
        no_kind = body(); no_kind.pop("kind")
        cases = {
            "not base64": self.cli("add", "--b64", "%%%not-base64%%%", "--json"),
            "not utf-8": self.cli("add", "--b64", b64(b"\xff\xfe"), "--json"),
            "not json": self.cli("add", "--b64", b64(b"{nope"), "--json"),
            "a list": self.push([1, 2]),
            "no version": self.push(no_version),
            "no id": self.push(body(nid=None)),
            "a bad id": self.push(body(nid="../x")),
            "no kind": self.push(no_kind),
            "no --b64": self.cli("add", "--json"),
        }
        for name, (code, data) in cases.items():
            self.assertEqual(code, 1, name)
            self.assertEqual(list(data), ["error"], name)
        self.assertEqual(self.files(), [])

    def test_the_clouds_sample_body_added_unchanged_updated(self):
        sample = ('{"schema_version":1,"id":42,"kind":"recording","staff_name":"Dana","cameras":["Gate"],'
                  '"from_utc":"2026-10-09T07:05:03Z","to_utc":"2026-10-09T07:05:03Z",'
                  '"message":"Home Guard support viewed recordings from Gate (10:05)"}')
        value = base64.b64encode(sample.encode("utf-8")).decode("ascii")
        self.assertEqual(self.cli("add", "--b64", value, "--json"), (0, {"result": "added", "id": 42}))
        self.assertEqual(self.files(), ["20261009T070503_42.json"])
        self.assertEqual(self.cli("add", "--b64", value, "--json"), (0, {"result": "unchanged", "id": 42}))
        later = base64.b64encode(sample.replace('"to_utc":"2026-10-09T07:05:03Z"',
                                                '"to_utc":"2026-10-09T07:20:41Z"').encode("utf-8")).decode("ascii")
        self.assertEqual(self.cli("add", "--b64", later, "--json"), (0, {"result": "updated", "id": 42}))
        self.assertEqual(self.files(), ["20261009T070503_42.json"])
        row = notices.listing(self.folder, "en")["notices"][0]
        self.assertEqual((row["id"], row["cameras"], row["to_utc"]), ("42", ["Gate"], "2026-10-09T07:20:41Z"))

    def test_every_rejection_is_one_error_line_and_exit_1_even_from_argparse(self):
        for argv in (["add"], ["add", "--b64"], ["add", "--b64", "x", "--json", "--bogus"], ["nonsense"], []):
            code, data = self.cli(*argv)
            self.assertEqual(code, 1, argv)
            self.assertEqual(list(data), ["error"], argv)

    def test_a_b64_longer_than_max_b64_is_refused_plainly(self):
        data = body(message="x" * 6000)
        value = b64(data)
        self.assertGreater(len(value), notices.MAX_B64)
        code, answer = self.cli("add", "--b64", value, "--json")
        self.assertEqual(code, 1)
        self.assertIn("too long", answer["error"])
        size = 6000
        while len(b64(body(message="x" * size))) > notices.MAX_B64:
            size -= 1
        just_fits = body(message="x" * size)
        self.assertGreater(len(b64(just_fits)), notices.MAX_B64 - 4)        # right at the limit
        self.assertEqual(self.push(just_fits)[0], 0)

    def test_the_cloud_s_longest_command_fits_cmd_exe(self):
        longest = cloud_command(body()).replace(b64(body()), "A" * notices.MAX_B64)
        self.assertLess(len(longest), 8191)

    def test_a_disk_failure_is_exit_2_so_the_cloud_sends_it_again(self):
        with mock.patch.object(notices.os, "replace", side_effect=OSError("disk full")):
            code, data = self.push(body())
        self.assertEqual(code, 2)
        self.assertIn("could not be saved", data["error"])
        self.assertEqual(self.files(), [])                          # no half-written or temporary file

    def test_files_are_written_atomically(self):
        with mock.patch.object(notices.os, "replace", wraps=os.replace) as replace:
            self.push(body())
        self.assertEqual(replace.call_args.args[1], os.path.join(self.folder, "20261009T101500_7.json"))
        self.assertTrue(os.path.basename(replace.call_args.args[0]).startswith(".notice-"))

    def test_the_cloud_s_exact_command_line_runs_through_the_box_cli(self):
        from home_guard_project.box import __main__ as box_main
        line = cloud_command(body(nid=12, cameras=("דלת הכניסה",)))
        argv = line.split(" -m home_guard_project.box ", 1)[1].split(" ")
        self.assertEqual(argv[:3], ["notices", "add", "--b64"])
        with mock.patch.object(box_main.sys, "argv", ["box", *argv]), \
                mock.patch.object(notices, "notices_dir", return_value=self.folder), \
                contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
            box_main.main()
        self.assertEqual(stop.exception.code, 0)
        self.assertEqual(json.loads(out.getvalue()), {"result": "added", "id": 12})
        self.assertEqual(self.files(), ["20261009T101500_12.json"])

    def test_the_store_is_the_boxs_state_folder(self):
        from home_guard_project.box import paths
        self.assertEqual(notices.notices_dir(), os.path.join(paths.state_dir(), "notices"))


class ListTest(Case):
    def test_newest_first_with_read_flags_unread_count_and_local_times(self):
        self.push(body(nid=1, frm="2026-10-08T09:00:00Z", to="2026-10-08T09:00:00Z"))
        self.push(body(nid=2, kind="chat", cameras=(), frm="2026-10-09T10:00:00Z", to="2026-10-09T10:20:00Z"))
        self.assertEqual(self.cli("read", "--id", "1", "--json"), (0, {"id": "1", "marked": 1, "unread": 1}))
        with mock.patch.object(notices, "_local", side_effect=lambda utc: "L" + utc[:16]):
            code, data = self.cli("list", "--json", "--lang", "en")
        self.assertEqual((code, data["unread"]), (0, 1))
        self.assertEqual([n["id"] for n in data["notices"]], ["2", "1"])
        chat = data["notices"][0]
        self.assertEqual((chat["kind"], chat["read"], chat["from_local"], chat["to_local"], chat["message"]),
                         ("chat", False, "L2026-10-09T10:00", "L2026-10-09T10:20", ""))   # no message for a known kind

    def test_read_and_read_all(self):
        self.push(body(nid=1)); self.push(body(nid=2))
        self.assertEqual(self.cli("read", "--id", "99", "--json"), (1, {"error": "no notice 99"}))
        self.assertEqual(self.cli("read", "--id", "1", "--json"), (0, {"id": "1", "marked": 1, "unread": 1}))
        self.assertEqual(self.cli("read-all", "--json"), (0, {"marked": 1, "unread": 0}))
        self.assertEqual(self.cli("read-all", "--json"), (0, {"marked": 0, "unread": 0}))

    def test_an_empty_box_lists_nothing(self):
        self.assertEqual(self.cli("list", "--json"), (0, {"notices": [], "unread": 0}))
        self.assertEqual(self.cli("read-all", "--json"), (0, {"marked": 0, "unread": 0}))

    def test_a_damaged_file_or_read_json_is_left_out(self):
        self.push(body(nid=1))
        with open(os.path.join(self.folder, "20261001T000000_5.json"), "w", encoding="utf-8") as f:
            f.write("{broken")
        with open(os.path.join(self.folder, "read.json"), "w", encoding="utf-8") as f:
            f.write("[1")
        self.assertEqual([n["id"] for n in notices.listing(self.folder)["notices"]], ["1"])

    def test_a_name_that_looks_like_an_id_goes_through_display_name(self):
        self.push(body(cameras=("ameer_week_0_1_ch6", "Front door", "דלת הכניסה")))
        with mock.patch("home_guard_project.box.camera_names._load", return_value={}):
            he = notices.listing(self.folder, "he")["notices"][0]["cameras"]
            en = notices.listing(self.folder, "en")["notices"][0]["cameras"]
        self.assertEqual(he, ["מצלמה 6", "Front door", "דלת הכניסה"])
        self.assertEqual(en, ["Camera 6", "Front door", "דלת הכניסה"])

    def test_local_time_is_the_boxs_clock(self):
        self.assertRegex(notices._local("2026-10-09T10:15:00Z"), r"^2026-10-(08|09|10)T\d\d:\d\d$")


if __name__ == "__main__":
    unittest.main()
