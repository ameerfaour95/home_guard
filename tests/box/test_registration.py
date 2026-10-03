from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest import mock

from home_guard_project.box import __main__ as box_main
from home_guard_project.box.registration import (
    RegistrationError,
    load_registration,
    put_registration,
    register_from_json,
    registration_summary,
    update_site,
    write_registration,
)


class FakeS3:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class RegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(tmp.name, "registration.json")
        self.box_yaml = os.path.join(tmp.name, "box.yaml")
        self.published = os.path.join(tmp.name, "registration.published")
        with open(self.box_yaml, "w", encoding="utf-8") as f:
            f.write('site: "cohen_haifa"\n')

    def _write(self, **over: Any) -> Dict[str, Any]:
        fields: Dict[str, Any] = dict(owner_name="Dana Cohen", owner_phone="+972501234567", installer="Ameer",
                                      consent_live=True, consent_recordings=False, consent_training=False,
                                      tailscale_host="100.1.2.3", box_host="BEELINK1", app_version="abc1234")
        fields.update(over)
        return write_registration(self.path, box_yaml=self.box_yaml, **fields)

    def test_writes_schema_v1(self) -> None:
        reg = self._write()
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), reg)
        self.assertEqual(reg["schema_version"], 1)
        self.assertEqual(reg["site"], "cohen_haifa")
        self.assertEqual(reg["owner_name"], "Dana Cohen")
        self.assertEqual(reg["consent"]["live"], True)
        self.assertEqual(reg["consent"]["recordings"], False)
        self.assertEqual(reg["consent"]["recorded_by"], "Ameer")
        self.assertTrue(reg["consent"]["recorded_utc"].endswith("Z"))
        self.assertTrue(reg["installed_utc"].endswith("Z"))
        self.assertEqual(reg["tailscale_host"], "100.1.2.3")
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".tmp")])

    def test_phone_optional(self) -> None:
        self.assertEqual(self._write(owner_phone=None)["owner_phone"], "")

    def test_validation(self) -> None:
        for bad in ({"owner_name": ""}, {"owner_name": "x" * 121}, {"owner_name": "a\nb"}, {"owner_name": "a\x00"},
                    {"site": "Other"}, {"site": "other_site"}, {"consent_live": "yes"}, {"consent_training": 1}):
            with self.subTest(bad=bad), self.assertRaises(RegistrationError):
                self._write(**bad)
        self.assertFalse(os.path.exists(self.path))

    def test_rerun_keeps_other_fields_and_consent_time(self) -> None:
        first = self._write()
        second = write_registration(self.path, box_yaml=self.box_yaml, owner_name="Dana C.")
        self.assertEqual(second["owner_name"], "Dana C.")
        self.assertEqual(second["owner_phone"], first["owner_phone"])
        self.assertEqual(second["installed_utc"], first["installed_utc"])
        self.assertEqual(second["consent"], first["consent"])

    def test_consent_change_updates_recorded(self) -> None:
        self._write()
        with mock.patch("home_guard_project.box.registration._now_utc", return_value="2030-01-01T00:00:00Z"):
            reg = write_registration(self.path, box_yaml=self.box_yaml, consent_training=True, installer="Sam")
        self.assertTrue(reg["consent"]["training"])
        self.assertEqual(reg["consent"]["recorded_utc"], "2030-01-01T00:00:00Z")
        self.assertEqual(reg["consent"]["recorded_by"], "Sam")

    def test_load_tolerant(self) -> None:
        self.assertIsNone(load_registration(self.path))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertIsNone(load_registration(self.path))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("[1]")
        self.assertIsNone(load_registration(self.path))
        self._write()
        reg = load_registration(self.path)
        self.assertIsNotNone(reg)
        self.assertEqual(reg["site"], "cohen_haifa")

    def test_put_only_on_change(self) -> None:
        reg = self._write()
        s3 = FakeS3()
        key = put_registration(reg, "bkt", "dataset_cohen_haifa", s3, published_path=self.published)
        self.assertEqual(key, "dataset_cohen_haifa/_status/registration.json")
        self.assertEqual(json.loads(s3.calls[0]["Body"]), reg)
        self.assertEqual(s3.calls[0]["Bucket"], "bkt")
        self.assertIsNone(put_registration(reg, "bkt", "dataset_cohen_haifa", s3, published_path=self.published))
        self.assertEqual(len(s3.calls), 1)
        changed = self._write(owner_name="Dana K")
        self.assertIsNotNone(put_registration(changed, "bkt", "dataset_cohen_haifa", s3, published_path=self.published))
        self.assertEqual(len(s3.calls), 2)

    def test_failed_put_is_retried_next_time(self) -> None:
        reg = self._write()

        class Boom:
            def put_object(self, **kw: Any) -> None:
                raise OSError("down")

        with self.assertRaises(OSError):
            put_registration(reg, "b", "dataset_cohen_haifa", Boom(), published_path=self.published)
        s3 = FakeS3()
        self.assertIsNotNone(put_registration(reg, "b", "dataset_cohen_haifa", s3, published_path=self.published))

    def test_site_change_republishes_under_new_prefix(self) -> None:
        reg = self._write()
        s3 = FakeS3()
        put_registration(reg, "b", "dataset_cohen_haifa", s3, published_path=self.published)
        new = update_site(self.path, "cohen_tlv")
        self.assertEqual(new["site"], "cohen_tlv")
        self.assertEqual(new["owner_name"], "Dana Cohen")
        key = put_registration(new, "b", "dataset_cohen_tlv", s3, published_path=self.published)
        self.assertEqual(key, "dataset_cohen_tlv/_status/registration.json")
        self.assertEqual(len(s3.calls), 2)

    def test_update_site_without_registration_is_noop(self) -> None:
        self.assertIsNone(update_site(self.path, "cohen_tlv"))
        self.assertFalse(os.path.exists(self.path))
        self._write()
        with self.assertRaises(RegistrationError):
            update_site(self.path, "Bad Site")

    def test_from_json_deletes_file_and_summary_hides_phone(self) -> None:
        src = os.path.join(self.dir, "answers.json")
        with open(src, "w", encoding="utf-8") as f:
            json.dump({"owner_name": "Dana Cohen", "owner_phone": "+972501234567", "installer": "Ameer",
                       "consent_live": True, "consent_recordings": False, "consent_training": False,
                       "tailscale_host": "100.1.2.3"}, f)
        reg = register_from_json(src, self.path, box_yaml=self.box_yaml)
        self.assertFalse(os.path.exists(src))
        self.assertEqual(reg["owner_name"], "Dana Cohen")
        line = registration_summary(reg)
        self.assertIn("cohen_haifa", line)
        self.assertIn("Dana Cohen", line)
        self.assertNotIn("972", line)

    def test_from_json_old_answers_default_no_consent(self) -> None:
        src = os.path.join(self.dir, "a.json")
        with open(src, "w", encoding="utf-8") as f:
            json.dump({"installer": "Ameer"}, f)
        reg = register_from_json(src, self.path, box_yaml=self.box_yaml)
        self.assertEqual(reg["owner_name"], "cohen_haifa")
        self.assertEqual(reg["consent"]["live"], False)
        self.assertEqual(reg["consent"]["training"], False)

    def test_from_json_bad_file_still_deleted(self) -> None:
        src = os.path.join(self.dir, "a.json")
        with open(src, "w", encoding="utf-8") as f:
            f.write("nope")
        with self.assertRaises(RegistrationError):
            register_from_json(src, self.path, box_yaml=self.box_yaml)
        self.assertFalse(os.path.exists(src))

    def test_status_reports_registration(self) -> None:
        self.assertEqual(box_main.registration_status(None), {"registered": False})
        reg = self._write()
        st = box_main.registration_status(reg)
        self.assertEqual(st, {"registered": True, "consent": {"live": True, "recordings": False, "training": False}})
        self.assertNotIn("owner_phone", json.dumps(st))

    def test_cli_arguments_parse(self) -> None:
        args = box_main.parse_register_args(["--owner", "Dana", "--phone", "1", "--consent-live", "yes",
                                             "--consent-recordings", "no", "--consent-training", "no",
                                             "--installer", "Ameer", "--tailscale-host", "h"])
        self.assertEqual(args["owner_name"], "Dana")
        self.assertIs(args["consent_live"], True)
        self.assertIs(args["consent_recordings"], False)
        self.assertEqual(args["tailscale_host"], "h")


if __name__ == "__main__":
    unittest.main()
