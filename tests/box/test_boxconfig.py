from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box.boxconfig import (
    PRODUCTION_PREFIX_ROOT,
    PRODUCTION_RETENTION_DAYS,
    BoxConfig,
    BoxConfigError,
    get_option,
    load_box_config,
    load_box_settings,
    production_prefix,
    s3_prefix,
    set_option,
    set_site,
)


class BoxConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "box.yaml")

    def _write(self, text: str) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_valid_file_uses_default_min_age(self) -> None:
        self._write('site: "house2"\n')
        self.assertEqual(load_box_config(self.path), BoxConfig(site="house2", min_age_minutes=10.0))

    def test_explicit_min_age(self) -> None:
        self._write("site: house2\nmin_age_minutes: 3\n")
        self.assertEqual(load_box_config(self.path).min_age_minutes, 3.0)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(BoxConfigError):
            load_box_config(self.path)

    def test_missing_site_raises(self) -> None:
        self._write("min_age_minutes: 3\n")
        with self.assertRaises(BoxConfigError):
            load_box_config(self.path)

    def test_invalid_site_raises(self) -> None:
        self._write('site: "House 2"\n')
        with self.assertRaises(BoxConfigError):
            load_box_config(self.path)

    def test_s3_prefix(self) -> None:
        self.assertEqual(s3_prefix("house2"), "dataset_house2")

    def test_production_prefix_is_a_separate_folder_that_expires(self) -> None:
        self.assertEqual(production_prefix("house2"), "production_house2")
        self.assertTrue(production_prefix("house2").startswith(PRODUCTION_PREFIX_ROOT))
        self.assertEqual(PRODUCTION_RETENTION_DAYS, 14)

    def test_mode_defaults_to_data_collection(self) -> None:
        self._write("site: house2\n")
        self.assertEqual(load_box_config(self.path).mode, "data_collection")

    def test_mode_inference(self) -> None:
        self._write("site: house2\nmode: inference\n")
        self.assertEqual(load_box_config(self.path).mode, "inference")

    def test_unknown_mode_raises(self) -> None:
        self._write("site: house2\nmode: turbo\n")
        with self.assertRaises(BoxConfigError):
            load_box_config(self.path)

    def test_load_box_settings_returns_every_key(self) -> None:
        self._write("site: house2\nmode: inference\nalert_start_hour: 22\nowner_phone: '+972500000000'\n")
        settings = load_box_settings(self.path)
        self.assertEqual(settings["alert_start_hour"], 22)
        self.assertEqual(settings["owner_phone"], "+972500000000")

    def test_load_box_settings_missing_file_raises(self) -> None:
        with self.assertRaises(BoxConfigError):
            load_box_settings(self.path)

    def test_set_site_replaces_only_the_site_line(self) -> None:
        self._write('# a comment\nsite: "old_house"\nmode: inference\nalert_start_hour: 22\n')
        set_site("house2", self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(
                f.read(),
                '# a comment\nsite: "house2"\nmode: inference\nalert_start_hour: 22\n',
            )
        self.assertEqual(load_box_config(self.path), BoxConfig(site="house2", min_age_minutes=10.0, mode="inference"))

    def test_set_site_creates_the_file(self) -> None:
        set_site("house2", self.path)
        self.assertEqual(load_box_config(self.path).site, "house2")

    def test_set_site_adds_the_line_when_missing(self) -> None:
        self._write("min_age_minutes: 5\n")
        set_site("house2", self.path)
        cfg = load_box_config(self.path)
        self.assertEqual((cfg.site, cfg.min_age_minutes), ("house2", 5.0))

    def test_option_defaults_to_false(self) -> None:
        self.assertFalse(get_option("show_cameras", self.path))  # no file at all
        self._write("site: house2\n")
        self.assertFalse(get_option("show_cameras", self.path))

    def test_set_option_stores_true_and_false_and_keeps_other_lines(self) -> None:
        self._write('site: "house2"\nmode: inference\n')
        self.assertTrue(set_option("show_cameras", "Yes", self.path))
        self.assertTrue(get_option("show_cameras", self.path))
        self.assertFalse(set_option("show_cameras", "false", self.path))
        self.assertFalse(get_option("show_cameras", self.path))
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), 'site: "house2"\nmode: inference\nshow_cameras: false\n')

    def test_set_option_rejects_unknown_keys_and_values(self) -> None:
        with self.assertRaises(BoxConfigError):
            set_option("site", "true", self.path)
        with self.assertRaises(BoxConfigError):
            set_option("show_cameras", "maybe", self.path)
        with self.assertRaises(BoxConfigError):
            get_option("nope", self.path)

    def test_inference_options_are_validated_and_stored(self) -> None:
        self._write('site: "house2"\n')
        self.assertEqual(set_option("mode", "inference", self.path), "inference")
        self.assertEqual(set_option("alert_start_hour", "22", self.path), 22)
        self.assertEqual(set_option("alert_end_hour", "6", self.path), 6)
        self.assertEqual(set_option("alert_channel", "telegram", self.path), "telegram")
        self.assertEqual(set_option("telegram_chat_ids", "-1001234567,987654", self.path), "-1001234567,987654")
        self.assertFalse(set_option("notify_dry_run", "off", self.path))

        settings = load_box_settings(self.path)
        self.assertEqual(settings["alert_start_hour"], 22)
        self.assertEqual(settings["alert_end_hour"], 6)
        self.assertEqual(settings["alert_channel"], "telegram")
        self.assertEqual(settings["telegram_chat_ids"], "-1001234567,987654")
        self.assertIs(settings["notify_dry_run"], False)
        self.assertEqual(load_box_config(self.path), BoxConfig(site="house2", min_age_minutes=10.0, mode="inference"))
        self.assertEqual(get_option("alert_start_hour", self.path), 22)
        self.assertEqual(get_option("mode", self.path), "inference")

    def test_unset_options_have_defaults(self) -> None:
        self._write('site: "house2"\n')
        self.assertFalse(get_option("notify_dry_run", self.path))
        self.assertEqual(get_option("mode", self.path), "data_collection")
        self.assertIsNone(get_option("alert_start_hour", self.path))
        self.assertIsNone(get_option("telegram_chat_ids", self.path))

    def test_inference_options_reject_bad_values_and_write_nothing(self) -> None:
        self._write('site: "house2"\n')
        for key, bad in (
            ("mode", "turbo"),
            ("alert_start_hour", "24"),
            ("alert_end_hour", "-1"),
            ("alert_start_hour", "ten"),
            ("alert_channel", "pigeon"),
            ("telegram_chat_ids", "12, 34"),
            ("telegram_chat_ids", "abc"),
            ("telegram_chat_ids", ""),
        ):
            with self.assertRaises(BoxConfigError, msg=f"{key}={bad!r}"):
                set_option(key, bad, self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), 'site: "house2"\n')

    def test_set_site_rejects_bad_names(self) -> None:
        for bad in ("House 2", "", "house-2", "../x"):
            with self.assertRaises(BoxConfigError):
                set_site(bad, self.path)
        self.assertFalse(os.path.exists(self.path))


if __name__ == "__main__":
    unittest.main()
