from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box.boxconfig import (
    BoxConfig,
    BoxConfigError,
    load_box_config,
    s3_prefix,
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


if __name__ == "__main__":
    unittest.main()
