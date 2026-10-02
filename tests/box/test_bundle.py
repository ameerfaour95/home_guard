from __future__ import annotations

import os
import unittest

from home_guard_project.box.boxconfig import PROJECT_ROOT
from home_guard_project.box.make_bundle import CREDENTIAL_URL, bundle_files

MUST_INCLUDE = [
    "pyproject.toml",
    "uv.lock",
    "home_guard_project/data_collection/config.py",
    "home_guard_project/data_collection/config.yaml",
    "home_guard_project/data_collection/data_collection.py",
    "home_guard_project/data_collection/discover.py",
    "home_guard_project/s3_upload/s3_upload.py",
    "home_guard_project/s3_upload/config.yaml",
    "home_guard_project/labeling/utils/ffmpeg.py",
    "home_guard_project/box/run_collector.sh",
    "home_guard_project/box/setup_box.ps1",
    "home_guard_project/box/config.box.yaml",
]

MUST_EXCLUDE_NAMES = {
    "cameras.yaml",
    "zones.yaml",
    "box.yaml",
    "api_key.env",
    "run_with_gpt.py",
    "open_camera.py",
    "fortified_security_smolvlm.py",
}


class BundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.files = bundle_files(PROJECT_ROOT)

    def test_required_files_are_included(self) -> None:
        for rel in MUST_INCLUDE:
            self.assertIn(rel, self.files)

    def test_secret_and_private_files_are_excluded(self) -> None:
        for rel in self.files:
            self.assertNotIn(os.path.basename(rel), MUST_EXCLUDE_NAMES, rel)
        self.assertNotIn("home_guard_project/labeling/config.yaml", self.files)
        self.assertFalse([f for f in self.files if "__pycache__" in f or f.endswith(".pyc")])

    def test_no_file_contains_a_camera_url_with_credentials(self) -> None:
        for rel in self.files:
            if rel.endswith(".pt"):
                continue
            with open(os.path.join(PROJECT_ROOT, rel), encoding="utf-8", errors="ignore") as f:
                self.assertIsNone(CREDENTIAL_URL.search(f.read()), rel)

    def test_paths_are_sorted_and_relative(self) -> None:
        self.assertEqual(self.files, sorted(self.files))
        self.assertFalse([f for f in self.files if os.path.isabs(f) or "\\" in f])


if __name__ == "__main__":
    unittest.main()
