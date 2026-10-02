from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.data_collection import config as dc_config

OVERLAY_ENV = "HOME_GUARD_CONFIG_OVERLAY"
BOX_OVERLAY = os.path.join(
    os.path.dirname(os.path.abspath(dc_config.__file__)), "..", "box", "config.box.yaml"
)

BASE_YAML = """\
display:
  show_windows: true
  show_plotted_boxes: true
vlm:
  enabled: true
"""

OVERLAY_YAML = """\
display:
  show_windows: false
vlm:
  enabled: false
"""


class ConfigOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.base = os.path.join(self.dir, "config.yaml")
        with open(self.base, "w", encoding="utf-8") as f:
            f.write(BASE_YAML)
        self.overlay = os.path.join(self.dir, "overlay.yaml")
        with open(self.overlay, "w", encoding="utf-8") as f:
            f.write(OVERLAY_YAML)

    def _load(self, config_path: str | None = None) -> dc_config.Config:
        return dc_config.load_config(
            config_path=config_path or self.base,
            cameras_path=os.path.join(self.dir, "no_cameras.yaml"),
            zones_path=os.path.join(self.dir, "no_zones.yaml"),
        )

    def test_overlay_overrides_nested_keys_and_keeps_siblings(self) -> None:
        with mock.patch.dict(os.environ, {OVERLAY_ENV: self.overlay}):
            cfg = self._load()
        self.assertFalse(cfg.SHOW_WINDOWS)
        self.assertFalse(cfg.RUN_VLM_ON_SAVED_CLIPS)
        self.assertTrue(cfg.SHOW_PLOTTED_BOXES)

    def test_without_env_base_values_are_used(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != OVERLAY_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = self._load()
        self.assertTrue(cfg.SHOW_WINDOWS)
        self.assertTrue(cfg.RUN_VLM_ON_SAVED_CLIPS)

    def test_missing_overlay_file_raises(self) -> None:
        missing = os.path.join(self.dir, "nope.yaml")
        with mock.patch.dict(os.environ, {OVERLAY_ENV: missing}):
            with self.assertRaises(FileNotFoundError):
                self._load()

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 1}
        overlay = {"a": {"x": 9}, "c": 3}
        merged = dc_config._deep_merge(base, overlay)
        self.assertEqual(merged, {"a": {"x": 9, "y": 2}, "b": 1, "c": 3})
        self.assertEqual(base, {"a": {"x": 1, "y": 2}, "b": 1})
        self.assertEqual(overlay, {"a": {"x": 9}, "c": 3})

    def test_box_overlay_makes_real_config_headless(self) -> None:
        with mock.patch.dict(os.environ, {OVERLAY_ENV: BOX_OVERLAY}):
            cfg = self._load(config_path=dc_config._CONFIG_PATH)
        self.assertFalse(cfg.SHOW_WINDOWS)
        self.assertFalse(cfg.RUN_VLM_ON_SAVED_CLIPS)
        self.assertTrue(cfg.RANDOM_CLIP_ENABLED)


if __name__ == "__main__":
    unittest.main()
