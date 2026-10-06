from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from home_guard_project.box.brain.aliases import add_alias, load_aliases, normalize, remap_aliases, remove_alias
from home_guard_project.box.find_cameras import apply_changes

CAMS = ["main_entrance", "back_door", "front_side"]


class AliasesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "camera_aliases.yaml")

    def test_missing_file_is_empty(self) -> None:
        self.assertEqual(load_aliases(self.path), {})

    def test_normalize(self) -> None:
        self.assertEqual(normalize("  Front_Door  "), "front door")
        self.assertEqual(normalize("הכניסה"), "הכניסה")

    def test_add_alias_keeps_the_owners_spelling_and_dedupes(self) -> None:
        self.assertEqual(add_alias("main_entrance", "Entrance", CAMS, self.path), ["Entrance"])
        self.assertEqual(add_alias("main_entrance", "entrance", CAMS, self.path), ["Entrance"])
        self.assertEqual(add_alias("main_entrance", "כניסה", CAMS, self.path), ["Entrance", "כניסה"])
        self.assertEqual(load_aliases(self.path), {"main_entrance": ["Entrance", "כניסה"]})

    def test_add_alias_refuses_collisions_unknown_cameras_and_empty_names(self) -> None:
        add_alias("main_entrance", "front", CAMS, self.path)
        with self.assertRaises(ValueError):
            add_alias("front_side", "front", CAMS, self.path)
        with self.assertRaises(ValueError):
            add_alias("front_side", "back door", CAMS, self.path)   # another camera's own name
        with self.assertRaises(ValueError):
            add_alias("garage", "garage", CAMS, self.path)
        with self.assertRaises(ValueError):
            add_alias("front_side", "   ", CAMS, self.path)

    def test_remove_alias_takes_back_one_name(self) -> None:
        add_alias("main_entrance", "Entrance", CAMS, self.path)
        add_alias("main_entrance", "פרגולה", CAMS, self.path)
        self.assertEqual(remove_alias("main_entrance", "פרגולה", self.path), ["Entrance"])
        self.assertEqual(remove_alias("main_entrance", "nothing", self.path), ["Entrance"])
        self.assertEqual(remove_alias("main_entrance", " entrance ", self.path), [])
        self.assertEqual(load_aliases(self.path), {})

    def test_remap_follows_renames_and_swaps(self) -> None:
        add_alias("main_entrance", "entrance", CAMS, self.path)
        add_alias("back_door", "back", CAMS, self.path)
        remap_aliases({"main_entrance": "back_door", "back_door": "main_entrance"}, self.path)
        self.assertEqual(load_aliases(self.path), {"back_door": ["entrance"], "main_entrance": ["back"]})

    def test_damaged_file_reads_as_empty(self) -> None:
        with open(self.path, "wb") as f:
            f.write(b"\xff\xfe\x00bad")
        self.assertEqual(load_aliases(self.path), {})

    def test_remap_renamed_entry_beats_a_stale_one(self) -> None:
        for text in ("a: [x]\nb: [y]\n", "b: [y]\na: [x]\n"):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(text)
            remap_aliases({"a": "b"}, self.path)
            self.assertEqual(load_aliases(self.path), {"b": ["x"]})

    def test_apply_changes_carries_aliases_on_rename(self) -> None:
        cameras = os.path.join(self.dir, "cameras.yaml")
        with open(cameras, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": {"test_ch6": "rtsp://a", "test_ch8": "rtsp://b"}}, f)
        add_alias("test_ch6", "entrance", ["test_ch6", "test_ch8"], self.path)
        apply_changes({"cameras": [{"name": "test_ch6", "new_name": "main_entrance", "enabled": True}]},
                      cameras, zones_path=os.path.join(self.dir, "zones.yaml"), restart=False,
                      aliases_path=self.path)
        # The aliases follow the camera, and its old name becomes one of them.
        self.assertEqual(load_aliases(self.path), {"main_entrance": ["entrance", "test_ch6"]})


if __name__ == "__main__":
    unittest.main()
