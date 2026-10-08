"""fleet_contract/camera_names.py is a verbatim copy of home_guard_project/box/camera_names.py (branch beelink-collector-box).

The admin shows the owner's camera names by the box's own rule (alias, else "Camera N" from the channel, never the
raw id). This test fails when the two drift: copy the box file again
(`git show beelink-collector-box:home_guard_project/box/camera_names.py`).
"""
import pathlib
import subprocess

import pytest

from home_guard_project.fleet_contract import camera_names
from home_guard_project.fleet_contract.camera_names import channel_of, display_name

VENDORED = pathlib.Path(camera_names.__file__)
SOURCES = ("beelink-collector-box:home_guard_project/box/camera_names.py",
           "origin/beelink-collector-box:home_guard_project/box/camera_names.py")


def _box_copy():
    for ref in SOURCES:
        try:
            out = subprocess.run(["git", "show", ref], capture_output=True, cwd=VENDORED.parent, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode == 0:
            return out.stdout.decode("utf-8")
    return None


def _normal(text):
    return text.replace("\r\n", "\n").strip()


def test_vendored_copy_matches_the_box_camera_names():
    box = _box_copy()
    if box is None:
        pytest.skip("branch beelink-collector-box is not available in this checkout")
    assert _normal(VENDORED.read_text(encoding="utf-8")) == _normal(box)


def test_without_aliases_the_channel_names_the_camera():
    assert channel_of("ameer_week_0_1_ch6") == "6" and channel_of("front_door") is None
    assert display_name("ameer_tes2_ch6", "en", {}) == "Camera 6"
    assert display_name("ameer_tes2_ch6", "he", {}) == "מצלמה 6"
    assert display_name("front_door", "en", {}) == "front door"


def test_an_alias_saved_before_a_site_rename_still_names_the_camera():
    aliases = {"ameer_tes2_ch1": ["entrance", "כניסה ראשית"]}
    assert display_name("ameer_week_0_1_ch1", "he", aliases) == "כניסה ראשית"
    assert display_name("ameer_week_0_1_ch2", "en", aliases) == "Camera 2"
