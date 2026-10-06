"""fleet_contract/taxonomy.py is a verbatim copy of home_guard_project/box/taxonomy.py (branch beelink-collector-box).

The Admin Center checkout has no box/ package, so the categories are vendored. This test fails when the two
drift: copy the box file again (`git show beelink-collector-box:home_guard_project/box/taxonomy.py`).
"""
import pathlib
import subprocess

import pytest

from home_guard_project.fleet_contract import taxonomy

VENDORED = pathlib.Path(taxonomy.__file__)
SOURCES = ("beelink-collector-box:home_guard_project/box/taxonomy.py",
           "origin/beelink-collector-box:home_guard_project/box/taxonomy.py")


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


def test_vendored_copy_matches_the_box_taxonomy():
    box = _box_copy()
    if box is None:
        pytest.skip("branch beelink-collector-box is not available in this checkout")
    assert _normal(VENDORED.read_text(encoding="utf-8")) == _normal(box)


def test_the_fixed_list():
    ids = [c.id for c in taxonomy.CATEGORIES]
    assert ids[:2] == ["N1", "N2"] and len([i for i in ids if i[0] == "N"]) == 10
    assert len([i for i in ids if i[0] == "S"]) == 9 and len([i for i in ids if i[0] == "E"]) == 8
    assert taxonomy.CATEGORY_IDS[-1] == taxonomy.OTHER
    assert all(c.he for c in taxonomy.CATEGORIES)
    table = taxonomy.as_table()
    assert table["categories"][-1]["id"] == "other" and table["zones"] and table["flags"]
