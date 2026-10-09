"""fleet_contract/judgement.py carries a verbatim copy of the box's "is a message a judgement of an alert at all?"
section (home_guard_project/box/feedback.py, branch beelink-collector-box). This test fails when the two drift: copy
the section again. The box branch is read from this checkout, else from the home_guard repo next to it."""
import pathlib
import re
import subprocess

import pytest

from home_guard_project.fleet_contract import judgement

VENDORED = pathlib.Path(judgement.__file__)
BOX_REF = "origin/beelink-collector-box"          # the one place the branch is named
BOX_FILE = "home_guard_project/box/feedback.py"
REPOS = (VENDORED.parent, pathlib.Path("C:/Users/ameer/Ameer/home_guard"))
START = "# Is a message a judgement of an alert at all?"


def _box_copy():
    for repo in REPOS:
        for ref in (BOX_REF, BOX_REF.split("/", 1)[1]):
            try:
                out = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{BOX_FILE}"], capture_output=True,
                                     timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            if out.returncode == 0 and START.encode() in out.stdout:
                return out.stdout.decode("utf-8")
    return None


def _section(text):
    """From the section's title to the end of not_a_judgement, without trailing separators and blank lines."""
    text = text.replace("\r\n", "\n")
    start = text.index(START)
    end = text.index("\n", text.index("def not_a_judgement", start))
    body = text[end:]
    stop = re.search(r"\n(?!    |\n)", body)       # the first line that is not part of the function
    return text[start:end + (stop.start() if stop else len(body))].rstrip()


def test_vendored_section_matches_the_box():
    box = _box_copy()
    if box is None:
        pytest.skip("branch beelink-collector-box (with not_a_judgement) is not available here")
    vendored = VENDORED.read_text(encoding="utf-8").replace("\r\n", "\n")
    block = vendored.split("# >>> vendored from box/feedback.py\n", 1)[1].split("# <<< vendored", 1)[0]
    assert _section(block) == _section(box)


@pytest.mark.parametrize("text, expected", [
    ("זה הגנן שלנו", False), ("it was the gardener", False), ("delivery guy", False),
    ("למה המצלמה שולחת לי כל כך הרבה התראות?", True), ("איזה תזכורת, אתה מטומטם ומגזים", True),
    ("די עם ההודעה המטופשת הזאת", True), ("stop sending these", True), ("show me the video", True),
    ("", True), (None, True)])
def test_the_rule(text, expected):
    assert judgement.not_a_judgement(text) is expected
