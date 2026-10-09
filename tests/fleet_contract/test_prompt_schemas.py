"""fleet_contract/model_input.py is a verbatim copy of home_guard_project/data_collection/model_input.py, and
fleet_contract/prompt_schemas.py carries verbatim copies of the answer schemas of box/inference.py (the legacy prompt)
and box/eye_prompt.py (eye-v3), all on branch beelink-collector-box. These tests fail when the box's copies change:
copy them again. The branch is read from this checkout, else from the home_guard repo next to it."""
import pathlib
import subprocess

import pytest

from home_guard_project.fleet_contract import model_input, prompt_schemas as ps

BOX_REF = "origin/beelink-collector-box"          # the one place the branch is named
REPOS = (pathlib.Path(ps.__file__).parent, pathlib.Path("C:/Users/ameer/Ameer/home_guard"))


def box_file(path, needle):
    for repo in REPOS:
        for ref in (BOX_REF, BOX_REF.split("/", 1)[1]):
            try:
                out = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            if out.returncode == 0 and needle.encode() in out.stdout:
                return out.stdout.decode("utf-8").replace("\r\n", "\n")
    pytest.skip(f"{BOX_REF}:{path} is not available here")


def _normal(text):
    lines = [ln for ln in text.replace("\r\n", "\n").strip().split("\n")]
    while lines and (not lines[-1].strip() or set(lines[-1].strip()) <= {"#", "-", " "}):
        lines.pop()
    return "\n".join(lines).strip()


def _vendored(name):
    text = pathlib.Path(ps.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return text.split(f"# >>> vendored from {name}\n", 1)[1].split("# <<< vendored", 1)[0]


def test_model_input_is_a_verbatim_copy():
    box = box_file("home_guard_project/data_collection/model_input.py", "def render_model_input")
    assert _normal(pathlib.Path(model_input.__file__).read_text(encoding="utf-8")) == _normal(box)


def test_legacy_schema_block_matches_box_inference():
    box = box_file("home_guard_project/box/inference.py", "VLM_SCHEMA")
    start = box.index("# Bumped whenever the prompt or the answer's schema changes")
    end = box.index("\n}\n", box.index("VLM_SCHEMA: Dict", start)) + 2
    assert _normal(_vendored("box/inference.py")) == _normal(box[start:end])


def test_eye_schema_block_matches_box_eye_prompt():
    box = box_file("home_guard_project/box/eye_prompt.py", "EYE_PROMPT_VERSION")
    start = box.index("EYE_PROMPT_VERSION = ")
    end = box.index("# Prompt modules", start)
    assert _normal(_vendored("box/eye_prompt.py")) == _normal(box[start:end])


def test_schema_by_prompt_version():
    assert ps.field_order(ps.PROMPT_VERSION) == ("summary", "label", "raw_label", "applied_fact_id",
                                                 "serious_behaviour", "people", "vehicle_moving", "animals", "why",
                                                 "summary_owner")
    assert ps.field_order("v1") == ps.field_order(ps.PROMPT_VERSION)
    assert ps.schema_kind(None) == ps.schema_kind("") == ps.EYE      # an old clip no prompt answered
    eye = ps.field_order(ps.EYE_PROMPT_VERSION + "+tf1")
    assert eye[:3] == ("summary", "category", "other_text") and eye[-1] == "why"
    assert ps.answer_schema(ps.EYE_PROMPT_VERSION)[1]["additionalProperties"] is False
