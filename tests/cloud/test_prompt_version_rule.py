"""Which schema a clip's tag follows when its prompt version was not recorded: a clip the box's AI answered was
answered by the legacy prompt (the only one before the Eye; Eye answers always record eye-*), so the legacy form is
used and saved as LEGACY_ASSUMED; only a clip no AI answered gets the Eye's category form."""
from datetime import datetime, timezone
from types import SimpleNamespace

from home_guard_project.cloud.tagstudio.config import StudioPaths
from home_guard_project.admin.tagging_client import _MemoryStudio
from home_guard_project.fleet_contract import prompt_schemas as ps

from .tagstudio_fixtures import alert_meta, dataset_row, make_dataset, put_owner_clip

STEM = "house2_ch2_1791091199_alert"


def _studio(tmp_path, **meta_extra):
    ds = make_dataset(str(tmp_path / "ds"), [dataset_row("front_side_1771696865_trigger", "A man walks past.")])
    put_owner_clip(ds, "production_house2", alert_meta("house2_ch2", STEM, label="normal", **meta_extra), STEM)
    return _MemoryStudio(StudioPaths.resolve(env={}, dataset=ds, eval_dir=str(tmp_path / "eval"),
                                             exports=str(tmp_path / "exports")))


def test_an_ai_answer_without_a_recorded_version_is_legacy_assumed(tmp_path):
    s = _studio(tmp_path, teacher=None)                       # the AI answered (alert + model_response), no teacher
    key = f"of:production_house2/{STEM}"
    detail = s.detail(None, key)
    assert detail["prompt_version"] == ps.LEGACY_ASSUMED and detail["answer_schema"]["kind"] == ps.LEGACY
    assert ps.schema_kind(ps.LEGACY_ASSUMED) == ps.LEGACY
    staff = SimpleNamespace(id=1, name="me")
    s.save(None, staff, key, {"raw_label": "normal", "description": "Two children play."}, datetime.now(timezone.utc))
    assert s.tags(None)[key].fields["prompt_version"] == ps.LEGACY_ASSUMED      # saved, and says it was assumed


def test_a_recorded_version_wins_and_a_clip_no_ai_answered_gets_the_category_form(tmp_path):
    s = _studio(tmp_path, teacher={"model": "qwen", "prompt_version": ps.EYE_PROMPT_VERSION})
    assert s.detail(None, f"of:production_house2/{STEM}")["answer_schema"]["kind"] == ps.EYE
    assert s.detail(None, "ds:front_side_1771696865_trigger")["prompt_version"] == ""


def test_the_model_comes_from_the_teacher_or_the_alert(tmp_path):
    meta = alert_meta("house2_ch2", STEM, teacher=None)
    meta["alert"]["model"] = "qwen/qwen3.5-9b"
    s = _studio(tmp_path / "a", teacher=None, alert=meta["alert"])
    ai = s.detail(None, f"of:production_house2/{STEM}")["opinions"]["ai"]
    assert ai["detail"]["model"] == "qwen/qwen3.5-9b"
