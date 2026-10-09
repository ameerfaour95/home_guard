""""In my words" -> the tag (tagstudio/convert.py): only the words, the taxonomy and the schema are sent (no picture),
the answer is forced into the clip's own prompt-version schema in its field order, values outside it are dropped, and
POST /v1/tagging/convert is audited."""
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.tagstudio import convert as cv
from home_guard_project.fleet_contract import prompt_schemas as ps

from .test_tagging_routes import STEM, studio  # noqa: F401 (fixture)


class FakeModel:
    model = "test/converter"

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.answer)))])


LEGACY_ANSWER = {"summary": "A man looks over the back fence.", "label": "suspicious", "raw_label": "suspicious",
                 "applied_fact_id": "", "serious_behaviour": True, "people": 1, "vehicle_moving": False,
                 "animals": 0, "why": "He looks into the yard.", "summary_owner": "גבר מציץ מעל הגדר"}


def test_legacy_words_become_the_legacy_answer_in_its_order():
    fake = FakeModel(LEGACY_ANSWER)
    out = cv.Converter(cv.ConvertConfig(model="m", api_key="k"), client=fake).convert(
        "גבר מציץ מעל הגדר האחורית", ps.PROMPT_VERSION)
    call = fake.calls[0]
    fmt = call["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert tuple(fmt["json_schema"]["schema"]["properties"]) == ps.field_order(ps.PROMPT_VERSION)
    # only text leaves the machine: one user message, a string, the words inside it
    assert [msg["role"] for msg in call["messages"]] == ["user"] and isinstance(call["messages"][0]["content"], str)
    assert "גבר מציץ מעל הגדר האחורית" in call["messages"][0]["content"]
    assert out["language"] == "he" and out["schema"] == "legacy_alert" and out["model"] == "m"
    assert list(out["fields"]) == ["description", *ps.field_order(ps.PROMPT_VERSION)[1:]]
    assert out["fields"]["description"] == "A man looks over the back fence." and out["fields"]["serious_behaviour"]


def test_eye_words_and_values_outside_the_schema_are_dropped():
    answer = {"summary": "A courier leaves a parcel.", "category": "Q9", "other_text": "x", "zone": "moon",
              "movement": "approaching", "flags": ["carrying_item", "nope"], "people": -2, "vehicles": 1,
              "vehicle_moving": "yes", "animals": 0, "visibility": "clear", "appearance": ["blue cap"],
              "evidence_frame": 3, "raw_label": "normal", "label": "normal", "applied_fact_id": "",
              "serious_behaviour": False, "why": ""}
    out = cv.Converter(cv.ConvertConfig(api_key="k"), client=FakeModel(answer)).convert(
        "courier with a parcel", ps.EYE_PROMPT_VERSION)
    f = out["fields"]
    assert out["language"] == "en" and out["schema"] == "eye_alert_triage"
    assert f["category"] == "" and f["zone"] == "" and f["movement"] == "approaching" and f["people"] == 0
    assert f["vehicle_moving"] is False and f["evidence_frame"] == 3
    assert all(flag in ps.tx.FLAGS for flag in f["flags"])


def test_no_words_or_no_key_say_why():
    with pytest.raises(cv.ConvertError, match="Write what you saw"):
        cv.Converter(cv.ConvertConfig(api_key="k"), client=FakeModel({})).convert("  ", None)
    with pytest.raises(cv.ConvertError, match="OPENROUTER_API_KEY"):
        cv.Converter(cv.ConvertConfig(api_key="")).convert("words", None)
    assert cv.ConvertConfig.resolve(env={}).model == "google/gemini-3.1-flash-lite"
    assert cv.ConvertConfig.resolve(env={"HG_CONVERT_MODEL": "x/y"}).model == "x/y"
    assert cv.language("مرحبا") == "ar" and cv.language("привет") == "ru"


def test_route_converts_with_the_clip_prompt_version_and_audits(client, staff_factory, studio):  # noqa: F811
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    fake = FakeModel(LEGACY_ANSWER)
    s.convert_client = fake
    r = client.post("/v1/tagging/convert", headers=h, json={"key": f"ev:{ids['consenting']}",
                                                            "words": "a man looks over the fence"})
    assert r.status_code == 200, r.text
    body = r.json()
    # the clip's meta says teacher.prompt_version "2026-10-03.test": a legacy-schema clip
    assert body["prompt_version"] == "2026-10-03.test" and body["schema_name"] == "legacy_alert"
    assert body["model"] == "test/converter" and body["fields"]["why"] == "He looks into the yard."
    with session_scope(client.app.state.engine) as session:
        row = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "tag_converted"))
        assert row.detail == {"model": "test/converter", "language": "en", "prompt_version": "2026-10-03.test"}
        assert session.scalars(select(m.TagEvent)).all() == []          # a suggestion: nothing is saved
