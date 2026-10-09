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

    def __init__(self, answer, answered_by=None):
        self.answer, self.calls, self.answered_by = answer, [], answered_by
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(model=self.answered_by,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.answer)))])


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
        assert row.detail == {"model": "test/converter", "model_id": "test/converter", "language": "en",
                              "prompt_version": "2026-10-03.test", "words_source": "staff"}
        assert session.scalars(select(m.TagEvent)).all() == []          # a suggestion: nothing is saved


# ---------------------------------------------------------------- privacy: whose words, consent, nothing that names a home

def _privacy_setup(client, s, ids):
    """The consenting household's camera has an owner name; the refusing household's clip has an owner answer."""
    with session_scope(client.app.state.engine) as session:
        house2 = session.scalar(select(m.Device).where(m.Device.site == "house2"))
        session.add(m.Camera(device_pk=house2.id, name="house2_ch2", display_name="חניה"))
        other = session.scalar(select(m.Device).where(m.Device.site == "other"))
        session.add(m.Feedback(device_pk=other.id, event_id=ids["refusing"], alert_stem="other_ch1_1791091300_alert",
                               verdict="real_but_wrong", raw_text="it was our gardener Moshe", s3_key="fb/other/1",
                               owner_text="it was our gardener Moshe"))
        consenting_fb = m.Feedback(device_pk=house2.id, event_id=ids["consenting"], alert_stem=STEM,
                                   verdict="real_but_wrong", owner_text="זה הגנן שלנו", s3_key="fb/house2/1")
        session.add(consenting_fb); session.flush()
        return consenting_fb.id


def test_customer_words_need_training_consent_and_staff_words_never_do(client, staff_factory, studio):  # noqa: F811
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    fb_id = _privacy_setup(client, s, ids)
    s.convert_client = FakeModel(LEGACY_ANSWER)
    refusing, consenting = f"ev:{ids['refusing']}", f"ev:{ids['consenting']}"
    r = client.post("/v1/tagging/convert", headers=h, json={"key": refusing, "words": "it was our gardener Moshe",
                                                            "words_source": "owner_answer"})
    assert r.status_code == 403 and r.json()["detail"] == ("This customer hasn't agreed to training use, so their "
                                                           "words can't be sent to the converter")
    # the server decides: the owner's own answer typed in as "staff" words is still the owner's
    r = client.post("/v1/tagging/convert", headers=h, json={"key": refusing, "words": "Note: it was our gardener Moshe!",
                                                            "words_source": "staff"})
    assert r.status_code == 403
    assert len(s.convert_client.calls) == 0                              # nothing left the machine
    r = client.post("/v1/tagging/convert", headers=h, json={"key": refusing, "words": "a man walks to the gate",
                                                            "words_source": "staff"})
    assert r.status_code == 200 and r.json()["words_source"] == "staff"   # the founder's own words: always
    r = client.post("/v1/tagging/convert", headers=h, json={"key": consenting, "words": "זה הגנן שלנו",
                                                            "words_source": "transcript", "feedback_id": fb_id})
    assert r.status_code == 200 and r.json()["words_source"] == "transcript"
    r = client.post("/v1/tagging/convert", headers=h, json={"key": refusing, "words": "x y z",
                                                            "words_source": "owner_answer", "feedback_id": fb_id})
    assert r.status_code == 422                                          # that answer is about another clip


def test_the_outgoing_payload_names_no_camera_house_or_customer(client, staff_factory, studio):  # noqa: F811
    s, ids = studio
    _, _, _, h = staff_factory("admin")
    _privacy_setup(client, s, ids)
    fake = FakeModel(LEGACY_ANSWER, answered_by="google/gemini-3.1-flash-lite-001")
    s.convert_client = fake
    words = ("At house2_ch2 (the חניה camera of House two, site house2) a man walked to the gate; "
             "the clip is under production_house2 and the owner Two said he left")
    r = client.post("/v1/tagging/convert", headers=h, json={"key": f"ev:{ids['consenting']}", "words": words})
    assert r.status_code == 200, r.text
    sent = json.dumps(fake.calls[-1], ensure_ascii=False)
    for private in ("house2_ch2", "חניה", "House two", "house2", "production_house2"):
        assert private not in sent, private
    assert "a man walked to the gate" in sent and "camera" in sent and "the house" in sent
    body = r.json()
    assert body["words_sent"] in fake.calls[-1]["messages"][0]["content"]
    # the exact model id that answered is returned and stored with the tag
    assert body["model"] == "test/converter" and body["model_id"] == "google/gemini-3.1-flash-lite-001"
    fields = {"raw_label": "normal", "description": "A man walks to the gate.", "tagger_words": words,
              "converted_by": body["model"], "converted_model": body["model_id"], "words_source": body["words_source"]}
    r = client.post("/v1/tagging/tag", headers=h, json={"key": f"ev:{ids['consenting']}", "fields": fields})
    assert r.status_code == 200, r.text
    saved = r.json()["tag"]["fields"]
    assert saved["converted_model"] == "google/gemini-3.1-flash-lite-001" and saved["words_source"] == "staff"


def test_redact_replaces_names_longest_first_and_id_shapes():
    out = cv.redact("Camera Front Door at Front door house, cam ameer_week_0_1_ch6, prefix dataset_ameer_tes2",
                    cameras=["Front Door", "Front Door house"], places=["Ameer"])
    assert "Front" not in out and "ch6" not in out and "ameer" not in out.lower()
    assert out.count("camera") >= 3
