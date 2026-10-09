"""The training export as LLaMA-Factory sharegpt: every studio tag becomes the box's own JSON answer in the exact field
order of its clip's prompt version, and every row round-trips through the loader's rules (alternating turns, one
<video> tag per video, a JSON-object answer)."""
import json

from home_guard_project.cloud.tagstudio import export as ex
from home_guard_project.cloud.tagstudio.fields import Tag
from home_guard_project.fleet_contract import prompt_schemas as ps

from .test_tagstudio import TAGS, _items

LEGACY_TAG = Tag("ds:old_ok_1771696865_trigger", {
    "raw_label": "suspicious", "label": "suspicious", "description": "A man looks into the car.",
    "serious_behaviour": True, "people": 1, "why": "He looks into a car.", "summary_owner": "גבר מציץ לרכב",
    "prompt_version": ps.PROMPT_VERSION, "tagger_words": "גבר מציץ לתוך הרכב", "tagger_language": "he",
    "converted_by": "google/gemini-3.1-flash-lite"}, "2026-10-09T10:00:00Z", "me")


def _rows(tmp_path):
    tags = {**TAGS, LEGACY_TAG.key: LEGACY_TAG}
    training, _, _ = ex.build(_items(tmp_path).values(), tags)
    return training, ex.sharegpt_rows(training)


def load_like_llamafactory(rows):
    """The sharegpt loader's checks (LLaMA-Factory data/converter.py SharegptDatasetConverter + mm_plugin): roles
    alternate starting with the user, and the <video> placeholders match the videos one to one."""
    out = []
    for row in rows:
        msgs = row["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert sum(m["content"].count("<video>") for m in msgs) == len(row["videos"])
        out.append((msgs[0]["content"], json.loads(msgs[1]["content"]), row["videos"]))
    return out


def test_each_studio_tag_is_its_prompt_versions_answer_in_order(tmp_path):
    training, rows = _rows(tmp_path)
    by_id = {r["clip_id"]: r for r in training}
    legacy = by_id["old_ok_1771696865_trigger"]
    assert legacy["prompt_version"] == ps.PROMPT_VERSION and legacy["tagger_words"] == "גבר מציץ לתוך הרכב"
    assert legacy["converted_by"] == "google/gemini-3.1-flash-lite" and legacy["tagger_language"] == "he"
    assert tuple(legacy["answer"]) == ps.field_order(ps.PROMPT_VERSION)
    assert legacy["answer"]["summary"] == "A man looks into the car." and legacy["answer"]["serious_behaviour"] is True
    eye = by_id["tagged_1771696870_trigger"]         # no prompt version: the Eye's category form
    assert tuple(eye["answer"]) == ps.field_order(None) and eye["answer"]["category"] == "S3"
    assert eye["answer"]["label"] == eye["answer"]["raw_label"] == "suspicious"
    # a migrated old tag holds no structured answer: no sharegpt row for it
    assert by_id["old_alert_1771696867_trigger"]["answer"] is None
    assert {r["clip_id"] for r in rows} == {"old_ok_1771696865_trigger", "tagged_1771696870_trigger"}


def test_every_row_round_trips_through_the_loader(tmp_path):
    _, rows = _rows(tmp_path)
    assert ex.check_sharegpt(rows) == []
    loaded = load_like_llamafactory(json.loads(json.dumps(rows, ensure_ascii=False)))
    for (question, answer, videos), row in zip(loaded, rows):
        assert question.startswith("<video>") and len(videos) == 1
        assert tuple(answer) == ps.field_order(row["prompt_version"] or None)


def test_the_checker_catches_what_the_loader_refuses():
    good = {"messages": [{"role": "user", "content": "<video>q"}, {"role": "assistant", "content": "{}"}],
            "videos": ["a.mp4"]}
    assert ex.check_sharegpt([good]) == []
    two = dict(good, videos=["a.mp4", "b.mp4"])
    swapped = dict(good, messages=good["messages"][::-1])
    text = dict(good, messages=[good["messages"][0], {"role": "assistant", "content": "plain words"}])
    problems = ex.check_sharegpt([two, swapped, text])
    assert any("1 <video> tag(s) for 2" in p for p in problems)
    assert any("alternate" in p for p in problems) and any("JSON object" in p for p in problems)
