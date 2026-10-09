"""The tagging studio's logic, no database: sources (old tags, owner feedback), the fold of tag events, the
contradiction queue, teachers (eval results; an OpenAI-compatible model with a fake client), exports, media grants
and where the data lives."""
import json
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from home_guard_project.cloud.tagstudio import evalfmt
from home_guard_project.cloud.tagstudio import export as ex
from home_guard_project.cloud.tagstudio import queue as wq
from home_guard_project.cloud.tagstudio.config import StudioPaths
from home_guard_project.cloud.tagstudio.fields import Tag, TagError, clean_fields, fold
from home_guard_project.cloud.tagstudio.items import AI, OLD, OWNER, TEACHER, ClipItem, Opinion
from home_guard_project.cloud.tagstudio.service import TagStudio
from home_guard_project.cloud.tagstudio.sources import DatasetSource, OwnerFeedbackSource, owner_opinion
from home_guard_project.cloud.tagstudio.teacher import (EvalResultsTeacher, OpenAICompatibleTeacher, TeacherRefused,
                                                        teacher_prompt)

from .tagstudio_fixtures import (VIDEO_BYTES, alert_meta, dataset_row, eval_results, feedback_record, make_dataset,
                                 put_owner_clip, write)

STEM = "house2_ch2_1791091199_alert"


# ---------------------------------------------------------------- the unified dataset (old tags)
def test_dataset_reads_old_tags(tmp_path):
    root = make_dataset(str(tmp_path / "ds"), [
        dataset_row("front_side_1771696865_trigger", "A man walks past.", alert=False),
        dataset_row("front_side_1771696897_trigger", "No special activity", alert=False),
        dataset_row("Abuse_Abuse001_x264_w0000", "A man punches a woman.", alert=True, source="uca", camera="Abuse",
                    crop=False),
        dataset_row("main_door_1772290677_trigger", "[delete] blurry", alert=None),
        dataset_row("main_door_1772290999_trigger", "", alert=None, batch="claude_tagged_2026-10-06"),
    ])
    items = {i.clip_id: i for i in DatasetSource(root).load()}
    walk = items["front_side_1771696865_trigger"]
    assert walk.key == "ds:front_side_1771696865_trigger" and walk.opinions[OLD].label == "normal"
    assert walk.video_s3 == "s3://bucket/home_guard_dataset/clips/house/front_side_1771696865_trigger.mp4"
    assert walk.sort_ts == 1771696865.0 and walk.info["kind"] == "trigger" and os.path.isfile(walk.video)
    assert items["front_side_1771696897_trigger"].opinions[OLD].label == "empty"
    assert items["Abuse_Abuse001_x264_w0000"].opinions[OLD].label == "alert"
    deleted = items["main_door_1772290677_trigger"].opinions[OLD]
    assert deleted.label == "" and deleted.detail["delete"] is True
    assert OLD not in items["main_door_1772290999_trigger"].opinions      # Claude-tagged, waiting for the owner


def test_dataset_tolerates_bad_lines_and_missing_vlm_file(tmp_path):
    root = str(tmp_path / "ds")
    write(os.path.join(root, "annotations", "clips.jsonl"),
          "{broken\n" + json.dumps(dataset_row("x_1771696865_trigger", new_field=1)) + "\n")
    src = DatasetSource(root, bucket="b", s3_prefix="p")
    assert [i.clip_id for i in src.load()] == ["x_1771696865_trigger"]
    assert src.load()[0].video_s3 == "s3://b/p/clips/house/x_1771696865_trigger.mp4"
    assert src.status()["bad_lines"] == 1


def test_dataset_reloads_only_when_the_file_changes(tmp_path):
    root = make_dataset(str(tmp_path / "ds"), [dataset_row("a_1771696865_trigger")])
    src = DatasetSource(root)
    first = src.load()
    assert src.load() is first
    later = time.time() + 5
    os.utime(os.path.join(root, "annotations", "clips.jsonl"), (later, later))
    assert src.load() is not first


# ---------------------------------------------------------------- the customers' answers
def test_owner_feedback_folder(tmp_path):
    ds = str(tmp_path / "ds")
    put_owner_clip(ds, "production_house2", alert_meta("house2_ch2", STEM, label="suspicious"), STEM, answers=[
        feedback_record(STEM, "house2_ch2", "2026-10-04T05:40:00Z", "false_alarm", "empty"),
        feedback_record(STEM, "house2_ch2", "2026-10-04T05:41:00Z", "none", raw_text="is the car gone?")])
    observed = alert_meta("house2_ch3", "house2_ch3_1791091300_alert", label="suspicious")
    observed["observation"] = {"category": "s2", "zone": "window", "flags": ["flashlight"]}
    put_owner_clip(ds, "production_house2", observed, "house2_ch3_1791091300_alert")
    items = {i.clip_id: i for i in OwnerFeedbackSource(os.path.join(ds, "owner_feedback")).load()}
    item = items[STEM]
    assert item.key == f"of:production_house2/{STEM}" and item.source == "house2" and os.path.isfile(item.video)
    assert item.opinions[AI].label == "suspicious" and item.opinions[AI].knows_empty is False
    assert item.opinions[OWNER].label == "empty" and item.opinions[OWNER].detail["verdict"] == "false_alarm"
    ai = items["house2_ch3_1791091300_alert"].opinions[AI]
    assert ai.category == "S2" and ai.detail["zone"] == "window" and ai.knows_empty


@pytest.mark.parametrize("answers, ai_label, expect", [
    ([("2026-10-04T05:00:00Z", "true_alert", "")], "suspicious", ("suspicious", False)),
    ([("2026-10-04T05:00:00Z", "true_alert", "")], "normal", ("alert", False)),
    ([("2026-10-04T05:00:00Z", "real_but_wrong", "")], "normal", ("", True)),
    ([("2026-10-04T05:00:00Z", "real_but_wrong", "escalation")], "suspicious", ("escalation", False)),
    ([("2026-10-04T05:00:00Z", "expected", "normal"), ("2026-10-04T05:01:00Z", "false_alarm", "empty")],
     "normal", ("empty", False)),
])
def test_owner_answer_mapping(answers, ai_label, expect):
    op = owner_opinion([feedback_record(STEM, "cam", t, v, owner_label=o) for t, v, o in answers], ai_label)
    assert (op.label, op.disputes_ai) == expect


def test_owner_undo_takes_the_answer_back():
    records = [feedback_record(STEM, "cam", "2026-10-04T05:00:00Z", "false_alarm", "empty"),
               feedback_record(STEM, "cam", "2026-10-04T05:01:00Z", "none", note="tag undone")]
    assert owner_opinion(records) is None
    records.append(feedback_record(STEM, "cam", "2026-10-04T05:03:00Z", "expected", "normal"))
    assert owner_opinion(records).label == "normal"


# ---------------------------------------------------------------- tags
def test_fold_merges_partial_events_in_order():
    tags = fold([{"key": "a", "at": "t1", "by": "x", "fields": {"category": "S1", "description": "tries the gate"}},
                 {"key": "a", "at": "t2", "by": "y", "fields": {"needs_check": True}},
                 {"key": "a", "at": "t3", "by": "y", "fields": {"category": "S2"}},
                 {"key": "", "fields": {"category": "N1"}}, {"key": "c", "fields": "nope"}])
    a = tags["a"]
    assert set(tags) == {"a"} and a.events == 3 and (a.at, a.by) == ("t3", "y")
    assert a.fields == {"category": "S2", "description": "tries the gate", "needs_check": True}
    assert a.raw_label == "suspicious"


def test_clean_fields_validates_against_the_taxonomy():
    out = clean_fields({"category": "s3", "zone": "fence", "flags": ["running", "flashlight", "running"],
                        "raw_label": "suspicious", "evidence_sec": "2.5", "evidence_frame": 17.4, "needs_check": "true",
                        "description": "  walks the fence  "})
    assert out["category"] == "S3" and out["flags"] == ["flashlight", "running"]
    assert out["evidence_sec"] == 2.5 and out["evidence_frame"] == 17 and out["needs_check"] is True
    assert out["description"] == "walks the fence"
    for bad in ({"category": "N99"}, {"zone": "moon"}, {"flags": ["laser"]}, {"raw_label": "alert"},
                {"evidence_sec": -1}, {"evidence_sec": "nan"}, {"needs_check": "maybe"}, {"colour": "red"},
                {"description": "x" * 2001}):
        with pytest.raises(TagError):
            clean_fields(bad)


# ---------------------------------------------------------------- the work order
def item(key, sort_ts=0.0, **opinions):
    it = ClipItem(key, key, "dataset", sort_ts=sort_ts)
    for who, op in opinions.items():
        op.who = who
        it.opinions[who] = op
    return it


def op(label="", category="", **kw):
    return Opinion("", label=label, category=category, **kw)


def test_conflict_levels():
    assert wq.conflict(op("normal"), op("suspicious")) == 2
    assert wq.conflict(op("alert"), op("empty")) == 2
    assert wq.conflict(op("suspicious"), op("escalation")) == 2
    assert wq.conflict(op("alert"), op("escalation")) == 0
    assert wq.conflict(op("empty"), op("normal")) == 1
    assert wq.conflict(op("empty"), op("normal", knows_empty=False)) == 0
    assert wq.conflict(op(category="N1"), op(category="N7")) == 1
    assert wq.conflict(op(category="N1"), op(category="S3")) == 2
    assert wq.conflict(Opinion(OWNER, disputes_ai=True), Opinion(AI, label="normal")) == 1


def test_queue_order():
    rows = wq.build([
        item("done_migrated", old=op("normal")),
        item("untagged_old", sort_ts=1.0),
        item("untagged_new_alert", sort_ts=2.0, ai=op("suspicious")),
        item("minor_empty_vs_normal", old=op("empty"), owner=op("normal")),
        item("major_quiet", old=op("normal"), teacher=op("suspicious")),
        item("major_s_vs_e", ai=op("suspicious"), owner=op("escalation")),
        item("major_two", old=op("normal"), ai=op("suspicious"), teacher=op("escalation")),
        item("teacher_agrees", old=op("alert"), teacher=op("escalation")),
        item("customer_only", owner=op("normal"), ai=op("normal")),
    ], {})
    assert [i.key for i, _ in rows] == ["major_two", "major_quiet", "major_s_vs_e", "minor_empty_vs_normal",
                                        "untagged_new_alert", "untagged_old", "customer_only", "teacher_agrees",
                                        "done_migrated"]
    by = {i.key: a for i, a in rows}
    assert "old normal vs teacher suspicious" in by["major_quiet"].reasons
    assert by["customer_only"].reasons == ["untagged", "customer answered"]


def test_our_tag_settles_a_clip_until_a_later_answer_contradicts_it():
    it = item("c", old=op("normal"), teacher=op("suspicious"))
    tag = Tag("c", {"category": "N1", "raw_label": "normal"}, at="2026-10-06T10:00:00.000000Z", by="x")
    assert wq.assess(it).tier == wq.MAJOR and wq.assess(it, tag).tier == wq.DONE
    it.opinions[OWNER] = Opinion(OWNER, label="suspicious", at="2026-10-06T10:00:01Z")
    assert wq.assess(it, tag).tier == wq.MAJOR
    assert wq.assess(it, Tag("c", {"category": "N1", "needs_check": True}, at="2026-10-06T11:00:00Z")).tier == wq.MINOR
    assert wq.assess(it, Tag("c", {"delete": True}, at="x")).tier == wq.DONE


# ---------------------------------------------------------------- teachers
def test_eval_teacher_takes_the_strongest_model_per_clip(tmp_path):
    results = str(tmp_path / "results")
    eval_results(results, "p__weak", "weak", {"a": "normal", "b": "normal"}, 0.5, 0.3)
    eval_results(results, "p__strong", "strong", {"a": "suspicious"}, 0.9, 0.1)
    eval_results(results, "p__broken", "broken", {"a": "escalation"}, 1.0, 0.0, errors=60, rows=100)
    eval_results(results, "fake-p__fake", "fake", {"a": "escalation"}, 1.0, 0.0)
    teacher = EvalResultsTeacher(results)
    assert [r["model"] for r in teacher.ranking()] == ["strong", "weak"]
    a = teacher.suggest(ClipItem("ds:a", "a", "dataset"))
    assert (a.who, a.label, a.detail["model"], a.at, a.knows_empty) == (TEACHER, "suspicious", "strong", "", False)
    assert teacher.suggest(ClipItem("ds:b", "b", "dataset")).detail["model"] == "weak"
    assert EvalResultsTeacher(results, prefer=["weak"]).suggest(ClipItem("ds:a", "a", "dataset")).label == "normal"


def test_eval_teacher_reads_eval_set_and_eval_set_v2(tmp_path):
    v1, v2 = tmp_path / "eval" / "eval_set" / "results", tmp_path / "eval" / "eval_set_v2" / "results"
    eval_results(str(v1), "p__m", "m", {"a": "normal"}, 0.5, 0.3)
    eval_results(str(v2), "p__m", "m", {"a": "escalation", "b": "suspicious"}, 0.9, 0.1)
    teacher = EvalResultsTeacher(StudioPaths.resolve(env={"HOMEGUARD_EVAL_DIR": str(tmp_path / "eval")}).eval_results)
    assert [r["tag"] for r in teacher.ranking()] == ["eval_set_v2/p__m", "eval_set/p__m"]
    assert teacher.suggest(ClipItem("ds:a", "a", "dataset")).label == "escalation"
    assert teacher.suggest(ClipItem("ds:b", "b", "dataset")).label == "suspicious"


def test_openai_teacher_asks_once_and_caches(tmp_path):
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            answer = '{"summary": "a courier leaves a box", "category": "N3", "zone": "entrance", "raw_label": "normal"}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    cache = str(tmp_path / "teacher_cache.jsonl")
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 3
    teacher = OpenAICompatibleTeacher("http://gpu:8000/v1", "Qwen/Qwen3.6-27B", cache, client=client,
                                      frames_for=lambda path: frames)
    it = ClipItem("ds:a", "a", "dataset")
    assert teacher.suggest(it) is None and completions.calls == []      # suggest never calls the model
    op = teacher.ask(it, "video.mp4")
    assert (op.category, op.label, op.detail["zone"]) == ("N3", "normal", "entrance")
    assert completions.calls[0]["temperature"] == 0 and len(completions.calls[0]["messages"][0]["content"]) == 4
    again = OpenAICompatibleTeacher("http://gpu:8000/v1", "Qwen/Qwen3.6-27B", cache, client=client)
    assert again.suggest(it).category == "N3" and len(completions.calls) == 1
    assert "N3 delivery or service" in teacher_prompt()


@pytest.mark.parametrize("url, model", [("https://generativelanguage.googleapis.com/v1beta/openai", "x"),
                                        ("http://gpu:8000/v1", "gemini-2.5-pro")])
def test_gemini_is_never_the_teacher(tmp_path, url, model):
    with pytest.raises(TeacherRefused):
        OpenAICompatibleTeacher(url, model, str(tmp_path / "c.jsonl"))


# ---------------------------------------------------------------- export
def _items(tmp_path):
    root = make_dataset(str(tmp_path / "ds"), [
        dataset_row("old_ok_1771696865_trigger", "A man walks past.", alert=False),
        dataset_row("old_empty_1771696866_trigger", "No special activity", alert=False),
        dataset_row("old_alert_1771696867_trigger", "Climbs the fence.", alert=True),
        dataset_row("old_contradicted_1771696868_trigger", "A man walks past.", alert=False),
        dataset_row("old_deleted_1771696869_trigger", "[delete] blurry", alert=None),
        dataset_row("tagged_1771696870_trigger", "old words", alert=False),
        dataset_row("tagged_delete_1771696871_trigger", "x", alert=False),
        dataset_row("tagged_check_1771696872_trigger", "x", alert=False),
        dataset_row("untagged_1771696873_trigger", "", alert=None),
    ])
    items = {i.clip_id: i for i in DatasetSource(root).load()}
    items["old_contradicted_1771696868_trigger"].opinions[TEACHER] = Opinion(TEACHER, label="escalation")
    return items


TAGS = {"ds:tagged_1771696870_trigger": Tag("ds:tagged_1771696870_trigger", {
            "category": "S3", "zone": "fence", "movement": "moving_around", "flags": ["flashlight"],
            "visibility": "clear", "evidence_sec": 4.2, "evidence_frame": 29, "description": "Walks along the fence."},
            "2026-10-06T10:00:00Z", "me"),
        "ds:tagged_delete_1771696871_trigger": Tag("ds:tagged_delete_1771696871_trigger", {"delete": True}, "t", "me"),
        "ds:tagged_check_1771696872_trigger": Tag("ds:tagged_check_1771696872_trigger",
                                                  {"category": "N1", "needs_check": True}, "t", "me")}


def test_training_export_keeps_the_contract_and_adds_the_taxonomy(tmp_path):
    training, evals, counts = ex.build(_items(tmp_path).values(), TAGS)
    by_id = {r["clip_id"]: r for r in training}
    assert set(by_id) == {"old_ok_1771696865_trigger", "old_empty_1771696866_trigger", "old_alert_1771696867_trigger",
                          "tagged_1771696870_trigger"}
    for r in training:
        for key in ("video_s3_path", "vlm_crop_s3_path", "description", "camera_name", "duration_sec", "num_persons",
                    "num_cars", "kind", "date", "clip_id"):
            assert key in r
    t = by_id["tagged_1771696870_trigger"]
    assert (t["category"], t["category_name"], t["raw_label"], t["alert"]) == ("S3", "surveying", "suspicious", True)
    assert t["observation"] == {"zone": "fence", "movement": "moving_around", "flags": ["flashlight"],
                                "visibility": "clear", "evidence_frame": 29, "evidence_sec": 4.2}
    old_alert = by_id["old_alert_1771696867_trigger"]
    # a migrated old tag is a level only: no invented category, and an [alert] never said suspicious or escalation
    assert (old_alert["category"], old_alert["category_name"], old_alert["raw_label"]) == (None, None, None)
    assert (old_alert["tag_source"], old_alert["old_label"], old_alert["alert"]) == ("migrated", "alert", True)
    assert by_id["old_empty_1771696866_trigger"]["category"] is None
    assert by_id["old_empty_1771696866_trigger"]["raw_label"] == "normal"
    assert counts["delete"] == 2 and counts["needs_check"] == 1 and counts["contradicted"] == 1
    assert counts["untagged"] == 1 and counts["training"] == 4


def test_eval_rows_match_eval_prompt(tmp_path):
    _, evals, _ = ex.build(_items(tmp_path).values(), TAGS)
    by_id = {r["clip_id"]: r for r in evals}
    assert {k: r["ours_label"] for k, r in by_id.items()} == {
        "old_ok_1771696865_trigger": "normal", "old_empty_1771696866_trigger": "empty",
        "old_alert_1771696867_trigger": "alert", "tagged_1771696870_trigger": "alert"}
    row = by_id["tagged_1771696870_trigger"]
    assert row["frames"] == [f"frames/tagged_1771696870_trigger_{i}.jpg" for i in range(5)]
    assert row["local_time"] == evalfmt.clip_local_time(row["clip_id"]) and row["category"] == "S3"
    assert row["s3_key"] == "home_guard_dataset/clips/house/tagged_1771696870_trigger.mp4"


def test_write_frames(tmp_path):
    _, evals, _ = ex.build(_items(tmp_path).values(), TAGS)
    frames = [np.full((6, 6, 3), i * 40, dtype=np.uint8) for i in range(5)]
    out = str(tmp_path / "evalset")
    assert ex.write_frames(evals, out, lambda cid: "v.mp4", sample=lambda p: frames)["written"] == 4
    assert os.path.isfile(os.path.join(out, "frames", "tagged_1771696870_trigger_4.jpg"))
    assert ex.write_frames(evals, out, lambda cid: None)["cached"] == 4


# ---------------------------------------------------------------- the studio over files (no database)
def _studio(tmp_path):
    ds = make_dataset(str(tmp_path / "ds"), [dataset_row("front_side_1771696865_trigger", "A man walks past.")])
    put_owner_clip(ds, "production_house2", alert_meta("house2_ch2", STEM, label="suspicious"), STEM, answers=[
        feedback_record(STEM, "house2_ch2", "2026-10-04T05:40:00Z", "false_alarm", "empty")])
    results = tmp_path / "eval" / "eval_set" / "results"
    eval_results(str(results), "p__m", "m", {"front_side_1771696865_trigger": "suspicious"}, 0.9, 0.1)
    return TagStudio(StudioPaths.resolve(env={}, dataset=ds, eval_dir=str(tmp_path / "eval"),
                                         exports=str(tmp_path / "exports")))


def test_studio_without_database(tmp_path):
    studio = _studio(tmp_path)
    items = studio.items(None)
    assert set(items) == {"ds:front_side_1771696865_trigger", f"of:production_house2/{STEM}"}
    assert items["ds:front_side_1771696865_trigger"].opinions[TEACHER].label == "suspicious"
    assert studio.prefill(items["ds:front_side_1771696865_trigger"])["description"] == "A man walks past."


def test_media_grants_are_signed_and_expire(tmp_path):
    studio = _studio(tmp_path)
    path = studio.local_media(studio.items(None)["ds:front_side_1771696865_trigger"], "clip")
    now = datetime.now(timezone.utc).timestamp()
    token = studio.sign("secret-0123456789-0123456789-01234567", path, now)
    assert TagStudio.verify("secret-0123456789-0123456789-01234567", token, now) == path
    assert TagStudio.verify("another-secret-0123456789-0123456789", token, now) is None
    assert TagStudio.verify("secret-0123456789-0123456789-01234567", token, now + 3600) is None
    payload, mac = token.split(".")
    assert TagStudio.verify("secret-0123456789-0123456789-01234567", payload[:-2] + "AA." + mac, now) is None


def test_paths_flag_then_env_then_default(tmp_path):
    env = {"HOMEGUARD_DATASET_DIR": str(tmp_path / "env_ds"), "HOMEGUARD_EVAL_DIR": str(tmp_path / "ev")}
    for folder in ("eval_set/results", "eval_set_v2/results"):
        (tmp_path / "ev" / folder).mkdir(parents=True)
    p = StudioPaths.resolve(env=env)
    assert p.dataset == tmp_path / "env_ds"
    assert p.eval_results == (tmp_path / "ev" / "eval_set" / "results", tmp_path / "ev" / "eval_set_v2" / "results")
    assert p.exports == tmp_path / "studio_exports"
    flagged = StudioPaths.resolve(env=env, dataset=str(tmp_path / "flag"), exports=str(tmp_path / "out"))
    assert flagged.dataset == tmp_path / "flag" and flagged.exports == tmp_path / "out"
    assert StudioPaths.resolve(env={}).dataset.name in ("dataset", "home_guard_dataset")
    assert VIDEO_BYTES


def test_prefill_from_an_old_tag_never_invents_a_category(tmp_path):
    root = make_dataset(str(tmp_path / "ds2"), [
        dataset_row("alert_1772734904_trigger", "Three men cover their faces at the gate.", alert=True),
        dataset_row("normal_1772734905_trigger", "A man walks past.", alert=False),
        dataset_row("empty_1772734906_trigger", "No special activity", alert=False)])
    studio = TagStudio(StudioPaths.resolve(env={}, dataset=root, eval_dir=str(tmp_path / "none"),
                                           exports=str(tmp_path / "out")))
    items = studio.items(None)
    alert = studio.prefill(items["ds:alert_1772734904_trigger"])
    assert (alert["category"], alert["raw_label"]) == ("", "suspicious")
    assert alert["description"] == "Three men cover their faces at the gate."
    normal = studio.prefill(items["ds:normal_1772734905_trigger"])
    assert (normal["category"], normal["raw_label"]) == ("", "normal")
    empty = studio.prefill(items["ds:empty_1772734906_trigger"])
    assert (empty["category"], empty["raw_label"]) == ("N10", "normal")       # the only category an old tag implies
    # done = the old level stands; its category stays empty until someone tags it
    rows = {i.key: a for i, a in wq.build(items.values(), {})}
    assert rows["ds:alert_1772734904_trigger"].reasons == ["migrated old tag (no category yet)"]


def test_a_saved_tag_needs_a_category_unless_deleted():
    from home_guard_project.cloud.tagstudio.service import StudioError, require_category
    with pytest.raises(StudioError, match="Choose a category"):
        require_category(None, {"raw_label": "suspicious", "description": "x"})
    require_category(None, {"delete": True})
    require_category(Tag("k", {"category": "S2"}), {"needs_check": True})        # a later partial save is fine
    # a clip answered with the legacy prompt has no category in its schema: its raw label is what it needs
    legacy = "2026-10-03.tagged-rules-label-animals-why-owner-facts"
    require_category(None, {"raw_label": "normal", "prompt_version": legacy})
    with pytest.raises(StudioError, match="raw label"):
        require_category(None, {"why": "x", "prompt_version": legacy})
