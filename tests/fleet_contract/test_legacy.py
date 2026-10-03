import json, pathlib
from home_guard_project.fleet_contract.legacy import parse_meta, parse_feedback, parse_heartbeat
F = pathlib.Path("tests/fleet_contract/fixtures")
def load(n): return json.loads((F / n).read_text(encoding="utf-8"))
P = "production_test/meta/front_side/2026-10-03/front_side_1791020177_alert.meta.json"
T = "dataset_test/meta/front_side/2026-10-03/front_side_1791020177_alert.meta.json"

def test_production_alert_has_no_ai_but_has_dispatch():
    r = parse_meta(P, load("prod_alert.meta.json"))
    assert r.root == "production" and r.kind == "alert" and r.ai.status == "none"
    assert r.alert_command == "[send_message]" and r.dispatch["channel"] == "telegram"
    assert r.clip_rel == "clips/front_side/2026-10-03/front_side_1791020177_alert.mp4"
    assert r.trigger_ts == 1791020177 and r.frame_size == [704, 576]

def test_training_alert_has_real_teacher():
    r = parse_meta(T, load("train_alert.meta.json"))
    assert r.ai.status == "real" and r.ai.model == "gpt-4o"
    assert r.ai.input_frames_rel[0].startswith("vlm_crops/front_side/")
    assert r.ai.parsed["people"] == 1

def test_false_positive():
    r = parse_meta("dataset_test/meta/back_door/2026-10-03/back_door_1791013299_fp.meta.json", load("train_fp.meta.json"))
    assert r.kind == "false_positive" and r.alert_command == "[none]" and r.ai.status == "real"

def test_fallback_is_not_real():
    r = parse_meta("dataset_test/meta/back_door/2026-10-03/back_door_1791013299_fp.meta.json", load("fallback.meta.json"))
    assert r.ai.status == "fallback"

def test_paused():
    r = parse_meta("dataset_test/meta/left_side_1/2026-10-03/left_side_1_1791014883_paused.meta.json", load("paused.meta.json"))
    assert r.paused and r.muted and r.ai.status == "none" and r.detected == ["car"]

def test_collection_sampled_frames_and_conf():
    r = parse_meta("dataset_ameer_house/meta/back_door/2026-10-02/back_door_1790944263_trigger.meta.json", load("collect.meta.json"))
    assert r.kind == "trigger" and r.sampled_frames[0]["label_path"].startswith("yolo/labels/")
    assert 0.76 < r.class_max_conf["person"] < 0.77

def test_string_model_response():
    r = parse_meta("dataset_ameer_house/meta/back_door/2026-10-02/back_door_1790944263_trigger.meta.json", load("string_response.meta.json"))
    assert r.ai.status == "real" and r.summary == "A person walks."

def test_traversal_recorded_as_problem():
    r = parse_meta(P, load("traversal.meta.json"))
    assert r.clip_rel is None and "clip_path outside site" in r.problems

def test_garbage_never_raises():
    r = parse_meta(P, {"camera_name": 5, "yolo": "x", "alert": [], "model_response": 3})
    assert r.problems

def test_feedback_and_heartbeat():
    f = parse_feedback("production_test/feedback/test_ch6/2026-10-03/x.feedback.json", load("feedback.json"))
    assert f.alert_id == "test_ch6_1790979739_alert" and f.camera == "test_ch6" and f.verdict == "none"
    h = parse_heartbeat(load("heartbeat.json"))
    assert h.site == "test" and h.collector_running and h.cameras["front_side"].year == 2026


import copy
from datetime import datetime, timezone
import pytest


@pytest.mark.parametrize("extra,status", [
    ({}, "none"),
    ({"teacher": None, "model_response": None}, "none"),
    ({"teacher": {}}, "failed"),
    ({"teacher": {"model": "gpt-4o"}, "model_response": None}, "failed"),
    ({"teacher": {"raw_path": None}, "model_response": {"summary": "", "people": 0}}, "fallback"),
    ({"teacher": {"raw_path": "responses/x.txt"}, "model_response": {"summary": ""}}, "real"),
    ({"model_response": {"summary": "", "label": "normal"}}, "real"),
    ({"model_response": ""}, "real"),
])
def test_ai_status_rules(extra, status):
    r = parse_meta(P, load("prod_alert.meta.json") | extra)
    assert r.ai.status == status
    if isinstance(extra.get("model_response"), str):
        assert r.ai.parsed == {"text": extra["model_response"]}


@pytest.mark.parametrize("response", [0, 3, 1.5, False, [], ["text"]])
@pytest.mark.parametrize("teacher", [None, {"model": "gpt-4o"}])
def test_invalid_model_response_is_failed(response, teacher):
    r = parse_meta(P, {"model_response": response, "teacher": teacher})
    assert r.ai.status == "failed"
    assert r.ai.parsed is None
    assert "invalid model_response" in r.problems


@pytest.mark.parametrize("extra,expected,problem", [
    ({}, "responses/parsed.json", None),
    ({"model_raw_text_path": "responses/raw.txt"}, "responses/raw.txt", None),
    ({"teacher": {"raw_path": "responses/teacher.txt"}, "model_raw_text_path": "../unused"},
     "responses/teacher.txt", None),
    ({"teacher": {"raw_path": None}}, None, None),
    ({"teacher": {"raw_path": "../invalid"}}, None, "teacher.raw_path outside site"),
    ({"model_raw_text_path": None}, None, None),
    ({"model_raw_text_path": "../invalid"}, None, "model_raw_text_path outside site"),
])
def test_raw_path_precedence_preserves_explicit_values(extra, expected, problem):
    r = parse_meta(P, load("prod_alert.meta.json") | {"model_response_path": "responses/parsed.json"} | extra)
    assert r.ai.raw_rel == expected
    assert [p for p in r.problems if "outside site" in p] == ([problem] if problem else [])


@pytest.mark.parametrize("alert,response,expected", [
    ({"summary": "owner-facing"}, {"summary": "model"}, "owner-facing"),
    ({"summary": ""}, {"summary": "model"}, ""),
    ({}, {"summary": "model"}, "model"),
    ({}, "model text", "model text"),
    ({}, None, ""),
])
def test_summary_precedence(alert, response, expected):
    assert parse_meta(P, {"alert": alert, "model_response": response}).summary == expected


def test_collection_prompt_and_raw_response_paths():
    body = load("string_response.meta.json") | {
        "prompt_used": "Describe this clip.",
        "model_raw_text_path": "responses\\back_door\\x.model_raw.txt",
    }
    r = parse_meta(T, body)
    assert r.ai.prompt == "Describe this clip."
    assert r.ai.raw_rel == "responses/back_door/x.model_raw.txt"


def test_paths_are_validated_without_mutating_input():
    body = load("train_alert.meta.json")
    body["teacher"]["raw_path"] = "../other/raw.txt"
    body["teacher"]["input_frames"] = ["./vlm_crops\\a.jpg", "C:\\bad.jpg", None]
    body["yolo_export"] = {"exported_frames": [
        {"image_path": "./yolo\\images\\a.jpg", "label_path": "../bad.txt", "frame_index": 2}, 7
    ]}
    original = copy.deepcopy(body)
    r = parse_meta(T, body)
    assert body == original
    assert r.ai.raw_rel is None and r.ai.input_frames_rel == ["vlm_crops/a.jpg"]
    assert r.sampled_frames == [{"image_path": "yolo/images/a.jpg", "label_path": None, "frame_index": 2}]
    assert "teacher.raw_path outside site" in r.problems
    assert "yolo_export.exported_frames[0].label_path outside site" in r.problems


@pytest.mark.parametrize("value", [None, [], "bad", 7, True, {"nested": []}])
def test_odd_shapes_across_meta_fields(value):
    fields = ["camera_name", "kind", "clip_path", "clip_start_ts", "clip_end_ts",
              "duration_sec", "fps_estimated", "codec", "buffer", "yolo", "alert",
              "model_response", "teacher", "yolo_export", "owner_feedback"]
    r = parse_meta(P, dict.fromkeys(fields, value))
    assert r.problems
    assert isinstance(r.detected, list) and isinstance(r.sampled_frames, list)


def test_nested_garbage_and_missing_timestamps():
    r = parse_meta(P, {"buffer": {"store_size": [True, "large"]},
        "yolo": {"class_counts": [], "class_max_conf": {"person": "high", "car": float("inf")}},
        "alert": {"muted": "false", "paused": [], "dispatch": 7},
        "teacher": {"input_frames": "not a list"},
        "yolo_export": {"exported_frames": {}}, "owner_feedback": [3]})
    assert r.frame_size is None and r.class_max_conf == {}
    assert not r.muted and not r.paused and r.dispatch is None
    assert r.owner_feedback == [] and "missing clip_start_ts" in r.problems


def test_invalid_key_and_non_object_body_never_raise():
    for body in [None, [], "bad", 7]:
        assert parse_meta("unrecognised", body).problems


def test_owner_feedback_kind_and_capture_epoch_are_preserved():
    body = load("prod_alert.meta.json") | {"kind": "owner_feedback", "owner_feedback": [{"verdict": "false_alarm"}]}
    r = parse_meta(P, body)
    assert r.kind == "owner_feedback" and r.owner_feedback == body["owner_feedback"]
    assert r.start_ts == body["clip_start_ts"] and r.trigger_ts == 1791020177


def test_feedback_action_scope_is_separate():
    body = load("feedback.json") | {"camera": "back_door"}
    f = parse_feedback("production_test/feedback/_general/2026-10-03/x.feedback.json", body)
    assert f.camera == "test_ch6" and f.scope_camera == "back_door"
    assert f.time_utc == datetime(2026, 10, 2, 22, 23, 57, tzinfo=timezone.utc)
    f = parse_feedback("production_test/feedback/_general/2026-10-03/x.feedback.json", body | {"alert": None})
    assert f.camera is None and f.alert_id is None and f.scope_camera == "back_door"


def test_heartbeat_bad_values_remain_unknown():
    h = parse_heartbeat({"time_utc": "invalid", "collector_running": "false", "stopped": 0,
        "disk_free_gb": float("nan"), "clips_outbox": "many", "cameras": {"a": {}, "b": None}})
    assert h.time_utc is None and h.collector_running is None and h.stopped is None
    assert h.disk_free_gb is None and h.clips_outbox == 0 and h.cameras == {"a": None, "b": None}


def test_heartbeat_utc_normalization():
    h = parse_heartbeat(load("heartbeat.json") | {"time_utc": "2026-10-03T14:15:14+03:00"})
    assert h.time_utc == datetime(2026, 10, 3, 11, 15, 14, tzinfo=timezone.utc)
    assert h.newest_clip_utc.tzinfo == timezone.utc


@pytest.mark.parametrize("value", [
    "2026-10-03T11:15:14", "2026-10-03T11:15:14Z", "2026-10-03T14:15:14+03:00",
    None, "invalid", 42, [], datetime(2026, 10, 3), "0001-01-01T00:00:00+01:00",
])
def test_legacy_time_fields_keep_utc_parsing_semantics(value):
    expected = datetime(2026, 10, 3, 11, 15, 14, tzinfo=timezone.utc) if isinstance(value, str) and value.startswith("2026-") else None
    h = parse_heartbeat({"time_utc": value, "newest_clip_utc": value,
                         "cameras": {"front_side": {"newest_clip_utc": value}}})
    f = parse_feedback("production_test/feedback/_general/2026-10-03/x.feedback.json", {"time_utc": value})
    assert h.time_utc == h.newest_clip_utc == h.cameras["front_side"] == f.time_utc == expected

