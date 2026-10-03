"""Task 8: S3 wrapper + indexer that merges the production and training copies of a clip."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from home_guard_project.cloud import models as m
from . import builders as b


@pytest.fixture()
def s3client(monkeypatch):
    import boto3
    from moto import mock_aws

    for name, value in (("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing"),
                        ("AWS_SESSION_TOKEN", "testing"), ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")


@pytest.fixture()
def s3(s3client):
    from home_guard_project.cloud.s3 import S3

    s3client.create_bucket(Bucket=b.BUCKET)
    return S3(s3client, b.BUCKET)


@pytest.fixture()
def session(db_engine):
    s = sessionmaker(db_engine, expire_on_commit=False)()
    yield s
    s.close()


@pytest.fixture()
def device(session):
    dev = b.enroll(session)
    session.commit()
    return dev


def _index(session, s3, device, **kw):
    from home_guard_project.cloud.indexer import index_device

    return index_device(session, s3, device, **kw)


def _event(session, device, stem=b.STEM):
    return session.scalars(select(m.Event).where(m.Event.device_pk == device.id, m.Event.stem == stem)).one()


def _artifact(session, key):
    return session.scalars(select(m.Artifact).where(m.Artifact.s3_key == key)).one()


def _count(session, model):
    return session.scalar(select(func.count()).select_from(model))


# ---------------------------------------------------------------- S3 wrapper

def test_s3_wrapper_list_get_put_presign_exists(s3client, s3):
    b.put(s3client, "dataset_test/meta/a/2026-10-03/a_1_alert.meta.json", {"x": 1})
    b.put(s3client, "dataset_test/raw.txt", "hello")
    objs = list(s3.list("dataset_test/"))
    assert sorted(o.key for o in objs) == ["dataset_test/meta/a/2026-10-03/a_1_alert.meta.json", "dataset_test/raw.txt"]
    for o in objs:
        assert o.etag and '"' not in o.etag
        assert o.size > 0 and o.last_modified.tzinfo is not None
    assert s3.get_json("dataset_test/meta/a/2026-10-03/a_1_alert.meta.json") == {"x": 1}
    assert s3.get_text("dataset_test/raw.txt") == "hello"
    assert s3.exists("dataset_test/raw.txt") and not s3.exists("dataset_test/nope.txt")
    s3.put_json("fleet/dev/state/desired.json", {"generation": 1})
    assert s3.get_json("fleet/dev/state/desired.json") == {"generation": 1}
    with pytest.raises(ValueError):
        s3.put_json("production_test/meta/x.meta.json", {})  # phase 1 is read-only toward boxes
    url = s3.presign("dataset_test/raw.txt", ttl=120)
    assert "dataset_test/raw.txt" in url and "Expires=" in url


def test_s3_list_paginates(s3client, s3):
    for i in range(1005):
        s3client.put_object(Bucket=b.BUCKET, Key=f"dataset_test/many/{i:05d}.txt", Body=b"x")
    assert len(list(s3.list("dataset_test/many/"))) == 1005


# ---------------------------------------------------------------- merge

def test_production_and_training_copies_merge_into_one_event(session, s3client, s3, device):
    b.seed_bucket(s3client)
    stats = _index(session, s3, device)

    ev = _event(session, device)
    assert session.scalar(select(func.count()).select_from(m.Event).where(
        m.Event.device_pk == device.id, m.Event.stem == b.STEM)) == 1
    assert ev.site == "test" and ev.camera == "front_side" and ev.kind == "alert"
    assert ev.day == "2026-10-03" and ev.trigger_ts == 1791020177
    assert ev.start_ts == pytest.approx(1791020174.1329982) and ev.end_ts == pytest.approx(1791020183.9285913)
    assert ev.alert_command == "[send_message]" and ev.label == "normal"
    assert ev.summary.startswith("A person appears")
    assert ev.dispatch["channel"] == "telegram"
    assert ev.clip_start_local == "2026-10-03 12:36:14"
    assert ev.frame_size == [704, 576] and ev.fps == pytest.approx(4.4)
    assert ev.owner_verdicts == ["true_alert"]
    assert ev.completeness == {"video": True, "boxes": "none", "ai": "real", "owner_feedback": True,
                               "expired": False, "copies": ["production", "training"]}

    # both copies' artifacts are kept and attached
    roles = {a.s3_key: a.role for a in session.scalars(select(m.Artifact).where(m.Artifact.event_id == ev.id))}
    assert roles[b.PROD_CLIP] == roles[b.TRAIN_CLIP] == "original_video"
    assert roles[b.PROD_META] == roles[b.TRAIN_META] == "meta"
    assert roles[b.RAW_ANSWER] == "raw_answer"
    assert all(roles[k] == "teacher_frame" for k in b.CROPS)
    assert roles[b.FEEDBACK] == "feedback"
    assert _artifact(session, b.PROD_CLIP).detail["copy"] == "production"
    assert _artifact(session, b.TRAIN_CLIP).detail["copy"] == "training"
    assert _artifact(session, b.PROD_CLIP).mime == "video/mp4"

    # raw revisions saved for both meta bodies
    assert {r.s3_key for r in session.scalars(select(m.RawRevision))} >= {b.PROD_META, b.TRAIN_META}

    # one guard AI run, from the training copy's teacher, linked to its raw answer + input frames
    runs = session.scalars(select(m.AiRun).where(m.AiRun.event_id == ev.id)).all()
    assert len(runs) == 1
    run = runs[0]
    assert run.purpose == "guard" and run.status == "real" and run.model == "gpt-4o"
    assert run.prompt_version == "2026-10-03.tagged-rules-label" and run.parsed["people"] == 1
    assert run.raw_artifact_id == _artifact(session, b.RAW_ANSWER).id
    assert run.input_artifact_ids == [_artifact(session, k).id for k in b.CROPS]  # f3/f4 absent from S3

    # feedback linked to the event; general and orphan feedback stay unlinked
    fb = session.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK)).one()
    assert fb.event_id == ev.id and fb.alert_stem == b.STEM and fb.verdict == "true_alert"
    assert fb.device_pk == device.id
    assert fb.received_at == datetime(2026, 10, 3, 9, 40, tzinfo=timezone.utc)
    general = session.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.GENERAL_FEEDBACK)).one()
    assert general.event_id is None and general.alert_stem is None
    orphan = session.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.ORPHAN_FEEDBACK)).one()
    assert orphan.event_id is None and orphan.alert_stem == "test_ch6_1790979739_alert"

    events = session.scalar(select(func.count()).select_from(m.Event).where(m.Event.device_pk == device.id))
    assert stats.new_events == events == 6  # front_side alert, traversal, fp, fallback, paused, collect
    assert stats.feedback == 3 and stats.artifacts == len(b.seed_objects())
    assert stats.problems >= 1


def test_reindex_is_idempotent(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    models = (m.Event, m.Artifact, m.RawRevision, m.Feedback, m.AiRun, m.IndexProblem)
    before = {mod: _count(session, mod) for mod in models}
    stats = _index(session, s3, device)
    assert stats.new_events == 0 and stats.updated_events == 0
    assert stats.artifacts == 0 and stats.feedback == 0
    assert {mod: _count(session, mod) for mod in models} == before


def test_changed_training_meta_updates_event_and_adds_revision(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    body = b.fixture_json("train_alert.meta.json")
    body["alert"]["summary"] = "Edited: a person in a red coat walks across the yard."
    body["model_response"]["summary"] = body["alert"]["summary"]
    b.put(s3client, b.TRAIN_META, body)

    stats = _index(session, s3, device)
    assert stats.new_events == 0 and stats.updated_events == 1
    revs = session.scalars(select(m.RawRevision).where(m.RawRevision.s3_key == b.TRAIN_META)).all()
    assert len(revs) == 2 and len({r.etag for r in revs}) == 2
    session.expire_all()
    ev = _event(session, device)
    assert ev.summary.startswith("Edited:")
    assert ev.completeness["copies"] == ["production", "training"]
    assert session.scalar(select(func.count()).select_from(m.AiRun).where(m.AiRun.event_id == ev.id)) == 1


def test_ai_run_is_not_downgraded_by_a_worse_status(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    body = b.fixture_json("train_alert.meta.json")
    body["model_response"] = {"summary": ""}
    body["teacher"]["raw_path"] = None
    b.put(s3client, b.TRAIN_META, body)
    _index(session, s3, device)
    session.expire_all()
    ev = _event(session, device)
    run = session.scalars(select(m.AiRun).where(m.AiRun.event_id == ev.id)).one()
    assert run.status == "real" and ev.completeness["ai"] == "real"


def test_ai_comes_from_best_copy_and_summary_follows_it(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    fallback = _event(session, device, b.FALLBACK_STEM)
    assert fallback.completeness["ai"] == "fallback" and fallback.completeness["copies"] == ["training"]
    fp = _event(session, device, b.FP_STEM)
    assert fp.kind == "false_positive" and fp.completeness["ai"] == "real" and fp.alert_command == "[none]"
    paused = _event(session, device, b.PAUSED_STEM)
    assert paused.completeness["ai"] == "none" and paused.kind == "paused"


def test_collection_clip_with_sampled_yolo_labels(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    ev = _event(session, device, b.COLLECT_STEM)
    assert ev.kind == "trigger" and ev.day == "2026-10-02"
    assert ev.completeness["boxes"] == "sampled" and ev.completeness["copies"] == ["training"]
    assert ev.class_max_conf == {"person": pytest.approx(0.765583872795105)}
    assert ev.detected == ["person"]
    roles = sorted(a.role for a in session.scalars(select(m.Artifact).where(m.Artifact.event_id == ev.id)))
    assert roles == ["meta", "original_video", "yolo_image", "yolo_image", "yolo_label", "yolo_label"]


def test_traversal_fixture_lands_in_index_problems_once(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    _index(session, s3, device, full_scan=True)
    problems = session.scalars(select(m.IndexProblem).where(m.IndexProblem.s3_key == b.TRAVERSAL_META)).all()
    assert len(problems) == 1 and "clip_path outside site" in problems[0].reason
    assert _count(session, m.IndexProblem) == 1


def test_unreadable_meta_is_a_problem_not_a_crash(session, s3client, s3, device):
    b.seed_bucket(s3client)
    bad = "production_test/meta/front_side/2026-10-03/front_side_1791029999_alert.meta.json"
    b.put(s3client, bad, "{not json")
    _index(session, s3, device)
    problem = session.get(m.IndexProblem, bad)
    assert problem is not None and "invalid json" in problem.reason
    assert not session.scalars(select(m.Event).where(m.Event.stem == "front_side_1791029999_alert")).all()
    # an unchanged broken object is not refetched or re-counted
    assert _index(session, s3, device).problems == 0


def test_full_scan_marks_deleted_objects_unavailable(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    s3client.delete_object(Bucket=b.BUCKET, Key=b.PROD_CLIP)

    _index(session, s3, device)  # a full scan just ran; the next one is at most every 30 minutes
    assert _artifact(session, b.PROD_CLIP).available is True

    _index(session, s3, device, full_scan=True)
    session.expire_all()
    assert _artifact(session, b.PROD_CLIP).available is False
    assert _event(session, device).completeness["video"] is True  # training copy still there

    s3client.delete_object(Bucket=b.BUCKET, Key=b.TRAIN_CLIP)
    _index(session, s3, device, full_scan=True)
    session.expire_all()
    assert _event(session, device).completeness["video"] is False

    b.put(s3client, b.TRAIN_CLIP, b.FAKE_MP4)
    _index(session, s3, device)
    session.expire_all()
    assert _artifact(session, b.TRAIN_CLIP).available is True
    assert _event(session, device).completeness["video"] is True


def test_full_scan_runs_when_last_one_is_older_than_30_minutes(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    s3client.delete_object(Bucket=b.BUCKET, Key=b.PROD_CLIP)
    for cur in session.scalars(select(m.S3Cursor)):
        cur.last_full_scan = datetime.now(timezone.utc) - timedelta(minutes=31)
    session.commit()
    _index(session, s3, device)
    session.expire_all()
    assert _artifact(session, b.PROD_CLIP).available is False
    assert {c.prefix for c in session.scalars(select(m.S3Cursor))} == {"production_test/", "dataset_test/"}


def test_expired_when_production_clip_gone_after_14_days(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    s3client.delete_object(Bucket=b.BUCKET, Key=b.PROD_CLIP)
    later = datetime.fromtimestamp(1791020177, timezone.utc) + timedelta(days=15)
    _index(session, s3, device, full_scan=True, now=later)
    session.expire_all()
    ev = _event(session, device)
    assert ev.completeness["expired"] is True and ev.completeness["video"] is True
    # a training-only collection clip never expires on production retention
    assert _event(session, device, b.COLLECT_STEM).completeness["expired"] is False


def test_artifacts_and_feedback_before_meta_attach_when_meta_arrives(session, s3client, s3, device):
    b.seed_bucket(s3client, exclude={b.PROD_META, b.TRAIN_META})
    _index(session, s3, device)
    assert _artifact(session, b.CROPS[0]).event_id is None
    assert session.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK)).one().event_id is None

    b.put(s3client, b.PROD_META, (b.FIXTURES / "prod_alert.meta.json").read_bytes())
    b.put(s3client, b.TRAIN_META, (b.FIXTURES / "train_alert.meta.json").read_bytes())
    stats = _index(session, s3, device)
    assert stats.new_events == 1
    session.expire_all()
    ev = _event(session, device)
    assert _artifact(session, b.CROPS[0]).event_id == ev.id
    assert _artifact(session, b.FEEDBACK).event_id == ev.id
    assert session.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK)).one().event_id == ev.id
    assert ev.owner_verdicts == ["true_alert"] and ev.completeness["owner_feedback"] is True
    run = session.scalars(select(m.AiRun).where(m.AiRun.event_id == ev.id)).one()
    assert run.input_artifact_ids == [_artifact(session, k).id for k in b.CROPS]


def test_late_feedback_attaches_and_recomputes_verdicts(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    late = b.FEEDBACK.replace("1791020300000", "1791020400000")
    b.put(s3client, late, b.feedback_body(verdict="false_alarm", time_utc="2026-10-03T09:45:00Z"))
    dup = b.FEEDBACK.replace("1791020300000", "1791020500000")
    b.put(s3client, dup, b.feedback_body(verdict="none", time_utc="2026-10-03T09:50:00Z"))
    stats = _index(session, s3, device)
    assert stats.feedback == 2 and stats.updated_events == 1
    session.expire_all()
    assert _event(session, device).owner_verdicts == ["true_alert", "false_alarm"]


def test_meta_owner_feedback_union_counts(session, s3client, s3, device):
    b.seed_bucket(s3client, exclude={b.FEEDBACK})
    body = b.fixture_json("train_alert.meta.json")
    entry = {"time_utc": "2026-10-03T09:41:00Z", "verdict": "false_alarm", "note": "", "raw_text": "cat",
             "source": "button", "from": "Owner"}
    body["owner_feedback"] = [entry, dict(entry)]
    b.put(s3client, b.TRAIN_META, body)
    _index(session, s3, device)
    ev = _event(session, device)
    assert ev.completeness["owner_feedback"] is True and ev.owner_verdicts == ["false_alarm"]


def test_heartbeat_stored_on_device_only_when_etag_changes(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    session.expire_all()
    dev = session.get(m.Device, device.id)
    assert dev.last_heartbeat["host"] == "box-test"
    assert dev.last_heartbeat_at == datetime(2026, 10, 3, 11, 15, 14, tzinfo=timezone.utc)
    _index(session, s3, device)
    assert session.scalar(select(func.count()).select_from(m.RawRevision).where(
        m.RawRevision.s3_key == b.HEARTBEAT)) == 1
    hb = b.fixture_json("heartbeat.json")
    hb["time_utc"] = "2026-10-03T11:20:14Z"
    b.put(s3client, b.HEARTBEAT, hb)
    _index(session, s3, device)
    session.expire_all()
    assert session.get(m.Device, device.id).last_heartbeat_at == datetime(2026, 10, 3, 11, 20, 14, tzinfo=timezone.utc)


def test_index_all_covers_every_device(session, s3client, s3, device):
    from home_guard_project.cloud.indexer import IndexStats, index_all

    other = b.enroll(session, site="bian", customer_name="Other")
    session.commit()
    b.seed_bucket(s3client)
    b.put(s3client, "production_bian/clips/bian_ch2/2026-10-02/bian_ch2_1790972298_alert.mp4", b.FAKE_MP4)
    result = index_all(session, s3)
    assert set(result) == {"test", "bian"}
    assert isinstance(result["bian"], IndexStats) and result["bian"].artifacts == 1
    art = _artifact(session, "production_bian/clips/bian_ch2/2026-10-02/bian_ch2_1790972298_alert.mp4")
    assert art.event_id is None and other.id


# ================================================================ fix round (Codex review of Task 8)

BASE = datetime.fromtimestamp(1791020177, timezone.utc)


def _put_etag(s3client, key, body) -> str:
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode("utf-8")
    return s3client.put_object(Bucket=b.BUCKET, Key=key, Body=body)["ETag"].strip('"')


def _run(session, device, stem=b.STEM):
    return session.scalars(select(m.AiRun).where(m.AiRun.event_id == _event(session, device, stem).id)).one()


def _teacher_meta(summary="Teacher: a man in a red coat tries the side gate.", label="suspicious", feedback=()):
    body = b.fixture_json("train_alert.meta.json")
    body["alert"]["summary"] = summary
    body["alert"]["label"] = label
    body["alert"]["alert_command"] = "[call_owner]"
    body["model_response"]["summary"] = summary
    body["owner_feedback"] = list(feedback)
    return body


def _recreated_meta(feedback):
    """What box/feedback.py writes after the training copy was uploaded: production meta + owner_feedback."""
    body = b.fixture_json("prod_alert.meta.json")
    body["kind"] = "owner_feedback"
    body["owner_feedback"] = list(feedback)
    return body


def _verdict(verdict, minute):
    return {"time_utc": f"2026-10-03T09:{minute:02d}:00Z", "verdict": verdict, "note": "", "raw_text": verdict,
            "source": "button", "from": "Owner"}


def _get_counter(monkeypatch, s3client):
    calls = []
    real = s3client.get_object

    def spy(**kw):
        calls.append(kw["Key"])
        return real(**kw)

    monkeypatch.setattr(s3client, "get_object", spy)
    return calls


# ---- 1: historical merge keeps the winning AI revision's fields and every verdict ever seen

def test_recreated_training_meta_keeps_teacher_fields_and_all_verdicts(session, s3client, s3, device):
    b.seed_bucket(s3client, exclude={b.FEEDBACK, b.TRAIN_META})
    teacher_etag = _put_etag(s3client, b.TRAIN_META, _teacher_meta(feedback=[_verdict("true_alert", 40)]))
    _index(session, s3, device)
    ev = _event(session, device)
    assert ev.label == "suspicious" and ev.alert_command == "[call_owner]"

    _put_etag(s3client, b.TRAIN_META, _recreated_meta([_verdict("false_alarm", 45)]))  # teacher-free
    _index(session, s3, device)
    _put_etag(s3client, b.TRAIN_META, _recreated_meta([_verdict("real_but_wrong", 50)]))  # recreated again
    _index(session, s3, device)
    session.expire_all()
    ev = _event(session, device)
    assert ev.summary.startswith("Teacher:") and ev.label == "suspicious" and ev.alert_command == "[call_owner]"
    assert ev.owner_verdicts == ["true_alert", "false_alarm", "real_but_wrong"]
    run = _run(session, device)
    assert run.status == "real" and run.ai_source_key == b.TRAIN_META and run.ai_source_etag == teacher_etag


# ---- 2: the retained AI picks up sidecars that arrive late

def test_retained_ai_links_sidecars_that_arrive_after_a_poorer_copy(session, s3client, s3, device):
    sidecars = {b.RAW_ANSWER, *b.CROPS}
    b.seed_bucket(s3client, exclude=sidecars)
    _index(session, s3, device)
    assert _run(session, device).raw_artifact_id is None
    _put_etag(s3client, b.TRAIN_META, _recreated_meta([_verdict("false_alarm", 45)]))
    _index(session, s3, device)
    for key in sidecars:
        b.put(s3client, key, b.RAW_TEXT if key == b.RAW_ANSWER else b.FAKE_JPG)
    _index(session, s3, device)
    session.expire_all()
    run = _run(session, device)
    assert run.status == "real"
    assert run.raw_artifact_id == _artifact(session, b.RAW_ANSWER).id
    assert run.input_artifact_ids == [_artifact(session, k).id for k in b.CROPS]


# ---- 3: dispatch and muted come only from the production copy

def _prod_muted_no_dispatch():
    body = b.fixture_json("prod_alert.meta.json")
    body["alert"].pop("dispatch")
    body["alert"]["muted"] = True
    return body


@pytest.mark.parametrize("production_first", [True, False])
def test_dispatch_and_muted_come_only_from_production(session, s3client, s3, device, production_first):
    b.seed_bucket(s3client, exclude={b.PROD_META, b.TRAIN_META})
    first, second = (b.PROD_META, _prod_muted_no_dispatch()), (b.TRAIN_META, b.fixture_json("train_alert.meta.json"))
    if not production_first:
        first, second = second, first
    b.put(s3client, *first)
    _index(session, s3, device)
    session.expire_all()
    ev = _event(session, device)
    if production_first:
        assert ev.dispatch is None and ev.muted is True
    else:  # no production copy yet: delivery is unknown, not the training copy's claim
        assert ev.dispatch is None and ev.muted is None
    b.put(s3client, *second)
    _index(session, s3, device)
    session.expire_all()
    ev = _event(session, device)
    assert ev.dispatch is None and ev.muted is True
    assert ev.completeness["copies"] == ["production", "training"]


# ---- 4: returning to a cached revision is re-applied without another GET

def test_heartbeat_returning_to_a_cached_revision_is_reapplied(session, s3client, s3, device, monkeypatch):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    hb = b.fixture_json("heartbeat.json")
    hb["time_utc"] = "2026-10-03T11:20:14Z"
    b.put(s3client, b.HEARTBEAT, hb)
    _index(session, s3, device)
    b.put(s3client, b.HEARTBEAT, (b.FIXTURES / "heartbeat.json").read_bytes())  # back to A
    gets = _get_counter(monkeypatch, s3client)
    _index(session, s3, device)
    session.expire_all()
    assert session.get(m.Device, device.id).last_heartbeat_at == datetime(2026, 10, 3, 11, 15, 14, tzinfo=timezone.utc)
    assert gets == []


def test_feedback_returning_to_a_cached_revision_is_reapplied(session, s3client, s3, device, monkeypatch):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    b.put(s3client, b.FEEDBACK, b.feedback_body(verdict="false_alarm"))
    _index(session, s3, device)
    b.put(s3client, b.FEEDBACK, b.feedback_body())  # back to A
    gets = _get_counter(monkeypatch, s3client)
    _index(session, s3, device)
    session.expire_all()
    assert session.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK)).one().verdict == "true_alert"
    assert _event(session, device).owner_verdicts == ["true_alert"]
    assert gets == []


# ---- 5: an object replaced between LIST and GET is never recorded under the listed ETag

class _RacingS3:
    """Lists the bucket, then replaces `key` before the indexer's GET (a box re-uploading mid-pass)."""

    def __init__(self, s3, s3client, key):
        self.s3, self.client, self.key, self.uploads = s3, s3client, key, {}

    def list(self, prefix):
        objs = list(self.s3.list(prefix))
        if any(o.key == self.key for o in objs):
            body = _teacher_meta(summary=f"Race {len(self.uploads)}")
            self.uploads[_put_etag(self.client, self.key, body)] = body["alert"]["summary"]
        yield from objs

    def __getattr__(self, name):
        return getattr(self.s3, name)


def test_object_replaced_between_list_and_get_is_not_recorded_under_the_listed_etag(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    listed = _put_etag(s3client, b.TRAIN_META, _teacher_meta(summary="Listed body"))
    racer = _RacingS3(s3, s3client, b.TRAIN_META)
    _index(session, racer, device)
    session.expire_all()
    for rev in session.scalars(select(m.RawRevision).where(m.RawRevision.s3_key == b.TRAIN_META)):
        assert rev.etag != listed, "the GET returned a newer body; it must not be saved under the listed ETag"
    assert session.get(m.IndexProblem, b.TRAIN_META) is None  # one race is not a problem
    assert _event(session, device).summary.startswith("A person appears")  # previous state kept
    _index(session, racer, device)
    _index(session, racer, device)
    session.expire_all()
    assert "changed during fetch" in session.get(m.IndexProblem, b.TRAIN_META).reason  # third race in a row
    _index(session, s3, device)  # the uploader settled
    session.expire_all()
    assert session.get(m.IndexProblem, b.TRAIN_META) is None
    last_etag, last_summary = list(racer.uploads.items())[-1]
    rev = session.scalars(select(m.RawRevision).where(m.RawRevision.s3_key == b.TRAIN_META,
                                                      m.RawRevision.etag == last_etag)).one()
    assert rev.body["alert"]["summary"] == last_summary == _event(session, device).summary


# ---- 6: expiry advances with time even when nothing in S3 changes

def test_expiry_advances_with_time_without_listing_changes(session, s3client, s3, device):
    from home_guard_project.cloud.indexer import refresh_expiry

    b.seed_bucket(s3client)
    _index(session, s3, device, now=BASE + timedelta(days=1))
    s3client.delete_object(Bucket=b.BUCKET, Key=b.PROD_CLIP)
    _index(session, s3, device, full_scan=True, now=BASE + timedelta(days=13))
    session.expire_all()
    assert _event(session, device).completeness["expired"] is False
    _index(session, s3, device, now=BASE + timedelta(days=15))  # listing unchanged
    session.expire_all()
    assert _event(session, device).completeness["expired"] is True
    assert _event(session, device, b.COLLECT_STEM).completeness["expired"] is False
    b.put(s3client, b.PROD_CLIP, b.FAKE_MP4)  # the production video reappears
    _index(session, s3, device, now=BASE + timedelta(days=16))
    session.expire_all()
    assert _event(session, device).completeness["expired"] is False
    # refresh_expiry alone also reverses a stale flag
    ev = _event(session, device)
    ev.completeness = {**ev.completeness, "expired": True}
    session.commit()
    refresh_expiry(session, device, BASE + timedelta(days=16))
    session.commit()
    session.expire_all()
    assert _event(session, device).completeness["expired"] is False


# ---- 7: one malformed timestamp is one problem, not a blocked site

@pytest.mark.parametrize("stem,start", [("front_side_1791030000_alert", 1e20),
                                        ("front_side_1791030001_alert", 1e9),  # 2001: before the system existed
                                        ("front_side_99999999999999_alert", None)])
def test_out_of_range_timestamp_is_one_problem_and_the_site_continues(session, s3client, s3, device, stem, start):
    b.seed_bucket(s3client)
    body = b.fixture_json("prod_alert.meta.json")
    if start is None:
        body.pop("clip_start_ts")
    else:
        body["clip_start_ts"] = start
    bad = f"production_test/meta/front_side/2026-10-03/{stem}.meta.json"
    b.put(s3client, bad, body)
    _index(session, s3, device)
    session.expire_all()
    assert "timestamp" in session.get(m.IndexProblem, bad).reason
    assert not session.scalars(select(m.Event).where(m.Event.stem == stem)).all()
    assert _event(session, device).completeness["ai"] == "real"
    assert session.get(m.Device, device.id).last_heartbeat is not None


def test_database_failure_still_rolls_back_only_that_device(session, s3client, s3, device, db_engine):
    from sqlalchemy import event as sa_event

    from home_guard_project.cloud.indexer import index_all

    b.enroll(session, site="bian", customer_name="Other")
    session.commit()
    b.seed_bucket(s3client)
    bian_clip = "production_bian/clips/bian_ch2/2026-10-02/bian_ch2_1790972298_alert.mp4"
    b.put(s3client, bian_clip, b.FAKE_MP4)

    def boom(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("INSERT INTO EVENTS"):
            raise RuntimeError("database went away")

    sa_event.listen(db_engine, "before_cursor_execute", boom)
    try:
        result = index_all(session, s3)
    finally:
        sa_event.remove(db_engine, "before_cursor_execute", boom)
    assert set(result) == {"bian"}
    session.expire_all()
    assert _count(session, m.Event) == 0
    assert session.scalars(select(m.Artifact).where(m.Artifact.s3_key.startswith("production_test/"))).all() == []
    assert _artifact(session, bian_clip).available is True


# ---- 8: listed keys with traversal or a bad layout never become artifacts

@pytest.mark.parametrize("key", [
    "production_test/meta/front_side/2026-10-03/../front_side_1791020177_alert.meta.json",
    "production_test/clips/front_side/2026-10-03" + "9" * 300 + "/front_side_1791020177_alert.mp4",
    "production_test/meta/front_side/2026-10-03/extra/front_side_1791020177_alert.meta.json",
    "production_test/meta/front side!/2026-10-03/front side!_1791020177_alert.meta.json",
])
def test_invalid_key_layout_is_a_problem_not_an_artifact(session, s3client, s3, device, key):
    b.seed_bucket(s3client)
    evil = b.fixture_json("prod_alert.meta.json")
    evil["alert"]["summary"] = "Injected summary"
    b.put(s3client, key, evil)
    _index(session, s3, device)
    session.expire_all()
    assert session.get(m.IndexProblem, key).reason == "invalid key layout"
    assert session.scalars(select(m.Artifact).where(m.Artifact.s3_key == key)).all() == []
    assert _event(session, device).summary.startswith("A person appears")


# ---- 9: problems describe the current revision and clear only when it applies

def test_fetch_failure_keeps_previous_state_and_the_problem(session, s3client, s3, device, monkeypatch):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    _put_etag(s3client, b.TRAIN_META, _teacher_meta(summary="Teacher: new revision"))
    real = s3client.get_object

    def failing(**kw):
        if kw["Key"] == b.TRAIN_META:
            raise RuntimeError("throttled")
        return real(**kw)

    monkeypatch.setattr(s3client, "get_object", failing)
    _index(session, s3, device)
    session.expire_all()
    problem = session.get(m.IndexProblem, b.TRAIN_META)
    assert problem is not None and "fetch failed" in problem.reason
    assert _event(session, device).summary.startswith("A person appears")
    monkeypatch.setattr(s3client, "get_object", real)
    _index(session, s3, device)
    session.expire_all()
    assert session.get(m.IndexProblem, b.TRAIN_META) is None
    assert _event(session, device).summary == "Teacher: new revision"


@pytest.mark.parametrize("key,good", [(b.FEEDBACK, lambda: b.feedback_body()),
                                      (b.HEARTBEAT, lambda: b.fixture_json("heartbeat.json"))])
def test_fixed_feedback_and_heartbeat_clear_their_problem(session, s3client, s3, device, key, good):
    b.seed_bucket(s3client)
    b.put(s3client, key, [1, 2, 3])  # valid JSON, wrong shape
    _index(session, s3, device)
    assert session.get(m.IndexProblem, key) is not None
    b.put(s3client, key, good())
    _index(session, s3, device)
    session.expire_all()
    assert session.get(m.IndexProblem, key) is None


# ---- 10: bulk-loaded passes: bounded statements, measured on a 5,000-key site

class _FakeS3:
    """In-memory listing/GET with the S3 semantics the indexer relies on (ETag = md5 of the body)."""

    def __init__(self, objects: dict):
        import hashlib

        self.objects = {k: (v, hashlib.md5(v).hexdigest()) for k, v in objects.items()}
        self.gets = 0

    def list(self, prefix):
        from home_guard_project.cloud.s3 import ObjInfo

        for key in sorted(self.objects):
            if key.startswith(prefix):
                body, etag = self.objects[key]
                yield ObjInfo(key, etag, len(body), BASE)

    def get_text(self, key, if_match=None):
        self.gets += 1
        return self.objects[key][0].decode("utf-8")


def _synthetic_site(n_events: int) -> dict:
    """n events x 5 keys: production + training meta, production + training clip, one teacher crop."""
    prod = (b.FIXTURES / "prod_alert.meta.json").read_text(encoding="utf-8")
    train = (b.FIXTURES / "train_alert.meta.json").read_text(encoding="utf-8")
    objects = {}
    for i in range(n_events):
        camera, ts = f"cam{i % 10}", 1791020177 + i * 60
        stem, day = f"{camera}_{ts}_alert", "2026-10-03"
        for root, text in (("production", prod), ("dataset", train)):
            body = text.replace("front_side_1791020177_alert", stem).replace("front_side", camera)
            body = body.replace("1791020174.1329982", f"{ts - 3}.5").replace("1791020183.9285913", f"{ts + 6}.5")
            objects[f"{root}_test/meta/{camera}/{day}/{stem}.meta.json"] = body.encode("utf-8")
            objects[f"{root}_test/clips/{camera}/{day}/{stem}.mp4"] = b.FAKE_MP4
        objects[f"dataset_test/vlm_crops/{camera}/{day}/{stem}_f0.jpg"] = b.FAKE_JPG
    return objects


def test_index_pass_statement_count_is_bounded(session, device, db_engine):
    import time

    from sqlalchemy import event as sa_event

    n = 1000
    fake = _FakeS3(_synthetic_site(n))
    statements = []

    def count(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    sa_event.listen(db_engine, "before_cursor_execute", count)
    try:
        t0 = time.perf_counter()
        stats = _index(session, fake, device)
        first_secs = time.perf_counter() - t0
        first = len(statements)
        statements.clear()
        t0 = time.perf_counter()
        again = _index(session, fake, device)
        second_secs = time.perf_counter() - t0
        second = len(statements)
    finally:
        sa_event.remove(db_engine, "before_cursor_execute", count)
    print(f"\n5,000-key site: first pass {first} statements in {first_secs:.2f}s; "
          f"unchanged re-pass {second} statements in {second_secs:.2f}s")
    assert stats.new_events == n and again.new_events == again.updated_events == 0
    assert first <= 60 * n // 100, first
    assert second <= 20, second
    assert fake.gets == 2 * n  # each meta fetched exactly once
    assert session.scalar(select(func.count()).select_from(m.AiRun)) == n


# ---- 11: a size-only change is still a change

class _ResizingS3:
    def __init__(self, s3, key, size):
        self.s3, self.key, self.size = s3, key, size

    def list(self, prefix):
        from dataclasses import replace

        for obj in self.s3.list(prefix):
            yield replace(obj, size=self.size) if obj.key == self.key else obj

    def __getattr__(self, name):
        return getattr(self.s3, name)


def test_size_only_change_updates_the_artifact(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    stats = _index(session, _ResizingS3(s3, b.PROD_CLIP, 200), device)
    session.expire_all()
    assert _artifact(session, b.PROD_CLIP).bytes == 200 and stats.artifacts == 1


# ---- 12: a feedback artifact follows its feedback row's event, including to none

def test_feedback_artifact_follows_feedback_event(session, s3client, s3, device):
    b.seed_bucket(s3client)
    _index(session, s3, device)
    b.put(s3client, b.FEEDBACK, b.feedback_body(alert_id=b.FP_STEM, verdict="false_alarm"))  # reassigned
    _index(session, s3, device)
    session.expire_all()
    fp = _event(session, device, b.FP_STEM)
    assert _artifact(session, b.FEEDBACK).event_id == fp.id
    assert fp.owner_verdicts == ["false_alarm"]
    assert _event(session, device).owner_verdicts == []  # the old event no longer claims it
    general = b.feedback_body()
    general["alert"] = None
    b.put(s3client, b.FEEDBACK, general)  # now general feedback
    _index(session, s3, device)
    session.expire_all()
    fb = session.scalars(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK)).one()
    assert fb.event_id is None and _artifact(session, b.FEEDBACK).event_id is None
