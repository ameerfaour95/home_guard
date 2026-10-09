"""The Inbox: owner answers from Telegram (the box feedback file's owner_label / owner_text / transcript, stored by the
indexer, exposed on FeedbackOut), listed for admins with filters, and the admin's audited decisions; migration 0017
backfills answers indexed before it."""
import json

from sqlalchemy import select, text

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import _event_id, s3client  # noqa: F401
from .test_migration_0007 import _engine, fresh_url  # noqa: F401 (fixture)
from .test_studio import _seed

TAG_OLD = b.FEEDBACK.replace("_1791020300000.", "_1791020400000.")
TAG_MID = b.FEEDBACK.replace("_1791020300000.", "_1791020450000.")
TAG_NEW = b.FEEDBACK.replace("_1791020300000.", "_1791020470000.")
MUTE_FEEDBACK = b.FEEDBACK.replace("_1791020300000.", "_1791020500000.")
QUESTION_FEEDBACK = b.FEEDBACK.replace("_1791020300000.", "_1791020600000.")


def _answers():
    """Three tags of one clip (the first superseded by the box when the owner retagged, the second older than the
    newest without the mark: an older box), an action record (a pause), plain chat and a verdict button."""
    old = b.feedback_body(verdict="real_but_wrong", time_utc="2026-10-03T09:50:00Z")
    old.update(owner_label="other", owner_text="זה הגנן שלנו", transcript="זה הגנן שלנו", tagged_by="Dana",
               raw_text="", superseded_by=TAG_NEW.rsplit("/", 1)[1], superseded_utc="2026-10-03T09:57:00Z")
    mid = b.feedback_body(verdict="true_alert", time_utc="2026-10-03T09:52:00Z")
    mid.update(owner_label="suspicious", tagged_by="Dana", raw_text="")
    new = b.feedback_body(verdict="true_alert", time_utc="2026-10-03T09:57:00Z")
    new.update(owner_label="escalation", tagged_by="Dana", raw_text="")
    mute = b.feedback_body(verdict="none", time_utc="2026-10-03T09:58:00Z")
    mute.update(action="mute", raw_text="")
    question = b.feedback_body(verdict="none", time_utc="2026-10-03T09:59:00Z")
    question.update(raw_text="למה המצלמה שולחת לי כל כך הרבה התראות?")
    return {k: json.dumps(v).encode() for k, v in ((TAG_OLD, old), (TAG_MID, mid), (TAG_NEW, new),
                                                    (MUTE_FEEDBACK, mute), (QUESTION_FEEDBACK, question))}


def _seeded(client, s3client, consent=True):
    _seed(client, s3client, consent_training=consent, overrides=_answers())


def test_indexer_stores_the_owner_answer_and_feedback_out_shows_it(client, staff_factory, s3client):
    _seeded(client, s3client)
    _, _, _, admin = staff_factory("admin")
    _, _, _, labeler = staff_factory("labeler")
    with session_scope(client.app.state.engine) as s:
        fb = s.scalar(select(m.Feedback).where(m.Feedback.s3_key == TAG_OLD))
        assert (fb.owner_label, fb.owner_text, fb.transcript, fb.tagged_by) == (
            "other", "זה הגנן שלנו", "זה הגנן שלנו", "Dana")
        assert fb.superseded_by == TAG_NEW.rsplit("/", 1)[1] and fb.superseded_at is not None
        plain = s.scalar(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK))
        assert plain.owner_label == "" and plain.tagged_by == "Owner" and plain.superseded_by == ""
    eid = _event_id(client, b.STEM)
    detail = client.get(f"/v1/events/{eid}", headers=admin).json()
    tagged = [f for f in detail["feedback"] if f["owner_label"] == "other"]
    assert [(f["owner_text"], f["transcript"]) for f in tagged] == [("זה הגנן שלנו", "זה הגנן שלנו")]
    # a labeler of a consenting customer reads the tag word only, never the owner's words
    shown = client.get(f"/v1/events/{eid}", headers=labeler).json()["feedback"]
    assert {(f["owner_label"], f["owner_text"], f["transcript"]) for f in shown} >= {("other", "", "")}


def test_the_inbox_lists_only_the_clips_current_tag_with_the_rest_as_history(client, staff_factory, s3client):
    _seeded(client, s3client)
    _, _, _, h = staff_factory("admin")
    items = client.get("/v1/inbox", headers=h, params={"handled": "all"}).json()
    # one clip, one current tag: the newest; the pause, the chat, the verdict button and _general are never tags
    assert [(i["owner_label"], i["probably_not_label"]) for i in items] == [("escalation", False)]
    tag = items[0]
    assert [(e["owner_label"], e["superseded_by"]) for e in tag["history"]] == [
        ("other", TAG_NEW.rsplit("/", 1)[1]), ("suspicious", "a later tag")]
    assert tag["clip_key"] == f"ev:{tag['event_id']}" and tag["customer"] == "Acme" and tag["camera"] == "front_side"
    assert tag["tagged_by"] == "Dana" and tag["decision"] is None and tag["prompt_version"]
    assert tag["model_summary"] and tag["consent_training"] is True
    assert client.get("/v1/inbox", headers=h, params={"owner_label": "other"}).json() == []   # history, not waiting
    assert client.get("/v1/inbox", headers=h, params={"camera": "nope"}).json() == []
    assert client.get("/v1/inbox", headers=h, params={"customer_id": tag["customer_id"] + 99}).json() == []
    assert len(client.get("/v1/inbox", headers=h, params={"from_utc": "2026-10-03T09:55:00Z"}).json()) == 1
    assert client.get("/v1/inbox", headers=h, params={"to_utc": "2026-10-03T09:55:00Z"}).json() == []
    # the replaced tags cannot be decided on: they are history
    old_id = tag["history"][0]["feedback_id"]
    assert client.post(f"/v1/inbox/{old_id}/decision", headers=h, json={"decision": "fixed"}).status_code == 404


def test_one_current_tag_rule_and_the_safety_net():
    from home_guard_project.cloud import inbox
    flagged = m.Feedback(owner_label="other", owner_text="למה זה שלח לי התראה?", transcript="", raw_text="")
    plain = m.Feedback(owner_label="other", owner_text="it was the gardener", transcript="", raw_text="")
    button = m.Feedback(owner_label="normal", owner_text="", transcript="", raw_text="why?")
    assert inbox.probably_not_label(flagged) and not inbox.probably_not_label(plain)
    assert not inbox.probably_not_label(button)          # a tag button is a judgement, whatever was typed around it


def test_the_current_tag_feeds_tag_ai_and_the_export_snapshot(client, staff_factory, s3client, tmp_path):
    from home_guard_project.cloud import inbox
    from home_guard_project.cloud.tagstudio.config import StudioPaths
    from home_guard_project.cloud.tagstudio.service import TagStudio
    from .tagstudio_fixtures import make_dataset
    _seeded(client, s3client)
    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    client.app.state.tagstudio = TagStudio(StudioPaths.resolve(
        env={}, dataset=make_dataset(str(tmp_path / "ds"), []), eval_dir=str(tmp_path / "eval"),
        exports=str(tmp_path / "exports")))
    clip = client.get("/v1/tagging/clip", headers=h, params={"key": f"ev:{eid}"}).json()
    assert clip["opinions"]["owner"]["label"] == "escalation"          # Tag · AI's "Customer's answer"
    with session_scope(client.app.state.engine) as s:
        current = s.scalars(select(m.Feedback.s3_key).where(m.Feedback.event_id == eid, ~inbox.superseded())).all()
    assert TAG_NEW in current and TAG_OLD not in current and TAG_MID not in current
    assert b.FEEDBACK in current                                       # a verdict button is never superseded


def test_decisions_are_audited_and_move_answers_out_of_the_waiting_list(client, staff_factory, s3client):
    _seeded(client, s3client)
    _, _, _, h = staff_factory("admin")
    (tag,) = client.get("/v1/inbox", headers=h).json()
    r = client.post(f"/v1/inbox/{tag['feedback_id']}/decision", headers=h, json={"decision": "accepted"})
    assert r.status_code == 200, r.text
    assert r.json()["decision"] == "accepted" and r.json()["decided_by"] == "Admin 1" and r.json()["history"]
    assert client.get("/v1/inbox", headers=h).json() == []
    assert len(client.get("/v1/inbox", headers=h, params={"handled": "handled"}).json()) == 1
    r = client.delete(f"/v1/inbox/{tag['feedback_id']}/decision", headers=h)
    assert r.json()["decision"] is None
    assert [i["feedback_id"] for i in client.get("/v1/inbox", headers=h).json()] == [tag["feedback_id"]]
    r = client.post(f"/v1/inbox/{tag['feedback_id']}/decision", headers=h,
                    json={"decision": "not_label", "note": "a question about the bill"})
    assert r.json()["decision"] == "not_label" and r.json()["decision_note"] == "a question about the bill"
    with session_scope(client.app.state.engine) as s:
        rows = s.execute(select(m.AuditLog.action, m.AuditLog.target, m.AuditLog.detail)
                         .where(m.AuditLog.action.like("inbox_%")).order_by(m.AuditLog.id)).all()
    actions = [a for a, _, _ in rows]
    assert actions.count("inbox_decision") == 2 and actions.count("inbox_reopen") == 1 and "inbox_view" in actions
    decision = next(d for a, t, d in rows if a == "inbox_decision")
    assert decision == {"decision": "accepted", "event_id": tag["event_id"], "owner_label": "escalation",
                        "prompt_version": tag["prompt_version"], "probably_not_label": False}
    with session_scope(client.app.state.engine) as s:
        assert s.get(m.InboxDecision, tag["feedback_id"]).prompt_version == tag["prompt_version"]
    assert client.post("/v1/inbox/999999/decision", headers=h, json={"decision": "fixed"}).status_code == 404
    assert client.post(f"/v1/inbox/{tag['feedback_id']}/decision", headers=h,
                       json={"decision": "maybe"}).status_code == 422


def test_no_training_label_without_consent_and_admins_only(client, staff_factory, s3client):
    _seeded(client, s3client, consent=False)
    _, _, _, h = staff_factory("admin")
    _, _, _, labeler = staff_factory("labeler")
    _, _, _, support = staff_factory("support")
    (tag,) = client.get("/v1/inbox", headers=h).json()
    r = client.post(f"/v1/inbox/{tag['feedback_id']}/decision", headers=h, json={"decision": "accepted"})
    assert r.status_code == 409 and "consent" in r.json()["detail"]
    assert client.post(f"/v1/inbox/{tag['feedback_id']}/decision", headers=h,
                       json={"decision": "not_label"}).status_code == 200
    for other in (labeler, support):
        assert client.get("/v1/inbox", headers=other).status_code == 403


def test_0017_backfills_answers_indexed_before_it(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0016")
    engine = _engine(fresh_url)
    key = "production_x/feedback/cam/2026-10-03/cam_1791020177_alert_1.feedback.json"
    body = {"owner_label": "suspicious", "owner_text": "who is that", "transcript": "", "from": {"name": "Eli"}}
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO customers (id, name) VALUES (1, 'Eli')"))
            conn.execute(text("INSERT INTO devices (id, device_id, site, customer_id) VALUES (1, 'd1', 'x', 1)"))
            conn.execute(text("INSERT INTO feedback (id, device_pk, s3_key) VALUES (1, 1, :k)"), {"k": key})
            conn.execute(text("INSERT INTO raw_revisions (s3_key, etag, fetched_at, body) "
                              "VALUES (:k, 'e1', now(), CAST(:b AS jsonb))"), {"k": key, "b": json.dumps(body)})
        command.upgrade(cfg, "0017")
        with engine.begin() as conn:
            row = conn.execute(text("SELECT owner_label, owner_text, transcript, tagged_by FROM feedback")).one()
            assert tuple(row) == ("suspicious", "who is that", "", "Eli")
        command.downgrade(cfg, "0016")
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
    finally:
        engine.dispose()


def test_owner_card_skips_a_superseded_tag():
    from home_guard_project.cloud.tagstudio.sources import owner_opinion
    old = {"time_utc": "2026-10-03T09:50:00Z", "verdict": "true_alert", "owner_label": "suspicious",
           "superseded_by": "x_2.feedback.json"}
    newer_but_marked = {"time_utc": "2026-10-03T09:59:00Z", "verdict": "true_alert", "owner_label": "escalation",
                        "superseded_by": "x_3.feedback.json"}
    current = {"time_utc": "2026-10-03T09:55:00Z", "verdict": "false_alarm", "owner_label": "normal"}
    assert owner_opinion([old, newer_but_marked, current]).label == "normal"
    assert owner_opinion([old]) is None


def test_0020_records_superseded_tags_indexed_before_it(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0019")
    engine = _engine(fresh_url)
    key = "production_x/feedback/cam/2026-10-03/cam_1791020177_alert_1.feedback.json"
    body = {"owner_label": "suspicious", "superseded_by": "cam_1791020177_alert_2.feedback.json"}
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO customers (id, name) VALUES (1, 'Eli')"))
            conn.execute(text("INSERT INTO devices (id, device_id, site, customer_id) VALUES (1, 'd1', 'x', 1)"))
            conn.execute(text("INSERT INTO feedback (id, device_pk, s3_key, owner_label) VALUES (1, 1, :k, 'suspicious')"),
                         {"k": key})
            conn.execute(text("INSERT INTO raw_revisions (s3_key, etag, fetched_at, body) "
                              "VALUES (:k, 'e1', now(), CAST(:b AS jsonb))"), {"k": key, "b": json.dumps(body)})
        command.upgrade(cfg, "0020")
        with engine.begin() as conn:
            assert conn.execute(text("SELECT superseded_by FROM feedback")).scalar() == body["superseded_by"]
        command.downgrade(cfg, "0019")
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
    finally:
        engine.dispose()


def test_0020_repairs_a_database_stamped_past_0017_without_its_schema(fresh_url):
    """A DB that went 0016 -> 0018 -> 0019 before 0017 existed: stamped 0019, no 0017 columns or table."""
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0019")
    engine = _engine(fresh_url)
    key = "production_x/feedback/cam/2026-10-03/cam_1791020177_alert_1.feedback.json"
    body = {"owner_label": "escalation", "owner_text": "someone at the gate", "from": {"name": "Eli"}}
    try:
        with engine.begin() as conn:     # what that database looks like: 0017 never ran
            conn.execute(text("DROP TABLE inbox_decisions"))
            for column in ("owner_label", "owner_text", "transcript", "tagged_by"):
                conn.execute(text(f"ALTER TABLE feedback DROP COLUMN {column}"))
            conn.execute(text("INSERT INTO customers (id, name) VALUES (1, 'Eli')"))
            conn.execute(text("INSERT INTO devices (id, device_id, site, customer_id) VALUES (1, 'd1', 'x', 1)"))
            conn.execute(text("INSERT INTO feedback (id, device_pk, s3_key) VALUES (1, 1, :k)"), {"k": key})
            conn.execute(text("INSERT INTO raw_revisions (s3_key, etag, fetched_at, body) "
                              "VALUES (:k, 'e1', now(), CAST(:b AS jsonb))"), {"k": key, "b": json.dumps(body)})
        command.upgrade(cfg, "0020")
        with engine.begin() as conn:
            row = conn.execute(text("SELECT owner_label, owner_text, tagged_by, superseded_by FROM feedback")).one()
            assert tuple(row) == ("escalation", "someone at the gate", "Eli", "")
            conn.execute(text("INSERT INTO inbox_decisions (feedback_id, decision, decided_at) VALUES (1, 'fixed', now())"))
        with engine.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
        # and re-running it changes nothing: 0020 down and up again on the repaired database
        command.downgrade(cfg, "0019")
        command.upgrade(cfg, "head")
        with engine.begin() as conn:
            assert conn.execute(text("SELECT owner_label FROM feedback")).scalar() == "escalation"
            assert conn.execute(text("SELECT decision FROM inbox_decisions")).scalar() == "fixed"
    finally:
        engine.dispose()
