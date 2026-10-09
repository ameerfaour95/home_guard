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

TAG_FEEDBACK = b.FEEDBACK.replace("_1791020300000.", "_1791020400000.")
MUTE_FEEDBACK = b.FEEDBACK.replace("_1791020300000.", "_1791020500000.")
QUESTION_FEEDBACK = b.FEEDBACK.replace("_1791020300000.", "_1791020600000.")


def _answers():
    tag = b.feedback_body(verdict="real_but_wrong", time_utc="2026-10-03T09:50:00Z")
    tag.update(owner_label="other", owner_text="זה הגנן שלנו", transcript="זה הגנן שלנו", tagged_by="Dana",
               raw_text="")
    mute = b.feedback_body(verdict="none", time_utc="2026-10-03T09:55:00Z")
    mute.update(action="mute", raw_text="")
    question = b.feedback_body(verdict="none", time_utc="2026-10-03T09:58:00Z")
    question.update(raw_text="למה המצלמה שולחת לי כל כך הרבה התראות?")
    return {TAG_FEEDBACK: json.dumps(tag).encode(), MUTE_FEEDBACK: json.dumps(mute).encode(),
            QUESTION_FEEDBACK: json.dumps(question).encode()}


def _seeded(client, s3client, consent=True):
    _seed(client, s3client, consent_training=consent, overrides=_answers())


def test_indexer_stores_the_owner_answer_and_feedback_out_shows_it(client, staff_factory, s3client):
    _seeded(client, s3client)
    _, _, _, admin = staff_factory("admin")
    _, _, _, labeler = staff_factory("labeler")
    with session_scope(client.app.state.engine) as s:
        fb = s.scalar(select(m.Feedback).where(m.Feedback.s3_key == TAG_FEEDBACK))
        assert (fb.owner_label, fb.owner_text, fb.transcript, fb.tagged_by) == (
            "other", "זה הגנן שלנו", "זה הגנן שלנו", "Dana")
        plain = s.scalar(select(m.Feedback).where(m.Feedback.s3_key == b.FEEDBACK))
        assert plain.owner_label == "" and plain.tagged_by == "Owner"       # "from": {"name": "Owner"}
    eid = _event_id(client, b.STEM)
    detail = client.get(f"/v1/events/{eid}", headers=admin).json()
    tagged = [f for f in detail["feedback"] if f["owner_label"]]
    assert [(f["owner_label"], f["owner_text"], f["transcript"]) for f in tagged] == [
        ("other", "זה הגנן שלנו", "זה הגנן שלנו")]
    # a labeler of a consenting customer reads the tag word only, never the owner's words
    shown = client.get(f"/v1/events/{eid}", headers=labeler).json()["feedback"]
    assert {(f["owner_label"], f["owner_text"], f["transcript"]) for f in shown} >= {("other", "", "")}


def test_inbox_lists_answers_once_with_filters(client, staff_factory, s3client):
    _seeded(client, s3client)
    _, _, _, h = staff_factory("admin")
    items = client.get("/v1/inbox", headers=h).json()
    # the question, the tag and the verdict button answer; the pause alone is not an answer; the orphan has no clip
    assert [(i["owner_label"], i["probably_not_label"]) for i in items] == [("", True), ("other", False), ("", False)]
    assert items[0]["raw_text"].startswith("למה")              # the box's own rule: a question is not a label
    assert {i["prompt_version"] for i in items} == {items[0]["prompt_version"]} and items[0]["prompt_version"]
    items = items[1:]
    tag = items[0]
    assert tag["clip_key"] == f"ev:{tag['event_id']}" and tag["customer"] == "Acme" and tag["camera"] == "front_side"
    assert tag["owner_text"] == "זה הגנן שלנו" and tag["tagged_by"] == "Dana" and tag["decision"] is None
    assert tag["model_summary"] and tag["consent_training"] is True
    assert [i["owner_label"] for i in client.get("/v1/inbox", headers=h, params={"owner_label": "other"}).json()] == ["other"]
    assert client.get("/v1/inbox", headers=h, params={"camera": "nope"}).json() == []
    assert client.get("/v1/inbox", headers=h, params={"customer_id": tag["customer_id"] + 99}).json() == []
    assert len(client.get("/v1/inbox", headers=h, params={"from_utc": "2026-10-03T09:45:00Z"}).json()) == 2
    assert len(client.get("/v1/inbox", headers=h, params={"to_utc": "2026-10-03T09:45:00Z"}).json()) == 1
    everything = client.get("/v1/inbox", headers=h).json()
    page = client.get("/v1/inbox", headers=h, params={"limit": 1}).json()
    rest = client.get("/v1/inbox", headers=h, params={"before_id": page[0]["feedback_id"]}).json()
    assert [i["feedback_id"] for i in page + rest] == [i["feedback_id"] for i in everything]


def test_decisions_are_audited_and_move_answers_out_of_the_waiting_list(client, staff_factory, s3client):
    _seeded(client, s3client)
    _, _, _, h = staff_factory("admin")
    question, tag, button = client.get("/v1/inbox", headers=h).json()
    r = client.post(f"/v1/inbox/{tag['feedback_id']}/decision", headers=h, json={"decision": "accepted"})
    assert r.status_code == 200, r.text
    assert r.json()["decision"] == "accepted" and r.json()["decided_by"] == "Admin 1"
    r = client.post(f"/v1/inbox/{button['feedback_id']}/decision", headers=h,
                    json={"decision": "not_label", "note": "a question about the bill"})
    assert r.json()["decision"] == "not_label" and r.json()["decision_note"] == "a question about the bill"
    assert [i["feedback_id"] for i in client.get("/v1/inbox", headers=h).json()] == [question["feedback_id"]]
    assert len(client.get("/v1/inbox", headers=h, params={"handled": "handled"}).json()) == 2
    r = client.delete(f"/v1/inbox/{button['feedback_id']}/decision", headers=h)
    assert r.json()["decision"] is None
    assert [i["feedback_id"] for i in client.get("/v1/inbox", headers=h).json()] == [
        question["feedback_id"], button["feedback_id"]]
    with session_scope(client.app.state.engine) as s:
        rows = s.execute(select(m.AuditLog.action, m.AuditLog.target, m.AuditLog.detail)
                         .where(m.AuditLog.action.like("inbox_%")).order_by(m.AuditLog.id)).all()
    actions = [a for a, _, _ in rows]
    assert actions.count("inbox_decision") == 2 and actions.count("inbox_reopen") == 1 and "inbox_view" in actions
    decision = next(d for a, t, d in rows if a == "inbox_decision")
    assert decision == {"decision": "accepted", "event_id": tag["event_id"], "owner_label": "other",
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
    tag = client.get("/v1/inbox", headers=h).json()[1]
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
