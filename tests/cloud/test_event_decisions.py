"""Events, not only clips: the indexer reads each clip's event-layer decision (session, sent / held and why,
baseline shadow) and the API shows the real outcome, groups by session and filters on 'would raise'."""
import copy
import json
from datetime import datetime, timezone

from sqlalchemy import null, select, update

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import s3client  # noqa: F401  (fixture)

NOW = datetime(2026, 10, 3, 14, tzinfo=timezone.utc)
SESSION = "5e55i0n0a1"


def _meta(ts, **alert):
    body = copy.deepcopy(b.fixture_json("prod_alert.meta.json"))
    stem = f"front_side_{ts}_alert"
    body["clip_path"] = f"clips\\front_side\\2026-10-03\\{stem}.mp4"
    body["clip_start_ts"], body["clip_end_ts"] = ts - 3.0, ts + 6.0
    body["alert"].update(alert)
    return f"production_test/meta/front_side/2026-10-03/{stem}.meta.json", body


def _event(notify, reason, **kw):
    return {"notify": notify, "session_id": SESSION, "reason": reason, "reply_to": None, "new_people": 0,
            "known_text": "", "entities": [], "fresh": [], "unmarked": False, "counted_by": "head-count", **kw}


def _seed(client, s3client):
    from home_guard_project.cloud.s3 import S3

    s3 = S3(s3client, b.BUCKET)
    b.seed_bucket(s3client)
    for key, body in (
            _meta(1791020400, sent=True, event=_event(True, "suspicious, first in this event"), dispatch={"sent": True}),
            _meta(1791020460, sent=False, event=_event(False, "normal: kept in the event, not sent"),
                  not_sent_reason="normal: kept in the event, not sent",
                  baseline={"mode": "shadow", "rarity": "rare", "would_raise": True, "raise": False,
                            "text_en": "Not usual for this camera at this hour"}),
            _meta(1791020520, sent=False, event=_event(False, "suspicious, but the owner said who is here",
                                                       known_text="Daniel's workers"),
                  not_sent_reason="suspicious, but the owner said who is here", downgraded="appearance only")):
        b.put(s3client, key, json.dumps(body))
    client.app.state.s3 = s3
    client.app.state.clock = lambda: NOW
    with session_scope(client.app.state.engine) as s:
        return b.index_fixture_bucket(s, s3, consent_training=True, now=NOW).id


def test_outcomes_sessions_and_the_shadow_filter(client, staff_factory, s3client):
    device_pk = _seed(client, s3client)
    _, _, _, h = staff_factory("admin")
    items = client.get("/v1/events", params={"limit": 50}, headers=h).json()["items"]
    by_session = [e for e in items if e["session_id"] == SESSION]
    assert [e["outcome"] for e in by_session] == [
        "Lowered: appearance only  ·  Not sent: owner said known (Daniel's workers)",
        "Kept in the event, not sent (normal)  ·  Would raise: rare for this camera",
        "Sent"]
    assert [e["outcome_code"] for e in by_session] == ["known", "held", "sent"]
    old = next(e for e in items if e["session_id"] is None and e["camera"] == "front_side")
    assert old["outcome_code"] in (None, "sent", "undelivered")  # a clip from before events keeps its delivery
    shadow = client.get("/v1/events", params={"would_raise": True}, headers=h).json()["items"]
    assert [e["outcome_code"] for e in shadow] == ["held"]
    sessions = client.get("/v1/events/sessions", params={"session_id": [SESSION, "nope"]}, headers=h).json()
    assert len(sessions) == 1 and sessions[0]["clips"] == 3 and sessions[0]["sent"] == 1
    assert sessions[0]["camera"] == "front_side" and sessions[0]["first_utc"] < sessions[0]["last_utc"]
    # labelers: pseudonymous camera, never the owner's own words
    _, _, _, lab = staff_factory("labeler")
    seen = [e for e in client.get("/v1/events", params={"limit": 50}, headers=lab).json()["items"]
            if e["session_id"] == SESSION]
    assert seen[0]["outcome"] == "Lowered: appearance only  ·  Not sent: owner said known"
    assert client.get("/v1/events/sessions", params={"session_id": SESSION}, headers=lab).json()[0]["camera"].startswith("cam-")
    # events indexed before migration 0018 are backfilled from stored revisions on the next pass
    with session_scope(client.app.state.engine) as s:
        s.execute(update(m.Event).where(m.Event.device_pk == device_pk).values(decision=null(), session_id=None))  # SQL NULL, as after 0018
    from home_guard_project.cloud.indexer import index_device
    with session_scope(client.app.state.engine) as s:
        index_device(s, client.app.state.s3, s.get(m.Device, device_pk), now=NOW)
    with session_scope(client.app.state.engine) as s:
        assert s.scalar(select(m.Event.id).where(m.Event.decision.is_(None))) is None
        assert len(s.scalars(select(m.Event.id).where(m.Event.session_id == SESSION)).all()) == 3


def test_ai_call_flags_filter_and_backfill(client, staff_factory, s3client):
    """alert.vlm_failed / vlm_rescued and model_input.rescue reach the events: a filter per flag (count them per day
    per camera) and the flags' words; events indexed with an older decision record are re-read from stored revisions."""
    from home_guard_project.cloud.s3 import S3

    s3 = S3(s3client, b.BUCKET)
    b.seed_bucket(s3client)
    failed_key, failed = _meta(1791021000, sent=False, vlm_failed=True, alert_command="[none]",
                               event=_event(False, "normal: kept in the event, not sent"))
    rescued_key, rescued = _meta(1791021060, sent=True, vlm_rescued=True, event=_event(True, "suspicious, first in this event"))
    rescued["model_input"] = {"frame_indices": [0, 2], "rescue": {"max_side": 768, "reason": "request too large"}}
    for key, body in ((failed_key, failed), (rescued_key, rescued)):
        b.put(s3client, key, json.dumps(body))
    client.app.state.s3 = s3
    client.app.state.clock = lambda: NOW
    with session_scope(client.app.state.engine) as s:
        device_pk = b.index_fixture_bucket(s, s3, now=NOW).id
    _, _, _, h = staff_factory("admin")
    only = client.get("/v1/events", params={"vlm": "failed", "with_total": True}, headers=h).json()
    assert only["total"] == 1 and only["items"][0]["ai_flags"] == ["AI failed"]
    saved = client.get("/v1/events", params={"vlm": "rescued"}, headers=h).json()["items"]
    assert [e["ai_flags"] for e in saved] == [["Rescued at 768 px", "AI answer rescued"]]
    assert client.get("/v1/events", params={"vlm": "maybe"}, headers=h).status_code == 422
    # an event indexed before the flags (decision version 1) is re-read on the next pass
    with session_scope(client.app.state.engine) as s:
        s.execute(update(m.Event).where(m.Event.device_pk == device_pk)
                  .values(decision={"v": 1, "sent": False}))
    from home_guard_project.cloud.indexer import index_device
    with session_scope(client.app.state.engine) as s:
        index_device(s, client.app.state.s3, s.get(m.Device, device_pk), now=NOW)
    with session_scope(client.app.state.engine) as s:
        versions = {d.get("v") for d in s.scalars(select(m.Event.decision).where(m.Event.device_pk == device_pk))}
        assert versions == {2}
    again = client.get("/v1/events", params={"vlm": "failed"}, headers=h).json()["items"]
    assert len(again) == 1 and again[0]["ai_flags"] == ["AI failed"]
