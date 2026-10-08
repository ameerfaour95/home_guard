"""The customer page's Chat tab: the box's uploaded Telegram conversation, read-only, consent-gated and audited."""
import json
from datetime import datetime, timezone

from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import s3client  # noqa: F401  (fixture)

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
SITE = "ameer_week_0_1"


def _ts(hour, minute=0, day=9):
    return datetime(2026, 10, day, hour, minute, tzinfo=timezone.utc).timestamp()


def _line(**kw):
    base = {"ts": _ts(9), "who": "box", "name": "", "kind": "message", "text": "", "camera": "", "alert_id": "",
            "image": "", "delivered": True, "error": ""}
    return json.dumps({**base, **kw}, ensure_ascii=False)


def _seed(client, s3client):
    from home_guard_project.cloud.s3 import S3

    s3client.create_bucket(Bucket=b.BUCKET)
    client.app.state.s3 = S3(s3client, b.BUCKET)
    client.app.state.clock = lambda: NOW
    prefix = f"production_{SITE}/chat/"
    today = "\n".join([
        _line(ts=_ts(10, 5), kind="alert", camera=f"{SITE}_ch1", alert_id=f"{SITE}_ch1_1791540300_alert",
              text=f"{SITE}_ch1: a person at the door", image=f"{SITE}_ch1_1791540300_alert.jpg"),
        _line(ts=_ts(10, 6), who="owner", name="Ameer", kind="button", text="It's me"),
        "{not json",
        _line(ts=_ts(10, 7), who="assistant", kind="answer", text=f"Noted, {SITE}_ch2 stays on."),
        _line(ts=_ts(10, 8), kind="alert", camera=f"{SITE}_ch2", text="a car", delivered=False, error="Forbidden"),
    ]) + "\n"
    yesterday = _line(ts=_ts(20, 0, day=8), who="owner", name="Ameer", text="who was at the gate?") + "\n"
    b.put(s3client, prefix + "2026-10-09.jsonl", today.encode("utf-8"))
    b.put(s3client, prefix + "2026-10-08.jsonl", yesterday.encode("utf-8"))
    b.put(s3client, prefix + f"images/{SITE}_ch1_1791540300_alert.jpg", b"\xff\xd8jpeg")
    with session_scope(client.app.state.engine) as s:
        dev = b.enroll(s, SITE, "Ameer")
        dev.last_heartbeat = {"site": SITE, "time_utc": "2026-10-09T11:59:00Z", "cameras": {},
                              "camera_list": [{"id": f"{SITE}_ch1", "name": "כניסה ראשית", "channel": "1", "enabled": True},
                                              {"id": f"{SITE}_ch2", "name": "Camera 2", "channel": "2", "enabled": True}]}
        return dev.customer_id, dev.device_id


def test_a_day_of_chat_with_owner_names_and_states(client, staff_factory, s3client):
    cid, device_id = _seed(client, s3client)
    _, _, _, h = staff_factory("support")
    r = client.get(f"/v1/customers/{cid}/chat", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["day"] == "2026-10-09" and body["days"] == ["2026-10-09", "2026-10-08"]
    msgs = body["messages"]
    assert [m_["kind"] for m_ in msgs] == ["alert", "button", "answer", "alert"]  # oldest first; the bad line skipped
    assert msgs[0]["text"] == "כניסה ראשית: a person at the door" and msgs[0]["camera_name"] == "כניסה ראשית"
    assert msgs[0]["image"] == f"{SITE}_ch1_1791540300_alert.jpg"
    assert msgs[2]["text"] == "Noted, Camera 2 stays on." and not any(SITE in m_["text"] for m_ in msgs)
    assert msgs[3]["delivered"] is False and msgs[3]["error"] == "Forbidden"
    older = client.get(f"/v1/customers/{cid}/chat", params={"day": "2026-10-08"}, headers=h).json()
    assert [m_["text"] for m_ in older["messages"]] == ["who was at the gate?"]
    assert client.get(f"/v1/customers/{cid}/chat", params={"day": "9 Oct"}, headers=h).status_code == 400
    # every view is audited (never the search text) and the owner is told
    found = client.get(f"/v1/customers/{cid}/chat", params={"q": "GATE"}, headers=h).json()
    assert found["day"] is None and [m_["text"] for m_ in found["messages"]] == ["who was at the gate?"]
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "chat_view").order_by(m.AuditLog.id)).all()
        assert [r_.detail["day"] for r_ in rows] == ["2026-10-09", "2026-10-08", None]
        assert rows[-1].detail == {"day": None, "search": True, "messages": 1} and "gate" not in repr(rows[-1].detail)
        notices = s.scalars(select(m.OwnerNotice).where(m.OwnerNotice.kind == "chat")).all()
        assert len(notices) == 1 and notices[0].s3_key.startswith(f"fleet/{device_id}/notices/")
    notice = json.loads(s3client.get_object(Bucket=b.BUCKET, Key=notices[0].s3_key)["Body"].read())
    from home_guard_project.fleet_contract.notices import NOTICE_KINDS
    assert notice["kind"] == "chat" in NOTICE_KINDS and notice["message"].startswith("Home Guard support viewed your chat")


def test_chat_pictures_are_presigned_and_audited(client, staff_factory, s3client):
    cid, _ = _seed(client, s3client)
    _, _, _, h = staff_factory("admin")
    image = f"{SITE}_ch1_1791540300_alert.jpg"
    r = client.post(f"/v1/customers/{cid}/chat/images/access", json={"site": SITE, "image": image}, headers=h)
    assert r.status_code == 200 and image in r.json()["url"] and r.json()["mime"] == "image/jpeg"
    for site, name in ((SITE, "missing.jpg"), (SITE, "../x.jpg"), ("other", image)):
        assert client.post(f"/v1/customers/{cid}/chat/images/access", json={"site": site, "image": name},
                           headers=h).status_code == 404
    with session_scope(client.app.state.engine) as s:
        row = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_view")).one()
        assert row.detail["role"] == "chat_image" and row.target.endswith(image)


def test_chat_needs_recordings_consent_and_a_staff_role(client, staff_factory, s3client):
    cid, _ = _seed(client, s3client)
    _, _, _, lab = staff_factory("labeler")
    _, _, _, h = staff_factory("admin")
    assert client.get(f"/v1/customers/{cid}/chat", headers=lab).status_code == 403
    with session_scope(client.app.state.engine) as s:
        s.get(m.Customer, cid).consent_recordings = False
    r = client.get(f"/v1/customers/{cid}/chat", headers=h)
    assert r.status_code == 403
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(m.AuditLog.action)).all().count("media_denied") == 1
        assert not s.scalars(select(m.AuditLog).where(m.AuditLog.action == "chat_view")).all()
    assert client.get("/v1/customers/99999/chat", headers=h).status_code == 404


def test_real_chat_shapes_old_site_ids_and_sent_files(client, staff_factory, s3client):
    """Shapes from the live files (production_ameer_week_0_1/chat, 2026-10-04..08): alerts and buttons name the
    cameras by the OLD site's ids (ameer_tes2_ch6) and the text starts with them; the assistant's photo and video
    lines carry only a file name. Old ids take the owner's name of the camera now on their channel."""
    cid, _ = _seed(client, s3client)
    lines = "\n".join([
        _line(ts=_ts(11, 0), kind="alert", camera="ameer_tes2_ch1", alert_id="ameer_tes2_ch1_1791115353_alert",
              text="🟢 Looks normal · ameer_tes2_ch1\nA person walks across the driveway",
              image="ameer_tes2_ch1_1791115353_alert.jpg"),
        _line(ts=_ts(11, 1), kind="video", text="Video of the alert", alert_id="ameer_tes2_ch1_1791115353_alert"),
        _line(ts=_ts(11, 2), who="owner", name="Ameer", kind="button", text="✏️ Other…", camera="ameer_tes2_ch1",
              alert_id="ameer_tes2_ch1_1791115353_alert"),
        _line(ts=_ts(11, 3), who="assistant", kind="photo", text="ameer_tes2_ch2_1791274712_2a87cfbd.jpg"),
        _line(ts=_ts(11, 4), who="assistant", kind="video", text=f"{SITE}_ch1_1791366411_alert.mp4"),
    ]) + "\n"
    b.put(s3client, f"production_{SITE}/chat/2026-10-07.jsonl", lines.encode("utf-8"))
    _, _, _, h = staff_factory("admin")
    got = client.get(f"/v1/customers/{cid}/chat", params={"day": "2026-10-07"}, headers=h).json()["messages"]
    assert [(m_["kind"], m_["camera_name"]) for m_ in got] == [
        ("alert", "כניסה ראשית"), ("video", "כניסה ראשית"), ("button", "כניסה ראשית"), ("photo", "Camera 2"),
        ("video", "כניסה ראשית")]
    assert got[0]["text"] == "🟢 Looks normal · כניסה ראשית\nA person walks across the driveway"
    assert got[3]["text"] == "" and got[4]["text"] == ""  # a file name is not shown (it holds the camera id)
    assert not any("ameer_tes2" in m_["text"] or SITE in m_["text"] for m_ in got)
