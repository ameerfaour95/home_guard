"""Owner notices are pushed to the box over Tailscale SSH (cloud/notice_delivery.py), with a fake ssh runner."""
import base64
import json
from datetime import timedelta

from sqlalchemy import select

from home_guard_project.cloud import audit, notice_delivery
from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope
from home_guard_project.fleet_contract import notices

from . import builders as b
from .test_event_routes import NOW, s3client  # noqa: F401  (fixture)
from .test_media_routes import _art, _body, _set_camera, _setup

HOME_BOX = "df7b4c2e1bde4e379774d703ca0a0d18"
ADDED = (0, '{"result":"added","id":1}\r\n', "")
UNREACHABLE = (255, "", "ssh: connect to host box-test port 22: Connection timed out")


class FakeSSH:
    """Answers each call with the next scripted (exit, stdout, stderr); the last one repeats."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers) or [ADDED], []

    def __call__(self, argv):
        self.calls.append(argv)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def body(self, i=-1):
        cmd = self.calls[i][-1]
        return json.loads(base64.b64decode(cmd.split(" --b64 ")[1].split(" ")[0]).decode("utf-8"))


def _house(client, s3client, host="box-test", user="boxuser"):
    dev_pk, device_id = _setup(client, s3client)
    with session_scope(client.app.state.engine) as s:
        dev = s.get(m.Device, dev_pk)
        dev.tailscale_host, dev.ssh_user, dev.last_heartbeat_at = host, user, NOW - timedelta(minutes=2)
    return dev_pk, device_id


def _view(client, h, path=b.PROD_CLIP):
    r = client.post(f"/v1/artifacts/{_art(client, path)}/access", json={"purpose": "support"}, headers=h)
    assert r.status_code == 200, r.text


def _notice(client):
    with session_scope(client.app.state.engine) as s:
        return s.scalars(select(m.OwnerNotice)).one()


def _attempts(client):
    with session_scope(client.app.state.engine) as s:
        return [(a.reason, a.detail, a.staff_name, a.device_id) for a in s.scalars(
            select(m.AuditLog).where(m.AuditLog.action == "notice_delivery").order_by(m.AuditLog.id))]


def _loop(client, ssh, now=NOW):
    with session_scope(client.app.state.engine) as s:
        return notice_delivery.deliver_pending(s, now, ssh)


def test_a_view_pushes_the_notice_to_the_box_after_the_commit(client, staff_factory, s3client):
    _, device_id = _house(client, s3client)
    _, _, _, h = staff_factory("support")
    client.app.state.notice_runner = ssh = FakeSSH(ADDED)
    _view(client, h)
    assert len(ssh.calls) == 1
    argv = ssh.calls[0]
    assert argv[:7] == ["ssh", "-i", argv[2], "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    assert argv[2].endswith("homeguard_box") and argv[7] == "boxuser@box-test"
    assert argv[8].startswith(r"cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box notices add")
    fleet_key = s3client.list_objects_v2(Bucket=b.BUCKET, Prefix=f"fleet/{device_id}/notices/")["Contents"][0]["Key"]
    pushed = ssh.body()
    assert pushed == _body(s3client, fleet_key) and pushed["schema_version"] == 1 and pushed["kind"] == "recording"
    row = _notice(client)
    assert pushed["id"] == row.id and row.delivery_state == "delivered" and row.delivery_attempts == 1
    [(outcome, detail, actor, audited_device)] = _attempts(client)
    assert (outcome, actor, audited_device) == ("delivered", "notice delivery", device_id)
    assert detail == {"notice_id": row.id, "ok": True, "outcome": "delivered", "attempt": 1, "host": "box-test",
                      "exit": 0, "result": "added"}
    assert "Front side" not in json.dumps(detail)  # never the notice text


def test_without_a_runner_the_view_leaves_the_notice_pending(client, staff_factory, s3client):
    _house(client, s3client)
    _, _, _, h = staff_factory("support")
    assert client.app.state.notice_runner is None  # a server that does not run the loops
    _view(client, h)
    assert _notice(client).delivery_state == "pending" and _attempts(client) == []


def test_unreachable_stays_pending_and_is_retried_on_the_next_heartbeat(client, staff_factory, s3client):
    dev_pk, _ = _house(client, s3client)
    _, _, _, h = staff_factory("support")
    client.app.state.notice_runner = ssh = FakeSSH(UNREACHABLE)
    _view(client, h)
    row = _notice(client)
    assert (row.delivery_state, row.delivery_attempts) == ("pending", 1)
    assert row.last_delivery_error == "retry: ssh: connect to host box-test port 22: Connection timed out"
    assert _loop(client, ssh, NOW + timedelta(minutes=5)) == 0 and len(ssh.calls) == 1  # no newer heartbeat: waits
    with session_scope(client.app.state.engine) as s:
        s.get(m.Device, dev_pk).last_heartbeat_at = NOW + timedelta(minutes=4)  # the indexer saw a new heartbeat
    ssh.answers = [ADDED]
    assert _loop(client, ssh, NOW + timedelta(minutes=10)) == 1 and len(ssh.calls) == 2
    assert _notice(client).delivery_state == "delivered"
    attempts = _attempts(client)
    assert [a[0] for a in attempts] == ["retry", "delivered"]
    assert attempts[0][1]["ok"] is False and attempts[0][1]["exit"] == 255
    assert attempts[0][1]["error"].startswith("ssh: connect to host")


def test_after_seven_days_of_waiting_the_notice_is_given_up(client, staff_factory, s3client):
    dev_pk, _ = _house(client, s3client)
    _, _, _, h = staff_factory("support")
    client.app.state.notice_runner = ssh = FakeSSH(UNREACHABLE)
    _view(client, h)
    with session_scope(client.app.state.engine) as s:
        s.get(m.Device, dev_pk).last_heartbeat_at = NOW + timedelta(days=6)
    assert _loop(client, ssh, NOW + timedelta(days=6)) == 0 and len(ssh.calls) == 2  # still trying on day 6
    with session_scope(client.app.state.engine) as s:
        s.get(m.Device, dev_pk).last_heartbeat_at = NOW + timedelta(days=7, hours=2)
    _loop(client, ssh, NOW + timedelta(days=7, hours=1))
    assert len(ssh.calls) == 2 and _notice(client).delivery_state == "gave_up"
    assert [a[0] for a in _attempts(client)] == ["retry", "retry", "gave_up"]
    _loop(client, ssh, NOW + timedelta(days=8))
    assert len(ssh.calls) == 2 and len(_attempts(client)) == 3  # given up once, never tried again


def test_a_rejected_body_is_not_resent_but_a_rewrite_is(client, staff_factory, s3client):
    dev_pk, _ = _house(client, s3client)
    _, _, _, h = staff_factory("support")
    client.app.state.notice_runner = ssh = FakeSSH((1, '{"error":"bad base64"}', ""))
    _view(client, h)
    assert _notice(client).delivery_state == "failed"
    assert _notice(client).last_delivery_error == "rejected: bad base64"
    with session_scope(client.app.state.engine) as s:
        s.get(m.Device, dev_pk).last_heartbeat_at = NOW + timedelta(minutes=4)
    _loop(client, ssh, NOW + timedelta(minutes=5))
    assert len(ssh.calls) == 1  # permanent for that body
    assert _attempts(client)[0][:2] == ("rejected", {"notice_id": _notice(client).id, "ok": False,
                                                     "outcome": "rejected", "attempt": 1, "host": "box-test",
                                                     "exit": 1, "error": "bad base64"})
    a2 = _art(client, b.FP_CLIP)
    _set_camera(client, a2, "driveway")
    ssh.answers = [(0, '{"result":"updated","id":1}', "")]
    client.clock = NOW + timedelta(minutes=10)
    _view(client, h, b.FP_CLIP)  # a new body (another camera) is a new attempt
    assert len(ssh.calls) == 2 and _notice(client).delivery_state == "delivered"


def test_a_rewrite_within_the_window_is_pushed_again_under_the_same_id(client, staff_factory, s3client):
    _house(client, s3client)
    _, _, _, h = staff_factory("support")
    client.app.state.notice_runner = ssh = FakeSSH(ADDED, (0, '{"result":"updated","id":1}', ""))
    a2 = _art(client, b.FP_CLIP)
    _set_camera(client, a2, "driveway")
    _view(client, h)
    client.clock = NOW + timedelta(minutes=18)
    _view(client, h, b.FP_CLIP)
    assert len(ssh.calls) == 2
    first, second = ssh.body(0), ssh.body(1)
    assert first["id"] == second["id"] == _notice(client).id
    assert first["cameras"] == ["Front side"] and second["cameras"] == ["Front side", "Driveway"]
    assert _notice(client).delivery_state == "delivered"
    assert [a[1]["result"] for a in _attempts(client)] == ["added", "updated"]


def test_the_notice_reaches_the_box_by_box_id_across_a_site_rename(client, staff_factory, s3client):
    """ameer_tes2 -> ameer_week_0_1: a view from the old row is pushed to the box's live row (its host and user)."""
    dev_pk, _ = _house(client, s3client, host="old-host-gone", user="olduser")
    staff, _, _, _ = staff_factory("support")
    with session_scope(client.app.state.engine) as s:
        old = s.get(m.Device, dev_pk)
        old.last_heartbeat = {"site": "test", "time_utc": (NOW - timedelta(hours=49)).isoformat(), "box_id": HOME_BOX}
        new = b.enroll(s, "test_new", "Acme new")
        new.tailscale_host, new.ssh_user, new.last_heartbeat_at = "desktop-43dp1ti", "ameer", NOW
        new.last_heartbeat = {"site": "test_new", "time_utc": (NOW - timedelta(minutes=3)).isoformat(),
                              "box_id": HOME_BOX}
        new_device_id = new.device_id
    with session_scope(client.app.state.engine) as s:
        audit.owner_notice(s, client.app.state.s3, s.get(m.Device, dev_pk), s.get(m.Staff, staff.id), "recording",
                           ["Front side"], NOW)
    ssh = FakeSSH(ADDED)
    assert _loop(client, ssh) == 1
    assert ssh.calls[0][7] == "ameer@desktop-43dp1ti"
    assert _attempts(client)[0][3] == new_device_id  # audited against the box's live row


def test_an_oversize_body_is_refused_before_sending(client, staff_factory, s3client):
    dev_pk, _ = _house(client, s3client)
    staff, _, _, _ = staff_factory("support")
    with session_scope(client.app.state.engine) as s:
        audit.owner_notice(s, client.app.state.s3, s.get(m.Device, dev_pk), s.get(m.Staff, staff.id), "recording",
                           [f"camera {i} " + "x" * 90 for i in range(40)], NOW)
    ssh = FakeSSH(ADDED)
    _loop(client, ssh)
    assert ssh.calls == [] and _notice(client).delivery_state == "failed"
    [(outcome, detail, _, _)] = _attempts(client)
    assert outcome == "too_long" and detail["limit"] == notices.MAX_B64 and detail["size"] > notices.MAX_B64


def test_a_box_without_a_tailscale_host_waits(client, staff_factory, s3client):
    _house(client, s3client, host="")
    _, _, _, h = staff_factory("support")
    client.app.state.notice_runner = ssh = FakeSSH(ADDED)
    _view(client, h)
    assert ssh.calls == [] and _notice(client).delivery_state == "pending"
    assert _notice(client).last_delivery_error == "no_target: the box has no tailscale host"
    assert _attempts(client)[0][0] == "no_target"
