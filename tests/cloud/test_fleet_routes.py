from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.models import AuditLog, Customer, Device, Event, Feedback

from . import builders as b

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _setup(client):
    client.app.state.clock = lambda: NOW


def _hb(site, when, **kw):
    body = {"site": site, "mode": "inference", "host": f"box-{site}", "time_utc": when.isoformat().replace("+00:00", "Z"),
            "collector_running": True, "disk_free_gb": 500.0, "clips_outbox": 0, "stopped": False,
            "newest_clip_utc": when.isoformat().replace("+00:00", "Z"),
            "cameras": {"front": {"newest_clip_utc": when.isoformat().replace("+00:00", "Z")},
                        "back": {"newest_clip_utc": (NOW - timedelta(days=3)).isoformat().replace("+00:00", "Z")}}}
    body.update(kw)
    return body


def _ts(dt):
    return dt.timestamp()


def _seed(client):
    with session_scope(client.app.state.engine) as s:
        fresh = b.enroll(s, "zeta", "Zed Co")
        old = b.enroll(s, "alpha", "Acme")
        none = b.enroll(s, "mid", "Mid")
        fresh.last_heartbeat = _hb("zeta", NOW - timedelta(minutes=5))
        old.last_heartbeat = _hb("alpha", NOW - timedelta(hours=3))
        s.flush()
        for i, (kind, age) in enumerate([("alert", 1), ("trigger", 2), ("alert", 30)]):
            s.add(Event(device_pk=fresh.id, site="zeta", camera="front", stem=f"e{i}", kind=kind,
                        start_ts=_ts(NOW - timedelta(hours=age))))
        for i, (verdict, days) in enumerate([("false_alarm", 1), ("false_alarm", 9), ("true_alert", 1)]):
            s.add(Feedback(device_pk=fresh.id, verdict=verdict, s3_key=f"k{i}", received_at=NOW - timedelta(days=days)))
        return fresh.device_id, old.device_id, none.device_id


def test_fleet_order_verdicts_and_counts(client, staff_factory):
    _setup(client)
    _, _, _, h = staff_factory("support")
    fresh_id, old_id, none_id = _seed(client)
    r = client.get("/v1/fleet", headers=h)
    assert r.status_code == 200
    devs = r.json()["devices"]
    assert [d["site"] for d in devs] == ["alpha", "zeta", "mid"]
    assert [d["verdict"] for d in devs] == ["offline", "warning", "unknown"]
    z = devs[1]
    assert (z["events_24h"], z["alerts_24h"], z["false_alarms_7d"]) == (2, 1, 1)
    assert z["cameras_total"] == 2 and z["cameras_stale"] == 1
    assert z["mode"] == "inference" and z["host"] == "box-zeta" and z["collector_running"] is True
    assert z["last_seen_utc"].startswith("2026-10-03T11:55")
    assert devs[2]["last_seen_utc"] is None and devs[2]["cameras_total"] == 0
    assert r.json()["generated_utc"].startswith("2026-10-03T12:00")


def test_roles(client, staff_factory):
    _setup(client)
    _, _, _, lab = staff_factory("labeler")
    _, _, _, sup = staff_factory("support")
    _, _, _, adm = staff_factory("admin")
    assert client.get("/v1/fleet", headers=lab).status_code == 403
    assert client.get("/v1/customers", headers=lab).status_code == 403
    assert client.get("/v1/customers", headers=sup).status_code == 200
    assert client.post("/v1/customers", json={"name": "X"}, headers=sup).status_code == 403
    assert client.post("/v1/customers", json={"name": "X"}, headers=lab).status_code == 403
    assert client.post("/v1/devices/enroll", json={"customer_id": 1, "site": "s"}, headers=sup).status_code == 403
    assert client.post("/v1/customers", json={"name": "X"}, headers=adm).status_code == 200


def test_enroll_and_duplicate_site(client, staff_factory):
    _setup(client)
    _, _, _, h = staff_factory("admin")
    cid = client.post("/v1/customers", json={"name": "Acme"}, headers=h).json()["id"]
    r = client.post("/v1/devices/enroll", json={"customer_id": cid, "site": "home1", "tailscale_host": "x"}, headers=h)
    assert r.status_code == 200
    d = r.json()
    assert d["site"] == "home1" and d["customer_name"] == "Acme" and d["verdict"] == "unknown"
    assert len(d["device_id"]) == 36
    assert client.post("/v1/devices/enroll", json={"customer_id": cid, "site": "home1"}, headers=h).status_code == 409
    assert client.post("/v1/devices/enroll", json={"customer_id": 9999, "site": "other"}, headers=h).status_code == 404
    with session_scope(client.app.state.engine) as s:
        acts = [a.action for a in s.scalars(select(AuditLog))]
    assert "device_enroll" in acts


def test_customer_crud_and_patch_audit(client, staff_factory):
    _setup(client)
    _, _, _, h = staff_factory("admin")
    r = client.post("/v1/customers", json={"name": "Acme", "notes": "secret note"}, headers=h)
    assert r.status_code == 200
    cid = r.json()["id"]
    client.post("/v1/devices/enroll", json={"customer_id": cid, "site": "home1"}, headers=h)
    got = client.get(f"/v1/customers/{cid}", headers=h).json()
    assert got["name"] == "Acme" and [d["site"] for d in got["devices"]] == ["home1"]
    assert client.get("/v1/customers/999", headers=h).status_code == 404
    # consent comes from the sales contract (on); switching some off records the withdrawal
    body = {"name": "Acme 2", "timezone": "Asia/Jerusalem", "consent_live": True, "consent_recordings": False,
            "consent_training": False, "notes": "other secret"}
    p = client.patch(f"/v1/customers/{cid}", json=body, headers=h)
    assert p.status_code == 200 and p.json()["name"] == "Acme 2" and p.json()["consent_live"] is True
    assert p.json()["consent_source"] == "withdrawn" and p.json()["consent_recordings"] is False
    assert len(client.get("/v1/customers", headers=h).json()) == 1
    with session_scope(client.app.state.engine) as s:
        row = s.scalars(select(AuditLog).where(AuditLog.action == "customer_update")).one()
    assert row.customer_id == cid
    assert row.detail == {"changed": ["consent_recordings", "consent_training", "name", "notes"]}  # names, never values
    assert "secret" not in str(row.detail) and "Acme" not in str(row.detail)


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def _renamed_house(client):
    """A box renamed from site ameer_tes2 to ameer_week_0_1: the heartbeat still lists the old ids (14-day archive)."""
    with session_scope(client.app.state.engine) as s:
        dev = b.enroll(s, "ameer_week_0_1", "Ameer")
        dev.last_heartbeat = _hb("ameer_week_0_1", NOW - timedelta(minutes=5), cameras={
            "ameer_tes2_ch6": {"newest_clip_utc": _iso(NOW - timedelta(days=5))},
            "ameer_tes2_ch3": {"newest_clip_utc": _iso(NOW - timedelta(days=6))},
            "ameer_week_0_1_ch6": {"newest_clip_utc": _iso(NOW - timedelta(minutes=10))},
            "ameer_week_0_1_ch2": {"newest_clip_utc": _iso(NOW - timedelta(hours=30))}})
        s.flush()
        s.add(Event(device_pk=dev.id, site="ameer_week_0_1", camera="ameer_tes2_ch1", stem="old",
                    kind="trigger", start_ts=_ts(NOW - timedelta(days=9))))
        return dev.customer_id


def test_warnings_cover_current_cameras_only_with_owner_names(client, staff_factory):
    _setup(client)
    _, _, _, h = staff_factory("support")
    cid = _renamed_house(client)
    dev = client.get("/v1/fleet", headers=h).json()["devices"][0]
    assert dev["verdict"] == "warning" and dev["cameras_total"] == 2 and dev["cameras_stale"] == 1
    assert [r["message"] for r in dev["reasons"]] == ["No clip for 30 h from Camera 2 — check it has power and network"]
    assert "tes2" not in repr(dev["reasons"])
    cams = client.get("/v1/cameras", params={"customer_id": cid}, headers=h).json()
    assert [(c["camera"], c["name"], c["current"]) for c in cams] == [
        ("ameer_week_0_1_ch2", "Camera 2", True), ("ameer_week_0_1_ch6", "Camera 6", True),
        ("ameer_tes2_ch1", "Camera 1", False), ("ameer_tes2_ch3", "Camera 3", False),
        ("ameer_tes2_ch6", "Camera 6", False)]
    assert cams[1]["newest_clip_utc"].startswith("2026-10-03T11:50")
    assert client.get("/v1/cameras", params={"customer_id": 99999}, headers=h).status_code == 404
    _, _, _, lab = staff_factory("labeler")
    assert client.get("/v1/cameras", headers=lab).status_code == 403


def test_the_box_camera_list_decides_and_names_when_present(client, staff_factory):
    _setup(client)
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        dev = b.enroll(s, "home", "Home")
        dev.last_heartbeat = _hb("home", NOW - timedelta(minutes=5), cameras={
            "home_ch1": {"newest_clip_utc": _iso(NOW - timedelta(hours=40))},
            "home_ch2": {"newest_clip_utc": _iso(NOW - timedelta(minutes=1))},
            "home_ch5": {"newest_clip_utc": _iso(NOW - timedelta(hours=50))},
            "old_ch9": {"newest_clip_utc": _iso(NOW - timedelta(minutes=30))}},
            camera_list=[{"id": "home_ch1", "name": "כניסה ראשית", "channel": "1", "enabled": True},
                         {"id": "home_ch2", "name": "Camera 2", "channel": "2", "enabled": True},
                         {"id": "home_ch3", "name": "Camera 3", "channel": "3", "enabled": True},
                         {"id": "home_ch5", "name": "Pool", "channel": "5", "enabled": False}])
    dev = client.get("/v1/fleet", headers=h).json()["devices"][0]
    # the list decides: old_ch9 (fresh clip, not configured) is retired; the switched-off Pool never warns;
    # a configured camera with no clip yet does
    assert [r["message"] for r in dev["reasons"]] == [
        "No clip for 40 h from כניסה ראשית — check it has power and network",
        "No clip recorded yet from Camera 3 — check its login and stream on the box"]
    assert dev["cameras_total"] == 3
    cams = client.get("/v1/cameras", headers=h).json()
    assert [(c["camera"], c["name"], c["current"], c["enabled"]) for c in cams] == [
        ("home_ch1", "כניסה ראשית", True, True), ("home_ch2", "Camera 2", True, True),
        ("home_ch3", "Camera 3", True, True), ("home_ch5", "Pool", False, False), ("old_ch9", "Camera 9", False, True)]


def test_old_site_names_of_one_box_are_one_house(client, staff_factory):
    """Live 2026-10-09: ameer_test, ameer_tes2 and ameer_week_0_1 (same box DESKTOP-43DP1TI) were three Fleet rows,
    two of them offline. The box heard from last is the house; the others are its old site names."""
    _setup(client)
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        rows = {}
        for site, age, host in (("ameer_test", timedelta(hours=114), "DESKTOP-43DP1TI"),
                                ("ameer_tes2", timedelta(hours=49), "desktop-43dp1ti"),
                                ("ameer_week_0_1", timedelta(minutes=3), "DESKTOP-43DP1TI"),
                                ("other_house", timedelta(minutes=3), "other-box")):
            dev = b.enroll(s, site, site.title())
            dev.last_heartbeat = _hb(site, NOW - age, host=host)
            rows[site] = dev.device_id
        quiet = b.enroll(s, "never_heard", "Never")  # no heartbeat and no host: its own row
    devs = {d["site"]: d for d in client.get("/v1/fleet", headers=h).json()["devices"]}
    assert devs["ameer_week_0_1"]["old_sites"] == ["ameer_tes2", "ameer_test"]
    assert devs["ameer_week_0_1"]["replaced_by"] is None
    for old in ("ameer_tes2", "ameer_test"):
        assert devs[old]["replaced_by"] == rows["ameer_week_0_1"] and devs[old]["replaced_by_site"] == "ameer_week_0_1"
    assert devs["other_house"]["replaced_by"] is None and devs["other_house"]["old_sites"] == []
    assert devs["never_heard"]["replaced_by"] is None
    cid = client.get("/v1/customers", headers=h).json()
    old_customer = next(c for c in cid if c["name"] == "Ameer_Tes2")
    assert client.get(f"/v1/customers/{old_customer['id']}", headers=h).json()["devices"][0]["replaced_by_site"] == "ameer_week_0_1"
