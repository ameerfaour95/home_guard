"""Task 16: boxes that publish _status/registration.json (or only a heartbeat) are enrolled automatically."""
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from home_guard_project.cloud import discovery, models as m, redact
from . import builders as b

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
PHONE = "+972501234567"


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


def registration(site="dana_house", owner="Dana Cohen", live=True, recordings=True, training=False,
                 recorded="2026-10-02T10:00:00Z", host="dana-box", version="1.4.0"):
    return {"schema_version": 1, "site": site, "owner_name": owner, "owner_phone": PHONE,
            "consent": {"live": live, "recordings": recordings, "training": training,
                        "recorded_utc": recorded, "recorded_by": "installer"},
            "installer": "Ameer", "installed_utc": "2026-10-02T10:00:00Z", "tailscale_host": host,
            "box_host": "BOX1", "app_version": version}


def put_reg(s3client, site, body):
    b.put(s3client, f"dataset_{site}/_status/registration.json", body)


def put_hb(s3client, site):
    b.put(s3client, f"dataset_{site}/_status/heartbeat.json", (b.FIXTURES / "heartbeat.json").read_bytes())


def run(session, s3, **kw):
    out = discovery.discover(session, s3, now=NOW, **kw)
    session.commit()
    return out


def device_of(session, site):
    return session.scalar(select(m.Device).where(m.Device.site == site))


def test_registration_creates_customer_and_device(session, s3, s3client):
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    dev = device_of(session, "dana_house")
    cust = session.get(m.Customer, dev.customer_id)
    assert (cust.name, cust.name_source, cust.owner_phone) == ("Dana Cohen", "setup", PHONE)
    assert (cust.consent_live, cust.consent_recordings, cust.consent_training) == (True, True, False)
    assert cust.notes == ""
    assert cust.consent_recorded_utc == datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)
    assert (dev.enrolled_by, dev.tailscale_host, dev.app_version) == ("setup", "dana-box", "1.4.0")
    row = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "auto_enroll"))
    assert row.staff_id is None and row.staff_name == "setup" and row.device_id == dev.device_id
    assert PHONE not in json.dumps(row.detail) and "Dana" not in json.dumps(row.detail)


def test_heartbeat_only_needs_details(session, s3, s3client):
    put_hb(s3client, "new_site-2")
    run(session, s3)
    dev = device_of(session, "new_site-2")
    cust = session.get(m.Customer, dev.customer_id)
    assert cust.name == "New Site 2" and cust.name_source == "discovered"
    assert not (cust.consent_live or cust.consent_recordings or cust.consent_training)
    assert dev.enrolled_by == "discovered"
    assert session.scalar(select(m.AuditLog).where(m.AuditLog.action == "auto_discover")) is not None


def test_idempotent_rerun(session, s3, s3client):
    put_reg(s3client, "dana_house", registration())
    put_hb(s3client, "other")
    run(session, s3)
    before = (session.scalar(select(func.count()).select_from(m.Customer)),
              session.scalar(select(func.count()).select_from(m.AuditLog)))
    run(session, s3)
    after = (session.scalar(select(func.count()).select_from(m.Customer)),
             session.scalar(select(func.count()).select_from(m.AuditLog)))
    assert before == after == (2, 2)
    assert session.scalar(select(func.count()).select_from(m.Device)) == 2


def test_consent_newer_wins_older_ignored(session, s3, s3client):
    put_reg(s3client, "dana_house", registration(training=False, recorded="2026-10-02T10:00:00Z"))
    run(session, s3)
    put_reg(s3client, "dana_house", registration(training=True, live=False, recorded="2026-10-03T08:00:00Z",
                                                host="new-host", version="1.5.0"))
    run(session, s3)
    dev = device_of(session, "dana_house")
    cust = session.get(m.Customer, dev.customer_id)
    assert (cust.consent_live, cust.consent_training) == (False, True)
    assert (dev.tailscale_host, dev.app_version) == ("new-host", "1.5.0")
    row = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "consent_update"))
    assert sorted(row.detail["changed"]) == ["consent_live", "consent_training"] and row.staff_name == "setup"
    # an older answer changes the host but never the consents
    put_reg(s3client, "dana_house", registration(training=False, live=True, recorded="2026-10-01T00:00:00Z",
                                                host="third"))
    run(session, s3)
    session.refresh(cust)
    session.refresh(dev)
    assert (cust.consent_live, cust.consent_training) == (False, True)
    assert dev.tailscale_host == "third"
    assert session.scalar(select(func.count()).select_from(m.AuditLog).where(
        m.AuditLog.action == "consent_update")) == 1


def test_admin_rename_is_preserved(session, s3, s3client):
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    dev = device_of(session, "dana_house")
    cust = session.get(m.Customer, dev.customer_id)
    cust.name, cust.name_source = "Dana C. (VIP)", "admin"
    session.commit()
    put_reg(s3client, "dana_house", registration(owner="Dana Cohen-Levi", host="h2"))
    run(session, s3)
    session.refresh(cust)
    assert cust.name == "Dana C. (VIP)"


def test_registration_upgrades_a_discovered_site(session, s3, s3client):
    put_hb(s3client, "dana_house")
    run(session, s3)
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    dev = device_of(session, "dana_house")
    cust = session.get(m.Customer, dev.customer_id)
    assert (dev.enrolled_by, cust.name, cust.name_source) == ("setup", "Dana Cohen", "setup")
    assert cust.consent_live is True


def test_admin_enrolled_site_keeps_its_customer(session, s3, s3client):
    dev = b.enroll(session, "dana_house", "Admin Chosen")
    session.commit()
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    session.refresh(dev)
    cust = session.get(m.Customer, dev.customer_id)
    assert cust.name == "Admin Chosen" and dev.enrolled_by == "admin" and dev.app_version == "1.4.0"
    assert session.scalar(select(func.count()).select_from(m.Customer)) == 1


def test_ignore_list_and_excluded_pools(session, s3, s3client, monkeypatch):
    for site in ("uca", "smarthome", "multi", "skipme", "keepme"):
        put_hb(s3client, site)
    monkeypatch.setenv("HG_CLOUD_DISCOVERY_IGNORE", "skipme, dataset_other")
    run(session, s3)
    assert [d.site for d in session.scalars(select(m.Device))] == ["keepme"]


def test_bad_registration_is_skipped_not_fatal(session, s3, s3client):
    b.put(s3client, "dataset_broken/_status/registration.json", b"{nope")
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    assert device_of(session, "broken") is None and device_of(session, "dana_house") is not None


def test_owner_name_becomes_an_identity_term_and_phone_stays_private(client, staff_factory, s3):
    from home_guard_project.cloud.db import session_scope

    s3.client.put_object(Bucket=b.BUCKET, Key="dataset_dana_house/_status/registration.json",
                         Body=json.dumps(registration()).encode())
    with session_scope(client.app.state.engine) as s:
        discovery.discover(s, s3, now=NOW)
        dev = device_of(s, "dana_house")
        terms = redact.identity_terms(s, dev)
        assert "dana cohen" in terms and "dana_house" in terms and "dana-box" in terms
        assert "Dana" not in redact.redact_text("Dana walked in", terms, {})
    _, _, _, admin = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    fleet = client.get("/v1/fleet", headers=admin).json()["devices"][0]
    assert (fleet["enrolled_by"], fleet["needs_details"], fleet["app_version"]) == ("setup", False, "1.4.0")
    assert PHONE not in client.get("/v1/audit", headers=admin).text
    assert PHONE not in client.get("/v1/customers", headers=admin).text
    assert client.get("/v1/fleet", headers=lab).status_code == 403
    assert PHONE not in client.get("/v1/events", headers=lab).text


def test_needs_details_clears_when_admin_renames(client, staff_factory, s3):
    from home_guard_project.cloud.db import session_scope

    s3.client.put_object(Bucket=b.BUCKET, Key="dataset_mystery/_status/heartbeat.json",
                         Body=(b.FIXTURES / "heartbeat.json").read_bytes())
    with session_scope(client.app.state.engine) as s:
        discovery.discover(s, s3, now=NOW)
    _, _, _, admin = staff_factory("admin")
    dev = client.get("/v1/fleet", headers=admin).json()["devices"][0]
    assert dev["needs_details"] is True and dev["enrolled_by"] == "discovered"
    r = client.patch(f"/v1/customers/{dev['customer_id']}", json={"name": "Real Name"}, headers=admin)
    assert r.status_code == 200
    assert client.get("/v1/fleet", headers=admin).json()["devices"][0]["needs_details"] is False


def test_loop_is_wired(db_engine):
    from home_guard_project.cloud import loops

    built = loops.build_loops(loops.sessionmaker(db_engine), object(), engine=db_engine)
    lp = next(lp for lp in built.loops if lp.name == "discovery")
    assert lp.interval == 300 and lp.lock_key == loops.DISCOVERY_LOCK
    assert loops.LOCKS["discovery"] == loops.DISCOVERY_LOCK
    assert len({x.lock_key for x in built.loops}) == len(built.loops)


def test_manage_discover_once_parser():
    from home_guard_project.cloud import manage

    assert manage.build_parser().parse_args(["discover-once"]).fn is manage.cmd_discover_once
