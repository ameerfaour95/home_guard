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
    # the box's consent is only a proposal: the customer's real consents stay false until an admin confirms
    assert (cust.consent_live, cust.consent_recordings, cust.consent_training) == (False, False, False)
    assert cust.consent_proposed == {"live": True, "recordings": True, "training": False,
                                     "recorded_utc": "2026-10-02T10:00:00Z", "installer": "Ameer"}
    assert cust.notes == ""
    assert cust.consent_recorded_utc is None
    assert (dev.enrolled_by, dev.tailscale_host, dev.app_version) == ("setup", "dana-box", "1.4.0")
    row = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "auto_enroll"))
    assert row.staff_id is None and row.staff_name == "system:setup" and row.device_id == dev.device_id
    assert PHONE not in json.dumps(row.detail) and "Dana" not in json.dumps(row.detail)
    prop = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "consent_proposed"))
    assert prop.staff_name == "system:setup" and sorted(prop.detail["fields"]) == [
        "consent_live", "consent_recordings", "consent_training"]


def test_heartbeat_only_needs_details(session, s3, s3client):
    put_hb(s3client, "new_site-2")
    run(session, s3)
    dev = device_of(session, "new_site-2")
    cust = session.get(m.Customer, dev.customer_id)
    assert cust.name == "New Site 2" and cust.name_source == "discovered"
    assert not (cust.consent_live or cust.consent_recordings or cust.consent_training)
    assert dev.enrolled_by == "discovered"
    row = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "auto_discover"))
    assert row is not None and row.staff_name == "system:discovery"


def test_idempotent_rerun(session, s3, s3client):
    put_reg(s3client, "dana_house", registration())
    put_hb(s3client, "other")
    run(session, s3)
    before = (session.scalar(select(func.count()).select_from(m.Customer)),
              session.scalar(select(func.count()).select_from(m.AuditLog)))
    run(session, s3)
    after = (session.scalar(select(func.count()).select_from(m.Customer)),
             session.scalar(select(func.count()).select_from(m.AuditLog)))
    assert before == after == (2, 3)
    assert session.scalar(select(func.count()).select_from(m.Device)) == 2


def _cust(session, site="dana_house"):
    dev = device_of(session, site)
    session.refresh(dev)
    cust = session.get(m.Customer, dev.customer_id)
    session.refresh(cust)
    return dev, cust


def _confirm(session, site="dana_house"):
    """What an admin PATCH does to the stored consents (the route is tested separately)."""
    _, cust = _cust(session, site)
    cust.consent_live, cust.consent_recordings, cust.consent_training = True, True, False
    cust.consent_proposed = None
    cust.consent_recorded_utc = NOW
    session.commit()


def test_grants_are_proposals_revocations_apply(session, s3, s3client):
    put_reg(s3client, "dana_house", registration(training=False, recorded="2026-10-02T10:00:00Z"))
    run(session, s3)
    _confirm(session)
    # a newer registration (after the admin's confirmation) grants training and revokes live
    put_reg(s3client, "dana_house", registration(training=True, live=False, recorded="2026-10-03T12:01:00Z",
                                                host="new-host", version="1.5.0"))
    discovery.discover(session, s3, now=datetime(2026, 10, 3, 12, 2, tzinfo=timezone.utc))
    session.commit()
    dev, cust = _cust(session)
    assert (cust.consent_live, cust.consent_training) == (False, False)  # revoked now, grant only proposed
    assert cust.consent_proposed["training"] is True and cust.consent_proposed["live"] is False
    assert (dev.tailscale_host, dev.app_version) == ("new-host", "1.5.0")
    rev = session.scalar(select(m.AuditLog).where(m.AuditLog.action == "consent_revoked_by_owner"))
    assert rev.detail["fields"] == ["consent_live"] and rev.staff_name == "system:discovery"
    prop = list(session.scalars(select(m.AuditLog).where(m.AuditLog.action == "consent_proposed")
                                .order_by(m.AuditLog.id)))[-1]
    assert prop.detail["fields"] == ["consent_training"]
    # an older answer changes the host but never consents or the proposal
    put_reg(s3client, "dana_house", registration(training=False, live=True, recorded="2026-10-01T00:00:00Z",
                                                host="third"))
    discovery.discover(session, s3, now=datetime(2026, 10, 3, 12, 3, tzinfo=timezone.utc))
    session.commit()
    dev, cust = _cust(session)
    assert (cust.consent_live, cust.consent_training) == (False, False)
    assert cust.consent_proposed["training"] is True and dev.tailscale_host == "third"


def test_pure_revocation_has_no_proposal(session, s3, s3client):
    put_reg(s3client, "dana_house", registration(recorded="2026-10-02T10:00:00Z"))
    run(session, s3)
    _confirm(session)
    put_reg(s3client, "dana_house", registration(recordings=False, recorded="2026-10-03T12:01:00Z"))
    discovery.discover(session, s3, now=datetime(2026, 10, 3, 12, 2, tzinfo=timezone.utc))
    session.commit()
    _, cust = _cust(session)
    assert (cust.consent_live, cust.consent_recordings) == (True, False) and cust.consent_proposed is None


def test_future_registration_is_rejected(session, s3, s3client):
    put_reg(s3client, "dana_house", registration(recorded="2026-10-03T12:06:00Z"))
    assert run(session, s3) == {}
    assert device_of(session, "dana_house") is None
    problem = session.scalar(select(m.IndexProblem).where(
        m.IndexProblem.s3_key == "dataset_dana_house/_status/registration.json"))
    assert problem is not None and "registration time in the future" in problem.reason
    # still rejected on the next pass (stored revision), and within 5 minutes is fine
    assert run(session, s3) == {} and device_of(session, "dana_house") is None
    put_reg(s3client, "dana_house", registration(recorded="2026-10-03T12:04:00Z"))
    run(session, s3)
    assert device_of(session, "dana_house") is not None


def _count_gets(monkeypatch, s3):
    calls = []
    real = type(s3).get_text
    monkeypatch.setattr(type(s3), "get_text",
                        lambda self, key, *a, **k: (calls.append(key), real(self, key, *a, **k))[1])
    return calls


def test_unchanged_registration_is_not_fetched_again(session, s3, s3client, monkeypatch):
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    calls = _count_gets(monkeypatch, s3)
    run(session, s3)
    run(session, s3)
    assert calls == []


def test_unknown_site_with_stored_revision_reads_the_stored_body(session, s3, s3client, monkeypatch):
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    session.execute(m.Device.__table__.delete())
    session.commit()
    calls = _count_gets(monkeypatch, s3)
    run(session, s3)
    assert calls == [] and device_of(session, "dana_house") is not None


def test_stored_revision_has_no_phone(session, s3, s3client):
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    row = session.scalar(select(m.RawRevision).where(m.RawRevision.s3_key.like("%registration.json")))
    assert "owner_phone" not in row.body and PHONE not in json.dumps(row.body) and row.body["owner_name"] == "Dana Cohen"
    assert session.get(m.Customer, device_of(session, "dana_house").customer_id).owner_phone == PHONE


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
    assert cust.consent_live is False and cust.consent_proposed["live"] is True


def test_admin_enrolled_site_keeps_its_customer(session, s3, s3client):
    dev = b.enroll(session, "dana_house", "Admin Chosen")
    session.commit()
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    session.refresh(dev)
    cust = session.get(m.Customer, dev.customer_id)
    assert cust.name == "Admin Chosen" and dev.enrolled_by == "admin" and dev.app_version == "1.4.0"
    assert session.scalar(select(func.count()).select_from(m.Customer)) == 1
    # consents never change from a registration; a differing one is only proposed
    assert (cust.consent_live, cust.consent_recordings, cust.consent_training) == (False, False, False)
    assert cust.consent_proposed["live"] is True


def test_admin_enrolled_site_matching_consents_makes_no_proposal(session, s3, s3client):
    dev = b.enroll(session, "dana_house", "Admin Chosen")
    cust = session.get(m.Customer, dev.customer_id)
    cust.consent_live, cust.consent_recordings, cust.consent_training = True, True, False
    session.commit()
    put_reg(s3client, "dana_house", registration())
    run(session, s3)
    session.refresh(cust)
    assert cust.consent_proposed is None and cust.consent_live is True
    put_reg(s3client, "dana_house", registration(live=False, recorded="2026-10-03T00:00:00Z"))
    run(session, s3)
    session.refresh(cust)
    assert cust.consent_live is True and cust.consent_proposed["live"] is False  # no auto-revocation either


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
    assert (fleet["enrolled_by"], fleet["needs_details"], fleet["app_version"]) == ("setup", True, "1.4.0")  # proposal pending
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


def test_admin_patch_of_consents_confirms_the_proposal(client, staff_factory, s3):
    from home_guard_project.cloud.db import session_scope

    s3.client.put_object(Bucket=b.BUCKET, Key="dataset_dana_house/_status/registration.json",
                         Body=json.dumps(registration()).encode())
    with session_scope(client.app.state.engine) as s:
        discovery.discover(s, s3, now=NOW)
    _, _, _, admin = staff_factory("admin")
    _, _, _, support = staff_factory("support")
    cust = client.get("/v1/customers", headers=support).json()[0]
    assert cust["consent_proposed"]["live"] is True and cust["consent_proposed"]["installer"] == "Ameer"
    assert cust["consent_proposed"]["recorded_utc"].startswith("2026-10-02T10:00:00")
    assert (cust["consent_live"], cust["consent_training"]) == (False, False)
    dev = client.get("/v1/fleet", headers=admin).json()["devices"][0]
    assert dev["needs_details"] is True  # a proposal is pending even though the box named the owner
    r = client.patch(f"/v1/customers/{cust['id']}", json={"name": cust["name"], "consent_live": True, "consent_recordings": True},
                     headers=admin)
    assert r.status_code == 200 and r.json()["consent_proposed"] is None and r.json()["consent_live"] is True
    assert client.get("/v1/fleet", headers=admin).json()["devices"][0]["needs_details"] is False
    audit_rows = client.get("/v1/audit", headers=admin).json()["items"]
    conf = next(a for a in audit_rows if a["action"] == "consent_confirmed")
    assert conf["detail"]["fields"] == ["consent_live", "consent_recordings"]
    assert conf["detail"]["proposal_recorded_utc"].startswith("2026-10-02T10:00:00")
    with session_scope(client.app.state.engine) as s:
        assert s.get(m.Customer, cust["id"]).consent_recorded_utc is not None


def test_renamed_owner_keeps_the_old_name_as_an_identity_term(client, staff_factory, s3):
    from home_guard_project.cloud.db import session_scope

    s3.client.put_object(Bucket=b.BUCKET, Key="dataset_dana_house/_status/registration.json",
                         Body=json.dumps(registration()).encode())
    with session_scope(client.app.state.engine) as s:
        discovery.discover(s, s3, now=NOW)
    _, _, _, admin = staff_factory("admin")
    cid = client.get("/v1/customers", headers=admin).json()[0]["id"]
    assert client.patch(f"/v1/customers/{cid}", json={"name": "D. Cohen-Levi"}, headers=admin).status_code == 200
    with session_scope(client.app.state.engine) as s:
        terms = redact.identity_terms(s, device_of(s, "dana_house"))
        assert "dana cohen" in terms and "d. cohen-levi" in terms


def test_staff_names_cannot_impersonate_system_actors(capsys):
    from home_guard_project.cloud import manage

    args = manage.build_parser().parse_args(["create-staff", "--email", "x@example.com", "--name", " System:Setup",
                                             "--role", "admin"])
    assert manage.cmd_create_staff(args) == 1
    assert "system:" in capsys.readouterr().err


def _discovered(client, s3):
    from home_guard_project.cloud.db import session_scope

    s3.client.put_object(Bucket=b.BUCKET, Key="dataset_dana_house/_status/registration.json",
                         Body=json.dumps(registration()).encode())
    with session_scope(client.app.state.engine) as s:
        discovery.discover(s, s3, now=NOW)


def test_confirm_settles_exactly_the_consent_the_customer_gave_at_setup(client, staff_factory, s3):
    from home_guard_project.cloud.db import session_scope

    _discovered(client, s3)
    staff, _, _, admin = staff_factory("admin")
    cust = client.get("/v1/customers", headers=admin).json()[0]
    url = f"/v1/customers/{cust['id']}/consent/confirm"
    # a grant beyond the proposal (training was not given) is refused, in PATCH as well
    r = client.patch(f"/v1/customers/{cust['id']}", headers=admin, json={"name": cust["name"], "consent_training": True})
    assert r.status_code == 422
    assert client.post(url, headers=admin, json={"recorded_utc": "2026-10-01T10:00:00Z"}).status_code == 409  # stale
    r = client.post(url, headers=admin, json={"recorded_utc": cust["consent_proposed"]["recorded_utc"]})
    assert r.status_code == 200, r.text
    got = r.json()
    assert (got["consent_live"], got["consent_recordings"], got["consent_training"]) == (True, True, False)
    assert got["consent_proposed"] is None
    conf = next(a for a in client.get("/v1/audit", headers=admin).json()["items"] if a["action"] == "consent_confirmed")
    assert conf["staff"] == staff.name and conf["ts"] and conf["detail"]["fields"] == ["consent_live", "consent_recordings"]
    assert conf["detail"]["proposal"]["installer"] == "Ameer"
    assert conf["detail"]["proposal_recorded_utc"].startswith("2026-10-02T10:00:00")
    with session_scope(client.app.state.engine) as s:
        assert s.get(m.Customer, cust["id"]).consent_recorded_utc == datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
    # nothing left to confirm: blocked with a clear message
    r = client.post(url, headers=admin, json={"recorded_utc": "2026-10-02T10:00:00Z"})
    assert r.status_code == 409 and "can only come from the customer" in r.json()["detail"]
    # withdrawing is always allowed
    r = client.patch(f"/v1/customers/{cust['id']}", headers=admin, json={"name": cust["name"], "consent_live": False})
    assert r.status_code == 200 and r.json()["consent_live"] is False


@pytest.mark.parametrize("role", ["support", "labeler"])
def test_only_admins_confirm_consent(client, staff_factory, s3, role):
    _discovered(client, s3)
    _, _, _, admin = staff_factory("admin")
    _, _, _, other = staff_factory(role)
    cust = client.get("/v1/customers", headers=admin).json()[0]
    r = client.post(f"/v1/customers/{cust['id']}/consent/confirm", headers=other,
                    json={"recorded_utc": cust["consent_proposed"]["recorded_utc"]})
    assert r.status_code == 403


def test_confirm_only_some_of_the_yes_answers_and_never_a_no(client, staff_factory, s3):
    _discovered(client, s3)
    _, _, _, admin = staff_factory("admin")
    cust = client.get("/v1/customers", headers=admin).json()[0]
    url = f"/v1/customers/{cust['id']}/consent/confirm"
    when = cust["consent_proposed"]["recorded_utc"]
    r = client.post(url, headers=admin, json={"recorded_utc": when, "fields": ["live", "training"]})
    assert r.status_code == 422 and "did not agree to training" in r.json()["detail"]
    r = client.post(url, headers=admin, json={"recorded_utc": when, "fields": ["recordings"]})
    assert r.status_code == 200, r.text
    got = r.json()
    assert (got["consent_live"], got["consent_recordings"], got["consent_training"]) == (False, True, False)
    assert got["consent_proposed"] is None
    # with the proposal settled, nothing more can be turned on
    r = client.patch(f"/v1/customers/{cust['id']}", headers=admin, json={"name": cust["name"], "consent_live": True})
    assert r.status_code == 422
    # what is already on may be sent again unchanged
    r = client.patch(f"/v1/customers/{cust['id']}", headers=admin, json={"name": cust["name"], "consent_recordings": True})
    assert r.status_code == 200
