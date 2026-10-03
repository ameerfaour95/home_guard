"""Final fix round, privacy and consent (fix-final-decisions.md): identity history (C1), no hidden-household oracle
in labeler-created names (I1), no real customer ids for labelers, and one media consent decision (I2)."""
import json

import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, _event_id, _index, s3client  # noqa: F401  (fixture)


def _set_summary(client, stem, text):
    with session_scope(client.app.state.engine) as s:
        ev = s.get(m.Event, _event_id(client, stem))
        ev.summary, ev.summary_redacted = text, None


def _labeler_text(client, h, event_id):
    detail = client.get(f"/v1/events/{event_id}", headers=h)
    listed = client.get("/v1/events", headers=h)
    assert detail.status_code == listed.status_code == 200, (detail.text, listed.text)
    return (detail.text + listed.text).lower()


# ---------------------------------------------------------------- C1: identity history

def test_old_customer_name_stays_redacted_after_a_rename(client, staff_factory, s3client):
    _, customer_id = _index(client, s3client, consent_training=True, customer_name="Zircona")
    _, _, _, adm = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    eid = _event_id(client, b.STEM)
    _set_summary(client, b.STEM, "A courier at Zircona's gate; Felvoria van outside")
    assert "zircona" not in _labeler_text(client, lab, eid)
    r = client.patch(f"/v1/customers/{customer_id}", headers=adm, json={"name": "Felvoria"})
    assert r.status_code == 200, r.text
    text = _labeler_text(client, lab, eid)
    assert "zircona" not in text and "felvoria" not in text and "customer-" in text
    # the stored search copy is recomputed with the history too
    with session_scope(client.app.state.engine) as s:
        stored = s.get(m.Event, eid).summary_redacted
    assert stored is not None and "Zircona" not in stored and "Felvoria" not in stored
    # and an admin cannot name a collection after the old name either
    r = client.post("/v1/studio/collections", headers=adm, json={"name": "zircona night"})
    assert r.status_code == 400 and r.json()["detail"] == "Name must not identify a household"


def test_old_camera_display_name_and_heartbeat_host_stay_redacted(client, staff_factory, s3client):
    from home_guard_project.cloud.indexer import index_device

    device_pk, _ = _index(client, s3client, consent_training=True)
    _, _, _, lab = staff_factory("labeler")
    eid = _event_id(client, b.STEM)
    with session_scope(client.app.state.engine) as s:
        s.add(m.Camera(device_pk=device_pk, name="front_side", display_name="Quorvex Gate"))
    hb = b.fixture_json("heartbeat.json")
    hb["host"] = "wendolyn-laptop"
    b.put(s3client, b.HEARTBEAT, hb)
    with session_scope(client.app.state.engine) as s:  # the indexer sees the display name and the host
        index_device(s, client.app.state.s3, s.get(m.Device, device_pk), now=NOW)
    with session_scope(client.app.state.engine) as s:  # both change later
        s.scalars(select(m.Camera).where(m.Camera.device_pk == device_pk)).one().display_name = "Front"
    hb["host"] = "box-2"
    b.put(s3client, b.HEARTBEAT, hb)
    with session_scope(client.app.state.engine) as s:
        index_device(s, client.app.state.s3, s.get(m.Device, device_pk), now=NOW)
    _set_summary(client, b.STEM, "Someone at Quorvex Gate, logged from wendolyn-laptop and Wendolyn")
    text = _labeler_text(client, lab, eid)
    assert "quorvex" not in text and "wendolyn" not in text
    with session_scope(client.app.state.engine) as s:
        kinds = {(a.kind, a.value) for a in s.scalars(select(m.IdentityAlias))}
    assert {("display_name", "Quorvex Gate"), ("host", "wendolyn-laptop"), ("host", "box-2"),
            ("site", "test"), ("customer", "Acme"), ("camera", "front_side")} <= kinds


def test_enroll_records_aliases(client, staff_factory):
    _, _, _, adm = staff_factory("admin")
    cid = client.post("/v1/customers", headers=adm, json={"name": "Yarrowby"}).json()["id"]
    r = client.post("/v1/devices/enroll", headers=adm,
                    json={"customer_id": cid, "site": "yarrow_home", "tailscale_host": "yarrow-box"})
    assert r.status_code == 200, r.text
    with session_scope(client.app.state.engine) as s:
        kinds = {(a.kind, a.value) for a in s.scalars(select(m.IdentityAlias))}
    assert {("customer", "Yarrowby"), ("site", "yarrow_home"), ("host", "yarrow-box")} <= kinds


# ---------------------------------------------------------------- I1: no oracle in labeler-created names

def test_labeler_names_are_not_checked_against_hidden_households(client, staff_factory, s3client):
    _index(client, s3client, consent_training=False, customer_name="Zircona")  # hidden from labelers
    _, _, _, lab = staff_factory("labeler")
    answers = []
    for name in ("Zircona", "Velmora"):
        r = client.post("/v1/studio/collections", headers=lab, json={"name": name})
        answers.append(r.status_code)
        if r.status_code == 200:
            cid = r.json()["id"]
            req = {"collection_id": cid, "name": name.lower() + "-v1", "formats": ["clips"]}
            assert client.post("/v1/studio/exports/preview", headers=lab, json=req).status_code == 200
    assert answers == [200, 200]


def test_labeler_collections_are_private_to_their_creator_and_admins(client, staff_factory, s3client):
    _index(client, s3client, consent_training=True)
    _, _, _, adm = staff_factory("admin")
    _, _, _, lab_a = staff_factory("labeler")
    _, _, _, lab_b = staff_factory("labeler")
    mine = client.post("/v1/studio/collections", headers=lab_a, json={"name": "mine"}).json()["id"]
    shared = client.post("/v1/studio/collections", headers=adm, json={"name": "shared"}).json()["id"]
    assert {c["id"] for c in client.get("/v1/studio/collections", headers=lab_a).json()} == {mine, shared}
    assert {c["id"] for c in client.get("/v1/studio/collections", headers=lab_b).json()} == {shared}
    assert {c["id"] for c in client.get("/v1/studio/collections", headers=adm).json()} == {mine, shared}
    absent = client.get("/v1/studio/collections/999999/items", headers=lab_b)
    eid = _event_id(client, b.STEM)
    for method, url, body in (("GET", f"/v1/studio/collections/{mine}/items", None),
                              ("POST", f"/v1/studio/collections/{mine}/items", {"event_ids": [eid]}),
                              ("DELETE", f"/v1/studio/collections/{mine}/items", {"event_ids": [eid]}),
                              ("POST", "/v1/studio/exports/preview",
                               {"collection_id": mine, "name": "x", "formats": ["clips"]})):
        r = client.request(method, url, headers=lab_b, json=body)
        assert r.status_code == 404 and r.json()["detail"] == "Collection not found", (method, url, r.text)
    assert absent.status_code == 404
    assert client.get(f"/v1/studio/collections/{mine}/items", headers=adm).status_code == 200


# ---------------------------------------------------------------- labelers never get the real customer id

def test_labelers_get_customer_id_zero_and_cannot_filter_by_it(client, staff_factory, s3client):
    _, customer_id = _index(client, s3client, consent_training=True)
    _, _, _, lab = staff_factory("labeler")
    _, _, _, adm = staff_factory("admin")
    items = client.get("/v1/events", headers=lab).json()["items"]
    assert items and {it["customer_id"] for it in items} == {0}
    eid = items[0]["id"]
    assert client.get(f"/v1/events/{eid}", headers=lab).json()["customer_id"] == 0
    assert client.patch(f"/v1/events/{eid}/review", headers=lab, json={"reviewed": True}).json()["customer_id"] == 0
    assert {it["customer_id"] for it in client.get("/v1/events", headers=adm).json()["items"]} == {customer_id}
    answers = set()
    for cid in (customer_id, customer_id + 1000):
        for url, extra in (("/v1/events", {}),
                           ("/v1/events/density", {"from_utc": "2026-10-02T00:00:00Z",
                                                   "to_utc": "2026-10-04T00:00:00Z"})):
            r = client.get(url, headers=lab, params={"customer_id": cid, **extra})
            assert r.status_code == 400, r.text
            answers.add(r.text)
    assert len(answers) == 1  # one uniform refusal, whatever the id


# ---------------------------------------------------------------- I2: one consent decision for every media route

ROLES = ("admin", "support", "labeler")


def _expected(role, purpose, recordings, training):
    """The decision restated: labelers see only training-consent households (else 404) and only for training;
    support never opens training data; recordings need recordings consent, training needs training consent."""
    if role == "labeler":
        if not training:
            return 404
        return 200 if purpose == "training" else 403
    if purpose == "training":
        return 200 if role == "admin" and training else 403
    return 200 if recordings else 403


@pytest.fixture()
def media_household(client, s3client):
    from home_guard_project.cloud.s3 import S3

    b.seed_bucket(s3client)
    s3 = S3(s3client, b.BUCKET)
    client.app.state.s3 = s3
    client.app.state.clock = lambda: NOW
    for key in ("admin_cache/thumbs/1.jpg", "admin_cache/filmstrips/1.jpg"):
        b.put(s3client, key, b.FAKE_JPG)
    with session_scope(client.app.state.engine) as s:
        dev = b.index_fixture_bucket(s, s3, now=NOW)
        eid = s.scalar(select(m.Event.id).where(m.Event.stem == b.STEM))
        thumb = m.Artifact(event_id=eid, role="thumbnail", s3_key="admin_cache/thumbs/1.jpg", provenance="cloud",
                           etag="t")
        strip = m.Artifact(event_id=eid, role="filmstrip", s3_key="admin_cache/filmstrips/1.jpg",
                           provenance="cloud", etag="f")
        s.add_all([thumb, strip])
        s.flush()
        clip = s.scalar(select(m.Artifact.id).where(m.Artifact.s3_key == b.PROD_CLIP))
        return {"customer_id": dev.customer_id, "event_id": eid, "clip": clip, "filmstrip": strip.id}


def test_media_consent_is_one_decision_for_every_route(client, staff_factory, media_household):
    staff = {role: staff_factory(role)[3] for role in ROLES}
    hh = media_household
    failures = []
    for recordings in (False, True):
        for training in (False, True):
            with session_scope(client.app.state.engine) as s:
                c = s.get(m.Customer, hh["customer_id"])
                c.consent_recordings, c.consent_training = recordings, training
            for role in ROLES:
                h = staff[role]
                for purpose in ("review", "support", "training"):
                    for what in ("clip", "filmstrip"):
                        got = client.post(f"/v1/artifacts/{hh[what]}/access", json={"purpose": purpose},
                                          headers=h).status_code
                        want = _expected(role, purpose, recordings, training)
                        if got != want:
                            failures.append((what, role, purpose, recordings, training, got, want))
                purpose = "training" if role == "labeler" else "review"
                want = _expected(role, purpose, recordings, training)
                got = client.get(f"/v1/events/{hh['event_id']}/thumbnail", headers=h,
                                 follow_redirects=False).status_code
                if got != (307 if want == 200 else want):
                    failures.append(("thumbnail", role, recordings, training, got, want))
                page = client.get("/v1/events", headers=h)
                if page.status_code == 200:
                    url = {it["id"]: it["thumbnail_url"] for it in page.json()["items"]}.get(hh["event_id"])
                    if (url is not None) != (want == 200):
                        failures.append(("thumbnail_url", role, recordings, training, url, want))
    assert not failures, json.dumps(failures, indent=1)
