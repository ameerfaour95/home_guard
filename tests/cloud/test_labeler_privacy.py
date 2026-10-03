"""Fix round for Task 10 (Codex review): labelers must never learn which household a clip comes from.

The seeded household is deliberately distinctive: site `bian`, camera `bian_ch2` shown to the owner as
`BianHouse`, customer `Daniel Levi`, Tailscale host `bian-box`. Its summaries, alert reasons, AI output, prompt,
artifact details and storage keys all carry those names. Every labeler response is serialised and searched for
them (case-insensitive, with the usual spelling variants).
"""
import json

import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, s3client  # noqa: F401  (fixture)

SITE = "bian"
CAMERA = "bian_ch2"
DISPLAY = "BianHouse"
CUSTOMER = "Daniel Levi"
HOST = "bian-box.tail9c3.ts.net"
LEAKY = ("Person at BianHouse near the bian_ch2 gate, walking to Daniel Levi's car past MyBIANHome and BIANHome, "
         "then myBIANhome")
REASON = "Daniel Levi asked us to watch bian ch2 at BianHouse (MyBIANHome, BIANHome)"
STEM = b.STEM.replace("front_side", CAMERA)

IDENTITY = ["bian", "bian_ch2", "bian ch2", "bianch2", "bianhouse", "bian house", "daniel levi", "daniellevi",
            "daniel", "levi", "bian-box", "tail9c3", "back_door", "back door", "left_side_1", "mybianhome", "bianhome"]
STORAGE = ["dataset_", "production_", "admin_cache", "clips/", "clips\\\\", "-100100", "test message"]


def _assert_clean(where: str, payload, extra=STORAGE) -> None:
    text = (payload if isinstance(payload, str) else json.dumps(payload)).lower()
    for term in IDENTITY + list(extra):
        at = text.find(term)
        assert at < 0, f"{where} leaks {term!r}: ...{text[max(0, at - 60):at + 60]}..."


def _leaky_meta(body: bytes) -> bytes:
    meta = json.loads(body)
    alert = meta.get("alert")
    if isinstance(alert, dict):
        alert["summary"] = LEAKY
        alert["alert_reason"] = REASON
    if isinstance(meta.get("model_response"), dict):
        meta["model_response"]["summary"] = LEAKY
        meta["model_response"]["where"] = f"{DISPLAY} / {CAMERA}"
    teacher = meta.get("teacher")
    if isinstance(teacher, dict) and isinstance(teacher.get("prompt"), str):
        teacher["prompt"] += f" The owner is {CUSTOMER} of {DISPLAY} (site {SITE})."
    return json.dumps(meta).encode("utf-8")


def _bian_objects() -> dict[str, bytes]:
    out = {}
    for key, body in b.seed_objects().items():
        new_key = key.replace("_test/", f"_{SITE}/").replace("front_side", CAMERA)
        if new_key.endswith((".meta.json", ".feedback.json")):
            body = body.replace(b"front_side", CAMERA.encode())
        if new_key.endswith(".meta.json") and CAMERA in new_key:
            body = _leaky_meta(body)
        out[new_key] = body
    return out


@pytest.fixture()
def household(client, s3client):  # noqa: F811
    """Index the `bian` household (training consent given); returns {event stem: id} plus ids."""
    from home_guard_project.cloud.s3 import S3

    s3client.create_bucket(Bucket=b.BUCKET)
    for key, body in _bian_objects().items():
        b.put(s3client, key, body)
    b.put(s3client, f"admin_cache/{SITE}/{CAMERA}/thumb.jpg", b.FAKE_JPG)
    s3 = S3(s3client, b.BUCKET)
    client.app.state.s3 = s3
    client.app.state.clock = lambda: NOW
    with session_scope(client.app.state.engine) as s:
        dev = b.index_fixture_bucket(s, s3, site=SITE, customer_name=CUSTOMER, consent_training=True, now=NOW)
        dev = s.get(m.Device, dev.id)
        dev.tailscale_host = HOST
        s.add(m.Camera(device_pk=dev.id, name=CAMERA, display_name=DISPLAY))
        eid = s.scalar(select(m.Event.id).where(m.Event.stem == STEM))
        assert eid is not None
        ev = s.get(m.Event, eid)
        ev.label = DISPLAY
        thumb = m.Artifact(event_id=eid, role="thumbnail", s3_key=f"admin_cache/{SITE}/{CAMERA}/thumb.jpg",
                           provenance="cloud", etag="t1",
                           detail={"source_key": f"dataset_{SITE}/clips/{CAMERA}/x.mp4", "copy": "training",
                                   "count": 3, "owner": CUSTOMER})
        s.add(thumb)
        s.add(m.Feedback(device_pk=dev.id, event_id=eid, verdict="true_alert", action="", note=f"{CUSTOMER} here",
                         raw_text=f"{DISPLAY} is my house", source="telegram", s3_key="production_bian/fb/x.json",
                         received_at=NOW))
        s.flush()
        ids = list(s.scalars(select(m.Event.id).order_by(m.Event.id)))
        return {"event_id": eid, "thumb_id": thumb.id, "ids": ids, "device_pk": dev.id,
                "customer_id": dev.customer_id}


def test_no_labeler_response_names_the_household(client, staff_factory, household):
    _, _, _, admin = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    eid, ids = household["event_id"], household["ids"]

    # the household really is identifiable in what admins see (otherwise this test proves nothing)
    seen_by_admin = client.get(f"/v1/events/{eid}", headers=admin).text
    assert CUSTOMER in seen_by_admin and DISPLAY in seen_by_admin and CAMERA in seen_by_admin

    responses = []

    def get(where, url, **kw):
        r = client.get(url, headers=lab, **kw)
        assert r.status_code == 200, f"{where}: {r.status_code} {r.text}"
        responses.append((where, r.text))
        return r.json()

    page = get("list", "/v1/events", params={"limit": 500, "with_total": True})
    assert sorted(it["id"] for it in page["items"]) == ids
    get("list paged", "/v1/events", params={"limit": 1})
    r = client.get("/v1/events", params={"q": "walking"}, headers=lab)  # labelers cannot search at all
    assert r.status_code == 400
    responses.append(("search", r.text))
    for i in ids:
        get(f"detail {i}", f"/v1/events/{i}")
        get(f"detections {i}", f"/v1/events/{i}/detections")
    get("density", "/v1/events/density", params={"from_utc": "2026-10-02T00:00:00Z",
                                                  "to_utc": "2026-10-04T00:00:00Z"})
    get("density daily", "/v1/events/density", params={"from_utc": "2026-10-02T00:00:00Z",
                                                        "to_utc": "2026-10-04T00:00:00Z", "bucket": "day"})
    get("review-count", "/v1/events/review-count")
    r = client.patch(f"/v1/events/{eid}/review", json={"flagged": True}, headers=lab)
    assert r.status_code == 200
    responses.append(("review", r.text))
    with session_scope(client.app.state.engine) as s:
        col = m.Collection(name="privacy", created_at=NOW)
        s.add(col)
        s.flush()
        for i in ids:
            s.add(m.CollectionItem(collection_id=col.id, event_id=i, added_at=NOW))
        cid = col.id
    get("collection items", f"/v1/studio/collections/{cid}/items")
    get("collection items paged", f"/v1/studio/collections/{cid}/items", params={"limit": 2})
    r = client.post("/v1/studio/exports/preview", headers=lab,
                    json={"collection_id": cid, "name": "v1", "formats": ["yolo", "clips", "vlm_jsonl"]})
    assert r.status_code == 200
    responses.append(("export preview", r.text))

    for where, text in responses:
        _assert_clean(where, text)

    detail = client.get(f"/v1/events/{eid}", headers=lab).json()
    # what the labeler still gets: the redacted text, the AI answer and the frames to label
    assert "walking to" in detail["summary"] and "cam-" in detail["summary"] and "customer-" in detail["summary"]
    assert detail["alert_reason"] and detail["ai_runs"][0]["prompt"] is None  # labelers never get prompt text
    assert detail["ai_runs"][0]["parsed"]["summary"] == detail["summary"]
    assert set(detail["raw_meta"]) <= {"kind", "duration_sec", "fps_estimated", "frames_written", "buffer", "codec",
                                       "yolo", "model_response", "teacher"}
    assert set(detail["raw_meta"]["teacher"]) <= {"model", "prompt_version", "temperature"}
    assert detail["raw_meta"]["teacher"].get("prompt_version")
    assert detail["label"] is None  # the seeded label "BianHouse" is not one of the AI's labels
    assert all(a["s3_key"] == f"artifact-{a['id']}" for a in detail["artifacts"])
    thumb = next(a for a in detail["artifacts"] if a["id"] == household["thumb_id"])
    assert thumb["detail"] == {"copy": "training", "count": 3}
    assert all(f["note"] == "" and f["raw_text"] == "" for f in detail["feedback"])


SEARCH_DENIED = {"detail": "Search is not available for this role"}


def test_labeler_search_is_refused_uniformly(client, staff_factory, household):
    """Re-review probes: `walking OR bian` / `walking -bian` gave total=0 where an unknown word gave 2, and
    `q=tail9c3&cursor=` was 200 where an unknown host token was 400. Any labeler query is now one uniform 400."""
    _, _, _, admin = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    eid = household["event_id"]
    assert eid in [it["id"] for it in client.get("/v1/events", params={"q": "BianHouse"}, headers=admin)
                   .json()["items"]]  # admins still search
    probes = [{"q": q} for q in ("walking", "walking OR bian", "walking OR nobodyzz", "walking -bian",
                                 "walking -nobodyzz", "bian", "BianHouse", "Daniel Levi", "nobodyzz", " ")]
    probes += [{"q": "tail9c3", "cursor": ""}, {"q": "zzunknownhostzz", "cursor": ""},
               {"q": "bian", "with_total": True}, {"q": "nobodyzz", "with_total": True},
               {"q": "bian", "cursor": "!!", "with_total": True}, {"q": "nobodyzz", "filter": "nope"}]
    for params in probes:
        r = client.get("/v1/events", params=params, headers=lab)
        assert (r.status_code, r.json()) == (400, SEARCH_DENIED), params
    assert client.get("/v1/events", params={"q": ""}, headers=lab).status_code == 200  # empty q = no search


def test_labeler_media_urls_do_not_name_the_household(client, staff_factory, household, s3client):  # noqa: F811
    _, _, _, lab = staff_factory("labeler")
    _, _, _, admin = staff_factory("admin")
    eid = household["event_id"]
    with session_scope(client.app.state.engine) as s:
        clip_id = s.scalar(select(m.Artifact.id).where(m.Artifact.s3_key.like(f"dataset_{SITE}/clips/%"),
                                                       m.Artifact.event_id == eid))
    r = client.post(f"/v1/artifacts/{clip_id}/access", json={"purpose": "training"}, headers=lab)
    assert r.status_code == 200, r.text
    url = r.json()["url"]
    _assert_clean("access url", url, extra=("dataset_", "production_"))
    assert "response-content-disposition" not in url.lower()
    assert r.json()["mime"] == "video/mp4"
    # the copy is fetchable, is a real copy of the clip, and is reused on the next access
    key = url.split("?", 1)[0].split(f"/{b.BUCKET}/", 1)[-1].split(".amazonaws.com/", 1)[-1]
    assert key.startswith("admin_cache/opaque/") and key.endswith(".mp4")
    assert s3client.get_object(Bucket=b.BUCKET, Key=key)["Body"].read() == b.FAKE_MP4
    again = client.post(f"/v1/artifacts/{clip_id}/access", json={"purpose": "training"}, headers=lab).json()["url"]
    assert again.split("?", 1)[0] == url.split("?", 1)[0]
    with session_scope(client.app.state.engine) as s:
        copies = s.scalars(select(m.Artifact).where(m.Artifact.role == "opaque_copy")).all()
        assert len(copies) == 1 and copies[0].provenance == "cloud"
        assert copies[0].detail == {"source_artifact_id": clip_id} and copies[0].s3_key == key
    # the copy never shows up in what a labeler sees
    detail = client.get(f"/v1/events/{eid}", headers=lab).json()
    assert "opaque_copy" not in {a["role"] for a in detail["artifacts"]}
    # thumbnails redirect to an opaque copy too
    t = client.get(f"/v1/events/{eid}/thumbnail", headers=lab, follow_redirects=False)
    assert t.status_code == 307
    _assert_clean("thumbnail redirect", t.headers["location"], extra=("dataset_", "production_"))
    assert "admin_cache/opaque/" in t.headers["location"]
    # admins keep the real key
    ra = client.post(f"/v1/artifacts/{clip_id}/access", json={"purpose": "training"}, headers=admin).json()
    assert f"dataset_{SITE}/clips/" in ra["url"]
    # owner documents are not media: a labeler may not open them at all (the generic 404 of a missing artifact)
    with session_scope(client.app.state.engine) as s:
        meta_id = s.scalar(select(m.Artifact.id).where(m.Artifact.role == "meta", m.Artifact.event_id == eid))
    r = client.post(f"/v1/artifacts/{meta_id}/access", json={"purpose": "training"}, headers=lab)
    assert (r.status_code, r.json()) == (404, {"detail": "Artifact not found"})


# ---------------------------------------------------------------- fix round 2 (Codex re-review)

@pytest.fixture()
def hidden(client, household, s3client):  # noqa: F811
    """A second household (site `test`, customer `Acme`) WITHOUT training consent, next to `bian`."""
    for key, body in b.seed_objects().items():
        b.put(s3client, key, body)
    with session_scope(client.app.state.engine) as s:
        dev = b.index_fixture_bucket(s, client.app.state.s3, consent_training=False, now=NOW)
        ids = list(s.scalars(select(m.Event.id).where(m.Event.device_pk == dev.id).order_by(m.Event.id)))
        arts = s.scalars(select(m.Artifact).join(m.Event, m.Event.id == m.Artifact.event_id)
                         .where(m.Event.device_pk == dev.id, m.Artifact.role == "original_video")
                         .order_by(m.Artifact.id)).all()
        available, gone = arts[0], arts[1]
        gone.available = False
        return {"event_ids": ids, "available_artifact": available.id, "unavailable_artifact": gone.id}


class _Statements:
    """Every SQL statement the app runs while active."""

    def __init__(self, engine):
        from sqlalchemy import event

        self.engine, self.seen, self._event = engine, [], event

    def _record(self, conn, cursor, statement, *args):
        self.seen.append(" ".join(statement.lower().split()) + " ")

    def __enter__(self):
        self._event.listen(self.engine, "before_cursor_execute", self._record)
        return self

    def __exit__(self, *exc):
        self._event.remove(self.engine, "before_cursor_execute", self._record)

    def touched(self, *tables) -> list[str]:
        return [st for st in self.seen if any(f" {t} " in st for t in tables)]


def test_labeler_search_is_refused_before_any_event_query(client, staff_factory, household):
    _, _, _, lab = staff_factory("labeler")
    for params in ({"q": "bian", "with_total": True}, {"q": "nobodyzz", "cursor": "", "site": "x", "camera": "cam-1"}):
        with _Statements(client.app.state.engine) as st:
            assert client.get("/v1/events", params=params, headers=lab).status_code == 400
        assert st.seen, "the recorder saw nothing (auth queries the staff table)"
        assert st.touched("events", "customers", "devices", "cameras") == [], params


def test_cursor_and_filter_are_validated_before_any_count_or_lookup(client, staff_factory, household):
    """Re-review finding 5: with_total counted before the cursor was decoded."""
    _, _, _, admin = staff_factory("admin")
    for params in ({"with_total": True, "cursor": ""}, {"with_total": True, "cursor": "!!"},
                   {"with_total": True, "site": "bian", "camera": "bian_ch2", "filter": "nope"},
                   {"with_total": True, "site": "bian", "cursor": "MTow"}):
        with _Statements(client.app.state.engine) as st:
            assert client.get("/v1/events", params=params, headers=admin).status_code == 400, params
        assert st.touched("events", "customers", "devices", "cameras") == [], params


def test_acronym_and_case_run_spellings_are_redacted():
    from home_guard_project.cloud import redact

    terms = ["bianhouse", "bian"]
    mapping = {"bianhouse": "cam-222222", "bian": "customer-333333"}
    for text, want in (("MyBIANHome", "Mycustomer-333333Home"), ("BIANHome", "customer-333333Home"),
                       ("myBIANhome", "mycustomer-333333home"), ("XMLBian", "XMLcustomer-333333"),
                       ("BIAN2", "customer-3333332"), ("the BIANHOUSE", "the cam-222222"),
                       ("Bianca", "Bianca"), ("BIANCA", "BIANCA"), ("Fabian", "Fabian")):
        assert redact.redact_text(text, terms, mapping) == want, text


def test_acronym_variants_never_reach_a_labeler(client, staff_factory, household):
    _, _, _, lab = staff_factory("labeler")
    eid = household["event_id"]
    with session_scope(client.app.state.engine) as s:
        run = s.scalar(select(m.AiRun).where(m.AiRun.event_id == eid))
        run.parsed = {**(run.parsed or {}), "note": "MyBIANHome BIANHome"}
        run.prompt = (run.prompt or "") + " MyBIANHome BIANHome"
    r = client.get(f"/v1/events/{eid}", headers=lab)
    assert r.status_code == 200
    _assert_clean("detail", r.text)
    d = r.json()
    assert all(run["prompt"] is None for run in d["ai_runs"])
    assert "prompt" not in d["raw_meta"].get("teacher", {})


def test_injected_enum_and_format_fields_are_not_shown_to_labelers(client, staff_factory, household):
    """Re-review: alert_command="BianHouse" and clip_start_local="Daniel Levi" came back verbatim."""
    _, _, _, lab = staff_factory("labeler")
    eid = household["event_id"]
    with session_scope(client.app.state.engine) as s:
        for ev in s.scalars(select(m.Event)):
            ev.alert_command, ev.clip_start_local, ev.label = "BianHouse", "Daniel Levi", "Daniel Levi"
    page = client.get("/v1/events", headers=lab).json()
    assert {it["alert_command"] for it in page["items"]} == {None}
    assert {it["label"] for it in page["items"]} == {None}
    d = client.get(f"/v1/events/{eid}", headers=lab)
    assert d.json()["clip_start_local"] is None and d.json()["alert_command"] is None
    _assert_clean("detail", d.text)
    with session_scope(client.app.state.engine) as s:
        ev = s.get(m.Event, eid)
        ev.alert_command, ev.label = "[call_owner]", "suspicious"
    d = client.get(f"/v1/events/{eid}", headers=lab).json()
    assert d["alert_command"] == "[call_owner]" and d["label"] == "suspicious"


def test_indexer_drops_values_outside_the_enums_and_formats(client, s3client):  # noqa: F811
    from home_guard_project.cloud.s3 import S3

    objects = _bian_objects()
    for key in list(objects):
        if key.endswith(".meta.json"):
            meta = json.loads(objects[key])
            if isinstance(meta.get("alert"), dict):
                meta["alert"]["alert_command"] = "BianHouse"
                meta["alert"]["label"] = "Daniel Levi"
            meta["clip_start_local"] = "Daniel Levi"
            objects[key] = json.dumps(meta).encode("utf-8")
    s3client.create_bucket(Bucket=b.BUCKET)
    for key, body in objects.items():
        b.put(s3client, key, body)
    with session_scope(client.app.state.engine) as s:
        b.index_fixture_bucket(s, S3(s3client, b.BUCKET), site=SITE, customer_name=CUSTOMER, consent_training=True,
                               now=NOW)
        rows = s.execute(select(m.Event.alert_command, m.Event.label, m.Event.clip_start_local)).all()
    assert rows and all(tuple(row) == (None, None, None) for row in rows), rows


def test_indexer_keeps_valid_enums_and_formats(client, s3client):  # noqa: F811
    from home_guard_project.cloud.s3 import S3

    s3client.create_bucket(Bucket=b.BUCKET)
    for key, body in _bian_objects().items():
        b.put(s3client, key, body)
    with session_scope(client.app.state.engine) as s:
        b.index_fixture_bucket(s, S3(s3client, b.BUCKET), site=SITE, customer_name=CUSTOMER, consent_training=True,
                               now=NOW)
        rows = s.execute(select(m.Event.alert_command, m.Event.label, m.Event.clip_start_local)).all()
    assert {r[0] for r in rows} - {None} <= {"[none]", "[send_message]", "[call_owner]"}
    assert any(r[0] for r in rows) and any(r[2] for r in rows)


def test_hidden_artifacts_are_indistinguishable_from_missing_ones(client, staff_factory, hidden):
    """Re-review: hidden available 403 "not agreed to training use", hidden unavailable 410, missing 404."""
    _, _, _, lab = staff_factory("labeler")
    missing = client.post("/v1/artifacts/99999999/access", json={"purpose": "training"}, headers=lab)
    assert missing.status_code == 404
    with session_scope(client.app.state.engine) as s:
        opaque = m.Artifact(role="opaque_copy", s3_key="admin_cache/opaque/x.mp4", provenance="cloud",
                            available=True, detail={"source_artifact_id": hidden["available_artifact"]})
        s.add(opaque)
        s.flush()
        opaque_id = opaque.id
    for aid in (hidden["available_artifact"], hidden["unavailable_artifact"], opaque_id, 99999998):
        for purpose in ("training", "support"):
            r = client.post(f"/v1/artifacts/{aid}/access", json={"purpose": purpose}, headers=lab)
            assert (r.status_code, r.json()) == (404, missing.json()), (aid, purpose, r.text)
    with session_scope(client.app.state.engine) as s:
        denied = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "media_denied")).all()
        targets = {row.target for row in denied}
        assert all(row.detail["reason"] for row in denied)
        hidden_keys = set(s.scalars(select(m.Artifact.s3_key).where(m.Artifact.id.in_(
            [hidden["available_artifact"], hidden["unavailable_artifact"]]))))
        assert hidden_keys <= targets  # audited server-side with the real reason, never shown to the labeler
    for eid in hidden["event_ids"]:
        for url in (f"/v1/events/{eid}", f"/v1/events/{eid}/detections", f"/v1/events/{eid}/thumbnail"):
            r = client.get(url, headers=lab, follow_redirects=False)
            assert (r.status_code, r.json()) == (404, {"detail": "Event not found"}), url


def test_export_preview_drops_hidden_events_silently(client, staff_factory, hidden, household):
    """Re-review: the preview listed hidden event 7 under `no_training_consent`."""
    _, _, _, lab = staff_factory("labeler")
    _, _, _, admin = staff_factory("admin")
    visible = household["ids"]

    def collection(name, ids):
        with session_scope(client.app.state.engine) as s:
            col = m.Collection(name=name, created_at=NOW)
            s.add(col)
            s.flush()
            for i in ids:
                s.add(m.CollectionItem(collection_id=col.id, event_id=i, added_at=NOW))
            return col.id

    mixed, alone = collection("mixed", visible + hidden["event_ids"]), collection("visible", visible)
    req = {"collection_id": mixed, "name": "v1", "formats": ["yolo", "clips"]}
    seen = client.post("/v1/studio/exports/preview", json=req, headers=lab)
    assert seen.status_code == 200
    body = seen.json()
    shown = set(body["included_ids"]) | {e["event_id"] for e in body["excluded"]}
    assert shown <= set(visible) and not shown & set(hidden["event_ids"])
    assert all(e["reason"] != "no_training_consent" for e in body["excluded"])
    # the labeler's preview is exactly the preview of the visible events alone: no count, no warning differs
    assert body == client.post("/v1/studio/exports/preview", json={**req, "collection_id": alone},
                               headers=lab).json()
    # admins still see why the hidden ones are left out
    adm = client.post("/v1/studio/exports/preview", json=req, headers=admin).json()
    assert {e["event_id"] for e in adm["excluded"] if e["reason"] == "no_training_consent"} == set(hidden["event_ids"])


def _split_bits(client, headers, cid, names):
    bits = ""
    for name in names:
        r = client.post("/v1/studio/exports/preview", headers=headers,
                        json={"collection_id": cid, "name": name, "formats": ["clips"],
                              "split": {"train": 0.5, "val": 0.5}})
        assert r.status_code == 200, r.text
        assert r.json()["included_ids"], r.json()
        bits += "1" if r.json()["split_counts"]["train"] else "0"
    return bits


def test_export_split_is_not_an_offline_site_oracle(client, staff_factory, household):
    """Re-review: with one known event and names probe0..probe15 the train bits matched sha256(name+site|day)
    computed locally for `bian` only. The group hash is now keyed by a server secret."""
    import hashlib

    _, _, _, lab = staff_factory("labeler")
    eid = household["event_id"]
    with session_scope(client.app.state.engine) as s:
        ev = s.get(m.Event, eid)
        ev.completeness = {**ev.completeness, "video": True, "expired": False, "ai": "real"}
        day = ev.day
        col = m.Collection(name="one", created_at=NOW)
        s.add(col)
        s.flush()
        s.add(m.CollectionItem(collection_id=col.id, event_id=eid, added_at=NOW))
        cid = col.id
    names = [f"probe{i}" for i in range(16)]
    observed = _split_bits(client, lab, cid, names)

    def offline(site):
        out = ""
        for name in names:
            u = int.from_bytes(hashlib.sha256((name + f"{site}|{day}").encode()).digest()[:8], "big") / 2 ** 64
            out += "1" if u < 0.5 else "0"
        return out

    candidates = ["bian", "test", "acme", "levi", "home", "house1", "site2", "north"]
    assert observed not in {offline(c) for c in candidates}, observed
    assert observed == _split_bits(client, lab, cid, names)  # still deterministic


def test_assign_splits_is_keyed_and_group_aware():
    from types import SimpleNamespace

    from home_guard_project.cloud import studio

    items = [SimpleNamespace(id=g * 10 + k, site=f"s{g % 7}", day=f"2026-10-{g % 28 + 1:02d}", start_ts=0.0)
             for g in range(64) for k in range(2)]
    split = {"train": 0.5, "val": 0.5}
    a = studio.assign_splits(items, "v1", split, secret="secret-a")
    assert a == studio.assign_splits(list(reversed(items)), "v1", split, secret="secret-a")
    assert a != studio.assign_splits(items, "v1", split, secret="secret-b")
    assert all(a[g * 10] == a[g * 10 + 1] for g in range(64))
