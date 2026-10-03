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
LEAKY = "Person at BianHouse near the bian_ch2 gate, walking to Daniel Levi's car"
REASON = "Daniel Levi asked us to watch bian ch2 at BianHouse"
STEM = b.STEM.replace("front_side", CAMERA)

IDENTITY = ["bian", "bian_ch2", "bian ch2", "bianch2", "bianhouse", "bian house", "daniel levi", "daniellevi",
            "daniel", "levi", "bian-box", "tail9c3", "back_door", "back door", "left_side_1"]
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
    get("search", "/v1/events", params={"q": "walking"})
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
    assert detail["alert_reason"] and detail["ai_runs"][0]["prompt"].startswith("You are the eyes")
    assert detail["ai_runs"][0]["parsed"]["summary"] == detail["summary"]
    assert set(detail["raw_meta"]) <= {"kind", "duration_sec", "fps_estimated", "frames_written", "buffer", "codec",
                                       "yolo", "model_response", "teacher"}
    assert set(detail["raw_meta"]["teacher"]) <= {"model", "prompt_version", "temperature", "prompt"}
    assert all(a["s3_key"] == f"artifact-{a['id']}" for a in detail["artifacts"])
    thumb = next(a for a in detail["artifacts"] if a["id"] == household["thumb_id"])
    assert thumb["detail"] == {"copy": "training", "count": 3}
    assert all(f["note"] == "" and f["raw_text"] == "" for f in detail["feedback"])


def test_labeler_search_has_no_identity_oracle(client, staff_factory, household):
    _, _, _, admin = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    eid = household["event_id"]

    def found(h, q):
        r = client.get("/v1/events", params={"q": q}, headers=h)
        assert r.status_code == 200, r.text
        return [it["id"] for it in r.json()["items"]]

    assert eid in found(admin, "BianHouse") and eid in found(admin, "Daniel")
    for q in ("BianHouse", "bian", "Bian House", "bian_ch2", "Daniel", "levi", "Daniel Levi", "bian-box",
              "walking BianHouse", "back_door"):
        assert found(lab, q) == [], q
    assert eid in found(lab, "walking")  # ordinary words still search the redacted text


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
    # owner documents are not media: a labeler may not open them at all
    with session_scope(client.app.state.engine) as s:
        meta_id = s.scalar(select(m.Artifact.id).where(m.Artifact.role == "meta", m.Artifact.event_id == eid))
    assert client.post(f"/v1/artifacts/{meta_id}/access", json={"purpose": "training"},
                       headers=lab).status_code == 403
