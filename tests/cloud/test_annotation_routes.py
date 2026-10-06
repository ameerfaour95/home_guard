"""Contract 2f: in-app labeling routes -- suggestions from the weak labels, append-only versions with a 409 on a
stale base, review, history, audit, roles and labeler anonymity, EventSummary.annotation_status and the labeling
filters."""
import pytest
from sqlalchemy import select

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import _event_id, s3client  # noqa: F401
from .test_studio import _seed


@pytest.fixture()
def seeded(client, s3client):
    return _seed(client, s3client)


@pytest.fixture()
def seeded_sparse(client, s3client):
    """Only frames 0 and 2 of the collection clip have weak labels."""
    return _seed(client, s3client, all_frames=False)


def _url(event_id, tail=""):
    return f"/v1/events/{event_id}/annotation{tail}"


def _track(track_id="t1", label="person", kfs=((0, 0.0, [0.1, 0.1, 0.3, 0.5], True),
                                               (14, 2.0, [0.4, 0.1, 0.6, 0.5], True))):
    return {"track_id": track_id, "label": label, "source": "human",
            "keyframes": [{"frame": f, "t_sec": t, "xyxy": box, "enabled": en} for f, t, box, en in kfs]}


def _save(client, h, event_id, base=0, status="edited", tracks=None, description="A person walks to the door.",
          **kw):
    body = {"base_version": base, "tracks": [_track()] if tracks is None else tracks, "description": description,
            "status": status, **kw}
    return client.put(_url(event_id), headers=h, json=body)


# ---------------------------------------------------------------- GET: a clip nobody labeled yet

def test_new_annotation_starts_from_weak_label_suggestions(client, staff_factory, seeded_sparse):
    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.COLLECT_STEM)
    r = client.get(_url(eid), headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["event_id"] == eid and body["version"] == 0 and body["status"] == "new"
    assert body["fps"] == 7.0 and body["frame_count"] == 63 and body["frame_size"] == [704, 576]
    assert body["suggestions_used"] is True and body["author"] is None and body["updated_utc"] is None
    by_label = {t["label"]: t for t in body["tracks"]}
    assert set(by_label) == {"person", "dog"}  # class 4 of frame 0 is not one of the nine
    assert all(t["source"] == "yolo" for t in body["tracks"])  # preloaded and editable, nothing to accept first
    person = by_label["person"]
    assert [(k["frame"], k["enabled"]) for k in person["keyframes"]] == [(0, True), (2, True)]
    assert person["keyframes"][1]["t_sec"] == pytest.approx(2 / 7)
    assert person["keyframes"][0]["xyxy"] == pytest.approx([0.4, 0.3, 0.6, 0.7])
    dog = by_label["dog"]
    assert [(k["frame"], k["enabled"]) for k in dog["keyframes"]] == [(0, True), (2, False)]  # gone at frame 2


def test_new_annotation_description_is_the_ai_summary(client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.STEM)
    body = client.get(_url(eid), headers=h).json()
    assert body["description"].startswith("A person appears in the frame")
    assert body["ai_description"] == body["description"]
    assert body["ai_status"] == "real" and body["tracks"] == [] and body["suggestions_used"] is False


# ---------------------------------------------------------------- PUT: versions and conflicts

def test_save_appends_versions_and_refuses_a_stale_base(client, staff_factory, seeded):
    staff, _, _, h = staff_factory("labeler")
    eid = _event_id(client, b.COLLECT_STEM)
    r = _save(client, h, eid, base=0)
    assert r.status_code == 200, r.text
    one = r.json()
    from home_guard_project.cloud import pseudonym

    assert one["version"] == 1 and one["status"] == "edited"
    assert one["author"] == pseudonym.staff(client.app.state.settings.jwt_secret, staff.id)  # D5: never a name
    assert one["tracks"][0]["keyframes"][1]["xyxy"] == [0.4, 0.1, 0.6, 0.5] and one["updated_utc"]
    stale = _save(client, h, eid, base=0, description="other")
    assert stale.status_code == 409, stale.text
    r = _save(client, h, eid, base=1, status="submitted", description="Two people near the gate.")
    assert r.status_code == 200 and r.json()["version"] == 2 and r.json()["status"] == "submitted"
    assert client.get(_url(eid), headers=h).json()["description"] == "Two people near the gate."
    hist = client.get(_url(eid, "/history"), headers=h)
    assert hist.status_code == 200, hist.text
    assert [(v["version"], v["status"], v["tracks_count"], v["description_changed"]) for v in hist.json()] == [
        (2, "submitted", 1, True), (1, "edited", 1, True)]
    with session_scope(client.app.state.engine) as s:
        rows = s.scalars(select(m.AuditLog).where(m.AuditLog.action == "annotation_save")
                         .order_by(m.AuditLog.id)).all()
        assert [r.detail["version"] for r in rows] == [1, 2] and rows[0].target == f"event/{eid}"
        assert [a.version for a in s.scalars(select(m.Annotation).where(m.Annotation.event_id == eid))] == [1, 2]


def test_save_validates_tracks(client, staff_factory, seeded):
    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.COLLECT_STEM)
    bad = [
        [_track(label="unicorn")],
        [_track(kfs=((0, 0.0, [0.5, 0.1, 0.3, 0.5], True),))],  # x1 > x2
        [_track(kfs=((0, 0.0, [0.1, 0.1, 1.3, 0.5], True),))],  # outside 0..1
        [_track(kfs=((0, 0.0, [0.1, 0.1, 0.3, 0.5], True), (1, 0.0, [0.1, 0.1, 0.3, 0.5], True)))],  # same time
        [_track(kfs=((70, 60.0, [0.1, 0.1, 0.3, 0.5], True),))],  # after the clip
    ]
    for tracks in bad:
        r = _save(client, h, eid, tracks=tracks)
        assert r.status_code == 422, (tracks, r.text)
    assert client.get(_url(eid), headers=h).json()["version"] == 0  # nothing saved


def test_drop_clip_and_needs_review_are_kept(client, staff_factory, seeded):
    _, _, _, h = staff_factory("labeler")
    eid = _event_id(client, b.COLLECT_STEM)
    r = _save(client, h, eid, drop_clip=True, needs_review=True, tracks=[])
    assert r.status_code == 200 and r.json()["drop_clip"] and r.json()["needs_review"]


# ---------------------------------------------------------------- review

def test_admin_review_accept_and_reject(client, staff_factory, seeded):
    _, _, _, lab = staff_factory("labeler")
    admin, _, _, adm = staff_factory("admin")
    eid = _event_id(client, b.COLLECT_STEM)
    assert client.post(_url(eid, "/review"), headers=adm, json={"decision": "accept",
                                                                "version": 0}).status_code == 400
    _save(client, lab, eid, status="submitted")
    assert client.post(_url(eid, "/review"), headers=lab, json={"decision": "accept",
                                                                "version": 1}).status_code == 403
    r = client.post(_url(eid, "/review"), headers=adm, json={"decision": "reject", "note": "box too loose",
                                                              "frame": 12, "version": 1})
    assert r.status_code == 200, r.text
    assert (r.json()["status"], r.json()["review_note"], r.json()["review_frame"], r.json()["version"]) == (
        "rejected", "box too loose", 12, 1)
    assert client.get(_url(eid), headers=lab).json()["status"] == "rejected"
    r = _save(client, lab, eid, base=1, status="submitted")
    assert r.json()["status"] == "submitted" and r.json()["review_note"] == ""
    r = client.post(_url(eid, "/review"), headers=adm, json={"decision": "accept", "version": 2})
    assert r.json()["status"] == "reviewed" and r.json()["version"] == 2
    hist = client.get(_url(eid, "/history"), headers=adm).json()
    assert [(v["version"], v["status"]) for v in hist] == [(2, "reviewed"), (1, "rejected")]
    with session_scope(client.app.state.engine) as s:
        actions = [a.detail["decision"] for a in s.scalars(select(m.AuditLog).where(
            m.AuditLog.action == "annotation_review").order_by(m.AuditLog.id))]
    assert actions == ["reject", "accept"]


# ---------------------------------------------------------------- roles and anonymity

def test_roles(client, staff_factory, seeded):
    _, _, _, sup = staff_factory("support")
    eid = _event_id(client, b.COLLECT_STEM)
    assert client.get(_url(eid), headers=sup).status_code == 200
    assert client.get(_url(eid, "/history"), headers=sup).status_code == 200
    assert _save(client, sup, eid).status_code == 403
    assert client.post(_url(eid, "/review"), headers=sup, json={"decision": "accept", "version": 0}).status_code == 403
    assert client.get(_url(eid)).status_code == 401
    _, _, _, adm = staff_factory("admin")
    assert client.get(_url(999999), headers=adm).status_code == 404
    assert client.get(_url(2 ** 40), headers=adm).status_code == 404


def test_labeler_cannot_see_annotations_without_training_consent(client, s3client, staff_factory):
    _seed(client, s3client, consent_training=False)
    _, _, _, lab = staff_factory("labeler")
    eid = _event_id(client, b.COLLECT_STEM)
    for r in (client.get(_url(eid), headers=lab), client.get(_url(eid, "/history"), headers=lab),
              _save(client, lab, eid)):
        assert r.status_code == 404 and r.json()["detail"] == "Event not found"


def test_labeler_sees_descriptions_redacted(client, staff_factory, seeded):
    _, _, _, adm = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    eid = _event_id(client, b.COLLECT_STEM)
    assert _save(client, adm, eid, description="The Acme family dog at back_door").status_code == 200
    shown = client.get(_url(eid), headers=lab).json()
    assert "Acme" not in shown["description"] and "back_door" not in shown["description"]
    assert "customer-" in shown["description"] and "cam-" in shown["description"]
    assert client.get(_url(eid), headers=adm).json()["description"] == "The Acme family dog at back_door"


# ---------------------------------------------------------------- lists and filters

def test_summary_status_and_labeling_filters(client, staff_factory, seeded):
    _, _, _, adm = staff_factory("admin")
    collect, paused, alert = (_event_id(client, s) for s in (b.COLLECT_STEM, b.PAUSED_STEM, b.STEM))
    _save(client, adm, collect, status="submitted")
    _save(client, adm, paused, status="edited", needs_review=True)
    _save(client, adm, alert, status="submitted")
    client.post(_url(alert, "/review"), headers=adm, json={"decision": "reject", "version": 1})

    def ids(flt):
        r = client.get("/v1/events", headers=adm, params={"filter": flt, "limit": 500})
        assert r.status_code == 200, r.text
        return {i["id"] for i in r.json()["items"]}

    statuses = {i["id"]: i["annotation_status"] for i in
                client.get("/v1/events", headers=adm, params={"limit": 500}).json()["items"]}
    assert (statuses[collect], statuses[paused], statuses[alert]) == ("submitted", "edited", "rejected")
    assert sum(1 for v in statuses.values() if v is None) == len(statuses) - 3
    assert collect not in ids("needs_labeling") and {paused, alert} <= ids("needs_labeling")
    assert ids("to_review") == {collect, paused}
    assert ids("rejected") == {alert}
    assert client.get(f"/v1/events/{collect}", headers=adm).json()["annotation_status"] == "submitted"
    keys = [f["key"] for f in client.get("/v1/studio/filters", headers=adm).json()]
    assert keys[-3:] == ["needs_labeling", "to_review", "rejected"]


# ---------------------------------------------------------------- fix round D5: no staff identity reaches a labeler

def test_labeler_never_reads_a_staff_identity(client, staff_factory, seeded):
    import json
    import re

    from home_guard_project.cloud import pseudonym

    admin, _, _, adm = staff_factory("admin")
    labeler, _, _, lab = staff_factory("labeler")
    eid = _event_id(client, b.COLLECT_STEM)
    # a name-bearing track id from the client never comes back: ids are server-assigned
    r = _save(client, adm, eid, tracks=[_track(track_id=f"box by {admin.name}")], description="A person.")
    assert r.status_code == 200 and r.json()["author"] == admin.name  # admins see real names
    secret = client.app.state.settings.jwt_secret
    responses = [client.get(_url(eid), headers=lab), client.get(_url(eid, "/history"), headers=lab)]
    put = client.put(_url(eid), headers=lab, json={"base_version": 1, "tracks": responses[0].json()["tracks"],
                                                   "description": "A person.", "status": "submitted"})
    responses += [put, client.get(_url(eid, "/history"), headers=lab)]
    for resp in responses:
        assert resp.status_code == 200, resp.text
        text = json.dumps(resp.json())
        for secret_word in (admin.name, admin.email, labeler.name, labeler.email):
            assert secret_word not in text, (secret_word, text)
    assert responses[0].json()["author"] == pseudonym.staff(secret, admin.id)
    assert put.json()["author"] == pseudonym.staff(secret, labeler.id)
    assert [v["author"] for v in responses[3].json()] == [pseudonym.staff(secret, labeler.id),
                                                          pseudonym.staff(secret, admin.id)]
    assert all(re.fullmatch(r"t-\d+", t["track_id"]) for resp in responses[:1] + [put]
               for t in resp.json()["tracks"])


def test_track_ids_are_server_assigned_and_stable(client, staff_factory, seeded):
    _, _, _, h = staff_factory("labeler")
    eid = _event_id(client, b.COLLECT_STEM)
    one = _save(client, h, eid, tracks=[_track("alice"), _track("p1", label="dog")]).json()
    assert [t["track_id"] for t in one["tracks"]] == ["t-1", "t-2"]
    two = _save(client, h, eid, base=1, tracks=[_track("t-2", label="dog"), _track("t-99"), _track("bob")]).json()
    assert [t["track_id"] for t in two["tracks"]] == ["t-2", "t-3", "t-4"]  # only ids the server gave are kept
    same = _save(client, h, eid, base=2, tracks=[_track("x"), _track("x")])  # duplicates become two tracks
    assert same.status_code == 200 and [t["track_id"] for t in same.json()["tracks"]] == ["t-5", "t-6"]


# ---------------------------------------------------------------- fix round D6: a review names the version it saw

def test_review_requires_the_version_it_decides_on(client, staff_factory, seeded):
    _, _, _, lab = staff_factory("labeler")
    _, _, _, adm = staff_factory("admin")
    eid = _event_id(client, b.COLLECT_STEM)
    _save(client, lab, eid, status="submitted")
    seen = client.get(_url(eid), headers=adm).json()["version"]  # the admin opens v1
    missing = client.post(_url(eid, "/review"), headers=adm, json={"decision": "accept"})
    assert missing.status_code == 422 and missing.json()["detail"] == "version required"
    assert _save(client, lab, eid, base=1, status="submitted", description="changed").status_code == 200  # v2
    for decision in ("accept", "reject"):
        stale = client.post(_url(eid, "/review"), headers=adm, json={"decision": decision, "version": seen})
        assert stale.status_code == 409, stale.text
    assert client.get(_url(eid), headers=adm).json()["status"] == "submitted"  # nothing was approved unseen
    ok = client.post(_url(eid, "/review"), headers=adm, json={"decision": "accept", "version": 2})
    assert ok.status_code == 200 and (ok.json()["status"], ok.json()["version"]) == ("reviewed", 2)


# ---------------------------------------------------------------- fix round L1: suggestions are fast

class _CountingReads:
    """Wraps S3.get_text: counts reads and the most that ran at once (each read takes `delay` seconds)."""

    def __init__(self, real, delay=0.05):
        import threading

        self.real, self.delay, self.calls, self.now, self.peak = real, delay, 0, 0, 0
        self.lock = threading.Lock()

    def __call__(self, key, if_match=None):
        import time

        with self.lock:
            self.calls += 1
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(self.delay)
            return self.real(key, if_match=if_match)
        finally:
            with self.lock:
                self.now -= 1


def _fresh_label_cache():
    from home_guard_project.cloud.routes import events

    events.LABEL_CACHE.clear()


def test_suggestions_read_label_files_concurrently(client, staff_factory, seeded, monkeypatch):
    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.COLLECT_STEM)
    _fresh_label_cache()
    reads = _CountingReads(client.app.state.s3.get_text)
    monkeypatch.setattr(client.app.state.s3, "get_text", reads)
    r = client.get(_url(eid), headers=h)
    assert r.status_code == 200 and r.json()["tracks"], r.text
    assert reads.calls >= 8 and reads.peak > 1, (reads.calls, reads.peak)


def test_media_loop_precomputes_suggestions_and_get_reads_no_label_file(client, staff_factory, seeded, monkeypatch):
    import time

    from home_guard_project.cloud import labeling, loops, media

    _, _, _, h = staff_factory("admin")
    eid = _event_id(client, b.COLLECT_STEM)
    s3 = client.app.state.s3
    monkeypatch.setattr(media, "process_pending", lambda *a, **k: 0)  # the media part of the loop is tested elsewhere
    with session_scope(client.app.state.engine) as s:
        loops._media_job(s, s3)
        row = s.get(m.AnnotationSuggestion, eid)
        assert row is not None and row.tracks and row.sources
    expected = client.get(_url(eid), headers=h).json()["tracks"]
    _fresh_label_cache()
    reads = _CountingReads(s3.get_text)
    monkeypatch.setattr(s3, "get_text", reads)
    start = time.perf_counter()
    r = client.get(_url(eid), headers=h)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200 and r.json()["tracks"] == expected
    assert reads.calls == 0  # precomputed: no S3 read on the request path
    assert elapsed < 1.0, elapsed  # the target is < 300 ms; generous for a loaded test machine
    # a label file replaced since (the indexer recorded a new etag): the suggestions are recomputed on demand
    with session_scope(client.app.state.engine) as s:
        art = s.scalar(select(m.Artifact).where(m.Artifact.s3_key == b.YOLO_LABELS[0]))
        art.etag = "changed"
    client.get(_url(eid), headers=h)
    assert reads.calls >= 1
    assert labeling.SUGGESTION_MODEL  # unchanged model tag for predictions
