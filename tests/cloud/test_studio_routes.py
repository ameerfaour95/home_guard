"""Task 2c: event totals, collection items, export preview, selection and split functions."""
from datetime import datetime, timezone
from types import SimpleNamespace

from home_guard_project.cloud import models as m
from home_guard_project.cloud import studio
from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.schemas import ExportRequest

from .test_event_routes import _all_ids, indexed, indexed_consent, s3client  # noqa: F401

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _collection(client, event_ids):
    with session_scope(client.app.state.engine) as s:
        col = m.Collection(name="c", created_at=NOW)
        s.add(col)
        s.flush()
        for i in event_ids:
            s.add(m.CollectionItem(collection_id=col.id, event_id=i, added_at=NOW))
        return col.id


def _req(cid, **kw):
    return ExportRequest(collection_id=cid, name="v1", formats=kw.pop("formats", ["yolo", "clips"]), **kw)


def test_with_total_and_collection_filter(client, staff_factory, indexed):
    _, _, _, h = staff_factory("admin")
    ids = sorted(_all_ids(client))
    assert client.get("/v1/events", headers=h).json()["total"] is None
    body = client.get("/v1/events", params={"with_total": "true", "limit": 2}, headers=h).json()
    assert body["total"] == len(ids) and body["total_capped"] is False and len(body["items"]) == 2
    cid = _collection(client, ids[:2])
    body = client.get("/v1/events", params={"collection_id": cid, "with_total": "true"}, headers=h).json()
    assert sorted(i["id"] for i in body["items"]) == ids[:2] and body["total"] == 2


def test_total_is_capped(client, staff_factory, indexed, monkeypatch):
    from home_guard_project.cloud.routes import events

    monkeypatch.setattr(events, "TOTAL_CAP", 3)
    _, _, _, h = staff_factory("admin")
    body = client.get("/v1/events", params={"with_total": "true"}, headers=h).json()
    assert body["total"] == 3 and body["total_capped"] is True


def test_collection_items_roles_paging_and_pseudonyms(client, staff_factory, indexed_consent):
    ids = sorted(_all_ids(client))
    cid = _collection(client, ids)
    _, _, _, admin = staff_factory("admin")
    _, _, _, lab = staff_factory("labeler")
    _, _, _, sup = staff_factory("support")
    assert client.get(f"/v1/studio/collections/{cid}/items", headers=sup).status_code == 403
    assert client.get(f"/v1/studio/collections/{cid}/items").status_code == 401
    assert client.get("/v1/studio/collections/9999/items", headers=admin).status_code == 404
    seen, cursor = [], None
    while True:
        r = client.get(f"/v1/studio/collections/{cid}/items",
                       params={"limit": 2, **({"cursor": cursor} if cursor else {})}, headers=lab)
        assert r.status_code == 200, r.text
        page = r.json()
        seen += page["items"]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert sorted(i["id"] for i in seen) == ids
    assert all(i["customer_name"].startswith("customer-") and i["camera"].startswith("cam-") for i in seen)
    adm = client.get(f"/v1/studio/collections/{cid}/items", headers=admin).json()["items"]
    assert adm[0]["customer_name"] == "Acme"
    assert [i["id"] for i in adm] == [i["id"] for i in client.get(
        "/v1/events", params={"collection_id": cid}, headers=admin).json()["items"]]


def test_collection_items_hide_non_consenting_from_labeler(client, staff_factory, indexed):
    cid = _collection(client, sorted(_all_ids(client)))
    _, _, _, lab = staff_factory("labeler")
    assert client.get(f"/v1/studio/collections/{cid}/items", headers=lab).json()["items"] == []


def _flag(client, event_id, **comp):
    with session_scope(client.app.state.engine) as s:
        ev = s.get(m.Event, event_id)
        ev.completeness = {**ev.completeness, **comp}


def test_preview_reasons_and_formats(client, staff_factory, indexed_consent):
    ids = sorted(_all_ids(client))
    a, bad_video, expired, fallback = ids[:4]
    _flag(client, a, video=True, ai="real", expired=False)
    _flag(client, bad_video, video=False, ai="real", expired=False)
    _flag(client, expired, video=True, ai="real", expired=True)
    _flag(client, fallback, video=True, ai="fallback", expired=False)
    cid = _collection(client, [a, bad_video, expired, fallback])
    _, _, _, h = staff_factory("labeler")
    r = client.post("/v1/studio/exports/preview", headers=h,
                    json={"collection_id": cid, "name": "v1", "formats": ["yolo", "clips"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["included_ids"] == [a, fallback]
    assert {(e["event_id"], e["reason"]) for e in body["excluded"]} == {
        (bad_video, "video_unavailable"), (expired, "expired")}
    assert sum(body["split_counts"].values()) == 2 and body["groups"] >= 1
    r = client.post("/v1/studio/exports/preview", headers=h,
                    json={"collection_id": cid, "name": "v1", "formats": ["vlm_jsonl"]}).json()
    assert r["included_ids"] == [a]
    assert (fallback, "no_real_ai") in {(e["event_id"], e["reason"]) for e in r["excluded"]}
    r = client.post("/v1/studio/exports/preview", headers=h,
                    json={"collection_id": cid, "name": "v1", "formats": ["vlm_jsonl"],
                          "include_fallback_ai": True}).json()
    assert r["included_ids"] == [a, fallback]
    _, _, _, sup = staff_factory("support")
    assert client.post("/v1/studio/exports/preview", headers=sup, json={
        "collection_id": cid, "name": "v1", "formats": ["yolo"]}).status_code == 403
    assert client.post("/v1/studio/exports/preview", headers=h, json={
        "collection_id": 9999, "name": "v1", "formats": ["yolo"]}).status_code == 404


def test_preview_excludes_without_consent(client, staff_factory, indexed):
    ids = sorted(_all_ids(client))
    for i in ids:
        _flag(client, i, video=True, ai="real", expired=False)
    cid = _collection(client, ids)
    _, _, _, h = staff_factory("admin")
    body = client.post("/v1/studio/exports/preview", headers=h,
                       json={"collection_id": cid, "name": "v1", "formats": ["yolo"]}).json()
    assert body["included_ids"] == [] and {e["reason"] for e in body["excluded"]} == {"no_training_consent"}
    assert body["warnings"]


def test_select_export_items_function(client, indexed_consent):
    ids = sorted(_all_ids(client))
    _flag(client, ids[0], video=True, ai="failed", expired=False)
    cid = _collection(client, ids[:1])
    with session_scope(client.app.state.engine) as s:
        inc, exc = studio.select_export_items(s, cid, _req(cid, formats=["vlm_jsonl"]))
        assert inc == [] and exc[0].reason == "no_real_ai"
        inc, exc = studio.select_export_items(s, cid, _req(cid, formats=["vlm_jsonl"], include_fallback_ai=True))
        assert [e.id for e in inc] == [ids[0]] and exc == []


def test_assign_splits_group_aware_deterministic_and_proportional():
    split = {"train": 0.8, "val": 0.1, "test": 0.1}
    items = []
    for g in range(200):
        for k in range(3):
            items.append(SimpleNamespace(id=g * 10 + k, site="s" if g % 2 else "t", day=f"2026-{g // 28 + 1:02d}-{g % 28 + 1:02d}",
                                         start_ts=0.0))
    out = studio.assign_splits(items, "v1", split, secret="s")
    assert out == studio.assign_splits(list(reversed(items)), "v1", split, secret="s")
    by_group = {}
    for e in items:
        by_group.setdefault((e.site, e.day), set()).add(out[e.id])
    assert all(len(v) == 1 for v in by_group.values())
    share = sum(1 for v in out.values() if v == "train") / len(out)
    assert 0.65 < share < 0.95
    assert set(out.values()) <= {"train", "val", "test"}
    assert studio.assign_splits(items, "v2", split, secret="s") != out
    assert set(studio.assign_splits(items, "v1", {"train": 1.0}, secret="s").values()) == {"train"}
    assert studio.group_count(items) == len(by_group)
