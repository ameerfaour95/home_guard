"""Task 14: audit routes, index problems, background loops, manage CLI."""
import threading
import time
from datetime import datetime, timedelta, timezone

from home_guard_project.cloud import loops, models as m
from home_guard_project.cloud.db import session_scope

T0 = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _seed(client, n=5):
    with session_scope(client.app.state.engine) as s:
        for i in range(n):
            s.add(m.AuditLog(ts=T0 + timedelta(seconds=i), staff_id=None if i in (0, 1) else 1,
                             staff_name=f"Name {i}" if i != 1 else None,
                             action="export" if i % 2 else "mute", customer_id=7 if i < 2 else None,
                             target=f"t{i}", reason="r", detail={"k": i}))


def test_admin_pages_with_cursor_newest_first(client, staff_factory):
    _, _, _, h = staff_factory("admin")
    _seed(client, 5)
    p1 = client.get("/v1/audit", headers=h).json()
    assert [e["target"] for e in p1["items"]] == ["t4", "t3", "t2", "t1", "t0"]
    assert p1["items"][0]["detail"] == {"k": 4} and p1["items"][0]["staff"] == "Name 4"
    assert p1["items"][3]["staff"] == "deleted staff"  # no staff_id, no name
    assert p1["next_cursor"] is None


def test_cursor_walk_and_filters(client, staff_factory, monkeypatch):
    from home_guard_project.cloud.routes import audit as ra
    monkeypatch.setattr(ra, "PAGE_SIZE", 2)
    _, _, _, h = staff_factory("admin")
    _seed(client, 5)
    seen, cur = [], None
    for _ in range(5):
        r = client.get("/v1/audit", params={"cursor": cur} if cur else {}, headers=h).json()
        seen += [e["target"] for e in r["items"]]
        cur = r["next_cursor"]
        if not cur:
            break
    assert seen == ["t4", "t3", "t2", "t1", "t0"]
    assert client.get("/v1/audit", params={"cursor": "zz"}, headers=h).status_code == 400
    acts = client.get("/v1/audit", params={"action": "export"}, headers=h).json()["items"]
    assert {e["action"] for e in acts} == {"export"}
    cust = client.get("/v1/audit", params={"customer_id": 7}, headers=h).json()["items"]
    assert [e["target"] for e in cust] == ["t1", "t0"]
    by_id = client.get("/v1/audit", params={"staff": "1"}, headers=h).json()["items"]
    assert len(by_id) == 2 and all(e["target"] not in ("t0", "t1") for e in by_id)  # page size is patched to 2


def test_staff_filter_by_email_and_fallback_email(client, staff_factory):
    st, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        s.add(m.AuditLog(ts=T0, staff_id=st.id, staff_name=None, action="x", target="a", reason=""))
        s.add(m.AuditLog(ts=T0, staff_id=None, staff_name="Z", action="x", target="b", reason=""))
    items = client.get("/v1/audit", params={"staff": st.email}, headers=h).json()["items"]
    assert [e["target"] for e in items] == ["a"] and items[0]["staff"] == st.email


def test_support_and_labeler_forbidden(client, staff_factory):
    for role in ("support", "labeler"):
        _, _, _, h = staff_factory(role)
        assert client.get("/v1/audit", headers=h).status_code == 403
        assert client.get("/v1/index/problems", headers=h).status_code == 403
    assert client.get("/v1/audit").status_code == 401


def test_index_problems(client, staff_factory):
    _, _, _, h = staff_factory("admin")
    with session_scope(client.app.state.engine) as s:
        s.add(m.IndexProblem(s3_key="a", reason="bad", seen_at=T0))
        s.add(m.IndexProblem(s3_key="b", reason="worse", seen_at=T0 + timedelta(hours=1)))
    r = client.get("/v1/index/problems", headers=h).json()
    assert [p["s3_key"] for p in r] == ["b", "a"] and r[0]["seen_utc"].startswith("2026-10-03T13:00")


# ---------------------------------------------------------------- loops

def test_lifespan_without_loops_starts_no_threads(db_engine):
    from fastapi.testclient import TestClient
    from home_guard_project.cloud.app import create_app
    from home_guard_project.cloud.settings import Settings
    url = db_engine.url.render_as_string(hide_password=False)
    before = {t.name for t in threading.enumerate()}
    app = create_app(Settings.for_tests(url), s3=object(), init_db=False)
    assert app.state.settings.run_loops is False
    with TestClient(app):
        assert not [t for t in threading.enumerate() if t.name.startswith("hg-loop") and t.name not in before]
    app.state.engine.dispose()


def test_lifespan_with_loops_starts_and_stops(db_engine):
    from fastapi.testclient import TestClient
    from home_guard_project.cloud.app import create_app
    from home_guard_project.cloud.settings import Settings
    url = db_engine.url.render_as_string(hide_password=False)
    st = Settings.for_tests(url)
    st.run_loops = True
    app = create_app(st, s3=object(), init_db=False)
    with TestClient(app):
        assert [t for t in threading.enumerate() if t.name.startswith("hg-loop")]
    assert not [t for t in threading.enumerate() if t.name.startswith("hg-loop")]
    app.state.engine.dispose()


def test_loop_runs_survives_exceptions_and_stops_fast(db_engine):
    calls = []

    def job(session, s3):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    sm = loops.sessionmaker(db_engine)
    lp = loops.Loop("t1", 0.02, job, sm, None, lock_key=9001)
    lp.start()
    deadline = time.time() + 5
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.01)
    t = time.time()
    lp.stop(5)
    assert len(calls) >= 3 and time.time() - t < 2 and not lp.alive()


def test_advisory_lock_excludes_second_runner(db_engine):
    inside, peak, lock = [0], [0], threading.Lock()
    release = threading.Event()

    def job(session, s3):
        with lock:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        release.wait(0.3)
        with lock:
            inside[0] -= 1

    sm = loops.sessionmaker(db_engine)
    a = loops.Loop("a", 0.01, job, sm, None, lock_key=9002)
    b = loops.Loop("b", 0.01, job, sm, None, lock_key=9002)
    a.start(); b.start()
    time.sleep(1.0)
    a.stop(5); b.stop(5)
    assert peak[0] == 1 and a.runs + b.runs >= 1
    assert a.skipped + b.skipped >= 1


def test_different_lock_keys_run_concurrently(db_engine):
    started = threading.Event()
    both = threading.Barrier(2, timeout=3)

    def job(session, s3):
        both.wait()
        started.set()

    sm = loops.sessionmaker(db_engine)
    a = loops.Loop("a", 0.01, job, sm, None, lock_key=9003)
    b = loops.Loop("b", 0.01, job, sm, None, lock_key=9004)
    a.start(); b.start()
    assert started.wait(4)
    a.stop(5); b.stop(5)


def test_manage_media_once_and_index_once(db_engine, monkeypatch, capsys):
    from home_guard_project.cloud import manage, indexer, media
    monkeypatch.setenv("HG_CLOUD_DB_URL", db_engine.url.render_as_string(hide_password=False))
    monkeypatch.setattr(manage, "_s3", lambda: object())
    monkeypatch.setattr(indexer, "index_all", lambda s, s3, full_scan=False: {"siteA": "stats-A"})
    monkeypatch.setattr(media, "process_pending", lambda s, s3, limit=50: 3)
    assert manage.main(["index-once"]) == 0
    assert "siteA: stats-A" in capsys.readouterr().out
    assert manage.main(["media-once"]) == 0
    assert "3" in capsys.readouterr().out
