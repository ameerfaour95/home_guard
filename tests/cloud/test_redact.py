"""Fix round for Task 10: identity terms, redaction, the stored search copy and its backfill."""
from sqlalchemy import select, text

from home_guard_project.cloud import models as m
from home_guard_project.cloud import redact
from home_guard_project.cloud.db import session_scope

from . import builders as b
from .test_event_routes import NOW, s3client  # noqa: F401  (fixture)


def test_variants_cover_spellings_and_distinctive_parts():
    v = {x.lower() for x in redact.variants("bian_ch2")}
    assert {"bian_ch2", "bian ch2", "bian-ch2", "bianch2", "bian"} <= v and "ch2" not in v
    v = {x.lower() for x in redact.variants("BianHouse")}
    assert {"bianhouse", "bian house", "bian_house", "bian"} <= v and "house" not in v
    v = {x.lower() for x in redact.variants("Daniel Levi")}
    assert {"daniel levi", "daniellevi", "daniel_levi", "daniel", "levi"} <= v
    v = {x.lower() for x in redact.variants("front_side")}
    assert {"front_side", "front side", "frontside"} <= v and not {"front", "side"} & v
    assert "2024" not in redact.variants("cam_2024") and redact.variants("") == []


def test_redact_text_is_whole_word_ish_case_insensitive_and_single_pass():
    terms = ["bian_ch2", "bianhouse", "bian", "cam"]
    mapping = {"bian_ch2": "cam-111111", "bianhouse": "cam-222222", "bian": "customer-333333", "cam": "cam-444444"}
    out = redact.redact_text("BIANHOUSE: bian_ch2 at dataset_bian/clips, MyBianHome, Bianca, a cam", terms, mapping)
    assert out == ("cam-222222: cam-111111 at dataset_customer-333333/clips, Mycustomer-333333Home, Bianca, "
                   "a cam-444444")  # the inserted pseudonyms are not themselves redacted again
    assert redact.redact_json({"BianHouse": ["bian", 3, None]}, terms, mapping) == {
        "cam-222222": ["customer-333333", 3, None]}
    assert redact.redact_text("anything", [], {}) == "anything"


def test_identity_terms_of_a_device(client):
    with session_scope(client.app.state.engine) as s:
        dev = b.enroll(s, "bian", "Daniel Levi")
        dev.tailscale_host = "bian-box"
        dev.last_heartbeat = {"cameras": {"hb_cam9": {}}}
        s.add(m.Camera(device_pk=dev.id, name="bian_ch2", display_name="Gate of Rosenthal"))
        s.add(m.Event(device_pk=dev.id, site="bian", camera="pool_east", stem="pool_east_1_alert", start_ts=1.0))
        s.flush()
        terms = set(redact.identity_terms(s, dev))
        ident = redact.identity(s, dev, "secret-for-this-test-0123456789abcdef")
    assert {"bian", "bian_ch2", "daniel levi", "levi", "bian-box", "hb_cam9", "pool_east", "rosenthal"} <= terms
    assert "east" not in terms and "gate" not in terms
    assert ident.text("Rosenthal").startswith("cam-") and ident.text("Daniel").startswith("customer-")
    assert ident.mentions("the Bian house") and not ident.mentions("walking person")


def test_indexer_fills_the_redacted_search_copy_and_backfill_refills(client, s3client, monkeypatch):  # noqa: F811
    from home_guard_project.cloud import manage
    from home_guard_project.cloud.s3 import S3

    b.seed_bucket(s3client)
    s3 = S3(s3client, b.BUCKET)
    with session_scope(client.app.state.engine) as s:
        dev = b.index_fixture_bucket(s, s3, now=NOW)
        rows = s.execute(select(m.Event.summary, m.Event.summary_redacted)).all()
        assert rows and all(red is not None for _, red in rows)
        ev = s.scalars(select(m.Event).where(m.Event.stem == b.STEM)).one()
        ev.summary = "Person near front_side at the test site"
        ev.summary_redacted = None
        dev_id = dev.id
    monkeypatch.setenv("HG_CLOUD_DB_URL", client.app.state.engine.url.render_as_string(hide_password=False))
    assert manage.main(["redact-backfill"]) == 0
    with session_scope(client.app.state.engine) as s:
        red = s.scalar(select(m.Event.summary_redacted).where(m.Event.stem == b.STEM))
        assert red == f"Person near {redact.NEUTRAL_CAMERA} at the {redact.NEUTRAL_CUSTOMER} site"
        hit = s.scalar(select(m.Event.id).where(m.Event.stem == b.STEM, text(
            "search_redacted @@ websearch_to_tsquery('simple', 'person')")))
        assert hit is not None
        # a rename makes old copies stale; --all recomputes every event of every device
        s.add(m.Camera(device_pk=dev_id, name="pool", display_name="Person"))
    assert manage.main(["redact-backfill", "--all"]) == 0
    with session_scope(client.app.state.engine) as s:
        red = s.scalar(select(m.Event.summary_redacted).where(m.Event.stem == b.STEM))
        assert red.startswith(redact.NEUTRAL_CAMERA)
