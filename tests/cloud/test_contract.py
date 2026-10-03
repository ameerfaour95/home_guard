from home_guard_project.cloud.app import create_app
from home_guard_project.cloud.settings import Settings


def test_openapi_has_every_contract_route():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    paths = app.openapi()["paths"]
    for p in ["/v1/auth/login", "/v1/fleet", "/v1/events", "/v1/events/{event_id}", "/v1/events/{event_id}/detections",
              "/v1/artifacts/{artifact_id}/access", "/v1/studio/filters", "/v1/studio/collections",
              "/v1/studio/exports", "/v1/audit", "/v1/devices/enroll", "/v1/customers",
              "/v1/events/density", "/v1/fleet/activity", "/v1/events/review-count"]:
        assert p in paths, p
    schemas = app.openapi()["components"]["schemas"]
    assert "detail" in schemas["ArtifactOut"]["properties"]
    assert "timezone" in schemas["EventSummary"]["properties"]
    for name in ["DensityOut", "DensityRow", "ReviewCount"]:
        assert name in schemas, name
    # static routes must be declared before /events/{event_id}
    order = list(paths)
    assert order.index("/v1/events/density") < order.index("/v1/events/{event_id}")
    assert order.index("/v1/events/review-count") < order.index("/v1/events/{event_id}")


def test_committed_openapi_is_current():
    import json
    import pathlib

    from home_guard_project.cloud.export_openapi import build

    committed = json.loads(pathlib.Path("docs/admin/openapi.json").read_text(encoding="utf-8"))
    assert committed == build()


def test_contract_amendment_2_additions():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    doc = app.openapi()
    schemas = doc["components"]["schemas"]
    assert {"total", "total_capped"} <= set(schemas["EventPage"]["properties"])
    assert "detail" in schemas["AuditEntry"]["properties"]
    assert set(schemas["ExportPreview"]["properties"]) == {"included_ids", "excluded", "split_counts", "groups",
                                                           "warnings"}
    assert schemas["ExportExclusion"]["properties"]["reason"]["enum"] == [
        "no_training_consent", "video_unavailable", "no_real_ai", "expired"]
    assert "get" in doc["paths"]["/v1/studio/collections/{collection_id}/items"]
    assert "/v1/studio/exports/preview" in doc["paths"]
    names = [p["name"] for p in doc["paths"]["/v1/events"]["get"]["parameters"]]
    assert "with_total" in names and "collection_id" in names
