from home_guard_project.cloud.app import create_app
from home_guard_project.cloud.settings import Settings


def test_openapi_has_every_contract_route():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    paths = app.openapi()["paths"]
    for p in ["/v1/auth/login", "/v1/auth/local", "/v1/fleet", "/v1/events", "/v1/events/{event_id}", "/v1/events/{event_id}/detections",
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


def test_contract_amendment_2d_device_summary_fields():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    schemas = app.openapi()["components"]["schemas"]
    props = schemas["DeviceSummary"]["properties"]
    assert props["needs_details"]["default"] is False
    assert props["enrolled_by"]["enum"] == ["admin", "setup", "discovered"] and props["enrolled_by"]["default"] == "admin"
    assert "app_version" not in schemas["DeviceSummary"]["required"] and "needs_details" not in schemas["DeviceSummary"]["required"]
    assert "owner_phone" not in schemas["CustomerOut"]["properties"]


def test_contract_amendment_2e_consent_proposal():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    schemas = app.openapi()["components"]["schemas"]
    assert set(schemas["ConsentProposal"]["properties"]) == {"live", "recordings", "training", "recorded_utc", "installer"}
    assert set(schemas["ConsentProposal"]["required"]) == {"live", "recordings", "training", "recorded_utc", "installer"}
    prop = schemas["CustomerOut"]["properties"]["consent_proposed"]
    assert any(o.get("$ref", "").endswith("/ConsentProposal") for o in prop["anyOf"])
    assert "consent_proposed" not in schemas["CustomerOut"].get("required", [])
    assert "consent_proposed" not in schemas["CustomerIn"]["properties"]


def test_contract_amendment_2f_annotation_contract():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    doc = app.openapi()
    paths, schemas = doc["paths"], doc["components"]["schemas"]
    base = "/v1/events/{event_id}/annotation"
    assert set(paths[base]) == {"get", "put"}
    assert "post" in paths[base + "/review"] and "get" in paths[base + "/history"]
    assert set(schemas["Keyframe"]["properties"]) == {"frame", "t_sec", "xyxy", "enabled"}
    assert schemas["Keyframe"]["properties"]["enabled"]["default"] is True
    assert schemas["Track"]["properties"]["source"]["enum"] == ["human", "suggestion"]
    assert schemas["AnnotationIn"]["properties"]["status"]["enum"] == ["edited", "submitted"]
    assert set(schemas["AnnotationIn"]["required"]) == {"base_version", "tracks", "description", "status"}
    assert schemas["AnnotationOut"]["properties"]["status"]["enum"] == [
        "new", "edited", "submitted", "reviewed", "rejected"]
    assert {"suggestions_used", "fps", "frame_count", "frame_size", "review_note", "review_frame", "ai_status",
            "ai_model", "ai_prompt_version"} <= set(schemas["AnnotationOut"]["properties"])
    assert schemas["ReviewDecision"]["properties"]["decision"]["enum"] == ["accept", "reject"]
    assert set(schemas["AnnotationVersion"]["properties"]) == {
        "version", "status", "author", "created_utc", "tracks_count", "description_changed"}
    assert "annotation_status" in schemas["EventSummary"]["properties"]
    assert "annotation_status" not in schemas["EventSummary"]["required"]


def test_contract_2f_publish_batch():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    doc = app.openapi()
    schemas = doc["components"]["schemas"]
    assert "post" in doc["paths"]["/v1/studio/collections/{collection_id}/publish"]
    assert "get" in doc["paths"]["/v1/studio/publishes"]
    assert schemas["PublishRequest"]["properties"]["batch_name"]["pattern"] == "^[a-z0-9_]+$"
    assert set(schemas["PublishMissing"]["properties"]) == {"event_id", "reason"}
    assert schemas["PublishOut"]["properties"]["state"]["enum"] == ["queued", "running", "ready", "failed", "partial"]
    assert set(schemas["PublishOut"]["properties"]) == {
        "batch_name", "s3_prefix", "state", "tasks", "yolo_frames", "vlm_lines", "missing", "created_utc", "created_by"}
