from home_guard_project.cloud.app import create_app
from home_guard_project.cloud.settings import Settings


def test_openapi_has_every_contract_route():
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    paths = app.openapi()["paths"]
    for p in ["/v1/auth/login", "/v1/fleet", "/v1/events", "/v1/events/{event_id}", "/v1/events/{event_id}/detections",
              "/v1/artifacts/{artifact_id}/access", "/v1/studio/filters", "/v1/studio/collections",
              "/v1/studio/exports", "/v1/audit", "/v1/devices/enroll", "/v1/customers"]:
        assert p in paths, p


def test_committed_openapi_is_current():
    import json
    import pathlib

    from home_guard_project.cloud.export_openapi import build

    committed = json.loads(pathlib.Path("docs/admin/openapi.json").read_text(encoding="utf-8"))
    assert committed == build()
