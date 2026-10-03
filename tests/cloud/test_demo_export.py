"""Task 15: the demo data for the desktop exe is produced by the real routes and validates against the contract."""
import json
import re
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from home_guard_project.cloud import schemas as S

REPO = Path(__file__).resolve().parents[2]
COMMITTED = REPO / "docs" / "admin" / "demo"

# file name (without ".labeler" and ".json") -> response schema
SCHEMAS = [
    (r"me", S.StaffOut),
    (r"customers", list[S.CustomerOut]),
    (r"customer_\d+", S.CustomerOut),
    (r"fleet", S.FleetResponse),
    (r"fleet_activity(_\d+h)?", S.DensityOut),
    (r"events(_customer_\d+)?", S.EventPage),
    (r"events_density(_customer_\d+)?", S.DensityOut),
    (r"events_review_count", S.ReviewCount),
    (r"event_\d+", S.EventDetail),
    (r"detections_\d+", S.DetectionsOut),
    (r"studio_filters", list[S.SavedFilter]),
    (r"collections", list[S.CollectionOut]),
    (r"collection_\d+_items", S.EventPage),
    (r"exports", list[S.ExportOut]),
    (r"export_\d+", S.ExportOut),
    (r"export_preview", S.ExportPreview),
    (r"audit", S.AuditPage),
    (r"index_problems", list[S.IndexProblem]),
]


def schema_for(path: Path):
    stem = path.name[:-len(".json")]
    stem = stem[:-len(".labeler")] if stem.endswith(".labeler") else stem
    for pattern, schema in SCHEMAS:
        if re.fullmatch(pattern, stem):
            return TypeAdapter(schema)
    raise AssertionError(f"no schema for {path.name}")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from home_guard_project.cloud import export_demo

    out = tmp_path_factory.mktemp("demo")
    export_demo.main(out)
    return out


def load(demo, name):
    return json.loads((demo / name).read_text(encoding="utf-8"))


def test_required_files_exist_and_validate(demo):
    for required in ("fleet.json", "events.json", "studio_filters.json", "collections.json", "exports.json",
                     "audit.json", "index_problems.json", "events_density.json", "events_review_count.json",
                     "fleet_activity.json", "export_preview.json", "customers.json"):
        assert (demo / required).exists(), required
    events = load(demo, "events.json")
    assert events["items"]
    for item in events["items"]:
        assert (demo / f"event_{item['id']}.json").exists()
        assert (demo / f"detections_{item['id']}.json").exists()
    files = sorted(demo.glob("*.json"))
    assert len(files) > 100
    for path in files:
        if path.name == "index.json":
            continue
        schema_for(path).validate_python(json.loads(path.read_text(encoding="utf-8")))


def test_every_index_entry_points_at_a_file(demo):
    index = load(demo, "index.json")
    assert index["now_utc"] == "2026-10-03T12:00:00Z"
    for entry in index["files"]:
        assert (demo / entry["file"]).exists()
    assert {e["file"] for e in index["files"]} == {p.name for p in demo.glob("*.json") if p.name != "index.json"}


def test_dataset_shape(demo):
    fleet = load(demo, "fleet.json")
    assert len(fleet["devices"]) == 4
    assert {d["verdict"] for d in fleet["devices"]} == {"healthy", "warning", "critical", "offline"}
    assert len({d["customer_id"] for d in fleet["devices"]}) == 3
    assert [d["verdict"] for d in fleet["devices"]] == ["offline", "critical", "warning", "healthy"]
    events = load(demo, "events.json")
    assert len(events["items"]) == 120 and events["next_cursor"] is None
    assert len({e["camera"] for e in events["items"]}) == 6
    assert {"alert", "false_positive", "trigger", "paused"} <= {e["kind"] for e in events["items"]}
    assert {"real", "fallback", "failed", "none"} <= {e["completeness"]["ai"] for e in events["items"]}
    assert {"true_alert", "false_alarm"} <= {v for e in events["items"] for v in e["owner_verdicts"]}
    assert len(load(demo, "collections.json")) == 2
    assert [e["state"] for e in load(demo, "exports.json")] == ["ready"]
    assert len(load(demo, "audit.json")["items"]) == 30
    assert load(demo, "index_problems.json")
    stamps = sorted(e["start_utc"] for e in events["items"])
    assert stamps[0] >= "2026-10-01T12:00:00" and stamps[-1] <= "2026-10-03T12:00:00"
    assert any(load(demo, f"detections_{e['id']}.json")["frames"] for e in events["items"])
    jpgs = {p.name for p in (demo / "media").glob("*.jpg")}
    for e in events["items"]:
        if e["thumbnail_url"]:
            assert f"thumb_{e['id']}.jpg" in jpgs and f"filmstrip_{e['id']}.jpg" in jpgs
    assert len(jpgs) >= 100


def test_labeler_files_hide_the_households(demo):
    customers = load(demo, "customers.json")
    fleet = load(demo, "fleet.json")
    terms = set()
    for c in customers:
        terms.add(c["name"].lower())
        terms.update(p.lower() for p in re.split(r"\W+", c["name"]) if len(p) >= 4)
    for d in fleet["devices"]:
        terms.update({d["site"].lower(), d["device_id"].lower(), (d["host"] or "").lower()} - {""})
        terms.update(p for p in re.split(r"[_\W]+", d["site"].lower()) if len(p) >= 4)
    terms.update(n for d in fleet["devices"] for n in ("front_door", "main_entrance", "side_gate"))
    assert "cohen" in terms and "mizrahi" in terms
    labeler = sorted(demo.glob("*.labeler.json"))
    assert len(labeler) > 20
    names = {p.name for p in labeler}
    assert {"events.labeler.json", "studio_filters.labeler.json", "collections.labeler.json"} <= names
    assert "fleet.labeler.json" not in names and "audit.labeler.json" not in names
    for path in labeler:
        text = path.read_text(encoding="utf-8").lower()
        for term in terms:
            assert term not in text, f"{term!r} in {path.name}"
    admin_events = load(demo, "events.json")["items"]
    lab_events = load(demo, "events.labeler.json")["items"]
    assert 0 < len(lab_events) < len(admin_events)
    assert all(e["customer_name"].startswith("customer-") and e["camera"].startswith("cam-") for e in lab_events)


def _stable(value):
    """Drop what depends on the local ffmpeg build (encoded file sizes)."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k != "bytes"}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def test_committed_demo_data_is_current(demo):
    """Regenerate with `python -m home_guard_project.cloud.export_demo` after any route change."""
    assert COMMITTED.exists(), "docs/admin/demo has not been generated"
    names = sorted(p.name for p in demo.glob("*.json"))
    assert names == sorted(p.name for p in COMMITTED.glob("*.json"))
    for name in names:
        assert json.loads((demo / name).read_text(encoding="utf-8")) == \
            json.loads((COMMITTED / name).read_text(encoding="utf-8")), name
