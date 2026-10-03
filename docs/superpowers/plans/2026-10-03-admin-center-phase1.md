# Admin Center — Phase 1 (read-only console) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working Home Guard Cloud service (FastAPI + Postgres) that indexes every registered box's S3 data and serves fleet health, event history, AI records, media and training exports to the HomeGuardAdmin.exe, with roles and an append-only audit log.

**Architecture:** Contract-first. `fleet_contract/` (stdlib only) parses legacy box files. `cloud/` is a FastAPI app over SQLAlchemy 2 + Postgres, with an S3 indexer, a media worker (ffmpeg) and an export builder. The API's Pydantic schemas are frozen early (Task 2) and exported as `docs/admin/openapi.json` plus demo JSON, so Codex builds the exe in parallel against `DemoBackend`.

**Tech Stack:** Python 3.12 (uv), FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, psycopg 3, pgserver (embedded Postgres for tests), boto3 + moto, argon2-cffi, pyotp, PyJWT, ffmpeg (system), pytest.

Spec: `docs/superpowers/specs/2026-10-03-admin-center-design.md`. Audit: `docs/admin/AUDIT_AND_PROPOSAL.md`.

## Global Constraints

- Work only in `C:\Users\ameer\Ameer\home_guard_admin` (branch `admin-console`). Never touch `home_guard`, `home_guard_ui*`, `home_guard_review`.
- Stage files by explicit path; `.gitignore` has `*.md` and `*.png` — force-add docs (`git add -f`).
- `home_guard_project/fleet_contract/` imports **stdlib only** (the box imports it later).
- Cloud dependencies go in a uv dependency group `cloud` in the root `pyproject.toml`; do not add them to `[project].dependencies`.
- Run tests with `uv run --group cloud pytest tests/fleet_contract tests/cloud -q`.
- Secrets never in git: JWT secret, DB URL, AWS creds come from env (`HG_CLOUD_*`). Fixtures are sanitised (chat ids `-100100`, names `Owner`).
- Phase 1 is **read-only** toward boxes; the only S3 writes are under `fleet/`, `admin_cache/`, `training_exports/`.
- Bucket `security-camera-project-v1`, region `us-east-1`. Prefixes per site: `dataset_<site>/`, `production_<site>/`.
- `alert_command` values are bracketed literals: `"[none]"`, `"[send_message]"`, `"[call_owner]"`.
- YOLO weak labels on S3 use COCO ids (0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck, 14 bird, 15 cat, 16 dog); exported training labels are contiguous 0–8 in that sorted order.
- On this laptop, any Python TLS call needs: `env -u SSLKEYLOGFILE -u PYTHONSTARTUP AWS_CA_BUNDLE=C:/Users/ameer/.homeguard/ca_bundle.pem` (antivirus interception). Tests must not touch the network (moto only).
- Roles: `admin` (all), `support` (view + config, no exports, no staff admin), `labeler` (studio + events with customer identity replaced by `customer-<short id>` and camera by `cam-<short id>`; no live, no conversation, no audit).

---

## File structure

```
home_guard_project/fleet_contract/
  __init__.py
  keys.py          parse S3 keys → KeyInfo (root, site, area, camera, day, stem, kind, ext)
  legacy.py        parse legacy .meta.json / feedback / heartbeat → ClipRecord / FeedbackRecord / Heartbeat
  health.py        heartbeat (+ now) → HealthVerdict
  classes.py       COCO id ↔ name, contiguous remap
home_guard_project/cloud/
  __init__.py
  settings.py      env config (HG_CLOUD_DB_URL, HG_CLOUD_JWT_SECRET, HG_CLOUD_BUCKET, ...)
  db.py            engine/session factory
  models.py        SQLAlchemy tables
  schemas.py       Pydantic API contract (frozen in Task 2)
  app.py           FastAPI factory create_app(settings, s3=None)
  auth.py          password/TOTP/JWT, current_staff + require_role dependencies
  audit.py         audit.record(...) + owner notice batching
  s3.py            thin S3 wrapper (list_changed, get_json, presign, put_json)
  indexer.py       S3 → raw_revisions / events / artifacts / ai_runs / feedback
  media.py         renditions + thumbnails + filmstrip (ffmpeg)
  studio.py        built-in filters, collections, export builder
  routes/          auth.py fleet.py customers.py events.py media.py studio.py audit.py
  manage.py        CLI: init-db, create-staff, enroll, index-once, serve
  export_demo.py   seed fixtures → dump demo JSON for the exe
alembic/ (under home_guard_project/cloud/migrations)
tests/fleet_contract/  tests/cloud/   (fixtures under tests/fleet_contract/fixtures/)
docs/admin/openapi.json   docs/admin/demo/*.json
```

---

### Task 1: Scaffold, dependencies, fixtures, embedded Postgres

**Files:**
- Modify: `pyproject.toml` (add `[dependency-groups] cloud = [...]`)
- Create: `home_guard_project/fleet_contract/__init__.py`, `home_guard_project/cloud/__init__.py`, `tests/fleet_contract/__init__.py`, `tests/cloud/__init__.py`, `tests/cloud/conftest.py`
- Create fixtures: `tests/fleet_contract/fixtures/` — copy from `C:\Users\ameer\AppData\Local\Temp\claude\C--Users-ameer-Ameer-home-guard\cdde5549-509e-4bc6-99f0-980d78c671d3\scratchpad\s3sample\` the files `prod_alert.meta.json`, `train_alert.meta.json`, `train_fp.meta.json`, `paused.meta.json`, `collect.meta.json`, `feedback.json`, `heartbeat.json`, and the three key listings `production_test.txt`, `production_bian.txt`, `dataset_test.txt`. Sanitise: every `chat_id` → `"-100100"`, `from.user_id` → `1`, `from.name` → `"Owner"`, truncate `teacher.prompt` to its first 200 chars, `raw_text` → `"test message"`.
- Add hand-written fixtures: `fallback.meta.json` (copy of `train_fp` with `model_response: {"summary": ""}` and `teacher.raw_path: null`), `string_response.meta.json` (copy of `collect` with `"model_response": "A person walks."`), `traversal.meta.json` (copy of `prod_alert` with `clip_path: "..\\..\\dataset_other\\x.mp4"`).

- [ ] **Step 1:** Add to `pyproject.toml`:

```toml
[dependency-groups]
cloud = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.8",
    "sqlalchemy>=2.0.30",
    "alembic>=1.13",
    "psycopg[binary]>=3.2",
    "argon2-cffi>=23.1",
    "pyotp>=2.9",
    "pyjwt>=2.9",
    "httpx>=0.27",
    "moto[s3]>=5.0",
    "pgserver>=0.1.4",
    "pytest>=8",
]
```

Run: `uv sync --group cloud` → succeeds. If `pgserver` has no Windows wheel, replace it with `testing.postgresql`-free fallback: tests run on SQLite via `HG_CLOUD_TEST_DB=sqlite` and models must avoid PG-only types except behind `JSON().with_variant(JSONB, "postgresql")`; note the choice in the commit message.

- [ ] **Step 2:** `tests/cloud/conftest.py`:

```python
import os, shutil, tempfile, pytest

@pytest.fixture(scope="session")
def pg_url():
    import pgserver
    root = tempfile.mkdtemp(prefix="hgpg_")
    srv = pgserver.get_server(root, cleanup_mode="stop")
    yield srv.get_uri()  # postgresql://... ; convert driver in db.py
    srv.cleanup()
    shutil.rmtree(root, ignore_errors=True)
```

- [ ] **Step 3:** `tests/cloud/test_smoke.py`: connect with SQLAlchemy to `pg_url` (driver `postgresql+psycopg`) and `SELECT 1` == 1. Run → PASS.
- [ ] **Step 4:** Commit: `git add pyproject.toml uv.lock home_guard_project/fleet_contract/__init__.py home_guard_project/cloud/__init__.py tests/fleet_contract tests/cloud && git commit -m "Admin cloud: scaffold, dependency group, sanitised S3 fixtures, embedded Postgres"`

---

### Task 2: API contract (Pydantic schemas) + stub app + OpenAPI export

This freezes what the exe consumes. Codex starts on the UI once this lands.

**Files:**
- Create: `home_guard_project/cloud/schemas.py`, `home_guard_project/cloud/app.py`, `home_guard_project/cloud/routes/__init__.py` and one stub module per route group, `home_guard_project/cloud/export_openapi.py`
- Create: `docs/admin/openapi.json` (generated)
- Test: `tests/cloud/test_contract.py`

**Interfaces — Produces (exact, used by every later task and by the exe):**

```python
# schemas.py
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["admin", "support", "labeler"]
Verdict = Literal["healthy", "warning", "critical", "offline", "unknown"]
EventKind = Literal["alert", "false_positive", "paused", "owner_feedback", "trigger", "random", "unknown"]
AiStatus = Literal["real", "fallback", "failed", "none"]
BoxesStatus = Literal["captured", "sampled", "recomputed", "none"]

class LoginRequest(BaseModel): email: str; password: str; totp: str
class TokenPair(BaseModel): access_token: str; refresh_token: str; expires_in: int; staff: "StaffOut"
class RefreshRequest(BaseModel): refresh_token: str
class StaffOut(BaseModel): id: int; email: str; name: str; role: Role

class HealthReason(BaseModel): code: str; message: str; severity: Verdict
class CameraHealth(BaseModel): name: str; newest_clip_utc: Optional[datetime]; clips_waiting: int = 0; stale: bool
class DeviceSummary(BaseModel):
    device_id: str; site: str; customer_id: int; customer_name: str
    verdict: Verdict; reasons: list[HealthReason]
    last_seen_utc: Optional[datetime]; mode: Optional[str]; host: Optional[str]
    cameras_total: int; cameras_stale: int; disk_free_gb: Optional[float]
    collector_running: Optional[bool]; stopped: Optional[bool]
    newest_clip_utc: Optional[datetime]; events_24h: int; alerts_24h: int; false_alarms_7d: int
class FleetResponse(BaseModel): devices: list[DeviceSummary]; generated_utc: datetime

class CustomerIn(BaseModel): name: str; timezone: str = "Asia/Jerusalem"; consent_live: bool = False; consent_recordings: bool = False; consent_training: bool = False; notes: str = ""
class CustomerOut(CustomerIn): id: int; devices: list[DeviceSummary] = []
class EnrollRequest(BaseModel): customer_id: int; site: str = Field(pattern=r"^[a-z0-9_]+$"); tailscale_host: str = ""; ssh_user: str = "ameer"

class Completeness(BaseModel):
    video: bool; boxes: BoxesStatus; ai: AiStatus; owner_feedback: bool; expired: bool; copies: list[Literal["production", "training"]]

class EventSummary(BaseModel):
    id: int; site: str; customer_id: int; customer_name: str; camera: str
    kind: EventKind; start_utc: datetime; end_utc: Optional[datetime]
    summary: str; label: Optional[str]; alert_command: Optional[str]
    detected: list[str]; owner_verdicts: list[str]; completeness: Completeness
    reviewed: bool; flagged: bool; thumbnail_url: Optional[str]
class EventPage(BaseModel): items: list[EventSummary]; next_cursor: Optional[str]

class AiRunOut(BaseModel):
    id: int; purpose: Literal["guard"]; status: AiStatus; model: Optional[str]; prompt_version: Optional[str]
    prompt: Optional[str]; parsed: Optional[dict]; raw_text_artifact_id: Optional[int]; input_frame_artifact_ids: list[int]
class FeedbackOut(BaseModel):
    id: int; verdict: str; action: str; note: str; raw_text: str; source: str; received_utc: datetime
class ArtifactOut(BaseModel): id: int; role: str; s3_key: str; bytes: Optional[int]; available: bool; provenance: str
class DispatchOut(BaseModel): channel: Optional[str]; sent: Optional[bool]; detail: dict
class EventDetail(EventSummary):
    clip_start_local: Optional[str]; duration_sec: Optional[float]; fps: Optional[float]; frame_size: Optional[list[int]]
    alert_reason: str; dispatch: Optional[DispatchOut]
    ai_runs: list[AiRunOut]; feedback: list[FeedbackOut]; artifacts: list[ArtifactOut]
    raw_meta: dict  # newest raw revision, for the "raw" tab

class Box(BaseModel): cls: int; label: str; conf: Optional[float]; xyxy: list[float]  # normalised 0..1
class FrameBoxes(BaseModel): frame_index: int; t_sec: float; status: Literal["ran", "ran_empty", "not_run"]; boxes: list[Box]
class DetectionsOut(BaseModel): provenance: BoxesStatus; model: Optional[str]; frames: list[FrameBoxes]

class MediaAccessRequest(BaseModel): purpose: Literal["review", "support", "training"]
class MediaAccess(BaseModel): url: str; expires_utc: datetime; mime: str

class ReviewUpdate(BaseModel): reviewed: Optional[bool] = None; flagged: Optional[bool] = None
class SavedFilter(BaseModel):
    key: str; title: str; description: str; builtin: bool; query: dict
class CollectionIn(BaseModel): name: str; description: str = ""
class CollectionOut(CollectionIn): id: int; event_count: int; created_by: str; created_utc: datetime
class CollectionItems(BaseModel): event_ids: list[int]
class ExportRequest(BaseModel):
    collection_id: int; name: str = Field(pattern=r"^[a-z0-9_-]+$")
    formats: list[Literal["yolo", "vlm_jsonl", "clips"]]; split: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
    include_fallback_ai: bool = False
class ExportOut(BaseModel):
    id: int; name: str; version: int; state: Literal["queued", "running", "ready", "failed", "partial"]
    item_count: int; s3_prefix: str; manifest_url: Optional[str]; error: Optional[str]; created_utc: datetime; created_by: str
class AuditEntry(BaseModel):
    id: int; ts: datetime; staff: str; action: str; customer_id: Optional[int]; device_id: Optional[str]; target: str; reason: str
class AuditPage(BaseModel): items: list[AuditEntry]; next_cursor: Optional[str]
class IndexProblem(BaseModel): s3_key: str; reason: str; seen_utc: datetime
```

Routes (all under `/v1`; every route except `/auth/login` and `/auth/refresh` requires Bearer):

| Method + path | Response | Roles |
|---|---|---|
| POST /auth/login | TokenPair | — |
| POST /auth/refresh | TokenPair | — |
| GET /me | StaffOut | all |
| GET /fleet | FleetResponse | admin, support |
| GET /customers ; POST /customers ; GET /customers/{id} ; PATCH /customers/{id} | CustomerOut | admin (write), admin+support (read) |
| POST /devices/enroll | DeviceSummary | admin |
| GET /events?site=&customer_id=&camera=&kind=&ai=&verdict=&q=&from_utc=&to_utc=&reviewed=&flagged=&filter=&cursor=&limit= | EventPage | all (labeler pseudonymised) |
| GET /events/{id} | EventDetail | all (labeler pseudonymised, `raw_meta` stripped of `dispatch`) |
| GET /events/{id}/detections | DetectionsOut | all |
| PATCH /events/{id}/review | EventSummary | all |
| POST /artifacts/{id}/access | MediaAccess | all (labeler only with consent_training) |
| GET /events/{id}/thumbnail | 307 to presigned | all |
| GET /studio/filters | list[SavedFilter] | all |
| GET/POST /studio/collections ; POST /studio/collections/{id}/items ; DELETE /studio/collections/{id}/items | CollectionOut | admin, labeler |
| GET/POST /studio/exports ; GET /studio/exports/{id} | ExportOut | admin, labeler |
| GET /audit?staff=&customer_id=&action=&cursor= | AuditPage | admin |
| GET /index/problems | list[IndexProblem] | admin |

- [ ] **Step 1:** Write `tests/cloud/test_contract.py`:

```python
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
    import json, pathlib
    from home_guard_project.cloud.export_openapi import build
    committed = json.loads(pathlib.Path("docs/admin/openapi.json").read_text(encoding="utf-8"))
    assert committed == build()
```

- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement `schemas.py` exactly as above; `settings.py` (`Settings` dataclass from env `HG_CLOUD_DB_URL`, `HG_CLOUD_JWT_SECRET`, `HG_CLOUD_BUCKET`, `HG_CLOUD_REGION`, `HG_CLOUD_ACCESS_TTL=900`, `HG_CLOUD_REFRESH_TTL=43200`; classmethod `for_tests(db_url)` with a fixed test secret); `app.py` `create_app(settings, s3=None, init_db=True) -> FastAPI` including routers with the signatures above whose bodies `raise HTTPException(501)` for now; `export_openapi.py` with `build() -> dict` (create_app with test settings, return `app.openapi()`) and `__main__` writing `docs/admin/openapi.json` (indent 2, sorted keys).
- [ ] **Step 4:** `uv run --group cloud python -m home_guard_project.cloud.export_openapi` then run tests → PASS.
- [ ] **Step 5:** Commit (force-add the json is not needed — only `*.md`/`*.png` are ignored): `git add home_guard_project/cloud tests/cloud/test_contract.py docs/admin/openapi.json && git commit -m "Admin cloud: frozen API contract and OpenAPI export"`

---

### Task 3: `fleet_contract.keys` + `fleet_contract.classes`

**Files:** Create `home_guard_project/fleet_contract/keys.py`, `classes.py`; Test `tests/fleet_contract/test_keys.py`, `test_classes.py`

**Produces:**

```python
@dataclass(frozen=True)
class KeyInfo:
    key: str; root: Literal["dataset", "production"]; site: str
    area: Literal["meta", "clips", "feedback", "status", "responses", "vlm_crops", "yolo_images", "yolo_labels", "other"]
    camera: Optional[str]; day: Optional[str]; stem: Optional[str]; kind: Optional[str]; ext: str
def parse_key(key: str) -> Optional[KeyInfo]          # None for keys outside dataset_/production_
def stem_kind(stem: str) -> tuple[str, Optional[int], Optional[str]]   # (camera, epoch, kind) from "<camera>_<epoch>_<kind>"
def normalize_rel(path: str) -> Optional[str]         # "\\"→"/", strips "./", None for absolute or ".." paths
# classes.py
COCO_NAMES: dict[int, str]   # {0:"person",1:"bicycle",2:"car",3:"motorcycle",5:"bus",7:"truck",14:"bird",15:"cat",16:"dog"}
CONTIGUOUS: dict[int, int]   # sorted COCO id → 0..8
def name_to_coco(name: str) -> Optional[int]
```

Rules: `site` is the text after `dataset_`/`production_` up to the first `/` (sites contain underscores, e.g. `ameer_batch_3`). Area from the second segment (`_status`→status; `yolo/images`→yolo_images). Camera = third segment for meta/clips/responses/vlm_crops/yolo_*; for feedback the camera segment may be `_general`. Stem = file name without `.meta.json`/`.feedback.json`/`.mp4`/`.model_raw.txt`/`_fNNNN.jpg|txt`/`_f<i>.jpg`. Kind = last `_` token of the stem when it is one of `alert|fp|paused|trigger|random` (`fp`→`false_positive`). Camera names contain underscores; `stem_kind` splits from the right.

- [ ] **Step 1:** Tests:

```python
from home_guard_project.fleet_contract.keys import parse_key, stem_kind, normalize_rel
import pathlib

def test_production_meta():
    k = parse_key("production_test/meta/left_side_1/2026-10-03/left_side_1_1791019693_alert.meta.json")
    assert (k.root, k.site, k.area, k.camera, k.day, k.stem, k.kind) == (
        "production", "test", "meta", "left_side_1", "2026-10-03", "left_side_1_1791019693_alert", "alert")

def test_site_with_underscores_and_fp():
    k = parse_key("dataset_ameer_batch_3/meta/back_door/2026-10-03/back_door_1791013299_fp.meta.json")
    assert k.site == "ameer_batch_3" and k.kind == "false_positive" and k.camera == "back_door"

def test_status_and_feedback_general():
    assert parse_key("dataset_test/_status/heartbeat.json").area == "status"
    f = parse_key("production_test/feedback/_general/2026-10-03/general_1791013449968.feedback.json")
    assert f.area == "feedback" and f.camera == "_general"

def test_yolo_label_and_crop_frames():
    k = parse_key("dataset_ameer_house/yolo/labels/back_door/2026-10-02/back_door_1790944263_trigger_f0002.txt")
    assert k.area == "yolo_labels" and k.stem == "back_door_1790944263_trigger"
    c = parse_key("dataset_test/vlm_crops/front_side/2026-10-03/front_side_1791020177_alert_f1.jpg")
    assert c.area == "vlm_crops" and c.stem == "front_side_1791020177_alert"

def test_outside_prefix():
    assert parse_key("tagging/x.json") is None and parse_key("dataset_uca") is None

def test_stem_kind():
    assert stem_kind("test_ch6_1790979739_alert") == ("test_ch6", 1790979739, "alert")
    assert stem_kind("weird") == ("weird", None, None)

def test_normalize_rel():
    assert normalize_rel("clips\\a\\2026-10-03\\x.mp4") == "clips/a/2026-10-03/x.mp4"
    assert normalize_rel("..\\..\\dataset_other\\x.mp4") is None
    assert normalize_rel("C:\\x.mp4") is None and normalize_rel("/x.mp4") is None

def test_every_fixture_listing_key_parses():
    for name in ["production_test.txt", "production_bian.txt", "dataset_test.txt"]:
        for line in pathlib.Path("tests/fleet_contract/fixtures", name).read_text().splitlines():
            key = line.split()[-1]
            assert parse_key(key) is not None, key
```

and `test_classes.py`: `CONTIGUOUS == {0:0,1:1,2:2,3:3,5:4,7:5,14:6,15:7,16:8}`, `name_to_coco("truck") == 7`, `name_to_coco("horse") is None`.

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `"Fleet contract: S3 key parsing and class maps"`.

---

### Task 4: `fleet_contract.legacy` — normalise meta / feedback / heartbeat

**Files:** Create `home_guard_project/fleet_contract/legacy.py`; Test `tests/fleet_contract/test_legacy.py`

**Produces:**

```python
@dataclass
class AiRecord:
    status: Literal["real", "fallback", "failed", "none"]
    model: Optional[str]; prompt_version: Optional[str]; prompt: Optional[str]
    parsed: Optional[dict]; raw_rel: Optional[str]; input_frames_rel: list[str]

@dataclass
class ClipRecord:
    site: str; root: Literal["dataset", "production"]; camera: str; stem: str; kind: str
    start_ts: Optional[float]; end_ts: Optional[float]; trigger_ts: Optional[int]
    clip_rel: Optional[str]; duration_sec: Optional[float]; fps: Optional[float]; frame_size: Optional[list[int]]
    codec: Optional[str]; detected: list[str]; class_max_conf: dict[str, float]
    summary: str; label: Optional[str]; alert_command: Optional[str]; alert_reason: str
    dispatch: Optional[dict]; muted: bool; paused: bool
    ai: AiRecord; sampled_frames: list[dict]   # yolo_export.exported_frames with normalised paths
    owner_feedback: list[dict]; problems: list[str]

def parse_meta(key: str, body: dict) -> ClipRecord        # never raises on odd shapes; records problems instead
def parse_feedback(key: str, body: dict) -> FeedbackRecord
@dataclass
class FeedbackRecord:
    site: str; alert_id: Optional[str]; camera: Optional[str]; verdict: str; action: str
    note: str; raw_text: str; source: str; time_utc: Optional[datetime]; scope_camera: Optional[str]
def parse_heartbeat(body: dict) -> Heartbeat
@dataclass
class Heartbeat:
    site: str; mode: Optional[str]; host: Optional[str]; time_utc: Optional[datetime]
    collector_running: Optional[bool]; stopped: Optional[bool]; disk_free_gb: Optional[float]
    newest_clip_utc: Optional[datetime]; cameras: dict[str, Optional[datetime]]; clips_outbox: int
```

AI status rules (audit A3/A7):
- no `teacher` and no `model_response` → `"none"` (production copies, paused, collection without VLM).
- `model_response` is a dict whose only non-empty value would be `summary` and `summary == ""`, and `teacher.raw_path` is null → `"fallback"`.
- `model_response` is None while `teacher` exists → `"failed"`.
- `model_response` is a string (collection VLM) → `"real"` with `parsed={"text": <string>}`.
- otherwise `"real"`.

`summary` precedence: `alert.summary` → `model_response.summary` (dict) → `model_response` (str) → `""`. `trigger_ts` from the stem epoch. `detected` = keys of `yolo.class_counts`. `problems` gets `"clip_path outside site"` when `normalize_rel(clip_path)` is None, `"missing clip_start_ts"` etc. Feedback alert camera = `body.alert.camera`; feedback `camera` field is the action scope → `scope_camera`.

- [ ] **Step 1:** Tests (one per fixture):

```python
import json, pathlib
from home_guard_project.fleet_contract.legacy import parse_meta, parse_feedback, parse_heartbeat
F = pathlib.Path("tests/fleet_contract/fixtures")
def load(n): return json.loads((F / n).read_text(encoding="utf-8"))
P = "production_test/meta/front_side/2026-10-03/front_side_1791020177_alert.meta.json"
T = "dataset_test/meta/front_side/2026-10-03/front_side_1791020177_alert.meta.json"

def test_production_alert_has_no_ai_but_has_dispatch():
    r = parse_meta(P, load("prod_alert.meta.json"))
    assert r.root == "production" and r.kind == "alert" and r.ai.status == "none"
    assert r.alert_command == "[send_message]" and r.dispatch["channel"] == "telegram"
    assert r.clip_rel == "clips/front_side/2026-10-03/front_side_1791020177_alert.mp4"
    assert r.trigger_ts == 1791020177 and r.frame_size == [704, 576]

def test_training_alert_has_real_teacher():
    r = parse_meta(T, load("train_alert.meta.json"))
    assert r.ai.status == "real" and r.ai.model == "gpt-4o"
    assert r.ai.input_frames_rel[0].startswith("vlm_crops/front_side/")
    assert r.ai.parsed["people"] == 1

def test_false_positive():
    r = parse_meta("dataset_test/meta/back_door/2026-10-03/back_door_1791013299_fp.meta.json", load("train_fp.meta.json"))
    assert r.kind == "false_positive" and r.alert_command == "[none]" and r.ai.status == "real"

def test_fallback_is_not_real():
    r = parse_meta("dataset_test/meta/back_door/2026-10-03/back_door_1791013299_fp.meta.json", load("fallback.meta.json"))
    assert r.ai.status == "fallback"

def test_paused():
    r = parse_meta("dataset_test/meta/left_side_1/2026-10-03/left_side_1_1791014883_paused.meta.json", load("paused.meta.json"))
    assert r.paused and r.muted and r.ai.status == "none" and r.detected == ["car"]

def test_collection_sampled_frames_and_conf():
    r = parse_meta("dataset_ameer_house/meta/back_door/2026-10-02/back_door_1790944263_trigger.meta.json", load("collect.meta.json"))
    assert r.kind == "trigger" and r.sampled_frames[0]["label_path"].startswith("yolo/labels/")
    assert 0.76 < r.class_max_conf["person"] < 0.77

def test_string_model_response():
    r = parse_meta("dataset_ameer_house/meta/back_door/2026-10-02/back_door_1790944263_trigger.meta.json", load("string_response.meta.json"))
    assert r.ai.status == "real" and r.summary == "A person walks."

def test_traversal_recorded_as_problem():
    r = parse_meta(P, load("traversal.meta.json"))
    assert r.clip_rel is None and "clip_path outside site" in r.problems

def test_garbage_never_raises():
    r = parse_meta(P, {"camera_name": 5, "yolo": "x", "alert": [], "model_response": 3})
    assert r.problems

def test_feedback_and_heartbeat():
    f = parse_feedback("production_test/feedback/test_ch6/2026-10-03/x.feedback.json", load("feedback.json"))
    assert f.alert_id == "test_ch6_1790979739_alert" and f.camera == "test_ch6" and f.verdict == "none"
    h = parse_heartbeat(load("heartbeat.json"))
    assert h.site == "test" and h.collector_running and h.cameras["front_side"].year == 2026
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `"Fleet contract: legacy meta/feedback/heartbeat parser with honest AI status"`.

---

### Task 5: `fleet_contract.health` — verdict from heartbeat

**Files:** Create `home_guard_project/fleet_contract/health.py`; Test `tests/fleet_contract/test_health.py`

**Produces:** `def verdict(hb: Optional[Heartbeat], now: datetime, alert_hours: Optional[tuple[int,int]] = None) -> tuple[str, list[dict]]` returning (`verdict`, reasons `[{code, message, severity}]`) and `def camera_stale(newest: Optional[datetime], now) -> bool` (stale = no clip for 24 h).

Rules (worst wins): no heartbeat → `unknown` (`no_heartbeat`); heartbeat older than 90 min → `offline` (`heartbeat_old`, message "Last heard 3 h ago"); `stopped` → `warning` (`stopped_by_owner`); `collector_running` false and not stopped → `critical` (`engine_down`); `disk_free_gb < 20` → `critical`, `< 50` → `warning`; any camera stale → `warning` (`camera_quiet`, message names cameras); `clips_outbox > 500` → `warning` (`upload_backlog`). Otherwise `healthy`.

- [ ] **Step 1:** Tests: one per rule, plus `test_worst_wins` (stopped + disk 10 → critical) and `test_reasons_are_human` (message contains camera name). Use `parse_heartbeat(load("heartbeat.json"))` and `dataclasses.replace` to vary.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Fleet contract: box health verdict"`.

---

### Task 6: Database models + migrations + manage CLI

**Files:** Create `home_guard_project/cloud/db.py`, `models.py`, `manage.py`, `migrations/` (Alembic env + baseline revision); Test `tests/cloud/test_models.py`

**Produces:** SQLAlchemy 2 declarative models exactly per spec §5 (phase-1 subset): `Staff`, `RefreshToken`, `Customer`, `Device` (`device_id` UUID string, `site`, `tailscale_host`, `ssh_user`, `customer_id`, `enrolled_at`), `Camera` (`device_pk`, `name`, `display_name`), `Event` (unique `(device_pk, camera, stem)`; columns from `EventSummary` + `start_ts`, `trigger_ts`, `completeness` JSON, `search` TSVECTOR generated from `summary` on Postgres), `Artifact` (unique `s3_key`; `event_id`, `role` in `original_video|thumbnail|rendition|filmstrip|meta|raw_answer|teacher_frame|yolo_image|yolo_label|feedback`, `etag`, `bytes`, `available`, `provenance` in `box|cloud`), `RawRevision` (`s3_key`, `etag`, `fetched_at`, `body` JSON; unique `(s3_key, etag)`), `AiRun`, `Feedback` (unique `s3_key`), `ReviewState` (`event_id`, `reviewed`, `flagged`, `by`, `at`), `Collection`, `CollectionItem`, `Export`, `AuditLog`, `IndexProblem`, `S3Cursor` (`prefix`, `last_full_scan`). `db.py`: `make_engine(url)` (rewrites `postgresql://` → `postgresql+psycopg://`), `session_scope(engine)` context manager.
`manage.py` (argparse): `init-db` (alembic upgrade head), `create-staff --email --name --role` (prints TOTP provisioning URI + generated password once), `enroll --customer NAME --site SITE [--tailscale-host H]`, `index-once`, `serve [--port 8600]`.

- [ ] **Step 1:** Tests: create all tables on `pg_url`; insert Customer→Device→Event→Artifact; unique `(device_pk,camera,stem)` violation raises `IntegrityError`; `AuditLog` rows cannot be updated: the migration creates a trigger `audit_log_no_update` that raises on UPDATE/DELETE — test that `session.execute(update(AuditLog)...)` raises.
- [ ] **Step 2–4:** fail → implement (Alembic baseline via `alembic revision --autogenerate`, then hand-add the trigger) → pass.
- [ ] **Step 5:** Commit `"Admin cloud: database models, migrations, manage CLI"`.

---

### Task 7: Auth — password + TOTP + JWT + roles

**Files:** Create `home_guard_project/cloud/auth.py`, implement `routes/auth.py`; Test `tests/cloud/test_auth.py`

**Produces:** `hash_password(pw) -> str`, `verify_password(hash, pw) -> bool` (argon2id), `issue_tokens(staff, settings) -> TokenPair`, dependency `current_staff(token) -> Staff`, dependency factory `require_role(*roles)`. Login: wrong password or wrong TOTP → 401 with the same message "Email, password or code is wrong"; 5 failures in 15 min for one email → 429. Refresh tokens stored hashed, single-use (rotation). Disabled staff → 401. Every login success/failure → `audit.record(action="login"|"login_failed")` (audit.py is introduced here with `record(session, staff_id, action, target="", reason="", customer_id=None, device_id=None, detail=None)`).

- [ ] **Step 1:** Tests with `fastapi.testclient.TestClient(create_app(Settings.for_tests(pg_url), s3=moto_s3))`: login success returns tokens; bad TOTP 401; lockout 429 on 6th try; `/v1/me` with access token OK, with expired token (freeze time via `settings.access_ttl=-1`) 401; refresh rotates (old refresh second use → 401); labeler on `/v1/fleet` → 403; audit rows exist for login and login_failed. Add fixture `staff_factory(role)` in `conftest.py` returning `(staff, password, totp_secret, auth_headers)`.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: sign-in with TOTP, rotating refresh tokens, roles, login audit"`.

---

### Task 8: S3 wrapper + indexer

**Files:** Create `home_guard_project/cloud/s3.py`, `indexer.py`; Test `tests/cloud/test_indexer.py`

**Produces:**
- `class S3:` `__init__(client, bucket)`; `list(prefix) -> Iterator[ObjInfo(key, etag, size, last_modified)]`; `get_json(key) -> dict`; `get_text(key) -> str`; `put_json(key, body)`; `presign(key, ttl=300) -> str`; `exists(key) -> bool`.
- `index_device(session, s3, device) -> IndexStats(new_events, updated_events, artifacts, feedback, problems)`; `index_all(session, s3) -> dict[site, IndexStats]`.

Algorithm per device (site S): list `dataset_S/` and `production_S/` (skip areas other than meta, clips, feedback, status, responses, vlm_crops, yolo_*). For each meta key whose `(key, etag)` is not in `raw_revisions`: fetch, store `RawRevision`, `parse_meta`, upsert `Event` keyed `(device, camera, stem)`, merging: `copies` gets `"production"`/`"training"`; teacher/AI comes from whichever copy has `ai.status != "none"` (prefer `real`); `dispatch` from the production copy; `owner_feedback` entries unioned (dedupe by `time_utc`+`verdict`). Upsert `AiRun` (one guard run per event, replaced only when a better status arrives: real > failed > fallback > none). For every non-meta key, upsert `Artifact` by `s3_key` and attach to the event by `(camera, stem)`; artifacts with no event yet are kept with `event_id NULL` and attached when the meta arrives. Feedback keys → `Feedback` rows linked by `alert_id` stem when present. `_status/heartbeat.json` → stored as RawRevision + `Device.last_heartbeat` JSON. Parse problems → `IndexProblem`. Completeness: `video` = clip artifact exists and available; `boxes` = `sampled` if `sampled_frames` and label artifacts exist else `none`; `ai`; `owner_feedback`; `expired` = production clip artifact missing and event older than 14 days; Events' `customer`/`site` from device.
Objects that disappear from a listing (only checked on a full scan, at most every 30 min per prefix) → `Artifact.available = False`.

- [ ] **Step 1:** Tests with moto: upload all fixtures under their real keys (use the key listings to place `prod_alert` at `production_test/meta/front_side/2026-10-03/front_side_1791020177_alert.meta.json`, `train_alert` at the `dataset_test` twin, plus a fake mp4 at both clip keys, three `vlm_crops/..._f{0,1,2}.jpg`, the raw answer txt, a feedback whose `alert.alert_id` is `front_side_1791020177_alert`). Assert: exactly one Event for `front_side_1791020177_alert` with `copies == ["production","training"]`, `ai == "real"`, `dispatch` present, feedback linked; re-running index creates no new rows (idempotent, `new_events == 0`); changing the training meta body (new ETag) updates the event and adds a second RawRevision; the traversal fixture lands in `IndexProblem`; deleting the production clip then forcing a full scan sets that artifact unavailable; a vlm_crop uploaded before its meta gets attached after the meta arrives.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: S3 indexer merging production and training copies"`.

---

### Task 9: Fleet + customers + enrolment routes

**Files:** Implement `routes/fleet.py`, `routes/customers.py`; Test `tests/cloud/test_fleet_routes.py`

Behaviour: `POST /v1/devices/enroll` creates a Device with `device_id = str(uuid4())` for an existing customer and a unique site (409 if the site is already assigned). `GET /v1/fleet` returns one `DeviceSummary` per device sorted by severity (critical, offline, warning, unknown, healthy) then name; verdict from `fleet_contract.health.verdict` over the stored heartbeat; `events_24h`, `alerts_24h` (kind alert), `false_alarms_7d` (feedback verdict `false_alarm`) from the DB. Customer CRUD per contract; PATCH audits `customer_update` with changed fields in `detail`.

- [ ] **Step 1:** Tests: enrol two devices, seed heartbeats (one fresh, one 3 h old), assert order and verdicts; support can GET fleet, cannot POST customers (403); labeler 403 on both; enrolling a duplicate site → 409.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: fleet health and customer registry routes"`.

---

### Task 10: Events routes — list, detail, detections, review, labeler pseudonyms

**Files:** Implement `routes/events.py`; Create `home_guard_project/cloud/pseudonym.py`; Test `tests/cloud/test_event_routes.py`

Behaviour:
- `GET /v1/events`: keyset pagination ordered `(start_ts DESC, id DESC)`; cursor = urlsafe base64 of `"<start_ts>:<id>"`; `limit` default 100 max 500; filters per contract; `q` uses Postgres `websearch_to_tsquery('simple', q)` against `Event.search`; `filter=<saved filter key>` applies the built-in query from `studio.BUILTIN_FILTERS` (Task 13 defines; until then accept keys `false_alarm`, `ai_dismissed_person`, `ai_failed`, `real_but_wrong`, `low_conf`, `paused` with these SQL meanings: owner verdict false_alarm; kind false_positive and `"person" in detected`; ai in (failed, fallback); owner verdict real_but_wrong; max class conf < 0.45; kind paused). `thumbnail_url` = `/v1/events/{id}/thumbnail` when a thumbnail artifact exists, else null.
- `GET /v1/events/{id}`: `EventDetail`; `raw_meta` = newest RawRevision body of the training copy if present else production.
- `GET /v1/events/{id}/detections`: for `boxes == "sampled"` read every sampled `label_path` from S3 (cache in-process LRU 512), convert YOLO `cls xc yc w h` → `xyxy` normalised, `label` via `COCO_NAMES`, `conf=None`, `t_sec = frame_index / fps`; empty label file → `status="ran_empty"`; provenance `"sampled"`, model `"yolo (collection export)"`. Otherwise `{"provenance": "none", "frames": []}`.
- `PATCH /v1/events/{id}/review` upserts ReviewState; audit `review`.
- Labeler: `customer_name` → `pseudonym.customer(customer_id)` = `"customer-" + sha256(secret+id)[:6]`; `camera` → `"cam-" + sha256(secret+site+camera)[:6]`; `site` → same as customer pseudonym; `raw_meta` without `dispatch`, `teacher.prompt` kept; feedback `raw_text` and `note` blanked.
- Every detail view audits `event_view` (batched: one audit row per staff per event per 10 min).

- [ ] **Step 1:** Tests over an indexed moto bucket (reuse Task 8 fixture builder moved into `tests/cloud/builders.py` as `seed_bucket(s3client)` + `index_fixture_bucket(session, s3)`): page through 3 events with `limit=1` and verify cursors cover all without duplicates; `kind=false_positive` filter; `q=walking` finds the front_side alert; detail returns ai_runs with `status=="real"`; detections for the collection clip return a box with `label=="person"` and `0<=xyxy[i]<=1`; labeler sees `customer-` pseudonym and no `dispatch`; review PATCH round-trips.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: event history, detail, sampled YOLO boxes, review state, labeler pseudonyms"`.

---

### Task 11: Media access + audit + owner notices

**Files:** Implement `routes/media.py`, extend `audit.py`; Test `tests/cloud/test_media_routes.py`

Behaviour: `POST /v1/artifacts/{id}/access {purpose}` → 404 if unknown, 410 if `available` false, 403 for labeler when customer `consent_training` is false, 403 for anyone when purpose `support` and `consent_recordings` is false; returns presigned URL (300 s) with `mime` by extension. Records `audit.record(action="media_view", target=s3_key, reason=purpose)` and `audit.owner_notice(session, s3, device, staff, kind="recording", cameras=[camera])`. `owner_notice` batches: if a notice for the same (staff, device, kind) was written in the last 30 min, append the camera to the stored notice row (DB `OwnerNotice` table: id, device_pk, staff_id, kind, cameras, first_ts, last_ts, s3_key) and rewrite the same S3 object; else create a new one at `fleet/<device_id>/notices/<YYYYmmddTHHMMSS>_<id>.json` with body `{"schema_version":1,"id":..., "kind":"recording","staff_name":..., "cameras":[...], "from_utc":..., "to_utc":..., "message":"Home Guard support viewed recordings from Front side"}` (display names from `Camera.display_name` or the camera name with `_`→space, capitalised). Labeler views of training data do not create owner notices (training consent covers them) but are audited. `GET /v1/events/{id}/thumbnail` → 307 redirect to presigned thumbnail.

- [ ] **Step 1:** Tests: access returns a URL containing the key; second view of another camera within 30 min updates the same notice object (one S3 object, two cameras); a view 31 min later (inject clock) creates a second object; labeler without consent 403; support without recordings consent 403; unavailable artifact 410; audit rows present.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: audited media access and batched owner notices"`.

---

### Task 12: Media worker — renditions, thumbnails, filmstrips

**Files:** Create `home_guard_project/cloud/media.py`; extend `manage.py` with `media-once`; Test `tests/cloud/test_media_worker.py`

**Produces:** `ensure_media(session, s3, event, ffmpeg="ffmpeg", workdir) -> list[Artifact]`; `process_pending(session, s3, limit=50) -> int`.
For each event with an available `original_video` and no thumbnail: download to temp; thumbnail = frame at 40% of duration, 480 px wide JPEG q80 → `admin_cache/thumbs/<event_id>.jpg`; filmstrip = 1 fps, 160 px wide frames tiled horizontally into one JPEG → `admin_cache/filmstrips/<event_id>.jpg` with sprite metadata in `Artifact.detail` (`{"fps":1,"tile_w":160,"tile_h":131,"count":N}`); rendition only when codec is not `h264` → `admin_cache/renditions/<event_id>.mp4` (`-c:v libx264 -preset veryfast -crf 26 -movflags +faststart -an`). All new artifacts `provenance="cloud"`. ffmpeg failure → `IndexProblem` row, event left without media, never raises.

- [ ] **Step 1:** Tests: generate a 3-second 320x240 mp4v clip with `cv2.VideoWriter` (cv2 is in the base env), upload to moto at a clip key, index, run `process_pending` → thumbnail + filmstrip + rendition artifacts exist in S3 and DB; second run creates nothing; a corrupt mp4 records an IndexProblem. Skip with `pytest.skip` if `shutil.which("ffmpeg")` is None.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: thumbnails, filmstrips and browser/Qt-safe renditions"`.

---

### Task 13: Training studio — filters, collections, exports

**Files:** Create `home_guard_project/cloud/studio.py`; implement `routes/studio.py`; Test `tests/cloud/test_studio.py`

**Produces:** `BUILTIN_FILTERS: list[SavedFilter]` (keys and meanings from Task 10, titles: "Owner said false alarm", "AI dismissed, YOLO saw a person", "AI call failed or fell back", "Owner said real but wrong", "Low detector confidence", "Paused-camera footage"); `build_export(session, s3, export_id) -> Export`.
Export (runs in a background thread for now, state machine queued→running→ready|partial|failed), into `training_exports/<name>/v<N>/`:
- `manifest.json`: `{"schema_version":1,"name","version","created_utc","created_by","collection_id","formats","split":{...},"class_map":{"0":"person",...,"8":"dog"},"items":[{"event_id","site","camera","stem","split","clip_key","sha256","ai_status","owner_verdicts":[...],"sources":[keys]}],"missing":[{"event_id","reason"}]}`.
- `yolo/`: for events with sampled labels: copy `images/<split>/<stem>_fNNNN.jpg` and `labels/<split>/<stem>_fNNNN.txt` with class ids remapped through `CONTIGUOUS` (drop rows whose COCO id is unmapped, count them in manifest `dropped_boxes`), plus `data.yaml` (`path: .`, `train: images/train`, `val: images/val`, `test: images/test`, `names:` contiguous names).
- `vlm.jsonl`: one line per event with real AI (fallback only if `include_fallback_ai`): `{"messages":[{"role":"user","content":"<video><prompt>"},{"role":"assistant","content": json.dumps(parsed)}],"videos":["clips/<split>/<stem>.mp4"],"event_id":...,"owner_verdicts":[...],"ai_status":...}` (LLaMA-Factory ShareGPT format with `videos`; the prompt is the teacher prompt).
- `clips/<split>/<stem>.mp4` when `clips` in formats.
- Split is group-aware: group key = `site + day`; groups assigned to splits by a seeded hash (`sha256(name+group)`) against cumulative split fractions so a group never spans splits.
- Missing source objects → item listed under `missing`, state `partial`.
- Version = previous max version for `name` + 1; nothing under an existing version prefix is overwritten.
- Audits `export_create`; collections audit `collection_add`.

- [ ] **Step 1:** Tests: filters list has 6 builtins; create collection, add 3 indexed events, export with all formats → manifest items 3, `data.yaml` names count 9, a label row `0 ...` stays 0 and a `16 ...` row becomes `8`, vlm.jsonl has only real-AI lines, the two events from the same site+day share a split, deleting one source clip before export → `partial` + `missing`; exporting the same name again → version 2 under a new prefix; support role → 403.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: training studio filters, collections and versioned exports"`.

---

### Task 14: Audit routes + index problems + background loops + local run

**Files:** Implement `routes/audit.py`; Create `home_guard_project/cloud/loops.py` (indexer every 120 s, media every 60 s, started from FastAPI lifespan when `settings.run_loops`), `home_guard_project/cloud/run_local.sh`; Test `tests/cloud/test_audit_routes.py`

`run_local.sh`: uses pgserver data dir `~/.homeguard/cloud_pg`, env from `~/.homeguard/cloud.env` (JWT secret generated on first run), applies the TLS env from Global Constraints, runs `manage init-db` then `uvicorn` on `127.0.0.1:8600` with loops on. It reads S3 with the laptop's AWS profile (read-mostly; writes only under `fleet/`, `admin_cache/`, `training_exports/`).

- [ ] **Step 1:** Tests: admin sees audit page with cursor; support 403; index problems listed; lifespan with `run_loops=False` starts no threads.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit `"Admin cloud: audit log routes, background loops, local runner"`.

---

### Task 15: Demo data for the exe + end-to-end local check

**Files:** Create `home_guard_project/cloud/export_demo.py`; Create `docs/admin/demo/*.json` (generated); Test `tests/cloud/test_demo_export.py`

`export_demo.py`: seeds an embedded Postgres + moto bucket from the fixtures (plus synthetic events: 3 customers, 4 devices with verdicts healthy/warning/critical/offline, 120 events across 6 cameras and 48 h, mixes of kinds/AI status/verdicts, 2 collections, 1 ready export, 30 audit rows), then calls every GET route through TestClient as admin and as labeler and writes responses to `docs/admin/demo/<route_name>[.labeler].json`, plus thumbnails/filmstrips as tiny generated JPEGs in `docs/admin/demo/media/`. The exe's `DemoBackend` reads only these files.

- [ ] **Step 1:** Test: running `export_demo.main(tmpdir)` produces `fleet.json`, `events.json`, `event_<id>.json`, `detections_<id>.json`, `studio_filters.json`, `collections.json`, `exports.json`, `audit.json`, each validating against its Pydantic schema.
- [ ] **Step 2–4:** fail → implement → pass. Generate into `docs/admin/demo/`.
- [ ] **Step 5:** End-to-end (manual, recorded in commit message): `bash home_guard_project/cloud/run_local.sh`, `manage create-staff` for the founder, `manage enroll --customer "Ameer home" --site test`, `--site bian`, `--site ameer_house`; wait one index cycle; `curl` `/v1/fleet` and `/v1/events?limit=5` with a token → real events from S3 with both copies merged. Commit `"Admin cloud: demo data for the exe and first local run against S3"` (force-add nothing; json is not ignored; media JPEGs are — store them as `.jpeg`? No: `*.png` is ignored, `.jpg` is not; verify with `git check-ignore`).

---

## Parallel track: HomeGuardAdmin.exe (Codex)

Starts after Task 2 (contract) using a hand-written subset of demo JSON; switches to generated demo data after Task 15. Separate brief: `docs/admin/CODEX_BRIEF_EXE_R1.md`, worktree `C:\Users\ameer\Ameer\home_guard_admin_ui`, branch `admin-console-ui` (from `admin-console` after Task 2). Rounds:

1. Shell + design system + sign-in + Fleet + DemoBackend + HttpBackend skeleton + screenshot harness + PyInstaller build script.
2. Customer page: timeline (density strip, event list, filters) + event player with YOLO overlay + AI record panel + feedback.
3. Review queue (keyboard), Ctrl+K palette, Training studio (filters, collections, export wizard + history), Audit.
4. HttpBackend against `run_local.sh` + polish from Claude's review.

Each round: Claude reviews screenshots + tests + diff, then merges into `admin-console`.

---

## Self-review notes

- Spec §4.2 items not in phase 1 by design: live sessions, recompute job, config, health alerts to Telegram, AWS deploy (phase 2/3 plans).
- Spec §4.3 box additions: phase 2 plan.
- Owner notice is written to S3 now; the box starts showing it in phase 2 (box pulls `notices/`).
