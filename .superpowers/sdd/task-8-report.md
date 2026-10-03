# Task 8 report: S3 wrapper + indexer

**Status:** DONE. Commit `c95da94` "Admin cloud: S3 indexer merging production and training copies" on `admin-console`.

**Tests:** RED first: all 18 new tests errored with `ModuleNotFoundError: home_guard_project.cloud.s3`. GREEN: `tests/cloud/test_indexer.py` 18 passed. Full suite `uv run --group cloud pytest tests/fleet_contract tests/cloud -q` gives 170 passed.

**Run note (important for Tasks 10-13):** any test that imports boto3 or moto must run with `env -u SSLKEYLOGFILE -u PYTHONSTARTUP AWS_CA_BUNDLE=C:/Users/ameer/.homeguard/ca_bundle.pem`. Without it, pytest exits with rc=1 and prints **nothing**. This is the antivirus `OPENSSL_Applink` crash, even though moto makes no network calls.

## Files
- `home_guard_project/cloud/s3.py`: `ObjInfo(key, etag, size, last_modified)` (ETag has its quotes stripped) and `S3(client, bucket)`, with `list` (paginated), `get_bytes`, `get_text`, `get_json`, `put_json`, `presign(key, ttl=300)` (`generate_presigned_url("get_object", ExpiresIn=ttl)`) and `exists`. `put_json` raises `ValueError` outside `fleet/`, `admin_cache/` and `training_exports/`, which enforces the phase-1 read-only rule.
- `home_guard_project/cloud/indexer.py`:
  - `IndexStats(new_events, updated_events, artifacts, feedback, problems)`
  - `index_device(session, s3, device, full_scan=False, now=None)`, which commits once at the end
  - `index_all(session, s3, full_scan=False) -> {site: IndexStats}`, one commit per device. A device that fails is rolled back and logged, and the other devices still run.
  - `run_once(engine)`, used by `manage index-once`. It builds the boto3 client from env (`HG_CLOUD_REGION`, `HG_CLOUD_BUCKET`).
  - `artifact_role(KeyInfo)`
- `tests/cloud/builders.py` (reusable):
  - Key constants: `PROD_META`, `TRAIN_META`, `PROD_CLIP`, `TRAIN_CLIP`, `RAW_ANSWER`, `CROPS`, `FEEDBACK`, `ORPHAN_FEEDBACK`, `GENERAL_FEEDBACK`, `HEARTBEAT`, `TRAVERSAL_META`, `FP_*`, `FALLBACK_*`, `PAUSED_*`, `COLLECT_*`, `YOLO_IMAGES/LABELS`
  - Functions: `seed_objects()`, `seed_bucket(s3client, bucket=BUCKET, exclude=())`, `put(s3client, key, body)`, `feedback_body(alert_id, verdict, time_utc)`, `fixture_json(name)`, `enroll(session, site="test", customer_name="Acme")`. `enroll` flushes but does not commit.
  - Import it with `from . import builders as b`. `tests.cloud...` does not import, because `tests/` has no `__init__`.
- `tests/cloud/test_indexer.py`: 18 tests that use moto and the per-test Postgres.

## What the seeded bucket yields
6 events for site `test`:
- `front_side_1791020177_alert`: production + training copies
- `front_side_1791020999_alert`: the traversal fixture, which also records an IndexProblem
- `back_door_1791013299_fp`: real AI
- `back_door_1791013421_fp`: fallback AI
- `left_side_1_1791014883_paused`
- `back_door_1790944263_trigger`: a collection clip with sampled YOLO labels, so `boxes == "sampled"`

It also yields 3 feedback rows: one linked, one general, and one orphan whose `alert_stem` names a missing event. The device heartbeat comes from `dataset_test/_status/heartbeat.json`.

## Algorithm (per device)
1. **List** `production_<site>/` and `dataset_<site>/` once each. Only the indexed areas get Artifact rows, with roles as in the decisions: crop jpg is `teacher_frame` and crop mp4 is `crop_video`. Each row has `detail={"copy": "production"|"training"}`, a mime type from the extension, plus camera and stem from the key.
   - An Artifact that is new, has a changed ETag, or reappears marks its event `(camera, stem)` dirty.
   - JSON (meta, feedback, `_status/heartbeat.json`) is queued for fetch only when `(key, etag)` is not already in `raw_revisions`. Media bodies are never fetched.
2. **Disappearance check** per prefix. It runs when `full_scan=True`, when there is no cursor yet, or when `S3Cursor.last_full_scan` is at least 30 minutes old. Known artifacts missing from the listing become `available=False` and their events become dirty. An object that comes back flips to `available=True`.
3. **Fetch** queued JSON and store each body as a RawRevision.
   - A body that is not JSON is stored as `{"_invalid_json": text}` with problem "invalid json", so it is not fetched again until its ETag changes.
   - Feedback is upserted by `s3_key`; `alert_stem` is set from `alert.alert_id`, and the row is linked when the event exists.
   - The heartbeat sets `Device.last_heartbeat` and `last_heartbeat_at` (parsed `time_utc`), and only when its ETag changed.
4. **Rebuild** each dirty event from the database, never from S3:
   - The meta artifacts of both copies (available or not) are parsed from the RawRevision that matches their current ETag. This keeps a copy whose S3 object later disappears.
   - Merge: AI comes from the best status. Summary, label, alert_command and alert_reason come from the copy whose AI is chosen when that status is not `none`; otherwise production first, then training. `dispatch` comes from production when present. `owner_feedback` is unioned and deduplicated by `(time_utc, verdict)`. `start_ts` falls back to `trigger_ts`. `day` comes from the key, `clip_start_local` from the raw body, and `expires_at` is start + 14 days when a production copy exists.
   - All event-role artifacts of either copy are attached, so orphan media gets attached once the meta arrives. Feedback rows and their artifacts are attached by `alert_stem`.
   - `owner_verdicts` holds the distinct verdicts other than "none", in time order, from Feedback rows and meta `owner_feedback`.
   - AiRun: one `guard` run per event. It is replaced by an equal or better status and never by a worse one. On replacement, `raw_artifact_id` and `input_artifact_ids` are re-linked, in `input_frames` order and only for frames that exist.
   - Completeness has exactly `video`, `boxes`, `ai`, `owner_feedback`, `expired` and `copies`. `expired` = a production copy exists, no available production clip remains, and the event is more than 14 days old, so training-only collection clips never expire.
   - Parse problems are upserted per key and cleared when a later revision of that key parses cleanly.

## Concerns / deviations
1. **IndexProblem's primary key is `s3_key` alone** (migration 0001), so "one row per (s3_key, reason)" is implemented as one row per key with its reasons joined by `"; "`. The row is upserted and only re-counted when the reason text changes. A true `(s3_key, reason)` key would need a migration.
2. **`muted` is not stored**, because Event has no `muted` column (nor customer_id). If Task 10 needs muted, add a column, or read it from the production RawRevision.
3. **A kept (better) AiRun is not re-linked** when a worse revision arrives, because AiRun records no source copy. Its earlier links stay. Re-linking does happen on every pass in which the run is (re)written.
4. **First-index cost:** the rebuild issues about 8 small queries per dirty event (event, artifacts, revisions, feedback, AI run, artifact ids, problem). Steady-state runs touch only changed events. A first index of thousands of clips is O(events) round trips; it could be batched later if needed.
5. `index_device` takes `now=` as a test hook (expiry and the full-scan interval). Production callers omit it.
6. `completeness.ai` reflects the stored AiRun status. Because a run is never downgraded, `ai` can stay `real` after a copy's meta is rewritten worse. This is intentional.
