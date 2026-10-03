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

## Fix round (Codex review)

All 12 findings of `task-8-codex-review.md` are fixed. Commits on `admin-console`:
- `ad90a1b` migration `0003_indexer_fixes` and the model changes: `Event.muted`, `AiRun.ai_source_key` and `ai_source_etag`, `Artifact.applied_etag` and `etag_mismatches`, `IndexProblem.etag`, the `text_pattern_ops` prefix indexes on `artifacts.s3_key` and `index_problems.s3_key`, and `ix_artifacts_camera_stem`.
- `5f5eb8c` `parse_key` layout validation (`fleet_contract/keys.py`, still stdlib-only).
- `2191b59` the indexer fixes, `S3.get_text(key, if_match=)` and `ETagMismatch`.
- `696a079` the statement-count regression test, and new artifacts are now inserted once instead of insert-then-update.

How it works now:
- `Artifact.etag` is the ETag S3 lists. `Artifact.applied_etag` is the revision whose content is in effect.
- A JSON key whose listed ETag differs from its applied ETag is (re)applied. If that (key, ETag) was stored before, the stored body is used. Otherwise the body comes from a GET with `IfMatch`.
- Rebuilds read only stored revisions.

RED evidence: the new tests were run against the pre-fix indexer (`c95da94`, unchanged in `74fb815`) before any fix. There were 22 failures; the RED column below gives each failure reason. GREEN is the final code.

| # | Change | Test(s) | RED (before) | GREEN |
|---|---|---|---|---|
| 1 | `best_ai()` picks the best AI status across all retained revisions of both copies. On a tie, the current revision wins (production first), then the newest. Summary, label, alert_command and alert_reason come from that revision. `AiRun.ai_source_key`/`ai_source_etag` record it. `owner_feedback` is unioned over every retained revision of both copies, deduplicated by (time_utc, verdict). | `test_recreated_training_meta_keeps_teacher_fields_and_all_verdicts` (teacher meta, then two teacher-free recreations) | summary regressed to the production text | pass |
| 2 | `upsert_ai_run` always re-links `raw_artifact_id` and `input_artifact_ids` from the winning revision's paths, whether or not the status changed. | `test_retained_ai_links_sidecars_that_arrive_after_a_poorer_copy` | `raw_artifact_id` None, expected 21 | pass |
| 3 | `Event.muted` added. `dispatch` comes only from the production copy's applied revision, and so does `muted` (from its raw `alert.muted`, which must be a bool). Both are NULL when there is no production copy. | `test_dispatch_and_muted_come_only_from_production[True/False]` (both arrival orders) | `Event` had no `muted`; the training copy's dispatch was reported | pass |
| 4 | The listed ETag is compared with `applied_etag`; when they differ, the cached revision is replayed with no GET. Applies to meta, feedback and heartbeat. | `test_heartbeat_returning_to_a_cached_revision_is_reapplied`, `test_feedback_returning_to_a_cached_revision_is_reapplied` (A→B→A, GETs counted) | stayed at B (11:20:14 / `false_alarm`) | pass, 0 GETs |
| 5 | Every GET sends `IfMatch=<listed ETag>`; moto supports it and returns 412 `PreconditionFailed`. A response ETag that does not match is treated the same way. On a mismatch the key is skipped: nothing is recorded, nothing advances, and `etag_mismatches` is incremented. On the 3rd consecutive pass this becomes the problem "object changed during fetch on 3+ passes in a row". A successful fetch resets the counter and clears the problem. | `test_object_replaced_between_list_and_get_is_not_recorded_under_the_listed_etag` | the newer body was saved under the listed ETag | pass |
| 6 | `refresh_expiry(session, device, now)` runs at the end of every pass. One SELECT with an EXISTS on available production video sets `expired` to true or false. `now` is injectable. | `test_expiry_advances_with_time_without_listing_changes` (day 13, then day 15 with the listing unchanged, then the video reappears; also called directly) | `ImportError: refresh_expiry`, and stayed false | pass |
| 7 | `clip_start_ts`, `clip_end_ts`, the stem trigger time and feedback/heartbeat `time_utc` must be finite and in [2020-01-01, 2100-01-01). Otherwise the key gets one IndexProblem ("timestamp out of range: ...") and is skipped. Bodies containing U+0000, which JSONB rejects, are stored as invalid JSON. A database error still rolls back that device only. | `test_out_of_range_timestamp_is_one_problem_and_the_site_continues[1e20, 2001, stem 99999999999999]`, `test_database_failure_still_rolls_back_only_that_device` | `OverflowError` aborted the device (1e20, huge stem); 2001 was accepted with no problem. The DB-failure test passed before as well (existing behaviour, kept). | pass |
| 8 | `parse_key` returns None for keys with `..`, `\`, NUL, empty or `.` segments, or the wrong depth per area (5; yolo 6; `_status` 3). It also returns None when the camera does not match `^[A-Za-z0-9_.-]{1,80}$` or the day does not match `YYYY-MM-DD`. The indexer records "invalid key layout" for rejected keys under meta/clips/feedback/_status/responses/vlm_crops/yolo and creates no artifact. All 232 fixture keys still parse. | `test_parse_key_rejects_traversal_and_bad_layouts`, `test_parse_key_accepts_the_general_feedback_camera_and_dotted_names`, `test_invalid_key_layout_is_a_problem_not_an_artifact[4 keys]` | the traversal key parsed; the bad day hit `OSError [Errno 22]`; no problem rows | pass |
| 9 | Each problem row stores its ETag. Problems are set and cleared only when a revision is applied, never during a rebuild. They clear only when the key's current revision applies with no problems. A fetch failure keeps the previous applied state and creates or keeps the problem; the fetch is retried every pass. | `test_fetch_failure_keeps_previous_state_and_the_problem`, `test_fixed_feedback_and_heartbeat_clear_their_problem[feedback, heartbeat]` | the fetch-failed problem was cleared by the rebuild; fixed bodies never cleared their problems | pass |
| 10 | Bulk loads per pass: artifacts, problems and cursors use 1 statement each through the prefix indexes (EXPLAIN shows a Bitmap Index Scan on `ix_artifacts_s3_key_prefix`). Revisions are fetched by tuple-IN (key, ETag) for the pending keys only, and in full only for the meta keys of rebuilt events, never heartbeat history. Events, feedback and AI runs are loaded per pass with chunked IN lists. Inserts and updates are batched by the ORM (insertmanyvalues/executemany). | `test_index_pass_statement_count_is_bounded` (1,000 events, 5,000 keys, in-memory S3) | 14,015 statements on the first pass | 20 statements on the first pass (limit 600), 5 on an unchanged re-pass (limit 20) |
| 11 | An artifact counts as changed when its ETag or size changes; `bytes` and `last_modified` are updated. JSON fetch deduplication still keys on the ETag. | `test_size_only_change_updates_the_artifact` | bytes stayed 88 | pass |
| 12 | When feedback is (re)applied, `Artifact.event_id` is set to `Feedback.event_id`, including NULL. The event it was previously linked to is rebuilt, so its verdicts are recomputed. | `test_feedback_artifact_follows_feedback_event` (linked → other event → general) | the artifact stayed on the old event | pass |

Perf, measured on a synthetic 5,000-key site (1,000 events, each with production and training meta, two clips and a crop). The test uses an in-memory S3, so S3 latency is excluded; the times are DB plus CPU against the local pgserver:
- First pass: 20 statements, 1.9–2.0 s.
- Unchanged re-pass: 5 statements, 0.07–0.25 s.
- Under heavy machine load (other agents sharing the laptop), the same first pass took up to ~19 s. The time goes to Postgres I/O; the statement count did not change.
- Before the fix, the first pass issued 14,015 statements.

Notes:
- `refresh_expiry` scans the device's events on every pass using the device index plus an EXISTS per candidate. If sites reach hundreds of thousands of events, a partial index on `(device_pk, expires_at)` would help.
- After migration 0003, existing artifacts have `applied_etag` NULL. The first pass therefore re-applies every JSON key once, from stored revisions with no GETs, and rebuilds the events.
- `muted` is NULL when the production meta has no boolean `alert.muted`. The false_positive and fallback writers omit it, so it is shown as unknown, not as "not muted".
- Concerns 2-4 of the original report are resolved by this round: muted is stored, the kept AiRun is re-linked, and the first index is batched.

Full suite (`uv run --group cloud pytest tests/fleet_contract tests/cloud -q`, with the TLS env from constraints.md): **198 passed** in 481 s on the loaded machine (174 before this round, plus 24 new).

## Fix round 2 (Codex re-review)

Fixes the 2 partial findings (#8, #9) and the 2 new issues (N1, N2) of `task-8-codex-rereview.md`. Each has a regression test that failed first against `23989ea` (RED below), then passed.

| # | Change | Test(s) | RED (before) | GREEN |
|---|---|---|---|---|
| 8 | `parse_key` rejects impossible calendar days (`datetime.date.fromisoformat` in a try, after the `YYYY-MM-DD` regex). Stems in meta/clips/responses/vlm_crops/yolo must start with `<camera>_` and `stem_kind` must yield an epoch. Feedback stems must be `<that kind of stem>_<ms>`. General feedback is exempt: anything under `_general/`, and `general_<ms>` in a camera folder, which box/feedback.py writes when an alert has a camera but no alert_id. Still stdlib-only, and all 232 fixture keys parse. | `test_parse_key_rejects_impossible_calendar_dates`, `test_parse_key_requires_the_folder_cameras_stem_with_an_epoch` (fleet_contract), `test_invalid_calendar_day_colliding_with_a_real_event_is_a_problem[2026-99-99, 2026-02-30]` | the keys parsed; in the DB, no IndexProblem was recorded for the colliding key (`'NoneType' object has no attribute 'reason'`) | pass: "invalid key layout", no artifact, and the real event keeps day 2026-10-03 and its summary |
| 9 | A JSON key is also re-applied when its listed ETag is the applied one but its open problem names another ETag (A → invalid B → A). Cached A is replayed with no GET, and the stale problem clears. | `test_return_to_the_applied_revision_after_an_invalid_one_clears_the_problem[meta, feedback, heartbeat]` | the stale "invalid json" problem for B stayed (all 3) | pass, 0 GETs |
| N1 | `load()` quarantines persisted artifacts whose key `parse_key` now rejects. Each one is removed from the pass's artifact map, set `available=False` and `event_id=NULL`, and gets the problem "invalid key layout (legacy)". Its old event is rebuilt. The key is not re-flagged when listed, and a Feedback row on such a key is detached. Every later `parse_key(...)` in grouping and rebuild therefore only sees accepted keys. | `test_legacy_artifacts_with_now_invalid_keys_are_quarantined` (a pre-existing `../` meta row that is still listed, plus a `2026-02-30` clip row, full scan) | `AttributeError: 'NoneType' object has no attribute 'root'` (indexer.py:473) | pass, and a second pass records 0 new problems |
| N2 | In rebuild, a meta artifact with `applied_etag` NULL takes its last usable stored RawRevision as applied, and `applied_etag` is persisted. This holds whether or not the key is listed or available. | `test_upgrade_replays_the_stored_production_meta_after_it_disappeared` (index; set `applied_etag=NULL` everywhere; delete the production meta; full-scan re-index) | `(dispatch, muted, expires_at)` became `(None, None, None)` | pass: all three preserved, both copies still listed, `applied_etag` = the production ETag |

Full suite (exact command from constraints.md): **228 passed** in 82 s. This includes the uncommitted route tests of the agent working in parallel.
