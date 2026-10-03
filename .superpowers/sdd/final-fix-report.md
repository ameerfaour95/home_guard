# Final fix round — report

Branch `admin-console`, on top of 5896c23. Inputs: fix-final-decisions.md (all applied), final-review-claude.md,
final-review-codex.md, fix-final.md R1–R4. `schemas.py` and `docs/admin/openapi.json` are unchanged.
One new migration, 0007, follows 0006.

Commits:
- 267f8a4: migration 0007 and the shared helpers (aliases, retry columns, audit index, label validator, consent
  decision, id/length bounds, `s3.delete_keys`, `loops.loop_lock`)
- 9d3e36c: privacy and consent fixes
- 16e0cfa: truthful evidence and recovery
- 4b1db2e: the regenerated demo data. The only change is `customer_id: 0` in the `*.labeler.json` files.

Test command (constraints.md): **401 passed in 311 s** (baseline 366).
RED: the new tests were written first and run against the old code: **33 failed, 2 passed**. The 2 that passed
are guard tests whose behaviour was already correct (empty label is `ran_empty`; unchanged feedback is exported).

## Item → change → test → RED / GREEN

| Item | Change | Test | RED (old code) | GREEN |
|---|---|---|---|---|
| Codex C1 identity history | New table `identity_aliases`. Aliases are recorded at enrol (route and CLI), on a customer rename (old and new name, and the stored search text is recomputed), and by every index pass (cameras, display names, heartbeat host; the heartbeat host is now also a term). `redact.identity` uses the union of the device's aliases plus the customer aliases of sibling devices. Migration 0007 backfills them. | test_final_privacy::test_old_customer_name_stays_redacted_after_a_rename, ::test_old_camera_display_name_and_heartbeat_host_stay_redacted, ::test_enroll_records_aliases; test_migration_0007 | the partial PATCH reset consent (labeler 404); display name "quorvex" leaked; no `IdentityAlias` | pass |
| Codex I1 oracle | Labeler-created collection and export names get no identity check. Collections made by a labeler are visible only to that labeler and to admins: list, items GET/POST/DELETE, preview and create all give one 404. Names chosen by admins keep the check. | ::test_labeler_names_are_not_checked_against_hidden_households, ::test_labeler_collections_are_private_to_their_creator_and_admins | `[400, 200]`; another labeler saw the collection | pass |
| Claude I5 labeler customer_id | `customer_id = 0` in every labeler response. A labeler's `customer_id` filter on /events and /events/density is one uniform 400. | ::test_labelers_get_customer_id_zero_and_cannot_filter_by_it | the real ids were returned | pass |
| Codex I2 / Claude I1 thumbnail consent | `access.media_refusal(role, purpose, consents)` is used by artifact access (clips, filmstrips), the thumbnail redirect (staff → 403 without recordings consent) and `thumbnail_url` in lists (null when the viewer may not open it). | ::test_media_consent_is_one_decision_for_every_route (3 roles × 4 consent combinations × 3 purposes × clip/filmstrip, plus thumbnail and list link) | support got 307 without consent | pass |
| Claude I2 expiry | `expired` = production retention passed **and no copy** of the video is available (indexer and `refresh_expiry`). The export reason "expired" therefore only applies when no video is left. | test_final_evidence::test_training_copy_stays_exportable_after_production_retention; test_indexer expiry tests updated | `expired` was true while `video` was true | pass |
| Claude I3 derived-media retention | `media.retire_orphans`, run on every media pass, deletes from S3 and marks unavailable the thumbnails, filmstrips and renditions of events with no available clip, plus the opaque copies whose source is gone or belongs to such an event. | ::test_derived_media_is_deleted_when_no_source_clip_is_left | the files stayed available | pass |
| Codex I3 frozen feedback | Snapshot v2 freezes the feedback (verdict, action, received_utc, source revision = feedback artifact's applied ETag). At build time a mismatch or missing row becomes `changed_since_request` (note "owner feedback") and the entry is left out. v1 snapshots are still read. | ::test_feedback_changed_after_the_request_is_reported_not_exported, ::test_unchanged_feedback_is_exported_from_the_snapshot | `missing` was empty and the false_alarm was exported | pass |
| Codex I4 labels | `fleet_contract/yolo_labels.parse_label` (stdlib, strict) is used by the detections route and by `studio.remap_label`. A malformed file → `not_run` plus an IndexProblem "invalid yolo label". Reads are conditional on the indexed ETag and cached per (key, etag). | ::test_parse_label_is_strict, ::test_exporter_and_viewer_share_the_validator, ::test_malformed_label_is_not_run_and_a_problem[×3], ::test_empty_label_is_ran_empty, ::test_detections_read_only_the_indexed_revision | ran_empty / ran for garbage, a fractional class and a negative size; an unindexed revision was served | pass |
| Codex I5 media recovery | `MediaError.transient`. Timeouts, missing ffmpeg, OSError, a full disk and S3 or other non-media errors are transient: retried after 2^n minutes, at most 8 attempts, then `next_retry_at` is NULL. Decode errors and the "too large" caps are permanent. `manage media-retry [--event ID]` clears the problems. | test_final_ops::test_transient_media_failure_is_retried_with_backoff, ::test_transient_failures_stop_after_eight_attempts, ::test_corrupt_clip_is_permanent_until_an_operator_retry | there was no `now`/retry support | pass |
| R1 / Codex I6 CLI locks | `loops.loop_lock(engine, kind)` is used by `Loop.run_once` and by `manage index-once` and `media-once`. When the lock is busy the CLI prints "the server is already …" and exits 0. The dead, unlocked `indexer.run_once` was removed. | ::test_cli_waits_for_the_server_loop[index-once, media-once] | the CLI ran while the lock was held | pass |
| R2 concurrent indexers | Artifacts, events, raw revisions, feedback, problems and cursors are inserted with INSERT … ON CONFLICT and re-read. Events another writer created are taken over and their AI run is loaded, so no run is duplicated. Media `_upsert` is conflict-safe too. The statement bound still holds (first pass 43, unchanged pass 16). | ::test_two_indexers_on_one_database_both_succeed | IntegrityError (UniqueViolation) | pass |
| R3 logging | `app.configure_logging()` (INFO, "%(asctime)s %(levelname)s %(name)s %(message)s", idempotent) runs in `create_app_from_env`. | ::test_server_logging_is_configured_with_timestamps | no handler | pass |
| R4 port | The run_local.sh header documents HG_CLOUD_PORT, the log format and the CLI lock and retry behaviour. | (docs) | — | — |
| Audit de-dup (Claude I4) | event_view and thumbnail_view lookups are `ts > now − window` with `LIMIT 1` on the new index (staff_id, action, target, ts). The thumbnail view takes an advisory lock like event_view. | ::test_view_audit_lookups_are_bounded_by_the_window | there was no `ts` bound | pass |
| M1 bounds | `deps.require_id` / `id_in_range`: path ids outside 1..2^31−1 → 404 (events ×4, artifacts, customers ×2, collections ×3, exports, enroll and export `customer_id`/`collection_id`). Query ids out of range match nothing (events, density, audit, audit staff). Lengths: name ≤ 120, notes and description ≤ 4000, other strings ≤ 200 (timezone, site and ssh_user ≤ 64, their column widths) → 422. | ::test_out_of_range_ids_are_404_not_500, ::test_too_long_strings_are_422_not_500 | DataError 500; a 121-character name returned 200 | pass |
| M2 caps | ffmpeg runs with `-threads 2 -filter_threads 2` (and `-threads 2` for the x264 encode). Clips over 500 MB (checked before download) or 15 min → permanent "too large". `studio.MAX_EXPORT_EVENTS = 20_000` → 400 on preview and create. | ::test_every_ffmpeg_run_is_limited_to_two_threads, ::test_huge_clip…, ::test_long_clip…, ::test_export_selection_is_capped, ::test_default_export_cap… | missing | pass |
| PATCH /customers (both reviews) | Only `model_fields_set` changes. The audit `detail = {"changed": [field names]}`. A no-op writes no audit row. A missing customer → 404. | ::test_customer_patch_updates_only_the_fields_sent; test_fleet_routes updated (names only) | consents and timezone were reset | pass |
| Migration 0007 | Upgrade, downgrade and re-upgrade on a populated DB; the backfill is checked; row counts are kept; `compare_metadata` drift is empty. | test_migration_0007 | no 0007 | pass |
| Demo data | Regenerated. Only the labeler `customer_id` changed. The committed-vs-fresh test passes. | test_demo_export | — | pass |

Behaviour the exe must know: labeler `customer_id` is always 0, and a labeler `customer_id` filter is a 400.
`thumbnail_url` is null when the viewer may not open the thumbnail. Collection names over 120 characters are a
422 (before, names over 255 were a 400). `completeness.expired` now means "no video left". PATCH /customers still
needs `name` in the body (CustomerIn is frozen); every other field may be omitted.

## Later (not done in this round)
- Claude M2: clock injection is partial. Routes now pass `ts=` to audit (customers, enrol, review,
  collections, exports), but auth, the exports sweep and `audit.record`'s default still use the wall clock.
- Claude M3: admin `purpose="training"` views send no owner notice (document or notify).
- Claude M4: a separate `HG_CLOUD_DATA_KEY` for pseudonyms, splits and opaque names.
- Claude M5: a queued export can be failed after 30 min while it is still waiting in the executor (wording or timing).
- Claude M6: remaining scaffolding (`manage._optional`, "serve: not available yet", `hasattr(studio, …)`, the
  `sessionmaker` re-export). `indexer.run_once` was removed.
- Claude M7: `/docs` is public; no `/healthz`; no staff disable/TOTP-reset/role CLI; no logout; `serve` binds
  0.0.0.0 over HTTP; ffmpeg runs without `nice`; audit indexes `(ts, id)` and `(customer_id, ts)`.
- Claude M8: validate `timezone` with `zoneinfo` (only its length is bounded now).
- Claude M9: `test_demo_export` is slow (about 100 s setup); wall-clock latency asserts in test_audit_routes; cwd-relative
  test_contract path; stale `test_zz_*` pyc files.
- Claude M10: event lists and /detections are not audited (decide and record).
- Claude M11: assignments with validity ranges before any device can be re-assigned.
- Claude M12: a bare `alembic upgrade head` has no URL. `manage init-db` is the entry point (now documented in
  `manage.alembic_config`).
- Claude I3 extras: an S3 lifecycle rule on `admin_cache/` (for example 15 days) as a backstop. The worker still renders
  media for customers without recordings consent.
- Codex M2 rest: measure peak RSS, disk and API latency on the target server. The snapshot is bounded only by the
  20,000 cap.
- Codex: the listing budget (every pass lists both roots in full); restarted queued exports fail rather than resume;
  re-run the real three-site acceptance and load a real exported bundle in the training tools.
- Deferred minors: customers list N+1; case-sensitive fleet and customer sort; document the DELETE collection-items
  JSON body for the exe; sanitise `message_id` in the traversal fixture; smoke-driver assert.
