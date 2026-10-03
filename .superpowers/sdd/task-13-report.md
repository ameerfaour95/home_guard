# Task 13 report: training studio (filters, collections, versioned exports)

**Status:** done. Commit `204978e` "Admin cloud: training studio filters, collections and versioned exports".
**Tests:** `tests/cloud/test_studio.py` (20 tests), written first and run to RED (501 from stub routes), then GREEN.
Full suite: `306 passed` (`tests/fleet_contract tests/cloud`). The OpenAPI contract test still passes: the frozen
document is unchanged, and the new route functions have no docstrings on purpose.

## What was built

- **Filters** (`cloud/studio.py`): `BUILTIN_FILTERS: list[SavedFilter]` holds the 6 keys with the brief's titles,
  `builtin=True` and `query={"filter": key}`. The SQL now lives only in studio (`builtin_condition`, `has_verdict`,
  `max_class_conf`). `routes/events.py` uses it and the meanings are unchanged. `GET /studio/filters` is open to
  all roles.
- **Collections** (`routes/studio.py`): admin and labeler can list, create, add and remove. Adding or removing an
  event the caller cannot see gives the same 404 body as a missing event, and nothing is half-applied. Labelers
  can only add events from customers who consented to training. `event_count` counts only what the viewer can
  see. Adding is idempotent (`ON CONFLICT DO NOTHING`). Ids outside int4 give a 404, and one call takes at most
  5000 ids. Audit actions: `collection_create`, `collection_add`, `collection_remove`.
- **Exports:**
  - `POST /studio/exports` (admin and labeler) works out the version and records `export_create` in the audit
    log. The version is one more than the highest found in the DB or in the S3 folder, under an advisory lock
    on the name.
  - It commits, then hands `run_export_job` to `app.state.export_runner`. The default runner starts a daemon
    thread; tests inject a synchronous runner.
  - `build_export(session, s3, export_id, *, secret)` only builds an export that is `queued`, and commits each
    state change: running, then ready, partial or failed.
  - It refuses a version folder that already contains objects.
  - It uses `select_export_items(labeler=<creator was labeler>)` and `assign_splits(..., secret=jwt_secret)`
    exactly as the preview does. A test checks that the preview and the export have equal ids and split counts.
  - Any exception sets `failed`, with the error text as the type plus the first line, and anything path-like
    replaced by `<path>`.
  - A startup lifespan sweep (`sweep_stale_exports`) fails `queued` or `running` exports older than 1 h.
  - `manifest_url` is a presigned link to `manifest.json` only, for exports that are ready or partial.
- **Export layout** under `training_exports/<name>/v<N>/`:
  - `manifest.json` is written last, so its presence means the export finished.
    - Items carry `customer`, `camera` (both pseudonyms), `group` (`group-<12 hex>`, keyed HMAC), `split`,
      `kind`, `clip` and `clip_sha256`.
    - They also carry `ai_status`, `owner_verdicts` (redacted), `yolo_frames` and `sources` (`artifact-<id>`).
    - Items contain no site, stem or S3 key.
    - Top-level fields: `split`, `split_counts`, `groups`, `class_map` (`0..8`), `missing`, `excluded`,
      `dropped_boxes`, `frames_without_image`.
  - `yolo/images|labels/<split>/<event_id>_fNNNN.*` and `yolo/data.yaml`, whose `names` are in contiguous order.
    - Labels are downloaded, remapped from COCO to contiguous ids, and uploaded. Rows with unmapped classes or
      malformed rows are dropped and counted.
    - Images are copied server-side, choosing the matching `yolo_image` (same frame, training copy first).
    - A frame with no image artifact is skipped and counted in `frames_without_image`.
  - `clips/<split>/<event_id>.mp4`: a server-side `copy_object` with `ChecksumAlgorithm=SHA256`, so the SHA comes
    from S3 and no clip bytes pass through the API. The H.264 rendition is used when the meta codec is not
    `h264` (or is unknown) and a rendition exists.
  - `vlm.jsonl`: ShareGPT messages plus `videos`, `event_id`, `split`, `owner_verdicts`, `ai_status`,
    `prompt_redacted`, `prompt_source` and `prompt_version`.
    - The prompt is the teacher prompt with identity redacted. If any identity term survives the check, the
      placeholder prompt is used instead.
    - The assistant answer is the redacted `parsed`.
    - Fallback and failed lines appear only with `include_fallback_ai`; those events still contribute clips
      and YOLO.
  - `_private/mapping.json`: per event, the site, camera, stem, customer and device ids, clip key, source and
    etag, YOLO keys and the AI source key. No route serves it.
  - Missing sources go into `missing` and set the state to `partial`. The reasons are `clip_missing`,
    `yolo_label_missing` and `yolo_image_missing`. An event whose clip is missing keeps its item (`clip: null`)
    but gets no `vlm.jsonl` line.
- **S3 wrapper:**
  - `copy(..., sha256=True)` returns the hex digest.
  - New helpers: `put_bytes`, `list_dirs` and `any_under`. All writes still go through the writable-prefix guard.

## Concerns / decisions for the controller

1. **`vlm.jsonl` without the `clips` format.** Its lines point at `clips/...` files that the export does not
   contain. This follows the brief literally. Consider forcing `clips` whenever `vlm_jsonl` is requested.
2. **`clip_sha256` is null** when `clips` is not among the formats, because no copy happens and the API never
   downloads clips to hash them.
3. **When the rendition is used.** It is chosen when the meta codec is not `h264`, including when the codec is
   unknown (old collection metas name no codec). It does not check that the rendition's `src_etag` matches the
   chosen original.
4. **Exports are visible to every admin and labeler.** That covers their names, staff `created_by`, `s3_prefix`
   and the manifest link. Manifests carry no household identity, so this seemed acceptable.
5. **The sweep also fails `queued` exports** older than 1 h, not only `running` ones, because a restart drops
   queued threads too.
6. **New audit actions:** `collection_create` and `collection_remove` were added alongside the required
   `collection_add` and `export_create`.

## Fix round (Codex review)

Spec: `fix-13.md`. Commits: `263aa82` (migration 0006 `exports.worker_id`/`heartbeat_at`, S3 `copy(if_match=)`,
`head`, streamed `sha256`, `delete_prefix`) and `57595e0` (the fixes). New tests: `tests/cloud/test_studio_fix13.py`
(33). RED: 32 of 33 failed before any code change; the one pass is a guard (`test_valid_split_accepted`). Two
tests that first passed for the wrong reason (a TypeError) were tightened to assert the exact error text and
then failed too. GREEN: the full suite has `361 passed`. The OpenAPI contract is unchanged; `ExportOut` was not
touched.

| Finding | Change | Test(s) |
|---|---|---|
| C1 | `identifies_household` checks the identity terms of every device (via `redact.identity_terms`/`redact_text`), plus customers without a device. Export names, and collection names and descriptions, get 400 "Name must not identify a household". | `test_export_and_collection_names_must_not_identify_a_household` |
| M1 | One `validate_export_request` used by both preview and create. Split keys must be within {train,val,test}; fractions must be finite, at least 0, sum to 1 (±1e-6), and train must be above 0. | `test_split_validated_identically_by_preview_and_export` (5 cases), `test_valid_split_accepted` |
| I4 | `export_visible_to`: a labeler sees only exports they created, and only while every snapshot event is still visible. Otherwise list omits it and get returns the same 404 as a missing export. For admins, `error` reads "warning: …consent…" at read time, only when there is no real error. The manifest only counts exclusions of households without training consent (`excluded_counts`); the ids go in `_private/excluded.json`. | `test_labelers_see_only_their_own_exports_of_visible_events`, `test_manifest_never_lists_exclusions_of_non_consenting_customers` |
| I1 | `effective_formats` adds clips to any `vlm_jsonl` request, and the preview warns about it. Each clip is verified by streaming its hash; a missing clip drops the line with `clip_not_in_export`. Every referenced key gets a `head` check before publishing. | `test_vlm_implies_clips_in_preview_and_export`, `test_vlm_line_dropped_when_its_clip_is_not_in_the_export` |
| I2 | `manage export-download <id> --dest DIR` skips `_private/`, rewrites `data.yaml` `path:` to the absolute `DIR/yolo`, and fills `<DEST>` in README.txt. It refuses unfinished exports and exports whose consent was withdrawn. The generated `README.txt` holds the Ultralytics and LLaMA-Factory commands. | `test_export_download_makes_data_yaml_absolute` |
| I3 | `remap_label` → (text, dropped, problem). A row needs exactly 5 fields, finite values, an integer class, 0≤xc,yc≤1 and 0<w,h≤1; overshoot up to 1e-6 is clipped. An invalid row, or a non-empty file with only unmapped classes, quarantines the sample (`quarantined_samples`). Unmapped rows are counted in `dropped_boxes`. | `test_remap_label_validates_rows`, `test_invalid_label_quarantines_the_sample` |
| I5 | A rendition is used only when the original is not H.264 and `detail.src_etag` equals the selected original's ETag. Copies use `CopySourceIfMatch`; moto ignores it, so a HEAD ETag check comes first. | `test_stale_rendition_is_never_exported`, updated `test_clip_rendition_used_only_when_original_is_not_h264` |
| I6 | Expected frames come from meta `yolo_export.exported_frames`. Each missing label or image becomes a `missing` entry with `frame`, and the export is `partial`. Events without weak labels go to `no_weak_labels`. | `test_missing_sampled_frames_make_the_export_partial`, `test_unavailable_images_are_missing_not_silently_skipped` |
| I7 | `take_snapshot` runs at creation and stores in `Export.request.snapshot` the ids, splits, artifact key+ETag+bytes, AI run id+digest, feedback ids and verdicts. The build reads only the snapshot. A changed source gives `changed_since_request`. Consent is rechecked at build start (events dropped and counted) and before publishing (fail, prefix deleted). | `test_export_builds_the_snapshot_taken_at_creation`, `test_consent_withdrawn_before_build_drops_the_events`, `test_consent_withdrawn_during_build_fails_and_removes_files` |
| I8/I9 | `claim_export` runs `UPDATE … WHERE state='queued' RETURNING id`. A lease thread sends a heartbeat every 30 s. The final state is written only while the worker still owns the export; a lost lease never publishes. `sweep_stale_exports(session, now)` fails running exports with no heartbeat for over 5 min and queued exports older than 30 min. The 1 h rule is gone. app.py passes `engine=` to `build_loops`, which runs the 60 s exports loop. | `test_claim_is_atomic_and_exclusive`, `test_build_records_its_worker_and_heartbeat`, `test_a_build_that_lost_its_lease_never_publishes`, `test_sweep_uses_heartbeats_not_creation_time` |
| I10 | `vlm/{train,val,test}.jsonl` plus `vlm/dataset_info.json` (sharegpt; messages/videos; role/content/user/assistant tags) register `homeguard_<split>` for each non-empty split. Media paths are relative to the export root; the manifest records `vlm.media_dir: "."`. | `test_vlm_files_per_split_with_llamafactory_registration` |
| I11 | The manifest carries `counts` per format and split, plus `warnings` for empty splits and for fewer than 3 groups (the preview gives the groups warning too). An empty YOLO val or test split points at train, with "validation reuses training data". | `test_counts_per_format_and_split_and_small_set_warnings`, `test_preview_warns_about_small_sets` |
| I12 | Each item carries `yolo{source, model, class_schema}`, `ai{run_id, status, prompt_version, model, input}` and `owner_feedback[{feedback_id, verdict, received_utc}]`. The ids are opaque and the text is redacted. | `test_item_provenance` |
| M2 | `prompt_redacted` is true when privacy filtering changed or replaced the prompt. `prompt_source` is teacher or placeholder. | `test_vlm_prompt_redaction_flag` |
| M3 | Builds run on `app.state.export_executor` (`ThreadPoolExecutor(2)`). Items, mapping and JSONL are spooled to temp files, then uploaded; events are read in pages of 200. | `test_exports_run_on_a_bounded_executor` |
| M4 | A source over 5 GB becomes `missing` {reason `too_large`, note "too large for server-side copy"}. | `test_objects_too_large_for_server_side_copy_are_missing` |
| M5 | `clip_sha256` is streamed once from the copied object, never taken from the ETag. `clip_hash` is "sha256" or "not_exported". | `test_clip_hash_streamed_or_marked_not_exported` |
| `<video>` | `<video>`, `<image>` and `<audio>` tokens are stripped from the prompt and the assistant JSON, then exactly one `<video>` is prepended. | `test_media_tokens_normalised_to_one_video` |

Other notes:
- The manifest `schema_version` is now 2. The root `vlm.jsonl` is gone.
- `routes.studio.default_export_runner` was removed; `app.state.export_runner = None` means the executor.
- The old tests were updated to the new behaviour.
- Exports queued before this round have no snapshot. Building one fails with "start it again", and labelers do not see it.
