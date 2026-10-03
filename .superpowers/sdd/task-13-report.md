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
