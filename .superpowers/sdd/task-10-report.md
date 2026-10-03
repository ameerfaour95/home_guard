# Task 10 report: event routes, labeler pseudonyms, density / review-count / fleet activity

**Status:** complete. Commit `d83260c` "Admin cloud: event history, detail, sampled YOLO boxes, review state, labeler pseudonyms".

## Tests
- RED first: the new tests failed with 501 from the stubs.
- GREEN: `tests/cloud/test_event_routes.py` has 21 tests, all passing.
- Full suite (exact command from constraints.md): **228 passed** in the working tree. That count includes another agent's uncommitted indexer/keys changes. The commit on its own, checked in a clean detached worktree at `d83260c`: **219 passed**.

## Files
- `home_guard_project/cloud/pseudonym.py` (new). `customer(secret, id)` gives `customer-<6 hex>` and `camera(secret, site, camera)` gives `cam-<6 hex>`. Both are HMAC-SHA256 under a sub-key derived from `settings.jwt_secret` with HMAC and a domain label. It is never plain sha256(secret+id).
- `home_guard_project/cloud/routes/events.py`: every events route plus `fleet_activity_density()`.
- `home_guard_project/cloud/routes/fleet.py`: `GET /v1/fleet/activity` now calls `fleet_activity_density`.
- `tests/cloud/builders.py`: adds `index_fixture_bucket(session, s3, site, customer_name, consent_training, now)`.
- `tests/cloud/test_event_routes.py` (new).
- `docs/admin/openapi.json`: regenerated. The **only** diff is the amended `limit` on `GET /v1/events` (default 50 → 100, plus minimum 1 and maximum 500). `progress.md` had already planned this ("Task 10 sets default 100 max 500 — tell exe"). Route docstrings were turned into comments so no `description` fields leaked into the frozen OpenAPI. `schemas.py` is untouched.

## Behaviour
- **List.** Keyset paging on `(start_ts DESC, id DESC)`. The cursor is urlsafe base64 of `"<repr(start_ts)>:<id>"`. A bad or non-finite cursor returns 400.
  - Filters: site, customer_id, camera, kind, `ai` (`completeness->>'ai'`), `verdict` (owner_verdicts JSONB `@>`), `q` (`websearch_to_tsquery('simple')` against `Event.search`), from_utc/to_utc on start, reviewed/flagged (a missing ReviewState counts as false).
  - `filter=`: `false_alarm`, `ai_dismissed_person`, `ai_failed`, `real_but_wrong`, `low_conf` (max of `class_max_conf` values below 0.45, computed with `jsonb_each_text`), `paused`. An unknown key returns 400. They live in `events.BUILTIN_FILTERS`, ready for Task 13 to move.
  - `thumbnail_url` is set when an available `thumbnail` artifact exists.
  - `timezone` comes from the customer.
  - Missing completeness keys get defaults.
- **Detail.**
  - `ai_runs`: guard runs only.
  - `feedback`: rows linked to the event. If `received_at` is null, `received_utc` falls back to the event start.
  - `artifacts`: include `detail`.
  - `dispatch`: a `DispatchOut` with `sent` true when any `results[].ok` is true or a `whatsapp.sent` is true, false when such fields exist but say no, otherwise null. It is null when unknown and always null for labelers.
  - `raw_meta`: the newest RawRevision of the training meta, otherwise the production meta.
  - `event_view` audit: at most one row per (staff, event) in any 10-minute window. The check compares the last audit row against the app clock.
- **Detections.** Only when `completeness.boxes == "sampled"`.
  - Sampled frames come from the applied meta revision, training copy first. Label keys are `<root>_<site>/<label_path>`, matched against indexed `yolo_label` artifacts.
  - The label text is cached in an LRU keyed by `(s3_key, etag)`, size 512.
  - Boxes are YOLO `cls xc yc w h` converted to clamped normalised xyxy, with `label` from `COCO_NAMES` and `conf=None`. `t_sec = frame_index / fps`.
  - Status per frame: an empty file is `ran_empty`. A label artifact that is missing, unavailable or unreadable is `not_run`.
  - If the app has no S3, the route returns 503. Unsampled events return `{"provenance": "none", "model": null, "frames": []}`.
- **Review PATCH.** A PostgreSQL `INSERT … ON CONFLICT DO UPDATE` of only the provided fields plus `by`/`at`, so it is race-safe. It writes a `review` audit row with the changed fields as detail and returns an `EventSummary`. An empty body is a no-op.
- **Density.**
  - Buckets are aligned to the hour or day in UTC and cover `[floor(from), ceil(to))`. `to <= from` or more than 5000 buckets returns 400.
  - Rows are Camera rows ∪ cameras seen in the device's events ∪ heartbeat cameras, including zero rows, sorted by name.
  - `timezone` is the customer's when exactly one customer is in scope, otherwise "UTC".
  - `false_alarms` counts events whose owner_verdicts contain `false_alarm`.
- **Review count.** `unreviewed_24h` counts events that started in the last 24 h (app clock) and are not reviewed. `flagged_open` counts flagged and not reviewed. Both are scoped to what the caller may see.
- **Fleet activity.** Admin and support only. One `"fleet"` row of `hours` hourly buckets, where the last bucket is the hour containing now.
- **Labelers.**
  - They only see events of customers with `consent_training = true`. Everything else returns 404 or empty: list, detail, detections, review, density and review-count.
  - `customer_name` and `site` show the customer pseudonym; `camera` shows the camera pseudonym. `customer_id` stays the numeric id, as the contract requires an int.
  - Site and camera filters take the pseudonyms the labeler sees; real names match nothing.
  - `raw_meta` has every `dispatch` key removed at any depth. `owner_feedback[*]` loses `raw_text`, `note`, `from` and `chat_id`. `camera_name` is replaced by the pseudonym. `teacher.prompt` is kept.
  - Feedback `raw_text` and `note` are blanked.
  - Artifact `s3_key`s have the site segment and camera name replaced by the pseudonyms.

## Concerns / notes for the controller
1. **OpenAPI changed** for the amended `limit` only. The exe should send `limit ≤ 500` and expect a default page of 100.
2. **Multi-device density naming.** When more than one device is in scope (for example admin with no site or customer filter), rows are named `<site>/<camera>` so that the same camera name on two sites does not merge. With a single device in scope, the row is the bare camera name. The exe should treat `camera` as an opaque label.
3. **Labeler camera-pseudonym filter** on `/events` scans distinct `(site, camera)` over the visible events. That is fine at phase-1 scale but is a full scan on a very large table.
4. **6-hex pseudonyms (24 bits)** match the brief. The collision risk is negligible at fleet scale, but they are not unique by construction.
5. **Beyond the brief:** I also pseudonymise `raw_meta.camera_name` and artifact `s3_key`s for labelers, because the site appears in every key. `raw_meta` paths (`clip_path` and similar) and the teacher prompt still contain the real **camera** name, which the brief keeps. They never contain the site or customer.
6. **`event_view` batching** reads the audit ts (wall clock) against `app.state.clock`. In production both are the same clock.
7. Another agent had uncommitted changes in `indexer.py`, `keys.py`, `test_indexer.py` and `test_keys.py` during this task. I did not stage them. My commit is green on its own: 219 passed in a clean worktree.
