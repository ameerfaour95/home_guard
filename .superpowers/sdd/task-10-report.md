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

## Fix round (Codex review)

Spec `.superpowers/sdd/fix-10.md`. Commits `43ca89e` (redaction module, migration 0005, indexer +
`manage redact-backfill`) and `f2591fe` (labeler projections, search guard, opaque media, findings 4-6).
`schemas.py` and `docs/admin/openapi.json` unchanged (contract test green).

RED first: the response-wide test was written before any code change and failed on `da4ec98`:
`test_no_labeler_response_names_the_household` -> `list leaks 'bian': ..."summary":"person at bianhouse near the
bian_ch2 gate, walking to daniel levi's c...`; `test_labeler_search_has_no_identity_oracle` -> labeler
`q=BianHouse` returned `[5, 4]`; `test_labeler_media_urls_do_not_name_the_household` -> `access url leaks 'bian':
...s3.amazonaws.com/dataset_bian/clips/bian_ch2/2026-10-03/bian_ch2_1791020177_alert.mp4`. The finding 4-6 tests
also failed first (empty cursor 200; density 10:50Z->10:10Z 200; concurrent views wrote 6 event_view rows).

| Finding | Change | Test | RED -> GREEN |
|---|---|---|---|
| 1-3 Critical: raw meta, free text, search and artifacts disclose the household | New `cloud/redact.py`: identity terms per device (site, every camera name from Camera rows / events / heartbeat, display names, customer name, Tailscale host) + variants (`_` / space / `-` / joined, camelCase) + distinctive parts (>= 4 chars, not digits, not on the stoplist; spec stoplist plus a few more generic words). Single-pass, case-insensitive, whole-word-ish (camelCase and `_` are boundaries) replacement; no re-redaction of inserted pseudonyms. Labeler responses are allowlisted projections: `summary`, `label`, `detected`, `owner_verdicts`, `alert_reason`, `ai_runs[].prompt/parsed/model/prompt_version` redacted (site/customer -> `customer-xxxxxx`, camera/display name -> `cam-xxxxxx`); `raw_meta` = the spec allowlist only, every string redacted; artifacts `s3_key = artifact-<id>`, `detail` only scalar fps/tile_w/tile_h/count/copy, `opaque_copy` rows never listed; feedback verdict/action/source/received_utc only. Search: migration 0005 (after 0004) adds `events.summary_redacted` + generated `search_redacted` tsvector + GIN index; the indexer nulls `summary_redacted` whenever it sets a summary and `redact.backfill` refills every NULL at the end of each pass; `manage redact-backfill [--all]`. Labeler `q` searches `search_redacted`; a `q` containing an identity term of any device in the labeler's scope returns an empty page. | `tests/cloud/test_labeler_privacy.py` (household `bian` / `bian_ch2` / `BianHouse` / `Daniel Levi` / `bian-box`; serialises list (plain, paged, searched), detail + detections of every event, density hour/day, review-count, review PATCH, collection items (plain, paged), export preview; asserts no identity term, no `dataset_`/`production_`/`clips/` path, no chat id, no owner text); `test_labeler_search_has_no_identity_oracle` (10 identity queries empty, `walking` still found); `tests/cloud/test_redact.py` (variants, single pass, camelCase, term sources, indexer fill, `redact-backfill` with and without `--all`) | RED as above -> GREEN |
| 7 (controller): presigned URLs name the household | Labeler `POST /artifacts/{id}/access` and the thumbnail redirect: server-side `S3.copy` (writable prefixes only, `MetadataDirective=REPLACE` so no stored Content-Disposition/metadata travels) to `admin_cache/opaque/<hmac(artifact id, etag)[:40]>.<ext>` (HMAC sub-key derived from the JWT secret) on first access, recorded as Artifact role `opaque_copy`, provenance `cloud`, detail `{"source_artifact_id": id}`, `event_id` NULL (never in event views); upsert so concurrent first accesses are safe; the copy is presigned with no filename/ResponseContentDisposition hints. Admin/support keep the real key. Extra: labelers get 403 for `meta`, `feedback`, `raw_answer` and `opaque_copy` artifacts (the meta/feedback JSON carry dispatch, chat ids and the owner's words, which bypassed every redaction above). | `test_labeler_media_urls_do_not_name_the_household`: URL and thumbnail Location carry no identity term and no `dataset_`/`production_`; copy bytes equal the clip; second access reuses the key; exactly one `opaque_copy` row; admin URL still real; labeler 403 on the meta artifact | RED (real key in URL) -> GREEN |
| 4 Important: event_view batching race | `_audit_view` takes `pg_advisory_xact_lock(hashtextextended('event_view:<staff>:<event>'))` before the check, rechecks under the lock, and stamps the row with the same app clock that decides the window | `test_event_view_audit_is_race_free`: 6 threads behind a barrier, `audit.record` slowed by 0.3 s -> exactly 1 row, `ts` == app clock | 6 rows -> 1 |
| 5 Important: invalid cursors reach PostgreSQL | Absent cursor = first page; empty or invalid -> 400 before any query: > 128 chars, bad base64, not exactly `<number>:<id>` (no extra parts, whitespace or sign on the id), non-finite ts, id outside 1..2^63-1. `events.id` is int4, so the keyset bound is compared as a bigint literal (2^63-1 returns 200 instead of the DataError the first attempt showed). Collection items share the path. | `test_invalid_cursors_are_400_before_any_query` (16 bad cursors incl. `""`, the 30-digit id and 2^63; 2^63-1 -> 200; collection items `""` and 2^63 -> 400) | `""` 200 -> 400 |
| 6 Minor: density validates rounded bounds | UTC-normalise the original endpoints (naive = UTC) and compare before aligning; `to <= from` -> 400 | `test_density_validates_the_requested_interval_not_the_rounded_one` (hour and day: reversed and equal endpoints inside one bucket -> 400; valid sub-bucket interval -> 200 with one bucket; naive start vs +03:00 end compared in UTC) | 200 -> 400 |

Decision (stored search copy): the indexer has no server secret, so `summary_redacted` uses neutral placeholders
(`customer-redacted` / `cam-redacted`) and serves search only. Responses redact the original text per request with
the current terms and the labeler's real pseudonyms, so what a labeler sees is never stale. A stale stored copy
(e.g. after a camera is added) can still hold a newly added name, but the `q` guard makes it unsearchable, and
`redact-backfill --all` refreshes it.

Full suite (exact command from constraints.md): **269 passed in 119.15s**. The tree includes Task 12's commits
`6b80967` and `6c98307`, which landed on the branch during this round.

Notes:
- `audit.retry_pending_notices` is still not scheduled; Task 14's background loops will call it.
- Opaque copies are not garbage-collected: when a source etag changes, the old copy stays under
  `admin_cache/opaque/`. A cache-cleanup task should drop `opaque_copy` rows whose source has a newer etag.
- File contents (video frames, the raw AI text file) are not redacted; raw answers are refused to labelers,
  video and images are the labeling material.
- `ExportPreview.excluded` still lists event ids of non-consenting customers with the reason (ids only, no
  identity; unchanged behaviour).

## Fix round 2 (Codex re-review)

Spec `.superpowers/sdd/fix-10b.md`, review `task-10-codex-rereview.md`. Commit `dceac30`. Design change:
labelers lose the capabilities that leaked (search, prompt text); redaction stays as defence in depth.
`schemas.py` and `docs/admin/openapi.json` are unchanged (contract test green; the preview's new labeler note is a
comment, not docstring, because route docstrings are in the frozen OpenAPI).

All new and changed tests were written first and run against `d6f9ce5` code: 14 failed, 2 passed (the two that
passed are a positive indexer check and the unchanged media-URL test).

| Probe (re-review) | Test (`tests/cloud/test_labeler_privacy.py` unless noted) | RED on old code | GREEN change |
|---|---|---|---|
| `q=walking OR bian` / `walking -bian` total 0 vs 2; `q=tail9c3&cursor=` 200 vs unknown host 400 | `test_labeler_search_is_refused_uniformly` (16 probes incl. OR/negation, identity, unknown words, empty cursor, `with_total`, bad cursor, bad filter; all `(400, {"detail": "Search is not available for this role"})`; admin search still works; `q=""` is 200) | `{'q': 'walking'}` -> 200 | `list_events`: labeler + non-empty `q` -> 400 first; identity early return (`_names_household`) and labeler `search_redacted` query removed |
| Search decided before any DB access | `test_labeler_search_is_refused_before_any_event_query` (SQL listener: no statement touches events/customers/devices/cameras) | 200 != 400 | same |
| with_total counted before the cursor was decoded | `test_cursor_and_filter_are_validated_before_any_count_or_lookup` (admin: empty/bad cursor with total, bad filter with site+camera, `MTow` = id 0; no events/customers/devices/cameras statement) | `{'with_total': True, 'cursor': ''}`: count ran first | filter and cursor validated at the top of `list_events`; `event_page` decodes before counting; collection items decode before the collection lookup |
| `MyBIANHome BIANHome` survive in summary, alert_reason, model_response, prompts, parsed | `test_acronym_and_case_run_spellings_are_redacted` (MyBIANHome, BIANHome, myBIANhome, XMLBian, BIAN2, BIANHOUSE; Bianca/BIANCA/Fabian untouched); `test_acronym_variants_never_reach_a_labeler`; response-wide test LEAKY/REASON now carry the variants and IDENTITY lists `mybianhome`, `bianhome` | `MyBIANHome` unchanged; detail leaked `mybianhome and bianhome` | `redact._START/_END` add boundaries: Upper->Upper+lower (acronym end), upper run (>=2) -> lower, letter<->digit |
| Prompt text reaches labelers | same detail test + response-wide test (`ai_runs[].prompt is None`, teacher keeps model/prompt_version/temperature only); `test_event_routes.py::test_labeler_sees_pseudonyms_and_no_dispatch` | prompt present | labeler `AiRunOut.prompt=None`; `_LABELER_META.teacher` drops `prompt` |
| `alert_command="BianHouse"`, `clip_start_local="Daniel Levi"` returned verbatim | `test_injected_enum_and_format_fields_are_not_shown_to_labelers` (DB-injected; valid `[call_owner]`/`suspicious` still shown) | `{'BianHouse'} == {None}` | labeler projection: `redact.alert_command` / `redact.label` (enum or null), `clip_start_local` null |
| Ingestion accepts those strings | `test_indexer_drops_values_outside_the_enums_and_formats`; `test_indexer_keeps_valid_enums_and_formats` | rows `('BianHouse', 'Daniel Levi', 'Daniel Levi')` | indexer stores `alert_command`/`label` only from their enums, `clip_start_local` only as `YYYY-MM-DD HH:MM[:SS[.f]][Z/offset]` or an import's `<sec>s` |
| Artifact enumeration: hidden available 403, hidden unavailable 410, missing 404 | `test_hidden_artifacts_are_indistinguishable_from_missing_ones` (non-consented household next to `bian`; available, unavailable, opaque_copy and missing ids, purposes training and support: identical `(404, body)`; media_denied audited with the real reason for the hidden keys; hidden event detail/detections/thumbnail identical 404) | `(26, 'training', 'This customer has not agreed to training use')` | `artifact_access`: for labelers, visibility (`_labeler_hidden_reason`: missing, no household, no consent, meta/feedback/raw_answer/opaque_copy) is decided before availability/consent; refused with the generic `Artifact not found` 404 and audited as `media_denied` (reason in detail, committed) |
| Export preview lists hidden event under `no_training_consent` | `test_export_preview_drops_hidden_events_silently` (mixed collection preview == preview of the visible events alone; admin still sees `no_training_consent`) | hidden ids in `excluded` | `select_export_items(..., labeler=True)` filters non-consented events in the query |
| Split bits `1110111111110110` identify `bian` offline | `test_export_split_is_not_an_offline_site_oracle` (one-event collection, 16 names, 0.5/0.5; bits match no unkeyed candidate, still deterministic); `test_assign_splits_is_keyed_and_group_aware`; `test_studio_routes.py` passes `secret=` | observed bits exactly `1110111111110110`; `assign_splits` had no `secret` | `assign_splits(items, name, split, *, secret)`: HMAC-SHA256 under `HMAC(jwt_secret, "home-guard-admin/export-split/v1")` over `name\0site\0day` (separate derivation label from pseudonyms); the preview passes `settings.jwt_secret` |

Changed expectations in existing tests: labeler search tests now expect the uniform 400; labeler meta-artifact access
and no-consent artifact access are 404 (was 403) in `test_labeler_privacy.py` and
`test_media_routes.py::test_labeler_needs_training_consent_and_creates_no_notice`.

Full suite (exact command from constraints.md): **286 passed in 119.71s**.

Notes:
- Collection item add/remove (`POST`/`DELETE /studio/collections/{id}/items`) are still 501 stubs; when Task 13
  implements them, labelers must get the same 404 for hidden and missing event ids (use `_load_one`-style
  visibility before existence).
- The export builder (Task 13) must call `assign_splits(..., secret=settings.jwt_secret)` so the preview and the
  export agree.
- `summary_redacted`/`search_redacted` are still filled by the indexer but unused for labelers; the stale-copy
  finding no longer reaches a labeler. `redact.Identity.mentions` is kept (tested) but no longer used by routes.


## Fix round 3 (privacy-pass3.md, fix-10c.md)

- A: `redact._CAMEL` splits identity sources at every case/digit boundary (lower->Upper, UPPER->Upper+lower, letter<->digit), so `OAKRIDGEHome` yields `OAKRIDGE Home`, `OAKRIDGE_Home`, `oakridgehome`, `OAKRIDGE`. Same tokenizer for all sources. Tests: unit variants + ingestion-to-response regression (`test_acronym_display_name_never_reaches_a_labeler`). An all-caps run followed by lower case with no capital (`BIANhome`) has no unambiguous source split; the matcher's boundaries still catch it in text.
- B: `redact.VERDICTS` / `verdict()`. Indexer: owner_feedback verdicts (meta) and Feedback.verdict outside the set are stored as "unknown" with IndexProblem "invalid verdict"; labelers also see out-of-vocabulary stored values as "unknown" (old rows, no migration). `verdict`, `kind`, `ai` filters (and density `kind`) are validated for every role: 400 before any query.
- C: density omits cameras without events in the interval for labelers (admin/support keep the zero rows).
- D: `media._labeler_lookup` is one SELECT (artifact joined to event/device/customer, consent + role predicates in WHERE). Hidden/missing/unavailable/opaque artifacts run the same statements; audit `media_denied` with reason `not_visible`, target `artifact/<id>`. Behavior change: event-less artifacts are no longer reachable by labelers.
- RED recorded first: 9 new tests failed (acronym x2, verdict ingest, filters, hidden-vs-missing work, density, 3 loop/pool tests); after the changes all pass. Existing hidden-artifact test updated for the uniform audit target.
