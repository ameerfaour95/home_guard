# HomeGuardAdmin.exe — Round 3

Implemented on `admin-console-ui` in the existing `home_guard_admin_ui` worktree. Review, Studio, the expanded palette, Audit, and Task 2b integration are delivered and tested. **Live collection browsing, export consent preview, and audit detail JSON remain limited by three missing contract capabilities**, described below. Their UI does not invent routes or report an unverified consent check as successful.

## Merge and boundaries

Merged local `admin-console` first, in `23d7f0c`. Resolved `pyproject.toml` by retaining both the admin and cloud dependency groups and the cloud pytest settings; regenerated `uv.lock` with `uv --system-certs lock`. Read the merged `docs/admin/openapi.json` and `home_guard_project/cloud/schemas.py` as the authoritative contract.

No cloud source edits after that merge, no push, SSH, AWS access, or edits in another worktree. The three pre-existing `codex_exe_r*_run.log` files remain untracked. No sub-agents were used.

## Delivered behavior

- **Fleet:** one `GET /v1/fleet/activity?hours=24`, 24 hourly bars, proportional event heights with a minimum scale of five events, proportional alert segments, false-alarm markers, faint baseline, and centered labels every three hours. Hover wording is tested, including `14:00–15:00 · 18 events · 2 alerts · 1 false alarm`.
- **Contract amendment:** timeline density makes one aggregate request and includes silent cameras; the Review badge uses `review-count`. Event rows, playback clocks, and AI records use `EventSummary.timezone`, including labelers. Nullable `ArtifactOut.detail` is supported. Event 101 now has a real six-tile filmstrip sprite wired to scrubber hover.
- **Review:** fleet-wide cursor-paged stream, saved-view selector and kind/AI/owner-verdict/reviewed/flagged filters at left, virtualized compact timeline rows, and the existing player, overlays, AI evidence, feedback, and role gates at right. Wide layouts place recording and AI alongside one another; compact layouts stack them. Rows retain thumbnails, completeness indicators, review/flag state, and owner verdicts.
- **Keyboard workflow:** j/k navigate and play muted at 1×; r optimistically marks reviewed and advances; f optimistically toggles the flag; c opens a collection picker which remembers the last successful choice for the session. Failed mutations restore the before-image and show an event-specific toast. Successful changes remove rows that no longer match the selected review/flag filter. Typing in filters does not invoke review shortcuts. Event-ID jumps outside the loaded page act on the displayed event. No 1–5 verdict shortcuts were added.
- **Progress:** the unfiltered queue reports completed reviews against the initial 24-hour unreviewed count. Refined views report loaded matches alongside the fleet count without claiming an unavailable filtered total. The rail badge updates after successful review mutations.
- **Ctrl+K:** loaded customers, devices and cameras, server-supplied saved filters, navigation commands, export, sign out, and `#1234` event jumps. Exact/prefix matches rank above substring and fuzzy matches. `>` limits results to commands. Labelers can use the palette without customer/device identity entries.
- **Studio:** server-supplied saved filters, lazy match checks, collections with name/description/count/creator/date, creation, a virtualized thumbnail grid with completeness indicators, multi-selection removal, and collection addition from Review. Labelers land here; support has no export controls.
- **Export wizard:** collection/name/formats, balanced train/validation/test sliders, household/day grouping explanation, fallback-AI option off by default, consent exclusion summary, and confirmation. Validation rejects malformed slugs, empty/unknown formats, invalid split keys/values, and totals other than 100%. The demo excludes nonconsenting events and, by default, fallback events; unknown consent cannot be confirmed.
- **Export history:** queued/running/ready/partial/failed chips, version, item count, creator/time, errors, clipboard S3 path, and opening the returned HTTP(S) `manifest_url`. Active jobs refresh every 15 seconds while history is visible. Demo manifest URLs are absent; the URL action is verified with a supplied URL and a mocked browser opener.
- **Audit:** admin-only virtualized table, staff/customer/action filters, cursor paging and scroll loading, stale-query rejection, plain-language actions, and a row drawer. The drawer shows the actual entry JSON and explicitly identifies the missing server detail field.
- **States and performance:** API calls, fixture reads and thumbnail loads run in workers. Review reuses timeline loading/empty/error states; Studio and Audit have skeletons and retryable failures; Audit also has a dedicated filtered-empty state. Lists use Qt models and views.

## Contract questions and live limitations

1. **Collection membership reads:** OpenAPI defines collection list/create and item POST/DELETE, but no item GET and no `collection_id` filter on `/v1/events`. Please add a cursor-paged collection-members route returning event summaries, or document a collection filter on events. HTTP collection grids currently show an explicit unavailable state. Only the demo backend has the deliberately named `demo_collection_events` helper; no fictional HTTP route is sent.
2. **Consent preview:** collection members and per-event training eligibility are needed before confirmation. `CustomerOut.consent_training` is available to identity-authorized roles, but event summaries do not expose it to labelers and there is no export-preview route. Please add an identity-free preview response with included/excluded IDs, counts and reasons, ideally against the proposed `ExportRequest`. Until then, live wizard confirmation is disabled at the consent step. The existing create-export HTTP method is implemented and wire-tested. Please also clarify whether `include_fallback_ai=false` excludes whole events or only fallback AI records from VLM output; the demo currently excludes whole fallback events.
3. **Audit detail:** both the merged schema and OpenAPI omit `AuditEntry.detail`; there is no audit-entry detail route. Please add the optional dictionary or a detail GET. The current drawer labels its JSON as the entry itself, not missing detail data.
4. **Saved-filter totals:** `EventPage` contains only `items` and `next_cursor`. A lazy `limit=1` check therefore displays `0`, `1`, or `1+`, never a fabricated exact total. An optional filtered-total field would permit exact counts without cursor walks.

The merged cloud implementations of event, Studio, Audit and aggregate routes currently return 501. No live service integration was claimed or attempted. These route implementations remain cloud-team work; the desktop wire behavior is covered by MockTransport.

## Files and fixtures

| Area | Files under `home_guard_project/admin/` |
|---|---|
| Review | `review.py`, shared `timeline.py`, `timeline_model.py`, `event_view.py` |
| Studio and export | `studio.py`, `collections.py`, `export_wizard.py`, `export_logic.py` |
| Audit and shared tables | `audit.py`, `widgets/data_table.py` |
| Shell and visual components | `shell.py`, `widgets/palette.py`, `widgets/activity.py`, `theme.py` |
| Contract and transport | `models.py`, `backend.py`, `http_backend.py`, `demo_backend.py`, `demo_studio.py` |
| Demo generation | `make_demo_r3.py`, updated `make_demo_media.py`, `demo_data/` |
| Verification | `screenshots_r3.py`, expanded `__main__.py` smoke flow, `tests/admin/test_round3.py` |

New JSON fixtures cover filters, collections, collection membership, five export states, audit pages, density, review counts, Fleet activity and a camera catalog with a silent camera. Studio mutations persist in the demo session. The demo retains 96 curated events; Fleet rollups illustrate the larger fleet workload. Summaries now describe camera activity; failure and fallback explanations remain in AI state displays. Parsed and raw demo answers were updated consistently. `make_demo_r3` regenerates the Round 3 fixtures and filmstrip without modifying cloud code.

## Verification and packaging

```powershell
uv run --only-group admin --no-sync pytest tests/admin -q --tb=short
```

**312 passed in 10.63 seconds** on the final source. The prepared admin environment avoids synchronizing the unrelated box/ML dependency set. Added coverage includes keyboard playback/navigation, review success and optimistic rollback, out-of-page event mutations, palette ranking/execution, export validation and consent exclusion, collection add/remove requests, Audit cursor paging and late-response rejection, aggregate-only Fleet loading, histogram math/tooltips, labeler timezone/landing, nullable filmstrip metadata, all new HTTP routes, export URL actions, support/admin restrictions, and retryable offline states. Existing Round 1/2 tests remain covered; the old labeler-palette assertion was updated for its new authorized functionality.

```powershell
uv run --only-group admin --no-sync python -m home_guard_project.admin --demo --smoke-test
uv run --only-group admin --no-sync python -m home_guard_project.admin --smoke-test
home_guard_project/admin/build_admin_exe.ps1 -SkipSync
```

Both **source smoke tests exited 0**. The expanded demo smoke decodes/seeks a real clip, verifies boxes, opens Review, opens Studio, checks the export consent preview, and loads Audit. The sign-in smoke verifies startup and the bundled icon.

**PyInstaller build succeeded.** Final output:

- `dist/HomeGuardAdmin/HomeGuardAdmin.exe`: **5,518,049 bytes / 5.26 MiB**.
- Complete one-directory bundle: **144,865,053 bytes / 138.15 MiB**. Distribute the entire directory.
- Local build log: `build/admin-r3-build.log`.

**Packaged execution is unverified:** Windows rejected both attempted EXE smoke launches with `An Application Control policy has blocked this file.` No policy changes or bypasses were attempted. This is a launch restriction, not a successful smoke-test result.

## Screenshots

Generated with `uv run --only-group admin --no-sync python -m home_guard_project.admin.screenshots_r3`. All 13 final screenshots were opened and visually inspected. Corrections included the wide-to-compact minimum-size constraint, a clipped Review heading, excessive splitter contrast, completeness/owner-verdict context, Studio row density, visible checkbox indicators, and fitting all consent exclusions. The harness asserts the compact Review is exactly 1366×768 and captures an actual decoded mid-clip frame.

| Screenshot | Size |
|---|---|
| [Review split view, wide](screenshots/r3-review-1920.png) | 1920×1080 |
| [Review split view, compact](screenshots/r3-review-1366.png) | 1366×768 |
| [Palette commands](screenshots/r3-palette-commands.png) | 640×432 |
| [Studio saved filters](screenshots/r3-studio-filters.png) | 1366×768 |
| [Collection grid](screenshots/r3-collection-grid.png) | 1366×768 |
| [Export: dataset](screenshots/r3-export-1-dataset.png) | 760×657 |
| [Export: split and evidence](screenshots/r3-export-2-split.png) | 760×657 |
| [Export: consent and confirmation](screenshots/r3-export-3-consent.png) | 760×657 |
| [Export history, all five states](screenshots/r3-export-history.png) | 1366×768 |
| [Audit table](screenshots/r3-audit-table.png) | 1366×768 |
| [Audit drawer](screenshots/r3-audit-drawer.png) | 1366×768 |
| [Fleet histogram](screenshots/r3-fleet-histogram.png) | 1920×1080 |
| [Labeler landing on Studio](screenshots/r3-labeler-studio.png) | 1366×768 |

## Commits

- `23d7f0c` — Merge `admin-console`, preserving both dependency groups.
- `4b6d1ac` — Task 2b models/transports/aggregates, realistic fixtures and filmstrip.
- `c957c91` — Review, Studio, palette, Audit and behavioral tests.
- `fbd3c18` — Evidence-row polish and retryable Studio/Audit states.
- Final report/capture commit — this force-added report, all 13 force-added screenshots, and their harness.

Every commit ends with `Co-Authored-By: Codex <noreply@openai.com>`.
