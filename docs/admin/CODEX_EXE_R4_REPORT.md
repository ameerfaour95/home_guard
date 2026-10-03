# HomeGuardAdmin — Round 4

Worktree: `home_guard_admin_ui`, branch `admin-console-ui`. Merged local `admin-console` first (`2d378d0`), then its newly completed media/privacy work (`48f8c74`, through `d6f9ce5`) during integration. All Cloud changes came from those merges; this branch made no Cloud source edits. No push, SSH, AWS access, or other worktree changes.

All Critical and Important review findings have client fixes. Minor findings are fixed except the server logout endpoint and the broad formatting/script relocation portion of M8, explained below. Real-server media playback passes after the second merge; packaged validation is recorded separately below.

## Review findings, fixes, and regressions

Tests below are in `tests/admin/test_round4.py` unless another file is named. The initial tests reproduced 14 failures before implementation. An isolated copy of the merged pre-fix code under `build/r4-review-baseline` subsequently produced **28 failures / 1 pass** against the 29-test regression suite at that checkpoint. Single-flight refresh already passed and was preserved. The current suite adds audit-detail, thumbnail-501, hidden-view and stale-response/debounce coverage.

| Finding | Fix | Regression evidence |
|---|---|---|
| C1 — Windows TLS trust | `SSLContext(PROTOCOL_TLS_CLIENT)` loads certifi, Windows default certificates, then optional certificate overrides. Verification and hostname checks remain enabled. Invalid certificate configuration and certificate-validation causes have a designed `TlsError`; startup retains the sign-in window. | `test_c1_windows_store_and_verification` verifies `CERT_REQUIRED` and inclusion of system CA certificates; `test_c1_invalid_environment_cert_is_designed`. |
| I1 — refresh-token reuse | Every unsuccessful refresh discards the token pair and raises “Your session needs a new sign-in”. No transport retry of a single-use refresh token. Last email remains in preferences. | `test_i1_refresh_is_burned_after_ambiguous_failure` covers transport loss and 503; eight-thread `test_m9_concurrent_401s_single_flight` verifies one refresh. |
| I2 — plaintext login | Reject HTTP except exact `localhost` / `127.0.0.1`; reject other schemes and URL credentials. Configuration errors appear on sign-in. | `test_i2_refuse_insecure_server`. |
| I3 — skipped review event | Pending navigation retains event identity plus query generation. The incoming cursor page supplies the next event ID, resolved after any removals. | `test_i3_page_landing_uses_event_id` reproduces PATCH completion before page arrival. |
| I4 — dropped r/f | Shared `ReviewController` serializes writes and keeps pending desired values per event/field, last action wins. Immutable before-images support rollback by ID. All three mutation entry points use it. | `test_i4_fast_review_actions_are_not_lost`; existing rollback and keyboard tests in rounds 2/3. |
| I5 — stale audited media | 150 ms event-open debounce, cancellation between evidence requests, cancellation on leaving the view, generation checks, lazy visible AI frames/raw disclosures. List/grid thumbnails request only visible rows after a debounce; no offscreen prefetch. | `test_i5_stale_evidence_does_not_request_media`, `test_thumbnails_only_request_visible_rows`. |
| I6 — unbounded exit | Signal cancellation, clear pending jobs, wait at most 2 seconds. Close the backend on a drained pool; if a socket remains stuck, terminate the process instead of letting Qt perform an unbounded destructor join. | `test_i6_shutdown_has_two_second_bound`; real child-process `test_application_shutdown_with_stalled_worker_is_bounded` (old version exceeds 8 seconds; fixed version exits within the bound). |
| I7 — invisible worker errors | Rotating `%LOCALAPPDATA%/HomeGuardAdmin/logs/admin.log`: five files total, 1 MiB each. Redaction filter removes credentials/URLs; traceback diagnostics contain stack locations and exception type, not arbitrary exception messages or source text. Token fields omit repr. UI exception hook shows a status message; worker bugs use a local-error message. | `test_i7_token_repr_and_redaction`, `test_log_redacts_quoted_secrets_and_arbitrary_exception_messages`, `test_worker_exception_is_logged_and_designed`. |
| I8 — unknown enums | Unknown presentation enum values decode as `unknown` per item; display lookups have fallbacks. Staff roles still validate strictly. | `test_i8_unknown_enums_keep_page`; existing malformed-shape tests retain strict validation. |
| I9 — demo-only collections/exports | `collection_events` and `export_preview` are protocol methods implemented by HTTP and demo backends. UI no longer calls demo membership/consent helpers. Typed 501 states; server preview governs confirmation. | `test_contract_2c_http_routes`, `test_contract_preview_uses_protocol_and_enables_create`; real HTTP integration tests. |
| M1 — auth/rate-limit messages | Separate login/session messages, general request rate-limit copy, dialog/grid/count auth failures forwarded to session expiry. | `test_m1_collection_dialog_expires_session`; existing sign-in/session tests. |
| M2 — hidden/forbidden Fleet polling | Pause polling while hidden; stop on 403. | `test_m2_fleet_hidden_and_forbidden_stops_poll`. |
| M3 — lost refresh requests | Dirty flags retain badge and Studio refresh requests received during a job. | `test_m3_badge_refresh_retained_while_busy`. |
| M4 — boxes outside coverage | Nearest-frame lookup rejects distances beyond 1.5 times median positive frame spacing; no boxes paint before a video frame exists. | `test_m4_overlay_outside_coverage`; updated round-2 alignment test. |
| M5 — widget leaks | Dialogs delete on close; filmstrip tooltip belongs to its slider. | `test_m5_dialog_and_tooltip_ownership`; test teardown now tolerates already-deleted widgets. |
| M6 — toast styling/lifetime | Repolish changed styles and restart a single owned timer. | `test_m6_toast_restarts_timer_and_repolishes`. |
| M7 — old-user requests / logout | Request session epoch prevents a pending old-user request from retrying with a new user's tokens. Sign-out invalidates the epoch. **Server token revocation is deferred:** no logout route exists in the merged contract/service, and editing Cloud is prohibited. | `test_m7_old_session_cannot_retry_as_new_user`. |
| M8 — inconsistent mutation paths / cleanup | Shared `ReviewController` handles review, timeline and event-view saves. Dev server and existing generator/screenshot modules are explicitly excluded from packaging. **Broad reformatting and relocating existing dev scripts are deferred:** moving their resource-relative paths adds unrelated churn; packaging exclusion addresses the shipping concern directly. | Fast-action, rollback and keyboard tests; inspected executable archive exclusions. |
| M9 — missing probes / expired media | Added lost-response, concurrent refresh, TLS store, paging race, HTTP/protocol and real-server tests. Media errors request fresh access once for the current event; no infinite retry. | `test_media_error_requests_access_only_once`, plus tests listed above. |

## Contract 2c

- `EventPage.total` and `total_capped` are modeled. Saved-filter counts use `with_total=true`; capped results display `10,000+`. `test_unknown_saved_filter_count_is_not_a_one_plus_estimate` covers this.
- `AuditEntry.detail` survives decoding and appears in drawer JSON. `test_audit_detail_survives_wire_decode_and_drawer` covers it.
- Collection grids consume cursor pages from `GET /studio/collections/{id}/items` through `AdminBackend`.
- The consent step calls `POST /studio/exports/preview`, shows included count, exclusions grouped by reason, split counts, house/day groups and warnings. Confirmation enables Create export only after a successful nonempty preview and acknowledgement.
- `include_fallback_ai=false` excludes fallback/failed answers only from `vlm.jsonl`; events still contribute clips/YOLO when selected. Copy and demo behavior reflect this. Removed the obsolete UI consent partitioner.
- Event calls respect the service's default 100 / maximum 500. UI timeline pages remain 25; collection pages default to 100.
- The merged OpenAPI document was used unchanged. Live Studio navigation remains unavailable until its list routes are implemented, even though collection-items and preview endpoints work with known seeded IDs.

## Local service and live results

`home_guard_project/admin/dev_server.py` starts embedded pgserver under `%LOCALAPPDATA%/HomeGuardAdmin/devdb`, applies Cloud migrations, and seeds the real Cloud SQLAlchemy models: three customers, four devices/heartbeats, 96 events, cameras, artifacts, AI runs, feedback, review states, collections/items and append-only audit rows. Sampled YOLO labels and applied metadata are seeded so the real detections route reads them. Reseeding replaces derived fixture pointers so a restarted in-memory S3 service is not asked for old keys. Cloud's redacted-search backfill is used. All media is synthetic.

It serves the unmodified `create_app(...)` at `127.0.0.1:8600`, with an in-memory `ThreadedMotoServer` at `127.0.0.1:8601`. Explicit dummy credentials and a loopback endpoint ensure no AWS access. `moto[server]` was added to the Cloud dependency group because `moto[s3]` alone lacks Flask. The Windows runtime's broken SSL key-log hook is disabled in this dev process only; TLS verification is not disabled.

Launch (PowerShell):

```powershell
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
$env:UV_SYSTEM_CERTS = '1'
uv sync --group cloud --group admin --system-certs
uv run --group cloud --group admin --system-certs python -m home_guard_project.admin.dev_server
```

The helper prints fresh admin/support/labeler email/password/TOTP secrets. For automation it writes `build/dev-login.json`; that file is ignored and not committed. Local service processes are left running for inspection. Generated credentials are deliberately absent from this report and admin logs. Repeated sign-ins must respect the service's single-use TOTP policy.

| Real route/flow | Result |
|---|---|
| TOTP sign-in, `/me`, Fleet, customer list/detail | Passed using `HttpBackend` and actual offscreen Qt sign-in. |
| Event pages, customer timeline, density, fleet activity, review counts | Passed; screenshots show real database responses. |
| Event details, AI record, sampled detections, PATCH review | Passed; saved review state verified by a new GET. |
| Labeler sign-in and role navigation | Passed; customer/camera pseudonyms verified. |
| Event exact totals, collection 1 items, export preview | Passed through the real HTTP routes. |
| Audited artifact access, presigned API media playback, input frames and thumbnail redirects | Passed after the second merge. First decoded frame, seeking and nonempty detection boxes are asserted; no endpoint is mocked or replaced. |
| Independent moto presigned URL → QMediaPlayer first frame + real detections overlay | Passed as additional storage/decoder coverage. API media access is also tested separately. |
| Packaged exe against local Cloud | **Passed.** Receipt: 4 fleet devices, 25 timeline events, 36 detection frames, `media_unavailable=false`, `frame_decoded=true`, `review_saved=true`. The test waits for the exact spawned PID to signal readiness before supplying a fresh single-use TOTP. Earlier cold-start attempts consumed codes twice and one intermediate binary was rejected by Windows Application Control; the final build passed without any policy change. |

The following routes were called live and returned **501**:

```text
GET    /v1/studio/filters
GET    /v1/studio/collections
POST   /v1/studio/collections
POST   /v1/studio/collections/1/items
DELETE /v1/studio/collections/1/items
GET    /v1/studio/exports
GET    /v1/studio/exports/1
POST   /v1/studio/exports
GET    /v1/audit
```

Studio and Audit use designed “not available yet” screens. Artifact access and thumbnail endpoints initially returned 501, but work after merging the newly implemented Cloud routes. Their designed fallback remains covered by regressions. Server logout is absent, rather than a 501 route.

## Validation and artifacts

- `uv run --group admin --group cloud --system-certs pytest tests/admin -q`: **344 passed, 5 skipped**. The five live tests are skipped unless `HG_ADMIN_IT=1`.
- `$env:HG_ADMIN_IT='1'; .venv/Scripts/python.exe -m pytest tests/admin/test_live_server.py -q`: **5 passed in 37.10 seconds**, including the packaged exe and real API media playback.
- Executable smoke probe is in the opt-in integration test and writes a nonsecret checkpoint receipt. Windows error 4551 is explicitly reported as an xfail when encountered; other failures remain failures.
- `git diff --check`: clean after whitespace cleanup. No post-merge changes under `home_guard_project/cloud/`.
- Build: `home_guard_project/admin/build_admin_exe.ps1 -SkipSync`. Final size is recorded below.
- Inspected the embedded PYZ archive: no dev server, Cloud package, moto, pgserver or screenshot module. No source dependency on these was added to the exe.

Live screenshots (1920×1080, inspected visually; all are the production UI backed by real HTTP responses):

- [Fleet](screenshots/r4-live-fleet.png)
- [Customer timeline and density](screenshots/r4-live-timeline.png)
- [Event playback, detection boxes and AI record](screenshots/r4-live-event.png)

The event screenshot uses the synthetic seed clip returned through the real audited media-access endpoint and a moto presigned URL; it does not use DemoBackend. Server refresh-token revocation still requires a logout contract/route.

Final build: `dist/HomeGuardAdmin/HomeGuardAdmin.exe` — **9,689,353 bytes (9.24 MiB)**. Complete one-directory bundle: **192,419,089 bytes (183.51 MiB)**. The full directory is required to run the exe.
