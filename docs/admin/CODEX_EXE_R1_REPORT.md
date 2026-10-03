# HomeGuardAdmin.exe — Round 1

Built in `C:\Users\ameer\Ameer\home_guard_admin_ui` on `admin-console-ui`, 2026-10-03.

## Delivered

The native PySide6 desktop app has staff sign-in, a role-aware shell, a virtualized Fleet table, device details, customer headers with consent flags, and a fuzzy Ctrl+K customer/device palette. Review, Studio and Audit have designed release placeholders. Labelers receive only Review and Studio; their sessions never fetch the fleet or customer registry. Support does not receive Audit.

The dark and light palettes copy the exact company tokens from `box/app/theme.py` without importing the box app. Segoe UI is explicitly registered for Windows offscreen rendering. The standard native window frame retains Windows move, resize, snap and keyboard behavior.

Fleet sorts offline, critical, warning, unknown, healthy, with stable customer/site ordering within each severity. It has verdict and text filters, arrow selection, Enter to open a customer, `/` to focus filtering, hover previews, and 30-second refresh. Selection, filters and both scroll positions survive a refresh. Empty enrollment, no matches, loading skeleton, initial offline, cached offline and server-error states are separate. Failed refreshes retain the last successful snapshot. Compact detail views use horizontal table scrolling to preserve readable column widths.

The HTTP adapter uses the frozen `/v1` routes, typed dataclasses, Bearer auth, one automatic refresh/retry, and safe error messages. Network operations run in a QThreadPool and deliver results through QObject slots on the GUI thread. Tokens remain in memory; only the email is saved to `%APPDATA%/HomeGuardAdmin/prefs.json`. Password and code fields are cleared after each login attempt. Six TOTP boxes support auto-advance, backspace and full-code paste.

Seven hand-authored JSON files contain three fictional customers, four devices and one complete example event. Device reasons, counts, consent and cross-file IDs agree. The demo display clock starts at the fixture's `generated_utc`, so screenshots remain useful when opened later. Absolute timestamps use the customer's timezone; relative ages use UTC instants.

## File map

All application paths below are relative to `home_guard_project/admin/`.

| File | Purpose |
|---|---|
| `models.py` | Frozen response fields, nested dataclasses, validating JSON decoder, nullable values and contract defaults |
| `backend.py` | AdminBackend protocol and auth, permission, offline, server and rate-limit errors |
| `http_backend.py` | Synchronous httpx adapter, verified TLS, auth refresh lock and route mapping |
| `demo_backend.py`, `demo_data/*.json` | Replaceable JSON fixtures and demo role handling |
| `workers.py` | QRunnable jobs and queued result delivery |
| `theme.py`, `formatting.py` | Company colors, control styles, relative and local timestamps |
| `prefs.py`, `signin.py` | Email-only preferences and asynchronous sign-in |
| `shell.py` | Native window, rail, staff/environment indicators and session transitions |
| `fleet.py`, `fleet_model.py` | Fleet screen, table model, row painter, filtering and refresh |
| `customer.py` | Async customer header, consent flags and round-2 placeholder |
| `widgets/common.py` | Labels, buttons, skeleton and empty-state components |
| `widgets/totp.py` | Six-digit authentication input |
| `widgets/device_detail.py` | Scrollable device health and host panel |
| `widgets/palette.py` | Virtualized fuzzy customer/device navigation |
| `__main__.py` | CLI and packaged startup smoke checks |
| `build_admin_exe.ps1` | Worktree-local PyInstaller build |
| `screenshots.py` | Deterministic offscreen captures, including the actual Qt dialog |

Tests are in `tests/admin/{conftest,test_backends,test_ui}.py`. `pyproject.toml` and `uv.lock` add an `admin` dependency group; PyInstaller is not in the main dependencies. Build outputs and the local uv cache are ignored. Screenshots and this report are force-added.

## Run and verify

For an admin-only environment without installing the repository's training stack:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv --system-certs sync --only-group admin
$env:UV_NO_SYNC = '1'
$env:PYTEST_ADDOPTS = '--basetemp=tmp/admin-pytest'
uv run --group admin pytest tests/admin -q
uv run --group admin python -m home_guard_project.admin --demo
uv run --group admin python -m home_guard_project.admin --server http://127.0.0.1:8000
uv run --group admin python -m home_guard_project.admin --demo --theme light
uv run --group admin python -m home_guard_project.admin.screenshots
```

**Test result: 42 passed in 0.68 seconds.** The requested pytest command was run with `UV_NO_SYNC=1` after syncing only the admin group. Tests cover every demo document, route/query mapping, auth rotation and retry bounds, error mapping, nullable timestamps, TLS verification, sorting/filtering, keyboard navigation, refresh stability with 80 devices, worker-thread isolation, sign-in errors and success, TOTP paste/advance, preferences, role restrictions, palette jumps, offline/error/empty states, session expiry and returning from demo to the configured server.

Live-client startup also passed from source. An additional live-client test exposed an OpenSSL crash in this Windows uv runtime when `create_default_context` inherits TLS key logging. The client now constructs a verified `PROTOCOL_TLS_CLIENT` context with the CA bundle and without implicit key logging. Hostname and certificate checks remain enabled and tested. The upstream runtime issue and fix are documented in [python-build-standalone #1132](https://github.com/astral-sh/python-build-standalone/pull/1132).

## Executable

```powershell
& home_guard_project/admin/build_admin_exe.ps1
```

The build uses PyInstaller 6.22.3, Python 3.12.13, PySide6 6.11.2 and the provided `box/assets/logo.ico`. It bundles the demo files, timezone data, CA certificates and required Qt libraries. The script resolves its project root from its own path, checks that the output stays within that root, and supports `-SkipSync` for an already-installed environment.

**Build succeeded.** Output: `dist/HomeGuardAdmin/HomeGuardAdmin.exe`.

- Executable: **5,377,012 bytes (5.13 MiB)**.
- Complete one-directory bundle: **101,302,656 bytes (96.61 MiB)**.
- Executable SHA-256: `EEA7CB7289D5B637D32D954882888FE63682186DAA409DC0C6068588DFBB7BFD`.
- Packaged `--demo --smoke-test`: **exit 0**; loaded four fixture devices, selected the critical box, opened details and verified the icon resource.
- Packaged `--smoke-test`: **exit 0**; constructed the real HTTP/TLS backend and rendered sign-in, without contacting a server.
- PyInstaller's warnings were inspected: absent Unix modules and optional HTTP/2, SOCKS, Trio and CLI dependencies do not affect this synchronous HTTP/1 client.

The final build used `& home_guard_project/admin/build_admin_exe.ps1 -SkipSync` after installing the admin group. The local build log is `build/admin-build.log`. Smoke tests ran offscreen in hidden child processes. The packaged icon lookup uses the package path because PyInstaller relocates the entry script.

Distribute the entire `dist/HomeGuardAdmin/` directory, including `_internal`; the executable alone is not the application. The bundle is unsigned. No installer, code-signing certificate or auto-updater is part of this round.

## Screenshot review

All 15 captures were opened and inspected. Fixes from visual review included missing offscreen fonts, clipped sign-in copy caused by inherited layout spacing, header alignment, the device-panel close control, narrow table columns, modal borders, panel backgrounds and the initial offline/loading messages. The set includes the requested states plus empty, cached-offline, customer, light-theme and compact-detail coverage.

| Screenshot | State |
|---|---|
| [r1-signin.png](screenshots/r1-signin.png) | Sign-in, 1366×768 |
| [r1-signin-error.png](screenshots/r1-signin-error.png) | Invalid credentials/code |
| [r1-fleet-1366x768.png](screenshots/r1-fleet-1366x768.png) | Default Fleet |
| [r1-fleet-1920x1080.png](screenshots/r1-fleet-1920x1080.png) | Large Fleet |
| [r1-fleet-detail.png](screenshots/r1-fleet-detail.png) | Critical box details, 1920×1080 |
| [r1-fleet-detail-1366x768.png](screenshots/r1-fleet-detail-1366x768.png) | Compact details with horizontal table scrolling |
| [r1-fleet-critical.png](screenshots/r1-fleet-critical.png) | Critical filter |
| [r1-command-palette.png](screenshots/r1-command-palette.png) | Ctrl+K open |
| [r1-labeler-navigation.png](screenshots/r1-labeler-navigation.png) | Identity-restricted Studio workspace |
| [r1-offline.png](screenshots/r1-offline.png) | Cloud unavailable, no snapshot |
| [r1-offline-cached.png](screenshots/r1-offline-cached.png) | Cloud unavailable, cached rows retained |
| [r1-fleet-empty.png](screenshots/r1-fleet-empty.png) | No enrolled boxes |
| [r1-fleet-loading.png](screenshots/r1-fleet-loading.png) | Loading skeleton |
| [r1-customer.png](screenshots/r1-customer.png) | Customer header and consent |
| [r1-fleet-light.png](screenshots/r1-fleet-light.png) | Light theme |

## Known gaps and contract decisions

- No live Cloud service exists yet. HTTP behavior is verified with MockTransport; deployment auth, latency and real server integration remain unverified.
- Review, Studio, Audit, timeline, video/YOLO overlays, live viewing and remote configuration are intentionally placeholders or deferred to later rounds.
- The contract exposes `cameras_stale`, not camera connectivity. The table says “reporting / total”; an offline box shows an unconfirmed numerator. Versions are explicitly “Not reported by the current API.” No invented response fields were added.
- Demo event fixtures support basic field/text filtering and limits. Advanced event cursors and saved-filter execution are not implemented in the demo adapter this round; HttpBackend forwards supplied filters to the Cloud route.
- Demo authentication is intentionally synthetic. Production TOTP verification, throttling, role enforcement, token revocation and audit writes belong to Cloud. Sign-out clears local tokens; the frozen contract has no logout/revocation route.
- The minimum window is 1200×720 logical pixels. At 1366×768 with details open, scroll horizontally for the final table columns and vertically for the remaining device fields. High-DPI multi-monitor behavior has not been manually tested on physical displays.
- This round makes no media requests, S3 requests, box connections, SSH calls or AWS changes. The pre-existing `docs/admin/codex_exe_r1_run.log` is left untracked and untouched. No commits were pushed.

## Questions for round 2

1. Should Cloud proxy all media through its own HTTP origin? The brief requires Cloud-only traffic, while the frozen media route describes presigned S3 URLs. Resolve this before the player is connected.
2. Which event/AI/feedback fixtures should be the first acceptance cases for the player, and what are the authoritative overlay timing conventions?
3. Will the API add versions, per-camera health and upload backlog fields, or should the UI continue relying on health-reason messages?
4. What server base URL and certificate trust setup should the first staff rollout use?
5. Should opening a customer with multiple boxes default to the selected site or a combined timeline, and which pseudonym fields will Cloud guarantee for labelers?
