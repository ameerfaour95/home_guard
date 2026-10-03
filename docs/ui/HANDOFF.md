# Home Guard desktop app handoff

## Round Z — watch areas (2026-10-03)

1. Fast-forwarded `origin/beelink-collector-box` to `bf88b0d`, then built three UI commits on `box-app-ui`; no push.
2. Local and SSH controls load, set and clear zones; demo controls keep independent in-memory zones without commands.
3. The split dialog edits the raw snapshot, with a blurred ambient enclosure, animated handles and live outside dimming.
4. Click up to 32 corners, drag to adjust, snap to the first, Undo with right-click/Backspace/Ctrl+Z, or Clear; coordinates use four-decimal picture fractions.
5. Empty saves the whole picture, one or two corners disable Save, and areas under 5% show a quiet warning; Enter saves and Escape cancels.
6. Saving runs off the UI thread with disabled controls and a thin bar; failures keep the drawing open, and success closes with an Area saved toast.
7. The reserved 36px tile row gains a pill and crossfading status after snapshots; saved polygons dim and outline thumbnails without modifying the raw image.
8. `python -m unittest discover -s tests/box`: 560 tests pass using this checkout's Python, with `VIRTUAL_ENV` and inherited antivirus `SSLKEYLOGFILE` unset for that process.
9. Fourteen synthetic screenshots at 1366×768 and 1920×1080 were opened and reviewed; see [SCREENSHOTS.md](SCREENSHOTS.md); RTL geometry is tested and copy remains in the existing English-only catalog.
10. No real box, camera, SSH or Telegram was contacted; real restart/masking, remote latency and native desktop animation remain unverified; engine scripts/runtime were not edited after the merge.

## Earlier handoff

Built on `box-app-ui`, without pushing or merging. The five protected engine files are unchanged: `setup_customer.ps1`, `setup_network.ps1`, `check_box.ps1`, `build_exe.ps1`, and `inference.py`.

`CLAUDE.md` is absent from this working copy and from its tracked file list. The box README and the current setup script were read before implementation. Existing clip, metadata, outbox and heartbeat formats were left unchanged.

## Run

From the checkout root, with this checkout's Python:

```powershell
uv --system-certs sync
.venv\Scripts\python.exe -m home_guard_project.box.app
.venv\Scripts\python.exe -m home_guard_project.box.app --demo
.venv\Scripts\python.exe -m home_guard_project.box.app --setup --demo --skip-cameras
.venv\Scripts\python.exe -m home_guard_project.box.app --setup --demo --skip-cameras --fail
```

`--setup` always uses a simulated backend in this branch, even without `--demo`. No remote installer is implemented. Address and house validation follow the branch-tip wizard. Network and camera passwords are masked, excluded from answer/result representations, never written to disk by the app, and cleared when a simulated run finishes or fails. Use Find cameras later for a practice run without entering any password.

Direct screen selection:

```powershell
.venv\Scripts\python.exe -m home_guard_project.box.app --demo --state offline --cameras 9
.venv\Scripts\python.exe -m home_guard_project.box.app --demo --details
.venv\Scripts\python.exe -m home_guard_project.box.app --setup --demo --page summary
.venv\Scripts\python.exe -m home_guard_project.box.app --setup --demo --page network --wifi
.venv\Scripts\python.exe -m home_guard_project.box.app --demo --size 1920x1080 --screenshot docs/ui/example.png
```

Dashboard states: `mixed`, `live`, `offline`, `stopped`, `hidden`, `empty`, `error`, `loading`, `inference`, `quiet`. Wizard pages: `address`, `network`, `house`, `cameras`, `progress`, `summary`, `failure`, `validation`. Additional setup switches: `--wifi`, `--alerts`, `--skip-cameras`, `--fail`. `--cameras` accepts 1 through 9. Direct progress/summary screenshots advance the same sequencer deterministically, without waiting for the practice timers.

`live_view.cmd` uses a hidden Python launcher and Python's windowless executable so the normal shortcut does not leave a console open. The launcher waits for the GUI exit; a successful close does nothing to the collector. A failed start opens the original Git Bash screen visibly. `screen.sh` also starts the GUI first, with `--legacy` available to bypass it. The old log and live-camera scripts stay in the repository. Their existing stop/rerun behavior is reachable only through the explicit legacy fallback after an unsuccessful GUI run.

## Choices

| Choice | Reason / behavior |
| --- | --- |
| Dark navy, cyan from the logo, Segoe UI | Calm Windows security-product styling; Windows offscreen capture explicitly loads the local Segoe UI font to avoid missing glyphs. No font is redistributed. |
| File bridge, with hashed camera filenames and a names-only `cameras.json` manifest | Works across Windows sessions without another camera connection. Names cannot create filesystem paths; the app does not read camera configuration or camera credentials. |
| JPEG quality 75, at most 2 frames/second per camera, image longest side 720 pixels | Limits CPU and disk cost. The data collector shares its existing annotated display image; no second detector runs. |
| Viewer expires after 15 seconds; picture expires after 12 seconds | No encoding while nobody is looking. A stale image becomes a clear no-picture tile rather than a deceptively live frozen image. Capture object identity prevents republishing the same frozen source indefinitely. |
| Atomic temporary write and replace, with recoverable sharing failures | Readers open/read/close before decoding. Failed replacements preserve the previous image, remove the temporary file and retry on a later throttled tick. The manifest also retries after a sharing failure. |
| `display.preview_enabled` off in the laptop base configuration, on in the box overlay | The existing laptop display and collection paths retain their behavior when previews are disabled. Box annotation is enabled for the shared display image; clip output flags/formats stay unchanged. |
| Separate `inference_preview.py` adapter | Keeps protected `inference.py` untouched. It reuses the current `_Stream.read` interface and publishes camera pictures without annotations, because that inference mode has no annotated display image. The runner selects the adapter only in inference mode; with preview disabled it runs the original inference entry point directly inside the adapter. |
| One column for 1 camera, two for 2-4, three for 5-9; preserve full-image aspect ratio | Predictable resizing with no cropped camera coverage. The camera area is replaced completely with a calm panel when pictures are switched off. |
| Preview poll every 500 ms; local state/log refresh every 5 seconds on a worker | Keeps the GUI responsive on the mini PC. The collector alone owns capture and detection. Closing the GUI does not issue any collector command. |
| Latest 12 meaningful events, bounded 64 KiB log tails, cached by file size/time | Progress-bar noise stays out of the activity list. Daily rollover is discovered automatically, with dates inferred for collector time-only lines. Upload timestamps use a bounded-memory reverse scan, including earlier successful runs before a large progress tail. Subsequent checks scan appended bytes and cache unchanged files. Older log days are searched when necessary. Optional details redact addresses and common secret assignments. |
| Upload activity says files when the upload log counts files | The uploader includes clip companion files, so reporting that number as clips would be inaccurate. Clip-ready activity uses the actual clip count from the outbox log. |
| Four input pages plus progress and summary | Box address; network; house/pictures/alerts/hours; camera discovery. Alerts and hours are included because the branch-tip setup script asks about them. Ethernet is the default; pictures and alerts default off; camera discovery defaults on but can be skipped. |
| Backend protocol returns typed results/checks, with optional rescue-hotspot values | A real backend can be substituted later. The current simulator reports PASS/WARN/FAIL, halts on FAIL and supports a new run after retry. It never opens a network connection. Result copy uses translation keys. Real hotspot values are intentionally absent; summary shows clearly labeled pending fields. |
| Practice step delays of 2.4 / 0.9 / 2.2 / 4.2 / 1.4 seconds | Makes update, preferences, network, discovery and checks visibly progress without pretending the simulation is a real installation. Backend execution runs on a worker. Power recovery is a manual-check WARN even on the success path. |
| Synthetic Qt-painted camera images, deterministic screenshot selection | No external images, camera streams, private scenes or credentials are needed to review the product. Every application string is in `app/strings.py`. |

The inference adapter intentionally depends on the current private stream interface (`name`, `_frame`, `read`). When the other developer's inference changes arrive, recheck this adapter or replace it with a direct `PreviewWriter` hook in their frame loop. It adds no camera reader, no detector, and does not change alert decisions.

## Screenshots

There are **52 PNG screenshots**, covering 26 cases at **1366x768** and **1920x1080**. Every file is listed in [SCREENSHOTS.md](SCREENSHOTS.md). [index.html](index.html) is an offline clickable gallery. The two contact sheets show all screens together.

Regenerate with:

```powershell
.venv\Scripts\python.exe docs/ui/capture.py
```

Screens were viewed and revised: the offscreen font issue, malformed punctuation, summary clipping, nested panel backgrounds, and failure wording were corrected. Single-camera and four-camera examples show live pictures; nine cameras demonstrates mixed live/offline states. Password fields are empty throughout the review images.

## Verification

- `python -m unittest discover -s tests/box`, using this checkout's activated environment: **146 tests passed**, including all 121 pre-existing tests and 25 new tests.
- Nonvisual tests cover viewer gating, per-camera throttling, real JPEG/atomic replacement, failed Windows-style replacement, temporary cleanup, manifest retry, safe camera names, stale/future/missing pictures, closed reader handles, frozen capture suppression, log parsing/redaction/rollover/bounded reads/older upload timestamps, heartbeat state conversion, and simulated step ordering/warnings/failure/retry/backend exceptions.
- The real data collector module's `main()` was exercised with synthetic capture and a mocked detector for enabled+viewer, enabled+no-viewer, and disabled+viewer cases. Actual plotting branch selection, resize to 720, JPEG encode, file replace, reader decode and resource cleanup were exercised. No model weights, camera configuration, camera connection, or cloud request was used. A supplied annotated array stood in for detector plot output.
- The inference adapter also exercised the actual existing `_Stream.read()` method on an object supplied with a synthetic frame, while `VideoCapture` was forbidden.
- `docs/ui/smoke.py` passed: 1-9 live tiles at both resolutions, exact window dimensions, timed graphical setup completion, failure stopping after the network step, retry navigation, and mocked normal/failing launcher branches.
- `bash -n` passed for both changed shell scripts. Changed shell scripts have LF endings; the changed CMD launcher is ASCII. The five protected engine files have no diff from the starting commit.
- An actual `python -S` GUI launch failed with missing PySide6 as expected; mocked launcher verification confirmed that an unsuccessful child exit selects the legacy screen. The legacy window itself was not opened or allowed to stop/restart a collector here.
- Every screenshot's pixel dimensions were checked. The dependency change adds PySide6 Essentials; the three existing uv exclusion/index rules are unchanged.

## Not verified on hardware / not implemented

No box or camera was contacted, no SSH or AWS call was made, and no SSH/AWS credential files were read. This machine has no camera/box.

Not exercised: scheduled-task session 0 to logged-in-user filesystem permissions; the actual Windows desktop shortcut and visible fallback window; real network outages/reconnects; real detector annotation output; box CPU/disk load and camera-frame cadence with 1-9 physical cameras; uploads, alerts, or a real customer installation. Permission contention is unit-tested with a simulated `PermissionError`; the file-open/read/close and subsequent replace are exercised on this Windows filesystem, but live concurrent session-0 contention still needs a box test.

The installer backend is deliberately simulation-only. It does not update a box, change networks, discover cameras, create rescue credentials or package the graphical wizard into HomeGuardSetup.exe. Those engine and real-backend tasks remain with the other developer. The inference adapter must be checked against their future inference interface before deployment.

## Environment boundary incident

One formatter installation command followed the inherited `VIRTUAL_ENV` into `C:/Users/ameer/Ameer/home_guard/.venv`, contrary to the requested boundary. It completed before the attempted cancellation. No project source files in that checkout were changed, but seven formatter-related environment packages were reinstalled: `black` 26.1.0 (pure Python build), `click` 8.3.1 -> 8.5.0, `mypy-extensions` 1.1.0, `packaging` 25.0 -> 26.3, `pathspec` 1.0.4 -> 1.1.1, `platformdirs` 4.9.2 -> 4.12.2, and `pytokens` 0.4.1. Subsequent environment commands explicitly used this checkout's Python or project environment. The other environment has not been restored; its owner should review those changes before relying on it.
