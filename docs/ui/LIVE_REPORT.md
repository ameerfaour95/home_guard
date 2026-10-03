# Round L3 — final live details (2026-10-03)

Merged `origin/beelink-collector-box` in **1a825de**, retaining the upstream
`entry_opacity` call and L/L2 live behavior. `uv sync` with `VIRTUAL_ENV` unset
installed OpenVINO. The merge passed **740 tests** with the standard command.
Per-type sensitivity settings were not added to the app Settings table.

**e6379b8** aligns demo and replay detections with the painted person and car.
Optical flow is bounded to one observed sample interval and half the last real
box's width; before cadence is known the interval is one second. New observations
ease to their real coordinates over `motion.HOVER_MS` (120 ms). After one interval
the estimate freezes, including beyond 1.5 intervals. Visibility and fading stay
owned by `detector_view.camera_view` and `entry_opacity`, including slow-box
`still_seen` behavior. The hero caption moves to the clear bottom corner over
`motion.PANE_MS` (240 ms); it is concealed while its animated rectangle would
cover a box or detection label, or when both corners are occupied.

**58ceac4** clips thumbnail detection outlines above the entire bottom scrim.
The extended L2 regression checks rectangle intersections and verifies identical
scrim pixels with detections on and off, at the 80×144 floor and both window sizes.

Final suite: **746 tests passed in 50.339 seconds, exit code 0**, using:

```text
python -m unittest discover -s tests/box
```

The obsolete test runner and `live_tests.log` were removed. Test output from this
round is outside the repository. The replacement [live demo](live_demo_l3.gif)
is **877,050 bytes** (854×480, 5 fps, 48 colors), below the 2 MB limit. The original
17,342,585-byte GIF was removed. The clip demonstrates status/chat changes and the
garage reconnecting; the synthetic figures remain stationary and boxes match them.

All six screenshots were recaptured and opened individually for visual inspection:
every visible box is on its subject, the hero caption covers no box, and thumbnail
outlines stop before the scrim. Files have the exact pixel dimensions below.

| State | 1366×768 | 1920×1080 |
| --- | --- | --- |
| Overview | [live_overview.png](screenshots/live_overview.png) | [live_overview_1920.png](screenshots/live_overview_1920.png) |
| AI looking | [live_looking.png](screenshots/live_looking.png) | [live_looking_1920.png](screenshots/live_looking_1920.png) |
| Reconnecting | [live_reconnecting.png](screenshots/live_reconnecting.png) | [live_reconnecting_1920.png](screenshots/live_reconnecting_1920.png) |

Capture metadata: [1366](live_l3_capture.json), [1920](live_l3_capture_1920.json).
These are visual proof runs with capture overhead, not new CPU-budget benchmarks.
On this workstation's 150% display, set `QT_SCALE_FACTOR=0.6666666666666667` for
one physical pixel per logical pixel. With the checkout's virtual environment
active, reproduce using:

```text
python docs/ui/measure_live.py --seconds 11 --capture --output docs/ui/live_l3_capture.json
python docs/ui/measure_live.py --seconds 11 --screenshots-only --size 1920x1080 --screenshot-suffix _1920 --output docs/ui/live_l3_capture_1920.json
```

All work stayed on local branch `box-app-ui`. No push or box, camera, Telegram,
or other external service access was performed. Post-merge changes are confined
to the app, app/live tests under `tests/box`, and `docs/ui`.

---

# Round L2 — thumbnail layout and bounded publisher cost

This section supersedes Round L's fixed 18/6 rates and its workstation-only
extra-CPU budget result. The earlier measurements below remain historical.
All L2 work is local/offline on `box-app-ui`; `uv sync --offline` used the
existing environment. No box, camera or external service was contacted.

## Part 1: thumbnail text geometry

Thumbnail detections now draw box corners only. The existing `Sees: 1 person`
caption carries the count; per-box name/confidence chips remain on the hero.
Camera names, captions and heartbeat badges use separate bounded rectangles.
Long names, captions and reconnecting text elide within the tile. Tests exercise
the actual paint path, assert every text rectangle is inside the tile and every
pair is disjoint, and reject any thumbnail chip. Coverage includes the rail's
80×144 minimum and actual six-camera layouts at 1366×768 and 1920×1080.

## Part 2: adaptive publisher and local box measurement

`display.preview_cpu_cap_percent` defaults to **10% of one core in total**.
It is read with the existing preview display settings and local overlay
precedence. Per-camera encode/resize/write elapsed time uses a moving average
(20% new sample). Projected cost sums every visible camera at its selected rate
and adds measured worker/handoff CPU overhead, with a one-point minimum reserve.
Lease reads are shared across cameras every 250 ms. Whole-worker CPU counters
include wait/lock bookkeeping, not just encode calls.

The controller lowers one step every two seconds when projected cost exceeds
the cap: **16/5 → 12/3 → 8/2 → 6/2 fps** (hero/each thumbnail). Recovery requires
five seconds and projected cost at the next faster step below 80% of the cap.
Only the large selected camera receives the hero rate. Hidden/minimized,
undemanded and expired viewers publish zero frames; old viewers retain 6/2.
The 6/2 floors take priority when the cap is physically impossible; telemetry
explicitly flags `floor_over_budget`. Startup/transition overshoot is possible.
This adaptive budget replaces the unconditional ≥15/5 target from Round L.

The installer runs [measure_on_box.py](measure_on_box.py) following
[measure_on_box.md](measure_on_box.md). It reads local telemetry only and prints
publisher CPU, actual per-camera and hero/thumbnail publication fps, and detector
outer-loop Hz for 60 seconds closed and 60 seconds open. It rejects stale data,
engine restarts and changed visibility. Detector loops are counted at the first
stream read in the existing preview adapter, independently of throttled status
files; inference/YOLO logic is untouched. The probe has **not** been run on a box.
Actual N150 CPU and detector rates remain unmeasured. Native codec helper-thread
CPU, if present in a different OpenCV build, is outside thread CPU counters.

## Part 3: workstation proof

Hardware: Windows, AMD Ryzen 9 9955HX3D, six synthetic 1280×720 captures.
[live_l2_publisher_cpu.json](live_l2_publisher_cpu.json) compares the original
6/2 publisher to the adaptive worker for **60 seconds each**, then hidden for
10 seconds each. CPU is independently timed across the entire worker lifetime
plus the offer caller; the new telemetry independently agrees at **8.36%**.

| Publisher metric | Original 6/2 | L2 adaptive |
| --- | ---: | ---: |
| Total CPU, percent of one core | 3.59% | **8.36%** |
| Hero actually published | 5.38 fps | 8.18 fps |
| Each thumbnail actually published | 1.92 fps | 2.28 fps |
| Hidden frames published | 0 | **0** |
| Hidden CPU, percent of one core | 0.16% | 1.88% |

L2 is **4.77 percentage points** above the old publisher and below the stricter
10% total cap in this run. The second half measured **8.33%**, with 7.50 hero
fps and 2.13 fps per thumbnail. The controller exercised all four levels and
both directions; it ended at 8/2, with a 7.14% projected share including 2.96%
measured overhead. Final per-camera encode/publish moving averages were
2.00–2.66 ms. Rate transitions are retained in the raw JSON, sampled once per
second. These are measurements, not a guarantee for different hardware or for
every short time window. The configured floors may prevent satisfying a cap
on a sufficiently slow/busy box; the installer probe exposes that condition.

[live_l2_after.json](live_l2_after.json) is a separate 15-second UI replay with
all six cameras active. Hero paint rate was 7.99 fps; thumbnails 2.21–2.28 fps.
File-to-paint p95 was 29.17 ms; AI status maximum 55.22 ms; chat maximum 54.03 ms;
posted input-to-paint maximum 19.13 ms. The measured UI callback maximum was
27.59 ms. As in Round L, this replay times individual publish calls only; use
the whole-worker CPU run above for budget acceptance.

Screenshots were visually reviewed at **1366×768** and **1920×1080** logical
window sizes. The corresponding thumbnail widths are 164 and 237–238 pixels;
the regression separately exercises the narrower 80×144 floor. Captures at this
display's 150% scale retain the composed high-DPI image. The 1920 run resizes
after initial show so Windows cannot silently constrain it to the monitor's
smaller logical work area. Capture JSON records the actual window and tile
dimensions: [1366 capture](live_l2_capture.json), [1920 capture](live_l2_capture_1920.json).
The garage intentionally stops after four seconds to demonstrate reconnecting.

Historical L2 suite: **676 tests passed**. Run the standard command
`python -m unittest discover -s tests/box`. New regressions cover painted thumbnail rectangles, CPU
backoff/recovery/floors and measured overhead, overlay settings, hidden-window
suppression, detector/publication counters, and the installer's delta/rate math
and stale/restarted-data handling.

Reproduce using the existing environment, with `VIRTUAL_ENV` and `SSLKEYLOGFILE`
unset:

```text
uv sync --offline
.venv/Scripts/python.exe docs/ui/measure_publisher.py --seconds 60
.venv/Scripts/python.exe docs/ui/measure_live.py --seconds 15 --output docs/ui/live_l2_after.json
.venv/Scripts/python.exe docs/ui/measure_live.py --seconds 11 --screenshots-only --output docs/ui/live_l2_capture.json
.venv/Scripts/python.exe docs/ui/measure_live.py --seconds 11 --screenshots-only --size 1920x1080 --screenshot-suffix _1920 --output docs/ui/live_l2_capture_1920.json
python -m unittest discover -s tests/box
```

The three requested 1366×768 screenshots are regenerated, with additional
1920×1080 versions for comparison:

- [live_overview.png](screenshots/live_overview.png)
- [live_reconnecting.png](screenshots/live_reconnecting.png)
- [live_looking.png](screenshots/live_looking.png)
- [live_overview_1920.png](screenshots/live_overview_1920.png)
- [live_reconnecting_1920.png](screenshots/live_reconnecting_1920.png)
- [live_looking_1920.png](screenshots/live_looking_1920.png)

The Round L recording has been replaced by the compact L3 clip linked above.
L2 captured only the three requested states, without recording overhead.

---

# Round L — live console (historical)

## Part 1: baseline (2026-10-03)

Run `uv sync` and all commands with `VIRTUAL_ENV` and `SSLKEYLOGFILE` unset.
`python docs/ui/measure_live.py --seconds 12 --output docs/ui/live_before.json`
uses the actual dashboard, JPEG publisher and reader with six generated 1280×720
frames and atomic AI-status writes in a temporary directory. All controls are
simulated. It never opens cameras or contacts any service. One second of startup
is excluded from paint statistics. FPS counts distinct published frames painted,
not repeated repaints. CPU is publisher thread CPU time, as a percent of one core.
The 10 ms event-loop probe measures scheduling lateness, not physical input latency.

Baseline raw measurements: [live_before.json](live_before.json).

Code inspection confirms a 6 fps hero / 2 fps thumbnail publisher, a 166 ms
dashboard and fullscreen poll, JPEG reads/decodes on the UI thread, one-second
AI/chat gates, and replacement of the entire timeline widget when rows change.
Inference publication happens inside detector `read()`, so actual frame rate can
be even lower than the publisher cap when YOLO visits cameras infrequently.

Remote correction: `backend.py` models setup; `engine_backend.py` runs setup
processes. `remote_cameras.py` executes SSH snapshot commands and SCP copies for
camera setup. There is no remote live-dashboard frame or `ai_status.json` transport.
The dashboard reads local `LOG_DIR` even when a previous setup used a remote box.
Further inspection found an existing HTTP file/status server in `serve.py`, but
no Qt dashboard consumer. Its old `/viewer` contract only carries the hero name.

The permitted environment is this Windows workstation. Actual N150 CPU and YOLO
loop rate with/without the app cannot be claimed from this replay. No box or camera
will be contacted. Remote throughput likewise requires an actual transport and
network validation; existing one-shot snapshots cannot satisfy 8 fps.

Measured baseline: hero 5.81 fps; each thumbnail 2.09 fps. File-to-paint 134.69 ms median / 186.20 ms p95 (519.61 ms max). AI write-to-paint 443.79 ms median / 834.48 ms max. UI scheduling lateness 28.45 ms p95 / 53.72 ms max. Publisher 4.16% of one workstation core.

## Part 2: frames

Retain the existing atomic JPEG transport: it works across the engine and app's
separate processes on Windows without new dependencies or shared-memory ownership.
The app sends a versioned three-second visibility lease listing cameras and hero.
Visible demand requests 18/6 fps (headroom for capture/scheduling); hidden,
minimised, other-page and closed windows request no frames. Old viewers retain
their 6/2 fps behaviour. JPEG quality is 75 for the new viewer; sub-stream geometry
is retained up to the existing 1280-pixel limit.

Inference capture hands its already-masked immutable frame to one bounded slot
per camera. A single encoding worker consumes the latest slot independently of
YOLO. No extra capture, inference or frame copy is added to detector `read()`.
The legacy collector still publishes at its existing loop cadence: that producer
cannot be decoupled from detection within this round's allowed files.

Qt watches the preview directory in a worker thread, reads and decodes changed
files to QImage, then signals the GUI to paint. One delivery is in flight, so a
busy GUI cannot accumulate a video backlog. A 250 ms recovery scan covers missed
directory notifications. Fullscreen mirrors frame arrivals instead of polling.
The decorative blurred backing is cached for one second rather than rebuilt on
every paint.

Intermediate replay: [live_frames.json](live_frames.json): 16.14 hero fps,
5.43–5.57 thumbnail fps, 7.23 ms median / 17.12 ms p95 file-to-paint,
23.46 ms maximum scheduling lateness. Publisher CPU 11.72% of one workstation
core, versus baseline 4.16%: **+7.56 percentage points**. This is not an N150 claim.

## Part 3: live status and visible freshness

The worker watches both AI-status atomic replacements and chat file appends, with
re-arming and a recovery scan. Changed status is delivered before the associated
frame batch. Alert-picture decoding also runs off the GUI thread. The dashboard's
one-second timer is now for age/status housekeeping, not video delivery.

Each camera shows a pulsing LIVE badge only for a recent distinct frame. After
three seconds, its retained last picture dims and a `Reconnecting… N s` badge
counts the outage. Fullscreen shares the original frame timestamp. Timeline
timestamps update every second, including quiet groups. AI thinking retains the
existing pulse. New rows use the shared 240 ms slide/fade, while unchanged rows,
group expansion state and the scroll content survive updates. A retained visible
row anchors scroll position; following the bottom continues to follow new rows.

Laptop path: `python -m home_guard_project.box.app --remote-box USER@BOX` explicitly
opens one existing-key SSH tunnel to loopback port 8765. `app/live_server.py`
extends the existing read-only server with the new visibility lease; protected
`serve.py` is unchanged. The remote worker requests hero frames every 60 ms and
thumbnails every 180 ms, status every 350 ms, chat every 500 ms. Requests and
decoding are off the GUI; each resource has at most one request in flight, GUI
delivery is bounded, repeated image versions cannot refresh LIVE, and timeouts
back off. Alert image downloads are limited to two and don't delay chat text.
This remote dashboard is read only; existing setup/camera management remains in
the setup flow. Both machines need this checkout's app code. No SSH or remote
network request was made during development; remote ≥8 fps is not yet certified.

After status changes, a 12-second local replay measured hero 16.07 fps,
thumbnails 5.45–5.54 fps, AI write-to-paint 28.27 ms median / 52.15 ms maximum,
and maximum UI scheduling lateness 21.98 ms. Final proof below supersedes these
intermediate measurements.

## Part 4: proof and limits

Hardware: Windows, AMD Ryzen 9 9955HX3D (16 cores / 32 logical processors), **not
the Intel N150 box**. All video is generated 1280×720 demo imagery. Six cameras
are active in performance runs, including moving detection overlays. Final
measurements also append owner messages to the real chat-feed file. No camera,
box, Telegram or OpenAI connection was made. The remote test starts a synthetic
loopback HTTP server in a separate process and replaces SSH startup with a stub.

`measure_baseline.py` loads the three original 7f59234 UI/preview modules from
git objects in memory, then runs the same current probe. It changes no worktree
files. This gives a matched before/after comparison with the added chat and
input probes; the original Part 1 measurements remain saved separately.

The paint probe counts distinct source timestamps at the end of CameraTile's
paint callback. Local latency starts at the JPEG file mtime. Remote age uses
the server's frame age and the request start, so an old first response does not
become falsely LIVE and clock offsets do not refresh frozen video. AI/chat
latency ends when the panel paints the corresponding written data. Posted input
is a synthetic Qt event sent from the source thread, ending at the tile paint;
it includes GUI queue delay but not physical display/compositor latency.
`ui_callback_block` measures dashboard tick, status update, timeline render and
tile paint durations; it does not claim to instrument every Qt callback. The
10 ms probe separately records scheduling stalls across the event loop.

Detection overlays now use sparse optical flow on 320-pixel grayscale frames in
the decode worker, and only when the eye toggle is on. Each new detector sample
resets the estimate. A synthetic translation regression checks box movement.
Lost tracks disappear; tracking never extends the detector's three-second TTL,
and reconnecting tiles hide boxes. This is motion estimation between observations,
not new detections or extra YOLO work. Settings acknowledgements reuse watched
AI status rather than repeatedly reading that JSON on the GUI thread.

### Final measured results

Matched 15-second replays, excluding the first second of paint samples:
[before](live_before_matched.json), [local after](live_after.json),
[simulated remote](live_remote.json).

| Metric | Original UI | New local UI | Target |
| --- | ---: | ---: | --- |
| Hero painted FPS | 5.85 | 16.20 | >=15: met locally |
| Thumbnail painted FPS | 1.93-2.00 | 5.42-5.50 | >=5 each: met locally |
| File-to-paint p95 | 190.25 ms | 27.65 ms | Arrival-driven paint |
| AI write-to-paint maximum | 894.91 ms | 61.92 ms | <1 second: met |
| Chat write-to-paint maximum | 893.91 ms | 60.16 ms | <1 second: met |
| Measured UI callback maximum | 60.06 ms | 28.85 ms | See probe definition above |
| Event-loop scheduling lateness maximum | 66.16 ms | 40.74 ms | Includes uninstrumented Qt work |
| Posted input-to-paint p95 | 27.75 ms | 9.36 ms | <50 ms |
| Posted input-to-paint maximum | 28.61 ms | 20.71 ms | <50 ms: met in local sample |

Simulated remote: **12.06 hero FPS**, thumbnails 5.07-5.78 FPS,
AI maximum 401.36 ms, chat maximum 554.71 ms,
posted input maximum 31.21 ms. The >=8 fps remote target is
met over synthetic loopback HTTP. There was no previous Qt remote dashboard to
benchmark. Real SSH/Wi-Fi/LAN throughput and long-duration worst-case input latency
remain unverified; these short runs are measurements, not hard real-time guarantees.
Earlier remote trials exposed request/GUI contention; the final adapter avoids
reapplying unchanged overview state and gives the hero extra request headroom.

### Publisher budget and detection

[live_publisher_cpu.json](live_publisher_cpu.json) measures **all publisher worker
CPU plus offer/publish caller CPU**, using six synthetic 30 fps captures. Source
generation, camera decoding, the UI and YOLO are excluded on both sides.

| Publisher measurement | Original | New |
| --- | ---: | ---: |
| Visible, percent of one workstation core | 4.10% | 13.67% |
| Hero publication | 5.38 fps | 15.88 fps |
| Each thumbnail publication | 2.00 fps | 5.88 fps |
| Hidden, frames published | 0 | 0 |
| Hidden, percent of one workstation core | 0.78% | 0.78% |

Extra publisher cost is **9.57 percentage points of one workstation core**,
within the requested ten-point extra budget in this test, with little margin.
The simpler paint replay's `publisher_one_core_percent` measures individual
publish calls only; use the full worker measurement above for the CPU budget.

Actual **N150 CPU cost: unmeasured**. Actual **YOLO detector loop rate, app closed:
unmeasured; app open: unmeasured**. Contacting the box/cameras was expressly
forbidden and no local model weights were available. A blocking-encoder test
proves capture hand-off and detector `read()` do not wait for encoding; it does
not prove absence of CPU contention on an N150. This hardware acceptance check
remains outstanding. The collector's original synchronous producer also remains
limited by its own loop rate because editing that producer was outside scope.

### Reproduction

Unset `VIRTUAL_ENV` and `SSLKEYLOGFILE`, run `uv sync`, then use the checkout's
`.venv/Scripts/python.exe` for each command:

```text
docs/ui/measure_baseline.py --seconds 15 --output docs/ui/live_before_matched.json
docs/ui/measure_live.py --seconds 15 --output docs/ui/live_after.json
docs/ui/measure_live.py --seconds 15 --remote-simulated --output docs/ui/live_remote.json
docs/ui/measure_publisher.py
docs/ui/measure_live.py --seconds 12 --capture --output docs/ui/live_capture.json
python -m unittest discover -s tests/box
```

The recording is a separate run: the garage source intentionally stops at four
seconds. Its low average FPS is an outage demonstration, not a throughput result.
Qt captures the actual composed 1366×768 logical window (2049×1152 pixels at this
display's 150% scaling), at approximately ten samples/second; GIF encoding happens
after the run. Sampling screenshots and a
recording adds work, so its timings are retained in `live_capture.json` separately
from the six-active-camera performance result.

### Evidence and tests

- [Hero and thumbnails with LIVE badges](screenshots/live_overview.png)
- [AI looking state](screenshots/live_looking.png)
- [Reconnecting and dimmed last frame](screenshots/live_reconnecting.png)
- [Short moving demo, decisions and chat](live_demo_l3.gif)

Run `python -m unittest discover -s tests/box` from the checkout with its
virtual environment active. New tests cover reconnect boundaries, relative times,
retained widgets and scroll, visible/hidden/expired demand, hero switching, masked
asynchronous capture, off-GUI decoding, lease revocation, optical-flow translation
and expiry, and remote frozen-frame handling.

Final result: **666 tests passed in 38.778 seconds, exit code 0**, including all
`test_serve.py` cases. The existing detection-layer screenshot comparison now
holds the intentional LIVE pulse clock still; otherwise two screenshots could
differ solely because the pulse advanced. The original recording has been replaced by the compact L3 clip above.

The branch received external merge `1d112f3` between Parts 1 and 2. It was retained;
the final full suite tests that merged tree plus this work. This round's commits
only change allowed app/preview, new box tests and `docs/ui` files. Nothing was
pushed.
