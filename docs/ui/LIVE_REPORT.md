# Round L — live console

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
docs/ui/run_box_tests.py
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
- [Short moving demo, decisions and chat](live_demo.gif)
- [Full test output](live_tests.log)

The exact requested command,
`.venv/Scripts/python.exe -m unittest discover -s tests/box -t .`, fails before
discovery: `Start directory is not importable`. This checkout has no test package
markers and an unrelated installed package named `tests` shadows its namespace.
`docs/ui/run_box_tests.py` installs those two namespaces **in memory** and discovers
every box test without changing files outside the allowlist. No tests are skipped;
`test_serve.py` completes. New tests cover reconnect boundaries, relative times,
retained widgets and scroll, visible/hidden/expired demand, hero switching, masked
asynchronous capture, off-GUI decoding, lease revocation, optical-flow translation
and expiry, and remote frozen-frame handling.

Final result: **666 tests passed in 38.778 seconds, exit code 0**, including all
`test_serve.py` cases. The existing detection-layer screenshot comparison now
holds the intentional LIVE pulse clock still; otherwise two screenshots could
differ solely because the pulse advanced. The GIF contains **109 frames / 10.9
seconds**. Its separate capture run painted the hero at 15.39 fps, with maximum
posted input-to-paint 45.14 ms while recording and demonstrating the outage.

The branch received external merge `1d112f3` between Parts 1 and 2. It was retained;
the final full suite tests that merged tree plus this work. This round's commits
only change allowed app/preview, new box tests and `docs/ui` files. Nothing was
pushed.
