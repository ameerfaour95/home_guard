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
