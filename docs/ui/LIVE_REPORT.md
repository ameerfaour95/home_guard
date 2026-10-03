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
