# Installer measurement on the N150

Run this after deploying this checkout's app/preview files and restarting the
existing inference-preview engine. Keep normal camera capture and YOLO running.
This procedure has **not** been run against a real box during development.

The preview reads `display.preview_cpu_cap_percent` from the existing
`data_collection/config.yaml`, then `HOME_GUARD_CONFIG_OVERLAY` (overlay wins).
Default: **10 percent of one core**, total publisher budget. For example, merge
`preview_cpu_cap_percent: 10` into the existing `display` mapping; retain
`preview_enabled: true`. Restart the engine after changing this setting.

On the box, using its existing Python environment:

```text
python docs/ui/measure_on_box.py --preview-dir /absolute/path/to/logs/preview
```

On Windows use the box's `.venv/Scripts/python.exe` and a local Windows path.
The script only reads the local `publisher_metrics.json` file. It does not
launch the app/engine, write demand, contact cameras, or use a network connection.

1. Close the app, keep the engine running, wait 15 seconds, press Enter.
   The script measures **60 seconds** with the app closed.
2. Open the Live dashboard, select the normal hero and camera set, wait 15
   seconds, press Enter. Leave the window visible for **60 seconds**.
3. Save the printed JSON with the box model, camera count, stream resolutions,
   CPU cap and detector configuration. Repeat under comparable scene activity;
   cooldown and alert windows affect detector throughput independently of preview.

Output includes publisher CPU as percent of **one** core (not all four N150
cores), actual successful hero publications/sec, aggregate thumbnail
publications/sec, per-camera publications/sec, selected rate limits, and
detector loop Hz for each sample plus open/closed percentage change. Snapshots
arrive once per second, so a sample may span slightly more than 60 seconds;
rates use its actual monotonic duration. Restarted/stale engines or changed
visibility fail the sample instead of silently printing misleading rates.

Publisher CPU includes the whole encoding-worker loop (lease reads, resizing,
encoding, disk writes, metrics) and capture handoff, via thread CPU counters.
It excludes capture/decoding, Qt rendering and YOLO. Native codec helper-thread
CPU, if enabled by a different OpenCV build, is not captured by thread counters.
The controller uses a conservative moving average of elapsed encode/publish
time per camera, sums projected work across demanded cameras, and adds measured
worker/handoff overhead (moving average, minimum one percentage point). The
worker CPU counter includes wait/lock bookkeeping; lease checks are shared
across cameras at four checks/second. Hysteresis lowers one step every
two seconds when projected cost exceeds the cap, and raises at most once every
five seconds only if the next step fits below 80% of the cap.

Rate pairs are 16/5, 12/3, 8/2 and 6/2 fps (hero/thumbnail). Only the selected
large camera receives the hero rate. Hidden/minimized/expired demand publishes
no frames. The minimum 6/2 rates take precedence if even those exceed the cap:
`floor_over_budget_observed` flags that condition for acceptance review. A CPU
cap below the cost of the minimum rates cannot be guaranteed while preserving
those floors. Short startup/transition overshoot is possible while costs settle.

The detector counter increments at the first stream read on each outer
inference loop. It is independent of throttled `ai_status.json` writes and
works with the app closed. It measures **outer loops**, not individual YOLO
predictions or camera fps. The script requires the inference-preview adapter;
the legacy collector does not expose this counter. No detector code is changed.
