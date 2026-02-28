# Dual-Stream Data Collection Design

## The Problem

We need training data for two models that run at different resolutions during inference:

- **YOLO** runs on the **sub-stream** (low-res ~704x480, `/s1/`) for real-time detection.
- **VLM** runs on the **main-stream** (high-res 1920x1080 or 2592x1520, `/s0/`), cropped around the detected trigger classes (person, cat, dog, etc.).

Training data must match inference conditions. Collecting everything from the sub-stream means the VLM trains on blurry low-res frames. Collecting from the main-stream freezes the system because high-res raw frames consume massive memory.

### Why main-stream alone freezes the machine

The system runs on an i5-1135G7 laptop with 8 GB RAM and no dedicated GPU.

Each 1920x1080 raw frame = ~6 MB. With 5 cameras, a 17-second rolling buffer at 10 fps, the memory math is:

```
5 cameras x 170 frames x 6 MB = ~5.1 GB
```

That alone exceeds available memory. The system swaps to disk and locks up.

---

## The Solution: dual-stream with three memory optimizations

### 1. Sub-stream for YOLO (always running)

`SubStreamThread` connects to each camera's sub-stream (`/s1/`, ~704x480). It reads frames continuously, keeps `latest_frame` as a raw numpy array for YOLO detection, and stores the rolling buffer as **JPEG-compressed bytes** instead of raw numpy.

Memory comparison for the sub-stream buffer (5 cameras, 12s at 7 fps = 84 frames):

| Format | Per frame | Per camera | 5 cameras |
|--------|-----------|------------|-----------|
| Raw numpy (704x480x3) | ~1 MB | ~84 MB | ~420 MB |
| JPEG bytes (q=92) | ~30-50 KB | ~3.5 MB | ~17 MB |

This single change saves ~400 MB of RAM.

### 2. Main-stream pre-connect on detection (zero cost when truly idle)

`MainStreamThread` is **not** created at startup. It connects to the main-stream RTSP at the **first detection** (when the hysteresis score goes above zero), well before the trigger threshold is reached. This gives the main-stream time to establish the RTSP connection and start buffering high-res frames *during* the score ramp-up (~2-3 seconds), plus the entire post-roll window (5 seconds).

The goal is **10 seconds of VLM crop footage** — matching `clip.seconds`. After the post-roll expires, saving is deferred until `buffer_duration() >= CLIP_SECONDS`, with an absolute cap of `POST_ROLL_SEC + CLIP_SECONDS` from trigger time to avoid waiting forever.

If the detection fades (score drops back to zero) without triggering, the pre-connected main-stream is released immediately, keeping idle cost at zero.

After the clip is saved, the main-stream thread is destroyed, freeing all resources. If the main-stream fails to connect within `OPEN_TIMEOUT_MSEC + 2s`, it is abandoned and the clip is saved without a VLM crop.

Idle state: 5 sub-stream connections only (lightweight).
During an event: 5 sub-stream + 1 main-stream (only the triggered camera).

### 3. Reduced defaults for low-end hardware

| Setting | Before | After | Why |
|---------|--------|-------|-----|
| `clip.store_fps` | 10 | 7 | Fewer frames in buffer, less CPU |
| `clip.yolo_imgsz` | 960 | 640 | Sub-stream is ~704px, 640 is a natural fit |
| `main_stream.store_fps` | 5 | 5 | On-demand so cost is zero when idle |
| `detection.yolo_every_n_frames_cpu` | 3 | 5 | Run YOLO less often on CPU |
| `display.show_windows` | true | false | Each OpenCV window copies + resizes every frame |

---

## Total memory budget

| Component | Memory |
|-----------|--------|
| Sub-stream JPEG buffers (5 cameras) | ~17 MB |
| Sub-stream `latest_frame` raw (5 cameras) | ~5 MB |
| YOLO model in RAM | ~50 MB |
| Python + OpenCV overhead | ~200 MB |
| Main-stream during event (1 camera, 5s at 3fps, JPEG) | ~2 MB |
| **Total during idle** | **~270 MB** |
| **Total during event** | **~275 MB** |

Compared to ~5.1 GB for raw main-stream buffers. The system stays well within 8 GB.

---

## How data flows

### Idle state (no detection)

```
Sub-stream /s1/ (per camera)
    |
    v
SubStreamThread
    |-- latest_frame (raw numpy) --> YOLO detection --> hysteresis scoring
    |-- JPEG buffer (rolling 12s)    [sits idle until trigger]
```

Main-stream: not connected. Zero CPU, zero RAM.

### Detection starts → pre-connect

```
1. YOLO detects a trigger class → score goes above 0
2. MainStreamThread created → begins RTSP connection to /s0/
   (score is still ramping; trigger hasn't fired yet)
3. If score drops back to 0 before trigger → MainStreamThread released (false alarm)
```

### Trigger event

```
4. Hysteresis score hits SCORE_MAX → trigger armed
   - If main-stream already connected (pre-connect): post-roll starts immediately
   - If not yet connected: poll until frame_shape appears, then start post-roll
   - TIMEOUT (OPEN_TIMEOUT_MSEC + 2s): give up, save without VLM crop
5. Main-stream accumulates JPEG frames during post-roll
6. Post-roll (5s) ends → check main-stream buffer_duration():
   - buffer_duration >= CLIP_SECONDS (10s) → save now
   - buffer_duration < CLIP_SECONDS → keep waiting (max extra = CLIP_SECONDS)
7. Save:
    a. Sub-stream buffer → full low-res clip (clips/)
    b. Sub-stream frames → YOLO export (yolo/images/ + yolo/labels/)
    c. Main-stream frames → decompress → YOLO on aligned sub-frames →
       compute trigger-class crop → crop → save (vlm_crops/)
8. MainStreamThread destroyed, resources freed
```

The pre-connect at first detection gives the main-stream a ~2-3 second head start (during score ramp-up). Combined with the post-roll and the extra wait for buffer fill, the VLM crop gets a full `CLIP_SECONDS` (10s) of high-res footage.

### Crop computation

YOLO never runs on the high-res main-stream frames. Instead, detections come from the sub-stream (already in memory, small, fast) and are scaled to main-stream coordinates.

**Time-aligned crop computation.** The sub-stream clip may span a slightly different window than the main-stream (e.g. sub covers 10s but main only has 9s because of connection delay). The crop function receives the sub-stream timestamps (`sub_start_ts`, `sub_end_ts`) and the main-stream timestamps (`m_start`, `m_end`). If `m_start > sub_start_ts`, the leading portion of sub-frames that predates the main-stream is skipped. This ensures the crop is based only on detections that correspond to actual main-stream footage — a person who moves from point A to point B won't have the crop stuck on their earlier position.

Steps:

1. Compute the overlap between sub-stream and main-stream time windows
2. Skip the leading sub-stream frames that predate the main-stream
3. Sample ~5 evenly-spaced frames from the aligned sub-stream portion
4. Run YOLO on each (fast — sub-stream is ~704x480)
5. Collect all trigger-class bounding boxes (person, cat, dog, etc.)
6. Scale every box from sub-stream to main-stream coordinates (`scale_x = main_w / sub_w`)
7. Compute the union bounding box across **all** samples
8. Add padding (30% of bbox dimensions)
9. Expand to square (VLMs prefer square input)
10. Clamp to frame boundaries, enforce minimum 384x384
11. Apply this single crop region to every main-stream frame

One stable crop for the whole clip avoids jitter and keeps the code simple. All cropped frames share the same resolution so they go straight into the MP4 without resizing.

---

## Dataset output

Each trigger event produces:

```
dataset_multi/
  clips/<camera>/<date>/<clip_id>.mp4          # sub-stream full video
  vlm_crops/<camera>/<date>/<clip_id>.mp4      # main-stream cropped video (new)
  meta/<camera>/<date>/<clip_id>.meta.json     # metadata linking both
  yolo/images/<camera>/<date>/<clip_id>_f*.jpg # sub-stream YOLO training frames
  yolo/labels/<camera>/<date>/<clip_id>_f*.txt # YOLO label files
```

- **YOLO training**: uses `yolo/images/` + `yolo/labels/` at sub-stream resolution. Matches inference.
- **VLM training**: uses `vlm_crops/` with high-res cropped clips. Matches inference.
- **Review/labeling**: uses `clips/` for full-scene context.

---

## cameras.yaml URL handling

`cameras.yaml` stores main-stream URLs (`/s0/`). The sub-stream URL is auto-derived by replacing `/s0/` with `/s1/` in `config.py`:

```
cameras.yaml:   rtsp://admin:pass@192.168.68.106:554/unicast/c8/s0/live
                                                              ^^
CAMERAS (sub):  rtsp://admin:pass@192.168.68.106:554/unicast/c8/s1/live
CAMERAS_MAIN:   rtsp://admin:pass@192.168.68.106:554/unicast/c8/s0/live  (original)
```

No changes needed to `cameras.yaml` or the discovery process.

---

## Key timing details

### Pre-connect gives a head start

The main-stream connects at **first detection** (score > 0), not at trigger time (score = SCORE_MAX). Since the score ramps at `score_reward` (2.0 pts/sec) toward `score_max` (4.0), the pre-connect gives a ~2 second head start — enough for the RTSP connection to establish and start buffering frames before the trigger fires.

### Post-roll waits for connection

If the main-stream wasn't pre-connected (rare edge case, e.g. after cooldown), the post-roll timer does **not** start until the main-stream delivers its first frame. If the connection fails within `OPEN_TIMEOUT_MSEC + 2s`, the clip is saved without a VLM crop.

### Save waits for full buffer

After the post-roll expires, saving is deferred if `main_cap.buffer_duration() < CLIP_SECONDS`. This ensures the VLM crop gets a full 10 seconds of high-res footage. An absolute cap of `POST_ROLL_SEC + CLIP_SECONDS` prevents infinite waiting.

```
first detection (score > 0)
    |
    v
MainStreamThread created (pre-connect, main_connect_ts = now)
    |  ~2s ramp-up, main-stream connecting + buffering
    v
score hits SCORE_MAX --> trigger armed
    |  if main already delivering frames: trigger_ts = now
    |  else: poll until frame_shape appears (or timeout)
    v
trigger_ts + POST_ROLL_SEC elapsed?
    |  YES --> check buffer_duration >= CLIP_SECONDS?
    |            YES --> save clip
    |            NO  --> keep waiting (max extra = CLIP_SECONDS)
    v
**snapshot main-stream frames NOW** (before any I/O)
save: sub-stream clip + YOLO export + VLM crop (uses snapshot)
    |
    v
MainStreamThread released
```

**Critical: main-stream frames are snapshotted before the sub-stream save.** The sub-stream MP4 write, YOLO export, etc. can take several seconds on CPU. If `get_clip_frames()` were called *after* those operations, `time.time()` would have advanced and the rolling-buffer cutoff (`now - CLIP_SECONDS`) would shift past the event — producing a clip where the person is already gone. By snapshotting the main-stream buffer *first*, the frames are frozen at the right moment.

Typical timeline for a 10-second VLM clip:
- t=0: First detection. Main-stream pre-connects.
- t=1: RTSP connection established, frames start buffering.
- t=2: Score hits SCORE_MAX, trigger fires.
- t=7: Post-roll ends. Main buffer has ~6s. Keep waiting.
- t=11: Main buffer hits 10s. **Snapshot main frames [t=1..t=11].**
- t=11-16: Sub-stream MP4 + YOLO export (expensive, takes ~5s).
- t=16: VLM crop uses the t=11 snapshot — person is in the clip.
- Total VLM footage: 10 seconds.

If the detection fades (score → 0) before triggering, the pre-connected main-stream is released immediately — no wasted resources.

---

## Files changed

| File | What changed |
|------|-------------|
| `config.yaml` | Added `main_stream` section, lowered `store_fps`/`yolo_imgsz`/`yolo_every_n_frames_cpu`, disabled display windows |
| `config.py` | Added `CAMERAS_MAIN`, 6 main-stream fields, `_derive_sub_url()` helper |
| `data_collection.py` | Renamed `VideoCaptureThread` to `SubStreamThread` with JPEG buffer, added `MainStreamThread` (pre-connect on first detection, `buffer_duration()` method), added crop helpers (`_scale_boxes`, `_compute_trigger_crop`, `_smooth_crops`), added `_save_vlm_crop_clip()` with timestamp-based sub/main alignment, updated `CameraState` with `main_connect_ts` field, updated `main()` with pre-connect/idle-disconnect logic and buffer-fill save condition |
