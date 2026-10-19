# Licence plates: benchmark on the owner's clips and integration design

Status: the benchmark below dates from 2026-10-09. On 2026-10-19 a smaller part of the design was built
(`box/car_plates.py`, `box/known_vehicles.py`, `box/assistant_v3/vehicles.py`; README "Car plates"). Its
differences from §5–6:

- **How cars get names.** A car gets the owner's own words for it, bound to the plate the box read in the picture
  he was answering, or to a plate he types. There is no "my car" registration step.
- **Which vehicles are read.** Every vehicle the tracker followed in an alert clip is read, parked ones included.
  The owner's case was a parked car.
- **What the alert says.** A match names the car inside the existing alert lines. No arrival or departure message
  is sent, and the house state does not change.
- **What is kept.** The full plates read in the last 3 days stay on the box in `reads.jsonl`, so a name the owner
  gives later can still be bound.
Benchmark date: 2026-10-09. Data: `home_guard_data/checks/lpr_bench_2026-10-09/`.
Tools: `home_guard_project/analysis/plates_bench/`. Code: `home_guard_project/box/plates.py`.

## 1. Verdict

- **Plates can be read only from main-stream pixels, by day, on ch6 (entrance/parking) and ch3 (pergola/parking).**
  On ch6 the owner car was matched in 22 of 31 clips where a plate was visible. On ch3 it was matched in 10 of 16.
- **The plate's width decides everything.** In native main-stream pixels, cct-s read the exact plate in:
  - 18 % of frames at 40–50 px;
  - about 72 % at 50–80 px;
  - 95 % at 80–100 px;
  - 100 % from 100 px.

  Under 50 px it fails. From 80 px it is reliable. Our cameras mostly give 40–60 px.
- **Use it to match the owner's registered cars, not to read strangers' plates.** About a fifth of the track votes
  were wrong, most of them by one digit; for example, 1609875 came out as 1605875, 1609375, 7609875 or 1608875.
  Matching against the owner's list with a tolerance of one digit absorbs most of those errors. Across 233 tracks with no plate or an unreadable
  plate there were 0 false owner matches.
- **Night is not supported.** No night clip held a readable plate in a main-stream crop, and 14 night vehicle clips
  had no main-stream crop at all. The IR bloom from the 2026-10-06 check still applies.
- **Reader:** fast-alpr parts (`yolo-v9-t-384` plate detector + `cct-s-v2-global` OCR, digits only) on the N150's
  iGPU through OpenVINO. Frigate's OCR stage (PP-OCR) and cct-xs both read less. Plate Recognizer was not tested
  because no key was available, so the spend was $0.

## 2. Dataset

All vehicle clips stored on the box between 2026-10-03 and 2026-10-09, on every camera, day and night, were
streamed read-only over ssh: 412 clips (meta, D1 clip, VLM crop mp4 and its jpg frames).

- **Pixels used.** Only the VLM crop, which is cut from the **main stream** (2592×1520 on ch3/ch6, 1920×1080 on ch2)
  around the sub-stream trigger boxes scaled to the main stream (`data_collection/vlm_crop.py`). Each frame keeps
  native pixels, apart from the resize to the clip's median crop size, which is about 1:1.
- **Pixels never used.**
  - The D1 sub-stream (704×576). On 2026-10-06 it read 0 of about 40 plates.
  - The model-input frames, which are now capped by `vlm_max_side`.
- **What was read.** 5,451 crop frames from 372 clips. The other 40 clips had no main-stream crop: they fell back to
  the whole sub-stream frame, and 14 of them were at night.
- **Per vehicle.** YOLO11s found the vehicle in the native crop. A 20 % padded box around it, with no resize, went to
  the plate detector. Each plate box was read by three OCRs: cct-s, cct-xs, and PP-OCRv4 (RapidOCR, the recognition
  stage Frigate uses).
- **Hand check.** 417 frames were checked by eye: per vehicle track, its 6 widest plate boxes of at least 40 px
  (`gt.csv`, contact sheets in `sheets2/`).
  - 224 frames show a known plate: 162 legible cold, 21 cut by the crop edge, and the rest legible only once the
    plate is known from other frames.
  - 110 frames show a plate nobody can read: a trailer plate, motion blur.
  - 83 frames are not a plate: a bin, a car door, a lamp.
- **Plates seen.** Six distinct plates:
  - 5654230, a white Toyota;
  - 1609875, a silver Subaru;
  - 35271104, a white Seat;
  - 88657702;
  - 10652002, a white Citroën van;
  - 49167901, a red car.

  The owner still has to say which of these are the family's.

Two audits were done:

- **Detector misses.** Of 24 sampled vehicle frames, from tracks with a known plate, where the detector found no
  plate of 40 px or more, 13 showed a plate to the eye. All 13 were small or oblique.
- **Night.** All 5 night vehicle crops were checked: none showed a plate.

## 3. Results per camera

Day is 06:00–19:00 local time. *Vehicle frames* counts every vehicle seen in the main-stream crops. *Exact* is the
cct-s read, per frame, on the hand-checked frames with a known plate. Votes use `plates.vote`, which needs two reads
within one digit, on plates of at least 50 px. *Owner match* is `plates.match` against the 6 plates above.

| Camera, time | Vehicle frames | Plate box ≥50 / ≥80 px | Exact per frame (eye-legible) | Track votes right / wrong / none | Owner match: tracks, clips | False reads per frame (unreadable plates) | False owner matches |
|---|---|---|---|---|---|---|---|
| ch6 day | 1,427 | 207 / 35 | 96/152 = 63 % (88/114 = 77 %) | 20 / 5 / 14 of 39 | 23/39, clips 22/31 | 4/16 | 0/85 tracks |
| ch3 day | 2,021 | 174 / 29 | 32/72 = 44 % (32/48 = 67 %) | 7 / 2 / 13 of 22 | 8/22, clips 10/16 | 30/94 (a trailer plate) | 0/148 tracks |
| ch2 day | 3 | 1 / 0 (not a plate) | – | – | – | – | – |
| ch6 night | 9 | 0 | no plate visible | – | – | – | – |
| ch1, ch5, ch8 | 0 vehicles in the crops | – | – | – | – | – | – |

**Detection.** On tracks with a known plate, a plate box of at least 40 px was found in 71 % of the vehicle frames
on ch6 and 55 % on ch3. In the other frames the plate was under 40 px, hidden (side view, a person in front), or
missed.

**Readers.** The table counts frames with a known plate, ch3 and ch6 together.

| Reader | Exact | Laptop CPU per plate |
|---|---|---|
| fast-alpr cct-s-v2 (digits-only decode) | 128/224 = 57 % (within one digit: 75 %) | 27 ms |
| fast-alpr cct-xs-v2 | 80/224 = 36 % | 9 ms |
| PP-OCRv4 detection and recognition (Frigate's OCR stage) | 63/224 = 28 % | 182 ms |

Frigate itself was not run. It is a whole NVR in Docker, and this laptop has no Docker. Its plate detector is a
YOLOv9 plate model of the same kind as fast-alpr's, and its OCR stage was tested in the row above.

**Readability by plate width.** The table covers known-plate frames with the plate whole (not cut by the crop edge).

| Plate width | Frames | Eye-legible | cct-s exact | cct-xs exact |
|---|---|---|---|---|
| 40–50 px | 55 | 24 | 18 % | 13 % |
| 50–60 px | 48 | 38 | 73 % | 52 % |
| 60–80 px | 68 | 68 | 71 % | 31 % |
| 80–100 px | 21 | 21 | 95 % | 76 % |
| ≥100 px | 11 | 11 | 100 % | 100 % |

**Sharpness matters as well as width.** A sharp, still plate read at 52–57 px; for example, 16-098-75 was read by
all three OCRs. A blurred plate failed at 56 px: the moving Seat, and the trailer.

**Voting.** Taking the 5 best frames instead of all frames cut wrong votes on ch6 from 7 to 3 at 40 px. It also lost
one right vote. "Best" means the biggest box, the sharpest crop (Laplacian variance) and the stillest car. At 50 px,
all frames gave 20 right and 5 wrong; the best 5 gave 16 right and 2 wrong. Two other rules did not help:

- adding cct-xs reads to the vote raised wrong votes;
- a per-position character vote changed nothing.

Raising the floor from 50 px to 60 px lost owner matches (23 → 16 on ch6) and gained no safety.
`MIN_PLATE_WIDTH` is now 50.

## 4. CPU cost

The N150 was measured on the live box through ssh: the models were embedded in the script and run with the box's
own OpenVINO, and nothing was installed or written. The box CPU sat at **100 %** from live inference during the
measurement, so the CPU numbers are what LPR gets on top of today's load.

| Model (median ms) | N150 CPU, 1 thread | N150 CPU, all threads | N150 iGPU | Laptop OpenVINO, 1 thread |
|---|---|---|---|---|
| Plate detector t-384 | 160 | 86 | **24** | 9.9 |
| Plate detector t-640 | 426 | 204 | 38 | 25 |
| OCR cct-s | 128 | 68 | **28** | 8.0 |
| OCR cct-xs | 13 | 13 | 23 | 1.3 |

**Per vehicle track:** 5 frames × (detector + OCR) on the iGPU is about 0.26 s. Add the decode of 5 main-stream
JPEGs (estimated at 30–50 ms each on the N150), and the total is about 0.5 s.

**Per day:** ch3 and ch6 together produced about 37 vehicle clips a day, which is under 30 s of iGPU time a day. On
the CPU it would be about 0.8 s per track, competing with live inference. So it runs on the iGPU, off the detection
loop, one track at a time.

## 5. Recommendation

1. **Turn LPR on for ch6 and ch3, by day only.** That means a colour frame, `is_ir` False, at 06:00–19:00. Keep it
   off on ch1, ch2, ch5 and ch8, and at night.
2. **Owner cars only.** The owner registers plates, for example "my car is 56-542-30, the Toyota". A vote that
   matches a registered plate within one digit is named, with the plate masked as `****230`.
   - Any other plate is dropped: it is not stored, logged or sent.
   - The event says only "a car that is not one of yours".
   - The clip keeps its normal retention.
3. **Reader:** `yolo-v9-t-384` and cct-s-v2 on the iGPU (`fast-alpr[onnx-openvino]==0.4.0`, plus the OpenVINO
   provider). The lead decides the deployment, because the box does not have these packages yet.
4. **Expect misses, not wrong names.** About 30 % of visits will not be recognised, and the plate is cut or blurred
   in many frames. For reliable reads, every plate should be at least 80 px wide. That needs a narrow-angle camera at
   the gate (a 4–6 mm lens, short shutter), or the car stopping closer to ch6.

## 6. Integration design (not built)

```
tracker (sub-stream) ── confirmed vehicle track on ch3/ch6, by day
        │  drain_vehicle_events(): track_id, first/last box (normalised), crossings, ended
        ▼
PlateSampler (per camera, off the detection loop)
        │  1. main_cap.get_clip_frames(window) → native main-stream frames (2592×1520 JPEG ring, 5 fps)
        │  2. per look: the track's box at that time, scaled sub→main exactly as vlm_crop.scale_boxes does
        │     (normalised box × main w/h), padded 20 % (plates.CROP_MARGIN), cut, NEVER resized
        │  3. plates.best_frames(boxes, sharpness, k=5, frame_size=main size)
        ▼
PlateReader.read(VehicleCrop…)   detector on the vehicle crop → plate box ≥ 50 px, centre inside the vehicle box
        │                         → OCR on the plate box at native pixels (digits only)
        ▼
plates.vote(reads)  → plates.match(vote, owner_vehicles)   (same length, ≤ 1 digit off, ambiguity = no match)
        ▼
vehicle event (only for a match): {"vehicle_id", "name", "plate_masked", "camera", "direction", "ts", "clip"}
        ▼
house state + message: "הרכב שלך (טויוטה ****230) נכנס" / "יצא"
```

### 6.1 Where the pixels come from

- The source is the main stream at native resolution, from the `MainStreamThread` ring, which is already kept for
  the VLM crop. The plate path takes its own copy of the window's frames. It never uses the D1 clip, never the VLM
  model-input frames (capped by `vlm_max_side`), and never a resized crop.
- The vehicle box comes from the tracker, which runs on the sub-stream. Today the tracker keeps only the last box and
  the foot points of a track (`tracker._Track.box`, `points`). The sampler needs a short per-look box history for
  each confirmed vehicle track: `(ts, x1, y1, x2, y2)`, normalised, for the last `CLIP_SECONDS`.
- The sub-to-main mapping is the VLM crop's own: the normalised box times the main-stream width and height. Both
  streams show the same view. A main-stream frame is paired with the sub look nearest in time; a pair more than
  0.3 s apart is skipped.
- Each look's box is padded 20 % and cut at native pixels. `plates.vehicle_crops` takes one box today; it
  needs a per-frame box. A plate box
  found inside the crop is read at its own pixels. This is the tighter second crop, and the OCR model resizes it
  internally.
- The watch-zone mask also blacks out the main stream (`MainStreamThread` applies `ZoneMask`). The owner's zone
  must therefore include the plate area of the parking spot. If it does not, every plate there is black.

### 6.2 When it runs

- **When:** on a `drain_vehicle_events()` event for a confirmed track, whose reason is `crossed`, `area` or `ended`.
  At most once per track, plus once more when the track ends if the first try gave no vote.
- **Not for parked cars.** A vehicle that has not moved is never a track (`PARKED_DISTANCE`), so a parked car costs
  nothing. "Left" and "returned" both involve a moving track.
- **Where:** in a worker thread with a queue of 4. When the queue is full, the oldest job is dropped. The worker runs
  on the OpenVINO GPU device, with the CPU as fallback; a job longer than 2 s is abandoned.
- **Off switches:**
  - night: `is_ir` on the middle frame, or a local hour outside the camera's `lpr_hours`;
  - a camera not in `lpr_cameras`;
  - a reader without its packages (`available()` False).

### 6.3 Vote, match, event

- **Vote:** `plates.vote`, needing 2 reads within one digit, over the best 5 frames, at 50 px or more.
- **Match:** `plates.match` against `owner_vehicles`. A new small store, `<state_dir>/vehicles.json`, holds entries
  like `[{"id","name","plate","added_by","added_at"}]`, written only by an owner command through the brain:
  "הרכב שלי 56-542-30 טויוטה לבנה". The brain shows the plate back masked and asks for confirmation.
- **Direction:**
  - `crossings` on the scene-map line give `in` (נכנס) or `out` (יצא).
  - Without a line, the direction comes from `area_path`: from the parking spot to the gate is `out`, and the
    reverse is `in`.
  - If the direction is unclear, the message says only "seen" (נראה).
- **House state.** A plate event never changes the state by itself. "Returned" while the house is `away` becomes a
  pending proposal ("הרכב שלך חזר, לעבור ל'בבית'?"), the same path as any relaxing change. "Left" sends a notice; it
  can suggest `away` only when every registered car is out.
- **Alerts.** A moving vehicle matched to an owner car can tell the Eye "the owner's car", which softens a
  vehicle-only alert. A person near it is still judged on their own. Unknown cars change nothing.
- **Privacy:**
  - Only matched owner plates are stored, and only masked to the last 3 digits in messages and logs.
  - Non-matches are discarded in memory.
  - Raw reads never reach `meta.json`, `events.jsonl` or the dataset.
  - The plate crop of a match is kept as an evidence keyframe under the clip's normal retention.

### 6.4 Config (box.yaml, proposed)

```yaml
plates:
  enabled: false          # shadow first
  cameras: [ameer_week_0_1_ch6, ameer_week_0_1_ch3]
  hours: "06:00-19:00"
  min_plate_px: 50
  best_frames: 5
  device: GPU             # OpenVINO; CPU fallback
  mode: shadow            # shadow: log matches to plates_shadow.jsonl (masked), send nothing
```

### 6.5 Rollout

1. **Shadow for one week** on ch6 and ch3. Log masked matches, misses and timings, and compare against the owner's
   own record of when the cars came and went.
2. **Gate:** 0 wrong-car matches, at least 60 % of the owner car's visits on ch6 recognised, and p95 below 1 s per
   track on the iGPU. Then send the messages.
3. **Tests to add during the build:**
   - the sampler maps sub boxes to main pixels at 2592×1520 and 1920×1080;
   - a crop is never resized;
   - IR frames and hours outside the window skip;
   - there is no event without an owner match;
   - masking applies in every output;
   - a full queue drops the oldest job.

## 7. Open items for the owner

1. Say which of the six plates are the family's cars, and add their names.
2. Make sure the watch zones on ch3 and ch6 include the parked cars' bumpers. The main stream is masked too.
3. Optional: a narrow-angle camera at the gate, so night and moving reads become possible (plates of 100 px or more,
   an LPR profile).

## 8. How to rerun

These scripts are in `plates_bench/`. Data goes to `LPR_BENCH_DIR`.

1. `box_pull_vehicle_clips.py` runs on the box through ssh stdin and streams a tar of the clips.
2. `extract.py <repo>` reads the plates.
3. `sheets.py 40 sheets2` makes the hand-check sheets. Write `gt.csv` from them.
4. `metrics.py <repo>` computes the results, and `vote_variants.py <repo>` compares vote rules.
5. `make_timing.py` builds the timing scripts. Then pipe each `timing_scripts/t_*.py` to the box's python over ssh.
