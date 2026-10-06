# Eval set v2 (frozen 2026-10-06)

Where: `C:\Users\ameer\Ameer\home_guard_eval\eval_set_v2` on the laptop (`manifest.jsonl`, `frames/`, `FROZEN.json`,
`picks_v2.jsonl`, `dropped_v2.jsonl`, `staged-clips-shot-list.md`). Not in git and not on S3. Run models on it with
`eval_prompt run --dir <eval_set_v2> --strict-frozen`.

`eval_set` (v1, 292 clips) is left as it was; every v1 clip is in v2 with the same `clip_id`, camera, time, label,
text and frame bytes, so v1 results score on v2 without a new model call.

## What is in it

489 clips (5 frames each): **181 alerts**, 281 normal, 27 empty. Set sha256
`2054eca82da212b6f710b5f13719e5ed8508ca4c8ca6628f675586a555ab2d11` (manifest sha256 `9834dcac...`).

| Source | Alerts | Normal | Empty | New in v2 |
|---|---|---|---|---|
| house (our cameras) | 4 | 175 | 25 | 0 |
| external (web videos in ameer_house_batch_2) | 14 | 2 | 0 | 0 |
| UCF-Crime (UCA) | 103 | 39 | 2 | 79 alerts, 22 normal |
| SmartHome-Bench | 60 | 65 | 0 | 48 alerts, 48 normal |

- Alerts by category: E1 27, E2 8, E3 18, E4 41, E5 10, E6 17, E7 7, S1 21, S2 8, S3 3, S4 6, S5 5, S6 6, S8 4.
- Day / night / unknown: alerts 39 / 113 / 29, normal and empty 139 / 133 / 36.
- Misses set (`subset: misses`): 108 alerts (UCA 53, SmartHome 44, external 7, house 4). Reasons (`hard`, several per
  clip): night 79, subtle 44, occlusion 14, loitering 8.

## How it was built

1. **The 292 v1 clips.** v1's frames copied, then `prepare --dataset home_guard_data/dataset`: every row matched v1
   on camera, label, text, time and frames, and 150 frames re-sampled from the dataset's clips were byte-identical to
   v1's.
2. **New alerts.** Candidates mined from the S3 metadata (read only):
   - *UCF-Crime (UCA)*, Burglary, Stealing, Vandalism and Arson videos not already in the set: one window per video,
     whose annotated sentence describes the act (pry, smash, climb, try a door, look in, squat by a car, set fire...)
     in a home-like scene (door, window, yard, gate, car, motorcycle) and not in a shop, office or bank. The `segment`
     is that sentence's time range inside the 10 s window, at least 2 s; the 5 frames are evenly spaced inside it
     (`add`, same sampling and 1280 px cap as the box's eval).
   - *SmartHome-Bench*, Security videos tagged `Abnormal` with a home-security reason (not weather, animals, children
     or pranks), not already in the set. Its labels are per video, not per second, so the segment is the whole 10 s
     window, and the window with the most person detections was taken.
3. **Every candidate was looked at** (5 frames side by side). 22 alerts were dropped because the act or the person is
   not visible in the 5 frames (nobody in frame, the act out of frame, title cards, a laundromat machine rather than
   a home); they are listed with the reason in `dropped_v2.jsonl`. The category was set by reading the sentence and
   the frames, not by the keyword rule.
4. **Normals from the same sources**, so false alarms stay measurable outside our house: UCA `Normal` windows of
   street, driveway and doorstep scenes (most UCA normals are shops, offices, halls and buses and were dropped) and
   SmartHome-Bench `Normal` Security windows (doorbells, deliveries, family at the door, yard life, a police officer
   at the door, a man fumbling his key card: good look-alikes).
5. **Truth on the v1 clips**: a category for each of the 54 alerts (from the text and the frames), `N10` for the empty
   scenes, `day_night` where the clip has no clock (seen on the frames; indoor UCA scenes are left unknown).
6. `freeze`.

New clips are test data: **keep `uca_eval_v2` and `smarthome_eval_v2` clips (`picks_v2.jsonl`) out of any training
set.** The v1 clips came from the tagged batches, which are training data; scores on them read optimistic for a model
fine-tuned on those batches. Report the `uca_eval_v2` / `smarthome_eval_v2` batches separately for fine-tuned models.

## Fields

`category` is a `taxonomy.py` id. `day_night` is `day` / `night` (from the clock in the clip name for our cameras, else
seen on the frames). `subset: misses` marks the hard alerts, and `hard` says why: `night` (dark or infrared),
`occlusion` (the person is partly hidden by a car, gate, wall or the frame edge), `subtle` (small or far, a short or
quiet act: trying a handle, taking a parcel, a faint figure), `loitering`. `segment` (added clips only) is the
annotated `[start, end]` in seconds of the clip the frames come from; `note` says which annotation sentence it is.

## Categories for public datasets

Set per clip by reading the annotation and the frames. The defaults by source class were:

| Source class | Category | Note |
|---|---|---|
| UCA Burglary | E1 forced entry; E2 when climbing in through a window or over a wall; S1/S2/S3/S6 when the frames only show trying a handle, looking in, walking around the house or entering the yard | |
| UCA Stealing | E4 car break-in (cars and motorcycles: prying, smashing a window, forcing the ignition); E3 theft when taking from an open car or trunk; S1 trying car doors; S2 looking into a car; S8 squatting by a car | |
| UCA Robbery | E6 weapon in use | every robbery in the set shows a gun or knife |
| UCA Shoplifting | E3 theft | none in v2 (shop scenes) |
| UCA Fighting, Assault, Abuse | E5 violence | |
| UCA Shooting | E6 weapon in use | |
| UCA Arson, Explosion | E7 fire, smoke or crash; E1/E4 when the frames show the break-in before the fire | |
| UCA RoadAccidents | E7 | none in v2 |
| UCA Vandalism | E1 when a door or window is attacked, E4 for cars; else `other` | |
| SmartHome-Bench Abnormal (Security) | read per clip: S1 trying a car door or front door, E4 rummaging in or breaking into a car, E3 taking a parcel or bike, S2 peering in, S4 loitering, S6 walking into the yard, E2 climbing, E7 fire | |
| our `[alert]` clips | read per clip: S5 for the hooded group walking in, E6 when a knife is held | |
| empty scenes | N10 | |

Normals carry no category in v2 (only the empty scenes are N10).

## Known limits

- **Only 4 alerts come from our own cameras.** The staged-clips shot list (`staged-clips-shot-list.md`) is the fix.
- UCA and SmartHome clips have no clock, so the prompt is told the time the eval runs (as for v1's external clips).
  The `day_night` field is truth for splitting the score, not something the model is told.
- Frames for the new alerts come from inside the annotated act; the box sends the 5 seconds up to the first detection,
  which may come before the act. Recall on this set is an upper bound on the box's.
- UCA is 320x240; SmartHome-Bench is mostly Ring doorbells (fish-eye, close up). Neither looks like our cameras.
