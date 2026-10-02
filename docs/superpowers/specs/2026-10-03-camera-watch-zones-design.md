# Camera watch zones ("look only here") — design

**Date:** 2026-10-03  **Status:** approved by the user, ready for an implementation plan

## Goal

The owner draws, with the mouse, the part of each camera's picture the system should watch
(for example the yard). Everything outside that area (the street) is blacked out before
anything reads the frame, so no detector, no VLM, no clip, no snapshot and no preview ever
contains it. One zone per camera, optional; a camera without a zone is watched whole.

Decisions the user made during design:

- The area outside the zone is **blacked out everywhere** (detection, AI analysis, Telegram
  snapshot, saved clips, live tiles). Not merely "ignored".
- The zone applies in **both box modes**: data collection (training clips) and inference
  (live alerts).
- Engine work is done here; the Qt drawing dialog is **briefed to Codex** (the user's standing
  preference for UI work; the box app is being edited by Codex at the time of writing).

## What exists today

- `home_guard_project/data_collection/zones.yaml` — `{zones: {<camera>: [[x, y], ...]}}`,
  normalised 0–1 polygons. Loaded by `data_collection/config.py` into `Config.ROI_ZONES`.
- `data_collection/data_collection.py` uses the polygon only as a trigger filter: a detection
  counts if its bounding-box centre lies inside (`_any_center_in_polygon`). Frames stay whole.
- `data_collection/roi_editor.py` — OpenCV click editor on the laptop, with `load_zones` /
  `save_zones`.
- `box/inference.py` — no zone support.
- The box app (PySide6, `box/app/`) shows one raw snapshot per camera on its camera page, in
  both the laptop wizard (over SSH) and the box window (local). Camera changes go through
  `box/find_cameras.py apply --changes` and then `control.request_restart()`.

## Design

### 1. Zone data

Unchanged file and format: `home_guard_project/data_collection/zones.yaml`, next to
`cameras.yaml`, keyed by camera name, normalised `[x, y]` pairs in 0–1. Rules:

- 3 to 32 points, every coordinate in `[0, 1]`; anything else is invalid.
- A missing camera, or an invalid/short polygon, means "watch the whole picture".
- Renaming a camera (`apply --changes`) renames its zone key. Disabling keeps the zone.
- Normalised coordinates are what let one zone serve both the ~704×480 sub-stream and the
  1080p/1520p main stream.

### 2. One module: `home_guard_project/data_collection/zones.py`

Pure helpers, no camera or model imports beyond numpy/cv2:

- `load_zones(path) -> Dict[str, List[Tuple[float, float]]]` (damaged file → `{}` + warning).
- `save_zone(camera, points, path)`, `clear_zone(camera, path)`, `rename_zone(old, new, path)`;
  all rewrite the YAML atomically (temp file + replace) and keep other cameras' zones.
- `validate_points(points) -> List[Tuple[float, float]]` (raises `ValueError` with a plain
  message on <3 or >32 points or out-of-range coordinates).
- `class ZoneMask`: built from a polygon (or `None`). `apply(frame) -> frame` returns the frame
  unchanged when there is no polygon; otherwise returns a copy with every pixel outside the
  polygon set to black. The uint8 pixel mask is cached per frame shape and rebuilt only when
  the shape changes, so the per-frame cost is one `cv2.bitwise_and`.
- `roi_editor.py` and `config.py` import their load/save from here (no second copy).

### 3. Readers: the only place the mask is applied

Right after a successful `cap.read()`, before any resize, encode, store or publish:

- `data_collection.SubStreamThread` (detection stream) and `MainStreamThread` (clip stream)
  each hold a `ZoneMask` built from `cfg.ROI_ZONES.get(name)`.
- `inference._Stream` holds one built from the same config (`cam_cfg.ROI_ZONES`, which
  `inference.run()` already loads via `load_config()`).

Everything downstream is therefore masked with no further code: YOLO input, the VLM frames,
the clip ring and saved clips, the alert snapshot, the preview tiles, false-positive clips.
The collector's centre-in-polygon trigger filter (`roi_polygon`, `_any_center_in_polygon`,
and the `roi` block in the clip meta that reports it) is removed: black pixels yield no
detections, and two mechanisms for one rule is one too many. The on-screen polygon outline in
the laptop display window may stay as a visual aid.

### 4. Box command (`box/find_cameras.py`)

New subcommands, same `--json` convention and camera-name validation as the existing ones:

- `zones` → `{"zones": {<camera>: [[x, y], ...]}}`
- `set-zone --camera NAME --points "x,y x,y x,y ..."` → `{"camera", "points"}`; invalid input
  → `{"error": "<plain message>"}` and exit 1.
- `clear-zone --camera NAME` → `{"camera", "points": null}`.
- `apply --changes` additionally renames zone keys with the cameras.

After a successful set/clear the caller requests a restart of the running mode (same path as
camera changes: `control.request_restart()` locally, the wizard's existing restart over SSH),
so the zone is live within seconds. Both modes read `zones.yaml` once at start.

### 5. Drawing dialog (brief for Codex; details in the implementation plan)

Camera page (wizard and box window), under each camera photo: a button "Set the area to
watch". It opens the raw snapshot large:

- Left-click adds a corner; drag moves a corner; right-click or an Undo button removes the
  last; Clear removes all (= watch everything); Save applies; Cancel discards.
- While drawing, the outside of the polygon is darkened live, so the user sees exactly what
  will be ignored. Coordinates are normalised against the photo's own size.
- A warning appears (and Save still works) when the zone covers under 5 % of the picture.
- Save calls `set-zone` / `clear-zone` through the same local/remote controls path as camera
  changes, then restarts the running mode. The tile shows "Watching: the area you drew" or
  "Watching: the whole picture". Demo mode draws on the demo picture and saves nowhere.
- The editor's photo stays unmasked (snapshots are captured separately from the readers), so
  the street remains visible to draw around. The dashboard tiles, fed by the readers, show the
  masked view.

### 6. Failure handling

- Damaged `zones.yaml`: warning, no zones, cameras watched whole (as today).
- Invalid points are rejected by `validate_points`; nothing is clamped silently.
- A restart that fails after Save surfaces the same failure the camera page shows today.
- A zone that blacks out a camera entirely is the user's explicit choice; the 5 % warning is
  the only guard.

### 7. Tests (written before the code)

In `tests/box/` (the suite the box runs: `.venv/Scripts/python.exe -m unittest discover -s tests/box`):

- `test_zones.py`: `ZoneMask` passthrough without a polygon; inside pixels kept, outside
  zeroed; the same normalised polygon masks the matching regions at two resolutions; the
  cached mask is rebuilt when the frame shape changes; `validate_points` accepts 3–32 in-range
  points and rejects the rest; save/load/clear/rename round trip on a temp file; a damaged
  file loads as `{}`.
- `test_find_zones.py`: `zones`, `set-zone`, `clear-zone` JSON output and exit codes on temp
  files; `apply --changes` renames a zone with its camera.
- Readers: the masking step is a small method on each reader (`_masked(frame)`), tested
  directly with fake frames; capture itself is not tested.
- `test_inference.py`: `_Stream` applies its mask before `read()` returns and before the ring
  receives the frame.

### 8. Order of work and ownership

1. `zones.py` + tests, config/roi_editor switched to it.
2. Readers (collector, inference) + removal of the centre-in-polygon filter. `inference.py` is
   owned by session home-guard-fa and `find_cameras.py` by its owner; coordinate before
   editing, commit by explicit path on `beelink-collector-box`.
3. Box command + tests.
4. Codex brief for the dialog (in the `home_guard_ui` worktree, branch `box-app-ui`).

## Out of scope

- Multiple zones per camera, exclusion zones, per-class zones.
- Changing the YOLO or VLM prompts; training-data re-labelling of already collected clips.
- The laptop OpenCV editor stays as is (it keeps working through the shared module).
