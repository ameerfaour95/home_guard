# Camera Watch Zones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the owner draw, per camera, the area the box should watch; everything outside it is blacked out before any detector, VLM, clip, snapshot or preview sees the frame.

**Architecture:** One pure module (`data_collection/zones.py`) owns the zone file and a cached pixel mask (`ZoneMask`). The mask is applied at the three places a frame enters the system (collector sub-stream, collector main-stream, inference stream), so nothing downstream changes. The box's camera CLI gains `zones` / `set-zone` / `clear-zone`; the PySide6 drawing dialog is specified in a brief for Codex.

**Tech Stack:** Python 3.12 (uv-managed `.venv`), numpy + OpenCV (`cv2.fillPoly`, `cv2.bitwise_and`), PyYAML, `unittest`; PySide6 for the dialog (Codex).

Spec: `docs/superpowers/specs/2026-10-03-camera-watch-zones-design.md`.

## Global Constraints

- Shell: Git Bash. Python: `.venv/Scripts/python.exe` from the repo root `C:\Users\ameer\Ameer\home_guard`.
- Test command for everything: `.venv/Scripts/python.exe -m unittest discover -s tests/box` (expected today: `Ran 418 tests ... OK`; the count grows with each task).
- Branch `beelink-collector-box`; the working tree is shared with other Claude sessions, so **commit by explicit path only** (`git commit -m ... -- path1 path2`), never `git add -A`, never stash/reset.
- `*.md` is gitignored: documentation files need `git add -f <file>` before the commit.
- Do **not** edit `home_guard_project/box/app/` (Codex is working there) or any file owned by another session without telling them first: `home_guard_project/box/inference.py` is owned by session `home-guard-fa` (message it before Task 3 and after). `find_cameras.py` is shared; announce the edit.
- Zone file format is fixed: `zones: {<camera>: [[x, y], ...]}`, normalised 0–1, 3–32 points, 4 decimals.
- `--points` on the command line uses **no quotes and no spaces**: `x,y;x,y;x,y` (the wizard's SSH path refuses quotes).
- Attribution line on every commit: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility |
|---|---|
| `home_guard_project/data_collection/zones.py` (new) | Zone file I/O (`load_zones`, `save_zones`, `save_zone`, `clear_zone`, `rename_zone`), validation (`validate_points`, `parse_points`), the mask (`ZoneMask`), `mask_for`. |
| `home_guard_project/data_collection/config.py` (modify) | Load zones through `zones.load_zones`. |
| `home_guard_project/data_collection/roi_editor.py` (modify) | Drop its private copies of load/save; import them from `zones`. |
| `home_guard_project/data_collection/data_collection.py` (modify) | Readers accept a `ZoneMask` and apply it; the centre-in-polygon trigger filter goes away. |
| `home_guard_project/box/inference.py` (modify) | `_Stream` accepts a mask and applies it in one `_ingest` step; `run()` builds masks from the config. |
| `home_guard_project/box/find_cameras.py` (modify) | `zones`, `set-zone`, `clear-zone` subcommands; `apply` renames zones with cameras. |
| `home_guard_project/box/README.md` (modify) | Document the zone command. |
| `tests/box/test_zones.py` (new) | Unit tests for `zones.py`. |
| `tests/box/test_find_zones.py` (new) | Unit tests for the CLI handlers. |
| `tests/box/test_inference.py` (modify) | `_Stream._ingest` masks before `read()` and before the ring. |
| `docs/superpowers/plans/2026-10-03-codex-brief-watch-zone-dialog.md` (committed), copied to `C:\Users\ameer\Ameer\home_guard_ui\CODEX_BRIEF_ZONES.md` | The brief for Codex (gpt-6-astra): the premium drawing dialog. |

---

### Task 1: `zones.py` — the zone file and the mask

**Files:**
- Create: `home_guard_project/data_collection/zones.py`
- Modify: `home_guard_project/data_collection/config.py:19` (`_ZONES_PATH`) and `:174-187` (`_load_zones`) and `:219` (its call)
- Modify: `home_guard_project/data_collection/roi_editor.py:31-66` (its `_ZONES_PATH`, `_HEADER`, `load_zones`, `save_zones`)
- Test: `tests/box/test_zones.py`

**Interfaces:**
- Produces (used by Tasks 2–5):
  - `ZONES_PATH: str` — `<data_collection dir>/zones.yaml`
  - `Point = Tuple[float, float]`
  - `validate_points(points) -> List[Point]` — raises `ValueError` (plain message) unless 3–32 points, each coordinate a number in `[0, 1]`; returns tuples rounded to 4 decimals.
  - `parse_points(text: str) -> List[Point]` — `"0.1,0.2;0.9,0.2;0.5,0.9"` (also whitespace separators) → validated list.
  - `load_zones(path=ZONES_PATH) -> Dict[str, List[Point]]` — only valid polygons; damaged file → `{}`.
  - `save_zones(zones, path=ZONES_PATH) -> None` — whole-file write (kept for `roi_editor.py`).
  - `save_zone(camera, points, path=ZONES_PATH) -> List[Point]`, `clear_zone(camera, path=ZONES_PATH) -> bool`, `rename_zone(old, new, path=ZONES_PATH) -> bool`.
  - `class ZoneMask(polygon: Optional[Sequence[Point]])` with `.active: bool` and `.apply(frame) -> frame`.
  - `mask_for(zones: Dict[str, List[Point]], camera: str) -> ZoneMask`.

- [ ] **Step 1: Write the failing tests**

Create `tests/box/test_zones.py`:

```python
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from home_guard_project.data_collection import zones as z

LEFT_HALF = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]


def white(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


class ValidatePointsTest(unittest.TestCase):
    def test_three_to_thirty_two_points_in_range_are_accepted_and_rounded(self) -> None:
        self.assertEqual(z.validate_points([(0, 0), (1, 0), (0.123456, 1)]), [(0.0, 0.0), (1.0, 0.0), (0.1235, 1.0)])
        self.assertEqual(len(z.validate_points([(i / 40, 0.5) for i in range(32)])), 32)

    def test_too_few_or_too_many_points_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            z.validate_points([(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            z.validate_points([(i / 40, 0.5) for i in range(33)])

    def test_points_outside_the_picture_or_not_numbers_are_refused(self) -> None:
        for bad in ([(0, 0), (1.2, 0), (0, 1)], [(0, 0), (1, -0.1), (0, 1)], [(0, 0), ("a", 0), (0, 1)], [(0, 0), (1,), (0, 1)]):
            with self.assertRaises(ValueError):
                z.validate_points(bad)


class ParsePointsTest(unittest.TestCase):
    def test_semicolon_pairs_are_the_command_line_form(self) -> None:
        self.assertEqual(z.parse_points("0.1,0.2;0.9,0.2;0.5,0.9"), [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)])

    def test_whitespace_between_pairs_is_accepted_too(self) -> None:
        self.assertEqual(z.parse_points("0.1,0.2 0.9,0.2\n0.5,0.9"), [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)])

    def test_garbage_is_refused_with_a_plain_message(self) -> None:
        with self.assertRaises(ValueError):
            z.parse_points("0.1,0.2;nope;0.5,0.9")
        with self.assertRaises(ValueError):
            z.parse_points("")


class ZoneMaskTest(unittest.TestCase):
    def test_no_polygon_returns_the_very_same_frame(self) -> None:
        frame = white(10, 20)
        self.assertIs(z.ZoneMask(None).apply(frame), frame)
        self.assertFalse(z.ZoneMask(None).active)

    def test_outside_the_polygon_is_black_and_the_original_is_untouched(self) -> None:
        frame = white(10, 20)
        out = z.ZoneMask(LEFT_HALF).apply(frame)
        self.assertEqual(int(out[:, :10].min()), 255)   # inside: kept
        self.assertEqual(int(out[:, 11:].max()), 0)     # outside: black
        self.assertEqual(int(frame.min()), 255)          # caller's frame not modified
        self.assertTrue(z.ZoneMask(LEFT_HALF).active)

    def test_one_normalised_polygon_fits_both_stream_resolutions(self) -> None:
        mask = z.ZoneMask(LEFT_HALF)
        small = mask.apply(white(10, 20))
        large = mask.apply(white(20, 40))               # same instance, bigger frame: mask rebuilt
        self.assertEqual(int(small[:, :10].min()), 255)
        self.assertEqual(int(small[:, 11:].max()), 0)
        self.assertEqual(int(large[:, :20].min()), 255)
        self.assertEqual(int(large[:, 22:].max()), 0)
        again = mask.apply(white(10, 20))               # and back again
        self.assertEqual(int(again[:, 11:].max()), 0)

    def test_mask_for_an_unknown_camera_watches_everything(self) -> None:
        zones = {"yard": LEFT_HALF}
        self.assertTrue(z.mask_for(zones, "yard").active)
        self.assertFalse(z.mask_for(zones, "street").active)


class ZoneFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "zones.yaml")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_file_means_no_zones(self) -> None:
        self.assertEqual(z.load_zones(self.path), {})

    def test_save_one_zone_then_load_it(self) -> None:
        saved = z.save_zone("yard", [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)], self.path)
        self.assertEqual(saved, [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)])
        self.assertEqual(z.load_zones(self.path), {"yard": [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)]})

    def test_saving_a_second_camera_keeps_the_first(self) -> None:
        z.save_zone("yard", LEFT_HALF, self.path)
        z.save_zone("gate", [(0, 0), (1, 0), (1, 1)], self.path)
        self.assertEqual(set(z.load_zones(self.path)), {"yard", "gate"})

    def test_clear_removes_only_that_camera(self) -> None:
        z.save_zone("yard", LEFT_HALF, self.path)
        z.save_zone("gate", [(0, 0), (1, 0), (1, 1)], self.path)
        self.assertTrue(z.clear_zone("yard", self.path))
        self.assertFalse(z.clear_zone("yard", self.path))     # already gone
        self.assertEqual(set(z.load_zones(self.path)), {"gate"})

    def test_rename_moves_the_zone_with_the_camera(self) -> None:
        z.save_zone("yard", LEFT_HALF, self.path)
        self.assertTrue(z.rename_zone("yard", "garden", self.path))
        self.assertFalse(z.rename_zone("yard", "garden", self.path))   # nothing left to move
        self.assertEqual(set(z.load_zones(self.path)), {"garden"})

    def test_a_damaged_file_loads_as_no_zones(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones: [1, 2\n")
        self.assertEqual(z.load_zones(self.path), {})

    def test_an_invalid_polygon_in_the_file_is_skipped(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones:\n  yard: [[0, 0], [1, 1]]\n  gate: [[0, 0], [1, 0], [1, 1]]\n  bad: [[0, 0], [2, 0], [1, 1]]\n")
        self.assertEqual(z.load_zones(self.path), {"gate": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]})

    def test_save_zones_writes_the_whole_dict_for_the_laptop_editor(self) -> None:
        z.save_zones({"yard": [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]}, self.path)
        self.assertEqual(z.load_zones(self.path), {"yard": [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)]})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests/box -p test_zones.py`
Expected: `ImportError: cannot import name 'zones'` (module does not exist yet).

- [ ] **Step 3: Write `zones.py`**

Create `home_guard_project/data_collection/zones.py`:

```python
"""Watch zones: the part of each camera's picture the box looks at.

One optional polygon per camera, in normalised 0-1 coordinates, kept in
``zones.yaml`` next to ``cameras.yaml``. :class:`ZoneMask` blacks out every
pixel outside the polygon; it is applied once, where a frame enters the
system, so the detector, the VLM, the clips, the snapshots and the previews
all see the same masked picture. A camera without a zone is watched whole.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

log = logging.getLogger(__name__)

Point = Tuple[float, float]

_DIR = os.path.dirname(os.path.abspath(__file__))
ZONES_PATH = os.path.join(_DIR, "zones.yaml")

MIN_POINTS = 3
MAX_POINTS = 32
DECIMALS = 4

_HEADER = (
    "# ──────────────────────────────────────────────────────────────────────────────\n"
    "#  Watch zones — normalised polygon corners per camera; everything outside is\n"
    "#  blacked out. Cameras without an entry are watched whole.\n"
    "#  DO NOT COMMIT (deployment-specific)\n"
    "# ──────────────────────────────────────────────────────────────────────────────\n\n"
)
_PAIR_SPLIT = re.compile(r"[;\s]+")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_points(points: Any) -> List[Point]:
    """The polygon as (x, y) tuples rounded to 4 decimals, or ValueError in plain words."""
    if not isinstance(points, (list, tuple)):
        raise ValueError("the zone must be a list of corners")
    if not MIN_POINTS <= len(points) <= MAX_POINTS:
        raise ValueError(f"a zone needs {MIN_POINTS} to {MAX_POINTS} corners (got {len(points)})")
    out: List[Point] = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError("every corner must be two numbers, x and y") from None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("every corner must lie inside the picture (0 to 1)")
        out.append((round(x, DECIMALS), round(y, DECIMALS)))
    return out


def parse_points(text: str) -> List[Point]:
    """``"x,y;x,y;x,y"`` (whitespace between pairs is fine too) -> validated corners."""
    pairs = [t for t in _PAIR_SPLIT.split(str(text or "").strip()) if t]
    points: List[Any] = []
    for pair in pairs:
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"corner {pair!r} must be x,y")
        points.append(parts)
    return validate_points(points)


# ----------------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------------
def _read_raw(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("Could not read %s (%s); no zones", path, exc)
        return {}
    raw = data.get("zones", {}) if isinstance(data, dict) else {}
    return raw if isinstance(raw, dict) else {}


def load_zones(path: str = ZONES_PATH) -> Dict[str, List[Point]]:
    """``{camera: [(x, y), ...]}`` with only the valid polygons; a damaged file is no zones."""
    zones: Dict[str, List[Point]] = {}
    for camera, pts in _read_raw(path).items():
        try:
            zones[str(camera)] = validate_points(pts)
        except ValueError as exc:
            log.warning("Zone for %s ignored: %s", camera, exc)
    return zones


def save_zones(zones: Dict[str, Sequence[Sequence[float]]], path: str = ZONES_PATH) -> None:
    """Write the whole file (temp file + replace, so a reader never sees half a file)."""
    data = {"zones": {str(k): [[float(x), float(y)] for x, y in v] for k, v in zones.items()}}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.dump(data, f, default_flow_style=None, allow_unicode=True)
    os.replace(tmp, path)
    log.info("Wrote %d zone(s) to %s", len(zones), path)


def save_zone(camera: str, points: Any, path: str = ZONES_PATH) -> List[Point]:
    """Set one camera's zone, keeping the others. Returns the corners as stored."""
    corners = validate_points(points)
    zones: Dict[str, Any] = dict(_read_raw(path))
    zones[str(camera)] = corners
    save_zones(zones, path)
    return corners


def clear_zone(camera: str, path: str = ZONES_PATH) -> bool:
    """Remove one camera's zone (watch everything). True if there was one."""
    zones: Dict[str, Any] = dict(_read_raw(path))
    if str(camera) not in zones:
        return False
    del zones[str(camera)]
    save_zones(zones, path)
    return True


def rename_zone(old: str, new: str, path: str = ZONES_PATH) -> bool:
    """Carry a zone along when its camera is renamed. True if there was one to move."""
    zones: Dict[str, Any] = dict(_read_raw(path))
    if str(old) not in zones or old == new:
        return False
    zones[str(new)] = zones.pop(str(old))
    save_zones(zones, path)
    return True


# ----------------------------------------------------------------------------
# The mask
# ----------------------------------------------------------------------------
class ZoneMask:
    """Blacks out everything outside a camera's zone. ``ZoneMask(None)`` passes frames through.

    The pixel mask is built once per frame size and cached, so a frame costs
    one ``cv2.bitwise_and``. Not thread-safe: give each reader its own.
    """

    def __init__(self, polygon: Optional[Sequence[Point]]) -> None:
        self.polygon: Optional[List[Point]] = validate_points(polygon) if polygon else None
        self._shape: Optional[Tuple[int, int]] = None
        self._mask: Optional[np.ndarray] = None

    @property
    def active(self) -> bool:
        return self.polygon is not None

    def apply(self, frame: Any) -> Any:
        """The frame with everything outside the zone black (a new array), or the frame itself when there is no zone."""
        if self.polygon is None or frame is None:
            return frame
        import cv2  # noqa: PLC0415 - keep the import cost off modules that never mask

        h, w = frame.shape[:2]
        if self._shape != (h, w) or self._mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            corners = np.array([[int(round(x * (w - 1))), int(round(y * (h - 1)))] for x, y in self.polygon],
                               dtype=np.int32)
            cv2.fillPoly(mask, [corners], 255)
            self._mask, self._shape = mask, (h, w)
        return cv2.bitwise_and(frame, frame, mask=self._mask)


def mask_for(zones: Dict[str, List[Point]], camera: str) -> ZoneMask:
    """The camera's mask from a loaded zones dict; a camera without a zone gets a pass-through."""
    return ZoneMask(zones.get(camera))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests/box -p test_zones.py`
Expected: `Ran 19 tests ... OK`

- [ ] **Step 5: Switch `config.py` to the shared loader**

In `home_guard_project/data_collection/config.py`:

Replace the whole `_load_zones` function (lines 174–187):

```python
def _load_zones(path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load normalised polygon zones from zones.yaml."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("zones", {})
    if not isinstance(raw, dict):
        return {}
    zones: Dict[str, List[Tuple[float, float]]] = {}
    for cam, pts in raw.items():
        if isinstance(pts, list) and len(pts) >= 3:
            zones[str(cam)] = [(float(p[0]), float(p[1])) for p in pts]
    return zones
```

with:

```python
def _load_zones(path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load the watch zones (normalised polygons) from zones.yaml; see zones.py."""
    from .zones import load_zones  # noqa: PLC0415

    return load_zones(path)
```

Leave `_ZONES_PATH` and the call at line 219 as they are.

- [ ] **Step 6: Switch `roi_editor.py` to the shared helpers**

In `home_guard_project/data_collection/roi_editor.py`, replace lines 31–66 (from `_DIR = ...` through the end of `save_zones`) with:

```python
_DIR = os.path.dirname(os.path.abspath(__file__))

try:  # run as a module
    from .zones import ZONES_PATH as _ZONES_PATH, load_zones, save_zones  # noqa: F401
except ImportError:  # run as a script: py home_guard_project/data_collection/roi_editor.py
    sys.path.insert(0, _DIR)
    from zones import ZONES_PATH as _ZONES_PATH, load_zones, save_zones  # type: ignore  # noqa: F401
```

(`yaml` stays imported: `main()` still reads `cameras.yaml` with it.)

- [ ] **Step 7: Run the whole box suite and the two import smoke checks**

Run:
```bash
.venv/Scripts/python.exe -m unittest discover -s tests/box 2>&1 | tail -3
.venv/Scripts/python.exe -c "from home_guard_project.data_collection import config, roi_editor; print('imports ok')"
```
Expected: `OK` with 418 + 19 = 437 tests, and `imports ok`.

- [ ] **Step 8: Commit**

```bash
git commit -m "Watch zones: one module for the zone file and the pixel mask" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- home_guard_project/data_collection/zones.py home_guard_project/data_collection/config.py home_guard_project/data_collection/roi_editor.py tests/box/test_zones.py
```

---

### Task 2: The collector's readers mask every frame

**Files:**
- Modify: `home_guard_project/data_collection/data_collection.py` — imports (top), `_any_center_in_polygon` (lines 145–165, delete), `SubStreamThread.__init__` (360), its `_reader` (~436), `MainStreamThread.__init__` (514), its `_reader` (~588), `CameraState` (851–872), meta `roi` block (1133–1139), state build (1244–1262), trigger filter (1332–1339), main-stream construction (1377, 1403).

**Interfaces:**
- Consumes: `ZoneMask`, `mask_for` from Task 1.
- Produces: `SubStreamThread(cfg, src, mask: Optional[ZoneMask] = None)`, `MainStreamThread(cfg, src, mask: Optional[ZoneMask] = None)`; both expose `.mask`.

There is no unit test for the capture threads (they open RTSP and start a thread in `__init__`); the masking itself is covered by Task 1, and this task is verified by the import smoke check and a diff review.

- [ ] **Step 1: Import the mask**

Find the config import: `grep -n "^from .config import\|^from \.config\|import Config" home_guard_project/data_collection/data_collection.py`. Directly below that line add:

```python
from .zones import ZoneMask, mask_for
```

- [ ] **Step 2: Give both readers a mask and apply it once per stored frame**

`SubStreamThread.__init__` — change the signature and add one attribute:

```python
    def __init__(self, cfg: Config, src: str, mask: Optional[ZoneMask] = None):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", cfg.OPENCV_FFMPEG_CAPTURE_OPTIONS)
        self.cfg = cfg
        self.src = src
        self.mask = mask or ZoneMask(None)   # blacks out everything outside the camera's watch zone
```

In `SubStreamThread._reader`, right after

```python
            if (now - self.last_store_ts) < self.store_interval:
                continue
            self.last_store_ts = now
```

insert:

```python
            frame = self.mask.apply(frame)   # before resize, latest_frame and the buffer: nothing sees the outside
```

`MainStreamThread.__init__` — same signature change and attribute:

```python
    def __init__(self, cfg: Config, src: str, mask: Optional[ZoneMask] = None):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", cfg.OPENCV_FFMPEG_CAPTURE_OPTIONS)
        self.cfg = cfg
        self.src = src
        self.mask = mask or ZoneMask(None)
```

In `MainStreamThread._reader`, right after its own `self.last_store_ts = now` and before `self._frame_shape = ...`, insert:

```python
            frame = self.mask.apply(frame)
```

- [ ] **Step 3: Build the masks where the readers are constructed**

Line 1244:

```python
        cap = SubStreamThread(cfg, rtsp_sub, mask=mask_for(cfg.ROI_ZONES, name))
        norm_pts = cfg.ROI_ZONES.get(name)
        if norm_pts:
            log.info("%s: watch zone active (%d corners); everything outside is blacked out", name, len(norm_pts))
```

Lines 1377 and 1403 (both places a main stream is opened):

```python
                    st.main_cap = MainStreamThread(cfg, st.rtsp_main, mask=ZoneMask(st.roi_norm))
```

- [ ] **Step 4: Remove the centre-in-polygon filter (the mask subsumes it)**

1. Delete the function `_any_center_in_polygon` (lines 145–165). Keep `_norm_polygon_to_px`: the display overlay still draws the outline.
2. In `CameraState` delete the field `roi_polygon: Optional[np.ndarray]` (keep `roi_norm`).
3. In the state build (line ~1260) delete `roi_polygon=None,`.
4. Delete the trigger block (lines 1332–1339):

```python
                    if st.roi_norm is not None:
                        if st.roi_polygon is None:
                            fh, fw = frame.shape[:2]
                            st.roi_polygon = _norm_polygon_to_px(st.roi_norm, fw, fh)
                        st.trigger_detected = _any_center_in_polygon(
                            results[0].boxes, class_ids=cfg.TRIGGER_CLASS_IDS,
                            polygon=st.roi_polygon,
                        )
```

5. The clip meta keeps its `roi` block (it is part of the `.meta.json` contract) but its flag now describes the mask:

```python
        "roi": {
            "active": st.roi_norm is not None,            # the clip was masked to this zone
            "polygon_normalized": (
                [[float(x), float(y)] for x, y in st.roi_norm]
                if st.roi_norm else None
            ),
        },
```

- [ ] **Step 5: Verify**

Run:
```bash
grep -n "roi_polygon\|_any_center_in_polygon" home_guard_project/data_collection/data_collection.py
.venv/Scripts/python.exe -c "import home_guard_project.data_collection.data_collection as d; import inspect; print('mask' in inspect.signature(d.SubStreamThread.__init__).parameters, 'mask' in inspect.signature(d.MainStreamThread.__init__).parameters)"
.venv/Scripts/python.exe -m unittest discover -s tests/box 2>&1 | tail -3
```
Expected: the grep prints nothing; `True True`; the suite is `OK` (437).

- [ ] **Step 6: Commit**

```bash
git commit -m "Collector: every frame is masked to the camera's watch zone as it is read" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- home_guard_project/data_collection/data_collection.py
```

---

### Task 3: The inference stream masks every frame

**Files:**
- Modify: `home_guard_project/box/inference.py` — `_Stream` (lines 455–497) and `run()` line 706.
- Test: `tests/box/test_inference.py` (append).

**Interfaces:**
- Consumes: `ZoneMask`, `mask_for` (Task 1).
- Produces: `_Stream(name, url, ring=None, mask=None)` and `_Stream._ingest(frame, now)`; `inference_preview.adapt_stream` already forwards `**kwargs`, so it needs no change.

Before editing: message session `home-guard-fa` ("touching inference.py `_Stream` for watch zones; will tell you when pushed").

- [ ] **Step 1: Write the failing test**

Append to `tests/box/test_inference.py`:

```python


class _FakeRing:
    def __init__(self) -> None:
        self.added: list = []

    def wants(self, now: float) -> bool:
        return True

    def add(self, now: float, encoded) -> None:
        self.added.append(encoded)


class StreamMaskTest(unittest.TestCase):
    def test_a_frame_is_masked_once_before_read_and_before_the_clip_ring(self) -> None:
        import threading

        import numpy as np

        from home_guard_project.data_collection.zones import ZoneMask

        stream = inf._Stream.__new__(inf._Stream)      # no capture, no thread
        stream._lock = threading.Lock()
        stream._frame = None
        stream._ring = _FakeRing()
        stream._mask = ZoneMask([(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])   # the left half
        frame = np.full((10, 20, 3), 255, dtype=np.uint8)

        with mock.patch("home_guard_project.box.alert_clips.encode_frame", side_effect=lambda f: f):
            stream._ingest(frame, now=1.0)

        seen = stream.read()
        self.assertEqual(int(seen[:, :10].min()), 255)
        self.assertEqual(int(seen[:, 11:].max()), 0)
        (ringed,) = stream._ring.added
        self.assertEqual(int(ringed[:, 11:].max()), 0)

    def test_without_a_mask_the_frame_is_kept_whole(self) -> None:
        import threading

        import numpy as np

        stream = inf._Stream.__new__(inf._Stream)
        stream._lock = threading.Lock()
        stream._frame = None
        stream._ring = None
        stream._mask = None
        stream._ingest(np.full((4, 4, 3), 7, dtype=np.uint8), now=1.0)
        self.assertEqual(int(stream.read().min()), 7)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests/box -p test_inference.py`
Expected: 2 errors, `AttributeError: '_Stream' object has no attribute '_ingest'`.

- [ ] **Step 3: Implement `_ingest` and the `mask` parameter**

In `home_guard_project/box/inference.py` replace the `_Stream` class body from `def __init__` through `def read` with:

```python
    def __init__(self, name: str, url: str, ring: Any = None, mask: Any = None) -> None:
        import cv2  # noqa: PLC0415

        self.name = name
        self.url = url
        self._ring = ring
        self._mask = mask                    # zones.ZoneMask, or None to watch the whole picture
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        import cv2  # noqa: PLC0415

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.5)
                self._cap.release()
                self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            self._ingest(frame, time.time())

    def _ingest(self, frame: Any, now: float) -> None:
        """One decoded frame: masked to the watch zone first, then kept as the latest frame and offered to the clip ring."""
        if self._mask is not None:
            frame = self._mask.apply(frame)
        with self._lock:
            self._frame = frame
        if self._ring is not None and self._ring.wants(now):
            from .alert_clips import encode_frame  # noqa: PLC0415

            self._ring.add(now, encode_frame(frame))

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()
```

In `run()`, replace line 706:

```python
    streams = {name: _Stream(name, url, ring=rings[name]) for name, url in cameras.items()}
```

with:

```python
    from ..data_collection.zones import mask_for  # noqa: PLC0415

    zones = dict(getattr(cam_cfg, "ROI_ZONES", {}) or {})
    for name in cameras:
        if name in zones:
            log.info("[%s] watch zone active (%d corners); everything outside is blacked out", name, len(zones[name]))
    streams = {name: _Stream(name, url, ring=rings[name], mask=mask_for(zones, name)) for name, url in cameras.items()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests/box 2>&1 | tail -3`
Expected: `OK` (439). The existing `PreviewAdapterTest` still passes because `adapt_stream` forwards `**kwargs`.

- [ ] **Step 5: Commit and tell home-guard-fa**

```bash
git commit -m "Box inference: the camera stream is masked to its watch zone before anything reads it" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- home_guard_project/box/inference.py tests/box/test_inference.py
```

Then message `home-guard-fa`: commit hash, "inference.py is yours again".

---

### Task 4: Box command — `zones`, `set-zone`, `clear-zone`, rename on `apply`

**Files:**
- Modify: `home_guard_project/box/find_cameras.py` — constants (after line 83), `apply_changes` (108–157), argparse + main (453–530).
- Modify: `home_guard_project/box/README.md` — new subsection under "Customer setup (from the laptop)".
- Test: `tests/box/test_find_zones.py`

**Interfaces:**
- Consumes: `zones.parse_points`, `save_zone`, `clear_zone`, `rename_zone`, `load_zones` (Task 1).
- Produces (the Codex dialog calls these over the existing `find_cameras --json` path):
  - `zone_rows(cameras_path=CAMERAS_PATH, zones_path=ZONES_PATH) -> {"cameras": [{"name": str, "points": [[x, y], ...]}]}` — every camera (active and disabled), `points` is `[]` for "whole picture".
  - `set_zone(camera, points_text, cameras_path, zones_path, restart=True) -> {"camera", "points"}` — `ValueError` on unknown camera / bad points.
  - `clear_zone_command(camera, cameras_path, zones_path, restart=True) -> {"camera", "points": []}`.
  - CLI: `find_cameras --json zones`, `find_cameras --json set-zone --camera NAME --points x,y;x,y;x,y`, `find_cameras --json clear-zone --camera NAME`. Errors print `{"error": "..."}` and exit 1.

- [ ] **Step 1: Write the failing tests**

Create `tests/box/test_find_zones.py`:

```python
from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box import find_cameras as fc
from home_guard_project.data_collection import zones as z

URL = "rtsp://admin:s3cret@192.168.1.50:554/unicast/c1/s0/live"


class ZoneCommandsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cameras = os.path.join(self.tmp.name, "cameras.yaml")
        self.zones = os.path.join(self.tmp.name, "zones.yaml")
        fc._write_cameras({"yard": URL, "gate": URL}, {"old_cam": URL}, self.cameras)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_zone_rows_list_every_camera_with_an_empty_zone_by_default(self) -> None:
        rows = fc.zone_rows(self.cameras, self.zones)["cameras"]
        self.assertEqual(sorted(r["name"] for r in rows), ["gate", "old_cam", "yard"])   # YAML stores keys sorted
        self.assertEqual([r["points"] for r in rows], [[], [], []])

    def test_set_zone_stores_the_corners_and_reports_them(self) -> None:
        out = fc.set_zone("yard", "0.1,0.2;0.9,0.2;0.5,0.9", self.cameras, self.zones, restart=False)
        self.assertEqual(out, {"camera": "yard", "points": [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]})
        self.assertEqual(z.load_zones(self.zones), {"yard": [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)]})
        rows = {r["name"]: r["points"] for r in fc.zone_rows(self.cameras, self.zones)["cameras"]}
        self.assertEqual(rows["yard"], [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]])

    def test_a_disabled_camera_can_have_a_zone_too(self) -> None:
        out = fc.set_zone("old_cam", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        self.assertEqual(out["camera"], "old_cam")

    def test_an_unknown_camera_or_bad_corners_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            fc.set_zone("street", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        with self.assertRaises(ValueError):
            fc.set_zone("yard", "0,0;1,0", self.cameras, self.zones, restart=False)
        with self.assertRaises(ValueError):
            fc.set_zone("yard", "0,0;1,0;2,2", self.cameras, self.zones, restart=False)
        self.assertEqual(z.load_zones(self.zones), {})

    def test_clear_zone_means_the_whole_picture(self) -> None:
        fc.set_zone("yard", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        out = fc.clear_zone_command("yard", self.cameras, self.zones, restart=False)
        self.assertEqual(out, {"camera": "yard", "points": []})
        self.assertEqual(z.load_zones(self.zones), {})
        with self.assertRaises(ValueError):
            fc.clear_zone_command("street", self.cameras, self.zones, restart=False)

    def test_renaming_a_camera_carries_its_zone(self) -> None:
        fc.set_zone("yard", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        fc.apply_changes({"cameras": [{"name": "yard", "new_name": "garden", "enabled": True}]},
                         self.cameras, zones_path=self.zones, restart=False)
        self.assertEqual(set(z.load_zones(self.zones)), {"garden"})

    def test_disabling_a_camera_keeps_its_zone(self) -> None:
        fc.set_zone("yard", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        fc.apply_changes({"cameras": [{"name": "yard", "enabled": False}]}, self.cameras, zones_path=self.zones,
                         restart=False)
        self.assertEqual(set(z.load_zones(self.zones)), {"yard"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests/box -p test_find_zones.py`
Expected: errors `AttributeError: module ... has no attribute 'zone_rows'` (and `set_zone`, `clear_zone_command`); the rename and disable tests fail with `TypeError: apply_changes() got an unexpected keyword argument 'zones_path'`.

- [ ] **Step 3: Implement**

In `home_guard_project/box/find_cameras.py`:

1. After line 83 (`CAMERAS_PATH = ...`) add:

```python
ZONES_PATH = os.path.join(os.path.dirname(CAMERAS_PATH), "zones.yaml")
```

2. Change the `apply_changes` signature and add the rename hand-off. Signature:

```python
def apply_changes(changes: Dict[str, Any], path: str = CAMERAS_PATH, zones_path: str = ZONES_PATH,
                  restart: bool = True) -> Dict[str, Any]:
```

(`restart=False` lets tests run it without touching the box's restart flag.)

Inside the `for ch in changes.get("cameras", []):` loop, after `(new_active if enabled else new_disabled)[new] = known[old]`, add:

```python
        if new != old:
            renames[old] = new
```

and declare `renames: Dict[str, str] = {}` next to `handled: set = set()`. Then replace the tail of the function, from `_write_cameras(new_active, new_disabled, path)` to the `return`, with:

```python
    _write_cameras(new_active, new_disabled, path)
    from ..data_collection.zones import rename_zone  # noqa: PLC0415

    for old, new in renames.items():
        rename_zone(old, new, zones_path)      # the watch zone follows its camera
    if restart:
        _restart_running_mode()
    return {"active": sorted(new_active), "disabled": sorted(new_disabled)}
```

(`_restart_running_mode` is defined in the next step; it is the same try/except that used to sit here.)

3. Add the three handlers after `apply_changes`:

```python
def _restart_running_mode() -> None:
    try:
        from . import control  # noqa: PLC0415

        control.request_restart()
    except Exception:  # noqa: BLE001
        pass


def _known_camera(name: str, cameras_path: str) -> str:
    raw = _read_cameras_raw(cameras_path)
    known = {**dict(raw.get("disabled") or {}), **dict(raw.get("cameras") or {})}
    if name not in known:
        raise ValueError(f"unknown camera: {name!r}")
    return name


def zone_rows(cameras_path: str = CAMERAS_PATH, zones_path: str = ZONES_PATH) -> Dict[str, Any]:
    """Every camera with its watch zone: ``{"cameras": [{"name", "points"}]}``; ``points`` is [] for the whole picture."""
    from ..data_collection.zones import load_zones  # noqa: PLC0415

    raw = _read_cameras_raw(cameras_path)
    zones = load_zones(zones_path)
    names = list(dict(raw.get("cameras") or {})) + list(dict(raw.get("disabled") or {}))
    return {"cameras": [{"name": n, "points": [[x, y] for x, y in zones.get(n, [])]} for n in names]}


def set_zone(camera: str, points_text: str, cameras_path: str = CAMERAS_PATH,
             zones_path: str = ZONES_PATH, restart: bool = True) -> Dict[str, Any]:
    """Store a camera's watch zone from ``x,y;x,y;...`` and ask the running mode to restart."""
    from ..data_collection.zones import parse_points, save_zone  # noqa: PLC0415

    name = _known_camera(str(camera), cameras_path)
    corners = save_zone(name, parse_points(points_text), zones_path)
    if restart:
        _restart_running_mode()
    return {"camera": name, "points": [[x, y] for x, y in corners]}


def clear_zone_command(camera: str, cameras_path: str = CAMERAS_PATH,
                       zones_path: str = ZONES_PATH, restart: bool = True) -> Dict[str, Any]:
    """Watch the whole picture again for this camera."""
    from ..data_collection.zones import clear_zone  # noqa: PLC0415

    name = _known_camera(str(camera), cameras_path)
    clear_zone(name, zones_path)
    if restart:
        _restart_running_mode()
    return {"camera": name, "points": []}
```

4. In `main()`, after the `applyp` parser definition add:

```python
    sub.add_parser("zones", help="List every camera with its watch zone (empty = whole picture).")
    setz = sub.add_parser("set-zone", help="Store a camera's watch zone; everything outside it is blacked out.")
    setz.add_argument("--camera", required=True)
    setz.add_argument("--points", required=True,
                      help="Corners as x,y;x,y;x,y in 0-1 picture fractions, 3 to 32 of them, no spaces or quotes.")
    clrz = sub.add_parser("clear-zone", help="Watch the whole picture again for a camera.")
    clrz.add_argument("--camera", required=True)
```

and, before the final `if args.command == "apply":` block, add:

```python
    if args.command in ("zones", "set-zone", "clear-zone"):
        try:
            if args.command == "zones":
                result = zone_rows()
            elif args.command == "set-zone":
                result = set_zone(args.camera, args.points)
            else:
                result = clear_zone_command(args.camera)
        except ValueError as exc:
            if args.json:
                print(json.dumps({"error": str(exc)}))
            else:
                print(f"Error: {exc}")
            sys.exit(1)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for row in result.get("cameras", [result]):
                corners = row.get("points") or []
                print(f"  {row['camera' if 'camera' in row else 'name']:<22} "
                      f"{'whole picture' if not corners else str(len(corners)) + ' corners'}")
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests/box 2>&1 | tail -3`
Expected: `OK` (446).

- [ ] **Step 5: Try the command for real on this laptop (temporary files, no restart side effect)**

```bash
.venv/Scripts/python.exe - <<'EOF'
import os, tempfile
from home_guard_project.box import find_cameras as fc
d = tempfile.mkdtemp(); c = os.path.join(d, "cameras.yaml"); zpath = os.path.join(d, "zones.yaml")
fc._write_cameras({"yard": "rtsp://x"}, {}, c)
print(fc.set_zone("yard", "0.1,0.2;0.9,0.2;0.5,0.9", c, zpath, restart=False))
print(open(zpath, encoding="utf-8").read())
EOF
```
Expected: the dict with three corners, then the YAML file with the header and `zones: {yard: [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]}`.

- [ ] **Step 6: Document the command**

In `home_guard_project/box/README.md`, after the "Customer setup (from the laptop)" section, add:

```markdown
### Watch zones: look only at part of a camera's picture

A camera can be given a zone (for example the yard, not the street). Everything outside it is
blacked out as the frame is read, so no detector, AI call, clip, snapshot or preview ever contains
it. The zone is a polygon in picture fractions (0–1), 3 to 32 corners, kept in
`home_guard_project/data_collection/zones.yaml` on the box (not in git). It applies in both modes.

Normally the owner draws it in the setup app (camera page → "Set the area to watch"). By hand, on
the box:

```
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json zones
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json set-zone --camera yard --points 0.1,0.2;0.9,0.2;0.9,0.9;0.1,0.9
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json clear-zone --camera yard
```

`set-zone` / `clear-zone` ask the running mode to restart so the change is live within seconds.
Renaming a camera carries its zone along; disabling keeps it.
```

- [ ] **Step 7: Commit**

```bash
git add -f home_guard_project/box/README.md
git commit -m "Box: zones / set-zone / clear-zone camera commands; a rename carries the zone" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- home_guard_project/box/find_cameras.py tests/box/test_find_zones.py home_guard_project/box/README.md
```

---

### Task 5: Push and hand the dialog to Codex

**Files:**
- Copy: `docs/superpowers/plans/2026-10-03-codex-brief-watch-zone-dialog.md` to `C:\Users\ameer\Ameer\home_guard_ui\CODEX_BRIEF_ZONES.md` (the UI worktree, branch `box-app-ui`; Codex edits `home_guard_project/box/app/` there — we do not).

**Interfaces:**
- Consumes: the CLI contract from Task 4 (`zones`, `set-zone --camera --points`, `clear-zone --camera`; JSON shapes above).

- [ ] **Step 1: Push the engine work**

```bash
git fetch origin beelink-collector-box && git push origin beelink-collector-box
```
Expected: fast-forward push. If it is refused, another session pushed first: `git pull --rebase origin beelink-collector-box` is NOT safe in the shared tree; instead tell the user and the other session and let them push, or retry after they confirm the tree is quiet.

- [ ] **Step 2: Put the brief in the UI worktree**

The brief is already written and committed: `docs/superpowers/plans/2026-10-03-codex-brief-watch-zone-dialog.md` (premium-quality requirements: split layout, nested enclosures, ambient fill, live dimming with eased motion, every state built, tile preview, tests, screenshots and a look pass). Copy it into the Codex worktree under the name its rounds use:

```bash
cp docs/superpowers/plans/2026-10-03-codex-brief-watch-zone-dialog.md /c/Users/ameer/Ameer/home_guard_ui/CODEX_BRIEF_ZONES.md
cd /c/Users/ameer/Ameer/home_guard_ui && git status --short | head -3 && git log --oneline -1
```
Expected: the worktree is clean (nothing but the new untracked brief) and at `f78b1dc` or later on `box-app-ui`.

- [ ] **Step 3: Launch Codex with the gpt-6-astra model, unattended, in the background**

One round at a time in that worktree (agreed with session home-guard-fa). Start a NEW Codex session (never `codex resume`):

```bash
cd /c/Users/ameer/Ameer/home_guard_ui && env -u VIRTUAL_ENV codex exec -C /c/Users/ameer/Ameer/home_guard_ui -m gpt-6-astra --color never -o /c/Users/ameer/Ameer/home_guard_ui/codex_zones_last.md - < CODEX_BRIEF_ZONES.md > codex_zones_run.log 2>&1
```
Run it with `run_in_background: true` (it takes about an hour). Do not start a second one. When it finishes, read `codex_zones_last.md` (its final message) and the tail of `codex_zones_run.log`, then in the worktree check `git log --oneline f78b1dc..HEAD` for its three commits and run `.venv/Scripts/python.exe -m unittest discover -s tests/box` there. Open its screenshots under `docs/ui/screenshots/` and look at them yourself; if something is clearly off, start ONE follow-up round with a short brief naming exactly what to fix.

- [ ] **Step 4: Tell the user and the other sessions**

Report: engine commits (hashes), test count, Codex's commits and screenshot paths. Message `home-guard-fa` that the collector/inference masking is pushed, that `find_cameras.py` gained zone commands, and that the Codex round on `box-app-ui` is done so it can merge it into `beelink-collector-box`, run the full suite and update the box.

---

## Self-review notes

- Spec §1 (data) → Task 1 (`validate_points`, 3–32, 4 decimals, rename carries zone in Task 4).
- Spec §2 (module) → Task 1. Spec §3 (readers) → Tasks 2 and 3; centre filter removed in Task 2. Deviation, documented in Task 2 step 4: the clip meta keeps its `roi` block because `.meta.json` is a data contract parsed downstream; its `active` flag now means "masked".
- Spec §4 (command) → Task 4. Deviation: `clear-zone` returns `"points": []` rather than `null`, so the wizard's existing JSON parser (which needs a list) can use it.
- Spec §5 (dialog) → Task 5 brief. Spec §6 (failures) → Task 1 (damaged file → `{}`), Task 4 (`{"error"}` + exit 1). Spec §7 (tests) → each task; readers' capture threads are not unit-tested (documented in Task 2). Spec §8 (ownership) → constraints + Task 3/5 messages.
- Names used across tasks: `ZoneMask`, `mask_for`, `parse_points`, `save_zone`, `clear_zone`, `rename_zone`, `load_zones`, `zone_rows`, `set_zone`, `clear_zone_command`, `_Stream._ingest` — consistent throughout.
