# Round Z: "Set the area to watch" — the zone drawing dialog, built to a premium bar

The owner wants to draw, with the mouse, the part of each camera's picture the box watches
(the yard, not the street). Everything outside is blacked out by the engine as the frame is
read. The engine is done; this round is the owner-facing part, and the owner's verdict on it
will be the same one he gave the window: it must *feel* premium without him being able to say
why. Build it so a demanding person opens it and relaxes.

Same boundaries as round 13: merge `origin/beelink-collector-box` first (it brings the zone
engine: `home_guard_project/data_collection/zones.py` and the `find_cameras` commands below),
`uv sync` with `VIRTUAL_ENV` unset; this checkout only; branch `box-app-ui`; no push; no box,
camera or Telegram; engine `.ps1` and box runtime modules untouched (edit only
`home_guard_project/box/app/`, `tests/box/test_app_*.py`, `docs/ui/`). PySide6 only, no new
dependencies. Commit after each part, one-line messages in the branch's style. About 60 minutes.

## What exists (read these first)

- `home_guard_project/box/app/camera_ui.py` — the camera page (`CameraPage.render`): one tile per
  camera with its snapshot (`AlertPicture` in `ai_activity_ui.py`), a name field and an
  "enabled" checkbox. Shared by the laptop wizard (`wizard=True`, remote box over SSH) and the
  box window (local).
- `home_guard_project/box/app/camera_controls.py` (`CameraControls`, local) and
  `remote_cameras.py` (`RemoteCameraControls`, over `ssh.exe`). Both run
  `python -m home_guard_project.box.find_cameras --json <args>`; the remote one builds the
  command in `ssh()` and **refuses quotes**, and parses the reply with `parse(result, key)`,
  which needs a list under `key`.
- `home_guard_project/box/app/theme.py` — the only colours allowed: `PALETTES['dark']`
  (`bg #0c1218`, `surface #121c24`, `raised #192731`, `border #273743`, `text #edf4f6`,
  `secondary #bec8ce`, `muted #a0adb8`, `action #42d6c3`, `ok #7edcb0`, `error #f17e86`,
  `warning #e2ba76`) and the matching `light` set. `stylesheet()` carries the control styles;
  `ui.py` has `card()`, `label(text, role)`, `layout_for()`.
- `home_guard_project/box/app/strings.py` — every user-facing string goes through `tr(key)` in
  every language the file carries. No literal strings in widgets.
- `docs/ui/capture.py` and `docs/ui/SCREENSHOTS.md` — offscreen screenshot tooling; `--demo`
  mode has synthetic camera pictures.

## The engine contract (already on `beelink-collector-box`)

All under `find_cameras --json`:

| Command | Output |
|---|---|
| `zones` | `{"cameras": [{"name": "yard", "points": [[0.1, 0.2], ...]}, ...]}` — every camera, `points` is `[]` for "whole picture" |
| `set-zone --camera yard --points 0.1,0.2;0.9,0.2;0.5,0.9` | `{"camera": "yard", "points": [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]}` |
| `clear-zone --camera yard` | `{"camera": "yard", "points": []}` |
| any error | `{"error": "<plain message>"}`, exit code 1 |

Corners are picture fractions 0–1, 3 to 32 of them, 4 decimals; on the command line
`x,y;x,y;x,y` with **no spaces and no quotes**: `";".join(f"{x:.4f},{y:.4f}" for x, y in pts)`.
`set-zone` / `clear-zone` already restart the running mode on the box. For the remote parser use
`key='cameras'` for `zones` and `key='points'` for set/clear.

## Part 1 — controls (commit 1)

Add to both `CameraControls` and `RemoteCameraControls`, same names, same shapes:

```python
def zones(self):                    # -> {name: [[x, y], ...]}   ([] = whole picture)
def set_zone(self, name, points):   # points: list of (x, y) fractions -> stored [[x, y], ...]
def clear_zone(self, name):         # -> []
```

Demo mode (`box.demo`): keep zones in a dict on the controls object; nothing is run.

Tests (`tests/box/test_app_cameras.py` style): the remote `set_zone` builds an ssh command that
contains `set-zone --camera yard --points 0.1000,0.2000;0.9000,0.2000;0.5000,0.9000` and no
quote characters; `zones()` maps the JSON to the dict; an `{"error": ...}` reply raises.

## Part 2 — the dialog (commit 2)

`home_guard_project/box/app/zone_editor.py`, class `ZoneEditorDialog(QDialog)`, opened from a
new pill button under each camera photo: **"Set the area to watch"**.

### Layout: split, not centred

- The dialog opens at 1280×800 (clamped to the screen, minimum 1100×700) with a fade-in and an
  8 px rise over 260 ms, `QEasingCurve.OutCubic`; it closes with the reverse over 180 ms. No
  element ever appears or vanishes instantly.
- Left ~72 %: the picture stage. Right ~28 %: a column with, top to bottom: an eyebrow tag (small
  caps, letter-spaced, muted: "CAMERA · FRONT DOOR"), the title ("Draw the area to watch",
  weight and colour carry the hierarchy, not size), two lines of help, a status block, the
  warning slot, and at the bottom the buttons. In Hebrew/Arabic the whole dialog mirrors
  (Qt layout direction follows the app's language).
- The stage is a nested enclosure: an outer shell (`raised` fill, hairline border at 8 % white,
  radius 22 px, 6 px padding) holding the inner core (the picture, radius 16 px, a 1 px inner
  highlight at 10 % white). Shadows, if any, are tinted to the navy background, never grey or
  black. No flat 1 px grey borders anywhere.
- The picture is the camera's **raw** snapshot (the one the tile already shows; it is not
  masked, so the street stays visible to draw around). It is fitted inside the core by height
  or width, centred, and the rest of the core is filled with the same picture blurred
  (~40 px) and darkened to ~35 % — the "ambient" fill from round 13. Letterboxing with flat
  colour is forbidden.

### Drawing

- Left-click adds a corner. Hover a corner: it grows from 12 to 16 px with a soft 2 px ring in
  `action`; drag moves it with the cursor changed to a hand. Right-click, Backspace or the
  **Undo** button removes the last corner. **Clear** removes all. Clicking within 12 px of the
  first corner when there are 3+ corners snaps to it (a small pulse on the first corner while
  the pointer is near it).
- Corners: 12 px circles, `bg` fill, 2 px `action` stroke. Edges: 2 px `action`, anti-aliased.
  While the pointer moves, a dashed rubber-band edge follows it from the last corner.
- With 3+ corners the outside darkens **live**: an overlay of the palette `bg` colour at 65 %
  (never `#000000`), the polygon cut out, with the polygon edge drawn on top. The overlay fades
  in over 220 ms the moment the third corner lands, and fades out when Undo drops below three.
- Corners are stored as fractions of the picture's own pixel size, 4 decimals, independent of
  how the dialog is sized. Resizing the dialog re-fits the picture and the corners follow.
- Keyboard: Enter = Save, Escape = Cancel, Ctrl+Z = Undo. All buttons have accessible names.

### States (every one must be built and visible)

| State | What the right column shows |
|---|---|
| Empty (0 corners) | Status "Click the first corner of the area to watch." Save is enabled and means "watch the whole picture" (it clears the zone). |
| Drawing (1–2 corners) | Status "Add at least 3 corners." Save disabled. |
| Closed (3+ corners) | Status "Everything outside the shape is ignored." Save enabled. |
| Tiny (closed, under 5 % of the picture) | A warning chip in `warning` colour: "This area is very small; the camera will see almost nothing." Save still enabled. |
| Saving | Save shows "Saving…", all controls disabled, the picture keeps its state; a thin indeterminate bar under the button, no spinner. |
| Error | Inline line in `error` colour under the buttons ("The area could not be saved. Try again."); dialog stays open, controls re-enabled. |

### Buttons

Two pills at the bottom of the column: **Save** (primary, `action` fill, dark text, with the
trailing check icon nested in its own small circle inside the pill) and **Cancel** (ghost,
hairline border). **Undo** and **Clear** are smaller ghost pills above them. Pressed state:
scale to 0.98 (animate the geometry 1 px in and back, 90 ms OutCubic). Hover: fill lightens
by one step, 160 ms. No neon glows, no gradients on text, no purple.

### Save

- 3+ corners → `controls.set_zone(name, points)`; fewer → `controls.clear_zone(name)`.
- On success the dialog closes (reverse animation) and the tile updates (Part 3). On an exception
  the Error state shows.

## Part 3 — the tile (commit 3)

Under each camera photo on the camera page:

- The pill button "Set the area to watch" and, beside it, a status line: "Watching: the whole
  picture" or "Watching: the area you drew". It is loaded once with `controls.zones()` after
  the snapshots arrive, and updated from the dialog's reply with a 300 ms crossfade.
- When a zone exists, the thumbnail itself shows it: the saved polygon as a 1.5 px `action`
  outline over the picture and the outside dimmed to 35 % — the owner sees at a glance what is
  watched. Tiles without a zone are unchanged.

## Strings (add to `strings.py` in every language the file carries)

`camera_zone_button` "Set the area to watch" · `camera_zone_whole` "Watching: the whole picture"
· `camera_zone_drawn` "Watching: the area you drew" · `camera_zone_eyebrow` "Camera · {camera}"
· `camera_zone_title` "Draw the area to watch" · `camera_zone_help` "Click the corners of the
area the camera should watch. Everything outside it is ignored by the box." ·
`camera_zone_status_empty` "Click the first corner of the area to watch." ·
`camera_zone_status_drawing` "Add at least 3 corners." · `camera_zone_status_closed`
"Everything outside the shape is ignored." · `camera_zone_small` "This area is very small; the
camera will see almost nothing." · `camera_zone_undo` "Undo" · `camera_zone_clear` "Clear" ·
`camera_zone_save` "Save" · `camera_zone_saving` "Saving…" · `camera_zone_cancel` "Cancel" ·
`camera_zone_save_failed` "The area could not be saved. Try again."

## Tests (`tests/box/test_app_zones.py`, offscreen Qt, same style as `test_app_cameras.py`)

- A click at the picture's centre becomes (0.5, 0.5) and back, for a 4:3 picture in the
  wider stage (the fit is by height, so the x fraction accounts for the side fill).
- Undo removes the last corner; Clear empties; Backspace undoes; Ctrl+Z undoes.
- Save with 3+ corners calls `set_zone(name, points)` with 4-decimal fractions; with 0–2
  corners it calls `clear_zone(name)`; Save is disabled with 1–2 corners.
- The small-area warning appears for a polygon under 5 % and not for one over it.
- A failed save shows `camera_zone_save_failed` and keeps the dialog open and enabled.
- The tile status flips to "Watching: the area you drew" after a successful save.
- Snapping: the fourth click within 12 px of the first corner adds no corner.

## Screenshots and the look pass

Add to `docs/ui/capture.py` four dialog states (empty, closed, tiny-warning, error) at
1366×768 and 1920×1080, plus the camera page with one zoned and one unzoned tile; list them
in `docs/ui/SCREENSHOTS.md`. Then **open the screenshots and look** at each with the owner's
eye, and fix what you see before the final commit: clipped text, awkward gaps, a corner
handle that looks cheap, a warning that shouts, an overlay that reads as a black hole instead
of a quiet dimming, anything that would make him say "it doesn't look premium".

## Hand back

`python -m unittest discover -s tests/box` green; the three commits on `box-app-ui`; the
screenshot list; and ten lines in `docs/ui/HANDOFF.md` on what the dialog does and what was
not verified without a real box.
