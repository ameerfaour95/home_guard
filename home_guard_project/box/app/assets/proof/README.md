# Round 14 recorded UI proof

All captures use local demonstration pictures and the documented ai_status.py JSON contract, replayed at 1 Hz. The JSON fixture is a protocol example, not a capture from a customer's box. No box, camera, or Telegram connection was made. Existing integration tests use mocks or local loopback fixtures.

| View | 1920 x 1080 | 1366 x 768 |
| --- | --- | --- |
| Setup, details closed | [PNG](1920x1080-wizard-closed.png) | [PNG](1366x768-wizard-closed.png) |
| Setup, details open | [PNG](1920x1080-wizard-open.png) | [PNG](1366x768-wizard-open.png) |
| Detections on | [PNG](1920x1080-detections.png) | [PNG](1366x768-detections.png) |
| Off camera on stage | [PNG](1920x1080-off-stage.png) | [PNG](1366x768-off-stage.png) |
| Off camera in Cameras | [PNG](1920x1080-off-cameras.png) | [PNG](1366x768-off-cameras.png) |
| Settings | [PNG](1920x1080-settings.png) | [PNG](1366x768-settings.png) |
| Page change at 120 ms | [PNG](1920x1080-transition.png) | [PNG](1366x768-transition.png) |

The `*-press-{0,60,120}.png` and `*-page-{0,60,120,180,239}.png` files are frame sequences sampled from the actual Qt animations. Hover/press: 120 ms; toggles/chips: 180 ms; pages/panes: 240 ms; OutCubic easing; slide distance: 12 px. Toast dismissal starts at 3 seconds. Layout tests toggle setup details twice at 1920, 1366, and 1000 px and compare exact row/indicator geometry.

## Run

```powershell
$env:VIRTUAL_ENV = $null
.venv/Scripts/python.exe -m home_guard_project.box.app --demo --state live-detections --detections
.venv/Scripts/python.exe -m home_guard_project.box.app --setup --demo --page progress
.venv/Scripts/python.exe -m home_guard_project.box.app --demo --state off-camera --panel cameras
.venv/Scripts/python.exe -m home_guard_project.box.app.proof
```

Full suite: 536 tests passed. In this workstation the inherited antivirus SSLKEYLOGFILE points at a device unsupported by Python's OpenSSL build; it was unset for the test process. No system setting was changed.

```powershell
$env:VIRTUAL_ENV = $null
$env:SSLKEYLOGFILE = $null
.venv/Scripts/python.exe -m unittest discover -s tests/box
```

Regression coverage includes the documented status payload; detection ts versus checked_ts; wall-clock freshness, fade, and picture-relative geometry; immediate eye-toggle repaint on hero and thumbnails; stopped hint; disabled-camera retention, optimistic enable and failure rollback; preserving unsaved renames; reserved empty camera action slots; one-second replay cadence; page, switch, press, and busy feedback.
