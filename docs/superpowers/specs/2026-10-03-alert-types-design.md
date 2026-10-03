# What alerts the owner: people, vehicles, animals (house default + per camera)

Date: 2026-10-03. Approved by the owner ("approve everything").

## Why

A car driving into the driveway sent a Telegram alert. Not a regression: since the first
inference build a *moving* vehicle alerts (only parked ones are filtered, 077e2b9); the cameras
now face the driveway. The owner wants to choose what alerts them.

## What professional systems do (researched 2026-10-03)

| System | Scope | Types | Notify vs record | Zones x objects |
|---|---|---|---|---|
| Frigate | global default + per-camera override (`review.alerts.labels`) | any label; default person + car | alerts vs detections | zone `objects`, `required_zones` |
| UniFi Protect | per camera | person, vehicle, animal, package | notify on smart detections only | smart zones per type |
| Ring | per device | person, vehicle, package, other | notification / recording / both per type | person + zones |
| Arlo, Nest | per camera | people, vehicles, animals, packages | - | Nest: anywhere or only in zones |
| Reolink | per camera | person, vehicle, animal | push vs record, hourly schedule per type | - |

Common shape: per camera, a short fixed list of types, notification separate from recording.

## Decisions

- Types: **person, vehicle, animal**. Animals = cat, dog, horse, sheep, cow, bear (COCO).
  Birds are left out (they would fire all day). Packages are out (no detector class).
- **House default** = `alert_on` in box.yaml (live option), unset = `person`.
- **Per-camera override** = `camera_alerts.yaml` next to `cameras.yaml` / `zones.yaml`
  (`{alerts: {camera: [person, vehicle]}}`); a camera without an entry uses the house default.
  Re-read while running (mtime poll, like box.yaml). A damaged file or entry = house default,
  never a crash. Renaming cameras carries the entries (swaps included).
- Recording is unchanged: every escalated clip is still kept (alert, or false_positive for training).
  The setting only decides whether the AI is asked and whether Telegram fires.

## Engine behaviour

1. Detector gate (`should_escalate`), per camera with its effective types:
   person -> asks the AI; vehicle -> only if the vehicles moved; animal -> asks the AI.
   A type the camera does not alert on never wakes the AI.
2. The AI's answer gains `"animals": <count>` (schema + prompt line, PROMPT_VERSION bump).
3. `vlm_confirms(parsed, alert_on)`: True if it saw something the camera alerts on
   (people > 0 / vehicle_moving / animals > 0); False -> no message, clip kept as false_positive;
   None (no usable answer) -> trust the detector.
4. `ai_status.json` settings carry `alert_on` (house, list) and `camera_alert_on`
   (`{camera: [types]}`, overrides only).

## Commands (find_cameras, JSON for the app)

- `camera-alerts` -> `{"house": [...], "cameras": [{"name", "alert_on": [...] | null}]}`
- `set-camera-alerts --camera N --on person,vehicle` / `--default`. No restart (live).
- House default: `set-option alert_on=person,vehicle,animal` (existing command).

## App (Codex, branch box-app-ui)

Settings -> "Alert me about": three tiles (People / Vehicles / Animals), house default, at least
one on. Each camera card's action slot: "Alerts: House default (People)" -> House default |
Custom with the same tiles. Dashboard status line shows the house choice and marks cameras with
their own. Live apply with the existing Applying/Applied confirmation.

## Addendum: sensitivity per type (approved 2026-10-03)

Seen live: with one house threshold of 0.7 the detector reported only the car while a man walked
past; with people-only alerts such a look no longer reaches the AI. Frigate (`objects.filters.<label>
.threshold`, per camera) and Reolink (a slider per detection type) set certainty per type. So:
`conf_person` / `conf_vehicle` / `conf_animal` in box.yaml (live; unset = `inference_conf`), per-camera
overrides in the `sensitivity:` section of camera_alerts.yaml (some or all types). The detector runs
at the lowest certainty in use, then each find is held to its own type's value
(`filter_by_thresholds`). Command: `set-camera-sensitivity --camera N --values person=0.5,vehicle=0.8 |
--default`; `camera-alerts` also reports `house_sensitivity` and each camera's `sensitivity`;
ai_status settings carry `sensitivity` and `camera_sensitivity`.

## Out of scope

Objects per zone, hourly schedules per type, packages, notify-vs-record per type.
