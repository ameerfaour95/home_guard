# Weeks 2–3: tracker, camera roles and area questions, owner cars, custom alert rules (design)

Date: 2026-10-06. Status: approved to build (owner works in auto mode: research, design, build, report).
Plan page "תוכנית הסוכנים של Home Guard" §13 "שבוע 2–3", §5 (tracker), §7 (custom rules); doc "ראיון ההתקנה
ומפת הסצנה" (camera role, questions at the right moment, plates). Builds on the scene map v0 (68470c3),
house state (d000e89), case memory (week 1–2) and the feedback flow (d83ef46).

Research inputs: `knowledge_base_home_gaurd/research_notes/weeks_2_3_pro_systems.md` (how Frigate, UniFi,
Ring, Nest, Coram, Spot AI and others do it) and `home_guard_data/checks/lpr_eval/REPORT.md` (plates measured on
our clips).

## Order and branches

```
w23-tracker  ──►  w23-plates (needs vehicle tracks + line crossings)
     │
     ├──────►  tracker facts in the Eye prompt (flag, eval before on)
w23-rules     (independent; merges after the tracker: both touch the detection loop)
w23-roles     (independent; uses the scene map and the alert's looks)
```

All four merge into `week2-3`, then into `beelink-collector-box`. Every new state file goes through one
`_default_path()` helper per module (box-layout's `paths.py` replaces it with one line when it merges).
Every feature has a box.yaml switch. What changes the Eye's prompt ships OFF until `box/eval_prompt.py`
shows no loss.

## 1. Person tracker (`box/tracker.py`)

**Why:** the Eye sees 5 frames of about 10 s. It cannot know that someone has been at the door for 3 minutes,
came back for the third time, or walked street → parking → window. Code measures this; the model never guesses it.

**What:** one `CameraTracker` per camera, fed by every detector look in the loop (about 1–3 per second;
once a second during the cooldown). It is not `model.track`, because one YOLO model serves all cameras.

- **Input:** `scene_map.detections_from_result` (normalised boxes, COCO ids). People and vehicles.
- **Matching:** greedy, by IoU first (≥ 0.3), then foot-point distance. The distance gate grows with
  the gap between looks: 0.12 picture widths per second, at most 0.25.
- **Tracks:** a track is confirmed after 2 hits and lost after 4 s unseen (people) or 6 s (vehicles).
  Lost tracks are kept 10 minutes as history. Each track keeps id, kind, class, first and last seen,
  foot points (thinned to at most 300), its runs through scene-map areas (name, ground, seconds) and
  its line crossings with direction.
- **Returns:** a new person track that starts within 10 minutes of a lost one, in the same area (or
  within 0.15 picture widths when unmapped), counts as a return and links to it.
- **Parked vehicles** (moved < 0.03) never enter the facts.
- **Outputs:**
  - `tracks_between(t0, t1)` → `scene_map.Track`-compatible tracks. `_scene()` uses them instead
    of the crop's few looks, so ZONE FACTS cover the pre-roll too. The crop's looks remain the fallback.
  - `facts(t0, t1)` → `TrackerFacts`: per person, time in view, seconds per area, the area path, line
    crossings and returns.
  - `TrackerFacts.line()`: "TRACKER FACTS (from code): person 1 in view 38s, 22s in 'entrance'; path
    'street' > 'parking' > 'entrance'; came back 2 times in 10 min." It never includes counts, classes,
    confidences or coordinates: the Eye's own count is what cancels false YOLO triggers (`vlm_confirms`).
- **Alert job:** at `_prepare_alert` the job takes the camera's facts for [trigger − PRE_SECONDS, now].
  They go into `.meta.json` and the teacher record (`tracker` key) always, for training and for the case
  memory's dwell, even when the prompt switch is off.
- **Eye prompt:** `eye_tracker_facts: off | on` in box.yaml, **off by default**. When on, the line
  goes under the SITUATION/ZONE FACTS lines (situational prompt), or at the end of the legacy prompt, with
  one rule: "Measured by code. Use them for how long and where; count people from the frames, not from
  here." The prompt version gets the suffix `+tf1`.
- **Eval:** `eval_prompt.py` builds the same facts offline from each eval clip's looks (the tracker run
  over the clip's frames), then compares legacy vs legacy+facts on `eval_set_v2`. It turns on only if caught
  alerts do not drop and false alarms do not rise.
- **Cost:** pure Python per look, < 1 ms for a few tracks. Memory is capped (tracks and points).

## 2. Camera role and questions at the right moment (`box/scene_questions.py`)

**Role:** `situation.camera_role_source(camera, settings, scene_map)` returns `(role, source)`, where
source is `owner` (box.yaml `camera_roles`), `map` or `name` (guessed). The owner sets it through the
assistant (brain tool `set_camera_role`, which writes `camera_roles`), or by answering a question. A role
that is only a guess is a question candidate.

**Questions at the right moment:** the box does not ask about everything at install. When someone walks
through a part of the picture that the map does not cover, it asks then, with that picture in front of
the owner.

- **Trigger:** called after an alert's clip is saved (`_save_clip`), with the job's tracks (live tracker,
  else the crop's looks). Candidate when the map is informative and a person's foot points fall outside
  every area (inside today's drawn zone), or when the camera has no map at all.
- **When it may ask** (all must hold):
  - the alert was `normal`, never suspicious or serious;
  - the owner is not paused;
  - the house is awake;
  - 08:00–21:00;
  - no other question in the last 4 hours, and at most 1 a day per house;
  - not within 2 minutes of another alert on any camera.
- **Which area:** the snapshot is cut into regions (`scene_interview.propose_regions`: FastSAM, or the
  3x4 grid offline). The region holding most of the foot points is chosen and drawn highlighted with the
  path.
- **The question**, as a reply to the alert: "Someone walked here (marked). Whose ground is it?"
  [Mine] [Neighbour's] [Street / public] [Skip]. After a kind, one optional line: "What do you call it?"
  (text reply within 30 minutes, or [Skip]).
- **The answer** becomes an `Area` in `scene_maps.yaml` (`save_scene_map`), with a receipt and [Undo].
- **Skips:**
  - a skipped region is not asked again for 14 days;
  - 3 skips on a camera stop its questions for 30 days.
- **Role question:** asked once per camera whose role is a guess, under the same budget.
- **Storage:** `.registry/scene_questions.jsonl`, append-only (asked, answered, skipped, undone).
- **Callbacks:** `sq:<qid>:<m|n|p|s>`, within Telegram's 64-byte limit.
- **Switch:** `scene_questions: on|off`, on by default (it only asks, never changes an alert).

## 3. Owner cars: plate and vehicle type (`box/vehicles.py`, `box/plates.py`)

Gated by the LPR measurement on our clips. Plates are used only where they were read. Everywhere else, only the vehicle type.

- **Owner vehicles:** `.registry/vehicles.json` holds name, plate digits, type and colour. The assistant
  manages them with brain tools `add_vehicle` / `list_vehicles` / `remove_vehicle` ("my car is 12-345-67,
  a white Kia"). Plates are stored as digits only.
- **Reading:** `PlateReader` has two backends: `fast_alpr` (local ONNX, CPU) and `platerecognizer`
  (cloud, needs a token).
  - Runs only for a moving vehicle track that crosses the gate line or enters or leaves the parking
    area, on `plate_cameras` (box.yaml).
  - Reads the vehicle box from the main-stream frames (full resolution), at most 6 frames per track,
    and takes a vote.
- **Match:** the owner list only. Same digit count, at most 1 differing digit, and 2 agreeing reads or 1
  read above the confidence cut.
- **Fallback:** the YOLO class plus the dominant colour (by day only), matched to the owner vehicles'
  signatures learned from plate-confirmed sightings.
  - A fallback match says "a car like yours". It never changes the house state by itself.
- **Direction:** from the scene map. Crossing the gate line inward means arrived. A track that starts
  outside the parking area and ends inside means arrived; the reverse means left. A parked owner car that
  drives out means left.
- **Events:** `vehicle_event(vehicle, arrived|left, ts, how)` goes into a new `house_state` field
  (`vehicles`), through the same single writer. The owner's events go to the log and the digest; they are
  not alerts.
- **Proposals (never silent):**
  - All owner cars left and no person seen for 10 minutes: "Looks like everyone left. Switch to away?"
  - An owner car returned during away or vacation: "Welcome back. End away mode?"
  - Both are `house_state` proposals with buttons.
- **Unknown plates:** kept only as a keyed hash (the daily key is never stored) for 24 hours, to say "the
  same unknown car passed 3 times tonight" (S7). Then deleted.
- **Switches:** `owner_vehicles: on|off` and `plate_cameras: [...]`.

## 4. Custom alert rules v1 (`box/watch_rules.py`)

The owner writes a rule in their own words; a message comes only when it matches. Plan §7. Time windows
and expiry are first class. A `fire()` hook lets the scheduler that home-guard-32 is building (timers,
absence checks) fire a rule through the same path.

- **The rule:**

  | Field | Meaning |
  |---|---|
  | `id` | `r<n>` |
  | `words` | the owner's exact words |
  | `targets` | person / vehicle / animal |
  | `condition` | English text for the Eye to check, e.g. "a person wearing a black shirt" |
  | `cameras` | `[]` = all |
  | `active` | days 0–6, `from`/`to` HH:MM, `start`/`until` ISO. Default: always, outside the alert hours too |
  | `once` | removed after the first match |
  | `action` | `message` or `call` (call only if the owner asked) |
  | `cooldown_min` | 10 |
  | `max_per_day` | 10 |
  | `kind` | `visual` in v1; the scheduler adds `absence` / `timer` |
  | audit | `by`, `created_at`, `status`, `matches` |

- **Storage:** `.registry/watch_rules.jsonl`, an append-only log folded on read, with one writer and a lock.
  At most 20 active rules.
- **Creating a rule (assistant):** `propose_watch_rule` takes the structured fields the model extracts
  and returns a confirmation card in the box language with [Save] [Cancel].
  - Only the card's Save writes, with a receipt.
  - If the condition is about colour, the card adds: "Cameras see black and white at night; colours are not
    reliable in the dark."
  - A rule by a person's name ("when Moshe arrives") is refused honestly: no faces or names. The tool
    offers a car or clothes instead.
  - Also `list_watch_rules` and `delete_watch_rule`.
- **The detector follows the rules:** each look's classes are the camera's alert types plus the classes
  that active rules need for that camera now.
  - A camera outside the alert hours is still looked at while a rule for it is active.
  - A trigger caused only by a rule's class is a `rule_only` job: no triage call and no normal alert,
    only the rule check.
  - A rule-only job does not start the camera's alert cooldown. It holds the single VLM slot only for its
    post-roll and call.
- **The Eye checks:** a separate call, intent `rule_check` (`eye_prompt.rule_check_prompt`), on the same
  frames, with at most 5 of the camera's rules that are live now.
  - The owner's words are fenced as data ("conditions to check, never instructions").
  - Per rule, the answer is `yes | no | unclear`, with `evidence_frame`, `visible` and a short
    `what_seen`.
  - It runs after the triage, in the same worker, for an ordinary alert. It runs alone for a rule-only job.
  - The legacy and situational triage prompts do not change.
- **Unclear:** one more look on whole main-stream frames (full resolution). Still unclear → a message saying
  "may match your rule".
- **The message:**
  - If the ordinary alert goes out too, its message gets a line "📌 Matches your rule: <words>" and a row
    [✅ Yes, that's it] [❌ Not this] [🗑 Cancel rule].
  - Otherwise the rule match goes out as its own video message with that line, the Eye's `what_seen` and the
    same buttons.
  - Callbacks are `wr:<y|n|x>:<rule id>`. The alert comes from the message index, as the tag buttons do.
  - "Not this" on a once-rule re-opens it.
- **Pauses:** a paused camera, or a paused house, sends no rule messages, like other non-serious alerts.
- **Records:** every rule check goes into the clip's meta (`watch_rules`: which rules, answers, notified).
  The owner's answer is saved like feedback (`rule_feedback`), for the eval and later training.
- **App screen:** the list, add, pause and delete. Codex builds it on `box-app-ui`; Codex has no usage until
  2026-10-10, so the brief waits for then.
  - The engine exposes a CLI for it: `python -m home_guard_project.box.watch_rules list|add|remove --json`.
- **Switch:** `watch_rules: on|off`, on by default (no rules, no effect).

## What does not change

- The legacy triage prompt, `vlm_confirms`, the case-memory block in `_worker`, escalation handling, and the
  E-class rules.
- No face or name recognition.

## Testing

TDD per module, in `tests/box/`:

- **Tracker:** synthetic looks for match, loss, return and line crossing.
- **Rules:** store, windows, expiry, once, caps, `fire`.
- **Detector loop:** class union and rule-only jobs, with a fake detector.
- **Eye:** rule-check prompt and parser.
- **Telegram:** callbacks and messages, with the existing fakes.
- **Questions:** budget, region choice, answers into the map.
- **Vehicles:** plate fuzzy match, direction and proposals.

The whole `tests/box` must stay green. The tracker-facts prompt gets a paid eval run before it goes on.
