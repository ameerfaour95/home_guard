# Situation-aware Eye and the Investigator (design)

Date: 2026-10-05. Status: draft for the owner's review. Builds on
`2026-10-03-assistant-brain-design.md` (Guard/Assistant modes, house registry, facts) and on the
research report `knowledge_base_home_gaurd/reports/Video LLM ומערכת סוכנים.md` (model choice: base
Qwen by API now, fine-tuned Qwen3.5-4B on video later).

## Goal

Two changes that turn "one prompt, one model, one guess" into a system that reasons about the house:

1. **The Eye sees the same way always but judges by the situation.** What a scene *is* (a courier
   leaves a package) does not depend on the hour. What it *means* does: a courier at 14:00 is
   nothing; someone at the door at 02:30 while the family sleeps is not a courier. The vision prompt is
   composed per look from a fixed taxonomy plus the current situation (time of day, house state,
   camera role, why we are looking).
2. **The Investigator.** An agent with memory that opens a case for a non-trivial event, gathers
   evidence (tracker facts, the other cameras, this house's history, the owner's notes and past
   verdicts), can look again, and recommends a verdict with reasons. It is the part a customer
   pays for: fewer, better alerts, told as one story.

## Why

- The baseline (gpt-4o, 220 tagged clips, 2026-10-03): 21/177 normal clips flagged, all
  `suspicious`. The causes were appearance (dark clothes, hoods, some invented) and standing or talking
  read as loitering. One prompt for all hours can't know that loitering needs minutes, not 5 frames,
  or that a delivery at 02:30 is not a delivery.
- 5 cameras each send their own alert for one person walking around the house. The real story
  ("rang, nobody answered, went to the side passage, tried the back door") lives *across* cameras and
  minutes, where no single VLM call can see it.
- An always-on product has to justify its price. That means measurable false alarms per home per week, time to
  alert, and incidents told correctly. A cheaper model alone does not create that value. Context and memory do.

## Principles

- **Observation is universal; judgment is situational.** The Eye's observation fields are
  context-free (trainable, cacheable, reusable by every agent). The label is conditioned on the
  situation, and the raw context-free label is always kept (as `raw_label` today).
- **One taxonomy, many situations.** Category ids N1–N10, S1–S9 and E1–E8 never change per prompt. Only
  the expectations around them change. One fine-tuned model learns it from a situation header, so we
  don't need one model or one hand-written prompt per hour.
- **Code decides what code can measure.** Duration, path, zones, rarity, the hour and house
  state are computed, never guessed by a model.
- **Models recommend, policy acts.** No model places a call. The policy engine (code) maps the verdict
  plus owner rules (hours, mutes, alert types, caps) to an action. Escalation can never be softened by
  context, facts or the Investigator. Nothing E-class is ever silent, in any mode.
- **Pay for intelligence only where it changes the outcome.** Most events never reach an LLM beyond
  the Eye.

## Part 1: The situation

`box/situation.py` (pure code, no model) builds a `Situation` for every look:

| Field | Values | Source |
|---|---|---|
| `phase` | `day` / `evening` / `late_night` (00:00 to first light) / `dawn` | Sunrise and sunset for the house location (a fixed table for Israel is fine), plus the clock. `dark: bool` separately. |
| `house_state` | `home_awake` / `home_asleep` / `away` | The owner tells the assistant ("going to sleep", "we left", "back home"). Default schedule: `home_asleep` 00:00–06:00 unless told otherwise. `away` only when said, or from a vacation date range. Shown in the status line. |
| `intent` | `alert_triage` / `snapshot` / `event_question` / `follow_up` | Who asked: the guard loop, the owner's "what's there now", the owner asking about a saved clip, the Investigator. |
| `camera_role` | `street` / `entrance` / `private` (yard, side passage, roof) / `parking` | Per camera, chosen at setup (default from the camera name; editable). |
| `zones` | named areas inside the watch zone (`entrance`, `window`, `gate`, `fence`, `car`) | Optional, drawn in the existing watch-zone dialog. |
| `house_notes` | live facts for this camera and hour (existing `facts_for`) | Unchanged. |
| `expecting` | short owner-set notes with an expiry ("a package today", "the plumber at 10") | New: one assistant tool, stored with the facts. |

House state is the industry's Home/Away/Night (Ring, SimpliSafe) and costs nothing to compute. Owners
already understand it.

## Part 2: The Eye prompt composer

`box/eye_prompt.py` replaces the single `build_prompt` (and `brain/vision.look_prompt` later) with
modules:

1. **Base** (always): role, honesty ("appears to", never names/age/ethnicity), the taxonomy with
   short definitions, the JSON schema for the intent.
2. **Situation header** (always, structured, identical at training and inference):
   ```
   SITUATION: time 02:14, late_night, dark; house: home_asleep; camera: back_yard (private);
   intent: alert_triage; expecting: none
   ```
3. **Expectations block**, generated from the priors table below, only the lines that matter now.
   For example, at `late_night` + `home_asleep`:
   ```
   Right now the family is asleep. Deliveries, workers and visitors are NOT expected at this hour:
   treat someone at the door or in the yard as unexplained unless you see clear proof (uniform and a
   package left, a key used, the door opened from inside). Testing doors or windows, looking in,
   hiding, or being in a private area are serious now.
   ```
4. **Attention list** per situation: what to look for *now*. At night: flashlights, hands on handles
   and windows, crouching, carrying things out. In the day: the act, not the clothes.
5. **Intent module**: what to return.

### Intents

| Intent | Who | Returns | Label? |
|---|---|---|---|
| `alert_triage` | guard loop | full observation + `raw_label` + contextual `label` + `why` | yes |
| `snapshot` | owner: "what's there now?" | `description`, `quality`, counts, plus `safety_note` that is non-empty only if an S/E sign is visible | no label, but never silent on E |
| `event_question` | owner asks about a saved clip | `answer` to the question, grounded in the frames; "can't tell from the pictures" allowed | no |
| `follow_up` | Investigator | answers to 1–3 targeted yes/no questions, each with `evidence_frame` and `visible: clear/partial/no` | no |

The snapshot answer is short and friendly, with no suspicion hunting. The owner asked "what's there", not "is it
dangerous".

### Taxonomy (fixed ids; full wording in `box/taxonomy.py`)

- **Normal:** N1 passing by · N2 coming home / leaving · N3 delivery or service · N4 visitor at the door ·
  N5 work · N6 household life · N7 vehicle routine · N8 animals · N9 soldier or guard with a slung weapon
  · N10 nothing.
- **Suspicious:** S1 testing access (handles, windows) · S2 looking in · S3 surveying (moving between entry
  points, along the fence) · S4 lingering without purpose (time from the box, never from the frames) ·
  S5 hiding the face *while approaching* · S6 in a private area with no reason · S7 vehicle watching ·
  S8 hiding · S9 camera tampering.
- **Escalation:** E1 forced entry · E2 climbing in · E3 theft (picks up and leaves) · E4 car break-in ·
  E5 violence · E6 weapon in use (in hand, aimed) · E7 fire, smoke, crash · E8 person down.
- Plus `other` with `other_text`. Counting `other` tells us what to add.
- Appearance alone is never a category.

### Priors: what each category means in each situation

`expected` adds nothing. `unusual` means the event is worth a case. `serious` means at least `suspicious`,
and a candidate for a call.

| Category | day, home_awake | late_night / home_asleep | away |
|---|---|---|---|
| N1 passing by (street) | expected | expected | expected |
| N2 coming home / leaving | expected | expected only with a key or the door opened from inside, otherwise unusual | unusual unless `expecting` or a fact covers it |
| N3 delivery | expected | expected until 00:00 (food deliveries), unusual after | expected (package left) |
| N4 visitor at the door | expected | unusual | expected, and watch what they do next (the "knock to check if anyone's home" pattern) |
| N5 work | expected in daylight, or per a fact | unusual | unusual unless a fact or `expecting` covers it |
| N6 household life | expected | expected in private zones | unusual |
| N7 vehicle routine | expected | an unknown vehicle stopping is unusual | unusual |
| N8 animals, N10 nothing | expected | expected | expected |
| N9 soldier/guard, slung | expected | expected walking by, unusual at the door | expected walking by |
| S1, S2, S6, S8, S9 | serious → suspicious | serious → escalation candidate | serious → escalation candidate |
| S3, S4, S5, S7 | suspicious | serious → escalation candidate | serious → escalation candidate |
| E1–E8 | escalation | escalation | escalation |

"Escalation candidate" means the Investigator confirms (and the judge, before a phone call). The Eye alone
never turns an S into a call.

### Schema (the `alert_triage` intent)

Observations first, label last, so the model looks before it judges:

`summary`, `category`, `other_text`, `zone`, `movement`
(passing/approaching/leaving/staying/moving_around), `flags[]` (face_covered, touching_handle,
item_carried_away, tool_in_hand, weapon_visible, flashlight, crouching, running, uniform_or_helmet),
`people`, `vehicle_moving`, `animals`, `visibility` (clear/partial), `evidence_frame`, `raw_label`,
`label`, `applied_fact_id`, `serious_behaviour`, `why`.

- **Consistency checks in code:** a `normal` label with an S/E category, or a `partial` visibility on a serious category, opens a case.
- **Server enforcement:** the schema is enforced by `json_schema` on OpenAI-compatible APIs and by guided decoding on vLLM
  (the current `json_object` fallback drops the enums and must not be used with Qwen).
- **Hebrew:** the Eye answers in English. The owner-facing Hebrew comes from the messenger.

### Fine-tuning consequence

The fine-tuned Qwen is trained with the same situation header.

- **Taggers** label observations and `raw_label`, which are context-free.
- **The contextual `label` target** is derived by rule (the priors table) from `raw_label` + category + situation.
- **Augmentation:** the same clip can be paired with different `house_state` and `intent` headers to teach
  conditioning. We do **not** augment `phase`: a daylight clip with a "02:00" header teaches the model to
  ignore the picture.

## Part 3: The tracker and the facts it gives

`box/tracker.py`:
- **One person tracker per camera** (box-overlap matching like `VehicleMemory`; not `model.track`,
  whose state would mix the 5 cameras sharing one model).
- **Fed on every detector look** (about 3/s), not only inside the 10 s clip.
- **Per track:** first and last seen, zone sequence, seconds per zone, returns to a zone.

The Eye receives only what it cannot see in its frames: time in view, time per zone, the path, and the
hour. Never the YOLO counts (the Eye's own count is what cancels false YOLO triggers in `vlm_confirms`),
class names, confidences or coordinates.

## Part 4: Incidents

`box/incidents.py` (code): events on any camera of the house belong to one **incident** while a track
is alive or within 3 minutes of the last one.
- **One incident means one Telegram thread:** the first message is sent and then *edited* as the story
  grows.
- **The incident closes** when nobody has been seen for 3 minutes. Its summary goes to the event store and the digest.
- **This alone cuts notifications:** one walk around the house today means 5 alerts.

## Part 5: The Investigator

### When it runs (cost gate, in code)

It opens a case only when one of these holds:
- the label is `suspicious` or `escalation`
- the category is `unusual` or `serious` for the current situation (priors table)
- the Eye's checks are inconsistent, or `visibility: partial` on a serious category
- the incident spans 2 or more cameras in `late_night` or `away`
- the owner asks ("check what happened at the gate last night")

Everything else is logged with the Eye's answer only. Expected: 5–15% of events.

### The case file (built by code, about 3–6k tokens)

1. **The incident timeline:** every event's Eye observation, in order, across cameras.
2. **Tracker facts:** time in view, path, zones, returns.
3. **Situation:** phase, house state, `expecting`, live house notes.
4. **This house's history, as statistics:** from the Historian's baseline: "person events at back_yard between 02:00
   and 03:00 in the last 30 nights: 0", "N3 at the gate on weekdays: about 4 a day".
5. **Similar past events and their outcome:** the top 5 by category + camera + hour, each with the
   owner's verdict ("false alarm", "the gardener") and the Investigator's earlier verdict.
6. **Known routines and entities:** confirmed facts ("workers at the pergola Sun–Thu 07–15"); vehicle
   descriptions the owner confirmed. No face templates (see Privacy).

### Tools (each budgeted per case)

- `look_again(camera, seconds)`: fresh frames from any camera now (the main-stream crop), sent to the Eye
  with intent `follow_up`.
- `ask_eye(event, questions[])`: targeted questions on saved frames ("Is a hand on the door handle?
  Which frame?").
- `wait_and_watch(seconds ≤ 180)`: re-open the case when the track is still alive at that time. This is how
  S4 is decided, from measured minutes.
- `search_history(query, days)`: the existing event search, with outcomes.
- `recommend(verdict, level, reasons[], evidence[])`: the only way out. `level` is none / log / notify /
  call_candidate. Reasons must cite case-file items or tool results. A code check (like
  `brain/claims.py`) drops any reason that cites nothing.

### Rules

- **What it may not change:**
  - It can raise but not lower E-class.
  - It cannot lower a label that a hard rule raised.
  - It never messages the owner directly; the policy engine and the messenger do.
- **Before a phone call:**
  - `call_candidate` goes to the judge (big model, capped per box per day).
  - If the judge is unavailable or over budget, the policy engine's rule-based fallback applies: E-class
    still calls.
- **Speed:**
  - E-class gets an immediate rule-based heads-up; the Investigator edits it within seconds.
  - S-class waits for the Investigator up to 20 s, then goes out with what is known.
- **When it fails:** an exception, a timeout or a refusal falls back to the Eye's label through today's path. It is never silent.

### Memory (files under `<live_dir>/.registry/`, like `facts.jsonl`)

| Layer | What | Written by |
|---|---|---|
| Episodic | events, incidents, case files, verdicts, owner feedback | guard loop, Investigator, feedback |
| Baseline | per camera × hour-of-week: counts by category, median time in view | Historian, nightly, code |
| Routines | proposed regularities ("a person at the gate about 05:30 daily", "N3: Sun/Tue/Thu"), shown to the owner as a fact proposal with buttons; never silent | Historian, weekly, small LLM in batch, from the baseline |
| Lessons | owner corrections turned into rules ("the neighbour's cat is not an alert"), and the verdicts the owner overruled | feedback specialist; fleet-level weekly improver |

The facts flow (proposed after 3 `expected` verdicts, buttons, Undo) is reused, not duplicated.

### Models

- **Investigator:** a mid model with tools, rare calls. Start with gpt-6-luna; move to gpt-5-mini if the case
  eval says so.
- **Judge:** a big model, only for `call_candidate`.
- **Historian routines:** batch.
- All of them go behind the existing `brain/models.make_model` spec strings.

## Part 6: The agent map after this change

| Agent | Kind | Runs |
|---|---|---|
| Watcher (YOLO + tracker) | code + small model on the box | always |
| Situation engine | code | every look |
| Eye (vision) | small model; API Qwen now, fine-tuned Qwen3.5-4B later | every trigger, snapshot, follow-up |
| Investigator | mid LLM + tools + memory | 5–15% of events, and on request |
| Judge | big model | before a call, capped |
| Policy engine | code | every decision |
| Messenger (Hebrew) | template + small LLM | every owner message |
| Concierge (owner chat: router + specialists) | existing brain v2 | owner messages |
| Historian | code nightly + small LLM weekly in batch | background |
| Coach (weekly improver) | big model in batch, fleet level | weekly; turns feedback into rules, prompt changes and training data |

## Cost per box per month (estimates; measure in the eval)

| Item | Assumption | Cost |
|---|---|---|
| Eye, alert_triage | 150/day, API Qwen | $0.4–2.1 |
| Eye, snapshot / follow-up | 20/day | ≈ $0.1–0.3 |
| Investigator | 10–20 cases/day × about 6k in + 400 out, luna | ≈ $0.3–0.6 |
| Judge | ≤ 2/day cap | $0.25–0.8 |
| Messenger + concierge + voice | from the report | ≈ $0.5–1.3 |
| Historian / coach | batch | ≈ $0.05–0.1 |
| **Total** | | **≈ $1.6–5.2** |

## Privacy

- **Activity only.** Descriptions name activities, never identities.
- **No face templates by default** (Ring's Familiar Faces is blocked in Illinois/Texas/Portland and faces a class
  action; Israel's Privacy Protection Law Amendment 13 tightens biometric data).
- **Regulars** are learned from routine (time, zone, vehicle, clothing type). Entity memory beyond that is a
  later, opt-in, on-box feature after a legal check.

## Data contract changes (must flow through meta, eval and training)

- **New fields in `.meta.json` and `teacher_record`:** `situation{phase,dark,house_state,intent,camera_role}`,
  `observation{category,zone,movement,flags,visibility,evidence_frame}`, `incident_id`, `case_id`,
  `investigator{verdict,level,reasons}`, `prompt_version`.
- **Existing fields:** `raw_label` stays the training target for judgment; `label` stays what was acted on.
- **`eval_prompt.py`** gains `--situation` (fixed, or from the clip's timestamp) and reports per category.

## Evaluation

- **Eye:**
  - the existing 220-clip set plus at least 100 alert clips
  - per-category confusion
  - false alarms per situation (a day clip and a night clip are scored against their own priors)
  - refusals, invented flags (checked against the tracker)
- **Investigator:** a case eval of about 50 scripted incidents (multi-camera sequences built from real clips,
  including "knock then side passage", the gardener on a work day, a courier at 23:30, a person at
  the back door at 02:30). Measured: right level, cited evidence valid, tools used within budget, latency.
- **Product metrics on live boxes:** notifications per home per week, false alarms per home per week, time
  to first alert (p50/p95), calls per month, cost per box.

## Out of scope (for this spec)

Face recognition, licence plates, a full on-box VLM, voice out, and the Admin Center views of cases.

## Open questions for the owner

1. House state source: explicit commands plus a default night schedule (recommended), or also
   auto-detect from phones on Wi-Fi later?
2. Should the first phone call ever require the judge's agreement (recommended), or can E-class call on the Eye alone?
