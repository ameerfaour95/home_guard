# Owner assistant v2: "the brain" — design

Date: 2026-10-03
Status: approved section by section by the founder in brainstorming (approach 1, "receipts-first")
Scope: sub-project **A** of the assistant upgrade. B (richer per-event understanding at capture time),
C (proactive reports, health alerts) and D (more settings by chat) get their own specs later.
Second opinion: an independent design memo from Codex (gpt-6-astra, read-only) was compared against
this design; where it disagreed, the decision and the reason are recorded below.

## Goal

The Telegram assistant on a box in inference mode must be good enough to demo to a buyer:

1. It never says it did something it did not do.
2. It picks the right action for what the owner asked, in Hebrew, Arabic or English.
3. It knows the house: which camera is "the entrance", which cameras are off, muted, or offline.
4. It asks when a message is ambiguous instead of guessing (above all, before saving a training label).
5. It works in two modes with different prompts and tools: **Guard** inside the owner's alert hours
   (everything about "suspicious or not, and why") and **Assistant** outside them (a general helper).
6. Every one of these is measured by an eval suite and gated before release.

## Why: the failures that motivate it

From the real conversation on the home box (`production_multi/.conversations/-5326761586.json`,
2026-10-03), translated from Hebrew:

| # | Owner wrote | Assistant did | Root cause |
|---|---|---|---|
| 1 | "Get me a picture of what's happening now at the entrance" | Sent an old saved clip (01:22) | `find_alerts` advertises pictures and competes with `check_camera` |
| 2 | "Describe what's happening now on the back camera" | "Nothing was saved" | Searched the archive instead of looking live; cameras named `test_chN`, no aliases |
| 3 | "Turn off the front camera only" | Muted alerts, said "camera turned off until 01:35" | Wrong tool, then a false action claim in free prose |
| 4 | "this one" | Saved verdict `false_alarm` | Unthreaded message bound to the latest alert (up to 6 h old); verdict not tied to owner's words |
| 5 | "Give me both cases" | "Here are the two videos" | Prose claims delivery; `send_clip` only queues, Telegram's `ok` is never checked |
| 6 | "The picture isn't clear" | "The picture is clear" | Live prompt says "view looks clear" for *no activity*, mixing up picture quality and activity |
| 7 | "The video from 5 seconds before this event" | "Nothing was saved" | Clips already hold 4 s before + 6 s after; no tool to send a segment |
| 8 | Hebrew questions | Some answers in English | Heuristic language choice; hard-coded English system text |
| 9 | "Update me what's happening there" | "There is no camera test_ch6" | Camera list taken at startup; no camera/mute state; history keeps words but drops tool results and IDs |

The ranking (agreed with the Codex memo): the model controls both the actions and the story about them,
with no authoritative record linking the two. A stronger model reduces mistakes but does not fix this.

## What already exists (reused)

- `agent.py`: native JSON tool calling, `chat(messages, tools, tool_choice)` model interface, tool loop,
  `_quoted_from` owner-words check for pauses, never-raise `handle()`, always saves the owner's message.
- `inference.py` guard loop: YOLO gate → VLM with strict JSON
  `{summary, label: normal|suspicious|escalation, people, vehicle_moving}` (the tagging format),
  `LABEL_COMMANDS`, `vlm_confirms` false-positive rule, `VehicleMemory` parked-car rule, watch-zone masking,
  per-camera cooldown, `LiveSettings` (hours, cooldown, conf change while running).
- `alert_clips.py`: 4 s pre-roll + 6 s post-roll clips; `production_multi/` (14 days on box and S3).
- `feedback.py` (`MuteState`, verdict validation), `archive.py` (records + embeddings search),
  `live_view.py` (`look_now`), `telegram_agent.py` (alerts, buttons, inbox), `find_cameras.apply_changes`.

## Architecture

```
Telegram message ─▶ TurnBuilder ─▶ ModeResolver ─▶ Profile (Guard | Assistant)
                                                     ├─ system prompt (versioned)
                                                     └─ allowed tools
                         HouseRegistry (live state) ──┤
                         ChatMemory (refs, pending) ──┤
                                                     ▼
                                  ChatModel (pluggable) ⇄ Tools ─▶ Receipts (on disk)
                                                     ▼
                                  ReplyRenderer: model answer (facts) + code-written
                                  confirmations from receipts, in the owner's language
```

### Units

| Unit | New/changed | Purpose |
|---|---|---|
| `box/brain/mode.py` | NEW | `resolve_mode(now, start_hour, end_hour) -> "guard" \| "assistant"`, same window rule as `inference.in_alert_window` (including overnight windows). Also builds the status line. |
| `box/brain/profiles.py` + `prompts/guard.txt`, `prompts/assistant.txt` | NEW | The two system prompts (versioned, `PROMPT_VERSION` per profile) and each profile's tool list. Shared rules (honesty, language, clarification) in one common block. |
| `box/brain/registry.py` | NEW | `HouseRegistry.snapshot()` built fresh each turn: cameras, aliases, enabled, live/offline + last frame time, mute expiry, "sees" line, mode, retention and quiet-log coverage. Alias resolution in code. |
| `box/brain/memory.py` | NEW (replaces `conversation.py` use) | Per-chat: turns, event handles (E1…→ real IDs), delivered media with Telegram message IDs, pending clarification, per-speaker language. |
| `box/brain/receipts.py` | NEW | `Receipt` dataclass, append-only JSONL per day under `production_multi/.receipts/`, idempotency key per (turn, tool, target). |
| `box/brain/render.py` | NEW | Builds the outgoing message: answer text + ✓/✗ lines from receipts, localized templates (he/ar/en) for every system string, action-word safety net. |
| `box/brain/tools.py` | NEW | The tool handlers (contracts below). Each returns a JSON result and, if it acts, a receipt. |
| `box/brain/agent.py` | NEW | `OwnerAgentV2.handle(...)`: same signature and never-raise contract as today's `OwnerAgent.handle`, so `telegram_agent.py` can switch on `agent_version`. |
| `box/brain/models.py` | NEW | `ChatModel` adapters: OpenAI (moved from `agent.py`), Anthropic, Gemini via its OpenAI-compatible endpoint. Selected by `agent_model: <provider>:<model>`. Structured final answer enforced by JSON schema. |
| `box/agent_tools_v2.json` | NEW | JSON-Schema tool definitions; profile picks the subset. |
| `box/archive.py` | CHANGED | `AlertRecord` gains `kind`, `label`, `people`, `mode`, `described`; reads `<stem>.desc.json`; metadata filters before semantic ranking; keyword fallback never returns "everything" for a non-matching query. |
| `box/inference.py` | CHANGED | Quiet event log outside the alert hours (below). |
| `box/live_view.py` | CHANGED | Separate `quality` (clear/blurry/dark/no_signal) from activity; never "the view looks clear" for "nothing is happening"; returns the photo even when the description fails. |
| `box/telegram_agent.py` | CHANGED | Chooses v1/v2 by `agent_version`; media sends return Telegram's `ok` + `message_id`; `sendChatAction typing` during a turn; mode-switch announcements. |
| `box/agent.py` (v1) | KEPT | Fallback until v2 passes the live eval; then removed in a later change. |
| `tests/agent_eval/` | NEW | Eval cases, runner, scorecard (below). |

## The two modes

- Mode is computed in code from `alert_start_hour`/`alert_end_hour` (the same values the guard loop reads
  live). It is fixed for a whole turn. A conversation crossing the boundary keeps its handles and pending
  question and gets the new profile on its next turn; a tool not in the new profile is not executed.
- Each event records the mode at event time, separately from the current mode.
- **Guard profile:** answers about events lead with the label and the why, highest label first:
  🟢 normal, 🟡 suspicious, 🔴 escalation (the tagging labels, unchanged). Short and decisive tone.
  Straightforward requests ("send the entrance picture") are served plainly, not turned into a verdict.
- **Assistant profile:** factual helper; time-ordered results; neutral descriptions; answers the owner's
  specific question about a clip.
- **Status line, identical in the app and Telegram:** `🛡️ Guarding until 06:00`,
  `🛡️ Guarding · Entrance alerts paused until 08:00`, `🛡️ Guarding · ⚠️ Back camera offline`,
  `💬 Assistant · quiet logging`. A schedule alone never claims protection: offline cameras are shown.
  Written to `logs/ai_status.json` (`mode`, `status_line`) for the app. The app UI change itself is a
  separate Codex brief.
- **Mode switch announcement** in Telegram, once per switch:
  "🛡️ Guarding started, until 06:00, 6 cameras live" / "💬 Guarding ended. Quiet logging until 22:00."

### Guard alert delivery (graded by label)

| Label | Telegram | Command |
|---|---|---|
| 🟢 `normal` | message sent with `disable_notification: true` (silent) | `[send_message]` |
| 🟡 `suspicious` | normal message, with "Why: …" | `[send_message]` |
| 🔴 `escalation` | loud message (+ call when Twilio is upgraded) | `[call_owner]` |

`LABEL_COMMANDS` is unchanged. The new part is the silent flag for `normal` and the "Why" line, which
comes from a new optional `why` field in the VLM schema (one short clause; empty for `normal`).
The label prompt stays as tagged by the founder (a hidden face is a behaviour and stays `suspicious`);
one rule is added: dark clothing alone never raises the label.

## House registry

Injected as a compact block at the top of every turn:

```
CAMERAS (6 · 5 live)
main_entrance  aka: entrance, front door, כניסה, المدخل   live   alerts on
back_door      aka: back, backyard, אחורית, الخلف          OFF (turned off 10:44)
front_side     aka: front, street, קדמית                  live   alerts paused until 08:00
  sees: "driveway with two parked cars, front gate"
MODE: Guard until 06:00 · clips kept 14 days · quiet log since 06:00
```

- `aliases:` list per camera in `cameras.yaml` (new optional key; renames carry it, like zones).
  Set in the app or by chat (`set_alias`).
- Resolution in code: exact name → alias (case/space-insensitive, all languages) → unique prefix.
  Several matches → `ask_clarification` with one button per camera. No match → error listing the cameras.
- `sees:` one line per camera from the VLM, generated on first run and refreshed weekly, cached in
  `production_multi/.registry/sees.json`.
- Disabled cameras stay in the registry, marked OFF. No RTSP URLs or credentials in prompts.
- Live/offline from the guard loop's per-camera `checked_ts` in `logs/ai_status.json` (the last time the
  detector looked at a frame, about once a second; with the quiet log this runs all day); offline when it
  is more than 60 s old.

## Chat memory

Stored per chat in `production_multi/.conversations/<chat_id>.json` (format v2; v1 files are read and
upgraded on first write).

- **Handles:** every event, photo or clip shown in a turn gets a handle `E1`, `E2`… mapped to its real
  ID, kept for the conversation (last 50). The model refers to handles; tools resolve them in code.
  "Both", "the second one" resolve against the handles of the previous turn.
- **Delivered media:** handle, Telegram `message_id`, bounds.
- **Pending clarification:** the question, its choices, and the action waiting on it. The next message
  (or button tap) is resolved against it first; a tap answers it exactly.
- **Language per speaker:** from the letters of the message (today's heuristic); a short reply with no
  letters ("8", "ok", "👍") inherits that speaker's last language. An explicit request ("answer in
  English") overrides until changed.
- History sent to the model: the last 12 turns, each with its handles and receipt summaries, not only
  the words.

## Tools

Every tool that acts returns a receipt. Camera arguments go through registry resolution.

**Both modes**

| Tool | Contract |
|---|---|
| `check_camera(camera)` | Fresh photo + description + `quality` (clear/blurry/dark/no_signal). Photo is sent even if the description fails. Guard: also `label` + `why`. Receipt for the photo delivery. |
| `record_clip(camera, seconds)` | Records a new clip now (1–30 s, default 10), masked to the watch zone, sends it. Receipt with real bounds. One at a time per camera. |
| `find_events(cameras?, day?, time_from?, time_to?, last_hours?, latest?, what?, label?, kind?)` | Saved events as handles with time, camera, kind, label, people, has_video, described. Metadata filter first (time, camera, kind, label, detector classes), then meaning-ranking among described events. Returns `coverage`: cameras on/off in the range, quiet-log start, retention edge. Max 8 results; says how many more matched. |
| `summarize_period(day? \| last_hours?, cameras?)` | Totals by camera, by label, by kind; notable events (all 🔴/🟡, then a sample); coverage. Max 60 events detailed. |
| `send_media(handle, from_sec?, seconds?)` | Sends a clip, a segment of it (cut with ffmpeg, relative to the trigger: `from_sec=-4, seconds=4` is the pre-roll), or a photo. `done` only after Telegram returns `ok`. Max 3 per reply. |
| `record_verdict(handle, verdict, owner_words, note?)` | Verdicts as today (`true_alert`, `false_alarm`, `real_but_wrong`, `expected`, `missed_event`). Saved only if the handle resolves and `owner_words` is a real ≥2-word quote of the latest message (`_quoted_from`). Unthreaded messages bind to an alert only if exactly one alert arrived in the last 30 minutes; otherwise `ask_clarification`. |
| `ask_clarification(question, choices[])` | One question with 2–5 buttons. Nothing that depends on the answer runs this turn. |
| `pause_alerts(owner_words, cameras?, until?, minutes?)` | Mutes notifications only, always with an end (max as today, `MAX_MUTE_HOURS`). Available in both modes so "don't alert me tonight until 23:00" works in the afternoon. Receipt says "alerts paused", never "camera off". |
| `resume_alerts(cameras?)` | Ends a pause, per camera or all. |
| `set_camera_active(camera, active)` | Real on/off through `find_cameras.apply_changes`. Receipt `requested` until the restarted loop reports the camera's new state, then updated to `done` (follow-up message). Kept against the Codex memo's advice: the owner already uses it and it works. |
| `set_alias(camera, alias)` | Adds an alias in `cameras.yaml`. Rejects an alias that already names another camera. |

**Guard only:** `assess_event(handle)` re-watches a saved clip with the guard prompt; returns `label`,
`why`, `uncertainty`, or explicit `refused` / `insufficient_evidence` (neither means normal; the reply
says "activity detected; assessment unavailable").

**Assistant only:** `describe_event(handle, question?)` re-watches a saved clip with a neutral prompt and
answers the owner's question ("what was he holding?", "which way did the car go?").

Both cache to `<stem>.desc.json`, keyed by prompt version and question. Describing one event marks it
`described` and adds its text to the meaning search.

**Removed from v1:** `find_alerts` (replaced by `find_events`), `send_clip` (replaced by `send_media`).

## Receipts and the reply

**The rule: the model states facts; code states actions.**

1. **Receipt** (written to disk before the tool returns):
   ```json
   {"id": "R7", "turn": "…", "tool": "send_media", "status": "done", "target": "E2",
    "detail": {"bounds": "01:24:03–01:24:13", "telegram_msg": 51234}, "ts": "…"}
   ```
   `status`: `done` | `requested` | `failed` (+ `reason`). Same idempotency key → the earlier receipt is
   returned and nothing runs twice.
2. **Structured final answer**, enforced with a strict JSON schema:
   `{"answer": "<facts from tool results only>", "uses": ["E1", "R7", …]}`.
   The prompt forbids describing actions in `answer`.
3. **ReplyRenderer**: media is sent first (so ✓ is true when read), then one message:
   the answer text, then one ✓/✗ line per receipt from localized templates
   (`✓ Video sent (01:24:03–01:24:13)` / `✗ Video E1 could not be sent: it is older than 14 days`).
   All system strings (errors, confirmations, "unavailable") come from he/ar/en templates.
4. **Safety net:** the answer text is checked against a short list of action words per language
   (en: sent, turned off, disabled, paused, muted, saved, recorded; he: שלחתי, כיביתי, השתקתי, שמרתי,
   הקלטתי; ar: أرسلت، أطفأت، أوقفت، كتمت، حفظت). A hit without a matching receipt in this turn →
   one retry with a correction message; if it still fails, the reply is only the receipt lines plus the
   tool facts, and the incident is logged (`brain.claim_guard`). The eval measures how often it fires.
5. **Clarification** is a reply by itself: the question and buttons, nothing else.

Guarantee: every ✓ happened and every failure is reported. Not fully guaranteed: unusual wording in the
answer text that implies an action; the safety net and the eval gate cover that.

## Quiet event log (change to `inference.py`)

- The detector runs all day. Inside the alert hours: unchanged.
- Outside the hours, a trigger (person, or a vehicle that moved per `VehicleMemory`) saves a quiet event:
  same clip (4 s pre, 6 s post), same watch-zone masking, into `production_multi/` (14 days on box + S3).
  Meta: `kind: "quiet"`, `mode: "assistant"`, detector classes and counts, `people` (detector count),
  `described: false`. **No VLM call.**
- **Merging:** while the detector keeps seeing the same class on the same camera (gap < 10 s), the event
  is extended instead of a new clip being started, up to 60 s; then a new event starts.
- Paused cameras: as today (clip kept as `paused`, nothing sent).
- Quiet clips are not copied to `dataset_multi`. If the owner gives a verdict on one, it is copied with its
  feedback exactly like alert verdicts today (`owner_feedback`).
- Disk: existing 14-day retention plus a size cap (`quiet_max_gb`, default 20) that deletes the oldest
  quiet clips first.
- Answering "was anyone near the house?": detector hits first; up to 5 described on demand per question,
  most relevant first; the reply says how many were not checked. Undescribed events are shown as
  "detector saw a person, not confirmed". "Nothing recorded" is always said together with coverage.

## Model layer

- `agent_model` in `box.yaml`: `openai:<model>`, `anthropic:<model>`, `gemini:<model>`. Keys in
  `api_key.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). All use the OS trust store
  (antivirus TLS interception), as today.
- The vision model behind `check_camera` / `describe_event` / `assess_event` stays `vlm_model`, chosen
  separately on its own numbers.
- Which models are candidates is decided when the live eval is run (current availability and prices are
  checked then). Baseline: today's `gpt-4o-mini`. **We ship the cheapest model that passes the gates.**

## Eval suite (`tests/agent_eval/`)

- **Case format** (YAML): `history`, `clock`, `mode`, `registry`, `message` (original language),
  `speaker`, `reply_to` (alert or none), `tool_results` (recorded responses per tool+args pattern),
  `expect`: `tools` (allowed sequences), `forbidden_tools`, `args` (checks), `language`,
  `must_clarify`, `no_false_claims` (always on).
- **Seed set:** the 9 failures above as written (`critical/`), plus ~40 more across he/ar/en and both
  modes: the 06:00 boundary, alias collisions, offline camera, expired clip, failed delivery, two family
  members interleaving, quiet-log questions, pre-roll requests, pause vs turn-off.
- **Growth:** `python -m tests.agent_eval.from_chat <conversation.json>` drafts cases from a saved box
  conversation; the founder marks the wrong answers.
- **Layer 1, offline** (pytest, no network): scripted model outputs test receipts, renderer, safety net,
  verdict gate, mode switching, handle resolution.
- **Layer 2, live** (`python -m tests.agent_eval.run --model <provider:model> --repeat 3`): run by hand,
  costs money. Simulated tools answer from the recorded results. Writes a scorecard (JSON + HTML):
  tool+argument accuracy, false action claims, unintended actions, guessed verdicts, language adherence,
  clarification precision/recall, `assess_event` refusals, latency, cost per completed request;
  per-language and per-mode breakdowns so an average cannot hide a broken slice.
- **Release gates:** critical set: 0 false confirmations, 0 unintended actions, 0 guessed verdicts, every
  repeat. Overall: tool+args accuracy ≥ 95% (raised to 98% when the suite passes 150 cases), language
  adherence ≥ 99%.

## Errors

- Model unreachable or timed out: owner's message saved; localized "I couldn't work on that, your
  message was saved" plus any receipts already done.
- Tool exception: a `failed` receipt with a plain reason; the turn continues.
- Telegram media send fails: one retry, then ✗.
- Turn budget 30 s (model rounds ≤ 5, as today); Telegram shows "typing…" during the turn.
- Registry source unreadable (cameras.yaml, ai_status.json): the turn runs with what is known, and the
  registry block says "camera state unknown" rather than claiming cameras are live.

## Testing

Unit tests (pytest, existing `tests/box/` layout): `resolve_mode` (overnight window, exact boundary
minutes, start == end), alias resolution and collisions, verdict gate (threaded, unthreaded one/many
recent alerts, one-word messages), receipts (idempotency, status transitions), renderer in he/ar/en,
safety net (hit/miss per language, retry, fallback), quiet-event merging and size cap, archive new fields
and v1 meta compatibility, memory v1→v2 upgrade, `send_media` segment bounds. The existing box suite must
stay green.

## Rollout

- `agent_version: 1 | 2` in `box.yaml` (default 1 until the live eval passes, then 2).
- Home box first; customer boxes after the gates pass on the home box's own conversations.
- v1 `agent.py` removed in a later change once v2 has run for a week without a claim-guard incident.

## Out of scope

- B: richer per-event descriptions at capture time.
- C: daily/morning reports, camera-offline push alerts, proactive "why it alerted" follow-ups.
- D: settings by chat beyond `set_alias` and camera on/off (hours, cooldown, sensitivity).
- Family permissions (who may pause or turn off cameras): all members of the configured chat are trusted,
  as today.
- The app's status-line UI (a separate Codex brief reads `ai_status.json`).
- Multi-agent orchestration, model fine-tuning for the chat brain.
