# Collector box: inference (production) mode — design

Date: 2026-10-02
Status: approved by delegation (founder chose Twilio, GPT-4V now, N150 + cloud VLM)

## Goal

Give a sold box two modes, chosen in the setup app:
- **`data_collection`** (today): collect trigger clips for Label Studio tagging. Unchanged.
- **`inference`** (production): watch the cameras, and when something happens inside the owner's configured time window, send the owner a WhatsApp message (or place a call for serious events).

The box is the same N150. YOLO runs locally; the VLM call goes to the cloud (GPT-4V) because the N150 cannot run a VLM while YOLO already saturates its CPU. A future open-source VLM that runs locally will drop in behind the same interface.

## What already exists (reused, not rebuilt)

- Alert-window logic: `is_in_alert_window(now, start_hour, end_hour)` in `run_with_gpt.py`.
- Alert decision + prompt: GPT-4V returns `{summary, alert_command, alert_reason}` with `alert_command` in `[none]|[send_message]|[call_owner]`, and the policy override (person/car in-window with `[none]` -> `[send_message]`).
- YOLO + ByteTrack + frame buffer loop in `run_with_gpt.py`.

## What is new

- **Delivery does not exist.** `run_with_gpt.py` only `print()`s the decision. Need a Twilio notify module (WhatsApp message + Voice call).
- **`run_with_gpt.py` is a windowed single-camera desktop app** (`cv2.imshow`, `waitKey`) and is excluded from the box bundle. Inference mode needs a **headless, multi-camera runner** built from its logic — a new file, leaving `run_with_gpt.py` as the reference implementation.
- **Mode switch**: `box.yaml` gains `mode`; the box runs the collector or the inference runner accordingly.
- **Secrets on the box**: OpenAI key + Twilio SID/token.

## Components

| Unit | Status | Purpose |
|---|---|---|
| `box/notify.py` | NEW (this spec, built first) | Twilio WhatsApp message + Voice call, via stdlib `urllib` (no new pip dep). Dry-run by default; real sends only when credentials are present and dry-run is off. `notify(cfg, command, summary, reason)` dispatches on `alert_command`. |
| `box/infer.py` | NEW | Headless, multi-camera inference loop: YOLO -> per-camera frame buffer -> on escalation inside the window, call the VLM backend -> `notify`. Cooldown per camera. No windows. |
| VLM backend iface | NEW | `analyze(frames) -> {summary, alert_command, alert_reason}`. `GptBackend` (GPT-4V) now; a `LocalBackend` later. Chosen by config. |
| `box/run_inference.sh` | NEW (coordinate w/ peer) | Runner wrapper, restart loop, `collector.alive`-style heartbeat. Mirror of `run_collector.sh`. |
| `box.yaml` fields | COORDINATE w/ peer | `mode`, `alert_start_hour`, `alert_end_hour`, `owner_phone`, `owner_whatsapp`, `twilio_from`, `vlm_backend`. |
| `api_key.env` (gitignored) | COORDINATE | `OPENAI_API_KEY`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`. Already in bundle EXCLUDE_NAMES. |
| `setup_box.ps1` task | COORDINATE w/ peer | Register `HomeGuard-Collector` OR `HomeGuard-Inference` based on mode (peer owns task wiring). |
| wizard (`setup_customer.ps1`) | EXTEND | Ask mode; if inference, ask alert window + owner phone; collect secrets into `api_key.env` on the box. |

## Decisions

- **Twilio for both** WhatsApp (`Messages.json`) and calls (`Calls.json` with inline TwiML `<Say>`). One vendor, and it can actually place the `[call_owner]` call.
- **GPT-4V now, pluggable.** The VLM call sits behind a backend interface so the future local model replaces it without touching the runner.
- **This N150 + cloud VLM.** YOLO local, VLM remote. Documented limits: needs internet; each escalation costs an API call; a per-camera cooldown bounds cost.
- **Modes are mutually exclusive per box.** A production box does not also collect for training.
- **Delivery is dry-run-safe.** No message or call goes out unless credentials are configured and dry-run is explicitly off, so a misconfigured box is silent, not wrong.
- **Secrets never in git**, never on a command line, never in the heartbeat.

## Error handling

- VLM call fails / no internet -> log, skip this escalation, keep running (no alert rather than a crash).
- Twilio send fails -> log; for `[call_owner]`, if Voice is not configured, fall back to a WhatsApp message.
- Outside the window -> `[none]`, no VLM call at all (saves cost).
- Per-camera cooldown prevents alert storms.

## Testing

- `notify.py`: unit tests with the HTTP transport patched — dry-run sends nothing; enabled path posts the right URL/fields; `[send_message]` vs `[call_owner]` dispatch; `[call_owner]` falls back to WhatsApp when Voice unconfigured. (Built first, this turn.)
- `infer.py`: unit tests for the escalation gate (window + detection -> VLM called; outside window -> not called) and cooldown, with YOLO and the backend mocked.
- On the box: a real end-to-end dry-run (decisions logged, no Twilio) before any live credentials.
- Live Twilio + GPT-4V: a bench test the founder runs with real keys, to a test phone.

## Out of scope (now)

- The local open-source VLM backend (interface only; built when the model exists).
- Recording/clip upload in inference mode (production box alerts; it does not tag).
- Any UI beyond the setup wizard's prompts.
