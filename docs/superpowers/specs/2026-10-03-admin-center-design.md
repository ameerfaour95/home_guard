# Home Guard Admin Center — design

Date: 2026-10-03. Status: approved by the founder ("yes") after research + Codex audit.
Branch: `admin-console` (worktree `home_guard_admin`).
Inputs: `docs/admin/AUDIT_AND_PROPOSAL.md` (Codex static audit of every box artifact, with file:line evidence),
research on Verkada Command, Rhombus, UniFi Site Manager, Frigate, FiftyOne, Encord, Nucleus, Roboflow,
AWS IoT shadow, Balena, Mender, Viam, NVIDIA Fleet Command, FTC v. Ring (2023).

## 1. Goal

One professional console where Home Guard staff can, for every customer box:

1. see fleet health at a glance;
2. see what the customer sees — live cameras (on demand) and recorded events;
3. see every AI answer (prompt version, input frames, raw + parsed answer, decision) and the YOLO boxes over the video;
4. review history very fast (keyboard-first) and export clips + YOLO + AI answers + owner feedback as versioned training data;
5. change a customer's configuration remotely and see it confirmed by the box;
6. later: roll out model updates (YOLO weights, VLM prompt/model) with canary + rollback.

Non-goals for the first release: model rollout (phase 4), web/mobile console, customer-facing portal,
continuous cloud recording, semantic image search.

## 2. Decisions (founder)

| Topic | Decision |
|---|---|
| Form | Windows desktop **.exe** in the style of the existing Home Guard app (PySide6). Not a web app. |
| Users | Founder + a few staff. Roles: **admin** (everything), **support** (view + config, no export), **labeler** (training studio only: no live view, no customer identity, cameras shown as pseudonyms). |
| Privacy | Every live view, recording view, export and config change is written to an append-only audit log, and the owner sees a notice **inside his Home Guard app** (access history list + notice). **No Telegram message.** |
| First release | Fleet + see-what-they-see, training studio, remote config. |
| Architecture | **A**: the exe talks only to the Home Guard Cloud service; staff laptops never hold AWS keys or SSH keys to customer boxes. |

## 3. Architecture

```
 ┌──────────────────────┐   HTTPS (JWT, TOTP sign-in)   ┌──────────────────────────────────────────┐
 │ HomeGuardAdmin.exe   │ ────────────────────────────▶ │ Home Guard Cloud (one AWS EC2, same      │
 │ PySide6              │ ◀── presigned S3 media URLs ─ │ region as bucket; docker compose)         │
 │ fleet · customer ·   │                               │  api      FastAPI                         │
 │ review · studio ·    │                               │  worker   indexer, renditions, recompute, │
 │ config · audit       │                               │           exports, health alerts          │
 └──────────────────────┘                               │  db       Postgres 16                     │
                                                        │  tailscale (tag:hg-cloud)                 │
                                                        └──────┬───────────────────────┬────────────┘
                                         IAM instance role     │                       │ Tailscale SSH (port 22 only,
                                         (no stored keys)      ▼                       ▼  ACL tag:hg-cloud → tag:hg-box)
                                              s3://security-camera-project-v1     Customer box (Beelink)
                                              dataset_<site>/  production_<site>/  serve.py on 127.0.0.1 (via ssh -L)
                                              fleet/<device_id>/ …                 publisher + reconciler (new)
                                              admin_cache/ training_exports/
```

Rules:
- The cloud is the single place that holds S3 read/write for admin purposes, the SSH key to boxes, and the audit log.
- Boxes stay autonomous: alerts never depend on the cloud being up.
- No extra YOLO runs on the N150 for admin purposes (CPU is saturated). Heavy media work happens in the cloud worker.

## 4. Components

### 4.1 `home_guard_project/fleet_contract/` (shared, stdlib only)
Versioned JSON documents used by box, cloud and exe: `EventManifest`, `Detections` sidecar header/rows, `AiRun`,
`DesiredState`, `ReportedState`, `ApplyReceipt`, `AccessNotice`. Each has `schema_version`, a `validate(dict) -> list[str]`
and `to_dict()/from_dict()`. Pure dataclasses, no third-party deps (the box imports it). Golden fixtures in
`tests/fleet_contract/fixtures/`.

### 4.2 Home Guard Cloud — `home_guard_project/cloud/`
Own `pyproject.toml` (FastAPI, SQLAlchemy 2, Alembic, psycopg, boto3, argon2-cffi, pyotp, PyJWT, paramiko). No torch in the
API image; the recompute worker image adds ultralytics (CPU).

- **Auth:** email + password (argon2id) + TOTP (pyotp) at sign-in. Access JWT 15 min, refresh 12 h, revocable.
  Roles enforced server-side on every route; the exe only hides what the role cannot do.
- **Registry:** `customers`, `devices`, `assignments` (device ↔ customer ↔ site alias, with validity range), `cameras`
  (+ aliases for renames). Phase 1 enrolment is manual (admin enters site + Tailscale host); phase 2 boxes report
  `device_id` in heartbeat/reported state.
- **Indexer:** every 2 minutes lists only registered prefixes (`dataset_<site>/meta/`, `production_<site>/meta/`,
  `production_<site>/feedback/`, `dataset_<site>/_status/`, `fleet/<device_id>/`), compares ETags, fetches changed JSON,
  stores **raw revisions** and a normalised `events` row. Production + training copies of the same clip
  (same site, camera, stem) form one event with several artifacts. Unknown fields are kept. Legacy rules from the audit
  (B3): normalise `\` to `/`, reject traversal/cross-tenant paths, epoch fields over file names, missing ≠ empty,
  `model_response` may be object/string/null. Completeness badges per event: `video`, `boxes: captured|sampled|recomputed|none`,
  `ai: real|failed|fallback|none`, `owner feedback`, `expired`.
- **AI runs:** legacy teacher blocks become `ai_runs` rows; a run whose `model_response` is `{summary: ""}` with no raw
  answer is marked `fallback` (NullBackend), never a real teacher.
- **Media:** `POST /artifacts/{id}/access` returns a 5-minute presigned GET (no raw S3 keys accepted from clients).
  The worker creates `admin_cache/renditions/<event_id>.mp4` (H.264, faststart, Qt-playable) and a thumbnail when
  the source is not H.264, and a 1-fps filmstrip for fast scrubbing.
- **Recompute boxes:** worker job runs YOLO (yolo11s, CPU) over a legacy clip and stores a detections sidecar marked
  `provenance: recomputed` with model hash. Never shown as captured.
- **Live sessions:** `POST /devices/{id}/view-sessions {reason, cameras}` checks role + consent, opens an SSH tunnel
  (paramiko) to the box, starts `serve.py` with `--viewer "<staff name>"`, and proxies `/status`, `/preview/<camera>.jpg`
  and `/ai_status.json` through the cloud with frame age. Lease 10 minutes, renewed while the window is open, hard cap 60.
  One tunnel per box shared by all viewers.
- **Audit + owner notices:** every sensitive read/write appends to `audit_log` (DB, no UPDATE/DELETE grant for the app
  role) and to `fleet/<device_id>/notices/<ts>_<id>.json` (owner notice). Recording views are batched into one notice per
  staff member per customer per 30 minutes. Live views are noticed by the box itself (see 4.3).
- **Config:** `POST /devices/{id}/config/validate` and `PUT /devices/{id}/desired` (expected-revision check, 409 on
  conflict) write the next generation of `fleet/<device_id>/state/desired.json`; `GET /commands/{id}` returns the
  box's receipt. Allowed keys: `alert_start_hour`, `alert_end_hour`, `alert_cooldown_sec`, `inference_conf`, `alert_on`,
  `mode`, `alert_channel`, per-camera `enabled`/`display name`/`alert_on`, zones. Never credentials, site, network,
  Telegram chat ids (admin-only, phase 3b).
- **Training studio:** saved filters (SQL over events/ai_runs/feedback), collections, `exports` that freeze a selection
  into `training_exports/<name>/v<N>/` with `manifest.json` (hashes, source events, label provenance, consent), YOLO labels
  remapped to contiguous 0–8 (via `analysis/` remap tables), VLM SFT JSONL (LLaMA-Factory ShareGPT, qwen2_vl template),
  and a group-aware split (house + day never crosses train/test). "Send to Label Studio" reuses `labeling/` task generation.
- **Fleet health:** per box verdict `healthy | warning | critical | offline | unknown` from heartbeat age (offline > 90 min
  until phase 2 reported state, then > 5 min), collector running, disk free, upload backlog, newest clip age per camera,
  AI error rate. Health alerts go to the **founder's** Telegram (not the customer's).

### 4.3 Box additions — `home_guard_project/box/`
- `device_id`: UUID created once into `box.yaml`; included in heartbeat and reported state.
- **Detections sidecar:** `alert_clips.save_clip` also writes `detections/<camera>/<day>/<stem>.detections.json.gz` from
  the detector results inference already computed for frames in the clip ring, keyed by output frame index + capture time,
  with `status: ran|ran_empty|not_run`. No extra YOLO.
- **AI run records:** `ai_runs/<camera>/<day>/<stem>.<run_id>.json` per guard VLM call, including failures/refusals,
  written once, never rewritten.
- **Feedback no longer degrades training meta:** `feedback.keep_for_training` stops recreating a training copy without
  teacher data; feedback stays an independent record linked by `alert_id` (the cloud joins them).
- **Upload allowlist:** outbox stages `detections/`, `ai_runs/` and `manifest.json` (written last) alongside today's dirs.
- **Reconciler (`box/fleet_sync.py`)**, run from the heartbeat task every ~60 s with jitter: publishes `reported.json`
  (effective config + hash, versions, camera health, disk, upload backlog), pulls `desired.json`, validates with the
  existing validators (`boxconfig.set_option`, `camera_alerts`, `zones`, `find_cameras apply`), applies as one revision
  with at most one restart, writes an apply receipt (`received → validated → applying → applied | rejected | failed`).
  Owner-local changes bump `local_revision` and are reported as drift, never overwritten by an old generation.
  An owner stop/pause always survives. Pulls `notices/` into `logs/access_log.jsonl`.
- **Access notices in the owner's app:** `serve.py --viewer NAME` appends a live-view notice to `logs/access_log.jsonl`
  at session start/end; the PySide6 box app shows an "Access history" panel and a notice banner for new entries.
- **IAM:** new `make_box_key --device` policy: Put under its own `dataset_<site>/`, `production_<site>/`,
  `fleet/<device_id>/{reported,results,events}`; Get/List only `fleet/<device_id>/{state/desired.json,notices/}`.

### 4.4 HomeGuardAdmin.exe — `home_guard_project/admin/`
PySide6, same theme tokens as `box/app/theme.py`, packaged with PyInstaller (`build_admin_exe.ps1`), logo from `box/assets`.
An `AdminBackend` protocol with `HttpBackend` (real) and `DemoBackend` (fixtures, used by tests and screenshots) — the same
pattern as the box app.

Screens:
1. **Sign-in** (email, password, TOTP).
2. **Fleet**: table/cards of customers → boxes with health verdict chips, last seen, cameras up/total, newest clip,
   disk, backlog, mode, versions, config drift. Filter by verdict. Sorted worst-first.
3. **Customer**: header (customer, site, box, verdict, consent state) + tabs:
   - *Live* — camera grid on demand (reason prompt, lease countdown, frame age, detections toggle);
   - *Timeline* — per-camera density strip by hour, event list (kind colour, AI decision, owner verdict chip), filters;
   - *Event* — player with YOLO overlay layer (provenance label, offset slider), filmstrip scrubber, AI record panel
     (prompt version, input frames, raw/parsed, decision, delivery), owner feedback, chat thread around the event;
   - *Conversation* — the owner's Telegram history (role-gated);
   - *Config* — current vs desired diff, editor with validation, restart impact, command status timeline;
   - *Access* — audit entries for this customer.
4. **Review queue**: fleet-wide stream of events; `j/k` move, `space` play, `r` reviewed, `f` flag, `c` add to collection,
   `1–5` quick verdicts; `Ctrl+K` command palette (jump to customer, camera, event, saved filter).
5. **Training studio**: saved filters (built-ins: owner said false alarm; owner said real but wrong; AI dismissed but YOLO
   saw a person; AI call failed/fallback; low detector confidence; paused-camera footage), collections, export wizard
   (formats, split, consent check), export history with versions, "send to Label Studio".
6. **Audit**: fleet-wide audit log, filter by staff/customer/action.

UX bar: dark theme from the box app, no layout jumps, every list virtualised, every network call off the UI thread with
skeleton states, all failures shown as designed messages (never tracebacks), keyboard-first.

## 5. Data model (Postgres)

`staff(id, email, name, role, password_hash, totp_secret, disabled)`, `sessions`,
`customers(id, name, timezone, consent_live, consent_recordings, consent_training, notes)`,
`devices(id, device_id, tailscale_host, ssh_user, hardware, enrolled_at)`,
`assignments(id, device_id, customer_id, site, valid_from, valid_to)`,
`cameras(id, assignment_id, name, display_name, enabled)`, `camera_aliases`,
`events(id, assignment_id, camera_id, legacy_stem, kind, mode, start_ts, end_ts, trigger_ts, label, summary,
alert_command, completeness jsonb, expires_at, reviewed_by, reviewed_at, flags)`,
`artifacts(id, event_id, role, s3_key, etag, bytes, mime, provenance, available)`,
`raw_revisions(id, s3_key, etag, fetched_at, body jsonb)`,
`ai_runs(id, event_id, purpose, model, prompt_version, prompt_hash, status, parsed jsonb, raw_artifact_id, input_artifact_ids)`,
`feedback(id, event_id null, customer_id, verdict, action, note, source, received_at, s3_key)`,
`chat_messages` (phase 2), `device_state(device_id, desired_generation, desired_doc, reported_doc, reported_at, drift)`,
`commands(id, device_id, generation, author, reason, state, receipt jsonb)`,
`audit_log(id, ts, staff_id, action, customer_id, device_id, target, reason, detail jsonb)` append-only,
`saved_filters`, `collections`, `collection_items`, `exports`, `export_items`.

Indexes: `(assignment_id, start_ts desc, id)`, `(camera_id, start_ts desc)`, GIN on `completeness`, full-text on `summary`.
Target: p95 < 500 ms for a 100-event page; measured, not promised.

## 6. Phases and acceptance

| Phase | Scope | Accepted when |
|---|---|---|
| **1 — read-only console** | contract package; cloud API + DB + auth/roles + registry + indexer + media access + renditions + audit; exe sign-in, fleet, customer timeline, event player (overlay when sidecar/recomputed exists), AI record, review queue, studio filters/collections/export (YOLO + VLM JSONL + manifest). Runs locally against real S3 (read) before AWS deploy. | Founder browses `ameer_house`/`test`/`bian` history from both prefixes with copies merged, sees AI records with honest badges, exports a versioned dataset that `analysis/`-style consumers accept; every view audited. |
| **2 — see what they see + publication** | box detections sidecar, AI-run records, feedback fix, device_id, reported state, access notices in box app; cloud live sessions via Tailscale; recompute job; founder health alerts; AWS deploy. | New alert clips replay with captured boxes; live grid works with lease + frame age; owner's box app lists the access; disconnect ends "live". |
| **3 — remote config** | desired/reported reconciler on box; config editor + command status in exe. | Offline edit queues and applies once on reconnect; stale revision → 409; owner pause survives; invalid zone rejected with reason; "applied" shown only after the box's receipt. |
| **4 — model releases** | signed manifests, staged download, supervisor rollback, canary cohorts, shadow mode, re-ask-the-AI. | Bad hash, crash on start, power loss mid-activation and explicit rollback all leave the box on a working release. |

## 7. Error handling

- Cloud down → exe shows designed offline state; boxes unaffected.
- Box offline → fleet shows `offline` with last-seen; live session refused with reason; config commands queue.
- S3 object missing/expired → artifact `available=false`, badge `expired`; exports fail or mark partial explicitly.
- Indexer meets malformed JSON → raw revision stored, event quarantined with reason, visible in an "Index problems" view.
- Live tunnel drops → frames freeze with age counter turning red; auto-reconnect 3× then designed error.

## 8. Testing

- `tests/fleet_contract/`: golden fixtures for every document incl. legacy oddities from the audit (backslash paths,
  string `model_response`, NullBackend fallback, late feedback, renamed camera, duplicate copies, orphan video).
- `tests/cloud/`: API + indexer against a real Postgres via `pgserver` (embedded binaries, no Docker), S3 via `moto`.
  Every route has an auth/role test.
- `tests/admin/`: Qt offscreen tests against `DemoBackend`; screenshot set under `docs/admin/screenshots/` reviewed each Codex round.
- `tests/box/`: new tests for sidecar, ai-run records, feedback fix, reconciler (incl. interrupted apply), access notices.
- Cross-model review: Codex reviews Claude's backend/box diffs; Claude reviews Codex's UI rounds.

## 9. Work split

- **Codex** (worktree `home_guard_admin_ui`, branch `admin-console-ui`, one round at a time, high reasoning): the exe
  against `DemoBackend` + the published OpenAPI schema; screenshots + tests per round; independent reviews.
- **Claude subagents** (worktree `home_guard_admin`): fleet_contract, cloud, box changes, test-first.
- Integration and box deployment: this session, after review. Other sessions own `home_guard_ui` / `box-app-ui`; the box
  app's access-history panel is done in coordination with them.

## 10. Open items for the founder (not blocking phase 1)

- Permission to create the EC2 instance, IAM instance role, Tailscale tag/ACL, and ~$15–30/month.
- Customer consent text (live / recordings / training) — legal review per jurisdiction before selling widely.
- Whether training copies of customer footage are retained beyond 14 days (today: yes, indefinitely).
