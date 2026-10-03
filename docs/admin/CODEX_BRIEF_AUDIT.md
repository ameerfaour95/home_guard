# Codex brief — Admin Console: data-surface audit + architecture proposal (NO CODE)

You are working in a git worktree `C:\Users\ameer\Ameer\home_guard_admin` (branch `admin-console`).
This is a research/design task. **Do not write or change any code.** Your only output is one file:
`docs/admin/AUDIT_AND_PROPOSAL.md`. Do not commit, do not push, do not ssh anywhere, do not touch AWS,
do not touch any other directory.

## Context

Home Guard sells a Beelink mini-PC ("the box") to households. Each box (Windows 11, Intel N150, no GPU)
runs `home_guard_project/box/`: YOLO on RTSP cameras, triggered clips sent to a VLM (GPT-4o today),
alerts to the owner on Telegram, an owner assistant (chat agent), owner feedback on alerts (false alarm /
true alert ...), clips saved locally and uploaded to S3 bucket `security-camera-project-v1`. A box runs one
mode: `data_collection` or `inference` (production). Boxes are reached over Tailscale SSH. There is a
PySide6 desktop app in `home_guard_project/box/app/` (per-box UI) and a read-only HTTP server `box/serve.py`.
Read `CLAUDE.md`, `home_guard_project/box/README.md`, and `docs/superpowers/specs/*.md` first.

The founder now wants an **Admin Center**: one console listing every customer/box, where staff can
(1) see what the customer sees (live + recorded), (2) see every AI response and the YOLO boxes over the video,
(3) export video + YOLO + AI answers + owner feedback as training data, (4) update a customer's config,
(5) browse history very fast, (6) monitor fleet health, (7) push model updates (YOLO weights, VLM prompt/model)
to boxes.

## What to produce in `docs/admin/AUDIT_AND_PROPOSAL.md`

### Part A — Audit (facts, with file:line references)
1. Every artifact a box writes, per mode: local path, S3 key pattern, retention, who writes it, schema
   (list the real JSON fields of `.meta.json` for kinds `alert`, `false_positive`, `owner_feedback`, `paused`,
   and data-collection clips; `ai_status.json`; heartbeat; telegram chat feed; feedback; responses/; vlm_crops/; yolo/).
2. Are YOLO boxes stored **per frame** for production alert clips, or only counts/summary? Same for data-collection.
   Exactly what would be needed to draw boxes over the saved video in a browser.
3. Config: every key in `box.yaml` (OPTIONS / LIVE_OPTIONS / RESTART_OPTIONS), `cameras.yaml`, `camera_alerts.yaml`,
   zones. How each is changed today (CLI `set-option`, flag files, restart) and what is secret.
4. Models on the box: where YOLO weights and the VLM prompt/model come from, how `PROMPT_VERSION` works,
   how code updates happen today (`update.sh`, git pull), what a "model update" would need to change.
5. What a box knows about its identity (site name, Tailscale host, IAM key scope) and what S3 can tell a
   central console with **no** connection to the box.
6. Gaps and risks: credentials in files, things a console cannot see today, data lost after 14 days,
   anything that would break if a console read it naively.

### Part B — Proposal
Propose an architecture for the Admin Center with your reasoning. At minimum compare:
- **Cloud-first**: console reads S3 (+ a small index DB); boxes publish more to S3 (per-frame detections,
  a "reported state" doc) and pull a "desired state" doc (device-shadow pattern) for config + model versions.
- **Direct-to-box**: console talks to each box's API over Tailscale (extend `serve.py`).
- **Hybrid** (likely): history/training from cloud, live + "see what they see" on demand over Tailscale.

For your recommended option give: components, the data model (tables), the box-side changes (new files the
box must publish, the desired/reported state contract, how a box applies a model update safely with rollback),
the API surface, and a phased build order where phase 1 is usable in days. Keep the box's CPU budget in mind
(YOLO already saturates the N150). Flag privacy/consent requirements for staff viewing a customer's cameras.

Be concrete and terse. Tables over prose. Cite file:line for every factual claim in Part A.
