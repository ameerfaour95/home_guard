# Beelink Collector Box Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `data_collection` pipeline run unattended on a Windows 11 Beelink box at a second house, with automatic upload to S3 and a heartbeat.

**Architecture:** A new package `home_guard_project/box/` wraps the unchanged collector and uploader. A config overlay switches the collector to headless mode; finished clips are moved to an outbox so the uploader never touches files the collector is writing; three Windows scheduled tasks run collector, upload and heartbeat.

**Tech Stack:** Python 3.12 (stdlib + boto3 + pyyaml, already pinned), bash (Git Bash) runners, one PowerShell setup script, `unittest` for tests.

**Spec:** `docs/superpowers/specs/2026-10-02-beelink-collector-box-design.md`

**Execution note:** executed inline by the plan author in the same session, at the founder's request not to be asked for further approvals. Test cases below are the behaviour contract; implementation bodies are in the commits.

## Global Constraints

- Do not change collector trigger, crop or dataset format; extend, do not replace (project hard rule 5).
- No secrets in tracked files or in the bundle: no `cameras.yaml`, no `api_key.env`, no `labeling/config.yaml`, no tracked script that contains an RTSP URL with credentials.
- `home_guard_project/*/config.py` and `config.yaml` are gitignored by repo policy; changes to them are made locally and shipped in the bundle, not committed.
- Shell scripts are bash; PowerShell only for Windows-only setup.
- Python: `from __future__ import annotations`, type hints, `logging` not `print` in library code.
- Tests use `unittest` and temp directories; run with `.venv/Scripts/python.exe -m unittest discover -s tests/box -v`.
- S3 prefix for a site is exactly `dataset_<site>`; `<site>` matches `^[a-z0-9_]+$`.
- Heartbeat key is exactly `<prefix>/_status/heartbeat.json` and contains no URLs or log text.

## File structure

| File | Responsibility |
|---|---|
| `home_guard_project/data_collection/config.py` (local, untracked) | add overlay merge via env `HOME_GUARD_CONFIG_OVERLAY` |
| `home_guard_project/box/__init__.py` | package marker |
| `home_guard_project/box/config.box.yaml` | headless overrides |
| `home_guard_project/box/boxconfig.py` | paths, `box.yaml` loader, `s3_prefix()` |
| `home_guard_project/box/outbox.py` | move finished clips live → outbox |
| `home_guard_project/box/heartbeat.py` | build and put status JSON |
| `home_guard_project/box/__main__.py` | CLI: `upload`, `heartbeat`, `status` |
| `home_guard_project/box/run_collector.sh` | headless restart loop + alive file |
| `home_guard_project/box/run_upload.sh`, `run_heartbeat.sh` | scheduled-task wrappers with logging |
| `home_guard_project/box/setup_box.ps1` | one-time Windows setup + scheduled tasks |
| `home_guard_project/box/make_bundle.py` | build `dist/home_guard_box.zip` |
| `home_guard_project/box/README.md` | founder's step-by-step guide |
| `tests/box/test_*.py` | unit tests |

---

### Task 1: Config overlay

**Files:** modify `home_guard_project/data_collection/config.py`; create `home_guard_project/box/config.box.yaml`, `tests/box/test_config_overlay.py`.

**Interfaces — produces:** env var `HOME_GUARD_CONFIG_OVERLAY` (path to YAML); `_deep_merge(base: dict, overlay: dict) -> dict` (nested dicts merge, other values replace, inputs not mutated).

- [ ] Write tests: (a) overlay sets `display.show_windows: false` and `vlm.enabled: false` → `cfg.SHOW_WINDOWS is False`, `cfg.RUN_VLM_ON_SAVED_CLIPS is False`, and an un-overridden sibling key (`display.show_plotted_boxes: true`) survives; (b) env unset → values from base file; (c) env set to a missing file → `FileNotFoundError`; (d) `_deep_merge` does not mutate its inputs.
- [ ] Run tests, confirm they fail (no overlay support).
- [ ] Implement `_deep_merge` and the overlay block in `load_config` right after `config.yaml` is read.
- [ ] Write `config.box.yaml`: `display.show_windows: false`, `display.show_plotted_boxes: false`, `vlm.enabled: false`, `random_clip.enabled: true`, `random_clip.interval_sec: 3600.0`, `random_clip.allow_person: false`.
- [ ] Add a test that loading the real `config.yaml` with the real `config.box.yaml` gives `SHOW_WINDOWS False`, `RUN_VLM_ON_SAVED_CLIPS False`, `RANDOM_CLIP_ENABLED True`.
- [ ] Run tests, confirm pass. Commit tracked files.

### Task 2: Box config and paths

**Files:** create `home_guard_project/box/__init__.py`, `boxconfig.py`, `tests/box/test_boxconfig.py`; add `home_guard_project/box/box.yaml` and `dataset_outbox/`, `dist/` to `.gitignore`.

**Interfaces — produces:**
- `PROJECT_ROOT`, `LIVE_DIR` (`<root>/dataset_multi`), `OUTBOX_DIR` (`<root>/dataset_outbox`), `LOG_DIR` (`<root>/logs`), `ALIVE_FILE` (`<root>/logs/collector.alive`), `BOX_YAML` (`<package>/box.yaml`)
- `@dataclass(frozen=True) BoxConfig(site: str, min_age_minutes: float)`
- `load_box_config(path: str = BOX_YAML) -> BoxConfig` — raises `BoxConfigError` when the file is missing, `site` is absent, or `site` does not match `^[a-z0-9_]+$`; `min_age_minutes` defaults to `10.0`.
- `s3_prefix(site: str) -> str` returning `f"dataset_{site}"`.

- [ ] Write tests: valid file → `BoxConfig("house2", 10.0)`; explicit `min_age_minutes: 3` → `3.0`; missing file, missing site, and `site: "House 2"` each raise `BoxConfigError`; `s3_prefix("house2") == "dataset_house2"`.
- [ ] Run, confirm fail. Implement. Run, confirm pass. Commit.

### Task 3: Outbox

**Files:** create `home_guard_project/box/outbox.py`, `tests/box/test_outbox.py`.

**Interfaces — produces:**
- `finished_clip_metas(live_dir: str, min_age_sec: float, now: float | None = None) -> list[str]` — `*.meta.json` under `live_dir/meta` with mtime ≤ `now - min_age_sec`, sorted.
- `move_clip(meta_path: str, live_dir: str, outbox_dir: str) -> int` — moves, preserving relative paths, every file of the clip from `clips/`, `vlm_crops/`, `responses/`, `yolo/images/`, `yolo/labels/` under the same `<camera>/<date>` folder whose name starts with `<stem>.` or `<stem>_f`; moves the meta last; returns files moved.
- `move_finished_clips(live_dir: str, outbox_dir: str, min_age_sec: float, now: float | None = None) -> tuple[int, int]` — `(clips_moved, files_moved)`; a clip whose move raises `OSError` is logged and skipped with its meta left in place.

- [ ] Write tests with a helper that creates a full clip (mp4, crop, response, 2 images, 2 labels, meta) and sets the meta mtime: (a) old clip → all 8 files exist at the same relative path in the outbox and none in live, returns `(1, 8)`; (b) recent clip → `(0, 0)` and files untouched; (c) stems `cam_100_trigger` (old) and `cam_1000_trigger` (recent) → only the first moves; (d) old clip with no crop file → moves 7 files; (e) empty or missing live dir → `(0, 0)`.
- [ ] Run, confirm fail. Implement with `os.replace`. Run, confirm pass. Commit.

### Task 4: Heartbeat

**Files:** create `home_guard_project/box/heartbeat.py`, `tests/box/test_heartbeat.py`.

**Interfaces — produces:**
- `collector_alive(alive_path: str, now: float | None = None, max_age_sec: float = 180.0) -> bool`
- `build_heartbeat(site: str, live_dir: str, outbox_dir: str, alive_path: str, now: float | None = None) -> dict` with keys `site`, `host`, `time_utc`, `collector_running`, `disk_free_gb`, `clips_live`, `clips_outbox`, `newest_clip_utc` (or `None`), `cameras` (`{camera: {"clips_waiting": int, "newest_clip_utc": str}}`).
- `put_heartbeat(payload: dict, bucket: str, prefix: str, s3_client=None) -> str` — puts JSON at `f"{prefix}/_status/heartbeat.json"` with `ContentType="application/json"`, returns the key.

- [ ] Write tests: alive file touched now → `True`, 10 minutes old → `False`, missing → `False`; two clips for `front` in live and one for `yard` in outbox → `clips_live 2`, `clips_outbox 1`, per-camera counts `front 2`, `yard 1`, `newest_clip_utc` equals the newest meta mtime in ISO UTC; no clips → `newest_clip_utc None` and `cameras {}`; `put_heartbeat` with a fake client records bucket, key `dataset_house2/_status/heartbeat.json`, content type, and a body that parses back to the payload; the serialized payload contains no `rtsp`.
- [ ] Run, confirm fail. Implement. Run, confirm pass. Commit.

### Task 5: CLI

**Files:** create `home_guard_project/box/__main__.py`, `tests/box/test_cli.py`.

**Interfaces — consumes:** `load_box_config`, `s3_prefix`, paths, `move_finished_clips`, `build_heartbeat`, `put_heartbeat`, `home_guard_project.s3_upload.config.load_config`, `home_guard_project.s3_upload.s3_upload.run`.
**Produces:** `python -m home_guard_project.box {upload,heartbeat,status}`; `run_upload(cfg: BoxConfig, uploader=..., s3_bucket: str, workers: int) -> tuple[int, int]`.

- `upload`: ensure outbox dir exists → `move_finished_clips(LIVE_DIR, OUTBOX_DIR, cfg.min_age_minutes * 60)` → if the outbox contains any file, call `run(dataset_dir=OUTBOX_DIR, bucket=..., prefix=s3_prefix(cfg.site), workers=..., skip_reencode=False, no_cleanup=True, allowed_labels=frozenset(), delete_local=True)` → put heartbeat.
- `heartbeat`: build and put.
- `status`: build and print JSON, no network.

- [ ] Write tests for `run_upload` with a fake uploader: old clip in live → uploader called once with `dataset_dir == outbox`, `prefix == "dataset_house2"`, `no_cleanup True`, `delete_local True`, `allowed_labels == frozenset()`; empty live and empty outbox → uploader not called.
- [ ] Run, confirm fail. Implement. Run, confirm pass. Run `python -m home_guard_project.box status` with a temporary `box.yaml` and confirm JSON prints. Commit.

### Task 6: Runner scripts

**Files:** create `home_guard_project/box/run_collector.sh`, `run_upload.sh`, `run_heartbeat.sh`.

- `run_collector.sh`: cd to project root; `mkdir -p logs`; export `PYTHONUNBUFFERED=1` and `HOME_GUARD_CONFIG_OVERLAY` (default `home_guard_project/box/config.box.yaml`, keep an existing value); loop: if `cameras.yaml` is missing, log one line and sleep 60; else start `uv run python -u home_guard_project/data_collection/data_collection.py` in the background appending to `logs/collector-$(date +%F).log`, touch `logs/collector.alive` every 60 s while it lives, log its exit code, sleep 15, restart. On TERM/INT kill the child and exit.
- `run_upload.sh` / `run_heartbeat.sh`: cd to root, run `uv run python -m home_guard_project.box upload|heartbeat` appending to `logs/upload-$(date +%F).log` / `logs/heartbeat.log`.

- [ ] Write the three scripts; `bash -n` each.
- [ ] Smoke test on the laptop: run `run_collector.sh` for about 60 s with an overlay whose `output_dir` points to a scratch folder; confirm the log shows the collector started without display windows or VLM load, and `logs/collector.alive` exists; stop it and confirm the child is gone.
- [ ] Commit.

### Task 7: Windows setup script

**Files:** create `home_guard_project/box/setup_box.ps1`.

Parameters: `-Site <name>` (required, `^[a-z0-9_]+$`), `-UploadTime "03:00"`. Requires an elevated shell. Idempotent. Steps, each reporting OK or the error:

1. `winget install --id Git.Git`, `astral-sh.uv`, `Gyan.FFmpeg`, `tailscale.tailscale` (each with `-e --accept-source-agreements --accept-package-agreements`), skipped when the command already exists.
2. Power: `powercfg /change standby-timeout-ac 0`, `hibernate-timeout-ac 0`, `disk-timeout-ac 0`; `powercfg /hibernate off`.
3. Remote Desktop: set `fDenyTSConnections` to 0 and enable the "Remote Desktop" firewall group; warn instead of failing on Windows Home.
4. OpenSSH server: add capability `OpenSSH.Server~~~~0.0.1.0`, start `sshd`, startup automatic, firewall rule for TCP 22.
5. Write `home_guard_project/box/box.yaml` with `site` and `min_age_minutes: 10`.
6. Register scheduled tasks, replacing existing ones: `HomeGuard-Collector` (at startup, 30 s delay; restart every 1 minute up to 999 times; no execution time limit; runs `bash.exe -lc "<root>/home_guard_project/box/run_collector.sh"`), `HomeGuard-Upload` (daily at `-UploadTime`), `HomeGuard-Heartbeat` (hourly). Principal: current user, `LogonType S4U`, `RunLevel Highest`. Settings: allow on batteries, start when available.
7. Print the remaining manual steps: BIOS "restore on AC power loss", `tailscale up`, AWS credentials, camera discovery.

- [ ] Write the script. Check it parses: `[System.Management.Automation.Language.Parser]::ParseFile(...)` reports zero errors.
- [ ] Commit. (First real run happens on the box.)

### Task 8: Bundle and guide

**Files:** create `home_guard_project/box/make_bundle.py`, `home_guard_project/box/README.md`, `tests/box/test_bundle.py`; update `.gitignore`.

**Interfaces — produces:** `bundle_files(root: str) -> list[str]` (relative paths, sorted) and `python -m home_guard_project.box.make_bundle` writing `dist/home_guard_box.zip`.

Include: `pyproject.toml`, `uv.lock`, `.python-version` if present, `yolo11s.pt` if present, everything under `home_guard_project/box/`, `home_guard_project/s3_upload/`, `home_guard_project/labeling/__init__.py` and `home_guard_project/labeling/utils/`, and from `home_guard_project/data_collection/`: `__init__.py`, `config.py`, `config.yaml`, `data_collection.py`, `discover.py`, `roi_editor.py`, `start.sh`. Exclude always: `cameras.yaml`, `zones.yaml`, `box.yaml`, `__pycache__`, `*.pyc`, `.pt` files other than the root model, and any file whose text matches `rtsp://<user>:<password>@` with a real-looking password.

- [ ] Write tests: `bundle_files` on the real repo contains `home_guard_project/data_collection/config.py` and `home_guard_project/box/run_collector.sh`; contains none of `cameras.yaml`, `zones.yaml`, `box.yaml`, `api_key.env`, `labeling/config.yaml`, `run_with_gpt.py`, `fortified_security_smolvlm.py`; no included text file matches the credential pattern.
- [ ] Run, confirm fail. Implement. Run, confirm pass.
- [ ] Build the bundle; list its contents and size.
- [ ] Write `README.md`: what the box does; first-boot checklist; copying and unpacking the bundle to `C:\home_guard`; running `setup_box.ps1`; AWS key with the exact IAM policy JSON; camera discovery over Remote Desktop with site-prefixed camera names; the bench test; moving the box; checking the heartbeat from the laptop; troubleshooting (logs, restarting tasks, changing YOLO cadence).
- [ ] Commit.

## Self-review

- Spec coverage: overlay (T1), box config (T2), outbox (T3), heartbeat (T4), CLI/upload decisions (T5), runner and error handling (T6), scheduled tasks and Windows setup (T7), bundle, guide and secret exclusion (T8). Bench test is a founder step documented in the README.
- Names are consistent across tasks: `HOME_GUARD_CONFIG_OVERLAY`, `BoxConfig`, `s3_prefix`, `move_finished_clips`, `build_heartbeat`, `put_heartbeat`, `ALIVE_FILE`.
