# Beelink collector box — design

Date: 2026-10-02
Status: approved by delegation (founder asked for the box to be set up and not to be asked further; OS choice confirmed as Windows 11)

## Goal

Turn a Beelink mini PC (Intel N150, 16 GB, 1 TB, Windows 11) into an unattended data-collection box for a second house. Once set up, plugging in power and Ethernet is enough: it connects to that house's cameras, saves trigger clips with the existing `data_collection` pipeline, and sends them to S3 for tagging.

This is a data-collection box, not the product. It reuses the v1 pipeline unchanged where possible.

## Non-goals

- No Ubuntu, Hailo, OpenVINO, WhatsApp or VLM on the box.
- No zero-touch credential discovery. Camera login is entered once per house.
- No change to the collector's trigger, crop or dataset format.
- No fix for the repo's gitignore policy on `config.py` / `config.yaml` (the bundle carries those files instead).

## What "plug in and it works" means

1. One-time preparation at the founder's house (Windows setup, `setup_box.ps1`, AWS key, bench test).
2. At the second house: power and Ethernet only.
3. One remote session (Tailscale + Remote Desktop) to run camera discovery and enter that house's camera login. This writes `cameras.yaml`.
4. From then on every boot starts collection without a login, and a crash restarts it.

## Components

All new code lives in `home_guard_project/box/`.

| Unit | Purpose | Depends on |
|---|---|---|
| `config.box.yaml` | Overrides for unattended running: no display windows, no VLM, hourly random clips | overlay support in `data_collection/config.py` |
| overlay in `data_collection/config.py` | If env `HOME_GUARD_CONFIG_OVERLAY` names a YAML file, deep-merge it over `config.yaml` | — |
| `run_collector.sh` | Runs the collector headless in a restart loop, logging to `logs/collector-YYYY-MM-DD.log`; waits if `cameras.yaml` is missing | uv, overlay |
| `outbox.py` | Moves finished clips (meta older than N minutes) with all their files from the live dataset to `dataset_outbox/` | — |
| `heartbeat.py` | Builds a status JSON (time, site, host, disk free, live and outbox clip counts, newest clip time per camera, collector running) and puts it at `<prefix>/_status/heartbeat.json` | boto3 |
| `__main__.py` | `python -m home_guard_project.box upload` (outbox move → existing `s3_upload.run` on the outbox with `delete_local`) and `... heartbeat` | `outbox`, `heartbeat`, `s3_upload` |
| `run_upload.sh` | Wrapper for the scheduled upload; logs to `logs/upload-YYYY-MM-DD.log` | — |
| `box.yaml` (gitignored) | Per-box settings: `site`, `min_age_minutes` | — |
| `setup_box.ps1` | One-time Windows setup: installs Git, uv, ffmpeg, Tailscale; power settings; Remote Desktop and OpenSSH; writes `box.yaml`; registers three scheduled tasks | Windows |
| `make_bundle.py` | Builds `dist/home_guard_box.zip` with only what the box needs | — |
| `README.md` | Step-by-step guide for the founder | — |

## Data flow

```
boot ─► task "HomeGuard-Collector" ─► run_collector.sh ─► data_collection.py (overlay) ─► dataset_multi/   (live)
03:00 ─► task "HomeGuard-Upload"   ─► box upload: move finished clips ─► dataset_outbox/ ─► s3_upload.run ─► s3://bucket/dataset_<site>/  (local copy deleted)
hourly ─► task "HomeGuard-Heartbeat" ─► box heartbeat ─► s3://bucket/dataset_<site>/_status/heartbeat.json
```

The outbox exists so that the uploader, which re-encodes clips in place and deletes local files, never touches a clip the collector is still writing. The collector writes a clip's meta JSON last, so "meta older than `min_age_minutes`" marks a finished clip.

## Decisions

- **S3 layout:** each site gets its own top-level dataset prefix, `dataset_<site>/`, using the uploader's existing `--prefix`. The labeling pipeline is pointed at it the same way it was for `dataset_uca` and `dataset_smarthome`. No code change in labeling.
- **Upload everything:** on the box the uploader runs with orphan cleanup off and no label filter. Cleanup deletes clips that lack a crop, and the label filter would hold back random background clips; both decisions belong at tagging time.
- **Random clips on:** one per hour per camera, without requiring a person. The labeled set has no "nothing happening" footage.
- **VLM off:** descriptions are written by humans; SmolVLM would only cost CPU on the N150.
- **Scheduled tasks run as the box user with S4U logon:** start at boot without anyone logging in and without storing the Windows password.
- **Code reaches the box as a zip bundle,** not `git clone`. The repo ignores the config loaders, and tracked scripts contain the founder's home camera credentials, which must not be copied into another house. The bundle is built from an explicit include list.
- **AWS key on the box is a dedicated IAM user** limited to `PutObject`/`ListBucket` on `dataset_<site>/*`, created by the founder. The founder's own keys never go on the box.
- **Heartbeat carries no URLs or log text,** so camera passwords cannot leak through it.

## Error handling

- Collector exits or crashes → loop restarts it after 15 s; the scheduled task also restarts on failure.
- `cameras.yaml` missing → runner logs a clear message and re-checks every 60 s, so discovery can be run later without a reboot.
- Camera or network drop → existing reconnect logic in the collector.
- Upload fails → files stay in the outbox; the next run retries. `delete_local` only removes files confirmed on S3 with matching size.
- Power cut → BIOS "restore on AC power loss" (manual setting) plus the at-startup task.

## Known limits

- N150 speed with YOLO11s on CPU across several cameras is unmeasured. The bench test reports it; the knobs are `detection.yolo_every_n_frames_cpu` and the model size in `config.box.yaml`.
- 1 TB holds roughly 200,000 clips at the observed ~4 MB per clip, so no disk guard is built.
- Clip IDs are `<camera>_<timestamp>`; two sites could reuse a camera name. Cameras on the box should be named with a site prefix at discovery time.
- The other household's footage is uploaded as full clips. The founder obtains their consent and chooses which cameras to include.

## Testing

- Unit tests (pytest, temp directories): outbox move rules, heartbeat payload, config overlay merge.
- Local smoke test on the laptop: `run_collector.sh` starts headless with the overlay.
- PowerShell script: parser check only; first real run is on the box.
- Bench test at the founder's house before moving the box: reboot, power pull, network pull, next-morning S3 check, heartbeat present, CPU and RAM noted.
