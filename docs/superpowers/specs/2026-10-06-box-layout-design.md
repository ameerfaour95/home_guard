# Box layout (phase 1 of the repo split) — design

Date: 2026-10-06. Branch `box-layout`. Map of every path this touches: the layout map research of the same day (sections 1 and 5).

Owner decision (2026-10-06): the box's folders are messy and unprofessional. There will be a client repo, an admin repo and separate data. This phase only moves the box's per-machine config, secrets, data, logs and models out of the code folder, behind one module. It does not split repos or rename packages: module paths stay `home_guard_project.box...`, because the laptop-to-box SSH protocol depends on them.

## Layout

The code stays where it is, `C:\home_guard` (a git checkout; phase 2 moves it). Everything else:

| Place | Holds | Was |
|---|---|---|
| `HOME\config\` | box.yaml, cameras.yaml, camera_alerts.yaml, camera_aliases.yaml, zones.yaml, scene_maps.yaml, registration.json, registration.published, network.json | `box\` and `data_collection\` in the code; registration.published in `logs\` |
| `HOME\secrets\` | api_key.env | the repo root |
| `HOME\data\live\` | collector clips | `dataset_multi\` |
| `HOME\data\outbox\` | clips waiting for S3 | `dataset_outbox\` |
| `HOME\data\production\` | alert clips | `production_multi\` |
| `HOME\data\archive\` | answered alerts (14 days) | `production_archive\` |
| `HOME\data\state\` | house_state.jsonl, cases.jsonl, quiet_since.json, vision_budget.json, sees.json; `.conversations`, `.receipts`, `.desc`, `.live`, `.alert_embeddings.json` | `production_multi\.registry\` and `production_multi\.*` |
| `HOME\data\scene_interview\` | install interview pictures | `scene_interview\` |
| `HOME\logs\` | everything that was `logs\` | `logs\` |
| `HOME\models\` | yolo11s.pt, `yolo11s_openvino_model\`, FastSAM-s.pt | the repo root |

`HOME = %ProgramData%\HomeGuard`. Code defaults stay in the repo: `config.box.yaml`, `config.live.yaml`, `data_collection/config.yaml`, `s3_upload/config.yaml`.

ACLs (icacls): `HOME` without inherited entries, full control for Administrators (`*S-1-5-32-544`) and SYSTEM (`*S-1-5-18`), modify for the collector task's account and the auto sign-in account (the window). `secrets\` gets the same list again on its own, so a later change to `HOME` does not open it. The AWS key stays in the box account's `%USERPROFILE%\.aws\credentials` (boto3's default); phase 2 can move it to `secrets\aws_credentials` with `AWS_SHARED_CREDENTIALS_FILE`.

## Resolution rule

1. `HOMEGUARD_HOME`: a folder means that folder; `legacy` means the old places.
2. Else `%ProgramData%\HomeGuard`, when `layout.json` is in it.
3. Else legacy: every accessor gives today's place relative to the code folder.

The marker, not the folder, decides, so a rollback (which leaves the folder) is a switch back. One implementation per language, kept in step by tests:

- `home_guard_project/box/paths.py`: typed accessors (`config_dir()`, `box_yaml()`, `cameras_yaml()`, `secrets_env()`, `live_dir()`, `outbox_dir()`, `production_dir()`, `archive_dir()`, `state_dir()`, `assistant_dir()`, `logs_dir()`, `models_dir()`, `yolo_model()`, `resolve_model()`, ...), `mode()`, pure stdlib. `boxconfig`'s constants are aliases of it. `data_collection` imports it in script mode too.
- `_common.sh`: the same rule; exports `HOMEGUARD_HOME` (the folder, or `legacy`), so every Python program a runner starts agrees with it, and sets `LOG_DIR` and `CAMERAS_YAML`.
- `box_paths.ps1`: the same rule for setup_box.ps1, setup_network.ps1 and check_box.ps1 (works before the Python environment exists).
- The laptop asks a box `python -m home_guard_project.box paths --json`. No answer (old software) means the old layout. The install dir is one constant (`app/box_layout.py`, `$InstallDir` in setup_customer.ps1).

State that belongs to the production folder moves with `paths.state_paths_for(live_dir)`: for the production folder it gives `data\state`; for any other folder (tests, evals) it keeps the old `<live_dir>\.registry` and `<live_dir>`.

Models given by bare name (`yolo11s.pt`, `FastSAM-s.pt`) resolve to `models\`; ultralytics downloads a missing known model to the path it is given, and the OpenVINO copy is made next to the `.pt`.

## Migration and rollback

`python -m home_guard_project.box migrate-layout` (`migrate_layout.ps1` wraps it for the wizard):

- `--dry-run`: the plan (each item, file count, size, from, to). Changes nothing.
- run: check disk (copy size × 1.1 + 1 GB); pause `HomeGuard-Upload` and `HomeGuard-Heartbeat` (`schtasks /Change /DISABLE`, then wait while one is running; the heartbeat's self-heal would otherwise restart the collector mid-copy); `schtasks /End HomeGuard-Collector` and `stop_collector.sh` with `HOMEGUARD_HOME=legacy`; check the runner and collector pids are gone; lock `HOME` down before anything is copied in; plan again (the box may have saved or uploaded clips meanwhile) and copy with `copy2`, skipping files already there with the same size and time (a second run resumes); verify every file's size and config and secrets byte for byte; write `migration_manifest.json`, then `layout.json`; re-enable the tasks and `schtasks /Run` the collector. Any failure before the marker leaves the old layout and starts the box again as it was. Not carried: `*.winpid` and `collector.restart` (processes of the old layout). `collector.stop` is carried: a box stopped on purpose stays stopped.
- `--rollback`: stop the box the same way (with `HOMEGUARD_HOME=<HOME>`), copy back every file that changed or appeared in `HOME`, delete from the old places the copied files the box has since deleted (uploaded clips, expired answers), remove `layout.json`, start. `HOME` is left as it is.
- `--finalize`: only on a migrated box, only when git can list the tracked files. Lists the manifest's old copies that are still exactly as copied, plus `api_key.env.bak-*` and an empty `production_outbox`; deletes them only with `--yes`, then the folders left empty. Never a git-tracked file, nothing under `.git`, nothing outside the code folder. Works on a detached HEAD (the live box is detached at 12ea71e).
- After a run or a rollback the Home Guard window must be restarted (or the box rebooted): it resolves the places when it starts.
- Re-migrating after a rollback copies only what changed; files the old layout uploaded and deleted in between can still be in `HOME\data\outbox` and are uploaded again (same S3 keys).

A new box: `setup_box.ps1` creates `HOME` with its folders, ACLs and marker when the code folder has no box.yaml or cameras.yaml yet, and writes box.yaml into `config\`. A box on the old places keeps them and is told to run the migration.

## Phase 2 and 3 (outline)

- **Client repo `home-guard-box`**, installed in `C:\Program Files\HomeGuard`: `box/` (minus the admin and research files), the client half of `data_collection`, `s3_upload`, `labeling/utils/ffmpeg.py` and `cleanup.py`, and a client-only `pyproject.toml` (no label-studio, transformers, langchain; CPU or OpenVINO torch). The scheduled tasks and the desktop shortcut are registered again for the new code path; `update.sh` becomes a release update (tag or bundle), not `git pull` on whatever branch is checked out. Data stays in `HOME`, untouched by an update.
- **Admin repo `home-guard-admin`**: HomeGuardAdmin (Admin Center), the setup wizard (`setup_customer.ps1`, `build_exe.ps1`, the app's `--setup` pages), the gateway, and the fleet tools (`make_box_key`, `retention`, `make_bundle`, `prepare_remote`, `enable_remote`). `fleet_contract` becomes the shared contract package (heartbeat, registration, S3 keys, taxonomy) imported by both, ending the drifting copies.
- **Research stays in `home_guard`**: analysis, labeling, imports, eval (`eval_prompt`, `eval_translation`, `case_memory/evaluation`), training.
- **Data only in `home_guard_data` and S3**: no clips, datasets or per-machine files in any code repo.
- The module path `home_guard_project.box` changes only together with the laptop tools that call it over SSH, behind the `paths --json` style of asking the box.
