# HomeGuardAdmin.exe — Round 5

Completed on `admin-console-ui` in `C:\Users\ameer\Ameer\home_guard_admin_ui`.

## Result

Merged `admin-console` first (`d97e0b8`). Fleet, customer timeline/event, Review, Studio, Audit, and the new Index problems view now run against the real local Cloud service. No Cloud files were edited after the merge. No push, SSH, or real AWS access was performed.

### Behaviour and polish

- Labeler Review and shell search fields are hidden. Ctrl+K and the Commands button still find commands and saved filters. Event and density requests omit `customer_id` and `q` for labelers, including a camera-command guard; the HTTP client also enforces the omission. Live responses use pseudonyms and customer ID zero without displaying “customer 0.”
- Null thumbnails render a designed **Preview not available** tile. Consent refusal is tested against real customer updates; the labeler screenshot also shows an event with no thumbnail alongside its permitted recording.
- Collection/export names are validated at 120 characters. Cloud's **Name must not identify a household** error appears inline. Labeler creation dialogs explain that their collections/exports are visible only to them and administrators. Lists use the server's visibility decisions.
- Delivery says **Sent to the owner on Telegram**, **Not delivered (reason)**, or **Delivery not recorded**. Expired video says **No copy of this video remains**.
- Camera labels use an optional `display_name`, otherwise humanise the raw name; `cam-…` pseudonyms remain intact. Raw names remain in tooltips. Camera density selections now split Cloud's `site/camera` rows into the separate request fields.
- Export previews warn that VLM includes clips. Export details fetch the current signed manifest URL in a worker and display schema v2 counts, warnings, and VLM file layout. They show **Download for training** with `manage export-download <id> --dest DIR`.
- Added a full-body customer PATCH client method, retaining required name, timezone, consents and notes. Live tests change recordings consent and restore it in `finally`.
- Audit has readable action choices and verb-based action labels, accurate staff ID/email filtering guidance, and the real saved detail JSON in its drawer. Removed duplicated UTC text.
- Support Studio loads saved filters without requesting forbidden collections/exports.
- Settings works before/after sign-in: validated server URL, persisted dark/light theme, and Open log folder. Server changes sign out; command-line server/theme override saved preferences for that launch.

## Local data and integration

`admin/dev_server.py` uses an embedded Postgres database under `build/devdb-r5` and moto on loopback. It seeds 96 synthetic events, six real built-in filter views, four collections, audit details, and two index problems. Detector confidence is populated for the low-confidence view.

Startup signs in through the actual API with FastAPI TestClient, creates `training_starter` through `POST /v1/studio/exports`, and invokes the real export builder synchronously against moto. There are no route overrides or fabricated ready states. Startup asserts the export reaches ready. Normal wizard exports use Cloud's background executor. On restart, synthetic export rows are reset because moto's objects are transient; an export is built afresh.

Exact seeded filter totals verified independently from fixtures: false alarm **19**, AI dismissed/person **0**, AI failed/fallback **32**, real but wrong **0**, low confidence **19**, paused **16**. Zero-match views are intentional. The starter collection has three members; two are eligible in the VLM-only preview and the ready manifest contains two clips and two VLM records. Empty validation/test partitions and small dataset warnings remain visible.

## Validation

Commands run with the existing worktree environment (`VIRTUAL_ENV` set to this worktree's `.venv`; `--no-sync` avoids replacing its installed Cloud/admin dependencies):

```powershell
uv run --no-sync --group admin pytest tests/admin -q
$env:HG_ADMIN_IT='1'
uv run --no-sync --group cloud --group admin pytest tests/admin/test_live_server.py -q -rx
powershell -NoProfile -ExecutionPolicy Bypass -File home_guard_project/admin/build_admin_exe.ps1 -SkipSync
dist/HomeGuardAdmin/HomeGuardAdmin.exe --demo --smoke-test
```

- Unit/offscreen suite: **357 passed, 8 skipped**, 21.89 seconds. The eight skips are opt-in live tests.
- Full live suite: **8 passed**, 50.12 seconds, including the rebuilt EXE. Covers fleet, timeline and density camera filtering, real media decode/boxes, review writes, labeler privacy, all Studio totals, collection opening, wizard preview/create/ready/manifest, Audit drawer, Index problems, inline household-name refusal, and customer PATCH/consent thumbnails.
- Packaged live result: **4 devices, 25 timeline rows, 36 detection frames, video frame decoded, review saved**, no unavailable-media fallback.
- Packaged demo smoke: **exit 0**.
- `git diff --check`: clean. No post-merge changes under `home_guard_project/cloud/`.
- One intermediate run missed the existing 2.8-second shutdown timing threshold under concurrent load (3.16 seconds); the unchanged test passed in subsequent complete runs. Earlier assertion failures from changed copy/manifest semantics were updated, and a preferences regression was fixed.

## Packaging

Built `dist/HomeGuardAdmin/HomeGuardAdmin.exe` successfully:

- Executable: **9,703,746 bytes**.
- Complete one-directory bundle: **192,433,482 bytes**.
- SHA-256: `942017C56BDC9A13AF5EFE2CFB2AE8957F75F48B5705297B23D70B61F1DAB051`.

Inspected the final executable's embedded PYZ archive: `dev_server`, `screenshots`, `screenshots_r2`, `screenshots_r3`, `make_demo_media`, and `make_demo_r3` are absent. Settings and Index problems are present. The binary and build logs remain local build outputs, outside source control.

Staff instructions: [INSTALL_ADMIN.md](INSTALL_ADMIN.md), including whole-folder/ZIP installation, first sign-in, `--server`, Settings, log locations, updates and training downloads.

## Screenshots

All ten images were visually inspected. The first five are the requested live screenshots; the remaining images give supporting evidence. The live images are 1920×1080 and come from HttpBackend against the dev server, not DemoBackend. Camera pixels are the existing synthetic fixtures stored in moto.

- [Studio](screenshots/r5-live-studio.png)
- [Export ready with manifest](screenshots/r5-live-export-ready.png)
- [Audit drawer](screenshots/r5-live-audit.png)
- [Labeler Review](screenshots/r5-live-labeler-review.png)
- [Fleet](screenshots/r5-live-fleet.png)
- [Index problems](screenshots/r5-live-index-problems.png)
- [Customer timeline](screenshots/r5-live-timeline.png)
- [Event](screenshots/r5-live-event.png)
- [Settings dark](screenshots/r5-settings-dark.png)
- [Settings light](screenshots/r5-settings-light.png)

## File map and limits

`settings.py`, `prefs.py`, `shell.py`, `__main__.py`: settings, persistence and client lifecycle. `index_problems.py`: new admin view. `http_backend.py`, `backend.py`, `models.py`: protocol/validation/full customer PATCH. `studio.py`, `export_wizard.py`, `collections.py`: export details, role handling and validation. Presentation changes are in `formatting.py`, `ai_record.py`, `event_view.py`, `timeline*.py`, `review.py`, and shared widgets. `dev_server.py` and `tests/admin/test_live_server.py` provide live seeding and coverage; `test_round5.py` covers new UI/client behaviour.

The EXE is unsigned. The existing future-phase Conversation/Config/customer Access placeholders are not new Cloud routes in this round; the live Audit page provides the implemented audit functionality. Customer PATCH is available in the client and verified live; this round does not add a customer-editing form. Actual training dataset downloads remain an administrator management command. No unresolved failures remain in the Round 5 test suites.
