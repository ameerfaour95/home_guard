# Box registration review and Owner & consent wizard

Completed locally on `box-registration` in `C:\Users\ameer\Ameer\home_guard_registration`.

## Delivered

- Reviewed Claude's commit `07e417f`; findings and fixes are in [REGISTRATION_REVIEW.md](REGISTRATION_REVIEW.md).
- Added Owner & consent immediately after Home, before Cameras. Owner name requires 1–120 characters and no control characters; phone is optional. All three independent consents start off, using the requested sentences and support line verbatim.
- Installer name is saved atomically in the installer's local `.homeguard/installer.json`; owner details and permissions are not saved in local preferences. Demo mode neither reads nor writes that preference.
- Added keyboard labels, accessible control names, Space/Tab operation, clickable consent sentences, focused inline validation, and scroll fallback for small windows.
- AnswersFile contains `owner_name`, `owner_phone`, `installer`, `consent_live`, `consent_recordings`, and `consent_training`. The existing temporary-file engine transport is retained.
- Progress and readable details display `Adding the customer to Home Guard`; warnings remain visible and setup continues. Page names now keep camera retry, review, summary, and camera checking aligned with the added page.
- Fixed owner output leaks, consent coercion, stale customer data on reruns, atomic cleanup, destination-aware publication, interrupted site changes, and registration CLI failure codes.
- Built `dist/HomeGuardSetup.exe` locally using installed ps2exe 1.0.18. The generated binary is ignored by Git; source and build recipe are committed.

## Validation

- Regression tests were added and observed failing before the corresponding registration and wizard fixes.
- Final suite: **824 passed, 1 skipped, 91 subtests passed** (79.37 seconds); command `.venv/Scripts/python.exe -m pytest tests/box -q --ignore=tests/box/test_serve.py --tb=short`.
- Offscreen coverage includes name boundaries/control characters, Hebrew text, defaults off, navigation, keyboard toggles, actual engine AnswersFile payload and cleanup, owner redaction, installer preference atomicity, register events, and both screen sizes including validation.
- Fake S3 tests cover unchanged content, reordered keys, bucket/site changes, failure retries, and publication after site repair. No AWS requests were made.
- Windows PowerShell tests execute the actual answers/registration/helper blocks with mocked box commands, including old AnswersFiles, non-boolean consent, legacy private output, remote-cleanup failure, and stderr under ErrorActionPreference Stop.
- A compiled copy of the full engine passed old/new AnswersFile dry runs in the focused run (42 tests passed). A later run was blocked by Windows Application Control, WinError 4551; the test reports that environmental condition as a skip without bypassing policy.
- `test_serve.py` was tried separately and exited 1 without output, reproducing the behavior noted in Claude's report. It remains excluded from the broad suite; unrelated server code was not changed.
- `git diff --check` passed.

## Screenshots inspected

- [Owner, 1366×768](ui/screenshots/setup-owner-consent-1366x768.png)
- [Owner, 1920×1080](ui/screenshots/setup-owner-consent-1920x1080.png)
- [Validation, 1366×768](ui/screenshots/setup-owner-consent-validation-1366x768.png)
- [Validation, 1920×1080](ui/screenshots/setup-owner-consent-validation-1920x1080.png)
- [Registration progress, 1366×768](ui/screenshots/setup-owner-consent-progress-1366x768.png)

Fixed overflow at laptop size, aligned fields when validation appears, kept all consent text and the support line visible, expanded progress to eight stable rows, and removed its mismatched background. Screenshots contain no customer data.

No push, SSH to boxes, or AWS actions. Existing staged brief/report files and the untracked run log were preserved outside this commit. Cloud discovery and a real box installation were not exercised.
