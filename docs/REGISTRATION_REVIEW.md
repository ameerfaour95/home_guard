# Registration review — 07e417f

Reviewed registration.py, CLI/status/heartbeat, setup_customer.ps1, and wizard integration.
No remaining Critical/Important findings identified in the reviewed local paths.

Important findings fixed (regression tests written and observed failing first):
- Privacy: CLI summaries printed owner names, and setup copied that output into logs.
  Summaries omit owner details; setup suppresses legacy raw responses; errors omit payloads.
  Wizard output redacts the supplied owner name/phone; Answers repr omits both.
- Consent: PowerShell converted string "true" and integer 1 into consent.
  Only JSON booleans opt in; malformed saved consent is rejected instead of truth-coerced.
- Reruns: empty phone/installer and absent owner could retain the previous customer.
  JSON setup replaces details, clears blanks, defaults owner to site and permissions off.
- Atomicity: fixed-name .tmp files collided and leaked private data on replace failure.
  Unique same-directory temporary files, flush/fsync, replace, and finally cleanup now apply.
  The publish marker is also atomic and is written only after a successful PUT.
- Publication: the cache ignored bucket changes and depended on JSON key order.
  Its digest now covers bucket, key, and canonical content; failures remain retryable.
- Site changes: interrupted registration updates could keep publishing the old site.
  set-site and heartbeat reconcile the saved site before publishing; failed repair is retried.
  The old remote object remains, as specified; unchanged sites avoid rewriting the file.
- CLI: main discarded validation failure exit codes. Invalid registration now exits 1;
  publication failure still exits 0 and the setup step reports a retry warning.
- GUI protocol: register was unknown, so the following network step failed ordering checks.
  Parser, progress rows, details, demo, fixtures, and failed-step ownership now include it.

PowerShell 5.1 / ps2exe:
- SSH/SCP already use Invoke-Native, preserving Continue for native stderr and restoring Stop.
  Regression execution verifies stderr does not terminate the helper and exit codes survive.
- Registration uses UTF-8 JSON files and unique remote paths, never owner details in SSH args.
  Local cleanup runs even when remote cleanup throws; Python deletes input even on bad JSON.
- Old AnswersFiles and BOM input are covered; old/new compiled dry runs passed once.
  Later generated executables were blocked by Windows Application Control (environment skip).

Heartbeat/status expose only registration presence and consent flags, never owner fields.
Cloud PUTs use fakes; no SSH or AWS was exercised. Remote cleanup remains best effort offline.
Validation results, screenshots, and the pre-existing test_serve issue: CODEX_REGISTRATION_REPORT.md.
