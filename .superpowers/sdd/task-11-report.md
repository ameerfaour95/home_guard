# Task 11 report: audited media access and batched owner notices

Status: done. Suite: 249 passed (12 new in tests/cloud/test_media_routes.py).

- routes/media.py: POST /artifacts/{id}/access (404, 410, 403 rules, 300 s presign, mime by extension, media_view audit,
  owner notice for support/review purposes only) and GET /events/{id}/thumbnail (307, thumbnail_view audit batched 10 min,
  no notice; labelers 404 without training consent). Presigned URLs are never logged or audited.
- audit.py: record(ts=) param; owner_notice() (pg advisory xact lock keyed on staff/device/kind, 30 min window on last_ts,
  DB row decided before S3 write, S3 failure sets pending_upload); retry_pending_notices(); camera_label().
- Migration 0004 adds owner_notices.pending_upload (nullable bool); model updated.
- events.py not touched (the thumbnail stub lived in media.py; it imports _Viewer/_load_one from events).
- Decisions: purpose training requires consent_training for everyone (admin too); support/review need consent_recordings and
  are refused for labelers; message times are in the customer's timezone.
- Notice cleanup/retry is a function only; nothing schedules retry_pending_notices yet.
