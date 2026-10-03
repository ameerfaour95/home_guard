# Task 14 report
Implemented: GET /v1/audit (admin, keyset cursor base64 "<ts iso>:<id>", filters staff id/email, customer_id, action, page 50, detail included, staff = snapshot name -> email -> "deleted staff"), GET /v1/index/problems (admin, newest first, 500).
loops.py: indexer 120 s (full scan every 30 min), media 60 s (limit 50), notices 300 s; thread + own sessions per loop, jitter, exceptions logged, pg_try_advisory_lock per kind on a dedicated connection, stop <= 5 s. Lifespan: crash sweep, then loops when settings.run_loops (env HG_CLOUD_RUN_LOOPS) and s3 present.
Added app.create_app_from_env (uvicorn factory, boto3 S3). manage: media-once, real index-once (--full), serve uses the env factory. run_local.sh added (not executed against S3).
Test change: test_hardening_models index-once swallow case replaced by serve/app case (index-once no longer optional).
Suite: 317 passed.
