# Task 7 report
Status: DONE. Tests: 20 passed (fleet_contract + cloud), incl. 9 new in tests/cloud/test_auth.py (RED first: import errors).
Added: cloud/auth.py, cloud/audit.py; real deps.py (get_session, current_staff 401, require_role 403); routes/auth.py (login/refresh/me); fleet router requires admin/support; app.py builds engine+sessionmaker on app.state (lazy), init_db=True runs migrations; manage.create-staff uses auth.hash_password; fixtures client, staff_factory.
Notes: lockout counts login_failed rows (detail.email, lowercased) in 15 min; locked attempts are audited as login_locked (not counted). Failure audit is committed before raising 401/429. Reuse of a rotated refresh token revokes all that staff's tokens and commits. Refresh rows carry a family id across rotations. Other routers still use the bare bearer dependency (not enforced) until their tasks.

## Fix round
Tests in tests/cloud/test_auth_hardening.py (all written first; 11 of 13 new auth tests were seen failing before the code).
7. Commit before response: login/refresh use `Depends(get_session, scope="function")` (FastAPI 0.142 supports it) and commit explicitly -> test_login_visible_to_a_new_session_and_committed_in_handler
8. Lockout race: `pg_advisory_xact_lock(hashtext(email))` before counting failures; failure row committed in the same transaction -> test_concurrent_wrong_attempts_cannot_beat_lockout
9. TOTP replay: staff.totp_last_counter; staff row FOR UPDATE; counter match across c-1,c,c+1, must be > last; counter stored only on full success -> test_totp_code_cannot_be_replayed, test_failed_password_does_not_burn_totp_counter
10. `refresh_reuse` audit row (detail.family) before revoke -> test_refresh_reuse_writes_audit_row
11. refresh_tokens.family_started_at; rotation copies it, expiry capped at start + refresh_ttl, rejected past the cap -> test_session_cap_family_started_at
12. All routers depend on current_staff (was bare bearer) -> test_every_v1_route_requires_token (walks every OpenAPI path; FastAPI 0.142 wraps routers so app.routes has no paths). docs/admin/openapi.json regenerated: unchanged.
13. Settings.from_env rejects JWT secret < 32 bytes (ValueError) -> test_jwt_secret_must_be_32_bytes
14. login_locked records staff_id when the email matches -> test_login_locked_records_staff_id
15. password > 1024 or totp > 16 chars: same 401, no hashing, still audited as login_failed; index ix_audit_log_action_ts -> test_oversized_credentials_rejected_without_hashing (index: test_keyset_indexes)
16. test_padded_uppercase_email_shares_lockout, test_refresh_for_disabled_staff_401, test_role_change_applies_to_issued_access_token (these passed against the existing behaviour; they pin it)
Suite: `cd /c/Users/ameer/Ameer/home_guard_admin && unset VIRTUAL_ENV; export UV_SYSTEM_CERTS=1; uv run --group cloud --system-certs pytest tests/fleet_contract tests/cloud -q`
Output: `41 passed in 27.60s`
