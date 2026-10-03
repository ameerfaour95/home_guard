# Task 16 report: auto-enrolment from box registrations

- cloud/discovery.py: lists dataset_*/ (Delimiter), HEAD/GET registration.json with ETag dedup in raw_revisions; skips pools uca/smarthome/multi and HG_CLOUD_DISCOVERY_IGNORE; per-site savepoint, one bad site never stops the pass.
- Unknown site + registration -> Customer(name_source setup, consents, owner_phone) + Device(enrolled_by setup), aliases (owner, site, host), audit auto_enroll (staff NULL, "setup"). Heartbeat only -> "Site Name" customer, enrolled_by discovered, audit auto_discover.
- Known site: host/app_version updated; consents only from newer consent.recorded_utc (audit consent_update, field names only); a discovered site upgrades when its registration arrives; admin-named customers never renamed (PATCH name sets name_source admin).
- Contract 2d: DeviceSummary.needs_details/enrolled_by/app_version; openapi.json + demo regenerated (fleet/customers files).
- Migration 0009; loop "discovery" every 300 s (DISCOVERY_LOCK); manage discover-once; audit.record gained staff_name for non-staff actors.
- Tests: tests/cloud/test_discovery.py (13), test_contract extension, loops wiring. Full suite: 424 passed.

## Fix round (fix-16.md)

RED first: 16 of the new/changed tests failed before any code (only the rename-alias test #6 was already green, as expected).

- Critical #1 (consent = proposal): migration 0010 adds `customers.consent_proposed` (jsonb). Contract 2e: `ConsentProposal`, `CustomerOut.consent_proposed` (openapi regenerated, `test_contract_amendment_2e_consent_proposal`). Revocations apply and audit `consent_revoked_by_owner`; grants and all first-enrolment consents become the proposal (`consent_proposed`); stale answers are compared against max(consent_recorded_utc, proposal time). Admin PATCH on consents clears the proposal, stamps `consent_recorded_utc`, audits `consent_confirmed`. `needs_details` is also true while a proposal is pending. Future `recorded_utc` (> now + 5 min) is rejected with IndexProblem "registration time in the future", nothing applied. Tests: grants/revocations, pure revocation, future, PATCH confirm route, fleet needs_details.
- #2 GET storm: ETag already in raw_revisions -> no GET; unknown site reads the stored body. Tests count `get_text` calls.
- #3 stored raw_revisions body drops `owner_phone` (migration 0010 also scrubs existing rows). Test: stored body has no phone, customer keeps it.
- #4 admin-enrolled devices never change consents from a registration; a differing one is stored as a proposal only (tests incl. no auto-revocation).
- #5 audit actors `system:setup` / `system:discovery`; `create-staff` rejects names starting with `system:`.
- #6 test: renamed owner keeps the old name as identity term (already passed; pins behaviour).
- #7 `tests/cloud/test_migration_0010.py`: 0009 -> 0010 -> 0008 -> head, rows kept, no metadata drift.
- Demo regenerated (customer "Tamar Mizrahi" has a pending proposal; fleet needs_details true for her devices); demo comparison test green.
- Full suite after the round: 435 passed (test_final_ops customer PATCH equality gained the additive `consent_proposed: None`).
