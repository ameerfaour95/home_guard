# Task 16 report: auto-enrolment from box registrations

- cloud/discovery.py: lists dataset_*/ (Delimiter), HEAD/GET registration.json with ETag dedup in raw_revisions; skips pools uca/smarthome/multi and HG_CLOUD_DISCOVERY_IGNORE; per-site savepoint, one bad site never stops the pass.
- Unknown site + registration -> Customer(name_source setup, consents, owner_phone) + Device(enrolled_by setup), aliases (owner, site, host), audit auto_enroll (staff NULL, "setup"). Heartbeat only -> "Site Name" customer, enrolled_by discovered, audit auto_discover.
- Known site: host/app_version updated; consents only from newer consent.recorded_utc (audit consent_update, field names only); a discovered site upgrades when its registration arrives; admin-named customers never renamed (PATCH name sets name_source admin).
- Contract 2d: DeviceSummary.needs_details/enrolled_by/app_version; openapi.json + demo regenerated (fleet/customers files).
- Migration 0009; loop "discovery" every 300 s (DISCOVERY_LOCK); manage discover-once; audit.record gained staff_name for non-staff actors.
- Tests: tests/cloud/test_discovery.py (13), test_contract extension, loops wiring. Full suite: 424 passed.
