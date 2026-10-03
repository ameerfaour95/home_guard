# Admin Center phase 1 — progress ledger
Task 1: review: 1 Important open (heartbeat.json host/local_ip real) — fix queued after Task 2; minors: message_id real, conftest try/finally, smoke driver assert
Task 2: complete (commits 30dc424..ff2b957, review clean). Minors for final review: events limit unbounded (Task 10 sets default 100 max 500 — tell exe), PATCH customers needs full CustomerIn, DELETE collection items uses JSON body (exe must use request()), test_contract cwd-relative path, pytest pythonpath added to pyproject.
Task 1: complete (commits 7d06621..a5b5bef, review clean after fix)
Task 6: review — Important: TRUNCATE not blocked; fix queued after Task 7 (also: desc+id indexes, audit staff_name snapshot, manage init-db resolver/enroll side effect/ImportError, server_defaults)
DECISION: fleet sort order = offline, critical, warning, unknown, healthy (Task 9 brief must say this; exe uses it)
Tasks 3-5: complete (Codex, merged b3a61d6, review clean after fix a385d33)
Task 6: complete (8c6e62c + fixes ed0f7e4, review clean)
Task 7: complete (837d949 + fixes 7d00ab1 + 13cd4c2, review clean; race test proven to fail without lock)
Task 9: complete (6a8da04..74fb815, review clean). Minors for final review: customers list N+1, PATCH audit stores old/new values (decision said names only), case-sensitive sort, PATCH 404/no-op tests
Task 2b: complete (b263861, verified by controller: additive schema only)
Task 8: complete (13cd4c2..eccb29f; Codex review + rereview; 2 fix rounds; 228 passing)
Task 2c: complete (1e3aedc; controller-checked decisions: exclusion order consent>expired>video>no_real_ai)
Task 11: impl da4ec98; review: fix-11.md queued after fix-10 (systemic commit-before-response)
Task 12: complete (6b80967 + fix 6c98307; media tests 10/10; full-suite check pending privacy-fix completion; media-once CLI deferred to Task 14)
Task 11: complete (da4ec98 + fix d6f9ce5; systemic SessionDep scope=function guard test; 275 passing)
Task 10: complete (d83260c + privacy fixes 43ca89e,f2591fe,dceac30; 286 passing; final Codex adversarial pass running)
Task 14: complete (b887698 + fix 864083f, review clean)
Privacy round 3: complete (0190f35; fix-10c A-D)
Task 13: complete (204978e + fixes 263aa82,57595e0; 361 passing; manifest schema v2)
Task 15: complete (ad93f44; 366 passing)
Final fix round: complete (267f8a4..4b1db2e + report; 401 passing; migration 0007; later-list in final-fix-report.md)
FINAL: backend phase 1 complete at 6d0898a — 410 passing; Claude+Codex final reviews + 2 fix rounds; later-list in final-fix-report.md
