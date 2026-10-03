# Fleet contract Tasks 3–5

Worktree: `C:\Users\ameer\Ameer\home_guard_admin_contract`; branch: `admin-console-contract`.
The user's explicit worktree overrides the older path/branch in `constraints.md`.
No network, AWS, SSH, push, or other worktree operations were used.

## Test environment

PowerShell equivalent of the required shell setup, with networking disabled:

```powershell
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
$env:UV_SYSTEM_CERTS='1'
$env:UV_OFFLINE='1'
uv run --group cloud --system-certs python -m pytest tests/fleet_contract -q
```

The literal requested `... pytest tests/fleet_contract -q` command failed before
test collection: `Failed to spawn: pytest` / `An Application Control policy has
blocked this file. (os error 4551)`. Calling the same pytest through `python -m`
works. Dependencies were installed offline into this worktree's `.venv` from
the existing uv cache; no dependency declarations were changed.

## Task 3 — S3 keys and classes

Tests from the brief were written before implementation, plus path/filename edges.

RED excerpt:

```text
E   ModuleNotFoundError: No module named 'home_guard_project.fleet_contract.classes'
E   ModuleNotFoundError: No module named 'home_guard_project.fleet_contract.keys'
Interrupted: 2 errors during collection
2 errors in 0.15s
```

GREEN (entire contract suite):

```text
.............                                                            [100%]
13 passed in 0.14s
```

Files: `home_guard_project/fleet_contract/{keys,classes}.py`,
`tests/fleet_contract/test_{keys,classes}.py`, this report.

Edges: all 232 keys in the three fixture listings parse; underscore-containing
sites/cameras; `_general` feedback; YOLO's extra directory level; compound raw
response suffix; padded/unpadded frame suffixes; unknown areas; Windows drive,
UNC and parent traversal rejection; repeated `./` prefixes. `ext` is the final
suffix including its dot (e.g. `.json`, `.txt`). Unknown kind tokens stay unknown.

Contradictions: none between Task 3 rules and fixtures. The phrase "third
segment" for YOLO is interpreted after its two-segment `yolo/images` or
`yolo/labels` area, as required by the explicit example.

Task 3 commit: `5b0020f`.

## Task 4 — legacy metadata, feedback and heartbeat

The ten supplied tests and additional status/shape/path tests preceded the parser.

RED excerpt:

```text
E   ModuleNotFoundError: No module named 'home_guard_project.fleet_contract.legacy'
Interrupted: 1 error during collection
1 error in 0.18s
```

GREEN (entire contract suite):

```text
..................................................                       [100%]
50 passed in 0.12s
```

Files: `home_guard_project/fleet_contract/legacy.py`,
`tests/fleet_contract/test_legacy.py`, this report.

Edges: all supplied JSON fixtures; production copies without AI; NullBackend
fallback despite a configured model/prompt; teacher present with null response;
empty string responses; empty summary precedence; collection string responses,
`prompt_used` and raw-response paths; `owner_feedback` kind preservation;
malformed nested fields, non-finite numbers, bad frame sizes and top-level
non-object bodies; rejected paths in teacher inputs/raw outputs and sampled
frames; inputs are not mutated; feedback alert camera differs from action scope;
bad heartbeat values remain unknown; dates become timezone-aware UTC.

Contradictions/semantic caveats: the collection fixture's stem epoch
`1790944263` matches the integer part of `clip_end_ts`, as audit A7 describes,
not an independently recorded trigger timestamp. Per the brief, `trigger_ts`
is still taken from the stem, while `start_ts`/`end_ts` retain body epochs.
No fixture required changing an AI status or parsing rule. Metadata outside a
recognised meta key returns a record with `invalid meta key` in `problems`;
without a recognised root it uses `dataset` and an empty site as placeholders.

Task 4 commit: `f41ca85`.

## Task 5 — heartbeat health verdict

Tests for each rule, worst-wins behavior, human-readable reasons and threshold
edges preceded the implementation.

RED excerpt:

```text
E   ModuleNotFoundError: No module named 'home_guard_project.fleet_contract.health'
Interrupted: 1 error during collection
1 error in 0.16s
```

GREEN (entire contract suite):

```text
........................................................................ [ 91%]
.......                                                                  [100%]
79 passed in 0.11s
```

Files: `home_guard_project/fleet_contract/health.py`,
`tests/fleet_contract/test_health.py`, this report.

Edges: exactly 90 minutes is not offline; exactly 24 hours without a clip is
stale; missing camera clip dates are stale; 20 GB is warning and 50 GB healthy;
500 outbox clips is not a backlog warning; owner stop suppresses engine-down
only; disk critical beats owner-stop warning; offline wins while retaining all
other reasons; unknown collector state is not treated as false; future dates;
naive UTC and offset-aware timestamps; camera names appear in reasons.
Disk warnings/critical reasons both use `disk_low` with the applicable severity.
`alert_hours` is accepted with the exact required signature and does not alter
rules: the brief specifies no schedule-based exception. Missing optional
heartbeat values do not establish faults beyond the explicitly specified
missing-camera-date rule.

Contradictions/semantic caveats: no health fixture contradicts the required
thresholds. Audit A7 warns that `clips_outbox` includes retained archive metadata
and is not proof of unuploaded clips; the mandated `> 500` rule is implemented
unchanged. Camera recency likewise uses the heartbeat's metadata-mtime values,
not independently verified capture times.

## Final verification

- 79 tests pass: Task 3 contributes 13, Task 4 contributes 37, Task 5 contributes 29.
- AST audit of all five package files (including `__init__.py`) allows only
  standard-library absolute imports and internal relative imports.
- All contract modules import under Python `-I -S` with site-packages disabled.
- `git diff --check` passes; dependency files and fixtures are unchanged.
- Three task commits use the requested subjects and the exact
  `Co-Authored-By: Codex <noreply@openai.com>` trailer.
- The pre-existing untracked `docs/admin/codex_contract_run.log` was not edited
  or staged. The report is force-added despite the `*.md` ignore rule.
