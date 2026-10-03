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
