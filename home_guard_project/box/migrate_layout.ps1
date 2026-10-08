# ============================================================================
#  migrate_layout.ps1 - move this box's config, secrets, data, logs and models
#  out of the code folder into C:\ProgramData\HomeGuard, or back
#
#  Run on the box from an elevated PowerShell (locally, over SSH, or from the
#  HomeGuardSetup wizard):
#      powershell -ExecutionPolicy Bypass -File C:\home_guard\home_guard_project\box\migrate_layout.ps1 -DryRun
#      powershell -ExecutionPolicy Bypass -File ...\migrate_layout.ps1              # stop, copy, check, switch, start
#      powershell -ExecutionPolicy Bypass -File ...\migrate_layout.ps1 -Rollback    # back to the old places
#      powershell -ExecutionPolicy Bypass -File ...\migrate_layout.ps1 -Finalize    # list the old copies
#      powershell -ExecutionPolicy Bypass -File ...\migrate_layout.ps1 -Finalize -Yes   # delete them
#
#  The work is done by: python -m home_guard_project.box migrate-layout
#  (layout_migration.py). Exit code 0 when it finished, 1 when it stopped with
#  the box left as it was.
# ============================================================================
#Requires -RunAsAdministrator
param(
    [switch]$DryRun,
    [switch]$Rollback,
    [switch]$Finalize,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw "No Python environment at $Python - run setup_box.ps1 first." }

$argv = @('-m', 'home_guard_project.box', 'migrate-layout')
if ($DryRun) { $argv += '--dry-run' }
if ($Rollback) { $argv += '--rollback' }
if ($Finalize) { $argv += '--finalize' }
if ($Yes) { $argv += '--yes' }

Push-Location $Root
try {
    # Python and the scripts it runs write progress to stderr; keep it as plain output.
    $ErrorActionPreference = 'Continue'
    & $Python @argv 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
