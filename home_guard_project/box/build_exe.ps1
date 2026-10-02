# ============================================================================
#  build_exe.ps1 - compile setup_customer.ps1 into dist\HomeGuardSetup.exe
#  Run on the founder's laptop (needs internet once to fetch the ps2exe module):
#      powershell -ExecutionPolicy Bypass -File build_exe.ps1 [-WithBundle]
#
#  -WithBundle also builds home_guard_box.zip and copies it next to the exe,
#  so the exe can install a box on a machine that does not have this repo.
# ============================================================================
param([switch]$WithBundle)
$ErrorActionPreference = 'Stop'

$BoxDir  = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $BoxDir '..\..')).Path
$DistDir = Join-Path $RepoRoot 'dist'
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

Write-Host 'Ensuring ps2exe module...' -ForegroundColor Cyan
if (-not (Get-Module -ListAvailable -Name ps2exe)) {
    try {
        # On a fresh machine Install-Module throws a null-reference error unless the
        # NuGet package provider is present first; install it non-interactively.
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        if (-not (Get-PackageProvider -Name NuGet -ErrorAction SilentlyContinue)) {
            Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Scope CurrentUser -Force | Out-Null
        }
        Install-Module -Name ps2exe -Scope CurrentUser -Force -AllowClobber
    } catch {
        throw "Could not install ps2exe. On a machine with no PSGallery access, run 'Install-Module ps2exe' manually, or build on a connected machine. ($($_.Exception.Message))"
    }
}
Import-Module ps2exe

if ($WithBundle) {
    # The bundle is the no-git fallback install path (see README). make_bundle
    # writes it straight into dist\, beside the exe this script produces.
    Write-Host 'Building home_guard_box.zip...' -ForegroundColor Cyan
    Push-Location $RepoRoot
    try { uv run python -m home_guard_project.box.make_bundle } finally { Pop-Location }
}

$src = Join-Path $BoxDir 'setup_customer.ps1'
$exe = Join-Path $DistDir 'HomeGuardSetup.exe'
$icon = Join-Path $BoxDir 'assets\home_guard.ico'
Write-Host "Compiling $src -> $exe" -ForegroundColor Cyan
$ps2exeArgs = @{
    inputFile  = $src
    outputFile = $exe
    title      = 'Home Guard Setup'
    product    = 'Home Guard'
    company    = 'Home Guard'
    noConsole  = $false
}
if (Test-Path $icon) { $ps2exeArgs.iconFile = $icon; Write-Host "  using icon $icon" -ForegroundColor DarkGray }
else { Write-Host "  (no assets\logo.ico found; building without a custom icon)" -ForegroundColor Yellow }
Invoke-ps2exe @ps2exeArgs

if (Test-Path $exe) {
    Write-Host "[OK] Built $exe" -ForegroundColor Green
    if ($WithBundle) {
        Write-Host "[OK] home_guard_box.zip is in $DistDir - ship it alongside the exe." -ForegroundColor Green
    } else {
        Write-Host "Note: run with -WithBundle to also produce home_guard_box.zip for machines without this repo." -ForegroundColor DarkGray
    }
} else {
    throw 'Build failed: HomeGuardSetup.exe was not produced.'
}
