# ============================================================================
#  setup_box.ps1 - one-time Windows setup for a collector box
#
#  Run on the box from an elevated PowerShell (locally or over SSH):
#      powershell -ExecutionPolicy Bypass -File C:\home_guard\home_guard_project\box\setup_box.ps1 -Site house2
#
#  What it does (safe to re-run):
#    1. Installs Git (for bash), uv, ffmpeg and Tailscale if missing
#    2. Stops the PC from sleeping or hibernating, and from waiting at a recovery screen after a power cut
#    3. Enables Remote Desktop (Windows Pro only)
#    4. Writes box.yaml with the site name
#    5. Installs the Python environment (uv sync)
#    6. Registers three scheduled tasks: collector (at boot), upload (nightly), heartbeat (hourly)
#
#  Not done here (see README.md): BIOS power-on setting, Tailscale sign-in,
#  AWS credentials, camera discovery.
# ============================================================================
#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9_]+$')]
    [string]$Site,

    [string]$UploadTime = '03:00'
)

$ErrorActionPreference = 'Stop'

$BoxDir = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $BoxDir '..\..')).Path
$GitBash = 'C:\Program Files\Git\bin\bash.exe'
$LocalBin = Join-Path $env:USERPROFILE '.local\bin'

function Step($msg) { Write-Host "`n-- $msg --" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }

function Test-Command($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Install-WithWinget($id) {
    if (-not (Test-Command 'winget')) { return $false }
    winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements | Out-Host
    return ($LASTEXITCODE -eq 0)
}

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
        [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $LocalBin
}

# ----------------------------------------------------------------------------
Step '1/6 Tools'

if (Test-Path $GitBash) {
    Ok 'Git Bash already installed.'
} elseif (Install-WithWinget 'Git.Git') {
    Ok 'Git installed with winget.'
} else {
    Warn 'winget unavailable - downloading Git for Windows directly.'
    $release = Invoke-RestMethod 'https://api.github.com/repos/git-for-windows/git/releases/latest'
    $asset = $release.assets | Where-Object { $_.name -match '^Git-.*-64-bit\.exe$' } | Select-Object -First 1
    $installer = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest $asset.browser_download_url -OutFile $installer
    Start-Process $installer -ArgumentList '/VERYSILENT', '/NORESTART' -Wait
    Ok 'Git installed.'
}
if (-not (Test-Path $GitBash)) { throw "Git Bash not found at $GitBash after install." }

Update-SessionPath
if (Test-Command 'uv') {
    Ok 'uv already installed.'
} else {
    Invoke-RestMethod 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    Update-SessionPath
    if (-not (Test-Command 'uv')) { throw 'uv install failed.' }
    Ok 'uv installed.'
}

if (Test-Command 'ffmpeg') {
    Ok 'ffmpeg already installed.'
} elseif (Install-WithWinget 'Gyan.FFmpeg') {
    Ok 'ffmpeg installed with winget.'
} else {
    Warn 'ffmpeg not installed (winget unavailable). Uploads need it to re-encode clips - install it before the first upload.'
}

if (Test-Path 'C:\Program Files\Tailscale\tailscale.exe') {
    Ok 'Tailscale already installed.'
} elseif (Install-WithWinget 'tailscale.tailscale') {
    Ok 'Tailscale installed with winget.'
} else {
    Warn 'winget unavailable - downloading Tailscale directly.'
    $msi = Join-Path $env:TEMP 'tailscale-setup.msi'
    Invoke-WebRequest 'https://pkgs.tailscale.com/stable/tailscale-setup-latest-amd64.msi' -OutFile $msi
    Start-Process msiexec.exe -ArgumentList '/i', "`"$msi`"", '/qn', '/norestart' -Wait
    Ok 'Tailscale installed.'
}
# Without unattended mode Tailscale disconnects when nobody is logged in to Windows.
& 'C:\Program Files\Tailscale\tailscale.exe' set --unattended=true 2>$null
if ($LASTEXITCODE -eq 0) { Ok 'Tailscale unattended mode on.' } else { Warn 'Could not set Tailscale unattended mode (sign in first, then re-run).' }

# ----------------------------------------------------------------------------
Step '2/6 Power: never sleep'

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /hibernate off
Ok 'Sleep and hibernate disabled on AC power.'

# After a power cut Windows can stop at a recovery screen and wait for a keyboard. The box has none.
bcdedit /set '{current}' bootstatuspolicy ignoreallfailures | Out-Null
bcdedit /set '{current}' recoveryenabled no | Out-Null
Ok 'Windows will boot straight through after a power cut (no recovery screen).'

# ----------------------------------------------------------------------------
Step '3/6 Remote Desktop'

$edition = (Get-CimInstance Win32_OperatingSystem).Caption
if ($edition -match 'Home') {
    Warn "This is '$edition' - Remote Desktop is not available. Use SSH, or another remote-desktop tool for the camera preview and ROI editor."
} else {
    Set-ItemProperty 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name fDenyTSConnections -Value 0
    Get-NetFirewallRule -Group '@FirewallAPI.dll,-28752' -ErrorAction SilentlyContinue | Enable-NetFirewallRule
    Ok 'Remote Desktop enabled.'
}

# ----------------------------------------------------------------------------
Step '4/6 box.yaml'

$boxYaml = Join-Path $BoxDir 'box.yaml'
@(
    '# Per-box settings. Created by setup_box.ps1 - not committed.',
    "site: `"$Site`"            # clips upload to s3://<bucket>/dataset_$Site/",
    'min_age_minutes: 10       # a clip is uploaded once its meta file is this old'
) | Set-Content -Path $boxYaml -Encoding ascii
Ok "Wrote $boxYaml (site: $Site)"

# ----------------------------------------------------------------------------
Step '5/6 Python environment (first run downloads several GB)'

Push-Location $Root
try {
    uv sync --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed.' }
} finally {
    Pop-Location
}
Ok 'Python environment ready.'

# ----------------------------------------------------------------------------
Step '6/6 Scheduled tasks'

# C:\home_guard -> /c/home_guard
$posixRoot = '/' + $Root.Substring(0, 1).ToLower() + $Root.Substring(2).Replace('\', '/')

function Register-BoxTask($name, $script, $trigger, $description) {
    $action = New-ScheduledTaskAction -Execute $GitBash `
        -Argument "-lc `"'$posixRoot/home_guard_project/box/$script'`"" -WorkingDirectory $Root
    # S4U: runs without anyone logged in and without storing the Windows password.
    # Take the account from the token: over SSH, $env:USERDOMAIN is not the computer name.
    $account = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $account -LogonType S4U -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Description $description -Force | Out-Null
    Ok "Task registered: $name"
}

$atBoot = New-ScheduledTaskTrigger -AtStartup
$atBoot.Delay = 'PT30S'
Register-BoxTask 'HomeGuard-Collector' 'run_collector.sh' $atBoot 'Home Guard: headless camera data collection'

$nightly = New-ScheduledTaskTrigger -Daily -At $UploadTime
Register-BoxTask 'HomeGuard-Upload' 'run_upload.sh' $nightly 'Home Guard: upload finished clips to S3'

$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Hours 1)
Register-BoxTask 'HomeGuard-Heartbeat' 'run_heartbeat.sh' $hourly 'Home Guard: hourly status to S3'

Start-ScheduledTask -TaskName 'HomeGuard-Collector'
Ok 'Collector task started (it waits until cameras.yaml exists).'

Write-Host ''
Write-Host 'Setup finished. Still to do by hand (details in box\README.md):' -ForegroundColor Green
Write-Host '  1. BIOS: set "restore on AC power loss" to Power On.'
Write-Host '  2. Tailscale: run  tailscale up  and approve the login link.'
Write-Host "  3. AWS: put the box's own access key in $env:USERPROFILE\.aws\credentials."
Write-Host '  4. Cameras: run camera discovery once to create cameras.yaml.'
