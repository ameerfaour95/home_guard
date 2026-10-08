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
#    4. Creates C:\ProgramData\HomeGuard (config, secrets, data, logs, models;
#       Administrators, SYSTEM and this account only) and writes box.yaml with the
#       site name in its config folder. A box already using the old places in the
#       code folder keeps them (move them with migrate_layout.ps1)
#    5. Installs the Python environment (uv sync)
#    6. Registers three scheduled tasks: collector (at boot), upload (every 15 minutes), heartbeat (hourly)
#    7. Windows signs in by itself at boot (enable_autologon.ps1, asks for the
#       password once), so the Home Guard window comes back after a power cut
#
#  Not done here (see README.md): BIOS power-on setting, Tailscale sign-in,
#  AWS credentials, camera discovery.
# ============================================================================
#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9_]+$')]
    [string]$Site,

    # What the box runs: data_collection (clips for tagging) or inference (alerts).
    # Left out: an existing box.yaml keeps its mode; a new one gets data_collection.
    [ValidateSet('data_collection', 'inference')]
    [string]$Mode,

    # Clips leave the box for S3 this often; local copies are deleted once S3 has them.
    [ValidateRange(5, 1440)]
    [int]$UploadEveryMinutes = 15,

    # Leave auto sign-in as it is (the alert program does not need it, only the window does).
    [switch]$SkipAutologon
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
Step '1/7 Tools'

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
Step '2/7 Power: never sleep'

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /hibernate off
# Fast Startup is a half-hibernate; off, so every start is a clean boot.
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power' -Name HiberbootEnabled -Value 0 -Type DWord
Ok 'Sleep, hibernate and Fast Startup disabled on AC power.'

# After a power cut Windows can stop at a recovery screen and wait for a keyboard. The box has none.
bcdedit /set '{current}' bootstatuspolicy ignoreallfailures | Out-Null
bcdedit /set '{current}' recoveryenabled no | Out-Null
Ok 'Windows will boot straight through after a power cut (no recovery screen).'

# ----------------------------------------------------------------------------
Step '3/7 Remote Desktop'

$edition = (Get-CimInstance Win32_OperatingSystem).Caption
if ($edition -match 'Home') {
    Warn "This is '$edition' - Remote Desktop is not available. Use SSH, or another remote-desktop tool for the camera preview and ROI editor."
} else {
    Set-ItemProperty 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name fDenyTSConnections -Value 0
    Get-NetFirewallRule -Group '@FirewallAPI.dll,-28752' -ErrorAction SilentlyContinue | Enable-NetFirewallRule
    Ok 'Remote Desktop enabled.'
}

# ----------------------------------------------------------------------------
Step '4/7 Box folder + box.yaml'

# Where the box keeps its config, secrets, data and logs (box_paths.ps1, the same rule as paths.py).
. (Join-Path $BoxDir 'box_paths.ps1')
$hg = Get-HomeGuardPaths -CodeDir $Root
$oldPlaces = (Test-Path (Join-Path $BoxDir 'box.yaml')) -or
             (Test-Path (Join-Path $Root 'home_guard_project\data_collection\cameras.yaml'))
if ($hg.Mode -eq 'legacy' -and -not $env:HOMEGUARD_HOME -and -not $oldPlaces) {
    # A new box: its files go to C:\ProgramData\HomeGuard from the start.
    $taskAccount = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    Initialize-HomeGuardHome (Get-HomeGuardDefaultHome) $taskAccount
    $hg = Get-HomeGuardPaths -CodeDir $Root
    Ok "Box folder: $($hg.Home) (Administrators, SYSTEM and $taskAccount only)"
} elseif ($hg.Mode -eq 'legacy') {
    Warn "This box keeps its files in the code folder (the old layout). Move them with: $BoxDir\migrate_layout.ps1"
} else {
    Ok "Box folder: $($hg.Home)"
}

# Replace "key: ..." in the lines of a YAML file, or append it. Other lines are kept as they are.
function Set-YamlValue([string[]]$lines, [string]$key, [string]$value) {
    $found = $false
    $out = @(foreach ($line in $lines) {
        if ($line -match "^\s*$key\s*:") { $found = $true; "${key}: $value" } else { $line }
    })
    if (-not $found) { $out += "${key}: $value" }
    return $out
}

$boxYaml = $hg.BoxYaml
if (Test-Path $boxYaml) {
    # Keep everything already in the file (alert settings and so on); only set what was asked.
    $lines = @(Get-Content $boxYaml)
    $lines = Set-YamlValue $lines 'site' "`"$Site`""
    if ($Mode) { $lines = Set-YamlValue $lines 'mode' $Mode }
} else {
    $newMode = if ($Mode) { $Mode } else { 'data_collection' }
    $lines = @(
        '# Per-box settings. Created by setup_box.ps1 - kept outside the code (box_paths.ps1).',
        "site: `"$Site`"",
        "mode: $newMode",
        'min_age_minutes: 10'
    )
}
$lines | Set-Content -Path $boxYaml -Encoding ascii
Ok "Wrote $boxYaml ($(($lines | Where-Object { $_ -match '^(site|mode)\s*:' }) -join ', '))"

# ----------------------------------------------------------------------------
Step '5/7 Python environment (first run downloads several GB)'

Push-Location $Root
try {
    uv sync --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed.' }
} finally {
    Pop-Location
}
Ok 'Python environment ready.'

# ----------------------------------------------------------------------------
Step '6/7 Scheduled tasks'

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

$uploadEvery = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes $UploadEveryMinutes)
Register-BoxTask 'HomeGuard-Upload' 'run_upload.sh' $uploadEvery 'Home Guard: upload finished clips to S3'

$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Hours 1)
Register-BoxTask 'HomeGuard-Heartbeat' 'run_heartbeat.sh' $hourly 'Home Guard: hourly status to S3'

Start-ScheduledTask -TaskName 'HomeGuard-Collector'
Ok 'Collector task started (it waits until cameras.yaml exists).'

# ----------------------------------------------------------------------------
Step 'Home Guard desktop + startup shortcut'
# Shared with the customer wizard (setup_customer.ps1) so both create the same
# "Home Guard" shortcut on the desktop and in Startup.
& (Join-Path $BoxDir 'make_shortcut.ps1')

# ----------------------------------------------------------------------------
Step '7/7 Auto sign-in (the Home Guard window comes back after a power cut)'
# The alert program needs nobody signed in; the window in Startup does.
$autoLogon = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon').AutoAdminLogon
if ($SkipAutologon) {
    Warn 'Auto sign-in left as it is (-SkipAutologon).'
} elseif ($autoLogon -eq '1') {
    Ok 'Auto sign-in already on.'
} else {
    try {
        & (Join-Path $BoxDir 'enable_autologon.ps1')
    } catch {
        Warn "Auto sign-in not set: $($_.Exception.Message)"
        Warn "Run at the box (or over ssh -t): powershell -ExecutionPolicy Bypass -File $BoxDir\enable_autologon.ps1"
    }
}

Write-Host ''
Write-Host 'Setup finished. Still to do by hand (details in box\README.md):' -ForegroundColor Green
Write-Host '  1. BIOS: set "State After G3" / "restore on AC power loss" to Power On (S0).'
Write-Host '  2. Tailscale: run  tailscale up  and approve the login link.'
Write-Host "  3. AWS: put the box's own access key in $env:USERPROFILE\.aws\credentials (API keys go in $($hg.SecretsEnv))."
Write-Host '  4. Cameras: run camera discovery once to create cameras.yaml.'
