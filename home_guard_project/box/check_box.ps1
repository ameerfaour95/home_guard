# ============================================================================
#  check_box.ps1 - readiness report for a collector box. Read-only.
#  Exit code 1 if anything essential is FAIL.
#      powershell -ExecutionPolicy Bypass -File check_box.ps1
# ============================================================================
$ErrorActionPreference = 'SilentlyContinue'
$script:fails = 0
function Pass($m) { Write-Host "[PASS] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[FAIL] $m" -ForegroundColor Red; $script:fails++ }

# --- Remote access ---
if ((Get-Service sshd).Status -eq 'Running') { Pass 'sshd running' } else { Fail 'sshd not running' }
if ((Get-Service Tailscale).Status -eq 'Running') { Pass 'Tailscale running' } else { Fail 'Tailscale not running' }
$ts = & 'C:\Program Files\Tailscale\tailscale.exe' debug prefs 2>$null | Out-String
if ($ts -match '"ForceDaemon":\s*true') { Pass 'Tailscale unattended on' } else { Warn 'Tailscale unattended OFF (run: tailscale set --unattended=true)' }

# --- Power / boot ---
$sb = powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Out-String
if ($sb -match 'AC Power Setting Index:\s*0x00000000') { Pass 'Sleep on AC disabled' } else { Warn 'Box may sleep on AC power' }
$boot = bcdedit /enum '{current}' | Out-String
if ($boot -match 'bootstatuspolicy\s+IgnoreAllFailures') { Pass 'No recovery screen on bad boot' } else { Warn 'Box may stop at a recovery screen after a power cut' }
$bl = manage-bde -status C: | Out-String
if ($bl -match 'Protection Status:\s*Protection On') { Warn 'BitLocker / device encryption is on: a BIOS change can stop the boot at a recovery-key screen' } else { Pass 'BitLocker off (no recovery-key screen at boot)' }
$wl = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
if ($wl.AutoAdminLogon -eq '1') { Pass "Auto sign-in on ($($wl.DefaultUserName)): the Home Guard window comes back after a power cut" }
else { Warn 'Auto sign-in off: alerts still run after a power cut, but the Home Guard window waits for a sign-in (run enable_autologon.ps1)' }

# --- Scheduled tasks ---
foreach ($t in 'HomeGuard-Collector', 'HomeGuard-Upload', 'HomeGuard-Heartbeat') {
    if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) { Pass "Task present: $t" } else { Fail "Task missing: $t" }
}

# --- Network config ---
$netJson = Join-Path $PSScriptRoot 'network.json'
if (Test-Path $netJson) {
    $n = Get-Content $netJson -Raw | ConvertFrom-Json
    Pass "network.json: mode=$($n.mode)"
    $profiles = netsh wlan show profiles | Out-String
    if ($n.mode -eq 'wifi') {
        if ($profiles -match [Regex]::Escape($n.wifi_ssid)) { Pass "Customer Wi-Fi profile present: $($n.wifi_ssid)" } else { Fail "Customer Wi-Fi profile missing: $($n.wifi_ssid)" }
    }
    if ($n.rescue_ssid -and ($profiles -match [Regex]::Escape($n.rescue_ssid))) { Pass "Rescue Wi-Fi profile present: $($n.rescue_ssid)" } else { Warn 'Rescue Wi-Fi profile missing' }
} else {
    Fail 'network.json missing (run the HomeGuardSetup wizard / setup_network.ps1)'
}

# --- Link + cameras ---
$link = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.InterfaceAlias -ne 'Tailscale' }
if ($link) { Pass "Network link up ($($link[0].InterfaceAlias))" } else { Warn 'No non-Tailscale default route (box may be offline to the LAN)' }
if (Test-Path 'C:\home_guard\home_guard_project\data_collection\cameras.yaml') { Pass 'cameras.yaml present' } else { Warn 'cameras.yaml missing (run camera discovery)' }

# --- AWS upload key ---
if (Test-Path (Join-Path $env:USERPROFILE '.aws\credentials')) { Pass 'AWS credentials present' } else { Warn 'AWS credentials missing (uploads fail; run make_box_key <site> and copy to .aws\credentials)' }

# --- Telegram assistant can hear the owner (inference mode) ---
$py = 'C:\home_guard\.venv\Scripts\python.exe'
if (Test-Path $py) {
    Push-Location 'C:\home_guard'
    $tg = (& $py -m home_guard_project.box.telegram_check 2>$null | Out-String).Trim()
    Pop-Location
    if ($tg -match '^READY' -or $tg -match '^SKIP') { Pass "Telegram: $tg" }
    elseif ($tg -ne '') { Warn "Telegram: $tg" }
    else { Warn 'Telegram readiness check could not run' }
} else {
    Warn 'Telegram readiness check skipped (.venv python not found)'
}

Write-Host ''
if ($script:fails -gt 0) { Write-Host "$script:fails essential check(s) FAILED" -ForegroundColor Red; exit 1 } else { Write-Host 'All essential checks passed' -ForegroundColor Green }
