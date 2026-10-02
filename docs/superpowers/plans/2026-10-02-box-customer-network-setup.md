# Box customer network setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the founder prepare a screenless collector box for a customer with one laptop command that configures Ethernet or the customer's Wi-Fi, so the box joins the network by itself and rejoins after a Wi-Fi drop or power cut.

**Architecture:** New files in `home_guard_project/box/`. A PowerShell wizard on the laptop (`setup_customer.ps1`, compiled to `HomeGuardSetup.exe` with PS2EXE) collects answers and drives the box over Windows' built-in `ssh.exe`/`scp.exe` — no Git Bash needed on the laptop running the exe. A PowerShell script on the box writes all-users auto-connect Wi-Fi profiles (customer network + rescue hotspot) and `network.json`. A second PowerShell script reports readiness. Reconnection relies on built-in Windows/BIOS/Tailscale behaviour — no new daemon.

**Tech Stack:** PowerShell 5.1 (laptop wizard + box scripts), PS2EXE (compile the wizard), Windows OpenSSH client (`ssh.exe`/`scp.exe`), Python 3.12 + pytest (bundle test), `netsh wlan` for Wi-Fi profiles, existing Tailscale SSH access.

**Build output:** `dist/HomeGuardSetup.exe` — the double-clickable prep tool for the founder's laptop. Not committed (dist/ is gitignored); built by `build_exe.ps1`.

## Global Constraints

- Secrets never go in the repo, in `network.json`, or on the command line. Wi-Fi passwords travel only in a temp answers file that is deleted after use.
- The bundle (`make_bundle.py`) must never include `network.json` (same class as `box.yaml`).
- PowerShell is 5.1 on the box: no `&&`/`||`, no ternary, no `??`. ASCII-only script output (the box's console code page mangles non-ASCII), matching the existing box scripts.
- `.sh` files must use LF line endings (bash on the box rejects CRLF); `make_bundle.py` already normalises these.
- Wi-Fi: WPA2-Personal (`authentication=WPA2PSK`, `encryption=AES`). Password length 8–63 characters.
- Follow existing box conventions: `_common.sh` for shared bash setup, `Step`/`Ok`/`Warn` helpers in PowerShell, S4U tasks, `logs/` for logs.

---

## Task 1: Keep `network.json` out of the bundle

**Files:**
- Modify: `home_guard_project/box/make_bundle.py:47`
- Test: `tests/box/test_bundle.py`

**Interfaces:**
- Consumes: `bundle_files()` from `make_bundle.py` (existing).
- Produces: nothing new; tightens an existing guarantee.

- [ ] **Step 1: Add the failing assertions**

In `tests/box/test_bundle.py`, add `"network.json"` to the `MUST_EXCLUDE_NAMES` set, and extend `MUST_INCLUDE` with the scripts this plan adds so the bundle is proven to carry them:

```python
# in MUST_INCLUDE, add:
    "home_guard_project/box/setup_network.ps1",
    "home_guard_project/box/check_box.ps1",
# in MUST_EXCLUDE_NAMES, add:
    "network.json",
```

- [ ] **Step 2: Run the test, expect failure**

Run: `uv run python -m pytest tests/box/test_bundle.py -v`
Expected: `test_required_files_are_included` FAILS (the two new scripts do not exist yet). The exclude test passes already but is now guarded.

- [ ] **Step 3: Add `network.json` to the exclude list in code**

In `home_guard_project/box/make_bundle.py`, line 47, add `network.json`:

```python
EXCLUDE_NAMES = frozenset({"cameras.yaml", "zones.yaml", "box.yaml", "network.json", "api_key.env"})
```

- [ ] **Step 4: Leave the include failure for Task 2/3**

The `MUST_INCLUDE` additions stay red until the scripts exist. That is expected; do not create empty stubs just to pass. Commit the exclude change now.

- [ ] **Step 5: Commit**

```bash
git add home_guard_project/box/make_bundle.py tests/box/test_bundle.py
git commit -m "box: keep network.json out of the bundle; assert new setup scripts are bundled"
```

---

## Task 2: `setup_network.ps1` — apply the network choice on the box

**Files:**
- Create: `home_guard_project/box/setup_network.ps1`

**Interfaces:**
- Consumes: an answers file written by `setup_customer.sh` (Task 3) with `key=base64(value)` lines. Keys: `mode` (`ethernet`|`wifi`), `wifi_ssid`, `wifi_password`, `rescue_ssid`, `rescue_password`. Values are base64 of UTF-8.
- Produces: all-users auto-connect Wi-Fi profiles via `netsh wlan add profile`; `home_guard_project/box/network.json` with `{mode, wifi_ssid, rescue_ssid}` (no passwords). `-ForgetOtherWifi` switch deletes every saved Wi-Fi profile except the customer and rescue SSIDs.

- [ ] **Step 1: Write the script**

Create `home_guard_project/box/setup_network.ps1`:

```powershell
# ============================================================================
#  setup_network.ps1 - configure how a collector box reaches the network.
#
#  Run on the box from an elevated PowerShell (usually over SSH, driven by
#  setup_customer.sh on the laptop):
#      powershell -ExecutionPolicy Bypass -File setup_network.ps1 -AnswersFile C:\Users\ameer\net_answers.txt
#
#  The answers file has one "key=base64value" per line. Passwords never appear
#  on the command line or in network.json. The caller deletes the file after.
#
#  What it does:
#    wifi mode: installs an all-users, auto-connect Wi-Fi profile for the
#               customer's network plus a rescue-hotspot profile, so the box
#               joins at boot with nobody logged in and rejoins after a drop.
#    ethernet mode: installs only the rescue-hotspot profile.
#    Writes network.json (mode + SSID names, no passwords).
#    -ForgetOtherWifi also removes every other saved Wi-Fi network.
# ============================================================================
#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory = $true)][string]$AnswersFile,
    [switch]$ForgetOtherWifi
)
$ErrorActionPreference = 'Stop'

function Step($m) { Write-Host "`n-- $m --" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[OK]    $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN]  $m" -ForegroundColor Yellow }

# ---- read answers -----------------------------------------------------------
if (-not (Test-Path $AnswersFile)) { throw "Answers file not found: $AnswersFile" }
$answers = @{}
foreach ($line in Get-Content -LiteralPath $AnswersFile -Encoding UTF8) {
    if ($line -notmatch '=') { continue }
    $k = $line.Substring(0, $line.IndexOf('='))
    $v = $line.Substring($line.IndexOf('=') + 1)
    if ($v -eq '') { $answers[$k] = ''; continue }
    $answers[$k] = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($v))
}

$mode = $answers['mode']
if ($mode -ne 'ethernet' -and $mode -ne 'wifi') { throw "mode must be 'ethernet' or 'wifi' (got '$mode')" }

$BoxDir = $PSScriptRoot

function Test-WifiAdapter {
    [bool](Get-NetAdapter -Physical -ErrorAction SilentlyContinue |
        Where-Object { $_.InterfaceDescription -match 'Wi-?Fi|Wireless|802\.11' })
}

# Build a WPA2-Personal profile XML and install it for all users, auto-connect.
function Install-WifiProfile($ssid, $password, $label) {
    if ($ssid -eq '')       { throw "$label SSID is empty" }
    if ($password.Length -lt 8 -or $password.Length -gt 63) {
        throw "$label password must be 8-63 characters (got $($password.Length))"
    }
    # SSID can contain XML-special characters; hex-encode it for <hex>, and
    # XML-escape the key material.
    $ssidHex = -join ([Text.Encoding]::UTF8.GetBytes($ssid) | ForEach-Object { '{0:X2}' -f $_ })
    $keyXml  = [Security.SecurityElement]::Escape($password)
    $nameXml = [Security.SecurityElement]::Escape($ssid)
    $xml = @"
<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>$nameXml</name>
  <SSIDConfig><SSID><hex>$ssidHex</hex><name>$nameXml</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>auto</connectionMode>
  <MSM><security>
    <authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
    <sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>$keyXml</keyMaterial></sharedKey>
  </security></MSM>
</WLANProfile>
"@
    $tmp = Join-Path $env:TEMP ("wlan_" + [IO.Path]::GetRandomFileName() + ".xml")
    try {
        # ASCII-safe file: XML entities cover non-ASCII SSIDs.
        [IO.File]::WriteAllText($tmp, $xml, [Text.Encoding]::UTF8)
        netsh wlan add profile filename="$tmp" user=all | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "netsh failed to add the $label profile for '$ssid'" }
        Ok "$label Wi-Fi profile installed: $ssid"
    } finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

$rescueSsid = $answers['rescue_ssid']
$wifiSsid   = $answers['wifi_ssid']

Step '1/3 Wi-Fi profiles'
if ($mode -eq 'wifi') {
    if (-not (Test-WifiAdapter)) { throw "Wi-Fi mode requested but this box has no Wi-Fi adapter." }
    Install-WifiProfile $wifiSsid $answers['wifi_password'] 'Customer'
} else {
    Ok 'Ethernet mode: no customer Wi-Fi profile needed.'
}
# Rescue hotspot profile goes on every box that has a Wi-Fi radio.
if (Test-WifiAdapter) {
    if ($rescueSsid -ne '') {
        Install-WifiProfile $rescueSsid $answers['rescue_password'] 'Rescue'
    } else {
        Warn 'No rescue SSID supplied; skipping rescue profile.'
    }
    # Prefer the customer network over the rescue hotspot when both are present.
    if ($mode -eq 'wifi' -and $wifiSsid -ne '') {
        netsh wlan set profileorder name="$wifiSsid" interface="Wi-Fi" priority=1 | Out-Null
    }
    # Stop the Wi-Fi radio from being powered down to save energy.
    try {
        Get-NetAdapter -Physical | Where-Object { $_.InterfaceDescription -match 'Wi-?Fi|Wireless|802\.11' } |
            ForEach-Object { Disable-NetAdapterPowerManagement -Name $_.Name -ErrorAction SilentlyContinue }
        Ok 'Wi-Fi power saving disabled.'
    } catch { Warn "Could not change Wi-Fi power settings: $($_.Exception.Message)" }
} else {
    Warn 'No Wi-Fi adapter: rescue hotspot not available on this box.'
}

Step '2/3 Forget other Wi-Fi networks'
if ($ForgetOtherWifi) {
    $keep = @($wifiSsid, $rescueSsid) | Where-Object { $_ -ne '' }
    $current = (netsh wlan show interfaces | Select-String 'SSID\s+:\s+(.+)$').Matches |
        ForEach-Object { $_.Groups[1].Value.Trim() } | Select-Object -First 1
    $profiles = (netsh wlan show profiles | Select-String 'All User Profile\s+:\s+(.+)$').Matches |
        ForEach-Object { $_.Groups[1].Value.Trim() }
    # Delete the currently-connected one last so earlier deletes finish before the link drops.
    $ordered = @($profiles | Where-Object { $_ -ne $current }) + @($profiles | Where-Object { $_ -eq $current })
    foreach ($p in $ordered) {
        if ($keep -contains $p) { continue }
        netsh wlan delete profile name="$p" | Out-Null
        Ok "Forgot Wi-Fi network: $p"
    }
} else {
    Ok 'Left other saved Wi-Fi networks in place (pass -ForgetOtherWifi before delivery).'
}

Step '3/3 network.json'
$net = [ordered]@{ mode = $mode; wifi_ssid = $wifiSsid; rescue_ssid = $rescueSsid }
$net | ConvertTo-Json | Set-Content -Path (Join-Path $BoxDir 'network.json') -Encoding ascii
Ok "Wrote network.json (mode: $mode)"
Write-Host ''
Ok 'Network setup finished.'
```

- [ ] **Step 2: Parse-check locally**

Run: `powershell.exe -NoProfile -Command "$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'home_guard_project/box/setup_network.ps1').Path,[ref]$null,[ref]$e); if($e.Count){$e|%{$_.ToString()};exit 1}else{'parse OK'}"`
Expected: `parse OK`

(If the permission system blocks the inline parser call, parse-check on the box in Step 4 instead.)

- [ ] **Step 3: Commit**

```bash
git add home_guard_project/box/setup_network.ps1
git commit -m "box: add setup_network.ps1 (customer + rescue Wi-Fi profiles, network.json)"
```

- [ ] **Step 4: Verify on the real box (Wi-Fi profile round-trip)**

Write a throwaway answers file locally with a made-up SSID and a dummy 8+ char password, scp it and the script to the box, run the script, confirm the profile is all-users + auto, then delete the profile and the files:

```bash
printf 'mode=%s\nwifi_ssid=%s\nwifi_password=%s\nrescue_ssid=%s\nrescue_password=%s\n' \
  wifi "$(printf hg_test_net|base64)" "$(printf testpass123|base64)" \
  "$(printf hg_rescue|base64)" "$(printf rescuepass123|base64)" > /tmp/net_answers.txt
scp -i ~/.ssh/homeguard_box home_guard_project/box/setup_network.ps1 /tmp/net_answers.txt ameer@100.121.29.9:C:/Users/ameer/
ssh -i ~/.ssh/homeguard_box ameer@100.121.29.9 "powershell -ExecutionPolicy Bypass -File C:\\Users\\ameer\\setup_network.ps1 -AnswersFile C:\\Users\\ameer\\net_answers.txt"
ssh -i ~/.ssh/homeguard_box ameer@100.121.29.9 "netsh wlan show profiles & del C:\\Users\\ameer\\net_answers.txt"
# cleanup the test profiles:
ssh -i ~/.ssh/homeguard_box ameer@100.121.29.9 "netsh wlan delete profile name=hg_test_net & netsh wlan delete profile name=hg_rescue"
```
Expected: both `hg_test_net` and `hg_rescue` appear under "All User Profile" before cleanup.

---

## Task 3: `setup_customer.sh` — the laptop wizard

**Files:**
- Create: `home_guard_project/box/setup_customer.sh`

**Interfaces:**
- Consumes: `make_bundle` (via `uv run python -m home_guard_project.box.make_bundle`), `setup_box.ps1`, `setup_network.ps1`, `check_box.ps1` on the box; the rescue credentials file `~/.homeguard/rescue_wifi` (created on first run).
- Produces: a configured box; prints the manual steps and the rescue hotspot name/password.

- [ ] **Step 1: Write the script**

Create `home_guard_project/box/setup_customer.sh`:

```bash
#!/usr/bin/env bash
# ============================================================================
#  setup_customer.sh - prepare a collector box for a customer, from the laptop.
#
#  Usage:
#    ./home_guard_project/box/setup_customer.sh <user>@<box-tailscale-ip>
#      [--network-only]       skip the code install, just (re)configure the network
#      [--forget-other-wifi]  also remove Wi-Fi networks other than the customer's
#      [--key <path>]         SSH key (default ~/.ssh/homeguard_box)
#
#  Asks: site name, Ethernet or Wi-Fi, and (for Wi-Fi) the customer's network
#  name and password. Passwords travel in a temp file, never on the command
#  line, and are deleted from the box afterwards.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

KEY="$HOME/.ssh/homeguard_box"
NETWORK_ONLY=0
FORGET_OTHER=0
TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --network-only) NETWORK_ONLY=1; shift ;;
        --forget-other-wifi) FORGET_OTHER=1; shift ;;
        --key) KEY="$2"; shift 2 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) TARGET="$1"; shift ;;
    esac
done
[[ -n "$TARGET" ]] || { echo "Usage: setup_customer.sh <user>@<box-ip> [--network-only] [--forget-other-wifi] [--key PATH]" >&2; exit 2; }

SSH=(ssh -i "$KEY" -o ConnectTimeout=15 "$TARGET")
b64() { printf '%s' "$1" | base64 | tr -d '\n'; }

ask() { local prompt="$1" var; read -r -p "$prompt" var; printf '%s' "$var"; }
ask_secret() { local prompt="$1" var; read -r -s -p "$prompt" var; echo >&2; printf '%s' "$var"; }

# ---- rescue hotspot credentials (same for all of this founder's boxes) ------
RESCUE_FILE="$HOME/.homeguard/rescue_wifi"
if [[ ! -f "$RESCUE_FILE" ]]; then
    mkdir -p "$HOME/.homeguard"; chmod 700 "$HOME/.homeguard"
    r_ssid="homeguard-rescue"
    r_pass="$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 16)"
    printf 'ssid=%s\npassword=%s\n' "$r_ssid" "$r_pass" > "$RESCUE_FILE"
    chmod 600 "$RESCUE_FILE"
    echo "Created rescue hotspot credentials at $RESCUE_FILE"
fi
RESCUE_SSID="$(sed -n 's/^ssid=//p' "$RESCUE_FILE")"
RESCUE_PASS="$(sed -n 's/^password=//p' "$RESCUE_FILE")"

# ---- questions --------------------------------------------------------------
if [[ "$FORGET_OTHER" -eq 1 && "$NETWORK_ONLY" -eq 0 ]]; then : ; fi

SITE=""
if [[ "$NETWORK_ONLY" -eq 0 ]]; then
    while [[ ! "$SITE" =~ ^[a-z0-9_]+$ ]]; do
        SITE="$(ask 'Site name (lowercase letters, digits, underscores): ')"
    done
fi

MODE=""
while [[ "$MODE" != "ethernet" && "$MODE" != "wifi" ]]; do
    echo "How does this box reach the internet?"
    echo "  1) Ethernet cable"
    echo "  2) Wi-Fi"
    case "$(ask 'Choose 1 or 2: ')" in
        1) MODE="ethernet" ;;
        2) MODE="wifi" ;;
    esac
done

WIFI_SSID=""; WIFI_PASS=""
if [[ "$MODE" == "wifi" ]]; then
    while [[ -z "$WIFI_SSID" ]]; do WIFI_SSID="$(ask "Customer Wi-Fi name (SSID): ")"; done
    while (( ${#WIFI_PASS} < 8 || ${#WIFI_PASS} > 63 )); do
        WIFI_PASS="$(ask_secret 'Customer Wi-Fi password (8-63 chars): ')"
    done
fi

# ---- install code (unless --network-only) -----------------------------------
if [[ "$NETWORK_ONLY" -eq 0 ]]; then
    echo "Building bundle..."
    ( cd "$PROJECT_ROOT" && uv run python -m home_guard_project.box.make_bundle )
    echo "Copying bundle to the box..."
    scp -i "$KEY" "$PROJECT_ROOT/dist/home_guard_box.zip" "$TARGET:C:/home_guard_box.zip"
    echo "Stopping collector, unpacking, running setup_box.ps1..."
    "${SSH[@]}" "\"C:\\Program Files\\Git\\bin\\bash.exe\" -lc /c/home_guard/home_guard_project/box/stop_collector.sh" || true
    "${SSH[@]}" "powershell -Command \"Expand-Archive -Force C:\\home_guard_box.zip C:\\home_guard\""
    "${SSH[@]}" "powershell -ExecutionPolicy Bypass -File C:\\home_guard\\home_guard_project\\box\\setup_box.ps1 -Site $SITE"
fi

# ---- send network answers and apply -----------------------------------------
ANSWERS="$(mktemp)"
trap 'rm -f "$ANSWERS"' EXIT
{
    printf 'mode=%s\n' "$(b64 "$MODE")"
    printf 'wifi_ssid=%s\n' "$(b64 "$WIFI_SSID")"
    printf 'wifi_password=%s\n' "$(b64 "$WIFI_PASS")"
    printf 'rescue_ssid=%s\n' "$(b64 "$RESCUE_SSID")"
    printf 'rescue_password=%s\n' "$(b64 "$RESCUE_PASS")"
} > "$ANSWERS"
# mode is base64 too, so decode on read matches the box script.

REMOTE_ANSWERS="C:/Users/ameer/net_answers.txt"
scp -i "$KEY" "$ANSWERS" "$TARGET:$REMOTE_ANSWERS"
FORGET_ARG=""; [[ "$FORGET_OTHER" -eq 1 ]] && FORGET_ARG=" -ForgetOtherWifi"
"${SSH[@]}" "powershell -ExecutionPolicy Bypass -File C:\\home_guard\\home_guard_project\\box\\setup_network.ps1 -AnswersFile C:\\Users\\ameer\\net_answers.txt$FORGET_ARG"
"${SSH[@]}" "del C:\\Users\\ameer\\net_answers.txt" || true

# ---- readiness report -------------------------------------------------------
echo ""
echo "===== Readiness ====="
"${SSH[@]}" "powershell -ExecutionPolicy Bypass -File C:\\home_guard\\home_guard_project\\box\\check_box.ps1" || true

cat <<EOF

===== Still to do by hand =====
 1. BIOS: set "restore on AC power loss" to Power On (needs a screen once).
 2. Tailscale admin console: disable key expiry for this box.
 3. Camera discovery at the customer's house (writes cameras.yaml).
 4. Before delivery, re-run with --forget-other-wifi to drop your own Wi-Fi.

Rescue hotspot (make a phone hotspot with these to recover a box):
   name:     $RESCUE_SSID
   password: $RESCUE_PASS
EOF
```

**Note on `mode`:** the box script base64-decodes every value, so the wizard base64-encodes `mode` too (done above). Keep both sides consistent.

- [ ] **Step 2: Syntax-check**

Run: `bash -n home_guard_project/box/setup_customer.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add home_guard_project/box/setup_customer.sh
git commit -m "box: add setup_customer.sh laptop wizard for per-customer network setup"
```

- [ ] **Step 4: Live run against the box (`--network-only`, Ethernet)**

With the box reachable, run the wizard in network-only mode, answer Ethernet, and confirm it writes `network.json` and prints the readiness report and rescue credentials. (Choosing Ethernet avoids touching the box's live Wi-Fi connection during the test.)

Run: `./home_guard_project/box/setup_customer.sh ameer@100.121.29.9 --network-only`
Expected: `network.json` written with `"mode": "ethernet"`; rescue profile installed; readiness report prints.

---

## Task 4: `check_box.ps1` — readiness report

**Files:**
- Create: `home_guard_project/box/check_box.ps1`

**Interfaces:**
- Consumes: box state only (services, tasks, power config, profiles, `network.json`, `cameras.yaml`).
- Produces: stdout PASS/WARN/FAIL lines; exit code 1 if any FAIL.

- [ ] **Step 1: Write the script**

Create `home_guard_project/box/check_box.ps1`:

```powershell
# ============================================================================
#  check_box.ps1 - readiness report for a collector box. Read-only.
#  Exit code 1 if anything essential is FAIL.
# ============================================================================
$ErrorActionPreference = 'SilentlyContinue'
$fails = 0
function Pass($m) { Write-Host "[PASS] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[FAIL] $m" -ForegroundColor Red; $script:fails++ }

# Remote access
if ((Get-Service sshd).Status -eq 'Running') { Pass 'sshd running' } else { Fail 'sshd not running' }
if ((Get-Service Tailscale).Status -eq 'Running') { Pass 'Tailscale running' } else { Fail 'Tailscale not running' }
$ts = & 'C:\Program Files\Tailscale\tailscale.exe' debug prefs 2>$null | Out-String
if ($ts -match '"ForceDaemon":\s*true') { Pass 'Tailscale unattended on' } else { Warn 'Tailscale unattended OFF (tailscale set --unattended=true)' }

# Power / boot
$sb = (powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String 'AC Power').ToString()
if ($sb -match '0x00000000') { Pass 'Sleep on AC disabled' } else { Warn 'Box may sleep on AC power' }
$boot = bcdedit /enum '{current}' | Out-String
if ($boot -match 'bootstatuspolicy\s+IgnoreAllFailures') { Pass 'No recovery screen on bad boot' } else { Warn 'Box may stop at a recovery screen after a power cut' }

# Tasks
foreach ($t in 'HomeGuard-Collector','HomeGuard-Upload','HomeGuard-Heartbeat') {
    if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) { Pass "Task present: $t" } else { Fail "Task missing: $t" }
}

# Network config
$netJson = Join-Path $PSScriptRoot 'network.json'
if (Test-Path $netJson) {
    $n = Get-Content $netJson -Raw | ConvertFrom-Json
    Pass "network.json: mode=$($n.mode)"
    $profiles = netsh wlan show profiles | Out-String
    if ($n.mode -eq 'wifi') {
        if ($profiles -match [Regex]::Escape($n.wifi_ssid)) { Pass "Customer Wi-Fi profile present: $($n.wifi_ssid)" } else { Fail "Customer Wi-Fi profile missing: $($n.wifi_ssid)" }
    }
    if ($n.rescue_ssid -and $profiles -match [Regex]::Escape($n.rescue_ssid)) { Pass "Rescue Wi-Fi profile present: $($n.rescue_ssid)" } else { Warn "Rescue Wi-Fi profile missing" }
} else {
    Fail 'network.json missing (run setup_network.ps1 / setup_customer.sh)'
}

# Link + cameras
$link = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.InterfaceAlias -ne 'Tailscale' }
if ($link) { Pass "Network link up ($($link[0].InterfaceAlias))" } else { Warn 'No non-Tailscale default route (box may be offline to the LAN)' }
if (Test-Path 'C:\home_guard\home_guard_project\data_collection\cameras.yaml') { Pass 'cameras.yaml present' } else { Warn 'cameras.yaml missing (run camera discovery)' }

Write-Host ''
if ($fails -gt 0) { Write-Host "$fails essential check(s) FAILED" -ForegroundColor Red; exit 1 } else { Write-Host 'All essential checks passed' -ForegroundColor Green }
```

- [ ] **Step 2: Parse-check**

Run the parser check as in Task 2 Step 2 against `check_box.ps1` (or on the box if blocked locally).
Expected: `parse OK`.

- [ ] **Step 3: Commit**

```bash
git add home_guard_project/box/check_box.ps1
git commit -m "box: add check_box.ps1 readiness report"
```

- [ ] **Step 4: Run on the real box**

Run: `ssh -i ~/.ssh/homeguard_box ameer@100.121.29.9 "powershell -ExecutionPolicy Bypass -File C:\\home_guard\\home_guard_project\\box\\check_box.ps1"`
Expected: a PASS/WARN/FAIL list; exit non-zero only if a task or service is missing.

---

## Task 5: Confirm the bundle test is green and document

**Files:**
- Modify: `home_guard_project/box/README.md`

**Interfaces:** none.

- [ ] **Step 1: Run the full box test suite**

Run: `uv run python -m pytest tests/box/ -v`
Expected: all pass, including `test_required_files_are_included` (now that both scripts exist) and the `network.json` exclusion.

- [ ] **Step 2: Add a README section**

Add a "Setting up a box for a customer" section to `home_guard_project/box/README.md` documenting `setup_customer.sh` usage, the Ethernet vs Wi-Fi choice, the rescue hotspot, and the `--forget-other-wifi` pre-delivery step.

- [ ] **Step 3: Commit**

```bash
git add home_guard_project/box/README.md
git commit -m "box: document customer network setup wizard in README"
```

---

## Self-review notes

- Spec coverage: Ethernet/Wi-Fi choice (Task 3), customer Wi-Fi profile + rescue (Task 2), network.json no-passwords (Task 2), bundle exclusion (Task 1), readiness (Task 4), forget-own-wifi (Tasks 2–3), docs (Task 5). Watchdog intentionally absent per the founder's choice.
- No placeholders: every script is complete.
- Type/name consistency: answers keys (`mode`, `wifi_ssid`, `wifi_password`, `rescue_ssid`, `rescue_password`) are identical in `setup_customer.sh` and `setup_network.ps1`; `network.json` fields (`mode`, `wifi_ssid`, `rescue_ssid`) match between writer (Task 2) and reader (Task 4).
