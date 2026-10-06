# ============================================================================
#  setup_network.ps1 - configure how a collector box reaches the network.
#
#  Run on the box from an elevated PowerShell (usually over SSH, driven by
#  the HomeGuardSetup wizard on the laptop):
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
    # XML-escape the name and key material.
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
# In the box's config folder (box_paths.ps1): C:\ProgramData\HomeGuard\config, or box\ on the old layout.
. (Join-Path $BoxDir 'box_paths.ps1')
$netJson = (Get-HomeGuardPaths).NetworkJson
New-Item -ItemType Directory -Force -Path (Split-Path $netJson) | Out-Null
$net | ConvertTo-Json | Set-Content -Path $netJson -Encoding ascii
Ok "Wrote network.json (mode: $mode)"
Write-Host ''
Ok 'Network setup finished.'
