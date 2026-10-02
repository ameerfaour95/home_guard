# ============================================================================
#  add_wifi.ps1 - teach the box a Wi-Fi network so it joins it by itself
#
#  Use this when the box will not be cabled to the router (Ethernet needs no setup).
#  Run on the box from an elevated PowerShell (locally or over SSH), BEFORE it
#  is moved if you already know the other house's Wi-Fi name and password:
#      powershell -ExecutionPolicy Bypass -File add_wifi.ps1 -Ssid "HouseWifi" -Password "secret"
#
#  What it does:
#    1. Saves the network for all users with automatic connection, so the box
#       joins it at boot with nobody logged in
#    2. Lists the saved networks
#
#  -ConnectNow switches to the network immediately. Over a remote session this
#  drops the connection for a moment; if the network is out of range nothing changes.
#  -Remove deletes a saved network instead of adding one.
# ============================================================================
#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory = $true)]
    [string]$Ssid,

    [string]$Password,

    [ValidateSet('WPA2', 'WPA3')]
    [string]$Security = 'WPA2',

    [switch]$ConnectNow,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

function Show-Profiles {
    Write-Host ''
    Write-Host 'Saved Wi-Fi networks:'
    netsh wlan show profiles | Select-String 'All User Profile' | ForEach-Object { '  ' + ($_.Line -split ':', 2)[1].Trim() }
}

if ($Remove) {
    netsh wlan delete profile name="$Ssid" | Out-Host
    Show-Profiles
    return
}

if (-not $Password -or $Password.Length -lt 8) {
    throw 'A Wi-Fi password of at least 8 characters is required (-Password).'
}

$auth = if ($Security -eq 'WPA3') { 'WPA3SAE' } else { 'WPA2PSK' }
$name = [Security.SecurityElement]::Escape($Ssid)
$key = [Security.SecurityElement]::Escape($Password)

$xml = @"
<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>$name</name>
  <SSIDConfig>
    <SSID>
      <name>$name</name>
    </SSID>
  </SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>auto</connectionMode>
  <MSM>
    <security>
      <authEncryption>
        <authentication>$auth</authentication>
        <encryption>AES</encryption>
        <useOneX>false</useOneX>
      </authEncryption>
      <sharedKey>
        <keyType>passPhrase</keyType>
        <protected>false</protected>
        <keyMaterial>$key</keyMaterial>
      </sharedKey>
    </security>
  </MSM>
</WLANProfile>
"@

# The profile file holds the password in clear text: keep it only as long as netsh needs it.
$tmp = Join-Path $env:TEMP ("wifi-" + [Guid]::NewGuid().ToString('N') + '.xml')
try {
    Set-Content -Path $tmp -Value $xml -Encoding UTF8
    $out = netsh wlan add profile filename="$tmp" user=all
    if ($LASTEXITCODE -ne 0) { throw "netsh could not add the network: $out" }
    Write-Host "[OK]    Saved Wi-Fi network '$Ssid' ($Security, connects automatically)." -ForegroundColor Green
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}

if ($ConnectNow) {
    netsh wlan connect name="$Ssid" | Out-Host
}

Show-Profiles
