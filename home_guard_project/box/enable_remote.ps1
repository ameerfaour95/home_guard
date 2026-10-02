# ============================================================================
#  enable_remote.ps1 - let the laptop log in to this box over SSH
#
#  Run ONCE on the box, in a PowerShell window opened with "Run as administrator":
#      powershell -ExecutionPolicy Bypass -File <path>\enable_remote.ps1
#
#  What it does:
#    1. Installs and starts the Windows OpenSSH server (starts automatically at boot)
#    2. Opens TCP port 22 in the Windows firewall
#    3. Authorizes one SSH public key (the laptop's) - no password login is added
#    4. Prints the user name and IP address to give back to the laptop
# ============================================================================
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

# Filled in by prepare_remote.sh on the laptop.
$PublicKey = '__PUBLIC_KEY__'

if ($PublicKey -notmatch '^ssh-') {
    throw 'No public key in this script. Generate it on the laptop with home_guard_project/box/prepare_remote.sh.'
}

Write-Host '[1/4] OpenSSH server...'
$cap = Get-WindowsCapability -Online -Name 'OpenSSH.Server*'
if ($cap.State -ne 'Installed') {
    Add-WindowsCapability -Online -Name $cap.Name | Out-Null
}
Set-Service -Name sshd -StartupType Automatic
Start-Service -Name sshd

Write-Host '[2/4] Firewall rule for port 22...'
$ruleName = 'HomeGuard-SSH-In'
if (-not (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name $ruleName -DisplayName 'Home Guard SSH (sshd)' -Enabled True `
        -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow -Profile Any | Out-Null
}

Write-Host '[3/4] Authorizing the laptop key...'
# Members of Administrators are checked against this file, not ~\.ssh\authorized_keys.
$adminKeys = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
$existing = if (Test-Path $adminKeys) { Get-Content $adminKeys } else { @() }
if ($existing -notcontains $PublicKey) {
    Add-Content -Path $adminKeys -Value $PublicKey -Encoding ascii
}
# sshd ignores the file unless only Administrators and SYSTEM can access it (SIDs, so any Windows language works).
icacls $adminKeys /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null

# Same key for a non-administrator account, in case this user is not an admin.
$userSsh = Join-Path $env:USERPROFILE '.ssh'
New-Item -ItemType Directory -Force -Path $userSsh | Out-Null
$userKeys = Join-Path $userSsh 'authorized_keys'
$existing = if (Test-Path $userKeys) { Get-Content $userKeys } else { @() }
if ($existing -notcontains $PublicKey) {
    Add-Content -Path $userKeys -Value $PublicKey -Encoding ascii
}

Write-Host '[4/4] Done.'
$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -ExpandProperty IPAddress
$edition = (Get-CimInstance Win32_OperatingSystem).Caption

Write-Host ''
Write-Host '================ GIVE THESE TO CLAUDE ================' -ForegroundColor Green
Write-Host "User:    $env:USERNAME"
Write-Host "IP:      $($ips -join ', ')"
Write-Host "Windows: $edition"
Write-Host '======================================================' -ForegroundColor Green
