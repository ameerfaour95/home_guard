# One-click launcher for the Home Guard Admin Center on the founder's Windows laptop.
# Shortcut: powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File launch_admin.ps1
param(
    [string]$App = "",
    [int]$Port = 8610,
    [int]$WaitSeconds = 180
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
# The app is built into this folder's dist\ by home_guard_project\admin\build_admin_exe.ps1.
if (-not $App) { $App = Join-Path $repo "dist\HomeGuardAdmin\HomeGuardAdmin.exe" }
$logDir = Join-Path $env:LOCALAPPDATA "HomeGuardAdmin\logs"
$log = Join-Path $logDir "cloud.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Test-Port([int]$p) {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $c.BeginConnect("127.0.0.1", $p, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(500) -and $c.Connected) { return $true }
        return $false
    } catch { return $false } finally { $c.Close() }
}

if (-not (Test-Port $Port)) {
    $bash = "C:\Program Files\Git\bin\bash.exe"
    $script = ((Join-Path $repo "home_guard_project\cloud\run_local.sh") -replace "\\", "/")
    $errLog = Join-Path $logDir "cloud.err.log"
    Start-Process -FilePath $bash -ArgumentList @("`"$script`"") -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError $errLog | Out-Null
    $up = $false
    for ($i = 0; $i -lt $WaitSeconds; $i++) {
        if (Test-Port $Port) { $up = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $up) {
        $tail = ""
        foreach ($f in @($log, $errLog)) {
            if (Test-Path $f) { $tail += ((Get-Content $f -Tail 20) -join "`n") + "`n" }
        }
        $lines = ($tail -split "`n" | Select-Object -Last 20) -join "`n"
        [System.Windows.Forms.MessageBox]::Show(
            "The Admin Center service did not start within $WaitSeconds seconds.`n`nLast log lines:`n$lines",
            "Home Guard Admin Center", "OK", "Error") | Out-Null
        exit 1
    }
}

Start-Process -FilePath $App -ArgumentList @("--server", "http://127.0.0.1:$Port", "--local")
