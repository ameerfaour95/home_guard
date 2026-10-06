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

# A server started before the folder was updated keeps serving the old code: stop it, so the start below runs
# the current code (and its database migrations). run_local.sh writes the commit it started from.
$versionFile = Join-Path $env:USERPROFILE ".homeguard\cloud_server.version"
$head = (& git -C $repo rev-parse HEAD 2>$null)
if ((Test-Port $Port) -and $head) {
    $running = if (Test-Path $versionFile) { (Get-Content $versionFile -Raw).Trim() } else { "" }
    if ($running -ne $head.Trim()) {
        foreach ($conn in @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
        for ($i = 0; $i -lt 15 -and (Test-Port $Port); $i++) { Start-Sleep -Seconds 1 }
    }
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

$appArgs = @("--server", "http://127.0.0.1:$Port", "--local")
# The app runs from this folder's code, so it is always the current version. The packaged exe in dist\ is a
# snapshot that is only rebuilt by hand (and Windows Smart App Control can block it): it is the fallback.
$pythonw = Join-Path $repo ".venv\Scripts\pythonw.exe"
if (Test-Path $pythonw) {
    Start-Process -FilePath $pythonw -ArgumentList (@("-m", "home_guard_project.admin") + $appArgs) -WorkingDirectory $repo
} else {
    Start-Process -FilePath $App -ArgumentList $appArgs
}
