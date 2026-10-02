# ============================================================================
#  make_shortcut.ps1 - create/refresh the "Home Guard" shortcut on the box.
#
#  Puts a "Home Guard" shortcut (with the logo icon) on the all-users Desktop
#  and in the Startup folder, pointing at live_view.cmd. Safe to re-run.
#  Run on the box:  powershell -ExecutionPolicy Bypass -File make_shortcut.ps1
#  Called by setup_box.ps1 (first boot) and by setup_customer.ps1 (the wizard).
# ============================================================================
$BoxDir   = $PSScriptRoot
$liveCmd  = Join-Path $BoxDir 'live_view.cmd'
$liveIcon = Join-Path $BoxDir 'assets\home_guard.ico'
$targets = @(
    (Join-Path ([Environment]::GetFolderPath('CommonDesktopDirectory')) 'Home Guard.lnk'),
    (Join-Path ([Environment]::GetFolderPath('Startup')) 'Home Guard.lnk')
)
foreach ($lnk in $targets) {
    try {
        # Delete any existing .lnk first so Explorer re-reads the (new) icon
        # instead of showing the icon it cached when the shortcut was created.
        Remove-Item -Path $lnk -Force -ErrorAction SilentlyContinue
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($lnk)
        $sc.TargetPath = $liveCmd
        $sc.WorkingDirectory = $BoxDir
        if (Test-Path $liveIcon) { $sc.IconLocation = $liveIcon }
        $sc.Description = 'Home Guard - open the box window (logs, and cameras if enabled)'
        $sc.Save()
        Write-Host "[OK] shortcut: $lnk"
    } catch {
        Write-Host "[WARN] could not create $lnk - $($_.Exception.Message)"
    }
}
