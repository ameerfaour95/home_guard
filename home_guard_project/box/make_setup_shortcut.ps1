# ============================================================================
#  make_setup_shortcut.ps1 - a "Home Guard Setup" shortcut on the setup laptop
#
#  Lives in:  home_guard_project/box/make_setup_shortcut.ps1
#  Run once on the laptop that has this repository:
#      powershell -ExecutionPolicy Bypass -File make_setup_shortcut.ps1 [-Desktop]
#
#  The shortcut starts the graphical setup wizard through the repository's own
#  Python (app\start.pyw, so no console window). Nothing is compiled, so there is no
#  new, unsigned program for an antivirus to stop and examine on every run -
#  which is what happens with a freshly built HomeGuardSetup.exe.
#
#  It is written to dist\ next to the exe; -Desktop also puts one on the Desktop.
# ============================================================================
param([switch]$Desktop)
$ErrorActionPreference = 'Stop'

$BoxDir   = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $BoxDir '..\..')).Path
$Icon     = Join-Path $BoxDir 'assets\home_guard.ico'
$Starter  = Join-Path $BoxDir 'app\start.pyw'

# The environment's own pythonw.exe starts the console interpreter, which opens a
# terminal window beside the wizard. Use the windowless interpreter it was made from.
$cfg = Join-Path $RepoRoot '.venv\pyvenv.cfg'
if (-not (Test-Path $cfg)) { throw "No Python environment in $RepoRoot\.venv. Run 'uv sync' in $RepoRoot first." }
$base = (Select-String -Path $cfg -Pattern '^\s*home\s*=\s*(.+)$').Matches[0].Groups[1].Value.Trim()
$Python = Join-Path $base 'pythonw.exe'
if (-not (Test-Path $Python)) { throw "pythonw.exe was not found in $base." }

$targets = @(Join-Path $RepoRoot 'dist')
if ($Desktop) { $targets += [Environment]::GetFolderPath('Desktop') }

$shell = New-Object -ComObject WScript.Shell
foreach ($dir in $targets) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $path = Join-Path $dir 'Home Guard Setup.lnk'
    if (Test-Path $path) { Remove-Item $path -Force }    # so Explorer reads the icon again
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $Python
    $link.Arguments = "`"$Starter`" --setup"
    $link.WorkingDirectory = $RepoRoot
    $link.IconLocation = "$Icon,0"
    $link.Description = 'Set up a Home Guard box'
    $link.Save()
    Write-Host "[OK] $path" -ForegroundColor Green
}
