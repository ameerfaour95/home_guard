param([switch]$SkipSync)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$bundlePath = [IO.Path]::GetFullPath((Join-Path $projectRoot 'dist/HomeGuardAdmin'))
if (-not $bundlePath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Build output must remain inside this worktree.'
}
Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $projectRoot 'build/pyinstaller-cache'
    if (-not $SkipSync) {
        # Windows PowerShell turns redirected native stderr into ErrorRecords.
        # uv and PyInstaller log progress there; their exit codes decide failure.
        $ErrorActionPreference = 'Continue'
        uv --system-certs sync --only-group admin
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -ne 0) { throw 'Admin dependency installation failed.' }
    }
    $ErrorActionPreference = 'Continue'
    & .venv/Scripts/python.exe -m PyInstaller --noconfirm --clean --windowed --onedir `
        --name HomeGuardAdmin --distpath dist --workpath build/admin --specpath build `
        --paths $projectRoot --icon "$projectRoot/home_guard_project/box/assets/logo.ico" `
        --add-data "$projectRoot/home_guard_project/admin/demo_data;home_guard_project/admin/demo_data" `
        --add-data "$projectRoot/home_guard_project/box/assets/logo.ico;home_guard_project/box/assets" `
        --collect-data tzdata home_guard_project/admin/__main__.py
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    $exe = Get-Item -LiteralPath 'dist/HomeGuardAdmin/HomeGuardAdmin.exe'
    $bytes = (Get-ChildItem -LiteralPath 'dist/HomeGuardAdmin' -Recurse -File | Measure-Object Length -Sum).Sum
    Write-Output "Executable: $($exe.FullName) ($($exe.Length) bytes)"
    Write-Output "Complete one-directory bundle: $bytes bytes"
} finally {
    Pop-Location
}
