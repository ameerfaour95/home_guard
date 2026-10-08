# ============================================================================
#  box_paths.ps1 - where this box keeps its config, secrets, data and logs.
#  Dot-source it from a script in this folder:
#      . (Join-Path $PSScriptRoot 'box_paths.ps1')
#      $hg = Get-HomeGuardPaths
#
#  The same rule as paths.py and _common.sh: $env:HOMEGUARD_HOME (a folder, or
#  "legacy"), else C:\ProgramData\HomeGuard once the box has moved there (its
#  layout.json marker), else the old places inside the code folder (legacy).
#  Works before the Python environment exists (setup_box.ps1).
# ============================================================================

$HgBoxDir = $PSScriptRoot
$HgAdministrators = '*S-1-5-32-544'
$HgSystem = '*S-1-5-18'

function Get-HomeGuardDefaultHome {
    $base = $env:ProgramData
    if (-not $base) { $base = 'C:\ProgramData' }
    Join-Path $base 'HomeGuard'
}

function Get-HomeGuardPaths {
    param([string]$CodeDir = (Resolve-Path (Join-Path $HgBoxDir '..\..')).Path)
    $value = "$env:HOMEGUARD_HOME".Trim()
    $default = Get-HomeGuardDefaultHome
    $hgHome = $null
    if ($value -ieq 'legacy') { $hgHome = $null }
    elseif ($value) { $hgHome = $value }
    elseif (Test-Path (Join-Path $default 'layout.json')) { $hgHome = $default }

    if ($hgHome) {
        $config = Join-Path $hgHome 'config'
        $secrets = Join-Path $hgHome 'secrets'
        return [pscustomobject]@{
            Mode = 'home'; Home = $hgHome; CodeDir = $CodeDir; ConfigDir = $config
            BoxYaml = Join-Path $config 'box.yaml'; NetworkJson = Join-Path $config 'network.json'
            CamerasYaml = Join-Path $config 'cameras.yaml'
            SecretsDir = $secrets; SecretsEnv = Join-Path $secrets 'api_key.env'
            LogsDir = Join-Path $hgHome 'logs'
        }
    }
    $box = Join-Path $CodeDir 'home_guard_project\box'
    $dc = Join-Path $CodeDir 'home_guard_project\data_collection'
    [pscustomobject]@{
        Mode = 'legacy'; Home = $null; CodeDir = $CodeDir; ConfigDir = $dc
        BoxYaml = Join-Path $box 'box.yaml'; NetworkJson = Join-Path $box 'network.json'
        CamerasYaml = Join-Path $dc 'cameras.yaml'
        SecretsDir = $CodeDir; SecretsEnv = Join-Path $CodeDir 'api_key.env'
        LogsDir = Join-Path $CodeDir 'logs'
    }
}

# A new box: create C:\ProgramData\HomeGuard with its folders, lock it down to
# Administrators, SYSTEM and $Account (the scheduled tasks' user), and write the
# marker. The box then keeps its files there from the start (no migration).
function Initialize-HomeGuardHome([string]$HgHome, [string]$Account) {
    New-Item -ItemType Directory -Force -Path $HgHome | Out-Null
    $grants = @('/grant:r', "${HgAdministrators}:(OI)(CI)F", "${HgSystem}:(OI)(CI)F", "${Account}:(OI)(CI)M")
    & icacls $HgHome /inheritance:r @grants | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not set the permissions of $HgHome (icacls)." }
    foreach ($sub in 'config', 'secrets', 'data\live', 'data\outbox', 'data\production', 'data\archive',
                     'data\state', 'data\scene_interview', 'logs', 'models') {
        New-Item -ItemType Directory -Force -Path (Join-Path $HgHome $sub) | Out-Null
    }
    & icacls (Join-Path $HgHome 'secrets') /inheritance:r @grants | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not set the permissions of $HgHome\secrets (icacls)." }
    $marker = Join-Path $HgHome 'layout.json'
    if (-not (Test-Path $marker)) {
        @{ layout_version = 1; created = (Get-Date -Format s); created_by = 'setup_box.ps1' } |
            ConvertTo-Json | Set-Content -Path $marker -Encoding ascii
    }
}
