# ============================================================================
#  setup_customer.ps1 - configure a collector box for a customer, from the
#  laptop. Compiled to HomeGuardSetup.exe by build_exe.ps1 (uses Windows'
#  built-in ssh.exe / scp.exe - no Git Bash needed on the machine running it).
#
#  Use it AFTER the box has had its first-boot setup (git checkout at
#  C:\home_guard + setup_box.ps1; see README). This tool does the repeatable
#  part: pick Ethernet or Wi-Fi, join the customer's Wi-Fi, optionally locate
#  the cameras, update the box's software, and report readiness.
#
#  Usage (or double-click HomeGuardSetup.exe and answer the prompts):
#    setup_customer.ps1 -Target ameer@100.121.29.9
#        [-ForgetOtherWifi]   also remove Wi-Fi networks other than the customer's
#        [-SkipUpdate]        do not git-pull the box's software first
#        [-KeyPath <path>]    SSH key (default ~\.ssh\homeguard_box)
#        [-DryRun]            print the planned steps without touching the box
#
#  Passwords travel in a temp file that is deleted from the box afterwards -
#  never on a command line, never in network.json.
# ============================================================================
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [switch]$ForgetOtherWifi,
    [switch]$SkipUpdate,
    [string]$KeyPath = (Join-Path $HOME '.ssh\homeguard_box'),
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'

function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[OK]  $m" -ForegroundColor Green }
function Note($m) { Write-Host $m -ForegroundColor DarkGray }

if ($Target -notmatch '^[^@]+@.+$') { throw "Target must be user@host, e.g. ameer@100.121.29.9 (got '$Target')" }
$BoxUser = $Target.Substring(0, $Target.IndexOf('@'))
$RemoteHome = "C:\Users\$BoxUser"
$InstallDir = 'C:\home_guard'
$BoxBox = "$InstallDir\home_guard_project\box"
$Bash = '"C:\Program Files\Git\bin\bash.exe"'

# ---- helpers ----------------------------------------------------------------
function To-B64([string]$s) { [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($s)) }

function Read-NonEmpty([string]$prompt, [string]$pattern = '.+') {
    do { $v = (Read-Host $prompt).Trim() } while ($v -notmatch "^$pattern$")
    $v
}
function Read-Secret([string]$prompt) {
    $s = Read-Host -AsSecureString $prompt
    $b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s)
    try { [Runtime.InteropServices.Marshal]::PtrToStringAuto($b) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) }
}
function Invoke-Box([string]$command) {
    if ($DryRun) { Note "  ssh> $command"; return '' }
    & ssh -i $KeyPath -o ConnectTimeout=15 $Target $command
}
function Copy-ToBox([string]$local, [string]$remote) {
    if ($DryRun) { Note "  scp> $local  ->  ${Target}:$remote"; return }
    & scp -i $KeyPath $local "${Target}:$remote"
    if ($LASTEXITCODE -ne 0) { throw "scp failed copying $local" }
}

# ---- rescue hotspot credentials (same for all of this founder's boxes) ------
$RescueDir = Join-Path $HOME '.homeguard'
$RescueFile = Join-Path $RescueDir 'rescue_wifi.txt'
if (-not (Test-Path $RescueFile)) {
    New-Item -ItemType Directory -Force -Path $RescueDir | Out-Null
    $chars = 'abcdefghijkmnpqrstuvwxyz23456789'.ToCharArray()
    $rpw = -join (1..16 | ForEach-Object { $chars | Get-Random })
    @("ssid=homeguard-rescue", "password=$rpw") | Set-Content -Path $RescueFile -Encoding ascii
    Ok "Created rescue hotspot credentials at $RescueFile"
}
$RescueSsid = (Select-String '^ssid=(.+)$'     $RescueFile).Matches[0].Groups[1].Value
$RescuePass = (Select-String '^password=(.+)$' $RescueFile).Matches[0].Groups[1].Value

# ---- questions --------------------------------------------------------------
Info "`n=== Home Guard box setup ===`n"
Info 'How does this box reach the internet?'
Write-Host '  1) Ethernet cable'
Write-Host '  2) Wi-Fi'
$Mode = ''
do { switch (Read-Host 'Choose 1 or 2') { '1' { $Mode = 'ethernet' } '2' { $Mode = 'wifi' } } } while (-not $Mode)

$WifiSsid = ''; $WifiPass = ''
if ($Mode -eq 'wifi') {
    $WifiSsid = Read-NonEmpty 'Customer Wi-Fi name (SSID)'
    do { $WifiPass = Read-Secret 'Customer Wi-Fi password (8-63 chars)' } while ($WifiPass.Length -lt 8 -or $WifiPass.Length -gt 63)
}

$DoCameras = $false; $CamUser = ''; $CamPass = ''; $Site = ''
if ((Read-Host 'Find cameras now? (needs the camera/recorder login) [y/N]') -match '^[Yy]') {
    $DoCameras = $true
    $Site = Read-NonEmpty 'Site name for camera labels (lowercase letters, digits, underscores)' '[a-z0-9_]+'
    $CamUser = Read-NonEmpty 'Camera / recorder username'
    $CamPass = Read-Secret 'Camera / recorder password'
}

# ---- 1. update the box's software (git pull) --------------------------------
if (-not $SkipUpdate) {
    Info "`n[1] Updating the box software..."
    $isCheckout = $true
    if (-not $DryRun) {
        $isCheckout = ((Invoke-Box "powershell -Command `"Test-Path $InstallDir\.git`"" | Out-String).Trim() -eq 'True')
    }
    if ($isCheckout) {
        Invoke-Box "$Bash -lc /c/home_guard/home_guard_project/box/update.sh"
        Ok 'Box software updated (git pull + restart).'
    } else {
        Note "  $InstallDir is not a git checkout yet; skipping update (do the first-boot clone per README)."
    }
} else {
    Note '[1] Skipping software update (-SkipUpdate).'
}

# ---- 2. network configuration ----------------------------------------------
Info "`n[2] Configuring the network..."
$answersLocal = [IO.Path]::GetTempFileName()
$lines = @(
    "mode=$(To-B64 $Mode)"
    "wifi_ssid=$(To-B64 $WifiSsid)"
    "wifi_password=$(To-B64 $WifiPass)"
    "rescue_ssid=$(To-B64 $RescueSsid)"
    "rescue_password=$(To-B64 $RescuePass)"
)
[IO.File]::WriteAllLines($answersLocal, $lines, (New-Object Text.UTF8Encoding($false)))
$answersRemote = "$RemoteHome\net_answers.txt"
try {
    Copy-ToBox $answersLocal $answersRemote
    $forget = ''; if ($ForgetOtherWifi) { $forget = ' -ForgetOtherWifi' }
    Invoke-Box "powershell -ExecutionPolicy Bypass -File $BoxBox\setup_network.ps1 -AnswersFile $answersRemote$forget"
} finally {
    Invoke-Box "cmd /c del $answersRemote" | Out-Null
    Remove-Item $answersLocal -ErrorAction SilentlyContinue
}

# ---- 3. camera discovery (optional) ----------------------------------------
if ($DoCameras) {
    Info "`n[3] Locating cameras..."
    $camLocal = [IO.Path]::GetTempFileName()
    [IO.File]::WriteAllText($camLocal, (To-B64 $CamPass), (New-Object Text.UTF8Encoding($false)))
    $camRemote = "$RemoteHome\cam.b64"
    try {
        Copy-ToBox $camLocal $camRemote
        # Load the password into the env var find_cameras reads, keeping it off the command line.
        $find = 'powershell -ExecutionPolicy Bypass -Command "' +
                "`$env:HG_CAMERA_PASSWORD=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String((Get-Content $camRemote))); " +
                "Set-Location $InstallDir; " +
                "& .\.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json auto --user $CamUser --prefix $Site --write" +
                '"'
        $out = Invoke-Box $find
        if (-not $DryRun) {
            try { $j = ($out | Out-String | ConvertFrom-Json); Ok "cameras found: $($j.cameras.Count)" }
            catch { Note '  (could not parse camera result; raw output:)'; Write-Host ($out | Out-String) }
        }
    } finally {
        Invoke-Box "cmd /c del $camRemote" | Out-Null
        Remove-Item $camLocal -ErrorAction SilentlyContinue
    }
}

# ---- 4. readiness report ----------------------------------------------------
Info "`n[4] Readiness"
Invoke-Box "powershell -ExecutionPolicy Bypass -File $BoxBox\check_box.ps1"

# ---- summary ----------------------------------------------------------------
Write-Host ''
Info '=== Still to do by hand ==='
Write-Host ' 1. BIOS: set "restore on AC power loss" to Power On (needs a screen once).'
Write-Host ' 2. Tailscale admin console: disable key expiry for this box.'
if (-not $DoCameras) { Write-Host ' 3. Camera discovery at the customer site (re-run and answer yes to "Find cameras now").' }
Write-Host ' *. Before delivery, re-run with -ForgetOtherWifi to drop your own Wi-Fi.'
Write-Host ''
Info 'Rescue hotspot (make a phone hotspot with these to recover a box):'
Write-Host "   name:     $RescueSsid"
Write-Host "   password: $RescuePass"
if ($DryRun) { Write-Host ''; Note 'DryRun: nothing on the box was changed.' }
