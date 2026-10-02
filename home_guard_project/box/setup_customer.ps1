# ============================================================================
#  setup_customer.ps1 - configure a collector box for a customer, from the
#  laptop. Compiled to HomeGuardSetup.exe by build_exe.ps1 (uses Windows'
#  built-in ssh.exe / scp.exe - no Git Bash needed on the machine running it).
#
#  Use it AFTER the box has had its first-boot setup (git checkout at
#  C:\home_guard + setup_box.ps1; see README). This tool does the repeatable
#  part: pick Ethernet or Wi-Fi, join the customer's Wi-Fi, name the house
#  (which names the S3 folder its clips go to), optionally locate the cameras,
#  update the box's software, and report readiness.
#
#  Usage (or double-click HomeGuardSetup.exe and answer the prompts):
#    setup_customer.ps1 [-Target ameer@100.121.29.9]
#        [-ForgetOtherWifi]   also remove Wi-Fi networks other than the customer's
#        [-SkipUpdate]        do not git-pull the box's software first
#        [-KeyPath <path>]    SSH key (default ~\.ssh\homeguard_box)
#        [-DryRun]            print the planned steps without touching the box
#
#  Passwords travel in a temp file that is deleted from the box afterwards -
#  never on a command line, never in network.json.
#
#  Commands sent to the box contain no quotes and no "&" on purpose: Windows
#  PowerShell drops embedded quotes when it passes an argument to ssh.exe.
# ============================================================================
param(
    [string]$Target,
    [switch]$ForgetOtherWifi,
    [switch]$SkipUpdate,
    [string]$KeyPath = (Join-Path $HOME '.ssh\homeguard_box'),
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'

function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[OK]  $m" -ForegroundColor Green }
function Note($m) { Write-Host $m -ForegroundColor DarkGray }
function Bad($m)  { Write-Host "[!!]  $m" -ForegroundColor Red }

# Started by double-click, the window closes with the program. Wait, so the result can be read.
function Wait-BeforeClose {
    Write-Host ''
    [void](Read-Host 'Press Enter to close')
}
trap {
    Write-Host ''
    Bad "ERROR: $($_.Exception.Message)"
    Bad 'The setup stopped here. Fix the problem above and run the program again; running it twice is safe.'
    Wait-BeforeClose
    exit 1
}

$HgDir = Join-Path $HOME '.homeguard'
New-Item -ItemType Directory -Force -Path $HgDir | Out-Null

# ---- which box --------------------------------------------------------------
$LastTargetFile = Join-Path $HgDir 'last_target.txt'
if (-not $Target) {
    $last = ''
    if (Test-Path $LastTargetFile) { $last = (Get-Content $LastTargetFile -TotalCount 1).Trim() }
    $hint = ''
    if ($last) { $hint = " (Enter = $last)" }
    Info "`n=== Home Guard box setup ===`n"
    do {
        $Target = (Read-Host "Box address, written as user@address$hint").Trim()
        if (-not $Target) { $Target = $last }
    } while ($Target -notmatch '^[^@\s]+@\S+$')
}
if ($Target -notmatch '^[^@\s]+@\S+$') { throw "Target must be user@host, e.g. ameer@100.121.29.9 (got '$Target')" }
Set-Content -Path $LastTargetFile -Value $Target -Encoding ascii

if (-not $DryRun -and -not (Test-Path $KeyPath)) {
    throw "The access key for the box was not found at $KeyPath. This program must run on the setup laptop."
}

$BoxUser = $Target.Substring(0, $Target.IndexOf('@'))
$RemoteHome = "C:\Users\$BoxUser"
$InstallDir = 'C:\home_guard'
$BoxBox = "$InstallDir\home_guard_project\box"
$Bash = 'C:\PROGRA~1\Git\bin\bash.exe'      # short path of C:\Program Files: no spaces, so no quotes needed
$Python = '.venv\Scripts\python.exe'

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

# Inside the compiled program every line a native command writes to stderr
# becomes a terminating error under ErrorActionPreference Stop, and ssh, git
# and Python all write ordinary progress there. Run native commands with
# Continue, return what they wrote to stdout, and show stderr dimmed.
function Invoke-Native([string]$exe, [string[]]$arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $exe @arguments 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                Write-Host "    $($_.Exception.Message)" -ForegroundColor DarkGray
            } else {
                $_
            }
        }
    } finally {
        $ErrorActionPreference = $previous
    }
}
function Invoke-Box([string]$command) {
    if ($DryRun) { Note "  ssh> $command"; $global:LASTEXITCODE = 0; return '' }
    Invoke-Native 'ssh' @('-i', $KeyPath, '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR',
                          '-o', 'StrictHostKeyChecking=accept-new', $Target, $command)
}
function Copy-ToBox([string]$local, [string]$remote) {
    if ($DryRun) { Note "  scp> $local  ->  ${Target}:$remote"; return }
    Invoke-Native 'scp' @('-i', $KeyPath, '-o', 'LogLevel=ERROR', '-o', 'StrictHostKeyChecking=accept-new',
                          $local, "${Target}:$remote") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not copy a file to the box (scp failed)." }
}

# ---- can we reach the box at all? -------------------------------------------
if (-not $DryRun) {
    Note "Connecting to $Target ..."
    $hello = (Invoke-Box 'echo box-is-reachable' | Out-String)
    if ($hello -notmatch 'box-is-reachable') {
        throw "Cannot reach the box at $Target. Check that the box is on, that Tailscale shows it connected, and that this laptop has internet."
    }
    Ok 'Connected to the box.'
}

# ---- rescue hotspot credentials (same for all of this founder's boxes) ------
$RescueFile = Join-Path $HgDir 'rescue_wifi.txt'
if (-not (Test-Path $RescueFile)) {
    $chars = 'abcdefghijkmnpqrstuvwxyz23456789'.ToCharArray()
    $rpw = -join (1..16 | ForEach-Object { $chars | Get-Random })
    @("ssid=homeguard-rescue", "password=$rpw") | Set-Content -Path $RescueFile -Encoding ascii
    Ok "Created rescue hotspot credentials at $RescueFile"
}
$RescueSsid = (Select-String '^ssid=(.+)$'     $RescueFile).Matches[0].Groups[1].Value
$RescuePass = (Select-String '^password=(.+)$' $RescueFile).Matches[0].Groups[1].Value

# ---- questions --------------------------------------------------------------
Info "`nHow does this box reach the internet?"
Write-Host '  1) Ethernet cable'
Write-Host '  2) Wi-Fi'
$Mode = ''
do { switch (Read-Host 'Choose 1 or 2') { '1' { $Mode = 'ethernet' } '2' { $Mode = 'wifi' } } } while (-not $Mode)

$WifiSsid = ''; $WifiPass = ''
if ($Mode -eq 'wifi') {
    $WifiSsid = Read-NonEmpty 'Customer Wi-Fi name (SSID)'
    do { $WifiPass = Read-Secret 'Customer Wi-Fi password (8-63 chars)' } while ($WifiPass.Length -lt 8 -or $WifiPass.Length -gt 63)
}

Info "`nName for this house."
Write-Host '  The clips are saved online in a folder with this name (dataset_<name>),'
Write-Host '  and the cameras are labelled with it. Lowercase letters, digits and underscores only.'
$Site = Read-NonEmpty 'House name, for example cohen_haifa' '[a-z0-9_]+'

$DoCameras = $false; $CamUser = ''; $CamPass = ''
if ((Read-Host 'Find cameras now? (needs the camera/recorder login) [y/N]') -match '^[Yy]') {
    $DoCameras = $true
    $CamUser = Read-NonEmpty 'Camera / recorder username' '[A-Za-z0-9._@-]+'
    $CamPass = Read-Secret 'Camera / recorder password'
}

# ---- 1. update the box's software (git pull) --------------------------------
if (-not $SkipUpdate) {
    Info "`n[1] Updating the box software..."
    $isCheckout = $true
    if (-not $DryRun) {
        $isCheckout = ((Invoke-Box "if exist $InstallDir\.git (echo True) else (echo False)" | Out-String).Trim() -eq 'True')
    }
    if ($isCheckout) {
        Invoke-Box "$Bash -lc /c/home_guard/home_guard_project/box/update.sh" | ForEach-Object { Note "    $_" }
        if ($LASTEXITCODE -eq 0) { Ok 'Box software updated and restarted.' }
        else { Bad 'The update did not finish. Continuing with the software already on the box.' }
    } else {
        Note "  $InstallDir is not a git checkout yet; skipping update (do the first-boot clone per README)."
    }
} else {
    Note '[1] Skipping software update (-SkipUpdate).'
}

# ---- 2. name the house: this is the S3 folder the clips are saved in --------
Info "`n[2] Setting the house name..."
Invoke-Box "cd /d $InstallDir && $Python -m home_guard_project.box set-site $Site" | ForEach-Object { Note "    $_" }
if ($LASTEXITCODE -ne 0) { throw "The box did not accept the house name '$Site'." }
Ok "Clips from this box are saved in the online folder dataset_$Site"

# ---- 3. network configuration ----------------------------------------------
Info "`n[3] Configuring the network..."
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
    if ($LASTEXITCODE -ne 0) { throw 'The network settings could not be saved on the box.' }
} finally {
    Invoke-Box "cmd /c del $answersRemote" | Out-Null
    Remove-Item $answersLocal -ErrorAction SilentlyContinue
}

# ---- 4. camera discovery (optional) ----------------------------------------
$CamerasFound = -1
if ($DoCameras) {
    Info "`n[4] Locating cameras (about one minute)..."
    $camLocal = [IO.Path]::GetTempFileName()
    [IO.File]::WriteAllText($camLocal, (To-B64 $CamPass), (New-Object Text.UTF8Encoding($false)))
    $camRemote = "$RemoteHome\cam.b64"
    try {
        Copy-ToBox $camLocal $camRemote
        $out = Invoke-Box "cd /d $InstallDir && $Python -m home_guard_project.box.find_cameras --json auto --user $CamUser --prefix $Site --write --password-file $camRemote"
        if (-not $DryRun) {
            try {
                $j = ($out | Out-String | ConvertFrom-Json)
                $CamerasFound = @($j.cameras).Count
                if ($CamerasFound -gt 0) {
                    Ok "cameras found: $CamerasFound"
                    foreach ($c in $j.cameras) { Write-Host ("        {0,-24} {1}x{2}" -f $c.name, $c.width, $c.height) }
                } else {
                    Bad 'cameras found: 0. Check the camera login, and that the box is on the same network as the cameras.'
                }
            } catch {
                Bad 'Could not read the camera result. What the box answered:'
                Write-Host ($out | Out-String)
            }
        }
    } finally {
        Invoke-Box "cmd /c del $camRemote" | Out-Null
        Remove-Item $camLocal -ErrorAction SilentlyContinue
    }
}

# ---- 5. readiness report ----------------------------------------------------
Info "`n[5] Readiness"
Invoke-Box "powershell -ExecutionPolicy Bypass -File $BoxBox\check_box.ps1"

# ---- summary ----------------------------------------------------------------
Write-Host ''
Info '=== Still to do by hand ==='
Write-Host ' 1. BIOS: set "restore on AC power loss" to Power On (needs a screen once).'
Write-Host ' 2. Tailscale admin console: disable key expiry for this box.'
if (-not $DoCameras) { Write-Host ' 3. Camera discovery at the customer site (run again and answer y to "Find cameras now").' }
Write-Host ' *. Before delivery, run again with -ForgetOtherWifi to drop your own Wi-Fi.'
Write-Host ''
Info 'Rescue hotspot (make a phone hotspot with these to recover a box):'
Write-Host "   name:     $RescueSsid"
Write-Host "   password: $RescuePass"
if ($DryRun) { Write-Host ''; Note 'DryRun: nothing on the box was changed.' }
Wait-BeforeClose
