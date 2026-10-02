# ============================================================================
#  setup_customer.ps1 - configure a collector box for a customer, from the
#  laptop. Compiled to HomeGuardSetup.exe by build_exe.ps1 (uses Windows'
#  built-in ssh.exe / scp.exe - no Git Bash needed on the machine running it).
#
#  THREE ways to run:
#    * double-click HomeGuardSetup.exe  -> opens the graphical wizard (the
#        Python app), which gathers the answers and calls this script with
#        -AnswersFile. Falls back to the console wizard if no bundled Python.
#    * -Console                         -> force the text wizard (prompts).
#    * -AnswersFile <file.json>         -> non-interactive: read every answer
#        from the JSON, run with no prompts, and print @@ event lines for a UI.
#
#  Use it AFTER the box has had its first-boot setup (git checkout at
#  C:\home_guard + setup_box.ps1; see README). This tool does the repeatable
#  part: pick Ethernet or Wi-Fi, join the customer's Wi-Fi, name the house,
#  optionally locate the cameras, set the mode/alerts, update the software,
#  and report readiness.
#
#  Passwords travel in temp files that are deleted afterwards - never on a
#  command line, never in network.json. The -AnswersFile is deleted as soon as
#  it is read (it holds passwords) and is never echoed.
#
#  -AnswersFile JSON (UTF-8):
#    {"target":"user@addr","network":"ethernet"|"wifi","wifi_ssid":"",
#     "wifi_password":"","site":"cohen_haifa","show_cameras":false,
#     "find_cameras":true,"camera_user":"","camera_password":"",
#     "alerts":false,"alert_start_hour":0,"alert_end_hour":0,
#     "alert_cooldown_sec":120}
#  Event lines (stdout) in -AnswersFile mode:
#    @@step <id> start | ok|warn|fail|skip <text>   ids: connect update site
#                                                    network cameras alerts readiness
#    @@camera <name> <w>x<h>     @@check PASS|WARN|FAIL <text>
#    @@rescue <ssid> <password>  @@done ok|fail
#  Exit code 0 when no step failed, 1 otherwise.
# ============================================================================
param(
    [string]$Target,
    [string]$AnswersFile,
    [switch]$Console,
    [switch]$ForgetOtherWifi,
    [switch]$SkipUpdate,
    [string]$KeyPath = (Join-Path $HOME '.ssh\homeguard_box'),
    [switch]$DryRun
)

# ---- double-clicked exe: open the graphical wizard --------------------------
# The exe sits in <repo>\dist; the bundled interpreter is <repo>\.venv. When no
# -AnswersFile and no -Console is given and that interpreter exists, hand off to
# the Python app and exit; otherwise fall through to the console wizard below.
function Get-SetupRepoRoot {
    $cands = @()
    try {
        $exe = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
        if ($exe) { $cands += (Split-Path -Parent (Split-Path -Parent $exe)) }   # dist -> repo
    } catch { }
    if ($PSScriptRoot) { $cands += (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) }  # box -> hgp -> repo
    $cands += 'C:\home_guard'
    foreach ($c in $cands) {
        if ($c -and (Test-Path (Join-Path $c '.venv\Scripts\pythonw.exe'))) { return $c }
    }
    return $null
}
if (-not $AnswersFile -and -not $Console) {
    try {
        $repo = Get-SetupRepoRoot
        if ($repo) {
            Start-Process -FilePath (Join-Path $repo '.venv\Scripts\pythonw.exe') `
                -ArgumentList @('-m', 'home_guard_project.box.app', '--setup') -WorkingDirectory $repo
            exit 0
        }
    } catch { }   # fall through to the console wizard
}

$ErrorActionPreference = 'Stop'
$NonInteractive = [bool]$AnswersFile

function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[OK]  $m" -ForegroundColor Green }
function Note($m) { Write-Host $m -ForegroundColor DarkGray }
function Bad($m)  { Write-Host "[!!]  $m" -ForegroundColor Red }

# Structured events for the graphical wizard - printed only in -AnswersFile mode.
$script:StepFailed = $false
function Emit($s)          { if ($NonInteractive) { Write-Host $s } }
function Step-Start($id)   { Emit "@@step $id start" }
function Step-Ok($id, $t)  { Emit "@@step $id ok $t" }
function Step-Warn($id, $t){ Emit "@@step $id warn $t" }
function Step-Fail($id, $t){ Emit "@@step $id fail $t"; $script:StepFailed = $true }
function Step-Skip($id, $t){ Emit "@@step $id skip $t" }

# Started by double-click, the console closes with the program - wait so the
# result can be read. Never waits in -AnswersFile mode (a UI is reading stdout).
function Wait-BeforeClose {
    if ($NonInteractive) { return }
    Write-Host ''
    try { Write-Host 'Press any key to close...' -NoNewline; [void][Console]::ReadKey($true); Write-Host '' }
    catch { [void](Read-Host 'Press Enter to close') }
}
trap {
    Write-Host ''
    Bad "ERROR: $($_.Exception.Message)"
    if ($NonInteractive) {
        Emit '@@done fail'
    } else {
        Bad 'The setup stopped here. Fix the problem above and run the program again; running it twice is safe.'
        Wait-BeforeClose
    }
    exit 1
}

$HgDir = Join-Path $HOME '.homeguard'
New-Item -ItemType Directory -Force -Path $HgDir | Out-Null

# ---- answers: from the JSON file, or asked below ---------------------------
$Mode = ''; $WifiSsid = ''; $WifiPass = ''; $Site = ''
$ShowCameras = $false; $UseAI = $false; $AlertStart = 0; $AlertEnd = 0; $AlertCooldown = 120
$DoCameras = $false; $CamUser = ''; $CamPass = ''

if ($NonInteractive) {
    if (-not (Test-Path $AnswersFile)) { throw "Answers file not found: $AnswersFile" }
    $raw = Get-Content -Raw -Encoding UTF8 -Path $AnswersFile
    Remove-Item -Path $AnswersFile -Force -ErrorAction SilentlyContinue   # it holds passwords: gone once read
    $a = $raw | ConvertFrom-Json
    $Target      = [string]$a.target
    $Mode        = [string]$a.network
    $WifiSsid    = [string]$a.wifi_ssid
    $WifiPass    = [string]$a.wifi_password
    $Site        = [string]$a.site
    $ShowCameras = [bool]$a.show_cameras
    $DoCameras   = [bool]$a.find_cameras
    $CamUser     = [string]$a.camera_user
    $CamPass     = [string]$a.camera_password
    $UseAI       = [bool]$a.alerts
    if ($null -ne $a.alert_start_hour) { $AlertStart = [int]$a.alert_start_hour }
    if ($null -ne $a.alert_end_hour)   { $AlertEnd   = [int]$a.alert_end_hour }
    if ($a.alert_cooldown_sec)         { $AlertCooldown = [int]$a.alert_cooldown_sec }
    if ($Target -notmatch '^[^@\s]+@\S+$') { throw "answers: 'target' must be user@host (got '$Target')" }
    if (@('ethernet', 'wifi') -notcontains $Mode) { throw "answers: 'network' must be ethernet or wifi (got '$Mode')" }
    if ($Site -notmatch '^[a-z0-9_]+$') { throw "answers: 'site' must be lowercase letters, digits or underscores (got '$Site')" }
}

# ---- which box (console wizard asks; -AnswersFile already has it) -----------
$LastTargetFile = Join-Path $HgDir 'last_target.txt'
if (-not $NonInteractive) {
    if (-not $Target) {
        $last = ''
        if (Test-Path $LastTargetFile) { $last = (Get-Content $LastTargetFile -TotalCount 1).Trim() }
        $hint = ''; if ($last) { $hint = " (Enter = $last)" }
        Info "`n=== Home Guard box setup ===`n"
        do {
            $Target = (Read-Host "Box address, written as user@address$hint").Trim()
            if (-not $Target) { $Target = $last }
        } while ($Target -notmatch '^[^@\s]+@\S+$')
    }
    if ($Target -notmatch '^[^@\s]+@\S+$') { throw "Target must be user@host, e.g. ameer@100.121.29.9 (got '$Target')" }
}
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
    # -n: do not read from stdin, so ssh never swallows the console's keystrokes.
    Invoke-Native 'ssh' @('-n', '-i', $KeyPath, '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR',
                          '-o', 'StrictHostKeyChecking=accept-new', $Target, $command)
}
function Copy-ToBox([string]$local, [string]$remote) {
    if ($DryRun) { Note "  scp> $local  ->  ${Target}:$remote"; return }
    Invoke-Native 'scp' @('-i', $KeyPath, '-o', 'LogLevel=ERROR', '-o', 'StrictHostKeyChecking=accept-new',
                          $local, "${Target}:$remote") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not copy a file to the box (scp failed)." }
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
Emit "@@rescue $RescueSsid $RescuePass"

# ---- questions (console wizard only) ----------------------------------------
if (-not $NonInteractive) {
    Info "`nHow does this box reach the internet?"
    Write-Host '  1) Ethernet cable'
    Write-Host '  2) Wi-Fi'
    do { switch (Read-Host 'Choose 1 or 2') { '1' { $Mode = 'ethernet' } '2' { $Mode = 'wifi' } } } while (-not $Mode)

    if ($Mode -eq 'wifi') {
        $WifiSsid = Read-NonEmpty 'Customer Wi-Fi name (SSID)'
        do { $WifiPass = Read-Secret 'Customer Wi-Fi password (8-63 chars)' } while ($WifiPass.Length -lt 8 -or $WifiPass.Length -gt 63)
    }

    Info "`nName for this house."
    Write-Host '  The clips are saved online in a folder with this name (dataset_<name>),'
    Write-Host '  and the cameras are labelled with it. Lowercase letters, digits and underscores only.'
    $Site = Read-NonEmpty 'House name, for example cohen_haifa' '[a-z0-9_]+'

    if ((Read-Host "Show the live camera pictures on the box's own screen? [y/N]") -match '^[Yy]') { $ShowCameras = $true }

    Info "`nWhat is this box for?"
    Write-Host '  1) Data collection - save camera clips for later tagging/training'
    Write-Host '  2) Production      - AI watches and sends Telegram alerts'
    $picked = $null
    do { switch (Read-Host 'Choose 1 or 2') { '1' { $picked = $false } '2' { $picked = $true } } } while ($null -eq $picked)
    $UseAI = $picked
    if ($UseAI) {
        Info '  Production: when a person is seen during the hours below, the box sends a Telegram alert with a photo and logs what the AI saw.'
        $AlertStart = [int](Read-NonEmpty 'Send alerts from which hour (0-23)?' '([0-9]|1[0-9]|2[0-3])')
        $AlertEnd   = [int](Read-NonEmpty 'Send alerts until which hour (0-23)?' '([0-9]|1[0-9]|2[0-3])')
    }

    if ((Read-Host 'Find cameras now? (needs the camera/recorder login) [y/N]') -match '^[Yy]') {
        $DoCameras = $true
        $CamUser = Read-NonEmpty 'Camera / recorder username' '[A-Za-z0-9._@-]+'
        $CamPass = Read-Secret 'Camera / recorder password'
    }
}

# ---- connect ----------------------------------------------------------------
Step-Start 'connect'
if (-not $DryRun) {
    Note "Connecting to $Target ..."
    $hello = (Invoke-Box 'echo box-is-reachable' | Out-String)
    if ($hello -notmatch 'box-is-reachable') {
        Step-Fail 'connect' "cannot reach the box at $Target"
        throw "Cannot reach the box at $Target. Check that the box is on, that Tailscale shows it connected, and that this laptop has internet."
    }
    Ok 'Connected to the box.'
}
Step-Ok 'connect' "reachable at $Target"

# ---- update the box software (git pull) -------------------------------------
if (-not $SkipUpdate) {
    Step-Start 'update'
    Info "`n[1] Updating the box software..."
    $isCheckout = $true
    if (-not $DryRun) {
        $isCheckout = ((Invoke-Box "if exist $InstallDir\.git (echo True) else (echo False)" | Out-String).Trim() -eq 'True')
    }
    if ($isCheckout) {
        Invoke-Box "$Bash -lc /c/home_guard/home_guard_project/box/update.sh" | ForEach-Object { Note "    $_" }
        if ($LASTEXITCODE -eq 0) { Ok 'Box software updated and restarted.'; Step-Ok 'update' 'updated and restarted' }
        else { Bad 'The update did not finish. Continuing with the software already on the box.'; Step-Warn 'update' 'update did not finish; using existing software' }
    } else {
        Note "  $InstallDir is not a git checkout yet; skipping update (do the first-boot clone per README)."
        Step-Warn 'update' 'not a git checkout; skipped'
    }
} else {
    Note '[1] Skipping software update (-SkipUpdate).'
    Step-Skip 'update' '-SkipUpdate'
}

# ---- name the house + screen option -----------------------------------------
Step-Start 'site'
Info "`n[2] Setting the house name..."
Invoke-Box "cd /d $InstallDir && $Python -m home_guard_project.box set-site $Site" | ForEach-Object { Note "    $_" }
if ($LASTEXITCODE -ne 0) { Step-Fail 'site' "the box rejected the house name '$Site'"; throw "The box did not accept the house name '$Site'." }
Ok "Clips from this box are saved in the online folder dataset_$Site"
$showVal = 'false'; if ($ShowCameras) { $showVal = 'true' }
Invoke-Box "cd /d $InstallDir && $Python -m home_guard_project.box set-option show_cameras $showVal" | ForEach-Object { Note "    $_" }
Ok "Camera windows on the box's own screen: $showVal"
Invoke-Box "powershell -ExecutionPolicy Bypass -File $BoxBox\make_shortcut.ps1" | ForEach-Object { Note "    $_" }
Step-Ok 'site' "dataset_$Site, show_cameras=$showVal"

# ---- network configuration --------------------------------------------------
Step-Start 'network'
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
    if ($LASTEXITCODE -ne 0) { Step-Fail 'network' 'the box could not save the network settings'; throw 'The network settings could not be saved on the box.' }
} finally {
    Invoke-Box "cmd /c del $answersRemote" | Out-Null
    Remove-Item $answersLocal -ErrorAction SilentlyContinue
}
Step-Ok 'network' "mode=$Mode"

# ---- camera discovery (optional) --------------------------------------------
if ($DoCameras) {
    Step-Start 'cameras'
    Info "`n[4] Locating cameras (about one minute)..."
    $camLocal = [IO.Path]::GetTempFileName()
    [IO.File]::WriteAllText($camLocal, (To-B64 $CamPass), (New-Object Text.UTF8Encoding($false)))
    $camRemote = "$RemoteHome\cam.b64"
    try {
        Copy-ToBox $camLocal $camRemote
        $out = Invoke-Box "cd /d $InstallDir && $Python -m home_guard_project.box.find_cameras --json auto --user $CamUser --prefix $Site --write --password-file $camRemote"
        if ($DryRun) {
            Step-Ok 'cameras' 'dry run'
        } else {
            try {
                $j = ($out | Out-String | ConvertFrom-Json)
                $CamerasFound = @($j.cameras).Count
                $refused = @(); if ($j.login_refused) { $refused = @($j.login_refused) }
                $devices = $j.devices_found
                if ($null -eq $devices) { $devices = $CamerasFound + $refused.Count }
                if ($CamerasFound -gt 0) {
                    Ok "cameras found: $CamerasFound"
                    foreach ($c in $j.cameras) {
                        Write-Host ("        {0,-24} {1}x{2}" -f $c.name, $c.width, $c.height)
                        Emit "@@camera $($c.name) $($c.width)x$($c.height)"
                    }
                    if ($refused.Count -gt 0) {
                        Note "  ($($refused.Count) of $devices device(s) refused this login and were skipped.)"
                        Step-Ok 'cameras' "$CamerasFound found; $($refused.Count) refused the login"
                    } else {
                        Step-Ok 'cameras' "$CamerasFound found"
                    }
                } elseif ($refused.Count -gt 0) {
                    # Devices are there, the login is wrong - the most common real cause, and NOT a network fault.
                    $msg = "$($refused.Count) of $devices cameras answered but refused this login. Use the cameras' own user name and password."
                    Bad $msg
                    Step-Fail 'cameras' $msg
                } else {
                    Bad 'cameras found: 0. Check that the box is on the same network as the cameras.'
                    Step-Warn 'cameras' 'none found - the box may not be on the cameras network'
                }
            } catch {
                Bad 'Could not read the camera result. What the box answered:'
                Write-Host ($out | Out-String)
                Step-Warn 'cameras' 'could not read the camera result'
            }
        }
    } finally {
        Invoke-Box "cmd /c del $camRemote" | Out-Null
        Remove-Item $camLocal -ErrorAction SilentlyContinue
    }
} else {
    Step-Skip 'cameras' 'not requested'
}

# ---- mode / AI alerts -------------------------------------------------------
Step-Start 'alerts'
$SetOpt = "cd /d $InstallDir && $Python -m home_guard_project.box set-option"
if ($UseAI) {
    Info "`n[5] Turning on AI alerts..."
    # Secrets come from this laptop's ~/.homeguard store; they go to the box's
    # api_key.env as a file, never on a command line.
    $oaiKey = ''; $tgTok = ''; $tgChats = ''
    $oaiFile = Join-Path $HgDir 'openai.env'
    $tgFile  = Join-Path $HgDir 'telegram.env'
    if (Test-Path $oaiFile) { $m = Select-String '^OPENAI_API_KEY=(.+)$'    $oaiFile; if ($m) { $oaiKey  = $m.Matches[0].Groups[1].Value } }
    if (Test-Path $tgFile)  { $m = Select-String '^TELEGRAM_BOT_TOKEN=(.+)$' $tgFile;  if ($m) { $tgTok   = $m.Matches[0].Groups[1].Value } }
    if (Test-Path $tgFile)  { $m = Select-String '^TELEGRAM_CHAT_IDS=(.+)$'  $tgFile;  if ($m) { $tgChats = $m.Matches[0].Groups[1].Value } }
    if (-not $oaiKey -or -not $tgTok -or -not $tgChats) {
        Bad "AI alerts need OpenAI + Telegram configured on this laptop first (missing in $HgDir). Leaving the box in data-collection mode."
        Invoke-Box "$SetOpt mode data_collection" | Out-Null
        Step-Warn 'alerts' 'OpenAI/Telegram not configured on this laptop; left in data-collection mode'
    } else {
        $apiLocal = [IO.Path]::GetTempFileName()
        [IO.File]::WriteAllText($apiLocal, "OPENAI_API_KEY=$oaiKey`nTELEGRAM_BOT_TOKEN=$tgTok`n", (New-Object Text.UTF8Encoding($false)))
        try { Copy-ToBox $apiLocal "$InstallDir\api_key.env" } finally { Remove-Item $apiLocal -ErrorAction SilentlyContinue }
        Invoke-Box "$SetOpt mode inference"                | ForEach-Object { Note "    $_" }
        Invoke-Box "$SetOpt alert_start_hour $AlertStart"   | Out-Null
        Invoke-Box "$SetOpt alert_end_hour $AlertEnd"       | Out-Null
        Invoke-Box "$SetOpt alert_cooldown_sec $AlertCooldown" | Out-Null
        Invoke-Box "$SetOpt alert_channel telegram"         | Out-Null
        Invoke-Box "$SetOpt notify_dry_run false"           | Out-Null
        Invoke-Box "$SetOpt telegram_chat_ids=$tgChats"     | Out-Null
        # Restart so run_collector.sh picks up inference mode.
        Invoke-Box "$Bash -lc /c/home_guard/home_guard_project/box/stop_collector.sh" | Out-Null
        Invoke-Box 'schtasks /Run /TN HomeGuard-Collector' | Out-Null
        Ok "AI alerts ON: a person seen between ${AlertStart}:00 and ${AlertEnd}:00 sends a Telegram alert with a photo."
        Step-Ok 'alerts' "inference ${AlertStart}-${AlertEnd}, cooldown ${AlertCooldown}s"
    }
} else {
    Invoke-Box "$SetOpt mode data_collection" | ForEach-Object { Note "    $_" }
    Ok 'AI alerts off: the box collects clips for tagging (data-collection mode).'
    Step-Ok 'alerts' 'data-collection mode'
}

# ---- readiness report -------------------------------------------------------
Step-Start 'readiness'
Info "`n[6] Readiness"
$readyOut = (Invoke-Box "powershell -ExecutionPolicy Bypass -File $BoxBox\check_box.ps1" | Out-String)
Write-Host $readyOut
if ($NonInteractive) {
    foreach ($line in ($readyOut -split "`r?`n")) {
        if ($line -match '^\s*\[(PASS|WARN|FAIL)\]\s+(.*)$') { Emit "@@check $($matches[1]) $($matches[2].Trim())" }
    }
}
Step-Ok 'readiness' 'report complete'

# ---- summary ----------------------------------------------------------------
Write-Host ''
Info '=== Still to do by hand ==='
Write-Host ' 1. BIOS: set "restore on AC power loss" to Power On (needs a screen once).'
Write-Host ' 2. Tailscale admin console: disable key expiry for this box.'
if (-not $DoCameras) { Write-Host ' 3. Camera discovery at the customer site (run again and find cameras).' }
Write-Host ' *. Before delivery, run again with -ForgetOtherWifi to drop your own Wi-Fi.'
Write-Host ''
Info 'Rescue hotspot (make a phone hotspot with these to recover a box):'
Write-Host "   name:     $RescueSsid"
Write-Host "   password: $RescuePass"
if ($DryRun) { Write-Host ''; Note 'DryRun: nothing on the box was changed.' }

if ($NonInteractive) {
    if ($script:StepFailed) { Emit '@@done fail'; exit 1 }
    Emit '@@done ok'
    exit 0
}
Wait-BeforeClose
