# Collector Box

Turns a Windows mini PC (built for a Beelink Mini S13, Intel N150) into an unattended data-collection box for one house. After setup, power and Ethernet are all it needs: it connects to that house's cameras, saves trigger clips with the normal `data_collection` pipeline, and uploads them to S3 for tagging.

## What runs on the box

| Scheduled task | When | Script | What it does |
|---|---|---|---|
| `HomeGuard-Collector` | at boot | `run_collector.sh` | Runs the collector headless and restarts it if it exits |
| `HomeGuard-Upload` | every 15 minutes | `run_upload.sh` | Moves finished clips to `dataset_outbox/<site>/`, uploads them to `s3://<bucket>/dataset_<site>/`, deletes the local copies |
| `HomeGuard-Heartbeat` | hourly | `run_heartbeat.sh` | Writes `dataset_<site>/_status/heartbeat.json` to S3 |

The collector uses `config.box.yaml` on top of `data_collection/config.yaml`: no preview windows, no VLM, one random background clip per hour.

Logs are in `logs/` at the project root: `runner.log`, `collector-<date>.log`, `upload-<date>.log`, `heartbeat.log`.

## Desktop screen

The Home Guard desktop shortcut now opens a single PySide6 window. It shows the
house, background collector state, clips waiting, disk space, recent activity and
camera pictures when `show_cameras` is enabled. Closing it leaves the collector
running. If the GUI cannot start, the original console/live-camera screen remains
available as a fallback.

```powershell
.venv\Scripts\python.exe -m home_guard_project.box.app
.venv\Scripts\python.exe -m home_guard_project.box.app --demo
.venv\Scripts\python.exe -m home_guard_project.box.app --setup --demo --skip-cameras
```

The setup screens currently use only a simulated backend; the existing real
PowerShell wizard and compiled executable are unchanged. The box collector
publishes demand-driven local previews, with no extra camera connection or
inference process in the GUI. See [UI handoff](../../docs/ui/HANDOFF.md) and the
[screenshot gallery](../../docs/ui/SCREENSHOTS.md) for review and hardware checks.

## Setting up a new box

### 1. First boot (at the box, with monitor and keyboard)

1. Finish Windows setup, connect Ethernet, run Windows Update until nothing is left.
2. Install Tailscale and sign in with the same account as the laptop.
3. **Disable key expiry for this box** (required for lasting remote access). In the Tailscale admin console — [login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines) — find this box, open its **⋯** menu, and choose **Disable key expiry**. Without this, Tailscale re-authentication expires (~every 6 months by default) and the box silently drops off the network — and you'd need physical access at the customer's house to sign it back in. (`setup_box.ps1` already turns on unattended mode so it reconnects at boot; key expiry is the one thing that can only be set here in the web console, so do it now while you have the box in front of you.)
4. In the BIOS (press **Delete** at power-on), set **State After G3** (on other boards: "Restore on AC Power Loss") to **S0 State / Power On**, so the box starts by itself when the power comes back. On Beelink boards it is usually under *Chipset → PCH-IO Configuration*. Check it on every new model: pull the plug, plug it back, and the box must start without the button.

#### After a power cut, with nobody at the box

| What | How | Needs a sign-in? |
|---|---|---|
| Box turns on | BIOS setting above | - |
| No recovery screen | `setup_box.ps1` (bcdedit) | - |
| Wi-Fi joins again | all-users, auto-connect profile (`setup_network.ps1`) | no |
| Alert program starts | `HomeGuard-Collector` task at boot, session 0, like a service; restarts it forever, and the hourly heartbeat starts the task again if it died | no |
| Uses the iGPU | OpenVINO works in session 0 (checked 2026-10-06: 89 ms a frame next to the running program) | no |
| On/off state kept | **Stop** writes `logs/collector.stop`; it survives reboots, so a stopped box stays stopped and a running box starts again | no |
| Home Guard window | Startup shortcut, opens after the automatic sign-in from `enable_autologon.ps1` (step 7 of `setup_box.ps1`) | automatic |

`enable_autologon.ps1` asks for the Windows password once and keeps it like Sysinternals Autologon does (an LSA secret, not plain text in the registry). `-Off` undoes it. `check_box.ps1` warns while it is off.

### 2. Let the laptop in (once)

On the laptop:

```bash
./home_guard_project/box/prepare_remote.sh
```

This creates an SSH key and writes `dist/enable_remote.ps1`. Get that file onto the box (Tailscale file transfer or a USB stick), then on the box open PowerShell with **Run as administrator** and run:

```
powershell -ExecutionPolicy Bypass -File "$HOME\Downloads\enable_remote.ps1"
```

It prints the user name and IP. From then on the laptop can log in:

```bash
ssh -i ~/.ssh/homeguard_box <user>@<box-tailscale-ip>
```

### 3. Install the code and run setup

The install on the box is a git checkout, so later updates are a `git pull`. On the box, in an elevated PowerShell (or from the laptop over SSH):

```
git clone -b beelink-collector-box https://github.com/ameerfaour95/home_guard.git C:\Users\<user>\Desktop\home_guard
powershell -ExecutionPolicy Bypass -File C:\Users\<user>\Desktop\home_guard\home_guard_project\box\setup_box.ps1 -Site house2
```

`-Site` is the house name used in the S3 prefix: lowercase letters, digits and underscores only.

**Updating a box later:** in Git Bash on the box, `./home_guard_project/box/update.sh` (stops the collector, pulls, refreshes the environment, starts it again). `cameras.yaml`, `box.yaml`, the datasets and the logs are not in git and are left alone.

**Without git:** `uv run python -m home_guard_project.box.make_bundle` on the laptop writes `dist/home_guard_box.zip` with only what the box needs (never `cameras.yaml`, API keys, or scripts that hold camera passwords). Copy it to the box, unpack it, and run `setup_box.ps1` from there.

### 4. AWS key for the box

Give the box its own IAM user, not your personal keys. On the laptop:

```bash
uv run python -m home_guard_project.box.make_box_key house2
```

This creates the IAM user `homeguard-box-house2`, limits it to its own S3 folder, and writes `credentials` and `config` to `~/.homeguard/keys/house2/`. Copy both files to `C:\Users\<user>\.aws\` on the box and delete them from the laptop. The policy it applies, for site `house2`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::security-camera-project-v1",
      "Condition": { "StringLike": { "s3:prefix": "dataset_house2/*" } }
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::security-camera-project-v1/dataset_house2/*"
    }
  ]
}
```

By hand, the key goes in `C:\Users\<user>\.aws\credentials` on the box:

```
[default]
aws_access_key_id = ...
aws_secret_access_key = ...
```

and `C:\Users\<user>\.aws\config`:

```
[default]
region = us-east-1
```

### 5. Connect the box to the house network: two options

The box only has to be on the same network as the cameras or their recorder. Everything else (finding cameras, collecting, uploading) is the same either way.

| | Option A: Ethernet cable | Option B: Wi-Fi |
|---|---|---|
| Setup | None. Plug a cable from the box into the router, mesh unit or recorder switch | The box must know the Wi-Fi name and password (see below) |
| Reliability | Best. Use this when a cable can reach | Works, but video arrives with more damaged frames and the link can drop |
| When both are connected | Windows uses the cable | |

For Wi-Fi, the box has to be told the network name and password once, before it is moved. The setup wizard does this (see "Customer setup" below): it saves the customer's Wi-Fi on the box so the box joins it by itself at boot with nobody logged in, and rejoins after a drop. The script it runs on the box is `setup_network.ps1`.

The wizard also saves a rescue hotspot on every box. If the customer's Wi-Fi password turns out to be wrong or is changed later, either plug in a cable, or turn on a phone hotspot with the rescue name and password next to the box; the box joins it and can be fixed remotely.

If the house has more than one network (for example the internet provider's router and a separate mesh system), the box must be on the one the cameras use.

### 6. Find the cameras (once per house)

You need the camera or recorder username and password for that house.

**Without prompts** (works over SSH, and on Windows 11 Home which has no Remote Desktop):

```
cd /d <repo folder>
set HG_CAMERA_PASSWORD=...
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras auto --user admin --prefix house2 --write
```

`auto` scans the network, logs in to every camera device it finds with that one login, and with `--write` saves all channels as `house2_ch1`, `house2_ch2`, ... so clip names never collide between houses. `scan` alone needs no password and only lists the devices; `probe --host <address>` does one device. Add `--json` to any of them to get the result as JSON, for a setup program to read.

`auto` knows the common recorder address formats. For a brand it does not recognise, use the interactive tool below, which can also ask the device itself (ONVIF).

**With prompts and a live preview** (needs a screen), in Git Bash:

```bash
cd /c/home_guard
uv run python home_guard_project/data_collection/discover.py
```

Either way the result is `data_collection/cameras.yaml`. The collector task notices it within a minute and starts collecting; no restart is needed. Edit that file to remove indoor cameras the household does not want recorded.

A recorder's address can change when the router restarts. If the cameras stop connecting, run `scan` again and correct the address in `cameras.yaml`, or reserve the address in the router.

## Customer setup (from the laptop)

After a box has had its first setup (steps 1 to 3), preparing it for a customer is one program on the laptop:

```
dist\HomeGuardSetup.exe
```

or, without the compiled program:

```
powershell -ExecutionPolicy Bypass -File home_guard_project\box\setup_customer.ps1 -Target <user>@<box-tailscale-ip>
```

It asks whether the box will use Ethernet or Wi-Fi (and for Wi-Fi the customer's network name and password), the house name, whether to show the cameras on the box's own screen, whether to turn on AI alerts (and during which hours), and optionally the camera login. Then it:

1. Updates the box's software (`update.sh`).
2. Configures the network on the box (`setup_network.ps1`).
3. Finds the cameras (`find_cameras auto`) if a camera login was given.
4. Prints a readiness report (`check_box.ps1`: PASS / WARN / FAIL) and the rescue hotspot name and password.

| Option | Meaning |
|---|---|
| `-ForgetOtherWifi` | Remove every saved Wi-Fi network except the customer's and the rescue hotspot. Run this last before delivery: it removes your own Wi-Fi from the box, and takes the box offline if it is on that Wi-Fi |
| `-SkipUpdate` | Do not update the box's software first |
| `-KeyPath <path>` | SSH key to use (default `~\.ssh\homeguard_box`) |
| `-DryRun` | Show the planned steps without touching the box |

Passwords are sent in a temporary file that is deleted from the box afterwards. The rescue hotspot name and password are kept on the laptop in `~/.homeguard/rescue_wifi.txt`.

Build the program with `powershell -ExecutionPolicy Bypass -File home_guard_project\box\build_exe.ps1` (writes `dist\HomeGuardSetup.exe`).

### Watch zones: look only at part of a camera's picture

A camera can be given a zone (for example the yard, not the street). Everything outside it is
blacked out as the frame is read, so no detector, AI call, clip, snapshot or preview ever contains
it. The zone is a polygon in picture fractions (0–1), 3 to 32 corners, kept in
`home_guard_project/data_collection/zones.yaml` on the box (not in git). It applies in both modes.

Normally the owner draws it in the setup app (camera page → "Set the area to watch"). By hand, on
the box:

```
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json zones
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json set-zone --camera yard --points "0.1,0.2;0.9,0.2;0.9,0.9;0.1,0.9"
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json clear-zone --camera yard
```

Quote the corners as shown: in PowerShell an unquoted `;` ends the command. The setup app sends them without quotes over SSH, where the box's shell is cmd.exe.

`set-zone` / `clear-zone` ask the running mode to restart so the change is live within seconds.
Renaming a camera carries its zone along; disabling keeps it.

### What alerts the owner: people, vehicles, animals

In inference mode the owner chooses what sends an alert. The house default is `alert_on` in
box.yaml (unset = people only); a camera can have its own choice, kept in
`home_guard_project/data_collection/camera_alerts.yaml` on the box (not in git). Both apply
within seconds, without a restart. A vehicle alerts only when it moves (parked cars never do);
animals are cats, dogs and other animals, not birds. Whatever is not chosen never wakes the AI;
if the AI sees only something not chosen, the clip is kept for training and nothing is sent.

```
.venv\Scripts\python.exe -m home_guard_project.box set-option alert_on=person,vehicle
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json camera-alerts
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json set-camera-alerts --camera driveway --on person,vehicle
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json set-camera-alerts --camera driveway --default
```

Each type also has its own detector certainty, like Frigate's per-object thresholds: `conf_person`,
`conf_vehicle`, `conf_animal` in box.yaml (0.05-0.95; unset = `inference_conf`), and a camera can
override some or all of them. A lower value for people than for cars means a half-hidden person is
still caught while a car needs to be clear. The detector runs at the lowest value in use; each find
is then held to its own type's value.

```
.venv\Scripts\python.exe -m home_guard_project.box set-option conf_person=0.5
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json set-camera-sensitivity --camera driveway --values person=0.5,vehicle=0.8
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json set-camera-sensitivity --camera driveway --default
```

Renaming a camera carries its choices along. Design: `docs/superpowers/specs/2026-10-03-alert-types-design.md`.

## Registration with the Admin Center

The setup program asks for the owner's name, phone (optional) and three consents (live cameras, saved recordings, training), all default no, and runs the `register` step on the box. The box saves them in `home_guard_project/box/registration.json` (gitignored, never bundled) and publishes `dataset_<site>/_status/registration.json` next to the heartbeat, so the customer appears in the Admin Center by itself.

```bash
python -m home_guard_project.box register --owner "Dana Cohen" --phone +972...     --consent-live yes --consent-recordings no --consent-training no --installer Ameer
python -m home_guard_project.box register --from-json answers.json   # what setup uses; the file is deleted after reading
python -m home_guard_project.box status                              # "registration": registered + consents (no name or phone)
```

- CLI field updates preserve omitted fields; a changed consent or installer updates `consent.recorded_utc`. Setup/`--from-json` replaces owner details, including clearing a blank phone; missing permissions default off.
- The graphical Owner & consent page follows Home. The owner name is required (1-120 characters); the installer name alone is remembered locally. Owner details never appear in setup logs, SSH arguments, or heartbeat data.
- A failed publish only prints a warning (exit 0). The hourly upload retries; it sends the file only when its content changed (`logs/registration.published` holds the hash).
- `set-site` moves the registration to the new site and publishes it under the new folder; the old folder keeps its copy. The next heartbeat repairs an interrupted local site update before publishing.
- Re-running setup on an already registered box is safe. An older answers file without these fields registers with all consents off and the house name as the owner.
- Manual check of the setup step (no box needed): `powershell -File setup_customer.ps1 -AnswersFile a.json -DryRun -SkipUpdate` shows `@@step register ok`.

## Telegram alerts for a new customer

AI alerts are delivered over Telegram: it is free, and one message reaches a whole family group (so every family member is covered at no extra cost). **One bot serves every box; each customer gets their own group.**

Set up once on the laptop (already done for the founder — kept in `~/.homeguard/telegram.env`): a bot created with [@BotFather](https://t.me/BotFather) (`/newbot`). Its token is the `TELEGRAM_BOT_TOKEN` line in that file. The OpenAI key (for the AI descriptions) is in `~/.homeguard/openai.env`.

For each new customer:

1. **Create a Telegram group** for that home (for example "Cohen — Home Guard") and **add the bot to it** (search the bot's username, e.g. `@homeshield_ameer_bot`, → Add to group). Put the whole family in the group — everyone then gets the alerts.
2. **Make the bot a group admin.** ⚠️ **Required for the assistant to work.** Group info → **Administrators → Add Admin → pick the bot** (default rights are fine). Telegram's privacy rule hides ordinary group messages from a bot, so without this the family can tap the alert buttons and *reply* to an alert, but a **plain message they type won't reach the assistant at all** (no reply, no feedback saved). Making the bot an admin lets it read every message in the group. (Alternative: in [@BotFather](https://t.me/BotFather), `/mybots` → the bot → Bot Settings → Group Privacy → **Turn off**, then remove and re-add the bot — admin is the one-step way.)
3. **Send any message in the group** so the bot can see it.
4. **Find the group's chat id.** Open this in a browser, replacing `<TOKEN>` with the bot token from `~/.homeguard/telegram.env`:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
   Look for `"chat":{"id":-100...}` — a group id is a **negative** number.
5. **Point this box at that group.** Either edit `TELEGRAM_CHAT_IDS=` in `~/.homeguard/telegram.env` before running the wizard, or set it on the box afterwards:
   ```
   ssh -i ~/.ssh/homeguard_box <user>@<box-ip> "cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box set-option telegram_chat_ids=<id>"
   ```
   For several recipients, use comma-separated ids with no spaces (e.g. `-1001111,-1002222`).
6. **Turn alerts on in the wizard:** answer **y** to "Turn on AI alerts?" and give the hours. The wizard pushes the bot token and the OpenAI key onto the box (into `api_key.env`) and switches it to inference mode.

**Test it:** during the alert hours, have someone walk in front of a camera — a Telegram message with a photo should arrive in the group within a few seconds. Then have the family **type a message in the group** ("was that the mailman?") and check the assistant answers — if it doesn't, the bot isn't an admin (step 2). Everything the AI sees is also written to the box's log.

**Verify it without waiting:** `check_box.ps1` includes a Telegram row that confirms the bot can actually read messages (it calls `telegram_check`, which checks privacy/admin status — never `getUpdates`, so it won't disturb the live poller). A `[WARN] Telegram: NOT_READY` line means the bot still needs to be made an admin.

A customer who wants their **own** bot (not the shared one) creates one with @BotFather and you use that token for their box instead.

> Note: inference (AI) mode and data-collection mode are mutually exclusive — a box with AI alerts on does not also save training clips.

## Bench test before moving the box

Run it at your own house for a day first:

1. Restart the box. Within about two minutes `logs/runner.log` should show "Starting collector" without anyone logging in.
2. Walk in front of a camera. A clip should appear under `dataset_multi/clips/`.
3. Pull the power cable and plug it back in. The box should boot and collect again.
4. Unplug the network for ten minutes. `collector-<date>.log` should show the cameras reconnecting.
5. Run an upload by hand: `./home_guard_project/box/run_upload.sh`. Clips older than ten minutes should appear on S3 and disappear locally.
6. Note CPU and memory use in Task Manager with all cameras connected.

## Measured on the Beelink N150 (2026-10-02)

YOLO11s on the CPU takes about 327 ms per frame, so about 3 detections per second in total, shared between all cameras. That is comfortable for two or three cameras. With six or more, each camera is checked only about once every two seconds.

The collector uses the whole processor however many cameras there are. What matters is leaving room for reading the video. With 6 cameras over Wi-Fi, 150 seconds per setting, while Windows Update was also running:

| Setting | Damaged-frame messages per minute |
|---|---:|
| YOLO11s, 4 threads | 114 |
| YOLO11s, 3 threads (what the runner uses) | 23 |
| YOLO11n, 3 threads | 0 |

The runner therefore gives detection one core fewer than the machine has. If clips still show damaged frames, switch the box to the smaller model by adding `models: { yolo: "yolo11n.pt" }` to `config.box.yaml`; it finds people less reliably, so its boxes need more correcting when tagging.

## Checking the box from the laptop

```bash
uv run aws s3 cp s3://security-camera-project-v1/dataset_house2/_status/heartbeat.json -
```

`time_utc` should be less than an hour old, `collector_running` should be `true`, and `newest_clip_utc` tells you when a clip was last saved.

On the box itself, without network: `uv run python -m home_guard_project.box status`.

## Scoring the AI's prompt against our tags

`eval_prompt.py` gives one repeatable score for the prompt the box sends to the AI, over a frozen set of clips with our human labels. The current set is `home_guard_eval/eval_set_v2` on the laptop (see "The frozen eval set" below).

**1. On the laptop: build the set.** `prepare` reads `home_guard_dataset` (its `annotations/clips.jsonl` and `clips/`; a local folder, or `s3://security-camera-project-v1/home_guard_dataset` with the same paths; taken from `--dataset`, else `$HOMEGUARD_DATASET_DIR`, else `C:\Users\ameer\Ameer\home_guard_data\dataset`) and saves 5 frames spaced evenly across every reviewed clip. Only `dataset_row()` in `eval_prompt.py` reads the dataset's fields, so a renamed field is fixed in one place. Re-running it reads only what is missing and keeps what the eval added on a clip (`category`, `subset`, `day_night`, `hard`); `--sources` and `--batches` narrow the clips, `--limit N` reads at most N new ones. `add --picks picks.jsonl` adds clips from outside the dataset, with 5 frames evenly spaced inside an annotated `segment` (the same sampling and size). `freeze` writes `FROZEN.json` (sha256 of the manifest and every frame, one set hash and the set's make-up); after that, `prepare` and `add` refuse the folder.

```bash
.venv/Scripts/python.exe -m home_guard_project.box.eval_prompt prepare --out eval_set   # or --dataset <root>
env -u SSLKEYLOGFILE -u PYTHONSTARTUP AWS_CA_BUNDLE=<bundle.pem> \
  .venv/Scripts/python.exe -m home_guard_project.box.eval_prompt add --dir eval_set --picks picks.jsonl
.venv/Scripts/python.exe -m home_guard_project.box.eval_prompt freeze --dir eval_set
```

Python can only reach S3 from this laptop with that prefix. The antivirus sets `SSLKEYLOGFILE`, which crashes Python's TLS ("no OPENSSL_Applink"), and it intercepts TLS with a root certificate that is in the Windows store but not in Python's own list ("CERTIFICATE_VERIFY_FAILED"). Build `bundle.pem` once from certifi's bundle plus the Windows `ROOT` and `CA` stores (`ssl.enum_certificates`, converted with `ssl.DER_cert_to_PEM_cert`) and point `AWS_CA_BUNDLE` at it. Never turn certificate checks off. The box does not need any of this.

**2. Copy the folder to the box.**

```bash
scp -i ~/.ssh/homeguard_box -r eval_set <user>@<box-ip>:C:/home_guard/eval_set
```

**3. On the box: ask the AI and print the score.** It uses the OpenAI key in `api_key.env` and costs one call per clip. Each clip is told its own time of day, taken from its file name, so the score does not depend on when you run it.

```bash
.venv\Scripts\python.exe -m home_guard_project.box.eval_prompt run --dir eval_set
.venv\Scripts\python.exe -m home_guard_project.box.eval_prompt run --dir eval_set --prompt-file new_prompt.txt
.venv\Scripts\python.exe -m home_guard_project.box.eval_prompt summary --dir eval_set --tag <tag>
```

If the folder is frozen, `run` first checks every hash: a changed set prints a loud warning (its scores do not compare with the frozen ones), and `--strict-frozen` refuses with exit code 4, before any call, when the set changed or was never frozen. The summary names the set (`eval set  frozen <sha12>`).

`--prompt-file` tries a new wording without touching `inference.py`; `{camera_name}`, `{local_time_str}` and `{owner_language}` (default `en`) in the file are filled in. Results go to `eval_set/results/<tag>.jsonl`, `.csv` and `.summary.json`, named by the tag. A stopped run continues where it left off, and a clip that failed is asked again. `--fake` checks the setup without calling the AI.

The `.jsonl` holds the AI's answers only, and answers are only ever appended. The score, the `.csv` and the `.summary.json` are rebuilt every time from the last answer per clip and the tags in the current `manifest.jsonl`. So after re-tagging clips, re-run `prepare` and then `run` or `summary`: the new tags are scored without any new AI call. Each answer also records a fingerprint of what the AI saw (camera name, the clip's time of day, the frame files). If any of these changed, that clip is asked again; the old answer stays in the file. Answers for clips that left the manifest are kept, just not scored. Only one run at a time may write a results file: a second run on the same tag exits with code 3 before asking anything. If a run crashed and left `results/<tag>.lock` behind, the message says so; delete the lock and run again.

Paid answers are never deleted silently. The default tag is `<prompt version or file-hash>__<model>`, so a second model gets its own file. Every row also stores a hash of the prompt wording, so editing the prompt in `inference.py` without bumping its version is noticed. If a results file (usually one picked with `--tag`) holds answers from another prompt, wording or model, `run` prints which and exits with code 2 without touching it; pass `--overwrite` to replace them. A cut-off last line left by a killed run is skipped with a warning.

Before the first call, `run` prints `asking N clips (M already answered, K with errors to retry)`. For a real run of more than 20 clips it also names the model and waits 5 seconds (Ctrl+C aborts); `--yes` skips the wait.

**Caveat: the eval sees the whole clip.** The box sends the AI the last 5 buffered frames, 1 second apart, up to the trigger, with the camera's zone mask applied. The eval takes 5 frames evenly across the whole tagged clip, unmasked. So alert recall reads somewhat optimistic compared with the box. Frames are also JPEG-encoded twice (quality 90 on disk, 85 when sent); the effect is negligible.

What the score means:

| Line | Meaning |
|---|---|
| alerts caught | Clips we tagged `[alert]` that the AI called suspicious or escalation |
| escalation share | Of those, how many it called escalation |
| normal flagged | Ordinary clips the AI called suspicious or escalation (false alarms) |
| empty exact | Empty scenes where the AI wrote exactly "No special activity." |
| padding | Summaries that mention what is absent or the background ("without", "no one", "visible", "background", "parked") |
| words per summary | The AI's length against ours |
| errors | Clips the AI could not answer; they are left out of the other lines |
| outdated | Shown only when some saved answers were for other frames, camera or time (not yet asked again); they are left out |
| day / night, home / external | The same counts split by the clip's time (night 19:00-05:59; the manifest's `day_night` for clips without a time) and by where it comes from (our own cameras vs everything else, including the web videos imported into a home batch) |
| misses set | The same counts on the hard cases only (`subset: misses`: night, partial occlusion, subtle attempts, loitering) |
| missed alerts, false alarms | The clip names, to look at |
| tokens per call, cost | What the provider billed, and $ per box per month at 150 and 300 calls a day (hosted models only) |

### Comparing vision models

Run every model on the frozen set (`eval_set_v2`, see below) with `--strict-frozen`, so all the results in one table were made on the same frames.

Small models run on the laptop GPU through Ollama (no key; `ollama pull <model>` first, and start the server with `OLLAMA_CONTEXT_LENGTH=8192`, because 5 frames are about 6,000 tokens); hosted ones through OpenRouter (`OPENROUTER_API_KEY` in `api_key.env`). On the laptop, Python's TLS needs the antivirus workaround: `env -u SSLKEYLOGFILE -u PYTHONSTARTUP SSL_CERT_FILE=<bundle.pem>`.

    .venv\Scripts\python.exe -m home_guard_project.box.eval_prompt run --dir <eval_set> --provider ollama --model qwen3-vl:4b-instruct-bf16 --yes
    .venv\Scripts\python.exe -m home_guard_project.box.eval_prompt run --dir <eval_set> --provider openrouter --model qwen/qwen3-vl-32b-instruct --yes
    .venv\Scripts\python.exe -m home_guard_project.box.eval_prompt compare --dir <eval_set> --default <Qwen3-VL-4B tag> --challenger <Qwen3.5-4B tag> --others <tags...> --reference <gpt-4o tag>

`compare` prints one table and names the box's primary and fallback: Qwen3-VL-4B-Instruct primary and Qwen3.5-4B fallback, swapped only if Qwen3.5-4B catches more alerts (or as many with fewer false alarms) without newly missing a break-in and without more errors.

### The frozen eval set

`home_guard_eval/eval_set_v2` on the laptop, frozen 2026-10-06: 489 clips, 181 alerts (4 from our own cameras), 281 normal, 27 empty; a misses set of 108 hard alerts (night, partial occlusion, subtle attempts, loitering). Set sha256 `2054eca82da2...`. How it was built, the fields, the category mapping for the public datasets and the known limits are in `docs/eval/eval-set-v2.md`; what to film at home for the next set is in `docs/eval/staged-clips-shot-list.md`. The clips added in v2 (`picks_v2.jsonl`, batches `uca_eval_v2` and `smarthome_eval_v2`) are test data: keep them out of training.

### The owner's verdicts (eval_set_owner)

`prepare-owner --dir <folder>` builds a separate small set from the alerts the owner judged on Telegram (`owner_feedback/feedback_index.jsonl` in the dataset; `--dataset`, else `$HOMEGUARD_DATASET_DIR`, else the default). Each judged alert with a saved clip becomes one row. Its truth comes from the owner's latest verdict: a tag of suspicious or escalation is an alert, normal or empty is normal, `true_alert` is an alert, and `expected` or `false_alarm` is normal. `false_alarm` counts as normal rather than empty, so it shows up in "normal flagged". A `real_but_wrong` tagged `other` says the description was wrong, not the label: an earlier verdict on the same alert decides, otherwise the box's own label is kept (`truth_from` records which). The frames are the box's own 1-a-second sampling of the crop the AI saw (`vlm_input: crop`), otherwise of the alert clip, so a row has as many frames as the box sent (7 to 12). Each row also keeps the camera, the clip time, the box's label at the time and the owner's words. Freeze it like the main set. The first build (2026-10-06): 10 clips, 1 alert and 9 normal (all 9 are false alarms the box raised), set sha256 `f1c841bb78d7...`. It is too small to score on its own; read it next to `eval_set_v2`.

### The box's vision model settings (box.yaml)

    vlm_provider: vllm                     # openai | openrouter | ollama | vllm | dashscope-intl
    vlm_model: Qwen/Qwen3-VL-4B-Instruct
    vlm_fallback_provider: vllm            # empty: no fallback
    vlm_fallback_model: Qwen/Qwen3.5-4B

`vllm` needs `VLLM_BASE_URL` (and `VLLM_API_KEY` if the server has one) in `api_key.env`. If the main model fails, the same alert goes to the fallback once (`VLM fallback` in the log). Without these settings the box keeps gpt-4o.

### Alerts in Hebrew (box.yaml)

    owner_language: he
    owner_translation: translator          # the default; model: the vision model writes the Hebrew itself
    messenger_provider: openrouter         # openrouter | openai | google (GEMINI_API_KEY) | ...
    messenger_model: google/gemini-3.1-flash-lite
    messenger_timeout_sec: 4

With `translator`, a cheap text model (`messenger.py`; Gemini 3.1 Flash Lite, about $0.0003 an alert) translates the alert's summary and "why" in one call, keeping camera names, numbers and times. It never holds an alert longer than the timeout: on any failure the owner reads the vision model's own Hebrew when it wrote one, else the English (`Translation to he failed` in the log). Read at start. To judge the translation, `python -m home_guard_project.box.eval_translation <meta folder or eval results .jsonl> --limit 50` writes `translation_eval.csv` with Gemini Flash Lite and gpt-6-luna side by side (`--fake` checks the setup without calling anyone).

### The situational Eye (box.yaml)

    eye_prompt: situational                # the default; legacy: the 2026-10-03 prompt
    camera_roles: {front_side: street, left_side_1: private}   # optional; else guessed from the name
    camera_zones: {main_door: [entrance, gate]}                # optional

With `situational` the guard loop asks the vision model with `eye_prompt.py`: the fixed categories (`taxonomy.py`), one `SITUATION:` line (time, day/evening/late_night/dawn, dark, house state, camera role, what the owner expects) and what that situation means. The model names what it sees; code turns it into the label (a visitor at 02:30 is suspicious, at 14:00 normal), never below the model's own label and never softening escalation. The house state (awake / asleep / away / vacation, and "expecting" notes) lives in `production_multi/.registry/house_state.jsonl` (`house_state.py`); without commands the house is asleep 00:00-06:00. Restart after changing these. Score it first with `run --prompt eye` (each clip in its own situation, scored per category and per situation).

## Troubleshooting

| Problem | What to do |
|---|---|
| No "Starting collector" in `runner.log` | `cameras.yaml` is missing (run discovery), or the task is not running: `schtasks /Query /TN HomeGuard-Collector` |
| Collector starts but no clips | Read `collector-<date>.log`. "detecting" lines mean it sees people; none means the cameras show nobody, or are not connected |
| High CPU, detections lag | In `config.box.yaml` add `detection: { yolo_every_n_frames_cpu: 10 }`, or `models: { yolo: "yolo11n.pt" }`, then restart the task |
| Upload log shows credential errors | Check `.aws\credentials` on the box and the IAM policy prefix |
| Clips stay in `dataset_outbox/` | The last upload failed; the next run retries. Read `upload-<date>.log` |
| Restart the collector | `schtasks /End /TN HomeGuard-Collector` then `schtasks /Run /TN HomeGuard-Collector`. The new runner stops the old one |
| Stop collecting completely | `schtasks /End /TN HomeGuard-Collector`, then in Git Bash `./home_guard_project/box/stop_collector.sh`. Ending the task alone leaves the runner alive |
| Box drops off Tailscale after a reboot | Tailscale must be in unattended mode (`tailscale set --unattended=true`, done by `setup_box.ps1`), and key expiry should be disabled for the box in the Tailscale admin console |
| Change how often clips upload | Re-run `setup_box.ps1 -Site <site> -UploadEveryMinutes 30` |

## Privacy

The box uploads full clips from someone else's home. Get that household's agreement first, and agree with them which cameras are included.
