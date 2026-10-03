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
4. In the BIOS, set "restore on AC power loss" (or similar) to **Power On**, so the box starts by itself after a power cut.

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
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json set-zone --camera yard --points 0.1,0.2;0.9,0.2;0.9,0.9;0.1,0.9
.venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json clear-zone --camera yard
```

`set-zone` / `clear-zone` ask the running mode to restart so the change is live within seconds.
Renaming a camera carries its zone along; disabling keeps it.

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
