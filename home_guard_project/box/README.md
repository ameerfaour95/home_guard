# Collector Box

Turns a Windows mini PC (built for a Beelink Mini S13, Intel N150) into an unattended data-collection box for one house. After setup, power and Ethernet are all it needs: it connects to that house's cameras, saves trigger clips with the normal `data_collection` pipeline, and uploads them to S3 for tagging.

## What runs on the box

| Scheduled task | When | Script | What it does |
|---|---|---|---|
| `HomeGuard-Collector` | at boot | `run_collector.sh` | Runs the collector headless and restarts it if it exits |
| `HomeGuard-Upload` | nightly, 03:00 | `run_upload.sh` | Moves finished clips to `dataset_outbox/`, uploads them to `s3://<bucket>/dataset_<site>/`, deletes the local copies |
| `HomeGuard-Heartbeat` | hourly | `run_heartbeat.sh` | Writes `dataset_<site>/_status/heartbeat.json` to S3 |

The collector uses `config.box.yaml` on top of `data_collection/config.yaml`: no preview windows, no VLM, one random background clip per hour.

Logs are in `logs/` at the project root: `runner.log`, `collector-<date>.log`, `upload-<date>.log`, `heartbeat.log`.

## Setting up a new box

### 1. First boot (at the box, with monitor and keyboard)

1. Finish Windows setup, connect Ethernet, run Windows Update until nothing is left.
2. Install Tailscale and sign in with the same account as the laptop.
3. In the BIOS, set "restore on AC power loss" (or similar) to **Power On**, so the box starts by itself after a power cut.

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

It asks whether the box will use Ethernet or Wi-Fi, for Wi-Fi the customer's network name and password, and optionally the camera login. Then it:

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
| Change the upload time | Re-run `setup_box.ps1 -Site <site> -UploadTime 02:00` |

## Privacy

The box uploads full clips from someone else's home. Get that household's agreement first, and agree with them which cameras are included.
