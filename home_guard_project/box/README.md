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

### 3. Install the code and run setup (from the laptop)

```bash
uv run python -m home_guard_project.box.make_bundle                # writes dist/home_guard_box.zip
scp -i ~/.ssh/homeguard_box dist/home_guard_box.zip <user>@<ip>:C:/home_guard_box.zip
ssh -i ~/.ssh/homeguard_box <user>@<ip> "powershell -Command \"Expand-Archive -Force C:\home_guard_box.zip C:\home_guard\""
ssh -i ~/.ssh/homeguard_box <user>@<ip> "powershell -ExecutionPolicy Bypass -File C:\home_guard\home_guard_project\box\setup_box.ps1 -Site house2"
```

`-Site` is the house name used in the S3 prefix: lowercase letters, digits and underscores only.

The bundle contains only what the box needs. It never includes `cameras.yaml`, API keys, or scripts that hold camera passwords.

### 4. AWS key for the box

Give the box its own IAM user, not your personal keys. Policy for site `house2`:

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

Put the key in `C:\Users\<user>\.aws\credentials` on the box:

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

### 5. Camera discovery (once per house)

You need the camera or recorder username and password for that house. On the box (Remote Desktop, or at the box), in Git Bash:

```bash
cd /c/home_guard
uv run python home_guard_project/data_collection/discover.py
```

Name each camera with the site in front, for example `house2_front_door`, so clip names never collide between houses. Leave out indoor cameras the household does not want recorded.

This writes `data_collection/cameras.yaml`. The collector task notices it within a minute and starts collecting; no restart is needed.

## Bench test before moving the box

Run it at your own house for a day first:

1. Restart the box. Within about two minutes `logs/runner.log` should show "Starting collector" without anyone logging in.
2. Walk in front of a camera. A clip should appear under `dataset_multi/clips/`.
3. Pull the power cable and plug it back in. The box should boot and collect again.
4. Unplug the network for ten minutes. `collector-<date>.log` should show the cameras reconnecting.
5. Run an upload by hand: `./home_guard_project/box/run_upload.sh`. Clips older than ten minutes should appear on S3 and disappear locally.
6. Note CPU and memory use in Task Manager with all cameras connected.

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
| Restart the collector | `schtasks /End /TN HomeGuard-Collector` then `schtasks /Run /TN HomeGuard-Collector` |
| Change the upload time | Re-run `setup_box.ps1 -Site <site> -UploadTime 02:00` |

## Privacy

The box uploads full clips from someone else's home. Get that household's agreement first, and agree with them which cameras are included.
