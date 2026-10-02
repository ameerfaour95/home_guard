# Collector box: customer network setup — design

Date: 2026-10-02
Status: approved by delegation (founder asked for a setup application and not to be asked for sign-off)

## Goal

A box sold to a customer has no screen. The customer plugs in power (and a network cable, if wired) and nothing else. The box must get on the customer's network by itself, and get back on by itself after a Wi-Fi drop or a power cut.

The founder prepares each box before delivery with one command on the laptop that asks a few questions: site name, Ethernet or Wi-Fi, and for Wi-Fi the customer's network name and password.

## How reconnection works (no watchdog)

Reconnection is handled entirely by built-in Windows and firmware behaviour, so there is no new always-on process on the box:

- **Wi-Fi drops, then returns** → the all-users auto-connect Wi-Fi profile rejoins on its own, with nobody logged in.
- **Power cut, then back on** → BIOS "restore on AC power loss" powers the box up, and the at-boot `HomeGuard-Collector` task starts collection.
- **Tailscale tunnel** → unattended mode already reconnects it after any outage.

The founder chose two recovery paths for a wrong or changed Wi-Fi password (which cannot be verified before delivery): Ethernet always works, and a rescue hotspot profile on every box.

## Non-goals

- No on-site onboarding by the customer (no setup hotspot, no phone app). The Beelink's Wi-Fi driver reports "Hosted network supported: No", and the customer is not expected to do anything.
- No open or enterprise (802.1X) Wi-Fi. WPA2-Personal by default, WPA3-Personal on request.
- No change to the collector, upload or heartbeat.
- No self-update mechanism. Updating a box stays "run the wizard again".

## Decisions

- **The founder enters the Wi-Fi details, before delivery.** Windows stores a Wi-Fi profile for a network that is not in range and joins it when it appears. The profile is saved for all users with automatic connection, so it connects at boot with nobody logged in.
- **A wrong or changed Wi-Fi password must not brick the box.** The password cannot be verified at the founder's house. Two ways back in:
  1. Ethernet always works. Plugging a cable into the customer's router brings the box online whatever the Wi-Fi settings are.
  2. A rescue hotspot profile on every box. If anyone turns on a phone hotspot with that name and password next to the box, the box joins it and the founder can fix the Wi-Fi settings over Tailscale. The name and password are generated once on the laptop and kept in `~/.homeguard/rescue_wifi`, outside the repo.
- **No custom reconnection watchdog.** Built-in behaviour (auto-connect profiles, BIOS power-on, Tailscale unattended) covers every reconnect case the founder asked for. A self-reboot task was considered and dropped: it adds the risk of a box rebooting itself in a customer's home and buys nothing the auto-connect profile does not already give.
- **Answers travel as a file, not as command-line arguments.** Network names and passwords contain spaces, quotes and non-Latin letters; quoting through ssh, cmd and PowerShell is fragile. The laptop writes `key=base64(value)` lines, copies the file, and the box script deletes it after reading.
- **The founder's own Wi-Fi must not ship with the box.** A box tested on the founder's Wi-Fi stores that password, readable by the box's new owner. A separate, explicit command forgets every saved network except the customer's and the rescue hotspot. It is the last step before unplugging, because it takes the box offline if it is on that Wi-Fi.
- **A readiness report closes the setup.** One script prints PASS / WARN / FAIL for everything remote access and recovery depend on, so the founder sees what is left without knowing Windows.

## Components

All in `home_guard_project/box/`.

| Unit | Runs on | Purpose |
|---|---|---|
| `setup_customer.sh` | laptop | The wizard. Asks the questions, installs or updates the code on the box (bundle, copy, unpack, `setup_box.ps1`), sends the answers, runs `setup_network.ps1` and `check_box.ps1`, prints what is left to do by hand. `--network-only` skips the code install. `--forget-other-wifi` runs only the forget step. |
| `setup_network.ps1` | box (admin) | Applies the network choice: Wi-Fi profile for the customer's network, rescue profile, profile order, Wi-Fi power saving off, `network.json`. `-ForgetOtherWifi` deletes other saved networks. |
| `check_box.ps1` | box (admin) | Readiness report. Exit code 1 if anything is FAIL. |
| `network.json` | box | `mode`, `wifi_ssid`, `rescue_ssid`. No passwords. Not bundled, not committed. |

`setup_box.ps1` is unchanged apart from the boot-recovery lines added earlier today.

## Flow

```
laptop: setup_customer.sh ameer@<box-ip>
  ├─ questions: site, Ethernet/Wi-Fi, network name, password
  ├─ (unless --network-only) bundle ─► scp ─► stop collector ─► unpack ─► setup_box.ps1 -Site <site>
  ├─ answers file ─► scp ─► setup_network.ps1 -AnswersFile ...   (file deleted on the box)
  ├─ check_box.ps1 ─► PASS / WARN / FAIL list
  └─ prints the manual steps and the rescue hotspot name and password
```

Reconnection after that is built-in (see "How reconnection works" above); no new always-on process is added.

## Error handling

- Wi-Fi password shorter than 8 or longer than 63 characters, or an empty network name → the wizard asks again; the box script refuses.
- Wi-Fi mode on a box with no Wi-Fi adapter → `setup_network.ps1` stops with an error.
- Adding the customer profile never disconnects or deletes the network the box is currently on.
- `--forget-other-wifi` deletes the currently connected network last, so the other deletions finish before the link drops.

## Testing

- `setup_network.ps1` is run on the real box in Wi-Fi mode with a made-up network name; the profile is checked (all users, automatic) and removed again.
- `check_box.ps1` is run on the real box.
- `setup_customer.sh` is checked with `bash -n` and run against the real box with `--network-only`.
- Bundle test extended: `network.json` is never bundled, and the new scripts are included.
- Not testable from here, left as a bench test for the founder: a real Wi-Fi outage (router off for 20 minutes) and a real power pull.

## Known limits

- The customer's Wi-Fi password is unverified until the box is at the customer's house.
- The rescue hotspot name and password are the same on all of the founder's boxes. Someone who knows them and stands next to a box can make it join their hotspot. SSH on the box accepts keys only, so joining the hotspot alone grants no access.
- If the customer changes their Wi-Fi password, the box goes offline until the founder connects over Ethernet or the rescue hotspot and re-runs the wizard. There is no automatic recovery from a changed password by design.
