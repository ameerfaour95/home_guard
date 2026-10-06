# Install Home Guard Admin Center

## Copy and launch

Staff need a Windows 64-bit PC and the complete `HomeGuardAdmin` folder supplied by an administrator. Python is not required.

1. Copy the entire folder to a location you can write to, such as your Desktop or Documents. If you receive a ZIP, extract it first.
2. Keep `HomeGuardAdmin.exe` together with its `_internal` folder. Copying just the executable will not work.
3. Double-click `HomeGuardAdmin.exe`. A shortcut to this file is fine.
4. Open **Settings** at the top of the window and enter the Home Guard Cloud server URL supplied by your administrator. Save settings.
5. Sign in with your staff email, password, and current six-digit code from your authenticator. Ask your administrator for an account and authenticator enrolment if you do not have them.

The app remembers your email, server URL, and theme. It does not save your password, authenticator code, or session tokens. Labelers see Review and Studio; access also follows customer consent.

This build is unsigned. If your organisation's Windows Application Control blocks it, ask IT for an approved distribution. Do not disable Windows security controls.

## Server and appearance

**Settings** is available before and after sign-in. It supports a validated server URL, Dark or Light appearance, and **Open log folder**. Changing the server signs you out; changing appearance takes effect immediately.

Use HTTPS for a deployed server. HTTP is supported only for local development at `localhost` or `127.0.0.1`. Enter the server base URL, without `/v1`, a username/password, a query string, or a fragment.

You can also launch from PowerShell or put arguments in a Windows shortcut:

```powershell
& 'C:\Apps\HomeGuardAdmin\HomeGuardAdmin.exe' --server https://cloud.example.com
& 'C:\Apps\HomeGuardAdmin\HomeGuardAdmin.exe' --theme light
```

Command-line arguments override saved preferences for that launch. The default without either is `http://127.0.0.1:8000`. `--demo` opens synthetic offline data for a walkthrough; staff work requires the real server.

## Logs and updates

Logs live at `%LOCALAPPDATA%\HomeGuardAdmin\logs\admin.log`. **Settings → Open log folder** opens that directory. Logs rotate at 1 MiB, with four backups. Preferences are at `%APPDATA%\HomeGuardAdmin\prefs.json`.

If sign-in or a screen fails, verify the server URL and network connection, then give your administrator the error text and local log. Never send passwords or authenticator codes.

To update, close the app and replace the complete application folder with the new supplied folder. Preferences and logs remain in your Windows profile.

## Training downloads

In Studio, open an export marked **Ready** and choose **Open manifest** to inspect schema v2 counts and warnings. VLM exports include clips and write `vlm/{train,val,test}.jsonl` plus `vlm/dataset_info.json`.

An administrator with the Cloud management environment downloads the dataset using `manage export-download <id> --dest DIR`. The export detail shows the command with its actual ID. The desktop app reads the server-provided manifest URL; it does not need AWS credentials.

## Tagging studio (admins)

**Tag** in the navigation (or **Open tagging studio** in Studio) is where clips get their category tags for the AI.

| Area | What it shows |
| --- | --- |
| Work queue (left) | Contradictions first, alerts and suspicious-vs-escalation disagreements on top, then clips to check, then untagged clips. An old tag counts as done unless something contradicts it. |
| Clip (middle) | The crop the AI sees, or the full frame. Under it, side by side: the old tag, the customer's answer, the AI label and the teacher suggestion. A card is outlined in red when it is part of the contradiction. |
| Tag (right) | The category, grouped Normal / Suspicious / Escalation, with the Hebrew names. Then the raw label, zone, movement, flags, visibility, evidence frame, an English description, notes, Needs check and Delete. |

Press **?** in the studio for the keys. The most used are:

| Key | Action |
| --- | --- |
| `s` `3` | Category S3 (`n` `0` is N10) |
| **Ctrl+Enter** | Save and open the next clip |
| `f` | Mark the evidence frame |
| `a` | Use the teacher's suggestion |
| `j` / `k` | Next / previous clip |

Unsaved edits are kept when you move to another clip.

### Where the data comes from

| Source | Location | Override |
| --- | --- | --- |
| Customers' alerts and answers | The Cloud database (indexed from S3) | |
| Customers' alerts and answers (local copies) | `owner_feedback/` inside the dataset folder | |
| Old tags | The unified dataset: `home_guard_data/dataset` next to the app folder | `HOMEGUARD_DATASET_DIR` |
| Teacher suggestions | The eval results: `home_guard_eval/eval_set/results` | `HOMEGUARD_EVAL_DIR` |

A clip from a household that has not given training consent stays in the queue, but its video does not open. Confirm the box's consent proposal with **Confirm consent…** on the customer's page.

### Exports

**Export training set…** writes these files into a timestamped folder under `studio_exports` next to the dataset (override: `HOMEGUARD_STUDIO_EXPORT_DIR`):

- `vlm_training.jsonl`: the existing contract plus the category and observation fields.
- `eval_manifest.jsonl`: eval rows that `box/eval_prompt.py` can score against.

Nothing is written to S3. To export from the command line:

```
manage tagging-export [--dataset DIR] [--eval DIR] [--out DIR] [--eval-frames DIR]
```

`--eval-frames` also writes the frames and the `manifest.jsonl` that the eval needs.

### Self-hosted teacher

To ask a self-hosted teacher model on demand, set `HG_TEACHER_BASE_URL` and `HG_TEACHER_MODEL`, and `HG_TEACHER_API_KEY` if the server needs one. The server must speak the OpenAI-compatible API.

The teacher is asked only when you press **Ask teacher**. Gemini is refused: its terms forbid training a competing model on its outputs.
