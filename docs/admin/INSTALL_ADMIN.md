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
