# ============================================================================
#  enable_autologon.ps1 - sign the box's Windows user in by itself at boot
#
#  The alert program does not need this: it runs from the HomeGuard-Collector
#  task at boot with nobody signed in. This is for the Home Guard window
#  (Startup folder shortcut), which only opens once a user is signed in.
#
#  Run on the box from an elevated PowerShell (locally, or over "ssh -t"):
#      powershell -ExecutionPolicy Bypass -File enable_autologon.ps1
#      powershell -ExecutionPolicy Bypass -File enable_autologon.ps1 -Off
#
#  What it does (safe to re-run):
#    1. Asks for this user's Windows password and checks it
#    2. Stores it the way Sysinternals Autologon does: an LSA secret, never in
#       the registry as plain text
#    3. Turns on AutoAdminLogon for this user
#    4. Turns off the Windows 11 "passwordless" switch that hides auto sign-in
#    5. No sign-in screen when the display wakes up
# ============================================================================
#Requires -RunAsAdministrator
param(
    # Turn auto sign-in off and delete the stored password.
    [switch]$Off
)

$ErrorActionPreference = 'Stop'

function Ok($msg) { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }

Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class HgAutologon {
    [StructLayout(LayoutKind.Sequential)]
    struct LsaString { public ushort Length; public ushort MaximumLength; public IntPtr Buffer; }

    [StructLayout(LayoutKind.Sequential)]
    struct LsaObjectAttributes {
        public int Length; public IntPtr RootDirectory; public IntPtr ObjectName;
        public uint Attributes; public IntPtr SecurityDescriptor; public IntPtr SecurityQualityOfService;
    }

    [DllImport("advapi32.dll")]
    static extern uint LsaOpenPolicy(IntPtr system, ref LsaObjectAttributes attrs, uint access, out IntPtr policy);
    [DllImport("advapi32.dll")]
    static extern uint LsaStorePrivateData(IntPtr policy, ref LsaString key, IntPtr data);
    [DllImport("advapi32.dll")]
    static extern uint LsaClose(IntPtr policy);
    [DllImport("advapi32.dll")]
    static extern int LsaNtStatusToWinError(uint status);
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool LogonUser(string user, string domain, string password, int type, int provider, out IntPtr token);
    [DllImport("kernel32.dll")]
    static extern bool CloseHandle(IntPtr handle);

    const uint POLICY_CREATE_SECRET = 0x20;

    static LsaString Make(string s) {
        LsaString u = new LsaString();
        u.Buffer = Marshal.StringToHGlobalUni(s);
        u.Length = (ushort)(s.Length * 2);
        u.MaximumLength = (ushort)(s.Length * 2 + 2);
        return u;
    }

    // value == null deletes the secret. Returns a Win32 error code, 0 = success.
    public static int StoreSecret(string name, string value) {
        LsaObjectAttributes attrs = new LsaObjectAttributes();
        attrs.Length = Marshal.SizeOf(attrs);
        IntPtr policy;
        uint status = LsaOpenPolicy(IntPtr.Zero, ref attrs, POLICY_CREATE_SECRET, out policy);
        if (status != 0) return LsaNtStatusToWinError(status);
        LsaString key = Make(name);
        IntPtr data = IntPtr.Zero;
        LsaString val = new LsaString();
        try {
            if (value != null) {
                val = Make(value);
                data = Marshal.AllocHGlobal(Marshal.SizeOf(val));
                Marshal.StructureToPtr(val, data, false);
            }
            status = LsaStorePrivateData(policy, ref key, data);
            return status == 0 ? 0 : LsaNtStatusToWinError(status);
        } finally {
            LsaClose(policy);
            Marshal.FreeHGlobal(key.Buffer);
            if (val.Buffer != IntPtr.Zero) Marshal.FreeHGlobal(val.Buffer);
            if (data != IntPtr.Zero) Marshal.FreeHGlobal(data);
        }
    }

    // Win32 error of an interactive logon, 0 = the password is right.
    public static int CheckPassword(string user, string domain, string password) {
        IntPtr token;
        if (LogonUser(user, domain, password, 2, 0, out token)) { CloseHandle(token); return 0; }
        return Marshal.GetLastWin32Error();
    }
}
'@

$winlogon = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
# Take the account from the token: over SSH, $env:USERDOMAIN is not the computer name.
$account = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$domain, $user = $account.Split('\', 2)

if ($Off) {
    Set-ItemProperty $winlogon -Name AutoAdminLogon -Value '0'
    [void][HgAutologon]::StoreSecret('DefaultPassword', $null)
    Remove-ItemProperty $winlogon -Name DefaultPassword -ErrorAction SilentlyContinue
    Ok 'Auto sign-in off; stored password deleted.'
    return
}

$secure = Read-Host "Windows password for $account" -AsSecureString
$password = [Runtime.InteropServices.Marshal]::PtrToStringUni(
    [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($secure))

$err = [HgAutologon]::CheckPassword($user, $domain, $password)
if ($err -eq 1326) { throw 'Wrong password - nothing changed.' }
# 1327 (account restriction) is what a blank password gets here; Winlogon still accepts it.
if ($err -ne 0 -and -not ($err -eq 1327 -and $password -eq '')) {
    throw "Could not check the password (Windows error $err) - nothing changed."
}
Ok 'Password checked.'

$err = [HgAutologon]::StoreSecret('DefaultPassword', $password)
if ($err -ne 0) { throw "Could not store the password (Windows error $err)." }
# An old plain-text copy would win over the secret; remove it.
Remove-ItemProperty $winlogon -Name DefaultPassword -ErrorAction SilentlyContinue
Remove-ItemProperty $winlogon -Name AutoLogonCount -ErrorAction SilentlyContinue
Set-ItemProperty $winlogon -Name DefaultUserName -Value $user
Set-ItemProperty $winlogon -Name DefaultDomainName -Value $domain
Set-ItemProperty $winlogon -Name AutoAdminLogon -Value '1'
Ok "Auto sign-in on for $account (password kept as an LSA secret)."

# Windows 11 "For improved security, only allow Windows Hello sign-in": while on,
# Winlogon ignores AutoAdminLogon.
$pwless = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordLess\Device'
if (-not (Test-Path $pwless)) { New-Item $pwless -Force | Out-Null }
Set-ItemProperty $pwless -Name DevicePasswordLessBuildVersion -Value 0 -Type DWord
Ok 'Windows Hello-only sign-in off.'

# The box has no keyboard: the display turning off must not lock the session.
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_NONE CONSOLELOCK 0
powercfg /SETACTIVE SCHEME_CURRENT
Set-ItemProperty 'HKCU:\Control Panel\Desktop' -Name ScreenSaverIsSecure -Value '0'
Ok 'No sign-in screen when the display wakes.'
