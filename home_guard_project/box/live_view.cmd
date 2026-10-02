@echo off
rem Home Guard - launch the desktop app without a persistent console.
cd /d "%~dp0..\.."
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m home_guard_project.box.app.launcher
    exit /b 0
)
rem An older install can still use the original console screen.
"%ProgramFiles%\Git\bin\bash.exe" -l "%~dp0screen.sh"
