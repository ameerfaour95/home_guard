@echo off
rem Home Guard - launch the desktop app without a console window.
rem The environment's own pythonw.exe (a uv environment) hands over to the
rem console interpreter, which opens a terminal next to the app; the base
rem install's pythonw.exe on start.pyw does not. The base install is named in
rem .venv\pyvenv.cfg ("home = <folder>").
cd /d "%~dp0..\.."
set "BASE="
for /f "tokens=1,* delims== " %%a in ('findstr /b /c:"home" ".venv\pyvenv.cfg"') do set "BASE=%%b"
if defined BASE if exist "%BASE%\pythonw.exe" (
    start "" "%BASE%\pythonw.exe" "%~dp0app\start.pyw"
    exit /b 0
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m home_guard_project.box.app.launcher
    exit /b 0
)
rem An older install can still use the original console screen.
"%ProgramFiles%\Git\bin\bash.exe" -l "%~dp0screen.sh"
