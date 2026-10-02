@echo off
rem Home Guard - what the box shows on its own screen.
rem The log always opens; camera windows open only if that was chosen in the setup program.
title Home Guard
"%ProgramFiles%\Git\bin\bash.exe" -l "%~dp0screen.sh"
echo.
pause
