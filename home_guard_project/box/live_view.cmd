@echo off
rem Home Guard Live View - double-click on the box to watch the cameras and detections.
rem Stops the background collector, shows it live, and restarts the background one when you close it with q.
title Home Guard Live View
"%ProgramFiles%\Git\bin\bash.exe" -l "%~dp0watch_live.sh"
echo.
pause
