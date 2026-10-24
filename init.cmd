@echo off
rem Windows counterpart of init.sh, see init.ps1. Works from cmd.exe, PowerShell and Explorer,
rem and runs init.ps1 without changing the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0init.ps1" %*
exit /b %errorlevel%
