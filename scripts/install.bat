@echo off
REM MBO Utilities Installer - Windows batch wrapper
REM double-click this file to run the PowerShell installer

echo.
echo MBO Utilities Installer
echo.

REM run the PowerShell script with bypass execution policy
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"

pause
