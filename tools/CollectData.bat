@echo off
cd /d "%~dp0.."
SafetyCCTV.exe --run collect_data %*
