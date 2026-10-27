@echo off
cd /d "%~dp0.."
SafetyCCTV.exe --run calibrate_glove %*
