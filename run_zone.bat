@echo off
REM Zone_Alert launcher - double-click to run. Close the video window (Q) to stop.
cd /d "C:\Users\xczma\Documents\Default Project\safety-cctv"
python mode_zone.py --source 0 --camera CAM01
echo.
echo App stopped. Press any key to close this window.
pause >nul
