@echo off
REM PPE_Check launcher - double-click to run. Close the video window (Q) to stop.
cd /d "C:\Users\xczma\.gemini\antigravity\scratch\hazard-detection-cctv"
REM defaults: 640px + low per-class thresholds catch masks/gloves; 3-frame smoothing kills flicker
python mode_ppe.py --source 0 --camera CAM01 --imgsz 640 --conf 0.25
REM photo test (no camera): python mode_ppe.py --source live_ready.jpg --imgsz 640
REM gate gloves backup (2-3m only): add  --gloves-model ppe_v8m.pt
echo.
echo App stopped. Press any key to close this window.
pause >nul
