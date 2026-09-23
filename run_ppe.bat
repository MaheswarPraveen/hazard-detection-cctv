@echo off
REM PPE_Check launcher - double-click to run. Close the video window (Q) to stop.
cd /d "C:\Users\xczma\.gemini\antigravity\scratch\hazard-detection-cctv"
REM defaults: threaded (smooth video + ~10Hz AI), 480px main + face-zoom for masks, 3-frame smoothing
python mode_ppe.py --source 0 --camera CAM01 --imgsz 480 --conf 0.25
REM photo test (no camera): python mode_ppe.py --source live_ready.jpg --imgsz 640
REM gate gloves backup (2-3m only): add  --gloves-model ppe_v8m.pt
echo.
echo App stopped. Press any key to close this window.
pause >nul
