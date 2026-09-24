@echo off
REM PPE station launcher - pick a station.
cd /d "C:\Users\xczma\.gemini\antigravity\scratch\hazard-detection-cctv"
echo  1 = Mask + Helmet station
echo  2 = Vest + Blue-Gloves gate (2-3m)
set /p pick="Pick station [1/2]: "
if "%pick%"=="2" goto vg
:mh
python mode_ppe.py --source 0 --camera CAM01 --checks helmet,mask --imgsz 480 --conf 0.25 --silent
goto end
:vg
python mode_ppe.py --source 0 --camera CAM01 --checks vest,gloves --imgsz 416 --conf 0.25 --silent
REM gloves return with fine-tuned weights: add  --gloves-model best.pt  (ppe_v8m.pt is blind on our feed)
:end
echo.
echo App stopped. Press any key to close this window.
pause >nul
