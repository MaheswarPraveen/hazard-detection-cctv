@echo off
REM PPE station launcher - pick a station, or run both .bat files below directly.
cd /d "C:\Users\xczma\.gemini\antigravity\scratch\hazard-detection-cctv"
echo  1 = Helmet + Vest station
echo  2 = Mask + Gloves gate (2-3m)
set /p pick="Pick station [1/2]: "
if "%pick%"=="2" goto mg
:hhv
python mode_ppe.py --source 0 --camera CAM01 --checks helmet,vest --imgsz 480 --conf 0.25
goto end
:mg
python mode_ppe.py --source 0 --camera CAM01 --checks mask,gloves --gloves-model ppe_v8m.pt --imgsz 480 --conf 0.25
:end
echo.
echo App stopped. Press any key to close this window.
pause >nul
