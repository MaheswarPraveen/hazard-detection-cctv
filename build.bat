@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo =====================================================================
echo               Building Standalone Safety CCTV Package
echo =====================================================================

REM 1. Set up packaging venv if not present
if not exist ".venv" (
    echo [1/5] Creating dedicated build venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is on PATH.
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo [2/5] Installing CPU-only PyTorch and project dependencies...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pyinstaller reportlab

echo [3/5] Compiling tool helpers and generating Client Guide PDF...
python build_tools.py
python make_pdf_guide.py

echo [4/5] Running PyInstaller build...
pyinstaller SafetyCCTV.spec --clean --noconfirm
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    exit /b 1
)

echo [5/5] Assembling final distribution and creating ZIP archive...
python post_build.py
if errorlevel 1 (
    echo [ERROR] Post-build assembly failed.
    exit /b 1
)

echo =====================================================================
echo  BUILD SUCCESSFUL!
echo  Package: dist\SafetyCCTV_win64.zip
echo =====================================================================
