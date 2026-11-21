@echo off
setlocal enabledelayedexpansion
title Pipeline Optima™ — Build

echo ============================================================
echo  Pipeline Optima(tm) -- Windows Build
echo ============================================================
echo.

:: ── 0. Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.11+ and add to PATH.
    pause & exit /b 1
)

:: ── 0b. Check Node ───────────────────────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: node not found. Install Node.js 18+ and add to PATH.
    pause & exit /b 1
)

:: ── 1. Build React frontend ───────────────────────────────────────────────────
echo [1/4] Building React frontend...
cd frontend
call npm install --silent
if errorlevel 1 ( echo   npm install failed & cd .. & pause & exit /b 1 )
call npm run build
if errorlevel 1 ( echo   npm run build failed & cd .. & pause & exit /b 1 )
cd ..
echo   Done: frontend\dist\

:: ── 2. Create icon (if SVG exists and icon.ico missing) ──────────────────────
echo [2/4] Checking app icon...
if not exist "assets\icon.ico" (
    if exist "assets\icon.svg" (
        python assets\make_icon.py 2>nul
        if not exist "assets\icon.ico" (
            echo   WARNING: icon.ico not created -- build will use default icon
        ) else (
            echo   Done: assets\icon.ico
        )
    ) else (
        echo   WARNING: No icon.svg found -- build will use default icon
    )
) else (
    echo   OK: assets\icon.ico exists
)

:: ── 3. Install / upgrade Python deps ─────────────────────────────────────────
echo [3/4] Installing Python dependencies...
pip install -q -r requirements.txt
if errorlevel 1 ( echo   pip install failed & pause & exit /b 1 )
pip install -q pyinstaller
if errorlevel 1 ( echo   pyinstaller install failed & pause & exit /b 1 )
echo   Done.

:: ── 4. PyInstaller ───────────────────────────────────────────────────────────
echo [4/4] Building executable with PyInstaller...
pyinstaller build.spec --clean --noconfirm
if errorlevel 1 ( echo   PyInstaller failed & pause & exit /b 1 )

echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Executable : dist\PipelineOptima\PipelineOptima.exe
echo  Share the entire dist\PipelineOptima\ folder
echo  (do NOT move just the .exe -- it needs the folder's DLLs)
echo ============================================================
pause
