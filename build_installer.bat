@echo off
:: ============================================================
:: Pipeline Optima™ — Windows installer build script
:: Run this on a Windows machine with Python 3.11, Node 20,
:: PyInstaller and Inno Setup 6 installed.
:: ============================================================

setlocal enabledelayedexpansion
set ERRORLEVEL=0

echo.
echo ============================================================
echo  Pipeline Optima(tm) Build Script
echo ============================================================
echo.

:: --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11 from python.org
    exit /b 1
)

:: --- Check Node ---
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Install Node.js 20 from nodejs.org
    exit /b 1
)

:: --- Check PyInstaller ---
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: --- Step 1: Install Python dependencies ---
echo [1/4] Installing Python dependencies...
pip install -r desktop_requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed & exit /b 1 )

:: --- Step 2: Build React frontend ---
echo [2/4] Building React frontend...
cd frontend
call npm install
if errorlevel 1 ( echo ERROR: npm install failed & cd .. & exit /b 1 )
call npm run build
if errorlevel 1 ( echo ERROR: npm build failed & cd .. & exit /b 1 )
cd ..
echo     frontend/dist built successfully.

:: --- Step 3: PyInstaller bundle ---
echo [3/4] Bundling with PyInstaller...
pyinstaller pipeline_optima.spec --noconfirm
if errorlevel 1 ( echo ERROR: PyInstaller failed & exit /b 1 )
echo     dist\pipeline_optima\ created.

:: --- Step 4: Inno Setup installer ---
echo [4/4] Creating Windows installer...
set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
    set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
)
if not exist %ISCC% (
    echo WARNING: Inno Setup 6 not found at default path.
    echo Download from https://jrsoftware.org/isinfo.php and install.
    echo Skipping installer creation. The app folder is in dist\pipeline_optima\
    goto :done
)

mkdir dist\installer 2>nul
%ISCC% installer\setup.iss
if errorlevel 1 ( echo ERROR: Inno Setup failed & exit /b 1 )

:done
echo.
echo ============================================================
echo  BUILD COMPLETE
echo ============================================================
if exist dist\installer\PipelineOptima-Setup-v1.0.0.exe (
    echo  Installer: dist\installer\PipelineOptima-Setup-v1.0.0.exe
) else (
    echo  App folder: dist\pipeline_optima\PipelineOptima.exe
)
echo ============================================================
echo.
