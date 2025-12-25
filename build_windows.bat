@echo off
setlocal enabledelayedexpansion

REM Build a standalone Windows executable for the Pipeline Optimizer.

set REQUIRED_FILES=pipeline_optimization_app.py pipeline_desktop_app.py logo.png secrets.toml
for %%F in (%REQUIRED_FILES%) do (
  if not exist "%%F" (
    echo Missing required file: %%F
    exit /b 1
  )
)

dir /b *.csv >nul 2>&1
if errorlevel 1 (
  echo Missing required CSV files (*.csv).
  exit /b 1
)

py -m pip install --upgrade pip
py -m pip install -r requirements.txt pyinstaller pywebview

py -m PyInstaller --clean --onefile --name pipeline_optimizer ^
  --add-data "pipeline_optimization_app.py;." ^
  --add-data "logo.png;." ^
  --add-data "secrets.toml;." ^
  --add-data "*.csv;." ^
  pipeline_desktop_app.py

py verify_bundle.py dist\pipeline_optimizer.exe ^
  --require pipeline_optimization_app.py ^
  --require logo.png ^
  --require secrets.toml ^
  --write dist\bundle_manifest.txt

echo Executable built at dist\pipeline_optimizer.exe
pause
