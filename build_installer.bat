@echo off
setlocal

REM Build a Windows installer using Inno Setup (requires ISCC in PATH).

if not exist "dist\pipeline_optimizer.exe" (
  echo Missing dist\pipeline_optimizer.exe. Run build_windows.bat first.
  exit /b 1
)

where ISCC >nul 2>&1
if errorlevel 1 (
  echo Inno Setup compiler (ISCC) not found in PATH.
  echo Install Inno Setup and ensure ISCC.exe is in PATH.
  exit /b 1
)

ISCC pipeline_optimizer_installer.iss

echo Installer created in dist\PipelineOptimizerInstaller.exe
pause
