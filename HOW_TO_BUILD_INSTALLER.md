# How to Build the Windows Installer

## What you need (one-time setup on a Windows machine)

1. **Python 3.11** — https://www.python.org/downloads/
2. **Node.js 20** — https://nodejs.org/
3. **Inno Setup 6** (free) — https://jrsoftware.org/isdl.php
4. **PyInstaller** — `pip install pyinstaller`

## Steps

### 1. Clone the repository
```
git clone https://github.com/ParichayNITW/OPTIM2.git
cd OPTIM2
git checkout claude/optim2-file-count-ymv3s0
```

### 2. (Optional) Generate the app icon
```
pip install Pillow cairosvg
python assets/make_icon.py
```

### 3. Run the build script
```
build_installer.bat
```

This does everything automatically:
- Installs Python packages
- Builds the React frontend
- Bundles everything with PyInstaller
- Creates `dist\installer\PipelineOptima-Setup-v1.0.0.exe`

## End result

**`dist\installer\PipelineOptima-Setup-v1.0.0.exe`** — a standalone Windows installer.

Send this single file to any user. They:
1. Double-click `PipelineOptima-Setup-v1.0.0.exe`
2. Click Next → Next → Install (like any commercial software)
3. A Desktop shortcut is created
4. Double-click the shortcut → browser opens at `http://localhost:8765`
5. Register an account → start running optimizations

## No external dependencies for end users

The installer bundles:
- Python 3.11 runtime (embedded)
- All solver files (pipeline_model.py, etc.)
- React web interface
- SQLite database (created automatically)

**The end user needs nothing else installed.**
