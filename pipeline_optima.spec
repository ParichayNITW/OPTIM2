# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build spec for Pipeline Optima™
#
# Build command (run from the OPTIM2 directory):
#   pip install pyinstaller
#   pyinstaller pipeline_optima.spec
#
# Output:
#   dist/PipelineOptima/          (--onedir, recommended — faster startup)
#   dist/PipelineOptima.exe       (--onefile alternative — slower cold start)
#
# The dist/PipelineOptima/ folder can be zipped and distributed to any
# Windows machine — no Python installation required.

import os
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

# ── Collect Streamlit's own static assets (HTML, JS, CSS) ─────────────────
datas = collect_data_files('streamlit', include_py_files=False)

# ── Collect streamlit-agraph static assets ────────────────────────────────
try:
    datas += collect_data_files('streamlit_agraph')
except Exception:
    pass

# ── App source files & data ───────────────────────────────────────────────
app_files = [
    ('pipeline_optimization_app.py', '.'),
    ('pipeline_model.py', '.'),
    ('hydraulic_check.py', '.'),
    ('baseline_engine.py', '.'),
    ('dra_analysis.py', '.'),
    ('dra_utils.py', '.'),
    ('linefill_utils.py', '.'),
    ('schedule_utils.py', '.'),
    ('generate_thesis.py', '.'),
    ('logo.png', '.'),
    ('secrets.toml', '.'),
    ('.streamlit/config.toml', '.streamlit'),
]

# Add all DRA lookup CSV files (e.g. "1 cst.csv", "10 cst.csv", ...)
import glob
for csv_path in glob.glob('*.csv'):
    app_files.append((csv_path, '.'))

datas += app_files

# ── Hidden imports that PyInstaller may miss ──────────────────────────────
hiddenimports = [
    'streamlit',
    'streamlit.web.cli',
    'streamlit.web.server',
    'streamlit_agraph',
    'pyomo',
    'pyomo.environ',
    'fpdf',
    'fpdf2',
    'kaleido',
    'plotly',
    'plotly.graph_objects',
    'plotly.express',
    'numba',
    'scipy',
    'scipy.optimize',
    'scipy.interpolate',
    'xlsxwriter',
    'openpyxl',
    'matplotlib',
    'matplotlib.backends.backend_agg',
    'pandas',
    'numpy',
    'altair',
    'pydeck',
    'click',
    'tornado',
]

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5', 'wx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── One-directory build (recommended: faster startup, easier debugging) ───
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PipelineOptima',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,        # set False to hide the terminal window after testing
    icon='logo.png',     # requires logo.ico on Windows; use logo.png as fallback
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PipelineOptima',
)

# ── Uncomment below for single-file .exe (slower cold start ~30s) ──────────
# exe_onefile = EXE(
#     pyz,
#     a.scripts,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     name='PipelineOptima',
#     debug=False,
#     strip=False,
#     upx=True,
#     console=True,
#     icon='logo.png',
# )
