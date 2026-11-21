# -*- mode: python ; coding: utf-8 -*-
"""
Pipeline Optima™ — PyInstaller build spec.

Build with:
    pyinstaller build.spec --clean

Output: dist/PipelineOptima/PipelineOptima.exe  (Windows)
        dist/PipelineOptima/PipelineOptima       (Linux/macOS)
"""

import glob
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# ── Collect all data + hidden imports for framework packages ─────────────────
def _collect(*packages):
    datas, binaries, hidden = [], [], []
    for pkg in packages:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
    return datas, binaries, hidden

fw_datas, fw_binaries, fw_hidden = _collect(
    'uvicorn', 'fastapi', 'starlette', 'pydantic',
    'pydantic_core', 'sqlalchemy', 'passlib',
)

# ── DRA CSV lookup tables (all "N cst.csv" files in project root) ────────────
_csv_datas = [(f, '.') for f in glob.glob('*.csv')]

# ── Backend package — all .py modules ────────────────────────────────────────
_backend_hidden = collect_submodules('backend')

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=fw_binaries,
    datas=[
        # React build — served as static files by FastAPI
        ('frontend/dist', 'frontend/dist'),

        # Solver source files (read-only, never modified)
        ('pipeline_model.py', '.'),
        ('pipeline_optimization_app.py', '.'),
        ('dra_dispatcher.py', '.'),
        ('dra_utils.py', '.'),
        ('baseline_engine.py', '.'),
        ('pipeline_registry.py', '.'),
        ('linefill_utils.py', '.'),
        ('schedule_utils.py', '.'),

        # App icon
        ('assets/icon.ico', 'assets'),
    ] + _csv_datas + fw_datas,

    hiddenimports=[
        # uvicorn internals (not auto-discovered)
        'uvicorn.logging',
        'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.loops.asyncio',
        'uvicorn.protocols',
        'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl', 'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.websockets_impl',
        'uvicorn.lifespan', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off',
        # SQLite dialect
        'sqlalchemy.dialects.sqlite',
        # passlib bcrypt backend
        'passlib.handlers.bcrypt',
        # pkg_resources compat shim
        'pkg_resources.py2_compat',
        # h11 HTTP parser
        'h11',
        # anyio backends
        'anyio', 'anyio._backends._asyncio',
    ] + fw_hidden + _backend_hidden,

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'streamlit', 'pyomo', 'IPython', 'jupyter', 'notebook',
        'matplotlib.tests', 'numpy.testing',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
    console=False,       # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Icon — created by assets/make_icon.py; must exist before build
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['Qt*.dll', 'Qt*.so', 'PySide6*.so'],
    name='PipelineOptima',
)
