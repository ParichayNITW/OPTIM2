# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Pipeline Optima™ desktop application.

Build with:
    pyinstaller pipeline_optima.spec

Output: dist/pipeline_optima/PipelineOptima.exe  (Windows)
"""

import os
import glob
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
datas = []

# React frontend build output (must be built first: npm run build)
frontend_dist = os.path.join("frontend", "dist")
if os.path.isdir(frontend_dist):
    datas += [(frontend_dist, os.path.join("frontend", "dist"))]

# OPTIM2 solver files (called at runtime — not imported statically)
solver_files = [
    "pipeline_model.py",
    "dra_dispatcher.py",
    "dra_utils.py",
    "baseline_engine.py",
    "pipeline_registry.py",
]
for f in solver_files:
    if os.path.exists(f):
        datas += [(f, ".")]

# DRA CSV / model files
for pattern in ["*.csv", "*.pkl", "*.joblib"]:
    for f in glob.glob(pattern):
        datas += [(f, ".")]

# DRA models folder
if os.path.isdir("dra_models"):
    datas += [("dra_models", "dra_models")]

# Backend package (py files included as data so they are importable at runtime)
datas += collect_data_files("backend", include_py_files=True)

# App icon
if os.path.exists(os.path.join("assets", "icon.ico")):
    datas += [(os.path.join("assets", "icon.ico"), "assets")]

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
hiddenimports = [
    # FastAPI / Uvicorn
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi.staticfiles",
    # SQLAlchemy SQLite
    "sqlalchemy.dialects.sqlite",
    "sqlalchemy.dialects.sqlite.pysqlite",
    # Passlib
    "passlib.handlers.bcrypt",
    "passlib.handlers.sha2_crypt",
    # Pydantic
    "pydantic.deprecated.class_validators",
    # NumPy / SciPy
    "numpy",
    "scipy",
    "scipy.special._ufuncs_cxx",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    # Pandas
    "pandas",
    # Sklearn
    "sklearn",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._partition_nodes",
    # Backend submodules
    "backend",
    "backend.main",
    "backend.db",
    "backend.worker",
    "backend.models",
    "backend.deps",
    "backend.auth_utils",
    "backend.routers",
    "backend.routers.auth",
    "backend.routers.optimize",
    "backend.routers.results",
    "backend.routers.pipelines",
    # Qt WebEngine (PySide6)
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngine",
    "PySide6.QtNetwork",
    "PySide6.QtPrintSupport",
]

hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("fastapi")
hiddenimports += collect_submodules("starlette")
hiddenimports += collect_submodules("sqlalchemy")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "notebook", "jupyter"],
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
    name="PipelineOptima",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # windowed=True means no black console window; Qt handles its own window
    console=False,
    icon="assets/icon.ico" if os.path.exists("assets/icon.ico") else None,
    version_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="pipeline_optima",
)
