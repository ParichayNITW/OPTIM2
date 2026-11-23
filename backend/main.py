"""Pipeline Optima™ FastAPI application."""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.db import create_tables
from backend.routers import auth, optimize, pipelines, results, registry, baseline
from backend.routers import exports

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pipeline Optima™ API",
    description=(
        "Smart Scheduling and Optimization for Cross-country Pipeline Networks. "
        "Wraps the OPTIM2 DP solver — provably optimal on the discretized MINLP."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow React dev server and any configured origin
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth.router)
app.include_router(optimize.router)
app.include_router(results.router)
app.include_router(pipelines.router)
app.include_router(registry.router)
app.include_router(baseline.router)
app.include_router(exports.router)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    create_tables()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "pipeline-optima-api"}


# ---------------------------------------------------------------------------
# Serve React frontend static files (desktop / standalone mode)
# When running via PyInstaller the dist folder is next to the executable.
# When running from source the dist folder is frontend/dist relative to CWD.
# ---------------------------------------------------------------------------

def _find_dist() -> str | None:
    candidates = [
        # PyInstaller: sys._MEIPASS is the temp extraction dir
        os.path.join(getattr(sys, "_MEIPASS", ""), "frontend", "dist"),
        # Source / Docker build: relative to CWD
        os.path.join(os.getcwd(), "frontend", "dist"),
        # Relative to this file (backend/)
        os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return None


_dist = _find_dist()
if _dist:
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
