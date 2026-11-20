"""Pipeline Optima™ FastAPI application."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db import create_tables
from backend.routers import auth, optimize, pipelines, results

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pipeline Optima™ API",
    description=(
        "Globally optimal pipeline scheduling via Dynamic Programming. "
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
