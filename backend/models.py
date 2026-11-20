"""Pydantic request/response schemas for Pipeline Optima API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"


# ---------------------------------------------------------------------------
# Optimization request  (mirrors solve_pipeline_with_types signature)
# ---------------------------------------------------------------------------

class BranchInput(BaseModel):
    """A tap-off / branch line to solve after the mainline."""
    flow_m3h: float
    stations: list[dict[str, Any]]
    terminal: dict[str, Any]
    uid: str = ""
    name: str = "Branch"
    KV: float = 1.0
    rho: float = 850.0


class OptimizeRequest(BaseModel):
    # --- required core inputs ---
    stations: list[dict[str, Any]]
    terminal: dict[str, Any]           # {name, elev, min_residual}
    FLOW: float                         # m³/h
    KV_list: list[float]               # kinematic viscosities (cSt)
    rho_list: list[float]              # densities (kg/m³)
    RateDRA: float
    Price_HSD: float                   # ₹/litre
    Fuel_density: float                # kg/litre
    Ambient_temp: float                # °C

    # --- optional solver inputs ---
    segment_slices: list[list[dict[str, Any]]] | None = None
    linefill: list[dict[str, Any]] | None = None
    dra_reach_km: float = 0.0
    mop_kgcm2: float | None = None
    hours: float = 24.0
    start_time: str = "00:00"
    pump_shear_rate: float = 0.0

    # --- branch lines (solved after mainline) ---
    branches: list[BranchInput] | None = None

    # --- solver tuning (use engine defaults when None) ---
    rpm_step: int | None = None
    dra_step: int | None = None
    priority_feasibility: bool = False
    collect_state_audit: bool = False


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------

class OptimizeResponse(BaseModel):
    job_id: str
    status: JobStatus


class ResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Pipeline configs
# ---------------------------------------------------------------------------

class PipelineCreate(BaseModel):
    name: str
    description: str = ""
    config: dict[str, Any]           # the full case dict (JSON-serializable)


class PipelineOut(BaseModel):
    id: int
    name: str
    description: str
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
