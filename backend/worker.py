"""Celery worker — calls the OPTIM2 solver without modifying it."""

from __future__ import annotations

import os
import sys
from datetime import datetime

from celery import Celery

# ---------------------------------------------------------------------------
# Solver path — /solver in Docker (mounted read-only), or local dev path
# ---------------------------------------------------------------------------
SOLVER_PATH = os.environ.get("SOLVER_PATH", "/home/user/OPTIM2")
if SOLVER_PATH not in sys.path:
    sys.path.insert(0, SOLVER_PATH)

# These imports must come AFTER the path insertion
from pipeline_model import solve_pipeline_with_types, solve_branch  # noqa: E402

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "pipeline_optima",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
    result_expires=86400,     # keep results 24 h
)

# Keys that map directly onto solve_pipeline_with_types positional/keyword args
_MAINLINE_KEYS = (
    "stations", "terminal", "FLOW", "KV_list", "rho_list", "segment_slices",
    "RateDRA", "Price_HSD", "Fuel_density", "Ambient_temp",
    "linefill", "dra_reach_km", "mop_kgcm2", "hours", "start_time",
    "pump_shear_rate",
)
_MAINLINE_OPT = (
    "rpm_step", "dra_step", "priority_feasibility", "collect_state_audit",
)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=0, name="worker.run_optimization")
def run_optimization(self, job_id: str, payload: dict) -> dict:
    """Run the full pipeline optimization and persist the result to the DB."""
    from backend.db import OptimizationJob, get_session  # deferred to avoid circular import

    # --- mark job as running ---
    with get_session() as session:
        job = session.get(OptimizationJob, job_id)
        if job:
            job.status = "running"
            session.commit()

    try:
        # Phase 1: mainline
        kwargs: dict = {k: payload[k] for k in _MAINLINE_KEYS if k in payload}
        for k in _MAINLINE_OPT:
            if k in payload and payload[k] is not None:
                kwargs[k] = payload[k]

        mainline_result = solve_pipeline_with_types(**kwargs)

        if mainline_result.get("error"):
            raise RuntimeError(mainline_result.get("message", "Solver returned error"))

        # Phase 2: branch lines (optional)
        branch_results: list[dict] = []
        for branch_input in payload.get("branches") or []:
            junction_head = float(
                mainline_result.get("terminal_head")
                or mainline_result.get("delivery_head", 0.0)
            )
            branch_dict = {
                "flow_m3h": branch_input["flow_m3h"],
                "stations": branch_input["stations"],
                "terminal": branch_input["terminal"],
                "uid": branch_input.get("uid", ""),
                "name": branch_input.get("name", "Branch"),
            }
            br = solve_branch(
                junction_suction_head=junction_head,
                branch=branch_dict,
                KV=branch_input.get("KV", payload["KV_list"][0]),
                rho=branch_input.get("rho", payload["rho_list"][0]),
                RateDRA=payload["RateDRA"],
                Price_HSD=payload["Price_HSD"],
                Fuel_density=payload["Fuel_density"],
                Ambient_temp=payload["Ambient_temp"],
                mop_kgcm2=payload.get("mop_kgcm2") or 0.0,
                hours=payload.get("hours", 24.0),
            )
            branch_results.append(br)

        result_payload = {"mainline": mainline_result, "branches": branch_results}

        # --- persist success ---
        with get_session() as session:
            job = session.get(OptimizationJob, job_id)
            if job:
                job.status = "success"
                job.result = result_payload
                job.finished_at = datetime.utcnow()
                session.commit()

        return result_payload

    except Exception as exc:
        err = str(exc)
        with get_session() as session:
            job = session.get(OptimizationJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = err
                job.finished_at = datetime.utcnow()
                session.commit()
        raise
