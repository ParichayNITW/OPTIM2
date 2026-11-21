"""Thread-based job worker — no Celery or Redis required for desktop deployment."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ---------------------------------------------------------------------------
# Solver path — project root (desktop) or /solver (Docker)
# ---------------------------------------------------------------------------
SOLVER_PATH = os.environ.get(
    "SOLVER_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if SOLVER_PATH not in sys.path:
    sys.path.insert(0, SOLVER_PATH)

from pipeline_model import solve_pipeline_with_types, solve_branch  # noqa: E402

# ---------------------------------------------------------------------------
# Thread pool — max 2 concurrent solver runs on desktop
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)

_MAINLINE_KEYS = (
    "stations", "terminal", "FLOW", "KV_list", "rho_list", "segment_slices",
    "RateDRA", "Price_HSD", "Fuel_density", "Ambient_temp",
    "linefill", "dra_reach_km", "mop_kgcm2", "hours", "start_time",
    "pump_shear_rate",
)
_MAINLINE_OPT = (
    "rpm_step", "dra_step", "priority_feasibility", "collect_state_audit",
)


def submit_job(job_id: str, payload: dict) -> None:
    """Submit job to the thread pool. Returns immediately."""
    _executor.submit(_run_optimization, job_id, payload)


def _run_optimization(job_id: str, payload: dict) -> None:
    """Runs in a worker thread — mirrors the Celery task logic exactly."""
    from backend.db import OptimizationJob, get_session  # deferred to avoid circular import

    with get_session() as session:
        job = session.get(OptimizationJob, job_id)
        if job:
            job.status = "running"
            session.commit()

    try:
        kwargs: dict = {k: payload[k] for k in _MAINLINE_KEYS if k in payload}
        for k in _MAINLINE_OPT:
            if k in payload and payload[k] is not None:
                kwargs[k] = payload[k]

        mainline_result = solve_pipeline_with_types(**kwargs)

        if mainline_result.get("error"):
            raise RuntimeError(mainline_result.get("message", "Solver returned error"))

        # Inject aggregate metrics — solver stores per-station flat keys only;
        # the frontend needs totals at the top level for quick display.
        _stn_keys = [
            s['name'].strip().lower().replace(' ', '_')
            for s in kwargs.get('stations', [])
        ]
        mainline_result['power_cost'] = sum(
            float(mainline_result.get(f'power_cost_{k}', 0) or 0) for k in _stn_keys
        )
        mainline_result['dra_cost'] = sum(
            float(mainline_result.get(f'dra_cost_{k}', 0) or 0) for k in _stn_keys
        )
        mainline_result['num_pumps'] = sum(
            int(mainline_result.get(f'num_pumps_{k}', 0) or 0) for k in _stn_keys
        )
        mainline_result['FLOW'] = float(kwargs.get('FLOW', 0) or 0)

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

        with get_session() as session:
            job = session.get(OptimizationJob, job_id)
            if job:
                job.status = "success"
                job.result = result_payload
                job.finished_at = datetime.utcnow()
                session.commit()

    except Exception as exc:
        err = str(exc)
        with get_session() as session:
            job = session.get(OptimizationJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = err
                job.finished_at = datetime.utcnow()
                session.commit()
