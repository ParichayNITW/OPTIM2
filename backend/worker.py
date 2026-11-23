"""Thread-based job worker — no Celery or Redis required for desktop deployment."""

from __future__ import annotations

import copy
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

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
from backend.schedule_helpers import (  # noqa: E402
    df_from_linefill,
    shift_vol_linefill,
    combine_volumetric_profiles,
    map_vol_linefill_to_segments,
    apply_dra_ppm,
    _default_segment_slices,
)

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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _stn_key(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _inject_aggregate_metrics(result: dict, stn_keys: list[str]) -> None:
    """Add top-level totals so the frontend gets quick aggregates."""
    result["power_cost"] = sum(float(result.get(f"power_cost_{k}", 0) or 0) for k in stn_keys)
    result["dra_cost"]   = sum(float(result.get(f"dra_cost_{k}",   0) or 0) for k in stn_keys)
    result["num_pumps"]  = sum(int(result.get(f"num_pumps_{k}",    0) or 0) for k in stn_keys)


def _add_branch_deliveries(payload: dict, stations_in: list[dict]) -> list[dict]:
    """Return a copy of stations with branch-delivery amounts added in."""
    stations_copy = copy.deepcopy(stations_in)
    for branch_input in payload.get("branches") or []:
        jname  = (branch_input.get("junction_station_name") or "").strip().lower()
        bflow  = float(branch_input.get("flow_m3h") or 0.0)
        btype  = branch_input.get("branch_type", "offtake")
        if not jname or bflow <= 0:
            continue
        for stn in stations_copy:
            if _stn_key(stn.get("name", "")) == _stn_key(jname):
                if btype == "injection":
                    stn["supply"] = float(stn.get("supply") or 0.0) + bflow
                else:
                    stn["delivery"] = float(stn.get("delivery") or 0.0) + bflow
                break
    return stations_copy


def _solve_branches(payload: dict, mainline_result: dict, base_kwargs: dict) -> list[dict]:
    """Solve all branch lines using junction heads from the mainline result."""
    branch_results: list[dict] = []
    for branch_input in payload.get("branches") or []:
        jname_key = _stn_key(branch_input.get("junction_station_name") or "")
        junction_head = float(
            mainline_result.get(f"residual_head_in_{jname_key}")
            or mainline_result.get(f"sdh_{jname_key}")
            or mainline_result.get("terminal_head")
            or mainline_result.get("delivery_head", 0.0)
        )
        br = solve_branch(
            junction_suction_head=junction_head,
            branch={
                "flow_m3h": branch_input["flow_m3h"],
                "stations": branch_input["stations"],
                "terminal": branch_input["terminal"],
                "uid":      branch_input.get("uid", ""),
                "name":     branch_input.get("name", "Branch"),
            },
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
    return branch_results


def _post_process_engineering(result: dict, payload: dict, stn_keys: list[str]) -> dict:
    """Add engineering-accuracy fields: vapor check, motor overload, DRA corrections."""
    stations = payload.get("stations", [])
    ambient  = float(payload.get("Ambient_temp", 25.0) or 25.0)

    # Temperature-dependent DRA correction factor
    f_T = max(0.5, 1.0 - 0.005 * (ambient - 25.0))
    result["dra_temp_correction_factor"] = round(f_T, 4)

    motor_overload_warnings: list[dict] = []
    vapor_lock_warnings: list[dict] = []

    # Pump shear-degradation factor (20% loss per pump pass)
    f_shear = 0.80
    mop = float(payload.get("mop_kgcm2") or 0.0)
    mop_m = mop * 10000.0 / 850.0  # crude approximation kgf/cm² → m head (at ρ≈850)

    for idx, stn in enumerate(stations):
        k = stn_keys[idx] if idx < len(stn_keys) else _stn_key(stn.get("name", ""))
        num_pumps = int(result.get(f"num_pumps_{k}", 0) or 0)

        # DRA shear degradation
        dra_in = float(result.get(f"dra_ppm_{k}", 0) or 0)
        if num_pumps > 0:
            result[f"dra_ppm_post_shear_{k}"] = round(dra_in * (f_shear ** num_pumps), 2)
        else:
            result[f"dra_ppm_post_shear_{k}"] = dra_in

        # Motor overload: compare pump BkW vs motor kW capacity
        pump_bkw  = float(result.get(f"pump_bkw_{k}", 0) or 0)
        motor_kw  = float(stn.get("motor_kw") or stn.get("kW") or 0)
        if motor_kw > 0 and pump_bkw > motor_kw * 1.05:
            motor_overload_warnings.append({
                "station": stn.get("name", k),
                "pump_bkw": round(pump_bkw, 1),
                "motor_kw": round(motor_kw, 1),
                "overload_pct": round((pump_bkw / motor_kw - 1) * 100, 1),
            })

        # Vapor pressure / siphon check: residual head at station inlet vs elevation
        res_head   = float(result.get(f"residual_head_in_{k}") or result.get(f"sdh_{k}", 0) or 0)
        elev       = float(stn.get("elev", 0) or 0)
        vapor_head = 3.0  # conservative crude-oil vapor pressure head (m)
        if res_head > 0 and res_head < elev + vapor_head:
            vapor_lock_warnings.append({
                "station": stn.get("name", k),
                "residual_head_m": round(res_head, 1),
                "elevation_m": round(elev, 1),
                "min_safe_m": round(elev + vapor_head, 1),
            })

    if motor_overload_warnings:
        result["motor_overload_warnings"] = motor_overload_warnings
    if vapor_lock_warnings:
        result["vapor_lock_warnings"] = vapor_lock_warnings

    return result


# ---------------------------------------------------------------------------
# _run_single_point
# ---------------------------------------------------------------------------

def _run_single_point(payload: dict) -> dict:
    """Run a single steady-state optimisation (flow_rate mode)."""
    kwargs: dict = {k: payload[k] for k in _MAINLINE_KEYS if k in payload}
    for k in _MAINLINE_OPT:
        if k in payload and payload[k] is not None:
            kwargs[k] = payload[k]

    kwargs["stations"] = _add_branch_deliveries(payload, kwargs.get("stations", []))

    mainline_result = solve_pipeline_with_types(**kwargs)
    if mainline_result.get("error"):
        raise RuntimeError(mainline_result.get("message", "Solver returned error"))

    stn_keys = [_stn_key(s["name"]) for s in payload.get("stations", [])]
    _inject_aggregate_metrics(mainline_result, stn_keys)
    mainline_result["FLOW"] = float(kwargs.get("FLOW", 0))

    _post_process_engineering(mainline_result, payload, stn_keys)

    branch_results = _solve_branches(payload, mainline_result, kwargs)
    return {"mainline": mainline_result, "branches": branch_results}


# ---------------------------------------------------------------------------
# _run_daily_schedule
# ---------------------------------------------------------------------------

def _build_kwargs_for_hour(
    base_payload: dict,
    flow_m3h: float,
    kv_list: list[float],
    rho_list: list[float],
    segment_slices: list[list[dict]],
    hour_label: str,
    tou_rate: float | None,
) -> dict:
    """Build the per-hour solver kwargs from base payload + overrides."""
    kwargs: dict = {k: base_payload[k] for k in _MAINLINE_KEYS if k in base_payload}
    for k in _MAINLINE_OPT:
        if k in base_payload and base_payload[k] is not None:
            kwargs[k] = base_payload[k]

    kwargs["FLOW"]           = flow_m3h
    kwargs["KV_list"]        = kv_list
    kwargs["rho_list"]       = rho_list
    kwargs["segment_slices"] = segment_slices
    kwargs["start_time"]     = hour_label
    kwargs["hours"]          = 1.0
    # Remove full-run linefill (now handled via segment_slices directly)
    kwargs.pop("linefill", None)

    if tou_rate is not None:
        kwargs["Price_HSD"] = tou_rate

    kwargs["stations"] = _add_branch_deliveries(base_payload, kwargs.get("stations", []))
    return kwargs


def _get_tou_rate(tou_tariff: list[dict] | None, hour: int) -> float | None:
    """Return per-hour electricity rate from TOU tariff list (or None if not set)."""
    if not tou_tariff:
        return None
    h = hour % 24
    for slot in tou_tariff:
        try:
            s = int(slot.get("start_hr", 0))
            e = int(slot.get("end_hr",   24))
            r = float(slot.get("rate",   0))
        except (TypeError, ValueError):
            continue
        if s <= h < e:
            return r
    return None


def _variable_flow_candidates(
    base_flow: float,
    min_flow: float | None = None,
    max_flow: float | None = None,
    step: float = 25.0,
) -> list[float]:
    """Generate a coarse grid of candidate flows for variable-mode selection."""
    lo = float(min_flow) if min_flow else max(50.0, base_flow * 0.60)
    hi = float(max_flow) if max_flow else base_flow * 1.40
    cands: list[float] = []
    q = lo
    while q <= hi + 1e-6:
        cands.append(round(q, 1))
        q += step
    if not cands:
        cands = [base_flow]
    return cands


def _run_daily_schedule(payload: dict) -> dict:
    """
    Run the 24-hour daily schedule optimisation loop.

    For each hour:
      1. Advance the linefill queue by the previous hour's pumped volume.
      2. Compute hydraulic profiles from the new queue state.
      3. Solve the mainline (and all branches) for that hour.
      4. On infeasibility and retry_with_max_dra: apply max DRA and re-solve.
      5. Accumulate hourly results, linefill snapshots, costs.

    Returns a structured result dict suitable for JSON serialisation.
    """
    stations     = payload.get("stations", [])
    stn_keys     = [_stn_key(s["name"]) for s in stations]
    start_hour   = int(payload.get("start_hour", 7) or 7)
    base_flow    = float(payload.get("FLOW", 0) or 0)
    hourly_flow  = float(payload.get("hourly_flow") or base_flow or 0)
    flow_mode    = payload.get("flow_mode", "fixed")
    tou_tariff   = payload.get("tou_tariff")
    retry_dra    = bool(payload.get("retry_with_max_dra", False))
    daily_target = payload.get("daily_throughput_target")

    # Build initial linefill & day-plan DataFrames
    linefill_list = payload.get("linefill") or []
    day_plan_list = payload.get("day_plan")  or []
    vol_table = df_from_linefill(linefill_list)
    day_plan  = df_from_linefill(day_plan_list)
    if day_plan.empty:
        day_plan = None

    # Variable mode: compute per-hour flows that meet the daily throughput
    # using a simple proportional allocation weighted by hourly tariff.
    if flow_mode == "variable" and daily_target:
        daily_target = float(daily_target)
        # Distribute equally to start; tariff weighting below is a TODO
        hourly_flows = [daily_target / 24.0] * 24
    else:
        # Fixed mode: same flow every hour
        hourly_flows = [hourly_flow if hourly_flow > 0 else base_flow] * 24

    # -------------------------------------------------------------------------
    # Accumulation containers
    # -------------------------------------------------------------------------
    hourly_results:   list[dict]        = []
    branch_schedules: dict[str, list]   = {}
    linefill_snaps:   list[list[dict]]  = []  # JSON-safe per-hour snapshot
    hourly_errors:    list[str | None]  = []
    backtrack_notes:  list[str]         = []
    enforced_actions: list[dict]        = []
    total_cost        = 0.0
    total_volume      = 0.0

    for step in range(24):
        hour       = (start_hour + step) % 24
        hour_label = f"{hour:02d}:00"
        flow_h     = hourly_flows[step]
        tou_rate   = _get_tou_rate(tou_tariff, hour)

        # -- advance linefill queue (uses flow from PREVIOUS step) -----------
        if step > 0:
            prev_flow = hourly_flows[step - 1]
            vol_table, day_plan, _injected = shift_vol_linefill(
                vol_table, prev_flow, day_plan
            )

        # -- compute next-hour linefill for combine_volumetric_profiles ------
        if step < 23:
            next_flow  = hourly_flows[step + 1]
            vol_next, _, _ = shift_vol_linefill(
                vol_table.copy(), next_flow, day_plan.copy() if day_plan is not None else None
            )
        else:
            vol_next = vol_table.copy()

        # -- build hydraulic profiles for this hour --------------------------
        kv_list, rho_list, seg_slices = combine_volumetric_profiles(
            stations, vol_table, vol_next
        )

        # Fallback: if no linefill data, use payload KV/rho lists
        if not kv_list:
            kv_list    = list(payload.get("KV_list", [10.0]))
            rho_list   = list(payload.get("rho_list", [850.0]))
            seg_slices = _default_segment_slices(stations, kv_list, rho_list)

        # -- build per-hour kwargs -------------------------------------------
        kwargs = _build_kwargs_for_hour(
            payload, flow_h, kv_list, rho_list, seg_slices, hour_label, tou_rate
        )

        # -- solve mainline --------------------------------------------------
        hour_result: dict | None = None
        error_msg: str | None = None
        try:
            hour_result = solve_pipeline_with_types(**kwargs)
            if hour_result.get("error"):
                error_msg = hour_result.get("message", f"Hour {hour_label}: solver error")
                hour_result = None
        except Exception as exc:
            error_msg = f"Hour {hour_label}: {exc}"

        # Backtrack: retry with maximum DRA if infeasible
        if hour_result is None and retry_dra:
            try:
                kwargs_retry = dict(kwargs)
                kwargs_retry["RateDRA"] = payload.get("RateDRA", 0)
                hour_result = solve_pipeline_with_types(**kwargs_retry)
                if hour_result.get("error"):
                    hour_result = None
                else:
                    note = f"Hour {hour_label}: infeasible — retried with max DRA"
                    backtrack_notes.append(note)
                    enforced_actions.append({"hour": hour_label, "action": "retry_max_dra"})
            except Exception:
                hour_result = None

        if hour_result is None:
            # Use a zero/fallback record so the schedule continues
            hour_result = {
                "error": True,
                "message": error_msg or f"Hour {hour_label}: infeasible",
                "FLOW": flow_h,
                "power_cost": 0.0,
                "dra_cost": 0.0,
                "num_pumps": 0,
            }
            hourly_errors.append(error_msg or f"Hour {hour_label}: infeasible")
        else:
            _inject_aggregate_metrics(hour_result, stn_keys)
            hour_result["FLOW"] = flow_h
            _post_process_engineering(hour_result, payload, stn_keys)
            hourly_errors.append(None)

        hour_result["hour"]       = hour
        hour_result["hour_label"] = hour_label
        hour_result["step"]       = step
        hourly_results.append(hour_result)

        total_cost   += float(hour_result.get("power_cost", 0) or 0)
        total_cost   += float(hour_result.get("dra_cost",   0) or 0)
        total_volume += float(flow_h)

        # Linefill snapshot (JSON-safe)
        snap = vol_table.to_dict(orient="records") if not vol_table.empty else []
        linefill_snaps.append(snap)

        # Solve branches for this hour
        if hour_result.get("error") is None or hour_result.get("error") is False:
            br_list = _solve_branches(payload, hour_result, kwargs)
        else:
            br_list = []
        for idx, branch_input in enumerate(payload.get("branches") or []):
            bid = branch_input.get("uid") or branch_input.get("name", f"branch_{idx}")
            if bid not in branch_schedules:
                branch_schedules[bid] = []
            branch_schedules[bid].append(
                br_list[idx] if idx < len(br_list) else {"error": True, "hour": hour}
            )

    # Final linefill state = tomorrow's starting state
    final_vol     = vol_table.to_dict(orient="records") if not vol_table.empty else []
    final_day_plan = day_plan.to_dict(orient="records") if day_plan is not None and not day_plan.empty else []

    # Final segment-wise batch composition (from final vol_table)
    final_kv, final_rho, final_slices = map_vol_linefill_to_segments(vol_table, stations)
    final_segment_slices = [
        {
            "segment_idx": i,
            "station_name": stations[i]["name"] if i < len(stations) else f"seg_{i}",
            "slices": final_slices[i] if i < len(final_slices) else [],
            "avg_kv": final_kv[i]  if i < len(final_kv)  else 0.0,
            "avg_rho": final_rho[i] if i < len(final_rho) else 0.0,
        }
        for i in range(len(stations))
    ]

    return {
        "op_mode":          "daily_schedule",
        "mainline_schedule": hourly_results,
        "branch_schedules":  branch_schedules,
        "linefill_snaps":    linefill_snaps,
        "final_vol":         final_vol,
        "final_day_plan":    final_day_plan,
        "final_segment_slices": final_segment_slices,
        "hourly_flows":      hourly_flows,
        "total_cost":        round(total_cost, 2),
        "total_throughput":  round(total_volume, 2),
        "backtrack_notes":   backtrack_notes,
        "enforced_actions":  enforced_actions,
        "hourly_errors":     [e for e in hourly_errors if e is not None],
        # Also expose first-hour mainline result at top level for single-hour compatibility
        "mainline": hourly_results[0] if hourly_results else {},
        "branches": list(branch_schedules.values())[0] if branch_schedules else [],
    }


# ---------------------------------------------------------------------------
# _run_pumping_planner
# ---------------------------------------------------------------------------

def _run_pumping_planner(payload: dict) -> dict:
    """
    Run the multi-day pumping planner.

    proj_flow:  [{start: "YYYY-MM-DD HH:MM", end: "...", flow: m³/h}]
    proj_plan:  [{product, volume, visc, density, init_dra}] — supply batches
    planner_days: int (number of days to simulate)

    For each proj_flow interval:
      - Determine the number of hours in the interval
      - Run those hours using the schedule loop segment
      - Advance the linefill queue
    """
    proj_flow  = payload.get("proj_flow")  or []
    proj_plan  = payload.get("proj_plan")  or []
    planner_days = int(payload.get("planner_days") or 7)

    if not proj_flow:
        # Fall back to constant flow for planner_days × 24 hours
        base_flow = float(payload.get("FLOW") or payload.get("hourly_flow") or 0)
        proj_flow = [{
            "start": "2024-01-01 00:00",
            "end":   (datetime(2024, 1, 1) + timedelta(days=planner_days)).strftime("%Y-%m-%d %H:%M"),
            "flow":  base_flow,
        }]

    vol_table = df_from_linefill(payload.get("linefill") or [])
    day_plan  = df_from_linefill(proj_plan) if proj_plan else None
    if day_plan is not None and day_plan.empty:
        day_plan = None

    stations = payload.get("stations", [])
    stn_keys = [_stn_key(s["name"]) for s in stations]

    interval_results: list[dict] = []
    total_cost   = 0.0
    total_volume = 0.0

    for interval in proj_flow:
        try:
            dt_start = datetime.strptime(interval["start"], "%Y-%m-%d %H:%M")
            dt_end   = datetime.strptime(interval["end"],   "%Y-%m-%d %H:%M")
        except (KeyError, ValueError):
            continue

        interval_hours = max(1, int((dt_end - dt_start).total_seconds() / 3600))
        flow_h = float(interval.get("flow") or payload.get("FLOW") or 0)

        interval_cost   = 0.0
        interval_volume = 0.0

        for step in range(interval_hours):
            hour = (dt_start + timedelta(hours=step)).hour
            hour_label = f"{hour:02d}:00"
            tou_rate = _get_tou_rate(payload.get("tou_tariff"), hour)

            # Advance queue
            if step > 0:
                vol_table, day_plan, _ = shift_vol_linefill(vol_table, flow_h, day_plan)

            # Hydraulic profiles
            kv_list, rho_list, seg_slices = map_vol_linefill_to_segments(vol_table, stations)
            if not kv_list:
                kv_list    = list(payload.get("KV_list", [10.0]))
                rho_list   = list(payload.get("rho_list", [850.0]))
                seg_slices = _default_segment_slices(stations, kv_list, rho_list)

            kwargs = _build_kwargs_for_hour(
                payload, flow_h, kv_list, rho_list, seg_slices, hour_label, tou_rate
            )
            try:
                hr = solve_pipeline_with_types(**kwargs)
                if not hr.get("error"):
                    _inject_aggregate_metrics(hr, stn_keys)
                    pc = float(hr.get("power_cost", 0) or 0)
                    dc = float(hr.get("dra_cost",   0) or 0)
                    interval_cost   += pc + dc
                    interval_volume += flow_h
            except Exception:
                pass

        interval_results.append({
            "start":          interval["start"],
            "end":            interval["end"],
            "duration_hours": interval_hours,
            "flow_m3h":       flow_h,
            "volume_m3":      round(interval_volume, 1),
            "cost":           round(interval_cost,   2),
        })
        total_cost   += interval_cost
        total_volume += interval_volume

    final_vol = vol_table.to_dict(orient="records") if not vol_table.empty else []

    return {
        "op_mode":           "pumping_planner",
        "interval_results":  interval_results,
        "final_vol":         final_vol,
        "total_cost":        round(total_cost, 2),
        "total_throughput":  round(total_volume, 2),
        # Compatibility keys
        "mainline": {},
        "branches": [],
    }


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def _run_optimization(job_id: str, payload: dict) -> None:
    """Runs in a worker thread — dispatches to the right mode."""
    from backend.db import OptimizationJob, get_session  # deferred to avoid circular import

    with get_session() as session:
        job = session.get(OptimizationJob, job_id)
        if job:
            job.status = "running"
            session.commit()

    try:
        op_mode = payload.get("op_mode", "flow_rate")

        if op_mode == "daily_schedule":
            result_payload = _run_daily_schedule(payload)
        elif op_mode == "pumping_planner":
            result_payload = _run_pumping_planner(payload)
        else:
            result_payload = _run_single_point(payload)

        with get_session() as session:
            job = session.get(OptimizationJob, job_id)
            if job:
                job.status      = "success"
                job.result      = result_payload
                job.finished_at = datetime.utcnow()
                session.commit()

    except Exception as exc:
        err = str(exc)
        with get_session() as session:
            job = session.get(OptimizationJob, job_id)
            if job:
                job.status        = "failed"
                job.error_message = err
                job.finished_at   = datetime.utcnow()
                session.commit()
