import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_model import solve_pipeline


def _load_linefill() -> list[dict]:
    data_path = Path(__file__).resolve().parent / "data" / "representative_linefill.json"
    with data_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_representative_pipeline() -> tuple[list[dict], dict, list[float], list[float]]:
    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 2,
            "MinRPM": 1200,
            "DOL": 3000,
            "pump_type": "type1",
            "A": 0.0,
            "B": 0.0,
            "C": 190.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 85.0,
            "L": 60.0,
            "d": 0.7,
            "rough": 0.00004,
            "elev": 0.0,
            "min_residual": 35,
            "max_dr": 40,
            "power_type": "Grid",
            "rate": 0.0,
            "loopline": {
                "L": 55.0,
                "d": 0.68,
                "rough": 0.00004,
                "max_dr": 30,
            },
        },
        {
            "name": "Mid Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1100,
            "DOL": 2850,
            "pump_type": "type2",
            "A": 0.0,
            "B": 0.0,
            "C": 175.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 83.0,
            "L": 50.0,
            "d": 0.68,
            "rough": 0.00004,
            "elev": 5.0,
            "min_residual": 30,
            "max_dr": 30,
            "power_type": "Grid",
            "rate": 0.0,
        },
    ]
    terminal = {"name": "Terminal", "elev": 10.0, "min_residual": 30}
    kv_list = [1.3, 1.2]
    rho_list = [845.0, 842.0]
    return stations, terminal, kv_list, rho_list


def _representative_segment_profiles() -> list[list[dict[str, float]]]:
    return [
        [
            {"length_km": 30.0, "kv": 1.3, "rho": 845.0, "dra_ppm": 0.0},
            {"length_km": 30.0, "kv": 1.6, "rho": 846.0, "dra_ppm": 12.0},
        ],
        [
            {"length_km": 25.0, "kv": 1.2, "rho": 842.0, "dra_ppm": 0.0},
            {"length_km": 25.0, "kv": 1.4, "rho": 843.5, "dra_ppm": 8.0},
        ],
    ]


def test_daily_scheduler_path_completes_promptly() -> None:
    linefill = _load_linefill()
    stations, terminal, kv_list, rho_list = _build_representative_pipeline()

    start = time.perf_counter()
    current_linefill = copy.deepcopy(linefill)
    dra_reach_km = 40.0
    for step in range(6):
        start_hour = (step * 4) % 24
        result = solve_pipeline(
            copy.deepcopy(stations),
            terminal,
            FLOW=1700.0,
            KV_list=kv_list,
            rho_list=rho_list,
            RateDRA=5.0,
            Price_HSD=0.0,
            Fuel_density=820.0,
            Ambient_temp=25.0,
            linefill=copy.deepcopy(current_linefill),
            dra_reach_km=dra_reach_km,
            hours=4.0,
            start_time=f"{start_hour:02d}:00",
        )
        assert not result.get("error"), result.get("message")
        current_linefill = copy.deepcopy(result.get("linefill", current_linefill))
        dra_reach_km = result.get("dra_front_km", dra_reach_km)

    duration = time.perf_counter() - start
    assert duration < 25.0, f"Optimizer took too long: {duration:.2f}s"


def test_profile_cache_matches_baseline_and_improves_speed() -> None:
    import pipeline_model as pm

    linefill = _load_linefill()
    stations, terminal, kv_list, rho_list = _build_representative_pipeline()
    segment_profiles = _representative_segment_profiles()

    schedule_steps = 1
    solver_kwargs = dict(
        FLOW=1700.0,
        KV_list=kv_list,
        rho_list=rho_list,
        RateDRA=5.0,
        Price_HSD=0.0,
        Fuel_density=820.0,
        Ambient_temp=25.0,
        hours=4.0,
        dra_step=20,
        rpm_step=300,
    )

    def run_schedule() -> tuple[float, list[dict]]:
        current_linefill = copy.deepcopy(linefill)
        dra_reach_km = 40.0
        results: list[dict] = []
        start = time.perf_counter()
        for step in range(schedule_steps):
            start_hour = (step * 4) % 24
            result = pm.solve_pipeline(
                copy.deepcopy(stations),
                terminal,
                linefill=copy.deepcopy(current_linefill),
                dra_reach_km=dra_reach_km,
                start_time=f"{start_hour:02d}:00",
                segment_profiles=copy.deepcopy(segment_profiles),
                **solver_kwargs,
            )
            assert not result.get("error"), result.get("message")
            results.append(copy.deepcopy(result))
            current_linefill = copy.deepcopy(result.get("linefill", current_linefill))
            dra_reach_km = result.get("dra_front_km", dra_reach_km)
        duration = time.perf_counter() - start
        return duration, results

    original_flag = pm.HYDRAULICS_CACHE_ENABLED
    try:
        pm.HYDRAULICS_CACHE_ENABLED = False
        uncached_duration, uncached_results = run_schedule()

        pm.HYDRAULICS_CACHE_ENABLED = True
        cached_duration, cached_results = run_schedule()
    finally:
        pm.HYDRAULICS_CACHE_ENABLED = original_flag

    assert cached_results == uncached_results
    assert cached_duration <= uncached_duration * 0.6, (
        f"Caching did not materially reduce runtime: uncached={uncached_duration:.2f}s, "
        f"cached={cached_duration:.2f}s"
    )


def test_refine_recovers_lower_cost_when_coarse_hits_boundary() -> None:
    """Refinement should revisit the full DRA range when coarse hits an edge."""

    import pipeline_model as pm

    cost_map = {0: 1000.0, 5: 1000.0, 10: 900.0, 15: 850.0, 20: 1100.0}

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1200,
            "DOL": 3000,
            "A": 0.0,
            "B": 0.0,
            "C": 200.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 85.0,
            "L": 50.0,
            "d": 0.7,
            "rough": 4.0e-05,
            "elev": 0.0,
            "min_residual": 35,
            "max_dr": 20,
            "power_type": "Grid",
            "rate": 5.0,
        }
    ]
    terminal = {"name": "Terminal", "min_residual": 35, "elev": 0.0}

    common_kwargs = dict(
        FLOW=1500.0,
        KV_list=[3.0],
        rho_list=[850.0],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        hours=12.0,
        start_time="00:00",
        dra_step=5,
        rpm_step=50,
        enumerate_loops=False,
    )

    original_cache = pm._build_pump_option_cache
    original_solve = pm.solve_pipeline

    def fake_cache(station, opt, **kwargs):
        cache = original_cache(station, opt, **kwargs)
        dr_val = int(opt.get("dra_main", 0) or 0)
        target = cost_map.get(dr_val)
        if target is not None:
            delta = target - cache.get("power_cost", 0.0)
            cache["power_cost"] = target
            details = cache.get("pump_details")
            if isinstance(details, list) and details:
                per_detail = delta / len(details)
                for detail in details:
                    detail["power_cost"] = detail.get("power_cost", 0.0) + per_detail
        return cache

    coarse_results: list[dict] = []
    captured_ranges: list[dict[int, dict[str, tuple[int, int]]]] = []

    def wrapped_solve(*args, **kwargs):  # type: ignore[override]
        result = original_solve(*args, **kwargs)
        if kwargs.get("_internal_pass"):
            narrow = kwargs.get("narrow_ranges")
            if narrow is None:
                coarse_results.append(result)
            else:
                captured_ranges.append(narrow)
        return result

    with patch.object(pm, "_build_pump_option_cache", new=fake_cache):
        with patch.object(pm, "solve_pipeline", new=wrapped_solve):
            final_result = pm.solve_pipeline(stations, terminal, **common_kwargs)

    assert coarse_results, "Coarse optimisation did not run"
    assert captured_ranges, "Refinement pass did not receive narrowed ranges"

    coarse_cost = coarse_results[0].get("total_cost")
    assert coarse_cost == pytest.approx(cost_map[0])

    dra_range = captured_ranges[0][0]["dra_main"]
    assert dra_range == (0, 20)

    assert final_result.get("total_cost") == pytest.approx(cost_map[15])
    assert final_result.get("total_cost") < coarse_cost


def test_refine_considers_neighbourhood_when_coarse_prefers_zero_dra() -> None:
    """Refinement should explore coarse DRA neighbours even with tiny user steps."""

    import pipeline_model as pm

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1200,
            "DOL": 3000,
            "A": 0.0,
            "B": 0.0,
            "C": 200.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 85.0,
            "L": 50.0,
            "d": 0.7,
            "rough": 4.0e-05,
            "elev": 0.0,
            "min_residual": 35,
            "max_dr": 10,
            "power_type": "Grid",
            "rate": 5.0,
        }
    ]
    terminal = {"name": "Terminal", "min_residual": 35, "elev": 0.0}

    common_kwargs = dict(
        FLOW=1500.0,
        KV_list=[3.0],
        rho_list=[850.0],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        hours=12.0,
        start_time="00:00",
        dra_step=2,
        rpm_step=50,
        enumerate_loops=False,
    )

    original_cache = pm._build_pump_option_cache
    original_solve = pm.solve_pipeline

    stage_state = {"value": "outer"}

    stage_costs = {
        "coarse": {0: 900.0, 10: 1300.0},
        "refine": {0: 1400.0, 2: 1300.0, 4: 1200.0, 6: 1100.0, 8: 1000.0, 10: 800.0},
    }

    def staged_cache(station, opt, **kwargs):
        cache = original_cache(station, opt, **kwargs)
        dr_val = int(opt.get("dra_main", 0) or 0)
        stage_map = stage_costs.get(stage_state["value"], stage_costs["refine"])
        target = stage_map.get(dr_val)
        if target is not None:
            delta = target - cache.get("power_cost", 0.0)
            cache["power_cost"] = target
            details = cache.get("pump_details")
            if isinstance(details, list) and details:
                per_detail = delta / len(details)
                for detail in details:
                    detail["power_cost"] = detail.get("power_cost", 0.0) + per_detail
        return cache

    coarse_results: list[dict] = []

    def tracking_solve(*args, **kwargs):  # type: ignore[override]
        prev_stage = stage_state["value"]
        if kwargs.get("_internal_pass"):
            stage_state["value"] = "refine" if kwargs.get("narrow_ranges") else "coarse"
        else:
            stage_state["value"] = "outer"
        try:
            result = original_solve(*args, **kwargs)
        finally:
            stage_state["value"] = prev_stage
        if kwargs.get("_internal_pass") and kwargs.get("narrow_ranges") is None:
            coarse_results.append(result)
        return result

    with patch.object(pm, "_build_pump_option_cache", new=staged_cache):
        with patch.object(pm, "solve_pipeline", new=tracking_solve):
            final_result = pm.solve_pipeline(stations, terminal, **common_kwargs)

    assert coarse_results, "Coarse optimisation did not run"
    coarse_result = coarse_results[0]
    assert coarse_result.get("drag_reduction_origin_pump") == 0
    assert coarse_result.get("total_cost") == pytest.approx(stage_costs["coarse"][0])

    assert final_result.get("dra_ppm_origin_pump") == 1
    assert final_result.get("drag_reduction_origin_pump") > coarse_result.get("drag_reduction_origin_pump")
    assert final_result.get("total_cost") == pytest.approx(stage_costs["refine"][10])
    assert final_result.get("total_cost") < coarse_result.get("total_cost")
