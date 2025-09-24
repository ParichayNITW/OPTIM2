import contextlib
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_model import (
    solve_pipeline as _solve_pipeline,
    solve_pipeline_with_types as _solve_pipeline_with_types,
    _volume_from_km,
)
from schedule_utils import kv_rho_from_vol


def _null_spinner(_msg):
    return contextlib.nullcontext()


def _ensure_segment_slices(args, kwargs) -> None:
    if "segment_slices" in kwargs:
        return
    if len(args) >= 6:
        return
    if "stations" in kwargs:
        stations = kwargs["stations"]
    elif args:
        stations = args[0]
    else:
        return
    kwargs["segment_slices"] = [[] for _ in stations]


def solve_pipeline(*args, segment_slices=None, **kwargs):
    if "stations" in kwargs:
        stations = kwargs["stations"]
    elif args:
        stations = args[0]
    else:
        raise TypeError("stations must be provided")
    if segment_slices is None and "segment_slices" not in kwargs:
        kwargs["segment_slices"] = [[] for _ in stations]
    elif segment_slices is not None and "segment_slices" not in kwargs:
        kwargs["segment_slices"] = segment_slices
    return _solve_pipeline(*args, **kwargs)


def solve_pipeline_with_types(*args, segment_slices=None, **kwargs):
    if "stations" in kwargs:
        stations = kwargs["stations"]
    elif args:
        stations = args[0]
    else:
        raise TypeError("stations must be provided")
    if segment_slices is None and "segment_slices" not in kwargs:
        kwargs["segment_slices"] = [[] for _ in stations]
    elif segment_slices is not None and "segment_slices" not in kwargs:
        kwargs["segment_slices"] = segment_slices
    return _solve_pipeline_with_types(*args, **kwargs)


def test_run_all_updates_passes_segment_slices(monkeypatch):
    import pipeline_optimization_app as app

    session = app.st.session_state
    tracked_keys = [
        "stations",
        "terminal_name",
        "terminal_elev",
        "terminal_head",
        "FLOW",
        "RateDRA",
        "Price_HSD",
        "Fuel_density",
        "Ambient_temp",
        "MOP_kgcm2",
        "pump_shear_rate",
        "linefill_df",
    ]
    sentinel = object()
    previous_values = {key: session.get(key, sentinel) for key in tracked_keys}

    session["stations"] = [
        {
            "name": "Station A",
            "is_pump": True,
            "L": 12.0,
            "D": 0.7,
            "t": 0.007,
        },
        {
            "name": "Station B",
            "is_pump": False,
            "L": 8.0,
            "D": 0.7,
            "t": 0.007,
        },
    ]
    session["terminal_name"] = "Terminal"
    session["terminal_elev"] = 5.0
    session["terminal_head"] = 15.0
    session["FLOW"] = 1500.0
    session["RateDRA"] = 10.0
    session["Price_HSD"] = 0.0
    session["Fuel_density"] = 820.0
    session["Ambient_temp"] = 25.0
    session["MOP_kgcm2"] = 100.0
    session["pump_shear_rate"] = 0.0
    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 10000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": 15000.0,
                "Viscosity (cSt)": 3.0,
                "Density (kg/m³)": 830.0,
                app.INIT_DRA_COL: 0.0,
            },
        ]
    )
    session["linefill_df"] = vol_df
    session["linefill_vol_df"] = vol_df

    captured: dict[str, list] = {}

    def fake_solver(
        stations,
        terminal,
        flow,
        kv_list,
        rho_list,
        segment_slices,
        *args,
        **kwargs,
    ):
        captured["segment_slices"] = segment_slices
        result = {"stations_used": stations}
        for stn in stations:
            key = stn["name"].lower().replace(" ", "_")
            result[f"pipeline_flow_{key}"] = 0.0
        return result

    monkeypatch.setattr(app.pipeline_model, "solve_pipeline_with_types", fake_solver)
    monkeypatch.setattr(app, "build_station_table", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app.st, "spinner", _null_spinner)
    monkeypatch.setattr(app.st, "rerun", lambda: None)
    monkeypatch.setattr(app.st, "error", lambda msg: (_ for _ in ()).throw(AssertionError(msg)))

    try:
        app.run_all_updates()

        assert "segment_slices" in captured
        segment_slices = captured["segment_slices"]
        assert segment_slices is not None
        assert isinstance(segment_slices, list)
        assert len(segment_slices) == len(session["stations"])
        assert all(isinstance(entry, list) for entry in segment_slices)
        assert all(entry for entry in segment_slices)
        for slice_list in segment_slices:
            for entry in slice_list:
                assert {"length_km", "kv", "rho"} <= set(entry.keys())
    finally:
        app.invalidate_results()
        for key, value in previous_values.items():
            if value is sentinel:
                session.pop(key, None)
            else:
                session[key] = value


def test_kv_rho_from_vol_returns_segment_slices() -> None:
    stations = [
        {"name": "Station A", "L": 6.0, "D": 0.7, "t": 0.007},
        {"name": "Station B", "L": 4.0, "D": 0.7, "t": 0.007},
    ]
    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 8000.0,
                "Viscosity (cSt)": 2.0,
                "Density (kg/m³)": 820.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": 9000.0,
                "Viscosity (cSt)": 3.5,
                "Density (kg/m³)": 835.0,
            },
        ]
    )

    kv_list, rho_list, segment_slices = kv_rho_from_vol(vol_df, stations)

    assert kv_list and rho_list
    assert len(kv_list) == len(stations)
    assert len(rho_list) == len(stations)
    assert len(segment_slices) == len(stations)
    for idx, slices in enumerate(segment_slices):
        assert slices, f"Segment {idx} should have at least one slice"
        total_length = sum(entry["length_km"] for entry in slices)
        assert pytest.approx(total_length, rel=0.0, abs=1e-9) == stations[idx]["L"]


@pytest.mark.parametrize("mode", ["hourly", "daily"])
def test_scheduler_solver_receives_segment_slices(monkeypatch, mode):
    import importlib
    import pipeline_optimization_app as app

    stations = [
        {
            "name": "Station A",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1100,
            "DOL": 2800,
            "A": 0.0,
            "B": 0.0,
            "C": 180.0,
            "L": 6.0,
            "D": 0.7,
            "t": 0.007,
        },
        {
            "name": "Station B",
            "is_pump": False,
            "L": 4.0,
            "D": 0.7,
            "t": 0.007,
        },
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 25.0}
    current_vol = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 9000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 822.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": 7000.0,
                "Viscosity (cSt)": 3.2,
                "Density (kg/m³)": 830.0,
            },
        ]
    )
    future_vol = pd.DataFrame(
        [
            {
                "Product": "New Batch",
                "Volume (m³)": 5000.0,
                "Viscosity (cSt)": 4.0,
                "Density (kg/m³)": 840.0,
            },
            {
                "Product": "Batch 1",
                "Volume (m³)": 6000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 822.0,
            },
        ]
    )

    kv_list, rho_list, segment_slices = app.combine_volumetric_profiles(
        stations, current_vol, future_vol
    )

    captured: dict[str, list[list[dict]]] = {}

    def fake_solver(
        stations_arg,
        terminal_arg,
        flow_arg,
        kv_arg,
        rho_arg,
        segment_slices_arg,
        *args,
        **kwargs,
    ):  # type: ignore[override]
        captured["segment_slices"] = copy.deepcopy(segment_slices_arg)
        return {"error": False, "loop_usage": [], "linefill": []}

    monkeypatch.setattr(app.pipeline_model, "solve_pipeline", fake_solver)
    monkeypatch.setattr(importlib, "reload", lambda module: module)

    session = app.st.session_state
    tracked = [
        "MOP_kgcm2",
        "pump_shear_rate",
        "search_rpm_step",
        "search_dra_step",
        "search_coarse_multiplier",
        "search_state_top_k",
        "search_state_cost_margin",
    ]
    sentinel = object()
    previous = {key: session.get(key, sentinel) for key in tracked}

    try:
        session.setdefault("search_rpm_step", 25)
        session.setdefault("search_dra_step", 2)
        session.setdefault("search_coarse_multiplier", 5.0)
        session.setdefault("search_state_top_k", 50)
        session.setdefault("search_state_cost_margin", 5000.0)
        session.setdefault("MOP_kgcm2", 90.0)
        session.setdefault("pump_shear_rate", 0.0)

        flow = 1400.0 if mode == "daily" else 1200.0
        hours = 24.0 if mode == "daily" else 1.0
        app.solve_pipeline(
            stations=copy.deepcopy(stations),
            terminal=terminal,
            FLOW=flow,
            KV_list=kv_list,
            rho_list=rho_list,
            segment_slices=segment_slices,
            RateDRA=5.0,
            Price_HSD=0.0,
            Fuel_density=820.0,
            Ambient_temp=25.0,
            linefill_dict=[],
            hours=hours,
            start_time="00:00",
        )
    finally:
        for key, value in previous.items():
            if value is sentinel:
                session.pop(key, None)
            else:
                session[key] = value

    assert "segment_slices" in captured
    segment_slices = captured["segment_slices"]
    assert isinstance(segment_slices, list)
    assert segment_slices
    assert len(segment_slices) == len(stations)
    for slices in segment_slices:
        assert slices
        for entry in slices:
            assert {"length_km", "kv", "rho"} <= set(entry.keys())


def test_merge_segment_profiles_preserves_heterogeneity():
    import pipeline_optimization_app as app

    stations = [
        {
            "name": "Station A",
            "L": 10.0,
            "D": 0.7,
            "t": 0.007,
        }
    ]

    d_inner = stations[0]["D"] - 2.0 * stations[0]["t"]

    def make_vol_df(batches: list[tuple[float, float, float]], prefix: str) -> pd.DataFrame:
        rows = []
        for idx, (length_km, kv, rho) in enumerate(batches, start=1):
            volume_m3 = _volume_from_km(length_km, d_inner)
            rows.append(
                {
                    "Product": f"{prefix} Batch {idx}",
                    "Volume (m³)": volume_m3,
                    "Viscosity (cSt)": kv,
                    "Density (kg/m³)": rho,
                }
            )
        return pd.DataFrame(rows)

    current_df = make_vol_df(
        [(4.0, 2.0, 820.0), (6.0, 3.0, 830.0)],
        prefix="Current",
    )
    future_df = make_vol_df(
        [(5.0, 4.5, 845.0), (5.0, 2.5, 810.0)],
        prefix="Future",
    )

    kv_list, rho_list, segment_slices = app.combine_volumetric_profiles(
        stations, current_df, future_df
    )

    assert kv_list == pytest.approx([4.5])
    assert rho_list == pytest.approx([827.5])
    assert len(segment_slices) == 1

    slices = segment_slices[0]
    assert len(slices) == 2

    total_length = sum(entry["length_km"] for entry in slices)
    assert total_length == pytest.approx(stations[0]["L"], rel=0.0, abs=1e-6)

    assert slices[0]["length_km"] == pytest.approx(5.0, rel=0.0, abs=1e-6)
    assert slices[0]["kv"] == pytest.approx(4.5)
    assert slices[0]["rho"] == pytest.approx(845.0)

    assert slices[1]["length_km"] == pytest.approx(5.0, rel=0.0, abs=1e-6)
    assert slices[1]["kv"] == pytest.approx(3.0)
    assert slices[1]["rho"] == pytest.approx(830.0)

    assert any(entry["kv"] < kv_list[0] for entry in slices)


def _load_linefill() -> list[dict]:
    data_path = Path(__file__).resolve().parent / "data" / "representative_linefill.json"
    with data_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_daily_scheduler_path_completes_promptly() -> None:
    linefill = _load_linefill()

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
            "min_residual": 20,
            "max_dr": 30,
            "power_type": "Grid",
            "rate": 0.0,
        },
    ]
    terminal = {"name": "Terminal", "elev": 10.0, "min_residual": 30}
    kv_list = [1.3, 1.2]
    rho_list = [845.0, 842.0]

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
    assert duration < 45.0, f"Optimizer took too long: {duration:.2f}s"


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
        segment_slices=[[] for _ in stations],
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
        _ensure_segment_slices(args, kwargs)
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
        segment_slices=[[] for _ in stations],
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
            _ensure_segment_slices(args, kwargs)
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


def test_baseline_cases_run_even_with_aggressive_pruning() -> None:
    """Baseline feasibility checks should run alongside the refined search."""

    import pipeline_model as pm

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1100,
            "DOL": 3000,
            "A": 0.0,
            "B": 0.0,
            "C": 200.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 85.0,
            "L": 40.0,
            "d": 0.7,
            "rough": 4.0e-05,
            "elev": 0.0,
            "min_residual": 30,
            "max_dr": 6,
            "power_type": "Grid",
            "rate": 5.0,
        },
        {
            "name": "Mid Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1000,
            "DOL": 2800,
            "A": 0.0,
            "B": 0.0,
            "C": 180.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 82.0,
            "L": 35.0,
            "d": 0.68,
            "rough": 4.0e-05,
            "elev": 1.5,
            "min_residual": 18,
            "max_dr": 6,
            "power_type": "Grid",
            "rate": 5.0,
        },
    ]
    terminal = {"name": "Terminal", "min_residual": 18, "elev": 5.0}

    kwargs = dict(
        FLOW=900.0,
        KV_list=[3.0, 2.8],
        rho_list=[850.0, 848.0],
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
        state_top_k=1,
        state_cost_margin=0.0,
        enumerate_loops=False,
    )

    captured_ranges: list[dict[int, dict[str, tuple[int, int]]]] = []

    original_solve = pm.solve_pipeline

    def tracking_solve(*args, **kwargs):  # type: ignore[override]
        snapshot = None
        if kwargs.get("_internal_pass") and kwargs.get("narrow_ranges") is not None:
            snapshot = copy.deepcopy(kwargs["narrow_ranges"])
        _ensure_segment_slices(args, kwargs)
        result = original_solve(*args, **kwargs)
        if snapshot is not None:
            captured_ranges.append(snapshot)
        return result

    with patch.object(pm, "solve_pipeline", new=tracking_solve):
        result = pm.solve_pipeline(stations, terminal, **kwargs)

    assert not result.get("error"), result.get("message")
    assert captured_ranges, "No internal optimisation passes were recorded"
    assert len(captured_ranges) > 1, "Baseline feasibility runs were not executed"

    baseline_ranges = captured_ranges[1:]
    min_positive = max(1, kwargs["dra_step"])
    zero_case = None
    positive_found = False

    for case in baseline_ranges:
        dra_bounds = [entry.get("dra_main") for entry in case.values() if entry.get("dra_main")]
        positives = [bounds[0] for bounds in dra_bounds if bounds[0] > 0]
        if not positives and all(bounds == (0, 0) for bounds in dra_bounds):
            zero_case = case
        if positives:
            assert all(val == min_positive for val in positives)
            assert len(positives) == 1
            positive_found = True
        for entry in case.values():
            if "dra_loop" in entry:
                assert entry["dra_loop"] == (0, 0)

    assert zero_case is not None, "All-station zero-DRA baseline was not scheduled"
    assert positive_found, "Per-station positive baseline cases were missing"


def test_baseline_result_can_outperform_refine_when_cheaper() -> None:
    """Baseline runs should be able to win when they deliver a lower cost."""

    import pipeline_model as pm

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1100,
            "DOL": 3000,
            "A": 0.0,
            "B": 0.0,
            "C": 200.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 85.0,
            "L": 40.0,
            "d": 0.7,
            "rough": 4.0e-05,
            "elev": 0.0,
            "min_residual": 20,
            "max_dr": 6,
            "power_type": "Grid",
            "rate": 5.0,
        },
        {
            "name": "Mid Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1000,
            "DOL": 2800,
            "A": 0.0,
            "B": 0.0,
            "C": 180.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 82.0,
            "L": 35.0,
            "d": 0.68,
            "rough": 4.0e-05,
            "elev": 1.5,
            "min_residual": 18,
            "max_dr": 6,
            "power_type": "Grid",
            "rate": 5.0,
        },
    ]
    terminal = {"name": "Terminal", "min_residual": 18, "elev": 5.0}

    kwargs = dict(
        FLOW=900.0,
        KV_list=[3.0, 2.8],
        rho_list=[850.0, 848.0],
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
        state_top_k=1,
        state_cost_margin=0.0,
        enumerate_loops=False,
    )

    original_solve = pm.solve_pipeline
    refine_totals: list[float] = []
    baseline_totals: list[float] = []
    seen_refine = {"done": False}

    def favour_baseline(*args, **kwargs):  # type: ignore[override]
        _ensure_segment_slices(args, kwargs)
        result = original_solve(*args, **kwargs)
        if kwargs.get("_internal_pass") and kwargs.get("narrow_ranges") is not None:
            if not seen_refine["done"]:
                seen_refine["done"] = True
                refine_totals.append(float(result.get("total_cost", 0.0)))
                return result
            if not baseline_totals:
                adjusted_total = max(0.0, float(result.get("total_cost", 0.0)) - 5000.0)
                adjusted = copy.deepcopy(result)
                adjusted["total_cost"] = adjusted_total
                baseline_totals.append(adjusted_total)
                return adjusted
            baseline_totals.append(float(result.get("total_cost", 0.0)))
        return result

    with patch.object(pm, "solve_pipeline", new=favour_baseline):
        final_result = pm.solve_pipeline(stations, terminal, **kwargs)

    assert not final_result.get("error"), final_result.get("message")
    assert refine_totals, "Refined search result was not captured"
    assert baseline_totals, "Baseline runs did not execute"
    assert final_result.get("total_cost") == pytest.approx(baseline_totals[0])
    assert final_result.get("total_cost") < refine_totals[0]


def test_type_expansion_respects_station_maximum() -> None:
    """Enumerated pump-type combinations must obey the station-level cap."""

    station = {
        "name": "Origin Pump",
        "is_pump": True,
        "min_pumps": 1,
        "max_pumps": 2,
        "MinRPM": 1100,
        "DOL": 2800,
        "A": 0.0,
        "B": 0.0,
        "C": 180.0,
        "P": 0.0,
        "Q": 0.0,
        "R": 0.0,
        "S": 0.0,
        "T": 82.0,
        "L": 45.0,
        "d": 0.7,
        "rough": 4.0e-05,
        "elev": 0.0,
        "min_residual": 30,
        "max_dr": 20,
        "power_type": "Grid",
        "rate": 0.0,
        "pump_types": {
            "A": {"available": 2, "MinRPM": 1100, "DOL": 2800},
            "B": {"available": 1, "MinRPM": 1100, "DOL": 2800},
        },
    }
    terminal = {"name": "Terminal", "min_residual": 25, "elev": 0.0}
    kv_list = [3.0]
    rho_list = [850.0]
    origin_key = station["name"].strip().lower().replace(" ", "_")

    captured_totals: list[list[int]] = []

    def fake_solver(stations_arg, *_args, **_kwargs):
        totals = []
        for unit in stations_arg:
            combo = unit.get("active_combo") or {}
            totals.append(int(combo.get("A", 0)) + int(combo.get("B", 0)))
        captured_totals.append(totals)
        active = totals[0] if totals else 0
        return {
            "error": False,
            "total_cost": 100 - active,
            f"num_pumps_{origin_key}": active,
        }

    with patch("pipeline_model.solve_pipeline", side_effect=fake_solver):
        result = solve_pipeline_with_types(
            stations=[copy.deepcopy(station)],
            terminal=terminal,
            FLOW=1500.0,
            KV_list=kv_list,
            rho_list=rho_list,
            RateDRA=0.0,
            Price_HSD=0.0,
            Fuel_density=0.85,
            Ambient_temp=25.0,
            linefill=[],
            dra_reach_km=0.0,
            hours=12.0,
            start_time="00:00",
        )

    assert not result.get("error"), result.get("message")
    assert result[f"num_pumps_{origin_key}"] <= 2
    assert captured_totals, "No solve_pipeline calls captured"
    for totals in captured_totals:
        assert all(total <= 2 for total in totals)


def test_search_depth_controls_expand_combinatorial_search() -> None:
    """Custom search-depth knobs should widen the explored option space."""

    import pipeline_model as pm

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1100,
            "DOL": 1700,
            "A": 0.0,
            "B": 0.0,
            "C": 210.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 83.0,
            "L": 40.0,
            "d": 0.7,
            "rough": 4.0e-05,
            "elev": 0.0,
            "min_residual": 32,
            "max_dr": 20,
            "power_type": "Grid",
            "rate": 5.0,
        }
    ]
    terminal = {"name": "Terminal", "min_residual": 32, "elev": 0.0}

    base_kwargs = dict(
        FLOW=1400.0,
        KV_list=[2.5],
        rho_list=[845.0],
        segment_slices=[[] for _ in stations],
        RateDRA=5.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        hours=6.0,
        start_time="00:00",
        enumerate_loops=False,
    )

    def run_solver(**kwargs):
        stations_local = copy.deepcopy(stations)
        coarse_steps: list[int] = []
        original_solve = pm.solve_pipeline

        with patch.object(pm, "_build_pump_option_cache", wraps=pm._build_pump_option_cache) as mock_cache:
            def tracking(*args, **call_kwargs):  # type: ignore[override]
                if call_kwargs.get("_internal_pass"):
                    rpm_used = call_kwargs.get("rpm_step")
                    if isinstance(rpm_used, (int, float)):
                        coarse_steps.append(int(rpm_used))
                _ensure_segment_slices(args, call_kwargs)
                return original_solve(*args, **call_kwargs)

            with patch.object(pm, "solve_pipeline", side_effect=tracking):
                result = pm.solve_pipeline(
                    stations_local,
                    terminal,
                    **base_kwargs,
                    **kwargs,
                )
        return result, coarse_steps, mock_cache.call_count

    default_result, default_coarse, default_calls = run_solver()
    expanded_result, expanded_coarse, expanded_calls = run_solver(
        rpm_step=10,
        dra_step=1,
        coarse_multiplier=2.0,
        state_top_k=200,
        state_cost_margin=20000.0,
    )

    assert not default_result.get("error"), default_result.get("message")
    assert not expanded_result.get("error"), expanded_result.get("message")
    assert default_coarse, "Coarse search did not record any step"
    assert expanded_coarse, "Expanded search did not record any coarse steps"
    assert expanded_coarse[0] < default_coarse[0]
    assert expanded_calls > default_calls
