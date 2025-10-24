import contextlib
import copy
import json
import math
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import dra_utils

from pipeline_model import (
    solve_pipeline as _solve_pipeline,
    solve_pipeline_with_types as _solve_pipeline_with_types,
    _volume_from_km,
    _km_from_volume,
    _update_mainline_dra,
    _merge_queue,
    _ensure_queue_floor,
    _segment_profile_from_queue,
    _take_queue_front,
    _trim_queue_front,
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
            "d": 0.7,
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


def test_segment_floor_without_injection_is_infeasible():
    stations = [
        {
            "name": "Station A",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1000,
            "DOL": 1000,
            "A": 0.0,
            "B": 0.0,
            "C": 190.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 85.0,
            "L": 6.0,
            "D": 0.7,
            "t": 0.007,
            "max_dr": 0,
        },
        {
            "name": "Station B",
            "is_pump": False,
            "L": 8.0,
            "D": 0.7,
            "t": 0.007,
            "max_dr": 0,
        },
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 5.0}
    untreated_queue = [{"length_km": 14.0, "dra_ppm": 0.0}]

    result = solve_pipeline(
        stations,
        terminal,
        FLOW=500.0,
        KV_list=[3.0, 3.0],
        rho_list=[850.0, 850.0],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=850.0,
        Ambient_temp=25.0,
        linefill=untreated_queue,
        dra_reach_km=0.0,
        mop_kgcm2=100.0,
        hours=1.0,
        start_time="00:00",
        pump_shear_rate=0.0,
        segment_floors=[
            {"station_idx": 1, "length_km": stations[1]["L"], "dra_ppm": 5.0}
        ],
        enumerate_loops=False,
    )

    assert result.get("error") is True


def test_segment_floor_without_injection_short_circuits_option() -> None:
    queue = [
        {"length_km": 5.0, "dra_ppm": 0.05},
    ]
    station = {
        "idx": 0,
        "is_pump": False,
        "d_inner": 0.7,
        "kv": 0.0,
        "name": "Station A",
    }
    option = {"nop": 0, "dra_ppm_main": 0.0}
    segment_floor = {"length_km": 5.0, "dra_ppm": 0.05}

    _, _, inj_ppm, requires_injection = _update_mainline_dra(
        queue,
        station,
        option,
        5.0,
        _volume_from_km(5.0, 0.7),
        1.0,
        pump_running=False,
        pump_shear_rate=0.0,
        dra_shear_factor=0.0,
        shear_injection=False,
        is_origin=False,
        segment_floor=segment_floor,
    )

    assert inj_ppm == pytest.approx(0.0)
    assert requires_injection is True


def test_floor_requirement_enforces_positive_injection_and_reporting() -> None:
    import pipeline_model as pm

    stations = [
        {
            "name": "Station A",
            "is_pump": False,
            "L": 5.0,
            "D": 0.7,
            "t": 0.007,
            "rough": 0.00004,
            "kv": 0.0,
            "max_dr": 6,
        }
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 0.0}

    call_log: list[dict] = []

    def stub_update(queue, stn_data, opt, segment_length, flow_m3h, hours, **kwargs):
        if stn_data.get("dra_floor_ppm_min", 0.0) > 0.0:
            assert opt.get("dra_main", 0) > 0
            ppm_floor = float(stn_data.get("dra_floor_ppm_min", 0.0) or 0.0)
            assert opt.get("dra_ppm_main", 0.0) >= ppm_floor - 1e-9
        call_log.append(opt.copy())
        ppm = float(opt.get("dra_ppm_main", 0.0) or 0.0)
        return (
            [(segment_length, ppm)],
            [{"length_km": segment_length, "dra_ppm": ppm}],
            ppm,
            False,
        )

    def stub_segment_hydraulics(*_args, **_kwargs):
        return (0.0, 1.0, 1.0, 0.01)

    def stub_effective_dra_response(*_args, **_kwargs):
        return (5.0, 1.0)

    with patch("pipeline_model._update_mainline_dra", side_effect=stub_update), patch(
        "pipeline_model._segment_hydraulics_composite", side_effect=stub_segment_hydraulics
    ), patch("pipeline_model._segment_hydraulics", side_effect=stub_segment_hydraulics), patch(
        "pipeline_model._effective_dra_response", side_effect=stub_effective_dra_response
    ):
        result = pm.solve_pipeline(
            stations,
            terminal,
            FLOW=500.0,
            KV_list=[0.0],
            rho_list=[850.0],
            segment_slices=[[{"length_km": 5.0, "kv": 0.0, "rho": 850.0}]],
            RateDRA=1000.0,
            Price_HSD=0.0,
            Fuel_density=850.0,
            Ambient_temp=25.0,
            linefill=[{"length_km": 50.0, "dra_ppm": 0.05}],
            dra_reach_km=0.0,
            mop_kgcm2=100.0,
            hours=1.0,
            start_time="00:00",
            pump_shear_rate=0.0,
            segment_floors=[
                {"station_idx": 0, "length_km": 5.0, "dra_ppm": 0.05, "limited_by_station": True}
            ],
            enumerate_loops=False,
        )

    assert call_log, "No station options evaluated"
    assert result.get("error") is False
    inj_field = result.get("dra_ppm_station_a", 0.0)
    assert inj_field > 0.0
    floor_ppm = result.get("floor_min_ppm_station_a", 0.0)
    assert floor_ppm > 0.0
    assert result.get("floor_injection_applied_station_a") is True
    recorded_floor = result.get("floor_injection_ppm_station_a", 0.0)
    assert recorded_floor >= floor_ppm


def test_hourly_floor_requirement_forces_injection_each_hour() -> None:
    import pipeline_model as pm

    diameter = 0.7 - 2 * 0.007
    floor_ppm = 1.0
    segment_length = 8.0
    flow_rate = _volume_from_km(segment_length, diameter)

    stations = [
        {
            "name": "Station A",
            "is_pump": False,
            "L": segment_length,
            "d": 0.7,
            "t": 0.007,
            "rough": 4.0e-05,
            "elev": 0.0,
            "max_dr": 20,
        }
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 0}
    segment_floor = {"station_idx": 0, "length_km": segment_length, "dra_ppm": floor_ppm, "limited_by_station": True}

    linefill = [
        {"volume": _volume_from_km(segment_length, diameter), "dra_ppm": floor_ppm},
        {"volume": _volume_from_km(2.0, diameter), "dra_ppm": 0.0},
    ]

    baseline_tol = max(floor_ppm * 1e-6, 1e-9)
    current_linefill = linefill
    for hour in range(3):
        result = pm.solve_pipeline(
            stations,
            terminal,
            FLOW=flow_rate,
            KV_list=[2.5],
            rho_list=[850.0],
            segment_slices=[[]],
            RateDRA=500.0,
            Price_HSD=0.0,
            Fuel_density=850.0,
            Ambient_temp=25.0,
            linefill=current_linefill,
            dra_reach_km=0.0,
            mop_kgcm2=100.0,
            hours=1.0,
            start_time=f"{hour:02d}:00",
            pump_shear_rate=0.0,
            segment_floors=[segment_floor],
            enumerate_loops=False,
        )

        assert result.get("error") is False
        inj_ppm = float(result.get("dra_ppm_station_a", 0.0) or 0.0)
        floor_min = float(result.get("floor_min_ppm_station_a", 0.0) or 0.0)
        assert floor_min >= floor_ppm - baseline_tol
        assert inj_ppm >= floor_min - baseline_tol
        reports = result.get("reports") or []
        for report_entry in reports:
            report_result = report_entry.get("result") or {}
            logged_ppm = float(report_result.get("floor_injection_ppm_station_a", 0.0) or 0.0)
            assert logged_ppm >= floor_min - baseline_tol
            profile = report_result.get("dra_profile_station_a") or []
            inj_val = float(report_result.get("dra_ppm_station_a", 0.0) or 0.0)
            assert inj_val >= floor_min - baseline_tol
            assert inj_val > 0.0
            assert profile

        summary = result.get("floor_injection_summary") or []
        assert any(
            entry.get("station") == "station_a" and entry.get("ppm", 0.0) >= floor_min - baseline_tol
            for entry in summary
        )

        current_linefill = result.get("linefill", current_linefill)


def test_floor_schedule_logs_from_laced_queue_each_hour() -> None:
    import pipeline_model as pm

    diameter = 0.7 - 2 * 0.007
    segment_lengths = (4.0, 6.0)
    floor_requirements = (1.2, 1.8)
    flow_rate = _volume_from_km(5.0, diameter)

    stations = [
        {
            "name": "Station A",
            "is_pump": False,
            "L": segment_lengths[0],
            "d": 0.7,
            "t": 0.007,
            "rough": 4.0e-05,
            "max_dr": 24,
        },
        {
            "name": "Station B",
            "is_pump": False,
            "L": segment_lengths[1],
            "d": 0.7,
            "t": 0.007,
            "rough": 4.0e-05,
            "max_dr": 28,
        },
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 0.0}

    segment_floors = [
        {"station_idx": 0, "length_km": segment_lengths[0], "dra_ppm": floor_requirements[0]},
        {"station_idx": 1, "length_km": segment_lengths[1], "dra_ppm": floor_requirements[1]},
    ]

    segment_slices = [
        [
            {"length_km": segment_lengths[0] / 2.0, "kv": 2.6, "rho": 832.0},
            {"length_km": segment_lengths[0] / 2.0, "kv": 3.1, "rho": 838.0},
        ],
        [
            {"length_km": 2.0, "kv": 2.7, "rho": 830.0},
            {"length_km": segment_lengths[1] - 2.0, "kv": 3.3, "rho": 842.0},
        ],
    ]

    initial_linefill: list[dict] = []
    for idx, length in enumerate(segment_lengths):
        segment_volume = _volume_from_km(length, diameter)
        ppm_floor = floor_requirements[idx]
        initial_linefill.append({"volume": 0.6 * segment_volume, "dra_ppm": ppm_floor})
        initial_linefill.append({"volume": 0.4 * segment_volume, "dra_ppm": ppm_floor + 0.4})

    ppm_tol = max(max(floor_requirements) * 1e-6, 1e-9)
    current_linefill = copy.deepcopy(initial_linefill)

    for hour in range(3):
        result = pm.solve_pipeline(
            stations,
            terminal,
            FLOW=flow_rate,
            KV_list=[2.6, 3.0],
            rho_list=[832.0, 838.0],
            segment_slices=segment_slices,
            RateDRA=600.0,
            Price_HSD=0.0,
            Fuel_density=850.0,
            Ambient_temp=25.0,
            linefill=copy.deepcopy(current_linefill),
            dra_reach_km=0.0,
            mop_kgcm2=100.0,
            hours=1.0,
            start_time=f"{hour:02d}:00",
            pump_shear_rate=0.0,
            segment_floors=segment_floors,
            enumerate_loops=False,
        )

        assert result.get("error") is False

        summary = result.get("floor_injection_summary") or []
        summary_ppm = {
            entry.get("station"): float(entry.get("ppm", 0.0) or 0.0)
            for entry in summary
            if entry.get("station")
        }
        reports = result.get("reports") or []

        for idx, stn in enumerate(stations):
            station_key = stn["name"].strip().lower().replace(" ", "_")
            floor_ppm = floor_requirements[idx]
            assert result.get(f"floor_injection_applied_{station_key}") is True
            logged_ppm = float(result.get(f"floor_injection_ppm_{station_key}", 0.0) or 0.0)
            recorded_ppm = float(result.get(f"dra_ppm_{station_key}", 0.0) or 0.0)
            assert logged_ppm >= floor_ppm - ppm_tol
            assert recorded_ppm >= floor_ppm - ppm_tol
            assert summary_ppm.get(station_key, 0.0) >= floor_ppm - ppm_tol
            for report_entry in reports:
                report_result = report_entry.get("result") or {}
                hourly_logged = float(report_result.get(f"floor_injection_ppm_{station_key}", 0.0) or 0.0)
                assert hourly_logged >= floor_ppm - ppm_tol
                hourly_profile = report_result.get(f"dra_profile_{station_key}") or []
                hourly_inj = float(report_result.get(f"dra_ppm_{station_key}", 0.0) or 0.0)
                assert hourly_inj >= floor_ppm - ppm_tol
                if hourly_inj > 0.0:
                    assert hourly_profile
                else:
                    assert not hourly_profile

        current_linefill = result.get("linefill", current_linefill)


def test_floor_schedule_logs_when_queue_meets_floor_each_hour() -> None:
    import pipeline_model as pm

    diameter = 0.7 - 2 * 0.007
    segment_length = 6.0
    floor_ppm = 1.4
    flow_rate = _volume_from_km(segment_length, diameter)

    stations = [
        {
            "name": "Station A",
            "is_pump": False,
            "L": segment_length,
            "d": 0.7,
            "t": 0.007,
            "rough": 4.0e-05,
            "max_dr": 24,
        }
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 0.0}

    segment_floor = {
        "station_idx": 0,
        "length_km": segment_length,
        "dra_ppm": floor_ppm,
        "limited_by_station": True,
    }

    batch_volume = _volume_from_km(segment_length, diameter)
    linefill = [
        {"volume": batch_volume, "dra_ppm": floor_ppm},
        {"volume": batch_volume, "dra_ppm": floor_ppm},
    ]

    ppm_tol = max(floor_ppm * 1e-6, 1e-9)
    current_linefill = copy.deepcopy(linefill)

    for hour in range(3):
        result = pm.solve_pipeline(
            stations,
            terminal,
            FLOW=flow_rate,
            KV_list=[2.5],
            rho_list=[850.0],
            segment_slices=[[]],
            RateDRA=600.0,
            Price_HSD=0.0,
            Fuel_density=850.0,
            Ambient_temp=25.0,
            linefill=copy.deepcopy(current_linefill),
            dra_reach_km=0.0,
            mop_kgcm2=100.0,
            hours=1.0,
            start_time=f"{hour:02d}:00",
            pump_shear_rate=0.0,
            segment_floors=[segment_floor],
            enumerate_loops=False,
        )

        assert result.get("error") is False

        station_key = "station_a"
        floor_min = float(result.get(f"floor_min_ppm_{station_key}", 0.0) or 0.0)
        inj_ppm = float(result.get(f"dra_ppm_{station_key}", 0.0) or 0.0)
        logged_ppm = float(result.get(f"floor_injection_ppm_{station_key}", 0.0) or 0.0)

        assert floor_min >= floor_ppm - ppm_tol
        assert inj_ppm >= floor_min - ppm_tol
        assert logged_ppm >= floor_min - ppm_tol
        assert result.get(f"floor_injection_applied_{station_key}") is True

        summary = result.get("floor_injection_summary") or []
        assert any(
            entry.get("station") == station_key
            and float(entry.get("ppm", 0.0) or 0.0) >= floor_min - ppm_tol
            for entry in summary
        )

        reports = result.get("reports") or []
        for report_entry in reports:
            report_result = report_entry.get("result") or {}
            hourly_logged = float(report_result.get(f"floor_injection_ppm_{station_key}", 0.0) or 0.0)
            if hourly_logged <= 0.0:
                hourly_logged = float(report_result.get(f"dra_ppm_{station_key}", 0.0) or 0.0)
            assert hourly_logged >= floor_min - ppm_tol
            hourly_profile = report_result.get(f"dra_profile_{station_key}") or []
            hourly_inj = float(report_result.get(f"dra_ppm_{station_key}", 0.0) or 0.0)
            assert hourly_inj >= floor_min - ppm_tol
            if hourly_inj > 0.0:
                assert hourly_profile
            else:
                assert all(
                    float(slice_entry.get("dra_ppm", 0.0) or 0.0) <= 0.0
                    for slice_entry in hourly_profile
                )

        current_linefill = result.get("linefill", current_linefill)


def _basic_terminal(min_residual: float = 10.0) -> dict:
    return {"name": "Terminal", "elev": 0.0, "min_residual": min_residual}


def test_coarse_pass_skipped_when_grid_identical():
    stations = [
        {
            "name": "Station A",
            "is_pump": False,
            "L": 5.0,
            "D": 0.7,
            "t": 0.007,
            "max_dr": 0,
        }
    ]
    terminal = _basic_terminal()

    result = solve_pipeline(
        stations,
        terminal,
        FLOW=500.0,
        KV_list=[1.0],
        rho_list=[850.0],
        segment_slices=[[]],
        RateDRA=100.0,
        Price_HSD=0.0,
        Fuel_density=850.0,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        mop_kgcm2=100.0,
        hours=1.0,
        start_time="00:00",
        pump_shear_rate=0.0,
    )

    passes = result.get("executed_passes")
    assert passes == ["exhaustive"], f"Unexpected pass order: {passes}"


def test_refine_pass_skipped_when_ranges_unrestricted():
    stations = [
        {
            "name": "Station A",
            "is_pump": False,
            "L": 5.0,
            "D": 0.7,
            "t": 0.007,
            "max_dr": 10,
        }
    ]
    terminal = _basic_terminal()

    result = solve_pipeline(
        stations,
        terminal,
        FLOW=500.0,
        KV_list=[1.0],
        rho_list=[850.0],
        segment_slices=[[]],
        RateDRA=1000.0,
        Price_HSD=0.0,
        Fuel_density=850.0,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        mop_kgcm2=100.0,
        hours=1.0,
        start_time="00:00",
        pump_shear_rate=0.0,
    )

    passes = result.get("executed_passes")
    assert passes == ["coarse", "exhaustive"], f"Unexpected pass order: {passes}"
    assert "refine" not in passes


def test_solver_includes_full_grid_candidate(monkeypatch):
    station = {
        "name": "Station A",
        "is_pump": False,
        "L": 10.0,
        "D": 0.7,
        "t": 0.007,
        "max_dr": 40,
    }
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 10.0}
    station_key = "station_a"
    term_key = "terminal"

    coarse_result = {
        "error": False,
        "total_cost": 200.0,
        f"residual_head_{term_key}": 12.0,
        "residual": 12.0,
        f"drag_reduction_{station_key}": 0,
        "linefill": [],
    }
    exhaustive_result = {
        "error": False,
        "total_cost": 120.0,
        f"residual_head_{term_key}": 11.5,
        "residual": 11.5,
        f"drag_reduction_{station_key}": 10,
        "linefill": [],
    }

    import pipeline_model as pipeline_module

    original = pipeline_module.solve_pipeline

    def intercept(*args, **kwargs):
        if kwargs.get("_internal_pass"):
            if kwargs.get("_exhaustive_pass"):
                return copy.deepcopy(exhaustive_result)
            if kwargs.get("narrow_ranges") is None:
                return copy.deepcopy(coarse_result)
            return {"error": True}
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "solve_pipeline", intercept)

    result = _solve_pipeline(
        [station],
        terminal,
        1500.0,
        [0.0],
        [820.0],
        [[]],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=820.0,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        mop_kgcm2=100.0,
        hours=24.0,
        start_time="00:00",
        pump_shear_rate=0.0,
        loop_usage_by_station=[],
        enumerate_loops=False,
        rpm_step=1,
        dra_step=1,
    )

    assert result["total_cost"] == pytest.approx(exhaustive_result["total_cost"])
    assert result[f"drag_reduction_{station_key}"] == exhaustive_result[f"drag_reduction_{station_key}"]


def test_successful_exhaustive_short_circuits(monkeypatch):
    import pipeline_model as pm

    original_solve = pm.solve_pipeline
    internal_passes: list[bool] = []

    def counting_solver(*args, **kwargs):
        if kwargs.get("_internal_pass"):
            internal_passes.append(bool(kwargs.get("_exhaustive_pass")))
            if len(args) > 1:
                terminal = args[1]
            else:
                terminal = kwargs.get("terminal", {})
            term_name = terminal.get("name", "terminal").strip().lower().replace(" ", "_")
            residual_val = 12.0
            total_cost = 150.0
            if kwargs.get("_exhaustive_pass"):
                residual_val = 11.0
                total_cost = 90.0
            return {
                "error": False,
                "total_cost": total_cost,
                f"residual_head_{term_name}": residual_val,
                "residual": residual_val,
            }
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(pm, "solve_pipeline", counting_solver)

    stations = [
        {
            "name": "Station A",
            "is_pump": False,
            "L": 10.0,
            "D": 0.7,
            "t": 0.007,
            "max_dr": 0,
        }
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 10.0}

    result = pm.solve_pipeline(
        stations,
        terminal,
        1500.0,
        [1.0],
        [850.0],
        [[]],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=820.0,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        mop_kgcm2=100.0,
        hours=24.0,
        start_time="00:00",
        pump_shear_rate=0.0,
        loop_usage_by_station=[],
        enumerate_loops=False,
        rpm_step=5,
        dra_step=5,
    )

    assert internal_passes in ([False, True], [True])
    assert result["total_cost"] == pytest.approx(90.0)


def test_time_series_solver_backtracks_to_enforce_dra(monkeypatch):
    import pipeline_optimization_app as app

    stations_base = [
        {
            "name": "Station A",
            "is_pump": True,
            "L": 20.0,
            "D": 0.7,
            "t": 0.007,
            "max_pumps": 1,
        }
    ]
    term_data = {"name": "Terminal", "elev": 0.0, "min_residual": 10.0}
    hours = [0, 1]

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 12000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )
    vol_df = app.ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)
    vol_df["DRA ppm"] = vol_df[app.INIT_DRA_COL]
    dra_linefill = app.df_to_dra_linefill(vol_df)
    current_vol = app.apply_dra_ppm(vol_df.copy(), dra_linefill)

    call_log: list[tuple[int, bool]] = []

    def fake_solver(*solver_args, **solver_kwargs):
        (
            stations,
            terminal,
            flow,
            kv_list,
            rho_list,
            segment_slices,
            RateDRA,
            Price_HSD,
            fuel_density,
            ambient_temp,
            dra_linefill_in,
            dra_reach_km,
            mop_kgcm2,
            *_,
        ) = solver_args

        start_time = solver_kwargs.get("start_time", "00:00")
        hour = int(str(start_time).split(":")[0])
        positive = any(float(entry.get("dra_ppm", 0.0) or 0.0) > 0 for entry in dra_linefill_in or [])
        call_log.append((hour, positive))
        forced_detail = solver_kwargs.get("forced_origin_detail")
        hours_val = float(solver_kwargs.get("hours", 1.0))
        if hour == 0:
            if positive and not forced_detail:
                ppm_val = 3.0
                cost_factor = flow * 1000.0 * hours_val / 1e6
                dra_cost = ppm_val * cost_factor * RateDRA
                return {
                    "error": False,
                    "total_cost": 12.0 + dra_cost,
                    "linefill": [{"length_km": 6.0, "dra_ppm": ppm_val, "volume": 1000.0}],
                    "dra_front_km": 6.0,
                    "pipeline_flow_station_a": flow,
                    "dra_ppm_station_a": ppm_val,
                    "dra_cost_station_a": dra_cost,
                    "dra_profile_station_a": [
                        {"length_km": 6.0, "dra_ppm": ppm_val},
                    ],
                }
            if forced_detail:
                forced_ppm = float(forced_detail.get("dra_ppm", 0.0) or 0.0)
                cost_factor = flow * 1000.0 * hours_val / 1e6
                dra_cost = forced_ppm * cost_factor * RateDRA
                result = {
                    "error": False,
                    "total_cost": 12.0 + dra_cost,
                    "linefill": [
                        {"length_km": 6.0, "dra_ppm": forced_ppm or 3.0, "volume": 1000.0}
                    ],
                    "dra_front_km": 6.0,
                    "pipeline_flow_station_a": flow,
                    "dra_ppm_station_a": forced_ppm,
                    "dra_cost_station_a": dra_cost,
                    "forced_origin_detail": copy.deepcopy(forced_detail),
                    "dra_profile_station_a": [
                        {"length_km": 6.0, "dra_ppm": forced_ppm or 3.0},
                    ],
                }
                return result
            return {
                "error": False,
                "total_cost": 10.0,
                "linefill": [{"length_km": 0.0, "dra_ppm": 0.0, "volume": 0.0}],
                "dra_front_km": 0.0,
            }
        if hour == 1:
            if positive or float(dra_reach_km) > 0.0:
                ppm_from_queue = 0.0
                for entry in dra_linefill_in or []:
                    val = float(entry.get("dra_ppm", 0.0) or 0.0)
                    if val > 0.0:
                        ppm_from_queue = val
                        break
                ppm_val = max(ppm_from_queue, 3.0)
                cost_factor = flow * 1000.0 * hours_val / 1e6
                dra_cost = ppm_val * cost_factor * RateDRA
                return {
                    "error": False,
                    "total_cost": 11.0 + dra_cost,
                    "linefill": [{"length_km": 5.0, "dra_ppm": ppm_val, "volume": 1000.0}],
                    "dra_front_km": 5.0,
                    "pipeline_flow_station_a": flow,
                    "dra_ppm_station_a": ppm_val,
                    "dra_cost_station_a": dra_cost,
                    "dra_profile_station_a": [
                        {"length_km": 5.0, "dra_ppm": ppm_val},
                    ],
                }
            return {
                "error": True,
                "message": "No feasible pump combination found for stations.",
            }
        return {
            "error": False,
            "total_cost": 0.0,
            "linefill": [
                {**entry, "volume": float(entry.get("volume", 0.0) or 0.0)}
                for entry in dra_linefill_in or []
            ],
            "dra_front_km": float(dra_reach_km),
        }

    monkeypatch.setattr(app, "solve_pipeline", fake_solver)

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch",
                "Volume (m³)": 8000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )

    result = app._execute_time_series_solver(
        stations_base,
        term_data,
        hours,
        flow_rate=500.0,
        plan_df=plan_df,
        current_vol=current_vol,
        dra_linefill=dra_linefill,
        dra_reach_km=0.0,
        RateDRA=5.0,
        Price_HSD=0.0,
        fuel_density=820.0,
        ambient_temp=25.0,
        mop_kgcm2=100.0,
        pump_shear_rate=0.0,
        total_length=sum(stn["L"] for stn in stations_base),
        sub_steps=1,
    )

    assert result["error"] is None
    assert result["backtracked"] is True
    assert len(result["reports"]) == 2
    first_front = result["reports"][0]["result"].get("dra_front_km", 0.0)
    assert first_front > 0.0
    enforced_actions = result.get("enforced_origin_actions") or []
    assert enforced_actions
    enforced_detail = enforced_actions[0]
    enforced_ppm = float(enforced_detail.get("dra_ppm", 0.0) or 0.0)
    expected_treatable = app._estimate_treatable_length(
        total_length_km=sum(stn["L"] for stn in stations_base),
        total_volume_m3=float(vol_df["Volume (m³)"].sum()),
        flow_m3_per_hour=500.0,
        hours=1.0,
    )
    assert enforced_detail.get("length_km") == pytest.approx(expected_treatable)
    assert enforced_detail.get("treatable_km") == pytest.approx(expected_treatable)
    first_result = result["reports"][0]["result"]
    assert first_result.get("dra_ppm_station_a", 0.0) == pytest.approx(enforced_ppm)
    flow_main = float(first_result.get("pipeline_flow_station_a", 0.0) or 0.0)
    expected_cost = enforced_ppm * (flow_main * 1000.0 * 1.0 / 1e6) * 5.0
    assert first_result.get("dra_cost_station_a", 0.0) == pytest.approx(expected_cost)
    assert len(call_log) >= 3
    backtrack_notes = result.get("backtrack_notes") or []
    assert backtrack_notes
    note_text = backtrack_notes[0]
    assert "Origin queue updated" in note_text
    assert "scheduled at" in note_text.lower()
    assert "approximately" in note_text.lower()
    assert "Scheduled plan slices:" in note_text
    assert f"{enforced_detail.get('length_km', 0.0):.1f} km" in note_text

    plan_injections = enforced_detail.get("plan_injections") or []
    assert plan_injections
    for injection in plan_injections:
        label = app._format_plan_injection_label(injection)
        vol_val = float(injection.get("volume_m3", 0.0) or 0.0)
        ppm_val = float(injection.get("dra_ppm", enforced_detail.get("dra_ppm", 0.0)) or 0.0)
        assert label in note_text
        assert f"{vol_val:.0f} m³" in note_text
        assert f"{ppm_val:.2f} ppm" in note_text

    ppm_floor = enforced_ppm
    ppm_tol = max(ppm_floor * 1e-6, 1e-9)
    for entry in result["reports"]:
        hour_result = entry["result"]
        inj_ppm = float(hour_result.get("dra_ppm_station_a", 0.0) or 0.0)
        logged_floor = float(hour_result.get("floor_injection_ppm_station_a", 0.0) or 0.0)
        if logged_floor <= 0.0:
            logged_floor = inj_ppm
        assert logged_floor >= ppm_floor - ppm_tol
        profile = hour_result.get("dra_profile_station_a") or []
        if inj_ppm <= 0.0:
            assert all(
                float(slice_entry.get("dra_ppm", 0.0) or 0.0) <= 0.0
                for slice_entry in profile
            )
            continue
        assert inj_ppm >= ppm_floor - ppm_tol
        assert profile
        first_slice = profile[0]
        assert float(first_slice.get("dra_ppm", 0.0) or 0.0) == pytest.approx(inj_ppm)
        flow_val = float(hour_result.get("pipeline_flow_station_a", 0.0) or 0.0)
        expected_cost_hour = inj_ppm * (flow_val * 1000.0 * 1.0 / 1e6) * 5.0
        assert hour_result.get("dra_cost_station_a", 0.0) == pytest.approx(expected_cost_hour)

    first_snapshot = result["linefill_snaps"][0]
    assert isinstance(first_snapshot, pd.DataFrame)
    assert first_snapshot[app.INIT_DRA_COL].max() > 0.0

    warning_text = app._build_enforced_origin_warning(
        backtrack_notes, result.get("enforced_origin_actions")
    )
    assert f"{int(enforced_detail.get('hour', 0)) % 24:02d}:00" in warning_text
    for injection in plan_injections:
        label = app._format_plan_injection_label(injection)
        vol_val = float(injection.get("volume_m3", 0.0) or 0.0)
        ppm_val = float(injection.get("dra_ppm", enforced_detail.get("dra_ppm", 0.0)) or 0.0)
        assert f"Scheduled {label}" in warning_text
        assert f"{vol_val:.0f} m³" in warning_text
        assert f"{ppm_val:.2f} ppm" in warning_text


def test_enforce_minimum_origin_dra_updates_plan_split():
    import pipeline_optimization_app as app

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch 1",
                "Volume (m³)": 400.0,
                "Viscosity (cSt)": 2.4,
                "Density (kg/m³)": 818.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Plan Batch 2",
                "Volume (m³)": 700.0,
                "Viscosity (cSt)": 2.8,
                "Density (kg/m³)": 822.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Plan Batch 3",
                "Volume (m³)": 600.0,
                "Viscosity (cSt)": 3.0,
                "Density (kg/m³)": 825.0,
                app.INIT_DRA_COL: 0.0,
            },
        ]
    )

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 3500.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": 3000.0,
                "Viscosity (cSt)": 2.9,
                "Density (kg/m³)": 824.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Batch 3",
                "Volume (m³)": 2500.0,
                "Viscosity (cSt)": 3.1,
                "Density (kg/m³)": 828.0,
                app.INIT_DRA_COL: 0.0,
            },
        ]
    )

    state = {
        "plan": plan_df,
        "vol": vol_df,
        "dra_linefill": [],
        "dra_reach_km": 0.0,
    }

    changed = app._enforce_minimum_origin_dra(
        state,
        total_length_km=40.0,
        min_ppm=2.0,
        min_fraction=0.1,
    )

    assert changed is True

    enforced_plan = state["plan"]
    assert isinstance(enforced_plan, pd.DataFrame)
    assert len(enforced_plan) == 4
    enforced_volumes = enforced_plan["Volume (m³)"].tolist()
    enforced_ppm = enforced_plan[app.INIT_DRA_COL].tolist()
    assert enforced_volumes[:2] == pytest.approx([400.0, 500.0])
    assert all(ppm >= 2.0 for ppm in enforced_ppm[:2])
    assert enforced_ppm[2] == pytest.approx(0.0)

    queue = state["dra_linefill"]
    assert queue
    assert pytest.approx(queue[0]["volume"], rel=1e-6) == 900.0
    assert queue[0]["dra_ppm"] >= 2.0

    vol_snapshot = state["vol"]
    assert float(vol_snapshot.iloc[0][app.INIT_DRA_COL]) >= 2.0

    detail = state.get("origin_enforced_detail")
    assert detail is not None
    assert detail["volume_m3"] == pytest.approx(900.0)
    assert detail["dra_ppm"] >= 2.0
    assert detail.get("treatable_km", 0.0) == pytest.approx(0.0)
    assert detail.get("floor_ppm", 0.0) >= 2.0
    assert detail.get("floor_length_km", 0.0) == pytest.approx(4.0)
    injections = detail.get("plan_injections")
    assert isinstance(injections, list) and injections
    total_injected = sum(entry.get("volume_m3", 0.0) for entry in injections)
    assert total_injected == pytest.approx(900.0)


def test_enforce_minimum_origin_dra_caps_length_by_flow():
    import pipeline_optimization_app as app

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch",
                "Volume (m³)": 1000.0,
                "Viscosity (cSt)": 2.6,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch",
                "Volume (m³)": 12000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )
    vol_df = app.ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)

    state = {
        "plan": plan_df,
        "vol": vol_df,
        "dra_linefill": [],
        "dra_reach_km": 0.0,
    }

    total_length = 200.0
    flow_rate = 300.0
    hours = 1.0
    treatable_expected = app._estimate_treatable_length(
        total_length_km=total_length,
        total_volume_m3=float(vol_df["Volume (m³)"].sum()),
        flow_m3_per_hour=flow_rate,
        hours=hours,
    )
    changed = app._enforce_minimum_origin_dra(
        state,
        total_length_km=total_length,
        min_ppm=2.0,
        min_fraction=0.05,
        hourly_flow_m3=flow_rate,
        step_hours=hours,
    )

    assert changed is True
    queue = state["dra_linefill"]
    assert queue
    head = queue[0]
    assert head["length_km"] == pytest.approx(treatable_expected)
    detail = state.get("origin_enforced_detail")
    assert detail
    assert detail["length_km"] == pytest.approx(treatable_expected)
    assert detail.get("treatable_km") == pytest.approx(treatable_expected)
    assert detail.get("floor_length_km") == pytest.approx(treatable_expected)


def test_enforce_minimum_origin_dra_uses_queue_volume_when_snapshot_missing():
    import pipeline_optimization_app as app

    diameter = 0.7
    total_length = 120.0
    total_volume = _volume_from_km(total_length, diameter)

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch",
                "Volume (m³)": total_volume / 3.0,
                "Viscosity (cSt)": 2.8,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )

    queue = [
        {
            "length_km": total_length,
            "dra_ppm": 0.0,
            "volume": total_volume,
        }
    ]

    state = {
        "plan": plan_df,
        "vol": None,
        "linefill_snapshot": None,
        "dra_linefill": queue,
        "dra_reach_km": 0.0,
    }

    flow_length_per_hour = 4.0
    flow_rate = _volume_from_km(flow_length_per_hour, diameter)

    treatable_expected = app._estimate_treatable_length(
        total_length_km=total_length,
        total_volume_m3=total_volume,
        flow_m3_per_hour=flow_rate,
        hours=1.0,
        queue_entries=queue,
        plan_volume_m3=float(plan_df["Volume (m³)"].sum()),
    )

    changed = app._enforce_minimum_origin_dra(
        state,
        total_length_km=total_length,
        min_ppm=3.0,
        min_fraction=0.05,
        hourly_flow_m3=flow_rate,
        step_hours=1.0,
    )

    assert changed is True
    detail = state.get("origin_enforced_detail")
    assert detail
    assert detail["length_km"] == pytest.approx(treatable_expected)
    assert detail.get("treatable_km") == pytest.approx(treatable_expected)
    assert detail.get("floor_length_km") == pytest.approx(treatable_expected)


def test_enforce_minimum_origin_dra_requires_volume_column():
    import pipeline_optimization_app as app

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch 1",
                "Viscosity (cSt)": 2.4,
                "Density (kg/m³)": 818.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 3500.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )
    vol_df = app.ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)

    state = {
        "plan": plan_df,
        "vol": vol_df,
        "dra_linefill": [],
    }

    changed = app._enforce_minimum_origin_dra(
        state,
        total_length_km=40.0,
        min_ppm=2.0,
        min_fraction=0.1,
    )

    assert changed is False
    assert "missing a volume column" in state.get("origin_error", "")
    assert "origin_enforced_detail" not in state


def test_enforce_minimum_origin_dra_respects_baseline_requirement():
    import pipeline_optimization_app as app

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch",
                "Volume (m³)": 2000.0,
                "Viscosity (cSt)": 2.7,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )

    state = {
        "plan": plan_df,
        "vol": plan_df.copy(),
        "dra_linefill": [],
        "dra_reach_km": 0.0,
    }

    baseline = {"dra_ppm": 5.0, "length_km": 12.0, "dra_perc": 8.0}

    changed = app._enforce_minimum_origin_dra(
        state,
        total_length_km=40.0,
        min_ppm=2.0,
        min_fraction=0.05,
        baseline_requirement=baseline,
    )

    assert changed is True
    detail = state.get("origin_enforced_detail")
    assert detail
    summary = app._summarise_baseline_requirement(baseline)
    assert detail["dra_ppm"] >= summary["dra_ppm"]
    assert detail["length_km"] >= summary["length_km"]
    assert detail.get("floor_ppm") >= summary["dra_ppm"]
    assert detail.get("floor_length_km") >= summary["length_km"]


def test_enforce_minimum_origin_dra_preserves_segment_floors():
    import pipeline_optimization_app as app

    baseline = {
        "segments": [
            {"station_idx": 0, "length_km": 12.0, "dra_ppm": 3.0},
            {"station_idx": 1, "length_km": 8.0, "dra_ppm": 6.0},
        ]
    }

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 3000.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": 2500.0,
                app.INIT_DRA_COL: 0.0,
            },
        ]
    )

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Linefill 1",
                "Volume (m³)": 5000.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )
    vol_df = app.ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)

    state = {
        "plan": plan_df,
        "vol": vol_df,
        "dra_linefill": [],
        "dra_reach_km": 0.0,
    }

    changed = app._enforce_minimum_origin_dra(
        state,
        total_length_km=20.0,
        min_ppm=0.0,
        min_fraction=0.0,
        baseline_requirement=baseline,
    )

    assert changed is True

    queue = state["dra_linefill"]
    assert isinstance(queue, list) and len(queue) >= 2
    assert queue[0]["dra_ppm"] == pytest.approx(3.0)
    assert queue[1]["dra_ppm"] == pytest.approx(6.0)
    assert queue[0]["length_km"] == pytest.approx(12.0)
    assert queue[1]["length_km"] == pytest.approx(8.0)

    detail = state.get("origin_enforced_detail")
    assert detail is not None
    segments_detail = detail.get("segments")
    assert isinstance(segments_detail, list) and len(segments_detail) == 2
    assert segments_detail[0]["dra_ppm"] == pytest.approx(3.0)
    assert segments_detail[1]["dra_ppm"] == pytest.approx(6.0)


def test_compute_minimum_lacing_requirement_finds_floor():
    import pipeline_model as model

    stations = [
        {
            "name": "Station A",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "pump_type": "type1",
            "MinRPM": 3000,
            "DOL": 3000,
            "A": 0.0,
            "B": 0.0,
            "C": 4.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 75.0,
            "L": 10.0,
            "d": 0.7,
            "t": 0.007,
            "rough": 0.00004,
            "delivery": 0.0,
            "supply": 0.0,
            "max_dr": 70.0,
        }
    ]
    terminal = {"min_residual": 0.0, "elev": 0.0}

    min_suction = 1.0
    result = model.compute_minimum_lacing_requirement(
        stations,
        terminal,
        max_flow_m3h=900.0,
        max_visc_cst=2.5,
        min_suction_head=min_suction,
        fluid_density=0.0,
        mop_kgcm2=0.0,
    )

    assert result["length_km"] is None
    assert result["dra_perc"] is None
    assert result["dra_ppm"] is None
    assert result.get("dra_perc_uncapped") is None
    segments = result.get("segments")
    assert isinstance(segments, list) and len(segments) == 1

    flow = 900.0
    head_loss, *_ = model._segment_hydraulics(
        flow,
        stations[0]["L"],
        stations[0]["d"],
        stations[0]["rough"],
        2.5,
        0.0,
        0.0,
    )
    pump_info = model._pump_head(stations[0], flow, {"*": stations[0]["DOL"]}, 1)
    max_head = sum(p.get("tdh", 0.0) for p in pump_info)
    sdh_required = max(head_loss, 0.0)
    suction_head = max(min_suction, 0.0)
    available_head = max_head + suction_head
    expected_gap = max(sdh_required - available_head, 0.0)
    expected_unbounded = expected_gap / head_loss * 100.0 if head_loss > 0 else 0.0
    expected_dr = min(expected_unbounded, 70.0)
    seg_entry = segments[0]
    assert seg_entry["station_idx"] == 0
    assert seg_entry["length_km"] == pytest.approx(10.0)
    assert seg_entry["dra_perc"] == pytest.approx(expected_dr, rel=1e-2, abs=1e-2)
    dra_curve = dra_utils.DRA_CURVE_DATA.get(2.5)
    assert dra_curve is not None and not dra_curve.empty
    expected_ppm = math.ceil(model.get_ppm_for_dr(2.5, expected_dr) * 10.0) / 10.0
    assert seg_entry["dra_ppm"] == pytest.approx(expected_ppm)
    assert seg_entry["suction_head"] == pytest.approx(suction_head)
    assert seg_entry["available_head_before_suction"] == pytest.approx(available_head)
    assert seg_entry["max_head_available"] == pytest.approx(available_head)
    assert seg_entry["friction_head"] == pytest.approx(head_loss)


def test_compute_minimum_lacing_requirement_accounts_for_residual_head():
    import pipeline_model as model

    stations = [
        {
            "name": "Station A",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "pump_type": "type1",
            "MinRPM": 3000,
            "DOL": 3000,
            "A": 0.0,
            "B": 0.0,
            "C": 4.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 75.0,
            "L": 10.0,
            "d": 0.7,
            "t": 0.007,
            "rough": 0.00004,
            "delivery": 0.0,
            "supply": 0.0,
            "min_residual": 8.0,
            "max_dr": 70.0,
        }
    ]
    terminal = {"min_residual": 4.0, "elev": 0.0}

    min_suction = 2.0
    result = model.compute_minimum_lacing_requirement(
        stations,
        terminal,
        max_flow_m3h=1200.0,
        max_visc_cst=2.5,
        min_suction_head=min_suction,
        fluid_density=0.0,
        mop_kgcm2=0.0,
    )

    segments = result.get("segments")
    assert isinstance(segments, list) and len(segments) == 1
    seg_entry = segments[0]

    flow = 1200.0
    head_loss, *_ = model._segment_hydraulics(
        flow,
        stations[0]["L"],
        stations[0]["d"],
        stations[0]["rough"],
        2.5,
        0.0,
        0.0,
    )
    pump_info = model._pump_head(stations[0], flow, {"*": stations[0]["DOL"]}, 1)
    max_head = sum(p.get("tdh", 0.0) for p in pump_info)
    residual_head = max(stations[0]["min_residual"], terminal["min_residual"])
    sdh_required = terminal["min_residual"] + head_loss

    suction_head = max(residual_head, min_suction)
    available_head = max_head + suction_head
    expected_gap = max(sdh_required - available_head, 0.0)
    expected_dr = expected_gap / head_loss * 100.0 if head_loss > 0 else 0.0

    assert seg_entry["residual_head"] == pytest.approx(residual_head)
    assert seg_entry["available_head_before_suction"] == pytest.approx(available_head)
    assert seg_entry["suction_head"] == pytest.approx(suction_head)
    assert seg_entry["max_head_available"] == pytest.approx(available_head)
    assert seg_entry["dra_perc"] == pytest.approx(expected_dr, rel=1e-3, abs=1e-3)


def test_compute_minimum_lacing_requirement_flags_station_cap():
    import pipeline_model as model

    stations = [
        {
            "name": "Station A",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "pump_type": "type1",
            "MinRPM": 3000,
            "DOL": 3000,
            "A": 0.0,
            "B": 0.0,
            "C": 4.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 75.0,
            "L": 10.0,
            "d": 0.7,
            "t": 0.007,
            "rough": 0.00004,
            "delivery": 0.0,
            "supply": 0.0,
            "max_dr": 30.0,
        }
    ]
    terminal = {"min_residual": 0.0, "elev": 0.0}

    result = model.compute_minimum_lacing_requirement(
        stations,
        terminal,
        max_flow_m3h=1200.0,
        max_visc_cst=2.5,
        min_suction_head=1.5,
    )

    assert result["dra_perc"] is None
    assert result.get("dra_perc_uncapped") is None
    warnings = result.get("warnings")
    assert isinstance(warnings, list) and warnings
    assert any(w.get("type") == "station_max_dr_exceeded" for w in warnings if isinstance(w, dict))
    assert result.get("enforceable") is False
    segments = result.get("segments")
    assert isinstance(segments, list) and len(segments) == 1
    seg_entry = segments[0]
    assert seg_entry["dra_perc"] == pytest.approx(30.0)
    assert seg_entry.get("dra_perc_uncapped", 0.0) > seg_entry["dra_perc"]
    assert seg_entry.get("limited_by_station") is True
    rounded_ppm = math.ceil(model.get_ppm_for_dr(2.5, 30.0) * 10.0) / 10.0
    assert seg_entry.get("dra_ppm") == pytest.approx(rounded_ppm)


def test_compute_minimum_lacing_requirement_respects_single_type_series():
    import pipeline_model as model

    stations = [
        {
            "name": "Blend Pump",
            "is_pump": True,
            "max_pumps": 3,
            "pump_types": {
                "A": {
                    "available": 1,
                    "DOL": 3000,
                    "MinRPM": 3000,
                    "head_data": [
                        {"Flow (m³/hr)": 0.0, "Head (m)": 120.0},
                        {"Flow (m³/hr)": 500.0, "Head (m)": 110.0},
                        {"Flow (m³/hr)": 1000.0, "Head (m)": 100.0},
                    ],
                    "eff_data": [
                        {"Flow (m³/hr)": 0.0, "Efficiency (%)": 0.0},
                        {"Flow (m³/hr)": 1000.0, "Efficiency (%)": 80.0},
                    ],
                },
                "B": {
                    "available": 2,
                    "DOL": 3000,
                    "MinRPM": 3000,
                    "head_data": [
                        {"Flow (m³/hr)": 0.0, "Head (m)": 220.0},
                        {"Flow (m³/hr)": 500.0, "Head (m)": 210.0},
                        {"Flow (m³/hr)": 1000.0, "Head (m)": 200.0},
                    ],
                    "eff_data": [
                        {"Flow (m³/hr)": 0.0, "Efficiency (%)": 0.0},
                        {"Flow (m³/hr)": 1000.0, "Efficiency (%)": 80.0},
                    ],
                },
            },
            "L": 12.0,
            "d": 0.7,
            "t": 0.007,
            "rough": 0.00004,
            "delivery": 0.0,
            "supply": 0.0,
            "max_dr": 70.0,
        }
    ]

    terminal = {"name": "Terminal", "min_residual": 40.0, "elev": 0.0}

    result = model.compute_minimum_lacing_requirement(
        stations,
        terminal,
        max_flow_m3h=900.0,
        max_visc_cst=3.5,
        min_suction_head=10.0,
        fluid_density=850.0,
        mop_kgcm2=75.0,
    )

    segments = result.get("segments")
    assert isinstance(segments, list) and len(segments) == 1
    entry = segments[0]
    assert entry["available_head_before_suction"] == pytest.approx(444.0, rel=1e-3)

    stations[0]["allow_mixed_pump_types"] = True
    mixed = model.compute_minimum_lacing_requirement(
        stations,
        terminal,
        max_flow_m3h=900.0,
        max_visc_cst=3.5,
        min_suction_head=10.0,
        fluid_density=850.0,
        mop_kgcm2=75.0,
    )
    mixed_entry = mixed["segments"][0]
    assert mixed_entry["available_head_before_suction"] > entry["available_head_before_suction"]


def test_compute_minimum_lacing_requirement_matches_sample_case():
    import pipeline_model as model

    paradip_head = [
        {"Flow (m³/hr)": 0.0, "Head (m)": 401.43},
        {"Flow (m³/hr)": 500.88, "Head (m)": 412.86},
        {"Flow (m³/hr)": 1007.64, "Head (m)": 409.96},
        {"Flow (m³/hr)": 1503.24, "Head (m)": 396.29},
        {"Flow (m³/hr)": 1998.88, "Head (m)": 379.03},
        {"Flow (m³/hr)": 2497.51, "Head (m)": 351.03},
        {"Flow (m³/hr)": 2999.07, "Head (m)": 315.86},
        {"Flow (m³/hr)": 3169.14, "Head (m)": 299.96},
        {"Flow (m³/hr)": 3336.36, "Head (m)": 285.85},
    ]

    balasore_head = [
        {"Flow (m³/hr)": 0.0, "Head (m)": 450.0},
        {"Flow (m³/hr)": 500.0, "Head (m)": 450.0},
        {"Flow (m³/hr)": 1000.0, "Head (m)": 450.0},
        {"Flow (m³/hr)": 1500.0, "Head (m)": 440.0},
        {"Flow (m³/hr)": 2000.0, "Head (m)": 420.0},
        {"Flow (m³/hr)": 2500.0, "Head (m)": 400.0},
        {"Flow (m³/hr)": 3000.0, "Head (m)": 360.0},
        {"Flow (m³/hr)": 3500.0, "Head (m)": 315.0},
    ]

    stations = [
        {
            "name": "Paradip",
            "is_pump": True,
            "L": 158.0,
            "D": 0.762,
            "t": 0.0079248,
            "rough": 4e-05,
            "min_residual": 50.0,
            "max_pumps": 2,
            "MinRPM": 2200.0,
            "DOL": 2990.0,
            "max_dr": 35.0,
            "maop_head": 1000.0,
            "pump_types": {
                "A": {"available": 0},
                "B": {
                    "available": 2,
                    "MinRPM": 2200.0,
                    "DOL": 2990.0,
                    "head_data": paradip_head,
                    "eff_data": [],
                },
            },
        },
        {
            "name": "Balasore",
            "is_pump": True,
            "L": 170.0,
            "D": 0.762,
            "t": 0.0079248,
            "rough": 4e-05,
            "min_residual": 50.0,
            "max_pumps": 2,
            "MinRPM": 2200.0,
            "DOL": 2990.0,
            "max_dr": 35.0,
            "maop_head": 1000.0,
            "pump_types": {
                "A": {
                    "available": 2,
                    "MinRPM": 2200.0,
                    "DOL": 2990.0,
                    "head_data": balasore_head,
                    "eff_data": [],
                },
                "B": {"available": 0},
            },
        },
    ]

    terminal = {"name": "Haldia", "elev": 2.0, "min_residual": 50.0}

    result = model.compute_minimum_lacing_requirement(
        stations,
        terminal,
        max_flow_m3h=3169.0,
        max_visc_cst=20.0,
        min_suction_head=50.0,
        fluid_density=880.0,
        mop_kgcm2=59.0,
    )

    assert result["segments"], "Expected segment-wise baseline output"
    assert len(result["segments"]) == 2
    first, second = result["segments"]
    assert first["dra_perc"] == pytest.approx(28.770138, rel=1e-6)
    assert first["dra_ppm"] == pytest.approx(24.0, abs=1e-9)
    assert second["dra_perc"] == pytest.approx(24.128056, rel=1e-6)
    assert second["dra_ppm"] == pytest.approx(15.0, abs=1e-9)


def test_compute_minimum_lacing_requirement_handles_invalid_input():
    import pipeline_model as model

    stations = []
    terminal = {"min_residual": 0.0}

    result = model.compute_minimum_lacing_requirement(
        stations,
        terminal,
        max_flow_m3h=0.0,
        max_visc_cst=-1.0,
        min_suction_head=-5.0,
    )

    assert result["dra_perc"] == 0.0
    assert result["dra_ppm"] == 0.0
    assert result["length_km"] == 0.0
    assert result.get("segments") == []
    assert result.get("warnings") == []
    assert result.get("enforceable") is True


def test_segment_floors_overlay_queue_minimum():
    diameter = 0.7
    segment_lengths = [6.0, 152.0]
    flow_rate = _volume_from_km(segment_lengths[0], diameter)

    initial_queue = [
        {"length_km": sum(segment_lengths), "dra_ppm": 4.0},
    ]

    origin_data = {
        "idx": 0,
        "name": "Origin",
        "L": segment_lengths[0],
        "d_inner": diameter,
        "kv": 3.0,
        "dra_shear_factor": 0.0,
        "shear_injection": False,
        "linefill_slices": [],
    }
    origin_opt = {"nop": 1, "dra_ppm_main": 0.0}
    origin_floor = {"length_km": segment_lengths[0], "dra_ppm": 3.0}

    dra_segments, queue_after_origin, _, requires_injection = _update_mainline_dra(
        initial_queue,
        origin_data,
        origin_opt,
        segment_lengths[0],
        flow_rate,
        1.0,
        pump_running=True,
        pump_shear_rate=0.0,
        dra_shear_factor=0.0,
        shear_injection=False,
        is_origin=True,
        segment_floor=origin_floor,
    )
    assert requires_injection is True

    merged_origin = _merge_queue(
        [(entry["length_km"], entry["dra_ppm"]) for entry in queue_after_origin]
    )
    assert not dra_segments
    assert merged_origin == pytest.approx(
        [(segment_lengths[0], 0.0), (segment_lengths[1], 4.0)], rel=1e-3
    )

    downstream_data = {
        "idx": 1,
        "name": "Station B",
        "L": segment_lengths[1],
        "d_inner": diameter,
        "kv": 3.0,
        "dra_shear_factor": 0.0,
        "shear_injection": False,
        "linefill_slices": [],
    }
    downstream_opt = {"nop": 0, "dra_ppm_main": 0.0}
    downstream_floor = {"length_km": segment_lengths[1], "dra_ppm": 5.0}

    _, queue_after_downstream, _, requires_injection = _update_mainline_dra(
        queue_after_origin,
        downstream_data,
        downstream_opt,
        segment_lengths[1],
        flow_rate,
        1.0,
        pump_running=False,
        pump_shear_rate=0.0,
        dra_shear_factor=0.0,
        shear_injection=False,
        is_origin=False,
        segment_floor=downstream_floor,
    )
    assert requires_injection is True

    merged_final = _merge_queue(
        [(entry["length_km"], entry["dra_ppm"]) for entry in queue_after_downstream]
    )
    assert merged_final == pytest.approx(
        [(segment_lengths[0], 0.0), (segment_lengths[1], 4.0)], rel=1e-3
    )


def test_time_series_solver_reports_error_without_plan(monkeypatch):
    import pipeline_optimization_app as app

    stations_base = [
        {
            "name": "Station A",
            "is_pump": True,
            "L": 20.0,
            "D": 0.7,
            "t": 0.007,
            "max_pumps": 1,
        }
    ]
    term_data = {"name": "Terminal", "elev": 0.0, "min_residual": 10.0}
    hours = [0, 1]

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 8000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )
    vol_df = app.ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)
    dra_linefill = app.df_to_dra_linefill(vol_df)
    current_vol = app.apply_dra_ppm(vol_df.copy(), dra_linefill)

    def fake_solver(*solver_args, **solver_kwargs):
        (
            stations,
            terminal,
            flow,
            kv_list,
            rho_list,
            segment_slices,
            RateDRA,
            Price_HSD,
            fuel_density,
            ambient_temp,
            dra_linefill_in,
            dra_reach_km,
            mop_kgcm2,
            *_,
        ) = solver_args

        start_time = solver_kwargs.get("start_time", "00:00")
        hour = int(str(start_time).split(":")[0])
        positive = any(float(entry.get("dra_ppm", 0.0) or 0.0) > 0 for entry in dra_linefill_in or [])
        if hour == 0:
            if positive:
                return {
                    "error": False,
                    "total_cost": 12.0,
                    "linefill": [{"length_km": 6.0, "dra_ppm": 3.0}],
                    "dra_front_km": 6.0,
                }
            return {
                "error": False,
                "total_cost": 10.0,
                "linefill": [{"length_km": 0.0, "dra_ppm": 0.0}],
                "dra_front_km": 0.0,
            }
        if hour == 1:
            if positive or float(dra_reach_km) > 0.0:
                return {
                    "error": False,
                    "total_cost": 11.0,
                    "linefill": [{"length_km": 5.0, "dra_ppm": 2.5}],
                    "dra_front_km": 5.0,
                }
            return {
                "error": True,
                "message": "No feasible pump combination found for stations.",
            }
        return {
            "error": False,
            "total_cost": 0.0,
            "linefill": dra_linefill_in,
            "dra_front_km": float(dra_reach_km),
        }

    monkeypatch.setattr(app, "solve_pipeline", fake_solver)

    result = app._execute_time_series_solver(
        stations_base,
        term_data,
        hours,
        flow_rate=500.0,
        plan_df=None,
        current_vol=current_vol,
        dra_linefill=dra_linefill,
        dra_reach_km=0.0,
        RateDRA=5.0,
        Price_HSD=0.0,
        fuel_density=820.0,
        ambient_temp=25.0,
        mop_kgcm2=100.0,
        pump_shear_rate=0.0,
        total_length=sum(stn["L"] for stn in stations_base),
        sub_steps=1,
    )

    assert result["error"] == (
        "Zero DRA infeasible: upstream plan is empty so the enforced slug cannot be injected."
    )
    assert result["backtracked"] is False


def test_time_series_solver_enforces_when_head_untreated(monkeypatch):
    import pipeline_optimization_app as app

    stations_base = [
        {
            "name": "Station A",
            "is_pump": True,
            "L": 15.0,
            "D": 0.7,
            "t": 0.007,
            "max_pumps": 1,
        }
    ]
    term_data = {"name": "Terminal", "elev": 0.0, "min_residual": 10.0}
    hours = [0, 1]

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 6000.0,
                "Viscosity (cSt)": 2.5,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )
    vol_df = app.ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)
    dra_linefill = [
        {"volume": 3000.0, "dra_ppm": 0.0},
        {"volume": 3000.0, "dra_ppm": 2.5},
    ]
    current_vol = app.apply_dra_ppm(vol_df.copy(), dra_linefill)

    call_log: list[tuple[int, float]] = []

    def fake_solver(*solver_args, **solver_kwargs):
        (
            stations,
            terminal,
            flow,
            kv_list,
            rho_list,
            segment_slices,
            RateDRA,
            Price_HSD,
            fuel_density,
            ambient_temp,
            dra_linefill_in,
            dra_reach_km,
            mop_kgcm2,
            *_,
        ) = solver_args

        start_time = solver_kwargs.get("start_time", "00:00")
        hour = int(str(start_time).split(":")[0])
        head_ppm = 0.0
        if dra_linefill_in:
            try:
                head_ppm = float(dra_linefill_in[0].get("dra_ppm", 0.0) or 0.0)
            except (TypeError, ValueError):
                head_ppm = 0.0
        call_log.append((hour, head_ppm))
        if hour == 0:
            if head_ppm > 0.0:
                return {
                    "error": False,
                    "total_cost": 9.5,
                    "linefill": [
                        {"length_km": 3.5, "dra_ppm": head_ppm},
                        {"length_km": 2.5, "dra_ppm": head_ppm},
                    ],
                    "dra_front_km": 6.0,
                }
            return {
                "error": False,
                "total_cost": 9.0,
                "linefill": [
                    {"length_km": 3.5, "dra_ppm": 0.0},
                    {"length_km": 2.5, "dra_ppm": 2.5},
                ],
                "dra_front_km": 6.0,
            }
        if hour == 1:
            if head_ppm <= 0.0:
                return {
                    "error": True,
                    "message": "No feasible pump combination found for stations.",
                }
            return {
                "error": False,
                "total_cost": 10.0,
                "linefill": [
                    {"length_km": 2.0, "dra_ppm": 2.0},
                    {"length_km": 4.0, "dra_ppm": 2.0},
                ],
                "dra_front_km": 6.0,
            }
        return {
            "error": False,
            "total_cost": 0.0,
            "linefill": dra_linefill_in,
            "dra_front_km": float(dra_reach_km),
        }

    monkeypatch.setattr(app, "solve_pipeline", fake_solver)

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch 1",
                "Volume (m³)": 5000.0,
                "Viscosity (cSt)": 2.3,
                "Density (kg/m³)": 818.0,
                app.INIT_DRA_COL: 0.0,
            }
        ]
    )

    result = app._execute_time_series_solver(
        stations_base,
        term_data,
        hours,
        flow_rate=450.0,
        plan_df=plan_df,
        current_vol=current_vol,
        dra_linefill=dra_linefill,
        dra_reach_km=0.0,
        RateDRA=5.0,
        Price_HSD=0.0,
        fuel_density=820.0,
        ambient_temp=25.0,
        mop_kgcm2=100.0,
        pump_shear_rate=0.0,
        total_length=sum(stn["L"] for stn in stations_base),
        sub_steps=1,
    )

    assert result["error"] is None
    assert result["backtracked"] is True
    backtrack_notes = result.get("backtrack_notes") or []
    assert backtrack_notes
    assert any("Origin queue updated" in note for note in backtrack_notes)
    assert len(result["reports"]) == 2
    actions = result.get("enforced_origin_actions")
    assert isinstance(actions, list) and actions
    first_action = actions[0]
    assert first_action["hour"] == 0
    assert first_action["dra_ppm"] > 0.0
    assert first_action["volume_m3"] > 0.0
    hours_called = [hour for hour, _ in call_log]
    assert hours_called.count(1) >= 2
    # Ensure the retried call carried a positive head slug
    assert any(hour == 1 and ppm > 0.0 for hour, ppm in call_log)


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
    assert duration < 30.0, f"Optimizer took too long: {duration:.2f}s"


def test_daily_time_series_solver_finishes_within_budget() -> None:
    import pipeline_optimization_app as app

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1500,
            "DOL": 1500,
            "A": 0.0,
            "B": 0.0,
            "C": 185.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 85.0,
            "L": 55.0,
            "D": 0.714,
            "d": 0.714 - 2 * 0.007,
            "t": 0.007,
            "rough": 0.00004,
            "elev": 0.0,
            "min_residual": 28,
            "max_dr": 20,
            "power_type": "Grid",
            "rate": 0.0,
        },
        {
            "name": "Booster",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1480,
            "DOL": 1480,
            "A": 0.0,
            "B": 0.0,
            "C": 180.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 84.0,
            "L": 45.0,
            "D": 0.714,
            "d": 0.714 - 2 * 0.007,
            "t": 0.007,
            "rough": 0.00004,
            "elev": 4.0,
            "min_residual": 24,
            "max_dr": 15,
            "power_type": "Grid",
            "rate": 0.0,
        },
    ]

    term_data = {"name": "Terminal", "elev": 6.0, "min_residual": 22}
    total_length = sum(stn["L"] for stn in stations)
    hours = list(range(24))

    d_inner = stations[0]["D"] - 2 * stations[0]["t"]

    def _vol(length_km: float) -> float:
        return _volume_from_km(length_km, d_inner)

    current_vol = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": _vol(50.0),
                "Viscosity (cSt)": 2.3,
                "Density (kg/m³)": 816.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": _vol(30.0),
                "Viscosity (cSt)": 2.9,
                "Density (kg/m³)": 824.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Batch 3",
                "Volume (m³)": _vol(20.0),
                "Viscosity (cSt)": 3.2,
                "Density (kg/m³)": 829.0,
                app.INIT_DRA_COL: 0.0,
            },
        ]
    )

    dra_linefill = [
        {"volume": _vol(35.0), "dra_ppm": 5.0},
        {"volume": _vol(25.0), "dra_ppm": 0.0},
        {"volume": _vol(25.0), "dra_ppm": 7.0},
        {"volume": _vol(15.0), "dra_ppm": 0.0},
    ]

    current_vol = app.apply_dra_ppm(current_vol, dra_linefill)

    plan_df = pd.DataFrame(
        [
            {
                "Product": "Plan Batch 1",
                "Volume (m³)": _vol(30.0),
                "Viscosity (cSt)": 2.1,
                "Density (kg/m³)": 814.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Plan Batch 2",
                "Volume (m³)": _vol(35.0),
                "Viscosity (cSt)": 2.7,
                "Density (kg/m³)": 822.0,
                app.INIT_DRA_COL: 0.0,
            },
            {
                "Product": "Plan Batch 3",
                "Volume (m³)": _vol(35.0),
                "Viscosity (cSt)": 3.0,
                "Density (kg/m³)": 827.0,
                app.INIT_DRA_COL: 0.0,
            },
        ]
    )

    start = time.perf_counter()
    result = app._execute_time_series_solver(
        copy.deepcopy(stations),
        term_data,
        hours,
        flow_rate=1100.0,
        plan_df=plan_df,
        current_vol=current_vol,
        dra_linefill=copy.deepcopy(dra_linefill),
        dra_reach_km=0.0,
        RateDRA=5.0,
        Price_HSD=0.0,
        fuel_density=820.0,
        ambient_temp=25.0,
        mop_kgcm2=100.0,
        pump_shear_rate=0.0,
        total_length=total_length,
        sub_steps=1,
    )
    duration = time.perf_counter() - start

    assert result["error"] is None
    assert not result["backtracked"]
    assert len(result["reports"]) == len(hours)
    assert duration < 12.0, f"Daily schedule exceeded time budget: {duration:.2f}s"
    for entry in result["reports"]:
        assert "total_cost" in entry["result"]


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


def test_zero_dra_option_retained_under_pruning() -> None:
    """Zero-DRA scenarios should survive pruning-based passes."""

    import pipeline_model as pm

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1100,
            "DOL": 1100,
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
            "DOL": 1000,
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
        segment_slices=[[] for _ in stations],
    )

    def build_zero_ranges(station_defs: list[dict]) -> dict[int, dict[str, tuple[int, int]]]:
        zero_ranges: dict[int, dict[str, tuple[int, int]]] = {}
        for idx, stn in enumerate(station_defs):
            entry: dict[str, tuple[int, int]] = {"dra_main": (0, 0)}
            if stn.get("loopline"):
                entry["dra_loop"] = (0, 0)
            if stn.get("is_pump"):
                min_rpm = int(stn.get("MinRPM", 0) or 0)
                max_rpm = int(stn.get("DOL", min_rpm) or min_rpm)
                if max_rpm < min_rpm:
                    min_rpm, max_rpm = max_rpm, min_rpm
                entry["rpm"] = (min_rpm, max_rpm)
            zero_ranges[idx] = entry
        return zero_ranges

    zero_ranges = build_zero_ranges(stations)

    zero_only = pm.solve_pipeline(
        stations,
        terminal,
        **kwargs,
        narrow_ranges=zero_ranges,
        _internal_pass=True,
    )
    full_result = pm.solve_pipeline(stations, terminal, **kwargs)

    assert not zero_only.get("error"), zero_only.get("message")
    assert not full_result.get("error"), full_result.get("message")

    assert full_result.get("total_cost") == pytest.approx(zero_only.get("total_cost"))

    for stn in stations:
        key = stn["name"].strip().lower().replace(" ", "_")
        assert full_result.get(f"drag_reduction_{key}", 0) == 0


def test_min_rpm_baseline_retained_under_pruning() -> None:
    """Minimum-RPM baselines should remain feasible without fallback passes."""

    import pipeline_model as pm

    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1100,
            "DOL": 1100,
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
            "max_dr": 0,
            "power_type": "Grid",
            "rate": 5.0,
        }
    ]
    terminal = {"name": "Terminal", "min_residual": 30, "elev": 5.0}

    kwargs = dict(
        FLOW=900.0,
        KV_list=[3.0, 2.8],
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
        state_top_k=1,
        state_cost_margin=0.0,
        enumerate_loops=False,
        segment_slices=[[] for _ in stations],
    )

    def build_min_rpm_ranges(station_defs: list[dict]) -> dict[int, dict[str, tuple[int, int]]]:
        rpm_ranges: dict[int, dict[str, tuple[int, int]]] = {}
        for idx, stn in enumerate(station_defs):
            entry: dict[str, tuple[int, int]] = {"dra_main": (0, 0)}
            if stn.get("is_pump"):
                min_rpm = int(stn.get("MinRPM", 0) or 0)
                entry["rpm"] = (min_rpm, min_rpm)
            rpm_ranges[idx] = entry
        return rpm_ranges

    baseline_ranges = build_min_rpm_ranges(stations)

    constrained = pm.solve_pipeline(
        stations,
        terminal,
        **kwargs,
        narrow_ranges=baseline_ranges,
        _internal_pass=True,
    )
    unrestricted = pm.solve_pipeline(stations, terminal, **kwargs)

    assert not constrained.get("error"), constrained.get("message")
    assert not unrestricted.get("error"), unrestricted.get("message")

    assert constrained.get("speed_origin_pump") == pytest.approx(1100)
    assert unrestricted.get("total_cost") == pytest.approx(constrained.get("total_cost"))


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


def test_coarse_failure_triggers_refined_retry(monkeypatch):
    import pipeline_model

    rpm_step = 2
    dra_step = 1
    coarse_multiplier = 3.0

    coarse_rpm_step = int(round(rpm_step * coarse_multiplier)) if rpm_step > 0 else int(round(coarse_multiplier))
    if coarse_rpm_step <= 0:
        coarse_rpm_step = rpm_step if rpm_step > 0 else 1
    if coarse_multiplier >= 1.0 and rpm_step > 0:
        coarse_rpm_step = max(coarse_rpm_step, rpm_step)

    stations = [
        {
            "name": "Station Alpha",
            "is_pump": True,
            "L": 10.0,
            "D": 0.7,
            "t": 0.007,
            "MinRPM": 1200,
            "DOL": 1800,
            "max_dr": 20,
        }
    ]
    terminal = {"name": "Terminal", "min_residual": 0.0}
    success_payload = {
        "total_cost": 42.0,
        "residual_head_terminal": 0.0,
        "num_pumps_station_alpha": 1,
        "speed_station_alpha": 1200,
        "drag_reduction_station_alpha": 5,
    }
    call_log: list[tuple[bool, int | None, int | None, object]] = []

    original_solver = pipeline_model.solve_pipeline

    def fake_solver(*args, **kwargs):
        internal = kwargs.get("_internal_pass", False)
        rpm = kwargs.get("rpm_step")
        dra = kwargs.get("dra_step")
        narrow = kwargs.get("narrow_ranges")
        call_log.append((internal, rpm, dra, narrow))
        if internal:
            if rpm == coarse_rpm_step and narrow is None:
                return {"error": "coarse-failure"}
            if rpm == rpm_step and narrow is None:
                return success_payload.copy()
            return {"error": f"narrow-failure-{rpm}"}
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(pipeline_model, "solve_pipeline", fake_solver)

    result = pipeline_model.solve_pipeline_with_types(
        stations,
        terminal,
        FLOW=1000.0,
        KV_list=[1.0],
        rho_list=[850.0],
        segment_slices=[[]],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=850.0,
        Ambient_temp=25.0,
        linefill=None,
        dra_reach_km=0.0,
        mop_kgcm2=None,
        hours=24.0,
        start_time="00:00",
        pump_shear_rate=0.0,
        rpm_step=rpm_step,
        dra_step=dra_step,
        coarse_multiplier=coarse_multiplier,
    )

    assert not result.get("error")
    assert result["total_cost"] == success_payload["total_cost"]

    coarse_calls = [entry for entry in call_log if entry[0] and entry[1] == coarse_rpm_step and entry[3] is None]
    refined_calls = [entry for entry in call_log if entry[0] and entry[1] == rpm_step and entry[3] is None]

    assert coarse_calls, f"expected coarse call in log, saw {call_log!r}"
    assert refined_calls, f"expected refined retry in log, saw {call_log!r}"


def test_refined_retry_caps_type_combinations(monkeypatch):
    import math
    import pipeline_model as pm

    rpm_step = 50
    dra_step = 5
    coarse_multiplier = 3.0

    coarse_rpm_step = int(round(rpm_step * coarse_multiplier)) if rpm_step > 0 else int(round(coarse_multiplier))
    if coarse_rpm_step <= 0:
        coarse_rpm_step = rpm_step if rpm_step > 0 else 1
    if coarse_multiplier >= 1.0 and rpm_step > 0:
        coarse_rpm_step = max(coarse_rpm_step, rpm_step)

    stations = [
        {
            "name": "Station Mixed",
            "is_pump": True,
            "L": 10.0,
            "D": 0.7,
            "t": 0.007,
            "MinRPM": 900,
            "DOL": 1700,
            "max_dr": 0,
            "pump_types": {
                "A": {"available": 1, "MinRPM": 1000, "DOL": 1600},
                "B": {"available": 1, "MinRPM": 1100, "DOL": 1700},
            },
            "min_pumps": 2,
            "max_pumps": 2,
        }
    ]
    terminal = {"name": "Terminal", "min_residual": 0.0}

    base_kwargs = dict(
        FLOW=900.0,
        KV_list=[1.0],
        rho_list=[850.0],
        segment_slices=[[]],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=850.0,
        Ambient_temp=25.0,
        linefill=None,
        dra_reach_km=0.0,
        mop_kgcm2=None,
        hours=12.0,
        start_time="00:00",
        pump_shear_rate=0.0,
        rpm_step=rpm_step,
        dra_step=dra_step,
        coarse_multiplier=coarse_multiplier,
    )

    original_solve = pm.solve_pipeline
    original_allowed = pm._allowed_values
    real_product = pm.product

    def fake_allowed(min_val: int, max_val: int, step: int) -> list[int]:
        if min_val == 1000 and max_val == 1600:
            return [1000, 1080, 1160, 1240, 1320, 1400, 1480, 1560, 1600]
        if min_val == 1100 and max_val == 1700:
            return [1100, 1180, 1260, 1340, 1420, 1500, 1580, 1660, 1700]
        return original_allowed(min_val, max_val, step)

    recorded_lengths: list[tuple[int, ...]] = []
    tracking_mode = {"active": False}

    def tracking_product(*iterables):
        sequences = [list(seq) for seq in iterables]
        if tracking_mode["active"] and sequences:
            recorded_lengths.append(tuple(len(seq) for seq in sequences))
        return real_product(*sequences)

    refined_retry_seen: list[bool] = []
    coarse_seen: list[bool] = []

    def selective_solver(*args, **kwargs):
        internal = kwargs.get("_internal_pass", False)
        rpm_local = kwargs.get("rpm_step")
        refined_flag = bool(kwargs.get("refined_retry"))
        narrow = kwargs.get("narrow_ranges")
        tracking_mode["active"] = False
        if internal and not refined_flag and narrow is None and rpm_local == coarse_rpm_step:
            coarse_seen.append(True)
            return {"error": "forced-coarse"}
        if refined_flag:
            refined_retry_seen.append(True)
            tracking_mode["active"] = True
            try:
                return original_solve(*args, **kwargs)
            finally:
                tracking_mode["active"] = False
        return original_solve(*args, **kwargs)

    combo_cap = 12
    monkeypatch.setattr(pm, "solve_pipeline", selective_solver)
    monkeypatch.setattr(pm, "_allowed_values", fake_allowed)
    monkeypatch.setattr(pm, "product", tracking_product)
    monkeypatch.setattr(pm, "REFINED_RETRY_COMBO_CAP", combo_cap)

    result = pm.solve_pipeline_with_types(stations, terminal, **base_kwargs)

    assert coarse_seen, "coarse pass should be attempted"
    assert refined_retry_seen, "refined retry should be triggered"
    assert recorded_lengths, "expected to record per-type rpm lengths"
    assert all(math.prod(lengths) <= combo_cap for lengths in recorded_lengths)
    # Original lists contained 9 entries per type; ensure the retry reduced at least one list.
    assert any(any(length < 9 for length in lengths) for lengths in recorded_lengths)
    assert not result.get("error"), result.get("message")


def test_sequential_two_station_run_retains_carry(monkeypatch):
    import pipeline_model as pm

    def fake_segment(
        flow: float,
        length: float,
        d_inner: float,
        rough: float,
        kv: float,
        dra: float,
        dra_len: float,
        slices=None,
    ) -> tuple[float, float, float, float]:
        return 0.0, 1.0, 1000.0, float(flow)

    def fake_parallel(
        flow: float,
        L_main: float,
        d_main: float,
        rough_main: float,
        dra_main: float,
        dra_len_main: float,
        L_loop: float,
        d_loop: float,
        rough_loop: float,
        dra_loop: float,
        dra_len_loop: float,
        kv: float,
        slices=None,
    ) -> tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]]:
        main_stats = (1.0, 1.0, 1000.0, float(flow))
        loop_stats = (1.0, 1.0, 1000.0, 0.0)
        return 0.0, main_stats, loop_stats

    def fake_pump_cache(
        stn_data: dict,
        opt: dict,
        *,
        flow_total: float,
        hours: float,
        start_time: str,
        ambient_temp: float,
        fuel_density: float,
        price_hsd: float,
    ) -> dict:
        nop = int(opt.get("nop", 0) or 0)
        rpm = float(opt.get("rpm", 0) or 0)
        if nop <= 0 or rpm <= 0:
            return {
                "pump_details": [],
                "tdh": 0.0,
                "efficiency": 0.0,
                "pump_bkw": 0.0,
                "prime_kw": 0.0,
                "power_cost": 0.0,
            }
        pump_details = [
            {
                "tdh": 5.0,
                "eff": 75.0,
                "count": nop,
                "power_type": "Grid",
                "ptype": "mock",
                "rpm": int(rpm),
                "data": {"sfc_mode": "manual", "sfc": 0.0, "DOL": rpm},
            }
        ]
        return {
            "pump_details": pump_details,
            "tdh": 5.0,
            "efficiency": 75.0,
            "pump_bkw": 0.0,
            "prime_kw": 0.0,
            "power_cost": 0.0,
        }

    monkeypatch.setattr(pm, "_segment_hydraulics_composite", fake_segment)
    monkeypatch.setattr(pm, "_segment_hydraulics", fake_segment)
    monkeypatch.setattr(pm, "_parallel_segment_hydraulics", fake_parallel)
    monkeypatch.setattr(pm, "_build_pump_option_cache", fake_pump_cache)
    monkeypatch.setattr(pm, "_downstream_requirement", lambda *args, **kwargs: 0)

    stations = [
        {
            "name": "Station A",
            "is_pump": True,
            "L": 10.0,
            "D": 0.7,
            "t": 0.007,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1000,
            "DOL": 1000,
            "max_dr": 0,
            "min_residual": 60,
        },
        {
            "name": "Station B",
            "is_pump": False,
            "L": 8.0,
            "D": 0.7,
            "t": 0.007,
            "max_dr": 0,
        },
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 10.0}
    linefill = [{"length_km": 12.0, "dra_ppm": 25.0}]

    result = pm.solve_pipeline(
        stations,
        terminal,
        500.0,
        [1.0, 1.0],
        [850.0, 850.0],
        [[] for _ in stations],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=850.0,
        Ambient_temp=25.0,
        linefill=linefill,
        dra_reach_km=0.0,
        mop_kgcm2=100.0,
        hours=1.0,
        start_time="00:00",
        pump_shear_rate=0.0,
        enumerate_loops=False,
    )

    assert not result.get("error"), result.get("message")

    downstream_key = stations[1]["name"].strip().lower().replace(" ", "_")
    profile = result.get(f"dra_profile_{downstream_key}")
    assert profile is not None, "Downstream station should report a DRA profile"

    flow_rate = 500.0
    hours = 1.0
    queue_initial = list(linefill)

    dra_a, queue_after_a, _, _ = _update_mainline_dra(
        queue_initial,
        {"is_pump": True, "d_inner": 0.7, "idx": 0},
        {"nop": 1, "dra_ppm_main": 0},
        stations[0]["L"],
        flow_rate,
        hours,
        pump_running=True,
    )
    assert queue_after_a, "Origin station should produce a downstream queue"

    dra_b, queue_after_b, _, _ = _update_mainline_dra(
        queue_after_a,
        {"is_pump": False, "d_inner": 0.7, "idx": 1},
        {"nop": 0, "dra_ppm_main": 0},
        stations[1]["L"],
        flow_rate,
        hours,
        pump_running=False,
    )

    carried = [length for length, ppm in dra_b if ppm > 0]
    assert carried, "Sequential update should retain the carried DRA"
    ppm_values = [ppm for _, ppm in dra_b if ppm > 0]
    assert ppm_values[0] == pytest.approx(25.0)
    assert any(float(entry.get("dra_ppm", 0.0) or 0.0) > 0 for entry in queue_after_b)


def test_consecutive_injections_extend_dra_slug() -> None:
    """Repeated injections should lengthen the treated reach hour by hour."""

    diameter = 0.8
    pumped_speed = 2.0
    hours = 1.0
    flow_m3h = _volume_from_km(pumped_speed, diameter)
    pumped_length = _km_from_volume(flow_m3h * hours, diameter)

    initial_queue = [{"length_km": 25.0, "dra_ppm": 10.0}]
    station = {"idx": 0, "is_pump": True, "d_inner": diameter}
    operating = {"nop": 1, "dra_ppm_main": 12.0}

    _, queue_after_hour1, _, _ = _update_mainline_dra(
        initial_queue,
        station,
        operating,
        5.0,
        flow_m3h,
        hours,
        pump_running=True,
        pump_shear_rate=1.0,
    )

    assert queue_after_hour1
    assert queue_after_hour1[0]["dra_ppm"] == pytest.approx(operating["dra_ppm_main"], rel=1e-6)
    assert queue_after_hour1[0]["length_km"] == pytest.approx(pumped_length, rel=1e-6)
    assert queue_after_hour1[1]["dra_ppm"] == pytest.approx(initial_queue[0]["dra_ppm"], rel=1e-6)
    assert queue_after_hour1[1]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"] - pumped_length,
        rel=1e-6,
    )

    _, queue_after_hour2, _, _ = _update_mainline_dra(
        queue_after_hour1,
        station,
        operating,
        5.0,
        flow_m3h,
        hours,
        pump_running=True,
        pump_shear_rate=1.0,
    )

    assert queue_after_hour2
    assert queue_after_hour2[0]["dra_ppm"] == pytest.approx(operating["dra_ppm_main"], rel=1e-6)
    assert queue_after_hour2[0]["length_km"] == pytest.approx(pumped_length * 2.0, rel=1e-6)
    assert queue_after_hour2[1]["dra_ppm"] == pytest.approx(initial_queue[0]["dra_ppm"], rel=1e-6)
    assert queue_after_hour2[1]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"] - pumped_length * 2.0,
        rel=1e-6,
    )


def test_update_mainline_dra_ignores_non_enforced_floor() -> None:
    """Baseline floors flagged as non-enforcing should not overwrite the queue."""

    diameter = 0.7
    flow_m3h = 1000.0
    hours = 1.0
    pumped_length = _km_from_volume(flow_m3h * hours, diameter)

    initial_queue = [{"length_km": 150.0, "dra_ppm": 1.0}]
    station = {"idx": 0, "is_pump": True, "d_inner": diameter, "kv": 5.0}
    option = {"nop": 1, "dra_ppm_main": 7.0}

    dra_segments, queue_after, _, requires_injection = _update_mainline_dra(
        initial_queue,
        station,
        option,
        158.0,
        flow_m3h,
        hours,
        pump_running=True,
        pump_shear_rate=0.0,
        segment_floor={
            "length_km": 158.0,
            "dra_ppm": 4.0,
            "enforce_queue": False,
        },
    )

    assert not requires_injection
    assert dra_segments
    assert queue_after
    assert queue_after[0]["dra_ppm"] == pytest.approx(option["dra_ppm_main"], rel=1e-9)
    assert queue_after[0]["length_km"] == pytest.approx(pumped_length, rel=1e-6)
    assert queue_after[-1]["dra_ppm"] == pytest.approx(initial_queue[0]["dra_ppm"], rel=1e-9)


def test_update_mainline_dra_enforces_floor_when_requested() -> None:
    """Explicit enforcement should raise the queue to the requested floor."""

    diameter = 0.7
    flow_m3h = 1000.0
    hours = 1.0
    pumped_length = _km_from_volume(flow_m3h * hours, diameter)

    initial_queue = [{"length_km": 150.0, "dra_ppm": 1.0}]
    station = {"idx": 0, "is_pump": True, "d_inner": diameter, "kv": 5.0}
    option = {"nop": 1, "dra_ppm_main": 7.0}

    _, queue_after, _, requires_injection = _update_mainline_dra(
        initial_queue,
        station,
        option,
        158.0,
        flow_m3h,
        hours,
        pump_running=True,
        pump_shear_rate=0.0,
        segment_floor={
            "length_km": 40.0,
            "dra_ppm": 4.0,
            "enforce_queue": True,
        },
    )

    assert not requires_injection
    assert queue_after
    assert queue_after[0]["dra_ppm"] == pytest.approx(option["dra_ppm_main"], rel=1e-9)
    assert queue_after[0]["length_km"] == pytest.approx(pumped_length, rel=1e-6)
    remaining = queue_after[1:]
    assert remaining
    assert remaining[0]["dra_ppm"] == pytest.approx(4.0, rel=1e-6)
    assert remaining[0]["length_km"] == pytest.approx(40.0 - pumped_length, rel=1e-6)
    if len(remaining) > 1:
        assert remaining[1]["dra_ppm"] == pytest.approx(initial_queue[0]["dra_ppm"], rel=1e-9)


def test_zero_injection_hour_advances_profile() -> None:
    """Zero-DRA decisions should prepend untreated volume and trim the tail."""

    diameter = 0.8
    pumped_speed = 6.53
    hours = 1.0
    flow_m3h = _volume_from_km(pumped_speed, diameter)
    pumped_length = _km_from_volume(flow_m3h * hours, diameter)

    initial_profile = [
        {"length_km": 2.0, "dra_ppm": 5.0},
        {"length_km": 100.0, "dra_ppm": 0.0},
        {"length_km": 56.0, "dra_ppm": 4.0},
    ]
    station = {"idx": 0, "is_pump": True, "d_inner": diameter}
    zero_option = {"nop": 1, "dra_ppm_main": 0.0}
    segment_length = 158.0

    _, queue_after_hour1, inj_ppm_hour1, _ = _update_mainline_dra(
        initial_profile,
        station,
        zero_option,
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        pump_shear_rate=1.0,
    )

    assert inj_ppm_hour1 == 0.0
    assert queue_after_hour1
    assert queue_after_hour1[0]["dra_ppm"] == pytest.approx(0.0, abs=1e-9)
    assert queue_after_hour1[0]["length_km"] == pytest.approx(pumped_length, rel=1e-6)
    assert queue_after_hour1[-1]["dra_ppm"] == pytest.approx(4.0, rel=1e-6)
    assert queue_after_hour1[-1]["length_km"] == pytest.approx(56.0 - pumped_length, rel=1e-6)
    total_length_hour1 = sum(float(entry["length_km"]) for entry in queue_after_hour1)
    assert total_length_hour1 == pytest.approx(segment_length, rel=1e-6)

    _, queue_after_hour2, inj_ppm_hour2, _ = _update_mainline_dra(
        queue_after_hour1,
        station,
        zero_option,
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        pump_shear_rate=1.0,
    )

    assert inj_ppm_hour2 == 0.0
    assert queue_after_hour2
    assert queue_after_hour2[0]["dra_ppm"] == pytest.approx(0.0, abs=1e-9)
    assert queue_after_hour2[0]["length_km"] == pytest.approx(pumped_length * 2.0, rel=1e-6)
    assert queue_after_hour2[-1]["dra_ppm"] == pytest.approx(4.0, rel=1e-6)
    assert queue_after_hour2[-1]["length_km"] == pytest.approx(56.0 - pumped_length * 2.0, rel=1e-6)
    total_length_hour2 = sum(float(entry["length_km"]) for entry in queue_after_hour2)
    assert total_length_hour2 == pytest.approx(segment_length, rel=1e-6)


def test_queue_floor_preserves_downstream_slug() -> None:
    """Applying the floor should retain richer slices beyond the baseline head."""

    initial_queue = ((6.0, 0.0), (152.0, 4.0))

    floored_queue = _ensure_queue_floor(initial_queue, 6.0, 3.0)

    assert floored_queue
    assert len(floored_queue) == 2
    assert floored_queue[0][0] == pytest.approx(6.0, rel=1e-9)
    assert floored_queue[0][1] == pytest.approx(3.0, rel=1e-9)
    assert floored_queue[1][0] == pytest.approx(152.0, rel=1e-9)
    assert floored_queue[1][1] == pytest.approx(4.0, rel=1e-9)
    assert sum(length for length, _ppm in floored_queue) == pytest.approx(158.0, rel=1e-9)

    full_profile = _segment_profile_from_queue(floored_queue, 0.0, 158.0)
    assert len(full_profile) == 2
    assert full_profile[0][0] == pytest.approx(6.0, rel=1e-9)
    assert full_profile[0][1] == pytest.approx(3.0, rel=1e-9)
    assert full_profile[1][0] == pytest.approx(152.0, rel=1e-9)
    assert full_profile[1][1] == pytest.approx(4.0, rel=1e-9)

    downstream_profile = _segment_profile_from_queue(floored_queue, 6.0, 152.0)
    assert len(downstream_profile) == 1
    assert downstream_profile[0][0] == pytest.approx(152.0, rel=1e-9)
    assert downstream_profile[0][1] == pytest.approx(4.0, rel=1e-9)


def test_queue_floor_splices_segment_requirements() -> None:
    """Segment floors should be inserted ahead of the existing queue."""

    initial_queue: tuple[tuple[float, float], ...] = ()
    segment_floors = [
        {"length_km": 4.0, "dra_ppm": 8.0},
        {"length_km": 2.0, "dra_ppm": 12.0},
    ]

    floored_queue = _ensure_queue_floor(initial_queue, 0.0, 0.0, segment_floors)

    assert floored_queue
    assert floored_queue[0][0] == pytest.approx(4.0, rel=1e-9)
    assert floored_queue[0][1] == pytest.approx(8.0, rel=1e-9)
    assert floored_queue[1][0] == pytest.approx(2.0, rel=1e-9)
    assert floored_queue[1][1] == pytest.approx(12.0, rel=1e-9)
    total_length = sum(length for length, _ppm in floored_queue)
    assert total_length == pytest.approx(6.0, rel=1e-9)


def test_bypassed_station_respects_segment_floor() -> None:
    """Stations in bypass should still honour the configured segment floor."""

    diameter = 0.7
    segment_length = 5.0
    hours = 1.0
    flow_m3h = _volume_from_km(segment_length, diameter)

    initial_queue = [
        {"length_km": 2.0, "dra_ppm": 70.0},
        {"length_km": 8.0, "dra_ppm": 30.0},
    ]

    station = {"idx": 1, "is_pump": False, "d_inner": diameter, "kv": 3.0}
    option = {"nop": 0, "dra_ppm_main": 0.0}
    segment_floor = {"length_km": segment_length, "dra_ppm": 50.0}

    dra_segments, queue_after, inj_ppm, requires_injection = _update_mainline_dra(
        initial_queue,
        station,
        option,
        segment_length,
        flow_m3h,
        hours,
        pump_running=False,
        pump_shear_rate=0.0,
        dra_shear_factor=0.0,
        shear_injection=False,
        is_origin=False,
        segment_floor=segment_floor,
    )

    assert inj_ppm == pytest.approx(0.0)
    assert dra_segments
    assert sum(length for length, _ppm in dra_segments) == pytest.approx(segment_length, rel=1e-6)
    assert dra_segments[0][0] == pytest.approx(2.0, rel=1e-6)
    assert dra_segments[0][1] == pytest.approx(70.0, rel=1e-6)
    min_ppm = min(ppm for _length, ppm in dra_segments)
    assert min_ppm == pytest.approx(30.0, rel=1e-6)

    assert queue_after
    total_length = sum(float(entry["length_km"]) for entry in queue_after)
    assert total_length == pytest.approx(10.0, rel=1e-6)
    assert queue_after[0]["length_km"] == pytest.approx(2.0, rel=1e-6)
    assert queue_after[0]["dra_ppm"] == pytest.approx(70.0, rel=1e-6)
    assert queue_after[1]["length_km"] == pytest.approx(8.0, rel=1e-6)
    assert queue_after[1]["dra_ppm"] == pytest.approx(30.0, rel=1e-6)
    assert requires_injection is True


def test_dra_profile_reflects_hourly_push_examples() -> None:
    """Profiles at successive stations should mirror the user's worked examples."""

    diameter = 0.8
    flow_m3h = _volume_from_km(2.0, diameter)
    hours = 1.0

    queue_initial = [{"length_km": 25.0, "dra_ppm": 10.0}]
    station_a = {"idx": 0, "is_pump": True, "d_inner": diameter}
    station_b = {"idx": 1, "is_pump": True, "d_inner": diameter}

    def _profiles_for_case(
        inj_a: float,
        pump_a: bool,
        inj_b: float,
        pump_b: bool,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        queue_after_a = _update_mainline_dra(
            queue_initial,
            station_a,
            {"nop": 1 if pump_a else 0, "dra_ppm_main": inj_a},
            5.0,
            flow_m3h,
            hours,
            pump_running=pump_a,
            pump_shear_rate=1.0,
        )[1]
        queue_a_full = tuple(
            (float(entry["length_km"]), float(entry["dra_ppm"]))
            for entry in queue_after_a
            if float(entry["length_km"]) > 0.0
        )
        merged_a = _merge_queue(queue_a_full)
        profile_a = [
            (float(length), float(ppm))
            for length, ppm in _segment_profile_from_queue(merged_a, 0.0, 5.0)
        ]

        prefix_a = _take_queue_front(merged_a, 5.0)
        inlet_b = _trim_queue_front(merged_a, 5.0)

        queue_after_b = _update_mainline_dra(
            [
                {"length_km": float(length), "dra_ppm": float(ppm)}
                for length, ppm in inlet_b
            ],
            station_b,
            {"nop": 1 if pump_b else 0, "dra_ppm_main": inj_b},
            20.0,
            flow_m3h,
            hours,
            pump_running=pump_b,
            pump_shear_rate=1.0,
        )[1]
        queue_b_full = _merge_queue(
            tuple(prefix_a)
            + tuple(
                (float(entry["length_km"]), float(entry["dra_ppm"]))
                for entry in queue_after_b
                if float(entry["length_km"]) > 0.0
            )
        )
        profile_b = [
            (float(length), float(ppm))
            for length, ppm in _segment_profile_from_queue(queue_b_full, 5.0, 20.0)
        ]
        return profile_a, profile_b

    def _assert_profile(actual, expected):
        assert len(actual) == len(expected)
        for (len_actual, ppm_actual), (len_expected, ppm_expected) in zip(actual, expected):
            assert len_actual == pytest.approx(len_expected, rel=1e-6)
            assert ppm_actual == pytest.approx(ppm_expected, rel=1e-6)

    profile_a, profile_b = _profiles_for_case(12.0, True, 12.0, True)
    _assert_profile(profile_a, [(2.0, 12.0), (3.0, 10.0)])
    _assert_profile(profile_b, [(2.0, 12.0), (18.0, 10.0)])

    _, profile_b_idle = _profiles_for_case(12.0, True, 12.0, False)
    _assert_profile(profile_b_idle, [(2.0, 22.0), (18.0, 10.0)])

    profile_a_zero, profile_b_zero = _profiles_for_case(0.0, True, 0.0, True)
    _assert_profile(profile_a_zero, [(2.0, 0.0), (3.0, 10.0)])
    _assert_profile(profile_b_zero, [(2.0, 0.0), (18.0, 10.0)])

    _, profile_b_no_injection = _profiles_for_case(12.0, True, 0.0, True)
    _assert_profile(profile_b_no_injection, [(2.0, 0.0), (18.0, 10.0)])


def test_dra_profile_preserves_baseline_after_injection() -> None:
    """Injected slugs should overlay a pre-laced baseline across the segment."""

    diameter_inner = 0.7461504
    segment_length = 158.0
    flow_m3h = 2600.0
    hours = 1.0
    pumped_length = _km_from_volume(flow_m3h * hours, diameter_inner)

    queue_initial = [{"length_km": segment_length, "dra_ppm": 4.0}]
    station = {"idx": 0, "is_pump": True, "d_inner": diameter_inner}
    segment_floor = {"length_km": segment_length, "dra_ppm": 4.0}

    dra_segments_hour1, queue_after_hour1, _, _ = _update_mainline_dra(
        queue_initial,
        station,
        {"nop": 1, "dra_ppm_main": 5.0},
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        segment_floor=segment_floor,
        is_origin=True,
    )

    assert dra_segments_hour1
    assert dra_segments_hour1[0][0] == pytest.approx(pumped_length, rel=1e-6)
    assert dra_segments_hour1[0][1] == pytest.approx(5.0, rel=1e-6)
    assert dra_segments_hour1[1][0] == pytest.approx(segment_length - pumped_length, rel=1e-6)
    assert dra_segments_hour1[1][1] == pytest.approx(4.0, rel=1e-6)

    dra_segments_hour2, _, _, _ = _update_mainline_dra(
        queue_after_hour1,
        station,
        {"nop": 1, "dra_ppm_main": 9.0},
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        segment_floor=segment_floor,
        is_origin=True,
    )

    assert dra_segments_hour2
    assert dra_segments_hour2[0][0] == pytest.approx(pumped_length, rel=1e-6)
    assert dra_segments_hour2[0][1] == pytest.approx(9.0, rel=1e-6)
    assert dra_segments_hour2[1][0] == pytest.approx(pumped_length, rel=1e-6)
    assert dra_segments_hour2[1][1] == pytest.approx(5.0, rel=1e-6)
    assert dra_segments_hour2[2][0] == pytest.approx(segment_length - pumped_length * 2.0, rel=1e-6)
    assert dra_segments_hour2[2][1] == pytest.approx(4.0, rel=1e-6)


def test_time_series_solver_uses_cached_baseline(monkeypatch):
    import copy

    import pipeline_optimization_app as app
    import streamlit as st

    baseline_requirement = {
        "dra_ppm": 6.0,
        "dra_perc": 15.0,
        "length_km": 100.0,
        "enforceable": True,
        "segments": [
            {"station_idx": 0, "length_km": 40.0, "dra_ppm": 4.0},
            {"station_idx": 1, "length_km": 60.0, "dra_ppm": 6.0},
        ],
    }

    summary = app._summarise_baseline_requirement(baseline_requirement)
    segments = app._collect_segment_floors(baseline_requirement)

    st.session_state["origin_lacing_baseline"] = copy.deepcopy(baseline_requirement)
    st.session_state["origin_lacing_baseline_summary"] = copy.deepcopy(summary)
    st.session_state["origin_lacing_segment_baseline"] = copy.deepcopy(segments)
    st.session_state["origin_lacing_baseline_warnings"] = []

    stations_base = [
        {
            "name": "Station A",
            "is_pump": True,
            "L": 12.0,
            "D": 0.7,
            "t": 0.007,
            "max_pumps": 1,
        },
        {
            "name": "Station B",
            "is_pump": True,
            "L": 18.0,
            "D": 0.7,
            "t": 0.007,
            "max_pumps": 1,
        },
    ]
    term_data = {"name": "Terminal", "elev": 0.0, "min_residual": 50.0}
    hours = [7]

    vol_df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 8000.0,
                "Viscosity (cSt)": 5.0,
                "Density (kg/m³)": 820.0,
                app.INIT_DRA_COL: 2.0,
            }
        ]
    )
    vol_df = app.ensure_initial_dra_column(vol_df, default=2.0, fill_blanks=True)
    dra_linefill = app.df_to_dra_linefill(vol_df)

    captured: list[dict] = []

    import importlib as importlib_module

    def fake_reload(module):
        return module

    def stub_solve_pipeline(*args, **kwargs):
        captured.append(
            {
                "forced": copy.deepcopy(kwargs.get("forced_origin_detail")),
                "floors": copy.deepcopy(kwargs.get("segment_floors")),
            }
        )
        linefill = args[10] if len(args) > 10 else kwargs.get("linefill_dict", [])
        return {
            "error": False,
            "total_cost": 0.0,
            "linefill": copy.deepcopy(linefill),
            "dra_front_km": 0.0,
            "stations_used": copy.deepcopy(args[0] if args else []),
        }

    monkeypatch.setattr(importlib_module, "reload", fake_reload)
    monkeypatch.setattr(app.pipeline_model, "solve_pipeline", stub_solve_pipeline)

    result = app._execute_time_series_solver(
        stations_base,
        term_data,
        hours,
        flow_rate=500.0,
        plan_df=None,
        current_vol=vol_df,
        dra_linefill=dra_linefill,
        dra_reach_km=0.0,
        RateDRA=5.0,
        Price_HSD=0.0,
        fuel_density=820.0,
        ambient_temp=25.0,
        mop_kgcm2=100.0,
        pump_shear_rate=0.0,
        total_length=sum(stn["L"] for stn in stations_base),
        sub_steps=1,
    )

    assert result["error"] is None
    assert captured, "Expected the solver wrapper to invoke pipeline_model.solve_pipeline"
    forced_detail = captured[0]["forced"]
    segment_floors = captured[0]["floors"]

    assert isinstance(segment_floors, list)
    assert segment_floors == segments
    assert forced_detail is not None
    assert forced_detail.get("enforce_queue") is False
    assert forced_detail.get("dra_ppm", 0.0) >= summary["dra_ppm"]

    for key in [
        "origin_lacing_baseline",
        "origin_lacing_baseline_summary",
        "origin_lacing_segment_baseline",
        "origin_lacing_baseline_warnings",
    ]:
        st.session_state.pop(key, None)


def test_solve_pipeline_rebuilds_segment_floors_when_cache_missing(monkeypatch):
    import importlib as importlib_module

    import pipeline_optimization_app as app
    import streamlit as st

    baseline_requirement = {
        "dra_ppm": 6.0,
        "dra_perc": 15.0,
        "length_km": 100.0,
        "enforceable": True,
        "segments": [
            {"station_idx": 0, "length_km": 40.0, "dra_ppm": 4.0},
            {"station_idx": 1, "length_km": 60.0, "dra_ppm": 6.0},
        ],
    }

    summary = app._summarise_baseline_requirement(baseline_requirement)
    expected_segments = app._collect_segment_floors(baseline_requirement)

    st.session_state["origin_lacing_baseline"] = copy.deepcopy(baseline_requirement)
    st.session_state["origin_lacing_baseline_summary"] = copy.deepcopy(summary)
    st.session_state.pop("origin_lacing_segment_baseline", None)

    captured: dict[str, object] = {}

    def fake_reload(module):
        return module

    def stub_solver(*args, **kwargs):
        captured["floors"] = copy.deepcopy(kwargs.get("segment_floors"))
        return {"error": False, "linefill": [], "total_cost": 0.0}

    monkeypatch.setattr(importlib_module, "reload", fake_reload)
    monkeypatch.setattr(app.pipeline_model, "solve_pipeline", stub_solver)

    stations = [
        {"name": "A", "is_pump": True, "L": 40.0, "D": 0.7, "t": 0.007, "max_pumps": 1},
        {"name": "B", "is_pump": True, "L": 60.0, "D": 0.7, "t": 0.007, "max_pumps": 1},
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 50.0}

    app.solve_pipeline(
        stations,
        terminal,
        1000.0,
        [5.0, 5.0],
        [820.0, 820.0],
        None,
        5.0,
        0.0,
        820.0,
        25.0,
        [],
        pump_shear_rate=0.0,
    )

    assert isinstance(captured.get("floors"), list)
    assert captured["floors"] == expected_segments

    for key in [
        "origin_lacing_baseline",
        "origin_lacing_baseline_summary",
        "origin_lacing_segment_baseline",
    ]:
        st.session_state.pop(key, None)


def test_get_speed_display_map_skips_multi_type_aggregation():
    import pipeline_optimization_app as app

    station = {"name": "Paradip", "pump_types": {"A": {}, "B": {}}}
    res = {
        "speed_paradip": 1450.0,
        "speed_paradip_a": 1450.0,
        # No entry for pump type B
    }

    speed_map = app.get_speed_display_map(res, "paradip", station)

    assert list(speed_map.keys()) == ["A", "B"]
    assert speed_map["A"] == 1450.0
    assert math.isnan(speed_map["B"])
