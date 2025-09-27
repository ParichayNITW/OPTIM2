from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dra_utils import get_ppm_for_dr, get_dr_for_ppm
import pipeline_model as pm
from pipeline_model import (
    _km_from_volume,
    _prepare_dra_queue_consumption,
    _segment_profile_from_queue,
    _take_queue_front,
    _trim_queue_front,
    _update_mainline_dra,
    _volume_from_km,
    _segment_hydraulics,
    solve_pipeline as _solve_pipeline,
)

from pipeline_optimization_app import (
    combine_volumetric_profiles,
    map_linefill_to_segments,
    map_vol_linefill_to_segments,
)


def solve_pipeline(*args, segment_slices=None, **kwargs):
    if "stations" in kwargs:
        stations = kwargs["stations"]
    elif args:
        stations = args[0]
    else:
        raise TypeError("stations must be provided")
    if segment_slices is None and "segment_slices" not in kwargs:
        segment_slices = [[] for _ in stations]
        kwargs["segment_slices"] = segment_slices
    elif segment_slices is not None and "segment_slices" not in kwargs:
        kwargs["segment_slices"] = segment_slices
    return _solve_pipeline(*args, **kwargs)


def test_map_linefill_to_segments_returns_segment_slices() -> None:
    stations = [
        {"name": "Station A", "L": 5.0, "D": 0.7, "t": 0.007},
        {"name": "Station B", "L": 3.0, "D": 0.7, "t": 0.007},
    ]
    linefill = pd.DataFrame(
        [
            {
                "Start (km)": 0.0,
                "End (km)": 5.0,
                "Viscosity (cSt)": 2.0,
                "Density (kg/m³)": 820.0,
            },
            {
                "Start (km)": 5.0,
                "End (km)": 8.0,
                "Viscosity (cSt)": 3.5,
                "Density (kg/m³)": 835.0,
            },
        ]
    )

    kv_list, rho_list, segment_slices = map_linefill_to_segments(linefill, stations)

    assert kv_list == [2.0, 3.5]
    assert rho_list == [820.0, 835.0]
    assert len(segment_slices) == len(stations)
    for idx, slices in enumerate(segment_slices):
        assert slices, f"Segment {idx} should include at least one slice"
        total_length = sum(entry["length_km"] for entry in slices)
        assert math.isclose(total_length, stations[idx]["L"], rel_tol=0.0, abs_tol=1e-9)
        for entry in slices:
            assert {"length_km", "kv", "rho"} <= set(entry.keys())


def test_combine_volumetric_profiles_merges_future_batches() -> None:
    station = {"name": "Only Station", "L": 10.0, "D": 0.7, "t": 0.007}
    d_inner = station["D"] - 2.0 * station["t"]

    current = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": _volume_from_km(4.0, d_inner),
                "Viscosity (cSt)": 2.0,
                "Density (kg/m³)": 820.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": _volume_from_km(6.0, d_inner),
                "Viscosity (cSt)": 3.0,
                "Density (kg/m³)": 830.0,
            },
        ]
    )
    future = pd.DataFrame(
        [
            {
                "Product": "New Batch",
                "Volume (m³)": _volume_from_km(2.0, d_inner),
                "Viscosity (cSt)": 5.0,
                "Density (kg/m³)": 840.0,
            },
            {
                "Product": "Batch 1",
                "Volume (m³)": _volume_from_km(2.0, d_inner),
                "Viscosity (cSt)": 2.0,
                "Density (kg/m³)": 820.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": _volume_from_km(6.0, d_inner),
                "Viscosity (cSt)": 3.0,
                "Density (kg/m³)": 830.0,
            },
        ]
    )

    kv_list, rho_list, segment_slices = combine_volumetric_profiles([station], current, future)

    assert kv_list == [5.0]
    assert rho_list[0] >= 830.0
    assert len(segment_slices) == 1
    slices = segment_slices[0]
    assert len(slices) >= 2
    assert math.isclose(sum(entry["length_km"] for entry in slices), station["L"], abs_tol=1e-6)
    assert slices[0]["kv"] == pytest.approx(5.0)
    assert slices[0]["rho"] == pytest.approx(840.0)
    assert slices[-1]["rho"] == pytest.approx(830.0)


def test_segment_head_loss_respects_multi_batch_profiles() -> None:
    """Head loss should equal the sum over per-batch slices."""

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
            "L": 10.0,
            "d": 0.7,
            "rough": 4.0e-05,
            "elev": 0.0,
            "min_residual": 20,
            "max_dr": 0,
            "power_type": "Grid",
            "rate": 0.0,
        }
    ]
    terminal = {"name": "Terminal", "min_residual": 5, "elev": 0.0}

    d_inner = stations[0]["d"]
    vol_df = pd.DataFrame(
        [
            {"Product": "Batch 1", "Volume (m³)": _volume_from_km(4.0, d_inner), "Viscosity (cSt)": 2.0, "Density (kg/m³)": 820.0},
            {"Product": "Batch 2", "Volume (m³)": _volume_from_km(6.0, d_inner), "Viscosity (cSt)": 4.0, "Density (kg/m³)": 870.0},
        ]
    )
    kv_list, rho_list, segment_slices = map_vol_linefill_to_segments(vol_df, stations)

    flow_rate = 1200.0
    result = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        FLOW=flow_rate,
        KV_list=kv_list,
        rho_list=rho_list,
        segment_slices=segment_slices,
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        hours=1.0,
        start_time="00:00",
        enumerate_loops=False,
    )

    assert not result.get("error"), result.get("message")
    head_loss_solver = result["head_loss_station_a"]

    manual_loss = 0.0
    for slice_entry in segment_slices[0]:
        hl, *_ = _segment_hydraulics(
            flow_rate,
            slice_entry["length_km"],
            d_inner,
            stations[0]["rough"],
            slice_entry["kv"],
            0.0,
            None,
        )
        manual_loss += hl

    assert head_loss_solver == pytest.approx(manual_loss, rel=1e-5)


def _make_pump_station(name: str, *, max_dr: int = 0) -> dict:
    """Return a minimal pump-station definition for regression tests."""

    return {
        "name": name,
        "is_pump": True,
        "L": 5.0,
        "d": 0.7,
        "rough": 4.0e-05,
        "min_pumps": 1,
        "max_pumps": 1,
        "MinRPM": 1000,
        "DOL": 1000,
        "A": 0.0,
        "B": 0.0,
        "C": 200.0,
        "P": 0.0,
        "Q": 0.0,
        "R": 0.0,
        "S": 0.0,
        "T": 85.0,
        "rate": 0.0,
        "tariffs": [],
        "power_type": "Grid",
        "sfc_mode": "manual",
        "sfc": 0.0,
        "max_dr": max_dr,
        "supply": 0.0,
        "delivery": 0.0,
        "elev": 0.0,
    }


def test_linefill_dra_persists_through_running_pumps() -> None:
    """Initial linefill DRA should reduce SDH even without new injections."""

    stations = [_make_pump_station("Station A"), _make_pump_station("Station B")]
    terminal = {"name": "Terminal", "min_residual": 5, "elev": 0.0}

    common_kwargs = dict(
        FLOW=3000.0,
        KV_list=[3.0, 3.0, 3.0],
        rho_list=[850.0, 850.0, 850.0],
        RateDRA=5.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        hours=24.0,
        start_time="00:00",
        enumerate_loops=False,
    )

    base_result = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        linefill=[],
        dra_reach_km=0.0,
        **common_kwargs,
    )
    assert not base_result.get("error"), base_result.get("message")

    linefill = [{"volume": 150000.0, "dra_ppm": 4}]
    dra_result = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        linefill=linefill,
        dra_reach_km=200.0,
        **common_kwargs,
    )

    assert not dra_result.get("error"), dra_result.get("message")

    # No new DRA should have been injected at either pump station.
    assert dra_result["dra_ppm_station_a"] == 0
    assert dra_result["dra_ppm_station_b"] == 0

    # The carried slug reduces the SDH at the downstream station and continues
    # travelling through the line (positive treated volume remains).
    assert dra_result["sdh_station_b"] < base_result["sdh_station_b"]
    treated_volume = sum(
        float(batch.get("volume", 0.0))
        for batch in dra_result["linefill"]
        if float(batch.get("dra_ppm", 0) or 0.0) > 0
    )
    assert treated_volume > 0.0


def test_zero_injection_benefits_from_inherited_slug() -> None:
    """Inherited DRA continues lowering SDH when no station injects."""

    def _treated_length(linefill_state: list[dict], diameter: float) -> float:
        if diameter <= 0:
            return 0.0
        area = 3.141592653589793 * (diameter ** 2) / 4.0
        total = 0.0
        for batch in linefill_state:
            try:
                ppm = float(batch.get("dra_ppm", 0) or 0.0)
            except Exception:
                ppm = 0.0
            if ppm <= 0:
                continue
            try:
                vol = float(batch.get("volume", 0.0))
            except Exception:
                vol = 0.0
            if vol <= 0:
                continue
            total += vol / area / 1000.0
        return total

    stations = [_make_pump_station("Station A"), _make_pump_station("Station B")]
    terminal = {"name": "Terminal", "min_residual": 30, "elev": 0.0}

    common_kwargs = dict(
        FLOW=2000.0,
        KV_list=[3.0, 3.0, 3.0],
        rho_list=[850.0, 850.0, 850.0],
        RateDRA=5.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        hours=12.0,
        start_time="00:00",
        enumerate_loops=False,
    )

    base = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        linefill=[],
        dra_reach_km=0.0,
        **common_kwargs,
    )
    assert not base.get("error"), base.get("message")
    base_sdh_b = base["sdh_station_b"]

    linefill_state = [{"volume": 180000.0, "dra_ppm": 6}]
    sdh_history: list[float] = []
    for _ in range(3):
        reach = _treated_length(linefill_state, stations[0]["d"])
        result = solve_pipeline(
            stations=copy.deepcopy(stations),
            terminal=terminal,
            linefill=copy.deepcopy(linefill_state),
            dra_reach_km=reach,
            **common_kwargs,
        )
        assert not result.get("error"), result.get("message")
        assert result["dra_ppm_station_a"] == 0
        assert result["dra_ppm_station_b"] == 0
        sdh_history.append(result["sdh_station_b"])
        linefill_state = copy.deepcopy(result["linefill"])

    assert sdh_history[0] < base_sdh_b
    assert all(b >= a for a, b in zip(sdh_history, sdh_history[1:]))
    assert sdh_history[-1] <= base_sdh_b


def test_update_mainline_dra_injects_when_pump_idle() -> None:
    """Idle pump injections should add to the traversed slug rather than replace it."""

    initial_queue = [{"length_km": 12.0, "dra_ppm": 40}]
    stn_data = {"is_pump": True, "d_inner": 0.7}
    opt = {"nop": 0, "dra_ppm_main": 55}
    flow_m3h = 3600.0
    hours = 1.0
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])
    segment_length = pumped_length / 2.0

    dra_segments, queue_after, inj_ppm = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        segment_length,
        flow_m3h,
        hours,
    )

    assert inj_ppm == opt["dra_ppm_main"]
    expected_ppm = initial_queue[0]["dra_ppm"] + inj_ppm
    assert dra_segments
    assert dra_segments[0][1] == expected_ppm
    assert dra_segments[0][0] == pytest.approx(segment_length, rel=1e-6)
    assert queue_after
    assert queue_after[0]["dra_ppm"] == expected_ppm
    assert queue_after[0]["length_km"] == pytest.approx(
        pumped_length,
        rel=1e-6,
    )
    assert queue_after[1]["dra_ppm"] == initial_queue[0]["dra_ppm"]
    assert queue_after[1]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"] - pumped_length,
        rel=1e-6,
    )


def test_idle_pump_injection_mass_balances_incoming_slices() -> None:
    """Case 2: idle pump injections add to each incoming slice."""

    initial_queue = [
        {"length_km": 4.0, "dra_ppm": 20},
        {"length_km": 3.0, "dra_ppm": 5},
    ]
    stn_data = {"is_pump": True, "d_inner": 0.7}
    opt = {"nop": 0, "dra_ppm_main": 15}
    flow_m3h = 3600.0
    hours = 0.5
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])

    dra_segments, queue_after, inj_ppm = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        0.0,
        flow_m3h,
        hours,
    )

    assert not dra_segments
    assert queue_after and len(queue_after) >= 3
    expected_front_ppm = initial_queue[0]["dra_ppm"] + inj_ppm
    assert queue_after[0]["dra_ppm"] == expected_front_ppm
    assert queue_after[0]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"],
        rel=1e-6,
    )

    consumed_second = pumped_length - initial_queue[0]["length_km"]
    assert consumed_second > 0
    expected_second_ppm = initial_queue[1]["dra_ppm"] + inj_ppm
    assert queue_after[1]["dra_ppm"] == expected_second_ppm
    assert queue_after[1]["length_km"] == pytest.approx(consumed_second, rel=1e-6)

    assert queue_after[2]["dra_ppm"] == initial_queue[1]["dra_ppm"]
    assert queue_after[2]["length_km"] == pytest.approx(
        initial_queue[1]["length_km"] - consumed_second,
        rel=1e-6,
    )


def test_segment_longer_than_pumped_length_consumes_downstream_slug() -> None:
    """Cases 1 & 3: downstream coverage persists when the segment extends further."""

    initial_queue = [{"length_km": 10.0, "dra_ppm": 10}]
    stn_base = {"is_pump": True, "d_inner": 0.7}
    initial_total_length = sum(float(entry["length_km"]) for entry in initial_queue)

    flow_m3h = 3600.0
    target_pumped_length = 2.0
    area = math.pi * (stn_base["d_inner"] ** 2) / 4.0
    hours = target_pumped_length * area * 1000.0 / flow_m3h
    segment_length = 5.0

    pumped_length = _km_from_volume(flow_m3h * hours, stn_base["d_inner"])
    assert pumped_length < segment_length

    cases = [
        {
            "label": "case1_idle",
            "pump_running": False,
            "dra_shear_factor": 0.0,
            "opt": {"nop": 0, "dra_ppm_main": 0},
            "expected_ten_length": segment_length,
        },
        {
            "label": "case3_running",
            "pump_running": True,
            "dra_shear_factor": 0.3,
            "opt": {"nop": 1, "dra_ppm_main": 0},
            "expected_ten_length": segment_length - pumped_length,
        },
    ]

    for case in cases:
        dra_segments, queue_after, _ = _update_mainline_dra(
            copy.deepcopy(initial_queue),
            dict(stn_base),
            case["opt"],
            segment_length,
            flow_m3h,
            hours,
            pump_running=case["pump_running"],
            dra_shear_factor=case["dra_shear_factor"],
        )

        assert dra_segments, case["label"]
        total_length = sum(length for length, _ppm in dra_segments)
        assert total_length == pytest.approx(segment_length, rel=1e-6), case["label"]

        ten_length = sum(length for length, ppm in dra_segments if abs(ppm - 10.0) <= 1e-9)
        assert ten_length == pytest.approx(case["expected_ten_length"], rel=1e-6), case["label"]

        assert queue_after, case["label"]
        queue_total = sum(
            float(entry.get("length_km", 0.0) or 0.0)
            for entry in queue_after
            if float(entry.get("length_km", 0.0) or 0.0) > 0
        )
        assert queue_total == pytest.approx(initial_total_length, rel=1e-6), case["label"]


def test_downstream_station_waits_for_advancing_front() -> None:
    """Station B should not see the upstream slug until it reaches the inlet."""

    diameter = 0.5
    pumped_speed_kmh = 2.0
    hours = 1.0
    flow_m3h = _volume_from_km(pumped_speed_kmh, diameter)
    pumped_length = _km_from_volume(flow_m3h * hours, diameter)
    assert pumped_length == pytest.approx(pumped_speed_kmh, rel=1e-6)

    segment_a = 5.0
    segment_b = 20.0
    initial_queue = [
        {"length_km": segment_a, "dra_ppm": 10},
        {"length_km": segment_b, "dra_ppm": 0},
    ]

    opt_idle = {"nop": 0, "dra_ppm_main": 12}

    _, queue_after_a, _ = _update_mainline_dra(
        initial_queue,
        {"idx": 0, "is_pump": True, "d_inner": diameter},
        opt_idle,
        segment_a,
        flow_m3h,
        hours,
    )

    assert queue_after_a
    assert queue_after_a[0]["dra_ppm"] == opt_idle["dra_ppm_main"] + initial_queue[0]["dra_ppm"]
    assert queue_after_a[0]["length_km"] == pytest.approx(pumped_length, rel=1e-6)
    total_after_a = sum(
        float(entry.get("length_km", 0.0) or 0.0)
        for entry in queue_after_a
        if float(entry.get("length_km", 0.0) or 0.0) > 0
    )
    assert total_after_a == pytest.approx(segment_a + segment_b, rel=1e-6)
    queue_full_a = tuple(
        (float(entry["length_km"]), float(entry["dra_ppm"]))
        for entry in queue_after_a
    )
    queue_for_b = _trim_queue_front(queue_full_a, segment_a)
    assert queue_for_b
    assert queue_for_b[0][0] == pytest.approx(segment_b, rel=1e-6)
    assert queue_for_b[0][1] == pytest.approx(initial_queue[1]["dra_ppm"], rel=1e-6)

    queue_for_b_dicts = [
        {"length_km": length, "dra_ppm": ppm}
        for length, ppm in queue_for_b
    ]

    dra_segments_b, queue_after_b, inj_ppm_b = _update_mainline_dra(
        queue_for_b_dicts,
        {"idx": 1, "is_pump": True, "d_inner": diameter},
        opt_idle,
        segment_b,
        flow_m3h,
        hours,
    )

    assert inj_ppm_b == opt_idle["dra_ppm_main"]
    assert dra_segments_b
    assert dra_segments_b[0][0] == pytest.approx(pumped_length, rel=1e-6)
    assert dra_segments_b[0][1] == opt_idle["dra_ppm_main"]
    assert queue_after_b
    first_treated = next(
        (batch for batch in queue_after_b if float(batch.get("dra_ppm", 0) or 0.0) > 0),
        None,
    )
    assert first_treated is not None
    assert first_treated["dra_ppm"] == opt_idle["dra_ppm_main"]
    assert first_treated["length_km"] == pytest.approx(pumped_length, rel=1e-6)
    queue_after_b_inlet = tuple(
        (
            float(entry.get("length_km", 0.0) or 0.0),
            float(entry.get("dra_ppm", 0.0) or 0.0),
        )
        for entry in queue_after_b
        if float(entry.get("length_km", 0.0) or 0.0) > 0
    )
    total_after_b_inlet = sum(length for length, _ppm in queue_after_b_inlet)
    assert total_after_b_inlet == pytest.approx(segment_b, rel=1e-6)
    prefix_for_b = _take_queue_front(queue_full_a, segment_a)
    combined_after_b = prefix_for_b + queue_after_b_inlet
    total_after_b_full = sum(length for length, _ppm in combined_after_b)
    assert total_after_b_full == pytest.approx(segment_a + segment_b, rel=1e-6)


def test_segment_profile_from_queue_origin_segment() -> None:
    """The profile helper should extract the origin segment without upstream offset."""

    queue_full = (
        (2.0, 12.0),
        (3.0, 10.0),
        (20.0, 10.0),
    )

    profile = _segment_profile_from_queue(queue_full, upstream_length=0.0, segment_length=5.0)

    assert len(profile) == 2
    assert profile[0][0] == pytest.approx(2.0, rel=1e-9)
    assert profile[0][1] == pytest.approx(12.0, rel=1e-9)
    assert profile[1][0] == pytest.approx(3.0, rel=1e-9)
    assert profile[1][1] == pytest.approx(10.0, rel=1e-9)


def test_segment_profile_from_queue_downstream_segment() -> None:
    """Downstream segments should ignore the upstream prefix before slicing."""

    queue_full = (
        (2.0, 12.0),
        (3.0, 10.0),
        (2.0, 12.0),
        (18.0, 10.0),
    )

    profile = _segment_profile_from_queue(queue_full, upstream_length=5.0, segment_length=20.0)

    assert len(profile) == 2
    assert profile[0][0] == pytest.approx(2.0, rel=1e-9)
    assert profile[0][1] == pytest.approx(12.0, rel=1e-9)
    assert profile[1][0] == pytest.approx(18.0, rel=1e-9)
    assert profile[1][1] == pytest.approx(10.0, rel=1e-9)


def test_zero_flow_still_delivers_initial_slug_downstream() -> None:
    """Station B should retain the inherited slug even when no flow is pumped."""

    diameter = 0.5
    flow_m3h = 0.0
    hours = 1.0
    segment_a = 5.0
    segment_b = 20.0
    initial_queue = [
        {"length_km": segment_a, "dra_ppm": 10},
        {"length_km": segment_b, "dra_ppm": 0},
    ]
    opt_idle = {"nop": 0, "dra_ppm_main": 0}

    precomputed_b = _prepare_dra_queue_consumption(
        initial_queue,
        segment_b,
        flow_m3h,
        hours,
        diameter,
    )
    pumped_length_b = float(precomputed_b[0])
    assert pumped_length_b == pytest.approx(0.0, abs=1e-9)

    dra_segments_b, queue_after_b, inj_ppm_b = _update_mainline_dra(
        initial_queue,
        {"idx": 1, "is_pump": True, "d_inner": diameter},
        opt_idle,
        segment_b,
        flow_m3h,
        hours,
        precomputed=precomputed_b,
    )

    assert inj_ppm_b == 0
    assert dra_segments_b
    assert dra_segments_b[0][1] == pytest.approx(10.0)
    assert dra_segments_b[0][0] == pytest.approx(segment_a, rel=1e-6)

    first_treated = next(
        (batch for batch in queue_after_b if float(batch.get("dra_ppm", 0) or 0.0) > 0),
        None,
    )
    assert first_treated is not None
    assert first_treated["dra_ppm"] == pytest.approx(10.0)
    assert first_treated["length_km"] == pytest.approx(segment_a, rel=1e-6)


def test_idle_downstream_pump_preserves_upstream_slug() -> None:
    """End-state linefill should retain the carried 10 ppm slug when pump B is idle."""

    diameter = 0.5
    flow_m3h = _volume_from_km(2.0, diameter)
    hours = 1.0
    stations = [_make_pump_station("Station A"), _make_pump_station("Station B")]
    stations[0]["d"] = stations[0]["d_inner"] = diameter
    stations[0]["L"] = 5.0
    stations[1]["d"] = stations[1]["d_inner"] = diameter
    stations[1]["L"] = 20.0
    stations[1]["min_pumps"] = 0
    stations[1]["max_pumps"] = 0
    terminal = {"name": "Terminal", "min_residual": 0.0, "elev": 0.0}
    linefill = [
        {"volume": _volume_from_km(5.0, diameter), "dra_ppm": 10},
        {"volume": _volume_from_km(20.0, diameter), "dra_ppm": 0},
    ]
    common_kwargs = dict(
        FLOW=flow_m3h,
        KV_list=[3.0, 3.0, 3.0],
        rho_list=[850.0, 850.0, 850.0],
        RateDRA=5.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        hours=hours,
        start_time="00:00",
        enumerate_loops=False,
    )

    result = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        linefill=copy.deepcopy(linefill),
        dra_reach_km=0.0,
        **common_kwargs,
    )

    assert not result.get("error"), result.get("message")

    final_linefill = result["linefill"]
    total_length = sum(
        _km_from_volume(float(batch.get("volume", 0.0) or 0.0), diameter)
        for batch in final_linefill
        if float(batch.get("volume", 0.0) or 0.0) > 0
    )
    assert total_length == pytest.approx(25.0, rel=1e-6)

    treated_batches = [
        batch for batch in final_linefill if float(batch.get("dra_ppm", 0) or 0.0) > 0
    ]
    assert treated_batches, "Expected the upstream slug to persist"
    treated_length = sum(
        _km_from_volume(float(batch.get("volume", 0.0) or 0.0), diameter)
        for batch in treated_batches
    )
    assert treated_length == pytest.approx(5.0, rel=1e-6)
    assert all(float(batch.get("dra_ppm", 0) or 0.0) == pytest.approx(10.0) for batch in treated_batches)


def test_running_pump_shears_trimmed_slug() -> None:
    """Shear factor should attenuate the slug passing through an active pump."""

    initial_queue = [{"length_km": 8.0, "dra_ppm": 40}]
    stn_data = {"is_pump": True, "d_inner": 0.7}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 3600.0
    hours = 0.5
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])

    dra_segments, queue_after, _ = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        pumped_length,
        flow_m3h,
        hours,
        pump_running=True,
        dra_shear_factor=0.25,
    )

    kv = float(stn_data.get("kv", 3.0) or 3.0)
    upstream_ppm = float(initial_queue[0]["dra_ppm"])
    upstream_dr = float(get_dr_for_ppm(kv, upstream_ppm))
    expected_dr = upstream_dr * (1.0 - 0.25)
    expected_ppm_float = float(get_ppm_for_dr(kv, expected_dr)) if expected_dr > 0 else 0.0
    expected_ppm = expected_ppm_float if expected_ppm_float > 0 else 0.0
    assert dra_segments
    assert dra_segments[0][1] == pytest.approx(expected_ppm)
    assert dra_segments[0][0] == pytest.approx(pumped_length, rel=1e-6)
    assert queue_after
    assert queue_after[0]["dra_ppm"] == pytest.approx(expected_ppm)
    assert queue_after[0]["length_km"] == pytest.approx(
        pumped_length,
        rel=1e-6,
    )


def test_global_shear_scales_drag_reduction_in_dr_domain() -> None:
    """Global pump shear should attenuate drag reduction in the %DR domain."""

    shear_rate = 0.5
    initial_queue = [{"length_km": 6.0, "dra_ppm": 60}]
    stn_data = {"is_pump": True, "d_inner": 0.7, "kv": 3.0}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 3600.0
    hours = 0.25
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])

    dra_segments, queue_after, _ = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        pumped_length,
        flow_m3h,
        hours,
        pump_running=True,
        pump_shear_rate=shear_rate,
    )

    assert dra_segments
    assert dra_segments[0][0] == pytest.approx(pumped_length, rel=1e-6)
    assert queue_after
    assert queue_after[0]["length_km"] == pytest.approx(pumped_length, rel=1e-6)

    kv = float(stn_data["kv"])
    upstream_ppm = float(initial_queue[0]["dra_ppm"])
    downstream_ppm = float(queue_after[0]["dra_ppm"])
    assert downstream_ppm > 0

    upstream_dr = float(get_dr_for_ppm(kv, upstream_ppm))
    downstream_dr = float(get_dr_for_ppm(kv, downstream_ppm))
    expected_dr_cont = upstream_dr * (1.0 - shear_rate)
    expected_ppm_float = float(get_ppm_for_dr(kv, expected_dr_cont)) if expected_dr_cont > 0 else 0.0
    expected_ppm = expected_ppm_float if expected_ppm_float > 0 else 0.0

    assert downstream_ppm == pytest.approx(expected_ppm)

    expected_dr = float(get_dr_for_ppm(kv, expected_ppm)) if expected_ppm > 0 else 0.0
    assert downstream_dr == pytest.approx(expected_dr, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    "label,opt,pump_running,shear,expected_segments,expected_queue,expected_trimmed",
    [
        (
            "idle_no_injection",
            {"nop": 0, "dra_ppm_main": 0},
            False,
            0.0,
            [(5.0, 10.0)],
            [(25.0, 10.0)],
            [(22.0, 10.0)],
        ),
        (
            "idle_injection",
            {"nop": 0, "dra_ppm_main": 12},
            False,
            0.0,
            [(2.0, 22.0), (3.0, 10.0)],
            [(2.0, 22.0), (23.0, 10.0)],
            [(22.0, 10.0)],
        ),
        (
            "running_no_injection",
            {"nop": 1, "dra_ppm_main": 0},
            True,
            1.0,
            [(3.0, 10.0)],
            [(2.0, 0.0), (23.0, 10.0)],
            [(22.0, 10.0)],
        ),
        (
            "running_injection",
            {"nop": 1, "dra_ppm_main": 12},
            True,
            1.0,
            [(2.0, 12.0), (3.0, 10.0)],
            [(2.0, 12.0), (23.0, 10.0)],
            [(22.0, 10.0)],
        ),
    ],
)
def test_two_station_case_profiles(
    label: str,
    opt: dict,
    pump_running: bool,
    shear: float,
    expected_segments: list[tuple[float, int]],
    expected_queue: list[tuple[float, int]],
    expected_trimmed: list[tuple[float, float]],
) -> None:
    """Validate Case 1–4 queue evolution for the 5 km + 20 km scenario."""

    diameter = 0.5
    flow_m3h = _volume_from_km(2.0, diameter)
    hours = 1.0
    segment_a = 5.0
    segment_b = 20.0
    initial_queue = [
        {"length_km": segment_a + segment_b, "dra_ppm": 10},
    ]

    precomputed = _prepare_dra_queue_consumption(
        initial_queue,
        segment_a,
        flow_m3h,
        hours,
        diameter,
    )
    pumped_length = float(precomputed[0])

    dra_segments, queue_after, _ = _update_mainline_dra(
        initial_queue,
        {
            "idx": 0,
            "is_pump": True,
            "d_inner": diameter,
            "dra_shear_factor": shear,
        },
        opt,
        segment_a,
        flow_m3h,
        hours,
        pump_running=pump_running,
        dra_shear_factor=shear,
        precomputed=precomputed,
    )

    assert len(dra_segments) == len(expected_segments), label
    for (length_actual, ppm_actual), (length_expected, ppm_expected) in zip(dra_segments, expected_segments):
        assert ppm_actual == ppm_expected, label
        assert length_actual == pytest.approx(length_expected, rel=1e-6), label

    queue_full = [
        (float(entry.get("length_km", 0.0) or 0.0), float(entry.get("dra_ppm", 0) or 0.0))
        for entry in queue_after
        if float(entry.get("length_km", 0.0) or 0.0) > 0
    ]
    assert len(queue_full) == len(expected_queue), label
    for (length_actual, ppm_actual), (length_expected, ppm_expected) in zip(queue_full, expected_queue):
        assert ppm_actual == pytest.approx(ppm_expected, rel=1e-9), label
        assert length_actual == pytest.approx(length_expected, rel=1e-6), label

    queue_full_floats = tuple((length, float(ppm)) for length, ppm in queue_full)
    trim_offset = segment_a - pumped_length
    queue_trimmed = _trim_queue_front(queue_full_floats, trim_offset)
    trimmed_list = [
        (float(length), float(ppm))
        for length, ppm in queue_trimmed
        if float(length) > 0
    ]
    assert len(trimmed_list) == len(expected_trimmed), label
    for (length_actual, ppm_actual), (length_expected, ppm_expected) in zip(trimmed_list, expected_trimmed):
        assert length_actual == pytest.approx(length_expected, rel=1e-6), label
        assert ppm_actual == pytest.approx(ppm_expected, rel=1e-6), label

def test_injected_slug_respects_shear_when_upstream() -> None:
    """Injection upstream of pumps should emerge at reduced ppm."""

    stn_data = {"is_pump": True, "d_inner": 0.7, "dra_injector_position": "upstream"}
    opt = {"nop": 1, "dra_ppm_main": 60}
    flow_m3h = 2400.0
    hours = 0.25
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])

    dra_segments, queue_after, inj_ppm = _update_mainline_dra(
        [],
        stn_data,
        opt,
        pumped_length,
        flow_m3h,
        hours,
        pump_running=True,
        dra_shear_factor=0.2,
        shear_injection=True,
    )

    assert inj_ppm == opt["dra_ppm_main"]
    kv = float(stn_data.get("kv", 3.0) or 3.0)
    inj_dr = float(get_dr_for_ppm(kv, float(opt["dra_ppm_main"])))
    sheared_dr = inj_dr * (1.0 - 0.2)
    expected_ppm_float = float(get_ppm_for_dr(kv, sheared_dr)) if sheared_dr > 0 else 0.0
    expected_ppm = expected_ppm_float if expected_ppm_float > 0 else 0.0
    assert dra_segments
    assert dra_segments[0][1] == pytest.approx(expected_ppm)
    assert dra_segments[0][0] == pytest.approx(pumped_length, rel=1e-6)
    if queue_after:
        assert queue_after[0]["dra_ppm"] == pytest.approx(expected_ppm)


def test_idle_pump_injection_reflected_in_results() -> None:
    """Idle pump injections should incur cost and appear in reporting."""

    stations = [
        _make_pump_station("Station A"),
        _make_pump_station("Station B", max_dr=12),
    ]
    stations[1]["min_pumps"] = 0
    stations[1]["max_pumps"] = 0

    terminal = {"name": "Terminal", "min_residual": 5, "elev": 0.0}

    hours = 12.0
    rate_dra = 5.0
    common_kwargs = dict(
        FLOW=3000.0,
        KV_list=[3.0, 3.0, 3.0],
        rho_list=[850.0, 850.0, 850.0],
        RateDRA=rate_dra,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        hours=hours,
        start_time="00:00",
        enumerate_loops=False,
    )

    narrow_ranges = {1: {"dra_main": (12, 12)}}

    result = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        linefill=[],
        dra_reach_km=0.0,
        narrow_ranges=narrow_ranges,
        _internal_pass=True,
        **common_kwargs,
    )

    assert not result.get("error"), result.get("message")

    expected_ppm = float(get_ppm_for_dr(3.0, 12))
    flow_station_b = result["pipeline_flow_station_b"]
    assert result["num_pumps_station_b"] == 0
    assert result["dra_ppm_station_b"] == pytest.approx(expected_ppm)
    assert result["drag_reduction_station_b"] > 0.0
    assert result["dra_cost_station_b"] == pytest.approx(
        expected_ppm * (flow_station_b * 1000.0 * hours / 1e6) * rate_dra,
        rel=1e-6,
    )


def test_shear_factor_reduces_downstream_effective_ppm() -> None:
    """Repeated pump stages should attenuate the slug according to shear."""

    shear_factor = 0.3
    initial_ppm = 12
    queue = [{"length_km": 20.0, "dra_ppm": initial_ppm}]
    stn_data = {"is_pump": True, "d_inner": 0.7}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 3600.0
    hours = 0.5

    # Advance through the first pump stage without consuming segment length so the sheared
    # slug remains available for the next stage.
    _, queue_after_stage1, _ = _update_mainline_dra(
        queue,
        stn_data,
        opt,
        0.0,
        flow_m3h,
        hours,
        pump_running=True,
        dra_shear_factor=shear_factor,
    )
    assert queue_after_stage1, "Expected a sheared slug after first stage"
    ppm_stage1 = float(queue_after_stage1[0]["dra_ppm"])
    kv = float(stn_data.get("kv", 3.0) or 3.0)

    def _expected_ppm(ppm_in: float) -> float:
        dr_val = float(get_dr_for_ppm(kv, float(ppm_in)))
        sheared_val = dr_val * (1.0 - shear_factor)
        if sheared_val <= 0:
            return 0.0
        ppm_float = float(get_ppm_for_dr(kv, sheared_val))
        return ppm_float if ppm_float > 0 else 0.0

    expected_stage1 = _expected_ppm(initial_ppm)
    assert ppm_stage1 == pytest.approx(expected_stage1)

    # Repeat for the second pump stage.
    _, queue_after_stage2, _ = _update_mainline_dra(
        queue_after_stage1,
        stn_data,
        opt,
        0.0,
        flow_m3h,
        hours,
        pump_running=True,
        dra_shear_factor=shear_factor,
    )
    assert queue_after_stage2, "Expected slug to persist after second stage"
    ppm_stage2 = float(queue_after_stage2[0]["dra_ppm"])
    expected_stage2 = _expected_ppm(expected_stage1)
    assert ppm_stage2 == pytest.approx(expected_stage2)

    base_dr = float(get_dr_for_ppm(kv, initial_ppm))
    stage2_dr = float(get_dr_for_ppm(kv, ppm_stage2))
    assert stage2_dr <= base_dr
    expected_stage2_dr = float(get_dr_for_ppm(kv, expected_stage2)) if expected_stage2 > 0 else 0.0
    assert stage2_dr == pytest.approx(expected_stage2_dr, rel=1e-6)


def test_full_shear_zeroes_trimmed_slug() -> None:
    """A 100% shear factor should erase the trimmed slug for the segment."""

    initial_queue = [{"length_km": 6.0, "dra_ppm": 40}]
    stn_data = {"is_pump": True, "d_inner": 0.7, "idx": 1}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 3600.0
    hours = 0.5
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])

    dra_segments, queue_after, _ = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        pumped_length,
        flow_m3h,
        hours,
        pump_running=True,
        dra_shear_factor=1.0,
    )

    assert not dra_segments
    assert queue_after
    assert queue_after[0]["dra_ppm"] == 0
    assert queue_after[0]["length_km"] == pytest.approx(
        pumped_length,
        rel=1e-6,
    )
    assert queue_after[1]["dra_ppm"] == initial_queue[0]["dra_ppm"]
    assert queue_after[1]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"] - pumped_length,
        rel=1e-6,
    )


def test_full_shear_retains_zero_front_for_partial_segment() -> None:
    """When the segment is shorter than the trimmed slug the 0 ppm zone persists."""

    initial_queue = [{"length_km": 10.0, "dra_ppm": 55}]
    stn_data = {"is_pump": True, "d_inner": 0.7, "idx": 2}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 3600.0
    hours = 0.25
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])
    segment_length = pumped_length / 2.0

    dra_segments, queue_after, _ = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        dra_shear_factor=1.0,
    )

    assert not dra_segments
    assert queue_after
    assert queue_after[0]["dra_ppm"] == 0
    assert queue_after[0]["length_km"] == pytest.approx(
        pumped_length,
        rel=1e-6,
    )
    assert queue_after[1]["dra_ppm"] == initial_queue[0]["dra_ppm"]
    assert queue_after[1]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"] - pumped_length,
        rel=1e-6,
    )


def test_origin_station_without_injection_zeroes_slug() -> None:
    """Origin pumps should drop inherited slugs to 0 ppm when not injecting."""

    initial_queue = [{"length_km": 5.0, "dra_ppm": 30}]
    stn_data = {"is_pump": True, "d_inner": 0.7, "idx": 0}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 2400.0
    hours = 0.25
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])
    segment_length = pumped_length / 2.0

    for shear_factor in (1.0, 0.25):
        dra_segments, queue_after, _ = _update_mainline_dra(
            initial_queue,
            stn_data,
            opt,
            segment_length,
            flow_m3h,
            hours,
            pump_running=True,
            dra_shear_factor=shear_factor,
        )

        assert not dra_segments
        assert queue_after
        assert queue_after[0]["dra_ppm"] == 0
        assert queue_after[0]["length_km"] == pytest.approx(
            pumped_length,
            rel=1e-6,
        )


def test_origin_zero_front_advances_with_repeated_updates() -> None:
    """Untreated origin fronts should accumulate across successive hours."""

    initial_queue = [(158.0, 4.0)]
    stn_data = {"is_pump": True, "d_inner": 0.82, "idx": 0}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 3600.0
    hours = 1.0
    segment_length = 0.0
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])

    precomputed_stage1 = _prepare_dra_queue_consumption(
        initial_queue,
        segment_length,
        flow_m3h,
        hours,
        stn_data["d_inner"],
    )
    _, queue_after_stage1, _ = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        is_origin=True,
        precomputed=precomputed_stage1,
    )

    assert queue_after_stage1
    zero_front_1 = queue_after_stage1[0]
    assert zero_front_1["dra_ppm"] == 0
    assert zero_front_1["length_km"] == pytest.approx(pumped_length, rel=1e-6)

    precomputed_stage2 = _prepare_dra_queue_consumption(
        queue_after_stage1,
        segment_length,
        flow_m3h,
        hours,
        stn_data["d_inner"],
    )
    _, queue_after_stage2, _ = _update_mainline_dra(
        queue_after_stage1,
        stn_data,
        opt,
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        is_origin=True,
        precomputed=precomputed_stage2,
    )

    assert queue_after_stage2
    zero_front_2 = queue_after_stage2[0]
    assert zero_front_2["dra_ppm"] == 0
    assert zero_front_2["length_km"] == pytest.approx(pumped_length * 2.0, rel=1e-6)


def test_full_shear_zero_front_propagates_downstream() -> None:
    """Downstream segments should consume the 0 ppm zone before any treated slug."""

    initial_queue = [{"length_km": 9.0, "dra_ppm": 60}]
    stn_data = {"is_pump": True, "d_inner": 0.7, "idx": 1}
    opt = {"nop": 1, "dra_ppm_main": 0}
    flow_m3h = 3600.0
    hours = 0.5
    pumped_length = _km_from_volume(flow_m3h * hours, stn_data["d_inner"])
    segment_length = pumped_length / 3.0

    _, queue_after, _ = _update_mainline_dra(
        initial_queue,
        stn_data,
        opt,
        segment_length,
        flow_m3h,
        hours,
        pump_running=True,
        dra_shear_factor=1.0,
    )

    assert queue_after
    zero_front = queue_after[0]
    assert zero_front["dra_ppm"] == 0
    zero_length = zero_front["length_km"]

    downstream_segment = zero_length + 1.0
    dra_segments, queue_final, _ = _update_mainline_dra(
        queue_after,
        {"is_pump": False, "d_inner": stn_data["d_inner"], "idx": 2},
        {"nop": 0, "dra_ppm_main": 0},
        downstream_segment,
        0.0,
        1.0,
        pump_running=False,
        dra_shear_factor=0.0,
    )

    assert dra_segments
    assert dra_segments[0][0] == pytest.approx(1.0, rel=1e-6)
    assert dra_segments[0][1] == initial_queue[0]["dra_ppm"]
    assert queue_final
    assert queue_final[0]["dra_ppm"] == 0
    assert queue_final[0]["length_km"] == pytest.approx(
        zero_length,
        rel=1e-6,
    )
    assert queue_final[1]["dra_ppm"] == initial_queue[0]["dra_ppm"]


def test_dra_queue_signature_preserves_optimal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """States with identical residuals but distinct DRA queues must persist."""

    def fake_segment(
        flow: float,
        length: float,
        d_inner: float,
        rough: float,
        kv: float,
        dra: float,
        dra_len: float,
    ) -> tuple[float, float, float, float]:
        base = 80.0 if float(length) < 6.0 else 60.0
        reduction = 0.0
        if dra > 0:
            reduction = 40.0 if float(length) < 6.0 else 60.0
        head_loss = base - reduction
        return head_loss, 1.0, 1.0, 0.01

    def fake_pump_head(stn: dict, flow_m3h: float, rpm_map, nop: int) -> list[dict]:
        if nop <= 0:
            return []
        return [
            {
                "tdh": 100.0 * nop,
                "eff": 80.0,
                "count": nop,
                "power_type": stn.get("power_type", "Grid"),
                "ptype": "mock",
                "rpm": 1000,
                "data": {
                    "sfc_mode": stn.get("sfc_mode", "manual"),
                    "sfc": stn.get("sfc", 0.0),
                    "DOL": stn.get("DOL", 1000),
                    "power_type": stn.get("power_type", "Grid"),
                },
            }
        ]

    def fake_update(
        queue,
        stn_data: dict,
        opt: dict,
        segment_length: float,
        flow_m3h: float,
        hours: float,
        *,
        pump_running: bool = False,
        pump_shear_rate: float = 0.0,
        dra_shear_factor: float = 0.0,
        shear_injection: bool = False,
        is_origin: bool = False,
        precomputed=None,
    ) -> tuple[list[tuple[float, float]], list[dict], float]:
        seg_len = float(segment_length or 0.0)
        ppm = float(opt.get("dra_ppm_main", 0) or 0.0)
        if ppm <= 0 and float(opt.get("dra_main", 0) or 0) > 0:
            ppm = 5.0
        queue_entries: list[tuple[float, float]] = []
        if queue:
            for raw in queue:
                if isinstance(raw, dict):
                    queue_entries.append(
                        (
                            float(raw.get("length_km", 0.0) or 0.0),
                            float(raw.get("dra_ppm", 0.0) or 0.0),
                        )
                    )
                elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                    queue_entries.append((float(raw[0] or 0.0), float(raw[1] or 0.0)))
        if ppm > 0:
            dra_segments = [(seg_len, ppm)]
            queue_after = [{"length_km": 100.0, "dra_ppm": ppm}]
        elif queue_entries:
            head_ppm = queue_entries[0][1]
            dra_segments = [(seg_len, head_ppm)]
            queue_after = [
                {"length_km": max(queue_entries[0][0], 100.0), "dra_ppm": head_ppm}
            ]
        else:
            dra_segments = [(seg_len, 0.0)]
            queue_after = []
        return dra_segments, queue_after, ppm

    def fake_get_ppm(kv: float, dr: float) -> float:
        return float(dr) * 10.0

    def fake_get_dr(kv: float, ppm: float) -> float:
        return float(ppm) / 10.0

    def fake_composite(
        flow: float,
        length: float,
        d_inner: float,
        rough: float,
        kv_default: float,
        dra_perc: float,
        dra_length: float | None = None,
        slices=None,
        limit: float | None = None,
    ) -> tuple[float, float, float, float]:
        target_length = length if limit is None else min(float(limit), float(length))
        total_head = 0.0
        stats: tuple[float, float, float] | None = None
        remaining = target_length
        remaining_dra = dra_length
        if slices:
            for entry in slices:
                seg_len = min(float(entry.get("length_km", 0.0) or 0.0), remaining)
                if seg_len <= 0:
                    continue
                use_kv = float(entry.get("kv", kv_default) or kv_default)
                dra_seg = 0.0
                if remaining_dra is not None:
                    dra_seg = float(min(remaining_dra, seg_len))
                    remaining_dra -= dra_seg
                head, v, Re, f = fake_segment(flow, seg_len, d_inner, rough, use_kv, dra_perc, dra_seg)
                total_head += head
                if stats is None:
                    stats = (v, Re, f)
                remaining -= seg_len
                if remaining <= 0:
                    break
        if remaining > 1e-9:
            extra_dra = float(remaining_dra) if remaining_dra is not None else 0.0
            head, v, Re, f = fake_segment(flow, remaining, d_inner, rough, kv_default, dra_perc, extra_dra)
            total_head += head
            if stats is None:
                stats = (v, Re, f)
        if stats is None:
            stats = (1.0, 1.0, 0.01)
        return total_head, stats[0], stats[1], stats[2]

    monkeypatch.setattr(pm, "_segment_hydraulics", fake_segment)
    monkeypatch.setattr(pm, "_segment_hydraulics_composite", fake_composite)
    monkeypatch.setattr(pm, "_pump_head", fake_pump_head)
    monkeypatch.setattr(pm, "_update_mainline_dra", fake_update)
    monkeypatch.setattr(pm, "get_ppm_for_dr", fake_get_ppm)
    monkeypatch.setattr(pm, "get_dr_for_ppm", fake_get_dr)

    stations = [
        _make_pump_station("Station A", max_dr=10),
        _make_pump_station("Station B", max_dr=0),
    ]
    stations[0]["rate"] = 1.0
    stations[1]["rate"] = 1.0
    stations[1]["min_pumps"] = 0
    stations[1]["max_pumps"] = 1
    stations[0]["L"] = 5.0
    stations[1]["L"] = 7.0

    terminal = {"name": "Terminal", "min_residual": 30, "elev": 0.0}
    common_kwargs = dict(
        FLOW=3000.0,
        KV_list=[3.0, 3.0, 3.0],
        rho_list=[850.0, 850.0, 850.0],
        segment_slices=[
            [{"length_km": stations[0]["L"], "kv": 3.0, "rho": 850.0}],
            [{"length_km": stations[1]["L"], "kv": 3.0, "rho": 850.0}],
        ],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        hours=24.0,
        start_time="00:00",
        enumerate_loops=False,
        _internal_pass=True,
        state_top_k=1,
        state_cost_margin=0.0,
    )

    optimal = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        linefill=[],
        dra_reach_km=0.0,
        **common_kwargs,
    )

    assert optimal["dra_ppm_station_a"] > 0
    assert optimal["num_pumps_station_b"] == 0

    forced_zero = solve_pipeline(
        stations=copy.deepcopy(stations),
        terminal=terminal,
        linefill=[],
        dra_reach_km=0.0,
        narrow_ranges={0: {"dra_main": (0, 0)}},
        **common_kwargs,
    )

    assert forced_zero["dra_ppm_station_a"] == 0
    assert forced_zero["num_pumps_station_b"] == 1
    assert forced_zero["total_cost"] > optimal["total_cost"]
    assert forced_zero["residual_head_station_a"] == optimal["residual_head_station_a"]
