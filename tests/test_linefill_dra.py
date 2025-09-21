from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dra_utils import get_ppm_for_dr, get_dr_for_ppm
from pipeline_model import _km_from_volume, _update_mainline_dra, solve_pipeline


PPM_TOL = 1e-6


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


def test_get_ppm_for_known_drag_reduction_positive() -> None:
    """Lookup tables should return positive PPM for known drag reductions."""

    ppm = get_ppm_for_dr(3.0, 12.0)
    assert ppm > 0
    assert ppm == pytest.approx(1.0)


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
    assert abs(dra_result["dra_ppm_station_a"]) <= PPM_TOL
    assert abs(dra_result["dra_ppm_station_b"]) <= PPM_TOL

    # The carried slug reduces the SDH at the downstream station and continues
    # travelling through the line (positive treated volume remains).
    assert dra_result["sdh_station_b"] < base_result["sdh_station_b"]
    treated_volume = sum(
        float(batch.get("volume", 0.0))
        for batch in dra_result["linefill"]
        if float(batch.get("dra_ppm", 0.0) or 0.0) > PPM_TOL
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
                ppm = float(batch.get("dra_ppm", 0.0) or 0.0)
            except Exception:
                ppm = 0.0
            if ppm <= PPM_TOL:
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
        assert abs(result["dra_ppm_station_a"]) <= PPM_TOL
        assert abs(result["dra_ppm_station_b"]) <= PPM_TOL
        sdh_history.append(result["sdh_station_b"])
        linefill_state = copy.deepcopy(result["linefill"])

    assert sdh_history[0] < base_sdh_b
    assert all(b >= a for a, b in zip(sdh_history, sdh_history[1:]))
    assert sdh_history[-1] <= base_sdh_b


def test_update_mainline_dra_injects_when_pump_idle() -> None:
    """Pump idle + injection should prepend a slug at the traversed length."""

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

    assert inj_ppm == pytest.approx(opt["dra_ppm_main"], rel=1e-9, abs=PPM_TOL)
    assert dra_segments
    assert dra_segments[0][1] == pytest.approx(inj_ppm, rel=1e-9, abs=PPM_TOL)
    assert dra_segments[0][0] == pytest.approx(segment_length, rel=1e-6)
    assert queue_after
    assert queue_after[0]["dra_ppm"] == pytest.approx(inj_ppm, rel=1e-9, abs=PPM_TOL)
    assert queue_after[0]["length_km"] == pytest.approx(
        pumped_length - segment_length, rel=1e-6
    )
    assert queue_after[1]["dra_ppm"] == pytest.approx(
        initial_queue[0]["dra_ppm"], rel=1e-9, abs=PPM_TOL
    )
    assert queue_after[1]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"] - pumped_length,
        rel=1e-6,
    )


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

    expected_ppm = initial_queue[0]["dra_ppm"] * 0.75
    assert dra_segments
    assert dra_segments[0][1] == pytest.approx(expected_ppm, rel=1e-9, abs=PPM_TOL)
    assert dra_segments[0][0] == pytest.approx(pumped_length, rel=1e-6)
    assert queue_after
    assert queue_after[0]["dra_ppm"] == pytest.approx(
        initial_queue[0]["dra_ppm"], rel=1e-9, abs=PPM_TOL
    )
    assert queue_after[0]["length_km"] == pytest.approx(
        max(initial_queue[0]["length_km"] - pumped_length, 0.0),
        rel=1e-6,
    )


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

    assert inj_ppm == pytest.approx(opt["dra_ppm_main"], rel=1e-9, abs=PPM_TOL)
    expected_ppm = opt["dra_ppm_main"] * 0.8
    assert dra_segments
    assert dra_segments[0][1] == pytest.approx(expected_ppm, rel=1e-9, abs=PPM_TOL)
    assert dra_segments[0][0] == pytest.approx(pumped_length, rel=1e-6)
    if queue_after:
        assert queue_after[0]["dra_ppm"] == pytest.approx(
            expected_ppm, rel=1e-9, abs=PPM_TOL
        )


def test_idle_pump_injection_reflected_in_results() -> None:
    """Idle pump injections should incur cost and appear in reporting."""

    stations = [
        _make_pump_station("Station A"),
        _make_pump_station("Station B", max_dr=12),
    ]
    stations[1]["min_pumps"] = 0
    stations[1]["max_pumps"] = 0

    terminal = {"name": "Terminal", "min_residual": 35, "elev": 0.0}

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
    assert expected_ppm > 0
    flow_station_b = result["pipeline_flow_station_b"]
    assert result["num_pumps_station_b"] == 0
    assert result["dra_ppm_station_b"] == pytest.approx(
        expected_ppm, rel=1e-9, abs=PPM_TOL
    )
    assert result["dra_ppm_station_b"] > PPM_TOL
    assert result["drag_reduction_station_b"] > 0.0
    assert result["dra_cost_station_b"] == pytest.approx(
        expected_ppm * (flow_station_b * 1000.0 * hours / 1e6) * rate_dra,
        rel=1e-6,
    )


def test_fractional_dra_ppm_produces_benefit() -> None:
    """Stations with small DR should retain fractional ppm and improve hydraulics."""

    station = _make_pump_station("Station A", max_dr=4)
    station["min_pumps"] = 1
    station["max_pumps"] = 1
    terminal = {"name": "Terminal", "min_residual": 35, "elev": 0.0}

    common_kwargs = dict(
        FLOW=2500.0,
        KV_list=[3.0, 3.0],
        rho_list=[850.0, 850.0],
        RateDRA=5.0,
        Price_HSD=0.0,
        Fuel_density=0.85,
        Ambient_temp=25.0,
        hours=12.0,
        start_time="00:00",
        enumerate_loops=False,
    )

    fractional_result = solve_pipeline(
        stations=[copy.deepcopy(station)],
        terminal=terminal,
        linefill=[],
        dra_reach_km=0.0,
        narrow_ranges={0: {"dra_main": (4, 4)}},
        dra_step=1,
        _internal_pass=True,
        **common_kwargs,
    )
    assert not fractional_result.get("error"), fractional_result.get("message")

    baseline_station = copy.deepcopy(station)
    baseline_station["max_dr"] = 0
    baseline_result = solve_pipeline(
        stations=[baseline_station],
        terminal=terminal,
        linefill=[],
        dra_reach_km=0.0,
        dra_step=1,
        _internal_pass=True,
        **common_kwargs,
    )
    assert not baseline_result.get("error"), baseline_result.get("message")
    assert abs(baseline_result["dra_ppm_station_a"]) <= PPM_TOL

    ppm = fractional_result["dra_ppm_station_a"]
    expected_ppm = float(get_ppm_for_dr(3.0, 4))
    expected_dr = float(get_dr_for_ppm(3.0, expected_ppm))
    assert ppm > PPM_TOL
    assert ppm == pytest.approx(expected_ppm, rel=1e-9, abs=PPM_TOL)
    assert ppm < 1.0
    assert fractional_result["drag_reduction_station_a"] == pytest.approx(
        expected_dr, rel=1e-9
    )
    assert (
        fractional_result["head_loss_station_a"]
        < baseline_result["head_loss_station_a"]
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
    ppm_stage1 = queue_after_stage1[0]["dra_ppm"]
    expected_stage1 = initial_ppm * (1 - shear_factor)
    assert ppm_stage1 == pytest.approx(expected_stage1, rel=1e-9, abs=PPM_TOL)

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
    ppm_stage2 = queue_after_stage2[0]["dra_ppm"]
    expected_stage2 = expected_stage1 * (1 - shear_factor)
    assert ppm_stage2 == pytest.approx(expected_stage2, rel=1e-9, abs=PPM_TOL)

    kv = 3.0
    base_dr = float(get_dr_for_ppm(kv, initial_ppm))
    stage2_dr = float(get_dr_for_ppm(kv, ppm_stage2))
    assert stage2_dr <= base_dr
    assert stage2_dr == pytest.approx(float(get_dr_for_ppm(kv, expected_stage2)), rel=1e-6)


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
    assert queue_after[0]["dra_ppm"] == pytest.approx(
        initial_queue[0]["dra_ppm"], rel=1e-9, abs=PPM_TOL
    )
    assert queue_after[0]["length_km"] == pytest.approx(
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
    assert abs(queue_after[0]["dra_ppm"]) <= PPM_TOL
    assert queue_after[0]["length_km"] == pytest.approx(
        pumped_length - segment_length,
        rel=1e-6,
    )
    assert queue_after[1]["dra_ppm"] == pytest.approx(
        initial_queue[0]["dra_ppm"], rel=1e-9, abs=PPM_TOL
    )
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
        assert abs(queue_after[0]["dra_ppm"]) <= PPM_TOL
        assert queue_after[0]["length_km"] == pytest.approx(
            pumped_length - segment_length,
            rel=1e-6,
        )


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
    assert abs(zero_front["dra_ppm"]) <= PPM_TOL
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
    assert dra_segments[0][1] == pytest.approx(
        initial_queue[0]["dra_ppm"], rel=1e-9, abs=PPM_TOL
    )
    assert queue_final
    assert queue_final[0]["dra_ppm"] == pytest.approx(
        initial_queue[0]["dra_ppm"], rel=1e-9, abs=PPM_TOL
    )
    assert queue_final[0]["length_km"] == pytest.approx(
        initial_queue[0]["length_km"] - pumped_length - 1.0,
        rel=1e-6,
    )
