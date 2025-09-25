from pathlib import Path
import copy
import math
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_model import (
    solve_pipeline as _solve_pipeline,
    _prepare_dra_queue_consumption,
    _update_mainline_dra,
)


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


def _dra_length_km(linefill: list[dict], diameter: float) -> float:
    """Return the total treated length represented by ``linefill``."""

    if diameter <= 0:
        return 0.0
    area = math.pi * (diameter ** 2) / 4.0
    if area <= 0:
        return 0.0
    total = 0.0
    for batch in linefill:
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


def _total_queue_length(queue: list[dict]) -> float:
    """Return the total length represented by ``queue`` entries."""

    total = 0.0
    for entry in queue:
        try:
            length = float(entry.get("length_km", 0.0) or 0.0)
        except Exception:
            length = 0.0
        if length <= 0:
            continue
        total += length
    return total


def _zero_front_within_segment(queue: list[dict], segment_length: float) -> float:
    """Return the distance from the segment inlet to the first 0-ppm slice."""

    remaining = float(segment_length)
    distance = 0.0
    for entry in queue:
        if remaining <= 0:
            break
        try:
            length = float(entry.get("length_km", 0.0) or 0.0)
        except Exception:
            length = 0.0
        if length <= 0:
            continue
        take = length if length <= remaining else remaining
        try:
            ppm_val = float(entry.get("dra_ppm", 0) or 0.0)
        except Exception:
            ppm_val = 0.0
        if ppm_val == 0:
            return distance
        distance += take
        remaining -= take
    return distance


def test_sdh_varies_smoothly_with_downstream_slug():
    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "min_pumps": 1,
            "max_pumps": 1,
            "MinRPM": 1000,
            "DOL": 3000,
            "pump_type": "type1",
            "A": 0.0,
            "B": 0.0,
            "C": 180.0,
            "P": 0.0,
            "Q": 0.0,
            "R": 0.0,
            "S": 0.0,
            "T": 80.0,
            "L": 50.0,
            "d": 0.7,
            "rough": 0.00004,
            "elev": 0.0,
            "min_residual": 30,
            "max_dr": 40,
            "fixed_dra_perc": 30,
            "power_type": "Grid",
            "rate": 0.0,
        }
    ]
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 30}

    slug_reaches = [45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
    sdh_values: list[float] = []

    for reach in slug_reaches:
        result = solve_pipeline(
            stations,
            terminal,
            FLOW=1500.0,
            KV_list=[1.0],
            rho_list=[850.0],
            RateDRA=5.0,
            Price_HSD=0.0,
            Fuel_density=820.0,
            Ambient_temp=25.0,
            linefill=[{"volume": 1000.0, "dra_ppm": 0}],
            dra_reach_km=reach,
            hours=1.0,
            start_time="00:00",
            enumerate_loops=False,
            rpm_step=50,
            dra_step=2,
        )
        sdh_values.append(result["sdh_origin_pump"])

    assert sdh_values == sorted(sdh_values)
    diffs = [abs(b - a) for a, b in zip(sdh_values, sdh_values[1:])]
    assert diffs, "Expected at least one SDH difference"
    assert max(diffs) <= 6.0


def test_partial_slug_advances_with_positive_injection() -> None:
    """A heavy slug should taper smoothly when new DRA is injected."""

    station = {
        "name": "Origin Pump",
        "is_pump": True,
        "min_pumps": 1,
        "max_pumps": 1,
        "MinRPM": 1000,
        "DOL": 2800,
        "pump_type": "type1",
        "A": 0.0,
        "B": 0.0,
        "C": 190.0,
        "P": 0.0,
        "Q": 0.0,
        "R": 0.0,
        "S": 0.0,
        "T": 82.0,
        "L": 40.0,
        "d": 0.7,
        "rough": 0.00004,
        "elev": 0.0,
        "min_residual": 25,
        "max_dr": 35,
        "fixed_dra_perc": 18,
        "power_type": "Grid",
        "rate": 0.0,
    }
    terminal = {"name": "Terminal", "elev": 0.0, "min_residual": 25}

    linefill_state = [
        {"volume": 9000.0, "dra_ppm": 12},
        {"volume": 6000.0, "dra_ppm": 0},
    ]

    sdh_progression: list[float] = []
    hours = 1.0
    for _ in range(4):
        reach = _dra_length_km(linefill_state, station["d"])
        result = solve_pipeline(
            [copy.deepcopy(station)],
            terminal,
            FLOW=1600.0,
            KV_list=[1.8],
            rho_list=[845.0],
            RateDRA=5.0,
            Price_HSD=0.0,
            Fuel_density=0.85,
            Ambient_temp=25.0,
            linefill=copy.deepcopy(linefill_state),
            dra_reach_km=reach,
            hours=hours,
            start_time="00:00",
            enumerate_loops=False,
            rpm_step=50,
            dra_step=2,
        )
        assert not result.get("error"), result.get("message")
        assert result["dra_ppm_origin_pump"] > 0
        sdh_progression.append(result["sdh_origin_pump"])
        linefill_state = copy.deepcopy(result["linefill"])

    assert all(b <= a for a, b in zip(sdh_progression, sdh_progression[1:]))
    diffs = [a - b for a, b in zip(sdh_progression, sdh_progression[1:])]
    assert diffs, "Expected SDH to change over successive hours"
    assert max(diffs) <= 6.0


def test_queue_preserves_length_and_zero_front_progression() -> None:
    """Low pumped lengths should not shorten the queue or skip 0-ppm fronts."""

    segment_length = 40.0
    flow_m3h = 1200.0
    hours = 0.25
    diameter = 0.7

    queue_state = [
        {"length_km": 10.0, "dra_ppm": 0},
        {"length_km": 30.0, "dra_ppm": 18},
    ]

    stn_data = {
        "idx": 0,
        "L": segment_length,
        "d_inner": diameter,
        "d": diameter,
        "kv": 3.0,
    }
    opt = {"dra_ppm_main": 18, "nop": 0}

    total_length = _total_queue_length(queue_state)
    zero_positions = [_zero_front_within_segment(queue_state, segment_length)]
    pumped_lengths: list[float] = []

    for _ in range(4):
        precomputed = _prepare_dra_queue_consumption(
            queue_state,
            segment_length,
            flow_m3h,
            hours,
            diameter,
        )
        pumped_length, _, _ = precomputed
        pumped_lengths.append(pumped_length)
        _, queue_after, _ = _update_mainline_dra(
            queue_state,
            stn_data,
            opt,
            segment_length,
            flow_m3h,
            hours,
            pump_running=False,
            pump_shear_rate=0.0,
            dra_shear_factor=0.0,
            shear_injection=False,
            is_origin=True,
            precomputed=precomputed,
        )
        assert math.isclose(
            _total_queue_length(queue_after),
            total_length,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        queue_state = queue_after
        zero_positions.append(_zero_front_within_segment(queue_state, segment_length))

    assert pumped_lengths, "Expected at least one pumped-length sample"
    reference = pumped_lengths[0]
    for pumped_length in pumped_lengths[1:]:
        assert math.isclose(pumped_length, reference, rel_tol=1e-9, abs_tol=1e-12)

    expected_first = zero_positions[0] + reference
    assert math.isclose(zero_positions[1], expected_first, rel_tol=1e-9, abs_tol=1e-9)
    for previous, current in zip(zero_positions, zero_positions[1:]):
        assert current + 1e-9 >= previous
        assert current - previous <= reference + 1e-9
