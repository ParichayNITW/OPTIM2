from pathlib import Path
import copy
import math
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_model import solve_pipeline


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
            ppm = int(batch.get("dra_ppm", 0) or 0)
        except Exception:
            ppm = 0
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

    assert all(b >= a for a, b in zip(sdh_progression, sdh_progression[1:]))
    diffs = [b - a for a, b in zip(sdh_progression, sdh_progression[1:])]
    assert diffs, "Expected SDH to change over successive hours"
    assert max(diffs) <= 6.0
