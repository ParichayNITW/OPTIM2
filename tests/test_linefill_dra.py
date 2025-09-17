from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_model import solve_pipeline


def _make_pump_station(name: str) -> dict:
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
        "max_dr": 0,
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
    # travelling through the line (positive downstream reach).
    assert dra_result["sdh_station_b"] < base_result["sdh_station_b"]
    assert dra_result["dra_front_km"] > 0
