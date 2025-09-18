from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_model import solve_pipeline


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
    # travelling through the line (positive downstream reach).
    assert dra_result["sdh_station_b"] < base_result["sdh_station_b"]
    assert dra_result["dra_front_km"] > 0


def test_zero_injection_benefits_from_inherited_slug() -> None:
    """Inherited DRA continues lowering SDH when no station injects."""

    def _treated_length(linefill_state: list[dict], diameter: float) -> float:
        if diameter <= 0:
            return 0.0
        area = 3.141592653589793 * (diameter ** 2) / 4.0
        total = 0.0
        for batch in linefill_state:
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
