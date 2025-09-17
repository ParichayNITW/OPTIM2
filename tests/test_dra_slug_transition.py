from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_model import solve_pipeline


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
            dra_step=1,
        )
        sdh_values.append(result["sdh_origin_pump"])

    assert sdh_values == sorted(sdh_values)
    diffs = [abs(b - a) for a, b in zip(sdh_values, sdh_values[1:])]
    assert diffs, "Expected at least one SDH difference"
    assert max(diffs) <= 6.0
