from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_model import solve_pipeline_with_types


def test_type_combo_without_station_max_pumps_uses_active_sum() -> None:
    stations = [
        {
            "name": "Origin Pump",
            "is_pump": True,
            "pump_types": {
                "A": {
                    "available": 3,
                    "MinRPM": 900,
                    "DOL": 3000,
                    "A": 0.0,
                    "B": 0.0,
                    "C": 185.0,
                    "P": 0.0,
                    "Q": 0.0,
                    "R": 0.0,
                    "S": 0.0,
                    "T": 80.0,
                }
            },
            "L": 100.0,
            "d": 0.66,
            "rough": 0.00004,
            "elev": 0.0,
            "min_residual": 40,
            "MinRPM": 900,
            "DOL": 3000,
            "max_dr": 0,
            "power_type": "Grid",
            "rate": 0.0,
        }
    ]
    terminal = {"name": "Terminal", "elev": 30.0, "min_residual": 45}

    result = solve_pipeline_with_types(
        stations,
        terminal,
        FLOW=2800.0,
        KV_list=[1.5],
        rho_list=[855.0],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=820.0,
        Ambient_temp=25.0,
        linefill=[],
        dra_reach_km=0.0,
        hours=6.0,
        start_time="00:00",
        dra_shear_factor=0.0,
    )

    assert not result.get("error"), result.get("message")
    message = result.get("message") or ""
    assert "Frequent issues" not in message
    assert result.get("num_pumps_origin_pump") == 3

    stations_used = result.get("stations_used")
    assert stations_used, "expanded station list missing"
    unit = stations_used[0]
    assert unit.get("active_combo") == {"A": 3, "B": 0}
    assert unit.get("max_pumps") == 3
