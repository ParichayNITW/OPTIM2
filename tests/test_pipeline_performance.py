import copy
import json
import time
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_model import solve_pipeline


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
            "min_residual": 30,
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
    assert duration < 6.0, f"Optimizer took too long: {duration:.2f}s"
