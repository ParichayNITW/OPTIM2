import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_model import solve_pipeline


def test_bolpur_bypass_feasible():
    stations = [
        {
            "name": "Origin",
            "L": 0.0,
            "d": 0.7,
            "loopline": {"L": 0.0, "d": 0.7},
            "is_pump": False,
            "min_residual": 0,
        },
        {
            "name": "Bolpur",
            "L": 0.0,
            "d": 0.7,
            "is_pump": False,
            "min_residual": 0,
        },
    ]
    terminal = {"min_residual": 0}
    kv = [1.0, 1.0]
    rho = [850.0, 850.0]
    result = solve_pipeline(
        stations,
        terminal,
        1900.0,
        kv,
        rho,
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=0.0,
        Ambient_temp=25.0,
        loop_usage_by_station=[2, 0],
        enumerate_loops=False,
    )
    assert not result.get("error")
    assert result.get("bypass_next_origin") == 1

