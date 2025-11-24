import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pipeline_model


def test_bruteforce_uses_full_dra_grid(monkeypatch):
    # Capture all DRA grids requested during the brute-force exhaustive pass.
    requested_grids = []

    original_allowed = pipeline_model._allowed_values

    def recorder(min_val, max_val, step):
        requested_grids.append((min_val, max_val, step))
        return original_allowed(min_val, max_val, step)

    monkeypatch.setattr(pipeline_model, "_allowed_values", recorder)

    stations = [
        {
            "name": "Origin",
            "idx": 0,
            "elev": 0.0,
            "L": 1.0,
            "D": 0.762,
            "t": 0.0079248,
            "rough": 4e-05,
            "is_pump": True,
            "max_dr": 10.0,
            "pump_types": {
                "A": {
                    "name": "Pump A",
                    "head_data": [
                        {"Flow (m³/hr)": 0.0, "Head (m)": 400.0},
                        {"Flow (m³/hr)": 1000.0, "Head (m)": 380.0},
                    ],
                    "eff_data": [
                        {"Flow (m³/hr)": 0.0, "Efficiency (%)": 70.0},
                        {"Flow (m³/hr)": 1000.0, "Efficiency (%)": 80.0},
                    ],
                    "power_type": "Grid",
                    "MinRPM": 1000,
                    "DOL": 1000,
                    "rate": 5.0,
                    "available": 1,
                }
            },
            "pump_names": ["Pump A"],
            "pump_name": "Pump A",
            "max_pumps": 1,
            "MinRPM": 1000,
            "DOL": 1000,
            "pump_combo": {"A": 1},
        }
    ]

    terminal = {"name": "Term", "elev": 0.0, "min_residual": 0.0}

    result = pipeline_model.solve_pipeline(
        stations,
        terminal,
        500.0,
        [5.0],
        [850.0],
        [[]],
        RateDRA=385.0,
        Price_HSD=72.0,
        Fuel_density=820.0,
        Ambient_temp=25.0,
        linefill=[{"volume": 1000.0, "dra_ppm": 5.0}],
        dra_reach_km=250.0,
        mop_kgcm2=None,
        hours=1.0,
        start_time="00:00",
        loop_usage_by_station=[0],
        enumerate_loops=False,
        rpm_step=100,
        dra_step=2,
        narrow_ranges={0: {"dra_main": (2, 4)}},
        state_top_k=5,
        state_cost_margin=0.0,
        state_cost_margin_pct=0.0,
        _exhaustive_pass=True,
        refined_retry=True,
        collect_state_audit=True,
    )

    # The brute-force pass should request a full 0..max grid using the fine step
    # instead of sticking to the narrowed 2..4 range.
    assert (0, 10, 2) in requested_grids
