import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pipeline_model


def test_exhaustive_pass_ignores_floor_filtering(monkeypatch):
    """The brute-force exhaustive pass should keep the full DRA grid even when a floor is set."""

    # Make drag-reduction percentage map 1:1 to ppm for a predictable grid.
    monkeypatch.setattr(pipeline_model, "get_ppm_for_dr", lambda kv, dr: float(dr))
    monkeypatch.setattr(pipeline_model, "get_dr_for_ppm", lambda kv, ppm: float(ppm))
    accepted_ppm: list[float] = []

    def record_injection(ppm_requested, *_, **__):
        accepted_ppm.append(float(ppm_requested or 0.0))
        return float(ppm_requested or 0.0)

    monkeypatch.setattr(pipeline_model, "_predict_effective_injection", record_injection)

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
        100.0,
        [5.0],
        [850.0],
        [[]],
        RateDRA=385.0,
        Price_HSD=72.0,
        Fuel_density=820.0,
        Ambient_temp=25.0,
        linefill=[{"volume": 1000.0, "dra_ppm": 6.0}],
        dra_reach_km=250.0,
        mop_kgcm2=None,
        hours=1.0,
        start_time="00:00",
        loop_usage_by_station=[0],
        enumerate_loops=False,
        rpm_step=100,
        dra_step=2,
        state_top_k=5,
        state_cost_margin=0.0,
        state_cost_margin_pct=0.0,
        _exhaustive_pass=True,
        refined_retry=True,
        collect_state_audit=True,
        segment_floors=[{"dra_ppm": 6.0, "length_km": 1.0}],
    )

    # With dra_step=2 and max_dr=10 we expect six drag-reduction points (0..10).
    # The exhaustive pass should still try them all even with a 6 ppm floor.
    assert len([ppm for ppm in accepted_ppm if ppm in {0.0, 2.0, 4.0, 6.0, 8.0, 10.0}]) >= 6
