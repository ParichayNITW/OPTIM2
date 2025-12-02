import inspect
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pipeline_model


def test_priority_feasibility_kw_present_on_solve_pipeline():
    params = inspect.signature(pipeline_model.solve_pipeline).parameters
    assert "priority_feasibility" in params


def test_priority_feasibility_kw_present_on_solve_pipeline_with_types(monkeypatch):
    captured = {}

    def _fake_solve_pipeline(*args, **kwargs):
        captured["priority_feasibility"] = kwargs.get("priority_feasibility")
        return {}

    monkeypatch.setattr(pipeline_model, "solve_pipeline", _fake_solve_pipeline)

    pipeline_model.solve_pipeline_with_types(
        stations=[{"pump_types": {"A": {"available": 1}}}],
        terminal={},
        FLOW=0.0,
        KV_list=[0.0],
        rho_list=[0.0],
        segment_slices=[[]],
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=0.0,
        Ambient_temp=0.0,
        priority_feasibility=True,
    )

    assert captured["priority_feasibility"] is True
