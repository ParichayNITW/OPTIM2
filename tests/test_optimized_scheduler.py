import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from optimized_scheduler import (
    DayResult,
    HourResult,
    PipelineConfig,
    Pump,
    SchedulerConfig,
    Station,
    compute_flow,
    pump_cost,
    refine_search,
    solve_day,
    solve_for_hour,
    solve_pipeline,
    solve_station,
    profile_solver,
)


@pytest.fixture(autouse=True)
def clear_caches():
    compute_flow.cache_clear()
    pump_cost.cache_clear()
    yield
    compute_flow.cache_clear()
    pump_cost.cache_clear()


def _simple_station(name: str) -> Station:
    pump = Pump(
        type_id=f"pump_{name}",
        rpm_range=tuple(range(900, 1301, 100)),
        dr_range=(0, 10, 20),
        flow_gain=0.15,
        base_flow=5.0,
        cost_coeff=0.02,
        base_cost=1.0,
        dra_penalty=0.25,
    )
    return Station(name=name, pumps=(pump,))


def test_solve_station_branch_and_bound_matches_naive():
    station = _simple_station("A")
    flow_in = 100.0
    result = solve_station(station, flow_in, cfg=SchedulerConfig())

    # Brute-force evaluation for comparison.
    naive_results = []
    for rpm in station.pumps[0].rpm_range:
        new_flow = compute_flow(station.pumps[0], rpm, flow_in)
        for dr in station.pumps[0].dr_range:
            cost = pump_cost(station.pumps[0], new_flow, dr)
            naive_results.append((cost, (rpm, dr)))
    expected_cost, expected_config = min(naive_results, key=lambda item: item[0])

    assert result.feasible
    assert math.isclose(result.cost, expected_cost, rel_tol=1e-9)
    assert result.config[0] == expected_config


def test_pump_cost_and_flow_are_cached():
    station = _simple_station("Cache")
    pump = station.pumps[0]

    compute_flow(pump, 1000, 120.0)
    compute_flow(pump, 1000, 120.0)
    info = compute_flow.cache_info()
    assert info.hits >= 1

    pump_cost(pump, 150.0, 10)
    pump_cost(pump, 150.0, 10)
    cost_info = pump_cost.cache_info()
    assert cost_info.hits >= 1


def test_solve_station_respects_dra_cap():
    station = _simple_station("Cap")
    cfg = SchedulerConfig(dra_cap_percent=5.0)
    result = solve_station(station, 100.0, cfg=cfg)
    assert result.feasible
    assert result.max_dr_percent <= cfg.dra_cap_percent + 1e-9


def test_solve_pipeline_parallelises_by_hour(monkeypatch):
    stations = (_simple_station("A"), _simple_station("B"))
    config = PipelineConfig(stations=stations, inlet_flow=(100.0, 120.0, 150.0), hours=3)

    calls = []

    class DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, *args, **kwargs):
            calls.append((func, args))
            return DummyFuture(func(*args, **kwargs))

    monkeypatch.setattr("optimized_scheduler.concurrent.futures.ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr("optimized_scheduler.concurrent.futures.as_completed", lambda futures: futures)

    results, backend = solve_pipeline(config, parallel=True, cfg=SchedulerConfig())

    assert backend in {"process", "thread"}
    assert len(results) == 3
    assert all(isinstance(res, HourResult) for res in results)
    assert all(call[0] is solve_for_hour for call in calls)


def test_refine_search_locates_minimum():
    func = lambda x: (x - 4.0) ** 2
    result = refine_search(func, 0.0, 10.0, steps=6, iterations=4)
    assert abs(result - 4.0) < 0.5


def test_solve_for_hour_returns_hour_result():
    station = _simple_station("A")
    config = PipelineConfig(stations=(station,), inlet_flow=(100.0,), hours=1)
    hour_result = solve_for_hour(config, 0, cfg=SchedulerConfig())
    assert isinstance(hour_result, HourResult)
    assert hour_result.feasible
    assert hour_result.rows
    assert hour_result.cost_currency is not None
    assert hour_result.legacy_payload.get("pump_settings")


def test_profile_solver_reports_functions():
    stations = (_simple_station("A"),)
    config = PipelineConfig(stations=stations, inlet_flow=(100.0,), hours=1)
    stats_output = profile_solver(config, limit=5)
    assert "solve_for_hour" in stats_output


def test_solve_day_wraps_pipeline_results():
    station = _simple_station("A")
    case = {"stations": [station], "inlet_flow": [100.0], "hours": 2}
    cfg = SchedulerConfig(parallel_hours=False)
    progress_calls = []

    def progress(kind, **kw):
        progress_calls.append((kind, kw))

    day = solve_day(case, cfg, terminal_min_residual_m=50.0, progress_callback=progress)
    assert isinstance(day, DayResult)
    assert len(day.hours) == 2
    assert progress_calls, "Expected progress callbacks"

