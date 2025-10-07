"""Optimised pump scheduling helpers."""

from __future__ import annotations

import cProfile
import concurrent.futures
from dataclasses import dataclass
from functools import lru_cache
import io
import math
import pstats
from typing import Callable, Sequence
import uuid

CACHE_DECIMALS = 6
RPM_REFINEMENT_THRESHOLD = 12


def _round_cache(value: float) -> float:
    """Return a consistently rounded value suitable for caching keys."""

    return round(float(value), CACHE_DECIMALS)


@dataclass(frozen=True)
class Pump:
    """Basic pump definition used by the optimiser."""

    type_id: str
    rpm_range: tuple[int, ...]
    dr_range: tuple[int, ...]
    flow_gain: float = 0.0
    base_flow: float = 0.0
    cost_coeff: float = 1.0
    base_cost: float = 0.0
    dra_penalty: float = 0.0


@dataclass(frozen=True)
class Station:
    """Collection of pumps located at a pipeline station."""

    name: str
    pumps: tuple[Pump, ...]
    refinement_steps: int = 5
    refinement_iterations: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    """Normalised configuration consumed by :func:`solve_pipeline`."""

    stations: tuple[Station, ...]
    inlet_flow: tuple[float, ...]
    hours: int

    def flow_for_hour(self, hour: int) -> float:
        if not self.inlet_flow:
            return 0.0
        if hour < len(self.inlet_flow):
            return self.inlet_flow[hour]
        return self.inlet_flow[-1]


def _normalise_pump(definition: Pump | dict | None) -> Pump:
    if isinstance(definition, Pump):
        return definition
    if definition is None:
        raise ValueError("Pump definition cannot be None")
    rpm_raw = definition.get("rpm_range") or definition.get("rpm_values") or []
    dr_raw = definition.get("dr_range") or definition.get("dra_range") or []
    rpm_vals = tuple(sorted({int(val) for val in rpm_raw}))
    dr_vals = tuple(sorted({int(val) for val in dr_raw}))
    if not rpm_vals:
        rpm_vals = (0,)
    if not dr_vals:
        dr_vals = (0,)
    type_id = (
        str(definition.get("type_id"))
        or str(definition.get("id"))
        or str(definition.get("name"))
        or f"pump_{uuid.uuid4()}"
    )
    return Pump(
        type_id=type_id,
        rpm_range=rpm_vals,
        dr_range=dr_vals,
        flow_gain=float(definition.get("flow_gain", 0.0)),
        base_flow=float(definition.get("base_flow", 0.0)),
        cost_coeff=float(definition.get("cost_coeff", 1.0)),
        base_cost=float(definition.get("base_cost", 0.0)),
        dra_penalty=float(definition.get("dra_penalty", 0.0)),
    )


def _normalise_station(definition: Station | dict) -> Station:
    if isinstance(definition, Station):
        return definition
    pumps_raw = definition.get("pumps") or []
    pumps = tuple(_normalise_pump(p) for p in pumps_raw)
    name = str(definition.get("name") or f"Station {uuid.uuid4().hex[:8]}")
    steps = int(definition.get("refinement_steps", 5))
    iterations = int(definition.get("refinement_iterations", 2))
    return Station(name=name, pumps=pumps, refinement_steps=steps, refinement_iterations=iterations)


def _normalise_pipeline_config(config: PipelineConfig | dict) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        return config
    stations_raw = config.get("stations") or []
    stations = tuple(_normalise_station(stn) for stn in stations_raw)
    flow_raw = config.get("inlet_flow") or config.get("flow") or []
    if isinstance(flow_raw, (int, float)):
        flow_seq: Sequence[float] = [float(flow_raw)]
    else:
        flow_seq = [float(val) for val in flow_raw]
    hours = int(config.get("hours", len(flow_seq) or 24))
    if hours <= 0:
        hours = 1
    if not flow_seq:
        flow_seq = [0.0] * hours
    if len(flow_seq) < hours:
        flow_seq.extend([flow_seq[-1]] * (hours - len(flow_seq)))
    inlet_flow = tuple(flow_seq[:hours])
    return PipelineConfig(stations=stations, inlet_flow=inlet_flow, hours=hours)


@lru_cache(maxsize=None)
def _compute_flow_cached(pump: Pump, rpm: int, flow_in: float) -> float:
    rpm_val = float(rpm)
    flow = float(flow_in)
    adjusted = flow + pump.base_flow + pump.flow_gain * rpm_val
    return max(adjusted, 0.0)


def compute_flow(pump: Pump, rpm: int, flow_in: float) -> float:
    """Return the new flow exiting ``pump`` at ``rpm`` for ``flow_in``."""

    return _compute_flow_cached(pump, int(rpm), _round_cache(flow_in))


def _clear_flow_cache() -> None:
    _compute_flow_cached.cache_clear()


compute_flow.cache_clear = _clear_flow_cache  # type: ignore[attr-defined]
compute_flow.cache_info = _compute_flow_cached.cache_info  # type: ignore[attr-defined]


@lru_cache(maxsize=None)
def _pump_cost_cached(pump: Pump, flow: float, dr: int) -> float:
    flow_val = float(flow)
    rpm_factor = max(flow_val, 0.0)
    dr_factor = 1.0 - pump.dra_penalty * (int(dr) / 100.0)
    cost = pump.base_cost + pump.cost_coeff * rpm_factor * dr_factor
    return max(cost, 0.0)


def pump_cost(pump: Pump, flow: float, dr: int) -> float:
    """Return cached operating cost for ``pump`` at ``flow`` and ``dr``."""

    return _pump_cost_cached(pump, _round_cache(flow), int(dr))


def _clear_cost_cache() -> None:
    _pump_cost_cached.cache_clear()


pump_cost.cache_clear = _clear_cost_cache  # type: ignore[attr-defined]
pump_cost.cache_info = _pump_cost_cached.cache_info  # type: ignore[attr-defined]


def refine_search(
    func: Callable[[float], float],
    low: float,
    high: float,
    *,
    steps: int = 5,
    iterations: int = 3,
) -> float:
    """Successively narrow an interval around the minimum of ``func``."""

    if high < low:
        low, high = high, low
    if math.isclose(high, low):
        return float(low)
    span = float(high - low)
    if span <= 0:
        return float(low)
    best_x = low
    best_val = float("inf")
    current_low = float(low)
    current_high = float(high)
    steps = max(1, int(steps))
    iterations = max(1, int(iterations))
    for _ in range(iterations):
        xs = [current_low + j * (current_high - current_low) / steps for j in range(steps + 1)]
        vals = [func(x) for x in xs]
        min_idx = min(range(len(vals)), key=lambda i: vals[i])
        best_x = xs[min_idx]
        best_val = vals[min_idx]
        window = (current_high - current_low) / (steps * 2)
        centre = xs[min_idx]
        current_low = max(low, centre - window)
        current_high = min(high, centre + window)
        if math.isclose(current_high, current_low):
            break
    return float(best_x if math.isfinite(best_val) else low)


def _adaptive_rpm_candidates(pump: Pump, flow: float, station: Station) -> tuple[int, ...]:
    rpm_vals = pump.rpm_range
    if len(rpm_vals) <= RPM_REFINEMENT_THRESHOLD or len(rpm_vals) <= station.refinement_steps:
        return rpm_vals

    low, high = rpm_vals[0], rpm_vals[-1]

    def rpm_proxy(rpm_value: float) -> float:
        rpm_int = int(round(rpm_value))
        flow_out = compute_flow(pump, rpm_int, flow)
        return pump_cost(pump, flow_out, pump.dr_range[0] if pump.dr_range else 0)

    centre = refine_search(
        rpm_proxy,
        low,
        high,
        steps=station.refinement_steps,
        iterations=station.refinement_iterations,
    )
    sorted_candidates = sorted(rpm_vals, key=lambda val: abs(val - centre))
    limit = min(len(sorted_candidates), station.refinement_steps + 1)
    return tuple(sorted(sorted_candidates[:limit]))


def solve_station(
    station: Station | dict,
    flow_in: float,
    global_best: float | None = None,
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Return the lowest-cost pump configuration for ``station``.

    The search uses branch-and-bound pruning combined with memoisation of partial
    states.  ``global_best`` optionally supplies a hard upper-bound on the final
    cost which helps prune additional branches when the caller already tracks a
    full-pipeline incumbent.
    """

    stn = _normalise_station(station)
    if not stn.pumps:
        return 0.0, ()

    pumps = stn.pumps
    rpm_choices = []
    dr_choices = []
    for pump in pumps:
        rpm_vals = _adaptive_rpm_candidates(pump, flow_in, stn)
        if not rpm_vals:
            rpm_vals = (0,)
        rpm_choices.append(rpm_vals)
        dr_vals = pump.dr_range or (0,)
        dr_choices.append(dr_vals)

    best_cost = float("inf")
    best_config: tuple[tuple[int, int], ...] = ()
    cache: dict[tuple[int, float], float] = {}
    current: list[tuple[int, int]] = []
    bound = float("inf") if global_best is None else float(global_best)

    def dfs(idx: int, flow: float, cost: float) -> None:
        nonlocal best_cost, best_config
        if cost >= best_cost or cost >= bound:
            return
        key = (idx, _round_cache(flow))
        seen_cost = cache.get(key)
        if seen_cost is not None and cost >= seen_cost - 1e-9:
            return
        cache[key] = cost if seen_cost is None or cost < seen_cost else seen_cost
        if idx == len(pumps):
            best_cost = cost
            best_config = tuple(current)
            return
        pump = pumps[idx]
        for rpm in rpm_choices[idx]:
            flow_next = compute_flow(pump, rpm, flow)
            for dr in dr_choices[idx]:
                new_cost = cost + pump_cost(pump, flow_next, dr)
                if new_cost >= best_cost or new_cost >= bound:
                    continue
                current.append((int(rpm), int(dr)))
                dfs(idx + 1, flow_next, new_cost)
                current.pop()

    dfs(0, float(flow_in), 0.0)
    return best_cost, best_config


def _apply_configuration_flow(
    station: Station,
    flow: float,
    config: Sequence[tuple[int, int]],
) -> float:
    updated = float(flow)
    for pump, setting in zip(station.pumps, config):
        rpm, _dr = setting
        updated = compute_flow(pump, rpm, updated)
    return updated


def solve_for_hour(
    pipeline_config: PipelineConfig | dict,
    hour: int,
) -> tuple[dict[str, tuple[tuple[int, int], ...]], float]:
    """Solve one hourly sub-problem returning the schedule and cost."""

    config = _normalise_pipeline_config(pipeline_config)
    flow = config.flow_for_hour(hour)
    schedule: dict[str, tuple[tuple[int, int], ...]] = {}
    total_cost = 0.0
    for station in config.stations:
        cost, combo = solve_station(station, flow)
        total_cost += cost
        schedule[station.name] = combo
        station_norm = _normalise_station(station)
        flow = _apply_configuration_flow(station_norm, flow, combo)
    return schedule, total_cost


def solve_pipeline(
    pipeline_config: PipelineConfig | dict,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
) -> list[tuple[dict[str, tuple[tuple[int, int], ...]], float]]:
    """Solve each hourly scenario, optionally in parallel."""

    config = _normalise_pipeline_config(pipeline_config)
    hours = range(config.hours)
    results: list[tuple[dict[str, tuple[tuple[int, int], ...]], float]] = [
        ({}, 0.0) for _ in hours
    ]
    if not parallel or config.hours <= 1:
        return [solve_for_hour(config, hr) for hr in hours]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_for_hour, config, hr): hr for hr in hours}
        for future in concurrent.futures.as_completed(futures):
            hr = futures[future]
            results[hr] = future.result()
    return results


def profile_solver(
    pipeline_config: PipelineConfig | dict,
    *,
    sort: str = "cumtime",
    limit: int = 20,
) -> str:
    """Profile a pipeline solve and return formatted statistics."""

    profiler = cProfile.Profile()
    profiler.enable()
    solve_pipeline(pipeline_config, parallel=False)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sort)
    stats.print_stats(limit)
    return stream.getvalue()


__all__ = [
    "Pump",
    "Station",
    "PipelineConfig",
    "compute_flow",
    "pump_cost",
    "refine_search",
    "solve_station",
    "solve_for_hour",
    "solve_pipeline",
    "profile_solver",
]

