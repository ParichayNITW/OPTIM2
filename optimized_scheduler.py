"""Optimised pump scheduling helpers."""

from __future__ import annotations

import cProfile
import concurrent.futures
from dataclasses import dataclass
from functools import lru_cache
import io
import math
import os
import pstats
from typing import Callable, Iterable, Mapping, Sequence
import uuid

CACHE_DECIMALS = 6
RPM_REFINEMENT_THRESHOLD = 12
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))


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
    label: str | None = None


@dataclass(frozen=True)
class PeakConstraint:
    """Residual head requirement at a station peak."""

    name: str
    available_head_m: float
    required_head: float | Callable[..., float]

    def residual_head(self, flow: float, temperature_c: float, dr_percent: float) -> float:
        """Return the residual head for this peak."""

        requirement = self.required_head
        required_val: float
        if callable(requirement):
            try:
                required_val = float(requirement(flow=flow, temperature_c=temperature_c, dr=dr_percent))
            except TypeError:
                try:
                    required_val = float(requirement(flow, dr_percent))
                except TypeError:
                    try:
                        required_val = float(requirement(flow))
                    except TypeError:
                        required_val = float(requirement())
        else:
            required_val = float(requirement)
        return float(self.available_head_m) - required_val


@dataclass(frozen=True)
class Station:
    """Collection of pumps located at a pipeline station."""

    name: str
    pumps: tuple[Pump, ...]
    refinement_steps: int = 5
    refinement_iterations: int = 2
    peaks: tuple[PeakConstraint, ...] = ()
    temperature_c: float | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Normalised configuration consumed by :func:`solve_pipeline`."""

    stations: tuple[Station, ...]
    inlet_flow: tuple[float, ...]
    hours: int
    ambient_temp_c: float = 25.0

    def flow_for_hour(self, hour: int) -> float:
        if not self.inlet_flow:
            return 0.0
        if hour < len(self.inlet_flow):
            return self.inlet_flow[hour]
        return self.inlet_flow[-1]


def _normalise_pump(definition: Pump | Mapping[str, object] | None) -> Pump:
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
    label = definition.get("label") or definition.get("display_name") or definition.get("name")
    label_str = str(label) if label not in (None, "") else type_id
    return Pump(
        type_id=type_id,
        rpm_range=rpm_vals,
        dr_range=dr_vals,
        flow_gain=float(definition.get("flow_gain", 0.0)),
        base_flow=float(definition.get("base_flow", 0.0)),
        cost_coeff=float(definition.get("cost_coeff", 1.0)),
        base_cost=float(definition.get("base_cost", 0.0)),
        dra_penalty=float(definition.get("dra_penalty", 0.0)),
        label=label_str,
    )


def _normalise_peak(entry: PeakConstraint | Mapping[str, object]) -> PeakConstraint:
    if isinstance(entry, PeakConstraint):
        return entry
    name = str(entry.get("name") or f"peak_{uuid.uuid4().hex[:6]}")
    available = float(entry.get("available_head_m", entry.get("available", float("inf"))))
    required = entry.get("required_head_m")
    if required is None:
        required = entry.get("required_head")
    if required is None:
        required = 0.0
    return PeakConstraint(name=name, available_head_m=available, required_head=required)


def _normalise_station(definition: Station | Mapping[str, object]) -> Station:
    if isinstance(definition, Station):
        return definition
    pumps_raw = definition.get("pumps") or []
    pumps = tuple(_normalise_pump(p) for p in pumps_raw)
    name = str(definition.get("name") or f"Station {uuid.uuid4().hex[:8]}")
    steps = int(definition.get("refinement_steps", 5))
    iterations = int(definition.get("refinement_iterations", 2))
    peaks_raw = definition.get("peaks") or ()
    peaks = tuple(_normalise_peak(peak) for peak in peaks_raw if peak is not None)
    temperature = definition.get("temperature_c")
    temperature_c = float(temperature) if temperature not in (None, "") else None
    return Station(
        name=name,
        pumps=pumps,
        refinement_steps=steps,
        refinement_iterations=iterations,
        peaks=peaks,
        temperature_c=temperature_c,
    )


def _normalise_pipeline_config(config: PipelineConfig | Mapping[str, object]) -> PipelineConfig:
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
    ambient = config.get("ambient_temp_c")
    if ambient is None:
        ambient = config.get("ambient_temp")
    try:
        ambient_temp_c = float(ambient)
    except (TypeError, ValueError):
        ambient_temp_c = 25.0
    return PipelineConfig(stations=stations, inlet_flow=inlet_flow, hours=hours, ambient_temp_c=ambient_temp_c)


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
    station: Station | Mapping[str, object],
    flow_in: float,
    global_best: float | None = None,
    *,
    cfg: SchedulerConfig | None = None,
    temperature_c: float | None = None,
) -> StationSearchResult:
    """Return the lowest-cost pump configuration for ``station``.

    The search uses branch-and-bound pruning combined with memoisation of partial
    states. ``cfg`` supplies feasibility constraints such as the maximum DRA
    percentage and minimum residual head at peaks.
    """

    stn = _normalise_station(station)
    if not stn.pumps:
        return StationSearchResult(True, 0.0, (), 0.0, None)

    pumps = stn.pumps
    rpm_choices: list[tuple[int, ...]] = []
    dr_choices: list[tuple[int, ...]] = []
    for pump in pumps:
        rpm_vals = _adaptive_rpm_candidates(pump, flow_in, stn)
        if not rpm_vals:
            rpm_vals = (0,)
        rpm_choices.append(rpm_vals)
        dr_vals = pump.dr_range or (0,)
        dr_choices.append(dr_vals)

    best_cost = float("inf")
    best_config: tuple[tuple[int, int], ...] = ()
    best_max_dr = 0.0
    cache: dict[tuple[int, float], float] = {}
    current: list[tuple[int, int]] = []
    bound = float("inf") if global_best is None else float(global_best)
    dr_cap = cfg.dra_cap_percent if cfg is not None else None
    min_residual = cfg.min_peak_head_m if cfg is not None else None
    ambient_temp = stn.temperature_c if stn.temperature_c is not None else temperature_c
    if ambient_temp is None:
        ambient_temp = 25.0

    infeasible_reason = "No feasible configuration found"

    def violates_peaks(flow_value: float, dr_percent: float) -> bool:
        if not stn.peaks or min_residual is None:
            return False
        for peak in stn.peaks:
            residual = peak.residual_head(flow_value, ambient_temp, dr_percent)
            if residual < min_residual - 1e-9:
                return True
        return False

    def dfs(idx: int, flow: float, cost: float, max_dr_used: float) -> None:
        nonlocal best_cost, best_config, best_max_dr, infeasible_reason
        if cost >= best_cost or cost >= bound:
            return
        if dr_cap is not None and max_dr_used > dr_cap + 1e-9:
            infeasible_reason = "DR cap violated"
            return
        key = (idx, _round_cache(flow))
        seen_cost = cache.get(key)
        if seen_cost is not None and cost >= seen_cost - 1e-9:
            return
        cache[key] = cost if seen_cost is None or cost < seen_cost else seen_cost
        if idx == len(pumps):
            best_cost = cost
            best_config = tuple(current)
            best_max_dr = max_dr_used
            return
        pump = pumps[idx]
        for rpm in rpm_choices[idx]:
            flow_next = compute_flow(pump, rpm, flow)
            for dr in dr_choices[idx]:
                dr_float = float(dr)
                new_max_dr = max(max_dr_used, dr_float)
                if dr_cap is not None and new_max_dr > dr_cap + 1e-9:
                    infeasible_reason = "DR cap violated"
                    continue
                if violates_peaks(flow_next, dr_float):
                    infeasible_reason = "peak residual head violated"
                    continue
                new_cost = cost + pump_cost(pump, flow_next, dr)
                if new_cost >= best_cost or new_cost >= bound:
                    continue
                current.append((int(rpm), int(dr)))
                dfs(idx + 1, flow_next, new_cost, new_max_dr)
                current.pop()

    dfs(0, float(flow_in), 0.0, 0.0)
    feasible = math.isfinite(best_cost)
    reason = None if feasible else infeasible_reason
    return StationSearchResult(feasible, best_cost, best_config, best_max_dr, reason)


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
    pipeline_config: PipelineConfig | Mapping[str, object],
    hour: int,
    *,
    cfg: SchedulerConfig | None = None,
) -> HourResult:
    """Solve one hourly sub-problem returning the schedule and cost."""

    config = _normalise_pipeline_config(pipeline_config)
    flow = config.flow_for_hour(hour)
    schedule: dict[str, list[dict[str, int | float | str]]] = {}
    total_cost = 0.0
    feasible = True
    max_dr = 0.0
    rows: list[tuple[str, str, int, int]] = []
    temperature = config.ambient_temp_c
    message: str | None = None

    for station in config.stations:
        station_norm = _normalise_station(station)
        result = solve_station(
            station_norm,
            flow,
            cfg=cfg,
            temperature_c=temperature,
        )
        if not result.feasible:
            feasible = False
            message = result.reason
            break
        total_cost += result.cost
        pretty_rows: list[dict[str, int | float | str]] = []
        for pump, (rpm, dr) in zip(station_norm.pumps, result.config):
            label = pump.label or pump.type_id
            rows.append((station_norm.name, label, rpm, dr))
            max_dr = max(max_dr, float(dr))
            pretty_rows.append({"pump": label, "rpm": int(rpm), "dr_percent": int(dr)})
        schedule[station_norm.name] = pretty_rows
        flow = _apply_configuration_flow(station_norm, flow, result.config)

    if not feasible:
        pretty_table = "No feasible configuration found."
        total_cost = float("inf")
    else:
        headers = "| Station | Pump | RPM | DRA (%) |\n|---|---|---:|---:|"
        body = "\n".join(
            f"| {station} | {pump} | {rpm} | {dr} |" for station, pump, rpm, dr in rows
        )
        pretty_table = f"{headers}\n{body}" if body else headers

    return HourResult(feasible, total_cost, schedule, max_dr, pretty_table, message)


def solve_pipeline(
    pipeline_config: PipelineConfig | Mapping[str, object],
    *,
    parallel: bool = True,
    max_workers: int | None = None,
    cfg: SchedulerConfig | None = None,
    progress_callback: Callable[[str, int | None, float | None], None] | None = None,
) -> tuple[list[HourResult], str]:
    """Solve each hourly scenario, optionally in parallel."""

    config = _normalise_pipeline_config(pipeline_config)
    hours = range(config.hours)
    if not parallel or config.hours <= 1:
        results: list[HourResult] = []
        for hr in hours:
            if progress_callback:
                progress_callback("hour_start", hour=hr, pct=None)
            hr_result = solve_for_hour(config, hr, cfg=cfg)
            results.append(hr_result)
            if progress_callback:
                progress_callback(
                    "hour_done", hour=hr, pct=(len(results) / config.hours) * 100.0
                )
        return results, "serial"

    worker_count = max_workers if max_workers is not None else DEFAULT_WORKERS
    try:
        executor: concurrent.futures.Executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        )
        backend = "process"
    except Exception:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_count, 4))
        backend = "thread"

    results: list[HourResult] = [
        HourResult(False, float("inf"), {}, 0.0, "No result", "Not started") for _ in hours
    ]
    if progress_callback:
        for hr in hours:
            progress_callback("hour_start", hour=hr, pct=None)
    with executor:
        futures = {
            executor.submit(solve_for_hour, config, hr, cfg=cfg): hr for hr in hours
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            hr = futures[future]
            try:
                results[hr] = future.result(timeout=cfg.per_hour_timeout_s if cfg else None)
            except Exception as exc:  # pragma: no cover - defensive
                results[hr] = HourResult(
                    False,
                    float("inf"),
                    {},
                    0.0,
                    "No feasible configuration found.",
                    f"Execution failed: {exc}",
                )
            completed += 1
            if progress_callback:
                progress_callback(
                    "hour_done", hour=hr, pct=(completed / config.hours) * 100.0
                )
    return results, backend


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


def solve_day(
    case: Mapping[str, object],
    cfg: SchedulerConfig,
    *,
    terminal_min_residual_m: float,
    progress_callback: Callable[[str, int | None, float | None, str | None], None] | None = None,
) -> DayResult:
    """Solve an entire day of hourly scenarios using ``cfg`` constraints."""

    def _progress(kind: str, hour: int | None, pct: float | None) -> None:
        if progress_callback is None:
            return
        msg = None
        if kind == "summary":
            msg = "Daily optimisation complete"
        progress_callback(kind, hour=hour, pct=pct, msg=msg)

    case_copy = dict(case)
    hours = int(case_copy.get("hours", 24) or 24)
    case_copy["hours"] = hours
    inlet_flow = case_copy.get("inlet_flow") or case_copy.get("flow")
    if inlet_flow is None:
        inlet_flow = [0.0] * hours
    if isinstance(inlet_flow, (int, float)):
        inlet_flow = [float(inlet_flow)] * hours
    if len(inlet_flow) < hours:
        inlet_flow = list(inlet_flow) + [float(inlet_flow[-1])] * (hours - len(inlet_flow))
    case_copy["inlet_flow"] = inlet_flow[:hours]

    if progress_callback:
        progress_callback("hour_start", hour=0, pct=0.0, msg="Starting optimisation")

    results, backend = solve_pipeline(
        case_copy,
        parallel=cfg.parallel_hours,
        cfg=cfg,
        progress_callback=lambda kind, hour, pct: _progress(kind, hour, pct),
    )

    if progress_callback:
        progress_callback(
            "summary",
            hour=None,
            pct=100.0,
            msg=f"Backend: {backend} · terminal ≥ {terminal_min_residual_m:.1f} m",
        )

    return DayResult(tuple(results), backend)


__all__ = [
    "Pump",
    "Station",
    "PipelineConfig",
    "SchedulerConfig",
    "StationSearchResult",
    "HourResult",
    "DayResult",
    "compute_flow",
    "pump_cost",
    "refine_search",
    "solve_station",
    "solve_for_hour",
    "solve_pipeline",
    "profile_solver",
    "solve_day",
]

@dataclass(frozen=True)
class SchedulerConfig:
    """Tunable knobs for the day solver."""

    rpm_refine_step: int = 25
    dr_step: float = 2.0
    coarse_mult: float = 5.0
    max_states: int = 50
    dp_cost_margin: float = 5_000.0
    dra_cap_percent: float = 30.0
    min_peak_head_m: float = 25.0
    parallel_hours: bool = True
    per_hour_timeout_s: float | None = None


@dataclass(frozen=True)
class StationSearchResult:
    """Detailed outcome of a station branch-and-bound search."""

    feasible: bool
    cost: float
    config: tuple[tuple[int, int], ...]
    max_dr_percent: float
    reason: str | None = None


@dataclass(frozen=True)
class HourResult:
    """User-facing summary for a solved hour."""

    feasible: bool
    cost_currency: float
    pump_settings: dict[str, list[dict[str, int | float | str]]]
    max_dr_percent: float
    pretty_table: str
    message: str | None = None


@dataclass(frozen=True)
class DayResult:
    """Aggregate of all hourly results."""

    hours: tuple[HourResult, ...]
    backend: str
