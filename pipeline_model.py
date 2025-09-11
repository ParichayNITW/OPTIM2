"""Simplified pipeline optimisation model without external solvers.

This module replaces the previous Pyomo/NEOS based optimisation with a
lightweight search that enumerates feasible pump operating points.  The goal is
not to be perfectly optimal but to provide reasonable results using only the
standard Python stack so the application can run in environments where no
solver is available.
"""

from __future__ import annotations

from math import log10, pi
import copy
import numpy as np

from dra_utils import get_ppm_for_dr, get_dr_for_ppm
from linefill_utils import advance_linefill

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def head_to_kgcm2(head_m: float, rho: float) -> float:
    """Convert a head value in metres to kg/cm²."""
    return head_m * rho / 10000.0


def generate_type_combinations(maxA: int = 3, maxB: int = 3) -> list[tuple[int, int]]:
    """Return all feasible pump count combinations for two pump types."""
    combos = [
        (a, b)
        for a in range(maxA + 1)
        for b in range(maxB + 1)
        if a + b > 0
    ]
    return sorted(combos, key=lambda x: (x[0] + x[1], x))

# ---------------------------------------------------------------------------
# Loop enumeration utilities
# ---------------------------------------------------------------------------

def _generate_loop_cases(num_loops: int) -> list[list[int]]:
    """Return a small set of representative loop-usage combinations.

    This helper produces a variety of loop-use vectors that are independent of
    pipe diameters.  Each element in a returned list corresponds to a looped
    segment and takes one of the following values:

      * ``0`` – loop disabled (flow only through the mainline)
      * ``1`` – loop used in parallel with the mainline (flows split)
      * ``2`` – loop used in bypass mode (loop rejoins downstream of the next pump)
      * ``3`` – loop-only mode (flow only through the loopline)

    The enumeration intentionally limits the number of combinations so that
    optimisation remains tractable.  When there is only a single looped
    segment it returns four cases.  For two loops it returns the most
    relevant permutations.  When there are more than two loops the helper
    constructs a handful of representative cases: all off, all parallel,
    each individual loop used in parallel, first bypass, last bypass and
    all loop-only.
    """
    if num_loops <= 0:
        return [[]]
    # One loop: off, parallel, bypass and loop-only
    if num_loops == 1:
        return [[0], [1], [2], [3]]
    # Two loops: no-loop, both parallel, first bypass, second parallel,
    # first parallel only, and both loop-only
    if num_loops == 2:
        return [[0, 0], [1, 1], [2, 0], [0, 1], [1, 0], [3, 3]]
    # More loops: all off, all parallel, each single loop in parallel,
    # first bypass, last bypass, and all loop-only
    combos: list[list[int]] = []
    combos.append([0] * num_loops)
    combos.append([1] * num_loops)
    for i in range(num_loops):
        c = [0] * num_loops
        c[i] = 1
        combos.append(c)
    # first bypass
    c = [0] * num_loops
    c[0] = 2
    combos.append(c)
    # last bypass
    c = [0] * num_loops
    c[-1] = 2
    combos.append(c)
    combos.append([3] * num_loops)
    # Remove duplicates while preserving order
    unique: list[list[int]] = []
    for c in combos:
        if c not in unique:
            unique.append(c)
    return unique

# ---------------------------------------------------------------------------
# Custom loop-case enumeration respecting pipe diameters
# ---------------------------------------------------------------------------

def _generate_loop_cases_by_diameter(num_loops: int, equal_diameter: bool) -> list[list[int]]:
    """Generate loop usage patterns tailored to pipe diameter equality.

    When ``equal_diameter`` is ``True`` the returned cases correspond to
    combinations required by Case‑1 in the problem description: no loops,
    parallel loops on all segments and each individual loop in parallel.  For
    instance, with two loops this yields four cases: `[0, 0]`, `[1, 1]`,
    `[0, 1]` and `[1, 0]`.  Bypass and loop‑only modes are not returned
    because they are irrelevant when the loop and mainline diameters are
    identical.

    When ``equal_diameter`` is ``False`` the returned cases reflect Case‑2:
    no loops, loop‑only across the entire pipeline and a bypass case.  With
    multiple loops the bypass directive applies only to the first looped
    segment because the specification assumes that the loop bypasses the
    next pump and rejoins the mainline downstream of that station.  Additional
    loops are disabled in this case.  For a single loop this yields three
    cases: `[0]`, `[3]` and `[2]`; for two loops: `[0, 0]`, `[3, 3]` and
    `[2, 0]`.  When more than two loops exist the patterns generalise to
    `[0, 0, ...]`, `[3, 3, ...]` and `[2, 0, 0, ...]`.
    """
    if num_loops <= 0:
        return [[]]
    if equal_diameter:
        # Case‑1: only consider off/on combinations without bypass or loop-only.
        cases: list[list[int]] = []
        # All loops off
        cases.append([0] * num_loops)
        # All loops on (parallel)
        cases.append([1] * num_loops)
        # Each loop individually on
        for i in range(num_loops):
            c = [0] * num_loops
            c[i] = 1
            if c not in cases:
                cases.append(c)
        return cases
    else:
        # Case‑2: consider mainline‑only, loop‑only and bypass for first loop.
        cases: list[list[int]] = []
        # All loops off (mainline only)
        cases.append([0] * num_loops)
        # All loops loop‑only
        cases.append([3] * num_loops)
        # Bypass on first loop and others off
        c = [0] * num_loops
        c[0] = 2
        cases.append(c)
        return cases

# ---------------------------------------------------------------------------
# Fine-grained loop-case enumeration based on per-loop diameter equality
# ---------------------------------------------------------------------------

def _generate_loop_cases_by_flags(flags: list[bool]) -> list[list[int]]:
    """Return loop usage cases for pipelines with mixed diameter equality.

    ``flags`` contains one boolean per looped segment indicating whether the
    loopline diameter matches the mainline (``True``) or differs (``False``).

    Behaviour follows the two cases described in the problem statement:

    * **Case‑1** – When all flags are ``True`` the returned patterns are the
      same as :func:`_generate_loop_cases_by_diameter` with
      ``equal_diameter=True``: all loops off, all parallel and each loop
      individually in parallel.
    * **Case‑2** – When at least one flag is ``False`` the solver considers
      only three global scenarios for the differing loops: all loops off
      (mainline only), all differing loops in loop-only mode with equal-diameter
      loops disabled, and a bypass on the first differing loop with all other
      loops disabled.  Equal-diameter loops may additionally operate in
      parallel while all differing loops remain off.

    This tailored enumeration avoids invalid combinations such as bypassing
    multiple loops simultaneously or running unequal pipes in parallel.
    When ``flags`` is empty an empty pattern is returned.
    """
    if not flags:
        return [[]]
    if all(flags):
        # All loops have equal diameter → reuse simpler helper
        return _generate_loop_cases_by_diameter(len(flags), True)

    combos: list[list[int]] = []
    n = len(flags)
    # Base case: all loops off
    combos.append([0] * n)

    # Equal-diameter loops may run in parallel when differing loops are off
    eq_positions = [i for i, eq in enumerate(flags) if eq]
    if eq_positions:
        # All equal loops on in parallel
        all_eq_parallel = [1 if eq else 0 for eq in flags]
        combos.append(all_eq_parallel)
        # Each equal loop individually on
        for i in eq_positions:
            c = [0] * n
            c[i] = 1
            combos.append(c)

    # Case-2 scenarios for differing loops
    diff_positions = [i for i, eq in enumerate(flags) if not eq]
    if diff_positions:
        # All differing loops in loop-only mode (others off)
        loop_only = [3 if not eq else 0 for eq in flags]
        combos.append(loop_only)
        # Bypass only the first differing loop
        bypass_first = [0] * n
        bypass_first[diff_positions[0]] = 2
        combos.append(bypass_first)

    # Remove duplicates while preserving order
    unique: list[list[int]] = []
    for c in combos:
        if c not in unique:
            unique.append(c)
    return unique

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

RPM_STEP = 100
DRA_STEP = 7.5
# Residual head precision (decimal places) used when bucketing states during the
# dynamic-programming search.  Using a modest precision keeps the state space
# tractable while still providing near-global optimality.
RESIDUAL_ROUND = 1
V_MIN = 0.5
V_MAX = 2.5

# Simple memoisation caches used to avoid repeatedly solving the same
# hydraulic sub-problems when many states evaluate identical conditions.
_SEGMENT_CACHE: dict[tuple, tuple] = {}
_PARALLEL_CACHE: dict[tuple, tuple] = {}


def _allowed_values(min_val: int, max_val: int, step: int) -> list[int]:
    vals = list(range(min_val, max_val + 1, step))
    if vals[-1] != max_val:
        vals.append(max_val)
    return vals


def _segment_hydraulics(
    flow_m3h: float,
    L: float,
    d_inner: float,
    rough: float,
    kv: float,
    dra_perc: float,
    dra_length: float | None = None,
) -> tuple[float, float, float, float]:
    """Return (head_loss, velocity, reynolds, friction_factor).

    ``dra_length`` expresses the portion of the segment length ``L`` (in km)
    that experiences drag reduction.  If ``dra_length`` is ``None`` or greater
    than ``L`` the drag reduction is assumed to act over the full length.  When
    the value is ``0`` only the base friction is applied.
    """

    # Cache look-up keyed by the rounded arguments.  Rounding keeps the number
    # of unique keys manageable while still distinguishing materially different
    # states.
    key = (
        round(flow_m3h, 3),
        round(L, 3),
        round(d_inner, 5),
        round(rough, 6),
        round(kv, 6),
        round(dra_perc, 1),
        round(-1.0 if dra_length is None else dra_length, 3),
    )
    if key in _SEGMENT_CACHE:
        return _SEGMENT_CACHE[key]

    g = 9.81
    flow_m3s = flow_m3h / 3600.0
    area = pi * d_inner ** 2 / 4.0
    v = flow_m3s / area if area > 0 else 0.0
    Re = v * d_inner / (kv * 1e-6) if kv > 0 else 0.0
    if Re > 0:
        if Re < 4000:
            f = 64.0 / Re
        else:
            arg = (rough / d_inner / 3.7) + (5.74 / (Re ** 0.9))
            f = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
    else:
        f = 0.0

    # Drag reduction may only apply to part of the segment.  Compute head losses
    # for the affected and unaffected lengths separately.
    if dra_length is None or dra_length >= L:
        hl_dra = f * ((L * 1000.0) / d_inner) * (v ** 2 / (2 * g)) * (1 - dra_perc / 100.0)
        head_loss = hl_dra
    elif dra_length <= 0:
        head_loss = f * ((L * 1000.0) / d_inner) * (v ** 2 / (2 * g))
    else:
        hl_dra = f * (((dra_length) * 1000.0) / d_inner) * (v ** 2 / (2 * g)) * (1 - dra_perc / 100.0)
        hl_nodra = f * (((L - dra_length) * 1000.0) / d_inner) * (v ** 2 / (2 * g))
        head_loss = hl_dra + hl_nodra

    result = (head_loss, v, Re, f)
    _SEGMENT_CACHE[key] = result
    return result


def _parallel_segment_hydraulics(
    flow_m3h: float,
    main: dict,
    loop: dict,
    kv: float,
) -> tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Split ``flow_m3h`` between ``main`` and ``loop`` so both see the same head loss.

    ``main`` and ``loop`` are dictionaries with keys ``L``, ``d_inner``, ``rough``,
    ``dra`` and ``dra_len`` describing each line.  The function returns the common
    head loss and the (velocity, Re, f, flow) tuples for each path.
    """

    def calc(line_flow: float, data: dict) -> tuple[float, float, float, float]:
        return _segment_hydraulics(
            line_flow,
            data['L'],
            data['d_inner'],
            data['rough'],
            kv,
            data.get('dra', 0.0),
            data.get('dra_len'),
        )
    key = (
        round(flow_m3h, 3),
        round(main['L'], 3),
        round(main['d_inner'], 5),
        round(main['rough'], 6),
        round(main.get('dra', 0.0), 1),
        round(-1.0 if main.get('dra_len') is None else main.get('dra_len'), 3),
        round(loop['L'], 3),
        round(loop['d_inner'], 5),
        round(loop['rough'], 6),
        round(loop.get('dra', 0.0), 1),
        round(-1.0 if loop.get('dra_len') is None else loop.get('dra_len'), 3),
        round(kv, 6),
    )
    if key in _PARALLEL_CACHE:
        return _PARALLEL_CACHE[key]

    lo, hi = 0.0, flow_m3h
    best = None
    for _ in range(20):
        mid = (lo + hi) / 2.0
        q_loop = mid
        q_main = flow_m3h - q_loop
        hl_main, v_main, Re_main, f_main = calc(q_main, main)
        hl_loop, v_loop, Re_loop, f_loop = calc(q_loop, loop)
        diff = hl_main - hl_loop
        best = (
            hl_main,
            (v_main, Re_main, f_main, q_main),
            (v_loop, Re_loop, f_loop, q_loop),
        )
        if abs(diff) < 1e-6:
            break
        if diff > 0:
            lo = mid
        else:
            hi = mid

    _PARALLEL_CACHE[key] = best
    return best

# ---------------------------------------------------------------------------
# Multi‑segment parallel flow splitting
# ---------------------------------------------------------------------------
def _split_flow_two_segments(
    flow_m3h: float,
    kv: float,
    main1: dict,
    main2: dict,
    loop1: dict,
    loop2: dict,
) -> tuple[float, float, float, float, float, float]:
    """Split ``flow_m3h`` between mainline and loopline over two segments.

    This helper solves for a loop flow ``q_loop`` and a mainline flow
    ``q_main`` such that the total head loss (friction only) across both
    segments is the same for the mainline and loopline.  ``main1`` and
    ``main2`` describe the first and second mainline segments; ``loop1``
    and ``loop2`` describe the corresponding loop segments.  Each dict must
    have keys ``L``, ``d_inner``, ``rough``, ``dra`` and ``dra_len``.

    The return tuple contains ``q_main``, ``q_loop``, the head loss for
    the first mainline segment, the head loss for the first loopline
    segment, the head loss for the second mainline segment and the
    head loss for the second loopline segment.  All head losses are
    returned in metres.
    """
    # Binary search on q_loop to equalise total head loss
    lo, hi = 0.0, flow_m3h
    best = None
    for _ in range(25):
        q_loop = (lo + hi) / 2.0
        q_main = flow_m3h - q_loop
        # Head loss for mainline segments
        hl_m1, _, _, _ = _segment_hydraulics(q_main, main1['L'], main1['d_inner'], main1['rough'], kv, main1.get('dra', 0.0), main1.get('dra_len'))
        hl_m2, _, _, _ = _segment_hydraulics(q_main, main2['L'], main2['d_inner'], main2['rough'], kv, main2.get('dra', 0.0), main2.get('dra_len'))
        hl_main_total = hl_m1 + hl_m2
        # Head loss for loopline segments
        hl_l1, _, _, _ = _segment_hydraulics(q_loop, loop1['L'], loop1['d_inner'], loop1['rough'], kv, loop1.get('dra', 0.0), loop1.get('dra_len'))
        hl_l2, _, _, _ = _segment_hydraulics(q_loop, loop2['L'], loop2['d_inner'], loop2['rough'], kv, loop2.get('dra', 0.0), loop2.get('dra_len'))
        hl_loop_total = hl_l1 + hl_l2
        diff = hl_main_total - hl_loop_total
        best = (q_main, q_loop, hl_m1, hl_l1, hl_m2, hl_l2)
        if abs(diff) < 1e-6:
            break
        if diff > 0:
            # mainline has higher head; increase loop flow
            lo = q_loop
        else:
            hi = q_loop
    return best


def _pump_head(stn: dict, flow_m3h: float, rpm: float, nop: int) -> list[dict]:
    """Return per-pump-type head and efficiency information.

    The return value is a list of dictionaries with keys ``tdh`` (total head
    contributed by that pump type), ``eff`` (efficiency at the operating
    point), ``count`` (number of pumps of that type), ``power_type`` and the
    original pump-type data under ``data``.  This allows callers to compute
    power and cost contributions for each pump type individually instead of
    relying on an averaged efficiency across all running pumps.
    """

    if nop <= 0:
        return []

    combo = (
        stn.get("active_combo")
        or stn.get("combo")
        or stn.get("pump_combo")
    )
    ptypes = stn.get("pump_types")
    results: list[dict] = []
    if combo and ptypes:
        for ptype, count in combo.items():
            if count <= 0:
                continue
            pdata = ptypes.get(ptype, {})
            dol = pdata.get("DOL", rpm)
            Q_equiv = flow_m3h * dol / rpm if rpm > 0 else flow_m3h
            A = pdata.get("A", 0.0)
            B = pdata.get("B", 0.0)
            C = pdata.get("C", 0.0)
            tdh_single = A * Q_equiv ** 2 + B * Q_equiv + C
            tdh_single = max(tdh_single, 0.0)
            tdh_type = tdh_single * (rpm / dol) ** 2 * count
            P = pdata.get("P", 0.0)
            Qc = pdata.get("Q", 0.0)
            R = pdata.get("R", 0.0)
            S = pdata.get("S", 0.0)
            T = pdata.get("T", 0.0)
            eff_single = (
                P * Q_equiv ** 4
                + Qc * Q_equiv ** 3
                + R * Q_equiv ** 2
                + S * Q_equiv
                + T
            )
            eff_single = min(max(eff_single, 0.0), 100.0)
            results.append(
                {
                    "tdh": tdh_type,
                    "eff": eff_single,
                    "count": count,
                    "power_type": pdata.get("power_type", stn.get("power_type")),
                    "data": pdata,
                }
            )
        return results

    # Single pump type
    dol = stn.get("DOL", rpm)
    Q_equiv = flow_m3h * dol / rpm if rpm > 0 else flow_m3h
    A = stn.get("A", 0.0)
    B = stn.get("B", 0.0)
    C = stn.get("C", 0.0)
    tdh_single = A * Q_equiv ** 2 + B * Q_equiv + C
    tdh_single = max(tdh_single, 0.0)
    tdh = tdh_single * (rpm / dol) ** 2 * nop
    P = stn.get("P", 0.0)
    Q = stn.get("Q", 0.0)
    R = stn.get("R", 0.0)
    S = stn.get("S", 0.0)
    T = stn.get("T", 0.0)
    eff = P * Q_equiv ** 4 + Q * Q_equiv ** 3 + R * Q_equiv ** 2 + S * Q_equiv + T
    eff = min(max(eff, 0.0), 100.0)
    results.append(
        {
            "tdh": tdh,
            "eff": eff,
            "count": nop,
            "power_type": stn.get("power_type"),
            "data": stn,
        }
    )
    return results


def _compute_iso_sfc(pdata: dict, rpm: float, pump_bkw_total: float, rated_rpm: float, elevation: float, ambient_temp: float) -> float:
    """Compute SFC (gm/bhp-hr) using ISO 3046 approximation."""
    params = pdata.get('engine_params', {})
    rated_power = params.get('rated_power', 0.0)
    sfc50 = params.get('sfc50', 0.0)
    sfc75 = params.get('sfc75', 0.0)
    sfc100 = params.get('sfc100', 0.0)
    # Step 1: engine shaft power (kW)
    engine_kw = pump_bkw_total / 0.98 if pump_bkw_total > 0 else 0.0
    # Step 2: engine power based on operating speed
    engine_power = rated_power * (rpm / rated_rpm) if rated_rpm > 0 else 0.0
    # Step 3: determine ISO 3046 power adjustment factor (formula ref. D)
    T_ref = 298.15  # 25 °C in kelvin
    T_K = ambient_temp + 273.15
    m = 0.7
    n = 1.2
    alpha = (T_ref / T_K) ** m * np.exp(-n * elevation / 1000.0)
    engine_derated_power = engine_power * alpha
    # Step 4: load ratio
    load = engine_kw / engine_derated_power if engine_derated_power > 0 else 0.0
    load_perc = load * 100.0
    # Interpolate test bed SFC at current load
    if load_perc <= 50:
        sfc_tb = sfc50
    elif load_perc <= 75:
        sfc_tb = sfc50 + (sfc75 - sfc50) * (load_perc - 50) / 25.0
    elif load_perc <= 100:
        sfc_tb = sfc75 + (sfc100 - sfc75) * (load_perc - 75) / 25.0
    else:
        sfc_tb = sfc100
    # ISO 3046 fuel consumption adjustment factor (β) ~ 1/α for ref. D
    beta = 1.0 / alpha if alpha > 0 else 1.0
    sfc_site = sfc_tb * beta
    return sfc_site


# ---------------------------------------------------------------------------
# Downstream requirements
# ---------------------------------------------------------------------------

def _downstream_requirement(
    stations: list[dict],
    idx: int,
    terminal: dict,
    segment_flows: list[float] | None,
    KV_list: list[float],
    flow_override: float | list[float] | None = None,
    loop_usage_by_station: list[int] | None = None,
    pump_flow_overrides: dict[int, float] | None = None,
) -> float:
    """Return minimum residual head needed immediately after station ``idx``.

    The previous implementation only accumulated losses across consecutive
    non-pump stations.  When multiple pump stations appear in sequence (e.g. to
    represent different pump types at an origin), upstream pumps were unaware of
    the downstream pressure requirement and the solver could deem a feasible
    configuration infeasible.  This version performs a backward recursion over
    *all* downstream stations, subtracting the maximum head each pump can
    deliver and adding line/elevation losses for every segment.

    ``segment_flows`` may supply the flow rate after each station.  When
    ``flow_override`` is given it takes precedence and may be either a constant
    flow value or a full per-segment list.  ``loop_usage_by_station`` can be used
    to indicate whether each looped segment is active (0 means the loop is
    disabled).  ``pump_flow_overrides`` optionally maps station indices to flow
    rates used when computing pump head; this is useful when a downstream pump
    is bypassed and only the mainline flow enters the pumps.  The returned value
    is therefore the minimum residual needed after station ``idx`` so that the
    terminal residual head constraint can still be met.
    """

    from functools import lru_cache

    N = len(stations)
    if flow_override is not None:
        if isinstance(flow_override, list):
            flows = flow_override
        else:
            flows = [flow_override] * (N + 1)
    else:
        if segment_flows is None:
            raise ValueError("segment_flows or flow_override must be provided")
        flows = segment_flows

    @lru_cache(None)
    def req_entry(i: int) -> float:
        if i >= N:
            return terminal.get('min_residual', 0.0)
        stn = stations[i]
        kv = KV_list[i]
        # ``flows`` holds the flow rate *after* each station; use the downstream
        # value so losses reflect the correct segment flow between station ``i``
        # and ``i+1``.
        flow = flows[i + 1]
        L = stn.get('L', 0.0)
        t = stn.get('t', 0.007)
        # ``D`` may be present but ``None`` in pump-type expansions.  Treat
        # a ``None`` value as absent and fall back to the ``d`` key.  Without
        # this check subtraction would error.
        if stn.get('D') is not None:
            d_inner = stn['D'] - 2 * t
        else:
            d_inner = stn.get('d', 0.7) - 2 * t
        rough = stn.get('rough', 0.00004)
        dra_down = stn.get('max_dr', 0.0)

        head_loss, *_ = _segment_hydraulics(flow, L, d_inner, rough, kv, dra_down, None)
        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N - 1 else stations[i + 1].get('elev', 0.0)
        downstream = req_entry(i + 1)
        req = downstream + head_loss + (elev_next - elev_i)

        # Check intermediate peaks on both mainline and loopline.  Each peak
        # requires sufficient upstream pressure to maintain at least 25 m of
        # residual head at the peak itself.  Consider whichever peak demands the
        # highest pressure.
        # Helper to compute the residual head requirement at intermediate peaks.
        # ``flow_rate`` is the volumetric flow rate (m³/h) used to compute friction to the peak.
        def peak_requirement(flow_rate: float, peaks, d_pipe: float, rough_pipe: float, dra_perc: float) -> float:
            req_local = 0.0
            for peak in peaks or []:
                # Peak location can be stored under various keys
                dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
                elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
                if dist is None or elev_peak is None:
                    continue
                head_peak, *_ = _segment_hydraulics(flow_rate, float(dist), d_pipe, rough_pipe, kv, dra_perc)
                req_p = head_peak + (float(elev_peak) - elev_i) + 25.0
                if req_p > req_local:
                    req_local = req_p
            return req_local

        # Compute peak requirement on the mainline using downstream flow ``flow``.
        peak_req_main = peak_requirement(flow, stn.get('peaks'), d_inner, rough, dra_down)
        peak_req = peak_req_main
        # Compute peak requirement on the loopline.  When the loop carries flow beyond this station
        # (e.g. under bypass), we conservatively use the upstream flow ``flows[i]`` to estimate
        # friction to the peak.  This avoids underestimating the head needed at peaks on the 18" line.
        loop = stn.get('loopline')
        usage = loop_usage_by_station[i] if loop_usage_by_station and i < len(loop_usage_by_station) else None
        loop_flow = flows[i] if usage != 0 else 0.0
        if loop and usage != 0:
            if loop.get('D') is not None:
                t_loop = loop.get('t', t)
                d_inner_loop = loop['D'] - 2 * t_loop
            else:
                d_inner_loop = loop.get('d', d_inner)
            rough_loop = loop.get('rough', rough)
            dra_loop = loop.get('max_dr', 0.0)
            # Use the upstream flow ``flows[i]`` for loop peaks to account for bypassed flow.
            peak_req_loop = peak_requirement(loop_flow, loop.get('peaks'), d_inner_loop, rough_loop, dra_loop)
            peak_req = max(peak_req_main, peak_req_loop)
        req = max(req, peak_req)

        if stn.get('is_pump', False):
            rpm_max = int(stn.get('DOL', stn.get('MinRPM', 0)))
            nop_max = stn.get('max_pumps', 0)
            flow_pump = pump_flow_overrides.get(i, flow) if pump_flow_overrides else flow
            if rpm_max and nop_max:
                pump_info = _pump_head(stn, flow_pump, rpm_max, nop_max)
                tdh_max = sum(max(p['tdh'], 0.0) for p in pump_info)
            else:
                tdh_max = 0.0
            req -= tdh_max
        return max(req, stn.get('min_residual', 0.0))

    return req_entry(idx + 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_pipeline(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    Fuel_density: float,
    Ambient_temp: float,
    linefill: list[dict] | dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
    *,
    loop_usage_by_station: list[int] | None = None,
    enumerate_loops: bool = True,
    # --- Adaptive search parameters ---
    rpm_step: int = RPM_STEP,
    dra_step: int = DRA_STEP,
    per_station_ranges: dict[int, dict] | None = None,
    beam_cap: int | None = None,
    residual_round_override: int | None = None,
) -> dict:
    """Enumerate feasible options across all stations to find the lowest-cost
    operating strategy.

    ``linefill`` describes the current batches in the pipeline as a sequence of
    dictionaries with at least ``volume`` (m³) and ``dra_ppm`` keys.  The
    leading batch's concentration is used as the upstream DRA level for the
    first station.  The function returns the updated linefill after pumping
    under the key ``"linefill"``.

    This function supports optional loop-use directives.  When
    ``enumerate_loops`` is ``True`` and no explicit
    ``loop_usage_by_station`` is provided the solver will automatically
    build a small set of loop-use patterns (e.g. Cases A–E) and run the
    optimisation for each.  The best result is returned.  When
    ``loop_usage_by_station`` is supplied the solver restricts which
    loop scenarios are considered at each station: 0=disabled, 1=parallel,
    2=bypass.  By default the function behaves like the original
    implementation with internal loop enumeration.
    """

    # When requested, perform an outer enumeration over loop usage patterns.
    # We only enter this branch when no explicit per-station loop usage is
    # specified.  Each candidate pattern is mapped onto the stations with
    # looplines, then the solver is invoked recursively with
    # ``enumerate_loops=False`` so that this block isn't re-entered.  The
    # best feasible result across all cases is returned.
    if enumerate_loops and loop_usage_by_station is None:
        # Identify the indices of stations with defined looplines
        loop_positions = [idx for idx, stn in enumerate(stations) if stn.get('loopline')]
        num_loops = len(loop_positions)
        # If there are no looped segments simply call solve_pipeline once
        if num_loops == 0:
            return solve_pipeline(
                stations,
                terminal,
                FLOW,
                KV_list,
                rho_list,
                RateDRA,
                Price_HSD,
                Fuel_density,
                Ambient_temp,
                linefill,
                dra_reach_km,
                mop_kgcm2,
                hours,
                loop_usage_by_station=[],
                enumerate_loops=False,
                rpm_step=rpm_step,
                dra_step=dra_step,
                per_station_ranges=per_station_ranges,
                beam_cap=beam_cap,
                residual_round_override=residual_round_override,
            )
        # Determine per-loop diameter equality flags.  For each looped
        # segment compute whether the inner diameters of the mainline and
        # loopline match within a small tolerance.  This allows the
        # optimiser to apply Case‑1 logic on loops with equal pipes and
        # Case‑2 logic on those with differing pipes independently.
        default_t_local = 0.007
        flags: list[bool] = []
        for idx in loop_positions:
            stn = stations[idx]
            # Inner diameter of mainline
            if stn.get('D') is not None:
                d_main_outer = stn['D']
                t_main = stn.get('t', default_t_local)
                d_inner_main = d_main_outer - 2 * t_main
            else:
                # When only an inner diameter is given treat it as inner
                d_inner_main = stn.get('d', 0.0)
            loop = stn.get('loopline') or {}
            if loop:
                if loop.get('D') is not None:
                    d_loop_outer = loop['D']
                    t_loop = loop.get('t', stn.get('t', default_t_local))
                    d_inner_loop = d_loop_outer - 2 * t_loop
                else:
                    d_inner_loop = loop.get('d', d_inner_main)
            else:
                # Should not happen as only stations with loopline are in loop_positions
                d_inner_loop = d_inner_main
            flags.append(abs(d_inner_main - d_inner_loop) <= 1e-6)
        # Generate loop-usage patterns based on per-loop diameter equality
        cases = _generate_loop_cases_by_flags(flags)
        best_res: dict | None = None
        for case in cases:
            usage = [0] * len(stations)
            for pos, val in zip(loop_positions, case):
                usage[pos] = val
            res = solve_pipeline(
                stations,
                terminal,
                FLOW,
                KV_list,
                rho_list,
                RateDRA,
                Price_HSD,
                Fuel_density,
                Ambient_temp,
                linefill,
                dra_reach_km,
                mop_kgcm2,
                hours,
                loop_usage_by_station=usage,
                enumerate_loops=False,
                rpm_step=rpm_step,
                dra_step=dra_step,
                per_station_ranges=per_station_ranges,
                beam_cap=beam_cap,
                residual_round_override=residual_round_override,
            )
            if res.get('error'):
                continue
            if best_res is None or res.get('total_cost', float('inf')) < best_res.get('total_cost', float('inf')):
                # Track which loop usage produced the best result.  Store a
                # copy to avoid mutating the result of nested calls.  Users
                # can inspect this field to derive human‑friendly names.
                res_with_usage = res.copy()
                res_with_usage['loop_usage'] = usage.copy()
                best_res = res_with_usage
        return best_res or {
            'error': True,
            'message': 'No feasible pump combination found for stations.',
        }
    # Normalise linefill input into a list of batches each carrying volume and
    # DRA concentration.  Accepts either a list of dictionaries or a dict of
    # columns as produced by ``DataFrame.to_dict()``.  The linefill is copied
    # so callers are not mutated.
    linefill_state: list[dict] = []
    if linefill:
        if isinstance(linefill, dict):
            vols = linefill.get('volume') or linefill.get('Volume (m³)') or linefill.get('Volume')
            ppms = linefill.get('dra_ppm') or linefill.get('DRA ppm') or {}
            if vols is not None:
                items = vols.items() if isinstance(vols, dict) else enumerate(vols)
                for idx, v in items:
                    try:
                        vol = float(v)
                    except Exception:
                        continue
                    if vol <= 0:
                        continue
                    if isinstance(ppms, dict):
                        ppm_val = ppms.get(idx, 0.0)
                    elif isinstance(ppms, list):
                        ppm_val = ppms[idx] if idx < len(ppms) else 0.0
                    else:
                        ppm_val = 0.0
                    try:
                        ppm = float(ppm_val)
                    except Exception:
                        ppm = 0.0
                    linefill_state.append({'volume': vol, 'dra_ppm': ppm})
        elif isinstance(linefill, list):
            for ent in linefill:
                try:
                    vol = float(ent.get('volume') or ent.get('Volume (m³)') or ent.get('Volume') or 0.0)
                except Exception:
                    continue
                if vol <= 0:
                    continue
                try:
                    ppm = float(ent.get('dra_ppm') or ent.get('DRA ppm') or 0.0)
                except Exception:
                    ppm = 0.0
                linefill_state.append({'volume': vol, 'dra_ppm': ppm})
    linefill_state = copy.deepcopy(linefill_state)

    N = len(stations)

    # -----------------------------------------------------------------------
    # Sanitize viscosity (KV_list) and density (rho_list) inputs
    #
    # In some scenarios the caller may provide ``KV_list`` or ``rho_list``
    # entries that are zero or ``None``.  A zero viscosity would result in a
    # division by zero when computing Reynolds numbers and friction factors, and
    # a zero density will preclude converting heads to pressure or computing
    # hydraulic power.  Such values frequently arise when the upstream UI has
    # no linefill information and defaults all entries to zero.  To ensure the
    # optimisation can progress we substitute conservative defaults when
    # encountering these values.  The defaults represent a moderately light
    # refined product at 25 °C: 1.0 cSt (~1×10⁻⁶ m²/s) for viscosity and
    # 850 kg/m³ for density.  Negative values are also treated as invalid.
    #
    KV_list = [float(kv) if (kv is not None and kv > 0) else 1.0 for kv in KV_list]
    rho_list = [float(rho) if (rho is not None and rho > 0) else 850.0 for rho in rho_list]
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        segment_flows.append(prev_flow - delivery + supply)

    default_t = 0.007
    default_e = 0.00004

    # Pre-compute static data for each station; head losses depend on DRA reach
    station_opts = []
    origin_enforced = False
    cum_dist = 0.0
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        flow = segment_flows[i]
        kv = KV_list[i - 1]
        rho = rho_list[i - 1]

        L = stn.get('L', 0.0)
        # Determine pipe dimensions.  Some pump-type expansions may carry a
        # ``D`` key with value ``None``.  Treat a ``None`` diameter as
        # unspecified and fall back to using ``d`` instead.  Likewise,
        # thickness defaults when not provided.  Without this check,
        # ``stn['D']`` could be ``None`` and arithmetic would raise an error.
        if stn.get('D') is not None:
            thickness = stn.get('t', default_t)
            # ``outer_d`` may be ``None`` if ``D`` exists but is explicitly
            # null.  Guard against this by falling back to the internal
            # diameter ``d`` if provided, otherwise the default 0.7 m.
            outer_d = stn['D'] if stn['D'] is not None else stn.get('d', 0.7)
            d_inner = outer_d - 2 * thickness
        else:
            # When ``D`` is absent or ``None`` fall back to ``d``
            d_inner = stn.get('d', 0.7)
            outer_d = d_inner
            thickness = stn.get('t', default_t)
        rough = stn.get('rough', default_e)

        # Use a default SMYS when the station provides ``None`` or omits the
        # parameter entirely.  A value of ``None`` would propagate and
        # break downstream multiplication.
        SMYS = stn.get('SMYS', 52000.0) or 52000.0
        design_factor = 0.72
        maop_psi = 2 * SMYS * design_factor * (thickness / outer_d) if outer_d > 0 else 0.0
        maop_kgcm2 = maop_psi * 0.0703069
        if mop_kgcm2 is not None:
            maop_kgcm2 = min(maop_kgcm2, float(mop_kgcm2))
        maop_head = maop_kgcm2 * 10000.0 / rho if rho > 0 else 0.0

        loop_info = stn.get('loopline')
        loop_dict = None
        if loop_info:
            L_loop = loop_info.get('L', L)
            if 'D' in loop_info:
                t_loop = loop_info.get('t', default_t)
                d_inner_loop = loop_info['D'] - 2 * t_loop
                outer_loop = loop_info['D']
            else:
                d_inner_loop = loop_info.get('d', d_inner)
                outer_loop = loop_info.get('d', outer_d)
                t_loop = loop_info.get('t', default_t)
            rough_loop = loop_info.get('rough', default_e)
            SMYS_loop = loop_info.get('SMYS', SMYS)
            maop_psi_loop = 2 * SMYS_loop * design_factor * (t_loop / outer_loop) if outer_loop > 0 else 0.0
            maop_kg_loop = maop_psi_loop * 0.0703069
            if mop_kgcm2 is not None:
                maop_kg_loop = min(maop_kg_loop, float(mop_kgcm2))
            maop_head_loop = maop_kg_loop * 10000.0 / rho if rho > 0 else 0.0
            loop_dict = {
                'name': loop_info.get('name', ''),
                'L': L_loop,
                'd_inner': d_inner_loop,
                'rough': rough_loop,
                'max_dr': loop_info.get('max_dr', 0.0),
                'maop_head': maop_head_loop,
                'maop_kgcm2': maop_kg_loop,
            }

        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N else stations[i].get('elev', 0.0)
        elev_delta = elev_next - elev_i

        opts = []
        flow_m3s = flow / 3600.0
        area = pi * d_inner ** 2 / 4.0
        if stn.get('is_pump', False):
            min_p = stn.get('min_pumps', 0)
            if not origin_enforced:
                min_p = max(1, min_p)
                origin_enforced = True
            max_p = stn.get('max_pumps', 2)
            # Determine RPM values either from user-provided per-station ranges or from the global step.
            if per_station_ranges and (i - 1) in per_station_ranges and per_station_ranges[i - 1].get('rpm'):
                rpm_vals = sorted(set(int(x) for x in per_station_ranges[i - 1]['rpm']))
            else:
                rpm_vals = _allowed_values(int(stn.get('MinRPM', 0)), int(stn.get('DOL', 0)), rpm_step)
            fixed_dr = stn.get('fixed_dra_perc', None)
            max_dr_main = int(stn.get('max_dr', 0))
            if fixed_dr is not None:
                dra_main_vals = [int(round(fixed_dr))]
            else:
                # Use per-station mainline DR range or global step
                if per_station_ranges and (i - 1) in per_station_ranges and per_station_ranges[i - 1].get('dra_main'):
                    dra_main_vals = sorted(set(int(x) for x in per_station_ranges[i - 1]['dra_main']))
                else:
                    dra_main_vals = _allowed_values(0, max_dr_main, dra_step)
            # Use per-station loopline DR range or global step
            if loop_dict:
                if per_station_ranges and (i - 1) in per_station_ranges and per_station_ranges[i - 1].get('dra_loop'):
                    dra_loop_vals = sorted(set(int(x) for x in per_station_ranges[i - 1]['dra_loop']))
                else:
                    dra_loop_vals = _allowed_values(0, int(loop_dict.get('max_dr', 0)), dra_step)
            else:
                dra_loop_vals = [0]
            for nop in range(min_p, max_p + 1):
                rpm_opts = [0] if nop == 0 else rpm_vals
                for rpm in rpm_opts:
                    for dra_main in dra_main_vals:
                        for dra_loop in dra_loop_vals:
                            ppm_main = get_ppm_for_dr(kv, dra_main) if dra_main > 0 else 0.0
                            ppm_loop = get_ppm_for_dr(kv, dra_loop) if dra_loop > 0 else 0.0
                            opts.append({
                                'nop': nop,
                                'rpm': rpm,
                                'dra_main': dra_main,
                                'dra_loop': dra_loop,
                                'dra_ppm_main': ppm_main,
                                'dra_ppm_loop': ppm_loop,
                            })
            if not any(o['nop'] == 0 for o in opts):
                opts.insert(0, {
                    'nop': 0,
                    'rpm': 0,
                    'dra_main': 0,
                    'dra_loop': 0,
                    'dra_ppm_main': 0.0,
                    'dra_ppm_loop': 0.0,
                })
        else:
            # Non-pump stations can inject DRA independently whenever a
            # facility exists (max_dr > 0).  If no injection is available the
            # upstream PPM simply carries forward.
            non_pump_opts: list[dict] = []
            max_dr_main = int(stn.get('max_dr', 0))
            if max_dr_main > 0:
                # Use per-station mainline DR range or global step for non-pump stations
                if per_station_ranges and (i - 1) in per_station_ranges and per_station_ranges[i - 1].get('dra_main'):
                    dra_vals = sorted(set(int(x) for x in per_station_ranges[i - 1]['dra_main']))
                else:
                    dra_vals = _allowed_values(0, max_dr_main, dra_step)
                for dra_main in dra_vals:
                    ppm_main = get_ppm_for_dr(kv, dra_main) if dra_main > 0 else 0.0
                    non_pump_opts.append({
                        'nop': 0,
                        'rpm': 0,
                        'dra_main': dra_main,
                        'dra_loop': 0,
                        'dra_ppm_main': ppm_main,
                        'dra_ppm_loop': 0.0,
                    })
            if not non_pump_opts:
                non_pump_opts.append({
                    'nop': 0,
                    'rpm': 0,
                    'dra_main': 0,
                    'dra_loop': 0,
                    'dra_ppm_main': 0.0,
                    'dra_ppm_loop': 0.0,
                })
            opts.extend(non_pump_opts)

        station_opts.append({
            'name': name,
            'orig_name': stn['name'],
            'idx': i - 1,
            'kv': kv,
            'rho': rho,
            'L': L,
            'd_inner': d_inner,
            'rough': rough,
            'cum_dist': cum_dist,
            'elev_delta': elev_delta,
            'maop_head': maop_head,
            'maop_kgcm2': maop_kgcm2,
            'loopline': loop_dict,
            'options': opts,
            'is_pump': stn.get('is_pump', False),
            'pump_combo': stn.get('pump_combo'),
            'pump_types': stn.get('pump_types'),
            'active_combo': stn.get('active_combo'),
            'coef_A': float(stn.get('A', 0.0)),
            'coef_B': float(stn.get('B', 0.0)),
            'coef_C': float(stn.get('C', 0.0)),
            'coef_P': float(stn.get('P', 0.0)),
            'coef_Q': float(stn.get('Q', 0.0)),
            'coef_R': float(stn.get('R', 0.0)),
            'coef_S': float(stn.get('S', 0.0)),
            'coef_T': float(stn.get('T', 0.0)),
            'min_rpm': int(stn.get('MinRPM', 0)),
            'dol': int(stn.get('DOL', 0)),
            'power_type': stn.get('power_type', 'Grid'),
            'rate': float(stn.get('rate', 0.0)),
            'sfc': float(stn.get('sfc', 0.0)),
            'sfc_mode': stn.get('sfc_mode', 'manual'),
            'engine_params': stn.get('engine_params', {}),
            'elev': float(stn.get('elev', 0.0)),
        })
        cum_dist += L
    # -----------------------------------------------------------------------
    # Dynamic programming over stations

    init_residual = stations[0].get('min_residual', 50.0)
    # Initial dynamic‑programming state.  Each state carries the cumulative
    # operating cost, the residual head after the current station, the full
    # sequence of record dictionaries (one per station), the last MAOP
    # limits, the current flow into the next segment and, importantly, a
    # ``carry_loop_dra`` field.  ``carry_loop_dra`` represents the drag
    # reduction percentage that remains effective on the loopline due to
    # upstream injection when a bypass scenario occurs.  At the origin
    # there is no upstream DRA on the loopline so this value starts at zero.
    #
    # Note: The earlier implementation tracked the advancement of DRA along the
    # pipeline via a ``reach`` parameter.  This has been removed because the
    # entire pipeline is assumed to be treated at the start of the day.  As
    # such, DRA either applies across an entire segment when injected or is
    # carried over unchanged on bypassed loops, but it does not advance
    # progressively.
    # Track the mainline injection PPM from the previous station so that
    # downstream segments inherit the same concentration.
    states: dict[float, dict] = {
        round(init_residual, 2): {
            'cost': 0.0,
            'residual': init_residual,
            'records': [],
            'last_maop': 0.0,
            'last_maop_kg': 0.0,
            'flow': segment_flows[0],
            'carry_loop_dra': 0.0,
            'prev_ppm_main': linefill_state[0]['dra_ppm'] if linefill_state else 0.0,
        }
    }

    for stn_data in station_opts:
        new_states: dict[float, dict] = {}
        for state in states.values():
            flow_total = state.get('flow', segment_flows[0])
            for opt in stn_data['options']:
                # -----------------------------------------------------------------
                # Enforce bypass rules on loopline injection:
                # if the previous station operated in bypass mode (Case‑G)
                # then no loopline DRA injection is permitted at this
                # station (dra_loop must be zero).  The upstream carry‑over
                # drag reduction is used instead.  We detect bypass using
                # ``loop_usage_by_station`` when provided.
                if stn_data['idx'] > 0 and loop_usage_by_station is not None:
                    usage_prev = loop_usage_by_station[stn_data['idx'] - 1]
                    if usage_prev == 2 and opt.get('dra_loop') not in (0, None):
                        continue
                # Determine the downstream PPM on the mainline.  When pumps run,
                # any upstream DRA is discarded and the selected injection sets the
                # new concentration.  If pumps are bypassed, a non-zero injection
                # overwrites the upstream PPM; otherwise the previous concentration
                # is carried forward.
                if stn_data['is_pump'] and opt['nop'] > 0:
                    ppm_main = opt['dra_ppm_main']
                else:
                    if opt['dra_ppm_main'] > 0:
                        ppm_main = opt['dra_ppm_main']
                    else:
                        ppm_main = state.get('prev_ppm_main', 0.0)
                inj_ppm_main = opt['dra_ppm_main'] if opt['dra_ppm_main'] > 0 else 0.0
                # Convert the PPM into an effective drag‑reduction percentage for
                # hydraulic calculations.
                eff_dra_main = get_dr_for_ppm(stn_data['kv'], ppm_main) if ppm_main > 0 else 0.0
                dra_len_main = stn_data['L'] if eff_dra_main > 0 else 0.0

                scenarios = []
                # Base scenario: flow through mainline only
                hl_single, v_single, Re_single, f_single = _segment_hydraulics(
                    flow_total,
                    stn_data['L'],
                    stn_data['d_inner'],
                    stn_data['rough'],
                    stn_data['kv'],
                    eff_dra_main,
                    dra_len_main,
                )
                scenarios.append({
                    'head_loss': hl_single,
                    'v': v_single,
                    'Re': Re_single,
                    'f': f_single,
                    'flow_main': flow_total,
                    'v_loop': 0.0,
                    'Re_loop': 0.0,
                    'f_loop': 0.0,
                    'flow_loop': 0.0,
                    'maop_loop': 0.0,
                    'maop_loop_kg': 0.0,
                    'bypass_next': False,
                })
                if stn_data.get('loopline'):
                    loop = stn_data['loopline']
                    # Drag reduction on loopline applies across the entire loop
                    eff_dra_loop = opt['dra_loop']
                    dra_len_loop = loop['L'] if eff_dra_loop > 0 else 0.0
                    # Parallel scenario (main + loop split by equal head)
                    hl_par, main_stats, loop_stats = _parallel_segment_hydraulics(
                        flow_total,
                        {
                            'L': stn_data['L'],
                            'd_inner': stn_data['d_inner'],
                            'rough': stn_data['rough'],
                            'dra': eff_dra_main,
                            'dra_len': dra_len_main,
                        },
                        {
                            'L': loop['L'],
                            'd_inner': loop['d_inner'],
                            'rough': loop['rough'],
                            'dra': eff_dra_loop,
                            'dra_len': dra_len_loop,
                        },
                        stn_data['kv'],
                    )
                    v_m, Re_m, f_m, q_main = main_stats
                    v_l, Re_l, f_l, q_loop = loop_stats
                    # Parallel scenario without bypass
                    scenarios.append({
                        'head_loss': hl_par,
                        'v': v_m,
                        'Re': Re_m,
                        'f': f_m,
                        'flow_main': q_main,
                        'v_loop': v_l,
                        'Re_loop': Re_l,
                        'f_loop': f_l,
                        'flow_loop': q_loop,
                        'maop_loop': loop['maop_head'],
                        'maop_loop_kg': loop['maop_kgcm2'],
                        'bypass_next': False,
                    })
                    # Bypass scenario: same flow split but bypass next pump on loop
                    scenarios.append({
                        'head_loss': hl_par,
                        'v': v_m,
                        'Re': Re_m,
                        'f': f_m,
                        'flow_main': q_main,
                        'v_loop': v_l,
                        'Re_loop': Re_l,
                        'f_loop': f_l,
                        'flow_loop': q_loop,
                        'maop_loop': loop['maop_head'],
                        'maop_loop_kg': loop['maop_kgcm2'],
                        'bypass_next': True,
                    })
                    # Loop‑only scenario: entire flow goes through loopline only.
                    # Only include when diameters differ; otherwise the parallel
                    # scenario already captures equal pipes.
                    if abs(stn_data['d_inner'] - loop['d_inner']) > 1e-6:
                        hl_loop, v_loop_only, Re_loop_only, f_loop_only = _segment_hydraulics(
                            flow_total,
                            loop['L'],
                            loop['d_inner'],
                            loop['rough'],
                            stn_data['kv'],
                            eff_dra_loop,
                            dra_len_loop,
                        )
                        scenarios.append({
                            'head_loss': hl_loop,
                            'v': 0.0,
                            'Re': 0.0,
                            'f': 0.0,
                            'flow_main': 0.0,
                            'v_loop': v_loop_only,
                            'Re_loop': Re_loop_only,
                            'f_loop': f_loop_only,
                            'flow_loop': flow_total,
                            'maop_loop': loop['maop_head'],
                            'maop_loop_kg': loop['maop_kgcm2'],
                            'bypass_next': False,
                        })

                if opt['nop'] > 0 and opt['rpm'] > 0:
                    pump_def = {
                        'A': stn_data['coef_A'],
                        'B': stn_data['coef_B'],
                        'C': stn_data['coef_C'],
                        'P': stn_data['coef_P'],
                        'Q': stn_data['coef_Q'],
                        'R': stn_data['coef_R'],
                        'S': stn_data['coef_S'],
                        'T': stn_data['coef_T'],
                        'DOL': stn_data['dol'],
                        'combo': stn_data.get('pump_combo'),
                        'pump_types': stn_data.get('pump_types'),
                        'active_combo': stn_data.get('active_combo'),
                        'power_type': stn_data.get('power_type'),
                        'sfc_mode': stn_data.get('sfc_mode'),
                        'sfc': stn_data.get('sfc'),
                        'engine_params': stn_data.get('engine_params', {}),
                    }
                    pump_details = _pump_head(pump_def, flow_total, opt['rpm'], opt['nop'])
                    tdh = sum(p['tdh'] for p in pump_details)
                else:
                    pump_details = []
                    tdh = 0.0

                eff_pct = (
                    sum(p['eff'] * p['count'] for p in pump_details) / opt['nop']
                    if pump_details and opt['nop'] > 0
                    else 0.0
                )

                pump_bkw_total = 0.0
                prime_kw_total = 0.0
                power_cost = 0.0
                for pinfo in pump_details:
                    eff_local = max(min(pinfo['eff'], 100.0), 1e-6)
                    tdh_local = max(pinfo['tdh'], 0.0)
                    pump_bkw_i = (
                        stn_data['rho'] * flow_total * 9.81 * tdh_local
                    ) / (3600.0 * 1000.0 * (eff_local / 100.0))
                    pump_bkw_total += pump_bkw_i
                    pdata = pinfo.get('data', {})
                    rated_rpm = pdata.get('DOL', stn_data['dol'])
                    if pinfo.get('power_type') == 'Diesel':
                        mech_eff = 0.98
                    else:
                        mech_eff = 0.95 if opt['rpm'] >= rated_rpm else 0.91
                    prime_kw_i = pump_bkw_i / mech_eff
                    prime_kw_total += prime_kw_i
                    if pinfo.get('power_type') == 'Diesel':
                        mode = pdata.get('sfc_mode', stn_data.get('sfc_mode', 'manual'))
                        if mode == 'manual':
                            sfc_val = pdata.get('sfc', stn_data.get('sfc', 0.0))
                        elif mode == 'iso':
                            sfc_val = _compute_iso_sfc(
                                pdata,
                                opt['rpm'],
                                pump_bkw_i,
                                pdata.get('DOL', stn_data['dol']),
                                stn_data.get('elev', 0.0),
                                Ambient_temp,
                            )
                        else:
                            sfc_val = 0.0
                        fuel_per_kWh = (sfc_val * 1.34102) / Fuel_density if sfc_val else 0.0
                        cost_i = prime_kw_i * hours * fuel_per_kWh * Price_HSD
                    else:
                        cost_i = prime_kw_i * hours * stn_data.get('rate', 0.0)
                    cost_i = max(cost_i, 0.0)
                    pinfo['pump_bkw'] = pump_bkw_i
                    pinfo['prime_kw'] = prime_kw_i
                    pinfo['power_cost'] = cost_i
                    power_cost += cost_i

                pump_bkw = pump_bkw_total
                motor_kw = prime_kw_total

                # Filter candidate scenarios based on explicit loop-usage directives.
                filtered_scenarios = []
                if loop_usage_by_station is not None and stn_data.get('loopline'):
                    usage = loop_usage_by_station[stn_data['idx']]
                    if usage == 0:
                        # Only the base (no-loop) scenario is allowed.  Pick the first
                        # scenario with zero loop flow.
                        for cand in scenarios:
                            if cand['flow_loop'] == 0.0:
                                filtered_scenarios.append(cand)
                                break
                    elif usage == 1:
                        # Use only the parallel scenario: loop flow > 0 and not bypass.
                        for cand in scenarios:
                            if cand['flow_loop'] > 0.0 and not cand.get('bypass_next', False):
                                filtered_scenarios.append(cand)
                                break
                    elif usage == 2:
                        # Use only the bypass scenario: loop flow > 0 and bypass flag set.
                        for cand in scenarios:
                            if cand['flow_loop'] > 0.0 and cand.get('bypass_next', False):
                                filtered_scenarios.append(cand)
                                break
                    elif usage == 3:
                        # Loop-only directive: select scenario where all flow goes through loopline
                        for cand in scenarios:
                            if cand['flow_loop'] > 0.0 and cand['flow_main'] == 0.0:
                                filtered_scenarios.append(cand)
                                break
                    else:
                        # Unrecognised directive: fall back to no-loop scenario.
                        for cand in scenarios:
                            if cand['flow_loop'] == 0.0:
                                filtered_scenarios.append(cand)
                                break
                else:
                    filtered_scenarios = scenarios
                for sc in filtered_scenarios:
                    # Skip scenarios with unacceptable velocities
                    if sc['flow_main'] > 0 and not (V_MIN <= sc['v'] <= V_MAX):
                        continue
                    if sc['flow_loop'] > 0 and not (V_MIN <= sc['v_loop'] <= V_MAX):
                        continue

                    # -----------------------------------------------------------------
                    # Special handling for bypass patterns across an entire pipeline.
                    #
                    # When there are exactly two stations and the first station's
                    # loopline bypasses the pumps at the next station, the loopline
                    # flow travels all the way from the origin to the terminal before
                    # rejoining the mainline.  In such cases the flow split between
                    # the mainline and loopline should be determined by equalising
                    # the total head loss (friction + elevation) from the origin to
                    # the terminal rather than on a per-segment basis.  The default
                    # implementation splits flow only over the current segment, which
                    # underestimates the required head for the loopline when the
                    # downstream segment contains peaks or substantial length.  The
                    # block below recomputes the flow split and corresponding head
                    # loss for the first segment based on the combined length of
                    # successive segments.  It then overwrites the candidate
                    # scenario's flow and velocity fields accordingly.  Only apply
                    # this correction when bypassing the next pump on the very
                    # first station in a two-station pipeline.
                    if (
                        sc.get('bypass_next')
                        and stn_data['idx'] == 0
                        and N == 2
                        and stn_data.get('loopline')
                    ):
                        # Identify the downstream station
                        next_stn = stations[1]
                        # Compute total mainline and loopline path lengths from
                        # the current station to the terminal
                        length_main_total = stn_data['L'] + next_stn['L']
                        # Loopline on the next station may not exist; use zero
                        length_loop_total = (
                            stn_data['loopline']['L'] + next_stn.get('loopline', {}).get('L', 0.0)
                        )
                        # If the downstream station does not define a loopline,
                        # treat the loopline length as only the current segment
                        if length_loop_total <= 0.0:
                            length_loop_total = stn_data['loopline']['L']
                        # Effective drag reduction for the entire path based on the
                        # inherited mainline PPM
                        eff_dra_main_tot = get_dr_for_ppm(stn_data['kv'], ppm_main) if ppm_main > 0 else 0.0
                        # Carry-over drag reduction on the loop from the previous state
                        carry_prev = state.get('carry_loop_dra', 0.0)
                        # In bypass mode the loopline may still inject additional DRA.
                        # Combine any upstream carry-over with the current option so
                        # the full effect is considered when splitting flow over the
                        # total path.
                        eff_dra_loop_tot = (
                            carry_prev + opt['dra_loop']
                            if sc.get('bypass_next')
                            else opt['dra_loop']
                        )
                        # Compute flow split to equalise head loss over the entire path
                        hl_tot, main_stats_tot, loop_stats_tot = _parallel_segment_hydraulics(
                            flow_total,
                            {
                                'L': length_main_total,
                                'd_inner': stn_data['d_inner'],
                                'rough': stn_data['rough'],
                                'dra': eff_dra_main_tot,
                                'dra_len': length_main_total if eff_dra_main_tot > 0 else 0.0,
                            },
                            {
                                'L': length_loop_total,
                                'd_inner': stn_data['loopline']['d_inner'],
                                'rough': stn_data['loopline']['rough'],
                                'dra': eff_dra_loop_tot,
                                'dra_len': length_loop_total if eff_dra_loop_tot > 0 else 0.0,
                            },
                            stn_data['kv'],
                        )
                        v_main_tot, Re_main_tot, f_main_tot, q_main_tot = main_stats_tot
                        v_loop_tot, Re_loop_tot, f_loop_tot, q_loop_tot = loop_stats_tot
                        # Recompute head loss for the first segment using the split
                        # flow on this segment.  Apply the same drag reduction
                        hl_main_seg, v_main_seg, Re_main_seg, f_main_seg = _segment_hydraulics(
                            q_main_tot,
                            stn_data['L'],
                            stn_data['d_inner'],
                            stn_data['rough'],
                            stn_data['kv'],
                            eff_dra_main_tot,
                            stn_data['L'] if eff_dra_main_tot > 0 else 0.0,
                        )
                        # Recompute loopline velocity and friction factor for the
                        # first segment.  The loopline may have different length
                        # than the mainline on this segment.
                        hl_loop_seg, v_loop_seg, Re_loop_seg, f_loop_seg = _segment_hydraulics(
                            q_loop_tot,
                            stn_data['loopline']['L'],
                            stn_data['loopline']['d_inner'],
                            stn_data['loopline']['rough'],
                            stn_data['kv'],
                            eff_dra_loop_tot,
                            stn_data['loopline']['L'] if eff_dra_loop_tot > 0 else 0.0,
                        )
                        # Overwrite the candidate scenario with corrected values
                        sc = sc.copy()
                        sc['flow_main'] = q_main_tot
                        sc['flow_loop'] = q_loop_tot
                        sc['v'] = v_main_seg
                        sc['Re'] = Re_main_seg
                        sc['f'] = f_main_seg
                        sc['v_loop'] = v_loop_seg
                        sc['Re_loop'] = Re_loop_seg
                        sc['f_loop'] = f_loop_seg
                        sc['head_loss'] = hl_main_seg
                        sc['bypass_next'] = True

                    # Determine the effective drag reduction on the loopline.  In bypass
                    # mode (Condition‑G) the DRA injection at this station is not
                    # performed on the loopline; instead the drag reduction from the
                    # upstream station persists.  Otherwise use the station's
                    # prescribed DRA for the loopline.  When there is no loop flow
                    # the value is irrelevant but carried forward.
                    carry_prev = state.get('carry_loop_dra', 0.0)
                    if sc['flow_loop'] > 0:
                        if sc.get('bypass_next'):
                            eff_dra_loop = carry_prev + opt['dra_loop']
                            inj_loop_current = opt['dra_loop']
                            inj_ppm_loop = opt['dra_ppm_loop']
                        else:
                            eff_dra_loop = opt['dra_loop']
                            inj_loop_current = opt['dra_loop']
                            inj_ppm_loop = opt['dra_ppm_loop']
                    else:
                        eff_dra_loop = 0.0
                        inj_loop_current = 0.0
                        inj_ppm_loop = 0.0

                    # Determine next carry-over drag reduction value for the loop.
                    if sc['flow_loop'] > 0:
                        if sc.get('bypass_next'):
                            new_carry = carry_prev + opt['dra_loop']
                        else:
                            new_carry = opt['dra_loop']
                    else:
                        new_carry = carry_prev

                    # Compute the resulting superimposed discharge head after the pump and
                    # check MAOP constraints.  Use the head delivered by the pumps on
                    # this segment.
                    sdh = state['residual'] + tdh
                    if sdh > stn_data['maop_head'] or (
                        sc['flow_loop'] > 0 and sdh > stn_data['loopline']['maop_head']
                    ):
                        continue

                    # Compute downstream residual head after segment loss and elevation
                    residual_next = sdh - sc['head_loss'] - stn_data['elev_delta']

                    # Recompute downstream flows if bypassing the next station; the flow
                    # through the mainline changes only for a bypass.  ``seg_flows_tmp``
                    # holds the flow after each station.
                    seg_flows_tmp = segment_flows.copy()
                    # When bypassing the next station, only the mainline flow enters that station; the
                    # loopline flow bypasses the pumps and rejoins downstream.  Therefore the flow for
                    # downstream segments should reflect either the mainline-only flow (in bypass) or the
                    # total flow (in parallel or loop-only modes).  This ensures head requirements are
                    # computed against the proper volumetric flow in each pipe.
                    next_flow = flow_total
                    seg_flows_tmp[stn_data['idx'] + 1] = next_flow
                    for j in range(stn_data['idx'] + 1, N):
                        delivery_j = float(stations[j].get('delivery', 0.0))
                        supply_j = float(stations[j].get('supply', 0.0))
                        seg_flows_tmp[j + 1] = seg_flows_tmp[j] - delivery_j + supply_j

                    # Compute minimum downstream requirement and skip infeasible states
                    pump_overrides = None
                    if sc.get('bypass_next') and stn_data['idx'] + 1 < N:
                        pump_overrides = {}
                        next_index = stn_data['idx'] + 1
                        next_orig = stations[next_index].get('orig_name') or stations[next_index].get('name')
                        j = next_index
                        while j < N:
                            if stations[j].get('orig_name') == next_orig or (
                                next_orig is None and stations[j].get('orig_name') is None
                            ):
                                pump_overrides[j] = sc['flow_main']
                                j += 1
                            else:
                                break
                    min_req = _downstream_requirement(
                        stations,
                        stn_data['idx'],
                        terminal,
                        seg_flows_tmp,
                        KV_list,
                        loop_usage_by_station=loop_usage_by_station,
                        pump_flow_overrides=pump_overrides,
                    )
                    if residual_next < min_req:
                        continue

                    # Compute DRA costs.  Only charge for injections performed at this
                    # station.  Mainline and loopline injections are handled
                    # separately and loopline cost is incurred only when
                    # an injection is made.
                    dra_cost = 0.0
                    if inj_ppm_main > 0:
                        dra_cost += inj_ppm_main * (sc['flow_main'] * 1000.0 * hours / 1e6) * RateDRA
                    # Loopline injection uses ``inj_ppm_loop`` computed
                    # earlier.  Charge cost only when an actual injection is
                    # performed at this station.
                    if sc['flow_loop'] > 0 and inj_loop_current > 0:
                        dra_cost += inj_ppm_loop * (sc['flow_loop'] * 1000.0 * hours / 1e6) * RateDRA

                    total_cost = power_cost + dra_cost

                    # Build the record for this station.  Update loop velocity and MAOP
                    # information based on the scenario.  Use the effective drag
                    # reduction for loopline in display.  Note: drag_reduction_loop
                    # reflects the value used in this segment (carry over for bypass).
                    record = {
                        f"pipeline_flow_{stn_data['name']}": sc['flow_main'],
                        f"pipeline_flow_in_{stn_data['name']}": flow_total,
                        f"loopline_flow_{stn_data['name']}": sc['flow_loop'],
                        f"head_loss_{stn_data['name']}": sc['head_loss'],
                        f"head_loss_kgcm2_{stn_data['name']}": head_to_kgcm2(sc['head_loss'], stn_data['rho']),
                        f"residual_head_{stn_data['name']}": state['residual'],
                        f"rh_kgcm2_{stn_data['name']}": head_to_kgcm2(state['residual'], stn_data['rho']),
                        f"sdh_{stn_data['name']}": sdh if stn_data['is_pump'] else state['residual'],
                        f"sdh_kgcm2_{stn_data['name']}": head_to_kgcm2(
                            sdh if stn_data['is_pump'] else state['residual'], stn_data['rho']
                        ),
                        f"rho_{stn_data['name']}": stn_data['rho'],
                        f"maop_{stn_data['name']}": stn_data['maop_head'],
                        f"maop_kgcm2_{stn_data['name']}": stn_data['maop_kgcm2'],
                        f"velocity_{stn_data['name']}": sc['v'],
                        f"reynolds_{stn_data['name']}": sc['Re'],
                        f"friction_{stn_data['name']}": sc['f'],
                        f"coef_A_{stn_data['name']}": stn_data['coef_A'],
                        f"coef_B_{stn_data['name']}": stn_data['coef_B'],
                        f"coef_C_{stn_data['name']}": stn_data['coef_C'],
                        f"coef_P_{stn_data['name']}": stn_data['coef_P'],
                        f"coef_Q_{stn_data['name']}": stn_data['coef_Q'],
                        f"coef_R_{stn_data['name']}": stn_data['coef_R'],
                        f"coef_S_{stn_data['name']}": stn_data['coef_S'],
                        f"coef_T_{stn_data['name']}": stn_data['coef_T'],
                        f"min_rpm_{stn_data['name']}": stn_data['min_rpm'],
                        f"dol_{stn_data['name']}": stn_data['dol'],
                    }
                    if sc['flow_loop'] > 0:
                        record.update({
                            f"velocity_loop_{stn_data['name']}": sc['v_loop'],
                            f"reynolds_loop_{stn_data['name']}": sc['Re_loop'],
                            f"friction_loop_{stn_data['name']}": sc['f_loop'],
                            f"maop_loop_{stn_data['name']}": sc['maop_loop'],
                            f"maop_loop_kgcm2_{stn_data['name']}": sc['maop_loop_kg'],
                        })
                    else:
                        record.update({
                            f"velocity_loop_{stn_data['name']}": 0.0,
                            f"reynolds_loop_{stn_data['name']}": 0.0,
                            f"friction_loop_{stn_data['name']}": 0.0,
                            f"maop_loop_{stn_data['name']}": 0.0,
                            f"maop_loop_kgcm2_{stn_data['name']}": 0.0,
                        })
                    if stn_data['is_pump']:
                        record.update({
                            f"pump_flow_{stn_data['name']}": flow_total,
                            f"num_pumps_{stn_data['name']}": opt['nop'],
                            f"speed_{stn_data['name']}": opt['rpm'],
                            # Store efficiency as a fraction on the raw key and as a percentage on a dedicated key.
                            f"efficiency_{stn_data['name']}": eff_pct / 100.0,
                            f"pump_efficiency_pct_{stn_data['name']}": eff_pct,
                            f"pump_bkw_{stn_data['name']}": pump_bkw,
                            f"motor_kw_{stn_data['name']}": motor_kw,
                            f"power_cost_{stn_data['name']}": power_cost,
                            f"dra_cost_{stn_data['name']}": dra_cost,
                            f"pump_details_{stn_data['name']}": [p.copy() for p in pump_details],
                            f"dra_ppm_{stn_data['name']}": ppm_main,
                            f"dra_ppm_loop_{stn_data['name']}": inj_ppm_loop,
                            f"drag_reduction_{stn_data['name']}": eff_dra_main,
                            f"drag_reduction_loop_{stn_data['name']}": eff_dra_loop,
                        })
                    else:
                        record.update({
                            f"pump_flow_{stn_data['name']}": 0.0,
                            f"num_pumps_{stn_data['name']}": 0,
                            f"speed_{stn_data['name']}": 0.0,
                            # For non-pump stations there is no efficiency or BKW, but preserve keys for consistency.
                            f"efficiency_{stn_data['name']}": 0.0,
                            f"pump_efficiency_pct_{stn_data['name']}": 0.0,
                            f"pump_bkw_{stn_data['name']}": 0.0,
                            f"motor_kw_{stn_data['name']}": 0.0,
                            f"power_cost_{stn_data['name']}": 0.0,
                            f"dra_cost_{stn_data['name']}": dra_cost,
                            f"pump_details_{stn_data['name']}": [],
                            f"dra_ppm_{stn_data['name']}": ppm_main,
                            f"dra_ppm_loop_{stn_data['name']}": inj_ppm_loop,
                            f"drag_reduction_{stn_data['name']}": eff_dra_main,
                            f"drag_reduction_loop_{stn_data['name']}": eff_dra_loop,
                        })
                    # Accumulate cost and update dynamic state.  When comparing states
                    # with the same residual bucket, prefer the one with lower cost
                    # or, when costs tie, the one with higher residual.  Carry
                    # forward the loop DRA carry value and the updated reach.
                    new_cost = state['cost'] + total_cost
                    bucket = round(residual_next, RESIDUAL_ROUND)
                    record[f"bypass_next_{stn_data['name']}"] = 1 if sc.get('bypass_next', False) else 0
                    new_record_list = state['records'] + [record]
                    existing = new_states.get(bucket)
                    flow_next = flow_total
                    if (
                        existing is None
                        or new_cost < existing['cost']
                        or (
                            abs(new_cost - existing['cost']) < 1e-9
                            and residual_next > existing['residual']
                        )
                    ):
                        new_states[bucket] = {
                            'cost': new_cost,
                            'residual': residual_next,
                            'records': new_record_list,
                            'last_maop': stn_data['maop_head'],
                            'last_maop_kg': stn_data['maop_kgcm2'],
                            # ``reach`` removed because DRA advancement is no longer tracked
                            'flow': flow_next,
                            'carry_loop_dra': new_carry,
                            # Propagate the resulting mainline PPM for downstream stations.
                            'prev_ppm_main': ppm_main,
                        }

        if not new_states:
            return {"error": True, "message": f"No feasible operating point for {stn_data['orig_name']}"}
        # Optionally apply a beam cap to bound the number of states retained.
        # When ``beam_cap`` is provided the new states are sorted by cost and
        # closeness to the terminal residual head and trimmed to the top
        # ``beam_cap`` entries.  This helps limit the combinatorial
        # explosion when many operating points exist.
        if beam_cap is not None and beam_cap > 0 and len(new_states) > beam_cap:
            term_req = terminal.get('min_residual', 0.0)
            items = sorted(
                new_states.items(),
                key=lambda kv: (kv[1]['cost'], abs(kv[1]['residual'] - term_req))
            )
            new_states = dict(items[:beam_cap])
        # Assign the (possibly trimmed) new states for the next iteration.  Previously we
        # pruned aggressively by residual and cost, which could discard viable
        # solutions.  Retaining all or capped states helps ensure that marginally
        # more expensive configurations remain available for later stations.
        states = new_states

    # Pick lowest-cost end state and, among equal-cost candidates,
    # prefer the one whose terminal residual head is closest to the
    # user-specified minimum.  This avoids unnecessarily high
    # pressures at the terminal which would otherwise waste energy.
    term_req = terminal.get('min_residual', 0.0)
    best_state = min(
        states.values(),
        key=lambda x: (x['cost'], x['residual'] - term_req),
    )
    result: dict = {}
    for rec in best_state['records']:
        result.update(rec)

    residual = best_state['residual']
    total_cost = best_state['cost']
    last_maop_head = best_state['last_maop']
    last_maop_kg = best_state['last_maop_kg']

    # Advance the linefill based on the selected origin injection.  The pumped
    # volume is the flow entering the first segment multiplied by the run
    # duration.  A single batch carrying the chosen ``dra_ppm`` is injected at
    # the origin.
    pumped_volume = segment_flows[0] * hours
    origin_name = stations[0]['name'].strip().lower().replace(' ', '_') if stations else ''
    inj_ppm = result.get(f"dra_ppm_{origin_name}", 0.0) if origin_name else 0.0
    schedule = [{'volume': pumped_volume, 'dra_ppm': inj_ppm}]
    advance_linefill(linefill_state, schedule, pumped_volume)
    result['linefill'] = linefill_state

    term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": segment_flows[-1],
        f"pipeline_flow_in_{term_name}": segment_flows[-2],
        f"pump_flow_{term_name}": 0.0,
        f"speed_{term_name}": 0.0,
        f"num_pumps_{term_name}": 0,
        # Terminal has no pumps; set efficiency and BKW fields to zero.
        f"efficiency_{term_name}": 0.0,
        f"pump_efficiency_pct_{term_name}": 0.0,
        f"pump_bkw_{term_name}": 0.0,
        f"motor_kw_{term_name}": 0.0,
        f"power_cost_{term_name}": 0.0,
        f"dra_cost_{term_name}": 0.0,
        f"dra_ppm_{term_name}": 0.0,
        f"dra_ppm_loop_{term_name}": 0.0,
        f"drag_reduction_{term_name}": 0.0,
        f"drag_reduction_loop_{term_name}": 0.0,
        f"head_loss_{term_name}": 0.0,
        f"velocity_{term_name}": 0.0,
        f"reynolds_{term_name}": 0.0,
        f"friction_{term_name}": 0.0,
        f"sdh_{term_name}": 0.0,
        f"residual_head_{term_name}": residual,
    })
    rho_term = rho_list[-1]
    result[f"rh_kgcm2_{term_name}"] = head_to_kgcm2(residual, rho_term)
    result[f"sdh_kgcm2_{term_name}"] = 0.0
    result[f"rho_{term_name}"] = rho_term
    result[f"maop_{term_name}"] = last_maop_head
    result[f"maop_kgcm2_{term_name}"] = last_maop_kg
    result['total_cost'] = total_cost
    # Propagate the run duration so callers can scale costs consistently.
    # ``solve_pipeline`` may be invoked for arbitrary intervals (e.g. <4 hours)
    # and downstream consumers need the actual duration to compute
    # per‑interval costs.  Record it on the result before returning.
    result['hours'] = hours
    result['dra_front_km'] = best_state.get('reach', 0.0)
    result['error'] = False
    return result


def solve_pipeline_with_types(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    Fuel_density: float,
    Ambient_temp: float,
    linefill: list[dict] | dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
    **solver_kwargs,
) -> dict:
    """Enumerate pump type combinations at all stations and call ``solve_pipeline``."""

    best_result = None
    best_cost = float('inf')
    best_stations = None
    N = len(stations)

    def expand_all(pos: int, stn_acc: list[dict], kv_acc: list[float], rho_acc: list[float]):
        nonlocal best_result, best_cost, best_stations
        if pos >= N:
            # When all stations have been expanded into individual pump units,
            # perform loop-case enumeration explicitly.  We determine the
            # positions of units with looplines (typically the last unit of each
            # physical station) and then build loop usage directives for each
            # representative case.  This avoids relying on the internal
            # loop-enumeration of ``solve_pipeline``, which can behave
            # unpredictably when stations are split into multiple units.
            loop_positions = [idx for idx, u in enumerate(stn_acc) if u.get('loopline')]
            # Always run at least once even if no loops exist
            if not loop_positions:
                cases = [[]]
            else:
                # Determine per-loop diameter equality flags for the expanded stations.
                default_t_local = 0.007
                flags_expanded: list[bool] = []
                for pidx in loop_positions:
                    stn_e = stn_acc[pidx]
                    # Inner diameter of the mainline segment
                    if stn_e.get('D') is not None:
                        d_main_outer = stn_e['D']
                        t_main = stn_e.get('t', default_t_local)
                        d_inner_main = d_main_outer - 2 * t_main
                    else:
                        d_inner_main = stn_e.get('d', 0.0)
                    lp = stn_e.get('loopline') or {}
                    if lp:
                        if lp.get('D') is not None:
                            d_loop_outer = lp['D']
                            t_loop = lp.get('t', stn_e.get('t', default_t_local))
                            d_inner_loop = d_loop_outer - 2 * t_loop
                        else:
                            d_inner_loop = lp.get('d', d_inner_main)
                    else:
                        d_inner_loop = d_inner_main
                    flags_expanded.append(abs(d_inner_main - d_inner_loop) <= 1e-6)
                # Generate loop-case combinations based on flags
                cases = _generate_loop_cases_by_flags(flags_expanded)
            for case in cases:
                usage = [0] * len(stn_acc)
                for pidx, val in zip(loop_positions, case):
                    usage[pidx] = val
                # Call solve_pipeline with explicit loop usage and disable
                # internal enumeration.  This ensures the provided directives
                # are respected even for split stations.
                # Remove any duplicate loop arguments from solver_kwargs to avoid passing
                # ``loop_usage_by_station`` or ``enumerate_loops`` twice.  The explicit
                # keywords below override whatever defaults may be present in
                # ``solver_kwargs``.
                _clean_kwargs = {k: v for k, v in solver_kwargs.items() if k not in ('loop_usage_by_station', 'enumerate_loops')}
                result = solve_pipeline(
                    stn_acc,
                    terminal,
                    FLOW,
                    kv_acc,
                    rho_acc,
                    RateDRA,
                    Price_HSD,
                    Fuel_density,
                    Ambient_temp,
                    linefill,
                    dra_reach_km,
                    mop_kgcm2,
                    hours,
                    loop_usage_by_station=usage,
                    enumerate_loops=False,
                    **_clean_kwargs,
                )
                if result.get("error"):
                    continue
                cost = result.get("total_cost", float('inf'))
                if cost < best_cost:
                    # Preserve usage directive for later labelling
                    result_with_usage = result.copy()
                    result_with_usage['loop_usage'] = usage.copy()
                    best_cost = cost
                    best_result = result_with_usage
                    best_stations = stn_acc
            return

        stn = stations[pos]
        kv = KV_list[pos]
        rho = rho_list[pos]

        if stn.get('pump_types'):
            # Determine available counts for each type
            availA = stn['pump_types'].get('A', {}).get('available', 0)
            availB = stn['pump_types'].get('B', {}).get('available', 0)
            combos = generate_type_combinations(availA, availB)
            for numA, numB in combos:
                total_units = numA + numB
                if total_units <= 0:
                    continue
                pdataA = stn['pump_types'].get('A', {})
                pdataB = stn['pump_types'].get('B', {})
                for actA in range(numA + 1):
                    for actB in range(numB + 1):
                        if actA + actB <= 0:
                            continue
                        unit = copy.deepcopy(stn)
                        unit['pump_combo'] = {'A': numA, 'B': numB}
                        unit['active_combo'] = {'A': actA, 'B': actB}
                        if actA > 0 and actB == 0:
                            pdata = pdataA
                        elif actB > 0 and actA == 0:
                            pdata = pdataB
                        else:
                            pdata = None
                        if pdata is not None:
                            for coef in ['A', 'B', 'C', 'P', 'Q', 'R', 'S', 'T']:
                                unit[coef] = pdata.get(coef, unit.get(coef, 0.0))
                            unit['MinRPM'] = pdata.get('MinRPM', unit.get('MinRPM', 0.0))
                            unit['DOL'] = pdata.get('DOL', unit.get('DOL', 0.0))
                            unit['power_type'] = pdata.get('power_type', unit.get('power_type', 'Grid'))
                            unit['rate'] = pdata.get('rate', unit.get('rate', 0.0))
                            unit['sfc'] = pdata.get('sfc', unit.get('sfc', 0.0))
                            unit['sfc_mode'] = pdata.get('sfc_mode', unit.get('sfc_mode', 'manual'))
                            unit['engine_params'] = pdata.get('engine_params', unit.get('engine_params', {}))
                        else:
                            unit['MinRPM'] = min(
                                pdataA.get('MinRPM', unit.get('MinRPM', 0.0)),
                                pdataB.get('MinRPM', unit.get('MinRPM', 0.0)),
                            )
                            unit['DOL'] = max(
                                pdataA.get('DOL', unit.get('DOL', 0.0)),
                                pdataB.get('DOL', unit.get('DOL', 0.0)),
                            )
                            unit['power_type'] = pdataA.get('power_type', unit.get('power_type', 'Grid'))
                            unit['rate'] = pdataA.get('rate', unit.get('rate', 0.0))
                            unit['sfc'] = pdataA.get('sfc', unit.get('sfc', 0.0))
                            unit['sfc_mode'] = pdataA.get('sfc_mode', unit.get('sfc_mode', 'manual'))
                            unit['engine_params'] = pdataA.get('engine_params', unit.get('engine_params', {}))
                        unit['max_pumps'] = actA + actB
                        unit['min_pumps'] = actA + actB
                        expand_all(pos + 1, stn_acc + [unit], kv_acc + [kv], rho_acc + [rho])
        else:
            expand_all(pos + 1, stn_acc + [copy.deepcopy(stn)], kv_acc + [kv], rho_acc + [rho])

    expand_all(0, [], [], [])

    if best_result is None:
        return {
            "error": True,
            "message": "No feasible pump combination found for stations.",
        }

    best_result['stations_used'] = best_stations
    return best_result

# -----------------------------------------------------------------------------
# Adaptive coarse→fine solver
#
# The following helper functions and wrapper implement a two-pass adaptive
# optimisation strategy.  In the first pass the pipeline is solved using
# coarser RPM and DRA discretisations to identify promising regions of the
# search space.  In the second pass the search is restricted to per-station
# ranges around the coarse winners at a finer resolution.  This greatly
# reduces the total number of operating points explored while retaining the
# final granularity specified by ``final_rpm_step`` and ``final_dr_step``.

def _make_range_around(center: int, low: int, high: int, step: int, span: int) -> list[int]:
    """Return a list of integer values centred on ``center`` within
    ``[low, high]`` using step increments and spanning ±``span``.

    The returned list always includes the exact ``high`` boundary even when
    ``high`` does not fall on a step multiple.  Values outside the
    ``[low, high]`` interval are discarded.
    """
    lo = max(low, center - span)
    hi = min(high, center + span)
    # Generate values on the step grid covering [lo, hi]
    start = (lo // step) * step
    end = ((hi + step - 1) // step) * step
    vals = list(range(start, end + 1, step))
    # Ensure exact upper bound is included
    if hi not in vals:
        vals.append(hi)
    return sorted(set(v for v in vals if low <= v <= high))


def solve_pipeline_adaptive(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    Fuel_density: float,
    Ambient_temp: float,
    linefill: list[dict] | dict | None = None,
    *,
    coarse_rpm_step: int = 300,
    coarse_dr_step: int = 20,
    final_rpm_step: int = 100,
    final_dr_step: int = 5,
    rpm_span: int = 400,
    dr_span: int = 20,
    beam_cap: int | None = None,
    loop_usage_by_station: list[int] | None = None,
    enumerate_loops: bool = True,
    residual_round_coarse: int | None = None,
) -> dict:
    """
    Perform a two-pass optimisation with coarse and fine discretisations.

    In the first pass the RPM and DRA step sizes are set to
    ``coarse_rpm_step`` and ``coarse_dr_step``, respectively, to quickly
    explore the entire search space.  The best resulting configuration is
    inspected and per-station RPM and DRA ranges are built by taking
    ±``rpm_span`` and ±``dr_span`` around the coarse optimum.  In the second
    pass the solver is invoked again with step sizes ``final_rpm_step`` and
    ``final_dr_step`` but only within the narrowed ranges.  This approach
    maintains the final resolution specified by the user while reducing
    computation time dramatically.

    Optional parameters:

    * ``beam_cap`` (int|None): limit the number of dynamic-programming states
      retained per station.  See ``solve_pipeline`` for details.
    * ``loop_usage_by_station`` (list[int]|None): explicit loop usage pattern
      (0=disabled, 1=parallel, 2=bypass).  When supplied the solver will not
      enumerate loop cases.
    * ``enumerate_loops`` (bool): whether to enumerate loop usage patterns
      internally.  Ignored when ``loop_usage_by_station`` is not None.
    * ``residual_round_coarse`` (int|None): override the residual rounding
      precision during the coarse pass.  This can reduce the number of DP
      buckets and speed up the coarse search.  The original rounding is
      restored for the fine pass.
    """
    # First pass: coarse search
    coarse_result = solve_pipeline_with_types(
        stations,
        terminal,
        FLOW,
        KV_list,
        rho_list,
        RateDRA,
        Price_HSD,
        Fuel_density,
        Ambient_temp,
        linefill,
        dra_reach_km=0.0,
        mop_kgcm2=None,
        hours=24.0,
        loop_usage_by_station=loop_usage_by_station,
        enumerate_loops=enumerate_loops,
        rpm_step=coarse_rpm_step,
        dra_step=coarse_dr_step,
        per_station_ranges=None,
        beam_cap=beam_cap,
        residual_round_override=residual_round_coarse,
    )
    # If an error occurred (e.g. no feasible point) return immediately
    if coarse_result.get("error"):
        return coarse_result
    # Build per-station ranges around the coarse winners
    per_station_ranges: dict[int, dict] = {}
    for idx, stn in enumerate(stations[:-1], start=0):
        name_key = stn['name'].strip().lower().replace(' ', '_')
        # RPM centre from coarse result (0 if pump was not active)
        rpm_center = int(coarse_result.get(f"speed_{name_key}", 0))
        min_rpm = int(stn.get('MinRPM', 0))
        max_rpm = int(stn.get('DOL', 0))
        if stn.get('is_pump', False) and max_rpm > 0:
            rpm_values = _make_range_around(rpm_center, min_rpm, max_rpm, final_rpm_step, rpm_span)
        else:
            rpm_values = [0]
        # Mainline DR centre
        dr_center = int(round(coarse_result.get(f"drag_reduction_{name_key}", 0.0)))
        max_dr_main = int(stn.get('max_dr', 0))
        dr_main_values = _make_range_around(dr_center, 0, max_dr_main, final_dr_step, dr_span)
        # Loopline DR centre
        if stn.get('loopline'):
            dr_loop_center = int(round(coarse_result.get(f"drag_reduction_loop_{name_key}", 0.0)))
            max_dr_loop = int(stn['loopline'].get('max_dr', 0))
            dr_loop_values = _make_range_around(dr_loop_center, 0, max_dr_loop, final_dr_step, dr_span)
        else:
            dr_loop_values = [0]
        per_station_ranges[idx] = {
            "rpm": rpm_values,
            "dra_main": dr_main_values,
            "dra_loop": dr_loop_values,
        }
    # Second pass: fine search restricted to the narrowed ranges
    return solve_pipeline_with_types(
        stations,
        terminal,
        FLOW,
        KV_list,
        rho_list,
        RateDRA,
        Price_HSD,
        Fuel_density,
        Ambient_temp,
        linefill,
        dra_reach_km=0.0,
        mop_kgcm2=None,
        hours=24.0,
        loop_usage_by_station=loop_usage_by_station,
        enumerate_loops=enumerate_loops,
        rpm_step=final_rpm_step,
        dra_step=final_dr_step,
        per_station_ranges=per_station_ranges,
        beam_cap=beam_cap,
    )
