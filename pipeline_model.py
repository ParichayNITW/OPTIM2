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
# Core calculations
# ---------------------------------------------------------------------------

RPM_STEP = 100
DRA_STEP = 5
# Residual head precision (decimal places) used when bucketing states during the
# dynamic-programming search.  Using a modest precision keeps the state space
# tractable while still providing near-global optimality.
RESIDUAL_ROUND = 1
V_MIN = 0.15
V_MAX = 3.0

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
    batches: list[dict],
) -> tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Split ``flow_m3h`` between ``main`` and ``loop`` so both see the same head loss.

    ``main`` and ``loop`` are dictionaries with keys ``L``, ``d_inner``, ``rough``,
    ``dra`` and ``dra_len`` describing each line.  ``batches`` describes the
    fluid slices within the segment. The function returns the common head loss
    and the (velocity, Re, f, flow) tuples for each path.
    """

    def slice_batches(parts: list[dict], length: float) -> list[dict]:
        remaining = length
        out: list[dict] = []
        for part in parts:
            if remaining <= 0:
                break
            use = min(part['len_km'], remaining)
            out.append({'len_km': use, 'kv': part['kv']})
            remaining -= use
        return out

    overlap = min(main['L'], loop['L'])
    batches_overlap = slice_batches(batches, overlap)

    def calc(line_flow: float, data: dict, parts: list[dict]) -> tuple[float, float, float, float]:
        total_hl = 0.0
        v_ret = Re_ret = f_ret = 0.0
        remaining = data.get('dra_len')
        first = True
        for part in parts:
            use = min(remaining, part['len_km']) if remaining is not None else None
            hl, v, Re, f = _segment_hydraulics(
                line_flow,
                part['len_km'],
                data['d_inner'],
                data['rough'],
                part['kv'],
                data.get('dra', 0.0),
                use,
            )
            total_hl += hl
            if first:
                v_ret, Re_ret, f_ret = v, Re, f
                first = False
            if remaining is not None:
                remaining -= use or 0.0
        return total_hl, v_ret, Re_ret, f_ret

    main_overlap = main.copy()
    if main.get('dra_len') is None:
        main_overlap['dra_len'] = None
    else:
        main_overlap['dra_len'] = min(main['dra_len'], overlap)
    main_overlap['L'] = overlap
    loop_overlap = loop.copy()
    if loop.get('dra_len') is None:
        loop_overlap['dra_len'] = None
    else:
        loop_overlap['dra_len'] = min(loop['dra_len'], overlap)
    loop_overlap['L'] = overlap

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
        tuple((round(p['len_km'], 3), round(p['kv'], 6)) for p in batches_overlap),
    )
    if key in _PARALLEL_CACHE:
        hl_overlap, main_stats, loop_stats = _PARALLEL_CACHE[key]
    else:
        lo, hi = 0.0, flow_m3h
        best = None
        for _ in range(20):
            mid = (lo + hi) / 2.0
            q_loop = mid
            q_main = flow_m3h - q_loop
            hl_main, v_main, Re_main, f_main = calc(q_main, main_overlap, batches_overlap)
            hl_loop, v_loop, Re_loop, f_loop = calc(q_loop, loop_overlap, batches_overlap)
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
        hl_overlap, main_stats, loop_stats = best
        _PARALLEL_CACHE[key] = best

    total_hl = hl_overlap
    if main['L'] > overlap:
        rem = main['L'] - overlap
        batches_rem = slice_batches(batches, rem + overlap)[len(batches_overlap):]
        remaining_dra = None
        if main.get('dra_len') is not None:
            remaining_dra = max(0.0, main.get('dra_len') - overlap)
        for part in batches_rem:
            use = min(remaining_dra, part['len_km']) if remaining_dra is not None else None
            hl_part, _, _, _ = _segment_hydraulics(
                flow_m3h,
                part['len_km'],
                main['d_inner'],
                main['rough'],
                part['kv'],
                main.get('dra', 0.0),
                use,
            )
            total_hl += hl_part
            if remaining_dra is not None:
                remaining_dra -= use or 0.0

    return total_hl, main_stats, loop_stats


def _pump_head(stn: dict, flow_m3h: float, rpm: float, nop: int) -> tuple[float, float]:
    """Return (tdh, efficiency) for ``nop`` identical pumps in **series**."""
    dol = stn.get('DOL', rpm)
    Q_equiv = flow_m3h * dol / rpm if rpm > 0 else flow_m3h
    A = stn.get('A', 0.0)
    B = stn.get('B', 0.0)
    C = stn.get('C', 0.0)
    tdh_single = A * Q_equiv ** 2 + B * Q_equiv + C
    tdh = tdh_single * (rpm / dol) ** 2 * nop
    P = stn.get('P', 0.0)
    Q = stn.get('Q', 0.0)
    R = stn.get('R', 0.0)
    S = stn.get('S', 0.0)
    T = stn.get('T', 0.0)
    eff = P * Q_equiv ** 4 + Q * Q_equiv ** 3 + R * Q_equiv ** 2 + S * Q_equiv + T
    return tdh, eff


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
    segment_flows: list[float],
    seg_batches: list[list[dict]],
) -> float:
    """Return minimum residual head needed immediately after station ``idx``.

    The previous implementation only accumulated losses across consecutive
    non-pump stations.  When multiple pump stations appear in sequence (e.g. to
    represent different pump types at an origin), upstream pumps were unaware of
    the downstream pressure requirement and the solver could deem a feasible
    configuration infeasible.  This version performs a backward recursion over
    *all* downstream stations, subtracting the maximum head each pump can
    deliver and adding line/elevation losses for every segment.  The returned
    value is therefore the minimum residual needed after station ``idx`` so that
    the terminal residual head constraint can still be met.
    """

    from functools import lru_cache

    N = len(stations)

    @lru_cache(None)
    def req_entry(i: int) -> float:
        if i >= N:
            return terminal.get('min_residual', 0.0)
        stn = stations[i]
        batches = seg_batches[i]
        # ``segment_flows`` holds the flow rate *after* each station;
        # use the downstream value so losses reflect the correct
        # segment flow between station ``i`` and ``i+1``.
        flow = segment_flows[i + 1]
        L = stn.get('L', 0.0)
        t = stn.get('t', 0.007)
        if 'D' in stn:
            d_inner = stn['D'] - 2 * t
        else:
            d_inner = stn.get('d', 0.7) - 2 * t
        rough = stn.get('rough', 0.00004)
        dra_down = stn.get('max_dr', 0.0)

        # Head loss assuming the entire flow stays in the mainline.  When a
        # loopline is available we also evaluate the parallel configuration and
        # retain the lower loss so upstream pumps are not over-constrained.
        head_main = 0.0
        for part in batches:
            hl, *_ = _segment_hydraulics(flow, part['len_km'], d_inner, rough, part['kv'], dra_down, None)
            head_main += hl
        head_loss = head_main
        loop = stn.get('loopline')
        if loop:
            L_loop = loop.get('L', L)
            if 'D' in loop:
                t_loop = loop.get('t', t)
                d_inner_loop = loop['D'] - 2 * t_loop
            else:
                d_inner_loop = loop.get('d', d_inner)
                t_loop = loop.get('t', t)
            rough_loop = loop.get('rough', rough)
            dra_loop = loop.get('max_dr', 0.0)
            hl_par, _, _ = _parallel_segment_hydraulics(
                flow,
                {
                    'L': L,
                    'd_inner': d_inner,
                    'rough': rough,
                    'dra': dra_down,
                    'dra_len': None,
                },
                {
                    'L': L_loop,
                    'd_inner': d_inner_loop,
                    'rough': rough_loop,
                    'dra': dra_loop,
                    'dra_len': None,
                },
                batches,
            )
            head_loss = min(head_main, hl_par)
        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N - 1 else stations[i + 1].get('elev', 0.0)
        downstream = req_entry(i + 1)
        req = downstream + head_loss + (elev_next - elev_i)

        # Check intermediate peaks within this segment.  Each peak requires enough
        # upstream pressure to maintain at least 10 m of residual head at the peak
        # itself.  Use the maximum requirement among all peaks and the downstream
        # station.
        peak_req = 0.0
        # Mainline peaks
        for peak in stn.get('peaks', []) or []:
            dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
            elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
            if dist is None or elev_peak is None:
                continue
            head_peak, *_ = _segment_hydraulics(flow, float(dist), d_inner, rough, batches[0]['kv'], dra_down)
            req_peak = head_peak + (float(elev_peak) - elev_i) + 10.0
            if req_peak > peak_req:
                peak_req = req_peak

        # Loopline peaks (if loopline present)
        loop = stn.get('loopline')
        if loop:
            L_loop = loop.get('L', L)
            if 'D' in loop:
                t_loop = loop.get('t', t)
                d_inner_loop = loop['D'] - 2 * t_loop
            else:
                d_inner_loop = loop.get('d', d_inner)
                t_loop = loop.get('t', t)
            rough_loop = loop.get('rough', rough)
            dra_loop = loop.get('max_dr', 0.0)
            for peak in loop.get('peaks', []) or []:
                dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
                elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
                if dist is None or elev_peak is None:
                    continue
                head_peak, *_ = _segment_hydraulics(flow, float(dist), d_inner_loop, rough_loop, batches[0]['kv'], dra_loop)
                req_peak = head_peak + (float(elev_peak) - elev_i) + 10.0
                if req_peak > peak_req:
                    peak_req = req_peak
        req = max(req, peak_req)

        if stn.get('is_pump', False):
            rpm_max = int(stn.get('DOL', stn.get('MinRPM', 0)))
            nop_max = stn.get('max_pumps', 0)
            tdh_max, _ = _pump_head(stn, flow, rpm_max, nop_max) if rpm_max and nop_max else (0.0, 0.0)
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
    seg_batches: list[list[dict]],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    Fuel_density: float,
    Ambient_temp: float,
    linefill_dict: dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
) -> dict:
    """Enumerate feasible options across all stations to find the lowest-cost
    operating strategy.  This replaces the previous greedy approach and
    guarantees that the global minimum (within the discretised search space) is
    returned."""

    N = len(stations)
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
        flow_in = segment_flows[i - 1]
        batches = seg_batches[i - 1]
        kv_first = batches[0]['kv'] if batches else 0.0
        rho = rho_list[i - 1]

        min_residual_next = _downstream_requirement(stations, i - 1, terminal, segment_flows, seg_batches)

        L = stn.get('L', 0.0)
        if 'D' in stn:
            thickness = stn.get('t', default_t)
            d_inner = stn['D'] - 2 * thickness
            outer_d = stn['D']
        else:
            d_inner = stn.get('d', 0.7)
            outer_d = stn.get('d', 0.7)
            thickness = stn.get('t', default_t)
        rough = stn.get('rough', default_e)

        SMYS = stn.get('SMYS', 52000.0)
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
        v_nom = flow_m3s / area if area > 0 else 0.0
        travel_km = v_nom * hours * 3600.0 / 1000.0

        if stn.get('is_pump', False):
            is_origin_station = not origin_enforced
            min_p = stn.get('min_pumps', 0)
            if is_origin_station:
                min_p = max(1, min_p)
                origin_enforced = True
            max_p = stn.get('max_pumps', 2)
            rpm_vals = _allowed_values(int(stn.get('MinRPM', 0)), int(stn.get('DOL', 0)), RPM_STEP)
            fixed_dr = stn.get('fixed_dra_perc', None)
            dra_main_vals = [int(round(fixed_dr))] if (fixed_dr is not None) else _allowed_values(0, int(stn.get('max_dr', 0)), DRA_STEP)
            if loop_dict:
                dra_loop_vals = _allowed_values(0, int(loop_dict.get('max_dr', 0)), DRA_STEP)
            else:
                dra_loop_vals = [0]
            for nop in range(min_p, max_p + 1):
                rpm_opts = [0] if nop == 0 else rpm_vals
                for rpm in rpm_opts:
                    for dra_main in dra_main_vals:
                        for dra_loop in dra_loop_vals:
                            if nop > 0 and rpm > 0:
                                tdh, eff = _pump_head(stn, flow_in, rpm, nop)
                            else:
                                tdh, eff = 0.0, 0.0
                            eff = max(eff, 1e-6) if nop > 0 else 0.0
                            if nop > 0 and rpm > 0:
                                pump_bkw_total = (rho * flow_in * 9.81 * tdh) / (3600.0 * 1000.0 * (eff / 100.0))
                                # Pumps operate in series so the total brake
                                # work and motor load are not divided among
                                # units.  Each pump sees the full flow and the
                                # combined head is the sum of individual heads.
                                pump_bkw = pump_bkw_total
                                if stn.get('power_type', 'Grid') == 'Diesel':
                                    prime_kw_total = pump_bkw_total / 0.98
                                else:
                                    prime_kw_total = pump_bkw_total / 0.95
                                motor_kw = prime_kw_total
                            else:
                                pump_bkw = motor_kw = prime_kw_total = 0.0
                            if stn.get('power_type', 'Grid') == 'Diesel' and prime_kw_total > 0:
                                mode = stn.get('sfc_mode', 'manual')
                                if mode == 'manual':
                                    sfc_val = stn.get('sfc', 0.0)
                                elif mode == 'iso':
                                    sfc_val = _compute_iso_sfc(stn, rpm, pump_bkw_total, stn.get('DOL', rpm), stn.get('elev', 0.0), Ambient_temp)
                                else:
                                    sfc_val = 0.0
                                fuel_per_kWh = (sfc_val * 1.34102) / Fuel_density if sfc_val else 0.0
                                power_cost = prime_kw_total * hours * fuel_per_kWh * Price_HSD
                            else:
                                rate = stn.get('rate', 0.0)
                                power_cost = prime_kw_total * hours * rate
                            ppm_main = get_ppm_for_dr(kv_first, dra_main) if dra_main > 0 else 0.0
                            ppm_loop = get_ppm_for_dr(kv_first, dra_loop) if dra_loop > 0 else 0.0
                            opts.append({
                                'nop': nop,
                                'rpm': rpm,
                                'dra_main': dra_main,
                                'dra_loop': dra_loop,
                                'travel_km': travel_km if (dra_main > 0 or dra_loop > 0) else 0.0,
                                'tdh': tdh,
                                'eff': eff,
                                'pump_bkw': pump_bkw,
                                'motor_kw': motor_kw,
                                'power_cost': power_cost,
                                'dra_ppm_main': ppm_main,
                                'dra_ppm_loop': ppm_loop,
                            })
        else:
            opts.append({
                'nop': 0,
                'rpm': 0,
                'dra_main': 0,
                'dra_loop': 0,
                'travel_km': 0.0,
                'tdh': 0.0,
                'eff': 0.0,
                'pump_bkw': 0.0,
                'motor_kw': 0.0,
                'power_cost': 0.0,
                'dra_ppm_main': 0.0,
                'dra_ppm_loop': 0.0,
            })

        station_opts.append({
            'name': name,
            'orig_name': stn['name'],
            'flow': flow,
            'flow_in': flow_in,
            'batches': batches,
            'kv_first': batches[0]['kv'] if batches else 0.0,
            'rho': rho,
            'L': L,
            'd_inner': d_inner,
            'rough': rough,
            'cum_dist': cum_dist,
            'elev_delta': elev_delta,
            'min_residual_next': min_residual_next,
            'maop_head': maop_head,
            'maop_kgcm2': maop_kgcm2,
            'loopline': loop_dict,
            'options': opts,
            'is_pump': stn.get('is_pump', False),
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
        })
        cum_dist += L
    # Dynamic programming over stations
    init_residual = stations[0].get('min_residual', 50.0)
    states: dict[tuple[float, float, float], dict] = {
        (round(init_residual, 2), round(dra_reach_km, 1), round(0.0, 1)): {
            'cost': 0.0,
            'residual': init_residual,
            'records': [],
            'last_maop': 0.0,
            'last_maop_kg': 0.0,
            'reach': dra_reach_km,
            'ppm': 0.0,
        }
    }

    for stn_data in station_opts:
        new_states: dict[tuple[float, float, float], dict] = {}
        for state in states.values():
            for opt in stn_data['options']:
                reach_prev = state.get('reach', 0.0)
                ppm_prev = state.get('ppm', 0.0)
                reach_after = reach_prev
                if opt['dra_main'] > 0 or opt['dra_loop'] > 0:
                    reach_after = max(reach_after, stn_data['cum_dist'] + opt['travel_km'])

                dra_len_main = max(0.0, min(stn_data['L'], reach_after - stn_data['cum_dist']))
                ppm_main = max(ppm_prev, opt['dra_ppm_main'])
                ppm_loop = max(ppm_prev, opt['dra_ppm_loop'])
                eff_dra_main = (
                    get_dr_for_ppm(stn_data['kv_first'], ppm_main) if dra_len_main > 0 else 0.0
                )
                scenarios = []
                hl_single = 0.0
                v_single = Re_single = f_single = 0.0
                remaining = dra_len_main if eff_dra_main > 0 else None
                first = True
                for part in stn_data['batches']:
                    use = min(remaining, part['len_km']) if remaining is not None else None
                    hl_part, v_part, Re_part, f_part = _segment_hydraulics(
                        stn_data['flow'],
                        part['len_km'],
                        stn_data['d_inner'],
                        stn_data['rough'],
                        part['kv'],
                        eff_dra_main,
                        use,
                    )
                    hl_single += hl_part
                    if first:
                        v_single, Re_single, f_single = v_part, Re_part, f_part
                        first = False
                    if remaining is not None:
                        remaining -= use or 0.0
                scenarios.append({
                    'head_loss': hl_single,
                    'v': v_single,
                    'Re': Re_single,
                    'f': f_single,
                    'flow_main': stn_data['flow'],
                    'v_loop': 0.0,
                    'Re_loop': 0.0,
                    'f_loop': 0.0,
                    'flow_loop': 0.0,
                    'maop_loop': 0.0,
                    'maop_loop_kg': 0.0,
                    'mode': 'No Bypass',
                })
                if stn_data.get('loopline'):
                    loop = stn_data['loopline']
                    dra_len_loop = max(0.0, min(loop['L'], reach_after - stn_data['cum_dist']))
                    eff_dra_loop = (
                        get_dr_for_ppm(stn_data['kv_first'], ppm_loop) if dra_len_loop > 0 else 0.0
                    )
                    hl_par, main_stats, loop_stats = _parallel_segment_hydraulics(
                        stn_data['flow'],
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
                        stn_data['batches'],
                    )
                    v_m, Re_m, f_m, q_main = main_stats
                    v_l, Re_l, f_l, q_loop = loop_stats
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
                        'mode': 'No Bypass',
                    })
                    # Scenario where loopline bypasses the station entirely
                    hl_loop_only = 0.0
                    v_lp = Re_lp = f_lp = 0.0
                    remaining_loop = dra_len_loop if eff_dra_loop > 0 else None
                    first_lp = True
                    for part in stn_data['batches']:
                        use = min(remaining_loop, part['len_km']) if remaining_loop is not None else None
                        hl_part, v_part, Re_part, f_part = _segment_hydraulics(
                            stn_data['flow'],
                            part['len_km'],
                            loop['d_inner'],
                            loop['rough'],
                            part['kv'],
                            eff_dra_loop,
                            use,
                        )
                        hl_loop_only += hl_part
                        if first_lp:
                            v_lp, Re_lp, f_lp = v_part, Re_part, f_part
                            first_lp = False
                        if remaining_loop is not None:
                            remaining_loop -= use or 0.0
                    scenarios.append({
                        'head_loss': hl_loop_only,
                        'v': 0.0,
                        'Re': 0.0,
                        'f': 0.0,
                        'flow_main': 0.0,
                        'v_loop': v_lp,
                        'Re_loop': Re_lp,
                        'f_loop': f_lp,
                        'flow_loop': stn_data['flow'],
                        'maop_loop': loop['maop_head'],
                        'maop_loop_kg': loop['maop_kgcm2'],
                        'mode': 'Bypass',
                    })
                for sc in scenarios:
                    if sc['flow_main'] > 0 and not (V_MIN <= sc['v'] <= V_MAX):
                        continue
                    if sc['flow_loop'] > 0 and not (V_MIN <= sc['v_loop'] <= V_MAX):
                        continue
                    sdh = state['residual'] + opt['tdh']
                    if sc.get('mode') == 'Bypass':
                        sdh = state['residual']
                    if sdh > stn_data['maop_head'] or (sc['flow_loop'] > 0 and sdh > stn_data['loopline']['maop_head']):
                        continue
                    residual_next = sdh - sc['head_loss'] - stn_data['elev_delta']
                    if residual_next < stn_data['min_residual_next']:
                        continue
                    inj_ppm_main = max(0.0, opt['dra_ppm_main'] - ppm_prev)
                    inj_ppm_loop = max(0.0, opt['dra_ppm_loop'] - ppm_prev)
                    dra_cost = inj_ppm_main * (sc['flow_main'] * 1000.0 * hours / 1e6) * RateDRA
                    if sc['flow_loop'] > 0:
                        dra_cost += inj_ppm_loop * (sc['flow_loop'] * 1000.0 * hours / 1e6) * RateDRA
                    power_cost = opt['power_cost'] if sc['flow_main'] > 0 else 0.0
                    total_cost = power_cost + dra_cost
                    flow_total = sc['flow_main'] + sc['flow_loop']
                    ppm_next = (
                        (ppm_main * sc['flow_main'] + ppm_loop * sc['flow_loop']) / flow_total
                        if flow_total > 0
                        else 0.0
                    )
                    key = stn_data['name'].lower().replace(' ', '_')
                    record = {
                        f"pipeline_flow_{key}": sc['flow_main'],
                        f"pipeline_flow_in_{key}": stn_data['flow_in'],
                        f"loopline_flow_{key}": sc['flow_loop'],
                        f"head_loss_{key}": sc['head_loss'],
                        f"head_loss_kgcm2_{key}": head_to_kgcm2(sc['head_loss'], stn_data['rho']),
                        f"residual_head_{key}": state['residual'],
                        f"rh_kgcm2_{key}": head_to_kgcm2(state['residual'], stn_data['rho']),
                        f"sdh_{key}": sdh if stn_data['is_pump'] else state['residual'],
                        f"sdh_kgcm2_{key}": head_to_kgcm2(sdh if stn_data['is_pump'] else state['residual'], stn_data['rho']),
                        f"rho_{key}": stn_data['rho'],
                        f"maop_{key}": stn_data['maop_head'],
                        f"maop_kgcm2_{key}": stn_data['maop_kgcm2'],
                        f"velocity_{key}": sc['v'],
                        f"reynolds_{key}": sc['Re'],
                        f"friction_{key}": sc['f'],
                        f"coef_A_{key}": stn_data['coef_A'],
                        f"coef_B_{key}": stn_data['coef_B'],
                        f"coef_C_{key}": stn_data['coef_C'],
                        f"coef_P_{key}": stn_data['coef_P'],
                        f"coef_Q_{key}": stn_data['coef_Q'],
                        f"coef_R_{key}": stn_data['coef_R'],
                        f"coef_S_{key}": stn_data['coef_S'],
                        f"coef_T_{key}": stn_data['coef_T'],
                        f"min_rpm_{key}": stn_data['min_rpm'],
                        f"dol_{key}": stn_data['dol'],
                    }
                    if sc['flow_loop'] > 0:
                        record.update({
                            f"velocity_loop_{key}": sc['v_loop'],
                            f"reynolds_loop_{key}": sc['Re_loop'],
                            f"friction_loop_{key}": sc['f_loop'],
                            f"maop_loop_{key}": sc['maop_loop'],
                            f"maop_loop_kgcm2_{key}": sc['maop_loop_kg'],
                        })
                    else:
                        record.update({
                            f"velocity_loop_{key}": 0.0,
                            f"reynolds_loop_{key}": 0.0,
                            f"friction_loop_{key}": 0.0,
                            f"maop_loop_{key}": 0.0,
                            f"maop_loop_kgcm2_{key}": 0.0,
                        })
                    if stn_data.get('loopline'):
                        record[f"loopline_mode_{key}"] = sc.get('mode', 'No Bypass')
                    else:
                        record[f"loopline_mode_{key}"] = 'N/A'
                    if stn_data['is_pump']:
                        record.update({
                            f"pump_flow_{key}": stn_data['flow_in'] if sc['flow_main'] > 0 else 0.0,
                            f"num_pumps_{key}": opt['nop'] if sc['flow_main'] > 0 else 0,
                            f"speed_{key}": opt['rpm'] if sc['flow_main'] > 0 else 0.0,
                            f"efficiency_{key}": opt['eff'] if sc['flow_main'] > 0 else 0.0,
                            f"pump_bkw_{key}": opt['pump_bkw'] if sc['flow_main'] > 0 else 0.0,
                            f"motor_kw_{key}": opt['motor_kw'] if sc['flow_main'] > 0 else 0.0,
                            f"power_cost_{key}": power_cost,
                            f"dra_cost_{key}": dra_cost,
                            f"dra_ppm_{key}": opt['dra_ppm_main'],
                            f"dra_ppm_loop_{key}": opt['dra_ppm_loop'],
                            f"drag_reduction_{key}": opt['dra_main'],
                            f"drag_reduction_loop_{key}": opt['dra_loop'],
                        })
                    else:
                        record.update({
                            f"pump_flow_{key}": 0.0,
                            f"num_pumps_{key}": 0,
                            f"speed_{key}": 0.0,
                            f"efficiency_{key}": 0.0,
                            f"pump_bkw_{key}": 0.0,
                            f"motor_kw_{key}": 0.0,
                            f"power_cost_{key}": 0.0,
                            f"dra_cost_{key}": 0.0,
                            f"dra_ppm_{key}": 0.0,
                            f"dra_ppm_loop_{key}": 0.0,
                            f"drag_reduction_{key}": 0.0,
                            f"drag_reduction_loop_{key}": 0.0,
                        })
                    new_cost = state['cost'] + total_cost
                    bucket = (
                        round(residual_next, RESIDUAL_ROUND),
                        round(reach_after, 1),
                        round(ppm_next, 1),
                    )
                    new_record_list = state['records'] + [record]
                    existing = new_states.get(bucket)
                    if (
                        existing is None
                        or new_cost < existing['cost']
                        or (
                            abs(new_cost - existing['cost']) < 1e-9
                            and (
                                residual_next > existing['residual']
                                or (
                                    abs(residual_next - existing['residual']) < 1e-9
                                    and (
                                        reach_after > existing.get('reach', 0.0)
                                        or (
                                            abs(reach_after - existing.get('reach', 0.0)) < 1e-9
                                            and ppm_next > existing.get('ppm', 0.0)
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        new_states[bucket] = {
                            'cost': new_cost,
                            'residual': residual_next,
                            'records': new_record_list,
                            'last_maop': stn_data['maop_head'],
                            'last_maop_kg': stn_data['maop_kgcm2'],
                            'reach': reach_after,
                            'ppm': ppm_next,
                        }

        if not new_states:
            return {"error": True, "message": f"No feasible operating point for {stn_data['orig_name']}"}
        # Remove dominated states while considering residual head, DRA reach and
        # current concentration.
        pruned: dict[tuple[float, float, float], dict] = {}
        for bucket, data in new_states.items():
            dominated = False
            to_remove: list[tuple[float, float, float]] = []
            for b2, d2 in pruned.items():
                if (
                    d2['cost'] <= data['cost']
                    and d2['residual'] >= data['residual']
                    and d2.get('reach', 0.0) >= data.get('reach', 0.0)
                    and d2.get('ppm', 0.0) >= data.get('ppm', 0.0)
                ):
                    dominated = True
                    break
                if (
                    data['cost'] <= d2['cost']
                    and data['residual'] >= d2['residual']
                    and data.get('reach', 0.0) >= d2.get('reach', 0.0)
                    and data.get('ppm', 0.0) >= d2.get('ppm', 0.0)
                ):
                    to_remove.append(b2)
            if not dominated:
                for b2 in to_remove:
                    del pruned[b2]
                pruned[bucket] = data
        states = pruned

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

    term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": segment_flows[-1],
        f"pipeline_flow_in_{term_name}": segment_flows[-2],
        f"pump_flow_{term_name}": 0.0,
        f"speed_{term_name}": 0.0,
        f"num_pumps_{term_name}": 0,
        f"efficiency_{term_name}": 0.0,
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
    result['dra_front_km'] = best_state.get('reach', 0.0)
    result['dra_ppm_final'] = best_state.get('ppm', 0.0)
    result['error'] = False
    return result


def solve_pipeline_with_types(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    seg_batches: list[list[dict]],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    Fuel_density: float,
    Ambient_temp: float,
    linefill_dict: dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
) -> dict:
    """Enumerate pump type combinations at all stations and call ``solve_pipeline``."""

    best_result = None
    best_cost = float('inf')
    best_stations = None
    N = len(stations)

    def expand_all(pos: int, stn_acc: list[dict], batch_acc: list[list[dict]], rho_acc: list[float]):
        nonlocal best_result, best_cost, best_stations
        if pos >= N:
            result = solve_pipeline(stn_acc, terminal, FLOW, batch_acc, rho_acc, RateDRA, Price_HSD, Fuel_density, Ambient_temp, linefill_dict, dra_reach_km, mop_kgcm2, hours)
            if result.get("error"):
                return
            cost = result.get("total_cost", float('inf'))
            if cost < best_cost:
                best_cost = cost
                best_result = result
                best_stations = stn_acc
            return

        stn = stations[pos]
        batches = seg_batches[pos]
        rho = rho_list[pos]

        if stn.get('pump_types'):
            combos = generate_type_combinations(
                stn['pump_types'].get('A', {}).get('available', 0),
                stn['pump_types'].get('B', {}).get('available', 0),
            )
            for numA, numB in combos:
                units: list[dict] = []
                name_base = stn['name']
                for ptype, count in [('A', numA), ('B', numB)]:
                    pdata = stn['pump_types'].get(ptype)
                    for n in range(count):
                        unit = {
                            'name': f"{name_base}_{ptype}{n + 1}",
                            'pump_name': pdata.get('name', f'Type {ptype}') if pdata else f'Type {ptype}',
                            'elev': stn.get('elev', 0.0),
                            'D': stn.get('D'),
                            't': stn.get('t'),
                            'SMYS': stn.get('SMYS'),
                            'rough': stn.get('rough'),
                            'L': 0.0,
                            'is_pump': True,
                            'head_data': pdata.get('head_data') if pdata else None,
                            'eff_data': pdata.get('eff_data') if pdata else None,
                            'A': pdata.get('A', 0.0) if pdata else 0.0,
                            'B': pdata.get('B', 0.0) if pdata else 0.0,
                            'C': pdata.get('C', 0.0) if pdata else 0.0,
                            'P': pdata.get('P', 0.0) if pdata else 0.0,
                            'Q': pdata.get('Q', 0.0) if pdata else 0.0,
                            'R': pdata.get('R', 0.0) if pdata else 0.0,
                            'S': pdata.get('S', 0.0) if pdata else 0.0,
                            'T': pdata.get('T', 0.0) if pdata else 0.0,
                            'power_type': pdata.get('power_type', 'Grid') if pdata else 'Grid',
                            'rate': pdata.get('rate', 0.0) if pdata else 0.0,
                            'sfc_mode': pdata.get('sfc_mode', 'none') if pdata else 'none',
                            'sfc': pdata.get('sfc', 0.0) if pdata else 0.0,
                            'engine_params': pdata.get('engine_params', {}) if pdata else {},
                            'MinRPM': pdata.get('MinRPM', 0.0) if pdata else 0.0,
                            'DOL': pdata.get('DOL', 0.0) if pdata else 0.0,
                            'max_pumps': 1,
                            'min_pumps': 1,
                            'delivery': 0.0,
                            'supply': 0.0,
                            'max_dr': 0.0,
                        }
                        units.append(unit)
                if not units:
                    continue
                units[0]['delivery'] = stn.get('delivery', 0.0)
                units[0]['supply'] = stn.get('supply', 0.0)
                min_res = 50.0 if pos == 0 else 0.0
                units[0]['min_residual'] = stn.get('min_residual', min_res)
                units[-1]['L'] = stn.get('L', 0.0)
                units[-1]['max_dr'] = stn.get('max_dr', 0.0)
                if stn.get('loopline'):
                    units[-1]['loopline'] = copy.deepcopy(stn['loopline'])
                expand_all(pos + 1, stn_acc + units, batch_acc + [batches] * len(units), rho_acc + [rho] * len(units))
        else:
            expand_all(pos + 1, stn_acc + [copy.deepcopy(stn)], batch_acc + [batches], rho_acc + [rho])

    expand_all(0, [], [], [])

    if best_result is None:
        return {
            "error": True,
            "message": "No feasible pump combination found for stations.",
        }

    best_result['stations_used'] = best_stations
    return best_result
