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

from dra_utils import get_ppm_for_dr

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


def _pump_head(stn: dict, flow_m3h: float, rpm: float, nop: int) -> tuple[float, float]:
    """Return (tdh, efficiency) for ``stn`` at ``rpm`` and ``nop`` pumps."""
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
    segment_flows: list[float] | None,
    KV_list: list[float],
    flow_override: float | list[float] | None = None,
    pump_plan: dict[int, tuple[int, int]] | None = None,
) -> float:
    """Return minimum residual head needed immediately after station ``idx``.

    ``pump_plan`` optionally maps a station index to the chosen number of pumps
    and rpm.  Only stations present in this plan have their head contribution
    subtracted when computing the requirement; others are treated as offering no
    additional head.  ``segment_flows`` may supply the flow rate after each
    station.  When ``flow_override`` is given it takes precedence and may be
    either a constant flow value or a full per-segment list.  The returned value
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
        if 'D' in stn:
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
        def peak_requirement(peaks, d_pipe, rough_pipe, dra_perc):
            req_local = 0.0
            for peak in peaks or []:
                dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
                elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
                if dist is None or elev_peak is None:
                    continue
                head_peak, *_ = _segment_hydraulics(flow, float(dist), d_pipe, rough_pipe, kv, dra_perc)
                req_p = head_peak + (float(elev_peak) - elev_i) + 25.0
                if req_p > req_local:
                    req_local = req_p
            return req_local

        peak_req = peak_requirement(stn.get('peaks'), d_inner, rough, dra_down)
        loop = stn.get('loopline')
        if loop:
            if 'D' in loop:
                t_loop = loop.get('t', t)
                d_inner_loop = loop['D'] - 2 * t_loop
            else:
                d_inner_loop = loop.get('d', d_inner)
            rough_loop = loop.get('rough', rough)
            dra_loop = loop.get('max_dr', 0.0)
            peak_req = max(peak_req, peak_requirement(loop.get('peaks'), d_inner_loop, rough_loop, dra_loop))
        req = max(req, peak_req)

        if stn.get('is_pump', False):
            # Assume downstream stations can contribute head.  When a specific
            # pump selection has already been made for the station (present in
            # ``pump_plan``) use that head value; otherwise subtract the
            # maximum head the station could deliver (``max_pumps`` running at
            # ``DOL``).  This prevents upstream stations from being forced to
            # supply the entire downstream requirement when later stations could
            # assist.
            if pump_plan and i in pump_plan:
                nop_sel, rpm_sel = pump_plan[i]
            else:
                nop_sel = stn.get('max_pumps', 0)
                rpm_sel = stn.get('DOL', 0)
            if nop_sel and rpm_sel:
                tdh_sel, _ = _pump_head(stn, flow, rpm_sel, nop_sel)
            else:
                tdh_sel = 0.0
            req -= tdh_sel
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
        kv = KV_list[i - 1]
        rho = rho_list[i - 1]

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
        if stn.get('is_pump', False):
            max_p = stn.get('max_pumps', 2)
            if not origin_enforced:
                min_p = max(1, stn.get('min_pumps', 0))
                origin_enforced = True
            else:
                min_p = stn.get('min_pumps', 0)
            rpm_vals = _allowed_values(int(stn.get('MinRPM', 0)), int(stn.get('DOL', 0)), RPM_STEP)
            fixed_dr = stn.get('fixed_dra_perc', None)
            dra_main_vals = [int(round(fixed_dr))] if (fixed_dr is not None) else _allowed_values(0, int(stn.get('max_dr', 0)), DRA_STEP)
            dra_loop_vals = _allowed_values(0, int(loop_dict.get('max_dr', 0)), DRA_STEP) if loop_dict else [0]
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
            if min_p == 0 and not any(o['nop'] == 0 for o in opts):
                opts.insert(0, {
                    'nop': 0,
                    'rpm': 0,
                    'dra_main': 0,
                    'dra_loop': 0,
                    'dra_ppm_main': 0.0,
                    'dra_ppm_loop': 0.0,
                })
        else:
            opts.append({
                'nop': 0,
                'rpm': 0,
                'dra_main': 0,
                'dra_loop': 0,
                'dra_ppm_main': 0.0,
                'dra_ppm_loop': 0.0,
            })

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
    # Dynamic programming over stations
    init_residual = stations[0].get('min_residual', 50.0)
    states: dict[float, dict] = {
        round(init_residual, 2): {
            'cost': 0.0,
            'residual': init_residual,
            'records': [],
            'last_maop': 0.0,
            'last_maop_kg': 0.0,
            'reach': dra_reach_km,
            'flow': segment_flows[0],
            'plan': {},
        }
    }

    for stn_data in station_opts:
        new_states: dict[float, dict] = {}
        for state in states.values():
            flow_total = state.get('flow', segment_flows[0])
            for opt in stn_data['options']:
                reach_prev = state.get('reach', 0.0)
                area = pi * stn_data['d_inner'] ** 2 / 4.0
                v_nom = flow_total / 3600.0 / area if area > 0 else 0.0
                travel_km = (
                    v_nom * hours * 3600.0 / 1000.0 if (opt['dra_main'] > 0 or opt['dra_loop'] > 0) else 0.0
                )
                reach_after = max(reach_prev, stn_data['cum_dist'] + travel_km)

                dra_len_main = max(0.0, min(stn_data['L'], reach_after - stn_data['cum_dist']))
                eff_dra_main = opt['dra_main'] if dra_len_main > 0 else 0.0
                scenarios = []
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
                    dra_len_loop = max(0.0, min(loop['L'], reach_after - stn_data['cum_dist']))
                    eff_dra_loop = opt['dra_loop'] if dra_len_loop > 0 else 0.0
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
                    }
                    tdh, eff = _pump_head(pump_def, flow_total, opt['rpm'], opt['nop'])
                else:
                    tdh, eff = 0.0, 0.0
                eff = max(eff, 1e-6) if opt['nop'] > 0 else 0.0
                if opt['nop'] > 0 and opt['rpm'] > 0:
                    pump_bkw_total = (stn_data['rho'] * flow_total * 9.81 * tdh) / (
                        3600.0 * 1000.0 * (eff / 100.0)
                    )
                    pump_bkw = pump_bkw_total / opt['nop']
                    prime_kw_total = pump_bkw_total / (0.98 if stn_data['power_type'] == 'Diesel' else 0.95)
                    motor_kw = prime_kw_total / opt['nop']
                else:
                    pump_bkw = motor_kw = prime_kw_total = 0.0
                if stn_data['power_type'] == 'Diesel' and prime_kw_total > 0:
                    mode = stn_data.get('sfc_mode', 'manual')
                    if mode == 'manual':
                        sfc_val = stn_data.get('sfc', 0.0)
                    elif mode == 'iso':
                        sfc_val = _compute_iso_sfc(
                            stn_data,
                            opt['rpm'],
                            pump_bkw_total,
                            stn_data['dol'],
                            stn_data.get('elev', 0.0),
                            Ambient_temp,
                        )
                    else:
                        sfc_val = 0.0
                    fuel_per_kWh = (sfc_val * 1.34102) / Fuel_density if sfc_val else 0.0
                    power_cost = prime_kw_total * hours * fuel_per_kWh * Price_HSD
                else:
                    power_cost = prime_kw_total * hours * stn_data.get('rate', 0.0)

                for sc in scenarios:
                    if sc['flow_main'] > 0 and not (V_MIN <= sc['v'] <= V_MAX):
                        continue
                    if sc['flow_loop'] > 0 and not (V_MIN <= sc['v_loop'] <= V_MAX):
                        continue
                    sdh = state['residual'] + tdh
                    if sdh > stn_data['maop_head'] or (
                        sc['flow_loop'] > 0 and sdh > stn_data['loopline']['maop_head']
                    ):
                        continue
                    residual_next = sdh - sc['head_loss'] - stn_data['elev_delta']
                    seg_flows_tmp = segment_flows.copy()
                    seg_flows_tmp[stn_data['idx'] + 1] = sc['flow_main'] if sc.get('bypass_next') else flow_total
                    for j in range(stn_data['idx'] + 1, N):
                        delivery_j = float(stations[j].get('delivery', 0.0))
                        supply_j = float(stations[j].get('supply', 0.0))
                        seg_flows_tmp[j + 1] = seg_flows_tmp[j] - delivery_j + supply_j
                    if sc.get('bypass_next') and stn_data['idx'] + 1 < N:
                        stations_skip = copy.deepcopy(stations)
                        stations_skip[stn_data['idx'] + 1]['is_pump'] = False
                        stations_skip[stn_data['idx'] + 1]['max_pumps'] = 0
                        min_req = _downstream_requirement(
                            stations_skip,
                            stn_data['idx'],
                            terminal,
                            seg_flows_tmp,
                            KV_list,
                            pump_plan=state.get('plan', {}),
                        )
                    else:
                        min_req = _downstream_requirement(
                            stations,
                            stn_data['idx'],
                            terminal,
                            seg_flows_tmp,
                            KV_list,
                            pump_plan=state.get('plan', {}),
                        )
                    if residual_next < min_req:
                        continue
                    dra_cost = 0.0
                    if opt['dra_ppm_main'] > 0:
                        dra_cost += opt['dra_ppm_main'] * (sc['flow_main'] * 1000.0 * hours / 1e6) * RateDRA
                    if opt['dra_ppm_loop'] > 0 and sc['flow_loop'] > 0:
                        dra_cost += opt['dra_ppm_loop'] * (sc['flow_loop'] * 1000.0 * hours / 1e6) * RateDRA
                    total_cost = power_cost + dra_cost
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
                            f"efficiency_{stn_data['name']}": eff,
                            f"pump_bkw_{stn_data['name']}": pump_bkw,
                            f"motor_kw_{stn_data['name']}": motor_kw,
                            f"power_cost_{stn_data['name']}": power_cost,
                            f"dra_cost_{stn_data['name']}": dra_cost,
                            f"dra_ppm_{stn_data['name']}": opt['dra_ppm_main'],
                            f"dra_ppm_loop_{stn_data['name']}": opt['dra_ppm_loop'],
                            f"drag_reduction_{stn_data['name']}": opt['dra_main'],
                            f"drag_reduction_loop_{stn_data['name']}": opt['dra_loop'],
                        })
                    else:
                        record.update({
                            f"pump_flow_{stn_data['name']}": 0.0,
                            f"num_pumps_{stn_data['name']}": 0,
                            f"speed_{stn_data['name']}": 0.0,
                            f"efficiency_{stn_data['name']}": 0.0,
                            f"pump_bkw_{stn_data['name']}": 0.0,
                            f"motor_kw_{stn_data['name']}": 0.0,
                            f"power_cost_{stn_data['name']}": 0.0,
                            f"dra_cost_{stn_data['name']}": 0.0,
                            f"dra_ppm_{stn_data['name']}": 0.0,
                            f"dra_ppm_loop_{stn_data['name']}": 0.0,
                            f"drag_reduction_{stn_data['name']}": 0.0,
                            f"drag_reduction_loop_{stn_data['name']}": 0.0,
                        })
                    new_cost = state['cost'] + total_cost
                    bucket = round(residual_next, RESIDUAL_ROUND)
                    record[f"bypass_next_{stn_data['name']}"] = 1 if sc.get('bypass_next', False) else 0
                    new_record_list = state['records'] + [record]
                    existing = new_states.get(bucket)
                    flow_next = sc['flow_main'] if sc.get('bypass_next') else flow_total
                    plan_new = state.get('plan', {}).copy()
                    if stn_data['is_pump']:
                        plan_new[stn_data['idx']] = (opt['nop'], opt['rpm'])
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
                            'reach': reach_after,
                            'flow': flow_next,
                            'plan': plan_new,
                        }

        if not new_states:
            return {"error": True, "message": f"No feasible operating point for {stn_data['orig_name']}"}
        # Previously we pruned higher-cost states across residual buckets to
        # limit combinatorial growth.  For full enumeration, retain every bucket
        # and keep only the cheapest entry within each bucket (handled above
        # when populating ``new_states``).
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
    linefill_dict: dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
) -> dict:
    """Apply selected pump types for each station and delegate to ``solve_pipeline``."""

    stations_copy = copy.deepcopy(stations)
    kv_copy = KV_list[:]
    rho_copy = rho_list[:]
    for stn in stations_copy:
        ptypes = stn.pop('pump_types', None)
        if not ptypes:
            continue
        chosen = None
        sel = stn.get('pump_name')
        for pdata in ptypes.values():
            if not isinstance(pdata, dict):
                continue
            names = pdata.get('names', [])
            if sel and names and sel in names:
                chosen = pdata
                break
            if chosen is None and pdata.get('available', 0) > 0:
                chosen = pdata
        if chosen:
            stn.update({
                'head_data': chosen.get('head_data'),
                'eff_data': chosen.get('eff_data'),
                'A': chosen.get('A', stn.get('A', 0.0)),
                'B': chosen.get('B', stn.get('B', 0.0)),
                'C': chosen.get('C', stn.get('C', 0.0)),
                'P': chosen.get('P', stn.get('P', 0.0)),
                'Q': chosen.get('Q', stn.get('Q', 0.0)),
                'R': chosen.get('R', stn.get('R', 0.0)),
                'S': chosen.get('S', stn.get('S', 0.0)),
                'T': chosen.get('T', stn.get('T', 0.0)),
                'power_type': chosen.get('power_type', stn.get('power_type', 'Grid')),
                'rate': chosen.get('rate', stn.get('rate', 0.0)),
                'sfc_mode': chosen.get('sfc_mode', stn.get('sfc_mode', 'manual')),
                'sfc': chosen.get('sfc', stn.get('sfc', 0.0)),
                'engine_params': chosen.get('engine_params', stn.get('engine_params', {})),
                'MinRPM': chosen.get('MinRPM', stn.get('MinRPM', 0.0)),
                'DOL': chosen.get('DOL', stn.get('DOL', 0.0)),
            })

    result = solve_pipeline(
        stations_copy,
        terminal,
        FLOW,
        kv_copy,
        rho_copy,
        RateDRA,
        Price_HSD,
        Fuel_density,
        Ambient_temp,
        linefill_dict,
        dra_reach_km,
        mop_kgcm2,
        hours,
    )
    result['stations_used'] = stations_copy
    return result