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

from dra_utils import get_ppm_for_dr

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def head_to_kgcm2(head_m: float, rho: float) -> float:
    """Convert a head value in metres to kg/cm²."""
    return head_m * rho / 10000.0


def generate_origin_combinations(maxA: int = 2, maxB: int = 2) -> list[tuple[int, int]]:
    """Return all feasible pump count combinations for the origin station."""
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

    g = 9.81
    flow_m3s = flow_m3h / 3600.0
    area = pi * d_inner ** 2 / 4.0
    v = flow_m3s / area if area > 0 else 0.0
    Re = v * d_inner / (kv * 1e-6) if kv > 0 else 0.0
    # Churchill (1977) explicit friction factor, valid for all Re
    if Re > 0 and d_inner > 0:
        rr = abs(rough / d_inner) if d_inner > 0 else 0.0
        # Use log safely; rr >= 0
        try:
            termA = (2.457 * math.log((7.0/Re)**0.9 + 0.27*rr))**16
            termB = (37530.0/Re)**16
            denom = (termA + termB)**1.5
            inv = ( (8.0/Re)**12 + (1.0/denom) )
            f = 8.0 * (inv ** (1.0/12.0))
        except ValueError:
            # Fallback to Swamee-Jain if numerical issue arises
            arg = (rr / 3.7) + (5.74 / (Re ** 0.9))
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

    return head_loss, v, Re, f


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


# ---------------------------------------------------------------------------
# Downstream requirements
# ---------------------------------------------------------------------------

def _downstream_requirement(
    stations: list[dict],
    idx: int,
    terminal: dict,
    segment_flows: list[float],
    KV_list: list[float],
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
        kv = KV_list[i]
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

        head_loss, *_ = _segment_hydraulics(flow, L, d_inner, rough, kv, dra_down, None)
        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N - 1 else stations[i + 1].get('elev', 0.0)
        downstream = req_entry(i + 1)
        req = downstream + head_loss + (elev_next - elev_i)

        # Check intermediate peaks within this segment.  Each peak requires enough
        # upstream pressure to maintain at least 25 m of residual head at the peak
        # itself.  Use the maximum requirement among all peaks and the downstream
        # station.
        peak_req = 0.0
        for peak in stn.get('peaks', []) or []:
            dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
            elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
            if dist is None or elev_peak is None:
                continue
            head_peak, *_ = _segment_hydraulics(flow, float(dist), d_inner, rough, kv, dra_down)
            req_peak = head_peak + (float(elev_peak) - elev_i) + 25.0
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

    _RPM_STEP_orig = globals().get('RPM_STEP', 100)
    _DRA_STEP_orig = globals().get('DRA_STEP', 5)
    globals()['RPM_STEP'] = rp_step
    global ADAPTIVE_IN_PROGRESS
    _was_adaptive = ADAPTIVE_IN_PROGRESS
    # Only outermost call performs fine pass
    do_fine = not ADAPTIVE_IN_PROGRESS
    globals()['DRA_STEP'] = dr_step
    # ==== Fine pass around coarse solution ====
    if do_fine:
        ADAPTIVE_IN_PROGRESS = True
    try:
        focus = {}
        if isinstance(result, dict):
            for i, stn in enumerate(stations, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ', '_')
                rpm0 = float(result.get(f'rpm_{key}', 0.0) or 0.0)
                dra0 = float(result.get(f'dra_{key}', 0.0) or 0.0)
                if rpm0 > 0 or dra0 > 0:
                    focus[i] = (rpm0, dra0)
        globals()['RPM_STEP'] = fine_rp_step
        globals()['DRA_STEP'] = fine_dr_step
        saved_bounds = []
        for i, stn in enumerate(stations, start=1):
            if not stn.get('is_pump', False):
                saved_bounds.append(None); continue
            saved_bounds.append((stn.get('MinRPM'), stn.get('DOL')))
            if i in focus:
                rpm0 = focus[i][0]
                lo = max(int(stn.get('MinRPM', 0)), int(rpm0 - 100))
                hi = min(int(stn.get('DOL', 0)), int(rpm0 + 100))
                if hi >= lo:
                    stn['MinRPM'] = lo
                    stn['DOL'] = hi
        res2 = solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict, dra_reach_km, mop_kgcm2, hours)
        if not res2.get('error') and res2.get('total_cost', float('inf')) < result.get('total_cost', float('inf')):
            result = res2
    finally:
        globals()['RPM_STEP'] = _RPM_STEP_orig
        globals()['DRA_STEP'] = _DRA_STEP_orig
        try:
            for (stn, ob) in zip(stations, saved_bounds):
                if ob is None: continue
                stn['MinRPM'], stn['DOL'] = ob
        except Exception:
            pass
def solve_pipeline_multi_origin(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill_dict: dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
) -> dict:
    """Enumerate pump type combinations at the origin and call ``solve_pipeline``."""

    origin_index = next(i for i, s in enumerate(stations) if s.get('is_pump', False))
    origin_station = stations[origin_index]
    pump_types = origin_station.get('pump_types', {})
    combos = generate_origin_combinations(
        pump_types.get('A', {}).get('available', 0),
        pump_types.get('B', {}).get('available', 0),
    )

    best_result = None
    best_cost = float('inf')
    best_stations = None

    for numA, numB in combos:
        if numA > 0 and not pump_types.get('A'):
            continue
        if numB > 0 and not pump_types.get('B'):
            continue

        stations_combo: list[dict] = []
        kv_combo: list[float] = []
        rho_combo: list[float] = []

        name_base = origin_station['name']
        pump_units = []
        for ptype, count in [('A', numA), ('B', numB)]:
            pdata = pump_types.get(ptype)
            for n in range(count):
                unit = {
                    'name': f"{name_base}_{ptype}{n + 1}",
                    'elev': origin_station.get('elev', 0.0),
                    'D': origin_station.get('D'),
                    't': origin_station.get('t'),
                    'SMYS': origin_station.get('SMYS'),
                    'rough': origin_station.get('rough'),
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
                    'sfc': pdata.get('sfc', 0.0) if pdata else 0.0,
                    'MinRPM': pdata.get('MinRPM', 0.0) if pdata else 0.0,
                    'DOL': pdata.get('DOL', 0.0) if pdata else 0.0,
                    'max_pumps': 1,
                    'min_pumps': 1,
                    'delivery': 0.0,
                    'supply': 0.0,
                    'max_dr': 0.0,
                }
                pump_units.append(unit)
                kv_combo.append(KV_list[0])
                rho_combo.append(rho_list[0])

        if not pump_units:
            continue

        pump_units[0]['delivery'] = origin_station.get('delivery', 0.0)
        pump_units[0]['supply'] = origin_station.get('supply', 0.0)
        pump_units[0]['min_residual'] = origin_station.get('min_residual', 50.0)
        pump_units[-1]['L'] = origin_station.get('L', 0.0)
        pump_units[-1]['max_dr'] = origin_station.get('max_dr', 0.0)

        stations_combo.extend(pump_units)
        stations_combo.extend(copy.deepcopy(stations[origin_index + 1:]))
        kv_combo.extend(KV_list[1:])
        rho_combo.extend(rho_list[1:])

        result = solve_pipeline(stations_combo, terminal, FLOW, kv_combo, rho_combo, RateDRA, Price_HSD, linefill_dict, dra_reach_km, mop_kgcm2, hours)
        if result.get("error"):
            continue
        cost = result.get("total_cost", float('inf'))
        if cost < best_cost:
            best_cost = cost
            best_result = result
            best_stations = stations_combo
            best_result['pump_combo'] = {'A': numA, 'B': numB}

    if best_result is None:
        return {
            "error": True,
            "message": "No feasible pump combination found for originating station.",
        }

    best_result['stations_used'] = best_stations
    return best_result
