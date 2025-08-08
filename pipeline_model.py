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
# Utilities
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


def _segment_hydraulics(flow_m3h: float, L: float, d_inner: float, rough: float,
                        kv: float, dra_perc: float) -> tuple[float, float, float, float]:
    """Return (head_loss, velocity, reynolds, friction_factor) with DRA applied to friction.

    DRA is modeled as reducing the Darcy friction factor: f' = f * (1 - DR),
    where DR = dra_perc/100 clipped to [0, 0.95]. This is closer to practice
    than scaling the entire head-loss term.
    """
    g = 9.81
    flow_m3s = flow_m3h / 3600.0
    area = pi * d_inner ** 2 / 4.0
    v = flow_m3s / area if area > 0 else 0.0
    Re = v * d_inner / (kv * 1e-6) if kv > 0 else 0.0

    # Base friction factor (no DRA)
    if Re > 0:
        if Re < 4000:
            f = 64.0 / Re
        else:
            arg = (rough / d_inner / 3.7) + (5.74 / (Re ** 0.9))
            f = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
    else:
        f = 0.0

    # Apply drag reduction to friction factor (clip DR to prevent negative f)
    DR = max(0.0, min(dra_perc / 100.0, 0.95))
    f_eff = f * (1.0 - DR)

    head_loss = f_eff * ((L * 1000.0) / d_inner) * (v ** 2 / (2 * g))
    return head_loss, v, Re, f_eff


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

    This version is conservative and **does not** subtract any hypothetical
    downstream pump head. It accumulates only hydraulic losses, elevation
    changes, and peak constraints, plus the terminal residual requirement.
    """

    from functools import lru_cache

    N = len(stations)

    @lru_cache(None)
    def req_entry(i: int) -> float:
        if i >= N:
            return float(terminal.get('min_residual', 0.0))

        stn = stations[i]
        kv = KV_list[i]
        flow = segment_flows[i]  # flow after station i
        L = stn.get('L', 0.0)
        thickness = stn.get('t', 0.007 if 'D' in stn or 'd' in stn else 0.007)
        d_inner = (stn.get('D', stn.get('d', 0.7)) - 2 * thickness)
        rough = stn.get('rough', 0.00004)
        dra_down = 0.0  # requirement shouldn't assume DRA downstream

        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N - 1 else stations[i + 1].get('elev', 0.0)

        # Base requirement from next station
        downstream = req_entry(i + 1)

        # Loss and elevation requirement across this segment
        head_loss, *_ = _segment_hydraulics(flow, L, d_inner, rough, kv, dra_down)
        req = downstream + head_loss + (elev_next - elev_i)

        # Intermediate peak checks with configurable margin
        peak_margin = stn.get('peak_margin', 25.0)
        peak_req = 0.0
        for peak in (stn.get('peaks', []) or []):
            dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
            elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
            if dist is None or elev_peak is None:
                continue
            head_to_peak, *_ = _segment_hydraulics(flow, float(dist), d_inner, rough, kv, dra_down)
            req_peak = head_to_peak + (float(elev_peak) - elev_i) + float(peak_margin)
            if req_peak > peak_req:
                peak_req = req_peak

        req = max(req, peak_req)

        # Do NOT subtract any downstream pump capability
        return max(req, float(stn.get('min_residual', 0.0)))

    return req_entry(idx + 1)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_pipeline(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill_dict: dict | None = None,
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

    # Pre-compute option lists for each station
    station_opts = []
    origin_enforced = False
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        flow = segment_flows[i]
        kv = KV_list[i - 1]
        rho = rho_list[i - 1]

        min_residual_next = _downstream_requirement(stations, i - 1, terminal, segment_flows, KV_list)

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
        maop_head = maop_kgcm2 * 10000.0 / rho if rho > 0 else 0.0

        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N else stations[i].get('elev', 0.0)
        elev_delta = elev_next - elev_i

        opts = []
        if stn.get('is_pump', False):
            min_p = stn.get('min_pumps', 0)
            if not origin_enforced:
                min_p = max(1, min_p)
                origin_enforced = True
            max_p = stn.get('max_pumps', 2)
            rpm_vals = _allowed_values(int(stn.get('MinRPM', 0)), int(stn.get('DOL', 0)), RPM_STEP)
            dra_vals = _allowed_values(0, int(stn.get('max_dr', 0)), DRA_STEP)
            for nop in range(min_p, max_p + 1):
                rpm_opts = [0] if nop == 0 else rpm_vals
                dra_opts = dra_vals
                for rpm in rpm_opts:
                    for dra in dra_opts:
                        head_loss, v, Re, f = _segment_hydraulics(flow, L, d_inner, rough, kv, dra)
                        # Velocity cap enforcement (default 3 m/s if not provided)
                        v_max = float(stn.get('v_max', 3.0))
                        if v_max > 0 and v > v_max:
                            continue
                        if nop > 0 and rpm > 0:
                            tdh, eff = _pump_head(stn, flow, rpm, nop)
                        else:
                            tdh, eff = 0.0, 0.0
                        eff = max(eff, 1e-6) if nop > 0 else 0.0
                        if nop > 0 and rpm > 0:
                            pump_bkw_total = (rho * flow * 9.81 * tdh) / (3600.0 * 1000.0 * (eff / 100.0))
                            pump_bkw = pump_bkw_total / nop
                            motor_kw_total = pump_bkw_total / 0.95
                            motor_kw = motor_kw_total / nop
                        else:
                            pump_bkw = motor_kw = motor_kw_total = 0.0
                        if stn.get('sfc', 0) and motor_kw_total > 0:
                            sfc_val = stn['sfc']
                            fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                            power_cost = motor_kw_total * 24.0 * fuel_per_kWh * Price_HSD
                        else:
                            rate = stn.get('rate', 0.0)
                            power_cost = motor_kw_total * 24.0 * rate
                        ppm = get_ppm_for_dr(kv, dra) if dra > 0 else 0.0
                        dra_cost = (ppm / 1e6) * flow * 24.0 * RateDRA if dra > 0 else 0.0
                        opts.append({
                            'nop': nop,
                            'rpm': rpm,
                            'dra': dra,
                            'tdh': tdh,
                            'eff': eff,
                            'pump_bkw': pump_bkw,
                            'motor_kw': motor_kw,
                            'power_cost': power_cost,
                            'dra_cost': dra_cost,
                            'dra_ppm': ppm,
                            'cost': power_cost + dra_cost,
                            'head_loss': head_loss,
                            'v': v,
                            'Re': Re,
                            'f': f,
                        })
        else:
            head_loss, v, Re, f = _segment_hydraulics(flow, L, d_inner, rough, kv, 0.0)
            opts.append({
                'nop': 0,
                'rpm': 0,
                'dra': 0,
                'tdh': 0.0,
                'eff': 0.0,
                'pump_bkw': 0.0,
                'motor_kw': 0.0,
                'power_cost': 0.0,
                'dra_cost': 0.0,
                'dra_ppm': 0.0,
                'cost': 0.0,
                'head_loss': head_loss,
                'v': v,
                'Re': Re,
                'f': f,
            })

        station_opts.append({
            'name': name,
            'orig_name': stn['name'],
            'flow': flow,
            'flow_in': segment_flows[i - 1],
            'kv': kv,
            'rho': rho,
            'maop_head': maop_head,
            'maop_kgcm2': maop_kgcm2,
            'maop_margin': float(stn.get('maop_margin', 0.0)),
            'elev_delta': elev_delta,
            'min_residual_next': min_residual_next,
            'options': opts,
            'is_pump': bool(stn.get('is_pump', False)),
            'L': L,
            'd_inner': d_inner,
            'rough': rough,
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

    # Dynamic programming over stations
    init_residual = stations[0].get('min_residual', 50.0)
    states: dict[float, dict] = {
        round(init_residual, 2): {
            'cost': 0.0,
            'residual': init_residual,
            'records': [],
            'last_maop': 0.0,
            'last_maop_kg': 0.0
        }
    }

    for stn_data in station_opts:
        new_states: dict[float, dict] = {}
        for state in states.values():
            for opt in stn_data['options']:
                sdh = state['residual'] + opt['tdh']
                # MAOP enforcement (sdh must not exceed MAOP - margin)
                if stn_data['maop_head'] > 0 and sdh > (stn_data['maop_head'] - stn_data.get('maop_margin', 0.0)):
                    continue
                residual_next = sdh - opt['head_loss'] - stn_data['elev_delta']
                if residual_next < stn_data['min_residual_next']:
                    continue
                record = {
                    f"pipeline_flow_{stn_data['name']}": stn_data['flow'],
                    f"pipeline_flow_in_{stn_data['name']}": stn_data['flow_in'],
                    f"head_loss_{stn_data['name']}": opt['head_loss'],
                    f"head_loss_kgcm2_{stn_data['name']}": head_to_kgcm2(opt['head_loss'], stn_data['rho']),
                    f"residual_head_{stn_data['name']}": state['residual'],
                    f"rh_kgcm2_{stn_data['name']}": head_to_kgcm2(state['residual'], stn_data['rho']),
                    f"sdh_{stn_data['name']}": sdh if stn_data['is_pump'] else state['residual'],
                    f"sdh_kgcm2_{stn_data['name']}": head_to_kgcm2(sdh if stn_data['is_pump'] else state['residual'], stn_data['rho']),
                    f"rho_{stn_data['name']}": stn_data['rho'],
                    f"maop_{stn_data['name']}": stn_data['maop_head'],
                    f"maop_kgcm2_{stn_data['name']}": stn_data['maop_kgcm2'],
                    f"velocity_{stn_data['name']}": opt['v'],
                    f"reynolds_{stn_data['name']}": opt['Re'],
                    f"friction_{stn_data['name']}": opt['f'],
                    f"num_pumps_{stn_data['name']}": opt['nop'],
                    f"speed_{stn_data['name']}": opt['rpm'],
                    f"pump_flow_{stn_data['name']}": stn_data['flow'] if opt['nop'] > 0 else 0.0,
                    f"efficiency_{stn_data['name']}": opt['eff'],
                    f"pump_bkw_{stn_data['name']}": opt['pump_bkw'],
                    f"motor_kw_{stn_data['name']}": opt['motor_kw'],
                    f"power_cost_{stn_data['name']}": opt['power_cost'],
                    f"dra_cost_{stn_data['name']}": opt['dra_cost'],
                    f"dra_ppm_{stn_data['name']}": opt['dra_ppm'],
                    f"drag_reduction_{stn_data['name']}": opt['dra'],
                }
                new_cost = state['cost'] + opt['cost']
                bucket = round(residual_next, RESIDUAL_ROUND)
                new_record_list = state['records'] + [record]
                if bucket not in new_states or new_cost < new_states[bucket]['cost']:
                    new_states[bucket] = {
                        'cost': new_cost,
                        'residual': residual_next,
                        'records': new_record_list,
                        'last_maop': stn_data['maop_head'],
                        'last_maop_kg': stn_data['maop_kgcm2'],
                    }
        if not new_states:
            return {"error": True, "message": f"No feasible operating point for {stn_data['orig_name']}"}
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
        f"drag_reduction_{term_name}": 0.0,
        f"head_loss_{term_name}": 0.0,
        f"velocity_{term_name}": 0.0,
        f"reynolds_{term_name}": 0.0,
        f"friction_{term_name}": 0.0,
        f"residual_head_{term_name}": residual,
        f"rh_kgcm2_{term_name}": head_to_kgcm2(residual, rho_list[-1]),
        f"sdh_{term_name}": residual,
        f"sdh_kgcm2_{term_name}": head_to_kgcm2(residual, rho_list[-1]),
        "total_cost": total_cost,
        "last_maop": last_maop_head,
        "last_maop_kgcm2": last_maop_kg,
    })
    return result


# ---------------------------------------------------------------------------
# Multi-origin (A/B types at origin) wrapper
# ---------------------------------------------------------------------------

def solve_pipeline_multi_origin(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill_dict: dict | None = None,
) -> dict:
    """Enumerate pump type combinations at the origin and call ``solve_pipeline``."""

    origin_index = next(i for i, s in enumerate(stations) if s.get('is_pump', False))
    origin_station = stations[origin_index]
    pump_types = origin_station.get('pump_types', {})
    combos = generate_origin_combinations(
        pump_types.get('A', {}).get('available', 2),
        pump_types.get('B', {}).get('available', 2),
    )

    best_cost = float('inf')
    best_result = None
    best_stations = None

    for (numA, numB) in combos:
        if numA + numB < 1:
            continue

        stations_combo = []
        kv_combo = []
        rho_combo = []

        # Expand origin into virtual serial units of A then B (L = 0 except final)
        pump_units = []
        for label, count in (('A', numA), ('B', numB)):
            for _ in range(count):
                unit = copy.deepcopy(origin_station)
                unit['name'] = f"{origin_station['name']}_{label}"
                unit['is_pump'] = True
                unit['L'] = 0.0
                unit['max_dr'] = origin_station.get('max_dr', 0.0)
                # Adopt that type's curve
                if label in pump_types:
                    curve = pump_types[label]
                    unit.update({
                        'A': curve.get('A', unit.get('A', 0.0)),
                        'B': curve.get('B', unit.get('B', 0.0)),
                        'C': curve.get('C', unit.get('C', 0.0)),
                        'P': curve.get('P', unit.get('P', 0.0)),
                        'Q': curve.get('Q', unit.get('Q', 0.0)),
                        'R': curve.get('R', unit.get('R', 0.0)),
                        'S': curve.get('S', unit.get('S', 0.0)),
                        'T': curve.get('T', unit.get('T', 0.0)),
                    })
                pump_units.append(unit)

        # The final origin segment takes the original distance
        if pump_units:
            pump_units[-1]['L'] = origin_station.get('L', 0.0)
            pump_units[-1]['max_dr'] = origin_station.get('max_dr', 0.0)

        stations_combo.extend(pump_units)
        stations_combo.extend(copy.deepcopy(stations[origin_index + 1:]))
        kv_combo.extend(KV_list[1:])
        rho_combo.extend(rho_list[1:])

        result = solve_pipeline(stations_combo, terminal, FLOW, kv_combo, rho_combo, RateDRA, Price_HSD, linefill_dict)
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
