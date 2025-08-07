"""Simplified pipeline optimisation model without external solvers.

This module replaces the previous Pyomo/NEOS based optimisation with a
lightweight search that enumerates feasible pump operating points.  The goal is
not to be perfectly optimal but to provide reasonable results using only the
standard Python stack so the application can run in environments where no
solver is available.
"""

from __future__ import annotations

from math import log10, pi
from itertools import product
import copy
import numpy as np

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


def _allowed_values(min_val: int, max_val: int, step: int) -> list[int]:
    vals = list(range(min_val, max_val + 1, step))
    if vals[-1] != max_val:
        vals.append(max_val)
    return vals


def _segment_hydraulics(flow_m3h: float, L: float, d_inner: float, rough: float,
                        kv: float, dra_perc: float) -> tuple[float, float, float, float]:
    """Return (head_loss, velocity, reynolds, friction_factor)."""
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
    head_loss = f * ((L * 1000.0) / d_inner) * (v ** 2 / (2 * g)) * (1 - dra_perc / 100.0)
    return head_loss, v, Re, f


def _pump_bep(stn: dict) -> float:
    """Return the best efficiency flow at DOL for the pump.

    The efficiency curve is defined by a 4th order polynomial in flow ``Q``.
    We compute the real positive root of its derivative that yields the maximum
    efficiency. If coefficients are missing the function returns 0.0 so no
    additional flow constraint is applied.
    """
    P = stn.get('P', 0.0)
    Q = stn.get('Q', 0.0)
    R = stn.get('R', 0.0)
    S = stn.get('S', 0.0)
    # derivative: 4P Q^3 + 3Q Q^2 + 2R Q + S = 0
    coeffs = [4 * P, 3 * Q, 2 * R, S]
    if all(abs(c) < 1e-12 for c in coeffs):
        return 0.0
    roots = np.roots(coeffs)
    roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    if not roots:
        return 0.0
    # select root giving maximum efficiency
    T = stn.get('T', 0.0)
    eff = lambda q: P * q ** 4 + Q * q ** 3 + R * q ** 2 + S * q + T
    best_q = max(roots, key=lambda r: eff(r))
    return float(best_q)


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
# Product movement utilities
# ---------------------------------------------------------------------------

def _segment_volumes(stations: list[dict]) -> list[tuple[float, float]]:
    """Return cumulative volume ranges for each pipeline segment.

    Each entry is a ``(start, end)`` pair representing the volume window (m³)
    occupied by the segment between station ``i-1`` and ``i``.  The start of the
    first segment is 0 and the end of the last segment equals the total linefill
    capacity.
    """

    ranges = []
    cum = 0.0
    default_t = 0.007
    for stn in stations:
        L = float(stn.get('L', 0.0)) * 1000.0  # km -> m
        if 'D' in stn:
            d_inner = stn['D'] - 2 * stn.get('t', default_t)
        else:
            d_inner = stn.get('d', 0.7)
        area = pi * d_inner ** 2 / 4.0
        vol = area * L
        start = cum
        cum += vol
        ranges.append((start, cum))
    return ranges


def _property_at_volume(slugs: list[dict], vol: float) -> tuple[float, float]:
    """Return (kv, rho) for the product occupying volume ``vol``."""
    for slug in slugs:
        if slug['start'] <= vol < slug['end']:
            return slug['kv'], slug['rho']
    # fallback to last slug if volume slightly exceeds due to rounding
    last = slugs[-1]
    return last['kv'], last['rho']


def _properties_from_slugs(slugs: list[dict], seg_ranges: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Return viscosity and density lists for each pipeline segment."""
    kv_list = []
    rho_list = []
    for start, end in seg_ranges:
        mid = (start + end) / 2.0
        kv, rho = _property_at_volume(slugs, mid)
        kv_list.append(kv)
        rho_list.append(rho)
    return kv_list, rho_list


def _advance_slugs(slugs: list[dict], pumped_vol: float, plan: list[dict], capacity: float) -> list[dict]:
    """Shift slugs downstream by ``pumped_vol`` and insert new product.

    ``plan`` is modified in place to remove pumped volume.
    """

    # Shift existing slugs downstream
    for slug in slugs:
        slug['start'] += pumped_vol
        slug['end'] += pumped_vol

    # Insert new volume at the start from the pumping plan
    volume_remaining = pumped_vol
    new_slugs = []
    offset = 0.0
    while volume_remaining > 0 and plan:
        prod = plan[0]
        take = min(prod['volume'], volume_remaining)
        new_slugs.append({
            'start': offset,
            'end': offset + take,
            'kv': prod['kv'],
            'rho': prod['rho'],
        })
        offset += take
        volume_remaining -= take
        prod['volume'] -= take
        if prod['volume'] <= 1e-9:
            plan.pop(0)

    slugs = new_slugs + slugs

    # Remove slugs that have exited the pipeline and trim the last one
    kept: list[dict] = []
    for slug in slugs:
        if slug['start'] >= capacity:
            continue
        if slug['end'] > capacity:
            slug['end'] = capacity
        kept.append(slug)

    # Merge adjacent slugs with identical properties
    merged: list[dict] = []
    for slug in kept:
        if merged and merged[-1]['kv'] == slug['kv'] and merged[-1]['rho'] == slug['rho'] and abs(merged[-1]['end'] - slug['start']) < 1e-9:
            merged[-1]['end'] = slug['end']
        else:
            merged.append(slug)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
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
    linefill: list[dict] | None = None,
    hours: float = 24.0,
) -> dict:
    """Greedy search over pump settings to minimise operating cost.

    Parameters
    ----------
    hours: float
        Duration (in hours) for which the pipeline operates at the supplied
        flow rate.  Defaults to 24 to mimic continuous daily operation but can
        be lowered when the optimiser is driven by a daily throughput target.
    """

    N = len(stations)
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        segment_flows.append(prev_flow - delivery + supply)

    residual = stations[0].get('min_residual', 50.0)
    result = {}
    total_cost = 0.0
    last_maop_head = 0.0
    last_maop_kg = 0.0

    default_t = 0.007
    default_e = 0.00004

    for i, stn in enumerate(stations, start=1):
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

        # Maximum allowable operating pressure (Barlow's formula)
        SMYS = stn.get('SMYS', 52000.0)  # psi
        design_factor = 0.72
        maop_psi = 2 * SMYS * design_factor * (thickness / outer_d) if outer_d > 0 else 0.0
        maop_kgcm2 = maop_psi * 0.0703069
        maop_head = maop_kgcm2 * 10000.0 / rho if rho > 0 else 0.0

        # Evaluate options
        if stn.get('is_pump', False):
            min_p = stn.get('min_pumps', 1)
            max_p = stn.get('max_pumps', 2)
            rpm_vals = _allowed_values(int(stn.get('MinRPM', 0)), int(stn.get('DOL', 0)), RPM_STEP)
            dra_vals = _allowed_values(0, int(stn.get('max_dr', 0)), DRA_STEP)
            bep_dol = _pump_bep(stn)
            best = None
            for nop, rpm, dra in product(range(min_p, max_p + 1), rpm_vals, dra_vals):
                # enforce flow limit of 120% of BEP for each pump
                dol = stn.get('DOL', rpm)
                if bep_dol > 0:
                    max_flow_total = 1.2 * bep_dol * rpm / dol * nop
                    if flow > max_flow_total:
                        continue
                head_loss, v, Re, f = _segment_hydraulics(flow, L, d_inner, rough, kv, dra)
                tdh, eff = _pump_head(stn, flow, rpm, nop)
                if tdh <= 0:
                    continue
                elev_i = stn.get('elev', 0.0)
                elev_next = terminal.get('elev', 0.0) if i == N else stations[i].get('elev', 0.0)
                residual_next = residual + tdh - head_loss - (elev_next - elev_i)
                if residual_next < stn.get('min_residual', 50.0):
                    continue
                eff = max(eff, 1e-6)
                hydraulic_kW = (rho * flow * 9.81 * tdh) / (3600.0 * 1000.0)
                bkw = hydraulic_kW / (eff / 100.0) if rpm > 0 else 0.0
                motor_kW = bkw / 0.95
                if motor_kW < 0:
                    continue
                if stn.get('sfc', 0):
                    sfc_val = stn['sfc']
                    fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                    power_cost = motor_kW * hours * fuel_per_kWh * Price_HSD
                else:
                    rate = stn.get('rate', 0.0)
                    power_cost = motor_kW * hours * rate
                ppm = max(get_ppm_for_dr(kv, dra), 0.0)
                dra_cost = ppm * (flow * 1000.0 * hours / 1e6) * RateDRA
                cost = power_cost + dra_cost
                if cost < 0:
                    continue
                if best is None or cost < best['cost']:
                    best = {
                        'nop': nop,
                        'rpm': rpm,
                        'dra': dra,
                        'residual_next': residual_next,
                        'head_loss': head_loss,
                        'v': v,
                        'Re': Re,
                        'f': f,
                        'tdh': tdh,
                        'eff': eff,
                        'power_cost': power_cost,
                        'dra_cost': dra_cost,
                        'dra_ppm': ppm,
                        'bkw': bkw,
                        'motor_kw': motor_kW,
                        'energy_kwh': motor_kW * hours,
                        'cost': cost,
                    }
            if best is None:
                return {"error": True, "message": f"No feasible operating point for {stn['name']}."}
            # Record
            name = stn['name'].strip().lower().replace(' ', '_')
            result[f"pipeline_flow_{name}"] = flow
            result[f"pipeline_flow_in_{name}"] = segment_flows[i - 1]
            result[f"pump_flow_{name}"] = flow
            result[f"num_pumps_{name}"] = best['nop']
            result[f"speed_{name}"] = best['rpm']
            result[f"efficiency_{name}"] = best['eff']
            result[f"power_cost_{name}"] = best['power_cost']
            result[f"dra_cost_{name}"] = best['dra_cost']
            result[f"dra_ppm_{name}"] = best['dra_ppm']
            result[f"drag_reduction_{name}"] = best['dra']
            result[f"bkw_{name}"] = best['bkw']
            result[f"motor_kw_{name}"] = best['motor_kw']
            result[f"kwh_{name}"] = best['energy_kwh']
            result[f"head_loss_{name}"] = best['head_loss']
            result[f"head_loss_kgcm2_{name}"] = head_to_kgcm2(best['head_loss'], rho)
            result[f"residual_head_{name}"] = residual
            result[f"rh_kgcm2_{name}"] = head_to_kgcm2(residual, rho)
            sdh_val = residual + best['tdh']
            result[f"sdh_{name}"] = sdh_val
            result[f"sdh_kgcm2_{name}"] = head_to_kgcm2(sdh_val, rho)
            # expose pump coefficients and speed limits for plotting
            result[f"coef_A_{name}"] = float(stn.get('A', 0.0))
            result[f"coef_B_{name}"] = float(stn.get('B', 0.0))
            result[f"coef_C_{name}"] = float(stn.get('C', 0.0))
            result[f"coef_P_{name}"] = float(stn.get('P', 0.0))
            result[f"coef_Q_{name}"] = float(stn.get('Q', 0.0))
            result[f"coef_R_{name}"] = float(stn.get('R', 0.0))
            result[f"coef_S_{name}"] = float(stn.get('S', 0.0))
            result[f"coef_T_{name}"] = float(stn.get('T', 0.0))
            result[f"min_rpm_{name}"] = int(stn.get('MinRPM', 0))
            result[f"dol_{name}"] = int(stn.get('DOL', 0))
            v = best['v']
            Re = best['Re']
            f = best['f']
            cost = best['cost']
            residual_next = best['residual_next']
        else:
            head_loss, v, Re, f = _segment_hydraulics(flow, L, d_inner, rough, kv, 0.0)
            elev_i = stn.get('elev', 0.0)
            elev_next = terminal.get('elev', 0.0) if i == N else stations[i].get('elev', 0.0)
            residual_next = residual - head_loss - (elev_next - elev_i)
            if residual_next < stn.get('min_residual', 50.0):
                return {"error": True, "message": f"Residual head below minimum after {stn['name']}"}
            result[f"pipeline_flow_{name}"] = flow
            result[f"pipeline_flow_in_{name}"] = segment_flows[i - 1]
            result[f"pump_flow_{name}"] = 0.0
            result[f"num_pumps_{name}"] = 0
            result[f"speed_{name}"] = 0.0
            result[f"efficiency_{name}"] = 0.0
            result[f"power_cost_{name}"] = 0.0
            result[f"dra_cost_{name}"] = 0.0
            result[f"dra_ppm_{name}"] = 0.0
            result[f"drag_reduction_{name}"] = 0.0
            result[f"bkw_{name}"] = 0.0
            result[f"motor_kw_{name}"] = 0.0
            result[f"kwh_{name}"] = 0.0
            result[f"head_loss_{name}"] = head_loss
            result[f"head_loss_kgcm2_{name}"] = head_to_kgcm2(head_loss, rho)
            result[f"residual_head_{name}"] = residual
            result[f"rh_kgcm2_{name}"] = head_to_kgcm2(residual, rho)
            result[f"sdh_{name}"] = residual  # no pump, SDH equals RH
            result[f"sdh_kgcm2_{name}"] = head_to_kgcm2(residual, rho)
            result[f"coef_A_{name}"] = 0.0
            result[f"coef_B_{name}"] = 0.0
            result[f"coef_C_{name}"] = 0.0
            result[f"coef_P_{name}"] = 0.0
            result[f"coef_Q_{name}"] = 0.0
            result[f"coef_R_{name}"] = 0.0
            result[f"coef_S_{name}"] = 0.0
            result[f"coef_T_{name}"] = 0.0
            result[f"min_rpm_{name}"] = 0
            result[f"dol_{name}"] = 0
            cost = 0.0
        result[f"rho_{name}"] = rho
        result[f"maop_{name}"] = maop_head
        result[f"maop_kgcm2_{name}"] = maop_kgcm2
        result[f"velocity_{name}"] = v
        result[f"reynolds_{name}"] = Re
        result[f"friction_{name}"] = f
        total_cost += cost
        residual = residual_next
        last_maop_head = maop_head
        last_maop_kg = maop_kgcm2

    # Terminal summary
    term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": segment_flows[-1],
        f"pipeline_flow_in_{term_name}": segment_flows[-2],
        f"pump_flow_{term_name}": 0.0,
        f"speed_{term_name}": 0.0,
        f"num_pumps_{term_name}": 0,
        f"efficiency_{term_name}": 0.0,
        f"power_cost_{term_name}": 0.0,
        f"dra_cost_{term_name}": 0.0,
        f"dra_ppm_{term_name}": 0.0,
        f"drag_reduction_{term_name}": 0.0,
        f"bkw_{term_name}": 0.0,
        f"motor_kw_{term_name}": 0.0,
        f"kwh_{term_name}": 0.0,
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
    result['operating_hours'] = hours
    result['opt_flow'] = FLOW
    result['error'] = False
    return result


def solve_pipeline_multi_origin(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill: list[dict] | None = None,
    hours: float = 24.0,
) -> dict:
    """Enumerate pump type combinations at the origin and call ``solve_pipeline``.

    The ``hours`` parameter is forwarded to :func:`solve_pipeline` so that cost
    calculations reflect the actual operating duration.
    """

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

        result = solve_pipeline(stations_combo, terminal, FLOW, kv_combo, rho_combo, RateDRA, Price_HSD, linefill, hours)
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


def optimise_throughput(
    stations: list[dict],
    terminal: dict,
    throughput_m3_day: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill: list[dict] | None = None,
) -> dict:
    """Enumerate operating hours to satisfy a daily throughput target.

    The search explores constant-flow operation for blocks of 2 hours up to a
    maximum of 24 hours.  For each candidate duration the corresponding flow
    rate is computed and :func:`solve_pipeline` is invoked.  The least-cost
    feasible schedule is returned along with a simple 2‑hour schedule showing
    the pump settings for each station.
    """

    best_res = None
    best_cost = float('inf')
    for hrs in range(2, 25, 2):
        flow = throughput_m3_day / hrs if hrs > 0 else 0.0
        res = solve_pipeline(
            stations, terminal, flow, KV_list, rho_list, RateDRA, Price_HSD, linefill, hrs
        )
        if res.get("error"):
            continue
        cost = res.get("total_cost", float('inf'))
        if cost < best_cost:
            best_cost = cost
            best_res = res
    if best_res is None:
        return {"error": True, "message": "No feasible operating schedule found."}

    # Build a 3-hourly schedule for reporting
    blocks = int(best_res.get('operating_hours', 0) // 2)
    flow = best_res.get('opt_flow', 0.0)
    schedule: list[dict] = []
    st_keys = [s['name'].strip().lower().replace(' ', '_') for s in stations]
    for b in range(blocks):
        entry = {'block': b + 1, 'duration_h': 2, 'flow_m3h': flow}
        for key in st_keys:
            entry[f'speed_{key}'] = best_res.get(f'speed_{key}', 0.0)
            entry[f'dra_ppm_{key}'] = best_res.get(f'dra_ppm_{key}', 0.0)
            entry[f'num_pumps_{key}'] = best_res.get(f'num_pumps_{key}', 0)
            entry[f'bkw_{key}'] = best_res.get(f'bkw_{key}', 0.0)
            entry[f'motor_kw_{key}'] = best_res.get(f'motor_kw_{key}', 0.0)
            entry[f'kwh_{key}'] = best_res.get(f'kwh_{key}', 0.0)
        schedule.append(entry)
    best_res['schedule'] = schedule

    return best_res


def optimise_throughput_multi_origin(
    stations: list[dict],
    terminal: dict,
    throughput_m3_day: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill: list[dict] | None = None,
) -> dict:
    """Throughput optimisation wrapper for multi-origin pipelines."""

    best_res = None
    best_cost = float('inf')
    for hrs in range(2, 25, 2):
        flow = throughput_m3_day / hrs if hrs > 0 else 0.0
        res = solve_pipeline_multi_origin(
            stations, terminal, flow, KV_list, rho_list, RateDRA, Price_HSD, linefill, hrs
        )
        if res.get("error"):
            continue
        cost = res.get("total_cost", float('inf'))
        if cost < best_cost:
            best_cost = cost
            best_res = res
    if best_res is None:
        return {"error": True, "message": "No feasible operating schedule found."}

    # Attach 2-hour schedule based on stations actually used
    blocks = int(best_res.get('operating_hours', 0) // 2)
    flow = best_res.get('opt_flow', 0.0)
    schedule: list[dict] = []
    st_list = best_res.get('stations_used', stations)
    st_keys = [s['name'].strip().lower().replace(' ', '_') for s in st_list if s.get('is_pump', False)]
    for b in range(blocks):
        entry = {'block': b + 1, 'duration_h': 2, 'flow_m3h': flow}
        for key in st_keys:
            entry[f'speed_{key}'] = best_res.get(f'speed_{key}', 0.0)
            entry[f'dra_ppm_{key}'] = best_res.get(f'dra_ppm_{key}', 0.0)
            entry[f'num_pumps_{key}'] = best_res.get(f'num_pumps_{key}', 0)
            entry[f'bkw_{key}'] = best_res.get(f'bkw_{key}', 0.0)
            entry[f'motor_kw_{key}'] = best_res.get(f'motor_kw_{key}', 0.0)
            entry[f'kwh_{key}'] = best_res.get(f'kwh_{key}', 0.0)
        schedule.append(entry)
    best_res['schedule'] = schedule

    return best_res


def simulate_pumping_plan(
    stations: list[dict],
    terminal: dict,
    flow_m3h: float,
    linefill: list[dict],
    plan: list[dict],
    RateDRA: float,
    Price_HSD: float,
    step_hours: float = 2.0,
    total_hours: float = 24.0,
) -> dict:
    """Simulate product movement and evaluate cost for a pumping plan.

    ``linefill`` and ``plan`` are lists of dictionaries with keys ``volume``
    (m³), ``kv`` (cSt) and ``rho`` (kg/m³).  The plan is modified during the
    simulation so callers should pass a copy if values are reused.
    """

    seg_ranges = _segment_volumes(stations)
    capacity = seg_ranges[-1][1] if seg_ranges else 0.0

    # Build initial slug list from linefill
    slugs: list[dict] = []
    start = 0.0
    for p in linefill:
        vol = float(p.get('volume', 0.0))
        slugs.append({'start': start, 'end': start + vol, 'kv': p.get('kv', 1.0), 'rho': p.get('rho', 800.0)})
        start += vol
    if start < capacity:
        # fill any remaining volume with the last known product
        if slugs:
            last = slugs[-1]
            slugs.append({'start': start, 'end': capacity, 'kv': last['kv'], 'rho': last['rho']})
        else:
            slugs.append({'start': 0.0, 'end': capacity, 'kv': plan[0]['kv'], 'rho': plan[0]['rho']})

    total_cost = 0.0
    schedule: list[dict] = []
    steps = int(total_hours // step_hours)
    for step in range(steps):
        kv_list, rho_list = _properties_from_slugs(slugs, seg_ranges)
        res = solve_pipeline(stations, terminal, flow_m3h, kv_list, rho_list, RateDRA, Price_HSD, None, step_hours)
        if res.get('error'):
            return res
        total_cost += res.get('total_cost', 0.0)

        entry = {'block': step + 1, 'duration_h': step_hours, 'flow_m3h': flow_m3h}
        for stn in stations:
            key = stn['name'].strip().lower().replace(' ', '_')
            if stn.get('is_pump'):
                entry[f'speed_{key}'] = res.get(f'speed_{key}', 0.0)
                entry[f'dra_ppm_{key}'] = res.get(f'dra_ppm_{key}', 0.0)
                entry[f'num_pumps_{key}'] = res.get(f'num_pumps_{key}', 0)
                entry[f'bkw_{key}'] = res.get(f'bkw_{key}', 0.0)
                entry[f'motor_kw_{key}'] = res.get(f'motor_kw_{key}', 0.0)
                entry[f'kwh_{key}'] = res.get(f'kwh_{key}', 0.0)
        schedule.append(entry)

        pumped_vol = flow_m3h * step_hours
        slugs = _advance_slugs(slugs, pumped_vol, plan, capacity)

    return {
        'error': False,
        'total_cost': total_cost,
        'schedule': schedule,
        'operating_hours': steps * step_hours,
        'opt_flow': flow_m3h,
    }


def optimise_pumping_plan(
    stations: list[dict],
    terminal: dict,
    linefill: list[dict],
    plan: list[dict],
    RateDRA: float,
    Price_HSD: float,
) -> dict:
    """Optimise flow rate and hours for a given pumping plan."""

    throughput = sum(p.get('volume', 0.0) for p in plan)
    best = None
    best_cost = float('inf')
    for hrs in range(2, 25, 2):
        flow = throughput / hrs if hrs > 0 else 0.0
        res = simulate_pumping_plan(
            stations,
            terminal,
            flow,
            [dict(p) for p in linefill],
            [dict(p) for p in plan],
            RateDRA,
            Price_HSD,
            step_hours=2.0,
            total_hours=float(hrs),
        )
        if res.get('error'):
            continue
        cost = res.get('total_cost', float('inf'))
        if cost < best_cost:
            best_cost = cost
            best = res
    if best is None:
        return {'error': True, 'message': 'No feasible operating schedule found.'}
    return best
