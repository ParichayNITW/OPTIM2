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
    linefill_dict: dict | None = None,
) -> dict:
    """Greedy search over pump settings to minimise operating cost."""

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
            best = None
            for nop, rpm, dra in product(range(min_p, max_p + 1), rpm_vals, dra_vals):
                head_loss, v, Re, f = _segment_hydraulics(flow, L, d_inner, rough, kv, dra)
                tdh, eff = _pump_head(stn, flow, rpm, nop)
                elev_i = stn.get('elev', 0.0)
                elev_next = terminal.get('elev', 0.0) if i == N else stations[i].get('elev', 0.0)
                residual_next = residual + tdh - head_loss - (elev_next - elev_i)
                if residual_next < stn.get('min_residual', 50.0):
                    continue
                eff = max(eff, 1e-6)
                power_kW = (rho * flow * 9.81 * tdh) / (3600.0 * 1000.0 * (eff / 100.0) * 0.95) if rpm > 0 else 0.0
                if stn.get('sfc', 0):
                    sfc_val = stn['sfc']
                    fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                    power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
                else:
                    rate = stn.get('rate', 0.0)
                    power_cost = power_kW * 24.0 * rate
                ppm = get_ppm_for_dr(kv, dra)
                dra_cost = ppm * (flow * 1000.0 * 24.0 / 1e6) * RateDRA
                cost = power_cost + dra_cost
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
            result[f"head_loss_{name}"] = best['head_loss']
            result[f"head_loss_kgcm2_{name}"] = head_to_kgcm2(best['head_loss'], rho)
            result[f"residual_head_{name}"] = residual
            result[f"rh_kgcm2_{name}"] = head_to_kgcm2(residual, rho)
            sdh_val = residual + best['tdh']
            result[f"sdh_{name}"] = sdh_val
            result[f"sdh_kgcm2_{name}"] = head_to_kgcm2(sdh_val, rho)
            result[f"maop_{name}"] = maop_head
            result[f"maop_kgcm2_{name}"] = maop_kgcm2
            result[f"velocity_{name}"] = best['v']
            result[f"reynolds_{name}"] = best['Re']
            result[f"friction_{name}"] = best['f']
            total_cost += best['cost']
            residual = best['residual_next']
            last_maop_head = maop_head
            last_maop_kg = maop_kgcm2
        else:
            head_loss, v, Re, f = _segment_hydraulics(flow, L, d_inner, rough, kv, 0.0)
            elev_i = stn.get('elev', 0.0)
            elev_next = terminal.get('elev', 0.0) if i == N else stations[i].get('elev', 0.0)
            residual_next = residual - head_loss - (elev_next - elev_i)
            if residual_next < stn.get('min_residual', 50.0):
                return {"error": True, "message": f"Residual head below minimum after {stn['name']}"}
            name = stn['name'].strip().lower().replace(' ', '_')
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
            result[f"head_loss_{name}"] = head_loss
            result[f"head_loss_kgcm2_{name}"] = head_to_kgcm2(head_loss, rho)
            result[f"residual_head_{name}"] = residual
            result[f"rh_kgcm2_{name}"] = head_to_kgcm2(residual, rho)
            result[f"sdh_{name}"] = residual  # no pump, SDH equals RH
            result[f"sdh_kgcm2_{name}"] = head_to_kgcm2(residual, rho)
            result[f"maop_{name}"] = maop_head
            result[f"maop_kgcm2_{name}"] = maop_kgcm2
            result[f"velocity_{name}"] = v
            result[f"reynolds_{name}"] = Re
            result[f"friction_{name}"] = f
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
        f"head_loss_{term_name}": 0.0,
        f"velocity_{term_name}": 0.0,
        f"reynolds_{term_name}": 0.0,
        f"friction_{term_name}": 0.0,
        f"sdh_{term_name}": residual,
        f"residual_head_{term_name}": residual,
    })
    rho_term = rho_list[-1]
    result[f"rh_kgcm2_{term_name}"] = head_to_kgcm2(residual, rho_term)
    result[f"sdh_kgcm2_{term_name}"] = head_to_kgcm2(residual, rho_term)
    result[f"maop_{term_name}"] = last_maop_head
    result[f"maop_kgcm2_{term_name}"] = last_maop_kg
    result['total_cost'] = total_cost
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
    linefill_dict: dict | None = None,
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
