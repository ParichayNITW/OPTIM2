# pipeline_model.py
# -----------------------------------------------------------------------------
# Discrete global-minimum pipeline optimizer with Pareto-front DP.
# - Objective: minimize TOTAL COST = power/fuel cost + DRA cost (fully included).
# - Hydraulics: Darcy–Weisbach head loss; laminar (64/Re) & turbulent (Haaland).
# - DRA physics: reduces Darcy friction factor (f' = f * (1 - DR)), DR in [0,0.95].
# - Constraints: MAOP (with margin), velocity cap, per-station min residual,
#                elevation + peak margins, terminal residual.
# - RPM bounds: STRICT MinRPM ≤ rpm ≤ DOL, per-station and per origin type A/B.
# - Inputs honored: rpm_list / RPM_STEP, dra_list / DRA_STEP, max_dr, min/max pumps,
#                   power_type (Grid/Diesel), rate, sfc + rate_hsd, v_max, SMYS, t,
#                   D or d (pipe size), rough, min_residual, peaks/peak_margin, etc.
# - Multi-origin: expands origin into serial A/B units; KV/rho arrays replicated.
# - Global optimality (within the discrete grids) via exhaustive enumeration +
#   safe dominance pruning (Pareto by cost↑, residual↓). No bucketing, no heuristics.
#
# Requires: dra_utils.get_ppm_for_dr(kv_cst: float, dra_percent: float) -> float
# -----------------------------------------------------------------------------

from __future__ import annotations

from math import log10, pi
import copy
from typing import List, Dict, Tuple

from dra_utils import get_ppm_for_dr


# ------------------------------ Tunables -------------------------------------

# Defaults (can be overridden per-station or per-type via 'RPM_STEP'/'DRA_STEP'
# or via explicit 'rpm_list'/'dra_list')
RPM_STEP_DEFAULT = 100        # RPM discretization step
DRA_STEP_DEFAULT = 5          # %DR discretization step
VEL_DEFAULT_MAX = 3.0         # m/s cap (if not provided at station)
MOTOR_EFF = 0.95              # overall motor/drive efficiency


# --------------------------- Helper conversions -------------------------------

def head_to_kgcm2(head_m: float, rho: float) -> float:
    """Convert head (m) to kg/cm² for the same fluid."""
    return head_m * rho / 10000.0


def _allowed_values(min_val: float | int, max_val: float | int, step: int) -> list[int]:
    """Inclusive grid from min to max with given step; always includes max."""
    lo = int(round(min_val))
    hi = int(round(max_val))
    st = max(1, int(step))
    if hi < lo:
        return [lo]
    vals = list(range(lo, hi + 1, st))
    if vals[-1] != hi:
        vals.append(hi)
    return vals


# --------------------------- Hydraulics & Pumps -------------------------------

def _segment_hydraulics(flow_m3h: float, L_km: float, d_inner: float, rough: float,
                        kv_cst: float, dra_perc: float) -> tuple[float, float, float, float]:
    """
    Return (head_loss[m], velocity[m/s], Re[-], f_eff[-]) for a segment.
    Darcy–Weisbach: h_f = f * (L/D) * (v^2 / 2g).  f: 64/Re (laminar) or Haaland (turb).
    DRA reduces friction: f' = f * (1 - DR), with DR in [0,0.95].
    """
    g = 9.81
    flow_m3s = flow_m3h / 3600.0
    area = pi * d_inner ** 2 / 4.0 if d_inner > 0 else 0.0
    v = flow_m3s / area if area > 0 else 0.0
    Re = v * d_inner / (kv_cst * 1e-6) if kv_cst > 0 else 0.0

    # Base friction factor
    if Re > 0:
        if Re < 4000.0:
            f = 64.0 / Re
        else:
            arg = (rough / d_inner / 3.7) + (5.74 / (Re ** 0.9))
            f = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
    else:
        f = 0.0

    DR = max(0.0, min(dra_perc / 100.0, 0.95))
    f_eff = f * (1.0 - DR)
    head_loss = f_eff * ((L_km * 1000.0) / max(d_inner, 1e-9)) * (v ** 2 / (2.0 * g))
    return head_loss, v, Re, f_eff


def _pump_head_and_eff(stn: dict, flow_m3h: float, rpm: float, nop: int) -> tuple[float, float]:
    """
    Pump head & efficiency at given flow, rpm, and number of pumps (similarity laws).
       Qe = Q * (DOL / rpm)
       H_rpm(Q) = (A*Qe^2 + B*Qe + C) * (rpm/DOL)^2
       TDH_total = H_rpm * nop
       Eff_rpm(Q) = P*Qe^4 + Q*Qe^3 + R*Qe^2 + S*Qe + T
    """
    dol = float(stn.get('DOL', rpm if rpm else 1.0))
    Qe = flow_m3h * dol / rpm if rpm > 0 else flow_m3h

    A = float(stn.get('A', 0.0)); B = float(stn.get('B', 0.0)); C = float(stn.get('C', 0.0))
    tdh_single = A * Qe ** 2 + B * Qe + C
    tdh = tdh_single * (rpm / dol) ** 2 * nop if dol > 0 else 0.0

    P = float(stn.get('P', 0.0)); Qc = float(stn.get('Q', 0.0)); R = float(stn.get('R', 0.0))
    S = float(stn.get('S', 0.0)); T = float(stn.get('T', 0.0))
    eff = P * Qe ** 4 + Qc * Qe ** 3 + R * Qe ** 2 + S * Qe + T
    return tdh, eff


# ----------------------- Peak margin feasibility check ------------------------

def _peaks_feasible(stn: dict, flow_m3h: float, kv: float, d_inner: float, rough: float,
                    elev_in: float, sdh: float, dra_perc: float) -> bool:
    """Check intermediate elevation peaks with margin, using actual SDH at the station."""
    peaks = (stn.get('peaks', []) or [])
    if not peaks:
        return True
    margin = float(stn.get('peak_margin', 25.0))
    for peak in peaks:
        dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location') or peak.get('dist_km')
        elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation') or peak.get('elev_m')
        if dist is None or elev_peak is None:
            continue
        hl_to_peak, *_ = _segment_hydraulics(flow_m3h, float(dist), d_inner, rough, kv, dra_perc)
        required_at_peak = (float(elev_peak) - float(elev_in)) + float(margin) + hl_to_peak
        if sdh + 1e-9 < required_at_peak:
            return False
    return True


# ---------------------------- Enumerations -----------------------------------

def _enumerate_rpm_list(entity: dict) -> List[int]:
    """
    RPM grid priority:
      1) explicit 'rpm_list'
      2) MinRPM..DOL using step = entity['RPM_STEP'] or default
    Always clipped to [MinRPM, DOL] (inclusive).
    """
    minrpm = float(entity.get('MinRPM', 0.0))
    maxrpm = float(entity.get('DOL', 0.0))
    if minrpm < 0: minrpm = 0.0
    if maxrpm < minrpm: maxrpm = minrpm

    # Explicit list
    if isinstance(entity.get('rpm_list'), list) and entity['rpm_list']:
        lst = sorted(set(int(round(x)) for x in entity['rpm_list']))
        return [r for r in lst if minrpm <= r <= maxrpm]

    # Stepped grid
    step = int(entity.get('RPM_STEP', RPM_STEP_DEFAULT))
    return _allowed_values(minrpm, maxrpm, step) if maxrpm > 0 else [0]


def _enumerate_dra_list(entity: dict) -> List[int]:
    """
    DRA grid priority:
      1) explicit 'dra_list'
      2) 0..max_dr using step = entity['DRA_STEP'] or default
    """
    if isinstance(entity.get('dra_list'), list) and entity['dra_list']:
        return sorted(set(max(0, int(round(x))) for x in entity['dra_list']))
    step = int(entity.get('DRA_STEP', DRA_STEP_DEFAULT))
    max_dr = int(round(float(entity.get('max_dr', 0))))
    return _allowed_values(0, max_dr, step)


# ---------------------------- Option generation ------------------------------

def _build_station_options(
    stations: list[dict],
    KV_list: list[float],
    rho_list: list[float],
    FLOW: float,
) -> tuple[list[dict], list[float]]:
    """Enumerate feasible options (per station) honoring *all* user inputs."""
    N = len(stations)

    # Flow profile after each station considering supplies/deliveries
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        segment_flows.append(segment_flows[-1] - delivery + supply)

    opts_all = []
    origin_seen = False
    default_t = 0.007
    default_e = 0.00004

    for i, stn in enumerate(stations, start=1):
        flow = segment_flows[i]
        kv = float(KV_list[i - 1])
        rho = float(rho_list[i - 1])

        # Geometry
        if 'D' in stn:
            thickness = float(stn.get('t', default_t))
            d_inner = float(stn['D']) - 2.0 * thickness
            outer_d = float(stn['D'])
        else:
            d_inner = float(stn.get('d', 0.7))
            outer_d = float(stn.get('d', 0.7))
            thickness = float(stn.get('t', default_t))
        rough = float(stn.get('rough', default_e))
        L_km = float(stn.get('L', 0.0))

        # Elevations
        elev_in = float(stn.get('elev', 0.0))
        if i < N:
            elev_out = float(stations[i].get('elev', elev_in))
        else:
            elev_out = float(stn.get('elev_out', elev_in))
        elev_delta = elev_out - elev_in

        # MAOP (thin-wall; default design factor 0.72 if not given)
        SMYS = float(stn.get('SMYS', 52000.0))
        design_factor = float(stn.get('design_factor', 0.72))
        maop_psi = 2.0 * SMYS * design_factor * (thickness / max(outer_d, 1e-9))
        maop_kgcm2 = maop_psi * 0.0703069
        maop_head = maop_kgcm2 * 10000.0 / max(rho, 1e-9)
        maop_margin = float(stn.get('maop_margin', 0.0))

        v_max = float(stn.get('v_max', VEL_DEFAULT_MAX))

        opts = []
        if stn.get('is_pump', False):
            min_p = int(stn.get('min_pumps', 0))
            if not origin_seen:
                min_p = max(1, min_p)  # origin must have ≥1 pump running
                origin_seen = True
            max_p = int(stn.get('max_pumps', 2))

            rpm_vals = _enumerate_rpm_list(stn)
            dra_vals = _enumerate_dra_list(stn)

            for nop in range(min_p, max_p + 1):
                for rpm in (rpm_vals if nop > 0 else [0]):
                    # strict bounds guard (inclusive)
                    if nop > 0 and (rpm < float(stn.get('MinRPM', 0.0)) or rpm > float(stn.get('DOL', 0.0))):
                        continue
                    for dra in dra_vals:
                        # Segment hydraulics under this DRA
                        head_loss, v, Re, f_eff = _segment_hydraulics(flow, L_km, d_inner, rough, kv, dra)
                        if v_max > 0.0 and v > v_max:
                            continue

                        # Pump physics
                        if nop > 0 and rpm > 0:
                            tdh, eff = _pump_head_and_eff(stn, flow, rpm, nop)
                            eff = max(eff, 1e-6)
                        else:
                            tdh, eff = 0.0, 0.0

                        # Power & DRA cost
                        if nop > 0 and rpm > 0:
                            hyd_W = (rho * (flow / 3600.0) * 9.81 * tdh) / max(eff / 100.0, 1e-9)  # hydraulic W for all pumps
                            motor_kw_total = hyd_W / 1000.0 / MOTOR_EFF
                            pump_bkw = hyd_W / 1000.0 / max(nop, 1)
                            motor_kw = motor_kw_total / max(nop, 1)
                        else:
                            pump_bkw = motor_kw = motor_kw_total = 0.0

                        # Energy cost/day
                        ptype = stn.get('power_type', 'Grid')
                        if ptype == 'Diesel' and motor_kw_total > 0:
                            # Diesel: use SFC (g/bhp-hr) and diesel price (₹/L)
                            sfc_val = float(stn.get('sfc', 150.0))
                            fuel_per_kWh = (sfc_val * 1.34102) / 820.0  # L/kWh
                            rate_hsd = float(stn.get('rate_hsd', 0.0))   # ₹/L
                            power_cost = motor_kw_total * 24.0 * fuel_per_kWh * rate_hsd
                        else:
                            # Grid
                            rate_elec = float(stn.get('rate', 0.0))      # ₹/kWh
                            power_cost = motor_kw_total * 24.0 * rate_elec

                        # DRA cost/day (ppm→fraction of volume)
                        ppm = get_ppm_for_dr(kv, dra) if dra > 0 else 0.0
                        dra_rate = float(stn.get('RateDRA', 0.0))       # ₹/L
                        dra_cost = (ppm / 1e6) * flow * 24.0 * dra_rate if dra_rate > 0 else 0.0

                        opts.append({
                            "nop": nop, "rpm": rpm, "dra": dra,
                            "head_loss": head_loss, "v": v, "Re": Re, "f": f_eff,
                            "tdh": tdh, "eff": eff,
                            "pump_bkw": pump_bkw, "motor_kw": motor_kw,
                            "power_cost": power_cost, "dra_cost": dra_cost, "dra_ppm": ppm,
                            "cost": power_cost + dra_cost,  # ✅ objective includes DRA + power/fuel
                            # for feasibility checks later:
                            "elev_in": elev_in, "elev_delta": elev_delta,
                            "d_inner": d_inner, "rough": rough, "kv": kv,
                        })
        else:
            # Non-pump segment
            head_loss, v, Re, f_eff = _segment_hydraulics(flow, L_km, d_inner, rough, kv, 0.0)
            opts.append({
                "nop": 0, "rpm": 0, "dra": 0,
                "head_loss": head_loss, "v": v, "Re": Re, "f": f_eff,
                "tdh": 0.0, "eff": 0.0,
                "pump_bkw": 0.0, "motor_kw": 0.0,
                "power_cost": 0.0, "dra_cost": 0.0, "dra_ppm": 0.0,
                "cost": 0.0,
                "elev_in": elev_in, "elev_delta": elev_delta,
                "d_inner": d_inner, "rough": rough, "kv": kv,
            })

        opts_all.append({
            "orig_name": stn.get("name", f"station_{i}"),
            "name": stn.get("name", f"station_{i}").strip().lower().replace(" ", "_"),
            "is_pump": bool(stn.get('is_pump', False)),
            "flow": flow,
            "flow_in": segment_flows[i - 1],
            "rho": rho,
            "kv": kv,
            "L": L_km,
            "d_inner": d_inner,
            "rough": rough,
            # Constraints
            "maop_head": maop_head,
            "maop_kgcm2": maop_kgcm2,
            "maop_margin": maop_margin,
            "min_residual_local": float(stn.get('min_residual', 0.0)),
            "peak_margin": float(stn.get('peak_margin', 25.0)),
            # Curves & bounds (for reporting)
            "coef_A": float(stn.get('A', 0.0)), "coef_B": float(stn.get('B', 0.0)), "coef_C": float(stn.get('C', 0.0)),
            "coef_P": float(stn.get('P', 0.0)), "coef_Q": float(stn.get('Q', 0.0)), "coef_R": float(stn.get('R', 0.0)),
            "coef_S": float(stn.get('S', 0.0)), "coef_T": float(stn.get('T', 0.0)),
            "min_rpm": int(stn.get('MinRPM', 0)), "dol": int(stn.get('DOL', 0)),
            "options": opts,
            "peaks": (stn.get('peaks', []) or []),
            "v_max": v_max,
        })

    return opts_all, segment_flows


# -------------------------- Pareto-front DP solver ----------------------------

def _pareto_prune(states: list[dict], eps_cost=1e-9, eps_res=1e-9) -> list[dict]:
    """Keep only non-dominated states: lower/equal cost & higher/equal residual dominates."""
    if not states:
        return []
    states = sorted(states, key=lambda s: (s['cost'], -s['residual']))
    best = []
    max_res_seen = -1e99
    for s in states:
        r = s['residual']
        if r <= max_res_seen + eps_res:
            continue
        best.append(s)
        if r > max_res_seen:
            max_res_seen = r
    return best


def solve_pipeline(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,   # kept for API compat; per-station rate_hsd preferred
    linefill_dict: dict | None = None,
) -> dict:
    """Global-min solver over the discrete grids using safe Pareto pruning."""
    N = len(stations)
    if not (len(KV_list) == N and len(rho_list) == N):
        return {"error": True, "message": f"Length mismatch: stations={N}, KV={len(KV_list)}, rho={len(rho_list)}"}

    # Make sure station-level tariffs exist; honor user-provided inputs, else fallback
    for stn in stations:
        if 'RateDRA' not in stn:
            stn['RateDRA'] = float(RateDRA)
        if 'rate_hsd' not in stn:
            stn['rate_hsd'] = float(Price_HSD)
        stn['rate'] = float(stn.get('rate', 0.0))
        stn['rate_hsd'] = float(stn.get('rate_hsd', 0.0))

    stn_opts, segment_flows = _build_station_options(stations, KV_list, rho_list, FLOW)

    # DP initial state
    init_res = float(stations[0].get('min_residual', 50.0))
    states = [{"residual": init_res, "cost": 0.0, "power_cost": 0.0, "dra_cost": 0.0, "records": []}]

    # Forward DP
    for S in stn_opts:
        new_states = []
        for state in states:
            rh_in = state['residual']
            for opt in S['options']:
                # SDH at station
                sdh = rh_in + opt['tdh'] if S['is_pump'] else rh_in

                # MAOP (with margin)
                maop_limit = S['maop_head'] - S['maop_margin']
                if S['maop_head'] > 0 and sdh > maop_limit + 1e-9:
                    continue

                # Peak feasibility using actual SDH
                if not _peaks_feasible(S, S['flow'], S['kv'], S['d_inner'], S['rough'],
                                       opt['elev_in'], sdh, opt['dra']):
                    continue

                # Residual after segment
                residual_next = sdh - opt['head_loss'] - opt['elev_delta']

                # Local min residual
                if residual_next + 1e-9 < S['min_residual_local']:
                    continue

                rec = {
                    f"pipeline_flow_{S['name']}": S['flow'],
                    f"pipeline_flow_in_{S['name']}": S['flow_in'],
                    f"head_loss_{S['name']}": opt['head_loss'],
                    f"head_loss_kgcm2_{S['name']}": head_to_kgcm2(opt['head_loss'], S['rho']),
                    f"residual_head_{S['name']}": rh_in,
                    f"rh_kgcm2_{S['name']}": head_to_kgcm2(rh_in, S['rho']),
                    f"sdh_{S['name']}": sdh if S['is_pump'] else rh_in,
                    f"sdh_kgcm2_{S['name']}": head_to_kgcm2(sdh if S['is_pump'] else rh_in, S['rho']),
                    f"rho_{S['name']}": S['rho'],
                    f"maop_{S['name']}": S['maop_head'],
                    f"maop_kgcm2_{S['name']}": S['maop_kgcm2'],
                    f"velocity_{S['name']}": opt['v'],
                    f"reynolds_{S['name']}": opt['Re'],
                    f"friction_{S['name']}": opt['f'],
                    f"num_pumps_{S['name']}": opt['nop'],
                    f"speed_{S['name']}": opt['rpm'],
                    f"pump_flow_{S['name']}": S['flow'] if opt['nop'] > 0 else 0.0,
                    f"efficiency_{S['name']}": opt['eff'],
                    f"power_cost_{S['name']}": opt['power_cost'],
                    f"dra_cost_{S['name']}": opt['dra_cost'],
                    f"dra_ppm_{S['name']}": opt['dra_ppm'],
                    f"drag_reduction_{S['name']}": opt['dra'],
                }

                new_states.append({
                    "residual": residual_next,
                    "cost": state['cost'] + opt['cost'],
                    "power_cost": state['power_cost'] + opt['power_cost'],
                    "dra_cost": state['dra_cost'] + opt['dra_cost'],
                    "records": state['records'] + [rec],
                })

        states = _pareto_prune(new_states)
        if not states:
            return {"error": True, "message": f"No feasible operating point after station: {S['orig_name']}"}

    # Terminal check & choice
    term_req = float(terminal.get('min_residual', 0.0))
    feasible = [s for s in states if s['residual'] + 1e-9 >= term_req]
    if not feasible:
        return {"error": True, "message": "No solution meets terminal residual requirement."}

    best = min(feasible, key=lambda s: (s['cost'], s['residual'] - term_req))

    # Build output
    out = {}
    for rec in best['records']:
        out.update(rec)

    term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    out.update({
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
        f"residual_head_{term_name}": best['residual'],
        f"rh_kgcm2_{term_name}": head_to_kgcm2(best['residual'], rho_list[-1]),
        f"sdh_{term_name}": best['residual'],
        f"sdh_kgcm2_{term_name}": head_to_kgcm2(best['residual'], rho_list[-1]),
        "total_cost": best['cost'],              # ✅ TOTAL = power + DRA
        "total_power_cost": best['power_cost'],  # split for clarity
        "total_dra_cost": best['dra_cost'],
    })
    return out


# ----------------------- Multi-origin (A/B) wrapper ---------------------------

def generate_origin_combinations(maxA: int = 2, maxB: int = 2) -> list[tuple[int, int]]:
    """All (A,B) with A+B>0, ordered by total then lexicographic (small → large)."""
    combos = [(a, b) for a in range(maxA + 1) for b in range(maxB + 1) if a + b > 0]
    return sorted(combos, key=lambda x: (x[0] + x[1], x))


def _apply_type_overrides_from_pump_type(unit: dict, type_dict: dict) -> None:
    """
    Override a virtual origin unit from pump type dict (curves, RPM grids, power settings).
    Honors user inputs: MinRPM, DOL, rpm_list, RPM_STEP, dra_list, DRA_STEP, rate/sfc, etc.
    """
    # Curves
    for k in ('A', 'B', 'C', 'P', 'Q', 'R', 'S', 'T'):
        if k in type_dict:
            unit[k] = float(type_dict[k])

    # RPM stuff
    if 'MinRPM' in type_dict: unit['MinRPM'] = float(type_dict['MinRPM'])
    if 'DOL' in type_dict:    unit['DOL'] = float(type_dict['DOL'])
    if 'RPM_STEP' in type_dict: unit['RPM_STEP'] = int(type_dict['RPM_STEP'])
    if 'rpm_list' in type_dict: unit['rpm_list'] = list(type_dict['rpm_list'])

    # DRA enumeration overrides
    if 'DRA_STEP' in type_dict: unit['DRA_STEP'] = int(type_dict['DRA_STEP'])
    if 'dra_list' in type_dict: unit['dra_list'] = list(type_dict['dra_list'])

    # Power settings
    if 'power_type' in type_dict: unit['power_type'] = type_dict['power_type']
    if unit.get('power_type', 'Grid') == 'Grid':
        if 'rate' in type_dict: unit['rate'] = float(type_dict['rate'])
        unit['sfc'] = 0.0
    else:
        if 'sfc' in type_dict: unit['sfc'] = float(type_dict['sfc'])
        unit['rate'] = 0.0


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
    """
    Expand origin into A/B virtual units (serial) and run the global-min solver.
    KV/rho arrays are replicated to match the expanded station list.
    """
    # Find origin (first pump station)
    try:
        origin_idx = next(i for i, s in enumerate(stations) if s.get('is_pump', False))
    except StopIteration:
        return {"error": True, "message": "No pump station found to act as origin."}

    origin = stations[origin_idx]
    pump_types = origin.get('pump_types', {})

    if not pump_types:
        # No A/B mixing; run single-origin solver
        return solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict)

    combos = generate_origin_combinations(
        int(pump_types.get('A', {}).get('available', 0)),
        int(pump_types.get('B', {}).get('available', 0)),
    )

    best = None
    best_cost = float('inf')
    best_stations = None

    for nA, nB in combos:
        if nA + nB < 1:
            continue

        prefix = copy.deepcopy(stations[:origin_idx])
        suffix = copy.deepcopy(stations[origin_idx + 1:])

        units = []
        for label, count in (('A', nA), ('B', nB)):
            tdict = pump_types.get(label, {})
            for _ in range(count):
                u = copy.deepcopy(origin)
                u['name'] = f"{origin['name']}_{label}"
                u['is_pump'] = True
                u['L'] = 0.0
                u['max_dr'] = origin.get('max_dr', 0.0)
                _apply_type_overrides_from_pump_type(u, tdict)
                units.append(u)

        if units:
            units[-1]['L'] = origin.get('L', 0.0)  # Only last virtual unit carries the segment length

        stations_combo = prefix + units + suffix

        # KV/rho aligned
        kv_combo  = KV_list[:origin_idx] + [KV_list[origin_idx]] * len(units) + KV_list[origin_idx + 1:]
        rho_combo = rho_list[:origin_idx] + [rho_list[origin_idx]] * len(units) + rho_list[origin_idx + 1:]

        if not (len(stations_combo) == len(kv_combo) == len(rho_combo)):
            return {"error": True, "message": "Internal alignment error in multi-origin expansion."}

        # Ensure cost defaults exist
        for stn in stations_combo:
            if 'RateDRA' not in stn: stn['RateDRA'] = float(RateDRA)
            if 'rate_hsd' not in stn: stn['rate_hsd'] = float(Price_HSD)
            stn['rate'] = float(stn.get('rate', 0.0))
            stn['rate_hsd'] = float(stn.get('rate_hsd', 0.0))

        res = solve_pipeline(stations_combo, terminal, FLOW, kv_combo, rho_combo, RateDRA, Price_HSD, linefill_dict)
        if res.get("error"):
            continue
        if res["total_cost"] < best_cost:
            best_cost = res["total_cost"]
            best = res
            best_stations = stations_combo
            best['pump_combo'] = {'A': nA, 'B': nB}

    if best is None:
        return {"error": True, "message": "No feasible pump combination found for origin."}

    best['stations_used'] = best_stations
    return best
