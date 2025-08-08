# pipeline_model.py
# Global-minimum (discrete) pipeline optimizer with Pareto-front DP
# Requirements: dra_utils.get_ppm_for_dr available in path

from __future__ import annotations

from math import log10, pi
import copy

from dra_utils import get_ppm_for_dr


# ------------------------------ Tunables -------------------------------------

RPM_STEP = 100       # RPM discretization
DRA_STEP = 5         # % drag reduction discretization
VEL_DEFAULT_MAX = 3.0  # m/s default velocity cap if station doesn't specify


# --------------------------- Helper conversions -------------------------------

def head_to_kgcm2(head_m: float, rho: float) -> float:
    """Convert head (m) to kg/cm² using rho (kg/m³)."""
    return head_m * rho / 10000.0


def _allowed_values(min_val: int, max_val: int, step: int) -> list[int]:
    """Inclusive integer sequence with step; always include max_val."""
    if max_val < min_val:
        return [min_val]
    vals = list(range(int(min_val), int(max_val) + 1, int(step)))
    if vals and vals[-1] != int(max_val):
        vals.append(int(max_val))
    elif not vals:
        vals = [int(max_val)]
    return vals


# --------------------------- Hydraulics & Pumps -------------------------------

def _segment_hydraulics(flow_m3h: float, L_km: float, d_inner: float, rough: float,
                        kv_cst: float, dra_perc: float) -> tuple[float, float, float, float]:
    """Compute (head_loss[m], velocity[m/s], Re[-], f_eff[-]) for a segment.

    DRA reduces Darcy friction: f' = f * (1 - DR), DR in [0, 0.95].
    """
    g = 9.81
    flow_m3s = flow_m3h / 3600.0
    area = pi * d_inner ** 2 / 4.0
    v = flow_m3s / area if area > 0 else 0.0
    Re = v * d_inner / (kv_cst * 1e-6) if kv_cst > 0 else 0.0

    # Base friction factor f (Haaland for turbulent, 64/Re for laminar)
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

    head_loss = f_eff * ((L_km * 1000.0) / d_inner) * (v ** 2 / (2.0 * g))
    return head_loss, v, Re, f_eff


def _pump_head_and_eff(stn: dict, flow_m3h: float, rpm: float, nop: int) -> tuple[float, float]:
    """TDH and efficiency at given flow, rpm, and pump count (curves scaled by similarity laws)."""
    dol = float(stn.get('DOL', rpm if rpm else 1.0))
    Q_equiv = flow_m3h * dol / rpm if rpm > 0 else flow_m3h
    A = float(stn.get('A', 0.0))
    B = float(stn.get('B', 0.0))
    C = float(stn.get('C', 0.0))
    tdh_single = A * Q_equiv ** 2 + B * Q_equiv + C
    tdh = tdh_single * (rpm / dol) ** 2 * nop

    P = float(stn.get('P', 0.0))
    Q = float(stn.get('Q', 0.0))
    R = float(stn.get('R', 0.0))
    S = float(stn.get('S', 0.0))
    T = float(stn.get('T', 0.0))
    eff = P * Q_equiv ** 4 + Q * Q_equiv ** 3 + R * Q_equiv ** 2 + S * Q_equiv + T
    return tdh, eff


# ----------------------- Peak margin feasibility check ------------------------

def _peaks_feasible(stn: dict, flow_m3h: float, kv: float, d_inner: float, rough: float,
                    elev_in: float, sdh_after_pump: float, dra_perc: float) -> bool:
    """Check intermediate elevation peaks are satisfied with margin for a given option."""
    peaks = (stn.get('peaks', []) or [])
    if not peaks:
        return True
    margin = float(stn.get('peak_margin', 25.0))
    for peak in peaks:
        # Accept any of the expected key variants
        dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location') or peak.get('dist_km')
        elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation') or peak.get('elev_m')
        if dist is None or elev_peak is None:
            continue
        hl_to_peak, *_ = _segment_hydraulics(flow_m3h, float(dist), d_inner, rough, kv, dra_perc)
        required_at_peak = (float(elev_peak) - float(elev_in)) + float(margin) + hl_to_peak
        if sdh_after_pump < required_at_peak - 1e-9:  # small tolerance
            return False
    return True


# ---------------------------- Option generation ------------------------------

def _build_station_options(
    stations: list[dict],
    KV_list: list[float],
    rho_list: list[float],
    FLOW: float,
) -> list[dict]:
    """Pre-compute per-station feasible options with physics and costs."""
    N = len(stations)
    # Flow profile after each station considering supplies/deliveries
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        segment_flows.append(prev_flow - delivery + supply)

    opts_all = []
    origin_seen = False
    default_t = 0.007
    default_e = 0.00004

    for i, stn in enumerate(stations, start=1):
        flow = segment_flows[i]
        kv = float(KV_list[i - 1])
        rho = float(rho_list[i - 1])

        # Geometry & materials
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

        elev_in = float(stn.get('elev', 0.0))
        elev_out = float(stations[i]['elev']) if i < N else float(stn.get('elev_out', elev_in))
        elev_out = float(stations[i].get('elev', elev_out)) if i < N else elev_out
        elev_delta = elev_out - elev_in

        # MAOP calculation (thin-wall, design factor 0.72)
        SMYS = float(stn.get('SMYS', 52000.0))
        design_factor = float(stn.get('design_factor', 0.72))
        maop_psi = 2.0 * SMYS * design_factor * (thickness / outer_d) if outer_d > 0 else 0.0
        maop_kgcm2 = maop_psi * 0.0703069
        maop_head = maop_kgcm2 * 10000.0 / rho if rho > 0 else 0.0
        maop_margin = float(stn.get('maop_margin', 0.0))

        # Velocity cap
        v_max = float(stn.get('v_max', VEL_DEFAULT_MAX))

        opts = []
        if stn.get('is_pump', False):
            min_p = int(stn.get('min_pumps', 0))
            if not origin_seen:
                min_p = max(1, min_p)  # ensure origin runs ≥1 pump
                origin_seen = True
            max_p = int(stn.get('max_pumps', 2))
            rpm_vals = _allowed_values(int(stn.get('MinRPM', 0)), int(stn.get('DOL', 0)), RPM_STEP)
            dra_vals = _allowed_values(0, int(stn.get('max_dr', 0)), DRA_STEP)

            for nop in range(min_p, max_p + 1):
                for rpm in (rpm_vals if nop > 0 else [0]):
                    for dra in dra_vals:
                        # segment loss with this DRA
                        head_loss, v, Re, f_eff = _segment_hydraulics(flow, L_km, d_inner, rough, kv, dra)
                        if v_max > 0.0 and v > v_max:
                            continue

                        if nop > 0 and rpm > 0:
                            tdh, eff = _pump_head_and_eff(stn, flow, rpm, nop)
                            eff = max(eff, 1e-6)
                        else:
                            tdh, eff = 0.0, 0.0

                        sdh_after_pump = (tdh)  # sdh is residual_before + tdh (added later), we check margin relative to inlet
                        # Peak feasibility depends on absolute SDH after pump; here we only can check *incremental* capability.
                        # We'll enforce peak check after SDH is formed with the running residual in DP.

                        # Power & DRA cost
                        if nop > 0 and rpm > 0:
                            pump_bkw_total = (rho * flow * 9.81 * tdh) / (3600.0 * 1000.0 * (eff / 100.0))
                            motor_kw_total = pump_bkw_total / 0.95
                            pump_bkw = pump_bkw_total / nop
                            motor_kw = motor_kw_total / nop
                        else:
                            pump_bkw = motor_kw = motor_kw_total = 0.0

                        if stn.get('sfc', 0) and motor_kw_total > 0:
                            # HSD-diesel driver
                            sfc_val = float(stn['sfc'])
                            fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                            rate_hsd = float(stn.get('rate_hsd', 0.0))
                            power_cost = motor_kw_total * 24.0 * fuel_per_kWh * rate_hsd
                        else:
                            rate_elec = float(stn.get('rate', 0.0))
                            power_cost = motor_kw_total * 24.0 * rate_elec

                        ppm = get_ppm_for_dr(kv, dra) if dra > 0 else 0.0
                        dra_cost = (ppm / 1e6) * flow * 24.0 * float(stn.get('RateDRA', 0.0))

                        opts.append({
                            "nop": nop, "rpm": rpm, "dra": dra,
                            "head_loss": head_loss, "v": v, "Re": Re, "f": f_eff,
                            "tdh": tdh, "eff": eff,
                            "pump_bkw": pump_bkw, "motor_kw": motor_kw,
                            "power_cost": power_cost, "dra_cost": dra_cost, "dra_ppm": ppm,
                            "cost": power_cost + dra_cost,
                            # stuff for feasibility checks later
                            "elev_in": elev_in, "elev_delta": elev_delta,
                            "d_inner": d_inner, "rough": rough, "kv": kv,
                        })
        else:
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
            # Curve coefficients for reporting
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
    """Keep only states that are not dominated (lower cost & higher residual dominate)."""
    if not states:
        return []
    # Sort by cost asc, residual desc -> easy sweep
    states = sorted(states, key=lambda s: (s['cost'], -s['residual']))
    best = []
    max_res_seen = -1e99
    for s in states:
        r = s['residual']
        # Since we sort by increasing cost, if this state has residual <= max seen, it's dominated
        if r <= max_res_seen + eps_res:
            continue
        best.append(s)
        max_res_seen = max(max_res_seen, r)
    return best


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
    """Global-min DP: exhaustive discrete enumeration with safe Pareto pruning.

    Guarantees global optimum within the discrete grids of RPM, DRA, and pump counts.
    """
    N = len(stations)
    if not (len(KV_list) == N and len(rho_list) == N):
        return {"error": True, "message": f"Length mismatch: stations={N}, KV={len(KV_list)}, rho={len(rho_list)}"}

    # Build options
    # Insert RateDRA & tariffs into station dict for local cost computation
    for stn in stations:
        stn['RateDRA'] = float(RateDRA)
        stn['rate'] = float(stn.get('rate', 0.0))
        stn['rate_hsd'] = float(stn.get('rate_hsd', 0.0))

    stn_opts, segment_flows = _build_station_options(stations, KV_list, rho_list, FLOW)

    # DP initial state: starting residual at origin
    init_res = float(stations[0].get('min_residual', 50.0))
    states = [{
        "residual": init_res,
        "cost": 0.0,
        "records": [],
    }]

    # Iterate stations forward
    for idx, S in enumerate(stn_opts):
        new_states = []
        for state in states:
            rh_in = state['residual']
            for opt in S['options']:
                # Compute SDH after pump & enforce MAOP at this station
                sdh = rh_in + opt['tdh'] if S['is_pump'] else rh_in
                maop_limit = S['maop_head'] - S['maop_margin']
                if S['maop_head'] > 0 and sdh > maop_limit + 1e-9:
                    continue

                # Peak feasibility (uses actual SDH)
                if not _peaks_feasible(S, S['flow'], S['kv'], S['d_inner'], S['rough'],
                                       opt['elev_in'], sdh, opt['dra']):
                    continue

                # Residual after segment
                residual_next = sdh - opt['head_loss'] - opt['elev_delta']

                # Enforce local min residual at this station's outlet
                if residual_next < S['min_residual_local'] - 1e-9:
                    continue

                # Velocity cap already enforced when building options (via v_max)

                # Build record for reporting
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
                    f"pump_bkw_{S['name']}": opt['pump_bkw'],
                    f"motor_kw_{S['name']}": opt['motor_kw'],
                    f"power_cost_{S['name']}": opt['power_cost'],
                    f"dra_cost_{S['name']}": opt['dra_cost'],
                    f"dra_ppm_{S['name']}": opt['dra_ppm'],
                    f"drag_reduction_{S['name']}": opt['dra'],
                }

                new_states.append({
                    "residual": residual_next,
                    "cost": state['cost'] + opt['cost'],
                    "records": state['records'] + [rec],
                })

        # Pareto prune (guarantee-preserving)
        states = _pareto_prune(new_states)

        if not states:
            return {"error": True, "message": f"No feasible operating point after station: {S['orig_name']}"}

    # Terminal feasibility & pick best
    term_req = float(terminal.get('min_residual', 0.0))
    feasible = [s for s in states if s['residual'] + 1e-9 >= term_req]
    if not feasible:
        return {"error": True, "message": "No solution meets terminal residual requirement."}

    best = min(feasible, key=lambda s: (s['cost'], s['residual'] - term_req))

    # Assemble result dict
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
        "total_cost": best['cost'],
    })
    return out


# ----------------------- Multi-origin (A/B) wrapper ---------------------------

def generate_origin_combinations(maxA: int = 2, maxB: int = 2) -> list[tuple[int, int]]:
    """Enumerate (A,B) counts where A+B>0, ordered by total pumps then lexicographic."""
    combos = [(a, b) for a in range(maxA + 1) for b in range(maxB + 1) if a + b > 0]
    return sorted(combos, key=lambda x: (x[0] + x[1], x))


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
    """Expand origin into serial virtual units for A/B combinations and call solver.

    KV_list / rho_list are replicated for virtual units so indices align.
    """
    # Find first pump station as origin
    try:
        origin_index = next(i for i, s in enumerate(stations) if s.get('is_pump', False))
    except StopIteration:
        return {"error": True, "message": "No pump station found to act as origin."}

    origin = stations[origin_index]
    pump_types = origin.get('pump_types', {})
    combos = generate_origin_combinations(
        pump_types.get('A', {}).get('available', 2),
        pump_types.get('B', {}).get('available', 2),
    )

    best_cost = float('inf')
    best_res = None
    best_stations = None

    for (numA, numB) in combos:
        if numA + numB < 1:
            continue

        prefix = copy.deepcopy(stations[:origin_index])
        suffix = copy.deepcopy(stations[origin_index + 1:])

        # Build virtual chain
        pump_units = []
        for label, count in (('A', numA), ('B', numB)):
            for _ in range(count):
                u = copy.deepcopy(origin)
                u['name'] = f"{origin['name']}_{label}"
                u['is_pump'] = True
                u['L'] = 0.0
                u['max_dr'] = origin.get('max_dr', 0.0)
                if label in pump_types:
                    curve = pump_types[label]
                    u.update({
                        'A': curve.get('A', u.get('A', 0.0)),
                        'B': curve.get('B', u.get('B', 0.0)),
                        'C': curve.get('C', u.get('C', 0.0)),
                        'P': curve.get('P', u.get('P', 0.0)),
                        'Q': curve.get('Q', u.get('Q', 0.0)),
                        'R': curve.get('R', u.get('R', 0.0)),
                        'S': curve.get('S', u.get('S', 0.0)),
                        'T': curve.get('T', u.get('T', 0.0)),
                    })
                pump_units.append(u)

        if pump_units:
            pump_units[-1]['L'] = origin.get('L', 0.0)

        stations_combo = prefix + pump_units + suffix

        # Build KV/rho aligned
        kv_combo = KV_list[:origin_index] + [KV_list[origin_index]] * len(pump_units) + KV_list[origin_index + 1:]
        rho_combo = rho_list[:origin_index] + [rho_list[origin_index]] * len(pump_units) + rho_list[origin_index + 1:]

        if not (len(stations_combo) == len(kv_combo) == len(rho_combo)):
            return {"error": True, "message": "Internal alignment error in multi-origin expansion."}

        res = solve_pipeline(stations_combo, terminal, FLOW, kv_combo, rho_combo, RateDRA, Price_HSD, linefill_dict)
        if res.get("error"):
            continue
        cost = float(res.get("total_cost", float('inf')))
        if cost < best_cost:
            best_cost = cost
            best_res = res
            best_stations = stations_combo
            best_res['pump_combo'] = {'A': numA, 'B': numB}

    if best_res is None:
        return {"error": True, "message": "No feasible pump combination found for origin."}

    best_res['stations_used'] = best_stations
    return best_res
