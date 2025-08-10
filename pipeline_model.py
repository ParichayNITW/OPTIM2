
# Certified A* global optimizer over a discrete mesh with adaptive refinement.
from __future__ import annotations

import math
from math import log10, pi
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import heapq

from dra_utils import get_ppm_for_dr

# ---- Tunables ----
RPM_STEP_INIT: int = 100
DRA_STEP_INIT: int = 5
RPM_TOL_TARGET: int = 25
DRA_TOL_TARGET: int = 1
RESIDUAL_ROUND: int = 1
MOTOR_EFF: float = 0.95

# ---- Helpers ----
def _allowed_values(min_val: int, max_val: int, step: int) -> List[int]:
    if step <= 0: step = 1
    if max_val < min_val:
        return [int(min_val)]
    vals = list(range(int(min_val), int(max_val)+1, int(step)))
    if not vals or vals[-1] != int(max_val):
        vals.append(int(max_val))
    return sorted(set(vals))

def head_to_kgcm2(head_m: float, rho: float) -> float:
    return head_m * rho / 10000.0

def _inner_diameter(D: float, t: float) -> float:
    return max(D - 2.0*t, 0.0)

def _friction_factor_churchill(Re: float, rough: float, d_inner: float) -> float:
    if Re <= 0 or d_inner <= 0:
        return 0.0
    rr = abs(rough / d_inner)
    try:
        termA = (2.457 * math.log((7.0/Re)**0.9 + 0.27*rr))**16
        termB = (37530.0/Re)**16
        denom = (termA + termB)**1.5
        inv = ((8.0/Re)**12 + (1.0/denom))
        f = 8.0 * (inv ** (1.0/12.0))
    except ValueError:
        arg = (rr / 3.7) + (5.74 / (Re ** 0.9))
        f = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
    return max(f, 0.0)

def _segment_hydraulics(flow_m3h: float, L_km: float, d_inner: float, rough: float,
                        kv_cst: float, dra_percent: float, dra_length_km: Optional[float]=None
                       ) -> Tuple[float, float, float, float]:
    g = 9.81
    flow_m3s = flow_m3h / 3600.0
    area = pi * (d_inner ** 2) / 4.0
    v = flow_m3s / area if area > 0 else 0.0
    Re = v * d_inner / (kv_cst * 1e-6) if kv_cst > 0 else 0.0
    f = _friction_factor_churchill(Re, rough, d_inner)

    def hl(length_km: float, apply_dra: bool) -> float:
        red = (1.0 - dra_percent/100.0) if apply_dra else 1.0
        return f * ((max(length_km, 0.0) * 1000.0) / max(d_inner, 1e-9)) * (v**2 / (2*g)) * red

    if dra_length_km is None or dra_length_km >= L_km:
        head_loss = hl(L_km, True)
    elif dra_length_km <= 0:
        head_loss = hl(L_km, False)
    else:
        head_loss = hl(dra_length_km, True) + hl(L_km - dra_length_km, False)

    return head_loss, v, Re, f

def _station_maop_head(stn: Dict, rho: float) -> float:
    D = float(stn.get("D", 0.711))
    t = float(stn.get("t", 0.007))
    SMYS = float(stn.get("SMYS", 52000.0))
    design_factor = 0.72
    outer_d = D
    thickness = t
    maop_psi = 2 * SMYS * design_factor * (thickness / outer_d) if outer_d > 0 else 0.0
    maop_kgcm2 = maop_psi * 0.0703069
    return maop_kgcm2 * 10000.0 / max(rho, 1.0)

def _pump_head_at(stn: Dict, flow_m3h: float, rpm: float, nop: int) -> Tuple[float, float]:
    A = float(stn.get("A", 0.0))
    B = float(stn.get("B", 0.0))
    C = float(stn.get("C", 0.0))
    P = float(stn.get("P", 0.0))
    Qc = float(stn.get("Q", 0.0))
    R = float(stn.get("R", 0.0))
    S = float(stn.get("S", 0.0))
    T = float(stn.get("T", 0.0))
    N0 = float(stn.get("DOL", stn.get("MinRPM", rpm)))
    nop = max(int(nop), 0)
    if nop <= 0 or rpm <= 0:
        return 0.0, 0.0
    q_each = flow_m3h / max(nop, 1)
    H0 = A*(q_each**2) + B*q_each + C
    H = H0 * (rpm / max(N0, 1e-9))**2
    eff = P*(q_each**4) + Qc*(q_each**3) + R*(q_each**2) + S*q_each + T
    eff = max(min(eff, float(stn.get("eff_max", 90.0))), float(stn.get("eff_min", 5.0)))
    return max(H, 0.0), eff

def _downstream_req_with_peaks(stations: List[Dict], terminal: Dict, FLOW: float,
                               KV_list: List[float], rho_list: List[float],
                               dra_percent_downstream: float, dra_reach_km: float, hours: float, i: int
                              ) -> float:
    N = len(stations) - 1
    def req_entry(idx: int) -> float:
        if idx > N:
            return float(terminal.get("min_residual", 30.0))
        stn = stations[idx-1]
        next_elev = terminal.get("elev", 0.0) if idx == N else stations[idx].get("elev", 0.0)
        elev_i = stn.get("elev", 0.0)
        D = float(stn.get("D", 0.711)); t = float(stn.get("t", 0.007)); rough = float(stn.get("rough", 4.5e-5))
        kv = float(KV_list[idx-1])
        L = float(stn.get("L", 0.0))
        d_inner = _inner_diameter(D, t)
        dra_len_here = min(max(dra_reach_km, 0.0), L) if dra_percent_downstream > 0 else 0.0
        head_loss, *_ = _segment_hydraulics(FLOW, L, d_inner, rough, kv, dra_percent_downstream, dra_len_here)
        req = head_loss + (next_elev - elev_i) + req_entry(idx+1)
        peak_req = 0.0
        for peak in (stn.get('peaks') or []):
            dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
            elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
            try:
                dist = float(dist); elev_peak = float(elev_peak)
            except Exception:
                continue
            head_to_peak, *_ = _segment_hydraulics(FLOW, float(dist), d_inner, rough, kv, dra_percent_downstream, min(dra_len_here, float(dist)))
            req_peak = head_to_peak + (elev_peak - elev_i) + 25.0
            peak_req = max(peak_req, req_peak)
        return max(req, peak_req)
    return req_entry(i)

@dataclass(order=True)
class PQItem:
    f: float
    g: float=field(compare=False)
    idx: int=field(compare=False)
    residual: float=field(compare=False)
    decisions: list=field(compare=False, default_factory=list)

def _lower_bound_cost_to_go(stations: List[Dict], terminal: Dict, FLOW: float,
                            KV_list: List[float], rho_list: List[float],
                            i: int, residual_at_i: float,
                            dra_max_allowed: float, dra_reach_km: float, hours: float) -> float:
    g = 9.81
    req_i = _downstream_req_with_peaks(stations, terminal, FLOW, KV_list, rho_list,
                                       dra_percent_downstream=dra_max_allowed,
                                       dra_reach_km=dra_reach_km, hours=hours, i=i)
    deficit = max(req_i - residual_at_i, 0.0)
    if deficit <= 0:
        return 0.0
    eff_best =  max(5.0, min(90.0, max(float(s.get('eff_max', 90.0)) for s in stations if s.get('is_pump')))) / 100.0
    rho = rho_list[min(i-1, len(rho_list)-1)] if rho_list else 850.0
    motor_kw = (rho * FLOW * g * deficit) / (3600.0 * 1000.0 * max(eff_best, 1e-6) * max(MOTOR_EFF, 1e-6))
    tariffs = []
    for s in stations[i-1:]:
        if not s.get('is_pump'): continue
        if s.get('sfc', 0):
            tariffs.append(0.0)
        else:
            tariffs.append(float(s.get('rate', 0.0)))
    rate_lb = min(tariffs) if tariffs else 0.0
    return max(motor_kw * hours * rate_lb, 0.0)

def _enumerate_options_for_station(stn: Dict, FLOW: float, kv: float, rho: float,
                                   rpm_values: List[int], dra_values: List[int],
                                   hours: float, dra_reach_km: float) -> List[Dict]:
    opts = []
    if not stn.get('is_pump'):
        opts.append({'nop': 0, 'rpm': 0, 'dra': 0, 'tdh': 0.0, 'eff': 0.0, 'power_cost': 0.0, 'dra_cost': 0.0})
        return opts

    min_p = int(stn.get('min_pumps', 0))
    max_p = int(stn.get('max_pumps', 2))
    min_rpm = int(stn.get('MinRPM', 0))
    max_rpm = int(stn.get('DOL', 0))
    eff_min = float(stn.get('eff_min', 5.0)); eff_max = float(stn.get('eff_max', 90.0))

    for nop in range(max(0, min_p), max_p+1):
        rpms = [0] if nop == 0 else [r for r in rpm_values if min_rpm <= r <= max_rpm]
        for rpm in rpms:
            for dra in dra_values:
                if nop > 0 and rpm > 0:
                    tdh, eff = _pump_head_at(stn, FLOW, rpm, nop)
                    eff = max(min(eff, eff_max), eff_min)
                else:
                    tdh, eff = 0.0, 0.0

                if nop > 0 and rpm > 0 and eff > 0:
                    pump_bkw_total = (rho * FLOW * 9.81 * tdh) / (3600.0 * 1000.0 * (eff/100.0))
                    motor_kw_total = pump_bkw_total / max(MOTOR_EFF, 1e-6)
                else:
                    motor_kw_total = 0.0

                if stn.get('sfc', 0) and motor_kw_total > 0:
                    sfc_val = float(stn['sfc'])
                    fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                    power_cost = motor_kw_total * hours * fuel_per_kWh * float(stn.get('Price_HSD', 0.0))
                else:
                    power_cost = motor_kw_total * hours * float(stn.get('rate', 0.0))

                ppm = get_ppm_for_dr(kv, dra) if dra > 0 else 0.0
                dra_cost = ppm * (FLOW * 1000.0 * hours / 1e6) * float(stn.get('RateDRA', 0.0)) if dra > 0 else 0.0

                opts.append({'nop': nop, 'rpm': rpm, 'dra': dra, 'tdh': tdh, 'eff': eff,
                             'power_cost': float(power_cost), 'dra_cost': float(dra_cost)})
    return opts

def _solve_one_mesh(stations: List[Dict], terminal: Dict, FLOW: float,
                    KV_list: List[float], rho_list: List[float],
                    rpm_step: int, dra_step: int, dra_reach_km: float,
                    mop_kgcm2: Optional[float], hours: float) -> Dict:
    N = len(stations)
    stn_data = []
    for i, stn in enumerate(stations, start=1):
        kv = float(KV_list[i-1]); rho = float(rho_list[i-1])
        D = float(stn.get("D", 0.711)); t = float(stn.get("t", 0.007)); rough = float(stn.get("rough", 4.5e-5))
        d_inner = _inner_diameter(D, t)
        L = float(stn.get("L", 0.0))
        elev = float(stn.get("elev", 0.0))
        maop_head = _station_maop_head(stn, rho)
        if mop_kgcm2 is not None:
            maop_head = min(maop_head, float(mop_kgcm2) * 10000.0 / max(rho,1.0))
        min_rpm = int(stn.get('MinRPM', 0)); max_rpm = int(stn.get('DOL', 0))
        rpm_vals = _allowed_values(min_rpm, max_rpm, rpm_step) if stn.get('is_pump') else [0]
        dra_vals = _allowed_values(0, int(stn.get('max_dr', 0)), dra_step) if stn.get('is_pump') else [0]
        stn_data.append({"name": stn.get("name", f"S{i}"), "is_pump": bool(stn.get("is_pump", False)),
                         "L": L, "elev": elev, "d_inner": d_inner, "rough": rough, "kv": kv, "rho": rho,
                         "maop_head": maop_head, "rpm_vals": rpm_vals, "dra_vals": dra_vals, "stn": stn})

    init_residual = float(stations[0].get('min_residual', 50.0))
    init_bucket = round(init_residual, RESIDUAL_ROUND)
    start = PQItem(f=0.0, g=0.0, idx=1, residual=init_bucket, decisions=[])

    open_pq: list[PQItem] = []
    heapq.heappush(open_pq, start)
    best_g = {}
    best_solution = None
    best_cost = float('inf')

    while open_pq:
        node = heapq.heappop(open_pq)
        key = (node.idx, node.residual)
        if key in best_g and node.g > best_g[key] + 1e-9:
            continue
        best_g[key] = node.g

        if node.idx > N:
            if node.g < best_cost:
                best_cost = node.g
                best_solution = node
            break

        sdn = stn_data[node.idx-1]
        stn = sdn["stn"]
        next_elev = terminal.get("elev", 0.0) if node.idx == N else stn_data[node.idx]["elev"]
        elev_delta = next_elev - sdn["elev"]

        opts = _enumerate_options_for_station(stn, FLOW, sdn["kv"], sdn["rho"], sdn["rpm_vals"], sdn["dra_vals"], hours, dra_reach_km)

        for opt in opts:
            sdh = node.residual + opt['tdh']
            if sdn["is_pump"] and sdh > sdn["maop_head"]:
                continue

            dra_len_here = min(dra_reach_km, sdn["L"]) if opt['dra'] > 0 else 0.0
            head_loss, *_ = _segment_hydraulics(FLOW, sdn["L"], sdn["d_inner"], sdn["rough"], sdn["kv"], opt['dra'], dra_len_here)
            residual_next = sdh - head_loss - elev_delta
            if residual_next < float(stn.get("min_residual", 0.0)) - 1e-9:
                continue

            violated = False
            for peak in (stn.get('peaks') or []):
                dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
                elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
                try:
                    dist = float(dist); elev_peak = float(elev_peak)
                except Exception:
                    continue
                head_to_peak, *_ = _segment_hydraulics(FLOW, float(dist), sdn["d_inner"], sdn["rough"], sdn["kv"], opt['dra'], min(dra_len_here, float(dist)))
                sdh_at_peak = sdh - head_to_peak - (elev_peak - sdn["elev"])
                if sdh_at_peak > sdn["maop_head"]:
                    violated = True; break
            if violated:
                continue

            g_new = node.g + opt['power_cost'] + opt['dra_cost']
            lb_to_go = _lower_bound_cost_to_go(stations, terminal, FLOW, KV_list, rho_list,
                                               node.idx+1, residual_next, dra_max_allowed=max(sdn["stn"].get('max_dr', 0), 0),
                                               dra_reach_km=dra_reach_km, hours=hours)
            f_new = g_new + lb_to_go
            if f_new >= best_cost - 1e-9:
                continue

            bucket_next = round(residual_next, RESIDUAL_ROUND)
            decisions_next = node.decisions + [{
                "station": sdn["name"], "nop": opt["nop"], "rpm": opt["rpm"], "dra": opt["dra"],
                "tdh": opt["tdh"], "eff": opt["eff"], "power_cost": opt["power_cost"],
                "dra_cost": opt["dra_cost"], "sdh": sdh, "residual_next": residual_next,
            }]
            heapq.heappush(open_pq, PQItem(f=f_new, g=g_new, idx=node.idx+1, residual=bucket_next, decisions=decisions_next))

    if not best_solution:
        return {"error": True, "message": "No feasible solution on this mesh."}

    result = {
        "total_cost": float(best_cost),
        "mesh_rpm_step": rpm_step,
        "mesh_dra_step": dra_step,
        "decisions": best_solution.decisions,
    }
    for dec in best_solution.decisions:
        key = dec["station"].lower().replace(" ", "_")
        result[f"num_pumps_{key}"] = dec["nop"]
        result[f"speed_{key}"] = dec["rpm"]
        result[f"drag_reduction_{key}"] = dec["dra"]
        result[f"sdh_{key}"] = dec["sdh"]
        result[f"residual_head_{key}"] = dec["residual_next"]
        result[f"power_cost_{key}"] = dec["power_cost"]
        result[f"dra_cost_{key}"] = dec["dra_cost"]
    return result

def solve_pipeline(stations: List[Dict], terminal: Dict, FLOW: float, KV_list: List[float], rho_list: List[float],
                   RateDRA: float, Price_HSD: float, linefill_dict: Optional[dict]=None, dra_reach_km: float=0.0,
                   mop_kgcm2: Optional[float]=None, hours: float=24.0,
                   rpm_step_init: int=RPM_STEP_INIT, dra_step_init: int=DRA_STEP_INIT,
                   rpm_tol_target: int=RPM_TOL_TARGET, dra_tol_target: int=DRA_TOL_TARGET) -> Dict:
    for s in stations:
        s["RateDRA"] = float(RateDRA)
        s["Price_HSD"] = float(Price_HSD)

    rpm_step = int(max(1, rpm_step_init))
    dra_step = int(max(1, dra_step_init))

    bounds = []
    for s in stations:
        if s.get('is_pump'):
            bounds.append({"rpm_min": int(s.get("MinRPM", 0)), "rpm_max": int(s.get("DOL", 0)),
                           "dra_min": 0, "dra_max": int(s.get("max_dr", 0))})
        else:
            bounds.append({"rpm_min": 0, "rpm_max": 0, "dra_min": 0, "dra_max": 0})

    last_result = None
    while True:
        stations_mesh = []
        for s, b in zip(stations, bounds):
            ss = dict(s)
            if ss.get('is_pump'):
                ss['MinRPM'] = b["rpm_min"]; ss['DOL'] = b["rpm_max"]; ss['max_dr'] = b["dra_max"]
            stations_mesh.append(ss)

        res = _solve_one_mesh(stations_mesh, terminal, FLOW, KV_list, rho_list,
                              rpm_step, dra_step, dra_reach_km, mop_kgcm2, hours)
        if res.get("error"):
            if last_result is not None:
                return last_result
            return res

        last_result = res
        if rpm_step <= rpm_tol_target and dra_step <= dra_tol_target:
            break

        chosen = {dec["station"]: {"rpm": int(dec["rpm"]), "dra": int(dec["dra"])} for dec in res["decisions"]}
        for i, (s, b) in enumerate(zip(stations, bounds)):
            if not s.get('is_pump'): continue
            name = s.get("name", f"S{i+1}")
            pick = chosen.get(name, {"rpm": b["rpm_min"], "dra": b["dra_min"]})
            rpm_c = pick["rpm"]; dra_c = pick["dra"]
            rpm_lo = max(int(s.get("MinRPM", 0)), int(rpm_c - max(rpm_step, rpm_tol_target)))
            rpm_hi = min(int(s.get("DOL", 0)),    int(rpm_c + max(rpm_step, rpm_tol_target)))
            dra_lo = 0
            dra_hi = min(int(s.get("max_dr", 0)), int(dra_c + max(dra_step, dra_tol_target)))
            b.update({"rpm_min": rpm_lo, "rpm_max": rpm_hi, "dra_min": dra_lo, "dra_max": dra_hi})

        rpm_step = max(rpm_tol_target, rpm_step // 2 if rpm_step > 1 else 1)
        dra_step = max(dra_tol_target, dra_step // 2 if dra_step > 1 else 1)

    last_result["certificate"] = {"optimal_on_mesh": True, "rpm_step": rpm_step, "dra_step": dra_step,
                                  "note": "Certified global optimum on the final discrete mesh."}
    return last_result

def solve_pipeline_multi_origin(stations: List[Dict], terminal: Dict, FLOW: float, KV_list: List[float], rho_list: List[float],
                                RateDRA: float, Price_HSD: float, linefill_dict: Optional[dict]=None, dra_reach_km: float=0.0,
                                mop_kgcm2: Optional[float]=None, hours: float=24.0, **kwargs) -> Dict:
    _solve = solve_pipeline
    origin_index = next((i for i, s in enumerate(stations) if s.get('is_pump')), 0)
    origin = stations[origin_index]
    pump_types = origin.get("pump_types", {})
    maxA = int(pump_types.get("A", {}).get("available", 0))
    maxB = int(pump_types.get("B", {}).get("available", 0))

    def combos():
        out = []
        for a in range(maxA+1):
            for b in range(maxB+1):
                if a+b == 0: 
                    continue
                if a+b > int(origin.get("max_pumps", 3)):
                    continue
                out.append((a,b))
        return out or [(0,0)]

    best = None; best_cost = float('inf')
    for (nA, nB) in combos():
        stns = [dict(s) for s in stations]
        st0 = dict(stns[origin_index])
        st0["min_pumps"] = max(1, nA+nB)
        st0["max_pumps"] = max(1, nA+nB)
        stns[origin_index] = st0
        res = _solve(stns, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD,
                     linefill_dict, dra_reach_km, mop_kgcm2, hours,
                     kwargs.get("rpm_step_init", RPM_STEP_INIT),
                     kwargs.get("dra_step_init", DRA_STEP_INIT),
                     kwargs.get("rpm_tol_target", RPM_TOL_TARGET),
                     kwargs.get("dra_tol_target", DRA_TOL_TARGET))
        if not res.get("error") and res.get("total_cost", float('inf')) < best_cost:
            best = res; best_cost = float(res["total_cost"])
    return best or {"error": True, "message": "No feasible origin combination."}
