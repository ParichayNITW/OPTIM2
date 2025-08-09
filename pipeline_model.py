
"""Lightweight pipeline optimizer (enumeration) with optional per-segment drag reduction profile.

Assumptions:
- Stations list defines N segments (between station i and i+1; last goes to terminal)
- Optional `dr_profile`: list of length N giving %DR applied to the *entire* segment.
- If unavailable, computed normally (no DR).
"""

from __future__ import annotations
from math import log10, pi
from typing import List, Dict, Tuple, Optional
import itertools

from dra_utils import get_ppm_for_dr

G = 9.80665  # m/s^2

def swamee_jain_f(Re: float, eps: float, D: float) -> float:
    if Re <= 0: return 0.02
    return 0.25/(log10((eps/(3.7*D)) + (5.74/(Re**0.9)))**2)

def headloss_darcy(f: float, L: float, D: float, v: float) -> float:
    return f * (L/D) * (v**2)/(2*G)

def solve_pipeline(
    stations: List[Dict],
    terminal: Dict,
    FLOW_m3h: float,
    KV_list: List[float],
    RHO_list: List[float],
    RateDRA: float,
    Price_HSD: float,
    context: Dict|None = None,
    dr_profile: Optional[List[float]] = None,
    interval_hours: float = 4.0,
    min_peak_head_m: float = 25.0
) -> Dict:
    """
    Returns dict with per-station keys used by the app.
    This enumerates pump combo and speeds in given ranges, and DRA %DR per station 0..max_dr.
    If dr_profile is given, it scales each segment friction by (1-DR/100).
    Costs are returned *per day*; app can scale to interval.
    """
    N = len(stations)
    segN = N  # segment per station to next
    # Segment geometry
    D_list = [st.get("D", 0.711) for st in stations]
    t_list = [st.get("t", 0.007) for st in stations]
    L_list = [st.get("L", 10.0) for st in stations]
    eps_list = [st.get("eps", 1e-5) for st in stations]

    # Flow & velocity per segment (assume single Q)
    Q = FLOW_m3h / 3600.0  # m3/s
    A_list = [3.141592653589793*((max(D-2*t, 1e-3))**2)/4.0 for D,t in zip(D_list,t_list)]
    v_list = [Q/max(A,1e-9) for A in A_list]
    rho_list = RHO_list
    mu_list = [kv*1e-6 for kv in KV_list]  # cSt ≈ mm2/s => m2/s (kinematic)

    # Decide per-station: number of pumps (0..max), speed (min..max), and DRA %DR (0..max_dr step)
    # For simplicity we assume one pump type per station with speed range and max count.
    results = {}
    best_cost = float("inf")
    best = None

    # Enumerations (keep reasonable)
    def enum_speeds(stn):
        smin = int(stn.get("min_speed_rpm", 1500))
        smax = int(stn.get("max_speed_rpm", 3000))
        step = int(stn.get("speed_step", 100))
        return range(smin, smax+1, step)

    for num_combo in itertools.product(*[range(0, int(st.get("max_pumps",1))+1) for st in stations]):
        # Build speeds enumeration
        speed_ranges = [enum_speeds(st) if n>0 else [0] for st,n in zip(stations, num_combo)]
        for speeds in itertools.product(*speed_ranges):
            # DRA: choose %DR per station 0..max_dr step
            dr_steps = [int(st.get("dr_step", 5)) for st in stations]
            max_drs = [int(st.get("max_dr", 0)) for st in stations]
            dr_ranges = [range(0, md+1, ds) for md,ds in zip(max_drs, dr_steps)]
            for drs in itertools.product(*dr_ranges):
                # Compute hydraulic feasibility and costs
                ok, power_kw, head_min, dra_ppm_per_st, station_cost_pf = hydraulic_and_power_costs(
                    stations, L_list, D_list, t_list, eps_list, A_list, v_list, rho_list, mu_list,
                    num_combo, speeds, dr_profile if dr_profile is not None else [0.0]*segN
                )
                if not ok or head_min < min_peak_head_m:
                    continue
                # Energy cost
                pf_cost_day = sum(station_cost_pf)
                # DRA cost per day from station ppm and Q
                dra_cost_day = 0.0
                for st,dr in zip(stations, drs):
                    visc = st.get("kv_ref", KV_list[0])
                    ppm = get_ppm_for_dr(visc, dr) if dr>0 else 0.0
                    st_name = st.get("name","").lower().replace(" ","_")
                    results[f"dra_ppm_{st_name}"] = ppm
                    # ppm => mg/L => g/m3
                    dra_cost_day += ppm * Q * 3600*24 / 1e6 * RateDRA  # (ppm mg/L) -> kg/day approx

                total_cost_day = pf_cost_day + dra_cost_day
                if total_cost_day < best_cost:
                    best_cost = total_cost_day
                    best = {
                        "error": False,
                        "total_cost": total_cost_day,
                        "power_fuel_cost_day": pf_cost_day,
                        "dra_cost_day": dra_cost_day,
                        "head_min": head_min,
                    }
                    # per station outputs
                    for i, st in enumerate(stations):
                        key = st.get("name","stn").lower().replace(" ","_")
                        best[f"num_pumps_{key}"] = num_combo[i]
                        best[f"speed_{key}"] = speeds[i]
                        best[f"dra_dr_{key}"] = drs[i]
                        best[f"power_kw_{key}"] = power_kw[i]
                    best["stations_used"] = stations
    if best is None:
        return {"error": True, "message": "No feasible combination found."}
    return best


def hydraulic_and_power_costs(
    stations, L_list, D_list, t_list, eps_list, A_list, v_list, rho_list, mu_list,
    num_combo, speeds, dr_profile
):
    """Compute feasibility, per-station power, min head across line, and per-station power cost/day.

    dr_profile: list[%DR] per segment.
    """
    # Simple hydraulic: sum friction per segment (Darcy) with DR factor
    power_kw = [0.0]*len(stations)
    station_cost_pf = [0.0]*len(stations)
    head_min = 1e9

    # Assume each station contributes head proportional to speed and pump count (toy)
    for i,(st, n, rpm) in enumerate(zip(stations, num_combo, speeds)):
        # friction on downstream segment i
        D = D_list[i]; t=t_list[i]; L=L_list[i]; A=A_list[i]; v=v_list[i]
        rho = rho_list[i]; mu=mu_list[i]
        Re = max(rho*v*max(D-2*t,1e-3)/max(mu,1e-9), 1.0)
        f = swamee_jain_f(Re, eps_list[i], max(D-2*t,1e-3))
        hf = headloss_darcy(f, L, max(D-2*t,1e-3), v)
        # Apply drag reduction to this segment:
        dr = max(0.0, min(100.0, dr_profile[i] if i < len(dr_profile) else 0.0))
        hf *= (1.0 - dr/100.0)

        # Pump head gain (toy): coeffs
        H_per_pump = st.get("H_const", 20.0) * (rpm/ max(st.get("min_speed_rpm",1500),1))**2
        head_gain = n * H_per_pump

        net_head = head_gain - hf
        head_min = min(head_min, net_head)

        # Power (toy): rho*g*Q*head / (eta*1000)
        eta = max(st.get("eff", 0.7), 0.05)
        Q = v*A
        power_kw[i] = rho*9.80665*Q*max(head_gain,0.0)/(max(eta,1e-3)*1000.0)
        station_cost_pf[i] = power_kw[i] * st.get("power_rate_inr_per_kwh", 8.0) * 24.0

    return True, power_kw, head_min, station_cost_pf
