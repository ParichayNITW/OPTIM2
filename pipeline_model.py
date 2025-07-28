import numpy as np
import pandas as pd
import itertools
import os

def solve_pipeline(
    stations,
    terminal,
    FLOW,
    kv_list,
    rho_list,
    RateDRA,
    Price_HSD,
    linefill_dict
):
    """
    Brute-force scenario enumeration for Pipeline Optima™ backend.
    Returns dict of optimized per-station and pipeline results (keys as per frontend expectation).
    """

    g = 9.81  # m/s2
    HOURS_PER_DAY = 24
    EPS = 1e-6

    # ---- DRA Curve Loader ----
    DRA_CSV_FILES = {
        10: "10 cst.csv",
        15: "15 cst.csv",
        20: "20 cst.csv",
        25: "25 cst.csv",
        30: "30 cst.csv",
        35: "35 cst.csv",
        40: "40 cst.csv"
    }
    DRA_CURVE_DATA = {}
    for cst, fname in DRA_CSV_FILES.items():
        if os.path.exists(fname):
            DRA_CURVE_DATA[cst] = pd.read_csv(fname)
        else:
            DRA_CURVE_DATA[cst] = pd.DataFrame({'%Drag Reduction':[0,10,20,30],'PPM':[0,10,30,50]})

    def get_ppm_for_dr(visc, dr, dra_curve_data=DRA_CURVE_DATA):
        """Interpolate PPM for required DR and viscosity (cSt)."""
        cst_list = sorted(dra_curve_data.keys())
        visc = float(visc)
        if visc <= cst_list[0]:
            df = dra_curve_data[cst_list[0]]
            return np.interp(dr, df['%Drag Reduction'], df['PPM'])
        elif visc >= cst_list[-1]:
            df = dra_curve_data[cst_list[-1]]
            return np.interp(dr, df['%Drag Reduction'], df['PPM'])
        else:
            lower = max([c for c in cst_list if c <= visc])
            upper = min([c for c in cst_list if c >= visc])
            df_lower = dra_curve_data[lower]
            df_upper = dra_curve_data[upper]
            ppm_lower = np.interp(dr, df_lower['%Drag Reduction'], df_lower['PPM'])
            ppm_upper = np.interp(dr, df_upper['%Drag Reduction'], df_upper['PPM'])
            ppm_interp = np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
            return ppm_interp

    # ---- Hydraulic Equations ----
    def darcy_weisbach_head_loss(Q, D, L, rough, visc, dens, drag_reduction):
        """Head loss in m for segment at given DRA (Swamee-Jain for Re>4000)."""
        A = np.pi * (D/2)**2  # m2
        v = Q / 3600 / A  # m/s
        Re = dens * v * D / (visc * 1e-6)
        eps_D = rough / D
        if Re < 2000:
            f = 64/Re
        else:
            # Swamee-Jain (no approximation)
            f = 0.25/(np.log10(eps_D/3.7 + 5.74/(Re**0.9)))**2
        f = f * (1 - drag_reduction/100)
        hl = f * (L*1000) / D * v**2 / (2*g)
        return hl, v, Re

    def get_peaks_extra_head(peaks, stn_elev):
        """Return max extra head (m) needed to cross peaks, relative to upstream station elevation."""
        extra = 0
        for pk in peaks:
            if pk["elev"] > stn_elev:
                extra = max(extra, pk["elev"] - stn_elev)
        return extra

    # --- MAOP calculation ---
    def calc_maop(SMYS, D, t):
        D_mm = D * 1000
        t_mm = t * 1000
        SMYS_MPa = SMYS * 0.006895
        maop = 2 * t_mm * SMYS_MPa / (D_mm * 1.5) * 10.1972  # bar
        return maop

    # --- Curve fitters ---
    def eval_head_curve(Q, coefs, N, DOL):
        """Head (m) at Q, N (rpm), given A,B,C at DOL."""
        A,B,C = coefs
        return A*Q**2 + B*Q*(N/DOL) + C*(N/DOL)**2

    def eval_eff_curve(Q, coefs, N, DOL):
        """Efficiency (%) at Q, N (rpm), given P,Q,R,S,T at DOL."""
        P, Qc, R, S, T = coefs
        Qeq = Q * DOL / N if N != 0 else 0
        eff = P*Qeq**4 + Qc*Qeq**3 + R*Qeq**2 + S*Qeq + T
        return np.clip(eff, 1, 99.5)

    # --- All possible pump combos ---
    def pump_combos(maxA, maxB):
        out = []
        for nA in range(0, maxA+1):
            for nB in range(0, maxB+1):
                if nA==0 and nB==0: continue
                out.append((nA, nB))
        return out

    # --- DRA steps per station ---
    def drag_steps(max_dr):
        return np.round(np.arange(0, max_dr+0.01, 2.5),2)

    # --- Build bypass scenarios (all ways to select operating pump stations, including skipping any) ---
    Nstn = len([s for s in stations if s.get("is_pump", False)])
    idx_pump_stations = [i for i,s in enumerate(stations) if s.get("is_pump", False)]
    bypass_scenarios = []
    from itertools import combinations
    # Each scenario: list of pump station indices that will be used in order (can skip any)
    for r in range(1, Nstn+1):
        for use in combinations(idx_pump_stations, r):
            bypass_scenarios.append(list(use))

    # --- Brute-force enumeration of all scenarios ---
    min_total_cost = np.inf
    best_result = {}
    scenario_id = 0

    for bypass in bypass_scenarios:
        # --- For these pumping stations (by index in stations list), enumerate combos ---
        per_station_opts = []
        for idx in bypass:
            stn = stations[idx]
            # Only 2 types: Type A and B (by default, assume one pump curve—extend if needed)
            maxA = min(3, stn.get("max_pumps",1))
            maxB = min(3, stn.get("max_pumps_B",0))  # If B not present, maxB=0
            if "A" in stn.get("pump_types",["A"]):
                combos = pump_combos(maxA, maxB)
            else:
                combos = pump_combos(maxA, maxB)
            drs = drag_steps(stn.get("max_dr",0.0))
            # For each: (nA, nB), DR%
            per_station_opts.append([(c, dr) for c in combos for dr in drs])

        # --- Each configuration is one tuple: [((nA,nB), DR) for each pump station in bypass] ---
        for config in itertools.product(*per_station_opts):
            scenario_id += 1
            # Prepare all segment results
            segment_results = []
            total_power_cost = 0
            total_dra_cost = 0
            feasible = True

            # --- Forward sweep along pipeline, station by station, using chosen bypasses only ---
            heads_out = []
            rh_out = []
            effs_out = []
            speeds_out = []
            dra_out = []
            pumpnum_out = []
            cost_out = []
            velocity_out = []
            reynolds_out = []
            flow_out = []
            headloss_out = []
            maop_out = []
            sdhs_out = []
            residual_heads = []
            drag_reductions = []
            dra_ppm_out = []

            curr_head = None
            curr_elev = stations[0]['elev']
            curr_rh = stations[0].get('min_residual',50.0)
            curr_idx = 0
            curr_flow = FLOW
            idx_bypass = 0
            for stn_idx, stn in enumerate(stations):
                key = stn['name'].lower().replace(' ','_')
                # Segment properties
                D = stn['D']
                t = stn['t']
                SMYS = stn['SMYS']
                rough = stn['rough']
                L = stn['L']
                max_dr = stn.get('max_dr', 0.0)
                visc = float(kv_list[stn_idx])
                dens = float(rho_list[stn_idx])
                is_pump = stn.get('is_pump', False)
                # --- Calculate peaks/elevation for this segment ---
                peaks = stn.get('peaks',[])
                elev_start = curr_elev
                elev_end = stations[stn_idx+1]['elev'] if stn_idx+1<len(stations) else terminal['elev']
                peak_head = get_peaks_extra_head(peaks, elev_start)
                # If this is a pumping station being operated in current scenario
                if stn_idx in bypass:
                    # --- Find combo/DR for this pump station in config ---
                    nA, nB = config[idx_bypass][0]
                    DR = config[idx_bypass][1]
                    idx_bypass += 1
                    # Only Type-A pumps supported in basic version; can extend to multi-type if you provide curves.
                    if nA>0:
                        # Use A/B/C and P/Q/R/S/T from stn
                        # Solve: For given SDH_required, find N so that total pump head matches
                        # ---- Compute system head for this segment at this flow and DRA
                        head_loss, vel, Re = darcy_weisbach_head_loss(curr_flow, D, L, rough, visc, dens, DR)
                        sdh_required = head_loss + curr_rh + (elev_end-elev_start) + peak_head
                        # --- Now find pump speed N where sum of nA pump heads at N matches sdh_required
                        A = stn.get('A',0); B = stn.get('B',0); C = stn.get('C',0)
                        P = stn.get('P',0); Qc = stn.get('Q',0); R = stn.get('R',0)
                        S = stn.get('S',0); T = stn.get('T',0)
                        DOL = stn.get('DOL',1500)
                        MinRPM = stn.get('MinRPM',1000)
                        max_rpm = DOL
                        # For single type pumps, just multiply by nA
                        found_N = None
                        eff_val = None
                        for rpm in range(int(MinRPM), int(max_rpm)+1, 10):
                            head_per_pump = eval_head_curve(curr_flow, (A,B,C), rpm, DOL)
                            total_head = nA * head_per_pump
                            if total_head >= sdh_required-EPS:
                                found_N = rpm
                                eff_val = eval_eff_curve(curr_flow, (P,Qc,R,S,T), rpm, DOL)
                                break
                        if not found_N:
                            feasible = False
                            break
                        # --- Power & cost
                        eff_val = max(eff_val, 1.0)
                        kw = (g * curr_flow/3600 * sdh_required * dens) / (eff_val/100) / 1000
                        if stn['power_type'] == 'Grid':
                            power_cost = kw * HOURS_PER_DAY * stn['rate']
                        else:
                            kwh = kw * HOURS_PER_DAY
                            # Convert to litres from kWh using SFC (gm/bhp.hr)
                            bhp = kw/0.746
                            fuel_lph = stn['sfc'] * bhp / 1e3
                            fuel_cost = fuel_lph * HOURS_PER_DAY * Price_HSD
                            power_cost = fuel_cost
                        # --- DRA cost
                        ppm = get_ppm_for_dr(visc, DR)
                        dra_vol_L = curr_flow * 1000 * HOURS_PER_DAY / 1e6 * ppm
                        dra_cost = dra_vol_L * RateDRA
                        # --- MAOP
                        maop_val = calc_maop(SMYS, D, t)
                        # --- Append to lists
                        pumpnum_out.append(nA)
                        speeds_out.append(found_N)
                        effs_out.append(eff_val)
                        cost_out.append(power_cost)
                        dra_out.append(dra_cost)
                        heads_out.append(sdh_required)
                        rh_out.append(curr_rh)
                        drag_reductions.append(DR)
                        dra_ppm_out.append(ppm)
                        velocity_out.append(vel)
                        reynolds_out.append(Re)
                        flow_out.append(curr_flow)
                        headloss_out.append(head_loss)
                        maop_out.append(maop_val)
                        sdhs_out.append(sdh_required)
                        residual_heads.append(curr_rh)
                        total_power_cost += power_cost
                        total_dra_cost += dra_cost
                        # For next segment
                        curr_rh = 50.0  # Always minimum RH for next (can be extended to float if desired)
                        curr_elev = elev_end
                        # If pressure exceeds MAOP, mark as infeasible
                        if sdh_required > maop_val:
                            feasible = False
                            break
                    else:
                        feasible = False
                        break
                else:
                    # Not pumping, just pipeline segment. Add head loss, DRA as 0.
                    head_loss, vel, Re = darcy_weisbach_head_loss(curr_flow, D, L, rough, visc, dens, 0)
                    sdh_required = head_loss + curr_rh + (elev_end-elev_start) + peak_head
                    heads_out.append(sdh_required)
                    rh_out.append(curr_rh)
                    pumpnum_out.append(0)
                    speeds_out.append(0)
                    effs_out.append(0)
                    cost_out.append(0)
                    dra_out.append(0)
                    drag_reductions.append(0)
                    dra_ppm_out.append(0)
                    velocity_out.append(vel)
                    reynolds_out.append(Re)
                    flow_out.append(curr_flow)
                    headloss_out.append(head_loss)
                    maop_val = calc_maop(SMYS, D, t)
                    maop_out.append(maop_val)
                    sdhs_out.append(sdh_required)
                    residual_heads.append(curr_rh)
                    curr_elev = elev_end
                    curr_rh = 50.0

            # --- After all segments, check global feasibility
            if not feasible:
                continue

            total_cost = total_power_cost + total_dra_cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                # --- Compose output dict for frontend ---
                res = {}
                for i, stn in enumerate(stations):
                    key = stn['name'].lower().replace(' ','_')
                    res[f"num_pumps_{key}"] = int(pumpnum_out[i]) if i<len(pumpnum_out) else 0
                    res[f"speed_{key}"] = float(speeds_out[i]) if i<len(speeds_out) else 0.0
                    res[f"sdh_{key}"] = float(sdhs_out[i]) if i<len(sdhs_out) else 0.0
                    res[f"residual_head_{key}"] = float(residual_heads[i]) if i<len(residual_heads) else 0.0
                    res[f"drag_reduction_{key}"] = float(drag_reductions[i]) if i<len(drag_reductions) else 0.0
                    res[f"power_cost_{key}"] = float(cost_out[i]) if i<len(cost_out) else 0.0
                    res[f"efficiency_{key}"] = float(effs_out[i]) if i<len(effs_out) else 0.0
                    res[f"velocity_{key}"] = float(velocity_out[i]) if i<len(velocity_out) else 0.0
                    res[f"reynolds_{key}"] = float(reynolds_out[i]) if i<len(reynolds_out) else 0.0
                    res[f"head_loss_{key}"] = float(headloss_out[i]) if i<len(headloss_out) else 0.0
                    res[f"maop_{key}"] = float(maop_out[i]) if i<len(maop_out) else 0.0
                    res[f"pipeline_flow_{key}"] = float(flow_out[i]) if i<len(flow_out) else 0.0
                    res[f"dra_ppm_{key}"] = float(dra_ppm_out[i]) if i<len(dra_ppm_out) else 0.0
                res["total_cost"] = float(total_cost)
                best_result = res

    if not best_result:
        raise Exception("No feasible configuration found.")
    return best_result
