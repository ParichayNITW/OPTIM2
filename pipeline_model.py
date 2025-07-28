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
    g = 9.81  # m/s²
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

    def darcy_weisbach_head_loss(Q, D, L, rough, visc, dens, drag_reduction):
        A = np.pi * (D/2)**2  # m²
        v = Q / 3600 / A      # m/s
        Re = dens * v * D / (visc * 1e-6)
        eps_D = rough / D
        if Re < 2000:
            f = 64/Re
        else:
            f = 0.25/(np.log10(eps_D/3.7 + 5.74/(Re**0.9)))**2
        f = f * (1 - drag_reduction/100)
        hl = f * (L*1000) / D * v**2 / (2*g)
        return hl, v, Re

    def get_peaks_extra_head(peaks, stn_elev):
        extra = 0
        for pk in peaks:
            if pk["elev"] > stn_elev:
                extra = max(extra, pk["elev"] - stn_elev)
        return extra

    def calc_maop(SMYS, D, t):
        D_mm = D * 1000
        t_mm = t * 1000
        SMYS_MPa = SMYS * 0.006895
        maop = 2 * t_mm * SMYS_MPa / (D_mm * 1.5) * 10.1972  # bar
        return maop

    def pump_combos(max_n):
        return list(range(1, max_n+1))

    def drag_steps(max_dr):
        return np.round(np.arange(0, max_dr+0.01, 2.5),2)

    pump_station_indices = [i for i,s in enumerate(stations) if s.get("is_pump",False)]
    bypass_scenarios = []
    from itertools import combinations
    for r in range(1, len(pump_station_indices)+1):
        for use in combinations(pump_station_indices, r):
            bypass_scenarios.append(list(use))

    min_total_cost = np.inf
    best_result = {}

    for bypass in bypass_scenarios:
        per_station_opts = []
        for idx in bypass:
            stn = stations[idx]
            combos = pump_combos(stn.get("max_pumps",1))
            drs = drag_steps(stn.get("max_dr",0.0))
            per_station_opts.append([(n, dr) for n in combos for dr in drs])

        for config in itertools.product(*per_station_opts):
            total_power_cost = 0
            total_dra_cost = 0
            feasible = True

            curr_elev = stations[0]['elev']
            curr_rh = stations[0].get('min_residual',50.0)
            curr_flow = FLOW
            idx_bypass = 0

            pumpnum_out = []
            speeds_out = []
            effs_out = []
            cost_out = []
            dra_out = []
            heads_out = []
            rh_out = []
            drag_reductions = []
            dra_ppm_out = []
            velocity_out = []
            reynolds_out = []
            flow_out = []
            headloss_out = []
            maop_out = []
            sdhs_out = []
            residual_heads = []

            for stn_idx, stn in enumerate(stations):
                key = stn['name'].lower().replace(' ','_')
                D = stn['D']
                t = stn['t']
                SMYS = stn['SMYS']
                rough = stn['rough']
                L = stn['L']
                max_dr = stn.get('max_dr', 0.0)
                visc = float(kv_list[stn_idx])
                dens = float(rho_list[stn_idx])
                is_pump = stn.get('is_pump', False)
                peaks = stn.get('peaks',[])
                elev_start = curr_elev
                elev_end = stations[stn_idx+1]['elev'] if stn_idx+1<len(stations) else terminal['elev']
                peak_head = get_peaks_extra_head(peaks, elev_start)
                if stn_idx in bypass:
                    n = config[idx_bypass][0]
                    DR = config[idx_bypass][1]
                    idx_bypass += 1
                    # ---- Use tabular head/eff curves ----
                    head_table = stn.get('head_curve', None)
                    eff_table = stn.get('eff_curve', None)
                    DOL = stn.get('DOL',1500)
                    MinRPM = stn.get('MinRPM',1000)
                    max_rpm = DOL
                    if head_table is None or eff_table is None or len(head_table) < 2 or len(eff_table) < 2:
                        print(f"[DEBUG] Station {stn['name']}: Missing or too short pump table. Skipping scenario.")
                        feasible = False
                        break
                    head_table = sorted(head_table, key=lambda x: x['flow'])
                    eff_table = sorted(eff_table, key=lambda x: x['flow'])

                    head_flows = np.array([row['flow'] for row in head_table])
                    head_heads = np.array([row['head'] for row in head_table])
                    eff_flows = np.array([row['flow'] for row in eff_table])
                    eff_effs = np.array([row['eff'] for row in eff_table])

                    head_loss, vel, Re = darcy_weisbach_head_loss(curr_flow, D, L, rough, visc, dens, DR)
                    sdh_required = head_loss + curr_rh + (elev_end-elev_start) + peak_head

                    found_N = None
                    eff_val = None

                    # Try each speed (down to MinRPM), use affinity laws + interpolation for each
                    for rpm in range(int(MinRPM), int(max_rpm)+1, 10):
                        speed_ratio = rpm / DOL
                        Q_equiv = curr_flow / speed_ratio
                        if Q_equiv < head_flows[0] or Q_equiv > head_flows[-1]:
                            continue
                        head_at_Q_equiv = np.interp(Q_equiv, head_flows, head_heads)
                        head_at_N = head_at_Q_equiv * (speed_ratio)**2  # Affinity law
                        total_head = n * head_at_N
                        if total_head >= sdh_required-EPS:
                            eff_at_Q_equiv = np.interp(Q_equiv, eff_flows, eff_effs)
                            found_N = rpm
                            eff_val = eff_at_Q_equiv
                            break

                    if not found_N or eff_val is None or eff_val < 1e-3:
                        print(f"[DEBUG] Station {stn['name']}: No RPM ({MinRPM}-{max_rpm}) gives enough head for SDH {sdh_required:.2f}m, n={n}. Max curve head: {np.max(head_heads):.2f}.")
                        feasible = False
                        break

                    kw = (g * curr_flow/3600 * sdh_required * dens) / (eff_val/100) / 1000
                    if stn['power_type'] == 'Grid':
                        power_cost = kw * HOURS_PER_DAY * stn['rate']
                    else:
                        kwh = kw * HOURS_PER_DAY
                        bhp = kw/0.746
                        fuel_lph = stn['sfc'] * bhp / 1e3
                        fuel_cost = fuel_lph * HOURS_PER_DAY * Price_HSD
                        power_cost = fuel_cost
                    ppm = get_ppm_for_dr(visc, DR)
                    dra_kg_per_day = curr_flow * 1000 * HOURS_PER_DAY * ppm / 1e9
                    dra_cost = dra_kg_per_day * RateDRA
                    maop_val = calc_maop(SMYS, D, t)
                    sdh_bar = sdh_required * dens * g / 1e5
                    if sdh_bar > maop_val:
                        print(f"[DEBUG] Station {stn['name']}: SDH (bar) {sdh_bar:.2f} > MAOP {maop_val:.2f}")
                        feasible = False
                        break
                    pumpnum_out.append(n)
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
                    curr_rh = 50.0
                    curr_elev = elev_end
                else:
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

            if not feasible:
                continue

            total_cost = total_power_cost + total_dra_cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
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
