import os
import numpy as np
import pandas as pd
from math import log10, pi

# --------- DRA Curve Data Loader ----------
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
        DRA_CURVE_DATA[cst] = None

def safe_int(x, default=0):
    # Handles dicts, floats, strings safely
    if isinstance(x, dict):
        for key in ['value', 'Value', 'min', 'Min', 'rpm', 'RPM']:
            if key in x:
                return int(x[key])
        if x:
            return int(list(x.values())[0])
        return int(default)
    try:
        return int(float(x))
    except Exception:
        return int(default)

def get_ppm_breakpoints(visc):
    cst_list = sorted([c for c in DRA_CURVE_DATA.keys() if DRA_CURVE_DATA[c] is not None])
    visc = float(visc)
    if not cst_list:
        return [0], [0]
    if visc <= cst_list[0]:
        df = DRA_CURVE_DATA[cst_list[0]]
    elif visc >= cst_list[-1]:
        df = DRA_CURVE_DATA[cst_list[-1]]
    else:
        lower = max([c for c in cst_list if c <= visc])
        upper = min([c for c in cst_list if c >= visc])
        df_lower = DRA_CURVE_DATA[lower]
        df_upper = DRA_CURVE_DATA[upper]
        x_lower, y_lower = df_lower['%Drag Reduction'].values, df_lower['PPM'].values
        x_upper, y_upper = df_upper['%Drag Reduction'].values, df_upper['PPM'].values
        dr_points = np.unique(np.concatenate((x_lower, x_upper)))
        ppm_points = np.interp(dr_points, x_lower, y_lower)*(upper-visc)/(upper-lower) + \
                     np.interp(dr_points, x_upper, y_upper)*(visc-lower)/(upper-lower)
        unique_dr, unique_indices = np.unique(dr_points, return_index=True)
        unique_ppm = ppm_points[unique_indices]
        return list(unique_dr), list(unique_ppm)
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    return list(unique_x), unique_y.tolist()

def fit_poly_curve(x, y, order=2):
    try:
        coeffs = np.polyfit(x, y, order)
        return coeffs.tolist()
    except Exception:
        return [0.0] * (order + 1)

def pump_head(flow, rpm, dol, coeffs):
    Q_equiv = flow * dol / rpm if rpm > 0 else 0
    if len(coeffs) == 3:
        return (coeffs[0] * Q_equiv ** 2 + coeffs[1] * Q_equiv + coeffs[2]) * (rpm / dol) ** 2
    else:
        return 0

def pump_eff(flow, rpm, dol, coeffs):
    Q_equiv = flow * dol / rpm if rpm > 0 else 0
    if len(coeffs) == 5:
        return coeffs[0]*Q_equiv**4 + coeffs[1]*Q_equiv**3 + coeffs[2]*Q_equiv**2 + coeffs[3]*Q_equiv + coeffs[4]
    elif len(coeffs) == 3:
        return coeffs[0]*Q_equiv**2 + coeffs[1]*Q_equiv + coeffs[2]
    else:
        return 0

def hydraulic_loss(f, L, d, v, g=9.81, drag_frac=0.0):
    return f * (L*1000.0/d) * (v**2/(2*g)) * (1-drag_frac)

def friction_factor(Re, e, d):
    if Re < 4000:
        return 64.0 / Re if Re > 0 else 0
    else:
        if d > 0 and e > 0 and Re > 0:
            return 0.25 / (log10(e/(3.7*d) + 5.74/(Re**0.9)))**2
        else:
            return 0

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, rpm_step=100, dra_step=5):
    # --- Brute-force grid search ---
    max_types = 2
    max_nop = 3

    N = len(stations)
    results_list = []

    # Calculate segment flows
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}; peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    for i, stn in enumerate(stations, start=1):
        length[i] = stn.get('L', 0.0)
        if 'D' in stn:
            D_out = stn['D']
            thickness[i] = stn.get('t', default_t)
            d_inner[i] = D_out - 2*thickness[i]
        elif 'd' in stn:
            d_inner[i] = stn.get('d', 0.7)
            thickness[i] = stn.get('t', default_t)
        else:
            d_inner[i] = 0.7
            thickness[i] = default_t
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
        peaks_dict[i] = stn.get('peaks', [])

    elev[N+1] = terminal.get('elev', 0.0)

    # Prepare search grids
    station_enum_list = []
    for idx, stn in enumerate(stations, start=1):
        if stn.get("is_pump", False):
            pump_types = stn["pumps"][:max_types]  # Only up to 2 types
            pump_choices = []
            for typidx, pt in enumerate(pump_types):
                nops = list(range(0, safe_int(pt.get("max_units", max_nop))+1))
                pump_choices.append(nops)
            combos = []
            for nopA in pump_choices[0]:
                if len(pump_choices) > 1:
                    for nopB in pump_choices[1]:
                        combos.append((nopA, nopB))
                else:
                    combos.append((nopA, 0))
            min_rpm1 = safe_int(pump_types[0].get("MinRPM", 1000))
            max_rpm1 = safe_int(pump_types[0].get("DOL", 3000))
            rpms1 = list(range(min_rpm1, max_rpm1+1, rpm_step))
            if rpms1[-1] != max_rpm1:
                rpms1.append(max_rpm1)
            if len(pump_types) > 1:
                min_rpm2 = safe_int(pump_types[1].get("MinRPM", 1000))
                max_rpm2 = safe_int(pump_types[1].get("DOL", 3000))
                rpms2 = list(range(min_rpm2, max_rpm2+1, rpm_step))
                if rpms2[-1] != max_rpm2:
                    rpms2.append(max_rpm2)
            else:
                rpms2 = [0]
            max_dr = safe_int(stn.get("max_dr", 30))
            drs = list(range(0, max_dr+1, dra_step))
            if drs[-1] != max_dr:
                drs.append(max_dr)
            station_enum_list.append((combos, rpms1, rpms2, drs))
        else:
            station_enum_list.append(([(0,0)], [0], [0], [0]))

    from itertools import product
    all_indices = []
    for station_opts in station_enum_list:
        combos, rpms1, rpms2, drs = station_opts
        indices = list(product(range(len(combos)), range(len(rpms1)), range(len(rpms2)), range(len(drs))))
        all_indices.append(indices)

    all_combination_indices = list(product(*all_indices))

    for cfg in all_combination_indices:
        station_config = []
        valid = True
        total_cost = 0.0
        sdh = {}; rh = {}; dra_ppm = {}; dra_cost = {}; velocity = {}; reynolds = {}; friction = {}
        rpm_out = {}; tdh_out = {}; eff_out = {}; pumpnum_out = {}; dra_perc_out = {}

        # Initial RH at start
        rh[1] = stations[0].get('min_residual', 50.0)
        for idx, sidx in enumerate(cfg, start=1):
            combos, rpms1, rpms2, drs = station_enum_list[idx-1]
            c_idx, r1_idx, r2_idx, d_idx = sidx
            nops = combos[c_idx]
            rpm1 = rpms1[r1_idx]
            rpm2 = rpms2[r2_idx]
            drag = drs[d_idx]
            stn = stations[idx-1]
            pump_types = stn["pumps"][:max_types] if stn.get("is_pump", False) else []
            stn_dict = {
                'NOP1': nops[0], 'NOP2': nops[1], 'RPM1': rpm1, 'RPM2': rpm2, 'DR': drag,
                'PumpType1': pump_types[0].get('name', f"Type {1}") if pump_types else '',
                'PumpType2': pump_types[1].get('name', f"Type {2}") if len(pump_types)>1 else ''
            }
            station_config.append(stn_dict)

        # First station must have at least one pump on if it is a pump station
        if stations[0].get("is_pump", False):
            if station_config[0]['NOP1'] + station_config[0]['NOP2'] == 0:
                continue

        rh[1] = stations[0].get('min_residual', 50.0)
        for i in range(1, N+1):
            stn = stations[i-1]
            pump_flag = stn.get("is_pump", False)
            flow = float(segment_flows[i])
            area = pi * (d_inner[i]**2)/4.0
            v = flow/3600.0/area if area > 0 else 0.0
            kv = float(KV_list[i-1])
            rho = float(rho_list[i-1])
            if kv > 0:
                Re = v*d_inner[i]/(kv*1e-6)
            else:
                Re = 0.0
            f = friction_factor(Re, roughness[i], d_inner[i])
            velocity[i] = v; reynolds[i] = Re; friction[i] = f
            stn_cfg = station_config[i-1]
            dra_frac = stn_cfg['DR']/100.0 if pump_flag and stn_cfg['DR']>0 else 0.0
            HL = hydraulic_loss(f, length[i], d_inner[i], v, drag_frac=dra_frac)
            D_out = d_inner[i] + 2 * thickness[i]
            MAOP_head = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/D_out)*10000.0/rho
            sdh[i] = rh[i] + (elev[i+1]-elev[i]) + HL
            tdh = 0.0
            eff = []
            power_cost = 0.0
            dra_cost[i] = 0.0
            dra_ppm[i] = 0.0
            rpm_out[i] = 0.0
            tdh_out[i] = 0.0
            eff_out[i] = 0.0
            pumpnum_out[i] = 0
            dra_perc_out[i] = 0.0
            if pump_flag:
                for typidx in range(max_types):
                    nops = stn_cfg['NOP1'] if typidx==0 else stn_cfg['NOP2']
                    rpm = stn_cfg['RPM1'] if typidx==0 else stn_cfg['RPM2']
                    if nops > 0 and rpm > 0:
                        pt = stn['pumps'][typidx]
                        coeffs_H = pt.get('head_coeffs', [0,0,0])
                        if not coeffs_H or len(coeffs_H)<3:
                            hdata = pt.get('head_data', [])
                            if hdata and len(hdata)>=3:
                                qv = [float(x['Flow (m³/hr)']) for x in hdata]
                                hv = [float(x['Head (m)']) for x in hdata]
                                coeffs_H = fit_poly_curve(qv, hv, 2)
                        coeffs_E = pt.get('eff_coeffs', [0,0,0,0,0])
                        if not coeffs_E or len(coeffs_E)<3:
                            edata = pt.get('eff_data', [])
                            if edata and len(edata)>=3:
                                qv = [float(x['Flow (m³/hr)']) for x in edata]
                                ev = [float(x['Efficiency (%)']) for x in edata]
                                coeffs_E = fit_poly_curve(qv, ev, 4 if len(edata)>=5 else 2)
                        _tdh = pump_head(flow, rpm, safe_int(pt.get("DOL", rpm)), coeffs_H) * nops
                        _eff = pump_eff(flow, rpm, safe_int(pt.get("DOL", rpm)), coeffs_E)
                        tdh += _tdh
                        eff.append(_eff)
                        if _eff > 0:
                            if pt.get("power_type","grid").lower() == "grid":
                                rate = float(pt.get("rate",0.0))
                                power_kW = (rho*flow*9.81*_tdh)/(3600.0*1000.0*(_eff/100.0)*0.95)
                                power_cost += power_kW*24.0*rate
                            else:
                                sfc = float(pt.get("sfc",0.0))
                                fuel_per_kWh = (sfc*1.34102)/820.0
                                power_kW = (rho*flow*9.81*_tdh)/(3600.0*1000.0*(_eff/100.0)*0.95)
                                power_cost += power_kW*24.0*fuel_per_kWh*Price_HSD
                        visc = kv
                        dr_points, ppm_points = get_ppm_breakpoints(visc)
                        ppm_interp = np.interp([stn_cfg['DR']], dr_points, ppm_points)[0] if len(dr_points)>1 else 0.0
                        dra_ppm[i] += ppm_interp
                        dra_cost[i] += ppm_interp*(flow*1000.0*24.0/1e6)*RateDRA*nops
                        rpm_out[i] += rpm * nops
                        tdh_out[i] += _tdh
                        eff_out[i] += _eff * nops
                        pumpnum_out[i] += nops
                        dra_perc_out[i] += stn_cfg['DR']
                    else:
                        eff.append(0.0)
                if pumpnum_out[i] > 0:
                    rpm_out[i] = rpm_out[i] / pumpnum_out[i]
                    eff_out[i] = eff_out[i] / pumpnum_out[i]
                    dra_perc_out[i] = dra_perc_out[i] / pumpnum_out[i]
                rh[i+1] = sdh[i] + tdh
            else:
                rh[i+1] = sdh[i]
            if rh[i+1] < 50:
                valid = False; break
            if sdh[i] > MAOP_head:
                valid = False; break
            for peak in peaks_dict[i]:
                L_peak = peak['loc']*1000.0
                elev_k = peak['elev']
                loss_no_dra = friction_factor(Re, roughness[i], d_inner[i])*(L_peak/d_inner[i])*(v**2/(2*9.81))
                if (rh[i]+tdh - (elev_k-elev[i]) - loss_no_dra) < 50:
                    valid = False; break
            total_cost += power_cost + dra_cost[i]
        if valid:
            result_out = {
                "total_cost": total_cost,
                "station_config": station_config,
                "sdh": sdh,
                "rh": rh,
                "dra_ppm": dra_ppm,
                "dra_cost": dra_cost,
                "velocity": velocity,
                "reynolds": reynolds,
                "friction": friction,
                "rpm": rpm_out,
                "tdh": tdh_out,
                "eff": eff_out,
                "num_pumps": pumpnum_out,
                "dra_perc": dra_perc_out,
            }
            results_list.append(result_out)

    # Sort and return top 3 (or just best)
    top_results = sorted(results_list, key=lambda x: x['total_cost'])[:3]
    return top_results
