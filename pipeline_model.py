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

def swamee_jain(Re, e, d):
    if Re < 4000 and Re > 0:
        return 64.0 / Re
    elif Re >= 4000 and d > 0 and e >= 0:
        A = e/(3.7*d)
        B = 5.74/(Re**0.9)
        return 0.25/(log10(A + B))**2
    else:
        return 0

def pump_head(flow, rpm, dol, coeffs):
    Q_equiv = flow * dol / rpm if rpm > 0 else 0
    if len(coeffs) == 3:
        return (coeffs[0]*Q_equiv**2 + coeffs[1]*Q_equiv + coeffs[2]) * (rpm/dol)**2
    else:
        return 0

def pump_eff(flow, rpm, dol, coeffs, base_rpm):
    if rpm <= 0:
        return 0
    shifted_flow = flow * (base_rpm / rpm)
    if len(coeffs) == 5:
        return coeffs[0]*shifted_flow**4 + coeffs[1]*shifted_flow**3 + coeffs[2]*shifted_flow**2 + coeffs[3]*shifted_flow + coeffs[4]
    elif len(coeffs) == 3:
        return coeffs[0]*shifted_flow**2 + coeffs[1]*shifted_flow + coeffs[2]
    else:
        return 0

def hydraulic_loss(f, L, d, v, g=9.81, drag_frac=0.0):
    return f * (L/d) * (v**2/(2*g)) * (1-drag_frac)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def fit_poly_curve(x, y, order=2):
    try:
        coeffs = np.polyfit(x, y, order)
        return coeffs.tolist()
    except Exception:
        return [0.0] * (order + 1)

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, *args, rpm_step=50, dra_step=5):
    max_types = 2
    max_nop = 3
    N = len(stations)
    results_list = []

    # Segment flows
    segment_flows = [safe_float(FLOW)]
    for stn in stations:
        delivery = safe_float(stn.get('delivery', 0.0))
        supply = safe_float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # Pipe properties and geometry
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}; peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    for i, stn in enumerate(stations, start=1):
        length[i] = safe_float(stn.get('L', 0.0)) * 1000.0  # Ensure L in meters!
        if 'D' in stn:
            D_out = safe_float(stn['D'])
            thickness[i] = safe_float(stn.get('t', default_t))
            d_inner[i] = D_out - 2*thickness[i]
        elif 'd' in stn:
            d_inner[i] = safe_float(stn['d'])
            thickness[i] = safe_float(stn.get('t', default_t))
        else:
            d_inner[i] = 0.7
            thickness[i] = default_t
        roughness[i] = safe_float(stn.get('rough', default_e))
        smys[i] = safe_float(stn.get('SMYS', default_smys))
        design_factor[i] = safe_float(stn.get('DF', default_df))
        elev[i] = safe_float(stn.get('elev', 0.0))
        peaks_dict[i] = stn.get('peaks', [])

    elev[N+1] = safe_float(terminal.get('elev', 0.0))

    # Prepare brute force search space
    from itertools import product
    station_options = []
    for idx, stn in enumerate(stations, start=1):
        if stn.get("is_pump", False):
            pump_types = stn.get("pumps", [])[:max_types]
            pump_choices = []
            for typidx, pt in enumerate(pump_types):
                nops = list(range(0, int(pt.get("max_units", max_nop))+1))
                pump_choices.append(nops)
            combos = []
            for nopA in pump_choices[0]:
                if len(pump_choices) > 1:
                    for nopB in pump_choices[1]:
                        combos.append((nopA, nopB))
                else:
                    combos.append((nopA, 0))
            min_rpm1 = int(pump_types[0].get("MinRPM", 1000)) if len(pump_types) > 0 else 0
            max_rpm1 = int(pump_types[0].get("DOL", 3000)) if len(pump_types) > 0 else 0
            rpms1 = list(range(min_rpm1, max_rpm1+1, rpm_step)) if min_rpm1 and max_rpm1 else [0]
            if rpms1 and (rpms1[-1] != max_rpm1):
                rpms1.append(max_rpm1)
            if len(pump_types) > 1:
                min_rpm2 = int(pump_types[1].get("MinRPM", 1000))
                max_rpm2 = int(pump_types[1].get("DOL", 3000))
                rpms2 = list(range(min_rpm2, max_rpm2+1, rpm_step))
                if rpms2 and (rpms2[-1] != max_rpm2):
                    rpms2.append(max_rpm2)
            else:
                rpms2 = [0]
            max_dr = int(stn.get("max_dr", 30))
            drs = list(range(0, max_dr+1, dra_step))
            if drs and drs[-1] != max_dr:
                drs.append(max_dr)
            station_options.append((combos, rpms1, rpms2, drs))
        else:
            station_options.append(([(0, 0)], [0], [0], [0]))

    all_indices = []
    for combos, rpms1, rpms2, drs in station_options:
        indices = list(product(range(len(combos)), range(len(rpms1)), range(len(rpms2)), range(len(drs))))
        all_indices.append(indices)
    all_combination_indices = list(product(*all_indices))

    for cfg in all_combination_indices:
        station_config = []
        valid = True
        total_cost = 0.0
        sdh = {}; rh = {}; dra_ppm = {}; dra_cost = {}; velocity = {}; reynolds = {}; friction = {}; head_loss = {}
        rh[1] = stations[0].get('min_residual', 50.0)
        per_station = {}
        for i in range(1, N+1):
            stn = stations[i-1]
            name = stn['name'].strip().lower().replace(' ', '_')
            flow = segment_flows[i]
            area = pi * (d_inner[i]**2)/4.0
            v = flow/3600.0/area if area > 0 else 0.0
            kv = safe_float(KV_list[i-1])
            rho = safe_float(rho_list[i-1])
            Re = v*d_inner[i]/(kv*1e-6) if kv > 0 else 0.0
            f = swamee_jain(Re, roughness[i], d_inner[i])
            velocity[i] = v; reynolds[i] = Re; friction[i] = f

            combos, rpms1, rpms2, drs = station_options[i-1]
            c_idx, r1_idx, r2_idx, d_idx = cfg[i-1]
            nops = combos[c_idx]
            rpm1 = rpms1[r1_idx]
            rpm2 = rpms2[r2_idx]
            drag = drs[d_idx]
            pump_types = stn.get("pumps", [])[:max_types] if stn.get("is_pump", False) else []
            type1 = pump_types[0]['type'] if len(pump_types) > 0 and 'type' in pump_types[0] else ''
            type2 = pump_types[1]['type'] if len(pump_types) > 1 and 'type' in pump_types[1] else ''
            stn_dict = {
                'NOP1': nops[0], 'NOP2': nops[1], 'RPM1': rpm1, 'RPM2': rpm2, 'DR': drag,
                'PumpType1': type1,
                'PumpType2': type2
            }
            station_config.append(stn_dict)
            dr_frac = drag/100.0 if drag > 0 else 0.0
            HL = hydraulic_loss(f, length[i], d_inner[i], v, drag_frac=dr_frac)
            head_loss[i] = HL
            rh_next = rh[i] if i == 1 else max(rh[i], 50.0)
            delta_z = elev[i+1] - elev[i]
            sdh[i] = HL + rh_next + delta_z
            tdh = 0.0; eff = []; power_cost = 0.0; dra_cost[i] = 0.0; dra_ppm[i] = 0.0
            for typidx in range(max_types):
                nop = nops[typidx]
                rpm = rpm1 if typidx == 0 else rpm2
                if nop > 0 and rpm > 0 and typidx < len(pump_types):
                    pt = pump_types[typidx]
                    head_data = pt.get('head_data', [])
                    eff_data = pt.get('eff_data', [])
                    dol = int(pt.get("DOL", 3000))
                    base_rpm = dol
                    if head_data and (not pt.get('head_coeffs')):
                        qv = [safe_float(x['Flow (m³/hr)']) for x in head_data]
                        hv = [safe_float(x['Head (m)']) for x in head_data]
                        pt['head_coeffs'] = fit_poly_curve(qv, hv, 2)
                    if eff_data and (not pt.get('eff_coeffs')):
                        qv = [safe_float(x['Flow (m³/hr)']) for x in eff_data]
                        ev = [safe_float(x['Efficiency (%)']) for x in eff_data]
                        pt['eff_coeffs'] = fit_poly_curve(qv, ev, 4 if len(ev) >= 5 else 2)
                    coeffs_H = pt.get('head_coeffs', [0, 0, 0])
                    coeffs_E = pt.get('eff_coeffs', [0, 0, 0, 0, 0])
                    PH = pump_head(flow, rpm, dol, coeffs_H) * nop
                    if PH + 1e-4 < sdh[i]:
                        valid = False
                        break
                    rh[i+1] = PH - HL + delta_z if PH > sdh[i] else max(50, rh[i])
                    _eff = pump_eff(flow, rpm, dol, coeffs_E, base_rpm)
                    eff.append(_eff)
                    if _eff > 0:
                        if pt.get("power_type", "grid").lower() == "grid":
                            rate = safe_float(pt.get("rate", 0.0))
                            power_kW = (rho * flow * 9.81 * PH) / (3600.0 * 1000.0 * (_eff / 100.0) * 0.95)
                            power_cost += power_kW * 24.0 * rate
                        else:
                            sfc = safe_float(pt.get("sfc", 0.0))
                            fuel_per_kWh = (sfc * 1.34102) / 820.0
                            power_kW = (rho * flow * 9.81 * PH) / (3600.0 * 1000.0 * (_eff / 100.0) * 0.95)
                            power_cost += power_kW * 24.0 * fuel_per_kWh * Price_HSD
                    else:
                        power_cost += 0.0
                    visc = kv
                    dr_points, ppm_points = get_ppm_breakpoints(visc)
                    ppm_interp = np.interp([drag], dr_points, ppm_points)[0] if len(dr_points) > 1 else 0.0
                    dra_ppm[i] += ppm_interp
                    dra_cost[i] += ppm_interp * (flow * 1000.0 * 24.0 / 1e6) * RateDRA * nop
                    per_station[f"num_pumps_{name}_type{typidx+1}"] = nop
                    per_station[f"speed_{name}_type{typidx+1}"] = rpm
                    per_station[f"efficiency_{name}_type{typidx+1}"] = _eff
                else:
                    eff.append(0.0)
                    per_station[f"num_pumps_{name}_type{typidx+1}"] = 0
                    per_station[f"speed_{name}_type{typidx+1}"] = 0
                    per_station[f"efficiency_{name}_type{typidx+1}"] = 0
            total_cost += power_cost + dra_cost[i]
            D_out = d_inner[i] + 2 * thickness[i]
            MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho if D_out > 0 and rho > 0 else 1e6
            if sdh[i] > MAOP_head:
                valid = False
                break
            if rh.get(i+1, 0) < 50:
                valid = False
                break
            for peak in peaks_dict[i]:
                L_peak = peak['loc'] * 1000.0
                elev_k = peak['elev']
                loss_no_dra = swamee_jain(Re, roughness[i], d_inner[i]) * (L_peak/d_inner[i]) * (v**2/(2*9.81))
                if (rh[i] - (elev_k - elev[i]) - loss_no_dra) < 50:
                    valid = False
                    break
            per_station[f"pipeline_flow_{name}"] = flow
            per_station[f"head_loss_{name}"] = HL
            per_station[f"residual_head_{name}"] = rh[i]
            per_station[f"sdh_{name}"] = sdh[i]
            per_station[f"dra_cost_{name}"] = dra_cost[i]
            per_station[f"dra_ppm_{name}"] = dra_ppm[i]
            per_station[f"friction_{name}"] = f
            per_station[f"velocity_{name}"] = v
            per_station[f"reynolds_{name}"] = Re
        if valid:
            result_out = dict(
                total_cost=total_cost,
                station_config=station_config,
                **per_station
            )
            results_list.append(result_out)

    top_results = sorted(results_list, key=lambda x: x['total_cost'])[:3]
    if not top_results:
        return {"error": True, "message": "No feasible solution found. Please check your input and relax constraints."}
    result = top_results[0]
    result['top3'] = top_results
    result['error'] = False
    return result
