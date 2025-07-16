import numpy as np
import pandas as pd
from itertools import product

def interpolate_curve(x, y, x0):
    x = np.array(x)
    y = np.array(y)
    if len(x) < 2 or len(y) < 2:
        return y[0] if len(y) > 0 else 0.0
    if x0 <= x[0]:
        return y[0]
    if x0 >= x[-1]:
        return y[-1]
    return np.interp(x0, x, y)

def get_ppm_from_dra_curve(dra_curve_dict, viscosity, dr_percent):
    cst_list = sorted(dra_curve_dict.keys())
    if not cst_list:
        return 0.0
    if viscosity <= cst_list[0]:
        df = dra_curve_dict[cst_list[0]]
    elif viscosity >= cst_list[-1]:
        df = dra_curve_dict[cst_list[-1]]
    else:
        lower = max([c for c in cst_list if c <= viscosity])
        upper = min([c for c in cst_list if c >= viscosity])
        df_lower = dra_curve_dict[lower]
        df_upper = dra_curve_dict[upper]
        ppm_lower = interpolate_curve(df_lower['%Drag Reduction'].values, df_lower['PPM'].values, dr_percent)
        ppm_upper = interpolate_curve(df_upper['%Drag Reduction'].values, df_upper['PPM'].values, dr_percent)
        if upper == lower:
            return ppm_lower
        return (ppm_lower * (upper - viscosity) + ppm_upper * (viscosity - lower)) / (upper - lower)
    return interpolate_curve(df['%Drag Reduction'].values, df['PPM'].values, dr_percent)

def pump_head(A, B, C, Q, RPM, DOL):
    if RPM == 0 or DOL == 0:
        return 0.0
    Q_equiv = Q * DOL / RPM
    H_DOL = A * Q_equiv**2 + B * Q_equiv + C
    return H_DOL * (RPM / DOL)**2

def pump_eff(P, Qc, Rc, Sc, Tc, Q, RPM, DOL):
    if RPM == 0 or DOL == 0:
        return 0.0
    Q_equiv = Q * DOL / RPM
    return P * Q_equiv**4 + Qc * Q_equiv**3 + Rc * Q_equiv**2 + Sc * Q_equiv + Tc

def friction_factor_swamee_jain(Re, rough, D):
    if Re < 1e-6 or D == 0:
        return 0.0
    if Re < 4000:
        return 64.0 / Re
    try:
        arg = (rough / D / 3.7) + (5.74 / (Re**0.9))
        return 0.25 / (np.log10(arg)**2) if arg > 0 else 0.0
    except Exception:
        return 0.0

def compute_system_head(L, D, rough, kv, flow, elev_start, elev_end, drag_reduction_percent):
    g = 9.81
    Q = flow / 3600.0
    area = np.pi * (D**2) / 4.0
    v = Q / area if area > 0 else 0.0
    if kv > 0:
        Re = v * D / (kv * 1e-6)
    else:
        Re = 0.0
    f = friction_factor_swamee_jain(Re, rough, D)
    frac = 1.0 - (drag_reduction_percent / 100.0)
    friction_head = f * ((L*1000.0) / D) * (v**2 / (2*g)) * frac
    elevation_head = elev_end - elev_start
    return friction_head + elevation_head, v, area, Re, f, friction_head, elevation_head

def check_maop(thickness, SMYS, DF, D_out, density, sdh_val):
    maop_head = (2 * thickness * (SMYS * 0.070307) * DF / D_out) * 10000.0 / density
    return sdh_val <= maop_head, maop_head

def check_peaks(L, D, rough, kv, flow, elev_start, peaks):
    g = 9.81
    peak_report = []
    Q = flow / 3600.0
    area = np.pi * (D**2) / 4.0
    v = Q / area if area > 0 else 0.0
    if kv > 0:
        Re = v * D / (kv * 1e-6)
    else:
        Re = 0.0
    f = friction_factor_swamee_jain(Re, rough, D)
    for pk in peaks:
        L_peak = pk.get('loc', 0.0) * L * 1000.0
        elev_k = pk.get('elev', 0.0)
        loss_no_dra = f * (L_peak / D) * (v**2 / (2*g))
        peak_head = elev_k - elev_start + loss_no_dra + 50.0
        peak_report.append({
            'L_peak': L_peak,
            'elev_peak': elev_k,
            'loss_no_dra': loss_no_dra,
            'required_head_at_peak': peak_head
        })
    return peak_report

def optimize_pipeline(
    stations,
    terminal,
    FLOW,
    kv_list,
    rho_list,
    RateDRA,
    Price_HSD,
    dra_curve_dict,
    rpm_step = 100,
    dra_step = 1
):
    N = len(stations)
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)
    elev = {}
    for i, stn in enumerate(stations, start=1):
        elev[i] = stn.get('elev', 0.0)
    elev[N+1] = terminal.get('elev', 0.0)
    total_cost = 0.0
    best_config = []
    upstream_head = stations[0].get('min_residual', 50.0)
    results = {}
    raw_results = []

    for idx, stn in enumerate(stations, start=1):
        name = stn['name']
        is_pump = stn.get('is_pump', False)
        flow = float(segment_flows[idx-1])
        kv = float(kv_list[idx-1])
        rho = float(rho_list[idx-1])
        L = float(stn.get('L', 0.0))
        D = float(stn.get('D', 0.0))
        t = float(stn.get('t', 0.007))
        rough = float(stn.get('rough', 0.00004))
        SMYS = float(stn.get('SMYS', 52000))
        DF = float(stn.get('DF', 0.72))
        elev_start = elev[idx]
        elev_end = elev[idx+1]
        peaks = stn.get('peaks', [])
        station_result = {
            "name": name,
            "is_pump": is_pump,
            "best_cost": float('inf'),
            "best_config": {},
            "all_feasible": [],
            "all_configs": []
        }
        if is_pump:
            max_pumps = int(stn.get('max_pumps', 2))
            min_nop = 1 if idx == 1 else 0
            A, B, C = stn.get('A', 0.0), stn.get('B', 0.0), stn.get('C', 0.0)
            P, Qc, Rc, Sc, Tc = stn.get('P', 0.0), stn.get('Q', 0.0), stn.get('R', 0.0), stn.get('S', 0.0), stn.get('T', 0.0)
            min_rpm, max_rpm = int(stn.get('MinRPM', 0)), int(stn.get('DOL', 0))
            max_dr = int(stn.get('max_dr', 0))
            allowed_nop = list(range(min_nop, max_pumps + 1))
            allowed_rpm = list(range(min_rpm, max_rpm + 1, rpm_step))
            if allowed_rpm and allowed_rpm[-1] != max_rpm:
                allowed_rpm.append(max_rpm)
            allowed_dr = list(range(0, max_dr + 1, dra_step))
            if allowed_dr and allowed_dr[-1] != max_dr:
                allowed_dr.append(max_dr)
            combos = product(allowed_nop, allowed_rpm, allowed_dr)
        else:
            combos = [(0, 0, 0)]
        for NOP, RPM, DR in combos:
            rejection_reason = None
            if is_pump and NOP == 0:
                continue
            sys_head, v, area, Re, f, friction_head, elevation_head = compute_system_head(L, D, rough, kv, flow, elev_start, elev_end, DR)
            required_head = sys_head - upstream_head
            if required_head < 0:
                required_head = 0.0
            peak_report = check_peaks(L, D, rough, kv, flow, elev_start, peaks) if peaks else []
            peak_ok = True
            peak_violated = None
            for j, pk in enumerate(peak_report):
                if sys_head < pk['required_head_at_peak']:
                    peak_ok = False
                    peak_violated = j
                    break
            if not peak_ok:
                rejection_reason = f"Peak constraint violated at peak {peak_violated}"
            if is_pump:
                H_pump = pump_head(A, B, C, flow, RPM, max_rpm) * NOP
                eff = pump_eff(P, Qc, Rc, Sc, Tc, flow, RPM, max_rpm)
            else:
                H_pump = 0.0
                eff = 1.0
            if is_pump and H_pump < required_head - 1e-2:
                rejection_reason = "Pump head insufficient"
            D_out = D + 2 * t
            sdh_val = sys_head
            maop_ok, maop_head = check_maop(t, SMYS, DF, D_out, rho, sdh_val)
            if not maop_ok:
                rejection_reason = "MAOP violated"
            if is_pump:
                ppm = get_ppm_from_dra_curve(dra_curve_dict, kv, DR)
                dra_mass_per_day = ppm * (flow * 1000.0 * 24.0 / 1e6)
                dra_cost = dra_mass_per_day * RateDRA
            else:
                dra_cost = 0.0
                ppm = 0.0
            if is_pump and eff > 0 and NOP > 0:
                power_kW = (rho * flow * 9.81 * H_pump) / (3600.0 * 1000.0 * (eff/100.0) * 0.95)
                if stn.get('sfc', 0) not in (None, 0):
                    sfc_val = stn.get('sfc', 0.0)
                    fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                    power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
                else:
                    rate = stn.get('rate', 0.0)
                    power_cost = power_kW * 24.0 * rate
            else:
                power_cost = 0.0
            total_station_cost = power_cost + dra_cost
            conf = {
                'NOP': NOP, 'RPM': RPM, 'DR': DR, 'H_pump': H_pump,
                'Head_req': required_head, 'Eff': eff, 'DRA_cost': dra_cost,
                'PPM': ppm, 'Power_cost': power_cost, 'Total_cost': total_station_cost,
                'Peak_ok': peak_ok, 'MAOP_ok': maop_ok, 'sys_head': sys_head,
                'velocity': v, 'area': area,
                'Re': Re, 'friction_factor': f,
                'friction_head': friction_head, 'elevation_head': elevation_head,
                'maop_head': maop_head,
                'peaks': peak_report, 'rejection_reason': rejection_reason
            }
            station_result['all_configs'].append(conf)
            if not rejection_reason:
                station_result['all_feasible'].append(conf)
                if total_station_cost < station_result['best_cost']:
                    station_result['best_cost'] = total_station_cost
                    station_result['best_config'] = conf
        if station_result['best_config']:
            best_c = station_result['best_config']
            upstream_head = best_c['sys_head']
            total_cost += best_c['Total_cost']
        else:
            upstream_head = upstream_head + 0.0
        best_config.append(station_result)
        results[f"station_{idx}_{name.strip().replace(' ','_').lower()}"] = station_result
        raw_results.append({
            "station": name,
            "configs": station_result['all_configs']
        })
    results['total_cost'] = total_cost
    results['best_config'] = best_config
    results['raw_results'] = raw_results

    summary_table = []
    station_tables = {}
    for station_info in results['best_config']:
        name = station_info['name']
        best = station_info['best_config']
        if not best:
            row = {
                "Station": name, "NOP": None, "RPM": None, "DR%": None, "Head_Required (m)": None,
                "Head_Generated (m)": None, "Eff (%)": None, "DRA_PPM": None, "DRA_Cost": None,
                "Power_Cost": None, "Total_Cost": None, "Peak_OK": False, "MAOP_OK": False
            }
        else:
            row = {
                "Station": name,
                "NOP": int(best['NOP']),
                "RPM": int(best['RPM']),
                "DR%": float(best['DR']),
                "Head_Required (m)": round(float(best['Head_req']),2),
                "Head_Generated (m)": round(float(best['H_pump']),2),
                "Eff (%)": round(float(best['Eff']),2),
                "DRA_PPM": round(float(best['PPM']),2),
                "DRA_Cost": round(float(best['DRA_cost']),2),
                "Power_Cost": round(float(best['Power_cost']),2),
                "Total_Cost": round(float(best['Total_cost']),2),
                "Peak_OK": bool(best['Peak_ok']),
                "MAOP_OK": bool(best['MAOP_ok'])
            }
        summary_table.append(row)
    for station_info in results['best_config']:
        name = station_info['name']
        feas = station_info['all_feasible']
        if feas:
            df = pd.DataFrame(feas)
            colmap = {
                "NOP": "NOP",
                "RPM": "RPM",
                "DR": "DR%",
                "Head_req": "Head_Required (m)",
                "H_pump": "Head_Generated (m)",
                "Eff": "Eff (%)",
                "PPM": "DRA_PPM",
                "DRA_cost": "DRA_Cost",
                "Power_cost": "Power_Cost",
                "Total_cost": "Total_Cost",
                "Peak_ok": "Peak_OK",
                "MAOP_ok": "MAOP_OK"
            }
            df = df[list(colmap.keys())]
            df = df.rename(columns=colmap)
            for col in ["DR%", "Head_Required (m)", "Head_Generated (m)", "Eff (%)", "DRA_PPM", "DRA_Cost", "Power_Cost", "Total_Cost"]:
                df[col] = df[col].round(2)
            station_tables[name] = df
        else:
            station_tables[name] = pd.DataFrame([{"No feasible configs": "See rejection reasons in raw_results"}])
    csv_tables = {k: v.to_csv(index=False) for k,v in station_tables.items()}

    return {
        "summary_table": summary_table,
        "station_tables": station_tables,
        "csv_tables": csv_tables,
        "best_config": results['best_config'],
        "total_cost": round(float(results['total_cost']),2),
        "raw_results": results['raw_results']
    }
