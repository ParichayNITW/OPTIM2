import numpy as np
import pandas as pd
import os

def load_dra_curves():
    dra_curve_dict = {}
    for fname in os.listdir():
        if fname.endswith("cst.csv"):
            try:
                vis = int(fname.split()[0])
                df = pd.read_csv(fname)
                dra_curve_dict[vis] = df
            except Exception:
                continue
    return dra_curve_dict

def interpolate_dra_ppm(dra_curve_dict, viscosity, dr_percent):
    keys = sorted(dra_curve_dict.keys())
    if not keys:
        return 0.0
    if viscosity <= keys[0]:
        df = dra_curve_dict[keys[0]]
    elif viscosity >= keys[-1]:
        df = dra_curve_dict[keys[-1]]
    else:
        lower = max(k for k in keys if k <= viscosity)
        upper = min(k for k in keys if k >= viscosity)
        df_l = dra_curve_dict[lower]
        df_u = dra_curve_dict[upper]
        y_l = np.interp(dr_percent, df_l['%Drag Reduction'], df_l['PPM'])
        y_u = np.interp(dr_percent, df_u['%Drag Reduction'], df_u['PPM'])
        return y_l + (y_u - y_l) * (viscosity - lower) / (upper - lower)
    return np.interp(dr_percent, df['%Drag Reduction'], df['PPM'])

def friction_factor(D, Q, nu, e=0.045):
    A = np.pi / 4 * D**2
    v = Q / 3600 / A
    Re = v * D / (nu / 1e6)
    eps = e/1000
    if Re < 2000:
        return 64/Re if Re > 0 else 0.03
    else:
        try:
            f = 0.25/(np.log10(eps/(3.7*D) + 5.74/(Re**0.9)))**2
            return max(f, 0.008)
        except:
            return 0.03

def optimize_pipeline(station_data, q_h_curve, q_eff_curve, flow_rate, viscosity):
    dra_curve_dict = load_dra_curves()
    N = len(station_data)
    results = []
    summary = []
    min_DR = 0
    max_DR = 60
    DR_step = 1
    allowed_DR = list(range(min_DR, max_DR+1, DR_step))
    NOP_range = [1,2,3]
    RPM_fixed = 3000
    elevs = [float(stn.get("Elevation (m)", 0)) for stn in station_data]
    lengths = [float(stn.get("Distance (km)", 50)) for stn in station_data]
    Ds = [float(stn.get("Diameter (mm)", 500))/1000 for stn in station_data]
    es = [float(stn.get("Roughness (mm)", 0.045)) for stn in station_data]
    maops = [float(stn.get("MAOP (m)", 900)) for stn in station_data]
    peaks = [[] for _ in range(N-1)]
    for i in range(N-1):
        peaks[i] = []
    for idx, stn in enumerate(station_data):
        name = stn['Station']
        if idx == 0:
            allowed_NOP = [n for n in NOP_range if n >= 1]
        else:
            allowed_NOP = [0] + NOP_range
        best_cost = 1e99
        best_config = None
        all_configs = []
        for NOP in allowed_NOP:
            for DR in allowed_DR:
                D = Ds[idx]
                e = es[idx]
                nu = viscosity
                L = lengths[idx] * 1000 if idx < N-1 else 1000
                MAOP = maops[idx]
                f = friction_factor(D, flow_rate, nu, e) * (1 - DR/100)
                A = np.pi / 4 * D**2
                v = flow_rate / 3600 / A
                hf = 8*f*L*flow_rate**2/(np.pi**2*9.81*D**5*3600**2)
                head_next = (elevs[idx+1] if idx+1<N else elevs[idx]) - elevs[idx] + hf + 50
                heads_peaks = []
                if peaks[idx] and idx < N-1:
                    for ep in peaks[idx]:
                        frac = (ep - elevs[idx]) / (elevs[idx+1] - elevs[idx]) if elevs[idx+1] != elevs[idx] else 0.5
                        Lpeak = L * frac
                        hf_peak = 8*f*Lpeak*flow_rate**2/(np.pi**2*9.81*D**5*3600**2)
                        heads_peaks.append((ep - elevs[idx]) + hf_peak + 50)
                all_heads = [head_next] + heads_peaks
                SDH = max(all_heads) if NOP else 0
                if NOP == 0:
                    Head = 0
                    Eff = 0.01
                else:
                    Head = np.interp(flow_rate, [row['Flow (m3/h)'] for row in q_h_curve], [row['Head (m)'] for row in q_h_curve])
                    Eff = np.interp(flow_rate, [row['Flow (m3/h)'] for row in q_eff_curve], [row['Efficiency (%)'] for row in q_eff_curve])
                Total_Head = Head * NOP if NOP >= 1 else 0
                ppm = interpolate_dra_ppm(dra_curve_dict, viscosity, DR)
                dra_cost = ppm * flow_rate * 24
                power = (flow_rate * Total_Head * 9.81) / (max(Eff, 0.01) / 100 * 3.6e6) if NOP else 0
                power_cost = power * 24 * float(stn.get('Fuel Rate (Rs/kWh)', 12.5))
                total = dra_cost + power_cost
                feasible = True
                constraint = ""
                outlet_head = SDH + elevs[idx]
                if outlet_head > MAOP:
                    feasible = False
                    constraint = "MAOP exceeded"
                if NOP == 0 and idx == 0:
                    feasible = False
                    constraint = "Origin cannot be bypassed"
                if feasible and total < best_cost:
                    best_cost = total
                    best_config = {
                        "NOP": NOP, "%DR": DR, "Head": Total_Head, "Eff": Eff, "SDH": SDH, "PPM": ppm,
                        "DRA Cost": dra_cost, "Power Cost": power_cost, "Total Cost": total,
                        "Bypass": (NOP==0), "Feasible": feasible, "Constraint": constraint
                    }
                all_configs.append({
                    "NOP": NOP, "%DR": DR, "Head": Total_Head, "Eff": Eff, "SDH": SDH, "PPM": ppm,
                    "DRA Cost": dra_cost, "Power Cost": power_cost, "Total Cost": total,
                    "Bypass": (NOP==0), "Feasible": feasible, "Constraint": constraint
                })
        summary.append({
            "Station": name,
            "NOP": best_config["NOP"] if best_config else 0,
            "%DR": best_config["%DR"] if best_config else 0,
            "Head (m)": best_config["Head"] if best_config else 0,
            "Eff (%)": best_config["Eff"] if best_config else 0,
            "SDH (m)": best_config["SDH"] if best_config else 0,
            "PPM": best_config["PPM"] if best_config else 0,
            "DRA Cost": best_config["DRA Cost"] if best_config else 0,
            "Power Cost": best_config["Power Cost"] if best_config else 0,
            "Total Cost": best_config["Total Cost"] if best_config else 0,
            "Bypass": best_config["Bypass"] if best_config else False,
            "Feasible": best_config["Feasible"] if best_config else False,
            "Constraint": best_config["Constraint"] if best_config else "",
        })
        results.append({"Station": name, "AllConfigs": all_configs})
    total_cost = sum([row["Total Cost"] for row in summary])
    out = {
        "summary_table": summary,
        "station_tables": {row["Station"]: pd.DataFrame(r["AllConfigs"]) for row, r in zip(summary, results)},
        "total_cost": total_cost,
        "raw_results": results,
        "csv_tables": {row["Station"]: pd.DataFrame(r["AllConfigs"]).to_csv(index=False) for row, r in zip(summary, results)}
    }
    return out

def solve_pipeline(*args, **kwargs):
    return optimize_pipeline(*args, **kwargs)
