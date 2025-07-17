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

def friction_factor_swamee_jain(D, Q, nu, e=0.045):
    # All units: D in m, Q in m3/h, nu in cSt, e in mm
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

def optimize_pipeline(station_data, q_h_curve, q_eff_curve, Qm3h, viscosity):
    dra_curve_dict = load_dra_curves()
    N = len(station_data)
    summary = []
    results = []

    # Decision space
    min_DR = 0
    max_DR = 60
    DR_step = 1
    allowed_DR = list(range(min_DR, max_DR+1, DR_step))
    NOP_range = [1,2,3]

    # Extract parameters
    elevs = [float(stn.get("Elevation (m)", 0)) for stn in station_data]
    dists = [float(stn.get("Distance (km)", 0)) for stn in station_data]
    lengths = [dists[i+1] - dists[i] if i+1 < len(dists) else 0 for i in range(len(dists))]
    Ds = [float(stn.get("Diameter (mm)", 500))/1000 for stn in station_data]
    es = [float(stn.get("Roughness (mm)", 0.045)) for stn in station_data]
    maops = [float(stn.get("MAOP (m)", 900)) for stn in station_data]

    # Peaks
    peaks = []
    for stn in station_data:
        p = stn.get("Peaks", [])
        if isinstance(p, str) and p.strip() == "":
            p = []
        if not isinstance(p, list):
            try:
                p = eval(p)
            except:
                p = []
        peaks.append(p if p else [])

    for idx, stn in enumerate(station_data):
        Station = stn["Station"]
        D = Ds[idx]
        e = es[idx]
        nu = viscosity
        L = lengths[idx]*1000 if idx < N-1 else 1000
        MAOP = maops[idx]

        # Origin station cannot be bypassed
        if idx == 0:
            allowed_NOP = [n for n in NOP_range if n >= 1]
        else:
            allowed_NOP = [0] + NOP_range

        best_cost = np.inf
        best_config = None
        all_configs = []

        for NOP in allowed_NOP:
            for DR in allowed_DR:
                # Friction Factor with DRA effect
                f = friction_factor_swamee_jain(D, Qm3h, nu, e) * (1 - DR/100)
                A = np.pi/4 * D**2
                v = Qm3h / 3600 / A
                hf = 8*f*L*Qm3h**2/(np.pi**2*9.81*D**5*3600**2)
                elev_dn = elevs[idx+1] if idx+1<N else elevs[idx]
                head_next = elev_dn - elevs[idx] + hf + 50  # 50 = RHmin

                # Peak pressure logic
                heads_peaks = []
                if peaks[idx] and idx < N-1:
                    for ep in peaks[idx]:
                        if idx+1 < N:
                            frac = (ep - elevs[idx]) / (elevs[idx+1] - elevs[idx]) if elevs[idx+1] != elevs[idx] else 0.5
                            Lpeak = L * frac
                            hf_peak = 8*f*Lpeak*Qm3h**2/(np.pi**2*9.81*D**5*3600**2)
                            heads_peaks.append((ep - elevs[idx]) + hf_peak + 50)
                all_heads = [head_next] + heads_peaks
                SDH = max(all_heads) if NOP else 0

                # Q-H, Q-Eff interpolated from curves
                if NOP == 0:
                    Head = 0
                    Eff = 0.01
                else:
                    Head = np.interp(Qm3h, [row['Flow (m3/h)'] for row in q_h_curve], [row['Head (m)'] for row in q_h_curve])
                    Eff = np.interp(Qm3h, [row['Flow (m3/h)'] for row in q_eff_curve], [row['Efficiency (%)'] for row in q_eff_curve])

                Total_Head = Head * NOP if NOP >= 1 else 0
                PPM = interpolate_dra_ppm(dra_curve_dict, viscosity, DR)
                dra_cost = PPM * Qm3h * 24

                # Robust fuel rate
                fr = stn.get('Fuel Rate (Rs/kWh)', 12.5)
                try:
                    fr_val = float(fr)
                    if np.isnan(fr_val): fr_val = 12.5
                except (ValueError, TypeError):
                    fr_val = 12.5

                # Power (kW), power cost (Rs/day)
                power = (Qm3h * Total_Head * 9.81) / (max(Eff, 0.01) / 100 * 3.6e6) if NOP else 0
                power_cost = power * 24 * fr_val

                total = dra_cost + power_cost

                # Constraints
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
                        "Station": Station,
                        "NOP": NOP,
                        "%DR": DR,
                        "Head": Total_Head,
                        "Eff": Eff,
                        "SDH": SDH,
                        "PPM": PPM,
                        "DRA Cost": dra_cost,
                        "Power Cost": power_cost,
                        "Total Cost": total,
                        "Bypass": (NOP == 0),
                        "Feasible": feasible,
                        "Constraint": constraint
                    }
                all_configs.append({
                    "Station": Station,
                    "NOP": NOP,
                    "%DR": DR,
                    "Head": Total_Head,
                    "Eff": Eff,
                    "SDH": SDH,
                    "PPM": PPM,
                    "DRA Cost": dra_cost,
                    "Power Cost": power_cost,
                    "Total Cost": total,
                    "Bypass": (NOP == 0),
                    "Feasible": feasible,
                    "Constraint": constraint
                })

        summary.append({
            "Station": best_config["Station"] if best_config else Station,
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
        results.append({"Station": Station, "AllConfigs": all_configs})

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
