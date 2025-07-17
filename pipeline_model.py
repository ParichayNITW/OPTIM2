# pipeline_model.py
import numpy as np
import pandas as pd
import os

#########################
# DRA Curves: Load ALL CSVs (10 cst.csv, 15 cst.csv, ...)
#########################
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

#########################
# Interpolate PPM from DRA curves for viscosity and DR%
#########################
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

#########################
# Friction Factor (Swamee-Jain for turbulent, else Churchill for full Re range)
#########################
def friction_factor(D, Q, nu, e=0.045):  # D [m], Q [m3/hr], nu [cSt], e=pipe roughness mm
    A = np.pi / 4 * D**2
    v = Q / 3600 / A   # m/s
    Re = v * D / (nu / 1e6)
    eps = e/1000
    if Re < 2000:  # Laminar
        return 64/Re if Re > 0 else 0.03
    else:  # Turbulent
        try:
            f = 0.25/(np.log10(eps/(3.7*D) + 5.74/(Re**0.9)))**2
            return max(f, 0.008)
        except:
            return 0.03

#########################
# SDH logic: returns the minimum required head at the current station to guarantee RH ≥ 50 m at downstream station and at all intermediate peaks (returns max)
#########################
def compute_SDH(elev_this, elev_next, elev_peaks, L, Q, D, nu, allowed_DR, dra_curve_dict, viscosity):
    # For all allowed DR%, compute required inlet head to maintain RH≥50m at next and all peaks
    SDHs = []
    for dr in allowed_DR:
        f = friction_factor(D, Q, nu) * (1 - dr/100)
        hf = 8*f*L*Q**2/(np.pi**2*9.81*D**5*3600**2)
        head_next = (elev_next - elev_this) + hf + 50
        heads_peaks = []
        if elev_peaks:
            for ep in elev_peaks:
                # Compute distance to peak, assume proportional friction
                frac = (ep - elev_this) / (elev_next - elev_this) if elev_next != elev_this else 0.5
                Lpeak = L * frac
                hf_peak = 8*f*Lpeak*Q**2/(np.pi**2*9.81*D**5*3600**2)
                heads_peaks.append((ep - elev_this) + hf_peak + 50)
        all_heads = [head_next] + heads_peaks
        SDHs.append(max(all_heads))
    return max(SDHs)

#########################
# Main optimization
#########################
def optimize_pipeline(station_data, q_h_curve, q_eff_curve, flow_rate, viscosity):
    dra_curve_dict = load_dra_curves()
    N = len(station_data)
    results = []
    summary = []
    min_DR = 0
    max_DR = 60
    DR_step = 1
    allowed_DR = list(range(min_DR, max_DR+1, DR_step))
    # For demo: NOP allowed from 0 to NOP_max at each except origin (origin: NOP >=1)
    # User can expand with per-station NOP_max, RPM etc
    NOP_range = [1,2,3]  # Change as per station
    RPM_fixed = 3000     # For now; can loop over RPM_range if variable speed
    D = 0.5  # m, pipe diameter (sample); you can pass per-station
    nu = viscosity  # cSt
    e = 0.045 # mm, roughness
    L = 50    # km, station spacing; in future, use user value

    # Prepare elevation and peak list per segment
    elevs = [float(stn.get("Elevation (m)", 0)) for stn in station_data]
    peaks = [[] for _ in range(N-1)]
    for i in range(N-1):
        # User can enter peaks; for now, use [] (no peak)
        peaks[i] = []

    # Main loop over stations
    for idx, stn in enumerate(station_data):
        name = stn['Station']
        # Allow NOP = 0 for all except origin
        if idx == 0:
            allowed_NOP = [n for n in NOP_range if n >= 1]
        else:
            allowed_NOP = [0] + NOP_range
        best_cost = 1e99
        best_config = None
        all_configs = []
        for NOP in allowed_NOP:
            for DR in allowed_DR:
                # For this (NOP, DR) combo
                # Compute SDH (required head to maintain RH≥50m at next/peaks, or RH if bypass)
                if NOP == 0:
                    SDH = None  # Will fill after
                else:
                    elev_this = elevs[idx]
                    elev_next = elevs[idx+1] if idx+1 < N else elevs[idx]
                    elev_peaks = peaks[idx] if idx < N-1 else []
                    SDH = compute_SDH(elev_this, elev_next, elev_peaks, L*1000, flow_rate, D, nu, [DR], dra_curve_dict, viscosity)
                # Compute pump head and eff at this flow
                if NOP == 0:
                    Head = 0
                    Eff = 0.01
                else:
                    Head = np.interp(flow_rate, [row['Flow (m3/h)'] for row in q_h_curve], [row['Head (m)'] for row in q_h_curve])
                    Eff = np.interp(flow_rate, [row['Flow (m3/h)'] for row in q_eff_curve], [row['Efficiency (%)'] for row in q_eff_curve])
                # If NOP > 1, assume head is divided among pumps in series; power is sum, eff can be adjusted (or user can input per-NOP eff)
                if NOP >= 1:
                    Total_Head = Head * NOP
                else:
                    Total_Head = 0
                # DRA calculation
                ppm = interpolate_dra_ppm(dra_curve_dict, viscosity, DR)
                dra_cost = ppm * flow_rate * 24 * 1.0  # Example: ppm*flow*time
                # Power
                power = (flow_rate * Total_Head * 9.81) / (max(Eff, 0.01) / 100 * 3.6e6) if NOP else 0
                power_cost = power * 24 * float(stn.get('Fuel Rate (Rs/kWh)', 12.5))
                total = dra_cost + power_cost
                # RH = SDH if NOP == 0, else as per hydraulics (implement as needed)
                if NOP == 0:
                    SDH = 0  # RH at station
                # Check peak pressure, constraints (expand as per actual logic)
                feasible = True
                constraint = ""
                # (insert advanced checks here)
                all_configs.append({
                    "NOP": NOP, "%DR": DR, "Head": Total_Head, "Eff": Eff, "SDH": SDH, "PPM": ppm,
                    "DRA Cost": dra_cost, "Power Cost": power_cost, "Total Cost": total,
                    "Bypass": (NOP==0), "Feasible": feasible, "Constraint": constraint
                })
                if feasible and total < best_cost:
                    best_cost = total
                    best_config = all_configs[-1]
        # Summarize for station
        summary.append({
            "Station": name,
            "NOP": best_config["NOP"],
            "%DR": best_config["%DR"],
            "Head (m)": best_config["Head"],
            "Eff (%)": best_config["Eff"],
            "SDH (m)": best_config["SDH"],
            "PPM": best_config["PPM"],
            "DRA Cost": best_config["DRA Cost"],
            "Power Cost": best_config["Power Cost"],
            "Total Cost": best_config["Total Cost"],
            "Bypass": best_config["Bypass"],
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

# End of pipeline_model.py
