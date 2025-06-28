import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
import numpy as np
import pandas as pd
import json

# Set NEOS email
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

# DRA CSV loading logic - as in your repository, adjust path as needed
DRA_CSV_FILES = {10: "10 cst.csv", 15: "15 cst.csv", 20: "20 cst.csv",
                 25: "25 cst.csv", 30: "30 cst.csv", 35: "35 cst.csv", 40: "40 cst.csv"}
DRA_CURVE_DATA = {}
for cst, fname in DRA_CSV_FILES.items():
    DRA_CURVE_DATA[cst] = pd.read_csv(fname) if os.path.exists(fname) else None

def get_ppm_breakpoints(visc):
    cst_list = sorted([c for c in DRA_CURVE_DATA if DRA_CURVE_DATA[c] is not None])
    visc = float(visc)
    if not cst_list:
        return [0, 100], [0, 0]
    if visc <= cst_list[0]:
        df = DRA_CURVE_DATA[cst_list[0]]
    elif visc >= cst_list[-1]:
        df = DRA_CURVE_DATA[cst_list[-1]]
    else:
        lower = max(c for c in cst_list if c <= visc)
        upper = min(c for c in cst_list if c >= visc)
        if lower != upper:
            df_low, df_up = DRA_CURVE_DATA[lower], DRA_CURVE_DATA[upper]
            x_low, y_low = df_low['%Drag Reduction'].values, df_low['PPM'].values
            x_up, y_up = df_up['%Drag Reduction'].values, df_up['PPM'].values
            dr_points = np.unique(np.concatenate((x_low, x_up)))
            ppm_interp = (np.interp(dr_points, x_low, y_low) * (upper - visc)/(upper - lower) +
                          np.interp(dr_points, x_up, y_up) * (visc - lower)/(upper - lower))
            unique_dr, idx = np.unique(dr_points, return_index=True)
            unique_ppm = ppm_interp[idx]
            if len(unique_dr) < 2:
                return [0, 100], [0, 0]
            return list(unique_dr), list(unique_ppm)
        df = DRA_CURVE_DATA[lower]
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    unique_x, idx = np.unique(x, return_index=True)
    unique_y = y[idx]
    if len(unique_x) < 2:
        return [0, 100], [0, 0]
    return list(unique_x), list(unique_y)

def solve_pipeline_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    stations = data['stations']
    terminal = data['terminal']
    FLOW = data['FLOW']
    Rate_DRA = data['RateDRA']
    Price_HSD = data['Price_HSD']
    linefill = data['linefill']

    # Build viscosity and density lists per station from linefill (assuming uniform here)
    KV_list = [float(linefill[0]['Viscosity (cSt)'])] * len(stations)
    rho_list = [float(linefill[0]['Density (kg/m³)'])] * len(stations)

    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)
    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=float(FLOW))
    model.Rate_DRA = pyo.Param(initialize=float(Rate_DRA))
    model.Price_HSD = pyo.Param(initialize=float(Price_HSD))

    # Geometry/parameters
    length = {}
    d_inner = {}
    thick = {}
    rough = {}
    smys = {}
    elev = {}
    design_fac = {}
    min_rpm = {}
    max_rpm = {}
    max_dr = {}
    elec_rate = {}
    sfc = {}
    pump_stations = []
    peaks = {}

    for i, stn in enumerate(stations, start=1):
        length[i] = float(stn['L'])
        thick[i] = float(stn['t'])
        d_inner[i] = float(stn['D']) - 2 * thick[i]
        rough[i] = float(stn['rough'])
        smys[i] = float(stn['SMYS'])
        elev[i] = float(stn['elev'])
        design_fac[i] = 0.72
        min_rpm[i] = int(stn.get('MinRPM', 1))
        max_rpm[i] = int(stn.get('DOL', 2975))
        max_dr[i] = float(stn.get('max_dr', 0.0))
        peaks[i] = stn.get('peaks', [])
        if stn.get('is_pump', False):
            pump_stations.append(i)
            if stn.get('power_type', '').lower() == 'grid':
                elec_rate[i] = float(stn['rate'])
            else:
                sfc[i] = float(stn['sfc'])

    elev[N+1] = float(terminal['elev'])

    model.L     = pyo.Param(model.I, initialize=length)
    model.d     = pyo.Param(model.I, initialize=d_inner)
    model.e     = pyo.Param(model.I, initialize=rough)
    model.SMYS  = pyo.Param(model.I, initialize=smys)
    model.DF    = pyo.Param(model.I, initialize=design_fac)
    model.z     = pyo.Param(model.Nodes, initialize=elev)
    model.pump_stations = pyo.Set(initialize=pump_stations)
    model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm)
    model.DOL    = pyo.Param(model.pump_stations, initialize=max_rpm)
    model.max_dr = pyo.Param(model.pump_stations, initialize=max_dr)
    model.elec_rate = pyo.Param(model.pump_stations, initialize=elec_rate, default=9.0)

    # Pump variables
    def pump_bounds(m, j):
        lb = 1
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=pump_bounds, initialize=1)
    def speed_bounds(m, j):
        lo = int(min_rpm.get(j, 1))
        hi = int(max_rpm.get(j, 2975))
        return (lo, hi)
    model.N = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals, bounds=speed_bounds, initialize=lambda m, j: int(max_rpm.get(j, 2975)))
    model.DR = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals, bounds=lambda m, j: (0, max_dr.get(j, 0.0)), initialize=0.0)
    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals)

    # Residual head
    model.RH[1].fix(float(stations[0].get('min_residual', 50.0)))
    for node in range(2, N+2):
        model.RH[node].setlb(50.0)

    # Piecewise DRA PPM constraints
    model.dra_piecewise = pyo.ConstraintList()
    for i in pump_stations:
        dr_pts, ppm_pts = get_ppm_breakpoints(kv_dict[i])
        for k in range(1, len(dr_pts)):
            slope = (ppm_pts[k] - ppm_pts[k-1]) / (dr_pts[k] - dr_pts[k-1]) if (dr_pts[k] - dr_pts[k-1]) != 0 else 0
            # Linear envelope (enforces the piecewise)
            model.dra_piecewise.add(
                model.PPM[i] >= ppm_pts[k-1] + slope * (model.DR[i] - dr_pts[k-1])
            )
            model.dra_piecewise.add(
                model.PPM[i] <= ppm_pts[k-1] + slope * (model.DR[i] - dr_pts[k-1]) + 1e-3
            )

    # Head loss and SDH calculations
    g = 9.81
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.segment_constraints = pyo.ConstraintList()
    model.power_use = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    model.tdh = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    model.eff = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)

    # Precompute flow (single segment flow for each station in this example)
    segment_flows = [float(FLOW)] * (N+1)

    # Main hydraulics constraints (friction, head, elevation)
    for i in range(1, N+1):
        # Friction factor (Swamee-Jain for turbulent)
        vel = (segment_flows[i-1] / 3600.0) / (np.pi * (d_inner[i] ** 2) / 4.0)
        Re = vel * d_inner[i] / (kv_dict[i] * 1e-6)
        if Re > 4000:
            eD = rough[i] / d_inner[i]
            f = 0.25 / (np.log10(eD / 3.7 + 5.74 / (Re ** 0.9))) ** 2
        elif Re > 0:
            f = 64.0 / Re
        else:
            f = 0.0
        DR_fraction = model.DR[i] / 100.0 if i in pump_stations else 0.0
        head_loss = f * ((length[i] * 1000.0) / d_inner[i]) * ((vel ** 2) / (2 * g)) * (1 - DR_fraction)
        elev_diff = model.z[i+1] - model.z[i]
        model.segment_constraints.add(model.SDH[i] >= model.RH[i+1] + elev_diff + head_loss)
        for peak in peaks[i]:
            Lp = peak['loc'] * 1000.0
            elev_peak = peak['elev']
            peak_loss = f * (Lp / d_inner[i]) * ((vel ** 2) / (2 * g)) * (1 - DR_fraction)
            model.segment_constraints.add(model.SDH[i] >= (elev_peak - model.z[i]) + peak_loss + 50.0)
    # Pump TDH constraints (affinity law)
    for idx, i in enumerate(pump_stations, 1):
        stn = stations[i-1]
        # Quadratic head curve: H = A*Q^2 + B*Q + C
        A, B, C = float(stn['A']), float(stn['B']), float(stn['C'])
        q = segment_flows[i-1]
        n = model.N[i]
        n_rated = float(stn['DOL'])
        H_full = A * q ** 2 + B * q + C
        tdh_expr = H_full * (n / n_rated) ** 2
        model.tdh[i].set_value(tdh_expr)
        model.segment_constraints.add(model.tdh[i] * model.NOP[i] >= model.SDH[i] - model.RH[i])
        # Efficiency: 4th order polynomial
        P, Qc, R, S, T = float(stn['P']), float(stn['Q']), float(stn['R']), float(stn['S']), float(stn['T'])
        eff_expr = P * q ** 4 + Qc * q ** 3 + R * q ** 2 + S * q + T
        model.eff[i].set_value(eff_expr / 100.0)
        # Power: (rho * Q * g * TDH * NOP) / (3.6e6 * eff)
        power_expr = (rho_dict[i] * q * 9.81 * model.tdh[i] * model.NOP[i]) / (3.6e6 * model.eff[i] + 1e-9)
        model.power_use[i].set_value(power_expr)
    # Objective
    total_cost = 0.0
    for i in pump_stations:
        dra_cost = model.PPM[i] * segment_flows[i-1] * 24 * 1000 / 1e6 * model.Rate_DRA
        power_cost = model.power_use[i] * 24 * model.elec_rate[i]
        total_cost += dra_cost + power_cost
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    results = SolverManagerFactory('neos').solve(model, solver='couenne', tee=True)
    status = results.solver.status
    term = results.solver.termination_condition
    if (status != pyo.SolverStatus.ok) or (term not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        return {"error": True, "message": f"Optimization failed: {term}.", "termination_condition": str(term), "solver_status": str(status)}
    model.solutions.load_from(results)

    output = {}
    for i, stn in enumerate(stations, start=1):
        name_key = stn['name'].strip().lower().replace(' ', '_')
        output[f"num_pumps_{name_key}"] = int(pyo.value(model.NOP[i])) if i in pump_stations else 0
        output[f"speed_{name_key}"] = pyo.value(model.N[i]) if i in pump_stations else 0.0
        output[f"drag_reduction_{name_key}"] = pyo.value(model.DR[i]) if i in pump_stations else 0.0
        output[f"dra_ppm_{name_key}"] = pyo.value(model.PPM[i]) if i in pump_stations else 0.0
        output[f"dra_cost_{name_key}"] = (pyo.value(model.PPM[i]) * segment_flows[i-1] * 24 * 1000 / 1e6 * Rate_DRA) if i in pump_stations else 0.0
        output[f"power_cost_{name_key}"] = pyo.value(model.power_use[i]) * 24 * elec_rate.get(i, 9.0) if i in pump_stations else 0.0
        output[f"residual_head_{name_key}"] = pyo.value(model.RH[i])
        output[f"sdh_{name_key}"] = pyo.value(model.SDH[i])
    output["residual_head_terminal"] = pyo.value(model.RH[N+1])
    return output

# Example usage:
# results = solve_pipeline_from_json("/mnt/data/HBPL.json")
# print(results)