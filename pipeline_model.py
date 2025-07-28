import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'parichay.nitwarangal@gmail.com')

# DRA curve files
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
    return list(unique_x), list(unique_y)

def safe_polyfit(x, y, degree):
    if len(x) >= degree + 1:
        return np.polyfit(x, y, degree)
    else:
        return [0] * (degree + 1)

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None):
    # Curve fits for each type at each station
    for idx, stn in enumerate(stations):
        pump_types = stn.get('pump_types', ['A'])
        for ptype in pump_types:
            head_df = pd.DataFrame(stn.get(f"head_data_{ptype}", []))
            if head_df is not None and len(head_df) >= 3:
                x = head_df.iloc[:,0].values
                y = head_df.iloc[:,1].values
                A, B, C = safe_polyfit(x, y, 2)
            else:
                A = B = C = 0.0
            eff_df = pd.DataFrame(stn.get(f"eff_data_{ptype}", []))
            if eff_df is not None and len(eff_df) >= 5:
                x = eff_df.iloc[:,0].values
                y = eff_df.iloc[:,1].values
                P, Q, R, S, T = safe_polyfit(x, y, 4)
            else:
                P = Q = R = S = T = 0.0
            stn[f'coef_{ptype}'] = dict(A=float(A), B=float(B), C=float(C), P=float(P), Q=float(Q), R=float(R), S=float(S), T=float(T))

    N = len(stations)
    pump_types_set = set()
    station_pump_types = {}
    for i, stn in enumerate(stations, 1):
        if stn.get('is_pump', False):
            types = stn.get('pump_types', ['A'])
            station_pump_types[i] = types
            pump_types_set.update(types)
    pump_types_all = sorted(list(pump_types_set))

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)
    model.KV = pyo.Param(model.I, initialize={i: float(KV_list[i-1]) for i in range(1, N+1)})
    model.rho = pyo.Param(model.I, initialize={i: float(rho_list[i-1]) for i in range(1, N+1)})

    # Segment flows
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # Pipeline & hydraulic parameters
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}; peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    for i, stn in enumerate(stations, 1):
        length[i] = stn.get('L', 0.0)
        D_out = stn.get('D', 0.711)
        thickness[i] = stn.get('t', default_t)
        d_inner[i] = D_out - 2*thickness[i]
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
        peaks_dict[i] = stn.get('peaks', [])
    elev[N+1] = terminal.get('elev', 0.0)
    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)

    pump_indices = [i for i, stn in enumerate(stations, 1) if stn.get('is_pump', False)]
    model.pump_stations = pyo.Set(initialize=pump_indices)
    model.pump_types = pyo.Set(initialize=pump_types_all)

    def n_bounds(m, i, t):
        max_ = stations[i-1].get(f"max_{t}", 0)
        return (0, max_)
    model.n_pumps = pyo.Var(model.pump_stations, model.pump_types, domain=pyo.NonNegativeIntegers, bounds=n_bounds)
    def max_total_rule(m, i):
        return sum(m.n_pumps[i, t] for t in station_pump_types[i]) <= stations[i-1].get('max_pumps', 2)
    model.max_total_pumps = pyo.Constraint(model.pump_stations, rule=max_total_rule)
    def min_origin(m):
        i = pump_indices[0]
        return sum(m.n_pumps[i, t] for t in station_pump_types[i]) >= 1
    model.min_origin = pyo.Constraint(rule=min_origin)
    def rpm_bounds(m, i, t):
        ptinfo = stations[i-1][f"pump_{t}"]
        return (ptinfo.get('MinRPM', 1000), ptinfo.get('DOL', 1500))
    model.rpm = pyo.Var(model.pump_stations, model.pump_types, domain=pyo.NonNegativeReals, bounds=rpm_bounds)

    # DRA, RH, as before
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        flow_m3s = float(segment_flows[i]) / 3600.0
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        kv = float(KV_list[i-1])
        if kv > 0:
            Re[i] = v[i] * d_inner[i] / (kv * 1e-6)
        else:
            Re[i] = 0.0
        if Re[i] > 0:
            if Re[i] < 4000:
                f[i] = 64.0 / Re[i]
            else:
                arg = (roughness[i] / d_inner[i] / 3.7) + (5.74 / (Re[i]**0.9))
                f[i] = 0.25 / (log10(arg)**2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    for i in range(1, N+1):
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2 / (2*g))
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g))
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

    model.head_balance = pyo.ConstraintList()
    for i in range(1, N+1):
        if i in pump_indices:
            tdh_total = sum(
                model.n_pumps[i, t] *
                (stations[i-1][f'coef_{t}']['A'] * (float(segment_flows[i]) * stations[i-1][f"pump_{t}"]['DOL'] / (model.rpm[i, t]+1e-6))**2 +
                 stations[i-1][f'coef_{t}']['B'] * (float(segment_flows[i]) * stations[i-1][f"pump_{t}"]['DOL'] / (model.rpm[i, t]+1e-6)) +
                 stations[i-1][f'coef_{t}']['C']
                ) * (model.rpm[i, t] / stations[i-1][f"pump_{t}"]['DOL'])**2
                for t in station_pump_types[i]
            )
            model.head_balance.add(model.RH[i] + tdh_total >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])

    # Objective & DRA
    total_cost = 0
    for i in pump_indices:
        for t in station_pump_types[i]:
            ptinfo = stations[i-1][f"pump_{t}"]
            coef = stations[i-1][f'coef_{t}']
            n = model.n_pumps[i, t]
            rpm = model.rpm[i, t]
            dol = ptinfo.get('DOL', 1500)
            pump_flow = float(segment_flows[i])
            Qeq = pump_flow * dol / (rpm+1e-6)
            H_DOL = coef['A'] * Qeq**2 + coef['B'] * Qeq + coef['C']
            TDH_one = H_DOL * (rpm/dol)**2
            eff = (coef['P']*Qeq**4 + coef['Q']*Qeq**3 + coef['R']*Qeq**2 + coef['S']*Qeq + coef['T']) / 100.0
            eff = max(eff, 0.01)
            rho_i = float(rho_list[i-1])
            power_kW = (rho_i * pump_flow * 9.81 * TDH_one * n) / (3600.0 * 1000.0 * eff * 0.95)
            rate = ptinfo.get('rate', 9.0)
            sfc_val = ptinfo.get('sfc', 0.0)
            if rate > 0:
                power_cost = power_kW * 24.0 * rate
            else:
                fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
            total_cost += power_cost
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # Solve
    results = SolverManagerFactory('neos').solve(model, solver='couenne', tee=False)
    status = results.solver.status
    term = results.solver.termination_condition
    if (status != pyo.SolverStatus.ok) or (term not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        return {
            "error": True,
            "message": f"Optimization failed: {term}. Please check your input values and relax constraints if necessary.",
            "termination_condition": str(term),
            "solver_status": str(status)
        }
    model.solutions.load_from(results)

    # Output: every hydraulic parameter and per-type outputs
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        result[f"pipeline_flow_{name}"] = outflow
        result[f"pipeline_flow_in_{name}"] = inflow

        # Hydraulics
        result[f"velocity_{name}"] = v[i]
        result[f"reynolds_{name}"] = Re[i]
        result[f"friction_{name}"] = f[i]
        result[f"residual_head_{name}"] = pyo.value(model.RH[i])
        result[f"sdh_{name}"] = pyo.value(model.SDH[i])
        result[f"head_loss_{name}"] = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1] - model.z[i]))))
        D_out = d_inner[i] + 2*thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / float(rho_list[i-1])
        result[f"maop_{name}"] = MAOP_head

        # Per-type pump data
        if i in pump_indices:
            for t in station_pump_types[i]:
                ptinfo = stn[f"pump_{t}"]
                coef = stn[f'coef_{t}']
                n_val = int(round(pyo.value(model.n_pumps[i, t])))
                rpm_val = float(pyo.value(model.rpm[i, t]))
                pump_flow = float(segment_flows[i])
                Qeq = pump_flow * ptinfo.get('DOL', 1500) / (rpm_val+1e-6)
                H_DOL = coef['A'] * Qeq**2 + coef['B'] * Qeq + coef['C']
                TDH_one = H_DOL * (rpm_val/ptinfo.get('DOL', 1500))**2
                eff = (coef['P']*Qeq**4 + coef['Q']*Qeq**3 + coef['R']*Qeq**2 + coef['S']*Qeq + coef['T']) / 100.0
                eff = max(eff, 0.01)
                rho_i = float(rho_list[i-1])
                power_kW = (rho_i * pump_flow * 9.81 * TDH_one * n_val) / (3600.0 * 1000.0 * eff * 0.95)
                rate = ptinfo.get('rate', 9.0)
                sfc_val = ptinfo.get('sfc', 0.0)
                if rate > 0:
                    power_cost = power_kW * 24.0 * rate
                else:
                    fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                    power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD

                result[f"num_pumps_{name}_{t}"] = n_val
                result[f"speed_{name}_{t}"] = rpm_val
                result[f"efficiency_{name}_{t}"] = eff*100   # %
                result[f"power_{name}_{t}"] = power_kW
                result[f"power_cost_{name}_{t}"] = power_cost
                result[f"pump_flow_{name}_{t}"] = pump_flow * n_val

            # Per station sum/avg
            n_sum = sum(int(round(pyo.value(model.n_pumps[i, t]))) for t in station_pump_types[i])
            power_sum = sum(result.get(f"power_{name}_{t}", 0.0) for t in station_pump_types[i])
            power_cost_sum = sum(result.get(f"power_cost_{name}_{t}", 0.0) for t in station_pump_types[i])
            eff_weighted = sum(result.get(f"efficiency_{name}_{t}",0.0)*result.get(f"num_pumps_{name}_{t}",0.0)
                               for t in station_pump_types[i])
            pump_flow_total = sum(result.get(f"pump_flow_{name}_{t}",0.0) for t in station_pump_types[i])
            if n_sum > 0:
                eff_station = eff_weighted / n_sum
            else:
                eff_station = 0.0
            result[f"num_pumps_{name}"] = n_sum
            result[f"pump_flow_{name}"] = pump_flow_total
            result[f"power_{name}"] = power_sum
            result[f"power_cost_{name}"] = power_cost_sum
            result[f"efficiency_{name}"] = eff_station
            result[f"speed_{name}"] = np.mean([result.get(f"speed_{name}_{t}",0.0) for t in station_pump_types[i]])
        else:
            # Not a pump station
            result[f"num_pumps_{name}"] = 0
            result[f"pump_flow_{name}"] = 0.0
            result[f"power_{name}"] = 0.0
            result[f"power_cost_{name}"] = 0.0
            result[f"efficiency_{name}"] = 0.0
            result[f"speed_{name}"] = 0.0

        # DRA: only at station level, not type
        result[f"dra_cost_{name}"] = 0.0
        result[f"dra_ppm_{name}"] = 0.0
        result[f"drag_reduction_{name}"] = 0.0

    # Terminal
    term_name = terminal.get('name','terminal').strip().lower().replace(' ', '_')
    result[f"pipeline_flow_{term_name}"] = segment_flows[-1]
    result[f"pipeline_flow_in_{term_name}"] = segment_flows[-2]
    result[f"pump_flow_{term_name}"] = 0.0
    result[f"speed_{term_name}"] = 0.0
    result[f"num_pumps_{term_name}"] = 0
    result[f"efficiency_{term_name}"] = 0.0
    result[f"power_{term_name}"] = 0.0
    result[f"power_cost_{term_name}"] = 0.0
    result[f"dra_cost_{term_name}"] = 0.0
    result[f"dra_ppm_{term_name}"] = 0.0
    result[f"drag_reduction_{term_name}"] = 0.0
    result[f"velocity_{term_name}"] = 0.0
    result[f"reynolds_{term_name}"] = 0.0
    result[f"friction_{term_name}"] = 0.0
    result[f"residual_head_{term_name}"] = pyo.value(model.RH[N+1])
    result[f"sdh_{term_name}"] = 0.0
    result[f"head_loss_{term_name}"] = 0.0
    result[f"maop_{term_name}"] = 0.0

    result['total_cost'] = float(pyo.value(model.Obj)) if model.Obj is not None else 0.0
    result["error"] = False
    return result
