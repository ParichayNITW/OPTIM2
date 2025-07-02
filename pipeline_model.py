import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

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
    return list(unique_x, unique_y)

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None):
    RPM_STEP = 100
    DRA_STEP = 5

    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)

    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # --- Segment flows as before ---
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # --- Geometry & general pipeline params ---
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    peaks_dict = {}

    for i, stn in enumerate(stations, start=1):
        length[i] = stn.get('L', 0.0)
        if 'D' in stn:
            D_out = stn['D']
            thickness[i] = stn.get('t', default_t)
            d_inner[i] = D_out - 2*thickness[i]
        elif 'd' in stn:
            d_inner[i] = stn['d']
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

    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)

    # --- Pump type extraction ---
    pump_pairs = []
    pump_type_info = {}
    station_types = {}
    for i, stn in enumerate(stations, start=1):
        station_types[i] = []
        if stn.get('is_pump', False):
            for pump in stn.get('pump_types', []):
                t = pump['model_no']
                pump_pairs.append((i, t))
                station_types[i].append(t)
                pump_type_info[(i, t)] = pump

    model.PUMP_PAIRS = pyo.Set(initialize=pump_pairs)
    model.PUMP_STATIONS = pyo.Set(initialize=[i for i in station_types if station_types[i]])

    # --- Variables for pumps ---
    model.NOP = pyo.Var(model.PUMP_PAIRS, domain=pyo.Integers, bounds=(0, 4))
    model.RPM = pyo.Var(model.PUMP_PAIRS, domain=pyo.NonNegativeReals)
    model.DRA = pyo.Var(model.PUMP_PAIRS, domain=pyo.NonNegativeReals)

    # --- RPM/DRA discrete choices ---
    rpm_dict = {}
    dra_dict = {}
    for (i, t) in pump_pairs:
        p = pump_type_info[(i, t)]
        minrpm = int(p.get('MinRPM', 0))
        dol = int(p.get('DOL', 0))
        rpm_dict[(i, t)] = [r for r in range(minrpm, dol+1, RPM_STEP)]
        if rpm_dict[(i, t)][-1] != dol:
            rpm_dict[(i, t)].append(dol)
        maxval_dra = int(p.get('max_dr', 0))
        dra_dict[(i, t)] = [d for d in range(0, maxval_dra+1, DRA_STEP)]
        if dra_dict[(i, t)] and dra_dict[(i, t)][-1] != maxval_dra:
            dra_dict[(i, t)].append(maxval_dra)

    model.rpm_bin = pyo.Var(
        ((i, t, j) for (i, t) in pump_pairs for j in range(len(rpm_dict[(i, t)]))),
        domain=pyo.Binary
    )
    model.dra_bin = pyo.Var(
        ((i, t, j) for (i, t) in pump_pairs for j in range(len(dra_dict[(i, t)]))),
        domain=pyo.Binary
    )
    def rpm_bin_sum_rule(m, i, t):
        return sum(m.rpm_bin[i, t, j] for j in range(len(rpm_dict[(i, t)]))) == 1
    def dra_bin_sum_rule(m, i, t):
        return sum(m.dra_bin[i, t, j] for j in range(len(dra_dict[(i, t)]))) == 1
    model.rpm_bin_sum = pyo.Constraint(model.PUMP_PAIRS, rule=rpm_bin_sum_rule)
    model.dra_bin_sum = pyo.Constraint(model.PUMP_PAIRS, rule=dra_bin_sum_rule)

    def rpm_value_rule(m, i, t):
        return m.RPM[i, t] == sum(rpm_dict[(i, t)][j] * m.rpm_bin[i, t, j] for j in range(len(rpm_dict[(i, t)])))
    def dra_value_rule(m, i, t):
        return m.DRA[i, t] == sum(dra_dict[(i, t)][j] * m.dra_bin[i, t, j] for j in range(len(dra_dict[(i, t)])))
    model.rpm_value = pyo.Constraint(model.PUMP_PAIRS, rule=rpm_value_rule)
    model.dra_value = pyo.Constraint(model.PUMP_PAIRS, rule=dra_value_rule)

    def total_pumps_limit(m, i):
        return sum(m.NOP[i, t] for t in station_types.get(i, [])) <= 4
    model.pump_limit = pyo.Constraint(model.PUMP_STATIONS, rule=total_pumps_limit)

    # --- Pyomo Expression for DR_frac per station ---
    def dr_frac_rule(m, i):
        if station_types.get(i, []):
            return (
                sum(m.DRA[i, t] * m.NOP[i, t] for t in station_types[i])
                /
                (sum(m.NOP[i, t] for t in station_types[i]) + 1e-6)
            )
        else:
            return 0.0
    model.DR_frac = pyo.Expression(model.PUMP_STATIONS, rule=dr_frac_rule)

    # --- Hydraulics ---
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
        kv = kv_dict[i]
        if kv > 0:
            Re[i] = v[i] * d_inner[i] / (float(kv) * 1e-6)
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
    TDH = {}; EFFP = {}
    for i in range(1, N+1):
        head_add = 0
        eff_add = 0
        if i in station_types and station_types[i]:
            for t in station_types[i]:
                pump = pump_type_info[(i, t)]
                DR_frac = model.DRA[i, t] / 100.0
                rpm_val = model.RPM[i, t]
                dol_val = pump['DOL']
                pump_flow_i = float(segment_flows[i])
                Q_equiv = pump_flow_i * dol_val / rpm_val
                A, B, C = pump['A'], pump['B'], pump['C']
                TDH_ = (A * Q_equiv**2 + B * Q_equiv + C) * (rpm_val / dol_val)**2
                head_add += TDH_ * model.NOP[i, t]
                P, Qc, R, S, T = pump['P'], pump['Q'], pump['R'], pump['S'], pump['T']
                EFF_ = (P * Q_equiv**4 + Qc * Q_equiv**3 + R * Q_equiv**2 + S * Q_equiv + T)
                eff_add += EFF_ * model.NOP[i, t]
        TDH[i] = head_add
        EFFP[i] = eff_add
        # Use Pyomo Expression for DR_frac
        dr_frac_expr = model.DR_frac[i] if i in station_types and station_types[i] else 0.0
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2 / (2*g)) * (1 - dr_frac_expr)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g)) * (1 - dr_frac_expr)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N+1):
        if i in station_types and station_types[i]:
            model.head_balance.add(model.RH[i] + TDH[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        D_out = d_inner[i] + 2 * thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            loss_no_dra = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g))
            if i in station_types and station_types[i]:
                model.peak_limit.add(model.RH[i] + TDH[i] - (elev_k - model.z[i]) - loss_no_dra >= 50.0)
            else:
                model.peak_limit.add(model.RH[i] - (elev_k - model.z[i]) - loss_no_dra >= 50.0)

    model.PPM = pyo.Var(model.PUMP_PAIRS, domain=pyo.NonNegativeReals)
    model.dra_cost = pyo.Expression(model.PUMP_PAIRS)
    for (i, t) in pump_pairs:
        visc = kv_dict[i]
        dr_points, ppm_points = get_ppm_breakpoints(visc)
        if dr_points and ppm_points:
            dr_points_fixed, ppm_points_fixed = zip(*sorted(set(zip(dr_points, ppm_points))))
            setattr(model, f'piecewise_dra_ppm_{i}_{t}',
                pyo.Piecewise(f'pw_dra_ppm_{i}_{t}', model.PPM[i, t], model.DRA[i, t],
                              pw_pts=dr_points_fixed,
                              f_rule=ppm_points_fixed,
                              pw_constr_type='EQ'))
        dra_cost_expr = model.PPM[i, t] * (segment_flows[i] * 1000.0 * 24.0 / 1e6) * RateDRA
        model.dra_cost[i, t] = dra_cost_expr

    total_cost = 0
    for (i, t) in pump_pairs:
        pump = pump_type_info[(i, t)]
        rho_i = rho_dict[i]
        pump_flow_i = float(segment_flows[i])
        rpm_val = model.RPM[i, t]
        dol_val = pump['DOL']
        Q_equiv = pump_flow_i * dol_val / rpm_val
        A, B, C = pump['A'], pump['B'], pump['C']
        TDH_ = (A * Q_equiv**2 + B * Q_equiv + C) * (rpm_val / dol_val)**2
        P, Qc, R, S, T = pump['P'], pump['Q'], pump['R'], pump['S'], pump['T']
        EFF_ = (P * Q_equiv**4 + Qc * Q_equiv**3 + R * Q_equiv**2 + S * Q_equiv + T) / 100.0
        NOP = model.NOP[i, t]
        if pump.get('power_type', 'Grid') == 'Grid':
            rate = pump.get('rate', 0.0)
            power_kW = (rho_i * pump_flow_i * 9.81 * TDH_ * NOP) / (3600.0 * 1000.0 * EFF_ * 0.95)
            power_cost = power_kW * 24.0 * rate
        else:
            sfc_val = pump.get('sfc', 0.0)
            fuel_per_kWh = (sfc_val * 1.34102) / 820.0 if sfc_val else 0.0
            power_kW = (rho_i * pump_flow_i * 9.81 * TDH_ * NOP) / (3600.0 * 1000.0 * EFF_ * 0.95)
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        dra_cost = model.dra_cost[i, t]
        total_cost += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

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

    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0

        if i in station_types and station_types[i]:
            for t in station_types[i]:
                pump = pump_type_info[(i, t)]
                model_no = str(t)
                NOP_val = int(round(pyo.value(model.NOP[i, t]))) if model.NOP[i, t].value is not None else 0
                rpm_val = None
                for j in range(len(rpm_dict[(i, t)])):
                    if round(pyo.value(model.rpm_bin[i, t, j])) == 1:
                        rpm_val = rpm_dict[(i, t)][j]
                        break
                dra_perc = None
                for j in range(len(dra_dict[(i, t)])):
                    if round(pyo.value(model.dra_bin[i, t, j])) == 1:
                        dra_perc = dra_dict[(i, t)][j]
                        break
                if rpm_val is None:
                    rpm_val = rpm_dict[(i, t)][0]
                if dra_perc is None:
                    dra_perc = dra_dict[(i, t)][0] if dra_dict[(i, t)] else 0
                dol_val = pump['DOL']
                pump_flow_i = float(segment_flows[i])
                Q_equiv = pump_flow_i * dol_val / rpm_val if rpm_val > 0 else 0
                A, B, C = pump['A'], pump['B'], pump['C']
                tdh_val = (A * Q_equiv**2 + B * Q_equiv + C) * (rpm_val/dol_val)**2 if rpm_val > 0 else 0.0
                P, Qc, R, S, T = pump['P'], pump['Q'], pump['R'], pump['S'], pump['T']
                eff = (P*Q_equiv**4 + Qc*Q_equiv**3 + R*Q_equiv**2 + S*Q_equiv + T) if NOP_val > 0 else 0.0
                eff = float(eff)
                dra_ppm = float(pyo.value(model.PPM[i, t])) if model.PPM[i, t].value is not None else 0.0
                dra_cost_i = float(pyo.value(model.dra_cost[i, t])) if model.dra_cost[i, t].expr is not None else 0.0
                if NOP_val == 0:
                    rpm_val = 0.0
                    eff = 0.0
                    dra_perc = 0.0
                    dra_ppm = 0.0
                    dra_cost_i = 0.0
                    tdh_val = 0.0
                rho_i = rho_dict[i]
                power_kW = (rho_i * pump_flow * 9.81 * tdh_val * NOP_val) / (3600.0 * 1000.0 * (eff/100.0) * 0.95) if eff > 0 else 0.0
                if pump.get('power_type', 'Grid') == 'Grid':
                    rate = pump.get('rate', 0.0)
                    power_cost = power_kW * 24.0 * rate
                else:
                    sfc_val = pump.get('sfc', 0.0)
                    fuel_per_kWh = (sfc_val * 1.34102) / 820.0 if sfc_val else 0.0
                    power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
                drag_red = dra_perc
                result[f"{name}_model_{model_no}_num_pumps"] = NOP_val
                result[f"{name}_model_{model_no}_speed"] = rpm_val
                result[f"{name}_model_{model_no}_efficiency"] = eff
                result[f"{name}_model_{model_no}_power_cost"] = power_cost
                result[f"{name}_model_{model_no}_dra_cost"] = dra_cost_i
                result[f"{name}_model_{model_no}_dra_ppm"] = dra_ppm
                result[f"{name}_model_{model_no}_drag_reduction"] = drag_red
                result[f"{name}_model_{model_no}_tdh"] = tdh_val
        total_pumps = sum(int(round(pyo.value(model.NOP[i, t]))) if model.NOP[i, t].value is not None else 0 for t in station_types.get(i, []))
        result[f"num_pumps_{name}"] = total_pumps

    term_name = terminal.get('name','terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": segment_flows[-1],
        f"pipeline_flow_in_{term_name}": segment_flows[-2],
        f"pump_flow_{term_name}": 0.0,
        f"speed_{term_name}": 0.0,
        f"num_pumps_{term_name}": 0,
        f"efficiency_{term_name}": 0.0,
        f"power_cost_{term_name}": 0.0,
        f"dra_cost_{term_name}": 0.0,
        f"dra_ppm_{term_name}": 0.0,
        f"drag_reduction_{term_name}": 0.0,
        f"head_loss_{term_name}": 0.0,
        f"velocity_{term_name}": 0.0,
        f"reynolds_{term_name}": 0.0,
        f"friction_{term_name}": 0.0,
        f"sdh_{term_name}": 0.0,
        f"residual_head_{term_name}": float(pyo.value(model.RH[N+1])) if model.RH[N+1].value is not None else 0.0,
    })
    result['total_cost'] = float(pyo.value(model.Obj)) if model.Obj is not None else 0.0
    result["error"] = False
    return result
