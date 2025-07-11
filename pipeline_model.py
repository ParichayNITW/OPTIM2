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
    return list(unique_x), list(unique_y)

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None, looplines=None):
    RPM_STEP = 100
    DRA_STEP = 5
    g = 9.81
    if looplines is None:
        looplines = []

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

    # -- Station pipeline parameters
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    max_dr = {}
    peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    allowed_rpms = {}
    allowed_dras = {}

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
        has_pump = stn.get('is_pump', False)
        if has_pump:
            pump_indices.append(i)
            Acoef[i] = stn.get('A', 0.0)
            Bcoef[i] = stn.get('B', 0.0)
            Ccoef[i] = stn.get('C', 0.0)
            Pcoef[i] = stn.get('P', 0.0)
            Qcoef[i] = stn.get('Q', 0.0)
            Rcoef[i] = stn.get('R', 0.0)
            Scoef[i] = stn.get('S', 0.0)
            Tcoef[i] = stn.get('T', 0.0)
            min_rpm[i] = stn.get('MinRPM', 0)
            max_rpm[i] = stn.get('DOL', 0)
            if stn.get('sfc', 0) not in (None, 0):
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc', 0.0)
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 0.0)
            max_dr[i] = stn.get('max_dr', 0.0)
            minval = int(min_rpm[i])
            maxval = int(max_rpm[i])
            allowed_rpms[i] = [r for r in range(minval, maxval+1, RPM_STEP)]
            if allowed_rpms[i][-1] != maxval:
                allowed_rpms[i].append(maxval)
            maxval_dra = int(max_dr[i])
            allowed_dras[i] = [d for d in range(0, maxval_dra+1, DRA_STEP)]
            if allowed_dras[i][-1] != maxval_dra:
                allowed_dras[i].append(maxval_dra)

    elev[N+1] = terminal.get('elev', 0.0)
    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)

    model.pump_stations = pyo.Set(initialize=pump_indices)
    if pump_indices:
        model.A = pyo.Param(model.pump_stations, initialize=Acoef)
        model.B = pyo.Param(model.pump_stations, initialize=Bcoef)
        model.C = pyo.Param(model.pump_stations, initialize=Ccoef)
        model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef)
        model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef)
        model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef)
        model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef)
        model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef)
        model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm)
        model.DOL = pyo.Param(model.pump_stations, initialize=max_rpm)

    originating_pump_index = None
    for idx, stn in enumerate(stations, start=1):
        if stn.get('is_pump', False):
            originating_pump_index = idx
            break
    if originating_pump_index is None:
        raise ValueError("No originating pump station found in input!")
    def nop_bounds(m, j):
        lb = 1 if j == originating_pump_index else 0
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=nop_bounds, initialize=1)
    def min_pump_origin_rule(m):
        return m.NOP[originating_pump_index] >= 1
    model.min_pump_origin = pyo.Constraint(rule=min_pump_origin_rule)
    model.rpm_bin = pyo.Var(((i, j) for i in pump_indices for j in range(len(allowed_rpms[i]))), domain=pyo.Binary)
    def rpm_bin_sum_rule(m, i):
        return sum(m.rpm_bin[i, j] for j in range(len(allowed_rpms[i]))) == 1
    model.rpm_bin_sum = pyo.Constraint(model.pump_stations, rule=rpm_bin_sum_rule)
    model.RPM_var = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    def rpm_value_rule(m, i):
        return m.RPM_var[i] == sum(allowed_rpms[i][j] * m.rpm_bin[i, j] for j in range(len(allowed_rpms[i])))
    model.rpm_value = pyo.Constraint(model.pump_stations, rule=rpm_value_rule)
    model.dra_bin = pyo.Var(((i, j) for i in pump_indices for j in range(len(allowed_dras[i]))), domain=pyo.Binary)
    def dra_bin_sum_rule(m, i):
        return sum(m.dra_bin[i, j] for j in range(len(allowed_dras[i]))) == 1
    model.dra_bin_sum = pyo.Constraint(model.pump_stations, rule=dra_bin_sum_rule)
    def dra_var_bounds(m, i):
        return (min(allowed_dras[i]), max(allowed_dras[i]))
    model.DR_var = pyo.Var(model.pump_stations, bounds=dra_var_bounds, domain=pyo.NonNegativeReals)
    def dra_value_rule(m, i):
        return m.DR_var[i] == sum(allowed_dras[i][j] * m.dra_bin[i, j] for j in range(len(allowed_dras[i])))
    model.dra_value = pyo.Constraint(model.pump_stations, rule=dra_value_rule)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    # -- Loopline logic: prepare per-segment flow variables
    # Base: segment_flows[i] = variable for mainline flow in i-th segment (from i to i+1)
    segment_flows = [None] * (N+1)
    model.segment_flow = pyo.Var(range(N+1), domain=pyo.NonNegativeReals)
    model.segment_flow[0].fix(FLOW)

    # -- Build loopline variables (for each loopline, assign a variable for flow in loopline)
    loopline_vars = []
    for idx, lp in enumerate(looplines):
        model.add_component(f'loopline_flow_{idx}', pyo.Var(domain=pyo.NonNegativeReals))
        loopline_vars.append(getattr(model, f'loopline_flow_{idx}'))

    # -- Build connectivity maps for each chainage to know where splits/joins happen
    split_map = {}
    join_map = {}
    for idx, lp in enumerate(looplines):
        start = lp['start_idx']
        end = lp['end_idx']
        split_map.setdefault(start, []).append(idx)
        join_map.setdefault(end, []).append(idx)

    # -- Add flow continuity constraints at split and join nodes
    def flow_continuity_rule(m, i):
        flow_in = m.segment_flow[i-1]
        flow_out = m.segment_flow[i]
        inflow = flow_in
        outflow = flow_out
        # At split, outgoing mainline = mainline + all loopline flows starting here
        if i-1 in split_map:
            for lpidx in split_map[i-1]:
                inflow -= loopline_vars[lpidx]
        # At join, incoming mainline = mainline + all loopline flows joining here
        if i in join_map:
            for lpidx in join_map[i]:
                inflow += loopline_vars[lpidx]
        return flow_out == inflow
    model.flow_continuity = pyo.Constraint(model.I, rule=flow_continuity_rule)

    # -- For each loopline, add pressure drop constraint (equal drop in mainline and loopline between split/join)
    for idx, lp in enumerate(looplines):
        s_idx = lp['start_idx'] + 1
        e_idx = lp['end_idx'] + 1
        L_lp = lp['L']
        D_lp = lp['D']
        t_lp = lp['t']
        rough_lp = lp['rough']
        kv_lp = lp.get('kv', KV_list[lp['start_idx']])
        dr_frac = model.DR_var[originating_pump_index] / 100.0 if pump_indices else 0.0

        def pressure_drop_lp_rule(m):
            # mainline: from s_idx to e_idx
            main_flow = m.segment_flow[lp['start_idx']]
            d_ml = d_inner[lp['start_idx']+1]
            f_ml = 0.0
            v_ml = main_flow / 3600.0 / (pi * d_ml**2 / 4.0)
            kv_ml = kv_dict[lp['start_idx']+1]
            if kv_ml > 0:
                Re_ml = v_ml * d_ml / (kv_ml * 1e-6)
                if Re_ml < 4000:
                    f_ml = 64.0 / Re_ml
                else:
                    arg = (roughness[lp['start_idx']+1] / d_ml / 3.7) + (5.74 / (Re_ml**0.9))
                    f_ml = 0.25 / (log10(arg)**2) if arg > 0 else 0.0
            dh_ml = f_ml * ((length[lp['start_idx']+1]*1000.0)/d_ml) * (v_ml**2 / (2*g)) * (1-dr_frac)
            # loopline:
            flow_lp = getattr(m, f'loopline_flow_{idx}')
            d_lp_in = D_lp - 2*t_lp
            v_lp = flow_lp / 3600.0 / (pi * d_lp_in**2 / 4.0)
            f_lp = 0.0
            if kv_lp > 0:
                Re_lp = v_lp * d_lp_in / (kv_lp * 1e-6)
                if Re_lp < 4000:
                    f_lp = 64.0 / Re_lp
                else:
                    arg = (rough_lp / d_lp_in / 3.7) + (5.74 / (Re_lp**0.9))
                    f_lp = 0.25 / (log10(arg)**2) if arg > 0 else 0.0
            dh_lp = f_lp * ((L_lp*1000.0)/d_lp_in) * (v_lp**2 / (2*g)) * (1-dr_frac)
            return dh_ml == dh_lp
        setattr(model, f'pressure_drop_loopline_{idx}', pyo.Constraint(rule=pressure_drop_lp_rule))

    # -- Now use segment_flows (mainline) for all subsequent logic
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        flow_m3s = model.segment_flow[i-1]
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area
        kv = kv_dict[i]
        Re[i] = v[i] * d_inner[i] / (float(kv) * 1e-6) if kv > 0 else 0.0
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
    TDH = {}
    EFFP = {}

    for i in range(1, N+1):
        if i in pump_indices:
            DR_frac = model.DR_var[i] / 100.0
        else:
            DR_frac = 0.0
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)
        if i in pump_indices:
            pump_flow_i = model.segment_flow[i]
            rpm_val = model.RPM_var[i]
            dol_val = model.DOL[i]
            Q_equiv = pump_flow_i * dol_val / rpm_val
            H_DOL = model.A[i] * Q_equiv**2 + model.B[i] * Q_equiv + model.C[i]
            TDH[i] = H_DOL * (rpm_val / dol_val)**2
            EFFP[i] = (model.Pcoef[i]*Q_equiv**4 + model.Qcoef[i]*Q_equiv**3 +
                       model.Rcoef[i]*Q_equiv**2 + model.Scoef[i]*Q_equiv +
                       model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N+1):
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
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
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    model.dra_cost = pyo.Expression(model.pump_stations)
    for i in pump_indices:
        visc = kv_dict[i]
        dr_points, ppm_points = get_ppm_breakpoints(visc)
        dr_points_fixed, ppm_points_fixed = zip(*sorted(set(zip(dr_points, ppm_points))))
        setattr(model, f'piecewise_dra_ppm_{i}',
            pyo.Piecewise(f'pw_dra_ppm_{i}', model.PPM[i], model.DR_var[i],
                          pw_pts=dr_points_fixed,
                          f_rule=ppm_points_fixed,
                          pw_constr_type='EQ'))
        dra_cost_expr = model.PPM[i] * (model.segment_flow[i] * 1000.0 * 24.0 / 1e6) * RateDRA
        model.dra_cost[i] = dra_cost_expr

    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        pump_flow_i = model.segment_flow[i]
        rpm_val = model.RPM_var[i]
        eff_val = EFFP[i]
        power_kW = (rho_i * pump_flow_i * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * eff_val * 0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i, 0.0)
        else:
            fuel_per_kWh = (sfc.get(i,0.0) * 1.34102) / 820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        dra_cost_i = model.dra_cost[i]
        total_cost += power_cost + dra_cost_i
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
        inflow = pyo.value(model.segment_flow[i-1])
        outflow = pyo.value(model.segment_flow[i])
        pump_flow = outflow if stn.get('is_pump', False) else 0.0
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i])) if model.NOP[i].value is not None else 0
            rpm_val = None
            for j in range(len(allowed_rpms[i])):
                if round(pyo.value(model.rpm_bin[i, j])) == 1:
                    rpm_val = allowed_rpms[i][j]
                    break
            dra_perc = None
            for j in range(len(allowed_dras[i])):
                if round(pyo.value(model.dra_bin[i, j])) == 1:
                    dra_perc = allowed_dras[i][j]
                    break
            if rpm_val is None:
                rpm_val = allowed_rpms[i][0]
            if dra_perc is None:
                dra_perc = allowed_dras[i][0]
            dol_val = model.DOL[i]
            pump_flow_i = pyo.value(model.segment_flow[i])
            Q_equiv = pump_flow_i * dol_val / rpm_val
            tdh_val = float(model.A[i] * Q_equiv**2 + model.B[i] * Q_equiv + model.C[i]) * (rpm_val/dol_val)**2
            eff = (model.Pcoef[i]*Q_equiv**4 + model.Qcoef[i]*Q_equiv**3 +
                   model.Rcoef[i]*Q_equiv**2 + model.Scoef[i]*Q_equiv +
                   model.Tcoef[i]) if num_pumps > 0 else 0.0
            eff = float(eff)
            dra_ppm = float(pyo.value(model.PPM[i])) if model.PPM[i].value is not None else 0.0
            dra_cost_i = float(pyo.value(model.dra_cost[i])) if model.dra_cost[i].expr is not None else 0.0
            if num_pumps == 0:
                rpm_val = 0.0
                eff = 0.0
                dra_perc = 0.0
                dra_ppm = 0.0
                dra_cost_i = 0.0
                tdh_val = 0.0
        else:
            num_pumps = 0
            rpm_val = 0.0
            eff = 0.0
            dra_perc = 0.0
            dra_ppm = 0.0
            dra_cost_i = 0.0
            tdh_val = 0.0
        if i in pump_indices and num_pumps > 0:
            rho_i = rho_dict[i]
            power_kW = (rho_i * pump_flow * 9.81 * tdh_val * num_pumps) / (3600.0 * 1000.0 * (eff/100.0) * 0.95) if eff > 0 else 0.0
            if i in electric_pumps:
                rate = elec_cost.get(i, 0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i, 0.0)
                fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        else:
            power_cost = 0.0
        drag_red = dra_perc
        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i+1].value is not None else 0.0
        res_head = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
        velocity = pyo.value(v[i]); reynolds = pyo.value(Re[i]); fric = pyo.value(f[i])
        result[f"pipeline_flow_{name}"] = outflow
        result[f"pipeline_flow_in_{name}"] = inflow
        result[f"pump_flow_{name}"] = pump_flow
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = rpm_val
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost
        result[f"dra_cost_{name}"] = dra_cost_i
        result[f"dra_ppm_{name}"] = dra_ppm
        result[f"drag_reduction_{name}"] = drag_red
        result[f"head_loss_{name}"] = head_loss
        result[f"residual_head_{name}"] = res_head
        result[f"velocity_{name}"] = velocity
        result[f"reynolds_{name}"] = reynolds
        result[f"friction_{name}"] = fric
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
        result[f"maop_{name}"] = maop_dict[i]
        if i in pump_indices:
            result[f"coef_A_{name}"] = float(model.A[i])
            result[f"coef_B_{name}"] = float(model.B[i])
            result[f"coef_C_{name}"] = float(model.C[i])
            result[f"dol_{name}"]    = float(model.DOL[i])
            result[f"min_rpm_{name}"]= float(model.MinRPM[i])
            result[f"tdh_{name}"]    = tdh_val
    term_name = terminal.get('name','terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": pyo.value(model.segment_flow[-1]),
        f"pipeline_flow_in_{term_name}": pyo.value(model.segment_flow[-2]),
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
    # Loopline results
    for idx, lp in enumerate(looplines):
        flowval = pyo.value(loopline_vars[idx])
        result[f"loopline_{idx}_flow"] = flowval
    return result