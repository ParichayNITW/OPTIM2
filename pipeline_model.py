import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'parichay.nitwarangal@gmail.com')

# DRA curve files (unchanged)
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
        for c in cst_list:
            if abs(visc - c) < 1e-6:
                df = DRA_CURVE_DATA[c]
                x = df['%Drag Reduction'].values
                y = df['PPM'].values
                unique_x, unique_indices = np.unique(x, return_index=True)
                unique_y = y[unique_indices]
                return list(unique_x), list(unique_y)
        lower = max([c for c in cst_list if c <= visc])
        upper = min([c for c in cst_list if c >= visc])
        if abs(upper - lower) < 1e-6:
            df = DRA_CURVE_DATA[lower]
            x = df['%Drag Reduction'].values
            y = df['PPM'].values
            unique_x, unique_indices = np.unique(x, return_index=True)
            unique_y = y[unique_indices]
            return list(unique_x), unique_y
        df_lower = DRA_CURVE_DATA[lower]
        df_upper = DRA_CURVE_DATA[upper]
        x_lower, y_lower = df_lower['%Drag Reduction'].values, df_lower['PPM'].values
        x_upper, y_upper = df_upper['%Drag Reduction'].values, df_upper['PPM'].values
        dr_points = np.unique(np.concatenate((x_lower, x_upper)))
        ppm_lower_interp = np.interp(dr_points, x_lower, y_lower)
        ppm_upper_interp = np.interp(dr_points, x_upper, y_upper)
        ppm_points = ppm_lower_interp * (upper-visc)/(upper-lower) + ppm_upper_interp * (visc-lower)/(upper-lower)
        unique_dr, unique_indices = np.unique(dr_points, return_index=True)
        unique_ppm = ppm_points[unique_indices]
        return list(unique_dr), list(unique_ppm)
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    return list(unique_x), unique_y

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None):
    RPM_STEP = 100  # RPM step
    DRA_STEP = 5    # DRA step

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

    # Compute segment flows
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # Pipeline and pump parameters
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

    # ----- Setup for ORIGIN STATION (first station) -----
    # Required input keys for stations[0]:
    #   For Type A: 'A1', 'B1', 'C1', 'P1', 'Q1', 'R1', 'S1', 'T1', 'MinRPM1', 'DOL1', 'sfc1' or 'rate1'
    #   For Type B: 'A2', 'B2', 'C2', 'P2', 'Q2', 'R2', 'S2', 'T2', 'MinRPM2', 'DOL2', 'sfc2' or 'rate2'
    #   Optional: 'max_pumps_typeA', 'max_pumps_typeB' (default 2)
    ORIGIN_TYPEA = {'A': stations[0]['A1'], 'B': stations[0]['B1'], 'C': stations[0]['C1'],
                    'P': stations[0]['P1'], 'Q': stations[0]['Q1'], 'R': stations[0]['R1'],
                    'S': stations[0]['S1'], 'T': stations[0]['T1'], 'MinRPM': stations[0]['MinRPM1'],
                    'DOL': stations[0]['DOL1'],
                    'sfc': stations[0].get('sfc1'), 'rate': stations[0].get('rate1')}
    ORIGIN_TYPEB = {'A': stations[0]['A2'], 'B': stations[0]['B2'], 'C': stations[0]['C2'],
                    'P': stations[0]['P2'], 'Q': stations[0]['Q2'], 'R': stations[0]['R2'],
                    'S': stations[0]['S2'], 'T': stations[0]['T2'], 'MinRPM': stations[0]['MinRPM2'],
                    'DOL': stations[0]['DOL2'],
                    'sfc': stations[0].get('sfc2'), 'rate': stations[0].get('rate2')}
    max_pumps_typeA = stations[0].get('max_pumps_typeA', 2)
    max_pumps_typeB = stations[0].get('max_pumps_typeB', 2)

    # Setup for rest of the stations (unchanged)
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
        if has_pump and i != 1:
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

    # -- Special variables for Origin station, two types --
    # Decision: number of each type, bounded 0 to 2; total at least 1.
    model.NOP_A_origin = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, max_pumps_typeA), initialize=1)
    model.NOP_B_origin = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, max_pumps_typeB), initialize=0)
    model.min_pump_origin = pyo.Constraint(expr= model.NOP_A_origin + model.NOP_B_origin >= 1)

    # Allowed RPMs (discrete) for each type at origin
    allowed_rpms_A = list(range(int(ORIGIN_TYPEA['MinRPM']), int(ORIGIN_TYPEA['DOL'])+1, RPM_STEP))
    if allowed_rpms_A[-1] != int(ORIGIN_TYPEA['DOL']):
        allowed_rpms_A.append(int(ORIGIN_TYPEA['DOL']))
    allowed_rpms_B = list(range(int(ORIGIN_TYPEB['MinRPM']), int(ORIGIN_TYPEB['DOL'])+1, RPM_STEP))
    if allowed_rpms_B[-1] != int(ORIGIN_TYPEB['DOL']):
        allowed_rpms_B.append(int(ORIGIN_TYPEB['DOL']))

    # RPM selection binaries and continuous vars for both types at origin
    model.rpm_bin_A = pyo.Var(range(len(allowed_rpms_A)), domain=pyo.Binary)
    model.rpm_bin_B = pyo.Var(range(len(allowed_rpms_B)), domain=pyo.Binary)
    model.RPM_A_origin = pyo.Var(bounds=(ORIGIN_TYPEA['MinRPM'], ORIGIN_TYPEA['DOL']), domain=pyo.NonNegativeReals)
    model.RPM_B_origin = pyo.Var(bounds=(ORIGIN_TYPEB['MinRPM'], ORIGIN_TYPEB['DOL']), domain=pyo.NonNegativeReals)
    model.rpm_bin_sum_A = pyo.Constraint(expr=sum(model.rpm_bin_A[j] for j in range(len(allowed_rpms_A))) == 1)
    model.rpm_bin_sum_B = pyo.Constraint(expr=sum(model.rpm_bin_B[j] for j in range(len(allowed_rpms_B))) == 1)
    model.rpm_value_A = pyo.Constraint(expr= model.RPM_A_origin == sum(allowed_rpms_A[j] * model.rpm_bin_A[j] for j in range(len(allowed_rpms_A))))
    model.rpm_value_B = pyo.Constraint(expr= model.RPM_B_origin == sum(allowed_rpms_B[j] * model.rpm_bin_B[j] for j in range(len(allowed_rpms_B))))

    # DRA, RH, etc. for origin and all stations (unchanged)
    # ... (rest of your variable/constraint setup for DRA, residual head, etc. remains unchanged from your original code) ...

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        if j == N+1:
            model.RH[j].setlb(terminal.get('min_residual', 50.0))
        else:
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
    TDH = {}
    EFFP = {}

    # ----------- Head & Efficiency for Origin (two types) and rest stations -----------
    for i in range(1, N+1):
        if i == 1:
            pump_flow = float(segment_flows[i])
            # Type A
            rpm_A = model.RPM_A_origin
            dol_A = ORIGIN_TYPEA['DOL']
            Q_equiv_A = pump_flow * dol_A / rpm_A
            H_DOL_A = ORIGIN_TYPEA['A'] * Q_equiv_A**2 + ORIGIN_TYPEA['B'] * Q_equiv_A + ORIGIN_TYPEA['C']
            TDH_A = H_DOL_A * (rpm_A / dol_A)**2
            EFF_A = (ORIGIN_TYPEA['P']*Q_equiv_A**4 + ORIGIN_TYPEA['Q']*Q_equiv_A**3 +
                     ORIGIN_TYPEA['R']*Q_equiv_A**2 + ORIGIN_TYPEA['S']*Q_equiv_A + ORIGIN_TYPEA['T']) / 100.0
            model.TDH_A_origin = pyo.Expression(expr=TDH_A)
            model.EFF_A_origin = pyo.Expression(expr=EFF_A)
            # Type B
            rpm_B = model.RPM_B_origin
            dol_B = ORIGIN_TYPEB['DOL']
            Q_equiv_B = pump_flow * dol_B / rpm_B
            H_DOL_B = ORIGIN_TYPEB['A'] * Q_equiv_B**2 + ORIGIN_TYPEB['B'] * Q_equiv_B + ORIGIN_TYPEB['C']
            TDH_B = H_DOL_B * (rpm_B / dol_B)**2
            EFF_B = (ORIGIN_TYPEB['P']*Q_equiv_B**4 + ORIGIN_TYPEB['Q']*Q_equiv_B**3 +
                     ORIGIN_TYPEB['R']*Q_equiv_B**2 + ORIGIN_TYPEB['S']*Q_equiv_B + ORIGIN_TYPEB['T']) / 100.0
            model.TDH_B_origin = pyo.Expression(expr=TDH_B)
            model.EFF_B_origin = pyo.Expression(expr=EFF_B)
        elif i in pump_indices:
            pump_flow_i = float(segment_flows[i])
            rpm_val = model.RPM_var[i]
            dol_val = model.DOL[i]
            Q_equiv = pump_flow_i * dol_val / rpm_val
            H_DOL = model.A[i]*Q_equiv**2 + model.B[i]*Q_equiv + model.C[i]
            TDH[i] = H_DOL * (rpm_val/dol_val)**2
            EFFP[i] = (model.Pcoef[i]*Q_equiv**4 + model.Qcoef[i]*Q_equiv**3 +
                       model.Rcoef[i]*Q_equiv**2 + model.Scoef[i]*Q_equiv + model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # ----------- Constraints for SDH, peaks, head balance -----------
    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N+1):
        if i == 1:
            # Head balance at origin: sum both types
            model.head_balance.add(model.RH[i] +
                model.TDH_A_origin * model.NOP_A_origin +
                model.TDH_B_origin * model.NOP_B_origin >= model.SDH[i])
        elif i in pump_indices:
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
            if i == 1:
                expr = model.RH[i] + model.TDH_A_origin * model.NOP_A_origin + model.TDH_B_origin * model.NOP_B_origin - (elev_k - model.z[i]) - loss_no_dra
            elif i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    # ---------- RPM & DRA binaries for rest of the pumps (unchanged) ----------
    model.rpm_bin = pyo.Var(
        ((i, j) for i in pump_indices for j in range(len(allowed_rpms[i]))),
        domain=pyo.Binary
    )
    def rpm_bin_sum_rule(m, i):
        return sum(m.rpm_bin[i, j] for j in range(len(allowed_rpms[i]))) == 1
    model.rpm_bin_sum = pyo.Constraint(model.pump_stations, rule=rpm_bin_sum_rule)
    model.RPM_var = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    def rpm_value_rule(m, i):
        return m.RPM_var[i] == sum(allowed_rpms[i][j] * m.rpm_bin[i, j] for j in range(len(allowed_rpms[i])))
    model.rpm_value = pyo.Constraint(model.pump_stations, rule=rpm_value_rule)

    model.dra_bin = pyo.Var(
        ((i, j) for i in pump_indices for j in range(len(allowed_dras[i]))),
        domain=pyo.Binary
    )
    def dra_bin_sum_rule(m, i):
        return sum(m.dra_bin[i, j] for j in range(len(allowed_dras[i]))) == 1
    model.dra_bin_sum = pyo.Constraint(model.pump_stations, rule=dra_bin_sum_rule)
    def dra_var_bounds(m, i):
        return (min(allowed_dras[i]), max(allowed_dras[i]))
    model.DR_var = pyo.Var(model.pump_stations, bounds=dra_var_bounds, domain=pyo.NonNegativeReals)
    def dra_value_rule(m, i):
        return m.DR_var[i] == sum(allowed_dras[i][j] * m.dra_bin[i, j] for j in range(len(allowed_dras[i])))
    model.dra_value = pyo.Constraint(model.pump_stations, rule=dra_value_rule)

    # ----------- DRA PPM and Cost for all pump stations, including origin -----------
    model.PPM = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.dra_cost = pyo.Expression(model.I)
    for i in range(1, N+1):
        visc = kv_dict[i]
        dr_points, ppm_points = get_ppm_breakpoints(visc)
        dr_points_fixed, ppm_points_fixed = zip(*sorted(set(zip(dr_points, ppm_points))))
        if i in pump_indices:
            setattr(model, f'piecewise_dra_ppm_{i}',
                pyo.Piecewise(f'pw_dra_ppm_{i}', model.PPM[i], model.DR_var[i],
                              pw_pts=dr_points_fixed,
                              f_rule=ppm_points_fixed,
                              pw_constr_type='EQ'))
        else:
            model.PPM[i].fix(0)

        dra_cost_expr = model.PPM[i] * (segment_flows[i] * 1000.0 * 24.0 / 1e6) * RateDRA
        model.dra_cost[i] = dra_cost_expr

    # ----------- Objective: Power and DRA cost for origin (A+B) and others -----------
    total_cost = 0
    for i in range(1, N+1):
        rho_i = rho_dict[i]
        pump_flow = float(segment_flows[i])
        if i == 1:
            # Power (kW) for Type A and B at origin
            power_kW_A = (rho_i * pump_flow * 9.81 * model.TDH_A_origin * model.NOP_A_origin) / (3600.0 * 1000.0 * model.EFF_A_origin * 0.95) if ORIGIN_TYPEA['A'] != 0 else 0
            power_kW_B = (rho_i * pump_flow * 9.81 * model.TDH_B_origin * model.NOP_B_origin) / (3600.0 * 1000.0 * model.EFF_B_origin * 0.95) if ORIGIN_TYPEB['A'] != 0 else 0
            # Cost for each type: diesel or electric
            cost_A = 0
            if ORIGIN_TYPEA['sfc'] not in (None, 0):
                fuel_per_kWh_A = (ORIGIN_TYPEA['sfc'] * 1.34102) / 820.0
                cost_A = power_kW_A * 24.0 * fuel_per_kWh_A * Price_HSD
            elif ORIGIN_TYPEA['rate'] not in (None, 0):
                cost_A = power_kW_A * 24.0 * ORIGIN_TYPEA['rate']
            cost_B = 0
            if ORIGIN_TYPEB['sfc'] not in (None, 0):
                fuel_per_kWh_B = (ORIGIN_TYPEB['sfc'] * 1.34102) / 820.0
                cost_B = power_kW_B * 24.0 * fuel_per_kWh_B * Price_HSD
            elif ORIGIN_TYPEB['rate'] not in (None, 0):
                cost_B = power_kW_B * 24.0 * ORIGIN_TYPEB['rate']
            dra_cost_origin = model.dra_cost[i]
            total_cost += cost_A + cost_B + dra_cost_origin
        elif i in pump_indices:
            rpm_val = model.RPM_var[i]
            eff_val = EFFP[i]
            power_kW = (rho_i * pump_flow * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * eff_val * 0.95)
            if i in electric_pumps:
                total_cost += power_kW * 24.0 * elec_cost.get(i, 0.0) + model.dra_cost[i]
            else:
                fuel_per_kWh = (sfc.get(i,0.0) * 1.34102) / 820.0
                total_cost += power_kW * 24.0 * fuel_per_kWh * Price_HSD + model.dra_cost[i]
        else:
            total_cost += model.dra_cost[i]

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # ----------- Solve -----------
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
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

    # ----------- Output Extraction -----------
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0

        if i == 1:
            numA = int(round(pyo.value(model.NOP_A_origin)))
            numB = int(round(pyo.value(model.NOP_B_origin)))
            rpmA = 0
            for j in range(len(allowed_rpms_A)):
                if round(pyo.value(model.rpm_bin_A[j])) == 1:
                    rpmA = allowed_rpms_A[j]
                    break
            rpmB = 0
            for j in range(len(allowed_rpms_B)):
                if round(pyo.value(model.rpm_bin_B[j])) == 1:
                    rpmB = allowed_rpms_B[j]
                    break
            effA = pyo.value(model.EFF_A_origin) * 100 if numA > 0 else 0.0
            effB = pyo.value(model.EFF_B_origin) * 100 if numB > 0 else 0.0
            tdhA = pyo.value(model.TDH_A_origin) if numA > 0 else 0.0
            tdhB = pyo.value(model.TDH_B_origin) if numB > 0 else 0.0
            dra_perc = float(pyo.value(model.DR_var[1])) if hasattr(model, 'DR_var') and 1 in model.DR_var else 0.0
            dra_ppm = float(pyo.value(model.PPM[1])) if model.PPM[1].value is not None else 0.0
            dra_cost_i = float(pyo.value(model.dra_cost[1])) if model.dra_cost[1].expr is not None else 0.0

            result[f"pipeline_flow_{name}"] = outflow
            result[f"pipeline_flow_in_{name}"] = inflow
            result[f"pump_flow_{name}"] = pump_flow
            result[f"num_pumps_typeA_{name}"] = numA
            result[f"num_pumps_typeB_{name}"] = numB
            result[f"speed_typeA_{name}"] = rpmA
            result[f"speed_typeB_{name}"] = rpmB
            result[f"efficiency_typeA_{name}"] = effA
            result[f"efficiency_typeB_{name}"] = effB
            result[f"tdh_typeA_{name}"] = tdhA
            result[f"tdh_typeB_{name}"] = tdhB
            result[f"drag_reduction_{name}"] = dra_perc
            result[f"dra_ppm_{name}"] = dra_ppm
            result[f"dra_cost_{name}"] = dra_cost_i
            result[f"head_loss_{name}"] = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i+1].value is not None else 0.0
            result[f"residual_head_{name}"] = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
            result[f"velocity_{name}"] = v[i]
            result[f"reynolds_{name}"] = Re[i]
            result[f"friction_{name}"] = f[i]
            result[f"sdh_{name}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
            result[f"maop_{name}"] = maop_dict[i]
            result[f"density_{name}"] = rho_dict[i]
        elif i in pump_indices:
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
            tdh_val = float(model.A[i] * (pump_flow * model.DOL[i] / rpm_val)**2 + model.B[i] * (pump_flow * model.DOL[i] / rpm_val) + model.C[i]) * (rpm_val / model.DOL[i])**2
            eff = (model.Pcoef[i]* (pump_flow * model.DOL[i] / rpm_val)**4 +
                   model.Qcoef[i]* (pump_flow * model.DOL[i] / rpm_val)**3 +
                   model.Rcoef[i]* (pump_flow * model.DOL[i] / rpm_val)**2 +
                   model.Scoef[i]* (pump_flow * model.DOL[i] / rpm_val) +
                   model.Tcoef[i]) if num_pumps > 0 else 0.0
            eff = float(eff)
            dra_ppm = float(pyo.value(model.PPM[i])) if model.PPM[i].value is not None else 0.0
            dra_cost_i = float(pyo.value(model.dra_cost[i])) if model.dra_cost[i].expr is not None else 0.0
            result[f"pipeline_flow_{name}"] = outflow
            result[f"pipeline_flow_in_{name}"] = inflow
            result[f"pump_flow_{name}"] = pump_flow
            result[f"num_pumps_{name}"] = num_pumps
            result[f"speed_{name}"] = rpm_val
            result[f"efficiency_{name}"] = eff
            result[f"power_cost_{name}"] = 0.0  # Add power cost if you want per station here
            result[f"dra_cost_{name}"] = dra_cost_i
            result[f"dra_ppm_{name}"] = dra_ppm
            result[f"drag_reduction_{name}"] = dra_perc
            result[f"head_loss_{name}"] = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i+1].value is not None else 0.0
            result[f"residual_head_{name}"] = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
            result[f"velocity_{name}"] = v[i]; result[f"reynolds_{name}"] = Re[i]; result[f"friction_{name}"] = f[i]
            result[f"sdh_{name}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
            result[f"maop_{name}"] = maop_dict[i]
            result[f"density_{name}"] = rho_dict[i]
        else:
            result[f"pipeline_flow_{name}"] = outflow
            result[f"pipeline_flow_in_{name}"] = inflow
            result[f"pump_flow_{name}"] = 0.0
            result[f"num_pumps_{name}"] = 0
            result[f"speed_{name}"] = 0.0
            result[f"efficiency_{name}"] = 0.0
            result[f"power_cost_{name}"] = 0.0
            result[f"dra_cost_{name}"] = 0.0
            result[f"dra_ppm_{name}"] = 0.0
            result[f"drag_reduction_{name}"] = 0.0
            result[f"head_loss_{name}"] = 0.0
            result[f"residual_head_{name}"] = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
            result[f"velocity_{name}"] = v[i]; result[f"reynolds_{name}"] = Re[i]; result[f"friction_{name}"] = f[i]
            result[f"sdh_{name}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
            result[f"maop_{name}"] = maop_dict[i]
            result[f"density_{name}"] = rho_dict[i]

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
        f"density_{term_name}": rho_dict.get(N, list(rho_dict.values())[-1]),
    })
    result['total_cost'] = float(pyo.value(model.Obj)) if model.Obj is not None else 0.0
    result["error"] = False
    return result
