import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

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

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None):
    import numpy as np
    import pandas as pd

    RPM_STEP = 100  # RPM step
    DRA_STEP = 5    # DRA step

    N = len(stations)
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Compute flow in each segment
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # Main pipe params (same for all)
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72

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

    # --- Dual pump type structure ---
    pump_station_types = []
    pump_types = {}
    pump_stations = []
    for i, stn in enumerate(stations, start=1):
        if stn.get('is_pump', False):
            pump_stations.append(i)
            if "pumps" in stn and isinstance(stn["pumps"], list):
                for t, pt in enumerate(stn["pumps"]):
                    pump_station_types.append((i, t))
                    pump_types[(i, t)] = pt

    model.pump_stations = pyo.Set(initialize=pump_stations)
    model.pump_station_types = pyo.Set(dimen=2, initialize=pump_station_types)

    # --- Head and efficiency coefficients per type ---
    def get_init(pn, default=0.0):
        return {(i, t): float(pt.get(pn, default)) for (i, t), pt in pump_types.items()}

    model.A = pyo.Param(model.pump_station_types, initialize=get_init("A"))
    model.B = pyo.Param(model.pump_station_types, initialize=get_init("B"))
    model.C = pyo.Param(model.pump_station_types, initialize=get_init("C"))
    model.Pcoef = pyo.Param(model.pump_station_types, initialize=get_init("P"))
    model.Qcoef = pyo.Param(model.pump_station_types, initialize=get_init("Q"))
    model.Rcoef = pyo.Param(model.pump_station_types, initialize=get_init("R"))
    model.Scoef = pyo.Param(model.pump_station_types, initialize=get_init("S"))
    model.Tcoef = pyo.Param(model.pump_station_types, initialize=get_init("T"))
    model.MinRPM = pyo.Param(model.pump_station_types, initialize=get_init("MinRPM"))
    model.DOL = pyo.Param(model.pump_station_types, initialize=get_init("DOL"))

    # RPM & DRA discrete options for each (i, t)
    allowed_rpms = {}
    allowed_dras = {}
    max_dr = {}
    for (i, t), pt in pump_types.items():
        min_rpm = int(pt.get("MinRPM", 0))
        max_rpm = int(pt.get("DOL", 0))
        allowed_rpms[(i, t)] = [r for r in range(min_rpm, max_rpm+1, RPM_STEP)]
        if allowed_rpms[(i, t)][-1] != max_rpm:
            allowed_rpms[(i, t)].append(max_rpm)
        max_dr[(i, t)] = pt.get("max_dr", stn.get("max_dr", 0.0))
        allowed_dras[(i, t)] = [d for d in range(0, int(max_dr[(i, t)])+1, DRA_STEP)]
        if allowed_dras[(i, t)][-1] != int(max_dr[(i, t)]):
            allowed_dras[(i, t)].append(int(max_dr[(i, t)]))

    # Number of pumps (variables) per (i, t)
    def nop_bounds(m, i, t):
        pt = pump_types[(i, t)]
        lb = 0
        # Enforce at least 1 pump running at first pump station (origin)
        if i == min(pump_stations):
            lb = 1
        ub = int(pt.get("max_units", 2))
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_station_types, domain=pyo.NonNegativeIntegers, bounds=nop_bounds, initialize=1)

    # Limit total pumps per station
    def max_pumps_at_station_rule(m, i):
        stn = stations[i-1]
        max_pumps = int(stn.get("max_pumps", 2))
        return sum(m.NOP[i, t] for t, pt in enumerate(stn["pumps"])) <= max_pumps
    model.max_pumps_at_station = pyo.Constraint(model.pump_stations, rule=max_pumps_at_station_rule)

    # RPM binaries and values
    model.rpm_bin = pyo.Var(
        ((i, t, j) for (i, t) in pump_station_types for j in range(len(allowed_rpms[(i, t)]))),
        domain=pyo.Binary
    )
    def rpm_bin_sum_rule(m, i, t):
        return sum(m.rpm_bin[i, t, j] for j in range(len(allowed_rpms[(i, t)]))) == 1
    model.rpm_bin_sum = pyo.Constraint(model.pump_station_types, rule=rpm_bin_sum_rule)
    model.RPM_var = pyo.Var(model.pump_station_types, domain=pyo.NonNegativeReals)
    for key in model.RPM_var:
        model.RPM_var[key].setlb(1.0)

    def rpm_value_rule(m, i, t):
        return m.RPM_var[i, t] == sum(allowed_rpms[(i, t)][j] * m.rpm_bin[i, t, j] for j in range(len(allowed_rpms[(i, t)])))
    model.rpm_value = pyo.Constraint(model.pump_station_types, rule=rpm_value_rule)

    # DRA binaries and values
    model.dra_bin = pyo.Var(
        ((i, t, j) for (i, t) in pump_station_types for j in range(len(allowed_dras[(i, t)]))),
        domain=pyo.Binary
    )
    def dra_bin_sum_rule(m, i, t):
        return sum(m.dra_bin[i, t, j] for j in range(len(allowed_dras[(i, t)]))) == 1
    model.dra_bin_sum = pyo.Constraint(model.pump_station_types, rule=dra_bin_sum_rule)
    def dra_var_bounds(m, i, t):
        return (min(allowed_dras[(i, t)]), max(allowed_dras[(i, t)]))
    model.DR_var = pyo.Var(model.pump_station_types, bounds=dra_var_bounds, domain=pyo.NonNegativeReals)
    def dra_value_rule(m, i, t):
        return m.DR_var[i, t] == sum(allowed_dras[(i, t)][j] * m.dra_bin[i, t, j] for j in range(len(allowed_dras[(i, t)])))
    model.dra_value = pyo.Constraint(model.pump_station_types, rule=dra_value_rule)

    # Residual head
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    # Flow/velocity/friction factors
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

    # SDH, head balance, peak, pressure constraints
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}
    EFFP = {}

    for i in range(1, N+1):
        # DRA: If any pump at this station, sum all DR
        DR_total = sum(model.DR_var[i, t]/100.0 * model.NOP[i, t] for t, pt in enumerate(stations[i-1].get("pumps", []))) if stations[i-1].get("is_pump", False) else 0.0
        DR_frac = DR_total if stations[i-1].get("is_pump", False) else 0.0
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

    # Calculate TDH/EFF per type
    for (i, t), pt in pump_types.items():
        pump_flow = float(segment_flows[i])
        rpm_val = model.RPM_var[i, t]
        dol_val = model.DOL[i, t]
        Q_equiv = pump_flow * dol_val / rpm_val
        TDH[(i, t)] = (model.A[i, t] * Q_equiv**2 + model.B[i, t] * Q_equiv + model.C[i, t]) * (rpm_val / dol_val)**2
        EFFP[(i, t)] = (model.Pcoef[i, t]*Q_equiv**4 + model.Qcoef[i, t]*Q_equiv**3 +
                       model.Rcoef[i, t]*Q_equiv**2 + model.Scoef[i, t]*Q_equiv +
                       model.Tcoef[i, t]) / 100.0

    # Head balance: sum TDH*NOP over all types at station
    model.head_balance = pyo.ConstraintList()
    for i in range(1, N+1):
        if stations[i-1].get('is_pump', False):
            types_here = [t for t, _ in enumerate(stations[i-1]['pumps'])]
            model.head_balance.add(
                model.RH[i] + sum(TDH[(i, t)]*model.NOP[i, t] for t in types_here) >= model.SDH[i]
            )
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])

    # MAOP/peak limits
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N+1):
        D_out = d_inner[i] + 2 * thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            loss_no_dra = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g))
            # If any type at this station, sum all TDH*NOP
            tdh_sum = sum(TDH[(i, t)]*model.NOP[i, t] for t, pt in enumerate(stations[i-1].get("pumps", []))) if stations[i-1].get("is_pump", False) else 0.0
            expr = model.RH[i] + tdh_sum - (elev_k - model.z[i]) - loss_no_dra if stations[i-1].get("is_pump", False) else model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    # DRA: piecewise mapping for each (i, t) to compute ppm, cost
    model.PPM = pyo.Var(model.pump_station_types, domain=pyo.NonNegativeReals)
    model.dra_cost = pyo.Expression(model.pump_station_types)
    for (i, t), pt in pump_types.items():
        visc = kv_dict[i]
        dr_points, ppm_points = get_ppm_breakpoints(visc)
        dr_points_fixed, ppm_points_fixed = zip(*sorted(set(zip(dr_points, ppm_points))))
        setattr(model, f'piecewise_dra_ppm_{i}_{t}',
            pyo.Piecewise(f'pw_dra_ppm_{i}_{t}', model.PPM[i, t], model.DR_var[i, t],
                          pw_pts=dr_points_fixed,
                          f_rule=ppm_points_fixed,
                          pw_constr_type='EQ'))
        dra_cost_expr = model.PPM[i, t] * (segment_flows[i] * 1000.0 * 24.0 / 1e6) * RateDRA * model.NOP[i, t]
        model.dra_cost[i, t] = dra_cost_expr

    # --- Objective: total power + DRA cost ---
    total_cost = 0
    for (i, t), pt in pump_types.items():
        rho_i = rho_dict[i]
        pump_flow = float(segment_flows[i])
        rpm_val = model.RPM_var[i, t]
        eff_val = EFFP[(i, t)]
        # Power type logic
        power_type = pt.get("power_type", "Grid").lower()
        if power_type == "grid":
            elec_cost = float(pt.get("rate", 0.0))
            power_kW = (rho_i * pump_flow * 9.81 * TDH[(i, t)] * model.NOP[i, t]) / (3600.0 * 1000.0 * (eff_val + 1e-6) * 0.95)
            power_cost = power_kW * 24.0 * elec_cost
        else:
            sfc = float(pt.get("sfc", 0.0))
            power_kW = (rho_i * pump_flow * 9.81 * TDH[(i, t)] * model.NOP[i, t]) / (3600.0 * 1000.0 * (eff_val + 1e-6) * 0.95)
            fuel_per_kWh = (sfc * 1.34102) / 820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        dra_cost_i = model.dra_cost[i, t]
        total_cost += power_cost + dra_cost_i

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

    # --- Collect results ---
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0

        if stn.get('is_pump', False):
            for t, pt in enumerate(stn["pumps"]):
                key = f"{name}_type{t+1}"
                num_pumps = int(round(pyo.value(model.NOP[i, t]))) if model.NOP[i, t].value is not None else 0
                rpm_val = None
                for j in range(len(allowed_rpms[(i, t)])):
                    if round(pyo.value(model.rpm_bin[i, t, j])) == 1:
                        rpm_val = allowed_rpms[(i, t)][j]
                        break
                dra_perc = None
                for j in range(len(allowed_dras[(i, t)])):
                    if round(pyo.value(model.dra_bin[i, t, j])) == 1:
                        dra_perc = allowed_dras[(i, t)][j]
                        break
                if rpm_val is None:
                    rpm_val = allowed_rpms[(i, t)][0]
                if dra_perc is None:
                    dra_perc = allowed_dras[(i, t)][0]
                dol_val = model.DOL[i, t]
                Q_equiv = pump_flow * dol_val / rpm_val if rpm_val else 0.0
                tdh_val = float(model.A[i, t] * Q_equiv**2 + model.B[i, t] * Q_equiv + model.C[i, t]) * (rpm_val/dol_val)**2 if rpm_val else 0.0
                eff = (model.Pcoef[i, t]*Q_equiv**4 + model.Qcoef[i, t]*Q_equiv**3 +
                       model.Rcoef[i, t]*Q_equiv**2 + model.Scoef[i, t]*Q_equiv +
                       model.Tcoef[i, t]) if num_pumps > 0 and rpm_val else 0.0
                eff = float(eff)
                dra_ppm = float(pyo.value(model.PPM[i, t])) if model.PPM[i, t].value is not None else 0.0
                dra_cost_i = float(pyo.value(model.dra_cost[i, t])) if hasattr(model.dra_cost[i, t], "expr") else 0.0

                # Zero all if optimizer turns off all
                if num_pumps == 0:
                    rpm_val = 0.0
                    eff = 0.0
                    dra_perc = 0.0
                    dra_ppm = 0.0
                    dra_cost_i = 0.0
                    tdh_val = 0.0

                # Power cost per type
                rho_i = rho_dict[i]
                if num_pumps > 0 and eff > 0 and rpm_val > 0:
                    power_type = pt.get("power_type", "Grid").lower()
                    if power_type == "grid":
                        rate = float(pt.get("rate", 0.0))
                        power_kW = (rho_i * pump_flow * 9.81 * tdh_val * num_pumps) / (3600.0 * 1000.0 * (eff/100.0) * 0.95)
                        power_cost = power_kW * 24.0 * rate
                    else:
                        sfc = float(pt.get("sfc", 0.0))
                        fuel_per_kWh = (sfc * 1.34102) / 820.0
                        power_kW = (rho_i * pump_flow * 9.81 * tdh_val * num_pumps) / (3600.0 * 1000.0 * (eff/100.0) * 0.95)
                        power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
                else:
                    power_cost = 0.0

                result[f"num_pumps_{key}"] = num_pumps
                result[f"speed_{key}"] = rpm_val
                result[f"efficiency_{key}"] = eff
                result[f"power_cost_{key}"] = power_cost
                result[f"dra_cost_{key}"] = dra_cost_i
                result[f"dra_ppm_{key}"] = dra_ppm
                result[f"drag_reduction_{key}"] = dra_perc
                result[f"tdh_{key}"] = tdh_val

        # Report sum (total) for station as a whole (all pump types)
        total_pumps = sum(int(round(pyo.value(model.NOP[i, t]))) for t, pt in enumerate(stn["pumps"])) if stn.get("is_pump", False) else 0
        total_power_cost = sum(
            float(pyo.value(model.dra_cost[i, t])) +
            (
                (rho_dict[i] * pump_flow * 9.81 *
                 float(model.A[i, t] * (pump_flow * model.DOL[i, t] / (model.RPM_var[i, t]) + 1e-3)**2 +
                       model.B[i, t] * (pump_flow * model.DOL[i, t] / (model.RPM_var[i, t]) + 1e-3) +
                       model.C[i, t]) * (model.RPM_var[i, t]/model.DOL[i, t])**2) *
                 int(round(pyo.value(model.NOP[i, t]))) / (3600.0 * 1000.0 *
                 ((model.Pcoef[i, t]*((pump_flow * model.DOL[i, t]/(model.RPM_var[i, t]) + 1e-3)**4) +
                   model.Qcoef[i, t]*((pump_flow * model.DOL[i, t]/(model.RPM_var[i, t]) + 1e-3)**3) +
                   model.Rcoef[i, t]*((pump_flow * model.DOL[i, t]/(model.RPM_var[i, t]) + 1e-3)**2) +
                   model.Scoef[i, t]*(pump_flow * model.DOL[i, t]/(model.RPM_var[i, t]) + 1e-3) +
                   model.Tcoef[i, t])/100.0) * 0.95) * 24.0 * float(pt.get("rate", 0.0)) if pt.get("power_type", "Grid").lower() == "grid" else 0.0
            )
            for t, pt in enumerate(stn["pumps"])
        ) if stn.get("is_pump", False) else 0.0


        result[f"pipeline_flow_{name}"] = outflow
        result[f"pipeline_flow_in_{name}"] = inflow
        result[f"pump_flow_{name}"] = pump_flow
        result[f"num_pumps_{name}"] = total_pumps

        # Head, velocity, etc. (from main pipe, not pump)
        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i+1].value is not None else 0.0
        res_head = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
        velocity = v[i]; reynolds = Re[i]; fric = f[i]
        result[f"head_loss_{name}"] = head_loss
        result[f"residual_head_{name}"] = res_head
        result[f"velocity_{name}"] = velocity
        result[f"reynolds_{name}"] = reynolds
        result[f"friction_{name}"] = fric
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
        result[f"maop_{name}"] = maop_dict[i]

    term_name = terminal.get('name','terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": segment_flows[-1],
        f"pipeline_flow_in_{term_name}": segment_flows[-2],
        f"pump_flow_{term_name}": 0.0,
        f"num_pumps_{term_name}": 0,
        f"speed_{term_name}": 0.0,
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
