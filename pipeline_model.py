import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

# Load DRA curves for all possible viscosities present
DRA_CSV_FILES = {10: "10 cst.csv", 15: "15 cst.csv", 20: "20 cst.csv", 25: "25 cst.csv", 30: "30 cst.csv", 35: "35 cst.csv", 40: "40 cst.csv"}
DRA_CURVE_DATA = {}
for cst, fname in DRA_CSV_FILES.items():
    DRA_CURVE_DATA[cst] = pd.read_csv(fname) if os.path.exists(fname) else None

def get_ppm_breakpoints(visc):
    """Interpolates (%DragReduction, PPM) points for any viscosity (cst)."""
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

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, Rate_DRA, Price_HSD, linefill_dict):
    try:
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

        # --- Segment Flows ---
        segment_flows = [float(FLOW)]
        for stn in stations:
            prev_flow = segment_flows[-1]
            delivery = float(stn.get('delivery', 0.0))
            supply = float(stn.get('supply', 0.0))
            segment_flows.append(prev_flow - delivery + supply)
        # segment_flows[1..N] aligns with station index 1..N
        segment_flows = segment_flows[:N+1] # exactly N+1

        # --- Station Data Prep ---
        length = {}; d_inner = {}; rough = {}; thick = {}; smys = {}; design_fac = {}; elev = {}
        Acoef = {}; Bcoef = {}; Ccoef = {}
        Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
        min_rpm = {}; max_rpm = {}
        sfc = {}; elec_rate = {}
        pump_stations = []; diesel_stations = []; electric_stations = []
        max_dr = {}; peaks = {}
        default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72

        for i, stn in enumerate(stations, start=1):
            length[i] = float(stn.get('L', 0.0))
            if 'D' in stn:
                thick[i] = float(stn.get('t', default_t))
                d_inner[i] = float(stn['D']) - 2 * thick[i]
            elif 'd' in stn:
                d_inner[i] = float(stn['d'])
                thick[i] = float(stn.get('t', default_t))
            else:
                d_inner[i] = 0.7
                thick[i] = default_t
            rough[i] = float(stn.get('rough', default_e))
            smys[i] = float(stn.get('SMYS', default_smys))
            design_fac[i] = float(stn.get('DF', default_df))
            elev[i] = float(stn.get('elev', 0.0))
            peaks[i] = stn.get('peaks', [])
            if stn.get('is_pump', False):
                dol = float(stn.get('DOL', 0))
                if dol <= 0:
                    return {"error": True, "message": f"Station '{stn.get('name', i)}' missing DOL (rated RPM)."}
                pump_stations.append(i)
                Acoef[i] = float(stn.get('A', 0.0))
                Bcoef[i] = float(stn.get('B', 0.0))
                Ccoef[i] = float(stn.get('C', 0.0))
                Pcoef[i] = float(stn.get('P', 0.0))
                Qcoef[i] = float(stn.get('Q', 0.0))
                Rcoef[i] = float(stn.get('R', 0.0))
                Scoef[i] = float(stn.get('S', 0.0))
                Tcoef[i] = float(stn.get('T', 0.0))
                min_rpm_val = max(1, int(stn.get('MinRPM', 1)))
                max_rpm_val = max(min_rpm_val, int(dol))
                min_rpm[i] = min_rpm_val
                max_rpm[i] = max_rpm_val
                if stn.get('sfc', 0):
                    diesel_stations.append(i)
                    sfc[i] = float(stn.get('sfc', 0.0))
                else:
                    electric_stations.append(i)
                    elec_rate[i] = float(stn.get('rate', 0.0))
                max_dr[i] = float(stn.get('max_dr', 0.0))
        elev[N+1] = float(terminal.get('elev', 0.0))

        # Pyomo sets/params
        model.L     = pyo.Param(model.I, initialize=length)
        model.d     = pyo.Param(model.I, initialize=d_inner)
        model.e     = pyo.Param(model.I, initialize=rough)
        model.SMYS  = pyo.Param(model.I, initialize=smys)
        model.DF    = pyo.Param(model.I, initialize=design_fac)
        model.z     = pyo.Param(model.Nodes, initialize=elev)
        model.pump_stations = pyo.Set(initialize=pump_stations)
        if pump_stations:
            model.A = pyo.Param(model.pump_stations, initialize=Acoef)
            model.B = pyo.Param(model.pump_stations, initialize=Bcoef)
            model.C = pyo.Param(model.pump_stations, initialize=Ccoef)
            model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef)
            model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef)
            model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef)
            model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef)
            model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef)
            model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm)
            model.DOL    = pyo.Param(model.pump_stations, initialize=max_rpm)
        model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=lambda m, j: (1, stations[j-1].get('max_pumps', 2)) if j == pump_stations[0] else (0, stations[j-1].get('max_pumps', 2)), initialize=1)
        if pump_stations:
            def speed_bounds(m, j):
                lo = max(1, (int(min_rpm.get(j, 1)) + 9)//10)
                hi = max(lo, int(max_rpm.get(j, 0))//10) if max_rpm.get(j, 0) else lo
                return (lo, hi)
            model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=speed_bounds)
            model.N = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.N_u[j])
        else:
            model.N_u = pyo.Var([], domain=pyo.NonNegativeIntegers)
        model.DR = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals, bounds=lambda m, j: (0, max_dr.get(j, 0.0)), initialize=0.0)
        model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals)
        model.RH[1].fix(float(stations[0].get('min_residual', 50.0)))
        for node in range(2, N+2):
            model.RH[node].setlb(50.0)
        g = 9.81
        v = {}; f = {}
        for i in range(1, N+1):
            flow = segment_flows[i]
            area = pi * (d_inner[i] ** 2) / 4.0
            vel = (flow/3600.0) / area if area > 0 else 0.0
            v[i] = vel
            Re = vel * d_inner[i] / (kv_dict[i] * 1e-6) if kv_dict[i] > 0 else 0.0
            if Re > 0:
                if Re < 4000:
                    f[i] = 64.0 / Re
                else:
                    arg = (rough[i] / d_inner[i] / 3.7) + (5.74 / (Re ** 0.9))
                    f[i] = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
            else:
                f[i] = 0.0
        model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals)
        model.segment_constraints = pyo.ConstraintList()
        if pump_stations:
            model.Q_equiv = pyo.Expression(model.pump_stations, rule=lambda m, j: (segment_flows[j] * m.DOL[j] / m.N[j]))
            model.EFFP = pyo.Expression(model.pump_stations, rule=lambda m, j: (m.Pcoef[j] * m.Q_equiv[j]**4 + m.Qcoef[j] * m.Q_equiv[j]**3 + m.Rcoef[j] * m.Q_equiv[j]**2 + m.Scoef[j] * m.Q_equiv[j] + m.Tcoef[j]) / 100.0)
        else:
            model.EFFP = pyo.Expression(model.I, initialize=1.0)
        TDH = {}
        for i in range(1, N+1):
            DR_fraction = model.DR[i] / 100.0 if i in pump_stations else 0.0
            head_loss = f[i] * ((length[i] * 1000.0) / d_inner[i]) * ((v[i] ** 2) / (2 * g)) * (1 - DR_fraction)
            elev_diff = model.z[i+1] - model.z[i]
            model.segment_constraints.add(model.SDH[i] >= model.RH[i+1] + elev_diff + head_loss)
            for peak in peaks[i]:
                Lp = peak['loc'] * 1000.0
                elev_peak = peak['elev']
                peak_loss = f[i] * (Lp / d_inner[i]) * ((v[i] ** 2) / (2 * g)) * (1 - DR_fraction)
                model.segment_constraints.add(model.SDH[i] >= (elev_peak - model.z[i]) + peak_loss + 50.0)
            if i in pump_stations:
                N_i = model.N[i]
                DOL_i = model.DOL[i]
                Q_equiv = segment_flows[i] * DOL_i / N_i
                H_full = model.A[i] * Q_equiv**2 + model.B[i] * Q_equiv + model.C[i]
                TDH[i] = H_full * (N_i / DOL_i) ** 2
            else:
                TDH[i] = 0.0
        model.pressure_constraints = pyo.ConstraintList()
        maop_head = {}
        for i in range(1, N+1):
            if i in pump_stations:
                model.pressure_constraints.add(model.RH[i] + TDH[i] * model.NOP[i] >= model.SDH[i])
            else:
                model.pressure_constraints.add(model.RH[i] >= model.SDH[i])
            D_out = d_inner[i] + 2 * thick[i]
            MAOP = (2 * thick[i] * (smys[i] * 0.070307) * design_fac[i] / D_out) * 10000.0 / rho_dict[i]
            maop_head[i] = MAOP
            model.pressure_constraints.add(model.SDH[i] <= MAOP)
            for peak in peaks[i]:
                elev_peak = peak['elev']
                Lp = peak['loc'] * 1000.0
                loss_no_dra = f[i] * (Lp / d_inner[i]) * ((v[i] ** 2) / (2 * g))
                if i in pump_stations:
                    model.pressure_constraints.add(model.RH[i] + TDH[i] * model.NOP[i] >= (elev_peak - model.z[i]) + loss_no_dra + 50.0)
                else:
                    model.pressure_constraints.add(model.RH[i] >= (elev_peak - model.z[i]) + loss_no_dra + 50.0)
        model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
        dra_piecewise_blocks = {}
        model.dra_cost = pyo.Expression(model.pump_stations, rule=lambda m, j: 0.0)
        for i in pump_stations:
            dr_pts, ppm_pts = get_ppm_breakpoints(kv_dict[i])
            if dr_pts is None or ppm_pts is None or all([p==0 for p in ppm_pts]):
                model.DR[i].fix(0.0)
                model.PPM[i].fix(0.0)
            else:
                # Pyomo Piecewise must be attached per-variable
                dra_piecewise_blocks[i] = pyo.Piecewise(
                    model.PPM[i], model.DR[i],
                    pw_pts=list(dr_pts), f_rule=list(ppm_pts),
                    pw_constr_type='EQ')
                setattr(model, f'dra_pw_{i}', dra_piecewise_blocks[i])
            volume_day = segment_flows[i] * 24.0
            model.dra_cost[i] = model.PPM[i] * (volume_day * 1000.0 / 1e6) * model.Rate_DRA
        model.power_use = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
        model.power_balance = pyo.ConstraintList()
        for i in pump_stations:
            lhs = rho_dict[i] * segment_flows[i] * 9.81 * TDH[i] * model.NOP[i]
            model.power_balance.add(lhs == model.power_use[i] * (3600.0 * 1000.0 * 0.95) * model.EFFP[i])
        total_cost = 0.0
        for i in pump_stations:
            if i in electric_stations:
                cost_per_kWh = elec_rate.get(i, 0.0)
                total_cost += model.power_use[i] * 24.0 * cost_per_kWh + model.dra_cost[i]
            else:
                fuel_factor = (sfc.get(i, 0.0) * 1.34102) / 820.0
                total_cost += model.power_use[i] * 24.0 * fuel_factor * model.Price_HSD + model.dra_cost[i]
        model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

        results = SolverManagerFactory('neos').solve(model, solver='couenne', tee=False)
        status = results.solver.status
        term = results.solver.termination_condition
        if (status != pyo.SolverStatus.ok) or (term not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
            return {"error": True, "message": f"Optimization failed: {term}. Check inputs or relax constraints if necessary.", "termination_condition": str(term), "solver_status": str(status)}
        model.solutions.load_from(results)

        output = {}
        for i, stn in enumerate(stations, start=1):
            name_key = stn['name'].strip().lower().replace(' ', '_')
            flow_out = segment_flows[i]
            output[f"pipeline_flow_{name_key}"] = float(flow_out)
            output[f"station_elevation_{name_key}"] = float(elev[i])
            output[f"residual_head_{name_key}"] = float(pyo.value(model.RH[i]))
            output[f"sdh_{name_key}"] = float(pyo.value(model.SDH[i]))
            output[f"maop_{name_key}"] = float(maop_head.get(i, 0.0))
            output[f"velocity_{name_key}"] = float(v.get(i, 0.0))
            output[f"reynolds_{name_key}"] = float(v.get(i, 0.0) * d_inner[i] / (kv_dict[i] * 1e-6)) if kv_dict[i] > 0 else 0.0
            output[f"friction_{name_key}"] = float(f.get(i, 0.0))
            if i in pump_stations:
                num_pumps = int(pyo.value(model.NOP[i]))
                output[f"num_pumps_{name_key}"] = num_pumps
                output[f"pump_flow_{name_key}"] = float(flow_out) if num_pumps > 0 else 0.0
                output[f"speed_{name_key}"] = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
                output[f"efficiency_{name_key}"] = float(pyo.value(model.EFFP[i]) * 100.0) if num_pumps > 0 else 0.0
                output[f"drag_reduction_{name_key}"] = float(pyo.value(model.DR[i]))
                output[f"dra_ppm_{name_key}"] = float(pyo.value(model.PPM[i]))
                output[f"dra_cost_{name_key}"] = float(pyo.value(model.dra_cost[i]))
                power_kW = float(pyo.value(model.power_use[i]))
                if i in electric_stations:
                    cost_per_kWh = elec_rate.get(i, 0.0)
                    output[f"power_cost_{name_key}"] = power_kW * 24.0 * cost_per_kWh
                else:
                    fuel_factor = (sfc.get(i, 0.0) * 1.34102) / 820.0
                    output[f"power_cost_{name_key}"] = power_kW * 24.0 * fuel_factor * float(pyo.value(model.Price_HSD))
            else:
                output[f"num_pumps_{name_key}"] = 0
                output[f"pump_flow_{name_key}"] = 0.0
                output[f"speed_{name_key}"] = 0.0
                output[f"efficiency_{name_key}"] = 0.0
                output[f"drag_reduction_{name_key}"] = 0.0
                output[f"dra_ppm_{name_key}"] = 0.0
                output[f"dra_cost_{name_key}"] = 0.0
                output[f"power_cost_{name_key}"] = 0.0
        term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
        output[f"pipeline_flow_{term_name}"] = float(segment_flows[-1])
        output[f"station_elevation_{term_name}"] = float(elev[N+1])
        output[f"residual_head_{term_name}"] = float(pyo.value(model.RH[N+1]))
        return output
    except Exception as e:
        return {"error": True, "message": f"Python backend exception: {str(e)}"}