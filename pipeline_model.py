import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
import pandas as pd
import numpy as np
from math import pi, log10

def load_dra_curves(base_dir='./'):
    dra_curves = {}
    for visc in range(10, 45, 5):
        csv_path = os.path.join(base_dir, f"{visc} cst.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)

        # --- Robustly detect correct columns ---
        # Lowercase and strip all column names, remove special chars
        clean_cols = [c.strip().lower().replace('%', '').replace(' ', '').replace('_','') for c in df.columns]
        col_map = {}
        for idx, c in enumerate(clean_cols):
            if 'dragreduction' in c:
                col_map['dr'] = df.columns[idx]
            if 'ppm' == c:
                col_map['ppm'] = df.columns[idx]

        # Check both columns are present
        if not ('dr' in col_map and 'ppm' in col_map):
            raise Exception(f"Columns not found in {csv_path}. Found: {df.columns}")

        pairs = sorted([
            (float(row[col_map['dr']]), float(row[col_map['ppm']]))
            for _, row in df.iterrows()
            if pd.notnull(row[col_map['dr']]) and pd.notnull(row[col_map['ppm']])
               and float(row[col_map['dr']]) >= 0 and float(row[col_map['ppm']]) >= 0
        ])
        if len(pairs) >= 2:
            dra_curves[visc] = pairs
    return dra_curves


def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, dra_curve_dir='./'
):
    os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')
    dra_curves = load_dra_curves(dra_curve_dir)
    available_visc = sorted(dra_curves.keys())
    if not available_visc:
        raise Exception("No DRA curves found!")

    N = len(stations)

    # Compute flow_in and flow_out at each station, including deliveries/supplies
    flow_in = []
    flow_out = []
    f = float(FLOW)
    for stn in stations:
        flow_in.append(f)
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        f = f - delivery + supply
        flow_out.append(f)
    flow_in.append(f)
    flow_out.append(f)

    # Segment flow is always flow_out[i-1]
    length = {}; d_inner = {}; thickness = {}; roughness = {}; elev = {}
    kv_dict = {}; rho_dict = {}; smys = {}; design_factor = {}; peaks_dict = {}
    pump_indices = []; min_rpm = {}; max_rpm = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    sfc = {}; elec_cost = {}; diesel_pumps = []; electric_pumps = []
    max_dr = {}; 
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
        elev[i] = stn.get('elev', 0.0)
        kv_dict[i] = float(KV_list[i-1])
        rho_dict[i] = float(rho_list[i-1])
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        peaks_dict[i] = stn.get('peaks', [])
        if stn.get('is_pump', False):
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
            if stn.get('sfc', 0):
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc', 0.0)
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 0.0)
        max_dr[i] = stn.get('max_dr', 40.0)

    elev[N+1] = terminal.get('elev', 0.0)

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)
    model.pump_stations = pyo.Set(initialize=pump_indices)

    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.t = pyo.Param(model.I, initialize=thickness)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.kv = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.z = pyo.Param(model.Nodes, initialize=elev)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.RateDRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)
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

    model.DR = pyo.Var(model.I, bounds=lambda m,i: (0.0, max_dr[i]), initialize=0.0)
    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=nop_bounds, initialize=1)
    def speed_bounds(m, j):
        lo = int(model.MinRPM[j]) if model.MinRPM[j] else 1
        hi = int(model.DOL[j]) if model.DOL[j] else lo
        return (lo, hi)
    model.N = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals, bounds=speed_bounds, initialize=lambda m,j: (speed_bounds(m,j)[0]+speed_bounds(m,j)[1])//2)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50.0)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0.0)
    model.PPM = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0.0)
    for i in range(1, N+1):
        visc = kv_dict[i]
        vlow = min(available_visc, key=lambda v: abs(v-visc))
        curve_pts = dra_curves[vlow]
        DR_pts, PPM_pts = zip(*curve_pts)
        model.add_component(f'pw_dra_{i}',
            pyo.Piecewise(
                f'PW_{i}',
                model.PPM[i], model.DR[i],
                pw_pts=list(DR_pts),
                pw_constr_type='EQ',
                f_rule=lambda *args: float(np.interp(args[-1], DR_pts, PPM_pts)),
                pw_repn='SOS2'
            )
        )
    g = 9.81
    v_expr = {}; f_expr = {}; Re_expr = {}
    for i in range(1, N+1):
        flow = flow_out[i-1]
        flow_m3s = flow / 3600.0
        area = pi * (d_inner[i] ** 2) / 4.0 if d_inner[i] > 0 else 0.0
        v = flow_m3s / area if area > 0 else 0.0
        v_expr[i] = v
        Re_expr[i] = v * d_inner[i] / (kv_dict[i] * 1e-6) if kv_dict[i] > 0 else 0.0
        f_val = 0.0
        if Re_expr[i] < 4000 and Re_expr[i] > 0:
            f_val = 64.0/Re_expr[i]
        elif Re_expr[i] > 4000:
            e_d = roughness[i] / d_inner[i] / 3.7
            B = 5.74 / (Re_expr[i]**0.9)
            arg = e_d + B
            f_val = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
        f_expr[i] = f_val

    model.sdh_constraint = pyo.ConstraintList()
    for i in range(1, N+1):
        DR_frac = model.DR[i] / 100.0
        head_loss = f_expr[i] * (length[i]*1000.0/d_inner[i]) * (v_expr[i]**2/(2*g)) * (1 - DR_frac)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + head_loss
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            head_loss_peak = f_expr[i] * (L_peak/d_inner[i]) * (v_expr[i]**2/(2*g)) * (1 - DR_frac)
            expr_peak = (elev_k - model.z[i]) + head_loss_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

    TDH = {}
    EFFP = {}
    for i in pump_indices:
        flow = flow_out[i-1]
        n_var = model.N[i]
        TDH[i] = (model.A[i]*flow**2 + model.B[i]*flow + model.C[i]) * ((n_var/model.DOL[i])**2)
        flow_eq = flow * model.DOL[i]/n_var
        EFFP[i] = (
            model.Pcoef[i]*flow_eq**4 + model.Qcoef[i]*flow_eq**3 + model.Rcoef[i]*flow_eq**2
            + model.Scoef[i]*flow_eq + model.Tcoef[i]
        ) / 100.0

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
            head_loss_no_dra = f_expr[i] * (L_peak/d_inner[i]) * (v_expr[i]**2/(2*g))
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - head_loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - head_loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    total_cost = 0
    for i in range(1, N+1):
        ppm_i = model.PPM[i]
        dra_m3day = flow_out[i-1]*24.0
        dra_kg_day = ppm_i * dra_m3day / 1e6
        dra_cost = dra_kg_day * model.RateDRA
        power_cost = 0.0
        if i in pump_indices:
            rho_i = rho_dict[i]
            pump_flow = flow_out[i-1]
            num_pumps = model.NOP[i]
            n_var = model.N[i]
            eff = EFFP[i]
            power_kW = (rho_i * pump_flow * 9.81 * TDH[i] * num_pumps) / (3600.0*1000.0*eff*0.95)
            if i in electric_pumps:
                rate = elec_cost.get(i,0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i,0.0)
                fuel_per_kWh = (sfc_val*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * model.Price_HSD
        total_cost += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        is_pump = (i in pump_indices)
        num_pumps = int(pyo.value(model.NOP[i])) if is_pump else 0
        speed_rpm = float(pyo.value(model.N[i])) if is_pump else 0.0
        eff = float(pyo.value(EFFP[i])*100.0) if is_pump else 0.0
        power_cost = 0.0
        if is_pump:
            rho_i = rho_dict[i]
            pump_flow = flow_out[i-1]
            power_kW = (rho_i * pump_flow * 9.81 * float(pyo.value(TDH[i])) * num_pumps) / (3600.0*1000.0*float(pyo.value(EFFP[i]))*0.95)
            if i in electric_pumps:
                rate = elec_cost.get(i,0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i,0.0)
                fuel_per_kWh = (sfc_val*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD

        viscosity = kv_dict[i]
        ppm_val = float(pyo.value(model.PPM[i]))
        dra_cost = ppm_val * flow_out[i-1]*24.0 / 1e6 * RateDRA
        drag_red = float(pyo.value(model.DR[i]))
        res_head = float(pyo.value(model.RH[i]))
        sdh_val = float(pyo.value(model.SDH[i]))

        flow = flow_out[i-1]
        area = pi * (d_inner[i] ** 2) / 4.0 if d_inner[i] > 0 else 0.0
        v = flow / 3600.0 / area if area > 0 else 0.0
        reynolds = v * d_inner[i] / (kv_dict[i] * 1e-6) if kv_dict[i] > 0 else 0.0

        # --- Head Loss Calculation for Output ---
        DR_frac = drag_red / 100.0
        f_val = f_expr[i]
        head_loss = f_val * (length[i]*1000.0/d_inner[i]) * (v**2/(2*9.81)) * (1-DR_frac)
        result[f"head_loss_{name}"] = head_loss

        D_out = d_inner[i] + 2 * thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        result[f"pipeline_flow_{name}"] = flow
        result[f"pump_flow_{name}"] = flow if is_pump else None
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost
        result[f"dra_cost_{name}"] = dra_cost
        result[f"drag_reduction_{name}"] = drag_red
        result[f"dra_ppm_{name}"] = ppm_val
        result[f"residual_head_{name}"] = res_head
        result[f"sdh_{name}"] = sdh_val
        result[f"maop_{name}"] = MAOP_head
        result[f"velocity_{name}"] = v
        result[f"reynolds_{name}"] = reynolds
        if is_pump:
            result[f"coef_A_{name}"] = float(Acoef[i])
            result[f"coef_B_{name}"] = float(Bcoef[i])
            result[f"coef_C_{name}"] = float(Ccoef[i])
            result[f"min_rpm_{name}"] = int(min_rpm[i])
            result[f"dol_{name}"] = int(max_rpm[i])

    # Terminal node
    term_name = terminal['name'].strip().lower().replace(' ', '_')
    result[f"pipeline_flow_{term_name}"] = flow_out[-1]
    result[f"pump_flow_{term_name}"] = None
    result[f"num_pumps_{term_name}"] = 0
    result[f"speed_{term_name}"] = 0.0
    result[f"efficiency_{term_name}"] = 0.0
    result[f"power_cost_{term_name}"] = 0.0
    result[f"dra_cost_{term_name}"] = 0.0
    result[f"drag_reduction_{term_name}"] = 0.0
    result[f"dra_ppm_{term_name}"] = 0.0
    result[f"residual_head_{term_name}"] = float(pyo.value(model.RH[N+1]))
    result[f"sdh_{term_name}"] = 0.0
    result[f"maop_{term_name}"] = 0.0
    result[f"velocity_{term_name}"] = 0.0
    result[f"reynolds_{term_name}"] = 0.0

    result['total_cost'] = float(pyo.value(model.Obj))

    return result
