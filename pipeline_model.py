import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

# --- DRA Curve Data Loading ---
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

def safe_value(val, fallback=1.0):
    """Returns val if it is not None/NaN/zero (unless zero is valid); else fallback."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return fallback
        return float(val)
    except Exception:
        return fallback

def validate_json(stations, terminal):
    # Check required keys
    required_keys = ['D', 't', 'SMYS', 'rough', 'L', 'elev', 'name']
    for idx, stn in enumerate(stations):
        for key in required_keys:
            if key not in stn or stn[key] in [None, ""]:
                raise ValueError(f"Station {idx+1} ('{stn.get('name','') or '[unnamed]'}') missing required key: {key}")
    if 'elev' not in terminal or terminal['elev'] in [None, ""]:
        raise ValueError("Terminal is missing 'elev'")
    if 'name' not in terminal or terminal['name'] in [None, ""]:
        raise ValueError("Terminal is missing 'name'")

def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None
):
    # ---------- INPUT VALIDATION ----------
    try:
        validate_json(stations, terminal)
    except Exception as e:
        return {"error": True, "message": str(e)}

    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    kv_dict = {i: safe_value(KV_list[i-1], 1.5) for i in range(1, N+1)}
    rho_dict = {i: safe_value(rho_list[i-1], 850.0) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)

    model.FLOW = pyo.Param(initialize=safe_value(FLOW))
    model.Rate_DRA = pyo.Param(initialize=safe_value(RateDRA))
    model.Price_HSD = pyo.Param(initialize=safe_value(Price_HSD))

    # Segment Flows
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = safe_value(stn.get('delivery', 0.0), 0.0)
        supply = safe_value(stn.get('supply', 0.0), 0.0)
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # Parameter Initialization
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    max_dr = {}
    peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72

    for i, stn in enumerate(stations, start=1):
        length[i] = safe_value(stn.get('L', 0.0), 1.0)
        if 'D' in stn:
            D_out = safe_value(stn['D'], 0.7)
            thickness[i] = safe_value(stn.get('t', default_t), default_t)
            d_inner[i] = D_out - 2*thickness[i]
        elif 'd' in stn:
            d_inner[i] = safe_value(stn['d'], 0.7)
            thickness[i] = safe_value(stn.get('t', default_t), default_t)
        else:
            d_inner[i] = 0.7
            thickness[i] = default_t
        roughness[i] = safe_value(stn.get('rough', default_e), default_e)
        smys[i] = safe_value(stn.get('SMYS', default_smys), default_smys)
        design_factor[i] = safe_value(stn.get('DF', default_df), default_df)
        elev[i] = safe_value(stn.get('elev', 0.0), 0.0)
        peaks_dict[i] = stn.get('peaks', [])
        has_pump = stn.get('is_pump', False)
        if has_pump:
            pump_indices.append(i)
            Acoef[i] = safe_value(stn.get('A', 0.0))
            Bcoef[i] = safe_value(stn.get('B', 0.0))
            Ccoef[i] = safe_value(stn.get('C', 0.0))
            Pcoef[i] = safe_value(stn.get('P', 0.0))
            Qcoef[i] = safe_value(stn.get('Q', 0.0))
            Rcoef[i] = safe_value(stn.get('R', 0.0))
            Scoef[i] = safe_value(stn.get('S', 0.0))
            Tcoef[i] = safe_value(stn.get('T', 0.0))
            min_rpm[i] = int(safe_value(stn.get('MinRPM', 0), 1))
            max_rpm[i] = int(safe_value(stn.get('DOL', 0), 1))
            if safe_value(stn.get('sfc', 0), 0) != 0:
                diesel_pumps.append(i)
                sfc[i] = safe_value(stn.get('sfc', 0.0))
            else:
                electric_pumps.append(i)
                elec_cost[i] = safe_value(stn.get('rate', 0.0), 9.0)
            max_dr[i] = safe_value(stn.get('max_dr', 0.0), 0.0)

    elev[N+1] = safe_value(terminal.get('elev', 0.0), 0.0)

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

    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)

    speed_min = {}; speed_max = {}
    for j in pump_indices:
        lo = max(1, (int(model.MinRPM[j]) + 9)//10) if model.MinRPM[j] else 1
        hi = max(lo, int(model.DOL[j])//10) if model.DOL[j] else lo
        speed_min[j], speed_max[j] = lo, hi
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (speed_min[j], speed_max[j]),
                        initialize=lambda m,j: (speed_min[j]+speed_max[j])//2)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])

    # Option B: DR is a continuous variable, not an Expression!
    model.DR = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals,
                       bounds=lambda m,j: (0, max_dr[j]), initialize=0)

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(safe_value(stations[0].get('min_residual', 50.0), 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        flow_m3s = safe_value(segment_flows[i], 1.0)/3600.0
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        kv = kv_dict[i]
        rho = rho_dict[i]
        if kv > 0:
            Re[i] = v[i]*d_inner[i]/(float(kv)*1e-6)
        else:
            Re[i] = 0.0
        if Re[i] > 0:
            if Re[i] < 4000:
                f[i] = 64.0/Re[i]
            else:
                arg = (roughness[i]/d_inner[i]/3.7) + (5.74/(Re[i]**0.9)) if d_inner[i] > 0 else 0
                f[i] = 0.25/(log10(arg)**2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}
    EFFP = {}

    for i in range(1, N+1):
        if i in pump_indices:
            DR_frac = model.DR[i] / 100.0
        else:
            DR_frac = 0.0
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2/(2*g)) * (1 - DR_frac) if d_inner[i] > 0 else 0.0
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = safe_value(peak.get('loc',0.0),0.0) * 1000.0
            elev_k = safe_value(peak.get('elev',0.0),0.0)
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2/(2*g)) * (1 - DR_frac) if d_inner[i] > 0 else 0.0
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)
        if i in pump_indices:
            pump_flow_i = safe_value(segment_flows[i], 1.0)
            # Correct affinity law: map flow to equivalent DOL reference
            N_val = model.N[i] if model.N[i] is not None else 1.0
            DOL_val = model.DOL[i] if model.DOL[i] is not None else 1.0
            Q_equiv = pump_flow_i * DOL_val / N_val if N_val != 0 else 1.0
            H_DOL = model.A[i] * Q_equiv**2 + model.B[i] * Q_equiv + model.C[i]
            TDH[i] = H_DOL * (N_val / DOL_val)**2 if DOL_val != 0 else H_DOL
            # Efficiency polynomial at equivalent flow (already correct)
            EFFP[i] = (
                model.Pcoef[i]*Q_equiv**4 + model.Qcoef[i]*Q_equiv**3 + model.Rcoef[i]*Q_equiv**2
                + model.Scoef[i]*Q_equiv + model.Tcoef[i]
            ) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N+1):
        kv = kv_dict[i]
        rho = rho_dict[i]
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        D_out = d_inner[i] + 2 * thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i] if D_out != 0 and rho_dict[i] != 0 else 1e5
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)
        peaks = peaks_dict[i]
        for peak in peaks:
            loc_km = safe_value(peak.get('loc',0.0),0.0)
            elev_k = safe_value(peak.get('elev',0.0),0.0)
            L_peak = loc_km*1000.0
            loss_no_dra = f[i] * (L_peak/d_inner[i]) * (v[i]**2/(2*g)) if d_inner[i] > 0 else 0.0
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    # ---- DRA PPM Piecewise and Cost Calculation ----
    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    model.dra_cost = pyo.Expression(model.pump_stations)
    for i in pump_indices:
        visc = kv_dict[i]
        dr_points, ppm_points = get_ppm_breakpoints(visc)
        # Add guard: if all values identical, only pass unique!
        dr_points_fixed, ppm_points_fixed = zip(*sorted(set(zip(dr_points, ppm_points)))) if len(dr_points) > 0 else ([0],[0])
        setattr(model, f'piecewise_dra_ppm_{i}',
            pyo.Piecewise(
                f'pw_dra_ppm_{i}',
                model.PPM[i], model.DR[i],
                pw_pts=dr_points_fixed,
                f_rule=ppm_points_fixed,
                pw_constr_type='EQ'
            )
        )
        dra_cost_expr = model.PPM[i] * (safe_value(segment_flows[i],1.0) * 1000.0 * 24.0 / 1e6) * RateDRA
        model.dra_cost[i] = dra_cost_expr

    # ---- Objective Function (Power/Fuel + DRA Cost) ----
    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        pump_flow_i = safe_value(segment_flows[i], 1.0)
        eff_val = float(EFFP[i]) if EFFP[i] != 0 else 0.01
        if i in pump_indices:
            power_kW = (rho_i * pump_flow_i * 9.81 * TDH[i] * model.NOP[i])/(3600.0*1000.0*eff_val*0.95) if eff_val != 0 else 0.0
        else:
            power_kW = 0.0
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i,9.0)
        else:
            fuel_per_kWh = (sfc.get(i,0.0)*1.34102)/820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        dra_cost = model.dra_cost[i]
        total_cost += power_cost + dra_cost
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # --- Solve ---
    try:
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
    except Exception as ex:
        return {"error": True, "message": f"Solver/NEOS failed: {ex}"}

    # --- Results Section (robust, never fails) ---
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0

        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i])) if model.NOP[i].value is not None else 0
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 and model.N[i]() is not None else 0.0
            eff = float(pyo.value(EFFP[i])*100.0) if num_pumps > 0 else 0.0
            dra_ppm = float(pyo.value(model.PPM[i])) if model.PPM[i].value is not None else 0.0
            dra_cost_i = float(pyo.value(model.dra_cost[i])) if model.dra_cost[i]() is not None else 0.0
        else:
            num_pumps = 0; speed_rpm = 0.0; eff = 0.0; dra_ppm = 0.0; dra_cost_i = 0.0

        if i in pump_indices and num_pumps > 0:
            rho_i = rho_dict[i]
            eff_val = float(pyo.value(EFFP[i])) if pyo.value(EFFP[i]) != 0 else 0.01
            power_kW = (rho_i * pump_flow * 9.81 * float(pyo.value(TDH[i])) * num_pumps)/(3600.0*1000.0*eff_val*0.95) if eff_val != 0 else 0.0
            if i in electric_pumps:
                rate = elec_cost.get(i,9.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i,0.0)
                fuel_per_kWh = (sfc_val*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        else:
            power_cost = 0.0

        drag_red = float(pyo.value(model.DR[i])) if i in pump_indices and model.DR[i].value is not None else 0.0
        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i])))) if model.SDH[i].value is not None and model.RH[i+1].value is not None else 0.0
        res_head = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
        velocity = v[i]; reynolds = Re[i]; fric = f[i]

        result[f"pipeline_flow_{name}"] = outflow
        result[f"pipeline_flow_in_{name}"] = inflow
        result[f"pump_flow_{name}"] = pump_flow
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
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
            result[f"coef_A_{name}"] = float(pyo.value(model.A[i]))
            result[f"coef_B_{name}"] = float(pyo.value(model.B[i]))
            result[f"coef_C_{name}"] = float(pyo.value(model.C[i]))
            result[f"dol_{name}"]    = float(pyo.value(model.DOL[i]))
            result[f"min_rpm_{name}"]= float(pyo.value(model.MinRPM[i]))

    term = terminal.get('name','terminal').strip().lower().replace(' ','_')
    result.update({
        f"pipeline_flow_{term}": segment_flows[-1],
        f"pipeline_flow_in_{term}": segment_flows[-2],
        f"pump_flow_{term}": 0.0,
        f"speed_{term}": 0.0,
        f"num_pumps_{term}": 0,
        f"efficiency_{term}": 0.0,
        f"power_cost_{term}": 0.0,
        f"dra_cost_{term}": 0.0,
        f"dra_ppm_{term}": 0.0,
        f"drag_reduction_{term}": 0.0,
        f"head_loss_{term}": 0.0,
        f"velocity_{term}": 0.0,
        f"reynolds_{term}": 0.0,
        f"friction_{term}": 0.0,
        f"sdh_{term}": 0.0,
        f"residual_head_{term}": float(pyo.value(model.RH[N+1])) if model.RH[N+1].value is not None else 0.0,
    })
    result['total_cost'] = float(pyo.value(model.Obj)) if model.Obj() is not None else 0.0
    result["error"] = False
    return result
