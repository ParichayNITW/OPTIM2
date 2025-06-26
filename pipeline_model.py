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

def _ppm_from_df(df, dr):
    if df is None:
        return 0
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    if dr <= x[0]:
        return y[0]
    elif dr >= x[-1]:
        return y[-1]
    else:
        return np.interp(dr, x, y)

def get_ppm_for_dr(visc, dr, dra_curve_data=DRA_CURVE_DATA):
    cst_list = sorted([c for c in dra_curve_data.keys() if dra_curve_data[c] is not None])
    visc = float(visc)
    if not cst_list:
        return 0
    if visc <= cst_list[0]:
        df = dra_curve_data[cst_list[0]]
        return _ppm_from_df(df, dr)
    elif visc >= cst_list[-1]:
        df = dra_curve_data[cst_list[-1]]
        return _ppm_from_df(df, dr)
    else:
        lower = max([c for c in cst_list if c <= visc])
        upper = min([c for c in cst_list if c >= visc])
        df_lower = dra_curve_data[lower]
        df_upper = dra_curve_data[upper]
        ppm_lower = _ppm_from_df(df_lower, dr)
        ppm_upper = _ppm_from_df(df_upper, dr)
        return np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])

def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None
):
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

    # ---- CORRECTED SEGMENT FLOW LOGIC ----
    # segment_flows[i]: flow after delivery/supply at station i (i=0 for inlet, i=1 after A, ..., i=N after last station)
    segment_flows = [float(FLOW)]  # segment_flows[0]: initial mainline flow at inlet
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # --- PUMP FLOW LOGIC (always pump_flow = pipeline_flow at pump station, after delivery/supply at that station) ---
    pump_flows = []
    for idx, stn in enumerate(stations):
        if stn.get('is_pump', False):
            pump_flows.append(segment_flows[idx+1])  # Use flow after delivery/supply at station
        else:
            pump_flows.append(0.0)

    # --- Parameter Initialization ---
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

    dr_max = {j: int(max_dr.get(j, 40)//10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                         bounds=lambda m,j: (0, dr_max[j]), initialize=0)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.DR_u[j])

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        # Use segment_flows[i] for all calculations (flow after delivery/supply at i)
        flow_m3s = float(segment_flows[i])/3600.0
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
                arg = (roughness[i]/d_inner[i]/3.7) + (5.74/(Re[i]**0.9))
                f[i] = 0.25/(log10(arg)**2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}
    EFFP = {}

    for i in range(1, N+1):
        # DRA only at pump stations
        if i in pump_indices:
            DR_frac = model.DR[i] / 100.0
        else:
            DR_frac = 0.0
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2/(2*g)) * (1 - DR_frac)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2/(2*g)) * (1 - DR_frac)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)
        # Pump equations: use pump_flows!
        if i in pump_indices:
            pump_flow_i = float(segment_flows[i])  # always use after delivery
            TDH[i] = (model.A[i]*pump_flow_i**2 + model.B[i]*pump_flow_i + model.C[i]) * ((model.N[i]/model.DOL[i])**2)
            flow_eq = pump_flow_i * model.DOL[i]/model.N[i]
            EFFP[i] = (
                model.Pcoef[i]*flow_eq**4 + model.Qcoef[i]*flow_eq**3 + model.Rcoef[i]*flow_eq**2
                + model.Scoef[i]*flow_eq + model.Tcoef[i]
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
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)
        peaks = peaks_dict[i]
        for peak in peaks:
            loc_km = peak['loc']
            elev_k = peak['elev']
            L_peak = loc_km*1000.0
            loss_no_dra = f[i] * (L_peak/d_inner[i]) * (v[i]**2/(2*g))
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        visc_i = kv_dict[i]
        pump_flow_i = float(segment_flows[i])
        power_kW = (rho_i * pump_flow_i * 9.81 * TDH[i] * model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i, 0.0)
        else:
            fuel_per_kWh = (sfc.get(i, 0.0) * 1.34102) / 820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD

        ppm_expr = 0.1425 * visc_i + 0.9294 * model.DR[i] + 0.00289 * visc_i * model.DR[i] - 16.5529
        dra_cost = ppm_expr * (pump_flow_i * 24 * 1000 / 1e6) * RateDRA

        total_cost += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    running_pumps = []
    for i in pump_indices:
        if int(pyo.value(model.NOP[i])) > 0:
            running_pumps.append(i)
    dra_map = [None]*(N+1)
    cumlen = [0]
    for stn in stations:
        cumlen.append(cumlen[-1] + stn["L"])
    for idx in range(1, N+1):
        dra_station = None
        for upidx in reversed(running_pumps):
            if idx <= upidx:
                continue
            start_km = cumlen[upidx-1]
            seg_start = cumlen[idx-1]
            next_running = min([i for i in running_pumps if i > upidx], default=N+1)
            limit_km = min(cumlen[next_running-1] if next_running <= N else cumlen[-1], start_km + 400)
            if seg_start < limit_km:
                dra_station = upidx
                break
        dra_map[idx] = dra_station

    # === RESULTS SECTION ===
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]  # before delivery/supply at station i
        outflow = segment_flows[i]   # after delivery/supply at station i (USE THIS FOR CALC & REPORT)
        pump_flow = outflow if stn.get('is_pump', False) else 0.0
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
            eff = float(pyo.value(EFFP[i])*100.0) if num_pumps > 0 else 0.0
        else:
            num_pumps = 0; speed_rpm = 0.0; eff = 0.0

        if i in pump_indices and num_pumps > 0:
            rho_i = rho_dict[i]
            power_kW = (rho_i * pump_flow * 9.81 * float(pyo.value(TDH[i])) * num_pumps)/(3600.0*1000.0*float(pyo.value(EFFP[i]))*0.95)
            if i in electric_pumps:
                rate = elec_cost.get(i,0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i,0.0)
                fuel_per_kWh = (sfc_val*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        else:
            power_cost = 0.0

        if i in pump_indices:
            drag_red = float(pyo.value(model.DR[i]))
        else:
            drag_red = 0.0

        visc_i = kv_dict[i]
        ppm_expr = 0.1425 * visc_i + 0.9294 * pyo.value(model.DR[i]) + 0.00289 * visc_i * pyo.value(model.DR[i]) - 16.5529
        dra_cost = ppm_expr * (pump_flow_i * 24 * 1000 / 1e6) * RateDRA

        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i]))))
        res_head = float(pyo.value(model.RH[i]))
        velocity = v[i]; reynolds = Re[i]; fric = f[i]

        result[f"pipeline_flow_{name}"] = outflow  # ALWAYS USE PIPELINE FLOW AFTER DELIVERY
        result[f"pipeline_flow_in_{name}"] = inflow  # (optional: before delivery, for reference)
        result[f"pump_flow_{name}"] = pump_flow
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost
        result[f"dra_cost_{name}"] = dra_cost
        result[f"drag_reduction_{name}"] = drag_red
        result[f"head_loss_{name}"] = head_loss
        result[f"residual_head_{name}"] = res_head
        result[f"velocity_{name}"] = velocity
        result[f"reynolds_{name}"] = reynolds
        result[f"friction_{name}"] = fric
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i]))
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
        f"drag_reduction_{term}": 0.0,
        f"head_loss_{term}": 0.0,
        f"velocity_{term}": 0.0,
        f"reynolds_{term}": 0.0,
        f"friction_{term}": 0.0,
        f"sdh_{term}": 0.0,
        f"residual_head_{term}": float(pyo.value(model.RH[N+1])),
    })
    result['total_cost'] = float(pyo.value(model.Obj))
    return result
