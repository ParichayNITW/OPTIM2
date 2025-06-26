# --- HEAD: Imports & DRA Setup ---
import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'your_email@example.com')

DRA_CSV_FILES = {
    10: "10 cst.csv", 15: "15 cst.csv", 20: "20 cst.csv", 25: "25 cst.csv",
    30: "30 cst.csv", 35: "35 cst.csv", 40: "40 cst.csv"
}
DRA_CURVE_DATA = {}
for cst, fname in DRA_CSV_FILES.items():
    if os.path.exists(fname):
        DRA_CURVE_DATA[cst] = pd.read_csv(fname)
    else:
        DRA_CURVE_DATA[cst] = None

def _ppm_from_df(df, dr):
    if df is None: return 0
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    return np.interp(dr, x, y)

def get_ppm_for_dr(visc, dr, dra_curve_data=DRA_CURVE_DATA):
    cst_list = sorted([c for c in dra_curve_data.keys() if dra_curve_data[c] is not None])
    visc = float(visc)
    if not cst_list: return 0
    if visc <= cst_list[0]:
        return _ppm_from_df(dra_curve_data[cst_list[0]], dr)
    elif visc >= cst_list[-1]:
        return _ppm_from_df(dra_curve_data[cst_list[-1]], dr)
    else:
        lower = max([c for c in cst_list if c <= visc])
        upper = min([c for c in cst_list if c >= visc])
        ppm_lower = _ppm_from_df(dra_curve_data[lower], dr)
        ppm_upper = _ppm_from_df(dra_curve_data[upper], dr)
        return np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None):
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

    # --- Segment Flow Logic ---
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        segment_flows.append(segment_flows[-1] - delivery + supply)

    pump_flows = []
    for idx, stn in enumerate(stations):
        if stn.get('is_pump', False):
            pump_flows.append(segment_flows[idx+1])
        else:
            pump_flows.append(0.0)

    # --- Geometry and Design Parameters ---
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    max_dr = {}; peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72

    for i, stn in enumerate(stations, start=1):
        length[i] = stn.get('L', 0.0)
        thickness[i] = stn.get('t', default_t)
        d_inner[i] = stn.get('d', stn.get('D', 0.7) - 2*thickness[i])
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
        peaks_dict[i] = stn.get('peaks', [])
        if stn.get('is_pump', False):
            pump_indices.append(i)
            Acoef[i] = stn.get('A', 0.0); Bcoef[i] = stn.get('B', 0.0); Ccoef[i] = stn.get('C', 0.0)
            Pcoef[i] = stn.get('P', 0.0); Qcoef[i] = stn.get('Q', 0.0); Rcoef[i] = stn.get('R', 0.0)
            Scoef[i] = stn.get('S', 0.0); Tcoef[i] = stn.get('T', 0.0)
            min_rpm[i] = stn.get('MinRPM', 0); max_rpm[i] = stn.get('DOL', 0)
            if stn.get('sfc', 0): diesel_pumps.append(i); sfc[i] = stn.get('sfc', 0.0)
            else: electric_pumps.append(i); elec_cost[i] = stn.get('rate', 0.0)
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

    # NOP and speed variables
    def nop_bounds(m, j): return (1 if j==1 else 0, stations[j-1].get('max_pumps', 2))
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=nop_bounds, initialize=1)

    speed_min = {j: max(1, (min_rpm[j]+9)//10) for j in pump_indices}
    speed_max = {j: max(speed_min[j], max_rpm[j]//10) for j in pump_indices}
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (speed_min[j], speed_max[j]),
                        initialize=lambda m,j: (speed_min[j]+speed_max[j])//2)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])

    dr_max = {j: int(max_dr[j]//10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                         bounds=lambda m,j: (0, dr_max[j]), initialize=0)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.DR_u[j])

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2): model.RH[j].setlb(50.0)

    # --- Hydraulic Calculations ---
    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        flow_m3s = float(segment_flows[i])/3600.0
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        Re[i] = v[i]*d_inner[i]/(kv_dict[i]*1e-6) if kv_dict[i] > 0 else 0.0
        if Re[i] > 0:
            if Re[i] < 4000: f[i] = 64.0 / Re[i]
            else:
                arg = (roughness[i]/d_inner[i]/3.7) + (5.74/(Re[i]**0.9))
                f[i] = 0.25/(log10(arg)**2) if arg > 0 else 0.01
        else: f[i] = 0.01
    # --- Variables ---
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}; EFFP = {}

    for i in range(1, N+1):
        DR_frac = model.DR[i] / 100.0 if i in pump_indices else 0.0
        DH_fric = f[i] * (length[i]*1000.0 / d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
        elev_diff = model.z[i+1] - model.z[i]
        model.sdh_constraint.add(model.SDH[i] >= model.RH[i+1] + elev_diff + DH_fric)

        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
            model.sdh_constraint.add(model.SDH[i] >= (elev_k - model.z[i]) + DH_peak + 50.0)

        if i in pump_indices:
            pump_flow_i = float(segment_flows[i])
            TDH[i] = (model.A[i]*pump_flow_i**2 + model.B[i]*pump_flow_i + model.C[i]) * ((model.N[i]/model.DOL[i])**2)
            flow_eq = pump_flow_i * model.DOL[i] / pyo.maximize(model.N[i], 1)  # prevent divide by zero
            EFFP[i] = (
                model.Pcoef[i]*flow_eq**4 + model.Qcoef[i]*flow_eq**3 + model.Rcoef[i]*flow_eq**2
                + model.Scoef[i]*flow_eq + model.Tcoef[i]
            ) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # --- Constraints ---
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    maop_dict = {}

    for i in range(1, N+1):
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])

        D_out = d_inner[i] + 2*thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i]*0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)

        for peak in peaks_dict[i]:
            elev_k = peak['elev']
            L_peak = peak['loc'] * 1000.0
            loss_no_dra = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g))
            expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra if i in pump_indices \
                else model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    # --- Objective: Power + Real DRA Cost ---
    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        visc_i = kv_dict[i]
        pump_flow_i = float(segment_flows[i])
        TDH_val = TDH[i]
        eff_val = EFFP[i]
        eff_val_safe = eff_val if isinstance(eff_val, float) and eff_val > 0 else 0.4

        power_kW = (rho_i * pump_flow_i * 9.81 * TDH_val * model.NOP[i]) / (3600.0 * 1000.0 * eff_val_safe * 0.95)

        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i, 0.0)
        else:
            fuel_per_kWh = (sfc.get(i, 0.0) * 1.34102) / 820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD

        drag = model.DR[i]
        ppm = get_ppm_for_dr(visc_i, pyo.value(drag))
        dra_cost = ppm * (pump_flow_i * 24 * 1000 / 1e6) * RateDRA

        total_cost += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)
    # --- Identify Running Pumps ---
    running_pumps = []
    for i in pump_indices:
        if int(pyo.value(model.NOP[i])) > 0:
            running_pumps.append(i)

    # --- DRA Injection Mapping ---
    dra_map = [None] * (N+1)
    cumlen = [0]
    for stn in stations:
        cumlen.append(cumlen[-1] + stn.get("L", 0.0))

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

    # --- FINAL OUTPUT ---
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0
        visc_i = kv_dict[i]

        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
            eff = float(pyo.value(EFFP[i]) * 100.0) if num_pumps > 0 else 0.0
            rho_i = rho_dict[i]
            TDH_val = float(pyo.value(TDH[i]))
            eff_safe = float(pyo.value(EFFP[i])) if float(pyo.value(EFFP[i])) > 0 else 0.4
            power_kW = (rho_i * pump_flow * 9.81 * TDH_val * num_pumps) / (3600.0 * 1000.0 * eff_safe * 0.95)

            if i in electric_pumps:
                rate = elec_cost.get(i, 0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i, 0.0)
                fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD

            drag_red = float(pyo.value(model.DR[i]))
            ppm = get_ppm_for_dr(visc_i, drag_red)
            dra_cost = ppm * (pump_flow * 24 * 1000 / 1e6) * RateDRA
        else:
            num_pumps = 0
            speed_rpm = 0.0
            eff = 0.0
            power_cost = 0.0
            dra_cost = 0.0
            drag_red = 0.0

        res_head = float(pyo.value(model.RH[i]))
        head_loss = float(pyo.value(model.SDH[i]) - (model.RH[i+1] + (model.z[i+1] - model.z[i])))

        result[f"pipeline_flow_{name}"] = outflow
        result[f"pipeline_flow_in_{name}"] = inflow
        result[f"pump_flow_{name}"] = pump_flow
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost
        result[f"dra_cost_{name}"] = dra_cost
        result[f"drag_reduction_{name}"] = drag_red
        result[f"head_loss_{name}"] = head_loss
        result[f"residual_head_{name}"] = res_head
        result[f"velocity_{name}"] = v[i]
        result[f"reynolds_{name}"] = Re[i]
        result[f"friction_{name}"] = f[i]
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i]))
        result[f"maop_{name}"] = maop_dict[i]
        if i in pump_indices:
            result[f"coef_A_{name}"] = float(pyo.value(model.A[i]))
            result[f"coef_B_{name}"] = float(pyo.value(model.B[i]))
            result[f"coef_C_{name}"] = float(pyo.value(model.C[i]))
            result[f"dol_{name}"] = float(pyo.value(model.DOL[i]))
            result[f"min_rpm_{name}"] = float(pyo.value(model.MinRPM[i]))

    term = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
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
