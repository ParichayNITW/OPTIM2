import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

# 1. Michaelis–Menten DRA Fit (run at module load)
def mm_func(ppm, dr_max, km):
    return dr_max * ppm / (km + ppm)

DRA_CSV_FILES = {
    10: "10 cst.csv",
    15: "15 cst.csv",
    20: "20 cst.csv",
    25: "25 cst.csv",
    30: "30 cst.csv",
    35: "35 cst.csv",
    40: "40 cst.csv"
}
MM_PARAMS = {}
for cst, fname in DRA_CSV_FILES.items():
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        x = df['PPM'].values
        y = df['%Drag Reduction'].values
        popt, _ = curve_fit(mm_func, x, y, bounds=([1, 0.01], [100, 200]))
        MM_PARAMS[cst] = (popt[0], popt[1])  # dr_max, km
    else:
        MM_PARAMS[cst] = (0.0, 1.0)

MM_CST_LIST = sorted(MM_PARAMS.keys())

def interpolate_mm_params(visc):
    viscs = np.array(MM_CST_LIST)
    dr_maxs = np.array([MM_PARAMS[v][0] for v in viscs])
    kms = np.array([MM_PARAMS[v][1] for v in viscs])
    if visc <= viscs[0]:
        return dr_maxs[0], kms[0]
    elif visc >= viscs[-1]:
        return dr_maxs[-1], kms[-1]
    else:
        dr_max = np.interp(visc, viscs, dr_maxs)
        km = np.interp(visc, viscs, kms)
        return dr_max, km

def ppm_from_dr(visc, dr):
    dr_max, km = interpolate_mm_params(visc)
    if dr <= 0:
        return 0.0
    if dr >= dr_max - 1e-6:
        return 999.0
    return max(0.0, (km * dr) / (dr_max - dr))

# 2. Pipeline Optimizer
def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None
):
    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    # Parameters
    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Pipeline Geometry and Parameters
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

    # --- DYNAMIC HYDRAULICS ---
    # Pipeline velocity (m/s) as Pyomo expression
    def velocity_rule(m, i):
        flow_m3hr = m.FLOW
        area = pi * (m.d[i] ** 2) / 4.0
        return m.VEL[i] == flow_m3hr / 3600.0 / area
    model.VEL = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.vel_con = pyo.Constraint(model.I, rule=velocity_rule)

    # Reynolds number as expression
    def reynolds_rule(m, i):
        return m.RE[i] == m.VEL[i] * m.d[i] / (m.KV[i] * 1e-6)
    model.RE = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.rey_con = pyo.Constraint(model.I, rule=reynolds_rule)

    # Friction factor (Swamee–Jain for turbulent flow)
    def friction_rule(m, i):
        eps = m.e[i]
        D = m.d[i]
        Re = m.RE[i]
        # For turbulent regime
        arg = eps/(3.7*D) + 5.74/(Re**0.9 + 1e-6)
        return m.f[i] == 0.25 / (pyo.log10(arg + 1e-10))**2
    model.f = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.fr_con = pyo.Constraint(model.I, rule=friction_rule)

    # Effective friction with DRA
    def f_eff_rule(m, i):
        dr_frac = m.DR[i]/100.0 if i in pump_indices else 0.0
        return m.f_eff[i] == m.f[i] * (1 - dr_frac)
    model.f_eff = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.fe_con = pyo.Constraint(model.I, rule=f_eff_rule)

    # Head loss as dynamic Pyomo variable
    def headloss_rule(m, i):
        L = m.L[i] * 1000.0
        D = m.d[i]
        v = m.VEL[i]
        return m.HL[i] == m.f_eff[i] * (L/D) * v**2 / (2*g)
    model.HL = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.hl_con = pyo.Constraint(model.I, rule=headloss_rule)

    # Nodewise head/energy conservation (including head loss and pump head)
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}
    EFFP = {}

    for i in range(1, N+1):
        if i in pump_indices:
            DR_frac = model.DR[i] / 100.0
        else:
            DR_frac = 0.0
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + model.HL[i]
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            HL_peak = model.f_eff[i] * (L_peak / model.d[i]) * model.VEL[i]**2 / (2*g)
            expr_peak = (elev_k - model.z[i]) + HL_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)
        if i in pump_indices:
            pump_flow_i = float(FLOW)  # Mainline flow at station, can replace if segment-wise
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
            loss_no_dra = model.f[i] * (L_peak/model.d[i]) * (model.VEL[i]**2/(2*g))
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    # ---- COST FUNCTION: INCLUDES MM DRA COST & POWER/FUEL ----
    def total_cost_rule(m):
        total_cost = 0
        for i in pump_indices:
            viscosity = kv_dict[i]
            pump_flow_i = float(FLOW)
            drag_red = pyo.value(m.DR[i])
            ppm = ppm_from_dr(viscosity, drag_red)
            dra_injection = ppm * pump_flow_i * 1000.0 * 24.0 / 1e6
            dra_cost = dra_injection * RateDRA
            # Power/fuel
            rho_i = rho_dict[i]
            num_pumps = max(1, int(pyo.value(m.NOP[i])))
            tdh_val = float(pyo.value(TDH[i]))
            eff_val = max(0.7, float(pyo.value(EFFP[i])))
            power_kW = (rho_i * pump_flow_i * 9.81 * tdh_val * num_pumps)/(3600.0*1000.0*eff_val*0.95)
            if i in electric_pumps:
                power_cost = power_kW * 24.0 * elec_cost.get(i,0.0)
            else:
                fuel_per_kWh = (sfc.get(i,0.0)*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
            total_cost += power_cost + dra_cost
        return total_cost
    model.Obj = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # ---- SOLVE ----
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    # === RESULTS ===
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        pump_flow = float(FLOW)
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
            eff = float(pyo.value(EFFP[i])*100.0) if num_pumps > 0 else 0.0
            drag_red = float(pyo.value(model.DR[i]))
            ppm = ppm_from_dr(kv_dict[i], drag_red)
            dra_injection = ppm * pump_flow * 1000.0 * 24.0 / 1e6
            dra_cost = dra_injection * RateDRA
        else:
            num_pumps = 0; speed_rpm = 0.0; eff = 0.0; drag_red = 0.0; ppm = 0.0; dra_cost = 0.0

        result[f"pump_flow_{name}"] = pump_flow
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"dra_cost_{name}"] = dra_cost
        result[f"dra_ppm_{name}"] = ppm
        result[f"drag_reduction_{name}"] = drag_red
        result[f"velocity_{name}"] = float(pyo.value(model.VEL[i]))
        result[f"reynolds_{name}"] = float(pyo.value(model.RE[i]))
        result[f"friction_{name}"] = float(pyo.value(model.f[i]))
        result[f"f_eff_{name}"] = float(pyo.value(model.f_eff[i]))
        result[f"head_loss_{name}"] = float(pyo.value(model.HL[i]))
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i]))
        result[f"residual_head_{name}"] = float(pyo.value(model.RH[i]))

    result["total_cost"] = float(pyo.value(model.Obj))
    return result
