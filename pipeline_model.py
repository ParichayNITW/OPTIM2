import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

# ---------------------------
# 1. DRA Data Fitting Section
# ---------------------------
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

DRA_CURVE_DATA = {}
MM_PARAMS = {}
for cst, fname in DRA_CSV_FILES.items():
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        DRA_CURVE_DATA[cst] = df
        x = df['PPM'].values
        y = df['%Drag Reduction'].values
        # Safe initial guess, bounded to avoid extreme outputs
        popt, _ = curve_fit(mm_func, x, y, bounds=([5,0.1], [100, 200]))
        MM_PARAMS[cst] = (popt[0], popt[1])
    else:
        DRA_CURVE_DATA[cst] = None
        MM_PARAMS[cst] = (0.0, 1.0)  # Fallback

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

def ppm_from_dr(visc, dr_target):
    dr_max, km = interpolate_mm_params(visc)
    if dr_target <= 0:
        return 0.0
    if dr_target >= dr_max - 1e-6:
        return 999.0
    ppm = (km * dr_target) / (dr_max - dr_target)
    return max(0.0, ppm)

# --------------------------------
# 2. Pipeline Optimization Section
# --------------------------------
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

    # Compute segment flows (accounting for delivery/supply)
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # Station-specific parameters
    pump_indices = []
    electric_pumps = []
    diesel_pumps = []
    max_dr = {}
    elec_cost = {}
    sfc = {}
    min_rpm = {}
    max_rpm = {}
    A, B, C, P, Qc, R, S, T = {}, {}, {}, {}, {}, {}, {}, {}
    for i, stn in enumerate(stations, start=1):
        if stn.get('is_pump', False):
            pump_indices.append(i)
            max_dr[i] = stn.get('max_dr', 60.0)
            min_rpm[i] = stn.get('MinRPM', 1000.0)
            max_rpm[i] = stn.get('DOL', 1500.0)
            if stn.get('sfc', 0) not in (None, 0):
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc', 150.0)
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 9.0)
            # Pump curve coefficients
            A[i] = stn.get('A', 0.0)
            B[i] = stn.get('B', 0.0)
            C[i] = stn.get('C', 0.0)
            P[i] = stn.get('P', 0.0)
            Qc[i] = stn.get('Q', 0.0)
            R[i] = stn.get('R', 0.0)
            S[i] = stn.get('S', 0.0)
            T[i] = stn.get('T', 0.0)

    # Pyomo variables
    model.pump_stations = pyo.Set(initialize=pump_indices)
    model.DR = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals,
                       bounds=lambda m, j: (0, max_dr.get(j, 60.0)), initialize=0)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=(0, 6), initialize=1)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(terminal.get('min_residual', 50.0))

    # Hydraulics: Friction factor (Swamee–Jain, turbulent only)
    def friction_factor(Re, eps, D):
        if Re < 4000: return 0.0  # Not valid, but should be ok
        return 0.25 / (log10(eps/(3.7*D) + 5.74/Re**0.9))**2

    # Friction and residual head constraints
    model.HL = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # head loss per segment
    model.VEL = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.RE = pyo.Var(model.I, domain=pyo.NonNegativeReals)

    def velocity_rule(m, i):
        Q = float(segment_flows[i-1]) / 3600.0  # m3/s
        D = stations[i-1]['D']
        area = pi * (D**2) / 4.0
        return m.VEL[i] == Q / area
    model.vel_con = pyo.Constraint(model.I, rule=velocity_rule)

    def reynolds_rule(m, i):
        v = m.VEL[i]
        D = stations[i-1]['D']
        kv = m.KV[i]
        return m.RE[i] == v * D / (kv * 1e-6)
    model.rey_con = pyo.Constraint(model.I, rule=reynolds_rule)

    def headloss_rule(m, i):
        L = stations[i-1]['L'] * 1000.0
        D = stations[i-1]['D']
        rough = stations[i-1]['rough']
        v = m.VEL[i]
        re = m.RE[i]
        eps = rough
        f = 0.25 / (pyo.log10(eps/(3.7*D) + 5.74/(re**0.9)))**2 if re.value and re.value > 4000 else 0.018  # fallback
        # Apply DRA effect
        key = i if i in pump_indices else None
        dr_pct = m.DR[key] if key else 0.0
        f_dra = f * (1.0 - dr_pct/100.0)
        hl = f_dra * (L / D) * v**2 / (2 * 9.81)
        return m.HL[i] == hl
    model.hl_con = pyo.Constraint(model.I, rule=headloss_rule)

    # Nodewise energy conservation: Residual head across all segments
    def energy_rule(m, i):
        return m.RH[i+1] == m.RH[i] - m.HL[i]
    model.energy_con = pyo.Constraint(pyo.RangeSet(1,N), rule=energy_rule)

    # Power/fuel/DRA cost computation and objective
    def total_cost_rule(m):
        total_cost = 0.0
        for i in pump_indices:
            viscosity = kv_dict[i]
            flow = float(segment_flows[i-1])
            dr_val = m.DR[i]
            dr_val_py = pyo.value(dr_val)
            ppm = ppm_from_dr(viscosity, dr_val_py)
            dra_injection = ppm * flow * 1000 * 24 / 1e6  # L/day
            dra_cost = dra_injection * m.Rate_DRA
            # Power calculation (based on user-provided pump curves and efficiency)
            # Calculate Q (flow) at this station, head = model.HL[i], efficiency (assume 70% min)
            head = pyo.value(m.HL[i])
            Q_pump = flow  # m3/hr
            eff = max(0.7, 0.7)  # You may replace this with polynomial fits if available
            rho = rho_dict[i]
            P_kW = (rho * Q_pump * 9.81 * head) / (3600 * 1000 * eff)
            # Decide electric/diesel
            if i in electric_pumps:
                power_cost = P_kW * 24 * elec_cost[i]
            else:
                # Diesel cost (approx): SFC (gm/bhp-hr), Price_HSD (INR/L)
                bhp = P_kW / 0.746
                fuel_used_per_hr = bhp * sfc[i] / 1000  # kg/hr
                fuel_used_L_per_hr = fuel_used_per_hr / 0.84  # diesel density
                power_cost = fuel_used_L_per_hr * 24 * m.Price_HSD
            total_cost += dra_cost + power_cost
        return total_cost

    model.Obj = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # Solve
    solver_manager = SolverManagerFactory('neos')
    results = solver_manager.solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    # Output all results in a summary dict
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        viscosity = kv_dict[i]
        dr_val = float(pyo.value(model.DR[i])) if i in pump_indices else 0.0
        ppm = ppm_from_dr(viscosity, dr_val) if i in pump_indices else 0.0
        flow = float(segment_flows[i-1])
        dra_injection = ppm * flow * 1000 * 24 / 1e6
        dra_cost = dra_injection * RateDRA
        result[f"station_{name}_drag_reduction"] = dr_val
        result[f"station_{name}_dra_ppm"] = ppm
        result[f"station_{name}_dra_cost"] = dra_cost
        result[f"station_{name}_flow"] = flow
        result[f"station_{name}_viscosity"] = viscosity
        result[f"station_{name}_head_loss"] = pyo.value(model.HL[i])
        result[f"station_{name}_velocity"] = pyo.value(model.VEL[i])
        result[f"station_{name}_reynolds"] = pyo.value(model.RE[i])
        result[f"station_{name}_residual_head"] = pyo.value(model.RH[i])
    # Total cost
    result["total_cost"] = pyo.value(model.Obj)
    return result
