import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

# Ensure your NEOS email is set in the environment
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

# Helper to interpolate PPM from a single viscosity curve
def _ppm_from_df(df, dr):
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    if dr <= x[0]:
        return y[0]
    if dr >= x[-1]:
        return y[-1]
    return float(np.interp(dr, x, y))

# Python-side PPM lookup using two curves interpolation
# visc in cSt, dr in %
def get_ppm_for_dr(visc, dr, dra_curve_data=DRA_CURVE_DATA):
    cst_list = sorted([c for c, df in dra_curve_data.items() if df is not None])
    if not cst_list:
        return 0.0
    if visc <= cst_list[0]:
        return _ppm_from_df(dra_curve_data[cst_list[0]], dr)
    if visc >= cst_list[-1]:
        return _ppm_from_df(dra_curve_data[cst_list[-1]], dr)
    # interpolate between nearest viscosity curves
    lower = max(c for c in cst_list if c <= visc)
    upper = min(c for c in cst_list if c >= visc)
    df_low = dra_curve_data[lower]
    df_high = dra_curve_data[upper]
    ppm_low = _ppm_from_df(df_low, dr)
    ppm_high = _ppm_from_df(df_high, dr)
    return float(np.interp(visc, [lower, upper], [ppm_low, ppm_high]))


def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None
):
    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    # Params
    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}       # kinematic viscosity cSt
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}     # density kg/m3
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)   # cost per ppm per MLD
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Compute segment flows before building model
    segment_flows = [float(FLOW)]
    for stn in stations:
        prev = segment_flows[-1]
        delivery = float(stn.get('delivery', 0.0))
        supply   = float(stn.get('supply',   0.0))
        segment_flows.append(prev - delivery + supply)

    # Identify pump stations and extract pump coeffs
    pump_indices = []
    diesel_pumps = []
    electric_pumps = []
    sfc = {}
    elec_cost = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}; max_dr = {}
    peaks_dict = {}

    for i, stn in enumerate(stations, start=1):
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
            if stn.get('sfc', 0) > 0:
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc', 0.0)
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 0.0)
            max_dr[i] = stn.get('max_dr', 0.0)

    # Hydraulic & pump params
    length = {i: stn.get('L', 0.0) for i, stn in enumerate(stations, start=1)}
    thickness = {}
    d_inner  = {}
    roughness = {}
    smys = {}
    design_factor = {}
    elev = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    
    for i, stn in enumerate(stations, start=1):
        thickness[i] = stn.get('t', default_t)
        if 'D' in stn:
            d_inner[i] = stn['D'] - 2*thickness[i]
        else:
            d_inner[i] = stn.get('d', 0.7)
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
    elev[N+1] = terminal.get('elev', 0.0)

    # Add params to model
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
        model.DOL    = pyo.Param(model.pump_stations, initialize=max_rpm)

    # Decision vars
    def nop_bounds(m,j):
        lb = 1 if j==1 else 0
        ub = int(stations[j-1].get('max_pumps', 2))
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations,
                        domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds,
                        initialize=1)
    
    speed_min = {}
    speed_max = {}
    for j in pump_indices:
        lo = max(1, (int(model.MinRPM[j]) + 9)//10)
        hi = max(lo, int(model.DOL[j])//10)
        speed_min[j], speed_max[j] = lo, hi
    model.N_u = pyo.Var(model.pump_stations,
                        domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (speed_min[j], speed_max[j]),
                        initialize=lambda m,j: (speed_min[j]+speed_max[j])//2)
    model.N   = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])

    dr_max_int = {j: int(max_dr[j]//10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations,
                         domain=pyo.NonNegativeIntegers,
                         bounds=lambda m,j: (0, dr_max_int[j]),
                         initialize=0)
    model.DR   = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.DR_u[j])

    # Residual head vars
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    # --- Piecewise mapping: DR->PPM at each pump station ---
    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    for i in pump_indices:
        visc = kv_dict[i]
        steps = dr_max_int[i]
        dr_pts = [10*k for k in range(0, steps+1)]
        ppm_pts = [get_ppm_for_dr(visc, dr) for dr in dr_pts]
        pyo.Piecewise(
            model.PPM[i], model.DR[i],
            pw_pts=list(zip(dr_pts, ppm_pts)),
            pw_constr_type='EQ'
        )

    # --- Hydraulic calculations (flows, Re, friction, SDH) ---
    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        flow_m3s = float(segment_flows[i]) / 3600.0
        area    = pi * d_inner[i]**2 / 4.0
        v[i]    = flow_m3s/area if area>0 else 0.0
        kv_val  = kv_dict[i]*1e-6
        Re[i]   = v[i]*d_inner[i]/kv_val if kv_val>0 else 0.0
        if Re[i] <= 0:
            f[i] = 0.0
        elif Re[i] < 4000:
            f[i] = 64.0/Re[i]
        else:
            arg = (roughness[i]/d_inner[i]/3.7) + 5.74/(Re[i]**0.9)
            f[i] = 0.25/(log10(arg)**2) if arg>0 else 0.0

    # SDH constraints & pump curves
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}; EFFP = {}
    for i in range(1, N+1):
        dr_frac = (model.DR[i]/100.0) if i in pump_indices else 0.0
        base_loss = f[i]*((length[i]*1000.0)/d_inner[i])*(v[i]**2/(2*g))*(1-dr_frac)
        expr_next = model.RH[i+1] + (model.z[i+1]-model.z[i]) + base_loss
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            Lp = peak['loc']*1000.0
            loss_peak = f[i]*(Lp/d_inner[i])*(v[i]**2/(2*g))*(1-dr_frac)
            expr_peak = (peak['elev']-model.z[i]) + loss_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)
        if i in pump_indices:
            Qp = float(segment_flows[i])
            TDH[i] = (model.A[i]*Qp**2 + model.B[i]*Qp + model.C[i])*(model.N[i]/model.DOL[i])**2
            EFFP[i] = (
                model.Pcoef[i]*(Qp/model.N[i]*model.DOL[i])**4 +
                model.Qcoef[i]*(Qp/model.N[i]*model.DOL[i])**3 +
                model.Rcoef[i]*(Qp/model.N[i]*model.DOL[i])**2 +
                model.Scoef[i]*(Qp/model.N[i]*model.DOL[i]) +
                model.Tcoef[i]
            )/100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # Pressure & head balance
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        MAOP = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/(d_inner[i]+2*thickness[i]))*10000.0/rho_dict[i]
        model.pressure_limit.add(model.SDH[i] <= MAOP)

    # Objective: minimize fuel + electricity + DRA cost
    total = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        Qp = float(segment_flows[i])
        power_kW = (rho_i*Qp*9.81*TDH[i]*model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps:
            cost_p = power_kW*24.0*elec_cost.get(i,0.0)
        else:
            fuel_kWh = (sfc.get(i,0.0)*1.34102)/820.0
            cost_p = power_kW*24.0*fuel_kWh*model.Price_HSD
        # DRA cost: PPM (from piecewise) * MLD * RateDRA
        mld = Qp*1000.0*24.0/1e6
        cost_dra = model.PPM[i] * mld * model.Rate_DRA
        total += cost_p + cost_dra
    model.Obj = pyo.Objective(expr=total, sense=pyo.minimize)

    # Solve with global MINLP solver
    results = SolverManagerFactory('neos').solve(model, solver='couenne', tee=False)
    model.solutions.load_from(results)

    # Build results dict
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow  = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0
        num_p = int(pyo.value(model.NOP[i])) if i in pump_indices else 0
        sp_rpm = float(pyo.value(model.N[i])) if num_p>0 and i in pump_indices else 0.0
        eff_pct = float(pyo.value(EFFP[i])*100.0) if num_p>0 and i in pump_indices else 0.0
        # power cost
        if num_p>0 and i in pump_indices:
            rho_i = rho_dict[i]
            pow_kW = (rho_i*pump_flow*9.81*float(pyo.value(TDH[i]))*num_p)/(3600.0*1000.0*float(pyo.value(EFFP[i]))*0.95)
            if i in electric_pumps:
                p_cost = pow_kW*24.0*elec_cost.get(i,0.0)
            else:
                fuel_kWh = (sfc.get(i,0.0)*1.34102)/820.0
                p_cost = pow_kW*24.0*fuel_kWh*float(model.Price_HSD)
        else:
            p_cost = 0.0
        # dra cost
        ppm_val = float(pyo.value(model.PPM[i])) if i in pump_indices else 0.0
        mld = pump_flow*1000.0*24.0/1e6
        dra_cost_val = ppm_val * mld * float(model.Rate_DRA) if i in pump_indices else 0.0

        # assemble
        result.update({
            f"pipeline_flow_{name}": outflow,
            f"pipeline_flow_in_{name}": inflow,
            f"pump_flow_{name}": pump_flow,
            f"num_pumps_{name}": num_p,
            f"speed_{name}": sp_rpm,
            f"efficiency_{name}": eff_pct,
            f"power_cost_{name}": p_cost,
            f"dra_cost_{name}": dra_cost_val,
            f"drag_reduction_{name}": float(pyo.value(model.DR[i])) if i in pump_indices else 0.0,
            f"sdh_{name}": float(pyo.value(model.SDH[i])),  # and other outputs as needed
        })

    # add terminal
    term = terminal.get('name','terminal').strip().lower().replace(' ', '_')
    result[f"pipeline_flow_{term}"]    = segment_flows[-1]
    result[f"pipeline_flow_in_{term}"] = segment_flows[-2]
    result['total_cost'] = float(pyo.value(model.Obj))
    return result

# End of pipeline_model.py
