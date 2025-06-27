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
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    if dr <= x[0]:
        return y[0]
    elif dr >= x[-1]:
        return y[-1]
    else:
        return float(np.interp(dr, x, y))

def get_ppm_for_dr(visc, dr):
    cst_list = sorted([c for c, df in DRA_CURVE_DATA.items() if df is not None])
    if not cst_list:
        return 0.0
    if visc <= cst_list[0]:
        return _ppm_from_df(DRA_CURVE_DATA[cst_list[0]], dr)
    if visc >= cst_list[-1]:
        return _ppm_from_df(DRA_CURVE_DATA[cst_list[-1]], dr)
    lower = max(c for c in cst_list if c <= visc)
    upper = min(c for c in cst_list if c >= visc)
    ppm_lo = _ppm_from_df(DRA_CURVE_DATA[lower], dr)
    ppm_hi = _ppm_from_df(DRA_CURVE_DATA[upper], dr)
    return float(np.interp(visc, [lower, upper], [ppm_lo, ppm_hi]))

def solve_pipeline(
    stations, terminal,
    FLOW, KV_list, rho_list,
    RateDRA, Price_HSD,
    linefill_dict=None
):
    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    kv_dict  = {i: float(KV_list[i-1]) for i in model.I}
    rho_dict = {i: float(rho_list[i-1]) for i in model.I}
    model.KV  = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW      = pyo.Param(initialize=FLOW)
    model.Rate_DRA  = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # segment flows after each station
    segment_flows = [FLOW]
    for stn in stations:
        segment_flows.append(segment_flows[-1] - float(stn.get('delivery', 0.0)) + float(stn.get('supply', 0.0)))

    # identify pumps and parameters
    pump_indices = []
    diesel_pumps = []
    electric_pumps = []
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    sfc = {}; elec_cost = {}
    max_dr = {}
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
            if stn.get('sfc', 0):
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc')
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 0.0)
            max_dr[i] = stn.get('max_dr', 40.0)

    # geometry and materials
    length = {}; d_inner = {}; roughness = {}; thickness = {}
    smys = {}; design_factor = {}; elev = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    for i, stn in enumerate(stations, start=1):
        D_out = stn.get('D', stn.get('d', 0.7))
        t = stn.get('t', default_t)
        thickness[i] = t
        d_inner[i] = D_out - 2*t
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
    elev[N+1] = terminal.get('elev', 0.0)

    model.L  = pyo.Param(model.I, initialize=length)
    model.d  = pyo.Param(model.I, initialize=d_inner)
    model.e  = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF   = pyo.Param(model.I, initialize=design_factor)
    model.z    = pyo.Param(model.Nodes, initialize=elev)

    model.pump_stations = pyo.Set(initialize=pump_indices)
    if pump_indices:
        model.A     = pyo.Param(model.pump_stations, initialize=Acoef)
        model.B     = pyo.Param(model.pump_stations, initialize=Bcoef)
        model.C     = pyo.Param(model.pump_stations, initialize=Ccoef)
        model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef)
        model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef)
        model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef)
        model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef)
        model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef)

    # decision variables
    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)

    min_u = {j: max(1, (stations[j-1].get('MinRPM', 1000)+9)//10) for j in pump_indices}
    max_u = {j: max(min_u[j], stations[j-1].get('DOL', 1000)//10) for j in pump_indices}
    model.N_u = pyo.Var(model.pump_stations,
                        bounds=lambda m,j: (min_u[j], max_u[j]),
                        domain=pyo.NonNegativeIntegers,
                        initialize=lambda m,j: (min_u[j]+max_u[j])//2)
    model.N   = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])

    dr_max_int = {j: int(max_dr.get(j,40)//10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations,
                         bounds=lambda m,j: (0, dr_max_int[j]),
                         domain=pyo.NonNegativeIntegers,
                         initialize=0)
    model.DR   = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.DR_u[j])

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual',50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    # hydraulics: velocity, Re, friction
    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        Qm3s = segment_flows[i]/3600.0
        area = pi*(d_inner[i]**2)/4.0
        v[i] = Qm3s/area if area>0 else 0.0
        kv = kv_dict[i]
        Re[i] = v[i]*d_inner[i]/(kv*1e-6) if kv>0 else 0.0
        if Re[i]>0 and Re[i]<4000:
            f[i] = 64.0/Re[i]
        elif Re[i]>=4000:
            arg = (roughness[i]/d_inner[i]/3.7)+(5.74/(Re[i]**0.9))
            f[i] = 0.25/(log10(arg)**2) if arg>0 else 0.0
        else:
            f[i] = 0.0

    # SDH and pump head/efficiency
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}; EFFP = {}
    for i in range(1, N+1):
        DR_frac = model.DR[i]/100.0 if i in pump_indices else 0.0
        DH_pipe = f[i]*((length[i]*1000.0)/d_inner[i])*(v[i]**2/(2*g))*(1-DR_frac)
        model.sdh_constraint.add(
            model.SDH[i] >= model.RH[i+1] + (model.z[i+1]-model.z[i]) + DH_pipe
        )
        for peak in peaks_dict[i]:
            Lp = peak['loc']*1000.0
            DHp = f[i]*(Lp/d_inner[i])*(v[i]**2/(2*g))*(1-DR_frac)
            model.sdh_constraint.add(
                model.SDH[i] >= (peak['elev']-model.z[i]) + DHp + 50.0
            )
        if i in pump_indices:
            Qp = segment_flows[i]/3600.0
            TDH[i] = (model.A[i]*Qp**2 + model.B[i]*Qp + model.C[i]) * ((model.N[i]/stations[i-1].get('DOL',model.N[i]))**2)
            ratio = Qp * stations[i-1].get('DOL',model.N[i]) / model.N[i]
            EFFP[i] = (
                model.Pcoef[i]*ratio**4 + model.Qcoef[i]*ratio**3 +
                model.Rcoef[i]*ratio**2 + model.Scoef[i]*ratio + model.Tcoef[i]
            )/100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        MAOP_h = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/(d_inner[i]+2*thickness[i]))*10000.0/rho_dict[i]
        model.pressure_limit.add(model.SDH[i] <= MAOP_h)
        for peak in peaks_dict[i]:
            Lp = peak['loc']*1000.0
            loss0 = f[i]*(Lp/d_inner[i])*(v[i]**2/(2*g))
            expr = model.RH[i] + (TDH[i]*model.NOP[i] if i in pump_indices else 0) - (peak['elev']-model.z[i]) - loss0
            model.peak_limit.add(expr >= 50.0)

    # --- DRA piecewise mapping ---
    dr_points = sorted({pt for df in DRA_CURVE_DATA.values() if df is not None for pt in df['%Drag Reduction'].values})
    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    for j in model.pump_stations:
        visc_j = float(stations[j-1].get('viscosity', dr_points[0]))
        ppm_points = [get_ppm_for_dr(visc_j, dr) for dr in dr_points]
        pyo.Piecewise(
            model.PPM[j], model.DR[j],
            pw_pts=list(zip(dr_points, ppm_points)),
            pw_constr_type='EQ'
        )

    # --- Objective: fuel + electricity + DRA cost ---
    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        Qm3h = segment_flows[i]
        power_kW = (rho_i * Qm3h * 9.81 * TDH[i] * model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost[i]
        else:
            fuel_per_kWh = (sfc[i]*1.34102)/820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        vol_mld = Qm3h * 24.0 / 1000.0
        dra_cost = model.PPM[i] * vol_mld * model.Rate_DRA
        total_cost += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # --- Solve with global MINLP solver ---
    results = SolverManagerFactory('neos').solve(model, solver='couenne', tee=False)
    model.solutions.load_from(results)

    # --- Collect results ---
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0
        num_pumps = int(pyo.value(model.NOP[i])) if i in pump_indices else 0
        speed_rpm = float(pyo.value(model.N[i])) if num_pumps>0 else 0.0
        eff_pct = float(pyo.value(EFFP[i])*100.0) if num_pumps>0 else 0.0
        if i in pump_indices and num_pumps>0:
            power_kW = (rho_dict[i] * pump_flow * 9.81 * float(pyo.value(TDH[i])) * num_pumps)/(3600.0*1000.0*float(pyo.value(EFFP[i]))*0.95)
            power_cost = power_kW * 24.0 * (elec_cost[i] if i in electric_pumps else ((sfc[i]*1.34102)/820.0 * Price_HSD))
        else:
            power_cost = 0.0
        drag_red = float(pyo.value(model.DR[i])) if i in pump_indices else 0.0
        dra_cost = float(pyo.value(model.PPM[i] * (pump_flow*24.0/1000.0) * model.Rate_DRA)) if i in pump_indices else 0.0
        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i]))))
        res_head = float(pyo.value(model.RH[i]))
        velocity = v[i]; reynolds = Re[i]; friction = f[i]
        sdh_val = float(pyo.value(model.SDH[i]))
        maop = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/(d_inner[i]+2*thickness[i]))*10000.0/rho_dict[i]
        coefA = float(pyo.value(model.A[i])) if i in pump_indices else 0.0
        coefB = float(pyo.value(model.B[i])) if i in pump_indices else 0.0
        coefC = float(pyo.value(model.C[i])) if i in pump_indices else 0.0
        dol    = float(stations[i-1].get('DOL', 0)) if i in pump_indices else 0.0
        minrpm = float(stations[i-1].get('MinRPM', 0)) if i in pump_indices else 0.0

        result[f"pipeline_flow_{name}"]    = outflow
        result[f"pipeline_flow_in_{name}"] = inflow
        result[f"pump_flow_{name}"]        = pump_flow
        result[f"num_pumps_{name}"]        = num_pumps
        result[f"speed_{name}"]            = speed_rpm
        result[f"efficiency_{name}"]       = eff_pct
        result[f"power_cost_{name}"]       = power_cost
        result[f"dra_cost_{name}"]         = dra_cost
        result[f"drag_reduction_{name}"]   = drag_red
        result[f"head_loss_{name}"]        = head_loss
        result[f"residual_head_{name}"]    = res_head
        result[f"velocity_{name}"]         = velocity
        result[f"reynolds_{name}"]         = reynolds
        result[f"friction_{name}"]         = friction
        result[f"sdh_{name}"]              = sdh_val
        result[f"maop_{name}"]             = maop
        result[f"coef_A_{name}"]           = coefA
        result[f"coef_B_{name}"]           = coefB
        result[f"coef_C_{name}"]           = coefC
        result[f"dol_{name}"]              = dol
        result[f"min_rpm_{name}"]          = minrpm

    term = terminal.get('name','terminal').strip().lower().replace(' ', '_')
    result[f"pipeline_flow_{term}"]     = segment_flows[-1]
    result[f"pipeline_flow_in_{term}"]  = segment_flows[-2]
    result[f"pump_flow_{term}"]         = 0.0
    result[f"num_pumps_{term}"]         = 0
    result[f"speed_{term}"]             = 0.0
    result[f"efficiency_{term}"]        = 0.0
    result[f"power_cost_{term}"]        = 0.0
    result[f"dra_cost_{term}"]          = 0.0
    result[f"drag_reduction_{term}"]    = 0.0
    result[f"head_loss_{term}"]         = 0.0
    result[f"residual_head_{term}"]     = float(pyo.value(model.RH[N+1]))
    result[f"velocity_{term}"]          = 0.0
    result[f"reynolds_{term}"]          = 0.0
    result[f"friction_{term}"]          = 0.0
    result[f"sdh_{term}"]               = 0.0

    result['total_cost'] = float(pyo.value(model.Obj))
    return result
