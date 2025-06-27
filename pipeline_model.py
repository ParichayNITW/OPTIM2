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

    # Parameters
    kv_dict = {i: float(KV_list[i-1]) for i in model.I}
    rho_dict = {i: float(rho_list[i-1]) for i in model.I}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Segment flows
    segment_flows = [FLOW]
    for stn in stations:
        segment_flows.append(
            segment_flows[-1] - float(stn.get('delivery', 0.0)) + float(stn.get('supply', 0.0))
        )

    # Pump station data
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

    # Geometry
    length = {}; d_inner = {}; roughness = {}; thickness = {}
    smys = {}; design_factor = {}; elev = {}
    default_t, default_e, default_smys, default_df = 0.007, 0.00004, 52000, 0.72
    for i, stn in enumerate(stations, start=1):
        length[i] = float(stn.get('L', 0.0))
        D_out = stn.get('D', stn.get('d', 0.7))
        t = stn.get('t', default_t)
        thickness[i] = t
        d_inner[i] = D_out - 2*t
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
    elev[N+1] = terminal.get('elev', 0.0)

    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)

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

    # Decision variables
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (1 if j==1 else 0, stations[j-1].get('max_pumps',2)), initialize=1)
    model.N_u = pyo.Var(model.pump_stations,
                        bounds=lambda m,j: (max(1,(stations[j-1].get('MinRPM',1000)+9)//10), stations[j-1].get('DOL',1000)//10),
                        domain=pyo.NonNegativeIntegers,
                        initialize=lambda m,j: max(1,(stations[j-1].get('MinRPM',1000)+9)//10))
    model.N   = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                         bounds=lambda m,j: (0,int(max_dr.get(j,40)//10)), initialize=0)
    model.DR_pct = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals,
                           bounds=lambda m,j: (0,max_dr.get(j,40)), initialize=0)
    model.link_dr = pyo.Constraint(model.pump_stations, rule=lambda m,j: m.DR_pct[j] == 10*m.DR_u[j])

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual',50.0))
    for j in range(2,N+2): model.RH[j].setlb(50.0)

    # Hydraulics
    g=9.81; v={} ; Re={} ; f={}
    for i in range(1,N+1):
        Qm3s=segment_flows[i]/3600.0; area=pi*(d_inner[i]**2)/4.0
        v[i]=Qm3s/area if area>0 else 0; Re[i]=v[i]*d_inner[i]/(kv_dict[i]*1e-6) if kv_dict[i]>0 else 0
        if 0<Re[i]<4000: f[i]=64.0/Re[i]
        elif Re[i]>=4000:
            arg=(roughness[i]/d_inner[i]/3.7)+(5.74/(Re[i]**0.9))
            f[i]=0.25/(log10(arg)**2) if arg>0 else 0
        else: f[i]=0

    # SDH & Pump
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    con_sdh = pyo.ConstraintList()
    TDH={}; EFFP={}
    for i in range(1,N+1):
        DRf=model.DR_pct[i]/100.0 if i in pump_indices else 0.0
        DH_pipe = f[i]*((length[i]*1000.0)/d_inner[i])*(v[i]**2/(2*g))*(1-DRf)
        con_sdh.add(model.SDH[i] >= model.RH[i+1] + (model.z[i+1]-model.z[i]) + DH_pipe)
        if i in pump_indices:
            Qp=segment_flows[i]/3600.0
            TDH[i] = (model.A[i]*Qp**2 + model.B[i]*Qp + model.C[i]) * ((model.N[i]/stations[i-1].get('DOL',1000))**2)
            ratio = Qp * stations[i-1].get('DOL',1000) / model.N[i]
            EFFP[i] = (model.Pcoef[i]*ratio**4 + model.Qcoef[i]*ratio**3 + model.Rcoef[i]*ratio**2 + model.Scoef[i]*ratio + model.Tcoef[i]) / 100.0
        else:
            TDH[i]=0.0; EFFP[i]=1.0

    # Piecewise DRA -> PPM
    dr_pts = sorted({pt for df in DRA_CURVE_DATA.values() if df is not None for pt in df['%Drag Reduction'].values})
    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    for j in model.pump_stations:
        visc = float(stations[j-1].get('viscosity',dr_pts[0]))
        ppm_vals=[get_ppm_for_dr(visc,dr) for dr in dr_pts]
        pyo.Piecewise(model.PPM[j], model.DR_pct[j], pw_pts=list(zip(dr_pts,ppm_vals)), pw_constr_type='EQ')

    # Objective
    total_cost=0
    for i in pump_indices:
        rho_i=rho_dict[i]; Qh=segment_flows[i]
        pw_kW=(rho_i*Qh*9.81*TDH[i]*model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps: pc= pw_kW*24*elec_cost[i]
        else: fc=(sfc[i]*1.34102)/820.0; pc=pw_kW*24*fc*Price_HSD
        vol=Qh*24/1000.0; dc=model.PPM[i]*vol*model.Rate_DRA
        total_cost+=pc+dc
    model.Obj=pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # Solve
    res=SolverManagerFactory('neos').solve(model, solver='couenne', tee=False)
    model.solutions.load_from(res)

    # Results
    out={}
    for i,stn in enumerate(stations, start=1):
        nm=stn['name'].lower().replace(' ','_')
        out[f"pipeline_flow_{nm}"]=segment_flows[i]
        out[f"power_cost_{nm}"]=float(pc if i in pump_indices else 0)
        out[f"dra_cost_{nm}"]=float(model.PPM[i]*segment_flows[i]*24/1000*model.Rate_DRA) if i in pump_indices else 0
    out['total_cost']=float(pyo.value(model.Obj))
    return out
