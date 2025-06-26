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
