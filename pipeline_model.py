import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# Ensure NEOS email is set (replace with your email in deployment)
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    model = pyo.ConcreteModel()

    # Set parameters
    model.FLOW = pyo.Param(initialize=FLOW)
    model.KV = pyo.Param(initialize=KV)
    model.rho = pyo.Param(initialize=rho)
    model.RateDRA = pyo.Param(initialize=RateDRA)
    model.PriceHSD = pyo.Param(initialize=Price_HSD)

    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N + 1)

    # Pipeline geometry and data
    length, d_inner, roughness, elev = {}, {}, {}, {}
    thickness, SMYS, DF = {}, {}, {}
    A, B, C, P, Q, R, S, T = {}, {}, {}, {}, {}, {}, {}, {}
    min_rpm, dol_rpm, max_pumps = {}, {}, {}
    sfc, rate = {}, {}
    max_dr = {}
    pump_stations = []

    for i, stn in enumerate(stations, start=1):
        length[i] = stn['L']
        thickness[i] = stn['t']
        d_inner[i] = stn['D'] - 2 * stn['t']
        roughness[i] = stn['rough']
        elev[i] = stn['elev']
        SMYS[i] = stn.get('SMYS', 52000)
        DF[i] = stn.get('DF', 0.72)

        if stn.get('is_pump', False):
            pump_stations.append(i)
            A[i], B[i], C[i] = stn['A'], stn['B'], stn['C']
            P[i], Q[i], R[i], S[i], T[i] = stn['P'], stn['Q'], stn['R'], stn['S'], stn['T']
            min_rpm[i] = stn['MinRPM']
            dol_rpm[i] = stn['DOL']
            max_pumps[i] = stn['max_pumps']
            sfc[i] = stn.get('SFC', 0)
            rate[i] = stn.get('rate', 0)
            max_dr[i] = stn['max_dr']

    elev[N+1] = terminal['elev']

    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.z = pyo.Param(model.Nodes, initialize=elev)

    model.pump_stations = pyo.Set(initialize=pump_stations)
    model.A = pyo.Param(model.pump_stations, initialize=A)
    model.B = pyo.Param(model.pump_stations, initialize=B)
    model.C = pyo.Param(model.pump_stations, initialize=C)
    model.P = pyo.Param(model.pump_stations, initialize=P)
    model.Q = pyo.Param(model.pump_stations, initialize=Q)
    model.R = pyo.Param(model.pump_stations, initialize=R)
    model.S = pyo.Param(model.pump_stations, initialize=S)
    model.T = pyo.Param(model.pump_stations, initialize=T)
    model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm)
    model.DOL = pyo.Param(model.pump_stations, initialize=dol_rpm)

    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=lambda m,j: (0, max_pumps[j]), initialize=0)
    model.Nu = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=lambda m,j: (int(min_rpm[j]/10), int(dol_rpm[j]/10)), initialize=lambda m,j: int((min_rpm[j] + dol_rpm[j]) / 20))
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10 * m.Nu[j])

    model.DRu = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=lambda m,j: (0, int(max_dr[j]/10)), initialize=0)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10 * m.DRu[j])

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, bounds=lambda m,j: (50.0, None))
    model.RH[1].fix(stations[0]['min_residual'])
    model.RH[N+1].fix(terminal['min_residual'])

    model.constraints = pyo.ConstraintList()

    g = 9.81
    v, Re, f = {}, {}, {}
    flow_m3s = FLOW / 3600
    for i in model.I:
        A_flow = pi * (d_inner[i] ** 2) / 4
        v[i] = flow_m3s / A_flow
        Re[i] = v[i] * d_inner[i] / (KV * 1e-6)
        if Re[i] < 4000:
            f[i] = 64 / Re[i]
        else:
            f[i] = 0.25 / log10((roughness[i]/d_inner[i]/3.7 + 5.74/(Re[i]**0.9)))**2

    TDH, EFF = {}, {}
    for i in model.I:
        dh = f[i] * (length[i]*1000/d_inner[i]) * (v[i]**2/(2*g))
        static = model.z[i+1] - model.z[i]
        SDHR = model.RH[i+1] + static + dh

        if i in pump_stations:
            TDH[i] = (model.A[i]*FLOW**2 + model.B[i]*FLOW + model.C[i]) * (model.N[i]/model.DOL[i])**2
            eq_flow = FLOW * model.DOL[i]/model.N[i]
            EFF[i] = (model.P[i]*eq_flow**4 + model.Q[i]*eq_flow**3 + model.R[i]*eq_flow**2 + model.S[i]*eq_flow + model.T[i])/100
            model.constraints.add(model.RH[i] + TDH[i]*model.NOP[i] >= SDHR)
        else:
            model.constraints.add(model.RH[i] >= SDHR)

    # Objective
    cost_expr = 0
    for i in pump_stations:
        power_kw = (rho * FLOW * g * TDH[i] * model.NOP[i]) / (3600 * 1000 * EFF[i] * 0.95)
        dra_cost = (model.DR[i] / 100) * (FLOW * 24 * 1000 / 1e6) * RateDRA
        if sfc[i] > 0:
            fuel_kwh = (sfc[i]*1.34102) / 820
            fuel_cost = power_kw * 24 * fuel_kwh * Price_HSD
        else:
            fuel_cost = power_kw * 24 * rate[i]
        cost_expr += fuel_cost + dra_cost

    model.obj = pyo.Objective(expr=cost_expr, sense=pyo.minimize)

    solver = SolverManagerFactory('neos')
    results = solver.solve(model, solver='couenne')
    model.solutions.load_from(results)

    out = {}
    for i, stn in enumerate(stations, start=1):
        key = stn['name'].strip().lower().replace(" ", "_")
        out[f"residual_head_{key}"] = pyo.value(model.RH[i])
        if i in pump_stations:
            out[f"num_pumps_{key}"] = pyo.value(model.NOP[i])
            out[f"speed_{key}"] = pyo.value(model.N[i])
            out[f"efficiency_{key}"] = pyo.value(EFF[i]) * 100
            out[f"drag_reduction_{key}"] = pyo.value(model.DR[i])

    term_key = terminal['name'].strip().lower().replace(" ", "_")
    out[f"residual_head_{term_key}"] = pyo.value(model.RH[N+1])
    out["total_cost"] = pyo.value(model.obj)
    return out
