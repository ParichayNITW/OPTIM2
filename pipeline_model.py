import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import pi, log10

# Ensure NEOS email is set
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'your_email@example.com')

def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    N = len(stations)
    model = pyo.ConcreteModel()
    model.seg = pyo.RangeSet(1, N)
    model.node = pyo.RangeSet(1, N+1)

    model.FLOW = pyo.Param(initialize=FLOW)
    model.KV = pyo.Param(initialize=KV)
    model.rho = pyo.Param(initialize=rho)
    model.Rate_DRA = pyo.Param(initialize=Rate_DRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Input dictionaries
    L, D, t, eps, z, smys, DF = {},{},{},{},{},{},{}
    is_pump, maxP, minRPM, maxRPM, rate, power_type, SFC = {},{},{},{},{},{},{}
    A,B,C,P,Q,R,S,T = {},{},{},{},{},{},{},{}

    for i, stn in enumerate(stations, start=1):
        L[i] = stn['L']
        D[i] = stn['D']
        t[i] = stn['t']
        eps[i] = stn['rough']
        z[i] = stn['elev']
        smys[i] = stn.get('SMYS', 52000)
        DF[i] = stn.get('DF', 0.72)
        is_pump[i] = stn.get('is_pump', False)
        if is_pump[i]:
            maxP[i] = stn['max_pumps']
            minRPM[i] = stn['MinRPM']
            maxRPM[i] = stn['DOL']
            rate[i] = stn['rate']
            power_type[i] = stn['power_type']
            SFC[i] = stn.get('SFC', 250)
            A[i],B[i],C[i] = stn['A'],stn['B'],stn['C']
            P[i],Q[i],R[i],S[i],T[i] = stn['P'],stn['Q'],stn['R'],stn['S'],stn['T']

    z[N+1] = terminal['elev']
    min_res = terminal['min_residual']
    init_res = stations[0].get('residual_head', 50)

    model.L = pyo.Param(model.seg, initialize=L)
    model.D = pyo.Param(model.seg, initialize=D)
    model.t = pyo.Param(model.seg, initialize=t)
    model.eps = pyo.Param(model.seg, initialize=eps)
    model.z = pyo.Param(model.node, initialize=z)
    model.SMYS = pyo.Param(model.seg, initialize=smys)
    model.DF = pyo.Param(model.seg, initialize=DF)

    pumps = [i for i in model.seg if is_pump.get(i, False)]
    model.pumps = pyo.Set(initialize=pumps)

    model.maxP = pyo.Param(model.pumps, initialize=maxP)
    model.minRPM = pyo.Param(model.pumps, initialize=minRPM)
    model.maxRPM = pyo.Param(model.pumps, initialize=maxRPM)
    model.A = pyo.Param(model.pumps, initialize=A)
    model.B = pyo.Param(model.pumps, initialize=B)
    model.C = pyo.Param(model.pumps, initialize=C)
    model.Pp = pyo.Param(model.pumps, initialize=P)
    model.Qp = pyo.Param(model.pumps, initialize=Q)
    model.Rp = pyo.Param(model.pumps, initialize=R)
    model.Sp = pyo.Param(model.pumps, initialize=S)
    model.Tp = pyo.Param(model.pumps, initialize=T)

    model.RH = pyo.Var(model.node, domain=pyo.NonNegativeReals)
    model.RH[1].fix(init_res)
    model.RH[N+1].fix(min_res)
    for i in range(2, N+1):
        model.RH[i].setlb(50)

    model.NOP = pyo.Var(model.seg, domain=pyo.NonNegativeIntegers, bounds=lambda m,i: (0, m.maxP[i]) if i in m.pumps else (0,0))
    model.Nu = pyo.Var(model.seg, domain=pyo.NonNegativeIntegers, bounds=lambda m,i: (int((m.minRPM[i]+9)//10), int(m.maxRPM[i]//10)) if i in m.pumps else (0,0))
    model.N = pyo.Expression(model.seg, rule=lambda m,i: 10*m.Nu[i])
    model.DRu = pyo.Var(model.seg, domain=pyo.NonNegativeIntegers, bounds=(0,4))
    model.DR = pyo.Expression(model.seg, rule=lambda m,i: 10*m.DRu[i])

    model.constraints = pyo.ConstraintList()
    power_costs = []; dra_costs = []

    for i in model.seg:
        d_in = pyo.value(model.D[i]) - 2*pyo.value(model.t[i])
        if d_in <= 0:
            raise ValueError(f"Invalid internal diameter at segment {i}")
        A_flow = pi*d_in**2/4
        v = pyo.value(model.FLOW)/3600/A_flow
        Re = v*d_in/(pyo.value(model.KV)*1e-6)
        if Re <= 4000:
            raise ValueError(f"Re = {Re} is too low for Swamee–Jain at segment {i}")
        arg = model.eps[i]/(3.7*d_in) + 5.74/(Re**0.9)
        f = 0.25/(log10(arg)**2)
        SH = model.RH[i+1] + (model.z[i+1] - model.z[i])
        HL = f*(pyo.value(model.L[i])*1000/d_in)*(v**2/(2*9.81))*(1 - model.DR[i]/100)

        if i in model.pumps:
            PH = (model.A[i]*model.FLOW**2 + model.B[i]*model.FLOW + model.C[i]) * (model.N[i]/model.maxRPM[i])**2
            model.constraints.add(model.RH[i] + PH*model.NOP[i] >= SH + HL)
            MAOP = (2*model.t[i]*model.SMYS[i]*0.070307*model.DF[i]/model.D[i])*10000/model.rho
            model.constraints.add(model.RH[i] + PH*model.NOP[i] <= MAOP)

            flow_eq = model.FLOW*model.maxRPM[i]/model.N[i]
            eff = model.Pp[i]*flow_eq**4 + model.Qp[i]*flow_eq**3 + model.Rp[i]*flow_eq**2 + model.Sp[i]*flow_eq + model.Tp[i]
            eff = pyo.Expression(expr=max(eff/100, 0.01)) if pyo.value(model.NOP[i]) > 0 else pyo.Expression(expr=0.0)

            if power_type[i] == 'Grid':
                base = model.rho*model.FLOW*9.81*PH*model.NOP[i]/(3600*1000*eff*0.95)
                power_cost = base*24*rate[i] if pyo.value(model.NOP[i]) > 0 else 0.0
            else:
                base = model.rho*model.FLOW*9.81*PH*model.NOP[i]/(3600*1000*eff*0.95)
                bhp_hr = SFC[i]*1.34102
                fuel = bhp_hr/820
                power_cost = base*24*fuel*Price_HSD if pyo.value(model.NOP[i]) > 0 else 0.0
        else:
            model.constraints.add(model.RH[i] >= SH + HL)
            MAOP = (2*model.t[i]*model.SMYS[i]*0.070307*model.DF[i]/model.D[i])*10000/model.rho
            model.constraints.add(model.RH[i] <= MAOP)
            power_cost = 0.0

        dra_cost = (model.DR[i]/1e6)*model.FLOW*24*1000*Rate_DRA
        power_costs.append(power_cost)
        dra_costs.append(dra_cost)

    model.Obj = pyo.Objective(expr=sum(power_costs)+sum(dra_costs), sense=pyo.minimize)

    solver = SolverManagerFactory('neos')
    results = solver.solve(model, solver='couenne', tee=False)
    model.solutions.load_from(results)

    return model
