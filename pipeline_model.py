import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10

if 'NEOS_EMAIL' not in os.environ:
    raise RuntimeError("NEOS_EMAIL environment variable not set")

def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    P = len(stations)
    if P < 1:
        raise ValueError("At least one pumping station required")

    m = pyo.ConcreteModel()
    m.segs = pyo.RangeSet(P)
    m.nodes = pyo.RangeSet(P+1)

    # Global parameters
    m.FLOW      = pyo.Param(initialize=FLOW)
    m.KV        = pyo.Param(initialize=KV)
    m.rho       = pyo.Param(initialize=rho)
    m.Rate_DRA  = pyo.Param(initialize=Rate_DRA)
    m.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Helper dicts
    L, Dout, t, eps, z = {},{},{},{},{}
    SMYS, DF = {},{}
    is_pump, max_pumps = {},{}
    A,B,C = {},{},{}
    Pcoef,Qcoef,Rcoef,Scoef,Tcoef = {},{},{},{},{}
    minRPM, maxRPM = {},{}
    isGrid, ElecRt, SFC = {},{},{}

    # Read station inputs
    for i, s in enumerate(stations, start=1):
        L[i]     = s['L']
        Dout[i]  = s['D']
        t[i]     = s['t']
        eps[i]   = s['rough']
        z[i]     = s['elev']
        SMYS[i]  = s['SMYS']
        DF[i]    = s.get('DF', 0.72)

        if s.get('is_pump', False):
            is_pump[i]   = True
            max_pumps[i] = s['max_pumps']
            A[i],B[i],C[i] = s['A'], s['B'], s['C']
            Pcoef[i] = s['P']; Qcoef[i] = s['Q']
            Rcoef[i] = s['R']; Scoef[i] = s['S']; Tcoef[i] = s['T']
            minRPM[i] = s['MinRPM']; maxRPM[i] = s['DOL']
            isGrid[i]  = 1 if s['power_type']=="Grid" else 0
            ElecRt[i]  = s['rate']
            SFC[i]    = s.get('SFC',0)
        else:
            is_pump[i] = False

    # Terminal node
    z[P+1]      = terminal['elevation']
    min_res     = terminal['min_residual']

    # Create params on model
    m.L    = pyo.Param(m.segs, initialize=L)
    m.Dout = pyo.Param(m.segs, initialize=Dout)
    m.t    = pyo.Param(m.segs, initialize=t)
    m.eps  = pyo.Param(m.segs, initialize=eps)
    m.SMYS = pyo.Param(m.segs, initialize=SMYS)
    m.DF   = pyo.Param(m.segs, initialize=DF)
    m.z    = pyo.Param(m.nodes, initialize=z)

    pumps = [i for i in m.segs if is_pump.get(i,False)]
    m.pumps = pyo.Set(initialize=pumps)
    if pumps:
        m.max_pumps = pyo.Param(m.pumps, initialize=max_pumps)
        m.A  = pyo.Param(m.pumps, initialize=A)
        m.B  = pyo.Param(m.pumps, initialize=B)
        m.C  = pyo.Param(m.pumps, initialize=C)
        m.Pp = pyo.Param(m.pumps, initialize=Pcoef)
        m.Qp = pyo.Param(m.pumps, initialize=Qcoef)
        m.Rp = pyo.Param(m.pumps, initialize=Rcoef)
        m.Sp = pyo.Param(m.pumps, initialize=Scoef)
        m.Tp = pyo.Param(m.pumps, initialize=Tcoef)
        m.minRPM = pyo.Param(m.pumps, initialize=minRPM)
        m.maxRPM = pyo.Param(m.pumps, initialize=maxRPM)
        m.isGrid = pyo.Param(m.pumps, initialize=isGrid)
        m.ElecRt = pyo.Param(m.pumps, initialize=ElecRt)
        m.SFC    = pyo.Param(m.pumps, initialize=SFC)

    # Variables
    m.RH = pyo.Var(m.nodes, domain=pyo.NonNegativeReals, initialize=min_res)
    m.term_con = pyo.Constraint(expr=m.RH[P+1]==min_res)

    m.NOP = pyo.Var(m.segs, domain=pyo.NonNegativeIntegers,
                    bounds=lambda m,i: (0, m.max_pumps[i]) if i in m.pumps else (0,0))
    m.Nu  = pyo.Var(m.segs, domain=pyo.NonNegativeIntegers,
                    bounds=lambda m,i: (
                        int((m.minRPM[i]+9)//10), int(m.maxRPM[i]//10)
                    ) if i in m.pumps else (0,0),
                    initialize=lambda m,i: (
                        int((m.minRPM[i]+9)//10) if i in m.pumps else 0
                    ))
    m.N   = pyo.Expression(m.segs, rule=lambda m,i: 10*m.Nu[i])
    m.DRu = pyo.Var(m.segs, domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=0)
    m.DR  = pyo.Expression(m.segs, rule=lambda m,i: 10*m.DRu[i])

    # Build objective pieces
    pwr_terms = []
    dra_terms = []
    for i in m.segs:
        # Python‐level diameter etc to avoid zero divisors
        d_inner = pyo.value(m.Dout[i]) - 2*pyo.value(m.t[i])
        v = pyo.value(m.FLOW)/(3.414*d_inner**2/4)/3600
        Re= v*d_inner/(pyo.value(m.KV)*1e-6)
        ff= 0.25/(log10((pyo.value(m.eps[i])/d_inner/3.7)+(5.74/(Re**0.9)))**2)
        SH = m.RH[i+1] + (m.z[i+1]-m.z[i])
        HL = ff*(pyo.value(m.L[i])*1000/d_inner)*(v**2/(2*9.81))*(1-m.DR[i]/100)

        if i in m.pumps:
            PH = (m.A[i]*m.FLOW**2 + m.B[i]*m.FLOW + m.C[i]) * (m.N[i]/m.maxRPM[i])**2
            m.add_component(f"balance_{i}",
                pyo.Constraint(expr=m.RH[i] + PH*m.NOP[i] >= SH + HL)
            )
            MAOP = (2*pyo.value(m.t[i])*(pyo.value(m.SMYS[i])*0.070307)*pyo.value(m.DF[i]) / pyo.value(m.Dout[i]))*10000/pyo.value(m.rho)
            m.add_component(f"maop_{i}",
                pyo.Constraint(expr=m.RH[i] + PH*m.NOP[i] <= MAOP)
            )
            eq = m.FLOW*m.maxRPM[i]/m.N[i] if pyo.value(m.N[i])>0 else 0
            eff = (m.Pp[i]*eq**4 + m.Qp[i]*eq**3 + m.Rp[i]*eq**2 + m.Sp[i]*eq + m.Tp[i]) / 100
            base = (m.rho*m.FLOW*9.81*PH*m.NOP[i])/(3600*1000*eff*0.95)
            elec_cost   = base*24*m.ElecRt[i]
            diesel_cost = base*24*(m.SFC[i]*1.34102/1000/820)*1000*m.Price_HSD
            pwr_terms.append(m.isGrid[i]*elec_cost + (1-m.isGrid[i])*diesel_cost)
        else:
            # no‐pump
            m.add_component(f"balance_{i}",
                pyo.Constraint(expr=m.RH[i] >= SH + HL)
            )
            MAOP = (2*pyo.value(m.t[i])*(pyo.value(m.SMYS[i])*0.070307)*pyo.value(m.DF[i]) / pyo.value(m.Dout[i]))*10000/pyo.value(m.rho)
            m.add_component(f"maop_{i}",
                pyo.Constraint(expr=m.RH[i] <= MAOP)
            )
            pwr_terms.append(0)

        dra_terms.append((m.DR[i]/1e6)*m.FLOW*24*1000*m.Rate_DRA)

    # Objective & solve
    m.Obj = pyo.Objective(expr=sum(pwr_terms)+sum(dra_terms), sense=pyo.minimize)
    results = SolverManagerFactory('neos').solve(m, solver='couenne', tee=False)
    m.solutions.load_from(results)

    # Extract
    out = {'total_cost': pyo.value(m.Obj)}
    for i in m.segs:
        key = stations[i-1]['name'].lower()
        out[f"num_pumps_{key}"]    = int(pyo.value(m.NOP[i]))
        out[f"speed_{key}"]         = float(pyo.value(m.N[i]))
        # recompute efficiency for reporting
        eq = pyo.value(m.FLOW)*pyo.value(m.maxRPM[i])/pyo.value(m.N[i]) if pyo.value(m.N[i])>0 else 0
        eff = (pyo.value(m.Pp[i])*eq**4 + pyo.value(m.Qp[i])*eq**3 + pyo.value(m.Rp[i])*eq**2 + pyo.value(m.Sp[i])*eq + pyo.value(m.Tp[i]))/100
        out[f"efficiency_{key}"]    = float(eff*100)
        out[f"power_cost_{key}"]    = float(pwr_terms[i-1])
        out[f"dra_cost_{key}"]      = float(dra_terms[i-1])
        out[f"residual_head_{key}"] = float(pyo.value(m.RH[i]))
    tkey = terminal['name'].lower()
    out[f"residual_head_{tkey}"] = float(pyo.value(m.RH[P+1]))
    return out
