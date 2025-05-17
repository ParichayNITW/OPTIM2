import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10

# Ensure NEOS email is set
if 'NEOS_EMAIL' not in os.environ:
    raise RuntimeError("NEOS_EMAIL environment variable not set")


def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    """
    stations: list of dicts for each pumping segment
    terminal: dict with keys name, elevation, min_residual
    """
    P = len(stations)
    if P < 1:
        raise ValueError("At least one pumping segment required")

    m = pyo.ConcreteModel()
    m.seg  = pyo.RangeSet(1, P)
    m.node = pyo.RangeSet(1, P+1)

    # Global parameters
    m.FLOW      = pyo.Param(initialize=FLOW)
    m.KV        = pyo.Param(initialize=KV)
    m.rho       = pyo.Param(initialize=rho)
    m.Rate_DRA  = pyo.Param(initialize=Rate_DRA)
    m.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Input dictionaries
    L, Dout, thickness, eps, z = {},{},{},{},{}
    SMYS, DF = {},{}
    is_pump, max_pumps = {},{}
    Acoef, Bcoef, Ccoef = {},{},{}
    Pcoef, Qcoef, Rcoef, Scoef, Tcoef = {},{},{},{},{}
    minRPM, maxRPM = {},{}
    isGrid, ElecRt, SFC = {},{},{}

    # Read station inputs
    for i, s in enumerate(stations, start=1):
        L[i]          = s['L']
        Dout[i]       = s['D']
        thickness[i]  = s['t']
        eps[i]        = s['rough']
        z[i]          = s['elev']
        SMYS[i]       = s['SMYS']
        DF[i]         = s.get('DF', 0.72)
        if s.get('is_pump', False):
            is_pump[i]       = True
            max_pumps[i]     = s['max_pumps']
            Acoef[i],Bcoef[i],Ccoef[i] = s['A'],s['B'],s['C']
            Pcoef[i],Qcoef[i]         = s['P'],s['Q']
            Rcoef[i],Scoef[i],Tcoef[i] = s['R'],s['S'],s['T']
            minRPM[i]       = s['MinRPM']
            maxRPM[i]       = s['DOL']
            isGrid[i]       = 1 if s['power_type']=='Grid' else 0
            ElecRt[i]       = s['rate']
            SFC[i]          = s.get('SFC',0)
        else:
            is_pump[i] = False

    # Terminal node elevation and required residual
    z[P+1]      = terminal['elevation']
    min_res     = terminal['min_residual']

    # Create model parameters
    m.L      = pyo.Param(m.seg,   initialize=L)
    m.Dout   = pyo.Param(m.seg,   initialize=Dout)
    m.t      = pyo.Param(m.seg,   initialize=thickness)
    m.eps    = pyo.Param(m.seg,   initialize=eps)
    m.SMYS   = pyo.Param(m.seg,   initialize=SMYS)
    m.DF     = pyo.Param(m.seg,   initialize=DF)
    m.z      = pyo.Param(m.node,  initialize=z)

    # Identify pump segments
    pumps = [i for i in m.seg if is_pump.get(i,False)]
    m.pumps = pyo.Set(initialize=pumps)
    if pumps:
        m.maxP   = pyo.Param(m.pumps, initialize=max_pumps)
        m.A      = pyo.Param(m.pumps, initialize=Acoef)
        m.B      = pyo.Param(m.pumps, initialize=Bcoef)
        m.C      = pyo.Param(m.pumps, initialize=Ccoef)
        m.Pp     = pyo.Param(m.pumps, initialize=Pcoef)
        m.Qp     = pyo.Param(m.pumps, initialize=Qcoef)
        m.Rp     = pyo.Param(m.pumps, initialize=Rcoef)
        m.Sp     = pyo.Param(m.pumps, initialize=Scoef)
        m.Tp     = pyo.Param(m.pumps, initialize=Tcoef)
        m.minRPM = pyo.Param(m.pumps, initialize=minRPM)
        m.maxRPM = pyo.Param(m.pumps, initialize=maxRPM)
        m.isGrid = pyo.Param(m.pumps, initialize=isGrid)
        m.ElecRt = pyo.Param(m.pumps, initialize=ElecRt)
        m.SFC    = pyo.Param(m.pumps, initialize=SFC)

    # Decision variables
    m.RH  = pyo.Var(m.node, domain=pyo.NonNegativeReals, initialize=min_res)
    m.term = pyo.Constraint(expr=m.RH[P+1] == min_res)

    m.NOP = pyo.Var(m.seg,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda mod,i: (0, mod.maxP[i]) if i in mod.pumps else (0,0)
    )
    m.Nu = pyo.Var(m.seg,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda mod,i: (
            int((mod.minRPM[i]+9)//10), int(mod.maxRPM[i]//10)
        ) if i in mod.pumps else (0,0),
        initialize=lambda mod,i: (int((mod.minRPM[i]+9)//10) if i in mod.pumps else 0)
    )
    m.N   = pyo.Expression(m.seg, rule=lambda mod,i: 10*mod.Nu[i])
    m.DRu = pyo.Var(m.seg, domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=0)
    m.DR  = pyo.Expression(m.seg, rule=lambda mod,i: 10*mod.DRu[i])

    # Build constraints & collect objective terms
    power_costs = []
    dra_costs   = []
    for i in m.seg:
        # Compute inner dia, velocity, friction in pure Python
        inner_d = pyo.value(m.Dout[i]) - 2*pyo.value(m.t[i])
        if inner_d <= 0:
            raise ZeroDivisionError(f"Segment {i} inner diameter <=0: {inner_d}")
        v = pyo.value(m.FLOW)/(3.414*inner_d**2/4)/3600
        Re = v*inner_d/(pyo.value(m.KV)*1e-6)
        expr = pyo.value(m.eps[i])/inner_d/3.7 + 5.74/(Re**0.9)
        if expr <= 0:
            raise ValueError(f"Invalid log10 argument: {expr}")
        ff = 0.25/(log10(expr)**2)
        SH = m.RH[i+1] + (m.z[i+1]-m.z[i])
        HL = ff*(pyo.value(m.L[i])*1000/inner_d)*(v**2/(2*9.81))*(1 - m.DR[i]/100)

        if i in m.pumps:
            PH = (m.A[i]*m.FLOW**2 + m.B[i]*m.FLOW + m.C[i])*(m.N[i]/m.maxRPM[i])**2
            m.add_component(f"bal_{i}", pyo.Constraint(
                expr=m.RH[i] + PH*m.NOP[i] >= SH + HL
            ))
            MAOP = (2*m.t[i]*(m.SMYS[i]*0.070307)*m.DF[i]/m.Dout[i]) * 10000/m.rho
            m.add_component(f"maop_{i}", pyo.Constraint(
                expr=m.RH[i] + PH*m.NOP[i] <= MAOP
            ))
            eqf = m.FLOW*m.maxRPM[i]/m.N[i]
            eff = (m.Pp[i]*eqf**4 + m.Qp[i]*eqf**3 + m.Rp[i]*eqf**2 + m.Sp[i]*eqf + m.Tp[i]) / 100
            base = (m.rho*m.FLOW*9.81*PH*m.NOP[i])/(3600*1000*eff*0.95)
            rcost = base * 24 * m.ElecRt[i]
            dcost = base * 24 * (m.SFC[i]*1.34102/1000/820)*1000*m.Price_HSD
            power_costs.append(m.isGrid[i]*rcost + (1-m.isGrid[i])*dcost)
        else:
            m.add_component(f"bal_{i}", pyo.Constraint(
                expr=m.RH[i] >= SH + HL
            ))
            MAOP = (2*m.t[i]*(m.SMYS[i]*0.070307)*m.DF[i]/m.Dout[i]) * 10000/m.rho
            m.add_component(f"maop_{i}", pyo.Constraint(
                expr=m.RH[i] <= MAOP
            ))
            power_costs.append(0)

        dra_costs.append((m.DR[i]/1e6)*m.FLOW*24*1000*m.Rate_DRA)

    # Objective
    m.Obj = pyo.Objective(expr=sum(power_costs)+sum(dra_costs), sense=pyo.minimize)

    # Solve
    sol = SolverManagerFactory('neos').solve(m, solver='couenne', tee=False)
    m.solutions.load_from(sol)

    # Extract
    out = {'total_cost': pyo.value(m.Obj)}
    for i in m.seg:
        key = stations[i-1]['name'].lower()
        out[f"num_pumps_{key}"]    = int(pyo.value(m.NOP[i]))
        out[f"speed_{key}"]         = float(pyo.value(m.N[i]))
        eqf = pyo.value(m.FLOW)*pyo.value(m.maxRPM[i])/pyo.value(m.N[i])
        eff = (pyo.value(m.Pp[i])*eqf**4 + pyo.value(m.Qp[i])*eqf**3 +
               pyo.value(m.Rp[i])*eqf**2 + pyo.value(m.Sp[i])*eqf +
               pyo.value(m.Tp[i]))/100
        out[f"efficiency_{key}"]    = float(eff*100)
        out[f"power_cost_{key}"]    = float(pyo.value(power_costs[i-1]))
        out[f"dra_cost_{key}"]      = float(pyo.value(dra_costs[i-1]))
        out[f"residual_head_{key}"] = float(pyo.value(m.RH[i]))
    tkey = terminal['name'].lower()
    out[f"residual_head_{tkey}"] = float(pyo.value(m.RH[P+1]))
    return out
