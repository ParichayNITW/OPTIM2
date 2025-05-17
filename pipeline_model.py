import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# Ensure NEOS email is set via environment or secrets
if 'NEOS_EMAIL' not in os.environ:
    raise RuntimeError("NEOS_EMAIL environment variable not set")


def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    """
    stations: list of dicts for pumping segments (excluding terminal)
    terminal: dict with keys name, elevation, min_residual
    global: FLOW, KV, rho, Rate_DRA, Price_HSD
    """
    P = len(stations)
    if P < 1:
        raise ValueError("At least one pumping station required")

    m = pyo.ConcreteModel()
    m.Seg = pyo.RangeSet(P)
    m.Node = pyo.RangeSet(P+1)

    # Global parameters
    m.FLOW      = pyo.Param(initialize=FLOW)
    m.KV        = pyo.Param(initialize=KV)
    m.rho       = pyo.Param(initialize=rho)
    m.Rate_DRA  = pyo.Param(initialize=Rate_DRA)
    m.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Gather inputs
    length, D_out, thickness, roughness, elev = {},{},{},{},{}
    SMYS, DF = {},{}
    pump_flag, max_pumps = {},{}
    Acoef,Bcoef,Ccoef = {},{},{}
    Pcoef,Qcoef,Rcoef,Scoef,Tcoef = {},{},{},{},{}
    minRPM_map, maxRPM_map = {},{}
    isGrid_map, ElecRt_map, SFC_map = {},{},{}

    for i, stn in enumerate(stations, start=1):
        length[i]    = stn['L']
        D_out[i]     = stn['D']
        thickness[i] = stn['t']
        roughness[i] = stn['rough']
        elev[i]      = stn['elev']
        SMYS[i]      = stn['SMYS']
        DF[i]        = stn.get('DF', 0.72)
        if stn.get('is_pump', False):
            pump_flag[i]       = True
            max_pumps[i]       = stn['max_pumps']
            Acoef[i], Bcoef[i], Ccoef[i] = stn['A'], stn['B'], stn['C']
            Pcoef[i], Qcoef[i] = stn['P'], stn['Q']
            Rcoef[i], Scoef[i], Tcoef[i] = stn['R'], stn['S'], stn['T']
            minRPM_map[i] = stn.get('MinRPM', 0)  # map front-end MinRPM
            maxRPM_map[i] = stn.get('DOL', 0)      # map front-end DOL
            isGrid_map[i]      = 1 if stn.get('power_source', 'Diesel') == 'Grid' else 0
            ElecRt_map[i]      = stn.get('power_rate', 0)
            SFC_map[i]         = stn.get('SFC', 0)
        else:
            pump_flag[i] = False

    elev[P+1]       = terminal['elevation']
    min_residual    = terminal['min_residual']

    # Define Params
    m.L    = pyo.Param(m.Seg, initialize=length)
    m.Dout = pyo.Param(m.Seg, initialize=D_out)
    m.t    = pyo.Param(m.Seg, initialize=thickness)
    m.eps  = pyo.Param(m.Seg, initialize=roughness)
    m.SMYS = pyo.Param(m.Seg, initialize=SMYS)
    m.DF   = pyo.Param(m.Seg, initialize=DF)
    m.z    = pyo.Param(m.Node, initialize=elev)

    pump_idxs = [i for i in m.Seg if pump_flag.get(i, False)]
    m.PSEGS = pyo.Set(initialize=pump_idxs)
    if pump_idxs:
        m.maxP     = pyo.Param(m.PSEGS, initialize=max_pumps)
        m.A        = pyo.Param(m.PSEGS, initialize=Acoef)
        m.B        = pyo.Param(m.PSEGS, initialize=Bcoef)
        m.C        = pyo.Param(m.PSEGS, initialize=Ccoef)
        m.Pcoef    = pyo.Param(m.PSEGS, initialize=Pcoef)
        m.Qcoef    = pyo.Param(m.PSEGS, initialize=Qcoef)
        m.Rcoef    = pyo.Param(m.PSEGS, initialize=Rcoef)
        m.Scoef    = pyo.Param(m.PSEGS, initialize=Scoef)
        m.Tcoef    = pyo.Param(m.PSEGS, initialize=Tcoef)
        m.minRPM   = pyo.Param(m.PSEGS, initialize=minRPM_map)
        m.maxRPM   = pyo.Param(m.PSEGS, initialize=maxRPM_map)
        m.isGrid   = pyo.Param(m.PSEGS, initialize=isGrid_map)
        m.ElecRt   = pyo.Param(m.PSEGS, initialize=ElecRt_map)
        m.SFC      = pyo.Param(m.PSEGS, initialize=SFC_map)

    # Variables
    m.RH  = pyo.Var(m.Node, domain=pyo.NonNegativeReals, initialize=min_residual)
    m.term_head = pyo.Constraint(expr=m.RH[P+1] == min_residual)

    m.NOP = pyo.Var(m.Seg, domain=pyo.NonNegativeIntegers,
                    bounds=lambda m,i: (0, m.maxP[i]) if i in m.PSEGS else (0,0))
    m.Nu  = pyo.Var(m.Seg, domain=pyo.NonNegativeIntegers,
                    bounds=lambda m,i: (int((m.minRPM[i]+9)//10), int(m.maxRPM[i]//10))
                    if i in m.PSEGS else (0,0))
    m.N   = pyo.Expression(m.Seg, rule=lambda m,i: 10*m.Nu[i])
    m.DRu = pyo.Var(m.Seg, domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=4)
    m.DR  = pyo.Expression(m.Seg, rule=lambda m,i: 10*m.DRu[i])

    # Build constraints and objective
    power_terms = []
    dra_terms   = []
    for i in m.Seg:
        # pure Python inner dia and velocity
        dia = pyo.value(m.Dout[i]) - 2*pyo.value(m.t[i])
        if dia <= 0:
            raise ZeroDivisionError(f"Zero inner dia seg {i}")
        v = pyo.value(m.FLOW)/(3.414*dia**2/4)/3600
        Re= v*dia/(pyo.value(m.KV)*1e-6)
        ff= 0.25/(log10((pyo.value(m.eps[i])/dia/3.7)+(5.74/(Re**0.9)))**2)
        SH = m.RH[i+1] + (m.z[i+1]-m.z[i])
        HL_const = ff*(pyo.value(m.L[i])*1000/dia)*(v**2/(2*9.81))
        if i in m.PSEGS:
            PH = (m.A[i]*m.FLOW**2 + m.B[i]*m.FLOW + m.C[i]) * (m.N[i]/m.maxRPM[i])**2
            m.add_component(f"bal_{i}", pyo.Constraint(
                expr=m.RH[i] + PH*m.NOP[i] >= SH + HL_const*(1-m.DR[i]/100)
            ))
            MAOP_val = (2*pyo.value(m.t[i])*(pyo.value(m.SMYS[i])*0.070307)*pyo.value(m.DF[i])/pyo.value(m.Dout[i]))*10000/pyo.value(m.rho)
            m.add_component(f"maop_{i}", pyo.Constraint(
                expr=m.RH[i] + PH*m.NOP[i] <= MAOP_val
            ))
            eqf = m.FLOW*m.maxRPM[i]/m.N[i] if pyo.value(m.N[i])>0 else 0
            eff = (m.Pcoef[i]*eqf**4 + m.Qcoef[i]*eqf**3 + m.Rcoef[i]*eqf**2 + m.Scoef[i]*eqf + m.Tcoef[i]) / 100
            base = (m.rho*m.FLOW*9.81*PH*m.NOP[i])/(3600*1000*eff*0.95)
            ec = base*24*m.ElecRt[i]
            dc = base*24*(m.SFC[i]*1.34102/1000/820)*1000*m.Price_HSD
            power_terms.append(m.isGrid[i]*ec + (1-m.isGrid[i])*dc)
        else:
            m.add_component(f"bal_{i}", pyo.Constraint(
                expr=m.RH[i] >= SH + HL_const*(1-m.DR[i]/100)
            ))
            MAOP_val = (2*pyo.value(m.t[i])*(pyo.value(m.SMYS[i])*0.070307)*pyo.value(m.DF[i])/pyo.value(m.Dout[i]))*10000/pyo.value(m.rho)
            m.add_component(f"maop_{i}", pyo.Constraint(
                expr=m.RH[i] <= MAOP_val
            ))
            power_terms.append(0)
        dra_terms.append((m.DR[i]/1e6)*m.FLOW*24*1000*m.Rate_DRA)

    m.Obj = pyo.Objective(expr=sum(power_terms) + sum(dra_terms), sense=pyo.minimize)

    results = SolverManagerFactory('neos').solve(m, solver='couenne', tee=False)
    m.solutions.load_from(results)

    # extract
    res = {'total_cost': pyo.value(m.Obj)}
    for i in m.Seg:
        key = stations[i-1]['name'].lower()
        res[f"num_pumps_{key}"]    = int(pyo.value(m.NOP[i]))
        res[f"speed_{key}"]         = float(pyo.value(m.N[i]))
        res[f"efficiency_{key}"]    = float(pyo.value(eff*100))
        res[f"power_cost_{key}"]    = float(pyo.value(power_terms[i-1]))
        res[f"dra_cost_{key}"]      = float(pyo.value(dra_terms[i-1]))
        res[f"residual_head_{key}"] = float(pyo.value(m.RH[i]))
    tkey = terminal['name'].lower()
    res[f"residual_head_{tkey}"] = float(pyo.value(m.RH[P+1]))
    return res
