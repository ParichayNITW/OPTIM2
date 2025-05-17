import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# Ensure NEOS email is set via environment or secrets
t = os.environ.get('NEOS_EMAIL')
if not t:
    raise RuntimeError("NEOS_EMAIL environment variable not set")


def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    """
    stations: list of dicts for pumping segments (excluding terminal)
    terminal: dict with keys name, elevation, min_residual
    global: FLOW, KV, rho, Rate_DRA, Price_HSD
    """
    # number of pumping segments
    P = len(stations)
    if P < 1:
        raise ValueError("At least one pumping station required")

    m = pyo.ConcreteModel()
    m.Seg = pyo.RangeSet(P)            # segments 1..P
    m.Node = pyo.RangeSet(P+1)         # nodes 1..P+1

    # global params
    m.FLOW      = pyo.Param(initialize=FLOW)
    m.KV        = pyo.Param(initialize=KV)
    m.rho       = pyo.Param(initialize=rho)
    m.Rate_DRA  = pyo.Param(initialize=Rate_DRA)
    m.Price_HSD = pyo.Param(initialize=Price_HSD)

    # gather input maps
    length, D_out, tmap, rough, elev = {},{},{},{},{}
    SMYS, DF = {},{}
    has_p, max_p = {},{}
    A,B,C = {},{},{}
    Pp,Qp,Rp,Sp,Tp = {},{},{},{},{}
    minRPM_map, maxRPM_map = {},{}
    power_src, elec_rt, sfc_map = {},{},{}

    for i, stn in enumerate(stations, start=1):
        length[i]    = stn['L']
        D_out[i]     = stn['D']
        tmap[i]      = stn['t']
        rough[i]     = stn['rough']
        elev[i]      = stn['elev']
        SMYS[i]      = stn['SMYS']
        DF[i]        = stn.get('DF', 0.72)
        # pumping data
        if stn.get('is_pump', False):
            has_p[i]            = True
            max_p[i]            = stn['max_pumps']
            A[i], B[i], C[i]    = stn['A'], stn['B'], stn['C']
            Pp[i], Qp[i]        = stn['P'], stn['Q']
            Rp[i], Sp[i], Tp[i] = stn['R'], stn['S'], stn['T']
            minRPM_map[i]       = stn.get('min_rpm', 0)
            maxRPM_map[i]       = stn.get('max_rpm', 0)
            power_src[i]        = stn.get('power_source', 'Diesel')
            elec_rt[i]          = stn.get('power_rate', 0)
            sfc_map[i]          = stn.get('SFC', 0)
        else:
            has_p[i] = False

    # terminal elevation + required residual
    elev[P+1]    = terminal['elevation']
    min_res      = terminal['min_residual']

    # parameters
    m.L    = pyo.Param(m.Seg, initialize=length)
    m.Dout = pyo.Param(m.Seg, initialize=D_out)
    m.t    = pyo.Param(m.Seg, initialize=tmap)
    m.eps  = pyo.Param(m.Seg, initialize=rough)
    m.SMYS = pyo.Param(m.Seg, initialize=SMYS)
    m.DF   = pyo.Param(m.Seg, initialize=DF)
    m.z    = pyo.Param(m.Node, initialize=elev)

    segs = [i for i in range(1, P+1) if has_p.get(i, False)]
    m.PSEGS = pyo.Set(initialize=segs)
    if segs:
        m.maxP    = pyo.Param(m.PSEGS, initialize=max_p)
        m.A       = pyo.Param(m.PSEGS, initialize=A)
        m.B       = pyo.Param(m.PSEGS, initialize=B)
        m.C       = pyo.Param(m.PSEGS, initialize=C)
        m.Pcoef   = pyo.Param(m.PSEGS, initialize=Pp)
        m.Qcoef   = pyo.Param(m.PSEGS, initialize=Qp)
        m.Rcoef   = pyo.Param(m.PSEGS, initialize=Rp)
        m.Scoef   = pyo.Param(m.PSEGS, initialize=Sp)
        m.Tcoef   = pyo.Param(m.PSEGS, initialize=Tp)
        m.minRPM  = pyo.Param(m.PSEGS, initialize=minRPM_map)
        m.maxRPM  = pyo.Param(m.PSEGS, initialize=maxRPM_map)
        m.isGrid  = pyo.Param(m.PSEGS, initialize={i:1 if power_src[i]=='Grid' else 0 for i in segs})
        m.ElecRt  = pyo.Param(m.PSEGS, initialize=elec_rt)
        m.SFC     = pyo.Param(m.PSEGS, initialize=sfc_map)

    # variables
    m.RH   = pyo.Var(m.Node, domain=pyo.NonNegativeReals, initialize=min_res)
    m.term = pyo.Constraint(expr=m.RH[P+1] == min_res)

    m.NOP  = pyo.Var(m.Seg, domain=pyo.NonNegativeIntegers,
                     bounds=lambda m,i: (0, m.maxP[i]) if i in m.PSEGS else (0,0))
    m.Nu   = pyo.Var(m.Seg, domain=pyo.NonNegativeIntegers,
                     bounds=lambda m,i: (int((m.minRPM[i]+9)//10), int(m.maxRPM[i]//10))
                     if i in m.PSEGS else (0,0))
    m.N    = pyo.Expression(m.Seg, rule=lambda m,i: 10*m.Nu[i])
    m.DRu  = pyo.Var(m.Seg, domain=pyo.NonNegativeIntegers,
                     bounds=(0,4), initialize=4)
    m.DR   = pyo.Expression(m.Seg, rule=lambda m,i: 10*m.DRu[i])

    # cost lists
    pwr_costs = []
    dra_costs = []

    for i in m.Seg:
        # Compute inner diameter at Python-level
        dia_val = pyo.value(m.Dout[i]) - 2*pyo.value(m.t[i])
        if dia_val <= 0:
            raise ZeroDivisionError(f"Inner diameter <=0 for segment {i}: computed {dia_val}")
        # Flow velocity (m/s) constant term
        area_factor = 3.414 * dia_val**2 / 4
        v_j = pyo.value(m.FLOW) / (area_factor * 3600)
        # Reynolds number (constant)
        Re_j = v_j * dia_val / (pyo.value(m.KV) * 1e-6)
        # Friction factor (constant)
        ff_j = 0.25/(log10((pyo.value(m.eps[i])/dia_val/3.7)+(5.74/(Re_j**0.9)))**2)
        # Static head term
        SH = m.RH[i+1] + (m.z[i+1] - m.z[i])
        # Constant head loss term (to be scaled by drag reduction)
        const_HL = ff_j * (pyo.value(m.L[i])*1000/dia_val) * (v_j**2/(2*9.81))
        if i in m.PSEGS:
            # Pump developed head (expression)
            PH = (m.A[i]*m.FLOW**2 + m.B[i]*m.FLOW + m.C[i]) * (m.N[i]/m.maxRPM[i])**2
            # Head balance constraint
            m.add_component(f"bal_{i}", pyo.Constraint(
                expr=m.RH[i] + PH*m.NOP[i] >= SH + const_HL*(1 - m.DR[i]/100)
            ))
            # MAOP constraint
            MAOP = (2*m.t[i]*(m.SMYS[i]*0.070307)*m.DF[i]/m.Dout[i]) * 10000/m.rho
            m.add_component(f"maop_{i}", pyo.Constraint(
                expr=m.RH[i] + PH*m.NOP[i] <= MAOP
            ))
            # Pump efficiency (expression)
            eq_flow = m.FLOW*m.maxRPM[i]/m.N[i]
            eff = (m.Pcoef[i]*eq_flow**4 + m.Qcoef[i]*eq_flow**3 +
                   m.Rcoef[i]*eq_flow**2 + m.Scoef[i]*eq_flow + m.Tcoef[i]) / 100
            # Power cost expression
            base_cost = (m.rho*m.FLOW*9.81*PH*m.NOP[i])/(3600*1000*eff*0.95)
            elec_cost = base_cost * 24 * m.ElecRt[i]
            diesel_cost = base_cost * 24 * (m.SFC[i]*1.34102/1000/820)*1000 * m.Price_HSD
            pwr_costs.append(m.isGrid[i]*elec_cost + (1-m.isGrid[i])*diesel_cost)
            # Drag reduction cost
            dra_costs.append((m.DR[i]/1e6)*m.FLOW*24*1000*m.Rate_DRA)
        else:
            # No-pump segment: only head loss
            m.add_component(f"bal_{i}", pyo.Constraint(
                expr=m.RH[i] >= SH + const_HL*(1 - m.DR[i]/100)
            ))
            # MAOP for no-pump
            MAOP = (2*m.t[i]*(m.SMYS[i]*0.070307)*m.DF[i]/m.Dout[i]) * 10000/m.rho
            m.add_component(f"maop_{i}", pyo.Constraint(
                expr=m.RH[i] <= MAOP
            ))
            pwr_costs.append(0)
            dra_costs.append((m.DR[i]/1e6)*m.FLOW*24*1000*m.Rate_DRA)

        # Objective
    m.Obj = pyo.Objective(expr=sum(pwr_costs) + sum(dra_costs), sense=pyo.minimize)

    # Solve
    results = SolverManagerFactory('neos').solve(m, solver='couenne', tee=False) SolverManagerFactory('neos').solve(m, solver='couenne', tee=False)
    m.solutions.load_from(results)

    # extract
    out = {'total_cost': pyo.value(m.Obj)}
    for i in m.Seg:
        key = stations[i-1]['name'].lower()
        out[f"num_pumps_{key}"]    = int(pyo.value(m.NOP[i]))
        out[f"speed_{key}"]         = float(pyo.value(m.N[i]))
        out[f"efficiency_{key}"]    = float(pyo.value(eff*100))
        out[f"power_cost_{key}"]    = float(pyo.value(pwr_costs[i-1]))
        out[f"dra_cost_{key}"]      = float(pyo.value(dra_costs[i-1]))
        out[f"residual_head_{key}"] = float(pyo.value(m.RH[i]))
    tkey = terminal['name'].lower()
    out[f"residual_head_{tkey}"] = float(pyo.value(m.RH[P+1]))
    return out
