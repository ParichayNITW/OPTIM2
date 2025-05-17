import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory

# Require NEOS_EMAIL in env
if 'NEOS_EMAIL' not in os.environ:
    raise RuntimeError("Please set NEOS_EMAIL as an environment variable or in Streamlit secrets")

def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    """
    stations: list of dicts for each pumping segment (exclude terminal)
      must include keys:
        name, elev, D, t, SMYS, DF,
        is_pump (bool),
        -- if is_pump True, also: max_pumps, MinRPM, DOL,
           power_type ("Grid"/"Diesel"), rate (INR/kWh) or SFC (gm/bhp-hr),
           A, B, C, P, Q, R, S, T
    terminal: dict with keys:
        name, elevation (m), min_residual (m)
    other args: scalars
    """
    P = len(stations)
    if P < 1:
        raise ValueError("At least one pumping segment required")

    m = pyo.ConcreteModel()
    m.seg   = pyo.RangeSet(1, P)
    m.node  = pyo.RangeSet(1, P+1)

    # Global params
    m.FLOW      = pyo.Param(initialize=FLOW)
    m.KV        = pyo.Param(initialize=KV)
    m.rho       = pyo.Param(initialize=rho)
    m.Rate_DRA  = pyo.Param(initialize=Rate_DRA)
    m.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Read inputs into param dicts
    L, Dout, t, eps, z = {},{},{},{},{}
    SMYS, DF = {},{}
    is_pump, max_pumps = {},{}
    A,B,C = {},{},{}
    Pp,Qp,Rp,Sp,Tp = {},{},{},{},{}
    minRPM, maxRPM = {},{}
    isGrid, ElecRt, SFC = {},{},{}

    for i, stn in enumerate(stations, start=1):
        L[i]     = stn['L']
        Dout[i]  = stn['D']
        t[i]     = stn['t']
        eps[i]   = stn['rough']
        z[i]     = stn['elev']
        SMYS[i]  = stn['SMYS']
        DF[i]    = stn.get('DF', 0.72)
        if stn.get('is_pump', False):
            is_pump[i]   = True
            max_pumps[i] = stn['max_pumps']
            A[i],B[i],C[i]    = stn['A'], stn['B'], stn['C']
            Pp[i],Qp[i]       = stn['P'], stn['Q']
            Rp[i],Sp[i],Tp[i] = stn['R'], stn['S'], stn['T']
            minRPM[i]         = stn['MinRPM']
            maxRPM[i]         = stn['DOL']
            isGrid[i]         = 1 if stn['power_type']=="Grid" else 0
            ElecRt[i]         = stn['rate']
            SFC[i]            = stn.get('SFC', 0)
        else:
            is_pump[i] = False

    # Terminal node
    z[P+1]      = terminal['elevation']
    min_res     = terminal['min_residual']

    # Construct Params on model
    m.L    = pyo.Param(m.seg,   initialize=L)
    m.Dout = pyo.Param(m.seg,   initialize=Dout)
    m.t    = pyo.Param(m.seg,   initialize=t)
    m.eps  = pyo.Param(m.seg,   initialize=eps)
    m.SMYS = pyo.Param(m.seg,   initialize=SMYS)
    m.DF   = pyo.Param(m.seg,   initialize=DF)
    m.z    = pyo.Param(m.node,  initialize=z)

    pump_list = [i for i in m.seg if is_pump.get(i,False)]
    m.psg = pyo.Set(initialize=pump_list)
    if pump_list:
        m.maxP    = pyo.Param(m.psg, initialize=max_pumps)
        m.A       = pyo.Param(m.psg, initialize=A)
        m.B       = pyo.Param(m.psg, initialize=B)
        m.C       = pyo.Param(m.psg, initialize=C)
        m.Pp      = pyo.Param(m.psg, initialize=Pp)
        m.Qp      = pyo.Param(m.psg, initialize=Qp)
        m.Rp      = pyo.Param(m.psg, initialize=Rp)
        m.Sp      = pyo.Param(m.psg, initialize=Sp)
        m.Tp      = pyo.Param(m.psg, initialize=Tp)
        m.minRPM  = pyo.Param(m.psg, initialize=minRPM)
        m.maxRPM  = pyo.Param(m.psg, initialize=maxRPM)
        m.isGrid  = pyo.Param(m.psg, initialize=isGrid)
        m.ElecRt  = pyo.Param(m.psg, initialize=ElecRt)
        m.SFC     = pyo.Param(m.psg, initialize=SFC)

    # Variables
    m.RH   = pyo.Var(m.node, domain=pyo.NonNegativeReals, initialize=min_res)
    m.term_con = pyo.Constraint(expr=m.RH[P+1] == min_res)

    m.NOP  = pyo.Var(m.seg,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda mod,i: (0, mod.maxP[i]) if i in mod.psg else (0,0))
    m.Nu   = pyo.Var(m.seg,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda mod,i: (
             int((mod.minRPM[i]+9)//10), int(mod.maxRPM[i]//10)
        ) if i in mod.psg else (0,0),
        initialize=lambda mod,i: (int((mod.minRPM[i]+9)//10) if i in mod.psg else 0)
    )
    m.N    = pyo.Expression(m.seg, rule=lambda mod,i: 10*mod.Nu[i])
    m.DRu  = pyo.Var(m.seg,
        domain=pyo.NonNegativeIntegers,
        bounds=(0,4),
        initialize=0
    )
    m.DR   = pyo.Expression(m.seg, rule=lambda mod,i: 10*mod.DRu[i])

    # Build constraints & objective pieces
    power_terms = []
    dra_terms   = []
    for i in m.seg:
        # pure-python inner diameter and velocity
        din = pyo.value(m.Dout[i]) - 2*pyo.value(m.t[i])
        if din <= 0:
            raise ZeroDivisionError(f"Segment {i} inner diameter <=0")
        v   = pyo.value(m.FLOW)/(3.414*din**2/4)/3600
        Re  = v*din/(pyo.value(m.KV)*1e-6)
        ff  = 0.25/(pyo.log((m.eps[i]/din/3.7)+(5.74/(Re**0.9)),10)**2)
        SH  = m.RH[i+1] + (m.z[i+1]-m.z[i])
        HL  = ff*(pyo.value(m.L[i])*1000/din)*(v**2/(2*9.81))*(1 - m.DR[i]/100)

        if i in m.psg:
            PH = (m.A[i]*m.FLOW**2 + m.B[i]*m.FLOW + m.C[i]) * (m.N[i]/m.maxRPM[i])**2
            m.add_component(f"bal_{i}",
                pyo.Constraint(expr=m.RH[i] + PH*m.NOP[i] >= SH + HL)
            )
            MAOP = (2*m.t[i]*(m.SMYS[i]*0.070307)*m.DF[i]/m.Dout[i])*10000/m.rho
            m.add_component(f"maop_{i}",
                pyo.Constraint(expr=m.RH[i] + PH*m.NOP[i] <= MAOP)
            )
            # pump efficiency
            eqf = m.FLOW*m.maxRPM[i]/m.N[i]
            eff = (m.Pp[i]*eqf**4 + m.Qp[i]*eqf**3 + m.Rp[i]*eqf**2 + m.Sp[i]*eqf + m.Tp[i]) / 100
            base= (m.rho*m.FLOW*9.81*PH*m.NOP[i])/(3600*1000*eff*0.95)
            elec_cost   = base*24*m.ElecRt[i]
            diesel_cost = base*24*(m.SFC[i]*1.34102/1000/820)*1000*m.Price_HSD
            power_terms.append(m.isGrid[i]*elec_cost + (1-m.isGrid[i])*diesel_cost)
        else:
            # no-pump
            m.add_component(f"bal_{i}",
                pyo.Constraint(expr=m.RH[i] >= SH + HL)
            )
            MAOP = (2*m.t[i]*(m.SMYS[i]*0.070307)*m.DF[i]/m.Dout[i])*10000/m.rho
            m.add_component(f"maop_{i}",
                pyo.Constraint(expr=m.RH[i] <= MAOP)
            )
            power_terms.append(0)

        dra_terms.append((m.DR[i]/1e6)*m.FLOW*24*1000*m.Rate_DRA)

    # Objective
    m.Obj = pyo.Objective(expr=sum(power_terms)+sum(dra_terms), sense=pyo.minimize)

    # Solve
    sol = SolverManagerFactory('neos').solve(m, solver='couenne', tee=False)
    m.solutions.load_from(sol)

    # Extract
    out = {'total_cost': pyo.value(m.Obj)}
    for i in m.seg:
        key = stations[i-1]['name'].lower()
        out[f"num_pumps_{key}"]    = int(pyo.value(m.NOP[i]))
        out[f"speed_{key}"]         = float(pyo.value(m.N[i]))
        # recompute efficiency
        eqf = pyo.value(m.FLOW)*pyo.value(m.maxRPM[i])/pyo.value(m.N[i])
        eff = (pyo.value(m.Pp[i])*eqf**4 + pyo.value(m.Qp[i])*eqf**3 +
               pyo.value(m.Rp[i])*eqf**2 + pyo.value(m.Sp[i])*eqf +
               pyo.value(m.Tp[i])) / 100
        out[f"efficiency_{key}"]    = float(eff*100)
        out[f"power_cost_{key}"]    = float(pyo.value(power_terms[i-1]))
        out[f"dra_cost_{key}"]      = float(pyo.value(dra_terms[i-1]))
        out[f"residual_head_{key}"] = float(pyo.value(m.RH[i]))

    tkey = terminal['name'].lower()
    out[f"residual_head_{tkey}"] = float(pyo.value(m.RH[P+1]))
    return out
