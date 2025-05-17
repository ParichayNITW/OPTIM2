import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# Ensure NEOS email is set via environment or secrets
if 'NEOS_EMAIL' not in os.environ:
    raise RuntimeError("NEOS_EMAIL environment variable not set")


def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    """
    stations: list of dicts for intermediate pumping stations (no terminal)
      keys: name, elev, D, t, SMYS, DF, L,
            is_pump, power_source, power_rate/SFC,
            max_pumps, min_rpm, max_rpm,
            DR_max, A,B,C,P,Q,R,S,T
    terminal: dict with keys: name, elevation, min_residual
    global params: FLOW, KV, rho, Rate_DRA, Price_HSD
    """
    P = len(stations)
    if P < 1:
        raise ValueError("At least one pumping station required")

    model = pyo.ConcreteModel()
    model.segments = pyo.RangeSet(1, P)
    model.nodes    = pyo.RangeSet(1, P+1)

    # Global parameters
    model.FLOW      = pyo.Param(initialize=FLOW)
    model.KV        = pyo.Param(initialize=KV)
    model.rho       = pyo.Param(initialize=rho)
    model.Rate_DRA  = pyo.Param(initialize=Rate_DRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Read station data
    length, d_inner, thickness, roughness, elevation = {},{},{},{},{}
    SMYS, DF = {},{}
    has_pump, max_pumps = {},{}
    Acoef,Bcoef,Ccoef = {},{},{}
    Pcoef,Qcoef,Rcoef,Scoef,Tcoef = {},{},{},{},{}
    min_rpm, max_rpm = {},{}

    for i, stn in enumerate(stations, start=1):
        length[i]    = stn['L']
        thickness[i] = stn['t']
        d_inner[i]   = stn['D'] - 2*stn['t']
        roughness[i] = stn['rough']
        elevation[i] = stn['elev']
        SMYS[i]      = stn['SMYS']
        DF[i]        = stn.get('DF', 0.72)  # default design factor
        if stn.get('is_pump', False):
            has_pump[i]      = True
            max_pumps[i]     = stn['max_pumps']
            Acoef[i],Bcoef[i],Ccoef[i] = stn['A'],stn['B'],stn['C']
            Pcoef[i],Qcoef[i] = stn['P'],stn['Q']
            Rcoef[i],Scoef[i],Tcoef[i] = stn['R'],stn['S'],stn['T']
            min_rpm[i] = stn.get('min_rpm', 0)
            max_rpm[i] = stn.get('max_rpm', 0)
        else:
            has_pump[i] = False

    # terminal node elevation + required residual
    elevation[P+1] = terminal['elevation']
    min_residual   = terminal['min_residual']

    # Parameters
    model.L    = pyo.Param(model.segments, initialize=length)
    model.d    = pyo.Param(model.segments, initialize=d_inner)
    model.t    = pyo.Param(model.segments, initialize=thickness)
    model.eps  = pyo.Param(model.segments, initialize=roughness)
    model.SMYS = pyo.Param(model.segments, initialize=SMYS)
    model.DF   = pyo.Param(model.segments, initialize=DF)
    model.z    = pyo.Param(model.nodes, initialize=elevation)

    pump_idxs = [i for i in model.segments if has_pump.get(i, False)]
    model.PUMPS = pyo.Set(initialize=pump_idxs)
    if pump_idxs:
        model.max_pumps = pyo.Param(model.PUMPS, initialize=max_pumps)
        model.A  = pyo.Param(model.PUMPS, initialize=Acoef)
        model.B  = pyo.Param(model.PUMPS, initialize=Bcoef)
        model.C  = pyo.Param(model.PUMPS, initialize=Ccoef)
        model.Pp = pyo.Param(model.PUMPS, initialize=Pcoef)
        model.Qp = pyo.Param(model.PUMPS, initialize=Qcoef)
        model.Rp = pyo.Param(model.PUMPS, initialize=Rcoef)
        model.Sp = pyo.Param(model.PUMPS, initialize=Scoef)
        model.Tp = pyo.Param(model.PUMPS, initialize=Tcoef)
        model.minRPM = pyo.Param(model.PUMPS, initialize=min_rpm)
        model.maxRPM = pyo.Param(model.PUMPS, initialize=max_rpm)

    # Decision variables
    model.RH  = pyo.Var(model.nodes, domain=pyo.NonNegativeReals,
                       initialize=min_residual)
    # terminal residual equality
    model.term_con = pyo.Constraint(expr=model.RH[P+1] == min_residual)

    def bound_nop(m,i):
        return (0, m.max_pumps[i]) if i in m.PUMPS else (0,0)
    model.NOP = pyo.Var(model.segments, domain=pyo.NonNegativeIntegers,
                        bounds=bound_nop)

    def bound_n(m,i):
        if i in m.PUMPS:
            lo = int(pyo.value(m.minRPM[i]) + 9)//10
            hi = int(pyo.value(m.maxRPM[i]))//10
            return (lo, hi)
        return (0,0)
    model.N_u = pyo.Var(model.segments, domain=pyo.NonNegativeIntegers,
                        bounds=bound_n)
    model.N   = pyo.Expression(model.segments, rule=lambda m,i: 10*m.N_u[i])

    model.DR_u = pyo.Var(model.segments, domain=pyo.NonNegativeIntegers,
                         bounds=(0,4), initialize=4)
    model.DR   = pyo.Expression(model.segments, rule=lambda m,i: 10*m.DR_u[i])

    # Helper for friction
    def ff(Re, eps, d): return 0.25/(log10((eps/d/3.7)+(5.74/(Re**0.9)))**2)

    # Objective components
    power_terms = []
    dra_terms   = []

    for i in model.segments:
        # flow & hydraulics
        v_i = model.FLOW/(3.414*model.d[i]**2/4)/3600
        Re_i= v_i*model.d[i]/(model.KV*1e-6)
        f_i = ff(Re_i, model.eps[i], model.d[i])
        SH  = model.RH[i+1] + (model.z[i+1]-model.z[i])
        DH  = f_i*(model.L[i]*1000/model.d[i])*(v_i**2/(2*9.81))*(1-model.DR[i]/100)
        # pump head EQ
        if i in model.PUMPS:
            TDH = (model.A[i]*model.FLOW**2 + model.B[i]*model.FLOW + model.C[i]) \
                  * (model.N[i]/model.maxRPM[i])**2
        else:
            TDH = 0
        # balance & MAOP
        model.add_component(f"cons_{i}_bal", pyo.Constraint(
            expr=(model.RH[i] + TDH*model.NOP[i] >= SH + DH)
        ))
        MAOP = (2*model.t[i]*(model.SMYS[i]*0.070307)*model.DF[i]/model.d[i]) *10000/model.rho
        model.add_component(f"cons_{i}_maop", pyo.Constraint(
            expr=(model.RH[i] + TDH*model.NOP[i] <= MAOP)
        ))
        # cost
        if i in model.PUMPS:
            eq_flow = model.FLOW*model.maxRPM[i]/model.N[i]
            eff = (model.Pp[i]*eq_flow**4 + model.Qp[i]*eq_flow**3 +
                   model.Rp[i]*eq_flow**2 + model.Sp[i]*eq_flow + model.Tp[i]) / 100
            # electric vs diesel
            base = (model.rho*model.FLOW*9.81*TDH*model.NOP[i])/(3600*1000*eff*0.95)
            e_cost = base * 24 * (model.ElecRate[i] if hasattr(model,'ElecRate') else 0)
            d_cost = base * 24 * (model.SFC[i]*1.34102/1000/820)*1000 * model.Price_HSD
            power_terms.append((model.isElectric[i]*e_cost + (1-model.isElectric[i])*d_cost))
        # DRA cost
        dra_terms.append((model.DR[i]/1e6)*model.FLOW*24*1000*model.Rate_DRA)

    model.Obj = pyo.Objective(expr=sum(power_terms)+sum(dra_terms), sense=pyo.minimize)

    # solve
    mgr = SolverManagerFactory('neos')
    sol = mgr.solve(model, solver='couenne', tee=False)
    model.solutions.load_from(sol)

    # extract
    result = {'total_cost': pyo.value(model.Obj)}
    for i in model.segments:
        key = stations[i-1]['name'].lower()
        result[f"num_pumps_{key}"]    = int(pyo.value(model.NOP[i]))
        result[f"speed_{key}"]         = float(pyo.value(model.N[i]))
        # assume eff var stored last in loop
        result[f"efficiency_{key}"]    = float(pyo.value(eff*100))
        result[f"power_cost_{key}"]    = float(pyo.value(power_terms[i-1]))
        result[f"dra_cost_{key}"]      = float(pyo.value(dra_terms[i-1]))
        result[f"residual_head_{key}"] = float(pyo.value(model.RH[i]))
    # terminal
    tkey = terminal['name'].lower()
    result[f"residual_head_{tkey}"] = float(pyo.value(model.RH[P+1]))

    return result
