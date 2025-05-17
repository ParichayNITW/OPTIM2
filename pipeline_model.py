import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# NEOS solver email
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'parichay.nitwarangal@gmail.com')

def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    model = pyo.ConcreteModel()

    # ───── GLOBAL PARAMS ─────
    model.FLOW       = pyo.Param(initialize=FLOW)
    model.KV         = pyo.Param(initialize=KV)
    model.rho        = pyo.Param(initialize=rho)
    model.Rate_DRA   = pyo.Param(initialize=Rate_DRA)
    model.Price_HSD  = pyo.Param(initialize=Price_HSD)

    N = len(stations)
    model.I     = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    # ───── READ STATION DATA ─────
    length, d_inner, thickness, roughness, elevation = {},{},{},{},{}
    smys_dict, df_dict = {},{}
    has_pump, max_pumps_dict = {},{}
    Acoef,Bcoef,Ccoef = {},{},{}
    Pcoef,Qcoef,Rcoef,Scoef,Tcoef = {},{},{},{},{}
    min_rpm, max_rpm = {},{}

    for i, stn in enumerate(stations, start=1):
        # geometry + fluid path
        length[i]    = stn['L']
        d_inner[i]   = stn['D'] - 2*stn['t']
        thickness[i] = stn['t']
        roughness[i] = stn['rough']
        elevation[i] = stn['elev']
        # structural parameters
        smys_dict[i] = stn.get('SMYS', 52000.0)
        df_dict[i]   = stn.get('DF',   0.72)
        # pump?
        if stn.get('is_pump', False):
            has_pump[i]        = True
            max_pumps_dict[i]  = stn['max_pumps']
            Acoef[i] = stn['A']; Bcoef[i] = stn['B']; Ccoef[i] = stn['C']
            Pcoef[i] = stn['P']; Qcoef[i] = stn['Q']
            Rcoef[i] = stn['R']; Scoef[i] = stn['S']; Tcoef[i] = stn['T']
            min_rpm[i] = stn['MinRPM']; max_rpm[i] = stn['DOL']
        else:
            has_pump[i] = False

    # terminal
    elevation[N+1] = terminal['elevation']
    min_residual   = terminal['min_residual']

    # ───── PARAMETERS ─────
    model.L    = pyo.Param(model.I, initialize=length)
    model.d    = pyo.Param(model.I, initialize=d_inner)
    model.t    = pyo.Param(model.I, initialize=thickness)
    model.e    = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys_dict)    # ← added
    model.DF   = pyo.Param(model.I, initialize=df_dict)      # ← added
    model.z    = pyo.Param(model.Nodes, initialize=elevation)

    pump_idxs = [i for i in model.I if has_pump[i]]
    model.PUMPS = pyo.Set(initialize=pump_idxs)

    if pump_idxs:
        model.max_pumps = pyo.Param(model.PUMPS, initialize=max_pumps_dict)
        model.A         = pyo.Param(model.PUMPS, initialize=Acoef)
        model.B         = pyo.Param(model.PUMPS, initialize=Bcoef)
        model.C         = pyo.Param(model.PUMPS, initialize=Ccoef)
        model.Pp        = pyo.Param(model.PUMPS, initialize=Pcoef)
        model.Qp        = pyo.Param(model.PUMPS, initialize=Qcoef)
        model.Rp        = pyo.Param(model.PUMPS, initialize=Rcoef)
        model.Sp        = pyo.Param(model.PUMPS, initialize=Scoef)
        model.Tp        = pyo.Param(model.PUMPS, initialize=Tp:=Tcoef)
        model.MinRPM    = pyo.Param(model.PUMPS, initialize=min_rpm)
        model.DOL       = pyo.Param(model.PUMPS, initialize=max_rpm)

    # ───── VARIABLES ─────
    def bounds_nop(m,i):
        return (0, m.max_pumps[i]) if i in m.PUMPS else (0,0)
    model.NOP = pyo.Var(model.I,
                        domain=pyo.NonNegativeIntegers,
                        bounds=bounds_nop,
                        initialize=0)

    def bounds_n(m,i):
        if i in m.PUMPS:
            lo = int(pyo.value(m.MinRPM[i]) + 9)//10
            hi = int(pyo.value(m.DOL[i]))   //10
            return (lo, hi)
        return (0,0)
    model.N_u = pyo.Var(model.I,
                        domain=pyo.NonNegativeIntegers,
                        bounds=bounds_n,
                        initialize=lambda m,i: bounds_n(m,i)[0])
    model.N   = pyo.Expression(model.I, rule=lambda m,i: 10*m.N_u[i])

    model.DR_u = pyo.Var(model.I,
                         domain=pyo.NonNegativeIntegers,
                         bounds=(0,4),
                         initialize=4)
    model.DR   = pyo.Expression(model.I, rule=lambda m,i: 10*m.DR_u[i])

    model.RH = pyo.Var(model.Nodes,
                       domain=pyo.NonNegativeReals,
                       initialize=min_residual)
    model.term_head = pyo.Constraint(expr=model.RH[N+1] == min_residual)

    # ───── HYDRAULICS ─────
    g = 9.81
    v, Re, f = {},{},{}
    for i in model.I:
        flow_m3s = pyo.value(model.FLOW)/3600.0
        A_cs     = pi*(pyo.value(model.d[i])**2)/4.0
        v[i]      = flow_m3s/A_cs if A_cs>0 else 0.0
        Re[i]     = v[i]*pyo.value(model.d[i])/(pyo.value(model.KV)*1e-6)
        arg       = pyo.value(model.e[i])/pyo.value(model.d[i])/3.7 + 5.74/( (Re[i]+1e-16)**0.9 )
        f[i]      = 0.25/(log10(arg)**2) if arg>0 else 0.0

    SDHR, TDH, EFFP = {},{},{}
    for i in model.I:
        SH   = model.RH[i+1] + (model.z[i+1] - model.z[i])
        DH   = f[i]*(model.L[i]*1000/model.d[i])*(v[i]**2/(2*g))*(1 - model.DR[i]/100)
        SDHR[i] = SH + DH
        if i in model.PUMPS:
            TDH[i] = (model.A[i]*model.FLOW**2 + model.B[i]*model.FLOW + model.C[i]) \
                     * (model.N[i]/model.DOL[i])**2
            flow_eq = model.FLOW*model.DOL[i]/model.N[i]
            EFFP[i] = (model.Pp[i]*flow_eq**4 + model.Qp[i]*flow_eq**3 +
                       model.Rp[i]*flow_eq**2 + model.Sp[i]*flow_eq + model.Tp[i]) / 100.0
        else:
            TDH[i]  = 0.0
            EFFP[i] = 1.0

    # ───── CONSTRAINTS ─────
    model.head_bal = pyo.ConstraintList()
    model.press_lim= pyo.ConstraintList()
    for i in model.I:
        if i in model.PUMPS:
            model.head_bal.add(model.RH[i] + TDH[i]*model.NOP[i] >= SDHR[i])
        else:
            model.head_bal.add(model.RH[i] >= SDHR[i])

        D_out = model.d[i] + 2*model.t[i]
        MAOP_h = (2*model.t[i]*(model.SMYS[i]*0.070307)*model.DF[i]/D_out)*10000.0/model.rho
        if i in model.PUMPS:
            model.press_lim.add(model.RH[i] + TDH[i]*model.NOP[i] <= MAOP_h)
        else:
            model.press_lim.add(model.RH[i] <= MAOP_h)

    # ───── OBJECTIVE ─────
    total_cost = 0.0
    for i in model.PUMPS:
        pow_kW = (model.rho*model.FLOW*9.81*TDH[i]*model.NOP[i]) \
                 /(3600*1000*EFFP[i]*0.95)
        if stations[i-1]['power_type']=='Grid':
            cost_pow = pow_kW*24*stations[i-1]['rate']
        else:
            sfc      = stations[i-1]['sfc']
            fuel_LpkW= (sfc*1.34102)/(1000*820)*1000
            cost_pow = pow_kW*24*fuel_LpkW*Price_HSD
        dra_cost = (model.DR[i]/4)/1e6*model.FLOW*1000*24*Rate_DRA
        total_cost += cost_pow + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # ───── SOLVE ─────
    mgr = SolverManagerFactory('neos')
    sol = mgr.solve(model, solver='couenne', tee=False)
    model.solutions.load_from(sol)

    # ───── EXTRACT ─────
    out = {}
    for i, stn in enumerate(stations, start=1):
        key = stn['name'].strip().lower()
        out[key] = {
          'num_pumps':      int(pyo.value(model.NOP[i])),
          'speed':          float(pyo.value(model.N[i])),
          'efficiency':     float(EFFP[i]*100),
          'power_cost':     float((model.rho*model.FLOW*9.81*TDH[i]*model.NOP[i]) \
                                 /(3600*1000*EFFP[i]*0.95)*24 \
                                 * (stations[i-1]['rate']
                                    if stn['power_type']=='Grid'
                                    else ((stn['sfc']*1.34102)/(1000*820)*1000*24*Price_HSD))),
          'dra_cost':       float((model.DR[i]/4)/1e6*model.FLOW*1000*24*Rate_DRA),
          'head_loss':      float(DH[i]),
          'residual_head':  float(pyo.value(model.RH[i])),
          'velocity':       float(v[i]),
          'reynolds_number':float(Re[i]),
          'drag_reduction': float(pyo.value(model.DR[i]))
        }
    # terminal
    tkey = terminal['name'].strip().lower()
    out[tkey] = {
        'num_pumps':0, 'speed':0, 'efficiency':0,
        'power_cost':0, 'dra_cost':0,
        'head_loss':0,
        'residual_head': float(pyo.value(model.RH[N+1])),
        'velocity':0, 'reynolds_number':0, 'drag_reduction':0
    }
    out['total_cost'] = float(pyo.value(model.Obj))
    return out
