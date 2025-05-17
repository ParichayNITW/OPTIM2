import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# Ensure NEOS email is set via environment or secrets
if 'NEOS_EMAIL' not in os.environ:
    raise RuntimeError("NEOS_EMAIL environment variable not set")


def solve_pipeline(stations,
                   FLOW, KV, rho,
                   Rate_DRA, Price_HSD):
    """
    stations: list of dicts
      For i in 1..P (pumping stations):
        stations[i-1] contains keys:
          name, elev, D, t, SMYS, DF, L,
          min_rpm, max_rpm, max_pumps,
          power_source ("Grid"|"Diesel"), power_rate (if Grid), SFC (if Diesel),
          DR_max, A,B,C, P,Q,R,S,T
      Last entry is terminal: only name, elev
    """
    # Number of stations (last is terminal)
    N = len(stations)
    if N < 2:
        raise ValueError("At least one pumping station and one terminal required")
    model = pyo.ConcreteModel()
    # Indexes: 1..N for stations, 1..N-1 for segments/pumps
    model.stations = pyo.RangeSet(N)
    model.segments = pyo.RangeSet(N-1)

    # Global Params
    model.FLOW       = pyo.Param(initialize=FLOW)
    model.KV         = pyo.Param(initialize=KV)
    model.rho        = pyo.Param(initialize=rho)
    model.Rate_DRA   = pyo.Param(initialize=Rate_DRA)
    model.Price_HSD  = pyo.Param(initialize=Price_HSD)

    # Station-specific parameters
    elev_map = {i+1: stations[i]["elev"] for i in range(N)}
    model.z = pyo.Param(model.stations, initialize=elev_map)

    def seg_param(name, default=None):
        return {i: stations[i-1].get(name, default) for i in range(1, N)}

    model.L      = pyo.Param(model.segments, initialize=seg_param("L"))
    model.D      = pyo.Param(model.segments, initialize=seg_param("D"))
    model.t      = pyo.Param(model.segments, initialize=seg_param("t"))
    model.SMYS   = pyo.Param(model.segments, initialize=seg_param("SMYS"))
    model.DF     = pyo.Param(model.segments, initialize=seg_param("DF"))
    model.d      = pyo.Param(model.segments,
                              initialize={i: seg_param("D")[i] - 2*seg_param("t")[i]
                                          for i in range(1, N)})
    model.eps    = pyo.Param(model.segments, initialize=seg_param("eps", 0.00004))
    model.DR_max = pyo.Param(model.segments, initialize=seg_param("DR_max", 0))

    model.A = pyo.Param(model.segments, initialize=seg_param("A"))
    model.B = pyo.Param(model.segments, initialize=seg_param("B"))
    model.C = pyo.Param(model.segments, initialize=seg_param("C"))
    model.P = pyo.Param(model.segments, initialize=seg_param("P"))
    model.Q = pyo.Param(model.segments, initialize=seg_param("Q"))
    model.R = pyo.Param(model.segments, initialize=seg_param("R"))
    model.S = pyo.Param(model.segments, initialize=seg_param("S"))
    model.T = pyo.Param(model.segments, initialize=seg_param("T"))

    model.minRPM   = pyo.Param(model.segments, initialize=seg_param("min_rpm"))
    model.maxRPM   = pyo.Param(model.segments, initialize=seg_param("max_rpm"))
    model.maxPumps = pyo.Param(model.segments, initialize=seg_param("max_pumps"))

    power_src = seg_param("power_source", "Diesel")
    model.isElectric = pyo.Param(model.segments,
                                 initialize={i: 1 if power_src[i]=="Grid" else 0
                                             for i in range(1, N)})
    elec_rate = seg_param("power_rate", 0)
    model.ElecRate = pyo.Param(model.segments, initialize=elec_rate)
    sfc_map = seg_param("SFC", 0)
    model.SFC = pyo.Param(model.segments, initialize=sfc_map)

    # Decision Variables
    def rh_bounds(m, i):
        min_r = stations[i-1].get("min_residual", 50)
        if i == 1:
            return (min_r, min_r)
        return (min_r, None)
    model.RH = pyo.Var(model.stations,
                       domain=pyo.NonNegativeReals,
                       bounds=rh_bounds,
                       initialize={i: stations[i-1].get("min_residual", 50)
                                   for i in range(1, N+1)})
    def rpm_bounds(m, j):
        lo = int(pyo.value(model.minRPM[j]) + 9)//10
        hi = int(pyo.value(model.maxRPM[j]))//10
        return (lo, hi)
    model.N_u = pyo.Var(model.segments,
                        domain=pyo.NonNegativeIntegers,
                        bounds=rpm_bounds)
    model.N   = pyo.Expression(model.segments,
                                rule=lambda m, j: 10*m.N_u[j])
    model.NOP = pyo.Var(model.segments,
                        domain=pyo.NonNegativeIntegers,
                        bounds=lambda m, j: (0, int(pyo.value(model.maxPumps[j]))),
                        initialize=lambda m, j: 0)
    def dr_bounds(m, j):
        return (0, int(pyo.value(model.DR_max[j])//10))
    model.DR_u = pyo.Var(model.segments,
                         domain=pyo.NonNegativeIntegers,
                         bounds=dr_bounds,
                         initialize=lambda m, j: int(pyo.value(model.DR_max[j])//10))
    model.DR = pyo.Expression(model.segments,
                               rule=lambda m, j: 10*m.DR_u[j])

    # Terminal head constraint
    model.term_head = pyo.Constraint(expr=model.RH[N] == stations[-1]["elev"])

    # Hydraulics & Pump Equations
    def friction_factor(Re, eps, d):
        return 0.25/(log10((eps/d/3.7) + (5.74/(Re**0.9)))**2)

    OF_power = {}
    OF_dra   = {}
    for j in model.segments:
        # Hydraulic calculations
        vj   = model.FLOW/(3.414*model.d[j]**2/4)/3600
        Rej  = vj*model.d[j]/(model.KV*1e-6)
        fj   = friction_factor(Rej, model.eps[j], model.d[j])
        SHj  = model.RH[j+1] + (model.z[j+1] - model.z[j])
        DHj  = fj*(model.L[j]*1000/model.d[j])*(vj**2/(2*9.81))*(1 - model.DR[j]/100)
        # Pump head
        HD_j = (model.A[j]*model.FLOW**2
                + model.B[j]*model.FLOW
                + model.C[j]) * (model.N[j]/model.maxRPM[j])**2
        # Efficiency (fraction)
        eqj  = model.FLOW*model.maxRPM[j]/model.N[j]
        eff_frac = (model.P[j]*eqj**4
                    + model.Q[j]*eqj**3
                    + model.R[j]*eqj**2
                    + model.S[j]*eqj
                    + model.T[j]) / 100

        # Cost terms
        elec_term = (model.rho*model.FLOW*9.81*HD_j*model.NOP[j]) \
                   /(3600*1000*eff_frac*0.95) * 24 * model.ElecRate[j]
        diesel_term = ((model.rho*model.FLOW*9.81*HD_j*model.NOP[j]) \
                       /(3600*1000*eff_frac*0.95)) \
                      * (model.SFC[j]*1.34102/1000/820) * 1000 * 24 * model.Price_HSD

        OF_power[j] = pyo.Expression(expr=(model.isElectric[j]*elec_term
                                           + (1-model.isElectric[j])*diesel_term))
        OF_dra[j]   = pyo.Expression(expr=(model.DR[j]/1e6)*model.FLOW*24*1000*model.Rate_DRA)

        # Constraints
        model.add_component(f"c_head_{j}",
            pyo.Constraint(expr=(model.RH[j] + HD_j*model.NOP[j] >= SHj + DHj))
        )
        MAOPj = (2*model.t[j]*(model.SMYS[j]*0.070307)*model.DF[j]/model.D[j]) \
                * 10000 / model.rho
        model.add_component(f"c_maop_{j}",
            pyo.Constraint(expr=(model.RH[j] + HD_j*model.NOP[j] <= MAOPj))
        )

    # Objective
    model.Obj = pyo.Objective(expr=sum(OF_power[j] + OF_dra[j] for j in model.segments),
                               sense=pyo.minimize)

    # Solve
    sol_mgr = SolverManagerFactory('neos')
    results = sol_mgr.solve(model, solver='couenne', keepfiles=False, tee=False)
    model.solutions.load_from(results)

    # Extract
    res = {'total_cost': pyo.value(model.Obj)}
    for j in model.segments:
        key = f"seg_{j}"
        res[f"efficiency_{j}"] = pyo.value(eff_frac*100)
    return res
