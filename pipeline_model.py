# pipeline_model.py (backend)

import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# Set NEOS credentials (required for remote solve)
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    """
    Build and solve the pipeline optimization model.
    """
    model = pyo.ConcreteModel()

    # Global fluid & cost parameters
    model.FLOW = pyo.Param(initialize=FLOW)        # flow (m^3/hr)
    model.KV = pyo.Param(initialize=KV)            # viscosity (cSt)
    model.rho = pyo.Param(initialize=rho)          # density (kg/m^3)
    model.Rate_DRA = pyo.Param(initialize=RateDRA) # DRA cost (currency/L)
    model.Price_HSD = pyo.Param(initialize=Price_HSD) # diesel price (currency/L)

    # Index sets: segments 1..N, nodes 1..N+1
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    # Prepare data dictionaries from input
    length = {}; d_inner = {}; thickness = {}; roughness = {}
    smys = {}; design_factor = {}; elevation = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    last_pump = None

    # Default values
    default_t = 0.0071374
    default_e = 0.00004
    default_smys = 52000
    default_df = 0.72

    # Process each station (segment i between station i and i+1)
    for i, stn in enumerate(stations, start=1):
        length[i] = stn.get('L', stn.get('length', 0.0))
        if 'D' in stn or 'diameter' in stn:
            D_out = stn.get('D', stn.get('diameter'))
            thickness[i] = stn.get('t', stn.get('thickness', default_t))
            d_inner[i] = D_out - 2 * thickness[i]
        elif 'd' in stn:
            d_inner[i] = stn['d']
            thickness[i] = stn.get('t', stn.get('thickness', default_t))
        else:
            d_inner[i] = 0.697
            thickness[i] = default_t

        roughness[i] = stn.get('e', stn.get('roughness', default_e))
        smys[i] = stn.get('SMYS', stn.get('smys', default_smys))
        design_factor[i] = stn.get('DF', stn.get('df', default_df))
        elevation[i] = stn.get('z', stn.get('elev', 0.0))

        # Pump curve coefficients
        Acoef[i] = stn.get('A', 0.0)
        Bcoef[i] = stn.get('B', 0.0)
        Ccoef[i] = stn.get('C', 0.0)
        Pcoef[i] = stn.get('P', 0.0)
        Qcoef[i] = stn.get('Q', 0.0)
        Rcoef[i] = stn.get('R', 0.0)
        Scoef[i] = stn.get('S', 0.0)
        Tcoef[i] = stn.get('T', 0.0)
        min_rpm[i] = stn.get('MinRPM', 0.0)
        max_rpm[i] = stn.get('DOL', stn.get('RatedRPM', 0.0))
        sfc[i] = stn.get('sfc', 0.0)
        elec_cost[i] = stn.get('rate', 0.0)

        # Identify pump stations
        if stn.get('is_pump', False):
            pump_indices.append(i)
            if stn.get('power_type', 'Diesel') == 'Diesel':
                diesel_pumps.append(i)
            else:
                electric_pumps.append(i)
            last_pump = i

    elevation[N+1] = terminal.get('z', terminal.get('elev', 0.0))

    # Convert lengths from km to m
    for i in length:
        if length[i] is not None:
            length[i] *= 1000.0

    # Create Pyomo parameters
    model.L = pyo.Param(model.I, initialize=length, default=0.0)
    model.d = pyo.Param(model.I, initialize=d_inner, default=0.0)
    model.e = pyo.Param(model.I, initialize=thickness, default=0.0)
    model.smys = pyo.Param(model.I, initialize=smys, default=0.0)
    model.df = pyo.Param(model.I, initialize=design_factor, default=0.0)
    model.z = pyo.Param(model.Nodes, initialize=elevation, default=0.0)

    # Define pump station set
    model.pump_stations = pyo.Set(initialize=pump_indices)
    model.A = pyo.Param(model.pump_stations, initialize=Acoef, default=0.0)
    model.B = pyo.Param(model.pump_stations, initialize=Bcoef, default=0.0)
    model.C = pyo.Param(model.pump_stations, initialize=Ccoef, default=0.0)
    model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef, default=0.0)
    model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef, default=0.0)
    model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef, default=0.0)
    model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef, default=0.0)
    model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef, default=0.0)
    model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm, default=0.0)
    model.DOL = pyo.Param(model.pump_stations, initialize=max_rpm, default=0.0)
    model.SFC = pyo.Param(model.pump_stations, initialize=sfc, default=0.0)
    model.Rate_elec = pyo.Param(model.pump_stations, initialize=elec_cost, default=0.0)

    # ---------------------
    # VARIABLES
    # ---------------------
    # Number of pumps at each station (integer), limited by available pumps
    nop_bounds = {}
    for idx in pump_indices:
        maxp = stations[idx-1].get('max_pumps', None)
        if maxp is not None:
            nop_bounds[idx] = (0, maxp)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=nop_bounds)

    # Pump speed (rpm)
    model.N = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)

    # Drag reduction index (0-4 corresponding to 0–40%)
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=(0,4))
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10 * m.DR_u[j])

    # Residual head at each node
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)

    # Apply fixed residual heads (user inputs)
    origin_val = stations[0].get('min_residual', 50.0)
    model.RH[1].fix(origin_val)               # fix origin residual head
    for j in range(2, N+1):
        model.RH[j].setlb(50.0)              # intermediate >= 50m
    term_val = terminal.get('min_residual', 50.0)
    model.RH[N+1].fix(term_val)             # fix terminal residual head

    # ---------------------
    # HYDRAULIC & PUMP EQUATIONS
    # ---------------------
    g = 9.81
    v = {}; Re = {}; f = {}
    flow_m3s = float(FLOW) / 3600.0 if FLOW is not None else 0.0

    for i in range(1, N+1):
        # Flow velocity
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0

        # Reynolds number
        if KV and KV > 0:
            Re[i] = v[i] * d_inner[i] / (KV * 1e-6)
        else:
            Re[i] = 0.0

        # Friction factor: Swamee-Jain if Re>4000, else laminar formula
        if Re[i] > 4000:
            arg = (roughness[i] / d_inner[i] / 3.7) + (5.74 / (Re[i]**0.9))
            f[i] = 0.25 / (log10(arg)**2) if arg > 0 else 0.0
        elif Re[i] > 0:
            f[i] = 64.0 / Re[i]
        else:
            f[i] = 0.0

    SDHR = {}; TDH = {}; EFFP = {}
    for i in range(1, N+1):
        # Static head (downstream elev diff + residual head)
        SH_expr = model.RH[i+1] + (model.z[i+1] - model.z[i])

        # Frictional (dynamic) head loss on segment
        DR_frac = 0
        if last_pump is not None and last_pump in pump_indices:
            # use carryover drag reduction if applicable
            DR_frac = model.DR[last_pump] / 100.0
        DH_expr = f[i] * (length[i] / d_inner[i]) * ((v[i]**2) / (2*g)) * (1 - DR_frac)

        SDHR[i] = SH_expr + DH_expr

        if i in pump_indices:
            # Total head a single pump can add at station i
            TDH[i] = (model.A[i]*model.FLOW**2 + model.B[i]*model.FLOW + model.C[i]) * ((model.N[i]/model.DOL[i])**2)
            flow_eq = model.FLOW * model.DOL[i] / model.N[i]
            EFFP[i] = (model.Pcoef[i]*flow_eq**4 + model.Qcoef[i]*flow_eq**3 +
                       model.Rcoef[i]*flow_eq**2 + model.Scoef[i]*flow_eq + model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # ---------------------
    # CONSTRAINTS
    # ---------------------
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        # Ensure available head >= required discharge head
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= SDHR[i])
        else:
            model.head_balance.add(model.RH[i] >= SDHR[i])

        # Pressure limit (MAOP) for segment i
        D_out = d_inner[i] + 2*thickness[i]
        MAOP_head = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/D_out)*10000.0/rho
        if i in pump_indices:
            model.pressure_limit.add(model.RH[i] + TDH[i]*model.NOP[i] <= MAOP_head)
        else:
            model.pressure_limit.add(model.RH[i] <= MAOP_head)

    # ---------------------
    # OBJECTIVE: MINIMIZE TOTAL COST
    # ---------------------
    total_cost = 0
    for i in pump_indices:
        # Compute pumping power cost per day
        power_kW = (model.rho * model.FLOW * 9.81 * TDH[i] * model.NOP[i])/(3600*1000*EFFP[i]*0.95)
        if i in electric_pumps:
            rate = elec_cost.get(i,0.0)
            cost_power = power_kW * 24.0 * rate
        else:
            sfc_val = sfc.get(i,0.0) or 0.0
            fuel_L_per_kWh = (sfc_val * 1.34102)/820.0
            cost_power = power_kW * 24.0 * fuel_L_per_kWh * model.Price_HSD

        # DRA cost per day
        cost_dra = (model.DR[i]/4.0)/1e6 * model.FLOW * 1000.0 * 24.0 * model.Rate_DRA

        total_cost += cost_power + cost_dra

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # Solve (using NEOS/Couenne)
    solver = SolverManagerFactory('neos')
    result = solver.solve(model, solver='couenne', tee=False)
    model.solutions.load_from(result)

    # ---------------------
    # EXTRACT RESULTS (flattened)
    # ---------------------
    sol = {}
    for i, stn in enumerate(stations, start=1):
        name = stn.get('name', f'Station{i}').strip().lower()
        # Pump station results
        if i in pump_indices:
            speed_val = pyo.value(model.N[i])
            num_val = int(pyo.value(model.NOP[i]))
            eff_pct = pyo.value(EFFP[i])*100.0
            # Calculate costs
            if i in electric_pumps:
                power_cost = (pyo.value(model.rho)*FLOW*9.81*pyo.value(TDH[i])*num_val)/(3600*1000*EFFP[i]*0.95)*24.0*elec_cost.get(i,0.0)
            else:
                sfc_val = sfc.get(i,0.0) or 0.0
                fuel_L_kWh = (sfc_val*1.34102)/820.0
                power_cost = (pyo.value(model.rho)*FLOW*9.81*pyo.value(TDH[i])*num_val)/(3600*1000*EFFP[i]*0.95)*24.0*fuel_L_kWh*Price_HSD
            dra_cost = (pyo.value(model.DR[i])/4.0)/1e6 * FLOW*1000.0*24.0*RateDRA
        else:
            speed_val = 0.0
            num_val = 0
            eff_pct = 0.0
            power_cost = 0.0
            dra_cost = 0.0

        head_loss = pyo.value(SDHR[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i])))
        resid_head = pyo.value(model.RH[i])
        velocity = v[i]
        reynolds = Re[i]
        # Calculate total discharge head (static + dynamic)
        elev_diff = pyo.value(model.z[i+1]) - pyo.value(model.z[i])
        resid_next = pyo.value(model.RH[i+1])
        sdh = head_loss + resid_next + elev_diff

        sol[f"speed_{name}"] = speed_val
        sol[f"num_pumps_{name}"] = num_val
        sol[f"efficiency_{name}"] = eff_pct
        sol[f"power_cost_{name}"] = power_cost
        sol[f"dra_cost_{name}"] = dra_cost
        sol[f"drag_reduction_{name}"] = pyo.value(model.DR[i]) if i in pump_indices else 0.0
        sol[f"head_loss_{name}"] = head_loss
        sol[f"residual_head_{name}"] = resid_head
        sol[f"velocity_{name}"] = velocity
        sol[f"reynolds_{name}"] = reynolds
        sol[f"sdh_{name}"] = sdh

    # Terminal node (no pump)
    term_name = terminal.get('name', 'terminal').strip().lower()
    sol[f"speed_{term_name}"] = 0.0
    sol[f"num_pumps_{term_name}"] = 0
    sol[f"efficiency_{term_name}"] = 0.0
    sol[f"power_cost_{term_name}"] = 0.0
    sol[f"dra_cost_{term_name}"] = 0.0
    sol[f"drag_reduction_{term_name}"] = 0.0
    sol[f"head_loss_{term_name}"] = 0.0
    sol[f"residual_head_{term_name}"] = pyo.value(model.RH[N+1])
    sol[f"velocity_{term_name}"] = 0.0
    sol[f"reynolds_{term_name}"] = 0.0
    sol[f"sdh_{term_name}"] = 0.0

    sol['total_cost'] = pyo.value(model.Obj)
    return sol
