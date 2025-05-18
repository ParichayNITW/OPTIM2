import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

# Tell NEOS who you are (required for NEOS solver)
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'parichay.nitwarangal@gmail.com')

def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    """
    Build and solve the pipeline optimization model.
    """
    # Create Pyomo model
    model = pyo.ConcreteModel()
    # Global fluid properties
    model.FLOW = pyo.Param(initialize=FLOW)        # flow rate (m^3/hr)
    model.KV = pyo.Param(initialize=KV)            # kinematic viscosity (cSt)
    model.rho = pyo.Param(initialize=rho)          # fluid density (kg/m^3)
    model.Rate_DRA = pyo.Param(initialize=RateDRA) # DRA cost (currency/L)
    model.Price_HSD = pyo.Param(initialize=Price_HSD) # diesel price (currency/L)

    # Index sets for segments (between stations) and nodes (stations plus terminal)
    N = len(stations)
    model.I = pyo.RangeSet(1, N)        # segments 1..N (between station i and i+1)
    model.Nodes = pyo.RangeSet(1, N+1) # nodes 1..N+1 (including terminal node)

    # Data dictionaries for parameters
    length = {}; d_inner = {}; thickness = {}; roughness = {}
    smys = {}; design_factor = {}; elevation = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}; max_pumps = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    # Track last pump index for DRA carryover
    last_pump_idx = None
    inj_source = {}
    # Default values if not provided
    default_t = 0.0071374  # default wall thickness (m)
    default_e = 0.00004    # default pipe roughness (m)
    default_smys = 52000   # default SMYS (psi)
    default_df = 0.72      # default design factor

    # ---------------------
    # PROCESS INPUT DATA
    # ---------------------
    for i, stn in enumerate(stations, start=1):
        # Pipeline geometry for segment i -> i+1
        length[i] = stn.get('L', stn.get('length'))
        # Determine inner diameter and thickness
        if 'D' in stn or 'diameter' in stn:
            # Outer diameter provided
            D_out = stn.get('D', stn.get('diameter'))
            thickness[i] = stn.get('t', stn.get('thickness', default_t))
            d_inner[i] = D_out - 2 * thickness[i]
        elif 'd' in stn:
            # Inner diameter provided
            d_inner[i] = stn['d']
            thickness[i] = stn.get('t', stn.get('thickness', default_t))
        else:
            # No diameter info, use default values
            d_inner[i] = 0.697
            thickness[i] = default_t
        roughness[i] = stn.get('e', stn.get('roughness', default_e))
        smys[i] = stn.get('SMYS', stn.get('smys', default_smys))
        design_factor[i] = stn.get('DF', stn.get('df', default_df))
        elevation[i] = stn.get('z', stn.get('elevation', 0.0))

        # Determine if station has a pump
        has_pump = any(k in stn for k in ['A','B','C','P','Q','R','S','T']) or stn.get('pump', False)
        if has_pump:
            pump_indices.append(i)
            # Pump head curve coefficients
            Acoef[i] = stn.get('A', stn.get('a', 0.0))
            Bcoef[i] = stn.get('B', stn.get('b', 0.0))
            Ccoef[i] = stn.get('C', stn.get('c', 0.0))
            # Pump efficiency curve coefficients
            Pcoef[i] = stn.get('P', stn.get('p', 0.0))
            Qcoef[i] = stn.get('Q', stn.get('q', 0.0))
            Rcoef[i] = stn.get('R', stn.get('r', 0.0))
            Scoef[i] = stn.get('S', stn.get('s', 0.0))
            Tcoef[i] = stn.get('T', stn.get('tcoef', 0.0))
            # Min and max pump speed
            min_rpm[i] = stn.get('MinRPM', stn.get('min_rpm', 0))
            max_rpm[i] = stn.get('DOL', stn.get('dol', None))
            # Maximum pumps at station (use given or default)
            max_pumps[i] = stn.get('max_pumps', (3 if i == 1 else 2))
            # Pump type and fuel data
            if (stn.get('SFC') not in (None,0)) or (stn.get('sfc') not in (None,0)):
                # Diesel pump
                diesel_pumps.append(i)
                sfc[i] = stn.get('SFC', stn.get('sfc', 0.0))
            else:
                # Electric pump
                electric_pumps.append(i)
                elec_cost[i] = stn.get('cost_per_kwh', stn.get('Cost_per_Kwh', 9.0))

        # Track injection source for segment i (last pump upstream)
        if has_pump:
            last_pump_idx = i
        inj_source[i] = last_pump_idx

    # Terminal node elevation
    elevation[N+1] = terminal.get('z', terminal.get('elevation', 0.0))

    # ---------------------
    # PARAMETERS
    # ---------------------
    model.L = pyo.Param(model.I, initialize=length)      # segment length (km)
    model.d = pyo.Param(model.I, initialize=d_inner)     # inner diameter (m)
    model.t = pyo.Param(model.I, initialize=thickness)   # wall thickness (m)
    model.e = pyo.Param(model.I, initialize=roughness)   # pipe roughness (m)
    model.SMYS = pyo.Param(model.I, initialize=smys)     # SMYS (psi)
    model.DF = pyo.Param(model.I, initialize=design_factor) # design factor
    model.z = pyo.Param(model.Nodes, initialize=elevation)   # elevations of nodes (m)

    # Pump-specific parameters (for stations with pumps)
    model.pump_stations = pyo.Set(initialize=pump_indices)
    if pump_indices:
        model.A = pyo.Param(model.pump_stations, initialize=Acoef)
        model.B = pyo.Param(model.pump_stations, initialize=Bcoef)
        model.C = pyo.Param(model.pump_stations, initialize=Ccoef)
        model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef)
        model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef)
        model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef)
        model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef)
        model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef)
        model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm)
        model.DOL = pyo.Param(model.pump_stations, initialize=max_rpm)
        model.MaxP = pyo.Param(model.pump_stations, initialize=max_pumps)

    # ---------------------
    # DECISION VARIABLES
    # ---------------------
    # Number of pumps in operation
    def nop_bounds(m,j):
        # At least one pump at station 1
        lb = 1 if j == 1 else 0
        # Upper bound from data (MaxP)
        ub = model.MaxP[j] if (j in model.MaxP.index_set()) else (3 if j==1 else 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)

    # Pump speed (discrete 10 RPM steps)
    speed_min = {}; speed_max = {}
    for j in pump_indices:
        min_val = 0
        max_val = 0
        if j in min_rpm and min_rpm[j] is not None:
            min_val = int(min_rpm[j]) // 10
        if j in max_rpm and max_rpm[j] is not None:
            max_val = int(max_rpm[j]) // 10
        if max_val < min_val:
            max_val = min_val
        speed_min[j] = min_val
        speed_max[j] = max_val
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (speed_min[j], speed_max[j]),
                        initialize=lambda m,j: (speed_min[j] + speed_max[j])//2 if j in speed_min else 0)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10 * m.N_u[j])  # actual RPM

    # Drag reduction (percent, discrete 10% steps)
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=0)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10 * m.DR_u[j])

    # Pump active binary
    model.y = pyo.Var(model.pump_stations, domain=pyo.Binary, initialize=1)

    # Pump efficiency (fraction)
    model.EFFP = pyo.Var(model.pump_stations, bounds=(0,1))

    # Residual head at nodes (m)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(50)  # fix source head at 50 m (example)
    for j in range(2, N+2):
        model.RH[j].setlb(50)  # minimum residual head 50 m

    # ---------------------
    # LINKING CONSTRAINTS
    # ---------------------
    model.link = pyo.ConstraintList()
    for j in model.pump_stations:
        # If no pump active, NOP=0 => N=0, DR=0, EFFP=0
        model.link.add(model.NOP[j] <= model.MaxP[j] * model.y[j])
        model.link.add(model.NOP[j] >= model.y[j])
        model.link.add(model.N[j] <= model.DOL[j] * model.y[j])
        model.link.add(model.N[j] >= model.MinRPM[j] * model.y[j])
        model.link.add(model.DR_u[j] <= 4 * model.y[j])
        model.link.add(model.EFFP[j] <= model.y[j])

    # Pump head (TDH) expression per pump
    model.TDH = pyo.Expression(model.pump_stations,
        rule=lambda m, j: (m.A[j]*m.FLOW**2 + m.B[j]*m.FLOW + m.C[j]) * ((m.N[j]/m.DOL[j])**2)
    )

    # Efficiency formula linking constraint (active only if pump on)
    def eff_rule(m, j):
        flow_eq = m.FLOW * m.DOL[j] / (m.N[j] + (1-m.y[j]) * m.DOL[j])
        return m.EFFP[j]*100*m.y[j] == (m.Pcoef[j]*flow_eq**4 + m.Qcoef[j]*flow_eq**3 +
                                       m.Rcoef[j]*flow_eq**2 + m.Scoef[j]*flow_eq + m.Tcoef[j]) * m.y[j]
    model.eff_con = pyo.Constraint(model.pump_stations, rule=eff_rule)

    # ---------------------
    # HYDRAULIC & PUMP EQUATIONS
    # ---------------------
    g = 9.81  # gravity (m/s^2)
    v = {}; Re = {}; f = {}
    flow_m3s = float(FLOW)/3600.0 if FLOW is not None else 0.0

    for i in range(1, N+1):
        # Flow velocity and Reynolds number for segment i
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        if KV and KV > 0:
            Re[i] = v[i] * d_inner[i] / (KV * 1e-6)
        else:
            Re[i] = 0.0

        # Darcy-Weisbach friction factor
        if Re[i] > 4000:
            arg = (roughness[i]/d_inner[i]/3.7) + (5.74/((Re[i]+1e-16)**0.9))
            if arg > 0:
                f[i] = 0.25/(log10(arg)**2)
            else:
                f[i] = 0.0
        elif Re[i] > 0:
            f[i] = 64.0/Re[i]
        else:
            f[i] = 0.0

    # Required head (static + friction) for each segment
    SDHR = {}
    for i in range(1, N+1):
        # Static head (downstream residual + elevation difference)
        SH = model.RH[i+1] + (model.z[i+1] - model.z[i])
        # Friction head loss
        DR_frac = 0
        if (inj_source[i] is not None) and (inj_source[i] in pump_indices):
            DR_frac = model.DR[inj_source[i]] / 100.0
        DH = f[i] * ((length[i]*1000.0)/d_inner[i]) * ((v[i]**2)/(2*g)) * (1 - DR_frac)
        SDHR[i] = SH + DH

    # ---------------------
    # CONSTRAINTS
    # ---------------------
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        # Head balance
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + model.TDH[i]*model.NOP[i] >= SDHR[i])
        else:
            model.head_balance.add(model.RH[i] >= SDHR[i])

        # Pressure (MAOP) limits
        D_out = d_inner[i] + 2*thickness[i]  # outer diameter
        MAOP_head = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/D_out)*10000.0/rho
        if i in pump_indices:
            model.pressure_limit.add(model.RH[i] + model.TDH[i]*model.NOP[i] <= MAOP_head)
        else:
            model.pressure_limit.add(model.RH[i] <= MAOP_head)

    # ---------------------
    # OBJECTIVE: MINIMIZE TOTAL COST
    # ---------------------
    total_cost = 0
    for i in pump_indices:
        # Pumping power cost
        if i in electric_pumps:
            rate = elec_cost.get(i, 0.0)
            total_power_kW = (model.rho*model.FLOW*9.81*model.TDH[i]*model.NOP[i])/(3600*1000*model.EFFP[i]*0.95)
            power_cost = total_power_kW * 24 * rate
        else:
            sfc_val = sfc.get(i, 0.0) or 0.0
            total_power_kW = (model.rho*model.FLOW*9.81*model.TDH[i]*model.NOP[i])/(3600*1000*model.EFFP[i]*0.95)
            fuel_per_kWh = (sfc_val * 1.34102)/820.0
            power_cost = total_power_kW * 24 * fuel_per_kWh * model.Price_HSD

        # DRA chemical cost
        dra_cost = (model.DR[i]/4.0)/1e6 * model.FLOW * 1000.0 * 24.0 * model.Rate_DRA
        total_cost += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # Solve MINLP with Couenne on NEOS
    neos = SolverManagerFactory('neos')
    results = neos.solve(model, solver='couenne', tee=False)
    model.solutions.load_from(results)

    # ---------------------
    # EXTRACT RESULTS
    # ---------------------
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn.get('name', f'station{i}').strip().lower()
        if i in pump_indices:
            speed_val = pyo.value(model.N[i])
            num_pumps_val = int(pyo.value(model.NOP[i]))
            eff_val = 100.0 * pyo.value(model.EFFP[i])
            # Compute costs using solved values
            if i in electric_pumps:
                rate = elec_cost.get(i, 0.0)
                power_cost_val = ((pyo.value(model.rho)*pyo.value(model.FLOW)*9.81*
                                  pyo.value(model.TDH[i])*pyo.value(model.NOP[i]))/
                                  (3600*1000*pyo.value(model.EFFP[i])*0.95)) * 24.0 * rate
            else:
                sfc_val = sfc.get(i, 0.0) or 0.0
                power_cost_val = ((pyo.value(model.rho)*pyo.value(model.FLOW)*9.81*
                                  pyo.value(model.TDH[i])*pyo.value(model.NOP[i]))/
                                  (3600*1000*pyo.value(model.EFFP[i])*0.95)) * ((sfc_val*1.34102)/820.0) * 24.0 * pyo.value(model.Price_HSD)
            dra_cost_val = ((pyo.value(model.DR[i])/4.0)/1e6) * pyo.value(model.FLOW)*1000.0*24.0 * pyo.value(model.Rate_DRA)
        else:
            speed_val = 0.0
            num_pumps_val = 0
            eff_val = 0.0
            power_cost_val = 0.0
            dra_cost_val = 0.0

        drag_reduction_val = pyo.value(model.DR[i]) if i in pump_indices else 0.0
        head_loss_val = pyo.value(SDHR[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i])))
        residual_head_val = pyo.value(model.RH[i])
        velocity_val = v[i]
        reynolds_val = Re[i]

        result[name] = {
            'speed': speed_val,
            'num_pumps': num_pumps_val,
            'efficiency': eff_val,
            'power_cost': power_cost_val,
            'dra_cost': dra_cost_val,
            'drag_reduction': drag_reduction_val,
            'head_loss': head_loss_val,
            'residual_head': residual_head_val,
            'velocity': velocity_val,
            'reynolds_number': reynolds_val
        }

    # Terminal node (no pump)
    term_name = terminal.get('name', 'terminal').strip().lower()
    result[term_name] = {
        'speed': 0.0,
        'num_pumps': 0,
        'efficiency': 0.0,
        'power_cost': 0.0,
        'dra_cost': 0.0,
        'drag_reduction': 0.0,
        'head_loss': 0.0,
        'residual_head': pyo.value(model.RH[N+1]),
        'velocity': 0.0,
        'reynolds_number': 0.0
    }

    # Total cost
    result['total_cost'] = pyo.value(model.Obj)
    return result
