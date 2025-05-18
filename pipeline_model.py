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

    # Index sets for stations and nodes
    N = len(stations)
    model.I = pyo.RangeSet(1, N)        # segment indices (1 between station i and i+1)
    model.Nodes = pyo.RangeSet(1, N+1) # node indices (including terminal node)

    # Data dictionaries for parameters
    length = {}; d_inner = {}; thickness = {}; roughness = {}
    smys = {}; design_factor = {}; elevation = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
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

        # Pump station data
        if stn.get('is_pump', False):
            pump_indices.append(i)
            # Save coefficients for pump head curve
            Acoef[i] = stn.get('A', stn.get('a', 0.0))
            Bcoef[i] = stn.get('B', stn.get('b', 0.0))
            Ccoef[i] = stn.get('C', stn.get('c', 0.0))
            # Save coefficients for pump efficiency curve
            Pcoef[i] = stn.get('P', stn.get('p', 0.0))
            Qcoef[i] = stn.get('Q', stn.get('q', 0.0))
            Rcoef[i] = stn.get('R', stn.get('r', 0.0))
            Scoef[i] = stn.get('S', stn.get('s', 0.0))
            Tcoef[i] = stn.get('T', stn.get('t_coef', 0.0))
            # Operating constraints and costs
            min_rpm[i] = stn.get('MinRPM', 0)
            max_rpm[i] = stn.get('DOL', 0)
            sfc[i] = stn.get('sfc', 0.0)
            elec_cost[i] = stn.get('rate', 0.0)
            if stn.get('power_type', '').lower().startswith('die'):
                diesel_pumps.append(i)
            else:
                electric_pumps.append(i)
            # Track drag reduction carryover
            last_pump_idx = i
        # Set injection source (for drag reduction carryover)
        inj_source[i] = last_pump_idx

    # Terminal node elevation and required residual head
    elevation[N+1] = terminal.get('elev', terminal.get('elevation', 0.0))

    # ---------------------
    # PARAMETERS
    # ---------------------
    model.z = pyo.Param(model.Nodes, initialize=elevation)
    model.length = pyo.Param(model.I, initialize=length)
    model.d_inner = pyo.Param(model.I, initialize=d_inner)
    model.thickness = pyo.Param(model.I, initialize=thickness)
    model.roughness = pyo.Param(model.I, initialize=roughness)
    model.smys = pyo.Param(model.I, initialize=smys)
    model.design_factor = pyo.Param(model.I, initialize=design_factor)

    # Pump curve and cost parameters for pump stations
    if pump_indices:
        model.A = pyo.Param(model.pump_stations, initialize=Acoef)
        model.B = pyo.Param(model.pump_stations, initialize=Bcoef)
        model.C = pyo.Param(model.pump_stations, initialize=Ccoef)
        model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef)
        model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef)
        model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef)
        model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef)
        model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef)
        model.sfc = pyo.Param(model.pump_stations, initialize=sfc)
        model.elec_cost = pyo.Param(model.pump_stations, initialize=elec_cost)

    # Number of pumps on each station (decision variables)
    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0  # ensure at least one pump on at first station
        ub = 3 if j == 1 else 2  # default max pumps (can be overridden)
        if 'max_pumps' in stations[j-1] and stations[j-1]['max_pumps'] is not None:
            ub = int(stations[j-1]['max_pumps'])
        return (lb, ub)

    model.pump_stations = pyo.Set(initialize=pump_indices)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=nop_bounds, initialize=1)

    # Pump speed (RPM), discretized in 10-unit steps
    speed_min = {}; speed_max = {}
    for j in pump_indices:
        min_val = int(min_rpm.get(j, 0)) // 10
        max_val = int(max_rpm.get(j, 0)) // 10
        if max_val < min_val: 
            max_val = min_val
        speed_min[j] = min_val
        speed_max[j] = max_val

    # Param for Design Operating Level (rated RPM)
    model.DOL = pyo.Param(model.pump_stations, initialize=max_rpm)

    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m, j: (speed_min[j], speed_max[j]),
                        initialize=lambda m, j: (speed_min[j] + speed_max[j]) // 2 if j in speed_min else 1)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.N_u[j])  # actual pump speed (RPM)

    # Drag reduction (% friction reduction), discretized in 10% steps
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=(0, 4), initialize=4)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.DR_u[j])

    # Residual head at each node (m of fluid column)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    # Set initial and terminal residual heads based on user input
    initial_RH = stations[0].get('min_residual', stations[0].get('RH', 50.0))
    model.RH[1].fix(initial_RH)
    terminal_RH = terminal.get('min_residual', 50.0)
    # Minimum residual head at intermediate nodes (enforce RH >= 50)
    for j in range(2, N+2):
        if j < N+1:
            model.RH[j].setlb(50.0)
        else:
            model.RH[j].fix(terminal_RH)

    # ---------------------
    # HYDRAULIC & PUMP EQUATIONS
    # ---------------------
    g = 9.81  # gravitational acceleration (m/s^2)
    # Compute velocity, Reynolds number, and friction factor for each segment
    v = {}; Re = {}; f = {}
    flow_m3s = float(FLOW) / 3600.0 if FLOW is not None else 0.0  # flow rate (m^3/s)

    for i in range(1, N+1):
        # Cross-sectional area (m^2) and flow velocity (m/s)
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0

        # Reynolds number (dimensionless)
        if KV and KV > 0:
            Re[i] = v[i] * d_inner[i] / (KV * 1e-6)
        else:
            Re[i] = 0.0

        # Darcy-Weisbach friction factor (different formula based on Re)
        if Re[i] > 4000:
            arg = (roughness[i] / d_inner[i] / 3.7) + (5.74 / ((Re[i] + 1e-16)**0.9))
            if arg > 0:
                f[i] = 0.25 / (log10(arg)**2)
            else:
                f[i] = 0.0
        elif Re[i] > 0:
            # Laminar flow friction factor
            f[i] = 64.0 / Re[i]
        else:
            f[i] = 0.0

    # Expressions for required head and pump performance
    SDHR = {}  # station discharge head required for each segment
    TDH = {}   # total dynamic head added by one pump (per pump station)
    EFFP = {}  # pump hydraulic efficiency (fraction) for each pump station

    for i in range(1, N+1):
        # Static head (elevation difference + downstream residual head)
        SH_expr = model.RH[i+1] + (model.z[i+1] - model.z[i])
        # Frictional head loss for segment i (m)
        DR_frac_expr = 0
        if inj_source[i] is not None and inj_source[i] in pump_indices:
            # Use drag reduction from the last pump station upstream
            DR_frac_expr = model.DR[inj_source[i]] / 100.0
        DH_expr = f[i] * ((length[i] * 1000.0) / d_inner[i]) * ((v[i]**2) / (2 * g)) * (1 - DR_frac_expr)
        # Total discharge head required for segment i
        SDHR[i] = SH_expr + DH_expr

        # Pump head added by one pump at station i (if applicable)
        if i in pump_indices:
            TDH[i] = (model.A[i] * model.FLOW**2 + model.B[i] * model.FLOW + model.C[i]) * ((model.N[i] / model.DOL[i])**2)
            # Pump efficiency (fraction) as a function of equivalent flow at design speed
            flow_eq_expr = model.FLOW * model.DOL[i] / model.N[i]
            EFFP[i] = (model.Pcoef[i] * flow_eq_expr**4 + model.Qcoef[i] * flow_eq_expr**3 +
                       model.Rcoef[i] * flow_eq_expr**2 + model.Scoef[i] * flow_eq_expr +
                       model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # ---------------------
    # CONSTRAINTS
    # ---------------------
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        # Head balance: available head (incoming + pump boost) >= required head
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i] * model.NOP[i] >= SDHR[i])
        else:
            model.head_balance.add(model.RH[i] >= SDHR[i])

        # Operating pressure limits (MAOP in head units) for segment i
        D_out = d_inner[i] + 2 * thickness[i]  # approximate outer diameter (m)
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho
        if i in pump_indices:
            model.pressure_limit.add(model.RH[i] + TDH[i] * model.NOP[i] <= MAOP_head)
        else:
            model.pressure_limit.add(model.RH[i] <= MAOP_head)

    # ---------------------
    # OBJECTIVE: MINIMIZE TOTAL COST
    # ---------------------
    total_cost_expr = 0
    for i in pump_indices:
        # Pumping power cost
        if i in electric_pumps:
            # Electric-driven pump: cost per kWh from elec_cost (currency per kWh)
            rate = elec_cost.get(i, 0.0)
            total_power_kW = (model.rho * model.FLOW * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * EFFP[i] * 0.95)
            power_cost = total_power_kW * 24.0 * rate
        else:
            # Diesel-driven pump: use SFC (gm/bhp/hr) and diesel price
            sfc_val = sfc.get(i, 0.0) or 0.0
            total_power_kW = (model.rho * model.FLOW * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * EFFP[i] * 0.95)
            # Convert SFC to fuel consumption in L/kWh (820 kg/m^3 diesel density)
            fuel_per_kWh = (sfc_val * 1.34102) / 820.0
            power_cost = total_power_kW * 24.0 * fuel_per_kWh * model.Price_HSD
        # DRA chemical cost
        dra_cost = (model.DR[i] / 4.0) / 1e6 * model.FLOW * 1000.0 * 24.0 * model.Rate_DRA
        total_cost_expr += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost_expr, sense=pyo.minimize)

    # Solve the MINLP using Couenne via NEOS
    neos = SolverManagerFactory('neos')
    results = neos.solve(model, solver='couenne', tee=False)
    model.solutions.load_from(results)

    # ---------------------
    # EXTRACT RESULTS
    # ---------------------
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower() if 'name' in stn else f'station{i}'
        # Pump station outputs
        if i in pump_indices:
            num_pumps_val = int(pyo.value(model.NOP[i]))
            if num_pumps_val > 0:
                speed_val = pyo.value(model.N[i])
                eff_val = pyo.value(EFFP[i]) * 100.0
                # Compute station costs using solved values
                if i in electric_pumps:
                    rate = elec_cost.get(i, 0.0)
                    power_cost_val = (pyo.value(model.rho) * pyo.value(model.FLOW) *
                                      9.81 * pyo.value(TDH[i]) * 
                                      pyo.value(model.NOP[i])) / (3600.0 * 1000.0 * 
                                      pyo.value(EFFP[i]) * 0.95) * 24.0 * rate
                else:
                    sfc_val = sfc.get(i, 0.0) or 0.0
                    power_cost_val = (pyo.value(model.rho) * pyo.value(model.FLOW) *
                                      9.81 * pyo.value(TDH[i]) * 
                                      pyo.value(model.NOP[i])) / (3600.0 * 1000.0 * 
                                      pyo.value(EFFP[i]) * 0.95) * ((sfc_val * 1.34102) / 820.0) * 24.0 * pyo.value(model.Price_HSD)
                dra_cost_val = (pyo.value(model.DR[i]) / 4.0) / 1e6 * pyo.value(model.FLOW) * 1000.0 * 24.0 * pyo.value(model.Rate_DRA)
            else:
                speed_val = 0.0
                eff_val = 0.0
                power_cost_val = 0.0
                dra_cost_val = 0.0
                num_pumps_val = 0
        else:
            speed_val = 0.0
            num_pumps_val = 0
            eff_val = 0.0
            power_cost_val = 0.0
            dra_cost_val = 0.0
        drag_reduction_val = pyo.value(model.DR[i]) if (i in pump_indices and num_pumps_val > 0) else 0.0
        # Dynamic head loss in segment i (m)
        head_loss_val = pyo.value(SDHR[i] - (model.RH[i+1] + (model.z[i+1] - model.z[i])))
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

    # Terminal node outputs (no pump at terminal)
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

    # Total network cost (objective value)
    result['total_cost'] = pyo.value(model.Obj)
    return result
