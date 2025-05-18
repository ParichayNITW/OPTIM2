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
    model.I = pyo.RangeSet(1, N)       # segment indices (between station i and i+1)
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
    max_dr = {}  # max drag reduction per station
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
        has_pump = any(k in stn for k in ['A', 'B', 'C', 'P', 'Q', 'R', 'S', 'T']) or stn.get('pump', False)
        if has_pump:
            pump_indices.append(i)
            # Pump head curve coefficients (A, B, C)
            Acoef[i] = stn.get('A', stn.get('a', 0.0))
            Bcoef[i] = stn.get('B', stn.get('b', 0.0))
            Ccoef[i] = stn.get('C', stn.get('c', 0.0))
            # Pump efficiency curve coefficients (P, Q, R, S, T)
            Pcoef[i] = stn.get('P', stn.get('p', 0.0))
            Qcoef[i] = stn.get('Q', stn.get('q', 0.0))
            Rcoef[i] = stn.get('R', stn.get('r', 0.0))
            Scoef[i] = stn.get('S', stn.get('s', 0.0))
            Tcoef[i] = stn.get('T', stn.get('tcoef', 0.0))
            # Min and max pump speed (RPM)
            min_rpm[i] = stn.get('MinRPM', stn.get('min_rpm', None))
            max_rpm[i] = stn.get('DOL', stn.get('dol', None))
            # Fuel type and consumption
            if ('SFC' in stn and stn.get('SFC') not in (None, 0)) or ('sfc' in stn and stn.get('sfc') not in (None, 0)):
                # Diesel-driven pump: specific fuel consumption provided
                diesel_pumps.append(i)
                sfc[i] = stn.get('SFC', stn.get('sfc', 0.0))
            else:
                # Electric-driven pump
                electric_pumps.append(i)
                elec_cost[i] = stn.get('cost_per_kwh', stn.get('Cost_per_Kwh', 0.0))
            # Maximum drag reduction (%) for this pump station
            max_dr[i] = stn.get('max_dr', 40.0)
        # Track injection source for segment i
        if has_pump:
            last_pump_idx = i
        inj_source[i] = last_pump_idx

    # Terminal node elevation
    elevation[N+1] = terminal.get('z', terminal.get('elevation', 0.0))

    # ---------------------
    # PARAMETERS
    # ---------------------
    model.L = pyo.Param(model.I, initialize=length)        # segment length (km)
    model.d = pyo.Param(model.I, initialize=d_inner)       # inner diameter (m)
    model.t = pyo.Param(model.I, initialize=thickness)     # wall thickness (m)
    model.e = pyo.Param(model.I, initialize=roughness)     # pipe roughness (m)
    model.SMYS = pyo.Param(model.I, initialize=smys)       # SMYS (psi)
    model.DF = pyo.Param(model.I, initialize=design_factor) # design factor
    model.z = pyo.Param(model.Nodes, initialize=elevation) # elevations of nodes (m)

    # Pump-specific parameters (for stations that have pumps)
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

    # ---------------------
    # DECISION VARIABLES
    # ---------------------
    # Number of pumps in operation (integer)
    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0  # ensure at least one pump on at station 1
        ub = 3 if j == 1 else 2  # default max pumps (can override below)
        if 'max_pumps' in stations[j-1] and stations[j-1]['max_pumps'] is not None:
            ub = max(lb, stations[j-1]['max_pumps'])
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, 
                        bounds=nop_bounds, initialize=1)
    # Pump speed (RPM), discretized in 10-unit steps
    speed_min = {}; speed_max = {}
    for j in pump_indices:
        # Determine discrete speed bounds from min/max RPM
        min_val = 1
        max_val = 1
        if j in min_rpm and min_rpm[j] is not None:
            min_val = (int(min_rpm[j]) + 9) // 10
            if min_val < 1: min_val = 1
        if j in max_rpm and max_rpm[j] is not None:
            max_val = int(max_rpm[j]) // 10
            if max_val < min_val: max_val = min_val
        speed_min[j] = min_val
        speed_max[j] = max_val
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m, j: (speed_min[j], speed_max[j]),
                        initialize=lambda m, j: (speed_min[j] + speed_max[j]) // 2 if j in speed_min else 1)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.N_u[j])  # actual pump speed (RPM)
    # Drag reduction (% friction reduction), discretized in 10% steps
    dr_max = {j: int(max_dr.get(j, 40.0) / 10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                         bounds=lambda m, j: (0, min(4, dr_max[j])),
                         initialize=lambda m, j: min(4, dr_max[j]))
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.DR_u[j])
    # Residual head at each node (m of fluid column)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    # Set initial station head
    model.RH[1].fix(stations[0].get('min_residual', 50))
    # Intermediate nodes minimum residual head
    for j in range(2, N+1):
        model.RH[j].setlb(50)
    # Terminal node head
    if 'min_residual' in terminal:
        model.RH[N+1].fix(terminal['min_residual'])
    else:
        model.RH[N+1].setlb(50)

    # ---------------------
    # HYDRAULIC CALCULATIONS
    # ---------------------
    g = 9.81  # gravitational acceleration (m/s^2)
    v = {}; Re = {}; f = {}
    flow_m3s = pyo.value(model.FLOW) / 3600.0 if FLOW is not None else 0.0  # flow rate in m^3/s
    for i in range(1, N+1):
        # Cross-sectional area (m^2) and flow velocity (m/s)
        area = pi * (pyo.value(model.d[i])**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        # Reynolds number (dimensionless)
        if pyo.value(model.KV) > 0:
            Re[i] = v[i] * pyo.value(model.d[i]) / (pyo.value(model.KV) * 1e-6)
        else:
            Re[i] = 0.0
        # Darcy-Weisbach friction factor (laminar or Swamee–Jain)
        if Re[i] > 0:
            if Re[i] < 4000:
                f[i] = 64.0 / Re[i]
            else:
                arg = (pyo.value(model.e[i]) / pyo.value(model.d[i]) / 3.7) + (5.74 / ((Re[i] + 1e-16)**0.9))
                f[i] = 0.25 / (log10(arg)**2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    # ---------------------
    # HEAD REQUIREMENTS AND PUMP PERFORMANCE
    # ---------------------
    SH = {}; SDHR = {}; TDH = {}; EFFP = {}
    for i in range(1, N+1):
        # Static head (elevation difference + downstream residual head)
        SH[i] = model.RH[i+1] + (pyo.value(model.z[i+1]) - pyo.value(model.z[i]))
        # Frictional head loss (m)
        DR_frac = 0.0
        if inj_source.get(i) is not None:
            upstream = inj_source[i]
            if upstream in pump_indices:
                DR_frac = model.DR[upstream] / 100.0
        DH_loss = f[i] * ((length[i] * 1000.0) / d_inner[i]) * ((v[i]**2) / (2 * g)) * (1 - DR_frac)
        # Total discharge head required for segment i
        SDHR[i] = SH[i] + DH_loss
        # Pump head added by one pump at station i (if applicable)
        if i in pump_indices:
            TDH[i] = (model.A[i] * model.FLOW**2 + model.B[i] * model.FLOW + model.C[i]) * ((model.N[i] / model.DOL[i])**2)
            # Pump efficiency (fraction) as a function of equivalent flow at design speed
            flow_eq = model.FLOW * model.DOL[i] / model.N[i]
            EFFP[i] = (model.Pcoef[i] * flow_eq**4 + model.Qcoef[i] * flow_eq**3 +
                       model.Rcoef[i] * flow_eq**2 + model.Scoef[i] * flow_eq + model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # ---------------------
    # CONSTRAINTS
    # ---------------------
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        # Head balance: available head (incoming + pump boost) must meet or exceed required head
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
            rate = elec_cost.get(i, 0.0)
            total_power_kW = (model.rho * model.FLOW * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * EFFP[i] * 0.95)
            power_cost = total_power_kW * 24.0 * rate
        else:
            sfc_val = sfc.get(i, 0.0) or 0.0
            total_power_kW = (model.rho * model.FLOW * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * EFFP[i] * 0.95)
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
        name = stn['name'].strip().lower()
        # Number of pumps
        if i in pump_indices:
            num_pumps_val = int(pyo.value(model.NOP[i]))
        else:
            num_pumps_val = 0
        # Pump speed and efficiency
        if i in pump_indices and num_pumps_val > 0:
            speed_val = pyo.value(model.N[i])
            eff_val = pyo.value(EFFP[i]) * 100.0
        else:
            speed_val = 0.0
            eff_val = 0.0
        # Power cost
        if i in pump_indices and num_pumps_val > 0:
            if i in electric_pumps:
                rate = elec_cost.get(i, 0.0)
                total_power_kW = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * pyo.value(TDH[i]) * num_pumps_val) / (3600.0 * 1000.0 * pyo.value(EFFP[i]) * 0.95)
                power_cost_val = total_power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i, 0.0) or 0.0
                total_power_kW = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * pyo.value(TDH[i]) * num_pumps_val) / (3600.0 * 1000.0 * pyo.value(EFFP[i]) * 0.95)
                fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                power_cost_val = total_power_kW * 24.0 * fuel_per_kWh * pyo.value(model.Price_HSD)
        else:
            power_cost_val = 0.0
        # DRA cost (allowed even if no pumps)
        if i in pump_indices:
            dra_cost_val = (pyo.value(model.DR[i]) / 4.0) / 1e6 * pyo.value(model.FLOW) * 1000.0 * 24.0 * pyo.value(model.Rate_DRA)
        else:
            dra_cost_val = 0.0
        # Drag reduction
        if i in pump_indices:
            drag_reduction_val = pyo.value(model.DR[i])
        else:
            drag_reduction_val = 0.0
        # Head loss and residual head
        head_loss_val = pyo.value(SDHR[i] - (model.RH[i+1] + (pyo.value(model.z[i+1]) - pyo.value(model.z[i]))))
        residual_head_val = pyo.value(model.RH[i])
        velocity_val = v[i]
        reynolds_val = Re[i]
        # Populate results (flat keys)
        result[f"speed_{name}"] = speed_val
        result[f"num_pumps_{name}"] = num_pumps_val
        result[f"efficiency_{name}"] = eff_val
        result[f"power_cost_{name}"] = power_cost_val
        result[f"dra_cost_{name}"] = dra_cost_val
        result[f"drag_reduction_{name}"] = drag_reduction_val
        result[f"head_loss_{name}"] = head_loss_val
        result[f"residual_head_{name}"] = residual_head_val
        result[f"velocity_{name}"] = velocity_val
        result[f"reynolds_{name}"] = reynolds_val
        result[f"sdh_{name}"] = pyo.value(SDHR[i])
        # Pump coefficients and limits
        if i in pump_indices:
            result[f"coef_A_{name}"] = pyo.value(model.A[i])
            result[f"coef_B_{name}"] = pyo.value(model.B[i])
            result[f"coef_C_{name}"] = pyo.value(model.C[i])
            result[f"dol_{name}"] = pyo.value(model.DOL[i])
            result[f"min_rpm_{name}"] = pyo.value(model.MinRPM[i])
    # Terminal node outputs
    term_name = terminal.get('name', 'terminal').strip().lower()
    result[f"speed_{term_name}"] = 0.0
    result[f"num_pumps_{term_name}"] = 0
    result[f"efficiency_{term_name}"] = 0.0
    result[f"power_cost_{term_name}"] = 0.0
    result[f"dra_cost_{term_name}"] = 0.0
    result[f"drag_reduction_{term_name}"] = 0.0
    result[f"head_loss_{term_name}"] = 0.0
    result[f"residual_head_{term_name}"] = pyo.value(model.RH[N+1])
    result[f"velocity_{term_name}"] = 0.0
    result[f"reynolds_{term_name}"] = 0.0
    result[f"sdh_{term_name}"] = 0.0
    # Total network cost (objective value)
    result['total_cost'] = pyo.value(model.Obj)

    return result
