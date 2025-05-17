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
    model.FLOW = pyo.Param(initialize=FLOW)       # flow rate (m^3/hr)
    model.KV = pyo.Param(initialize=KV)           # kinematic viscosity (cSt)
    model.rho = pyo.Param(initialize=rho)         # fluid density (kg/m^3)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)      # DRA cost (currency/L)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)   # diesel price (currency/L)
    # Index sets for stations and nodes
    N = len(stations)
    model.I = pyo.RangeSet(1, N)           # pumping station indices (segments)
    model.Nodes = pyo.RangeSet(1, N+1)     # node indices (including terminal node)
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
    default_t = 0.0071374   # default wall thickness (m)
    default_e = 0.00004     # default pipe roughness (m)
    default_smys = 52000    # default SMYS (psi)
    default_df = 0.72       # default design factor
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
            thickness[i] = stn.get('thickness', default_t)
        else:
            # No diameter info, use default values
            d_inner[i] = 0.697
            thickness[i] = default_t
        roughness[i] = stn.get('e', stn.get('roughness', default_e))
        smys[i] = stn.get('SMYS', stn.get('smys', default_smys))
        design_factor[i] = stn.get('DF', stn.get('df', default_df))
        elevation[i] = stn.get('z', stn.get('elevation', 0.0))
        # Determine if station has a pump
        max_pumps = stn.get('max_pumps', None)
        has_pump = True
        if (max_pumps is not None and max_pumps == 0) or ('A' not in stn or 'B' not in stn or 'C' not in stn):
            has_pump = False
        if has_pump:
            pump_indices.append(i)
            # Pump head curve coefficients
            Acoef[i] = stn['A']; Bcoef[i] = stn['B']; Ccoef[i] = stn['C']
            # Pump efficiency curve coefficients
            Pcoef[i] = stn['P']; Qcoef[i] = stn['Q']; Rcoef[i] = stn['R']
            Scoef[i] = stn['S']; Tcoef[i] = stn['T']
            # Min and max RPM
            min_rpm[i] = stn.get('MinRPM', stn.get('min_rpm', None))
            max_rpm[i] = stn.get('DOL', stn.get('max_rpm', None))
            if min_rpm[i] is None or max_rpm[i] is None:
                # Default RPM bounds if not provided
                min_rpm[i] = min_rpm.get(i, 1200)
                max_rpm[i] = max_rpm.get(i, 3000)
            # Fuel type and consumption
            if 'SFC' in stn or 'sfc' in stn:
                # Diesel-driven pump
                diesel_pumps.append(i)
                sfc[i] = stn.get('SFC', stn.get('sfc'))
            else:
                # Electric-driven pump
                electric_pumps.append(i)
                elec_cost[i] = stn.get('cost_per_kwh', stn.get('Cost_per_Kwh', 9.0))
        # Track injection source for segment i
        if has_pump:
            last_pump_idx = i
        inj_source[i] = last_pump_idx
    # Terminal node elevation
    elevation[N+1] = terminal.get('z', terminal.get('elevation', 0.0))
    # ---------------------
    # PARAMETERS
    # ---------------------
    model.L = pyo.Param(model.I, initialize=length)          # segment length (km)
    model.d = pyo.Param(model.I, initialize=d_inner)         # inner diameter (m)
    model.t = pyo.Param(model.I, initialize=thickness)       # wall thickness (m)
    model.e = pyo.Param(model.I, initialize=roughness)       # pipe roughness (m)
    model.SMYS = pyo.Param(model.I, initialize=smys)         # specified minimum yield strength (psi)
    model.DF = pyo.Param(model.I, initialize=design_factor)  # design factor
    model.z = pyo.Param(model.Nodes, initialize=elevation)   # node elevations (m)
    # Pump-specific parameters (only for pump stations)
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
        lb = 1 if j == 1 else 0
        ub = 3 if j == 1 else 2
        stn = stations[j-1]
        if 'max_pumps' in stn and stn['max_pumps'] is not None:
            ub = max(lb, stn['max_pumps'])
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=nop_bounds, initialize=1)
    # Pump speed (RPM) discretized in 10-unit steps
    speed_min = {}; speed_max = {}
    for j in pump_indices:
        min_val = int(pyo.value(model.MinRPM[j]) + 9) // 10
        max_val = int(pyo.value(model.DOL[j])) // 10
        if min_val < 1: min_val = 1
        if max_val < min_val: max_val = min_val
        speed_min[j] = min_val; speed_max[j] = max_val
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m, j: (speed_min[j], speed_max[j]),
                        initialize=lambda m, j: (speed_min[j] + speed_max[j]) // 2)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.N_u[j])  # actual pump RPM
    # Drag reduction (% friction reduction) discretized in 10% steps
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=(0, 4), initialize=4)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.DR_u[j])
    # Residual head at each node (m of fluid)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(50)  # fix initial station head (e.g. 50 m)
    for j in range(2, N+2):
        model.RH[j].setlb(50)  # minimum pressure constraint (50 m head)
    # ---------------------
    # HYDRAULIC RELATIONSHIPS
    # ---------------------
    g = 9.81  # gravitational acceleration (m/s^2)
    # Compute constant velocity, Reynolds number, friction factor for each segment
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        # Flow in m^3/s
        flow_m3s = pyo.value(model.FLOW) / 3600.0
        # Cross-sectional area (m^2)
        area = pi * (pyo.value(model.d[i])**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        Re[i] = v[i] * pyo.value(model.d[i]) / (pyo.value(model.KV) * 1e-6) if pyo.value(model.KV) > 0 else 0.0
        # Darcy friction factor (Swamee–Jain equation)
        arg = (pyo.value(model.e[i]) / pyo.value(model.d[i]) / 3.7) + (5.74 / ((Re[i] + 1e-16)**0.9))
        f[i] = 0.25 / (log10(arg)**2) if arg > 0 else 0.0
    # Expressions for head requirements and pump performance
    SH = {}; DH = {}; SDHR = {}; TDH = {}; EFFP = {}
    for i in range(1, N+1):
        # Static head difference (elevation change + downstream residual head)
        SH[i] = model.RH[i+1] + (pyo.value(model.z[i+1]) - pyo.value(model.z[i]))
        # Frictional head loss for segment i (m)
        if inj_source[i] is not None:
            j = inj_source[i]
            DR_frac = model.DR[j] / 100.0 if j in pump_indices else 0.0
        else:
            DR_frac = 0.0
        DH[i] = f[i] * ((pyo.value(model.L[i]) * 1000.0) / pyo.value(model.d[i])) * (v[i]**2 / (2 * g)) * (1 - DR_frac)
        # Total head required for segment i
        SDHR[i] = SH[i] + DH[i]
        # Pump head added by one pump (if station has pump)
        if i in pump_indices:
            TDH[i] = (model.A[i] * (pyo.value(model.FLOW)**2) + model.B[i] * pyo.value(model.FLOW) + model.C[i]) * ((model.N[i] / model.DOL[i])**2)
            # Pump efficiency (fraction) as a function of equivalent flow at design speed
            FLOW_eq = pyo.value(model.FLOW) * pyo.value(model.DOL[i]) / model.N[i]
            EFFP[i] = (model.Pcoef[i] * (FLOW_eq**4) + model.Qcoef[i] * (FLOW_eq**3) + model.Rcoef[i] * (FLOW_eq**2) + model.Scoef[i] * FLOW_eq + model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0
            EFFP[i] = 1.0
    # ---------------------
    # CONSTRAINTS
    # ---------------------
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        # Head balance: pump head + incoming head >= required head for segment
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i] * model.NOP[i] >= SDHR[i])
        else:
            model.head_balance.add(model.RH[i] >= SDHR[i])
        # Operating pressure: head at station outlet <= MAOP of that pipe segment
        D_out = pyo.value(model.d[i]) + 2 * pyo.value(model.t[i])  # approximate outer diameter (m)
        MAOP_head = (2 * pyo.value(model.t[i]) * (pyo.value(model.SMYS[i]) * 0.070307) * pyo.value(model.DF[i]) / D_out) * 10000.0 / pyo.value(model.rho)
        if i in pump_indices:
            model.pressure_limit.add(model.RH[i] + TDH[i] * model.NOP[i] <= MAOP_head)
        else:
            model.pressure_limit.add(model.RH[i] <= MAOP_head)
    # ---------------------
    # OBJECTIVE: MINIMIZE TOTAL COST
    # ---------------------
    total_cost_expr = 0.0
    for i in pump_indices:
        # Pumping power cost
        if i in electric_pumps:
            # Electric pump: cost per kWh
            rate = elec_cost.get(i, 0.0)
            total_power_kW = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * EFFP[i] * 0.95)
            power_cost = total_power_kW * 24.0 * rate
        else:
            # Diesel pump: use SFC and diesel price
            sfc_val = sfc.get(i, 0.0)  # (gm/bhp/hr)
            total_power_kW = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * EFFP[i] * 0.95)
            # Convert SFC to L/kWh: *1.34102 (hp to kW) /1000 (g->kg)/820 (kg->L) and multiply by 1000 to get L (since 820 kg per m^3)
            fuel_per_kWh = (sfc_val * 1.34102) / (1000.0 * 820.0) * 1000.0
            power_cost = total_power_kW * 24.0 * fuel_per_kWh * pyo.value(model.Price_HSD)
        # DRA cost
        dra_cost = (model.DR[i] / 4.0) / 1e6 * pyo.value(model.FLOW) * 1000.0 * 24.0 * pyo.value(model.Rate_DRA)
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
        speed_val = pyo.value(model.N[i]) if i in pump_indices else 0.0
        num_pumps_val = int(pyo.value(model.NOP[i])) if i in pump_indices else 0
        eff_val = (pyo.value(EFFP[i]) * 100.0) if i in pump_indices else 0.0
        if i in pump_indices:
            # Compute costs for this station using solved values
            if i in electric_pumps:
                rate = elec_cost.get(i, 0.0)
                power_cost_val = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * pyo.value(TDH[i]) * pyo.value(model.NOP[i])) / (3600.0 * 1000.0 * pyo.value(EFFP[i]) * 0.95) * 24.0 * rate
            else:
                sfc_val = sfc.get(i, 0.0)
                power_cost_val = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * pyo.value(TDH[i]) * pyo.value(model.NOP[i])) / (3600.0 * 1000.0 * pyo.value(EFFP[i]) * 0.95) * (sfc_val * 1.34102 / 1000.0 / 820.0) * 1000.0 * 24.0 * pyo.value(model.Price_HSD)
            dra_cost_val = (pyo.value(model.DR[i]) / 4.0) / 1e6 * pyo.value(model.FLOW) * 1000.0 * 24.0 * pyo.value(model.Rate_DRA)
        else:
            power_cost_val = 0.0
            dra_cost_val = 0.0
        drag_reduction_val = pyo.value(model.DR[i]) if i in pump_indices else 0.0
        head_loss_val = pyo.value(DH[i])
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
    # Terminal station results (no pump)
    term_name = terminal['name'].strip().lower()
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
    # Total cost (objective value)
    result['total_cost'] = pyo.value(model.Obj)
    return result
