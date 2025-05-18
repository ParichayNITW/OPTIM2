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
    model.FLOW = pyo.Param(initialize=float(FLOW))  # flow rate (m^3/s)
    model.KV = pyo.Param(initialize=KV)           # kinematic viscosity (cSt)
    model.rho = pyo.Param(initialize=rho)         # density (kg/m^3)
    # Station and pipe data
    N = len(stations)  # number of pipe segments
    model.Nodes = pyo.RangeSet(1, N+1)
    model.pump_stations = pyo.Set(initialize=[i for i, stn in enumerate(stations, start=1) if stn.get('Pump', False)])
    # Define parameters for pipe segments
    d_inner = {}
    length = {}
    roughness = {}
    elevation = {}
    inj_source = {}
    for i, stn in enumerate(stations, start=1):
        d_inner[i] = stn.get('D', 0.0) / 1000.0  # convert mm to m if needed
        length[i] = stn.get('L', 0.0)           # length in km
        roughness[i] = stn.get('e', 0.0)        # absolute roughness in m
        elevation[i] = stn.get('Elev', 0.0)     # elevation (m)
        inj_source[i] = stn.get('Inj', None)    # injection source (pump) index or None
    # Append terminal node elevation
    elevation[N+1] = terminal.get('Elev', 0.0)   # terminal node elevation
    
    # Assign elevations to Pyomo Param
    model.z = pyo.Param(model.Nodes, initialize=lambda model, i: elevation[i])
    # Residual head at each node (m)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=0.0)
    # Drag reduction variable for each pump station (in %)
    model.DR = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals, bounds=(0, RateDRA))
    # Other pump performance parameters (given or default)
    model.A = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('A', 0.0))
    model.B = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('B', 0.0))
    model.C = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('C', 0.0))
    model.DOL = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('DOL', 1.0))
    model.N = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('N', 1.0))
    # Pump performance curve coefficients
    model.Pcoef = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('Pcoef', 0.0))
    model.Qcoef = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('Qcoef', 0.0))
    model.Rcoef = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('Rcoef', 0.0))
    model.Scoef = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('Scoef', 0.0))
    model.Tcoef = pyo.Param(model.pump_stations, initialize=lambda model, i: stations[i-1].get('Tcoef', 0.0))
    # Pump indices and DRA costs if any
    pump_indices = []
    for i, stn in enumerate(stations, start=1):
        if stn.get('Pump', False):
            pump_indices.append(i)
            model.A[i] = stn.get('A', 0.0)
            model.B[i] = stn.get('B', 0.0)
            model.C[i] = stn.get('C', 0.0)
            model.DOL[i] = stn.get('DOL', 1.0)
            model.N[i] = stn.get('N', 1.0)
            model.Pcoef[i] = stn.get('Pcoef', 0.0)
            model.Qcoef[i] = stn.get('Qcoef', 0.0)
            model.Rcoef[i] = stn.get('Rcoef', 0.0)
            model.Scoef[i] = stn.get('Scoef', 0.0)
            model.Tcoef[i] = stn.get('Tcoef', 0.0)
    # Default values if not provided
    default_t = 0.0071374  # default wall thickness (m)
    default_e = 0.00004    # default pipe roughness (m)
    default_smys = 52000   # default SMYS (psi)
    default_design_factor = 0.72

    # Compute velocity, Reynolds number, and Darcy friction factor for each segment
    v = {}; Re = {}; f = {}
    flow_m3s = float(FLOW) / 3600.0 if FLOW is not None else 0.0  # flow rate in m^3/s
    for i in range(1, N+1):
        # Cross-sectional area (m^2) and flow velocity (m/s)
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        # Reynolds number (dimensionless)
        if KV and KV > 0:
            Re[i] = v[i] * d_inner[i] / (KV * 1e-6)
        else:
            Re[i] = 0.0
        # Darcy-Weisbach friction factor (Swamee–Jain equation, valid for turbulent flow, Re > 4000)
        if Re[i] <= 4000:
            raise ValueError("Reynolds number {} for segment {} is <= 4000; Swamee–Jain formula requires Re > 4000".format(Re[i], i))
        arg = (roughness[i] / (3.7 * d_inner[i])) + (5.74 / (Re[i]**0.9))
        f[i] = 0.25 / (log10(arg)**2)

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
            # Use drag reduction from the last pump station upstream (carryover)
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
                       model.Rcoef[i] * flow_eq_expr**2 + model.Scoef[i] * flow_eq_expr + model.Tcoef[i]) / 100.0
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
            speed_val = pyo.value(model.N[i])
            num_pumps_val = int(pyo.value(model.NOP[i]))
            eff_val = pyo.value(EFFP[i]) * 100.0
            # Compute station costs using solved values
            if i in electric_pumps:
                rate = elec_cost.get(i, 0.0)
                power_cost_val = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * pyo.value(TDH[i]) * 
                                  pyo.value(model.NOP[i])) / (3600.0 * 1000.0 * pyo.value(EFFP[i]) * 0.95) * 24.0 * rate
            else:
                sfc_val = sfc.get(i, 0.0) or 0.0
                power_cost_val = (pyo.value(model.rho) * pyo.value(model.FLOW) * 9.81 * pyo.value(TDH[i]) * 
                                  pyo.value(model.NOP[i])) / (3600.0 * 1000.0 * pyo.value(EFFP[i]) * 0.95) * ((sfc_val * 1.34102) / 820.0) * 24.0 * pyo.value(model.Price_HSD)
            dra_cost_val = (pyo.value(model.DR[i]) / 4.0) / 1e6 * pyo.value(model.FLOW) * 1000.0 * 24.0 * pyo.value(model.Rate_DRA)
        else:
            # No pump at this station
            speed_val = 0.0
            num_pumps_val = 0
            eff_val = 0.0
            power_cost_val = 0.0
            dra_cost_val = 0.0
        drag_reduction_val = pyo.value(model.DR[i]) if i in pump_indices else 0.0
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
