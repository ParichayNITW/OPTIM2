import pyomo.environ as pyo
import pandas as pd
import os
import json

# Load DRA performance curves into memory
# Each CSV contains columns: '%Drag Reduction' and 'PPM'
# We create a dictionary mapping fluid viscosity (as int) to a DataFrame of that curve.
dra_curves = {}
viscosity_values = [10, 15, 20, 25, 30, 35, 40]
for visc in viscosity_values:
    file_name = f"{visc} cst.csv"
    # Determine file path relative to this script for reliability
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        df = pd.read_csv(file_path)
        # Ensure columns are as expected
        if df.columns.tolist()[:2] != ['%Drag Reduction', 'PPM']:
            raise ValueError(f"Unexpected columns in {file_name}: {df.columns.tolist()}")
        # Sort by %Drag Reduction for safe interpolation
        df = df.sort_values('%Drag Reduction')
        dra_curves[visc] = df
    except FileNotFoundError:
        # If a file is missing, we alert (the model can still run for other viscosities if needed).
        dra_curves[visc] = None
        print(f"Warning: DRA data file '{file_name}' not found. Curve data for {visc} cst will be unavailable.")

def get_ppm_for_dr(dr_percent, fluid_visc):
    """
    Retrieve the required PPM of DRA to achieve a given drag reduction percentage
    for the specified fluid viscosity (in cSt). Interpolates between nearest data points if needed.
    Returns 0 for dr_percent = 0.
    """
    if dr_percent <= 0:
        return 0.0
    # Ensure we have the data for this viscosity
    if fluid_visc not in dra_curves or dra_curves[fluid_visc] is None:
        raise ValueError(f"No DRA curve data available for fluid viscosity {fluid_visc} cst.")
    df = dra_curves[fluid_visc]
    # Use pandas interpolation to find PPM at the exact drag reduction percentage
    # If percentage is outside the range in data, clip to min/max of available data.
    dr_value = float(dr_percent)
    if dr_value < df['%Drag Reduction'].min():
        ppm = df.iloc[0]['PPM']
    elif dr_value > df['%Drag Reduction'].max():
        ppm = df.iloc[-1]['PPM']
    else:
        # Interpolate within the DataFrame range
        ppm = float(pd.Series(df['PPM'].values, index=df['%Drag Reduction']).reindex(
                    df['%Drag Reduction'].tolist() + [dr_value]).sort_index().interpolate(method='index').loc[dr_value])
    return ppm

def solve_pipeline(flow=2000.0,                    # target flow in m3/h through the pipeline
                   lengths_km=None,               # list of segment lengths in kilometers
                   diameters_mm=None,             # list of pipeline diameters in mm for each segment
                   pump_efficiencies=None,        # list of pump efficiencies (fraction) for each station
                   fluid_viscosity=20,            # fluid kinematic viscosity in cSt (choose from [10,15,...,40])
                   source_pressure=0.0,           # source (inlet) pressure in bar (or any consistent unit, here treated as m head equivalent)
                   target_pressure=0.0,           # minimum required delivery pressure at end (same unit as source_pressure)
                   RatePower=0.1,                 # cost of electricity ($ per kWh)
                   RateDRA=0.005                  # cost factor for DRA ($ per PPM*m3, see formula usage)
                   ):
    """
    Optimize pump usage and DRA injection for a pipeline to minimize total cost.
    Returns a JSON string with per-station and total results.
    """
    # Default pipeline data if none provided (example with 4 segments)
    if lengths_km is None:
        lengths_km = [50.0, 50.0, 50.0, 50.0]  # lengths of each segment in km
    if diameters_mm is None:
        diameters_mm = [600.0, 600.0, 600.0, 600.0]  # diameters (mm) of each segment (approx 24 inch)
    if pump_efficiencies is None:
        pump_efficiencies = [0.85, 0.85, 0.85, 0.85]  # pump efficiencies for each station (fraction)
    # Validate input lengths
    num_segments = len(lengths_km)
    if not (len(diameters_mm) == num_segments == len(pump_efficiencies)):
        raise ValueError("Input lists lengths_km, diameters_mm, and pump_efficiencies must have the same length.")
    # Ensure fluid viscosity is one of the available datasets
    if fluid_viscosity not in dra_curves or dra_curves[fluid_viscosity] is None:
        raise ValueError(f"Fluid viscosity {fluid_viscosity} cst is not supported or data file missing.")
    # Convert units: km to m, mm to m for calculations
    lengths_m = [L * 1000.0 for L in lengths_km]
    diameters_m = [d / 1000.0 for d in diameters_mm]
    # Calculate base frictional head loss for each segment at given flow using Hazen-Williams formula (SI units) [oai_citation:7‡engineering.stackexchange.com](https://engineering.stackexchange.com/questions/45508/how-to-calculate-head-loss-in-vertical-pipe#:~:text=%241%29%24%20The%20the%20Hazen,in%20SI%20units%20is).
    # h_f = 10.67 * L * Q^(1.852) / (C^(1.852) * d^(4.8704)), where Q in m3/s, L in m, d in m.
    # We choose a Hazen-Williams roughness coefficient C based on typical values (e.g., ~120 for clean pipeline steel).
    # For simplicity, we adjust C slightly by viscosity (validity of Hazen-W ill decrease for high viscous fluids [oai_citation:8‡engineering.stackexchange.com](https://engineering.stackexchange.com/questions/45508/how-to-calculate-head-loss-in-vertical-pipe#:~:text=Note%201%3A%20The%20empirical%20nature,accurate%20prediction%20of%20head%20loss)).
    base_C = 120.0
    # Adjust C for viscosity: assume water (1 cSt) ~ C=120; heavier fluids (like 40 cSt) lower C (e.g. ~100).
    # This is a rough adjustment for demonstration.
    C = base_C - (fluid_viscosity - 10)  # reduce C by 1 per additional cSt above 10 (so at 40 cSt, C ~ 80)
    if C < 50: 
        C = 50  # keep C in a reasonable range
    Q_cms = flow / 3600.0  # flow in m3/s
    friction_loss_noDRA = []
    for L, d in zip(lengths_m, diameters_m):
        # Hazen-Williams head loss (m) for this segment:
        if Q_cms <= 0:
            hl = 0.0
        else:
            hl = 10.67 * L * (Q_cms ** 1.852) / ((C ** 1.852) * (d ** 4.8704))
        friction_loss_noDRA.append(hl)
    num_stations = num_segments  # assume one pump station at start of each segment
    # Create Pyomo model
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(1, num_stations)  # index of pumping stations / segments
    # Parameters
    model.flow = pyo.Param(initialize=flow, mutable=False)  # m3/h
    model.eff = pyo.Param(model.N, initialize=lambda m,i: pump_efficiencies[i-1], mutable=False)
    model.friction_noDRA = pyo.Param(model.N, initialize=lambda m,i: friction_loss_noDRA[i-1], mutable=False)
    model.source_pressure = pyo.Param(initialize=source_pressure, mutable=False)
    model.target_pressure = pyo.Param(initialize=target_pressure, mutable=False)
    model.RatePower = pyo.Param(initialize=RatePower, mutable=False)
    model.RateDRA = pyo.Param(initialize=RateDRA, mutable=False)
    # Decision Variables
    model.pump_on = pyo.Var(model.N, domain=pyo.Binary)    # 1 if pump at station i is on, 0 if off
    model.head = pyo.Var(model.N, domain=pyo.NonNegativeReals)  # head added by pump i (m)
    # DRA selection binary variables: for each station and each possible drag reduction level
    dr_levels = [0, 10, 15, 20, 25, 30, 35, 40]  # allowed drag reduction percentages
    model.DR_levels = pyo.Set(initialize=dr_levels)
    model.use_DR = pyo.Var(model.N, model.DR_levels, domain=pyo.Binary)
    # Ensure exactly one drag reduction level is chosen per station
    def one_dr_rule(m, i):
        return sum(m.use_DR[i, dr] for dr in m.DR_levels) == 1
    model.one_dr_constraint = pyo.Constraint(model.N, rule=one_dr_rule)
    # If pump is off, head added must be zero. If on, head <= Hmax (which we set as friction with no DRA for that segment).
    # Big-M formulation: head_i <= friction_noDRA_i * pump_on_i  (so head can only be positive if pump_on=1).
    def head_limit_rule(m, i):
        return m.head[i] <= m.friction_noDRA[i] * m.pump_on[i]
    model.head_limit = pyo.Constraint(model.N, rule=head_limit_rule)
    # Pressure balance constraints between stations:
    # Let P0 = source_pressure. Then for each segment i from station i to i+1 (or to end for last):
    # P_i (pressure at start of segment i) = P_{i-1} + head_{i-1} - friction_loss_{i-1} (for i>1).
    # We implement recurrence: P1 = source_pressure + head1 - friction1, P2 = P1 + head2 - friction2, ... P_end = P_N + ... - friction_N.
    model.pressure = pyo.Var(model.N + 1, domain=pyo.Reals)  # pressure at each station (and station N+1 as pipeline end)
    model.pressure[1] = model.source_pressure  # pressure at start (station1 inlet) = source
    # Define pressure after each segment:
    def pressure_balance_rule(m, i):
        # Pressure at end of segment i (which is station i+1 inlet) = pressure at station i + head_i - friction_loss_i*(1 - DR%)
        # friction_loss with DRA = friction_noDRA * (1 - chosen_DR/100)
        # We compute chosen_DR fraction via the binary variables.
        # First, get the chosen drag reduction percent as an expression:
        dr_expr = sum((dr/100.0) * m.use_DR[i, dr] for dr in m.DR_levels)
        return m.pressure[i+1] == m.pressure[i] + m.head[i] - m.friction_noDRA[i] * (1 - dr_expr)
    model.pressure_balance = pyo.Constraint(model.N, rule=pressure_balance_rule)
    # Delivery pressure constraint: final pressure (at station N+1, pipeline end) must meet target (e.g. >= 0)
    def delivery_pressure_rule(m):
        return m.pressure[num_stations+1] >= m.target_pressure
    model.delivery_pressure = pyo.Constraint(rule=delivery_pressure_rule)
    # Power cost calculation: 
    # Pump hydraulic power (kW) = 0.002725 * flow(m3/h) * head(m) / efficiency [oai_citation:9‡engineering.stackexchange.com](https://engineering.stackexchange.com/questions/45508/how-to-calculate-head-loss-in-vertical-pipe#:~:text=%24h_f%20%3D%2010.67%20L%20Q,4.8704).
    # So daily energy (kWh/day) = 0.002725 * flow * head / eff * 24.
    # Daily cost ($) = that * RatePower.
    def power_cost_expr(m, i):
        return 0.002725 * m.flow * m.head[i] / m.eff[i] * 24 * m.RatePower
    # DRA cost calculation:
    # DRA PPM needed for chosen drag reduction at station i:
    def dra_cost_expr(m, i):
        # Compute PPM via the helper for each possible DR selection:
        return m.flow * 24 * m.RateDRA * sum(get_ppm_for_dr(dr, fluid_viscosity) * m.use_DR[i, dr] for dr in m.DR_levels)
    # Objective: minimize total power cost + total DRA cost
    model.total_cost = pyo.Objective(
        expr=sum(power_cost_expr(model, i) + dra_cost_expr(model, i) for i in model.N),
        sense=pyo.minimize
    )
    # Solve the model using NEOS solver (CBC for MILP)
    # Ensure NEOS email is set in environment
    if 'NEOS_EMAIL' not in os.environ:
        raise EnvironmentError("NEOS email not set. Please set the 'NEOS_EMAIL' environment variable to use NEOS.")
    solver_manager = pyo.SolverManagerFactory('neos')
    try:
        results = solver_manager.solve(model, solver='cbc')
    except Exception as e:
        raise RuntimeError(f"Solver failed: {e}")
    # Load the solution into the model
    model.solutions.load_from(results)
    # Prepare results
    stations_output = []
    total_power_cost = 0.0
    total_dra_cost = 0.0
    pumps_on_count = 0
    for i in model.N:
        pump_status = int(round(pyo.value(model.pump_on[i])))  # 0 or 1
        pumps_on_count += pump_status
        eff = pyo.value(model.eff[i])
        # Determine chosen drag reduction percent at station i
        chosen_dr = None
        for dr in dr_levels:
            if round(pyo.value(model.use_DR[i, dr])) == 1:
                chosen_dr = dr
                break
        if chosen_dr is None:
            chosen_dr = 0  # default if not found, shouldn't happen due to constraint
        # Calculate power cost and DRA cost using the expressions defined
        power_cost_i = 0.002725 * flow * pyo.value(model.head[i]) / eff * 24 * pyo.value(model.RatePower)
        # Calculate DRA cost: PPM * flow * 24 * RateDRA for chosen DR
        ppm_required = get_ppm_for_dr(chosen_dr, fluid_viscosity)
        dra_cost_i = ppm_required * flow * 24 * pyo.value(model.RateDRA)
        total_power_cost += power_cost_i
        total_dra_cost += dra_cost_i
        stations_output.append({
            "station": int(i),
            "flow": flow,
            "pump_on": pump_status,
            "efficiency": round(eff, 4),
            "power_cost": round(power_cost_i, 2),
            "DRA_cost": round(dra_cost_i, 2),
            "drag_reduction": int(chosen_dr)
        })
    total_cost = total_power_cost + total_dra_cost
    result = {
        "stations": stations_output,
        "total": {
            "flow": flow,
            "pumps_on": pumps_on_count,
            "total_power_cost": round(total_power_cost, 2),
            "total_DRA_cost": round(total_dra_cost, 2),
            "total_cost": round(total_cost, 2)
        }
    }
    return json.dumps(result)