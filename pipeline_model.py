import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

# Ensure NEOS email for solver (required by NEOS if not set)
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

# Load DRA performance curves for various viscosities (if files are available)
DRA_CSV_FILES = {
    10: "10 cst.csv",
    15: "15 cst.csv",
    20: "20 cst.csv",
    25: "25 cst.csv",
    30: "30 cst.csv",
    35: "35 cst.csv",
    40: "40 cst.csv"
}
DRA_CURVE_DATA = {}
for cst, fname in DRA_CSV_FILES.items():
    DRA_CURVE_DATA[cst] = pd.read_csv(fname) if os.path.exists(fname) else None

def get_ppm_breakpoints(visc):
    """Return (%DragReduction, PPM) breakpoints for given viscosity via interpolation."""
    cst_list = sorted([c for c in DRA_CURVE_DATA if DRA_CURVE_DATA[c] is not None])
    visc = float(visc)
    if not cst_list:
        return [0, 100], [0, 0]
    # Select or interpolate between nearest viscosity curves
    if visc <= cst_list[0]:
        df = DRA_CURVE_DATA[cst_list[0]]
    elif visc >= cst_list[-1]:
        df = DRA_CURVE_DATA[cst_list[-1]]
    else:
        lower = max(c for c in cst_list if c <= visc)
        upper = min(c for c in cst_list if c >= visc)
        if lower != upper:
            df_low, df_up = DRA_CURVE_DATA[lower], DRA_CURVE_DATA[upper]
            x_low, y_low = df_low['%Drag Reduction'].values, df_low['PPM'].values
            x_up, y_up = df_up['%Drag Reduction'].values, df_up['PPM'].values
            # Interpolate PPM requirement at each unique DR point
            dr_points = np.unique(np.concatenate((x_low, x_up)))
            ppm_interp = (np.interp(dr_points, x_low, y_low) * (upper - visc)/(upper - lower) +
                          np.interp(dr_points, x_up, y_up) * (visc - lower)/(upper - lower))
            unique_dr, idx = np.unique(dr_points, return_index=True)
            unique_ppm = ppm_interp[idx]
            if len(unique_dr) < 2:
                return [0, 100], [0, 0]
            return list(unique_dr), list(unique_ppm)
        # If lower == upper (visc exactly matches an available curve)
        df = DRA_CURVE_DATA[lower]
    # If we have a single dataframe (no interpolation needed)
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    unique_x, idx = np.unique(x, return_index=True)
    unique_y = y[idx]
    if len(unique_x) < 2:
        return [0, 100], [0, 0]
    return list(unique_x), list(unique_y)

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, Rate_DRA, Price_HSD, linefill_dict):
    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)
    # Initialize parameters for fluid properties
    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=float(FLOW))
    model.Rate_DRA = pyo.Param(initialize=float(Rate_DRA))
    model.Price_HSD = pyo.Param(initialize=float(Price_HSD))
    # Compute flow through each segment (m3/hr)
    segment_flows = [float(FLOW)]
    for stn in stations:
        prev_flow = segment_flows[-1]
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        segment_flows.append(prev_flow - delivery + supply)
    # Prepare data for each station/segment
    length = {}; d_inner = {}; rough = {}; thick = {}; smys = {}; design_fac = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_rate = {}
    pump_stations = []; diesel_stations = []; electric_stations = []
    max_dr = {}
    peaks = {}
    # Defaults (if not provided in station data)
    default_t = 0.007      # 7 mm wall thickness
    default_e = 0.00004    # 0.04 mm roughness
    default_smys = 52000   # SMYS in psi (e.g., X52 steel)
    default_df = 0.72      # design factor
    for i, stn in enumerate(stations, start=1):
        length[i] = float(stn.get('L', 0.0))
        # Determine inner diameter
        if 'D' in stn:  # outer diameter given
            thick[i] = float(stn.get('t', default_t))
            d_inner[i] = float(stn['D']) - 2 * thick[i]
        elif 'd' in stn:  # inner diameter given
            d_inner[i] = float(stn['d'])
            thick[i] = float(stn.get('t', default_t))
        else:
            d_inner[i] = 0.7  # default 0.7 m inner diameter
            thick[i] = default_t
        rough[i] = float(stn.get('rough', default_e))
        smys[i] = float(stn.get('SMYS', default_smys))
        design_fac[i] = float(stn.get('DF', default_df))
        elev[i] = float(stn.get('elev', 0.0))
        peaks[i] = stn.get('peaks', [])  # list of peak dicts if any
        # Pump data if station has a pump
        if stn.get('is_pump', False):
            # Ensure rated speed (DOL) is provided and positive
            dol = float(stn.get('DOL', 0))
            if dol <= 0:
                return {"error": True, "message": f"Station '{stn.get('name', i)}' missing valid DOL (rated RPM).", 
                        "termination_condition": "invalid_data", "solver_status": "input_error"}
            pump_stations.append(i)
            # Pump head curve coefficients (H = A*Q^2 + B*Q + C at full speed)
            Acoef[i] = float(stn.get('A', 0.0))
            Bcoef[i] = float(stn.get('B', 0.0))
            Ccoef[i] = float(stn.get('C', 0.0))
            # Pump efficiency curve coefficients (eff% = P*Q^4 + Q*Q^3 + R*Q^2 + S*Q + T)
            Pcoef[i] = float(stn.get('P', 0.0))
            Qcoef[i] = float(stn.get('Q', 0.0))
            Rcoef[i] = float(stn.get('R', 0.0))
            Scoef[i] = float(stn.get('S', 0.0))
            Tcoef[i] = float(stn.get('T', 0.0))
            # Speed limits
            min_rpm_val = max(1, int(stn.get('MinRPM', 1)))
            max_rpm_val = max(min_rpm_val, int(dol))
            min_rpm[i] = min_rpm_val
            max_rpm[i] = max_rpm_val
            # Fuel or electric
            if stn.get('sfc', 0):  # if specific fuel consumption provided (non-zero)
                diesel_stations.append(i)
                sfc[i] = float(stn.get('sfc', 0.0))
            else:
                electric_stations.append(i)
                elec_rate[i] = float(stn.get('rate', 0.0))  # electricity cost rate
            max_dr[i] = float(stn.get('max_dr', 0.0))
    # Terminal elevation
    elev[N+1] = float(terminal.get('elev', 0.0))
    # Create Pyomo parameters
    model.L     = pyo.Param(model.I, initialize=length)
    model.d     = pyo.Param(model.I, initialize=d_inner)
    model.e     = pyo.Param(model.I, initialize=rough)
    model.SMYS  = pyo.Param(model.I, initialize=smys)
    model.DF    = pyo.Param(model.I, initialize=design_fac)
    model.z     = pyo.Param(model.Nodes, initialize=elev)
    model.pump_stations = pyo.Set(initialize=pump_stations)
    if pump_stations:
        model.A = pyo.Param(model.pump_stations, initialize=Acoef)
        model.B = pyo.Param(model.pump_stations, initialize=Bcoef)
        model.C = pyo.Param(model.pump_stations, initialize=Ccoef)
        model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef)
        model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef)
        model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef)
        model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef)
        model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef)
        model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm)
        model.DOL    = pyo.Param(model.pump_stations, initialize=max_rpm)
    # Decision variables:
    # Number of pumps on at each pump station (integer)
    def pump_bounds(m, j):
        lb = 1 if j == pump_stations[0] else 0  # ensure first station has at least 1 pump
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=pump_bounds, initialize=1)
    # Pump speed variable (in tens of RPM, integer for discrete control)
    if pump_stations:
        def speed_bounds(m, j):
            lo = max(1, (int(min_rpm.get(j, 1)) + 9)//10)
            hi = max(lo, int(max_rpm.get(j, 0))//10) if max_rpm.get(j, 0) else lo
            return (lo, hi)
        model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, bounds=speed_bounds, 
                             initialize=lambda m, j: (speed_bounds(m, j)[0] + speed_bounds(m, j)[1])//2)
        model.N = pyo.Expression(model.pump_stations, rule=lambda m, j: 10 * m.N_u[j])
    else:
        model.N_u = pyo.Var([], domain=pyo.NonNegativeIntegers)  # empty, no pumps
    # DRA drag reduction percentage variable at pump stations (continuous, %)
    model.DR = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals, 
                       bounds=lambda m, j: (0, max_dr.get(j, 0.0)), initialize=0.0)
    # Residual head (pressure head) at each node (m)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals)
    # Set minimum residual head constraints
    model.RH[1].fix(float(stations[0].get('min_residual', 50.0)))  # starting suction head
    for node in range(2, N+2):
        model.RH[node].setlb(50.0)  # at least 50 m at all other stations and terminal
    # Pre-calculate constant velocities and friction factors for each segment (using initial flows)
    g = 9.81
    v = {}; f = {}
    for i in range(1, N+1):
        flow = segment_flows[i]  # m3/hr
        area = pi * (d_inner[i] ** 2) / 4.0
        vel = (flow/3600.0) / area if area > 0 else 0.0  # convert flow to m3/s for velocity
        v[i] = vel
        Re = vel * d_inner[i] / (kv_dict[i] * 1e-6) if kv_dict[i] > 0 else 0.0
        if Re > 0:
            if Re < 4000:
                f[i] = 64.0 / Re  # laminar
            else:
                arg = (rough[i] / d_inner[i] / 3.7) + (5.74 / (Re ** 0.9))
                f[i] = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
        else:
            f[i] = 0.0
    # Define station discharge head (SDH) variables and constraints linking segments
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.segment_constraints = pyo.ConstraintList()
    # If pumps exist, define Q_equiv and efficiency expressions
    if pump_stations:
        model.Q_equiv = pyo.Expression(model.pump_stations, 
                                       rule=lambda m, j: (segment_flows[j] * m.DOL[j] / m.N[j]) )
        model.EFFP = pyo.Expression(model.pump_stations, 
                                     rule=lambda m, j: (m.Pcoef[j] * m.Q_equiv[j]**4 + 
                                                        m.Qcoef[j] * m.Q_equiv[j]**3 + 
                                                        m.Rcoef[j] * m.Q_equiv[j]**2 + 
                                                        m.Scoef[j] * m.Q_equiv[j] + 
                                                        m.Tcoef[j]) / 100.0 )
    else:
        model.EFFP = pyo.Expression(model.I, initialize=1.0)
    # Loop through each segment for pressure drop and head constraints
    TDH = {}  # will hold expression for pump head (or 0 if no pump)
    for i in range(1, N+1):
        # Friction head loss in segment i, accounting for drag reduction
        if i in pump_stations:
            DR_fraction = model.DR[i] / 100.0  # convert % to fraction
        else:
            DR_fraction = 0.0
        # Required head at start of segment (station discharge) to maintain min pressure at end
        head_loss = f[i] * ((length[i] * 1000.0) / d_inner[i]) * ((v[i] ** 2) / (2 * g)) * (1 - DR_fraction)
        elev_diff = model.z[i+1] - model.z[i]
        # Station discharge head must cover downstream residual head, elevation gain, and friction loss
        model.segment_constraints.add(model.SDH[i] >= model.RH[i+1] + elev_diff + head_loss)
        # Intermediate peak constraints for this segment
        for peak in peaks[i]:
            Lp = peak['loc'] * 1000.0  # peak distance (m) from station i
            elev_peak = peak['elev']
            peak_loss = f[i] * (Lp / d_inner[i]) * ((v[i] ** 2) / (2 * g)) * (1 - DR_fraction)
            # Ensure at peak: station i discharge head - loss up to peak - elevation >= 50 m
            model.segment_constraints.add(model.SDH[i] >= (elev_peak - model.z[i]) + peak_loss + 50.0)
        # Compute TDH for pump stations using affinity laws, or 0 if no pump
        if i in pump_stations:
            # Pump head at station i for one pump at chosen speed
            N_i = model.N[i]
            DOL_i = model.DOL[i]
            Q_equiv = segment_flows[i] * DOL_i / N_i  # equivalent flow at full speed
            H_full = model.A[i] * Q_equiv**2 + model.B[i] * Q_equiv + model.C[i]  # head at full speed
            TDH[i] = H_full * (N_i / DOL_i) ** 2  # actual head at current speed
        else:
            TDH[i] = 0.0
    # Pressure continuity (head balance) and limits
    model.pressure_constraints = pyo.ConstraintList()
    maop_head = {}  # store MAOP head limits for output
    for i in range(1, N+1):
        # Head balance: upstream residual + added pump head >= required discharge head
        if i in pump_stations:
            model.pressure_constraints.add(model.RH[i] + TDH[i] * model.NOP[i] >= model.SDH[i])
        else:
            model.pressure_constraints.add(model.RH[i] >= model.SDH[i])
        # MAOP pressure limit for segment i
        D_out = d_inner[i] + 2 * thick[i]
        # Convert design pressure (Barlow's formula) to head (m): P = 2 * t * SMYS * DF / D_out (psi), convert to head in m
        MAOP = (2 * thick[i] * (smys[i] * 0.070307) * design_fac[i] / D_out) * 10000.0 / rho_dict[i]
        maop_head[i] = MAOP
        model.pressure_constraints.add(model.SDH[i] <= MAOP)
        # Ensure at least 50 m head at each specified peak (conservative, using full friction without DRA)
        for peak in peaks[i]:
            elev_peak = peak['elev']
            Lp = peak['loc'] * 1000.0
            loss_no_dra = f[i] * (Lp / d_inner[i]) * ((v[i] ** 2) / (2 * g))
            if i in pump_stations:
                model.pressure_constraints.add(model.RH[i] + TDH[i] * model.NOP[i] >= (elev_peak - model.z[i]) + loss_no_dra + 50.0)
            else:
                model.pressure_constraints.add(model.RH[i] >= (elev_peak - model.z[i]) + loss_no_dra + 50.0)
    # DRA usage piecewise-linear relation and cost
    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    model.dra_cost = pyo.Expression(model.pump_stations, rule=lambda m, j: 0.0)  # initialize
    for i in pump_stations:
        dr_pts, ppm_pts = get_ppm_breakpoints(kv_dict[i])
        if dr_pts is None or ppm_pts is None:
            dr_pts_fixed, ppm_pts_fixed = [0, 100], [0, 0]
        else:
            # Sort and remove duplicates for piecewise
            pts = sorted(set(zip(dr_pts, ppm_pts)))
            dr_pts_fixed, ppm_pts_fixed = zip(*pts) if pts else ([0, 100], [0, 0])
        # If no effective DRA data (all ppm 0), fix DR and PPM at 0
        if all(ppm == 0 for ppm in ppm_pts_fixed):
            model.DR[i].fix(0.0)
            model.PPM[i].fix(0.0)
        else:
            model.dra_piecewise = pyo.Piecewise(
                model.PPM[i], model.DR[i],
                pw_pts=list(dr_pts_fixed), f_rule=list(ppm_pts_fixed),
                pw_constr_type='EQ')
        # Calculate DRA cost = PPM (mg/L) * volume flow (m3/day) * cost per mg (Rate_DRA converts properly)
        # segment_flows[i] is m3/hr, convert to m3/day by *24
        volume_day = segment_flows[i] * 24.0  # m3 per day through segment i
        # PPM is mg per liter, so mg per m3 = PPM * 1000; total mg/day = PPM * 1000 * m3/day
        # Divide by 1e6 to get metric tons (or simply multiply PPM * volume (m3) to get liters*ppm etc.)
        # Here Rate_DRA is assumed cost per metric ton or per whatever unit aligns with this calc.
        model.dra_cost[i] = model.PPM[i] * (volume_day * 1000.0 / 1e6) * model.Rate_DRA
    # Power usage variables and constraints
    model.power_use = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)  # kW per station
    model.power_balance = pyo.ConstraintList()
    for i in pump_stations:
        # (rho * flow_i (m3/hr) * g * TDH * NOP) / (3600*1000) = power_use_i * EFFP_i * 0.95
        lhs = rho_dict[i] * segment_flows[i] * 9.81 * TDH[i] * model.NOP[i]
        model.power_balance.add(lhs == model.power_use[i] * (3600.0 * 1000.0 * 0.95) * model.EFFP[i])
    # Objective: minimize total pumping cost + DRA cost
    total_cost = 0.0
    for i in pump_stations:
        # Electric cost = kW * 24h * cost_per_kWh; Diesel cost = kW * 24h * (fuel_per_kWh * diesel_price)
        if i in electric_stations:
            cost_per_kWh = elec_rate.get(i, 0.0)
            total_cost += model.power_use[i] * 24.0 * cost_per_kWh + model.dra_cost[i]
        else:
            fuel_factor = (sfc.get(i, 0.0) * 1.34102) / 820.0  # liters of fuel per kWh
            total_cost += model.power_use[i] * 24.0 * fuel_factor * model.Price_HSD + model.dra_cost[i]
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)
    # Solve using Couenne (global MINLP solver via NEOS)
    results = SolverManagerFactory('neos').solve(model, solver='couenne', tee=False)
    status = results.solver.status
    term = results.solver.termination_condition
    if (status != pyo.SolverStatus.ok) or (term not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        return {
            "error": True,
            "message": f"Optimization failed: {term}. Check inputs or relax constraints if necessary.",
            "termination_condition": str(term),
            "solver_status": str(status)
        }
    model.solutions.load_from(results)
    # Compile results
    output = {}
    for i, stn in enumerate(stations, start=1):
        name_key = stn['name'].strip().lower().replace(' ', '_')
        flow_out = segment_flows[i]
        output[f"pipeline_flow_{name_key}"] = flow_out
        output[f"station_elevation_{name_key}"] = elev[i]
        output[f"residual_head_{name_key}"] = pyo.value(model.RH[i])
        output[f"sdh_{name_key}"] = pyo.value(model.SDH[i])
        output[f"maop_{name_key}"] = maop_head.get(i, 0.0)
        output[f"velocity_{name_key}"] = v.get(i, 0.0)
        output[f"reynolds_{name_key}"] = v.get(i, 0.0) * d_inner[i] / (kv_dict[i] * 1e-6) if kv_dict[i] > 0 else 0.0
        output[f"friction_{name_key}"] = f.get(i, 0.0)
        # If pump at this station, include pump-specific results
        if i in pump_stations:
            num_pumps = int(pyo.value(model.NOP[i]))
            output[f"num_pumps_{name_key}"] = num_pumps
            output[f"pump_flow_{name_key}"] = flow_out if num_pumps > 0 else 0.0
            output[f"speed_{name_key}"] = pyo.value(model.N[i]) if num_pumps > 0 else 0.0
            output[f"efficiency_{name_key}"] = pyo.value(model.EFFP[i]) * 100.0 if num_pumps > 0 else 0.0
            output[f"drag_reduction_{name_key}"] = pyo.value(model.DR[i])
            output[f"dra_ppm_{name_key}"] = pyo.value(model.PPM[i])
            output[f"dra_cost_{name_key}"] = pyo.value(model.dra_cost[i])
            # Calculate daily power cost for output
            power_kW = pyo.value(model.power_use[i])
            if i in electric_stations:
                cost_per_kWh = elec_rate.get(i, 0.0)
                output[f"power_cost_{name_key}"] = power_kW * 24.0 * cost_per_kWh
            else:
                fuel_factor = (sfc.get(i, 0.0) * 1.34102) / 820.0
                output[f"power_cost_{name_key}"] = power_kW * 24.0 * fuel_factor * pyo.value(model.Price_HSD)
        else:
            # No pump at this station
            output[f"num_pumps_{name_key}"] = 0
            output[f"pump_flow_{name_key}"] = 0.0
            output[f"speed_{name_key}"] = 0.0
            output[f"efficiency_{name_key}"] = 0.0
            output[f"drag_reduction_{name_key}"] = 0.0
            output[f"dra_ppm_{name_key}"] = 0.0
            output[f"dra_cost_{name_key}"] = 0.0
            output[f"power_cost_{name_key}"] = 0.0
    # Terminal node outputs
    term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    output[f"pipeline_flow_{term_name}"] = segment_flows[-1]
    output[f"station_elevation_{term_name}"] = elev[N+1]
    output[f"residual_head_{term_name}"] = pyo.value(model.RH[N+1])
    return output