# pipeline_model.py
import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
from math import pi

# Ensure NEOS email is set (replace with your email in deployment)
os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

def solve_pipeline(stations, terminal, FLOW, rho, RateDRA, Price_HSD):
    """
    Build and solve the pipeline optimization model using Pyomo.
    :param stations: list of station dicts (with geometry, pump data, peaks, etc.)
    :param terminal: dict with terminal name, elev, min_residual.
    :param FLOW: volumetric flow (m^3/hr)
    :param KV: kinematic viscosity (cSt)
    :param rho: fluid density (kg/m^3)
    :param RateDRA: drag reducer cost (currency per L)
    :param Price_HSD: diesel price (currency per L)
    :return: dict of results (pump speeds, counts, costs, etc.)
    """
    model = pyo.ConcreteModel()

    # Set global parameters
    model.FLOW = pyo.Param(initialize=FLOW)           # flow rate (m^3/hr)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)    # DRA cost (INR/L)
    model.Price_HSD = pyo.Param(initialize=Price_HSD) # diesel cost (INR/L)

    N = len(stations)
    model.I = pyo.RangeSet(1, N)          # pipeline segments (1..N)
    model.Nodes = pyo.RangeSet(1, N+1)    # nodes (stations 1..N, plus terminal N+1)

    # Initialize data dictionaries
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    inj_source = {}    # tracks last pump upstream
    max_dr = {}
    kv_dict  = {}
    rho_dict = {}
    last_pump_idx = None

    # Process station inputs
    default_t = 0.007  # default wall thickness (m)
    default_e = 0.00004
    default_smys = 52000
    default_df = 0.72
    default_kv  = 10.0   
    default_rho = 850.0   

    for i, stn in enumerate(stations, start=1):
        # Geometry of segment i→i+1
        length[i] = stn.get('L', 0.0)  # km
        # Determine inner diameter
        if 'D' in stn:
            D_out = stn['D']
            thickness[i] = stn.get('t', default_t)
            d_inner[i] = D_out - 2*thickness[i]
        elif 'd' in stn:
            d_inner[i] = stn['d']
            thickness[i] = stn.get('t', default_t)
        else:
            d_inner[i] = 0.7   # fallback
            thickness[i] = default_t
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
        kv_dict[i]  = stn.get('KV',  default_kv)
        rho_dict[i] = stn.get('rho', default_rho)

        # Check if this station has a pump
        has_pump = stn.get('is_pump', False)
        if has_pump:
            pump_indices.append(i)
            # Pump head and efficiency coefficients
            Acoef[i] = stn.get('A', 0.0)
            Bcoef[i] = stn.get('B', 0.0)
            Ccoef[i] = stn.get('C', 0.0)
            Pcoef[i] = stn.get('P', 0.0)
            Qcoef[i] = stn.get('Q', 0.0)
            Rcoef[i] = stn.get('R', 0.0)
            Scoef[i] = stn.get('S', 0.0)
            Tcoef[i] = stn.get('T', 0.0)
            min_rpm[i] = stn.get('MinRPM', 0)
            max_rpm[i] = stn.get('DOL', 0)
            # Fuel or power
            if stn.get('sfc', 0) not in (None, 0):
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc', 0.0)
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 0.0)
            # Max drag reduction
            max_dr[i] = stn.get('max_dr', 0.0)
            last_pump_idx = i
        # Injection source for segments (carry DRA downstream)
        inj_source[i] = last_pump_idx

    # Terminal node elevation
    elev[N+1] = terminal.get('elev', 0.0)

    # Add Pyomo parameters
    model.L = pyo.Param(model.I, initialize=length)          # length (km)
    model.d = pyo.Param(model.I, initialize=d_inner)         # inner diameter (m)
    model.e = pyo.Param(model.I, initialize=roughness)       # roughness (m)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)        # elevations (m)
    model.KV  = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)

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

    # Decision variables
    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0          # ensure at least one pump at station 1
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)

    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)


    # Discretize pump speed in units of 10 RPM
    speed_min = {}; speed_max = {}
    for j in pump_indices:
        lo = max(1, (int(model.MinRPM[j]) + 9)//10) if model.MinRPM[j] else 1
        hi = max(lo, int(model.DOL[j])//10) if model.DOL[j] else lo
        speed_min[j], speed_max[j] = lo, hi
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                       bounds=lambda m,j: (speed_min[j], speed_max[j]),
                       initialize=lambda m,j: (speed_min[j]+speed_max[j])//2)
    # Actual RPM
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])

    # Drag reduction (%), in 10% increments
    dr_max = {j: int(max_dr.get(j, 40)//10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (0, dr_max[j]), initialize=0)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.DR_u[j])

    # Residual head at each node (m) — now including the terminal
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)

    # Station 1: user‐specified minimum residual
    model.RH[1].fix(stations[0].get('min_residual', 50.0))

    # All other nodes (2…N+1): let the solver choose, but enforce ≥50 m
    for j in range(2, N+2):      # note: up to N+1 inclusive
        model.RH[j].setlb(50.0)

    # Hydraulic calculations (outside Pyomo: compute flow velocity, Re, f)
    g = 9.81
    flow_m3s = pyo.value(model.FLOW)/3600.0 if FLOW is not None else 0.0
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area>0 else 0.0
        if model.KV[i] > 0:
            Re[i] = v[i]*d_inner[i]/(float(model.KV[i])*1e-6)
        else:
            Re[i] = 0.0
        if Re[i] > 0:
            if Re[i] < 4000:
                f[i] = 64.0/Re[i]
            else:
                arg = (roughness[i]/d_inner[i]/3.7) + (5.74/(Re[i]**0.9))
                f[i] = 0.25/(log10(arg)**2) if arg>0 else 0.0
        else:
            f[i] = 0.0

    # --- begin revised SDH calculation (with peaks) ---

    # per‐segment pump head & efficiency placeholders
    TDH = {}      # total dynamic head per pump
    EFFP = {}     # pump efficiency

    # a Pyomo Var to hold “the required SDH” for each segment i
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)

    # we’ll collect the >= constraints into a ConstraintList
    model.sdh_constraint = pyo.ConstraintList()

    for i in range(1, N+1):
        # 1) frictional loss to the next node (i → i+1)
        DR_frac = 0
        if inj_source.get(i) in pump_indices:
            DR_frac = model.DR[inj_source[i]]/100.0

        DH_next = f[i] * ( (length[i]*1000.0) / d_inner[i] ) * (v[i]**2 / (2*g)) * (1 - DR_frac)

        # Option A: water must clear the next station
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)

        # Option B: water must clear each intermediate peak by ≥50 m
        for peak in stations[i-1].get('peaks', []):
            L_peak = peak['loc'] * 1000.0    # metres from station i
            elev_k = peak['elev']           # absolute elevation of the peak

            DR_frac_peak = 0
            if inj_source.get(i) in pump_indices:
                DR_frac_peak = model.DR[inj_source[i]]/100.0

            DH_peak = f[i] * ( (L_peak) / d_inner[i] ) * (v[i]**2 / (2*g)) * (1 - DR_frac_peak)

            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

        # now compute the actual pump head & efficiency (unchanged)
        if i in pump_indices:
            TDH[i] = (model.A[i]*model.FLOW**2 +
                      model.B[i]*model.FLOW +
                      model.C[i]) * ((model.N[i]/model.DOL[i])**2)

            flow_eq = model.FLOW * model.DOL[i]/model.N[i]
            EFFP[i] = (
                model.Pcoef[i]*flow_eq**4 +
                model.Qcoef[i]*flow_eq**3 +
                model.Rcoef[i]*flow_eq**2 +
                model.Scoef[i]*flow_eq   +
                model.Tcoef[i]
            ) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # --- end revised SDH calculation ---

    # Constraints
    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        # Head balance: residual_in + (pump head if any) ≥ required head (static+friction)
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])

        # Pressure (MAOP) limit in head units
        D_out = d_inner[i] + 2*thickness[i]
        MAOP_head = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/D_out)*10000.0/model.rho[i]
        if i in pump_indices:
            model.pressure_limit.add(model.RH[i] + TDH[i]*model.NOP[i] <= MAOP_head)
        else:
            model.pressure_limit.add(model.RH[i] <= MAOP_head)

        # Peak constraints for segment i (if any peaks defined)
        peaks = stations[i-1].get('peaks', [])
        for peak in peaks:
            loc_km = peak['loc']
            elev_k = peak['elev']
            L_peak = loc_km*1000.0  # meters
            # Compute head loss up to peak
            DR_frac_peak = 0
            if inj_source.get(i) in pump_indices:
                DR_frac_peak = model.DR[inj_source[i]]/100.0
            loss_no_dra = f[i] * (L_peak/d_inner[i]) * (v[i]**2/(2*g))
            # Build constraint: (head into peak) - (elevation rise) - (friction to peak) >= 50
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra*(1-DR_frac_peak)
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra*(1-DR_frac_peak)
            model.peak_limit.add(expr >= 50.0)

    # Objective: minimize total daily cost (24h power + DRA)
    total_cost = 0
    for i in pump_indices:
        # Pumping power (kW)
        power_kW = (model.rho[i] * FLOW * 9.81 * TDH[i] * model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i,0.0)
        else:
            fuel_per_kWh = (sfc.get(i,0.0)*1.34102)/820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        dra_cost = (model.DR[i]/4) * (FLOW*1000.0*24.0/1e6) * RateDRA
        total_cost += power_cost + dra_cost
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # Solve with Couenne via NEOS
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    # Extract results
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps>0 else 0.0
            eff = float(pyo.value(EFFP[i])*100.0) if num_pumps>0 else 0.0
        else:
            num_pumps = 0; speed_rpm = 0.0; eff = 0.0

        # Costs
        if i in pump_indices and num_pumps>0:
            power_kW = (model.rho[i] * FLOW * 9.81 * float(pyo.value(TDH[i])) * num_pumps)/(3600.0*1000.0*float(pyo.value(EFFP[i]))*0.95)
            if i in electric_pumps:
                rate = elec_cost.get(i,0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i,0.0)
                fuel_per_kWh = (sfc_val*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        else:
            power_cost = 0.0

        if i in pump_indices:
            dra_cost = (float(pyo.value(model.DR[i]))/4)*(FLOW*1000.0*24.0/1e6)*RateDRA
            drag_red = float(pyo.value(model.DR[i]))
        else:
            dra_cost = 0.0; drag_red = 0.0

        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i]))))
        res_head = float(pyo.value(model.RH[i]))
        velocity = v[i]; reynolds = Re[i]

        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost
        result[f"dra_cost_{name}"] = dra_cost
        result[f"drag_reduction_{name}"] = drag_red
        result[f"head_loss_{name}"] = head_loss
        result[f"residual_head_{name}"] = res_head
        result[f"velocity_{name}"] = velocity
        result[f"reynolds_{name}"] = reynolds
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i]))
        if i in pump_indices:
            result[f"coef_A_{name}"] = float(pyo.value(model.A[i]))
            result[f"coef_B_{name}"] = float(pyo.value(model.B[i]))
            result[f"coef_C_{name}"] = float(pyo.value(model.C[i]))
            result[f"dol_{name}"]    = float(pyo.value(model.DOL[i]))
            result[f"min_rpm_{name}"]= float(pyo.value(model.MinRPM[i]))

    # Terminal node (no pumps)
    term = terminal.get('name','terminal').strip().lower().replace(' ','_')
    result.update({
        f"speed_{term}": 0.0,
        f"num_pumps_{term}": 0,
        f"efficiency_{term}": 0.0,
        f"power_cost_{term}": 0.0,
        f"dra_cost_{term}": 0.0,
        f"drag_reduction_{term}": 0.0,
        f"head_loss_{term}": 0.0,
        f"velocity_{term}": 0.0,
        f"reynolds_{term}": 0.0,
        f"sdh_{term}": 0.0,
        f"residual_head_{term}": float(pyo.value(model.RH[N+1])),
    })
    result['total_cost'] = float(pyo.value(model.Obj))

    return result
