import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
import pandas as pd
from math import log10, pi

# --- DRA Curve Loader ---
def load_dra_curves(base_dir='./'):
    dra_curves = {}
    for visc in range(10, 45, 5):
        csv_path = os.path.join(base_dir, f"{visc} cst.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]
        if 'ppm' in df.columns and ('drag_reduction' in df.columns or 'drag reduction' in df.columns):
            dr_col = 'drag_reduction' if 'drag_reduction' in df.columns else 'drag reduction'
            # Force to float, just in case
            curve = sorted(
                [(float(row['ppm']), float(row[dr_col])) for idx, row in df.iterrows()],
                key=lambda x: x[1]
            )
            dra_curves[visc] = curve
    return dra_curves

# --- DRA PPM Interpolation ---
def get_ppm_for_drag_reduction(dra_curves, viscosity, drag_reduction):
    # Use nearest available viscosity if exact not found
    all_visc = sorted(dra_curves.keys())
    if not all_visc:
        return 0.0
    if viscosity in dra_curves:
        curve = dra_curves[viscosity]
    else:
        # Interpolate between two viscosity curves
        lower = max([v for v in all_visc if v <= viscosity], default=all_visc[0])
        upper = min([v for v in all_visc if v >= viscosity], default=all_visc[-1])
        if lower == upper:
            curve = dra_curves[lower]
        else:
            # Interpolate/extrapolate between two curves
            curve_low = dra_curves[lower]
            curve_high = dra_curves[upper]
            ppm_low = interpolate_ppm(curve_low, drag_reduction)
            ppm_high = interpolate_ppm(curve_high, drag_reduction)
            ratio = (viscosity - lower) / (upper - lower)
            return ppm_low + (ppm_high - ppm_low) * ratio
    return interpolate_ppm(curve, drag_reduction)

def interpolate_ppm(curve, drag_reduction):
    # curve: list of (PPM, DR)
    xs = [dr for ppm, dr in curve]
    ys = [ppm for ppm, dr in curve]
    if drag_reduction <= xs[0]:
        # Extrapolate left
        if xs[0] == xs[1]:
            return ys[0]
        return ys[0] + (ys[1] - ys[0]) * (drag_reduction - xs[0]) / (xs[1] - xs[0])
    if drag_reduction >= xs[-1]:
        # Extrapolate right
        if xs[-1] == xs[-2]:
            return ys[-1]
        return ys[-2] + (ys[-1] - ys[-2]) * (drag_reduction - xs[-2]) / (xs[-1] - xs[-2])
    # Linear interpolation
    for i in range(1, len(xs)):
        if xs[i-1] <= drag_reduction <= xs[i]:
            if xs[i] == xs[i-1]:
                return ys[i]
            return ys[i-1] + (ys[i] - ys[i-1]) * (drag_reduction - xs[i-1]) / (xs[i] - xs[i-1])
    return 0.0

# --- Pipeline Optimization Backend ---
def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict=None, dra_curve_dir='./'
):
    os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')
    dra_curves = load_dra_curves(dra_curve_dir)

    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)

    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # --- NEW: Segment-wise flow, delivery/supply logic ---
    segment_flows = []
    pump_flows = []
    flow = float(FLOW)
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        has_pump = stn.get('is_pump', False)
        segment_flows.append(flow)  # Flow into this station
        if has_pump:
            pump_flow = flow - delivery + supply
            pump_flows.append(pump_flow)
            flow = pump_flow
        else:
            pump_flows.append(None)
            flow = flow - delivery + supply
    segment_flows.append(flow)  # For terminal node

    # Remaining param initialization as before
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    max_dr = {}; peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72

    for i, stn in enumerate(stations, start=1):
        length[i] = stn.get('L', 0.0)
        if 'D' in stn:
            D_out = stn['D']
            thickness[i] = stn.get('t', default_t)
            d_inner[i] = D_out - 2*thickness[i]
        elif 'd' in stn:
            d_inner[i] = stn['d']
            thickness[i] = stn.get('t', default_t)
        else:
            d_inner[i] = 0.7
            thickness[i] = default_t
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
        peaks_dict[i] = stn.get('peaks', [])
        has_pump = stn.get('is_pump', False)
        if has_pump:
            pump_indices.append(i)
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
            if stn.get('sfc', 0) not in (None, 0):
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc', 0.0)
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 0.0)
        max_dr[i] = stn.get('max_dr', 0.0)

    elev[N+1] = terminal.get('elev', 0.0)

    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)

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

    # --- Decision Variables ---
    # DRA variable at all stations, not just pumps
    dra_max = {i: int(max_dr.get(i, 40)//10) for i in range(1, N+1)}
    model.DR_u = pyo.Var(model.I, domain=pyo.NonNegativeIntegers,
                         bounds=lambda m,i: (0, dra_max[i]), initialize=0)
    model.DR = pyo.Expression(model.I, rule=lambda m,i: 10*m.DR_u[i])

    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)

    speed_min = {}; speed_max = {}
    for j in pump_indices:
        lo = max(1, (int(model.MinRPM[j]) + 9)//10) if model.MinRPM[j] else 1
        hi = max(lo, int(model.DOL[j])//10) if model.DOL[j] else lo
        speed_min[j], speed_max[j] = lo, hi
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (speed_min[j], speed_max[j]),
                        initialize=lambda m,j: (speed_min[j]+speed_max[j])//2)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        flow_m3s = float(segment_flows[i-1])/3600.0
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        kv = kv_dict[i]
        rho = rho_dict[i]
        if kv > 0:
            Re[i] = v[i]*d_inner[i]/(float(kv)*1e-6)
        else:
            Re[i] = 0.0
        if Re[i] > 0:
            if Re[i] < 4000:
                f[i] = 64.0/Re[i]
            else:
                arg = (roughness[i]/d_inner[i]/3.7) + (5.74/(Re[i]**0.9))
                f[i] = 0.25/(log10(arg)**2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}
    EFFP = {}

    for i in range(1, N+1):
        # DRA (drag reduction) at ALL stations
        DR_frac = model.DR[i] / 100.0
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2/(2*g)) * (1 - DR_frac)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2/(2*g)) * (1 - DR_frac)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)
        # Pump equations: use pump_flows!
        if i in pump_indices:
            pump_flow_i = pump_flows[i-1]
            TDH[i] = (model.A[i]*pump_flow_i**2 + model.B[i]*pump_flow_i + model.C[i]) * ((model.N[i]/model.DOL[i])**2)
            flow_eq = pump_flow_i * model.DOL[i]/model.N[i]
            EFFP[i] = (
                model.Pcoef[i]*flow_eq**4 + model.Qcoef[i]*flow_eq**3 + model.Rcoef[i]*flow_eq**2
                + model.Scoef[i]*flow_eq + model.Tcoef[i]
            ) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N+1):
        kv = kv_dict[i]
        rho = rho_dict[i]
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        D_out = d_inner[i] + 2 * thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)
        peaks = peaks_dict[i]
        for peak in peaks:
            loc_km = peak['loc']
            elev_k = peak['elev']
            L_peak = loc_km*1000.0
            loss_no_dra = f[i] * (L_peak/d_inner[i]) * (v[i]**2/(2*g))
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    # --- Objective: Total Power + DRA cost ---
    total_cost = 0
    for i in range(1, N+1):
        # DRA cost at this station
        viscosity = kv_dict[i]
        segment_flow = segment_flows[i-1]
        drag_red = float(pyo.value(model.DR[i]) if model.DR[i].is_fixed() else model.DR[i])
        ppm = get_ppm_for_drag_reduction(dra_curves, viscosity, drag_red)
        dra_cost = ppm * (segment_flow * 1000.0 * 24.0 / 1e6) * RateDRA  # PPM × m³/day × rate

        # Power/fuel cost (if pump station and running)
        power_cost = 0.0
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
            eff = float(pyo.value(EFFP[i]) * 100.0) if num_pumps > 0 else 0.0
            rho_i = rho_dict[i]
            pump_flow_i = pump_flows[i-1]
            power_kW = (rho_i * pump_flow_i * 9.81 * float(pyo.value(TDH[i])) * num_pumps)/(3600.0*1000.0*float(pyo.value(EFFP[i]))*0.95)
            if i in electric_pumps:
                rate = elec_cost.get(i,0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i,0.0)
                fuel_per_kWh = (sfc_val*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        total_cost += power_cost + dra_cost

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    # --- Build output ---
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
            eff = float(pyo.value(EFFP[i])*100.0) if num_pumps > 0 else 0.0
        else:
            num_pumps = 0; speed_rpm = 0.0; eff = 0.0

        viscosity = kv_dict[i]
        segment_flow = segment_flows[i-1]
        drag_red = float(pyo.value(model.DR[i]))
        ppm = get_ppm_for_drag_reduction(dra_curves, viscosity, drag_red)
        dra_cost = ppm * (segment_flow * 1000.0 * 24.0 / 1e6) * RateDRA

        velocity = v[i]; reynolds = Re[i]; fric = f[i]
        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i]))))
        res_head = float(pyo.value(model.RH[i]))

        result[f"pipeline_flow_{name}"] = segment_flow
        result[f"pump_flow_{name}"] = pump_flows[i-1] if pump_flows[i-1] is not None else 0.0
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost if i in pump_indices else 0.0
        result[f"dra_cost_{name}"] = dra_cost
        result[f"drag_reduction_{name}"] = drag_red
        result[f"dra_ppm_{name}"] = ppm
        result[f"head_loss_{name}"] = head_loss
        result[f"residual_head_{name}"] = res_head
        result[f"velocity_{name}"] = velocity
        result[f"reynolds_{name}"] = reynolds
        result[f"friction_{name}"] = fric
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i]))
        result[f"maop_{name}"] = maop_dict[i]
        if i in pump_indices:
            result[f"coef_A_{name}"] = float(pyo.value(model.A[i]))
            result[f"coef_B_{name}"] = float(pyo.value(model.B[i]))
            result[f"coef_C_{name}"] = float(pyo.value(model.C[i]))
            result[f"dol_{name}"]    = float(pyo.value(model.DOL[i]))
            result[f"min_rpm_{name}"]= float(pyo.value(model.MinRPM[i]))

    term = terminal.get('name','terminal').strip().lower().replace(' ','_')
    result.update({
        f"pipeline_flow_{term}": segment_flows[-1],
        f"pump_flow_{term}": 0.0,
        f"num_pumps_{term}": 0,
        f"speed_{term}": 0.0,
        f"efficiency_{term}": 0.0,
        f"power_cost_{term}": 0.0,
        f"dra_cost_{term}": 0.0,
        f"drag_reduction_{term}": 0.0,
        f"dra_ppm_{term}": 0.0,
        f"head_loss_{term}": 0.0,
        f"velocity_{term}": 0.0,
        f"reynolds_{term}": 0.0,
        f"friction_{term}": 0.0,
        f"sdh_{term}": 0.0,
        f"residual_head_{term}": float(pyo.value(model.RH[N+1])),
    })
    result['total_cost'] = float(pyo.value(model.Obj))
    return result
