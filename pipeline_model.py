import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD,
    linefill_dict=None, looplines=None, deliveries=None
):
    """
    PIPELINE OPTIMA™: Backend Solver (Loopline/Peaks/DRA/Deliveries)
    """
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

    # -- Station input extraction (Mainline geometry, elevation, pump data)
    length, d_inner, roughness, thickness, smys, design_factor, elev = {}, {}, {}, {}, {}, {}, {}
    Acoef, Bcoef, Ccoef, Pcoef, Qcoef, Rcoef, Scoef, Tcoef = {}, {}, {}, {}, {}, {}, {}, {}
    min_rpm, max_rpm = {}, {}
    sfc, elec_cost = {}, {}
    pump_indices, diesel_pumps, electric_pumps, max_dr = [], [], [], {}
    peaks_dict = {}
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

    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)

    speed_min, speed_max = {}, {}
    for j in pump_indices:
        lo = max(1, (int(model.MinRPM[j]) + 9)//10) if model.MinRPM[j] else 1
        hi = max(lo, int(model.DOL[j])//10) if model.DOL[j] else lo
        speed_min[j], speed_max[j] = lo, hi
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (speed_min[j], speed_max[j]),
                        initialize=lambda m,j: (speed_min[j]+speed_max[j])//2)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])

    dr_max = {j: int(max_dr.get(j, 40)//10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                         bounds=lambda m,j: (0, dr_max[j]), initialize=0)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.DR_u[j])

    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)

    # --- DELIVERY INPUTS ---
    if deliveries is None:
        deliveries = {}
    # --- LOOPLINES DATA ---
    if looplines is None:
        looplines = []
    # Build a list of all segments, including mainline and each loopline
    # Segment: (type, from_node, to_node, properties dict)
    segments = []
    # Add all mainline segments (default)
    for i in range(1, N+1):
        seg = {
            'type': 'main',
            'from_node': i,
            'to_node': i+1,
            'L': length[i],
            'D': d_inner[i],
            'rough': roughness[i],
            'SMYS': smys[i],
            't': thickness[i],
            'DF': design_factor[i],
            'idx': i,
            'start_elev': elev[i],
            'end_elev': elev[i+1],
            'max_dr': max_dr.get(i, 0.0),
            'peaks': peaks_dict[i]
        }
        segments.append(seg)
    # Add all loopline segments (custom)
    for idx, lp in enumerate(looplines, start=1):
        # Map start_km and end_km to nearest node indices
        chainages = [0]
        for stn in stations:
            chainages.append(chainages[-1] + stn['L'])
        def find_nearest_node(km):
            return min(range(len(chainages)), key=lambda i: abs(chainages[i]-km)) + 1
        from_node = find_nearest_node(lp['start_km'])
        to_node   = find_nearest_node(lp['end_km'])
        peaks = lp.get('peaks', [])
        seg = {
            'type': 'loop',
            'from_node': from_node,
            'to_node': to_node,
            'L': lp['L'],
            'D': lp['D'] - 2*lp['t'],
            'rough': lp['rough'],
            'SMYS': lp['SMYS'],
            't': lp['t'],
            'DF': default_df,
            'idx': N+idx,
            'start_elev': lp.get('start_elev', elev[from_node]),
            'end_elev': lp.get('end_elev', elev[to_node]),
            'max_dr': lp.get('max_dr', 0.0),
            'peaks': peaks
        }
        segments.append(seg)

    # ---- Build Segment and Node Flow Variables ----
    model.segments = pyo.Set(initialize=[seg['idx'] for seg in segments])
    segment_map = {seg['idx']: seg for seg in segments}
    model.Q = pyo.Var(model.segments, domain=pyo.NonNegativeReals, initialize=FLOW) # m3/hr

    # --- DRA variables for each segment (mainline pump stations + each loopline) ---
    dra_segments = [seg['idx'] for seg in segments if seg['type']=='loop'] + [i for i in pump_indices]
    model.DR_seg = pyo.Var(dra_segments, domain=pyo.NonNegativeReals)
    # Add bounds per segment
    for segidx in dra_segments:
        seg = segment_map[segidx] if segidx in segment_map else None
        maxval = seg['max_dr'] if seg else max_dr.get(segidx, 0.0)
        model.DR_seg[segidx].setub(maxval)
        model.DR_seg[segidx].setlb(0.0)

    # ---- Node Continuity Constraints (with delivery) ----
    def continuity_rule(m, node):
        if node == 1 or node == N+1:
            return pyo.Constraint.Skip
        inflow = []
        outflow = []
        for seg in segments:
            if seg['to_node'] == node:
                inflow.append(model.Q[seg['idx']])
            if seg['from_node'] == node:
                outflow.append(model.Q[seg['idx']])
        demand = deliveries.get(node, 0.0)
        return sum(inflow) - sum(outflow) == demand
    model.node_continuity = pyo.Constraint(model.Nodes, rule=continuity_rule)

    # --- Velocity, Reynolds, Friction Factor for each segment (at design flow) ---
    g = 9.81
    v, Re, f = {}, {}, {}
    for seg in segments:
        idx = seg['idx']
        def vfun(q): return (q/3600.0) / (pi * (seg['D']**2) / 4.0) if seg['D'] > 0 else 0.0
        def Refun(q, kv): return vfun(q) * seg['D'] / (kv * 1e-6) if kv > 0 else 0.0
        def ffun(q, kv, dr): 
            re = Refun(q, kv)
            if re == 0: return 0.0
            if re < 4000: return 64.0/re
            arg = (seg['rough']/seg['D']/3.7) + (5.74/(re**0.9))
            fval = 0.25/(log10(arg)**2) if arg > 0 else 0.0
            fval = fval * (1 - dr/100.0)
            return fval
        v[idx] = vfun
        Re[idx] = Refun
        f[idx] = ffun

    # ---- Head Loss Equations for Mainline and Looplines ----
    # For each segment, head loss = f*(L/D)*(v^2/(2g)), DR if allowed
    def headloss_expr(m, idx):
        seg = segment_map[idx]
        q = m.Q[idx]
        kv = kv_dict.get(seg['from_node'], 1.1)
        dr = m.DR_seg[idx] if idx in dra_segments else 0.0
        fseg = f[idx](q, kv, dr)
        vseg = v[idx](q)
        return fseg * ((seg['L']*1000.0)/seg['D']) * (vseg**2/(2*g))
    # --- For each pair of parallel segments between the same nodes, enforce equal head loss ---
    pairwise = {}
    for seg in segments:
        key = (seg['from_node'], seg['to_node'])
        pairwise.setdefault(key, []).append(seg['idx'])
    model.parallel_headloss = pyo.ConstraintList()
    for segids in pairwise.values():
        if len(segids) > 1:
            hl0 = headloss_expr(model, segids[0])
            for idx in segids[1:]:
                model.parallel_headloss.add(
                    headloss_expr(model, idx) == hl0
                )

    # --- Peak Pressure Constraints (applies to all segments: mainline & loopline) ---
    model.peak_limit = pyo.ConstraintList()
    for seg in segments:
        idx = seg['idx']
        peaks = seg.get('peaks', [])
        for peak in peaks:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            q = model.Q[idx]
            kv = kv_dict.get(seg['from_node'], 1.1)
            dr = model.DR_seg[idx] if idx in dra_segments else 0.0
            f_i = f[idx](q, kv, dr)
            v_i = v[idx](q)
            DH_peak = f_i * (L_peak / seg['D']) * (v_i**2/(2*g))
            static_h = elev_k - seg['start_elev']
            model.peak_limit.add(model.RH[seg['from_node']] - static_h - DH_peak >= 50.0)
    # No separate loopline peak constraint block needed; above covers all.

    # ---- Residual Head and Static Discharge Head at Each Node ----
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    for i in range(1, N+1):
        seg = next(s for s in segments if s['type'] == 'main' and s['idx']==i)
        hl = headloss_expr(model, i)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + hl
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        # For all peaks
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            f_i = f[i](model.Q[i], kv_dict[i], model.DR_seg[i] if i in dra_segments else 0.0)
            v_i = v[i](model.Q[i])
            DH_peak = f_i * (L_peak / seg['D']) * (v_i**2/(2*g))
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

    # --- Pump Equations, Objective, MAOP Constraints ---
    TDH, EFFP, maop_dict = {}, {}, {}
    model.head_balance = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        if i in pump_indices:
            TDH[i] = (model.A[i]*model.FLOW**2 + model.B[i]*model.FLOW + model.C[i]) * ((model.N[i]/model.DOL[i])**2)
            flow_eq = model.FLOW * model.DOL[i]/model.N[i]
            EFFP[i] = (
                model.Pcoef[i]*flow_eq**4 + model.Qcoef[i]*flow_eq**3 + model.Rcoef[i]*flow_eq**2
                + model.Scoef[i]*flow_eq + model.Tcoef[i]
            ) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0
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

    # --- Objective Function: Minimize Pumping + DRA (main + loopline) + Power/Fuel Cost ---
    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        power_kW = (rho_i * FLOW * 9.81 * TDH[i] * model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i,0.0)
        else:
            fuel_per_kWh = (sfc.get(i,0.0)*1.34102)/820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        # --- Mainline DRA cost (if DRA allowed on mainline)
        if i in dra_segments:
            dra_cost = (model.DR_seg[i]/4) * (FLOW*1000.0*24.0/1e6) * RateDRA
        else:
            dra_cost = 0
        total_cost += power_cost + dra_cost
    # --- Loopline DRA cost ---
    for seg in segments:
        if seg['type'] == 'loop':
            dra_cost = (model.DR_seg[seg['idx']]/4) * (model.Q[seg['idx']]*1000.0*24.0/1e6) * RateDRA
            total_cost += dra_cost
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # --- Solve ---
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    # --- Output Section ---
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
            eff = float(pyo.value(EFFP[i]) * 100.0) if num_pumps > 0 else 0.0
        else:
            num_pumps = 0
            speed_rpm = 0.0
            eff = 0.0
    
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"residual_head_{name}"] = float(pyo.value(model.RH[i]))
    
        # --- Stationwise hydraulic output ---
        seg = next(s for s in segments if s['type'] == 'main' and s['idx'] == i)
        q_val = pyo.value(model.Q[i])
        v_val = v[i](q_val)
        re_val = Re[i](q_val, kv_dict[i])
        f_val = f[i](q_val, kv_dict[i], pyo.value(model.DR_seg[i]) if i in dra_segments else 0.0)
        hl_val = headloss_expr(model, i)
        sdh_val = pyo.value(model.SDH[i])
        maop_val = maop_dict[i]
        result[f"flow_{name}"] = q_val
        result[f"velocity_{name}"] = v_val
        result[f"reynolds_{name}"] = re_val
        result[f"friction_{name}"] = f_val
        result[f"head_loss_{name}"] = hl_val
        result[f"sdh_{name}"] = sdh_val
        result[f"maop_{name}"] = maop_val
    
        if i in pump_indices:
            result[f"drag_reduction_{name}"] = float(pyo.value(model.DR_seg[i]))  # Mainline DRA (%)
            # Power Cost Output
            rho_i = rho_dict[i]
            TDH_val = (Acoef[i] * FLOW ** 2 + Bcoef[i] * FLOW + Ccoef[i]) * ((pyo.value(model.N[i]) / max_rpm[i]) ** 2)
            eff_val = float(pyo.value(EFFP[i]))
            num_pumps = int(pyo.value(model.NOP[i]))
            if i in electric_pumps:
                power_kW = (rho_i * FLOW * 9.81 * TDH_val * num_pumps) / (3600.0 * 1000.0 * eff_val * 0.95)
                power_cost = power_kW * 24.0 * elec_cost.get(i, 0.0)
            else:
                fuel_per_kWh = (sfc.get(i, 0.0) * 1.34102) / 820.0
                power_kW = (rho_i * FLOW * 9.81 * TDH_val * num_pumps) / (3600.0 * 1000.0 * eff_val * 0.95)
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
            result[f"power_cost_{name}"] = float(power_cost)
        else:
            result[f"power_cost_{name}"] = 0.0
    
        # Output peak pressures (for all peaks defined for the station)
        peaks = stn.get('peaks', [])
        for pidx, peak in enumerate(peaks, start=1):
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            q = q_val
            kv = kv_dict[i]
            dr = pyo.value(model.DR_seg[i]) if i in dra_segments else 0.0
            f_i = f[i](q, kv, dr)
            v_i = v[i](q)
            DH_peak = f_i * (L_peak / seg['D']) * (v_i ** 2 / (2 * 9.81))
            static_h = elev_k - seg['start_elev']
            peak_head = pyo.value(model.RH[i]) - static_h - DH_peak
            result[f"peak_head_{name}_{pidx}"] = peak_head

    # ---- Output for looplines: include flows, velocities, head loss, DR, etc. ----
    for seg in segments:
        if seg['type'] == 'loop':
            key = f"loopline_{seg['from_node']}_{seg['to_node']}"
            q_val = pyo.value(model.Q[seg['idx']])
            v_val = v[seg['idx']](q_val)
            re_val = Re[seg['idx']](q_val, kv_dict.get(seg['from_node'], 1.1))
            f_val = f[seg['idx']](q_val, kv_dict.get(seg['from_node'], 1.1), pyo.value(model.DR_seg[seg['idx']]))
            hl_val = headloss_expr(model, seg['idx'])
            result[f"{key}_flow_m3hr"] = q_val
            result[f"{key}_velocity_ms"] = v_val
            result[f"{key}_reynolds"] = re_val
            result[f"{key}_friction"] = f_val
            result[f"{key}_head_loss_m"] = hl_val
            result[f"{key}_drag_reduction_percent"] = pyo.value(model.DR_seg[seg['idx']])
            result[f"{key}_power_cost"] = 0.0
            # --- Loopline peaks output (if any) ---
            if seg.get('peaks'):
                for pidx, peak in enumerate(seg['peaks'], start=1):
                    L_peak = peak['loc'] * 1000.0
                    elev_k = peak['elev']
                    q = q_val
                    kv = kv_dict.get(seg['from_node'], 1.1)
                    dr = pyo.value(model.DR_seg[seg['idx']])
                    f_i = f[seg['idx']](q, kv, dr)
                    v_i = v[seg['idx']](q)
                    DH_peak = f_i * (L_peak / seg['D']) * (v_i ** 2 / (2 * 9.81))
                    static_h = elev_k - seg['start_elev']
                    peak_head = pyo.value(model.RH[seg['from_node']]) - static_h - DH_peak
                    result[f"{key}_peak_{pidx}"] = peak_head
    result['total_cost'] = float(pyo.value(model.Obj))
    return result