import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import pi

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

def solve_pipeline(
    stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD,
    linefill_dict=None, looplines=None, deliveries=None
):
    if deliveries is None:
        deliveries = {}
    elif isinstance(deliveries, list):
        deliveries = {i+1: float(val) for i, val in enumerate(deliveries)}
    deliveries.setdefault(len(stations)+1, 0.0)

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
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)

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

    dra_segments = []
    for i, stn in enumerate(stations, start=1):
        if stn.get('max_dr', 0) > 0:
            dra_segments.append(i)

    # ---- LOOPLINES ----
    if looplines is None:
        looplines = []
    segments = []
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
    for idx, lp in enumerate(looplines, start=1):
        # Compute length from km
        lp_L = lp.get('end_km', 0.0) - lp.get('start_km', 0.0)
        from_node = idx+N  # (Choose a unique node id, if desired use: find_nearest_node)
        to_node = from_node + 1
        seg = {
            'type': 'loop',
            'from_node': from_node,
            'to_node': to_node,
            'L': lp_L,
            'D': lp['D'] - 2*lp['t'],
            'rough': lp['rough'],
            'SMYS': lp['SMYS'],
            't': lp['t'],
            'DF': default_df,
            'idx': N+idx,
            'start_elev': lp.get('start_elev', elev.get(from_node, 0)),
            'end_elev': lp.get('end_elev', elev.get(to_node, 0)),
            'max_dr': lp.get('max_dr', 0.0),
            'peaks': lp.get('peaks', [])
        }
        if seg.get('max_dr', 0) > 0:
            dra_segments.append(seg['idx'])
        segments.append(seg)

    model.segments = pyo.Set(initialize=[seg['idx'] for seg in segments])
    segment_map = {seg['idx']: seg for seg in segments}
    model.Q = pyo.Var(model.segments, domain=pyo.NonNegativeReals, initialize=FLOW)

    # Only define DRA var for active DRA segments
    model.DR_seg = pyo.Var(dra_segments, domain=pyo.NonNegativeReals)
    for segidx in dra_segments:
        seg = segment_map[segidx] if segidx in segment_map else None
        maxval = seg['max_dr'] if seg else max_dr.get(segidx, 0.0)
        model.DR_seg[segidx].setub(maxval)
        model.DR_seg[segidx].setlb(0.0)
        if maxval <= 0:
            model.DR_seg[segidx].fix(0.0)

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

    g = 9.81
    def vfun(q, D):
        return (q/3600.0) / (pi * (D**2) / 4.0) if D > 0 else 0.0
    def Refun(q, D, kv):
        return vfun(q, D) * D / (kv * 1e-6) if kv > 0 else 0.0
    def ffun(q, D, rough, kv, dr):
        eps = 1e-8
        v = (q/3600.0) / (pi * (D**2) / 4.0 + eps)
        Re = v * D / (kv * 1e-6 + eps)
        arg = rough/D/3.7 + 5.74/(Re**0.9 + eps)
        fval = 0.25/(pyo.log10(arg+eps)**2) * (1-dr/100.0)
        return fval

    def headloss_expr(m, idx):
        seg = segment_map[idx]
        q = m.Q[idx]
        kv = kv_dict.get(seg['from_node'], 1.1)
        dr = m.DR_seg[idx] if idx in dra_segments else 0.0
        D = seg['D']
        rough = seg['rough']
        fseg = ffun(q, D, rough, kv, dr)
        vseg = vfun(q, D)
        return fseg * ((seg['L']*1000.0)/D) * (vseg**2/(2*g))

    # Parallel headloss constraints (DO NOT use pyo.value() here)
    pairwise = {}
    model.parallel_headloss = pyo.ConstraintList()
    for seg in segments:
        key = (seg['from_node'], seg['to_node'])
        pairwise.setdefault(key, []).append(seg['idx'])
    for segids in pairwise.values():
        if len(segids) > 1:
            hl0 = headloss_expr(model, segids[0])
            for idx in segids[1:]:
                model.parallel_headloss.add(
                    headloss_expr(model, idx) == hl0
                )

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
            D = seg['D']
            rough = seg['rough']
            f_i = ffun(q, D, rough, kv, dr)
            v_i = vfun(q, D)
            DH_peak = f_i * (L_peak / D) * (v_i**2/(2*g))
            static_h = elev_k - seg['start_elev']
            model.peak_limit.add(model.RH[seg['from_node']] - static_h - DH_peak >= 50.0)

    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    model.RH[N+1].fix(terminal.get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    for i in range(1, N+1):
        seg = next(s for s in segments if s['type'] == 'main' and s['idx']==i)
        hl = headloss_expr(model, i)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + hl
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            D = seg['D']
            rough = seg['rough']
            f_i = ffun(model.Q[i], D, rough, kv_dict[i], model.DR_seg[i] if i in dra_segments else 0.0)
            v_i = vfun(model.Q[i], D)
            DH_peak = f_i * (L_peak / D) * (v_i**2/(2*g))
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

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

    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        power_kW = (rho_i * FLOW * 9.81 * TDH[i] * model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i,0.0)
        else:
            fuel_per_kWh = (sfc.get(i,0.0)*1.34102)/820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        if i in dra_segments:
            dra_cost = (model.DR_seg[i]/4) * (FLOW*1000.0*24.0/1e6) * Rate_DRA
        else:
            dra_cost = 0
        total_cost += power_cost + dra_cost
    for seg in segments:
        if seg['type'] == 'loop':
            dra_cost = (model.DR_seg[seg['idx']]/4) * (model.Q[seg['idx']]*1000.0*24.0/1e6) * Rate_DRA if seg['idx'] in dra_segments else 0
            total_cost += dra_cost
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # ---- SOLVE ----
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)

    # === Defensive variable access helper ===
    def safe_value(var, idx):
        try:
            v = pyo.value(var[idx])
            if v is None:
                return 0.0
            return float(v)
        except Exception:
            return 0.0

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

        seg = next(s for s in segments if s['type'] == 'main' and s['idx'] == i)
        q_val = pyo.value(model.Q[i])
        D = seg['D']
        kv = kv_dict[i]
        dr = safe_value(model.DR_seg, i) if i in dra_segments else 0.0
        v_val = vfun(q_val, D)
        re_val = Refun(q_val, D, kv)
        f_val = ffun(q_val, D, seg['rough'], kv, dr)
        hl_val = pyo.value(headloss_expr(model, i))
        sdh_val = pyo.value(model.SDH[i])
        maop_val = maop_dict[i]
        result[f"flow_{name}"] = q_val
        result[f"velocity_{name}"] = v_val
        result[f"reynolds_{name}"] = re_val
        result[f"friction_{name}"] = f_val
        result[f"head_loss_{name}"] = hl_val
        result[f"sdh_{name}"] = sdh_val
        result[f"maop_{name}"] = maop_val

        if i in dra_segments:
            result[f"drag_reduction_{name}"] = dr
        else:
            result[f"drag_reduction_{name}"] = 0.0

        if i in pump_indices:
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

        peaks = stn.get('peaks', [])
        for pidx, peak in enumerate(peaks, start=1):
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            q = q_val
            dr = safe_value(model.DR_seg, i) if i in dra_segments else 0.0
            f_i = ffun(q, D, seg['rough'], kv, dr)
            v_i = vfun(q, D)
            DH_peak = f_i * (L_peak / D) * (v_i ** 2 / (2 * 9.81))
            static_h = elev_k - seg['start_elev']
            peak_head = pyo.value(model.RH[i]) - static_h - DH_peak
            result[f"peak_head_{name}_{pidx}"] = peak_head

    for seg in segments:
        if seg['type'] == 'loop':
            key = f"loopline_{seg['from_node']}_{seg['to_node']}"
            q_val = pyo.value(model.Q[seg['idx']])
            D = seg['D']
            kv = kv_dict.get(seg['from_node'], 1.1)
            dr = safe_value(model.DR_seg, seg['idx']) if seg['idx'] in dra_segments else 0.0
            v_val = vfun(q_val, D)
            re_val = Refun(q_val, D, kv)
            f_val = ffun(q_val, D, seg['rough'], kv, dr)
            hl_val = pyo.value(headloss_expr(model, seg['idx']))
            result[f"{key}_flow_m3hr"] = q_val
            result[f"{key}_velocity_ms"] = v_val
            result[f"{key}_reynolds"] = re_val
            result[f"{key}_friction"] = f_val
            result[f"{key}_head_loss_m"] = hl_val
            result[f"{key}_drag_reduction_percent"] = dr
            result[f"{key}_power_cost"] = 0.0
            if seg.get('peaks'):
                for pidx, peak in enumerate(seg['peaks'], start=1):
                    L_peak = peak['loc'] * 1000.0
                    elev_k = peak['elev']
                    q = q_val
                    f_i = ffun(q, D, seg['rough'], kv, dr)
                    v_i = vfun(q, D)
                    DH_peak = f_i * (L_peak / D) * (v_i ** 2 / (2 * 9.81))
                    static_h = elev_k - seg['start_elev']
                    peak_head = pyo.value(model.RH[seg['from_node']]) - static_h - DH_peak
                    result[f"{key}_peak_{pidx}"] = peak_head
    result['total_cost'] = float(pyo.value(model.Obj))
    return result
