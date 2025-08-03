"""
Simplified pipeline optimisation model using discrete enumeration.

This version abandons the previous Pyomo/Couenne MINLP formulation
and instead performs an exhaustive search over all feasible
combinations of pump counts, pump speeds and drag reduction levels
at each pump station.  It computes the hydraulic head required and
available for each configuration and returns the one with the lowest
total cost (power cost plus DRA cost).

Assumptions and simplifications:

* Pumps at the originating station can be of two types (A and B);
  downstream pump stations have a single type.
* All pumps at a given station operate at the same speed.
* Pump head curves follow the affinity law: head at a given RPM is
  scaled from the head at the rated (DOL) speed using
  ``(rpm/dol)**2`` and evaluated using a quadratic polynomial in
  equivalent flow (head = A*Q_eq^2 + B*Q_eq + C).
* Pump efficiency curves are evaluated via a quartic polynomial
  (efficiency = P*Q^4 + Q*Q^3 + R*Q^2 + S*Q + T) and clipped to
  [0.1, 1.0].  If efficiency is specified in percent form (values
  greater than 1), it is divided by 100.
* Drag‐reducing additive (DRA) is applied only at pump stations and
  reduces friction by a percentage.  The DRA level is discretised
  from 0 up to the station's ``max_dr`` with a step of 1 % and
  capped at 1 %.
* Residual head at each station is assumed to be the minimum
  residual head specified in the input (default 50 m).  Static
  head (elevation difference) and friction head losses must be
  overcome by the pumps.

This code is intended for demonstration purposes and may require
further refinement for production deployment (e.g. handling peaks,
MAOP constraints, and multiple downstream pump stations).  It
supports an arbitrary number of pump stations and will enumerate
combinations for all of them, but note that the search space grows
exponentially with the number of stations and should be limited
accordingly.
"""

import math
from typing import Dict, List, Tuple, Any

import numpy as np


def _calculate_friction_factor(kv: float, d: float, v: float) -> float:
    """Compute the Darcy friction factor using the Swamee–Jain equation.

    Parameters
    ----------
    kv : float
        Kinematic viscosity (cSt) converted to m^2/s via 1e-6.
    d : float
        Inner diameter of the pipe (m).
    v : float
        Flow velocity (m/s).

    Returns
    -------
    float
        Dimensionless Darcy friction factor.
    """
    if v <= 0 or d <= 0 or kv <= 0:
        return 0.0
    re = v * d / (kv * 1e-6)
    if re < 4000:
        return 64.0 / re
    roughness = 1e-4  # default roughness in metres (approx)
    arg = (roughness / (d * 3.7)) + (5.74 / (re ** 0.9))
    return 0.25 / (math.log10(arg) ** 2)


def solve_pipeline(
    stations: List[Dict[str, Any]],
    terminal: Dict[str, Any],
    FLOW: float,
    KV_list: List[float],
    rho_list: List[float],
    RateDRA: float,
    Price_HSD: float,
    linefill_dict: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Solve the pipeline optimisation by enumerating pump configurations.

    Parameters
    ----------
    stations : list of dicts
        List of station dictionaries containing geometry, pump and
        operating parameters.
    terminal : dict
        Dictionary with terminal information (e.g. elevation).
    FLOW : float
        Total pipeline flow (m^3/hr).
    KV_list : list of float
        Viscosity (cSt) for each pipeline segment.
    rho_list : list of float
        Density (kg/m^3) for each pipeline segment.
    RateDRA : float
        Cost of drag reducer per litre.
    Price_HSD : float
        Cost of diesel fuel per litre (used if pump is diesel).
    linefill_dict : dict, optional
        Unused in this simplified formulation.

    Returns
    -------
    dict
        Result dictionary containing pipeline flow, pump counts,
        speeds, efficiencies, total dynamic head, residual head,
        head losses and costs for each station, along with the
        total cost.
    """
    # Constants
    g = 9.81  # gravity (m/s^2)
    # Compute segment flows (same as original model)
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    N = len(stations)
    # Geometry and physical properties
    length = {}
    d_inner = {}
    roughness = {}
    elevation = {}
    default_t = 0.007
    default_e = 4e-5
    for i, stn in enumerate(stations, start=1):
        length[i] = stn.get('L', 0.0)
        if 'D' in stn:
            D_out = stn['D']
            thickness = stn.get('t', default_t)
            d_inner[i] = D_out - 2 * thickness
        elif 'd' in stn:
            d_inner[i] = stn['d']
        else:
            d_inner[i] = 0.7
        roughness[i] = stn.get('rough', default_e)
        elevation[i] = stn.get('elev', 0.0)
    elevation[N + 1] = terminal.get('elev', 0.0)

    # Viscosity and density per segment
    kv_dict = {i: float(KV_list[i - 1]) for i in range(1, N + 1)}
    rho_dict = {i: float(rho_list[i - 1]) for i in range(1, N + 1)}

    # Identify pump stations and origin index
    pump_indices = [i for i, stn in enumerate(stations, start=1) if stn.get('is_pump', False)]
    if not pump_indices:
        raise ValueError("No pump stations found in input")
    origin_index = pump_indices[0]

    # Build allowed RPM and DRA lists for each station
    allowed_rpms = {}
    allowed_dras = {}
    max_pumps = {}
    # For origin pump types
    orig_allowed_rpms_A = []
    orig_allowed_rpms_B = []
    max_pumps_A = 0
    max_pumps_B = 0
    for i, stn in enumerate(stations, start=1):
        if not stn.get('is_pump', False):
            continue
        if i == origin_index:
            # Pump type A
            MinRPM1 = int(stn.get('MinRPM1', 0))
            DOL1 = int(stn.get('DOL1', 0))
            orig_allowed_rpms_A = list(range(MinRPM1, DOL1 + 1, 25))
            if orig_allowed_rpms_A and orig_allowed_rpms_A[-1] != DOL1:
                orig_allowed_rpms_A.append(DOL1)
            # Pump type B
            MinRPM2 = int(stn.get('MinRPM2', 0))
            DOL2 = int(stn.get('DOL2', 0))
            orig_allowed_rpms_B = list(range(MinRPM2, DOL2 + 1, 25))
            if orig_allowed_rpms_B and orig_allowed_rpms_B[-1] != DOL2:
                orig_allowed_rpms_B.append(DOL2)
            max_pumps_A = int(stn.get('max_pumps_typeA', stn.get('max_pumps', 2)))
            max_pumps_B = int(stn.get('max_pumps_typeB', 0))
        else:
            MinRPM = int(stn.get('MinRPM', 0))
            DOL = int(stn.get('DOL', 0))
            rpm_list = list(range(MinRPM, DOL + 1, 25))
            if rpm_list and rpm_list[-1] != DOL:
                rpm_list.append(DOL)
            allowed_rpms[i] = rpm_list
            max_pumps[i] = int(stn.get('max_pumps', 2))
        # DRA levels capped at 1 %
        max_dr = min(int(stn.get('max_dr', 0)), 1)
        allowed_dras[i] = list(range(0, max_dr + 1, 1))
        if allowed_dras[i] and allowed_dras[i][-1] != max_dr:
            allowed_dras[i].append(max_dr)

    # Precompute friction heads for each segment without DRA (i.e. at 0 % DRA)
    friction_base = {}
    static_diff = {}
    for i in range(1, N + 1):
        flow_m3s = segment_flows[i] / 3600.0
        area = math.pi * (d_inner[i] ** 2) / 4.0
        v = flow_m3s / area if area > 0 else 0.0
        f = _calculate_friction_factor(kv_dict[i], d_inner[i], v)
        friction_head = f * ((length[i] * 1000.0) / d_inner[i]) * (v ** 2 / (2 * g))
        friction_base[i] = friction_head
        static_diff[i] = elevation[i + 1] - elevation[i]

    # Extract pump curves for origin and downstream pumps
    # For origin type A and B, coefficients A1,B1,C1 and A2,B2,C2; efficiency P1,Q1,R1,S1,T1 etc.
    stn0 = stations[origin_index - 1]
    A1 = float(stn0.get('A1', stn0.get('A', 0.0)))
    B1 = float(stn0.get('B1', stn0.get('B', 0.0)))
    C1 = float(stn0.get('C1', stn0.get('C', 0.0)))
    P1 = float(stn0.get('P1', stn0.get('P', 0.0)))
    Q1c = float(stn0.get('Q1', stn0.get('Q', 0.0)))
    R1c = float(stn0.get('R1', stn0.get('R', 0.0)))
    S1c = float(stn0.get('S1', stn0.get('S', 0.0)))
    T1 = float(stn0.get('T1', stn0.get('T', 0.0)))
    A2 = float(stn0.get('A2', 0.0))
    B2 = float(stn0.get('B2', 0.0))
    C2 = float(stn0.get('C2', 0.0))
    P2 = float(stn0.get('P2', 0.0))
    Q2c = float(stn0.get('Q2', 0.0))
    R2c = float(stn0.get('R2', 0.0))
    S2c = float(stn0.get('S2', 0.0))
    T2 = float(stn0.get('T2', 0.0))
    DOL1 = int(stn0.get('DOL1', stn0.get('DOL', 1)))
    DOL2 = int(stn0.get('DOL2', stn0.get('DOL', 1)))
    rate1 = float(stn0.get('rate1', stn0.get('rate', 0.0)))
    sfc1 = float(stn0.get('sfc1', stn0.get('sfc', 0.0)))
    rate2 = float(stn0.get('rate2', stn0.get('rate', 0.0)))
    sfc2 = float(stn0.get('sfc2', stn0.get('sfc', 0.0)))

    # Downstream pumps (if any) coefficients stored by station index
    downstream_head_coeffs: Dict[int, Tuple[float, float, float]] = {}
    downstream_eff_coeffs: Dict[int, Tuple[float, float, float, float, float]] = {}
    downstream_DOL: Dict[int, int] = {}
    downstream_rate: Dict[int, float] = {}
    downstream_sfc: Dict[int, float] = {}
    for i in pump_indices:
        if i == origin_index:
            continue
        stn = stations[i - 1]
        downstream_head_coeffs[i] = (
            float(stn.get('A', 0.0)),
            float(stn.get('B', 0.0)),
            float(stn.get('C', 0.0)),
        )
        downstream_eff_coeffs[i] = (
            float(stn.get('P', 0.0)),
            float(stn.get('Q', 0.0)),
            float(stn.get('R', 0.0)),
            float(stn.get('S', 0.0)),
            float(stn.get('T', 0.0)),
        )
        downstream_DOL[i] = int(stn.get('DOL', 1))
        # Determine power type: electric (use rate) or diesel (use sfc)
        if float(stn.get('sfc', 0.0)) not in (None, 0.0):
            downstream_sfc[i] = float(stn.get('sfc', 0.0))
            downstream_rate[i] = 0.0
        else:
            downstream_sfc[i] = 0.0
            downstream_rate[i] = float(stn.get('rate', 0.0))

    # Helper to compute head and efficiency for a pump type at a given rpm
    def pump_head_eff(
        A: float,
        B: float,
        C: float,
        P: float,
        Qc: float,
        Rc: float,
        Sc: float,
        T: float,
        DOL: int,
        rpm: float,
        flow: float,
    ) -> Tuple[float, float]:
        """Return pump head and efficiency for a given flow and speed.

        This function evaluates the pump head and efficiency based on the
        supplied polynomial coefficients.  It uses the affinity law to
        scale the head from the rated (DOL) speed and clips the
        efficiency to a reasonable range.
        """
        # Convert flow to equivalent flow at DOL speed
        Q_equiv = flow * DOL / rpm if rpm > 0 else 0.0
        # Head at rated speed
        head_DOL = A * Q_equiv ** 2 + B * Q_equiv + C
        # Scale head to the actual rpm
        head = head_DOL * (rpm / DOL) ** 2 if DOL > 0 else 0.0
        # Efficiency polynomial
        eff = P * Q_equiv ** 4 + Qc * Q_equiv ** 3 + Rc * Q_equiv ** 2 + Sc * Q_equiv + T
        # Convert efficiency to fraction if > 1
        if eff > 1.0:
            eff = eff / 100.0
        # Clip efficiency to avoid zero or negative values
        eff = max(0.1, min(1.0, eff))
        return head, eff

    # Evaluate a candidate combination and return cost and details
    def evaluate_combination(config: Dict[int, Any]) -> Tuple[float, Dict[str, Any]]:
        # Determine head requirement and available head at each pump station
        total_cost = 0.0
        results = {}
        # We'll maintain residual heads; start with fixed residual at first station
        residual_prev = stations[0].get('min_residual', 50.0)
        feasible = True
        for i in range(1, N + 1):
            stn = stations[i - 1]
            name = stn['name'].strip().lower().replace(' ', '_')
            flow_i = segment_flows[i]
            rho_i = rho_dict[i]
            friction_head = friction_base[i]  # base friction head (no DRA)
            # Apply DRA reduction if any at this segment
            if i in allowed_dras:
                dra_i = config.get(i, (0, 0, orig_allowed_rpms_A[0], orig_allowed_rpms_B[0], 0))
                # For origin, config[i] is a tuple (numA,numB,rpmA,rpmB,dra)
                # For downstream, config[i] is (num,rpm,dra)
                if i == origin_index:
                    dr_frac = dra_i[4] / 100.0
                else:
                    dr_frac = dra_i[2] / 100.0
            else:
                dr_frac = 0.0
            friction_head_i = friction_head * (1 - dr_frac)
            required_head = friction_head_i + static_diff[i]
            # Compute available head and initialise cost components
            available_head = 0.0
            station_cost = 0.0
            power_cost_i = 0.0
            dra_cost_i = 0.0
            # Check if station has pumps
            if stn.get('is_pump', False):
                if i == origin_index:
                    numA, numB, rpmA, rpmB, dra_val = config[i]
                    # Type A pumps
                    if numA > 0:
                        headA, effA = pump_head_eff(
                            A1, B1, C1, P1, Q1c, R1c, S1c, T1, DOL1, rpmA, flow_i
                        )
                        available_head += numA * headA
                        # Compute power cost for A
                        # Determine power type
                        if sfc1 not in (None, 0.0):
                            fuel_per_kWh = (sfc1 * 1.34102) / 820.0
                            power_kW = (rho_i * flow_i * 9.81 * headA) / (3600.0 * 1000.0 * effA) * numA
                            power_cost_i += power_kW * 24.0 * fuel_per_kWh * Price_HSD
                        else:
                            power_kW = (rho_i * flow_i * 9.81 * headA) / (3600.0 * 1000.0 * effA) * numA
                            power_cost_i += power_kW * 24.0 * rate1
                    # Type B pumps
                    if numB > 0:
                        headB, effB = pump_head_eff(
                            A2, B2, C2, P2, Q2c, R2c, S2c, T2, DOL2, rpmB, flow_i
                        )
                        available_head += numB * headB
                        if sfc2 not in (None, 0.0):
                            fuel_per_kWh = (sfc2 * 1.34102) / 820.0
                            power_kW = (rho_i * flow_i * 9.81 * headB) / (3600.0 * 1000.0 * effB) * numB
                            power_cost_i += power_kW * 24.0 * fuel_per_kWh * Price_HSD
                        else:
                            power_kW = (rho_i * flow_i * 9.81 * headB) / (3600.0 * 1000.0 * effB) * numB
                            power_cost_i += power_kW * 24.0 * rate2
                else:
                    num, rpm_i, dra_val = config[i]
                    if num > 0:
                        # Head and efficiency using downstream coefficients
                        Acoef, Bcoef, Ccoef = downstream_head_coeffs[i]
                        Pcoef, Qcoef, Rcoef, Scoef, Tcoef = downstream_eff_coeffs[i]
                        DOLi = downstream_DOL[i]
                        head_i, eff_i = pump_head_eff(
                            Acoef, Bcoef, Ccoef,
                            Pcoef, Qcoef, Rcoef, Scoef, Tcoef,
                            DOLi, rpm_i, flow_i
                        )
                        available_head += num * head_i
                        # Power cost
                        if downstream_sfc[i] not in (None, 0.0):
                            fuel_per_kWh = (downstream_sfc[i] * 1.34102) / 820.0
                            power_kW = (rho_i * flow_i * 9.81 * head_i) / (3600.0 * 1000.0 * eff_i) * num
                            power_cost_i += power_kW * 24.0 * fuel_per_kWh * Price_HSD
                        else:
                            power_kW = (rho_i * flow_i * 9.81 * head_i) / (3600.0 * 1000.0 * eff_i) * num
                            power_cost_i += power_kW * 24.0 * downstream_rate[i]
                # DRA cost for this station
                dra_cost_i = dra_val * (flow_i * 1000.0 * 24.0 / 1e6) * RateDRA
                station_cost = power_cost_i + dra_cost_i
            # Check feasibility
            if available_head < required_head - 1e-6:
                feasible = False
                break
            total_cost += station_cost
            # Update residual head for next segment (assume constant residual)
            residual_prev = stations[i - 1].get('min_residual', 50.0)
            # Store results for reporting
            results[f"pipeline_flow_{name}"] = flow_i
            results[f"pump_flow_{name}"] = flow_i if stn.get('is_pump', False) else 0.0
            if stn.get('is_pump', False):
                if i == origin_index:
                    numA, numB, rpmA, rpmB, dra_val = config[i]
                    results[f"num_pumps_{name}"] = numA + numB
                    results[f"num_pumps_typeA_{name}"] = numA
                    results[f"num_pumps_typeB_{name}"] = numB
                    results[f"speed_{name}"] = max(rpmA, rpmB) if (numA + numB) > 0 else 0.0
                    results[f"speed_typeA_{name}"] = rpmA if numA > 0 else 0.0
                    results[f"speed_typeB_{name}"] = rpmB if numB > 0 else 0.0
                    # Compute aggregated efficiency for display (average across all running pumps)
                    effs = []
                    if numA > 0:
                        # Recompute efficiency for reporting
                        headA_tmp, effA_tmp = pump_head_eff(
                            A1, B1, C1, P1, Q1c, R1c, S1c, T1, DOL1, rpmA, flow_i
                        )
                        effs.extend([effA_tmp] * numA)
                    if numB > 0:
                        headB_tmp, effB_tmp = pump_head_eff(
                            A2, B2, C2, P2, Q2c, R2c, S2c, T2, DOL2, rpmB, flow_i
                        )
                        effs.extend([effB_tmp] * numB)
                    avg_efficiency = (sum(effs) / len(effs)) if effs else 0.0
                    results[f"efficiency_{name}"] = avg_efficiency * 100.0
                    results[f"efficiency_typeA_{name}"] = (effA_tmp * 100.0) if numA > 0 else 0.0
                    results[f"efficiency_typeB_{name}"] = (effB_tmp * 100.0) if numB > 0 else 0.0
                    # TDH per pump approximated as available_head / (numA + numB)
                    tdh_per_pump = available_head / (numA + numB) if (numA + numB) > 0 else 0.0
                    results[f"tdh_{name}"] = tdh_per_pump
                    results[f"tdh_typeA_{name}"] = tdh_per_pump if numA > 0 else 0.0
                    results[f"tdh_typeB_{name}"] = tdh_per_pump if numB > 0 else 0.0
                else:
                    num, rpm_i, dra_val = config[i]
                    results[f"num_pumps_{name}"] = num
                    results[f"num_pumps_typeA_{name}"] = num
                    results[f"num_pumps_typeB_{name}"] = 0
                    results[f"speed_{name}"] = rpm_i if num > 0 else 0.0
                    results[f"speed_typeA_{name}"] = rpm_i if num > 0 else 0.0
                    results[f"speed_typeB_{name}"] = 0.0
                    # Efficiency reporting for downstream pumps
                    if num > 0:
                        results[f"efficiency_{name}"] = eff_i * 100.0
                        results[f"efficiency_typeA_{name}"] = eff_i * 100.0
                        results[f"efficiency_typeB_{name}"] = 0.0
                    else:
                        results[f"efficiency_{name}"] = 0.0
                        results[f"efficiency_typeA_{name}"] = 0.0
                        results[f"efficiency_typeB_{name}"] = 0.0
                    tdh_per_pump = available_head / num if num > 0 else 0.0
                    results[f"tdh_{name}"] = tdh_per_pump
                    results[f"tdh_typeA_{name}"] = tdh_per_pump if num > 0 else 0.0
                    results[f"tdh_typeB_{name}"] = 0.0
            else:
                results[f"num_pumps_{name}"] = 0
                results[f"num_pumps_typeA_{name}"] = 0
                results[f"num_pumps_typeB_{name}"] = 0
                results[f"speed_{name}"] = 0.0
                results[f"speed_typeA_{name}"] = 0.0
                results[f"speed_typeB_{name}"] = 0.0
                results[f"tdh_{name}"] = 0.0
                results[f"tdh_typeA_{name}"] = 0.0
                results[f"tdh_typeB_{name}"] = 0.0
            results[f"head_loss_{name}"] = friction_head_i
            results[f"residual_head_{name}"] = residual_prev
            results[f"sdh_{name}"] = required_head + residual_prev  # approximate SDH
            results[f"maop_{name}"] = float('inf')  # not evaluated
            results[f"drag_reduction_{name}"] = dr_frac * 100.0
            # Store cost components
            results[f"power_cost_{name}"] = power_cost_i
            results[f"dra_cost_{name}"] = dra_cost_i
        if not feasible:
            return float('inf'), {}
        return total_cost, results

    # Enumeration of all pump combinations
    best_cost = float('inf')
    best_results = None
    config: Dict[int, Any] = {}

    # Recursive enumeration function
    def enumerate_station(idx: int):
        nonlocal best_cost, best_results, config
        if idx > N:
            # Evaluate complete configuration
            cost, res = evaluate_combination(config)
            if cost < best_cost:
                best_cost = cost
                best_results = res.copy()
            return
        stn = stations[idx - 1]
        if not stn.get('is_pump', False):
            # No pumps at this station
            config[idx] = (0, 0, 0, 0, 0)  # placeholder
            enumerate_station(idx + 1)
            return
        if idx == origin_index:
            # Origin: enumerate type A and B pumps
            for numA in range(0, max_pumps_A + 1):
                for numB in range(0, max_pumps_B + 1):
                    if numA + numB == 0:
                        continue  # must have at least one pump
                    for rpmA in orig_allowed_rpms_A:
                        for rpmB in orig_allowed_rpms_B:
                            for dra in allowed_dras[idx]:
                                config[idx] = (numA, numB, rpmA, rpmB, dra)
                                enumerate_station(idx + 1)
        else:
            # Downstream pump station with single type
            maxp = max_pumps.get(idx, 0)
            for num in range(0, maxp + 1):
                for rpm in allowed_rpms.get(idx, [0]):
                    for dra in allowed_dras[idx]:
                        config[idx] = (num, rpm, dra)
                        enumerate_station(idx + 1)

    enumerate_station(1)

    # Add terminal results
    term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    if best_results is None:
        # No feasible configuration found
        return {
            "error": True,
            "message": "No feasible pump configuration found",
        }
    best_results[f"pipeline_flow_{term_name}"] = segment_flows[-1]
    best_results[f"pump_flow_{term_name}"] = 0.0
    best_results[f"num_pumps_{term_name}"] = 0
    best_results[f"num_pumps_typeA_{term_name}"] = 0
    best_results[f"num_pumps_typeB_{term_name}"] = 0
    best_results[f"speed_{term_name}"] = 0.0
    best_results[f"speed_typeA_{term_name}"] = 0.0
    best_results[f"speed_typeB_{term_name}"] = 0.0
    best_results[f"tdh_{term_name}"] = 0.0
    best_results[f"tdh_typeA_{term_name}"] = 0.0
    best_results[f"tdh_typeB_{term_name}"] = 0.0
    best_results[f"head_loss_{term_name}"] = 0.0
    best_results[f"residual_head_{term_name}"] = stations[-1].get('min_residual', 50.0)
    best_results[f"sdh_{term_name}"] = 0.0
    best_results[f"maop_{term_name}"] = float('inf')
    best_results[f"drag_reduction_{term_name}"] = 0.0
    best_results['total_cost'] = best_cost
    best_results['error'] = False
    return best_results
