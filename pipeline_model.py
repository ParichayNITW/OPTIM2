"""Core optimisation routines for Pipeline Optima.

This module contains the mathematical model for the pipeline optimisation
problem along with helper utilities.  The previous iterations of the project
grew organically and a number of small helper functions were scattered across
the file.  This refactor consolidates common conversions, documents the public
functions and removes unused code so the module is easier to reason about and
extend.
"""

from __future__ import annotations

import os
from math import log10, pi

import copy
import io
import contextlib
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
import logging
import socket

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'parichay.nitwarangal@gmail.com')

# Suppress verbose Pyomo warnings so infeasible runs don't flood the UI
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
logging.getLogger('pyomo.solvers').setLevel(logging.ERROR)

# DRA curve files
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
    if os.path.exists(fname):
        DRA_CURVE_DATA[cst] = pd.read_csv(fname)
    else:
        DRA_CURVE_DATA[cst] = None


def _neos_available(host: str = "neos-server.org", port: int = 3333, timeout: int = 5) -> bool:
    """Return ``True`` if the NEOS server appears reachable.

    Parameters
    ----------
    host: str
        Hostname of the NEOS server.
    port: int
        Port number for the NEOS XML-RPC interface.
    timeout: int
        Connection timeout in seconds.
    """

    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def head_to_kgcm2(head_m: float, rho: float) -> float:
    """Convert a head value in metres to kg/cm².

    Parameters
    ----------
    head_m: float
        Head value expressed in metres.
    rho: float
        Density of the fluid in kg/m³.
    """

    return head_m * rho / 10000.0

def get_ppm_breakpoints(visc: float) -> tuple[list[float], list[float]]:
    """Return drag-reduction/PPM breakpoints for ``visc``.

    Parameters
    ----------
    visc:
        Fluid viscosity in cSt.

    Returns
    -------
    tuple[list[float], list[float]]
        Matching lists of drag-reduction percentages and PPM values suitable
        for a :class:`pyomo.environ.Piecewise` definition.
    """

    cst_list = sorted([c for c in DRA_CURVE_DATA.keys() if DRA_CURVE_DATA[c] is not None])
    visc = float(visc)
    if not cst_list:
        return [0.0], [0.0]
    if visc <= cst_list[0]:
        df = DRA_CURVE_DATA[cst_list[0]]
    elif visc >= cst_list[-1]:
        df = DRA_CURVE_DATA[cst_list[-1]]
    else:
        lower = max([c for c in cst_list if c <= visc])
        upper = min([c for c in cst_list if c >= visc])
        df_lower = DRA_CURVE_DATA[lower]
        df_upper = DRA_CURVE_DATA[upper]
        x_lower, y_lower = df_lower['%Drag Reduction'].values, df_lower['PPM'].values
        x_upper, y_upper = df_upper['%Drag Reduction'].values, df_upper['PPM'].values
        dr_points = np.unique(np.concatenate((x_lower, x_upper)))
        ppm_points = np.interp(dr_points, x_lower, y_lower) * (upper - visc) / (upper - lower) + \
                     np.interp(dr_points, x_upper, y_upper) * (visc - lower) / (upper - lower)
        unique_dr, unique_indices = np.unique(dr_points, return_index=True)
        unique_ppm = ppm_points[unique_indices]
        return list(unique_dr), list(unique_ppm)
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    return list(unique_x), list(unique_y)

def generate_origin_combinations(
    maxA: int = 2,
    maxB: int = 2,
    max_total: int | None = None,
) -> list[tuple[int, int]]:
    """Return all feasible pump count combinations for the origin station.

    Parameters
    ----------
    maxA, maxB:
        Maximum available pumps of type ``A`` and ``B``.
    max_total:
        Optional upper bound on the total number of pumps to consider.

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of ``(numA, numB)`` tuples.
    """

    maxA = int(maxA)
    maxB = int(maxB)
    if max_total is not None:
        max_total = int(max_total)

    combos = []
    for a in range(maxA + 1):
        for b in range(maxB + 1):
            if a + b == 0:
                continue
            if max_total is not None and a + b > max_total:
                continue
            combos.append((a, b))

    return sorted(combos, key=lambda x: (x[0] + x[1], x))

def solve_pipeline_multi_origin(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill_dict: dict | None = None,
    solver_timeout: float = 600,
) -> dict:
    """Enumerate pump combinations at the origin and select the least cost."""

    origin_index = next(i for i, s in enumerate(stations) if s.get('is_pump', False))
    origin_station = stations[origin_index]
    pump_types = origin_station.get('pump_types', {})
    combos = generate_origin_combinations(
        pump_types.get('A', {}).get('available', 0),
        pump_types.get('B', {}).get('available', 0),
        origin_station.get('max_pumps')
    )

    if not combos:
        return {
            "error": True,
            "message": "No pump types enabled at origin station.",
        }

    # Pre- and post-origin segments stay untouched; only the origin station is expanded
    pre_stations = copy.deepcopy(stations[:origin_index])
    post_stations = copy.deepcopy(stations[origin_index + 1:])
    pre_kv = KV_list[:origin_index]
    post_kv = KV_list[origin_index + 1:]
    pre_rho = rho_list[:origin_index]
    post_rho = rho_list[origin_index + 1:]

    pump_visc = KV_list[origin_index]
    pump_rho = rho_list[origin_index]

    best_result = None
    best_stations = None
    evaluated = []
    attempts: list[dict] = []

    for numA, numB in combos:
        if numA > 0 and not pump_types.get('A'):
            continue
        if numB > 0 and not pump_types.get('B'):
            continue

        stations_combo: list[dict] = []
        kv_combo: list[float] = []
        rho_combo: list[float] = []

        name_base = origin_station['name']
        pump_units = []
        for ptype, count in [('A', numA), ('B', numB)]:
            pdata = pump_types.get(ptype)
            label = pdata.get('name', ptype) if pdata else ptype
            for n in range(count):
                unit = {
                    'name': f"{name_base}_{label}{n+1}",
                    'elev': origin_station.get('elev', 0.0),
                    'D': origin_station.get('D'),
                    't': origin_station.get('t'),
                    'SMYS': origin_station.get('SMYS'),
                    'rough': origin_station.get('rough'),
                    'L': 0.0,
                    'is_pump': True,
                    'head_data': pdata.get('head_data') if pdata else None,
                    'eff_data': pdata.get('eff_data') if pdata else None,
                    'power_type': pdata.get('power_type', 'Grid') if pdata else 'Grid',
                    'rate': pdata.get('rate', 0.0) if pdata else 0.0,
                    'sfc': pdata.get('sfc', 0.0) if pdata else 0.0,
                    'MinRPM': pdata.get('MinRPM', 0.0) if pdata else 0.0,
                    'DOL': pdata.get('DOL', 0.0) if pdata else 0.0,
                    'max_pumps': 1,
                    'min_pumps': 1,
                    'delivery': 0.0,
                    'supply': 0.0,
                    'max_dr': 0.0,
                }
                pump_units.append(unit)

        if not pump_units:
            continue

        # Attach origin-station data: deliveries occur after the last pump
        pump_units[0]['min_residual'] = origin_station.get('min_residual', 50.0)
        pump_units[-1]['delivery'] = origin_station.get('delivery', 0.0)
        pump_units[-1]['supply'] = origin_station.get('supply', 0.0)
        pump_units[-1]['L'] = origin_station.get('L', 0.0)
        pump_units[-1]['max_dr'] = origin_station.get('max_dr', 0.0)

        # Assemble full station list and corresponding property vectors
        stations_combo.extend(pre_stations)
        stations_combo.extend(pump_units)
        stations_combo.extend(post_stations)

        kv_combo.extend(pre_kv)
        kv_combo.extend([pump_visc] * len(pump_units))
        kv_combo.extend(post_kv)

        rho_combo.extend(pre_rho)
        rho_combo.extend([pump_rho] * len(pump_units))
        rho_combo.extend(post_rho)

        try:
            result = solve_pipeline(
                stations_combo,
                terminal,
                FLOW,
                kv_combo,
                rho_combo,
                RateDRA,
                Price_HSD,
                linefill_dict,
                solver_timeout=solver_timeout,
            )
        except Exception as exc:  # pragma: no cover - defensive
            result = {"error": True, "message": str(exc)}
        if result.get("error"):
            attempts.append({"A": numA, "B": numB, "message": result.get("message", "")})
            continue

        cost = result.get("total_cost", float('inf'))
        combo_names = {}
        if numA:
            combo_names[pump_types.get('A', {}).get('name', 'A')] = numA
        if numB:
            combo_names[pump_types.get('B', {}).get('name', 'B')] = numB
        evaluated.append((cost, result, stations_combo, combo_names))

    if evaluated:
        _, best_result, best_stations, combo_names = min(evaluated, key=lambda x: x[0])
        best_result['pump_combo'] = combo_names
        if attempts:
            best_result['attempted_combos'] = attempts

    if best_result is None:
        return {
            "error": True,
            "message": "No feasible pump combination found for originating station.",
            "attempted_combos": attempts,
        }

    best_result['stations_used'] = best_stations
    return best_result

def solve_pipeline(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    linefill_dict: dict | None = None,
    solver_timeout: float = 600,
) -> dict:
    """Solve the pipeline optimisation for a fixed station configuration."""

    import numpy as np
    import pandas as pd

    def safe_polyfit(x, y, degree):
        if len(x) >= degree + 1:
            return np.polyfit(x, y, degree)
        else:
            return [0] * (degree + 1)

    # ---- ALWAYS FIT COEFFICIENTS FRESH FROM DATA ----
    for idx, stn in enumerate(stations):
        # HEAD CURVE FIT (Quadratic)
        head_df = None
        if "head_data" in stn and stn["head_data"] is not None:
            if isinstance(stn["head_data"], pd.DataFrame):
                head_df = stn["head_data"]
            elif isinstance(stn["head_data"], list):
                head_df = pd.DataFrame(stn["head_data"])
            elif isinstance(stn["head_data"], dict):
                head_df = pd.DataFrame(stn["head_data"])
        if head_df is not None and len(head_df) >= 3:
            x = head_df.iloc[:,0].values
            y = head_df.iloc[:,1].values
            A, B, C = safe_polyfit(x, y, 2)
            stn['A'] = float(A)
            stn['B'] = float(B)
            stn['C'] = float(C)
        # EFFICIENCY CURVE FIT (Quartic)
        eff_df = None
        if "eff_data" in stn and stn["eff_data"] is not None:
            if isinstance(stn["eff_data"], pd.DataFrame):
                eff_df = stn["eff_data"]
            elif isinstance(stn["eff_data"], list):
                eff_df = pd.DataFrame(stn["eff_data"])
            elif isinstance(stn["eff_data"], dict):
                eff_df = pd.DataFrame(stn["eff_data"])
        if eff_df is not None and len(eff_df) >= 5:
            x = eff_df.iloc[:,0].values
            y = eff_df.iloc[:,1].values
            P, Q, R, S, T = safe_polyfit(x, y, 4)
            stn['P'] = float(P)
            stn['Q'] = float(Q)
            stn['R'] = float(R)
            stn['S'] = float(S)
            stn['T'] = float(T)
    # ---- END COEFFICIENT FIT BLOCK ----
    RPM_STEP = 100  # RPM step
    DRA_STEP = 5    # DRA step

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

    # Compute flow in each segment
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        if out_flow < -1e-6:
            name = stn.get('name', '?')
            raise ValueError(
                f"Negative downstream flow after station {name}: "
                f"{prev_flow} - {delivery} + {supply} = {out_flow}"
            )
        segment_flows.append(out_flow)

    # Pipeline and pump parameters
    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    max_dr = {}
    peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    allowed_rpms = {}
    allowed_dras = {}

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
            minval = int(min_rpm[i])
            maxval = int(max_rpm[i])
            allowed_rpms[i] = [r for r in range(minval, maxval+1, RPM_STEP)]
            if allowed_rpms[i][-1] != maxval:
                allowed_rpms[i].append(maxval)
            maxval_dra = int(max_dr[i])
            allowed_dras[i] = [d for d in range(0, maxval_dra+1, DRA_STEP)]
            if allowed_dras[i][-1] != maxval_dra:
                allowed_dras[i].append(maxval_dra)

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

    # Identify the originating pump station (first with is_pump=True)
    originating_pump_index = None
    for idx, stn in enumerate(stations, start=1):  # 1-based indexing!
        if stn.get('is_pump', False):
            originating_pump_index = idx
            break
    if originating_pump_index is None:
        raise ValueError("No originating pump station found in input!")
    
    def nop_bounds(m, j):
        st = stations[j-1]
        lb = st.get('min_pumps', 1 if j == originating_pump_index else 0)
        ub = st.get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)

    # ---- RPM selection via binaries ----
    model.rpm_bin = pyo.Var(
        ((i, j) for i in pump_indices for j in range(len(allowed_rpms[i]))),
        domain=pyo.Binary
    )
    def rpm_bin_sum_rule(m, i):
        return sum(m.rpm_bin[i, j] for j in range(len(allowed_rpms[i]))) == 1
    model.rpm_bin_sum = pyo.Constraint(model.pump_stations, rule=rpm_bin_sum_rule)
    model.RPM_var = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    def rpm_value_rule(m, i):
        return m.RPM_var[i] == sum(allowed_rpms[i][j] * m.rpm_bin[i, j] for j in range(len(allowed_rpms[i])))
    model.rpm_value = pyo.Constraint(model.pump_stations, rule=rpm_value_rule)

    # ---- DRA selection via binaries ----
    model.dra_bin = pyo.Var(
        ((i, j) for i in pump_indices for j in range(len(allowed_dras[i]))),
        domain=pyo.Binary
    )
    def dra_bin_sum_rule(m, i):
        return sum(m.dra_bin[i, j] for j in range(len(allowed_dras[i]))) == 1
    model.dra_bin_sum = pyo.Constraint(model.pump_stations, rule=dra_bin_sum_rule)
    def dra_var_bounds(m, i):
        return (min(allowed_dras[i]), max(allowed_dras[i]))
    model.DR_var = pyo.Var(model.pump_stations, bounds=dra_var_bounds, domain=pyo.NonNegativeReals)

    def dra_value_rule(m, i):
        return m.DR_var[i] == sum(allowed_dras[i][j] * m.dra_bin[i, j] for j in range(len(allowed_dras[i])))
    model.dra_value = pyo.Constraint(model.pump_stations, rule=dra_value_rule)

    # Residual head constraints
    term_min = terminal.get('min_residual', 50.0)
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=term_min)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+1):
        model.RH[j].setlb(50.0)
    model.RH[N+1].setlb(term_min)

    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        flow_m3s = float(segment_flows[i]) / 3600.0
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        kv = kv_dict[i]
        if kv > 0:
            Re[i] = v[i] * d_inner[i] / (float(kv) * 1e-6)
        else:
            Re[i] = 0.0
        if Re[i] > 0:
            if Re[i] < 4000:
                f[i] = 64.0 / Re[i]
            else:
                arg = (roughness[i] / d_inner[i] / 3.7) + (5.74 / (Re[i]**0.9))
                f[i] = 0.25 / (log10(arg)**2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}
    EFFP = {}

    for i in range(1, N+1):
        if i in pump_indices:
            DR_frac = model.DR_var[i] / 100.0
        else:
            DR_frac = 0.0
        DH_next = f[i] * ((length[i]*1000.0)/d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g)) * (1 - DR_frac)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

        if i in pump_indices:
            pump_flow_i = float(segment_flows[i])
            rpm_val = model.RPM_var[i]
            dol_val = model.DOL[i]
            Q_equiv = pump_flow_i * dol_val / rpm_val
            H_DOL = model.A[i] * Q_equiv**2 + model.B[i] * Q_equiv + model.C[i]
            TDH[i] = H_DOL * (rpm_val / dol_val)**2
            EFFP[i] = (model.Pcoef[i]*Q_equiv**4 + model.Qcoef[i]*Q_equiv**3 +
                       model.Rcoef[i]*Q_equiv**2 + model.Scoef[i]*Q_equiv +
                       model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N+1):
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        D_out = d_inner[i] + 2 * thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            loss_no_dra = f[i] * (L_peak / d_inner[i]) * (v[i]**2 / (2*g))
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    model.PPM = pyo.Var(model.pump_stations, domain=pyo.NonNegativeReals)
    model.dra_cost = pyo.Expression(model.pump_stations)
    for i in pump_indices:
        visc = kv_dict[i]
        dr_points, ppm_points = get_ppm_breakpoints(visc)
        dr_points_fixed, ppm_points_fixed = zip(*sorted(set(zip(dr_points, ppm_points))))
        setattr(model, f'piecewise_dra_ppm_{i}',
            pyo.Piecewise(f'pw_dra_ppm_{i}', model.PPM[i], model.DR_var[i],
                          pw_pts=dr_points_fixed,
                          f_rule=ppm_points_fixed,
                          pw_constr_type='EQ'))
        dra_cost_expr = model.PPM[i] * (segment_flows[i] * 1000.0 * 24.0 / 1e6) * RateDRA
        model.dra_cost[i] = dra_cost_expr

    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        pump_flow_i = float(segment_flows[i])
        rpm_val = model.RPM_var[i]
        eff_val = EFFP[i]
        power_kW = (rho_i * pump_flow_i * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * eff_val * 0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i, 0.0)
        else:
            fuel_per_kWh = (sfc.get(i,0.0) * 1.34102) / 820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        dra_cost_i = model.dra_cost[i]
        total_cost += power_cost + dra_cost_i
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # Solve without auto-loading so infeasible runs don't emit warnings
    if not _neos_available():
        return {
            "error": True,
            "message": "NEOS server unreachable. Check your internet connection and try again later.",
        }
    stream = io.StringIO()
    try:
        # Capture both stdout and stderr so parse errors from NEOS do not
        # leak stack traces to the caller.  Any messages returned by the
        # remote solver are consolidated into ``solver_output``.
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            results = SolverManagerFactory('neos').solve(
                model,
                solver='couenne',
                tee=False,
                load_solutions=False,
                options={'timelimit': solver_timeout},
            )
    except Exception as exc:  # pragma: no cover - network failure path
        output = stream.getvalue().strip()
        return {
            "error": True,
            "message": f"NEOS solver error: {exc}",
            "solver_output": output,
        }

    status = results.solver.status
    term = results.solver.termination_condition
    if term == pyo.TerminationCondition.maxTimeLimit:
        return {
            "error": True,
            "message": f"Optimization exceeded time limit of {solver_timeout} seconds.",
            "termination_condition": str(term),
            "solver_status": str(status),
        }
    if (status != pyo.SolverStatus.ok) or (term != pyo.TerminationCondition.optimal):
        return {
            "error": True,
            "message": f"Optimization failed: {term}. Please check your input values and relax constraints if necessary.",
            "termination_condition": str(term),
            "solver_status": str(status)
        }
    model.solutions.load_from(results)

    # Collect results
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        inflow = segment_flows[i-1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0

        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i])) if model.NOP[i].value is not None else 0
            # RPM and DRA value selection
            rpm_val = None
            for j in range(len(allowed_rpms[i])):
                if round(pyo.value(model.rpm_bin[i, j])) == 1:
                    rpm_val = allowed_rpms[i][j]
                    break
            dra_perc = None
            for j in range(len(allowed_dras[i])):
                if round(pyo.value(model.dra_bin[i, j])) == 1:
                    dra_perc = allowed_dras[i][j]
                    break
            if rpm_val is None:
                rpm_val = allowed_rpms[i][0]
            if dra_perc is None:
                dra_perc = allowed_dras[i][0]
            dol_val = model.DOL[i]
            pump_flow_i = float(segment_flows[i])
            Q_equiv = pump_flow_i * dol_val / rpm_val
            tdh_val = float(model.A[i] * Q_equiv**2 + model.B[i] * Q_equiv + model.C[i]) * (rpm_val/dol_val)**2
            eff = (model.Pcoef[i]*Q_equiv**4 + model.Qcoef[i]*Q_equiv**3 +
                   model.Rcoef[i]*Q_equiv**2 + model.Scoef[i]*Q_equiv +
                   model.Tcoef[i]) if num_pumps > 0 else 0.0
            eff = float(eff)
            dra_ppm = float(pyo.value(model.PPM[i])) if model.PPM[i].value is not None else 0.0
            dra_cost_i = float(pyo.value(model.dra_cost[i])) if model.dra_cost[i].expr is not None else 0.0
        
            # If optimizer turned off all pumps, zero all reporting values:
            if num_pumps == 0:
                rpm_val = 0.0
                eff = 0.0
                dra_perc = 0.0
                dra_ppm = 0.0
                dra_cost_i = 0.0
                tdh_val = 0.0
        
        else:
            num_pumps = 0
            rpm_val = 0.0
            eff = 0.0
            dra_perc = 0.0
            dra_ppm = 0.0
            dra_cost_i = 0.0
            tdh_val = 0.0


        if i in pump_indices and num_pumps > 0:
            rho_i = rho_dict[i]
            power_kW = (rho_i * pump_flow * 9.81 * tdh_val * num_pumps) / (3600.0 * 1000.0 * (eff/100.0) * 0.95) if eff > 0 else 0.0
            if i in electric_pumps:
                rate = elec_cost.get(i, 0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i, 0.0)
                fuel_per_kWh = (sfc_val * 1.34102) / 820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        else:
            power_cost = 0.0

        drag_red = dra_perc
        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i+1].value is not None else 0.0
        res_head = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
        sdh_val = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
        if i != originating_pump_index and stn.get('is_pump', False) and num_pumps == 0:
            sdh_val = res_head
        rho_i = rho_dict[i]
        velocity = v[i]; reynolds = Re[i]; fric = f[i]
        head_loss_kg = head_to_kgcm2(head_loss, rho_i)
        rh_kg = head_to_kgcm2(res_head, rho_i)
        sdh_kg = head_to_kgcm2(sdh_val, rho_i)
        maop_kg = head_to_kgcm2(maop_dict[i], rho_i)

        result[f"pipeline_flow_{name}"] = outflow
        result[f"pipeline_flow_in_{name}"] = inflow
        result[f"pump_flow_{name}"] = pump_flow
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = rpm_val
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost
        result[f"dra_cost_{name}"] = dra_cost_i
        result[f"dra_ppm_{name}"] = dra_ppm
        result[f"drag_reduction_{name}"] = drag_red
        result[f"head_loss_{name}"] = head_loss
        result[f"head_loss_kgcm2_{name}"] = head_loss_kg
        result[f"residual_head_{name}"] = res_head
        result[f"rh_kgcm2_{name}"] = rh_kg
        result[f"velocity_{name}"] = velocity
        result[f"reynolds_{name}"] = reynolds
        result[f"friction_{name}"] = fric
        result[f"sdh_{name}"] = sdh_val
        result[f"sdh_kgcm2_{name}"] = sdh_kg
        result[f"maop_{name}"] = maop_dict[i]
        result[f"maop_kgcm2_{name}"] = maop_kg
        if i in pump_indices:
            result[f"coef_A_{name}"] = float(model.A[i])
            result[f"coef_B_{name}"] = float(model.B[i])
            result[f"coef_C_{name}"] = float(model.C[i])
            result[f"dol_{name}"]    = float(model.DOL[i])
            result[f"min_rpm_{name}"]= float(model.MinRPM[i])
            result[f"tdh_{name}"]    = tdh_val

    term_name = terminal.get('name','terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": segment_flows[-1],
        f"pipeline_flow_in_{term_name}": segment_flows[-2],
        f"pump_flow_{term_name}": 0.0,
        f"speed_{term_name}": 0.0,
        f"num_pumps_{term_name}": 0,
        f"efficiency_{term_name}": 0.0,
        f"power_cost_{term_name}": 0.0,
        f"dra_cost_{term_name}": 0.0,
        f"dra_ppm_{term_name}": 0.0,
        f"drag_reduction_{term_name}": 0.0,
        f"head_loss_{term_name}": 0.0,
        f"velocity_{term_name}": 0.0,
        f"reynolds_{term_name}": 0.0,
        f"friction_{term_name}": 0.0,
        f"sdh_{term_name}": 0.0,
        f"residual_head_{term_name}": float(pyo.value(model.RH[N+1])) if model.RH[N+1].value is not None else 0.0,
    })
    term_rh = result[f"residual_head_{term_name}"]
    rho_term = rho_dict[N]
    result[f"rh_kgcm2_{term_name}"] = head_to_kgcm2(term_rh, rho_term)
    result['total_cost'] = float(pyo.value(model.Obj)) if model.Obj is not None else 0.0
    result["error"] = False
    return result
