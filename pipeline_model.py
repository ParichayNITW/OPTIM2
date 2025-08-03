import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

"""
pipeline_model.py
------------------
This module formulates and solves a pipeline optimisation problem using Pyomo.  It
supports drag-reducing additives (DRA) and multiple pump stations along a
pipeline.  At the originating pump station there may be two different pump
types (A and B) with independent operating curves.  The model minimises
operating cost (power plus DRA) while satisfying hydraulic constraints,
including minimum residual head, pressure limits and peak constraints.

The code is based on a simplified single-pump-type formulation and has been
extended to handle two pump types at the origin.  For non-origin pump
stations there is a single pump type.
"""

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'parichay.nitwarangal@gmail.com')

# -----------------------------------------------------------------------------
# DRA drag-reduction curves.  Each CSV file should contain two columns:
# ``%Drag Reduction`` and ``PPM``.  The keys correspond to the kinematic
# viscosity (cSt) and are used for piecewise interpolation of DRA ppm as a
# function of drag reduction percentage.
# -----------------------------------------------------------------------------
DRA_CSV_FILES = {
    10: "10 cst.csv",
    15: "15 cst.csv",
    20: "20 cst.csv",
    25: "25 cst.csv",
    30: "30 cst.csv",
    35: "35 cst.csv",
    40: "40 cst.csv",
}

# Load DRA curve data if files are available
DRA_CURVE_DATA = {}
for cst, fname in DRA_CSV_FILES.items():
    if os.path.exists(fname):
        DRA_CURVE_DATA[cst] = pd.read_csv(fname)
    else:
        DRA_CURVE_DATA[cst] = None

def get_ppm_breakpoints(visc: float):
    """Return breakpoints for drag reduction percentage (x) and DRA ppm (y).

    For a given kinematic viscosity ``visc`` (in cSt), interpolate between
    neighbouring DRA curves if necessary and return the unique drag reduction
    percentages and corresponding ppm values.  These breakpoints are used to
    construct a Pyomo piecewise function relating drag reduction to ppm.

    Parameters
    ----------
    visc : float
        Kinematic viscosity in cSt.

    Returns
    -------
    Tuple[List[float], List[float]]
        Two lists representing drag reduction percentages and corresponding
        ppm values.
    """
    cst_list = sorted([c for c in DRA_CURVE_DATA.keys() if DRA_CURVE_DATA[c] is not None])
    visc = float(visc)
    if not cst_list:
        return [0], [0]
    # Clamp to available data
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
        # Linear interpolation between lower and upper curves
        ppm_lower_interp = np.interp(dr_points, x_lower, y_lower)
        ppm_upper_interp = np.interp(dr_points, x_upper, y_upper)
        ppm_points = ppm_lower_interp * (upper - visc) / (upper - lower) + \
                     ppm_upper_interp * (visc - lower) / (upper - lower)
        unique_dr, unique_indices = np.unique(dr_points, return_index=True)
        unique_ppm = ppm_points[unique_indices]
        return list(unique_dr), list(unique_ppm)
    # If visc matches available data exactly
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    return list(unique_x), list(unique_y)

def solve_pipeline(
    stations: list,
    terminal: dict,
    FLOW: float,
    KV_list: list,
    rho_list: list,
    RateDRA: float,
    Price_HSD: float,
    linefill_dict: dict = None,
) -> dict:
    """Optimise pipeline operation with optional drag-reducing additive.

    Parameters
    ----------
    stations : list of dict
        Station data.  Each dictionary may contain:

        * ``is_pump``: bool, True if station has a pump.
        * ``delivery`` and ``supply``: mass balance adjustments.
        * ``L``, ``d``/``D``: pipeline length (km) and diameter (m).
        * ``rough``, ``SMYS``, ``DF``, ``elev``: pipeline parameters.
        * ``peaks``: list of dicts with ``loc`` (km) and ``elev`` (m) for peak constraints.
        * Pump coefficients ``A``, ``B``, ``C``, ``P``, ``Q``, ``R``, ``S``, ``T`` for
          non-origin pump stations.
        * ``MinRPM``, ``DOL``, ``sfc`` or ``rate`` and ``max_pumps`` for non-origin
          pump stations.
        * For the originating station (first in the list with ``is_pump`` True) the
          following additional keys specify the two pump types:
            - ``max_pumps_typeA``, ``max_pumps_typeB``: maximum number of pumps of each type.
            - ``A1``, ``B1``, ``C1``, ``P1``, ``Q1``, ``R1``, ``S1``, ``T1``, ``MinRPM1``, ``DOL1``, ``sfc1``, ``rate1``
              for pump type A.
            - ``A2``, ``B2``, ``C2``, ``P2``, ``Q2``, ``R2``, ``S2``, ``T2``, ``MinRPM2``, ``DOL2``, ``sfc2``, ``rate2``
              for pump type B.

    terminal : dict
        Terminal data, may contain ``elev`` and ``min_residual``.

    FLOW : float
        Initial pipeline flow (m3/hr).

    KV_list : list
        Kinematic viscosity (cSt) for each segment.

    rho_list : list
        Density (kg/m3) for each segment.

    RateDRA : float
        Cost per litre of DRA.

    Price_HSD : float
        Price of diesel (INR/litre) for diesel pumps.

    linefill_dict : dict, optional
        Not used in the current formulation.

    Returns
    -------
    dict
        A dictionary of results including flows, number of pumps, RPM, power cost,
        DRA cost, head loss, residual head, velocity, Reynolds number, friction
        factor, SDH, MAOP, and total cost.
    """
    RPM_STEP = 100  # RPM step size for discretisation
    DRA_STEP = 5    # DRA step size (ppm)

    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)

    # Create parameter dictionaries for viscosity and density
    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Compute flow in each pipeline segment (outflow after deliveries/supplies)
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        out_flow = prev_flow - delivery + supply
        segment_flows.append(out_flow)

    # -------------------------------------------------------------------------
    # Pipeline geometry and pump parameters
    # -------------------------------------------------------------------------
    length = {}; d_inner = {}; roughness = {}; thickness = {}
    smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    max_dr = {}; peaks_dict = {}
    default_t = 0.007; default_e = 0.00004; default_smys = 52000; default_df = 0.72
    allowed_rpms = {}; allowed_dras = {}

    # Identify the originating pump station (first with is_pump=True)
    originating_pump_index = None
    for idx, stn in enumerate(stations, start=1):
        if stn.get('is_pump', False) and originating_pump_index is None:
            originating_pump_index = idx
        # Populate pipeline geometry
        length[idx] = stn.get('L', 0.0)
        if 'D' in stn:
            D_out = stn['D']
            thickness[idx] = stn.get('t', default_t)
            d_inner[idx] = D_out - 2*thickness[idx]
        elif 'd' in stn:
            d_inner[idx] = stn['d']
            thickness[idx] = stn.get('t', default_t)
        else:
            d_inner[idx] = 0.7
            thickness[idx] = default_t
        roughness[idx] = stn.get('rough', default_e)
        smys[idx] = stn.get('SMYS', default_smys)
        design_factor[idx] = stn.get('DF', default_df)
        elev[idx] = stn.get('elev', 0.0)
        peaks_dict[idx] = stn.get('peaks', [])

        # Pump-specific data
        if stn.get('is_pump', False):
            pump_indices.append(idx)
            if idx != originating_pump_index:
                # Non-origin pump stations have one pump type
                Acoef[idx] = stn.get('A', 0.0)
                Bcoef[idx] = stn.get('B', 0.0)
                Ccoef[idx] = stn.get('C', 0.0)
                Pcoef[idx] = stn.get('P', 0.0)
                Qcoef[idx] = stn.get('Q', 0.0)
                Rcoef[idx] = stn.get('R', 0.0)
                Scoef[idx] = stn.get('S', 0.0)
                Tcoef[idx] = stn.get('T', 0.0)
                min_rpm[idx] = stn.get('MinRPM', 0)
                max_rpm[idx] = stn.get('DOL', 0)
                if stn.get('sfc', 0) not in (None, 0):
                    diesel_pumps.append(idx)
                    sfc[idx] = stn.get('sfc', 0.0)
                else:
                    electric_pumps.append(idx)
                    elec_cost[idx] = stn.get('rate', 0.0)
                max_dr[idx] = stn.get('max_dr', 0.0)
            # Determine allowed RPMs and DRA for non-origin pumps (origin handled separately)
            minval = int(stn.get('MinRPM', 0))
            maxval = int(stn.get('DOL', 0))
            allowed_rpms[idx] = [r for r in range(minval, maxval + 1, RPM_STEP)]
            if allowed_rpms[idx] and allowed_rpms[idx][-1] != maxval:
                allowed_rpms[idx].append(maxval)
            maxval_dra = int(stn.get('max_dr', 0))
            allowed_dras[idx] = [d for d in range(0, maxval_dra + 1, DRA_STEP)]
            if allowed_dras[idx] and allowed_dras[idx][-1] != maxval_dra:
                allowed_dras[idx].append(maxval_dra)

    # Add terminal elevation for node N+1
    elev[N+1] = terminal.get('elev', 0.0)

    # Define Pyomo parameters for pipeline geometry
    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)

    # The set of all pump stations (including origin)
    model.pump_stations = pyo.Set(initialize=pump_indices)

    # Parameters for non-origin pump stations
    if pump_indices:
        # Only populate these parameters for indices > originating station
        non_origin_pumps = [i for i in pump_indices if i != originating_pump_index]
        if non_origin_pumps:
            model.A = pyo.Param(non_origin_pumps, initialize={i: Acoef[i] for i in non_origin_pumps})
            model.B = pyo.Param(non_origin_pumps, initialize={i: Bcoef[i] for i in non_origin_pumps})
            model.C = pyo.Param(non_origin_pumps, initialize={i: Ccoef[i] for i in non_origin_pumps})
            model.Pcoef = pyo.Param(non_origin_pumps, initialize={i: Pcoef[i] for i in non_origin_pumps})
            model.Qcoef = pyo.Param(non_origin_pumps, initialize={i: Qcoef[i] for i in non_origin_pumps})
            model.Rcoef = pyo.Param(non_origin_pumps, initialize={i: Rcoef[i] for i in non_origin_pumps})
            model.Scoef = pyo.Param(non_origin_pumps, initialize={i: Scoef[i] for i in non_origin_pumps})
            model.Tcoef = pyo.Param(non_origin_pumps, initialize={i: Tcoef[i] for i in non_origin_pumps})
            model.MinRPM = pyo.Param(non_origin_pumps, initialize={i: min_rpm[i] for i in non_origin_pumps})
            model.DOL = pyo.Param(non_origin_pumps, initialize={i: max_rpm[i] for i in non_origin_pumps})

    # -------------------------------------------------------------------------
    # Variables for pump counts at non-origin stations
    # -------------------------------------------------------------------------
    def nop_bounds(m, j):
        # ``j`` is a station index in non-origin pumps; lower bound is 0, upper bound from station data
        ub = stations[j - 1].get('max_pumps', 2)
        return (0, ub)
    non_origin_pumps = [i for i in pump_indices if i != originating_pump_index]
    model.NOP = pyo.Var(non_origin_pumps, domain=pyo.NonNegativeIntegers, bounds=nop_bounds, initialize=1)

    # Variables for pump counts at the origin: two types A and B
    if originating_pump_index is not None:
        stn0 = stations[originating_pump_index - 1]
        max_pumps_typeA = int(stn0.get('max_pumps_typeA', 2))
        max_pumps_typeB = int(stn0.get('max_pumps_typeB', 2))
        model.NOP_A_origin = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, max_pumps_typeA), initialize=1)
        model.NOP_B_origin = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, max_pumps_typeB), initialize=0)
        # At least one pump must run at the origin (either type A or B)
        model.min_pump_origin = pyo.Constraint(expr= model.NOP_A_origin + model.NOP_B_origin >= 1)

    # -------------------------------------------------------------------------
    # RPM selection
    # -------------------------------------------------------------------------
    # RPM binaries and continuous variables for non-origin pumps
    model.rpm_bin = pyo.Var(
        ((i, j) for i in non_origin_pumps for j in range(len(allowed_rpms[i]))),
        domain=pyo.Binary
    )
    def rpm_bin_sum_rule(m, i):
        return sum(m.rpm_bin[i, j] for j in range(len(allowed_rpms[i]))) == 1
    model.rpm_bin_sum = pyo.Constraint(non_origin_pumps, rule=rpm_bin_sum_rule)
    model.RPM_var = pyo.Var(non_origin_pumps, domain=pyo.NonNegativeReals)
    def rpm_value_rule(m, i):
        return m.RPM_var[i] == sum(allowed_rpms[i][j] * m.rpm_bin[i, j] for j in range(len(allowed_rpms[i])))
    model.rpm_value = pyo.Constraint(non_origin_pumps, rule=rpm_value_rule)

    # RPM variables for origin pump types
    if originating_pump_index is not None:
        # Allowed RPM lists for origin pump types
        stn0 = stations[originating_pump_index - 1]
        MinRPM1 = int(stn0.get('MinRPM1', 0))
        DOL1    = int(stn0.get('DOL1', 0))
        MinRPM2 = int(stn0.get('MinRPM2', 0))
        DOL2    = int(stn0.get('DOL2', 0))
        allowed_rpms_A = [r for r in range(MinRPM1, DOL1 + 1, RPM_STEP)]
        if allowed_rpms_A and allowed_rpms_A[-1] != DOL1:
            allowed_rpms_A.append(DOL1)
        allowed_rpms_B = [r for r in range(MinRPM2, DOL2 + 1, RPM_STEP)]
        if allowed_rpms_B and allowed_rpms_B[-1] != DOL2:
            allowed_rpms_B.append(DOL2)
        # Binary selection
        model.rpm_bin_A = pyo.Var(range(len(allowed_rpms_A)), domain=pyo.Binary)
        model.rpm_bin_B = pyo.Var(range(len(allowed_rpms_B)), domain=pyo.Binary)
        model.RPM_A_origin = pyo.Var(bounds=(MinRPM1, DOL1), domain=pyo.NonNegativeReals)
        model.RPM_B_origin = pyo.Var(bounds=(MinRPM2, DOL2), domain=pyo.NonNegativeReals)
        model.rpm_bin_sum_A = pyo.Constraint(expr=sum(model.rpm_bin_A[j] for j in range(len(allowed_rpms_A))) == 1)
        model.rpm_bin_sum_B = pyo.Constraint(expr=sum(model.rpm_bin_B[j] for j in range(len(allowed_rpms_B))) == 1)
        model.rpm_value_A = pyo.Constraint(expr= model.RPM_A_origin == sum(allowed_rpms_A[j] * model.rpm_bin_A[j] for j in range(len(allowed_rpms_A))))
        model.rpm_value_B = pyo.Constraint(expr= model.RPM_B_origin == sum(allowed_rpms_B[j] * model.rpm_bin_B[j] for j in range(len(allowed_rpms_B))))

    # -------------------------------------------------------------------------
    # DRA variables via binaries (applies to all pump stations including origin)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Residual head and suction/discharge head
    # -------------------------------------------------------------------------
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    # Fix residual head at first node if specified
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    # Set lower bound on residual head for other nodes
    for j in range(2, N + 2):
        model.RH[j].setlb(50.0)

    # Compute velocity, Reynolds number and friction factor for each segment
    g = 9.81
    v = {}; Re = {}; f = {}
    for i in range(1, N + 1):
        flow_m3s = float(segment_flows[i]) / 3600.0
        area = pi * (d_inner[i] ** 2) / 4.0
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
                arg = (roughness[i] / d_inner[i] / 3.7) + (5.74 / (Re[i] ** 0.9))
                f[i] = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    # SDH and associated constraints
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    TDH = {}
    EFFP = {}

    # Loop through stations to set SDH constraints and compute pump head/efficiency
    for i in range(1, N + 1):
        # Drag reduction fraction for segment i
        DR_frac = model.DR_var[i] / 100.0 if i in pump_indices else 0.0
        # Friction head loss for the next segment (with DRA)
        DH_next = f[i] * ((length[i] * 1000.0) / d_inner[i]) * (v[i] ** 2 / (2 * g)) * (1 - DR_frac)
        expr_next = model.RH[i + 1] + (model.z[i + 1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        # Peak constraints at intermediate points within the segment
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DH_peak = f[i] * (L_peak / d_inner[i]) * (v[i] ** 2 / (2 * g)) * (1 - DR_frac)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)

        # Compute pump head and efficiency
        if i == originating_pump_index:
            # Origin pump has two types: A and B
            stn0 = stations[i - 1]
            pump_flow_i = float(segment_flows[i])
            # Type A
            rpm_A = model.RPM_A_origin
            dol_A = float(stn0.get('DOL1', 1))
            # Equivalent flow (dimensionless).  Avoid conditional on rpm_A by relying on
            # variable bounds (MinRPM1 > 0).  The division yields a Pyomo expression.
            # Equivalent flow per pump: divide station flow by the total number of running pumps
            Q_equiv_A = pump_flow_i * dol_A / (rpm_A * (model.NOP_A_origin + model.NOP_B_origin))
            A1 = stn0.get('A1', 0.0); B1 = stn0.get('B1', 0.0); C1 = stn0.get('C1', 0.0)
            P1 = stn0.get('P1', 0.0); Q1 = stn0.get('Q1', 0.0); R1 = stn0.get('R1', 0.0);
            S1 = stn0.get('S1', 0.0); T1 = stn0.get('T1', 0.0)
            # Dynamic head and efficiency expressions for type A
            H_DOL_A = A1 * Q_equiv_A ** 2 + B1 * Q_equiv_A + C1
            TDH_A = H_DOL_A * (rpm_A / dol_A) ** 2
            EFF_A = (P1 * Q_equiv_A ** 4 + Q1 * Q_equiv_A ** 3 + R1 * Q_equiv_A ** 2 + S1 * Q_equiv_A + T1) / 100.0
            model.TDH_A_origin = pyo.Expression(expr=TDH_A)
            model.EFF_A_origin = pyo.Expression(expr=EFF_A)
            # Type B
            rpm_B = model.RPM_B_origin
            dol_B = float(stn0.get('DOL2', 1))
            Q_equiv_B = pump_flow_i * dol_B / (rpm_B * (model.NOP_A_origin + model.NOP_B_origin))
            A2 = stn0.get('A2', 0.0); B2 = stn0.get('B2', 0.0); C2 = stn0.get('C2', 0.0)
            P2 = stn0.get('P2', 0.0); Q2 = stn0.get('Q2', 0.0); R2 = stn0.get('R2', 0.0);
            S2 = stn0.get('S2', 0.0); T2 = stn0.get('T2', 0.0)
            H_DOL_B = A2 * Q_equiv_B ** 2 + B2 * Q_equiv_B + C2
            TDH_B = H_DOL_B * (rpm_B / dol_B) ** 2
            EFF_B = (P2 * Q_equiv_B ** 4 + Q2 * Q_equiv_B ** 3 + R2 * Q_equiv_B ** 2 + S2 * Q_equiv_B + T2) / 100.0
            model.TDH_B_origin = pyo.Expression(expr=TDH_B)
            model.EFF_B_origin = pyo.Expression(expr=EFF_B)
            # For origin, we don't populate TDH/EFFP dictionary
        elif i in pump_indices:
            pump_flow_i = float(segment_flows[i])
            rpm_val = model.RPM_var[i]
            dol_val = model.DOL[i]
            Q_equiv = pump_flow_i * dol_val / rpm_val
            H_DOL = model.A[i] * Q_equiv ** 2 + model.B[i] * Q_equiv + model.C[i]
            TDH[i] = H_DOL * (rpm_val / dol_val) ** 2
            EFFP[i] = (model.Pcoef[i] * Q_equiv ** 4 + model.Qcoef[i] * Q_equiv ** 3 +
                       model.Rcoef[i] * Q_equiv ** 2 + model.Scoef[i] * Q_equiv +
                       model.Tcoef[i]) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    # -------------------------------------------------------------------------
    # Head balance and pressure/peak limits
    # -------------------------------------------------------------------------
    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    maop_dict = {}
    for i in range(1, N + 1):
        if i == originating_pump_index:
            # Sum of pump heads from type A and B at the origin
            model.head_balance.add(
                model.RH[i] + model.TDH_A_origin * model.NOP_A_origin + model.TDH_B_origin * model.NOP_B_origin >= model.SDH[i]
            )
        elif i in pump_indices:
            model.head_balance.add(
                model.RH[i] + TDH[i] * model.NOP[i] >= model.SDH[i]
            )
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        # MAOP (Maximum Allowable Operating Pressure) as head
        D_out = d_inner[i] + 2 * thickness[i]
        MAOP_head = (2 * thickness[i] * (smys[i] * 0.070307) * design_factor[i] / D_out) * 10000.0 / rho_dict[i]
        maop_dict[i] = MAOP_head
        model.pressure_limit.add(model.SDH[i] <= MAOP_head)
        # Peak elevation constraints
        for peak in peaks_dict[i]:
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            loss_no_dra = f[i] * (L_peak / d_inner[i]) * (v[i] ** 2 / (2 * g))
            if i == originating_pump_index:
                expr = model.RH[i] + model.TDH_A_origin * model.NOP_A_origin + model.TDH_B_origin * model.NOP_B_origin - (elev_k - model.z[i]) - loss_no_dra
            elif i in pump_indices:
                expr = model.RH[i] + TDH[i] * model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra
            model.peak_limit.add(expr >= 50.0)

    # -------------------------------------------------------------------------
    # DRA ppm and cost piecewise definitions
    # -------------------------------------------------------------------------
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
        # DRA cost per day: ppm * (flow [m3/hr] * 1000 [kg/m3] * 24 h / 1e6) * RateDRA
        dra_cost_expr = model.PPM[i] * (segment_flows[i] * 1000.0 * 24.0 / 1e6) * RateDRA
        model.dra_cost[i] = dra_cost_expr

    # -------------------------------------------------------------------------
    # Objective: minimise total power cost + DRA cost
    # -------------------------------------------------------------------------
    total_cost = 0
    # Power cost at the originating pump station (two pump types)
    if originating_pump_index is not None:
        i = originating_pump_index
        rho_i = rho_dict[i]
        pump_flow_i = float(segment_flows[i])
        stn0 = stations[i - 1]
        # Type A power cost
        power_kW_A = (rho_i * pump_flow_i * 9.81 * model.TDH_A_origin * model.NOP_A_origin) / (3600.0 * 1000.0 * model.EFF_A_origin * 0.95)
        sfc1 = stn0.get('sfc1', 0.0); rate1 = stn0.get('rate1', 0.0)
        if sfc1 not in (None, 0):
            fuel_per_kWh_A = (float(sfc1) * 1.34102) / 820.0
            cost_A = power_kW_A * 24.0 * fuel_per_kWh_A * Price_HSD
        else:
            cost_A = power_kW_A * 24.0 * rate1
        # Type B power cost
        power_kW_B = (rho_i * pump_flow_i * 9.81 * model.TDH_B_origin * model.NOP_B_origin) / (3600.0 * 1000.0 * model.EFF_B_origin * 0.95)
        sfc2 = stn0.get('sfc2', 0.0); rate2 = stn0.get('rate2', 0.0)
        if sfc2 not in (None, 0):
            fuel_per_kWh_B = (float(sfc2) * 1.34102) / 820.0
            cost_B = power_kW_B * 24.0 * fuel_per_kWh_B * Price_HSD
        else:
            cost_B = power_kW_B * 24.0 * rate2
        total_cost += cost_A + cost_B + model.dra_cost[i]

    # Power cost at non-origin pump stations
    for i in non_origin_pumps:
        rho_i = rho_dict[i]
        pump_flow_i = float(segment_flows[i])
        eff_val = EFFP[i]
        # Avoid divide-by-zero if efficiency is zero
        power_kW = (rho_i * pump_flow_i * 9.81 * TDH[i] * model.NOP[i]) / (3600.0 * 1000.0 * eff_val * 0.95) if eff_val is not None else 0.0
        if i in electric_pumps:
            rate = elec_cost.get(i, 0.0)
            power_cost = power_kW * 24.0 * rate
        else:
            sfc_val = sfc.get(i, 0.0)
            fuel_per_kWh = (float(sfc_val) * 1.34102) / 820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        total_cost += power_cost + model.dra_cost[i]

    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # -------------------------------------------------------------------------
    # Solve the optimisation problem using NEOS with couenne solver
    # -------------------------------------------------------------------------
    results = SolverManagerFactory('neos').solve(model, solver='couenne', tee=False)
    status = results.solver.status
    term = results.solver.termination_condition
    if (status != pyo.SolverStatus.ok) or (term not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        return {
            "error": True,
            "message": f"Optimization failed: {term}. Please check your input values and relax constraints if necessary.",
            "termination_condition": str(term),
            "solver_status": str(status)
        }
    model.solutions.load_from(results)

    # -------------------------------------------------------------------------
    # Collect results for each station
    # -------------------------------------------------------------------------
    result = {}
    for i, stn in enumerate(stations, start=1):
        # Use a lowercase key for internal consistency but also preserve the original
        # capitalization/spacing for front-end compatibility.  Many front-end
        # components expect keys like "num_pumps_Haldia" (capitalised) rather than
        # "num_pumps_haldia".  We therefore construct two name variants: a
        # lowercase version with underscores (``lc_name``) and a preserved
        # version with spaces replaced by underscores (``orig_name``).  All
        # results will be stored under both keys to ensure compatibility.
        lc_name = stn['name'].strip().lower().replace(' ', '_')
        orig_name = stn['name'].strip().replace(' ', '_')
        inflow = segment_flows[i - 1]
        outflow = segment_flows[i]
        pump_flow = outflow if stn.get('is_pump', False) else 0.0

        if i == originating_pump_index:
            # Origin: report type A and B separately
            numA = int(pyo.value(model.NOP_A_origin)) if model.NOP_A_origin.value is not None else 0
            numB = int(pyo.value(model.NOP_B_origin)) if model.NOP_B_origin.value is not None else 0
            # Determine RPM values from binaries
            rpmA = 0
            for j in range(len(allowed_rpms_A)):
                if round(pyo.value(model.rpm_bin_A[j])) == 1:
                    rpmA = allowed_rpms_A[j]
                    break
            rpmB = 0
            for j in range(len(allowed_rpms_B)):
                if round(pyo.value(model.rpm_bin_B[j])) == 1:
                    rpmB = allowed_rpms_B[j]
                    break
            # Efficiency and TDH values for reporting
            # Compute on the fly using selected RPMs
            stn0 = stn
            pump_flow_i = float(segment_flows[i])
            # Flow per pump if multiple pumps are running
            total_pumps = numA + numB if (numA + numB) > 0 else 1
            flow_per_pump = pump_flow_i / total_pumps
            # Type A
            if numA > 0:
                dol_A = float(stn0.get('DOL1', 1))
                Q_equiv_A = (flow_per_pump * dol_A / rpmA) if rpmA != 0 else 1.0
                H_DOL_A = stn0.get('A1', 0.0) * Q_equiv_A ** 2 + stn0.get('B1', 0.0) * Q_equiv_A + stn0.get('C1', 0.0)
                tdhA = H_DOL_A * (rpmA / dol_A) ** 2
                effA = (stn0.get('P1', 0.0) * Q_equiv_A ** 4 + stn0.get('Q1', 0.0) * Q_equiv_A ** 3 +
                        stn0.get('R1', 0.0) * Q_equiv_A ** 2 + stn0.get('S1', 0.0) * Q_equiv_A + stn0.get('T1', 0.0)) * 1.0
            else:
                tdhA = 0.0; effA = 0.0; rpmA = 0
            # Type B
            if numB > 0:
                dol_B = float(stn0.get('DOL2', 1))
                Q_equiv_B = (flow_per_pump * dol_B / rpmB) if rpmB != 0 else 1.0
                H_DOL_B = stn0.get('A2', 0.0) * Q_equiv_B ** 2 + stn0.get('B2', 0.0) * Q_equiv_B + stn0.get('C2', 0.0)
                tdhB = H_DOL_B * (rpmB / dol_B) ** 2
                effB = (stn0.get('P2', 0.0) * Q_equiv_B ** 4 + stn0.get('Q2', 0.0) * Q_equiv_B ** 3 +
                        stn0.get('R2', 0.0) * Q_equiv_B ** 2 + stn0.get('S2', 0.0) * Q_equiv_B + stn0.get('T2', 0.0)) * 1.0
            else:
                tdhB = 0.0; effB = 0.0; rpmB = 0
            # DRA and cost
            dra_perc = float(pyo.value(model.DR_var[i])) if model.DR_var[i].value is not None else 0.0
            dra_ppm = float(pyo.value(model.PPM[i])) if model.PPM[i].value is not None else 0.0
            dra_cost_i = float(pyo.value(model.dra_cost[i])) if model.dra_cost[i].expr is not None else 0.0
            # Power cost is accounted in objective; compute for reporting
            power_cost = 0.0
            rho_i = rho_dict[i]
            # Type A power
            if numA > 0 and rpmA != 0:
                dol_A = float(stn0.get('DOL1', 1))
                # Use flow per pump to compute equivalent flow for pump head curve
                flow_per_pump = pump_flow_i / (numA + numB if (numA + numB) > 0 else 1)
                Q_equiv_A = flow_per_pump * dol_A / rpmA
                H_DOL_A = stn0.get('A1', 0.0) * Q_equiv_A ** 2 + stn0.get('B1', 0.0) * Q_equiv_A + stn0.get('C1', 0.0)
                tdhA_calc = H_DOL_A * (rpmA / dol_A) ** 2
                effA_calc = (stn0.get('P1', 0.0) * Q_equiv_A ** 4 + stn0.get('Q1', 0.0) * Q_equiv_A ** 3 +
                             stn0.get('R1', 0.0) * Q_equiv_A ** 2 + stn0.get('S1', 0.0) * Q_equiv_A + stn0.get('T1', 0.0)) / 100.0
                power_kW_A = (rho_i * pump_flow_i * 9.81 * tdhA_calc * numA) / (3600.0 * 1000.0 * effA_calc * 0.95) if effA_calc != 0 else 0.0
                sfc1 = stn0.get('sfc1', 0.0); rate1 = stn0.get('rate1', 0.0)
                if sfc1 not in (None, 0):
                    fuel_per_kWh_A = (float(sfc1) * 1.34102) / 820.0
                    power_cost += power_kW_A * 24.0 * fuel_per_kWh_A * Price_HSD
                else:
                    power_cost += power_kW_A * 24.0 * rate1
            # Type B power
            if numB > 0 and rpmB != 0:
                dol_B = float(stn0.get('DOL2', 1))
                # Use flow per pump to compute equivalent flow
                flow_per_pump = pump_flow_i / (numA + numB if (numA + numB) > 0 else 1)
                Q_equiv_B = flow_per_pump * dol_B / rpmB
                H_DOL_B = stn0.get('A2', 0.0) * Q_equiv_B ** 2 + stn0.get('B2', 0.0) * Q_equiv_B + stn0.get('C2', 0.0)
                tdhB_calc = H_DOL_B * (rpmB / dol_B) ** 2
                effB_calc = (stn0.get('P2', 0.0) * Q_equiv_B ** 4 + stn0.get('Q2', 0.0) * Q_equiv_B ** 3 +
                             stn0.get('R2', 0.0) * Q_equiv_B ** 2 + stn0.get('S2', 0.0) * Q_equiv_B + stn0.get('T2', 0.0)) / 100.0
                power_kW_B = (rho_i * pump_flow_i * 9.81 * tdhB_calc * numB) / (3600.0 * 1000.0 * effB_calc * 0.95) if effB_calc != 0 else 0.0
                sfc2 = stn0.get('sfc2', 0.0); rate2 = stn0.get('rate2', 0.0)
                if sfc2 not in (None, 0):
                    fuel_per_kWh_B = (float(sfc2) * 1.34102) / 820.0
                    power_cost += power_kW_B * 24.0 * fuel_per_kWh_B * Price_HSD
                else:
                    power_cost += power_kW_B * 24.0 * rate2
            # Head loss, residual head
            head_loss = float(pyo.value(model.SDH[i] - (model.RH[i + 1] + (model.z[i + 1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i + 1].value is not None else 0.0
            res_head = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
            # Aggregate pump counts and weighted averages for speed, efficiency and TDH
            total_pumps = numA + numB
            if total_pumps > 0:
                avg_speed = ((numA * rpmA) + (numB * rpmB)) / total_pumps
                avg_eff   = ((numA * effA) + (numB * effB)) / total_pumps
                # Compute weighted average head based on pump curve values
                avg_tdh = ((numA * tdhA) + (numB * tdhB)) / total_pumps
            else:
                avg_speed = 0.0
                avg_eff = 0.0
                avg_tdh = 0.0
            # Pack results for both lowercase and original names
            for nm in (lc_name, orig_name):
                result[f"pipeline_flow_{nm}"] = outflow
                result[f"pipeline_flow_in_{nm}"] = inflow
                result[f"pump_flow_{nm}"] = pump_flow
                # Report total pumps for compatibility with front-end
                result[f"num_pumps_{nm}"] = total_pumps
                result[f"speed_{nm}"] = avg_speed
                result[f"efficiency_{nm}"] = avg_eff
                result[f"tdh_{nm}"] = avg_tdh
                # Also report type-specific details
                result[f"num_pumps_typeA_{nm}"] = numA
                result[f"num_pumps_typeB_{nm}"] = numB
                result[f"speed_typeA_{nm}"] = rpmA
                result[f"speed_typeB_{nm}"] = rpmB
                result[f"efficiency_typeA_{nm}"] = effA
                result[f"efficiency_typeB_{nm}"] = effB
                result[f"tdh_typeA_{nm}"] = tdhA
                result[f"tdh_typeB_{nm}"] = tdhB
                result[f"power_cost_{nm}"] = power_cost
                result[f"dra_cost_{nm}"] = dra_cost_i
                result[f"dra_ppm_{nm}"] = dra_ppm
                result[f"drag_reduction_{nm}"] = dra_perc
                result[f"head_loss_{nm}"] = head_loss
                result[f"residual_head_{nm}"] = res_head
                result[f"velocity_{nm}"] = v[i]
                result[f"reynolds_{nm}"] = Re[i]
                result[f"friction_{nm}"] = f[i]
                result[f"sdh_{nm}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
                result[f"maop_{nm}"] = maop_dict[i]
                # Provide pump curve coefficients and base RPM/DOL for plotting on the frontend
                # Use Type A pump coefficients for the origin station
                stn0 = stations[i - 1]
                A1 = stn0.get('A1', 0.0)
                B1 = stn0.get('B1', 0.0)
                C1 = stn0.get('C1', 0.0)
                dol1 = stn0.get('DOL1', stn0.get('DOL', 0.0))
                minrpm1 = stn0.get('MinRPM1', stn0.get('MinRPM', 0.0))
                result[f"coef_A_{nm}"] = float(A1)
                result[f"coef_B_{nm}"] = float(B1)
                result[f"coef_C_{nm}"] = float(C1)
                result[f"dol_{nm}"]    = float(dol1)
                result[f"min_rpm_{nm}"] = float(minrpm1)
        elif i in pump_indices:
            # Non-origin pump station
            num_pumps = int(pyo.value(model.NOP[i])) if model.NOP[i].value is not None else 0
            # Determine RPM value
            rpm_val = None
            for j in range(len(allowed_rpms[i])):
                if round(pyo.value(model.rpm_bin[i, j])) == 1:
                    rpm_val = allowed_rpms[i][j]
                    break
            if rpm_val is None:
                rpm_val = allowed_rpms[i][0] if allowed_rpms[i] else 0.0
            # Drag reduction
            dra_perc = None
            for j in range(len(allowed_dras[i])):
                if round(pyo.value(model.dra_bin[i, j])) == 1:
                    dra_perc = allowed_dras[i][j]
                    break
            if dra_perc is None:
                dra_perc = allowed_dras[i][0] if allowed_dras[i] else 0.0
            # Compute TDH and efficiency using selected RPM
            pump_flow_i = float(segment_flows[i])
            dol_val = model.DOL[i]
            Q_equiv = pump_flow_i * dol_val / rpm_val if rpm_val != 0 else 1.0
            tdh_val = float(Acoef[i] * Q_equiv ** 2 + Bcoef[i] * Q_equiv + Ccoef[i]) * (rpm_val / dol_val) ** 2
            eff = (Pcoef[i] * Q_equiv ** 4 + Qcoef[i] * Q_equiv ** 3 + Rcoef[i] * Q_equiv ** 2 + Scoef[i] * Q_equiv + Tcoef[i]) * 1.0
            # If no pumps installed
            if num_pumps == 0:
                rpm_val = 0.0; eff = 0.0; dra_perc = 0.0; tdh_val = 0.0
            # Power cost
            power_cost = 0.0
            rho_i = rho_dict[i]
            if num_pumps > 0 and rpm_val != 0:
                eff_calc = (Pcoef[i] * Q_equiv ** 4 + Qcoef[i] * Q_equiv ** 3 + Rcoef[i] * Q_equiv ** 2 + Scoef[i] * Q_equiv + Tcoef[i]) / 100.0
                power_kW = (rho_i * pump_flow * 9.81 * tdh_val * num_pumps) / (3600.0 * 1000.0 * eff_calc * 0.95) if eff_calc != 0 else 0.0
                if i in electric_pumps:
                    rate = elec_cost.get(i, 0.0)
                    power_cost = power_kW * 24.0 * rate
                else:
                    sfc_val = sfc.get(i, 0.0)
                    fuel_per_kWh = (float(sfc_val) * 1.34102) / 820.0
                    power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
            dra_ppm = float(pyo.value(model.PPM[i])) if model.PPM[i].value is not None else 0.0
            dra_cost_i = float(pyo.value(model.dra_cost[i])) if model.dra_cost[i].expr is not None else 0.0
            head_loss = float(pyo.value(model.SDH[i] - (model.RH[i + 1] + (model.z[i + 1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i + 1].value is not None else 0.0
            res_head = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
            # Pack results for both lowercase and original names
            for nm in (lc_name, orig_name):
                result[f"pipeline_flow_{nm}"] = outflow
                result[f"pipeline_flow_in_{nm}"] = inflow
                result[f"pump_flow_{nm}"] = pump_flow
                result[f"num_pumps_{nm}"] = num_pumps
                result[f"speed_{nm}"] = rpm_val
                result[f"efficiency_{nm}"] = eff
                result[f"power_cost_{nm}"] = power_cost
                result[f"dra_cost_{nm}"] = dra_cost_i
                result[f"dra_ppm_{nm}"] = dra_ppm
                result[f"drag_reduction_{nm}"] = dra_perc
                result[f"head_loss_{nm}"] = head_loss
                result[f"residual_head_{nm}"] = res_head
                result[f"velocity_{nm}"] = v[i]
                result[f"reynolds_{nm}"] = Re[i]
                result[f"friction_{nm}"] = f[i]
                result[f"sdh_{nm}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
                result[f"maop_{nm}"] = maop_dict[i]
                result[f"tdh_{nm}"] = tdh_val
        else:
            # Non-pump station
            # Head loss on a non-pump segment is the friction head (positive)
            head_loss = float(pyo.value(model.SDH[i] - (model.RH[i + 1] + (model.z[i + 1] - model.z[i])))) if model.SDH[i].value is not None and model.RH[i + 1].value is not None else 0.0
            res_head = float(pyo.value(model.RH[i])) if model.RH[i].value is not None else 0.0
            # Pack results for both lowercase and original names
            for nm in (lc_name, orig_name):
                result[f"pipeline_flow_{nm}"] = outflow
                result[f"pipeline_flow_in_{nm}"] = inflow
                result[f"pump_flow_{nm}"] = 0.0
                result[f"num_pumps_{nm}"] = 0
                result[f"speed_{nm}"] = 0.0
                result[f"efficiency_{nm}"] = 0.0
                result[f"power_cost_{nm}"] = 0.0
                result[f"dra_cost_{nm}"] = 0.0
                result[f"dra_ppm_{nm}"] = 0.0
                result[f"drag_reduction_{nm}"] = 0.0
                result[f"head_loss_{nm}"] = head_loss
                result[f"residual_head_{nm}"] = res_head
                result[f"velocity_{nm}"] = v[i]
                result[f"reynolds_{nm}"] = Re[i]
                result[f"friction_{nm}"] = f[i]
                result[f"sdh_{nm}"] = float(pyo.value(model.SDH[i])) if model.SDH[i].value is not None else 0.0
                result[f"maop_{nm}"] = maop_dict[i]

    # Terminal node results
    # Report terminal (node N+1) results for both lowercase and original names
    term_lc_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    term_orig_name = terminal.get('name', 'terminal').strip().replace(' ', '_')
    for nm in (term_lc_name, term_orig_name):
        result[f"pipeline_flow_{nm}"] = segment_flows[-1]
        result[f"pipeline_flow_in_{nm}"] = segment_flows[-2]
        result[f"pump_flow_{nm}"] = 0.0
        result[f"speed_{nm}"] = 0.0
        result[f"num_pumps_{nm}"] = 0
        result[f"efficiency_{nm}"] = 0.0
        result[f"power_cost_{nm}"] = 0.0
        result[f"dra_cost_{nm}"] = 0.0
        result[f"dra_ppm_{nm}"] = 0.0
        result[f"drag_reduction_{nm}"] = 0.0
        result[f"head_loss_{nm}"] = 0.0
        result[f"velocity_{nm}"] = 0.0
        result[f"reynolds_{nm}"] = 0.0
        result[f"friction_{nm}"] = 0.0
        result[f"sdh_{nm}"] = 0.0
        result[f"residual_head_{nm}"] = float(pyo.value(model.RH[N + 1])) if model.RH[N + 1].value is not None else 0.0
    result['total_cost'] = float(pyo.value(model.Obj)) if model.Obj is not None else 0.0
    result['error'] = False
    return result
