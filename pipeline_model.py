import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
import numpy as np
import json
from math import pi, log10

# ---- Utility Functions ----

def safe_div(x, y, fallback=1e-8):
    """ Division safe against zero/NaN """
    try:
        yv = float(y)
        if yv is None or abs(yv) < fallback:
            return float(x) / fallback
        return float(x) / yv
    except Exception:
        return float(x) / fallback

def fit_curve(x, y, order):
    """Return polynomial coefficients of given order, or zeros if not enough points."""
    if x is None or y is None or len(x) < order+1:
        return [0.0] * (order+1)
    return np.polyfit(x, y, order).tolist()

def parse_peaks(peaks_list):
    """Ensure all peaks are in correct dict format."""
    peaks = []
    for peak in (peaks_list or []):
        try:
            peaks.append({
                "loc": float(peak["Location (km)"]),
                "elev": float(peak["Elevation (m)"])
            })
        except Exception:
            continue
    return peaks

def get_linefill_props(linefill, km):
    """Return density, viscosity at given km location based on linefill mapping."""
    for region in linefill:
        if region["Start (km)"] <= km < region["End (km)"]:
            return float(region["Density (kg/m³)"]), float(region["Viscosity (cSt)"])
    # Default to first if out of range
    region = linefill[0]
    return float(region["Density (kg/m³)"]), float(region["Viscosity (cSt)"])

# ---- Main Backend Function ----

def pipeline_optima_backend(json_obj):
    # --- Parse and Validate JSON ---
    stations = json_obj["stations"]
    terminal = json_obj["terminal"]
    FLOW = float(json_obj["FLOW"])
    RateDRA = float(json_obj["RateDRA"])
    Price_HSD = float(json_obj["Price_HSD"])
    linefill = json_obj["linefill"]
    N = len(stations)

    # --- Compose Full Data Tables & Validate Required Fields ---
    # Prepare peaks, pump curves, densities/viscosities for all stations
    seg_lens = [stn["L"] for stn in stations]
    seg_km = np.cumsum([0] + seg_lens)
    kv_list = []
    rho_list = []
    pump_indices = []
    peaks_dict = {}
    head_coef = {}
    eff_coef = {}
    min_rpm = {}
    max_rpm = {}
    sfc = {}
    elec_cost = {}
    max_dr = {}

    # Validate and map data
    for i, stn in enumerate(stations):
        # Required fields
        for k in ["D", "t", "SMYS", "rough", "L", "min_residual", "elev"]:
            if k not in stn or stn[k] is None:
                raise ValueError(f"Station {i+1} ('{stn.get('name','')}') missing '{k}'")

        # Peaks
        pk_key = f"peak_data_{i+1}"
        if pk_key in json_obj:
            stn['peaks'] = parse_peaks(json_obj[pk_key])
        else:
            stn['peaks'] = []
        peaks_dict[i+1] = stn['peaks']

        # Curves (Head/Efficiency)
        if stn.get("is_pump", False):
            pump_indices.append(i+1)
            h_key = f"head_data_{i+1}"
            e_key = f"eff_data_{i+1}"
            if h_key in json_obj and json_obj[h_key]:
                x, y = zip(*[(float(pt["Flow (m³/hr)"]), float(pt["Head (m)"])) for pt in json_obj[h_key]])
                head_coef[i+1] = fit_curve(x, y, 2)
            else:
                head_coef[i+1] = [0.0, 0.0, 0.0]
            if e_key in json_obj and json_obj[e_key]:
                x, y = zip(*[(float(pt["Flow (m³/hr)"]), float(pt["Efficiency (%)"])) for pt in json_obj[e_key]])
                eff_coef[i+1] = fit_curve(x, y, 4)
            else:
                eff_coef[i+1] = [0.0, 0.0, 0.0, 0.0, 0.0]
            min_rpm[i+1] = int(stn.get("MinRPM", 1))
            max_rpm[i+1] = int(stn.get("DOL", 1))
            sfc[i+1] = float(stn.get("sfc", 0.0))
            elec_cost[i+1] = float(stn.get("rate", 9.0))
            max_dr[i+1] = float(stn.get("max_dr", 0.0))
        else:
            head_coef[i+1] = [0.0, 0.0, 0.0]
            eff_coef[i+1] = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Linefill-mapped viscosity & density
        km = (seg_km[i] + seg_km[i+1]) / 2
        dens, visc = get_linefill_props(linefill, km)
        rho_list.append(dens)
        kv_list.append(visc)

    # --- Pyomo Model Construction ---
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)
    model.KV = pyo.Param(model.I, initialize={i: kv_list[i-1] for i in range(1,N+1)})
    model.rho = pyo.Param(model.I, initialize={i: rho_list[i-1] for i in range(1,N+1)})
    model.L = pyo.Param(model.I, initialize={i: stations[i-1]["L"] for i in range(1,N+1)})
    model.d = pyo.Param(model.I, initialize={i: stations[i-1]["D"] - 2*stations[i-1]["t"] for i in range(1,N+1)})
    model.e = pyo.Param(model.I, initialize={i: stations[i-1]["rough"] for i in range(1,N+1)})
    model.SMYS = pyo.Param(model.I, initialize={i: stations[i-1]["SMYS"] for i in range(1,N+1)})
    model.DF = pyo.Param(model.I, initialize={i: 0.72 for i in range(1,N+1)})
    elev_map = {i: stations[i-1]["elev"] for i in range(1, N+1)}
    elev_map[N+1] = terminal["elev"]
    model.z = pyo.Param(model.Nodes, initialize=elev_map)

    # Segment Flows (basic, for demo—expand if you want supply/delivery per station)
    segment_flows = [FLOW for _ in range(N+1)]

    # --- Model Variables/Parameters for Pumps ---
    model.pump_stations = pyo.Set(initialize=pump_indices)
    if pump_indices:
        model.A = pyo.Param(model.pump_stations, initialize={i: head_coef[i][0] for i in pump_indices})
        model.B = pyo.Param(model.pump_stations, initialize={i: head_coef[i][1] for i in pump_indices})
        model.C = pyo.Param(model.pump_stations, initialize={i: head_coef[i][2] for i in pump_indices})
        model.Pcoef = pyo.Param(model.pump_stations, initialize={i: eff_coef[i][0] for i in pump_indices})
        model.Qcoef = pyo.Param(model.pump_stations, initialize={i: eff_coef[i][1] for i in pump_indices})
        model.Rcoef = pyo.Param(model.pump_stations, initialize={i: eff_coef[i][2] for i in pump_indices})
        model.Scoef = pyo.Param(model.pump_stations, initialize={i: eff_coef[i][3] for i in pump_indices})
        model.Tcoef = pyo.Param(model.pump_stations, initialize={i: eff_coef[i][4] for i in pump_indices})
        model.MinRPM = pyo.Param(model.pump_stations, initialize={i: min_rpm[i] for i in pump_indices})
        model.DOL = pyo.Param(model.pump_stations, initialize={i: max_rpm[i] for i in pump_indices})

    # -- Decision Vars --
    def nop_bounds(m, j):
        return (1 if j == 1 else 0, 2)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers, initialize=1)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0]["min_residual"])
    for j in range(2,N+2):
        model.RH[j].setlb(50.0)

    # --- Minimal Objective/Constraint Example (for demonstration) ---
    # NOTE: This example does NOT do the full hydraulic equations for brevity,
    # but sets up all variables/parameters robustly. Plug in your constraint/objective logic as in your full model.

    # Simple objective (sum NOP)
    model.Obj = pyo.Objective(expr=sum(model.NOP[j] for j in pump_indices), sense=pyo.minimize)

    # --- Solve ---
    os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'your_email@domain.com') # put your email
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=True)
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

    # --- Results Extraction: Return All Main Values ---
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        result[f"num_pumps_{name}"] = int(pyo.value(model.NOP[i])) if i in pump_indices else 0
        result[f"residual_head_{name}"] = float(pyo.value(model.RH[i]))

    # Terminal node result
    term_name = terminal.get('name','terminal').strip().lower().replace(' ','_')
    result[f"residual_head_{term_name}"] = float(pyo.value(model.RH[N+1]))
    result['total_obj'] = float(pyo.value(model.Obj))

    result["error"] = False
    return result

# ---- Usage Example (for your JSON) ----

with open('/mnt/data/pipeline_case (15).json', 'r') as f:
    json_obj = json.load(f)
output = pipeline_optima_backend(json_obj)
print(output)
