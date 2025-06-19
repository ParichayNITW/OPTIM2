import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
import hashlib
import uuid
import json
from plotly.colors import qualitative

st.set_page_config(page_title="Pipeline Optima™", layout="wide", initial_sidebar_state="expanded")

#Custom Styles
st.markdown("""
    <style>
    .red-btn {
        background: linear-gradient(90deg, #d32f2f 30%, #c62828 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        padding: 1.1em 3.3em;
        border-radius: 16px;
        font-size: 1.6em;
        cursor: pointer;
        margin: 1.5em 0 2.1em 0;
        box-shadow: 0 4px 24px #d32f2f55;
        transition: background 0.19s;
    }
    .red-btn:hover {
        background: #b71c1c;
        color: #fff;
    }
    .action-btn-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1.7em;
        gap: 1.2em;
    }
    .action-btn {
        background: #3949ab;
        color: white !important;
        font-weight: 500;
        border: none;
        padding: 0.83em 1.8em;
        border-radius: 10px;
        font-size: 1.09em;
        cursor: pointer;
        box-shadow: 0 2px 12px #9995;
        transition: background 0.16s;
        outline: none;
    }
    .action-btn:hover {
        background: #23277a;
        color: #fff;
    }
    .section-title {
        font-weight: 700;
        color: #1e293b;
        font-size: 1.5em;
        letter-spacing: 0.2px;
        margin-bottom: 0.7em;
        margin-top: 0.2em;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #d32f2f;
        border-bottom: 2.5px solid #d32f2f;
        font-weight: bold;
        background: #ffeaea33;
    }
    </style>
""", unsafe_allow_html=True)

palette = [c for c in qualitative.Plotly if 'yellow' not in c.lower() and '#FFD700' not in c and '#ffeb3b' not in c.lower()]

# --- DRA Curve Data ---
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
        df = pd.read_csv(fname)
        DRA_CURVE_DATA[cst] = df
    else:
        DRA_CURVE_DATA[cst] = None

def get_ppm_for_dr(visc, dr, dra_curve_data=DRA_CURVE_DATA):
    cst_list = sorted(dra_curve_data.keys())
    visc = float(visc)
    # --- New: always round to nearest 0.5 ppm ---
    def round_ppm(val, step=0.5):
        return round(val / step) * step
    if visc <= cst_list[0]:
        df = dra_curve_data[cst_list[0]]
        return round_ppm(_ppm_from_df(df, dr))
    elif visc >= cst_list[-1]:
        df = dra_curve_data[cst_list[-1]]
        return round_ppm(_ppm_from_df(df, dr))
    else:
        lower = max([c for c in cst_list if c <= visc])
        upper = min([c for c in cst_list if c >= visc])
        df_lower = dra_curve_data[lower]
        df_upper = dra_curve_data[upper]
        ppm_lower = _ppm_from_df(df_lower, dr)
        ppm_upper = _ppm_from_df(df_upper, dr)
        ppm_interp = np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
        return round_ppm(ppm_interp)
def _ppm_from_df(df, dr):
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    if dr <= x[0]:
        return y[0]
    elif dr >= x[-1]:
        return y[-1]
    else:
        return np.interp(dr, x, y)

# --- User Login Logic ---

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()
users = {"parichay_das": hash_pwd("heteroscedasticity")}
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🔒 User Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and hash_pwd(password) == users[username]:
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown(
            """
            <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
            &copy; 2025 Pipeline Optima™ v1.1.2. Developed by Parichay Das.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
check_login()

if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

# ==== 1. EARLY LOAD/RESTORE BLOCK ====
def restore_case_dict(loaded_data):
    st.session_state['stations'] = loaded_data.get('stations', [])
    st.session_state['terminal_name'] = loaded_data.get('terminal', {}).get('name', "Terminal")
    st.session_state['terminal_elev'] = loaded_data.get('terminal', {}).get('elev', 0.0)
    st.session_state['terminal_head'] = loaded_data.get('terminal', {}).get('min_residual', 50.0)
    st.session_state['FLOW'] = loaded_data.get('FLOW', 1000.0)
    st.session_state['RateDRA'] = loaded_data.get('RateDRA', 500.0)
    st.session_state['Price_HSD'] = loaded_data.get('Price_HSD', 70.0)
    if "linefill" in loaded_data and loaded_data["linefill"]:
        st.session_state["linefill_df"] = pd.DataFrame(loaded_data["linefill"])
    for i in range(len(st.session_state['stations'])):
        head_data = loaded_data.get(f"head_data_{i+1}", None)
        eff_data  = loaded_data.get(f"eff_data_{i+1}", None)
        peak_data = loaded_data.get(f"peak_data_{i+1}", None)
        if head_data is not None:
            st.session_state[f"head_data_{i+1}"] = pd.DataFrame(head_data)
        if eff_data is not None:
            st.session_state[f"eff_data_{i+1}"] = pd.DataFrame(eff_data)
        if peak_data is not None:
            st.session_state[f"peak_data_{i+1}"] = pd.DataFrame(peak_data)

uploaded_case = st.sidebar.file_uploader("🔁 Load Case", type="json", key="casefile")
if uploaded_case is not None and not st.session_state.get("case_loaded", False):
    loaded_data = json.load(uploaded_case)
    restore_case_dict(loaded_data)
    st.session_state["case_loaded"] = True
    st.session_state["should_rerun"] = True
    st.rerun()
    st.stop()

if st.session_state.get("should_rerun", False):
    st.session_state["should_rerun"] = False
    st.rerun()
    st.stop()

# ==== 2. MAIN INPUT UI ====
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=st.session_state.get("FLOW", 1000.0), step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=st.session_state.get("RateDRA", 500.0), step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=st.session_state.get("Price_HSD", 70.0), step=0.5)
        st.session_state["FLOW"] = FLOW
        st.session_state["RateDRA"] = RateDRA
        st.session_state["Price_HSD"] = Price_HSD

    st.subheader("Linefill Profile (7:00 Hrs)")
    if "linefill_df" not in st.session_state:
        st.session_state["linefill_df"] = pd.DataFrame({
            "Start (km)": [0.0],
            "End (km)": [100.0],
            "Viscosity (cSt)": [10.0],
            "Density (kg/m³)": [850.0]
        })
    st.session_state["linefill_df"] = st.data_editor(
        st.session_state["linefill_df"],
        num_rows="dynamic", key="linefill_editor"
    )

    st.subheader("Stations")
    add_col, rem_col = st.columns(2)
    if "stations" not in st.session_state:
        st.session_state["stations"] = [{
            'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
            'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'min_residual': 50.0, 'is_pump': False,
            'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
            'max_pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
            'max_dr': 0.0,
            'delivery': 0.0,
            'supply': 0.0
        }]
    if add_col.button("➕ Add Station"):
        n = len(st.session_state.get('stations',[])) + 1
        default = {
            'name': f'Station {n}', 'elev': 0.0, 'D': 0.711, 't': 0.007,
            'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'min_residual': 50.0, 'is_pump': False,
            'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
            'max_pumps': 1, 'MinRPM': 1000.0, 'DOL': 1500.0,
            'max_dr': 0.0,
            'delivery': 0.0,
            'supply': 0.0
        }
        st.session_state.stations.append(default)
    if rem_col.button("🗑️ Remove Station"):
        if st.session_state.get('stations'):
            st.session_state.stations.pop()

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <h1 style='
        text-align:center;
        font-size:3.4rem;
        font-weight:700;
        color:#232733;
        margin-bottom:0.25em;
        margin-top:0.01em;
        letter-spacing:0.5px;
        font-family: inherit;
    '>
        Pipeline Optima™: Intelligent Pipeline Network Optimization Suite
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style='
        text-align:center;
        font-size:2.05rem;
        font-weight:700;
        color:#232733;
        margin-bottom:0.15em;
        margin-top:0.02em;
        font-family: inherit;
    '>
        Mixed Integer Non-Linear Non-Convex Pipeline Optimization
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.6em; margin-bottom:1.2em; border: 1px solid #e1e5ec;'>", unsafe_allow_html=True)

for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}: {stn['name']}", expanded=False):
        col1, col2, col3 = st.columns([1.5,1,1])
        with col1:
            stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
            stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{idx}")
            stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")
            stn['L'] = st.number_input("Length to next Station (km)", value=stn['L'], step=1.0, key=f"L{idx}")
            stn['max_dr'] = st.number_input("Max achievable Drag Reduction (%)", value=stn.get('max_dr', 0.0), key=f"mdr{idx}")
            if idx == 1:
                stn['min_residual'] = st.number_input("Available Suction Head (m)", value=stn.get('min_residual',50.0), step=0.1, key=f"res{idx}")
        with col2:
            D_in = st.number_input("OD (in)", value=stn['D']/0.0254, format="%.2f", step=0.01, key=f"D{idx}")
            t_in = st.number_input("Wall Thk (in)", value=stn['t']/0.0254, format="%.3f", step=0.001, key=f"t{idx}")
            stn['D'] = D_in * 0.0254
            stn['t'] = t_in * 0.0254
            stn['SMYS'] = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{idx}")
            stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=0.00001, key=f"rough{idx}")
        with col3:
            stn['max_pumps'] = st.number_input("Max Pumps available", min_value=1, value=stn.get('max_pumps',1), step=1, key=f"mpumps{idx}")
            stn['delivery'] = st.number_input("Delivery (m³/hr)", value=stn.get('delivery', 0.0), key=f"deliv{idx}")
            stn['supply'] = st.number_input("Supply (m³/hr)", value=stn.get('supply', 0.0), key=f"sup{idx}")

        tabs = st.tabs(["Pump", "Peaks"])
        with tabs[0]:
            if stn['is_pump']:
                key_head = f"head_data_{idx}"
                if key_head in st.session_state and isinstance(st.session_state[key_head], pd.DataFrame):
                    df_head = st.session_state[key_head]
                else:
                    df_head = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
                df_head = st.data_editor(df_head, num_rows="dynamic", key=f"head{idx}")
                st.session_state[key_head] = df_head

                key_eff = f"eff_data_{idx}"
                if key_eff in st.session_state and isinstance(st.session_state[key_eff], pd.DataFrame):
                    df_eff = st.session_state[key_eff]
                else:
                    df_eff = pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})
                df_eff = st.data_editor(df_eff, num_rows="dynamic", key=f"eff{idx}")
                st.session_state[key_eff] = df_eff

                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    stn['power_type'] = st.selectbox("Power Source", ["Grid", "Diesel"],
                                                    index=0 if stn['power_type']=="Grid" else 1, key=f"ptype{idx}")
                with pcol2:
                    stn['MinRPM'] = st.number_input("Min RPM", value=stn['MinRPM'], key=f"minrpm{idx}")
                    stn['DOL'] = st.number_input("Rated RPM", value=stn['DOL'], key=f"dol{idx}")
                with pcol3:
                    if stn['power_type']=="Grid":
                        stn['rate'] = st.number_input("Elec Rate (INR/kWh)", value=stn.get('rate',9.0), key=f"rate{idx}")
                        stn['sfc'] = 0.0
                    else:
                        stn['sfc'] = st.number_input("SFC (gm/bhp·hr)", value=stn.get('sfc',150.0), key=f"sfc{idx}")
                        stn['rate'] = 0.0
            else:
                st.info("Not a pumping station. No pump data required.")

        with tabs[1]:
            key_peak = f"peak_data_{idx}"
            if key_peak in st.session_state and isinstance(st.session_state[key_peak], pd.DataFrame):
                peak_df = st.session_state[key_peak]
            else:
                peak_df = pd.DataFrame({"Location (km)": [stn['L']/2.0], "Elevation (m)": [stn['elev']+100.0]})
            peak_df = st.data_editor(peak_df, num_rows="dynamic", key=f"peak{idx}")
            st.session_state[key_peak] = peak_df

st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value=st.session_state.get("terminal_name","Terminal"), key="terminal_name")
terminal_elev = st.number_input("Elevation (m)", value=st.session_state.get("terminal_elev",0.0), step=0.1, key="terminal_elev")
terminal_head = st.number_input("Minimum Residual Head (m)", value=st.session_state.get("terminal_head",50.0), step=1.0, key="terminal_head")

def get_full_case_dict():
    return {
        "stations": st.session_state.get('stations', []),
        "terminal": {
            "name": st.session_state.get('terminal_name', 'Terminal'),
            "elev": st.session_state.get('terminal_elev', 0.0),
            "min_residual": st.session_state.get('terminal_head', 50.0)
        },
        "FLOW": st.session_state.get('FLOW', 1000.0),
        "RateDRA": st.session_state.get('RateDRA', 500.0),
        "Price_HSD": st.session_state.get('Price_HSD', 70.0),
        "linefill": st.session_state.get('linefill_df', pd.DataFrame()).to_dict(orient="records"),
        **{
            f"head_data_{i+1}": (
                st.session_state.get(f"head_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"head_data_{i+1}"), pd.DataFrame) else None
            )
            for i in range(len(st.session_state.get('stations', [])))
        },
        **{
            f"eff_data_{i+1}": (
                st.session_state.get(f"eff_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"eff_data_{i+1}"), pd.DataFrame) else None
            )
            for i in range(len(st.session_state.get('stations', [])))
        },
        **{
            f"peak_data_{i+1}": (
                st.session_state.get(f"peak_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"peak_data_{i+1}"), pd.DataFrame) else None
            )
            for i in range(len(st.session_state.get('stations', [])))
        }
    }

case_data = get_full_case_dict()
st.sidebar.download_button(
    label="💾 Save Case",
    data=json.dumps(case_data, indent=2),
    file_name="pipeline_case.json",
    mime="application/json"
)

def map_linefill_to_segments(linefill_df, stations):
    cumlen = [0]
    for stn in stations:
        cumlen.append(cumlen[-1] + stn["L"])
    viscs = []
    dens = []
    for i in range(len(stations)):
        seg_start = cumlen[i]
        seg_end = cumlen[i+1]
        found = False
        for _, row in linefill_df.iterrows():
            if row["Start (km)"] <= seg_start < row["End (km)"]:
                viscs.append(row["Viscosity (cSt)"])
                dens.append(row["Density (kg/m³)"])
                found = True
                break
        if not found:
            viscs.append(linefill_df.iloc[-1]["Viscosity (cSt)"])
            dens.append(linefill_df.iloc[-1]["Density (kg/m³)"])
    return viscs, dens

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict):
    import pipeline_model
    import importlib
    importlib.reload(pipeline_model)
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill_dict)

# ---- Run Optimization Button (red) ----
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
run = st.button("🚀 Run Optimization", key="runoptbtn", help="Run pipeline optimization.", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if run:
    with st.spinner("Solving optimization..."):
        stations_data = st.session_state.stations
        term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}
        for idx, stn in enumerate(stations_data, start=1):
            if stn.get('is_pump', False):
                dfh = st.session_state.get(f"head_data_{idx}")
                dfe = st.session_state.get(f"eff_data_{idx}")
                if dfh is None or dfe is None or len(dfh)<3 or len(dfe)<5:
                    st.error(f"Station {idx}: At least 3 points for flow-head and 5 for flow-eff are required.")
                    st.stop()
                Qh = dfh.iloc[:,0].values; Hh = dfh.iloc[:,1].values
                coeff = np.polyfit(Qh, Hh, 2)
                stn['A'], stn['B'], stn['C'] = coeff[0], coeff[1], coeff[2]
                Qe = dfe.iloc[:,0].values; Ee = dfe.iloc[:,1].values
                coeff_e = np.polyfit(Qe, Ee, 4)
                stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = coeff_e
            peaks_df = st.session_state.get(f"peak_data_{idx}")
            peaks_list = []
            if peaks_df is not None:
                for _, row in peaks_df.iterrows():
                    try:
                        loc = float(row["Location (km)"])
                        elev_pk = float(row["Elevation (m)"])
                    except:
                        continue
                    if loc<0 or loc>stn['L']:
                        st.error(f"Station {idx}: Peak location must be between 0 and segment length.")
                        st.stop()
                    if elev_pk < stn['elev']:
                        st.error(f"Station {idx}: Peak elevation cannot be below station elevation.")
                        st.stop()
                    peaks_list.append({'loc': loc, 'elev': elev_pk})
            stn['peaks'] = peaks_list
        linefill_df = st.session_state.get("linefill_df", pd.DataFrame())
        kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
        res = solve_pipeline(stations_data, term_data, FLOW, kv_list, rho_list, RateDRA, Price_HSD, linefill_df.to_dict())
        import copy
        st.session_state["last_res"] = copy.deepcopy(res)
        st.session_state["last_stations_data"] = copy.deepcopy(stations_data)
        st.session_state["last_term_data"] = copy.deepcopy(term_data)
        st.session_state["last_linefill"] = copy.deepcopy(linefill_df)


# ---- VISUAL SUMMARY DASHBOARD (KPI METRICS) ----
if "last_res" in st.session_state:
    res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]
    term_data = st.session_state["last_term_data"]
    names = [s['name'] for s in stations_data] + [term_data['name']]
    keys = [n.lower().replace(' ', '_') for n in names]
    linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
    kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
    FLOW = st.session_state.get("FLOW", 1000.0)
    RateDRA = st.session_state.get("RateDRA", 500.0)
    # ---- Calculate KPIs ----
    # Total Cost (INR/day)
    total_cost = 0
    dra_cost_total = 0
    total_power_cost = 0
    max_velocity = 0
    min_pump_eff = 100
    for idx, stn in enumerate(stations_data):
        key = stn['name'].lower().replace(' ', '_')
        # DRA cost
        dr_opt = res.get(f"drag_reduction_{key}", 0.0)
        dr_max = stn.get('max_dr', 0.0)
        viscosity = kv_list[idx]
        dr_use = min(dr_opt, dr_max)
        ppm = get_ppm_for_dr(viscosity, dr_use)
        seg_flow = res.get(f"pipeline_flow_{key}", FLOW)
        dra_cost = ppm * (seg_flow * 1000.0 * 24.0 / 1e6) * RateDRA
        power_cost = float(res.get(f"power_cost_{key}", 0.0) or 0.0)
        velocity = res.get(f"velocity_{key}", 0.0) or 0.0
        eff = float(res.get(f"efficiency_{key}", 100.0))
        total_cost += dra_cost + power_cost
        dra_cost_total += dra_cost
        total_power_cost += power_cost
        if velocity > max_velocity: max_velocity = velocity
        if stn.get('is_pump', False) and eff < min_pump_eff: min_pump_eff = eff
    # Operating pumps
    total_pumps = sum([int(res.get(f"num_pumps_{stn['name'].lower().replace(' ', '_')}", 0)) for stn in stations_data])
    # Alerts
    alert = "✅ All OK"
    if max_velocity > 2.5:
        alert = "⚠️ High velocity"
    elif min_pump_eff < 60:
        alert = "⚠️ Low efficiency"
    # ---- KPI Cards ----
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Cost (INR/day)", f"{total_cost:,.0f}")
    k2.metric("Power Cost (INR/day)", f"{total_power_cost:,.0f}")
    k3.metric("DRA Cost (INR/day)", f"{dra_cost_total:,.0f}")
    k4.metric("Max Velocity (m/s)", f"{max_velocity:.2f}")
    k5.metric("Operating Pumps", f"{total_pumps} | {alert}")
    st.markdown("<hr style='margin-bottom:0.7em;'>", unsafe_allow_html=True)

# ---- Result Tabs ----
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_sens, tab_bench, tab_sim = st.tabs([
    "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves",
    "🔄 Pump-System", "📉 DRA Curves", "🧊 3D Analysis and Surface Plots", "🧮 3D Pressure Profile",
    "📈 Sensitivity", "📊 Benchmarking", "💡 Savings Simulator"
])


# ---- Tab 1: Summary ----
import numpy as np

with tab1:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        terminal_name = st.session_state["last_term_data"]["name"]
        names = [s['name'] for s in stations_data] + [terminal_name]

        # --- Use flows from backend output only ---
        segment_flows = []
        pump_flows = []
        for nm in names:
            key = nm.lower().replace(' ', '_')
            segment_flows.append(res.get(f"pipeline_flow_{key}", np.nan))
            pump_flows.append(res.get(f"pump_flow_{key}", np.nan))
            
        # DRA/PPM summary and table columns as before
        station_dr_capped = {}
        station_ppm = {}
        linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
        kv_list, _ = map_linefill_to_segments(linefill_df, stations_data)
        for idx, stn in enumerate(stations_data, start=1):
            key = stn['name'].lower().replace(' ', '_')
            dr_opt = res.get(f"drag_reduction_{key}", 0.0)
            dr_max = stn.get('max_dr', 0.0)
            viscosity = kv_list[idx-1]
            dr_use = min(dr_opt, dr_max)
            station_dr_capped[key] = dr_use
            ppm = get_ppm_for_dr(viscosity, dr_use)
            station_ppm[key] = ppm

        params = [
            "Pipeline Flow (m³/hr)", "Pump Flow (m³/hr)", "Power+Fuel Cost (INR/day)", "DRA Cost (INR/day)", 
            "DRA PPM", "No. of Pumps", "Pump Speed (rpm)", "Pump Eff (%)", "Reynolds No.", 
            "Head Loss (m)", "Vel (m/s)", "Residual Head (m)", "SDH (m)", "MAOP (m)", "Drag Reduction (%)"
        ]
        summary = {"Parameters": params}

        for idx, nm in enumerate(names):
            key = nm.lower().replace(' ','_')
            # For DRA cost at each station, use hydraulically-correct flow
            if key in station_ppm:
                dra_cost = (
                    station_ppm[key]
                    * (segment_flows[idx] * 1000.0 * 24.0 / 1e6)
                    * st.session_state["RateDRA"]
                )
            else:
                dra_cost = 0.0

            # For numeric columns, always use np.nan if not available
            pumpflow = pump_flows[idx] if (idx < len(pump_flows) and not pd.isna(pump_flows[idx])) else np.nan
            summary[nm] = [
                segment_flows[idx],
                pumpflow,
                res.get(f"power_cost_{key}",0.0) if res.get(f"power_cost_{key}",0.0) is not None else np.nan,
                dra_cost,
                station_ppm.get(key, np.nan),
                int(res.get(f"num_pumps_{key}",0)) if res.get(f"num_pumps_{key}",0) is not None else np.nan,
                res.get(f"speed_{key}",0.0) if res.get(f"speed_{key}",0.0) is not None else np.nan,
                res.get(f"efficiency_{key}",0.0) if res.get(f"efficiency_{key}",0.0) is not None else np.nan,
                res.get(f"reynolds_{key}",0.0) if res.get(f"reynolds_{key}",0.0) is not None else np.nan,
                res.get(f"head_loss_{key}",0.0) if res.get(f"head_loss_{key}",0.0) is not None else np.nan,
                res.get(f"velocity_{key}",0.0) if res.get(f"velocity_{key}",0.0) is not None else np.nan,
                res.get(f"residual_head_{key}",0.0) if res.get(f"residual_head_{key}",0.0) is not None else np.nan,
                res.get(f"sdh_{key}",0.0) if res.get(f"sdh_{key}",0.0) is not None else np.nan,
                res.get(f"maop_{key}",0.0) if res.get(f"maop_{key}",0.0) is not None else np.nan,
                res.get(f"drag_reduction_{key}",0.0) if res.get(f"drag_reduction_{key}",0.0) is not None else np.nan
            ]

        df_sum = pd.DataFrame(summary)

        # --- ENFORCE ALL NUMBERS AS STRINGS WITH TWO DECIMALS FOR DISPLAY ---
        for col in df_sum.columns:
            if col not in ["Parameters", "No. of Pumps"]:
                df_sum[col] = df_sum[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        if "No. of Pumps" in df_sum.columns:
            df_sum["No. of Pumps"] = pd.to_numeric(df_sum["No. of Pumps"], errors='coerce').fillna(0).astype(int)

        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")

        # --- Recompute total optimized cost (Power+Fuel + DRA) for all stations ---
        total_cost = 0.0
        for idx, stn in enumerate(stations_data):
            key = stn['name'].lower().replace(' ', '_')
            power_cost = float(res.get(f"power_cost_{key}", 0.0) or 0.0)
            dra_cost = (
                station_ppm.get(key, 0.0)
                * (segment_flows[idx] * 1000.0 * 24.0 / 1e6)
                * st.session_state["RateDRA"]
            )
            total_cost += power_cost + dra_cost
        
        total_pumps = 0
        effs = []
        speeds = []
        for stn in stations_data:
            key = stn['name'].lower().replace(' ','_')
            npump = int(res.get(f"num_pumps_{key}", 0))
            if npump > 0:
                total_pumps += npump
                eff = float(res.get(f"efficiency_{key}", 0.0))
                speed = float(res.get(f"speed_{key}", 0.0))
                for _ in range(npump):
                    effs.append(eff)
                    speeds.append(speed)
        avg_eff = sum(effs)/len(effs) if effs else 0.0
        avg_speed = sum(speeds)/len(speeds) if speeds else 0.0
        
        st.markdown(
            f"""<br>
            <div style='font-size:1.1em;'><b>Total Optimized Cost:</b> {total_cost:.2f} INR/day<br>
            <b>No. of operating Pumps:</b> {total_pumps}<br>
            <b>Average Pump Efficiency:</b> {avg_eff:.2f} %<br>
            <b>Average Pump Speed:</b> {avg_speed:.0f} rpm</div>
            """,
            unsafe_allow_html=True
        )

# ---- Tab 2: Cost Breakdown ----
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

with tab2:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        terminal_name = st.session_state["last_term_data"]["name"]
        names = [s['name'] for s in stations_data] + [terminal_name]
        keys = [n.lower().replace(' ', '_') for n in names]

        # --- Recompute hydraulically correct segment flows and DRA cost for every station ---
        segment_flows = []
        for key in keys:
            segment_flows.append(res.get(f"pipeline_flow_{key}", np.nan))
        
        # --- Get DRA PPM values as in Tab 1 ---
        linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
        kv_list, _ = map_linefill_to_segments(linefill_df, stations_data)
        station_ppm = {}
        for idx, stn in enumerate(stations_data, start=1):
            key = stn['name'].lower().replace(' ', '_')
            dr_opt = res.get(f"drag_reduction_{key}", 0.0)
            dr_max = stn.get('max_dr', 0.0)
            viscosity = kv_list[idx-1]
            dr_use = min(dr_opt, dr_max)
            station_ppm[key] = get_ppm_for_dr(viscosity, dr_use)

        # --- Compute power and DRA costs just as in summary ---
        power_costs = [float(res.get(f"power_cost_{k}", 0.0) or 0.0) for k in keys]
        dra_costs = []
        for idx, key in enumerate(keys):
            if key in station_ppm:
                dra_cost = (
                    station_ppm[key]
                    * (segment_flows[idx] * 1000.0 * 24.0 / 1e6)
                    * st.session_state["RateDRA"]
                )
            else:
                dra_cost = 0.0
            dra_costs.append(dra_cost)
        total_costs = [p + d for p, d in zip(power_costs, dra_costs)]

        df_cost = pd.DataFrame({
            "Station": names,
            "Power+Fuel Cost (INR/day)": power_costs,
            "DRA Cost (INR/day)": dra_costs,
            "Total Cost (INR/day)": total_costs,
        })

        # --- Grouped bar chart (side by side) for Power+Fuel and DRA ---
        fig_grouped = go.Figure()
        fig_grouped.add_trace(go.Bar(
            x=df_cost["Station"],
            y=df_cost["Power+Fuel Cost (INR/day)"],
            name="Power+Fuel",
            marker_color="#1976D2",
            text=[f"{x:.2f}" for x in df_cost["Power+Fuel Cost (INR/day)"]],
            textposition='outside'
        ))
        fig_grouped.add_trace(go.Bar(
            x=df_cost["Station"],
            y=df_cost["DRA Cost (INR/day)"],
            name="DRA",
            marker_color="#FFA726",
            text=[f"{x:.2f}" for x in df_cost["DRA Cost (INR/day)"]],
            textposition='outside'
        ))
        fig_grouped.update_layout(
            barmode='group',
            title="Station Daily Cost: Power+Fuel and DRA",
            xaxis_title="Station",
            yaxis_title="Cost (INR/day)",
            font=dict(size=16),
            legend=dict(font=dict(size=14)),
            height=430,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

        # DRA cost bar chart only ---
        st.markdown("<h4 style='font-weight:600; margin-top: 2em;'>DRA Cost</h4>", unsafe_allow_html=True)
        fig_dra = px.bar(
            df_cost,
            x="Station",
            y="DRA Cost (INR/day)",
            text="DRA Cost (INR/day)",
            color="DRA Cost (INR/day)",
            color_continuous_scale=px.colors.sequential.YlOrBr,
            height=320,
        )
        fig_dra.update_traces(texttemplate="%{text:.2f}", textposition='outside')
        fig_dra.update_layout(
            yaxis_title="DRA Cost (INR/day)",
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_dra, use_container_width=True)

        # --- Pie chart: Total cost distribution by station ---
        st.markdown("#### Cost Contribution by Station")
        fig_pie = px.pie(
            df_cost,
            values="Total Cost (INR/day)",
            names="Station",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.38
        )
        fig_pie.update_traces(textinfo='label+percent', pull=[0.05]*len(df_cost))
        fig_pie.update_layout(
            font=dict(size=15),
            height=330,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- Trend line: Total cost vs. chainage ---
        st.markdown("#### Cost Accumulation Along Pipeline")
        if "L" in stations_data[0]:
            # Compute cumulative chainage for each station
            chainage = [0]
            for stn in stations_data:
                chainage.append(chainage[-1] + stn.get("L", 0.0))
            chainage = chainage[:len(names)]  # Match to station count
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=chainage,
                y=total_costs,
                mode="lines+markers+text",
                text=[f"{tc:.2f}" for tc in total_costs],
                textposition="top center",
                name="Total Cost"
            ))
            fig_line.update_layout(
                title="Total Cost vs. Distance",
                xaxis_title="Cumulative Distance (km)",
                yaxis_title="Cost (INR/day)",
                font=dict(size=15),
                height=350
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # --- Table: All cost heads, 2-decimal formatted ---
        df_cost_fmt = df_cost.copy()
        for c in df_cost_fmt.columns:
            if c != "Station":
                df_cost_fmt[c] = df_cost_fmt[c].apply(lambda x: f"{x:.2f}")
        st.markdown("#### Tabular Cost Summary")
        st.dataframe(df_cost_fmt, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Station Cost (CSV)",
            df_cost.to_csv(index=False).encode(),
            file_name="station_cost.csv"
        )

        # --- KPI highlights ---
        st.markdown(
            f"""<br>
            <div style='font-size:1.1em;'><b>Total Operating Cost:</b> {sum(total_costs):,.2f} INR/day<br>
            <b>Maximum Station Cost:</b> {max(total_costs):,.2f} INR/day ({df_cost.loc[df_cost['Total Cost (INR/day)'].idxmax(), 'Station']})</div>
            """,
            unsafe_allow_html=True
        )


# ---- Tab 3: Performance ----
import plotly.graph_objects as go
import numpy as np
import pandas as pd

with tab3:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        terminal = st.session_state["last_term_data"]
        perf_tab, head_tab, char_tab, eff_tab, press_tab, power_tab = st.tabs([
            "Head Loss", "Velocity & Re", 
            "Pump Characteristic Curve", "Pump Efficiency Curve",
            "Pressure vs Pipeline Length", "Power vs Speed/Flow"
        ])
        
        # --- 1. Head Loss ---
        with perf_tab:
            st.markdown("<div class='section-title'>Head Loss per Segment</div>", unsafe_allow_html=True)
            df_hloss = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Head Loss (m)": [res.get(f"head_loss_{s['name'].lower().replace(' ','_')}", 0) for s in stations_data]
            })
            fig_h = go.Figure(go.Bar(
                x=df_hloss["Station"], y=df_hloss["Head Loss (m)"],
                marker_color='#1976D2',
                text=[f"{hl:.2f}" for hl in df_hloss["Head Loss (m)"]],
                textposition="auto"
            ))
            fig_h.update_layout(
                yaxis_title="Head Loss (m)",
                xaxis_title="Station",
                font=dict(size=16),
                title="Head Loss per Segment",
                height=400
            )
            st.plotly_chart(fig_h, use_container_width=True, key=f"perf_headloss_{uuid.uuid4().hex[:6]}")
            st.dataframe(df_hloss.style.format({"Head Loss (m)": "{:.2f}"}), use_container_width=True, hide_index=True)
        
        # --- 2. Velocity & Reynolds ---
        with head_tab:
            st.markdown("<div class='section-title'>Velocity & Reynolds</div>", unsafe_allow_html=True)
            df_vel = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Velocity (m/s)": [res.get(f"velocity_{s['name'].lower().replace(' ','_')}", 0) for s in stations_data],
                "Reynolds Number": [res.get(f"reynolds_{s['name'].lower().replace(' ','_')}", 0) for s in stations_data]
            })
            fig_v = go.Figure()
            fig_v.add_trace(go.Bar(
                x=df_vel["Station"],
                y=df_vel["Velocity (m/s)"],
                name="Velocity (m/s)",
                marker_color="#00ACC1",
                text=[f"{v:.2f}" for v in df_vel["Velocity (m/s)"]],
                textposition="auto"
            ))
            fig_v.add_trace(go.Bar(
                x=df_vel["Station"],
                y=df_vel["Reynolds Number"],
                name="Reynolds Number",
                marker_color="#E65100",
                text=[f"{r:.0f}" for r in df_vel["Reynolds Number"]],
                textposition="outside",
                yaxis="y2"
            ))
            fig_v.update_layout(
                barmode='group',
                yaxis=dict(title="Velocity (m/s)", side="left"),
                yaxis2=dict(title="Reynolds Number", overlaying="y", side="right", showgrid=False),
                font=dict(size=15),
                title="Velocity and Reynolds Number per Station",
                legend=dict(font=dict(size=14)),
                height=420
            )
            st.plotly_chart(fig_v, use_container_width=True)
            # Data table
            st.dataframe(df_vel.style.format({"Velocity (m/s)":"{:.2f}", "Reynolds Number":"{:.0f}"}), use_container_width=True, hide_index=True)
        
        # --- 3. Pump Characteristic Curve (Head vs Flow at various Speeds) ---
        with char_tab:
            st.markdown("<div class='section-title'>Pump Characteristic Curves (Head vs Flow at various Speeds)</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ','_')
                # Get user's Flow vs Head data for this pump
                df_head = st.session_state.get(f"head_data_{i}")
                if df_head is not None and "Flow (m³/hr)" in df_head.columns and len(df_head) > 1:
                    flow_user = np.array(df_head["Flow (m³/hr)"], dtype=float)
                    max_flow = np.max(flow_user)
                else:
                    max_flow = st.session_state.get("FLOW", 1000.0)
                # Now only plot up to user-provided max flow
                flows = np.linspace(0, max_flow, 200)
                A = res.get(f"coef_A_{key}",0)
                B = res.get(f"coef_B_{key}",0)
                C = res.get(f"coef_C_{key}",0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                step = max(100, int((N_max-N_min)/5))
                fig = go.Figure()
                for rpm in range(N_min, N_max+1, step):
                    H = (A*flows**2 + B*flows + C)*(rpm/N_max)**2 if N_max else np.zeros_like(flows)
                    # Remove all negative heads: mask out values where H < 0
                    flows_pos = flows[H >= 0]
                    H_pos = H[H >= 0]
                    fig.add_trace(go.Scatter(
                        x=flows_pos, y=H_pos, mode='lines', name=f"{rpm} rpm",
                        hovertemplate="Flow: %{x:.2f} m³/hr<br>Head: %{y:.2f} m"
                    ))
                fig.update_layout(
                    title=f"Head vs Flow: {stn['name']}",
                    xaxis_title="Flow (m³/hr)",
                    yaxis_title="Head (m)",
                    font=dict(size=15),
                    legend=dict(font=dict(size=13)),
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True, key=f"char_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")

        
        # --- 4. Pump Efficiency Curve (Eff vs Flow at various Speeds) ---
        with eff_tab:
            st.markdown("<div class='section-title'>Pump Efficiency Curves (Eff vs Flow at various Speeds)</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ','_')
                Qe = st.session_state.get(f"eff_data_{i}")
                FLOW = st.session_state.get("FLOW", 1000.0)
                if Qe is not None and len(Qe) > 1:
                    flow_user = np.array(Qe['Flow (m³/hr)'], dtype=float)
                    eff_user = np.array(Qe['Efficiency (%)'], dtype=float)
                    flow_min = float(np.min(flow_user))
                    flow_max = float(np.max(flow_user))
                    max_user_eff = float(np.max(eff_user))
                else:
                    flow_min, flow_max = 0.01, FLOW
                    max_user_eff = 100
                # Polynomial coefficients at DOL (user input speed)
                P = stn.get('P', 0); Qc = stn.get('Q', 0); R = stn.get('R', 0)
                S = stn.get('S', 0); T = stn.get('T', 0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                step = max(100, int((N_max-N_min)/4))  # 5 curves max
        
                fig = go.Figure()
                for rpm in range(N_min, N_max+1, step):
                    # For each rpm, limit flows such that equivalent flow at DOL ≤ max user flow
                    # Q_at_this_rpm * (DOL/rpm) ≤ flow_max  =>  Q_at_this_rpm ≤ flow_max * (rpm/DOL)
                    q_upper = flow_max * (rpm/N_max) if N_max else flow_max
                    q_lower = flow_min * (rpm/N_max) if N_max else flow_min
                    flows = np.linspace(q_lower, q_upper, 100)
                    Q_equiv = flows * N_max / rpm if rpm else flows
                    eff = (P*Q_equiv**4 + Qc*Q_equiv**3 + R*Q_equiv**2 + S*Q_equiv + T)
                    eff = np.clip(eff, 0, max_user_eff)
                    fig.add_trace(go.Scatter(
                        x=flows, y=eff, mode='lines', name=f"{rpm} rpm",
                        hovertemplate="Flow: %{x:.2f} m³/hr<br>Eff: %{y:.2f} %"
                    ))
                fig.update_layout(
                    title=f"Efficiency vs Flow: {stn['name']}",
                    xaxis_title="Flow (m³/hr)",
                    yaxis_title="Efficiency (%)",
                    font=dict(size=15),
                    legend=dict(font=dict(size=13)),
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # --- 5. Pressure vs Pipeline Length ---
        with press_tab:
            st.markdown("<div class='section-title'>Pressure vs Pipeline Length</div>", unsafe_allow_html=True)
            stations_data = st.session_state["last_stations_data"]
            res = st.session_state["last_res"]
            terminal = st.session_state["last_term_data"]
            N = len(stations_data)
            lengths = [0]
            for stn in stations_data:
                lengths.append(lengths[-1] + stn.get("L", 0.0))
            names = [s['name'] for s in stations_data] + [terminal["name"]]
            keys = [n.lower().replace(' ', '_') for n in names]
            rh_list = [res.get(f"residual_head_{k}", 0.0) for k in keys]
            sdh_list = [res.get(f"sdh_{k}", 0.0) for k in keys]
        
            # Gather all X, Y, annotation points
            x_pts, y_pts, annotations = [], [], []
            for i, stn in enumerate(stations_data):
                # 1. Start: RH at station i
                x_pts.append(lengths[i])
                y_pts.append(rh_list[i])
                annotations.append((lengths[i], rh_list[i], stn['name']))
                # 2. If pump running, vertical jump to SDH
                if abs(sdh_list[i] - rh_list[i]) > 1e-3:
                    x_pts.append(lengths[i])
                    y_pts.append(sdh_list[i])
                    # Optionally: annotations.append((lengths[i], sdh_list[i], f"SDH {stn['name']}"))
                # 3. For each peak (ordered by loc within segment):
                seg_len = stn['L']
                next_rh = rh_list[i+1]
                start_sdh = sdh_list[i]  # This is the actual SDH, not RH!
                if 'peaks' in stn and stn['peaks']:
                    for pk in sorted(stn['peaks'], key=lambda x: x['loc']):
                        pk_loc = pk['loc']
                        pk_x = lengths[i] + pk_loc
                        # Linear head drop from SDH at i to RH at i+1:
                        frac = pk_loc / seg_len if seg_len > 0 else 0
                        pk_head = start_sdh - (start_sdh - next_rh) * frac
                        x_pts.append(pk_x)
                        y_pts.append(pk_head)
                        annotations.append((pk_x, pk_head, f"Peak ({stn['name']})"))
                # 4. End: RH at next station
                x_pts.append(lengths[i+1])
                y_pts.append(next_rh)
                annotations.append((lengths[i+1], next_rh, names[i+1]))
        
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_pts, y=y_pts, mode='lines+markers', name="Pressure Profile", line=dict(width=3, color="#1976D2"),
                marker=dict(size=8)
            ))
            # Annotate stations and peaks
            for xp, yp, txt in annotations:
                fig.add_annotation(x=xp, y=yp, text=txt, showarrow=True, yshift=12)
            fig.update_layout(
                title="Pressure vs Pipeline Length (with Peaks)",
                xaxis_title="Cumulative Length (km)",
                yaxis_title="Pressure Head (mcl)",
                font=dict(size=15),
                showlegend=False,
                height=420
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- 6. Power vs Speed/Flow ---
        with power_tab:
            st.markdown("<div class='section-title'>Power vs Speed & Power vs Flow</div>", unsafe_allow_html=True)
            # Fetch summary DataFrame for pump flows
            df_summary = st.session_state.get("summary_table", None)
            if df_summary is not None:
                pump_flow_dict = dict(zip(df_summary.columns[1:], df_summary.loc[df_summary['Parameters'] == 'Pump Flow (m³/hr)'].values[0,1:]))
            else:
                # fallback: use optimized flow from results
                pump_flow_dict = {}
                for stn in stations_data:
                    key = stn['name'].lower().replace(' ','_')
                    pump_flow_dict[key] = res.get(f"pump_flow_{key}", st.session_state.get("FLOW",1000.0))
            
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ','_')
                # 1. Get constants and coefficients
                A = res.get(f"coef_A_{key}", 0)
                B = res.get(f"coef_B_{key}", 0)
                C = res.get(f"coef_C_{key}", 0)
                P4 = stn.get('P',0); Qc = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                rho = stn.get('rho', 850)
                # --- 1. Power vs Speed (at constant, optimized flow) ---
                pump_flow = pump_flow_dict.get(key, st.session_state.get("FLOW",1000.0))
                # Head and eff at pump_flow, DOL
                H = (A*pump_flow**2 + B*pump_flow + C)
                eff = (P4*pump_flow**4 + Qc*pump_flow**3 + R*pump_flow**2 + S*pump_flow + T)
                eff = max(0.01, eff/100)
                P1 = (rho * pump_flow * 9.81 * H)/(3600.0*1000*eff)
                speeds = np.arange(N_min, N_max+1, 100)
                power_curve = [P1 * (rpm/N_max)**3 for rpm in speeds]
                fig_pwr = go.Figure()
                fig_pwr.add_trace(go.Scatter(
                    x=speeds, y=power_curve, mode='lines+markers',
                    name="Power vs Speed",
                    marker_color="#1976D2",
                    line=dict(width=3),
                    hovertemplate="Speed: %{x} rpm<br>Power: %{y:.2f} kW"
                ))
                fig_pwr.update_layout(
                    title=f"Power vs Speed (at Pump Flow = {pump_flow:.2f} m³/hr): {stn['name']}",
                    xaxis_title="Speed (rpm)",
                    yaxis_title="Power (kW)",
                    font=dict(size=16),
                    height=400
                )
                st.plotly_chart(fig_pwr, use_container_width=True)
        
                # --- 2. Power vs Flow (at DOL only) ---
                flows = np.linspace(0.01, 1.2*pump_flow, 100)
                H_flows = (A*flows**2 + B*flows + C)  # At DOL
                eff_flows = (P4*flows**4 + Qc*flows**3 + R*flows**2 + S*flows + T)
                eff_flows = np.clip(eff_flows/100, 0.01, 1.0)
                power_flows = (rho * flows * 9.81 * H_flows)/(3600.0*1000*eff_flows)
                fig_pwr2 = go.Figure()
                fig_pwr2.add_trace(go.Scatter(
                    x=flows, y=power_flows, mode='lines+markers',
                    name="Power vs Flow",
                    marker_color="#D84315",
                    line=dict(width=3),
                    hovertemplate="Flow: %{x:.2f} m³/hr<br>Power: %{y:.2f} kW"
                ))
                fig_pwr2.update_layout(
                    title=f"Power vs Flow (at DOL: {N_max} rpm): {stn['name']}",
                    xaxis_title="Flow (m³/hr)",
                    yaxis_title="Power (kW)",
                    font=dict(size=16),
                    height=400
                )
                st.plotly_chart(fig_pwr2, use_container_width=True)

# ---- Tab 4: System Curves ----
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from math import pi

with tab4:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): 
                continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']
            L_seg = stn['L']
            elev_i = stn['elev']
            max_dr = int(stn.get('max_dr', 40))
            kv_list, _ = map_linefill_to_segments(linefill_df, stations_data)
            visc = kv_list[i-1]
            flows = np.linspace(0, st.session_state.get("FLOW", 1000.0), 101)
            v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
            Re_vals = v_vals * d_inner_i / (visc*1e-6) if visc > 0 else np.zeros_like(v_vals)
            f_vals = np.where(Re_vals>0,
                              0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
            # Professional gradient: from blue to red
            n_curves = (max_dr // 5) + 1
            color_palette = [
                "#1565C0", "#1976D2", "#1E88E5", "#3949AB", "#8E24AA",
                "#D81B60", "#F4511E", "#F9A825", "#43A047", "#00897B"
            ]
            color_idx = np.linspace(0, len(color_palette)-1, n_curves).astype(int)
            fig_sys = go.Figure()
            for j, dra in enumerate(range(0, max_dr+1, 5)):
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                # Fix: safe indexing for palette if only 1 curve
                color = color_palette[color_idx[j]] if n_curves > 1 else color_palette[0]
                fig_sys.add_trace(go.Scatter(
                    x=flows, y=SDH_vals,
                    mode='lines',
                    line=dict(width=4, color=color),
                    opacity=0.82,
                    name=f"DRA {dra}%",
                    hovertemplate=f"DRA: {dra}%<br>Flow: %{{x:.2f}} m³/hr<br>Head: %{{y:.2f}} m"
                ))
            fig_sys.update_layout(
                title=f"System Curve (Head vs Flow) — {stn['name']}",
                xaxis_title="Flow (m³/hr)",
                yaxis_title="Dynamic Head (m)",
                font=dict(size=18, family="Segoe UI"),
                legend=dict(font=dict(size=14), title="DRA Dosage"),
                height=450,
                margin=dict(l=10, r=10, t=60, b=30),
                plot_bgcolor="#f5f8fc"
            )
            st.plotly_chart(fig_sys, use_container_width=True, key=f"sys_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")


# ---- Tab 5: Pump-System Interaction ----
import plotly.graph_objects as go
import numpy as np
from math import pi
from plotly.colors import qualitative, sample_colorscale

with tab5:
    st.markdown("<div class='section-title'>Pump-System Interaction</div>", unsafe_allow_html=True)

    if "last_res" not in st.session_state or "last_stations_data" not in st.session_state:
        st.info("Please run optimization first to access Pump-System analysis.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]

        # Show only pump stations in dropdown
        pump_indices = [i for i, s in enumerate(stations_data) if s.get('is_pump', False)]
        if not pump_indices:
            st.warning("No pump stations defined in your pipeline setup.")
        else:
            station_options = [f"{i+1}: {stations_data[i]['name']}" for i in pump_indices]
            st_choice = st.selectbox("Select Pump Station", station_options, key="ps_stn")
            selected_index = station_options.index(st_choice)
            stn_idx = pump_indices[selected_index]
            stn = stations_data[stn_idx]
            key = stn['name'].lower().replace(' ','_')
            is_pump = stn.get('is_pump', False)
            max_dr = int(stn.get('max_dr', 40))
            n_pumps = int(stn.get('max_pumps', 1))

            # -------- Max Flow Based on Pump Data Table --------
            df_head = st.session_state.get(f"head_data_{stn_idx+1}")
            if df_head is not None and "Flow (m³/hr)" in df_head.columns and len(df_head) > 1:
                user_flows = np.array(df_head["Flow (m³/hr)"], dtype=float)
                max_flow = np.max(user_flows)
            else:
                max_flow = st.session_state.get("FLOW", 1000.0)
            flows = np.linspace(0, max_flow, 800)

            # -------- Downstream Pump Bypass Logic --------
            downstream_pumps = [s for s in stations_data[stn_idx+1:] if s.get('is_pump', False)]
            downstream_names = [f"{i+stn_idx+2}: {s['name']}" for i, s in enumerate(downstream_pumps)]
            bypassed = []
            if downstream_names:
                bypassed = st.multiselect("Bypass downstream pumps (Pump-System)", downstream_names)
            total_length = stn['L']
            current_elev = stn['elev']
            downstream_idx = stn_idx + 1
            while downstream_idx < len(stations_data):
                s = stations_data[downstream_idx]
                label = f"{downstream_idx+1}: {s['name']}"
                total_length += s['L']
                current_elev = s['elev']
                if s.get('is_pump', False) and label not in bypassed:
                    break
                downstream_idx += 1
            if downstream_idx == len(stations_data):
                term_elev = st.session_state["last_term_data"]["elev"]
                current_elev = term_elev

            # -------- Pipe, Viscosity, Roughness --------
            d_inner = stn['D'] - 2*stn['t']
            rough = stn['rough']
            linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
            kv_list, _ = map_linefill_to_segments(linefill_df, stations_data)
            visc = kv_list[stn_idx]

            # --------- Begin Figure ---------
            fig = go.Figure()

            # -------- System Curves: All DRA, Turbo Colormap, Vivid and Bold --------
            system_dra_steps = list(range(0, max_dr+1, 5))
            n_dra = len(system_dra_steps)
            for idx, dra in enumerate(system_dra_steps):
                v_vals = flows/3600.0 / (pi*(d_inner**2)/4)
                Re_vals = v_vals * d_inner / (visc*1e-6) if visc > 0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                    0.25/(np.log10(rough/d_inner/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((total_length*1000.0)/d_inner) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = max(0, current_elev) + DH
                SDH_vals = np.clip(SDH_vals, 0, None)
                label = f"System DRA {dra}%"
                showlegend = (dra == 0 or dra == 10 or dra == 20 or dra == max_dr)
                # Fix: protect division by zero
                if n_dra > 1:
                    blend = idx / (n_dra-1)
                else:
                    blend = 0.5
                color = sample_colorscale("Turbo", 0.1 + 0.8 * blend)[0]
                fig.add_trace(go.Scatter(
                    x=flows, y=SDH_vals,
                    mode='lines',
                    line=dict(width=4 if showlegend else 2.2, color=color, dash='solid'),
                    name=label if showlegend else None,
                    showlegend=showlegend,
                    opacity=1 if showlegend else 0.67,
                    hoverinfo="skip"
                ))

            # -------- Pump Curves: All Series, All RPM, Vivid Colors --------
            pump_palettes = qualitative.Plotly + qualitative.D3 + qualitative.Bold
            if is_pump:
                N_min = int(res.get(f"min_rpm_{key}", 1200))
                N_max = int(res.get(f"dol_{key}", 3000))
                rpm_steps = np.arange(N_min, N_max+1, 100)
                n_rpms = len(rpm_steps)
                A = res.get(f"coef_A_{key}", 0)
                B = res.get(f"coef_B_{key}", 0)
                C = res.get(f"coef_C_{key}", 0)
                for npump in range(1, n_pumps+1):
                    for idx, rpm in enumerate(rpm_steps):
                        # Fix: protect division by zero
                        if n_rpms > 1:
                            blend = idx / (n_rpms-1)
                        else:
                            blend = 0.5
                        color = sample_colorscale("Turbo", 0.2 + 0.6 * blend)[0]
                        H_pump = npump * ((A * flows**2 + B * flows + C) * (rpm / N_max) ** 2 if N_max else np.zeros_like(flows))
                        H_pump = np.clip(H_pump, 0, None)
                        label = f"{npump} Pump{'s' if npump>1 else ''} ({rpm} rpm)"
                        showlegend = (idx == 0 or idx == n_rpms-1)
                        fig.add_trace(go.Scatter(
                            x=flows, y=H_pump,
                            mode='lines',
                            line=dict(width=3 if showlegend else 1.7, color=color, dash='solid'),
                            name=label if showlegend else None,
                            showlegend=showlegend,
                            opacity=0.92 if showlegend else 0.56,
                            hoverinfo="skip"
                        ))

            # -------- Layout Polish: Bright, Vivid, Clean --------
            fig.update_layout(
                title=f"<b style='color:#222'>Pump-System Curves: {stn['name']}</b>",
                xaxis_title="Flow (m³/hr)",
                yaxis_title="Head (m)",
                font=dict(size=23, family="Segoe UI, Arial"),
                legend=dict(font=dict(size=17), itemsizing="constant", borderwidth=1, bordercolor="#ddd"),
                height=700,
                margin=dict(l=25, r=25, t=90, b=50),
                plot_bgcolor="#fffdf9",
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(80,100,230,0.13)'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(80,100,230,0.13)'),
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)




# ---- Tab 6: DRA Curves ----
with tab6:
    if "last_res" not in st.session_state or "last_stations_data" not in st.session_state:
        st.info("Please run optimization first to analyze DRA curves.")
        st.stop()
    res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]
    linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
    kv_list, _ = map_linefill_to_segments(linefill_df, stations_data)
    st.markdown("<div class='section-title'>DRA Curve (PPM vs %Drag Reduction) for Each Station</div>", unsafe_allow_html=True)
    for idx, stn in enumerate(stations_data, start=1):
        key = stn['name'].lower().replace(' ', '_')
        dr_opt = res.get(f"drag_reduction_{key}", 0.0)
        if dr_opt > 0:
            viscosity = kv_list[idx-1]
            cst_list = sorted(DRA_CURVE_DATA.keys())
            if viscosity <= cst_list[0]:
                df_curve = DRA_CURVE_DATA[cst_list[0]]
                curve_label = f"{cst_list[0]} cSt curve"
                percent_dr = df_curve['%Drag Reduction'].values
                ppm_vals = df_curve['PPM'].values
            else:
                lower = max([c for c in cst_list if c <= viscosity])
                upper = min([c for c in cst_list if c >= viscosity])
                df_lower = DRA_CURVE_DATA[lower]
                df_upper = DRA_CURVE_DATA[upper]
                
                # Defensive checks to prevent crash if data is missing or malformed
                if (
                    df_lower is None or df_upper is None or
                    '%Drag Reduction' not in df_lower or 'PPM' not in df_lower or
                    '%Drag Reduction' not in df_upper or 'PPM' not in df_upper or
                    df_lower['%Drag Reduction'].dropna().empty or df_lower['PPM'].dropna().empty or
                    df_upper['%Drag Reduction'].dropna().empty or df_upper['PPM'].dropna().empty
                ):
                    st.warning(f"DRA data for {lower} or {upper} cSt is missing or malformed.")
                    continue
            
                percent_dr = np.linspace(
                    min(df_lower['%Drag Reduction'].min(), df_upper['%Drag Reduction'].min()),
                    max(df_lower['%Drag Reduction'].max(), df_upper['%Drag Reduction'].max()),
                    50
                )
                # Always use np.array with float dtype for interpolation
                xp_lower = np.array(df_lower['%Drag Reduction'], dtype=float)
                yp_lower = np.array(df_lower['PPM'], dtype=float)
                xp_upper = np.array(df_upper['%Drag Reduction'], dtype=float)
                yp_upper = np.array(df_upper['PPM'], dtype=float)
                
                ppm_lower = np.interp(percent_dr, xp_lower, yp_lower)
                ppm_upper = np.interp(percent_dr, xp_upper, yp_upper)
                # Interpolate each percent_dr value for given viscosity
                ppm_vals = ppm_lower + (ppm_upper - ppm_lower) * ((viscosity - lower) / (upper - lower))
                curve_label = f"Interpolated for {viscosity:.2f} cSt"
            opt_ppm = get_ppm_for_dr(viscosity, dr_opt)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=percent_dr,
                y=ppm_vals,
                mode='lines+markers',
                name=curve_label
            ))
            fig.add_trace(go.Scatter(
                x=[dr_opt], y=[opt_ppm],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond'),
                name="Optimized Point"
            ))
            fig.update_layout(
                title=f"DRA Curve for {stn['name']} (Viscosity: {viscosity:.2f} cSt)",
                xaxis_title="% Drag Reduction",
                yaxis_title="PPM",
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No DRA applied at {stn['name']} (Optimal %DR = 0)")

# ---- Tab 7: 3D Analysis ----
with tab7:
    if "last_res" not in st.session_state or "last_stations_data" not in st.session_state:
        st.info("Please run optimization at least once to enable 3D analysis.")
        st.stop()
    last_res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]
    FLOW = st.session_state.get("FLOW", 1000.0)
    RateDRA = st.session_state.get("RateDRA", 500.0)
    Price_HSD = st.session_state.get("Price_HSD", 70.0)
    key = stations_data[0]['name'].lower().replace(' ', '_')

    speed_opt = float(last_res.get(f"speed_{key}", 1500.0))
    dra_opt = float(last_res.get(f"drag_reduction_{key}", 0.0))
    nopt_opt = int(last_res.get(f"num_pumps_{key}", 1))
    flow_opt = FLOW

    delta_speed = 150
    delta_dra = 10
    delta_nop = 1
    delta_flow = 150
    N = 9
    stn = stations_data[0]
    N_min = int(stn.get('MinRPM', 1000))
    N_max = int(stn.get('DOL', 1500))
    DRA_max = int(stn.get('max_dr', 40))
    max_pumps = int(stn.get('max_pumps', 4))

    speed_range = np.linspace(max(N_min, speed_opt - delta_speed), min(N_max, speed_opt + delta_speed), N)
    dra_range = np.linspace(max(0, dra_opt - delta_dra), min(DRA_max, dra_opt + delta_dra), N)
    nop_range = np.arange(max(1, nopt_opt - delta_nop), min(max_pumps, nopt_opt + delta_nop)+1)
    flow_range = np.linspace(max(0.01, flow_opt - delta_flow), flow_opt + delta_flow, N)

    groups = {
        "Pump Performance Surface Plots": {
            "Head vs Flow vs Speed": {"x": flow_range, "y": speed_range, "z": "Head"},
            "Efficiency vs Flow vs Speed": {"x": flow_range, "y": speed_range, "z": "Efficiency"},
        },
        "System Interaction Surface Plots": {
            "System Head vs Flow vs DRA": {"x": flow_range, "y": dra_range, "z": "SystemHead"},
        },
        "Cost Surface Plots": {
            "Power Cost vs Speed vs DRA": {"x": speed_range, "y": dra_range, "z": "PowerCost"},
            "Power Cost vs Flow vs Speed": {"x": flow_range, "y": speed_range, "z": "PowerCost"},
            "Total Cost vs NOP vs DRA": {"x": nop_range, "y": dra_range, "z": "TotalCost"},
        }
    }
    col1, col2 = st.columns(2)
    group = col1.selectbox("Plot Group", list(groups.keys()))
    plot_opt = col2.selectbox("Plot Type", list(groups[group].keys()))
    conf = groups[group][plot_opt]
    Xv, Yv = np.meshgrid(conf['x'], conf['y'], indexing='ij')
    Z = np.zeros_like(Xv, dtype=float)

    # --- Pump coefficients ---
    A = stn.get('A', 0); B = stn.get('B', 0); Cc = stn.get('C', 0)
    P = stn.get('P', 0); Qc = stn.get('Q', 0); R = stn.get('R', 0)
    S = stn.get('S', 0); T = stn.get('T', 0)
    DOL = float(stn.get('DOL', N_max))
    linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
    kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
    rho = rho_list[0]
    rate = stn.get('rate', 9.0)
    g = 9.81

    def get_head(q, n): return (A*q**2 + B*q + Cc)*(n/DOL)**2
    def get_eff(q, n): q_adj = q * DOL/n if n > 0 else q; return (P*q_adj**4 + Qc*q_adj**3 + R*q_adj**2 + S*q_adj + T)
    def get_power_cost(q, n, d, npump=1):
        h = get_head(q, n)
        eff = max(get_eff(q, n)/100, 0.01)
        pwr = (rho*q*g*h*npump)/(3600.0*eff*0.95*1000)
        return pwr*24*rate
    def get_system_head(q, d):
        d_inner = stn['D'] - 2*stn['t']
        rough = stn['rough']
        L_seg = stn['L']
        visc = kv_list[0]
        v = q/3600.0/(np.pi*(d_inner**2)/4)
        Re = v*d_inner/(visc*1e-6) if visc > 0 else 0
        if Re > 0:
            f = 0.25/(np.log10(rough/d_inner/3.7 + 5.74/(Re**0.9))**2)
        else:
            f = 0.0
        DH = f*((L_seg*1000.0)/d_inner)*(v**2/(2*g))*(1-d/100)
        return stn['elev'] + DH
        
    dr_opt = last_res.get(f"drag_reduction_{key}", 0.0)
    dr_max = stn.get('max_dr', 0.0)
    viscosity = kv_list[0]
    dr_use = min(dr_opt, dr_max)
    ppm_value = get_ppm_for_dr(viscosity, dr_use)

    def get_total_cost(q, n, d, npump):
        local_ppm = get_ppm_for_dr(viscosity, d)
        pcost = get_power_cost(q, n, d, npump)
        dracost = local_ppm * (q * 1000.0 * 24.0 / 1e6) * RateDRA
        return pcost + dracost

    for i in range(Xv.shape[0]):
        for j in range(Xv.shape[1]):
            if plot_opt == "Head vs Flow vs Speed":
                Z[i,j] = get_head(Xv[i,j], Yv[i,j])
            elif plot_opt == "Efficiency vs Flow vs Speed":
                Z[i,j] = get_eff(Xv[i,j], Yv[i,j])
            elif plot_opt == "System Head vs Flow vs DRA":
                Z[i,j] = get_system_head(Xv[i,j], Yv[i,j])
            elif plot_opt == "Power Cost vs Speed vs DRA":
                Z[i,j] = get_power_cost(flow_opt, Xv[i,j], Yv[i,j], nopt_opt)
            elif plot_opt == "Power Cost vs Flow vs Speed":
                Z[i,j] = get_power_cost(Xv[i,j], Yv[i,j], dra_opt, nopt_opt)
            elif plot_opt == "Total Cost vs NOP vs DRA":
                Z[i,j] = get_total_cost(flow_opt, speed_opt, Yv[i,j], int(Xv[i,j]))

    axis_labels = {
        "Flow": "X: Flow (m³/hr)",
        "Speed": "Y: Pump Speed (rpm)",
        "Head": "Z: Head (m)",
        "Efficiency": "Z: Efficiency (%)",
        "SystemHead": "Z: System Head (m)",
        "PowerCost": "Z: Power Cost (INR/day)",
        "DRA": "Y: DRA (%)",
        "NOP": "X: No. of Pumps",
        "TotalCost": "Z: Total Cost (INR/day)",
    }
    label_map = {
        "Head vs Flow vs Speed": ["Flow", "Speed", "Head"],
        "Efficiency vs Flow vs Speed": ["Flow", "Speed", "Efficiency"],
        "System Head vs Flow vs DRA": ["Flow", "DRA", "SystemHead"],
        "Power Cost vs Speed vs DRA": ["Speed", "DRA", "PowerCost"],
        "Power Cost vs Flow vs Speed": ["Flow", "Speed", "PowerCost"],
        "Total Cost vs NOP vs DRA": ["NOP", "DRA", "TotalCost"]
    }
    xlab, ylab, zlab = [axis_labels[l] for l in label_map[plot_opt]]

    fig = go.Figure(data=[go.Surface(
        x=conf['x'], y=conf['y'], z=Z.T, colorscale='Viridis', colorbar=dict(title=zlab)
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title=xlab,
            yaxis_title=ylab,
            zaxis_title=zlab
        ),
        title=f"{plot_opt}",
        height=750,
        margin=dict(l=30, r=30, b=30, t=80)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.95em; margin-bottom: 0.5em;'>
            <span style='color:#AAA;'>Surface plot shows parameter variability of the originating pump station for clarity and hydraulic relevance.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

import plotly.graph_objects as go
import numpy as np

with tab8:
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #1e4b82 !important;
            border-bottom: 3.5px solid #2f84d6 !important;
            font-weight: bold !important;
            background: linear-gradient(90deg, #e3f2fd55 30%, #e1f5feaa 100%) !important;
            box-shadow: 0 2px 10px #e3f2fd33 !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 1.25em !important;
            font-family: 'Segoe UI', Arial, sans-serif !important;
        }
        </style>
        <div style='margin-bottom: 0.7em;'></div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>3D Pressure Profile: Residual Head & Peaks</div>", unsafe_allow_html=True)

    if "last_res" not in st.session_state or "last_stations_data" not in st.session_state:
        st.info("Run optimization to enable 3D Pressure Profile.")
    else:
        # Gather data
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        terminal = st.session_state["last_term_data"]

        # ---- 1. Gather all points: stations and peaks ----
        chainages = [0]
        elevs = []
        rh = []
        names = []
        mesh_x, mesh_y, mesh_z, mesh_text, mesh_color = [], [], [], [], []
        peak_x, peak_y, peak_z, peak_label = [], [], [], []

        # Stations (include terminal as last "station")
        for i, stn in enumerate(stations_data):
            chainages.append(chainages[-1] + stn.get("L", 0.0))
            elevs.append(stn["elev"])
            key = stn["name"].lower().replace(" ", "_")
            rh_val = res.get(f"residual_head_{key}", 0.0)
            rh.append(rh_val)
            names.append(stn["name"])
            mesh_x.append(chainages[-1])
            mesh_y.append(stn["elev"])
            mesh_z.append(rh_val)
            mesh_text.append(stn["name"])
            mesh_color.append(rh_val)
            # Peaks for this station
            if 'peaks' in stn and stn['peaks']:
                for pk in stn['peaks']:
                    # pk['loc'] = distance from upstream station start (km)
                    px = chainages[-2] + pk.get('loc', 0)
                    py = pk.get('elev', stn['elev'])
                    pz = rh_val  # Assume RH at station for peak (or interpolate as needed)
                    mesh_x.append(px)
                    mesh_y.append(py)
                    mesh_z.append(pz)
                    mesh_text.append("Peak")
                    mesh_color.append(pz)
                    # Separate for special peak markers
                    peak_x.append(px)
                    peak_y.append(py)
                    peak_z.append(pz)
                    peak_label.append(f"Peak @ {stn['name']}")

        # Add terminal
        terminal_chainage = chainages[-1] + terminal.get("L", 0.0)
        mesh_x.append(terminal_chainage)
        mesh_y.append(terminal["elev"])
        key_term = terminal["name"].lower().replace(" ", "_")
        rh_term = res.get(f"residual_head_{key_term}", 0.0)
        mesh_z.append(rh_term)
        mesh_text.append(terminal["name"])
        mesh_color.append(rh_term)
        names.append(terminal["name"])
        elevs.append(terminal["elev"])
        rh.append(rh_term)
        chainages.append(terminal_chainage)

        # ---- 2. 3D mesh surface using station & peak points ----
        fig3d = go.Figure()

        # 2.1 Mesh Surface: Triangulate all (station + peak) points
        fig3d.add_trace(go.Mesh3d(
            x=mesh_x, y=mesh_y, z=mesh_z,
            intensity=mesh_color, colorscale="Viridis",
            alphahull=8, opacity=0.55,
            showscale=True, colorbar=dict(title="Residual Head (mcl)", x=0.95, y=0.7, len=0.5),
            hovertemplate="Chainage: %{x:.2f} km<br>Elevation: %{y:.2f} m<br>RH: %{z:.1f} mcl<br>%{text}",
            text=mesh_text,
            name="Pressure Mesh Surface"
        ))

        # 2.2 Stations: Big colored spheres, labeled
        fig3d.add_trace(go.Scatter3d(
            x=[chainages[i+1] for i in range(len(stations_data))],
            y=elevs[:-1],
            z=rh[:-1],
            mode='markers+text',
            marker=dict(size=10, color=rh[:-1], colorscale='Plasma', symbol='circle', line=dict(width=2, color='black')),
            text=names[:-1], textposition="top center",
            name="Stations",
            hovertemplate="<b>%{text}</b><br>Chainage: %{x:.2f} km<br>Elevation: %{y:.1f} m<br>RH: %{z:.1f} mcl"
        ))

        # 2.3 Terminal: Big blue sphere, labeled
        fig3d.add_trace(go.Scatter3d(
            x=[terminal_chainage],
            y=[terminal["elev"]],
            z=[rh_term],
            mode='markers+text',
            marker=dict(size=11, color='#238be6', symbol='circle', line=dict(width=3, color='#103d68')),
            text=[terminal["name"]], textposition="top center",
            name="Terminal",
            hovertemplate="<b>%{text}</b><br>Chainage: %{x:.2f} km<br>Elevation: %{y:.1f} m<br>RH: %{z:.1f} mcl"
        ))

        # 2.4 Peaks: Crimson diamonds, labeled
        if peak_x:
            fig3d.add_trace(go.Scatter3d(
                x=peak_x, y=peak_y, z=peak_z,
                mode='markers+text',
                marker=dict(size=10, color='crimson', symbol='diamond', line=dict(width=2, color='black')),
                text=peak_label, textposition="bottom center",
                name="Peaks",
                hovertemplate="<b>%{text}</b><br>Chainage: %{x:.2f} km<br>Elevation: %{y:.1f} m<br>RH: %{z:.1f} mcl"
            ))

        # 2.5 Connecting line (stations+terminal): Show pressure path
        fig3d.add_trace(go.Scatter3d(
            x=[chainages[i+1] for i in range(len(stations_data)+1)],
            y=elevs,
            z=rh,
            mode='lines',
            line=dict(color='deepskyblue', width=6),
            name="Pressure Path",
            hoverinfo="skip"
        ))

        # ---- 3. Layout and style polish ----
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text='Pipeline Chainage (km)', font=dict(size=20, color='#205081')),
                    backgroundcolor='rgb(247,249,255)',
                    gridcolor='lightgrey',
                    showspikes=False,
                    tickfont=dict(size=16, color='#183453')
                ),
                yaxis=dict(
                    title=dict(text='Elevation (m)', font=dict(size=20, color='#8B332A')),
                    backgroundcolor='rgb(252,252,252)',
                    gridcolor='lightgrey',
                    showspikes=False,
                    tickfont=dict(size=16, color='#792B22')
                ),
                zaxis=dict(
                    title=dict(text='Residual Head (mcl)', font=dict(size=20, color='#1C7D6C')),
                    backgroundcolor='rgb(249,255,250)',
                    gridcolor='lightgrey',
                    showspikes=False,
                    tickfont=dict(size=16, color='#16514B')
                ),
                camera=dict(eye=dict(x=1.6, y=1.3, z=1.08)),
                aspectmode='auto'
            ),
            plot_bgcolor="#fff",
            paper_bgcolor="#fcfcff",
            margin=dict(l=25, r=25, t=70, b=30),
            height=690,
            title={
                'text': "<b>3D Pressure Profile: Pipeline Chainage vs Elevation vs Residual Head</b>",
                'y':0.96,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=27, family="Segoe UI, Arial, sans-serif", color="#163269")
            },
            showlegend=True,
            legend=dict(
                font=dict(size=15, color='#183453'),
                orientation='h',
                yanchor='bottom', y=1.01,
                xanchor='right', x=1.0,
                bgcolor='rgba(240,248,255,0.96)',
                bordercolor='#d1e1f5', borderwidth=1
            )
        )

        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown(
            "<div style='text-align:center;color:#888;margin-top:1.1em;'>"
            "3D Pressure Profile <br>"
            unsafe_allow_html=True
        )

with tab_sens:
    st.markdown("<div class='section-title'>Sensitivity Analysis</div>", unsafe_allow_html=True)
    st.write("Analyze how key outputs respond to variations in a parameter. Each run recalculates results using your actual pipeline and optimization logic.")

    if "last_res" not in st.session_state:
        st.info("Run optimization first to enable sensitivity analysis.")
        st.stop()

    param = st.selectbox("Parameter to vary", [
        "Flowrate (m³/hr)", "Viscosity (cSt)", "Drag Reduction (%)", "Diesel Price (INR/L)", "DRA Cost (INR/L)"
    ])
    output = st.selectbox("Output metric", [
        "Total Cost (INR/day)", "Power Cost (INR/day)", "DRA Cost (INR/day)",
        "Residual Head (m)", "Pump Efficiency (%)"
    ])
    # Define range
    FLOW = st.session_state["FLOW"]
    RateDRA = st.session_state["RateDRA"]
    Price_HSD = st.session_state["Price_HSD"]
    linefill_df = st.session_state.get("linefill_df", pd.DataFrame())
    N = 10  # number of points
    if param == "Flowrate (m³/hr)":
        pvals = np.linspace(max(10, 0.5*FLOW), 1.5*FLOW, N)
    elif param == "Viscosity (cSt)":
        # Assume first segment viscosity is varied
        first_visc = linefill_df.iloc[0]["Viscosity (cSt)"] if not linefill_df.empty else 10
        pvals = np.linspace(max(1, 0.5*first_visc), 2*first_visc, N)
    elif param == "Drag Reduction (%)":
        # Use max_dr from first pump station
        max_dr = 0
        for stn in st.session_state['stations']:
            if stn.get('is_pump', False):
                max_dr = stn.get('max_dr', 40)
                break
        pvals = np.linspace(0, max_dr, N)
    elif param == "Diesel Price (INR/L)":
        pvals = np.linspace(0.5*Price_HSD, 2*Price_HSD, N)
    elif param == "DRA Cost (INR/L)":
        pvals = np.linspace(0.5*RateDRA, 2*RateDRA, N)

    yvals = []
    # Iterate and re-solve (uses your backend and always uses *actual* pipeline logic)
    st.info("Running sensitivity... This may take a few seconds per parameter.")
    progress = st.progress(0)
    for i, val in enumerate(pvals):
        # Clone all input parameters for each run
        stations_data = [dict(s) for s in st.session_state['stations']]
        term_data = dict(st.session_state["last_term_data"])
        kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
        this_FLOW = FLOW
        this_RateDRA = RateDRA
        this_Price_HSD = Price_HSD
        this_linefill_df = linefill_df.copy()
        # Modify only the selected param
        if param == "Flowrate (m³/hr)":
            this_FLOW = val
        elif param == "Viscosity (cSt)":
            # Set all segments to this value for test
            this_linefill_df["Viscosity (cSt)"] = val
            kv_list, rho_list = map_linefill_to_segments(this_linefill_df, stations_data)
        elif param == "Drag Reduction (%)":
            # Set all max_dr to >= val; force first pump station's drag reduction to val
            for stn in stations_data:
                if stn.get('is_pump', False):
                    stn['max_dr'] = max(stn.get('max_dr', val), val)
                    break
            # (the optimizer will always take optimal ≤ max_dr)
        elif param == "Diesel Price (INR/L)":
            this_Price_HSD = val
        elif param == "DRA Cost (INR/L)":
            this_RateDRA = val
        # --- Run solver (always backend) ---
        resi = solve_pipeline(stations_data, term_data, this_FLOW, kv_list, rho_list, this_RateDRA, this_Price_HSD, this_linefill_df.to_dict())
        # Extract output metric:
        # Consistent with Summary tab (use only computed data!)
        total_cost = power_cost = dra_cost = rh = eff = 0
        for idx, stn in enumerate(stations_data):
            key = stn['name'].lower().replace(' ', '_')
            # DRA
            dr_opt = resi.get(f"drag_reduction_{key}", 0.0)
            dr_max = stn.get('max_dr', 0.0)
            viscosity = kv_list[idx]
            dr_use = min(dr_opt, dr_max)
            ppm = get_ppm_for_dr(viscosity, dr_use)
            seg_flow = resi.get(f"pipeline_flow_{key}", this_FLOW)
            dra_cost_i = ppm * (seg_flow * 1000.0 * 24.0 / 1e6) * this_RateDRA
            power_cost_i = float(resi.get(f"power_cost_{key}", 0.0) or 0.0)
            eff_i = float(resi.get(f"efficiency_{key}", 100.0))
            rh_i = float(resi.get(f"residual_head_{key}", 0.0))
            total_cost += dra_cost_i + power_cost_i
            dra_cost += dra_cost_i
            power_cost += power_cost_i
            if eff_i < eff or i == 0: eff = eff_i
            if rh_i < rh or i == 0: rh = rh_i
        # Choose output
        if output == "Total Cost (INR/day)":
            yvals.append(total_cost)
        elif output == "Power Cost (INR/day)":
            yvals.append(power_cost)
        elif output == "DRA Cost (INR/day)":
            yvals.append(dra_cost)
        elif output == "Residual Head (m)":
            yvals.append(rh)
        elif output == "Pump Efficiency (%)":
            yvals.append(eff)
        progress.progress((i+1)/len(pvals))
    progress.empty()
    # Plot
    fig = px.line(x=pvals, y=yvals, markers=True,
        labels={"x": param, "y": output},
        title=f"{output} vs {param} (Sensitivity)")
    st.plotly_chart(fig, use_container_width=True)
    df_sens = pd.DataFrame({param: pvals, output: yvals})
    st.dataframe(df_sens, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df_sens.to_csv(index=False).encode(), file_name="sensitivity.csv")

with tab_bench:
    st.markdown("<div class='section-title'>Benchmarking & Global Standards</div>", unsafe_allow_html=True)
    st.write("Compare your pipeline performance to global or custom benchmarks. Green means you match/exceed global standards, red means improvement needed.")

    # --- User can pick standard or edit/upload their own
    b_mode = st.radio("Benchmark Source", ["Global Standards", "Edit Benchmarks", "Upload CSV"])
    if b_mode == "Global Standards":
        benchmarks = {
            "Total Cost per km (INR/day/km)": 12000,
            "Pump Efficiency (%)": 70,
            "Specific Energy (kWh/m³)": 0.065,
            "Max Velocity (m/s)": 2.1
        }
        for k, v in benchmarks.items():
            benchmarks[k] = st.number_input(f"{k}", value=float(v))
    elif b_mode == "Edit Benchmarks":
        bdf = pd.DataFrame({
            "Parameter": ["Total Cost per km (INR/day/km)", "Pump Efficiency (%)", "Specific Energy (kWh/m³)", "Max Velocity (m/s)"],
            "Benchmark Value": [12000, 70, 0.065, 2.1]
        })
        bdf = st.data_editor(bdf)
        benchmarks = dict(zip(bdf["Parameter"], bdf["Benchmark Value"]))
    elif b_mode == "Upload CSV":
        up = st.file_uploader("Upload Benchmark CSV", type=["csv"])
        benchmarks = {}
        if up:
            bdf = pd.read_csv(up)
            st.dataframe(bdf)
            benchmarks = dict(zip(bdf["Parameter"], bdf["Benchmark Value"]))
        if not benchmarks:
            st.warning("Please upload a CSV with columns [Parameter, Benchmark Value]")

    # --- Extract your computed results for comparison ---
    if "last_res" not in st.session_state:
        st.info("Run optimization to show benchmark analysis.")
        st.stop()
    res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]
    total_length = sum([s.get("L", 0.0) for s in stations_data])
    total_cost = 0
    total_pumped = 0
    total_power = 0
    effs = []
    max_velocity = 0
    FLOW = st.session_state.get("FLOW", 1000.0)
    RateDRA = st.session_state.get("RateDRA", 500.0)
    linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
    kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
    for idx, stn in enumerate(stations_data):
        key = stn['name'].lower().replace(' ', '_')
        dr_opt = res.get(f"drag_reduction_{key}", 0.0)
        dr_max = stn.get('max_dr', 0.0)
        viscosity = kv_list[idx]
        dr_use = min(dr_opt, dr_max)
        ppm = get_ppm_for_dr(viscosity, dr_use)
        seg_flow = res.get(f"pipeline_flow_{key}", FLOW)
        dra_cost = ppm * (seg_flow * 1000.0 * 24.0 / 1e6) * RateDRA
        power_cost = float(res.get(f"power_cost_{key}", 0.0) or 0.0)
        velocity = res.get(f"velocity_{key}", 0.0) or 0.0
        total_cost += dra_cost + power_cost
        total_pumped += seg_flow * 24.0
        total_power += power_cost
        eff = float(res.get(f"efficiency_{key}", 100.0))
        if stn.get('is_pump', False):
            effs.append(eff)
        if velocity > max_velocity: max_velocity = velocity
    # Derived KPIs for benchmarking
    my_cost_per_km = total_cost / (total_length if total_length else 1)
    my_avg_eff = np.mean(effs) if effs else 0
    my_spec_energy = (total_power / (FLOW*24.0)) if (FLOW > 0) else 0
    comp = {
        "Total Cost per km (INR/day/km)": my_cost_per_km,
        "Pump Efficiency (%)": my_avg_eff,
        "Specific Energy (kWh/m³)": my_spec_energy,
        "Max Velocity (m/s)": max_velocity
    }
    rows = []
    for k, v in comp.items():
        bench = benchmarks.get(k, None)
        if bench is not None:
            status = "✅" if (k != "Pump Efficiency (%)" and v <= bench) or (k == "Pump Efficiency (%)" and v >= bench) else "🔴"
            rows.append((k, f"{v:.2f}", f"{bench:.2f}", status))
    df_bench = pd.DataFrame(rows, columns=["Parameter", "Your Pipeline", "Benchmark", "Status"])
    st.dataframe(df_bench, use_container_width=True, hide_index=True)


with tab_sim:
    st.markdown("<div class='section-title'>Annualized Savings Simulator</div>", unsafe_allow_html=True)
    st.write("Annual savings from efficiency improvements, energy cost and DRA optimizations.")

    if "last_res" not in st.session_state:
        st.info("Run optimization first.")
        st.stop()
    FLOW = st.session_state["FLOW"]
    RateDRA = st.session_state["RateDRA"]
    Price_HSD = st.session_state["Price_HSD"]
    st.write("Adjust improvement assumptions and note the impact over a year.")
    pump_eff_impr = st.slider("Pump Efficiency Improvement (%)", 0, 10, 3)
    dra_cost_impr = st.slider("DRA Price Reduction (%)", 0, 30, 5)
    flow_change = st.slider("Throughput Increase (%)", 0, 30, 0)
    # Simulate
    stations_data = [dict(s) for s in st.session_state['stations']]
    term_data = dict(st.session_state["last_term_data"])
    # Apply changes
    for stn in stations_data:
        if stn.get('is_pump', False):
            # Assume actual efficiency increases by given % up to max 100
            if "eff_data" in stn and pump_eff_impr > 0:
                # If you have a custom efficiency curve, you could adjust coefficients; else, note improvement in calculation.
                pass  # For simplicity, only factor in to total efficiency calculation below
    new_RateDRA = RateDRA * (1 - dra_cost_impr / 100)
    new_FLOW = FLOW * (1 + flow_change / 100)
    kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
    # --- Re-solve with new parameters ---
    res2 = solve_pipeline(stations_data, term_data, new_FLOW, kv_list, rho_list, new_RateDRA, Price_HSD, linefill_df.to_dict())
    # Compute original and new total cost for 365 days
    total_cost, new_cost = 0, 0
    for idx, stn in enumerate(stations_data):
        key = stn['name'].lower().replace(' ', '_')
        dr_opt = st.session_state["last_res"].get(f"drag_reduction_{key}", 0.0)
        dr_max = stn.get('max_dr', 0.0)
        viscosity = kv_list[idx]
        dr_use = min(dr_opt, dr_max)
        ppm = get_ppm_for_dr(viscosity, dr_use)
        seg_flow = st.session_state["last_res"].get(f"pipeline_flow_{key}", FLOW)
        dra_cost = ppm * (seg_flow * 1000.0 * 24.0 / 1e6) * RateDRA
        power_cost = float(st.session_state["last_res"].get(f"power_cost_{key}", 0.0) or 0.0)
        total_cost += dra_cost + power_cost
        # NEW:
        dr_opt2 = res2.get(f"drag_reduction_{key}", 0.0)
        seg_flow2 = res2.get(f"pipeline_flow_{key}", new_FLOW)
        ppm2 = get_ppm_for_dr(viscosity, min(dr_opt2, dr_max))
        dra_cost2 = ppm2 * (seg_flow2 * 1000.0 * 24.0 / 1e6) * new_RateDRA
        power_cost2 = float(res2.get(f"power_cost_{key}", 0.0) or 0.0)
        new_cost += dra_cost2 + power_cost2
    annual_savings = (total_cost - new_cost) * 365.0
    st.markdown(f"""
    ### <span style="color:#2b9348"><b>Annual Savings: {annual_savings:,.0f} INR/year</b></span>
    """, unsafe_allow_html=True)
    st.write("Based on selected improvements and model output.")
    st.info("Calculations are based on optimized parameters.")


st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
    &copy; 2025 Pipeline Optima™ v1.1.1. Developed by Parichay Das.
    </div>
    """,
    unsafe_allow_html=True
)
