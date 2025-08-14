import os
import sys
from pathlib import Path
import streamlit as st

# --- SAFE DEFAULTS (session state guards) ---
if "stations" not in st.session_state or not isinstance(st.session_state.get("stations"), list):
    st.session_state["stations"] = []
if "FLOW" not in st.session_state:
    st.session_state["FLOW"] = 1000.0
if "op_mode" not in st.session_state:
    st.session_state["op_mode"] = "Flow rate"
if "planner_days" not in st.session_state:
    st.session_state["planner_days"] = 1.0
if "terminal_name" not in st.session_state:
    st.session_state["terminal_name"] = "Terminal"
if "terminal_elev" not in st.session_state:
    st.session_state["terminal_elev"] = 0.0
if "terminal_head" not in st.session_state:
    st.session_state["terminal_head"] = 10.0
if "MOP_kgcm2" not in st.session_state:
    st.session_state["MOP_kgcm2"] = 100.0
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
import hashlib
import uuid
import json
import copy
from plotly.colors import qualitative

# Ensure local modules are importable when the app is run from an arbitrary
# working directory (e.g. `streamlit run path/to/pipeline_optimization_app.py`).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dra_utils import (
    get_ppm_for_dr,
    DRA_CURVE_DATA,
)

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
        if st.button("Login", key="login_btn"):
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
        if st.button("Logout", key="main_logout_btn"):
            st.session_state.authenticated = False
            st.rerun()
check_login()

# ==== 1. EARLY LOAD/RESTORE BLOCK ====
def restore_case_dict(loaded_data):
    """Populate ``st.session_state`` from a saved case dictionary."""

    st.session_state['stations'] = loaded_data.get('stations', [])
    st.session_state['terminal_name'] = loaded_data.get('terminal', {}).get('name', "Terminal")
    st.session_state['terminal_elev'] = loaded_data.get('terminal', {}).get('elev', 0.0)
    st.session_state['terminal_head'] = loaded_data.get('terminal', {}).get('min_residual', 50.0)
    st.session_state['FLOW'] = loaded_data.get('FLOW', 1000.0)
    st.session_state['RateDRA'] = loaded_data.get('RateDRA', 500.0)
    st.session_state['Price_HSD'] = loaded_data.get('Price_HSD', 70.0)
    st.session_state['op_mode'] = loaded_data.get('op_mode', "Flow rate")
    if loaded_data.get("linefill_vol"):
        st.session_state["linefill_vol_df"] = pd.DataFrame(loaded_data["linefill_vol"])
    if loaded_data.get("day_plan"):
        st.session_state["day_plan_df"] = pd.DataFrame(loaded_data["day_plan"])
    if loaded_data.get("proj_flow"):
        df_flow = pd.DataFrame(loaded_data["proj_flow"])
        for col in ["Start", "End"]:
            if col in df_flow.columns:
                df_flow[col] = pd.to_datetime(df_flow[col])
        st.session_state["proj_flow_df"] = df_flow
    if loaded_data.get("proj_plan"):
        df_proj = pd.DataFrame(loaded_data["proj_plan"])
        cols_to_drop = [c for c in ["Start", "End", "Flow", "Flow (m³/h)"] if c in df_proj.columns]
        if cols_to_drop:
            df_proj = df_proj.drop(columns=cols_to_drop)
        st.session_state["proj_plan_df"] = df_proj
    if loaded_data.get("planner_days"):
        st.session_state["planner_days"] = loaded_data["planner_days"]
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
    # Handle pump type data for originating station
    headA = loaded_data.get("head_data_1A", None)
    effA = loaded_data.get("eff_data_1A", None)
    headB = loaded_data.get("head_data_1B", None)
    effB = loaded_data.get("eff_data_1B", None)
    if headA is not None:
        st.session_state["head_data_1A"] = pd.DataFrame(headA)
    if effA is not None:
        st.session_state["eff_data_1A"] = pd.DataFrame(effA)
    if headB is not None:
        st.session_state["head_data_1B"] = pd.DataFrame(headB)
    if effB is not None:
        st.session_state["eff_data_1B"] = pd.DataFrame(effB)

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
        MOP_val   = st.number_input("MOP (kg/cm²)", value=st.session_state.get("MOP_kgcm2", 100.0), step=1.0)
        st.session_state["FLOW"] = FLOW
        st.session_state["RateDRA"] = RateDRA
        st.session_state["Price_HSD"] = Price_HSD
        st.session_state["MOP_kgcm2"] = MOP_val

    st.subheader("Operating Mode")
    if "linefill_df" not in st.session_state:
        st.session_state["linefill_df"] = pd.DataFrame({
            "Start (km)": [0.0],
            "End (km)": [100.0],
            "Viscosity (cSt)": [10.0],
            "Density (kg/m³)": [850.0]
        })
    input_modes = ["Flow rate", "Daily Pumping Schedule", "Pumping planner"]
    if st.session_state.get("op_mode") not in input_modes:
        st.session_state["op_mode"] = input_modes[0]
    mode = st.radio(
        "Choose input mode",
        input_modes,
        horizontal=True,
        key="op_mode",
    )

    if mode == "Flow rate":
        # Flow rate is already captured as FLOW above.
        st.markdown("**Linefill at 07:00 Hrs (Volumetric)**")
        if "linefill_vol_df" not in st.session_state:
            st.session_state["linefill_vol_df"] = pd.DataFrame({
                "Product": ["Product-1"],
                "Volume (m³)": [50000.0],
                "Viscosity (cSt)": [5.0],
                "Density (kg/m³)": [810.0],
            })
        lf_df = st.data_editor(
            st.session_state["linefill_vol_df"],
            num_rows="dynamic",
            key="linefill_vol_editor",
        )
        st.session_state["linefill_vol_df"] = lf_df
    elif mode == "Daily Pumping Schedule":
        st.markdown("**Linefill at 07:00 Hrs (Volumetric)**")
        if "linefill_vol_df" not in st.session_state:
            st.session_state["linefill_vol_df"] = pd.DataFrame({
                "Product": ["Product-1", "Product-2", "Product-3"],
                "Volume (m³)": [50000.0, 40000.0, 15000.0],
                "Viscosity (cSt)": [5.0, 12.0, 15.0],
                "Density (kg/m³)": [810.0, 825.0, 865.0],
            })
        lf_df = st.data_editor(
            st.session_state["linefill_vol_df"],
            num_rows="dynamic",
            key="linefill_vol_editor",
        )
        st.session_state["linefill_vol_df"] = lf_df
        st.markdown("**Pumping Plan for the Day (Order of Pumping)**")
        if "day_plan_df" not in st.session_state:
            st.session_state["day_plan_df"] = pd.DataFrame({
                "Product": ["Product-4", "Product-5", "Product-6", "Product-7"],
                "Volume (m³)": [12000.0, 6000.0, 10000.0, 8000.0],
                "Viscosity (cSt)": [3.0, 10.0, 15.0, 4.0],
                "Density (kg/m³)": [800.0, 840.0, 880.0, 770.0],
            })
        day_df = st.data_editor(
            st.session_state["day_plan_df"],
            num_rows="dynamic",
            key="day_plan_editor",
        )
        st.session_state["day_plan_df"] = day_df
    else:
        st.markdown("**Linefill at 07:00 Hrs (Volumetric)**")
        if "linefill_vol_df" not in st.session_state:
            st.session_state["linefill_vol_df"] = pd.DataFrame({
                "Product": ["Product-1", "Product-2", "Product-3"],
                "Volume (m³)": [50000.0, 40000.0, 15000.0],
                "Viscosity (cSt)": [5.0, 12.0, 15.0],
                "Density (kg/m³)": [810.0, 825.0, 865.0],
            })
        lf_df = st.data_editor(
            st.session_state["linefill_vol_df"],
            num_rows="dynamic",
            key="linefill_vol_editor",
        )
        st.session_state["linefill_vol_df"] = lf_df
        st.session_state["planner_days"] = st.number_input(
            "Number of days in Projected Pumping Plan",
            min_value=1.0,
            step=1.0,
            value=float(st.session_state.get("planner_days", 1.0)),
        )
        st.markdown("**Projected Flow Schedule**")
        if "proj_flow_df" not in st.session_state:
            now = pd.Timestamp.now().floor("H")
            st.session_state["proj_flow_df"] = pd.DataFrame({
                "Start": [now],
                "End": [now + pd.Timedelta(hours=24)],
                "Flow (m³/h)": [1000.0],
            })
        flow_df = st.data_editor(
            st.session_state["proj_flow_df"],
            num_rows="dynamic",
            key="proj_flow_editor",
            column_config={
                "Start": st.column_config.DatetimeColumn("Start", format="DD/MM/YY HH:mm"),
                "End": st.column_config.DatetimeColumn("End", format="DD/MM/YY HH:mm"),
                "Flow (m³/h)": st.column_config.NumberColumn("Flow (m³/h)", format="%.2f"),
            },
        )
        st.session_state["proj_flow_df"] = flow_df
        st.markdown("**Projected Pumping Plan (Order of Pumping)**")
        if "proj_plan_df" not in st.session_state:
            st.session_state["proj_plan_df"] = pd.DataFrame({
                "Product": ["Product-4", "Product-5"],
                "Volume (m³)": [12000.0, 8000.0],
                "Viscosity (cSt)": [3.0, 10.0],
                "Density (kg/m³)": [800.0, 840.0],
            })
        else:
            st.session_state["proj_plan_df"] = st.session_state["proj_plan_df"].drop(
                columns=[c for c in ["Start", "End", "Flow", "Flow (m³/h)"] if c in st.session_state["proj_plan_df"].columns],
                errors="ignore",
            )
        proj_df = st.data_editor(
            st.session_state["proj_plan_df"],
            num_rows="dynamic",
            key="proj_plan_editor",
            column_config={
                "Product": st.column_config.TextColumn("Product"),
                "Volume (m³)": st.column_config.NumberColumn("Volume (m³)", format="%.2f"),
                "Viscosity (cSt)": st.column_config.NumberColumn("Viscosity (cSt)", format="%.2f"),
                "Density (kg/m³)": st.column_config.NumberColumn("Density (kg/m³)", format="%.2f"),
            },
        )
        proj_df = proj_df.drop(
            columns=[c for c in ["Start", "End", "Flow", "Flow (m³/h)"] if c in proj_df.columns],
            errors="ignore",
        )
        st.session_state["proj_plan_df"] = proj_df


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

st.subheader("Stations")
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
with st.sidebar:
    st.markdown("### Stations")
    add_col, rem_col = st.columns(2)
    if add_col.button("➕ Add Station", key="add_station"):
        n = len(st.session_state.get('stations', [])) + 1
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
    if rem_col.button("🗑️ Remove Station", key="rem_station"):
        if st.session_state.get('stations'):
            st.session_state.stations.pop()


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
            stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.7f", step=0.0000001, key=f"rough{idx}")
        with col3:
            stn['max_pumps'] = st.number_input("Max Pumps available", min_value=1, value=stn.get('max_pumps',1), step=1, key=f"mpumps{idx}")
            stn['delivery'] = st.number_input("Delivery (m³/hr)", value=stn.get('delivery', 0.0), key=f"deliv{idx}")
            stn['supply'] = st.number_input("Supply (m³/hr)", value=stn.get('supply', 0.0), key=f"sup{idx}")

        tabs = st.tabs(["Pump", "Peaks"])
        with tabs[0]:
            if stn['is_pump']:
                if idx == 1:
                    stn.setdefault('pump_types', {})
                    pump_tabs = st.tabs(["Type A", "Type B"])
                    for tab_idx, ptype in enumerate(['A', 'B']):
                        with pump_tabs[tab_idx]:
                            pdata = stn.get('pump_types', {}).get(ptype, {})
                            enabled = st.checkbox(
                                f"Use Pump Type {ptype}",
                                value=pdata.get('available', 0) > 0,
                                key=f"enable{idx}{ptype}"
                            )
                            avail = st.number_input(
                                "Available Pumps",
                                min_value=0,
                                max_value=2,
                                step=1,
                                value=int(pdata.get('available', 0)),
                                key=f"avail{idx}{ptype}"
                            )
                            if not enabled or avail == 0:
                                st.info("Pump type disabled")
                                stn.setdefault('pump_types', {})[ptype] = {'available': 0}
                                continue

                            names = pdata.get('names', [])
                            if len(names) < avail:
                                names += [f'Pump {ptype} {i+1}' for i in range(len(names), avail)]
                            for j in range(avail):
                                names[j] = st.text_input(
                                    f"Pump {ptype} {j+1} Name",
                                    value=names[j],
                                    key=f"pname{idx}{ptype}{j}"
                                )

                            key_head = f"head_data_{idx}{ptype}"
                            if key_head not in st.session_state or not isinstance(st.session_state[key_head], pd.DataFrame):
                                st.session_state[key_head] = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
                            df_head = st.data_editor(
                                st.session_state[key_head],
                                num_rows="dynamic",
                                key=f"{key_head}_editor",
                            )
                            st.session_state[key_head] = df_head

                            key_eff = f"eff_data_{idx}{ptype}"
                            if key_eff not in st.session_state or not isinstance(st.session_state[key_eff], pd.DataFrame):
                                st.session_state[key_eff] = pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})
                            df_eff = st.data_editor(
                                st.session_state[key_eff],
                                num_rows="dynamic",
                                key=f"{key_eff}_editor",
                            )
                            st.session_state[key_eff] = df_eff

                            pcol1, pcol2, pcol3 = st.columns(3)
                            with pcol1:
                                ptype_sel = st.selectbox(
                                    "Power Source", ["Grid", "Diesel"],
                                    index=0 if pdata.get('power_type', 'Grid') == "Grid" else 1,
                                    key=f"ptype{idx}{ptype}"
                                )
                            with pcol2:
                                minrpm = st.number_input("Min RPM", value=pdata.get('MinRPM', 1000.0), key=f"minrpm{idx}{ptype}")
                                dol = st.number_input("Rated RPM", value=pdata.get('DOL', 1500.0), key=f"dol{idx}{ptype}")
                            with pcol3:
                                if ptype_sel == "Grid":
                                    rate = st.number_input("Elec Rate (INR/kWh)", value=pdata.get('rate', 9.0), key=f"rate{idx}{ptype}")
                                    sfc = 0.0
                                else:
                                    sfc = st.number_input("SFC (gm/bhp·hr)", value=pdata.get('sfc', 150.0), key=f"sfc{idx}{ptype}")
                                    rate = 0.0

                            stn.setdefault('pump_types', {})[ptype] = {
                                'names': names,
                                'name': names[0] if names else f'Pump {ptype}',
                                'head_data': df_head,
                                'eff_data': df_eff,
                                'power_type': ptype_sel,
                                'MinRPM': minrpm,
                                'DOL': dol,
                                'rate': rate,
                                'sfc': sfc,
                                'available': avail
                            }
                    # Aggregate all pump names for station-level display
                    all_names = []
                    for pdata in stn.get('pump_types', {}).values():
                        avail = int(pdata.get('available', 0))
                        pn = pdata.get('names', [])
                        if len(pn) < avail:
                            pn += [f"Pump {len(all_names)+i+1}" for i in range(avail - len(pn))]
                        all_names.extend(pn[:avail])
                    stn['pump_names'] = all_names
                    stn['pump_name'] = all_names[0] if all_names else ''
                else:
                    names = stn.get('pump_names', [])
                    if len(names) < stn['max_pumps']:
                        names += [f'Pump {idx}-{i+1}' for i in range(len(names), stn['max_pumps'])]
                    for j in range(stn['max_pumps']):
                        names[j] = st.text_input(
                            f"Pump {j+1} Name",
                            value=names[j],
                            key=f"pname{idx}_{j}"
                        )
                    stn['pump_names'] = names
                    stn['pump_name'] = names[0] if names else ''

                    key_head = f"head_data_{idx}"
                    if key_head not in st.session_state or not isinstance(st.session_state[key_head], pd.DataFrame):
                        st.session_state[key_head] = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
                    df_head = st.data_editor(
                        st.session_state[key_head],
                        num_rows="dynamic",
                        key=f"{key_head}_editor",
                    )
                    st.session_state[key_head] = df_head

                    key_eff = f"eff_data_{idx}"
                    if key_eff not in st.session_state or not isinstance(st.session_state[key_eff], pd.DataFrame):
                        st.session_state[key_eff] = pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})
                    df_eff = st.data_editor(
                        st.session_state[key_eff],
                        num_rows="dynamic",
                        key=f"{key_eff}_editor",
                    )
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
            if key_peak not in st.session_state or not isinstance(st.session_state[key_peak], pd.DataFrame):
                st.session_state[key_peak] = pd.DataFrame({"Location (km)": [stn['L']/2.0], "Elevation (m)": [stn['elev']+100.0]})
            peak_df = st.data_editor(
                st.session_state[key_peak],
                num_rows="dynamic",
                key=f"{key_peak}_editor",
            )
            st.session_state[key_peak] = peak_df

st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value=st.session_state.get("terminal_name","Terminal"), key="terminal_name")
terminal_elev = st.number_input("Elevation (m)", value=st.session_state.get("terminal_elev",0.0), step=0.1, key="terminal_elev")
terminal_head = st.number_input("Minimum Residual Head (m)", value=st.session_state.get("terminal_head",50.0), step=1.0, key="terminal_head")

def get_full_case_dict():
    """Collect the complete case description from ``st.session_state``."""

    import numpy as np
    import pandas as pd

    for idx, stn in enumerate(st.session_state.get('stations', []), start=1):
        if stn.get('is_pump', False):
            if idx == 1 and 'pump_types' in stn:
                for ptype in ['A', 'B']:
                    pdata = stn['pump_types'].get(ptype, {})
                    dfh = st.session_state.get(f"head_data_{idx}{ptype}")
                    dfe = st.session_state.get(f"eff_data_{idx}{ptype}")
                    if dfh is None and pdata.get('head_data') is not None:
                        dfh = pd.DataFrame(pdata['head_data'])
                    if dfe is None and pdata.get('eff_data') is not None:
                        dfe = pd.DataFrame(pdata['eff_data'])
                    if dfh is not None and len(dfh) >= 3:
                        Qh = dfh.iloc[:, 0].values
                        Hh = dfh.iloc[:, 1].values
                        coeff = np.polyfit(Qh, Hh, 2)
                        pdata['A'], pdata['B'], pdata['C'] = float(coeff[0]), float(coeff[1]), float(coeff[2])
                    if dfe is not None and len(dfe) >= 5:
                        Qe = dfe.iloc[:, 0].values
                        Ee = dfe.iloc[:, 1].values
                        coeff_e = np.polyfit(Qe, Ee, 4)
                        pdata['P'], pdata['Q'], pdata['R'], pdata['S'], pdata['T'] = [float(c) for c in coeff_e]
                    pdata['head_data'] = dfh.to_dict(orient="records") if isinstance(dfh, pd.DataFrame) else None
                    pdata['eff_data'] = dfe.to_dict(orient="records") if isinstance(dfe, pd.DataFrame) else None
                    pdata['available'] = pdata.get('available', 0)
                    stn['pump_types'][ptype] = pdata
            else:
                dfh = st.session_state.get(f"head_data_{idx}")
                dfe = st.session_state.get(f"eff_data_{idx}")
                if dfh is None and "head_data" in stn:
                    dfh = pd.DataFrame(stn["head_data"])
                if dfe is None and "eff_data" in stn:
                    dfe = pd.DataFrame(stn["eff_data"])
                if dfh is not None and len(dfh) >= 3:
                    Qh = dfh.iloc[:, 0].values
                    Hh = dfh.iloc[:, 1].values
                    coeff = np.polyfit(Qh, Hh, 2)
                    stn['A'], stn['B'], stn['C'] = float(coeff[0]), float(coeff[1]), float(coeff[2])
                if dfe is not None and len(dfe) >= 5:
                    Qe = dfe.iloc[:, 0].values
                    Ee = dfe.iloc[:, 1].values
                    coeff_e = np.polyfit(Qe, Ee, 4)
                    stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = [float(c) for c in coeff_e]

    flow_df = st.session_state.get('proj_flow_df', pd.DataFrame())
    if isinstance(flow_df, pd.DataFrame) and len(flow_df):
        flow_df = flow_df.copy()
        for c in ["Start", "End"]:
            if c in flow_df.columns:
                flow_df[c] = pd.to_datetime(flow_df[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
        proj_flow = flow_df.to_dict(orient="records")
    else:
        proj_flow = []

    plan_df = st.session_state.get('proj_plan_df', pd.DataFrame())
    if isinstance(plan_df, pd.DataFrame) and len(plan_df):
        plan_df = plan_df.drop(
            columns=[c for c in ["Start", "End", "Flow", "Flow (m³/h)"] if c in plan_df.columns],
            errors="ignore",
        )
        proj_plan = plan_df.to_dict(orient="records")
    else:
        proj_plan = []

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
        "op_mode": st.session_state.get('op_mode', "Flow rate"),
        "linefill": st.session_state.get('linefill_df', pd.DataFrame()).to_dict(orient="records"),
        "linefill_vol": st.session_state.get('linefill_vol_df', pd.DataFrame()).to_dict(orient="records"),
        "day_plan": st.session_state.get('day_plan_df', pd.DataFrame()).to_dict(orient="records"),
        "proj_flow": proj_flow,
        "proj_plan": proj_plan,
        "planner_days": st.session_state.get('planner_days', 1.0),
        **{
            f"head_data_{i+1}": (
                st.session_state.get(f"head_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"head_data_{i+1}"), pd.DataFrame) else None
            )
            for i in range(len(st.session_state.get('stations', [])))
        },
        **{
            f"head_data_{1}{ptype}": (
                st.session_state.get(f"head_data_{1}{ptype}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"head_data_{1}{ptype}"), pd.DataFrame) else None
            )
            for ptype in ['A', 'B']
        },
        **{
            f"eff_data_{i+1}": (
                st.session_state.get(f"eff_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"eff_data_{i+1}"), pd.DataFrame) else None
            )
            for i in range(len(st.session_state.get('stations', [])))
        },
        **{
            f"eff_data_{1}{ptype}": (
                st.session_state.get(f"eff_data_{1}{ptype}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"eff_data_{1}{ptype}"), pd.DataFrame) else None
            )
            for ptype in ['A', 'B']
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
    """Map linefill properties onto each pipeline segment.

    Accepts either a tabular linefill with "Start/End (km)" columns or a
    volumetric table containing "Volume (m³)" information. In the latter case,
    :func:`map_vol_linefill_to_segments` is used to derive segment properties.
    """

    if linefill_df is None or len(linefill_df) == 0:
        return [0.0] * len(stations), [0.0] * len(stations)

    cols = set(linefill_df.columns)

    # If Start/End columns are missing but volumetric info is present,
    # delegate to volumetric mapper and return directly.
    if "Start (km)" not in cols or "End (km)" not in cols:
        if "Volume (m³)" in cols or "Volume" in cols:
            return map_vol_linefill_to_segments(linefill_df, stations)
        # Fallback: assume uniform properties from the last row
        kv = float(linefill_df.iloc[-1].get("Viscosity (cSt)", 0.0))
        rho = float(linefill_df.iloc[-1].get("Density (kg/m³)", 0.0))
        return [kv] * len(stations), [rho] * len(stations)

    cumlen = [0]
    for stn in stations:
        cumlen.append(cumlen[-1] + stn["L"])
    viscs = []
    dens = []
    for i in range(len(stations)):
        seg_start = cumlen[i]
        seg_end = cumlen[i + 1]
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

# ==== NEW: Volumetric linefill helpers ====
def pipe_cross_section_area_m2(stations: list[dict]) -> float:
    """Return pipe internal cross-sectional area (m²) using the first station's D/t."""
    if not stations:
        return 0.0
    D = float(stations[0].get("D", 0.711))
    t = float(stations[0].get("t", 0.007))
    d_inner = max(D - 2.0*t, 0.0)
    return float((pi * d_inner**2) / 4.0)

def map_vol_linefill_to_segments(vol_table: pd.DataFrame, stations: list[dict]) -> tuple[list[float], list[float]]:
    """Convert a volumetric linefill table [Volume (m3), Visc, Density] to segment KV/rho.

    Assumes uniform diameter along the pipeline (uses first station D & t).
    """
    A = pipe_cross_section_area_m2(stations)
    if A <= 0:
        raise ValueError("Invalid pipe area (check D and t).")

    # Compute lengths occupied by each batch
    # Expected columns: Product, Volume (m³), Viscosity (cSt), Density (kg/m³)
    batches = []
    for _, r in vol_table.iterrows():
        vol = float(r.get("Volume (m³)", 0.0) or r.get("Volume", 0.0) or 0.0)
        if vol <= 0:
            continue
        length_km = (vol / A) / 1000.0  # m / 1000 => km
        visc = float(r.get("Viscosity (cSt)", 0.0))
        dens = float(r.get("Density (kg/m³)", 0.0))
        batches.append({"len_km": length_km, "kv": visc, "rho": dens})

    # Map to segments (each station defines a segment length L)
    seg_kv, seg_rho = [], []
    seg_lengths = [s.get("L", 0.0) for s in stations]
    i_batch = 0
    remaining = batches[0]["len_km"] if batches else 0.0
    kv_cur = batches[0]["kv"] if batches else 0.0
    rho_cur = batches[0]["rho"] if batches else 0.0

    for L in seg_lengths:
        need = L
        # Consume from batches until we cover this segment upstream-to-downstream
        while need > 1e-9:
            if remaining <= 1e-9:
                i_batch += 1
                if i_batch >= len(batches):
                    # If we ran out, extend with last known properties
                    remaining = need
                    # kv_cur, rho_cur unchanged
                else:
                    remaining = batches[i_batch]["len_km"]
                    kv_cur = batches[i_batch]["kv"]
                    rho_cur = batches[i_batch]["rho"]
            take = min(need, remaining)
            # For per-segment properties, use the property of the upstream-most fluid in the segment.
            # (Piecewise mixing could be done, but you asked to keep other logic unchanged.)
            # So we only need the first batch's kv/rho per segment.
            if need == L:
                seg_kv.append(kv_cur)
                seg_rho.append(rho_cur)
            need -= take
            remaining -= take

    return seg_kv, seg_rho


def shift_vol_linefill(
    vol_table: pd.DataFrame,
    pumped_m3: float,
    day_plan: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Update ``vol_table`` after ``pumped_m3`` m³ has left the pipeline.

    Fluid is removed from the terminal end of ``vol_table`` and the same volume
    is injected at the origin from ``day_plan`` if provided.  The updated
    ``vol_table`` and the (possibly shortened) ``day_plan`` are returned.
    """

    # Remove delivered volume from downstream end
    vol_table = vol_table.copy()
    vol_table["Volume (m³)"] = vol_table["Volume (m³)"].astype(float)
    remaining = pumped_m3
    idx = len(vol_table) - 1
    while remaining > 1e-9 and idx >= 0:
        v = vol_table.at[idx, "Volume (m³)"]
        take = min(v, remaining)
        vol_table.at[idx, "Volume (m³)"] = v - take
        remaining -= take
        if vol_table.at[idx, "Volume (m³)"] <= 1e-9:
            vol_table = vol_table.drop(index=idx)
        idx -= 1
    vol_table = vol_table.reset_index(drop=True)

    # Inject new product at upstream end according to day plan
    if day_plan is not None:
        day_plan = day_plan.copy()
        day_plan["Volume (m³)"] = day_plan["Volume (m³)"].astype(float)
        added = pumped_m3
        j = 0
        while added > 1e-9 and j < len(day_plan):
            v = day_plan.at[j, "Volume (m³)"]
            take = min(v, added)
            batch = {
                "Product": day_plan.at[j, "Product"],
                "Volume (m³)": take,
                "Viscosity (cSt)": day_plan.at[j, "Viscosity (cSt)"],
                "Density (kg/m³)": day_plan.at[j, "Density (kg/m³)"],
            }
            vol_table = pd.concat([pd.DataFrame([batch]), vol_table], ignore_index=True)
            day_plan.at[j, "Volume (m³)"] = v - take
            added -= take
            if day_plan.at[j, "Volume (m³)"] <= 1e-9:
                j += 1
        day_plan = day_plan.iloc[j:].reset_index(drop=True)

    return vol_table, day_plan


# Build a summary dataframe from solver results
def build_summary_dataframe(res: dict, stations_data: list[dict], linefill_df: pd.DataFrame | None, drop_unused: bool = True) -> pd.DataFrame:
    """Create station-wise summary table matching the Optimization Results view."""

    if linefill_df is not None and len(linefill_df):
        if "Start (km)" in linefill_df.columns:
            kv_list, _ = map_linefill_to_segments(linefill_df, stations_data)
        else:
            kv_list, _ = map_vol_linefill_to_segments(linefill_df, stations_data)
    else:
        kv_list = [0.0] * len(stations_data)

    names = [s['name'] for s in stations_data]
    keys = [n.lower().replace(' ', '_') for n in names]

    station_ppm = {}
    for idx, stn in enumerate(stations_data):
        key = keys[idx]
        dr_opt = res.get(f"drag_reduction_{key}", 0.0)
        dr_max = stn.get('max_dr', 0.0)
        viscosity = kv_list[idx] if idx < len(kv_list) else 0.0
        dr_use = min(dr_opt, dr_max)
        station_ppm[key] = get_ppm_for_dr(viscosity, dr_use)

    segment_flows = [res.get(f"pipeline_flow_{k}", np.nan) for k in keys]
    pump_flows = [res.get(f"pump_flow_{k}", np.nan) for k in keys]

    params = [
        "Pipeline Flow (m³/hr)", "Pump Flow (m³/hr)", "Power & Fuel Cost (INR)", "DRA Cost (INR)",
        "DRA PPM", "No. of Pumps", "Pump Speed (rpm)", "Pump Eff (%)", "Pump BKW (kW)",
        "Motor Input (kW)", "Reynolds No.", "Head Loss (m)", "Head Loss (kg/cm²)", "Vel (m/s)",
        "Residual Head (m)", "Residual Head (kg/cm²)", "SDH (m)", "SDH (kg/cm²)",
        "MAOP (m)", "MAOP (kg/cm²)", "Drag Reduction (%)"
    ]
    summary = {"Parameters": params}

    for idx, nm in enumerate(names):
        key = keys[idx]
        summary[nm] = [
            segment_flows[idx],
            pump_flows[idx] if idx < len(pump_flows) and not pd.isna(pump_flows[idx]) else np.nan,
            res.get(f"power_cost_{key}", 0.0),
            res.get(f"dra_cost_{key}", 0.0),
            station_ppm.get(key, np.nan),
            int(res.get(f"num_pumps_{key}", 0)),
            res.get(f"speed_{key}", 0.0),
            res.get(f"efficiency_{key}", 0.0),
            res.get(f"pump_bkw_{key}", 0.0),
            res.get(f"motor_kw_{key}", 0.0),
            res.get(f"reynolds_{key}", 0.0),
            res.get(f"head_loss_{key}", 0.0),
            res.get(f"head_loss_kgcm2_{key}", 0.0),
            res.get(f"velocity_{key}", 0.0),
            res.get(f"residual_head_{key}", 0.0),
            res.get(f"rh_kgcm2_{key}", 0.0),
            res.get(f"sdh_{key}", 0.0),
            res.get(f"sdh_kgcm2_{key}", 0.0),
            res.get(f"maop_{key}", 0.0),
            res.get(f"maop_kgcm2_{key}", 0.0),
            res.get(f"drag_reduction_{key}", 0.0),
        ]

    df_sum = pd.DataFrame(summary)
    if drop_unused:
        drop_cols = []
        for stn in stations_data:
            key = stn['name'].lower().replace(' ', '_')
            if stn.get('is_pump', False):
                if int(res.get(f"num_pumps_{key}", 0)) == 0 and float(res.get(f"drag_reduction_{key}", 0)) == 0:
                    drop_cols.append(stn['name'])
        if drop_cols:
            df_sum.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df_sum.round(2)


def build_station_table(res: dict, base_stations: list[dict]) -> pd.DataFrame:
    """Return per-station details used in the daily schedule table.

    The function iterates over the stations used in the optimisation (including
    individual pump units at the origin) and pulls the corresponding values from
    ``res``.  No aggregation is performed so the hydraulic linkage between
    pumps and stations (RH -> SDH propagation) is preserved.
    """

    rows: list[dict] = []
    stations_seq = res.get('stations_used') or base_stations
    origin_name = base_stations[0]['name'] if base_stations else ''
    base_map = {s['name']: s for s in base_stations}

    for idx, stn in enumerate(stations_seq):
        name = stn['name'] if isinstance(stn, dict) else str(stn)
        key = name.lower().replace(' ', '_')
        if f"pipeline_flow_{key}" not in res:
            continue

        station_display = stn.get('orig_name', stn.get('name', name)) if isinstance(stn, dict) else name
        base_stn = base_map.get(stn.get('orig_name', name) if isinstance(stn, dict) else name, {})
        pump_list = None
        if isinstance(stn, dict):
            pump_list = stn.get('pump_names') or base_stn.get('pump_names')
        else:
            pump_list = base_stn.get('pump_names') if base_stn else None
        n_pumps = int(res.get(f"num_pumps_{key}", 0) or 0)
        if pump_list and n_pumps > 0:
            pump_name = ", ".join(pump_list[:n_pumps])
        else:
            pump_name = (stn.get('pump_name') if isinstance(stn, dict) else '') or base_stn.get('pump_name', '')

        if origin_name and name != origin_name and name.startswith(origin_name):
            station_display = origin_name

        row = {
            'Station': station_display,
            'Pump Name': pump_name,
            'Pipeline Flow (m³/hr)': float(res.get(f"pipeline_flow_{key}", 0.0) or 0.0),
            'Pump Flow (m³/hr)': float(res.get(f"pump_flow_{key}", 0.0) or 0.0),
            'Power & Fuel Cost (INR)': float(res.get(f"power_cost_{key}", 0.0) or 0.0),
            'DRA Cost (INR)': float(res.get(f"dra_cost_{key}", 0.0) or 0.0),
            'DRA PPM': float(res.get(f"dra_ppm_{key}", 0.0) or 0.0),
            'No. of Pumps': n_pumps,
            'Pump Speed (rpm)': float(res.get(f"speed_{key}", 0.0) or 0.0),
            'Pump Eff (%)': float(res.get(f"efficiency_{key}", 0.0) or 0.0),
            'Pump BKW (kW)': float(res.get(f"pump_bkw_{key}", 0.0) or 0.0),
            'Motor Input (kW)': float(res.get(f"motor_kw_{key}", 0.0) or 0.0),
            'Reynolds No.': float(res.get(f"reynolds_{key}", 0.0) or 0.0),
            'Head Loss (m)': float(res.get(f"head_loss_{key}", 0.0) or 0.0),
            'Head Loss (kg/cm²)': float(res.get(f"head_loss_kgcm2_{key}", 0.0) or 0.0),
            'Vel (m/s)': float(res.get(f"velocity_{key}", 0.0) or 0.0),
            'Residual Head (m)': float(res.get(f"residual_head_{key}", 0.0) or 0.0),
            'Residual Head (kg/cm²)': float(res.get(f"rh_kgcm2_{key}", 0.0) or 0.0),
            'SDH (m)': float(res.get(f"sdh_{key}", 0.0) or 0.0),
            'SDH (kg/cm²)': float(res.get(f"sdh_kgcm2_{key}", 0.0) or 0.0),
            'MAOP (m)': float(res.get(f"maop_{key}", 0.0) or 0.0),
            'MAOP (kg/cm²)': float(res.get(f"maop_kgcm2_{key}", 0.0) or 0.0),
            'Drag Reduction (%)': float(res.get(f"drag_reduction_{key}", 0.0) or 0.0),
        }

        row['Total Cost (INR)'] = row['Power & Fuel Cost (INR)'] + row['DRA Cost (INR)']

        # Available suction head only needs to be reported at the origin suction
        if idx == 0:
            row['Available Suction Head (m)'] = row['Residual Head (m)']
            row['Available Suction Head (kg/cm²)'] = row['Residual Head (kg/cm²)']
        else:
            row['Available Suction Head (m)'] = np.nan
            row['Available Suction Head (kg/cm²)'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.round(2)

# Persisted DRA lock from 07:00 run
def lock_dra_in_stations_from_result(stations: list[dict], res: dict, kv_list: list[float]) -> list[dict]:
    """Freeze per-station DRA (as %DR) based on ppm chosen at 07:00 for each station.

    Uses inverse interpolation to compute %DR that corresponds to the chosen PPM at the station's viscosity.
    """
    from dra_utils import get_dr_for_ppm
    new_stations = []
    for idx, stn in enumerate(stations, start=1):
        key = stn['name'].lower().replace(' ', '_')
        ppm = float(res.get(f"dra_ppm_{key}", 0.0) or 0.0)
        stn2 = dict(stn)
        if ppm > 0.0:
            kv = float(kv_list[idx-1] if idx-1 < len(kv_list) else kv_list[-1])
            dr_fixed = get_dr_for_ppm(kv, ppm)
            stn2['fixed_dra_perc'] = float(dr_fixed)
            # Ensure max_dr allows this value
            stn2['max_dr'] = max(float(stn2.get('max_dr', 0.0)), float(dr_fixed))
        new_stations.append(stn2)
    return new_stations

def fmt_pressure(res, key_m, key_kg):
    """Format pressure values stored in metres and kg/cm²."""

    m = res.get(key_m, 0.0) or 0.0
    kg = res.get(key_kg, 0.0) or 0.0
    return f"{m:.2f} m / {kg:.2f} kg/cm²"

def solve_pipeline(
    stations,
    terminal,
    FLOW,
    KV_list,
    rho_list,
    RateDRA,
    Price_HSD,
    linefill_dict,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
):
    """Wrapper around :mod:`pipeline_model` with origin pump enforcement."""

    import pipeline_model
    import importlib
    import copy

    importlib.reload(pipeline_model)

    stations = copy.deepcopy(stations)
    first_pump = next((s for s in stations if s.get('is_pump')), None)
    if first_pump and first_pump.get('min_pumps', 0) < 1:
        first_pump['min_pumps'] = 1

    if mop_kgcm2 is None:
        mop_kgcm2 = st.session_state.get("MOP_kgcm2")

    try:
        if stations and stations[0].get('pump_types'):
            return pipeline_model.solve_pipeline_multi_origin(
                stations,
                terminal,
                FLOW,
                KV_list,
                rho_list,
                RateDRA,
                Price_HSD,
                linefill_dict,
                dra_reach_km,
                mop_kgcm2,
                hours,
            )
        return pipeline_model.solve_pipeline(
            stations,
            terminal,
            FLOW,
            KV_list,
            rho_list,
            RateDRA,
            Price_HSD,
            linefill_dict,
            dra_reach_km,
            mop_kgcm2,
            hours,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"error": True, "message": str(exc)}

# ==== Batch Linefill Scenario Analysis ====
st.markdown("---")
st.subheader("Batch Linefill Scenario Analysis")

auto_batch = st.checkbox("Run Auto Linefill Generator (Batch Interface Scenarios)")

if auto_batch:
    if not st.session_state.get('stations'):
        st.info("Define pipeline stations above to enable batch optimisation.")
        st.stop()
    total_length = sum(stn["L"] for stn in st.session_state.stations)
    FLOW = st.number_input("Flow rate (m³/hr)", value=st.session_state.get("FLOW", 1000.0), step=10.0, key="batch_flow")
    RateDRA = st.number_input("DRA Cost (INR/L)", value=st.session_state.get("RateDRA", 500.0), step=1.0, key="batch_dra")
    Price_HSD = st.number_input("Diesel Price (INR/L)", value=st.session_state.get("Price_HSD", 70.0), step=0.5, key="batch_diesel")
    st.session_state["FLOW"] = FLOW
    st.session_state["RateDRA"] = RateDRA
    st.session_state["Price_HSD"] = Price_HSD
    num_products = st.number_input("Number of Products", min_value=2, max_value=3, value=2)
    product_table = st.data_editor(
        pd.DataFrame({
            "Product": [f"Product {i+1}" for i in range(num_products)],
            "Viscosity (cSt)": [1.0 + i for i in range(num_products)],
            "Density (kg/m³)": [800 + 40*i for i in range(num_products)],
        }),
        num_rows="dynamic", key="batch_prod_tbl"
    )
    step_size = st.number_input("Step Size (%)", min_value=5, max_value=50, value=10, step=5)
    batch_run = st.button("Run Batch Optimization", key="runbatchbtn", type="primary")

    if batch_run:
        import pandas as pd
        import numpy as np
        with st.spinner("Running batch optimization..."):
            import copy
            stations_data = copy.deepcopy(st.session_state.stations)
            term_data = {
                "name": st.session_state.get("terminal_name", "Terminal"),
                "elev": st.session_state.get("terminal_elev", 0.0),
                "min_residual": st.session_state.get("terminal_head", 50.0)
            }
            FLOW = st.session_state.get("FLOW", 1000.0)
            RateDRA = st.session_state.get("RateDRA", 500.0)
            Price_HSD = st.session_state.get("Price_HSD", 70.0)
            result_rows = []
            segs = int(100 // step_size)
            try:
                # Ensure pump coefficients are updated for all stations
                for idx, stn in enumerate(stations_data, start=1):
                    if stn.get('pump_types'):
                        for ptype in ['A', 'B']:
                            pdata = stn['pump_types'].get(ptype)
                            if not pdata:
                                continue
                            dfh = st.session_state.get(f"head_data_{idx}{ptype}")
                            dfe = st.session_state.get(f"eff_data_{idx}{ptype}")
                            if dfh is not None and len(dfh) >= 3:
                                Qh = dfh.iloc[:, 0].values
                                Hh = dfh.iloc[:, 1].values
                                coeff = np.polyfit(Qh, Hh, 2)
                                pdata['A'], pdata['B'], pdata['C'] = [float(c) for c in coeff]
                            if dfe is not None and len(dfe) >= 5:
                                Qe = dfe.iloc[:, 0].values
                                Ee = dfe.iloc[:, 1].values
                                coeff_e = np.polyfit(Qe, Ee, 4)
                                pdata['P'], pdata['Q'], pdata['R'], pdata['S'], pdata['T'] = [float(c) for c in coeff_e]
                    elif stn.get('is_pump', False):
                        dfh = st.session_state.get(f"head_data_{idx}")
                        dfe = st.session_state.get(f"eff_data_{idx}")
                        if dfh is None and "head_data" in stn:
                            dfh = pd.DataFrame(stn["head_data"])
                        if dfe is None and "eff_data" in stn:
                            dfe = pd.DataFrame(stn["eff_data"])
                        if dfh is not None and len(dfh) >= 3:
                            Qh = dfh.iloc[:, 0].values
                            Hh = dfh.iloc[:, 1].values
                            coeff = np.polyfit(Qh, Hh, 2)
                            stn['A'], stn['B'], stn['C'] = [float(c) for c in coeff]
                        if dfe is not None and len(dfe) >= 5:
                            Qe = dfe.iloc[:, 0].values
                            Ee = dfe.iloc[:, 1].values
                            coeff_e = np.polyfit(Qe, Ee, 4)
                            stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = [float(c) for c in coeff_e]

                # -- 2 product batch --
                if num_products == 2:
                    # 100% A
                    kv_list = []
                    rho_list = []
                    for i in range(len(stations_data)):
                        prod_row = product_table.iloc[0]
                        kv_list.append(prod_row["Viscosity (cSt)"])
                        rho_list.append(prod_row["Density (kg/m³)"])
                    res = solve_pipeline(stations_data, term_data, FLOW, kv_list, rho_list, RateDRA, Price_HSD, {})
                    row = {"Scenario": f"100% {product_table.iloc[0]['Product']}, 0% {product_table.iloc[1]['Product']}"}
                    for idx, stn in enumerate(stations_data, start=1):
                        key = stn['name'].lower().replace(' ', '_')
                        row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                        row[f"Speed {stn['name']}"] = res.get(f"speed_{key}", "")
                        row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                        row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                        row[f"DRA PPM {stn['name']}"] = res.get(f"dra_ppm_{key}", "")
                        row[f"Power Cost {stn['name']}"] = res.get(f"power_cost_{key}", "")
                        row[f"Drag Reduction {stn['name']}"] = res.get(f"drag_reduction_{key}", "")
                    row["Total Cost"] = res.get("total_cost", "")
                    result_rows.append(row)
                    # 100% B
                    kv_list = []
                    rho_list = []
                    for i in range(len(stations_data)):
                        prod_row = product_table.iloc[1]
                        kv_list.append(prod_row["Viscosity (cSt)"])
                        rho_list.append(prod_row["Density (kg/m³)"])
                    res = solve_pipeline(stations_data, term_data, FLOW, kv_list, rho_list, RateDRA, Price_HSD, {})
                    row = {"Scenario": f"0% {product_table.iloc[0]['Product']}, 100% {product_table.iloc[1]['Product']}"}
                    for idx, stn in enumerate(stations_data, start=1):
                        key = stn['name'].lower().replace(' ', '_')
                        row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                        row[f"Speed {stn['name']}"] = res.get(f"speed_{key}", "")
                        row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                        row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                        row[f"DRA PPM {stn['name']}"] = res.get(f"dra_ppm_{key}", "")
                        row[f"Power Cost {stn['name']}"] = res.get(f"power_cost_{key}", "")
                        row[f"Drag Reduction {stn['name']}"] = res.get(f"drag_reduction_{key}", "")
                    row["Total Cost"] = res.get("total_cost", "")
                    result_rows.append(row)
                    # In-between scenarios
                    for pct_A in range(step_size, 100, step_size):
                        pct_B = 100 - pct_A
                        segment_limits = [0]
                        for stn in stations_data:
                            segment_limits.append(segment_limits[-1] + stn["L"])
                        cumlen = segment_limits
                        kv_list = []
                        rho_list = []
                        for i in range(len(stations_data)):
                            mid = (cumlen[i]+cumlen[i+1])/2
                            frac = 100*mid/total_length
                            if frac <= pct_A:
                                prod_row = product_table.iloc[0]
                            else:
                                prod_row = product_table.iloc[1]
                            kv_list.append(prod_row["Viscosity (cSt)"])
                            rho_list.append(prod_row["Density (kg/m³)"])
                        res = solve_pipeline(stations_data, term_data, FLOW, kv_list, rho_list, RateDRA, Price_HSD, {})
                        row = {"Scenario": f"{pct_A}% {product_table.iloc[0]['Product']}, {pct_B}% {product_table.iloc[1]['Product']}"}
                        for idx, stn in enumerate(stations_data, start=1):
                            key = stn['name'].lower().replace(' ', '_')
                            row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                            row[f"Speed {stn['name']}"] = res.get(f"speed_{key}", "")
                            row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                            row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                            row[f"DRA PPM {stn['name']}"] = res.get(f"dra_ppm_{key}", "")
                            row[f"Power Cost {stn['name']}"] = res.get(f"power_cost_{key}", "")
                            row[f"Drag Reduction {stn['name']}"] = res.get(f"drag_reduction_{key}", "")
                        row["Total Cost"] = res.get("total_cost", "")
                        result_rows.append(row)
                # -- 3 product batch --
                if num_products == 3:
                    for first, label in enumerate(["A", "B", "C"]):
                        kv_list = []
                        rho_list = []
                        for i in range(len(stations_data)):
                            prod_row = product_table.iloc[first]
                            kv_list.append(prod_row["Viscosity (cSt)"])
                            rho_list.append(prod_row["Density (kg/m³)"])
                        scenario_labels = ["0%"] * 3
                        scenario_labels[first] = "100%"
                        row = {"Scenario": f"{scenario_labels[0]} {product_table.iloc[0]['Product']}, {scenario_labels[1]} {product_table.iloc[1]['Product']}, {scenario_labels[2]} {product_table.iloc[2]['Product']}"}
                        res = solve_pipeline(stations_data, term_data, FLOW, kv_list, rho_list, RateDRA, Price_HSD, {})
                        for idx, stn in enumerate(stations_data, start=1):
                            key = stn['name'].lower().replace(' ', '_')
                            row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                            row[f"Speed {stn['name']}"] = res.get(f"speed_{key}", "")
                            row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                            row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                            row[f"DRA PPM {stn['name']}"] = res.get(f"dra_ppm_{key}", "")
                            row[f"Power Cost {stn['name']}"] = res.get(f"power_cost_{key}", "")
                            row[f"Drag Reduction {stn['name']}"] = res.get(f"drag_reduction_{key}", "")
                        row["Total Cost"] = res.get("total_cost", "")
                        result_rows.append(row)
                    # In-between scenarios
                    for pct_A in range(step_size, 100, step_size):
                        for pct_B in range(step_size, 100 - pct_A + step_size, step_size):
                            pct_C = 100 - pct_A - pct_B
                            if pct_C < 0 or pct_C > 100:
                                continue
                            segment_limits = [0]
                            for stn in stations_data:
                                segment_limits.append(segment_limits[-1] + stn["L"])
                            cumlen = segment_limits
                            kv_list = []
                            rho_list = []
                            for i in range(len(stations_data)):
                                mid = (cumlen[i]+cumlen[i+1])/2
                                frac = 100*mid/total_length
                                if frac <= pct_A:
                                    prod_row = product_table.iloc[0]
                                elif frac <= pct_A + pct_B:
                                    prod_row = product_table.iloc[1]
                                else:
                                    prod_row = product_table.iloc[2]
                                kv_list.append(prod_row["Viscosity (cSt)"])
                                rho_list.append(prod_row["Density (kg/m³)"])
                            row = {
                                "Scenario": f"{pct_A}% {product_table.iloc[0]['Product']}, {pct_B}% {product_table.iloc[1]['Product']}, {pct_C}% {product_table.iloc[2]['Product']}"
                            }
                            res = solve_pipeline(stations_data, term_data, FLOW, kv_list, rho_list, RateDRA, Price_HSD, {})
                            for idx, stn in enumerate(stations_data, start=1):
                                key = stn['name'].lower().replace(' ', '_')
                                row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                                row[f"Speed {stn['name']}"] = res.get(f"speed_{key}", "")
                                row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                                row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                                row[f"DRA PPM {stn['name']}"] = res.get(f"dra_ppm_{key}", "")
                                row[f"Power Cost {stn['name']}"] = res.get(f"power_cost_{key}", "")
                                row[f"Drag Reduction {stn['name']}"] = res.get(f"drag_reduction_{key}", "")
                            row["Total Cost"] = res.get("total_cost", "")
                            result_rows.append(row)
                df_batch = pd.DataFrame(result_rows)
                st.session_state['batch_df'] = df_batch
            except Exception as e:
                st.session_state.pop('batch_df', None)
                st.error(f"Batch optimization failed: {e}")

    if 'batch_df' in st.session_state:
        df_batch = st.session_state['batch_df']
        st.dataframe(df_batch, use_container_width=True)
        st.download_button("Download Batch Results", df_batch.to_csv(index=False), file_name="batch_results.csv")
        if len(df_batch) > 0:
            pc_cols = []
            for c in df_batch.columns:
                c_lower = c.lower()
                if (
                    '%prod' in c_lower or 
                    'speed' in c_lower or 
                    'num pumps' in c_lower or
                    'dra ppm' in c_lower or
                    'total cost' in c_lower
                ):
                    pc_cols.append(c)
            if len(pc_cols) < 3:
                pc_cols = [c for c in df_batch.columns if c != "Scenario"]
            pc_cols = [c for c in pc_cols if df_batch[c].notnull().sum() > 0]
            pc_df = df_batch[pc_cols].apply(pd.to_numeric, errors='coerce')
            st.markdown("#### Multi-dimensional Batch Optimization Visualization")
            import plotly.express as px
            fig = px.parallel_coordinates(
                pc_df,
                color=pc_df[pc_cols[-1]],
                labels={col: col for col in pc_cols},
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Scenario-wise Multi-Dimensional Results",
            )
            fig.update_layout(
                font=dict(size=20),
                title_font=dict(size=26),
                margin=dict(l=40, r=40, t=80, b=40),
                height=750,
                plot_bgcolor='white',
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("Each line = one scenario. Hover to see full parameter set for each scenario.")
else:
    st.session_state.pop('batch_df', None)



if not auto_batch:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    run = st.button("Run Instantaneous Flow Optimizer", key="runoptbtn", help="Run pipeline optimization.", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)
    if run:
        with st.spinner("Solving optimization..."):
            stations_data = st.session_state.stations
            term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}
            # Always ensure linefill_df, kv_list, rho_list are defined!
            linefill_df = st.session_state.get("linefill_df", pd.DataFrame())
            kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)

            
            # ------------- ADD THIS BLOCK -------------
            import pandas as pd
            import numpy as np
    
            for idx, stn in enumerate(stations_data, start=1):
                if stn.get('is_pump', False):
                    if idx == 1 and 'pump_types' in stn:
                        for ptype in ['A', 'B']:
                            if ptype not in stn['pump_types']:
                                continue
                            if stn['pump_types'][ptype].get('available', 0) == 0:
                                continue
                            dfh = st.session_state.get(f"head_data_{idx}{ptype}")
                            dfe = st.session_state.get(f"eff_data_{idx}{ptype}")
                            stn['pump_types'][ptype]['head_data'] = dfh
                            stn['pump_types'][ptype]['eff_data'] = dfe
                    else:
                        dfh = st.session_state.get(f"head_data_{idx}")
                        dfe = st.session_state.get(f"eff_data_{idx}")
                        if dfh is None and "head_data" in stn:
                            dfh = pd.DataFrame(stn["head_data"])
                        if dfe is None and "eff_data" in stn:
                            dfe = pd.DataFrame(stn["eff_data"])
                        if dfh is not None and len(dfh) >= 3:
                            Qh = dfh.iloc[:, 0].values
                            Hh = dfh.iloc[:, 1].values
                            coeff = np.polyfit(Qh, Hh, 2)
                            stn['A'], stn['B'], stn['C'] = float(coeff[0]), float(coeff[1]), float(coeff[2])
                        if dfe is not None and len(dfe) >= 5:
                            Qe = dfe.iloc[:, 0].values
                            Ee = dfe.iloc[:, 1].values
                            coeff_e = np.polyfit(Qe, Ee, 4)
                            stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = [float(c) for c in coeff_e]
            # ------------- END OF BLOCK -------------

            res = solve_pipeline(
                stations_data,
                term_data,
                FLOW,
                kv_list,
                rho_list,
                RateDRA,
                Price_HSD,
                linefill_df.to_dict(),
                dra_reach_km=0.0,
                mop_kgcm2=st.session_state.get("MOP_kgcm2"),
                hours=24.0,
            )

            import copy
            if not res or res.get("error"):
                msg = res.get("message") if isinstance(res, dict) else "Optimization failed"
                st.error(msg)
                for k in ["last_res", "last_stations_data", "last_term_data", "last_linefill"]:
                    st.session_state.pop(k, None)
            else:
                st.session_state["last_res"] = copy.deepcopy(res)
                st.session_state["last_stations_data"] = copy.deepcopy(res.get('stations_used', stations_data))
                st.session_state["last_term_data"] = copy.deepcopy(term_data)
                st.session_state["last_linefill"] = copy.deepcopy(linefill_df)
                # --- CRUCIAL LINE TO FORCE UI REFRESH ---
                st.rerun()

    st.markdown("<div style='text-align:center; margin-top: 0.6rem;'>", unsafe_allow_html=True)
    run_day = st.button("Run Daily Pumping Schedule Optimizer", key="run_day_btn", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_day:
        with st.spinner("Running 6 optimizations (07:00 to 03:00)..."):
            import copy
            stations_base = copy.deepcopy(st.session_state.stations)
            for stn in stations_base:
                if stn.get('pump_types'):
                    names_all = []
                    for pdata in stn['pump_types'].values():
                        avail = int(pdata.get('available', 0))
                        names = pdata.get('names', [])
                        if len(names) < avail:
                            names += [f"Pump {len(names)+i+1}" for i in range(avail - len(names))]
                        names_all.extend(names[:avail])
                    stn['pump_names'] = names_all
                    if names_all:
                        stn['pump_name'] = names_all[0]
                elif stn.get('is_pump'):
                    names = stn.get('pump_names', [])
                    if len(names) < stn.get('max_pumps', 1):
                        names += [f"Pump {stn['name']} {i+1}" for i in range(len(names), stn.get('max_pumps', 1))]
                    stn['pump_names'] = names
                    if names:
                        stn['pump_name'] = names[0]
            term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}

            # Prepare initial volumetric linefill
            vol_df = st.session_state.get("linefill_vol_df", pd.DataFrame())
            if vol_df is None or len(vol_df)==0:
                st.error("Please enter linefill (volumetric) data.")
                st.stop()

            # Determine FLOW for this mode
            if st.session_state.get("op_mode") == "Daily Pumping Schedule":
                plan_df = st.session_state.get("day_plan_df", pd.DataFrame())
                daily_m3 = float(plan_df["Volume (m³)"].astype(float).sum()) if len(plan_df) else 0.0
                FLOW_sched = daily_m3 / 24.0
            else:
                plan_df = None
                FLOW_sched = st.session_state.get("FLOW", 1000.0)

            # Helper to compute segment kv/rho from volumetric table
            def kv_rho_from_vol(vol_df_now):
                return map_vol_linefill_to_segments(vol_df_now, stations_base)

            # Time points
            hours = [7,11,15,19,23,27]
            reports = []
            linefill_snaps = []
            total_length = sum(stn.get('L', 0.0) for stn in stations_base)
            dra_reach_km = 0.0

            current_vol = vol_df.copy()

            for ti, hr in enumerate(hours):
                pumped_tmp = FLOW_sched * 4.0
                future_vol, future_plan = shift_vol_linefill(
                    current_vol.copy(), pumped_tmp, plan_df.copy() if plan_df is not None else None
                )
                # Determine worst-case fluid properties over this 4h window
                kv_now, rho_now = kv_rho_from_vol(current_vol)
                kv_next, rho_next = kv_rho_from_vol(future_vol)
                kv_list = [max(a, b) for a, b in zip(kv_now, kv_next)]
                rho_list = [max(a, b) for a, b in zip(rho_now, rho_next)]

                stns_run = copy.deepcopy(stations_base)

                res = solve_pipeline(
                    stns_run, term_data, FLOW_sched, kv_list, rho_list,
                    RateDRA, Price_HSD, current_vol.to_dict(), dra_reach_km,
                    st.session_state.get("MOP_kgcm2"), hours=4.0
                )

                if res.get("error"):
                    st.error(f"Optimization failed at {hr%24:02d}:00 -> {res.get('message','')}")
                    st.stop()

                reports.append({"time": hr%24, "result": res})
                linefill_snaps.append(current_vol.copy())

                if ti < len(hours)-1:
                    current_vol, plan_df = future_vol, future_plan
                    dra_reach_km = float(res.get('dra_front_km', dra_reach_km))

            # Build a consolidated station-wise table
            station_tables = []
            for rec in reports:
                res = rec["result"]
                hr = rec["time"]
                df_int = build_station_table(res, stations_base)
                df_int.insert(0, "Time", f"{hr:02d}:00")
                station_tables.append(df_int)
            df_day = pd.concat(station_tables, ignore_index=True).fillna(0.0).round(2)

            num_cols = [c for c in df_day.columns if c not in ["Time", "Station", "Pump Name"]]
            styled = df_day.style.format({c: "{:.2f}" for c in num_cols}).background_gradient(
                subset=num_cols, cmap="Blues"
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Daily Optimizer Output data",
                df_day.to_csv(index=False, float_format="%.2f"),
                file_name="daily_schedule_results.csv",
            )

            combined = []
            for idx, df_line in enumerate(linefill_snaps):
                hr = hours[idx] % 24
                temp = df_line.copy()
                temp['Time'] = f"{hr:02d}:00"
                combined.append(temp)
            lf_all = pd.concat(combined, ignore_index=True).round(2)
            st.download_button(
                "Download Daily Dynamic Linefill Output",
                lf_all.to_csv(index=False, float_format="%.2f"),
                file_name="linefill_snapshots.csv",
            )
    st.markdown("<div style='text-align:center; margin-top: 0.6rem;'>", unsafe_allow_html=True)
    run_plan = st.button("Run Dynamic Pumping Plan Optimizer", key="run_plan_btn", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_plan:
        with st.spinner("Running dynamic pumping plan optimization..."):
            import copy
            stations_base = copy.deepcopy(st.session_state.get("stations", []))
            for stn in stations_base:
                if stn.get('pump_types'):
                    names_all = []
                    for pdata in stn['pump_types'].values():
                        avail = int(pdata.get('available', 0))
                        names = pdata.get('names', [])
                        if len(names) < avail:
                            names += [f"Pump {len(names)+i+1}" for i in range(avail - len(names))]
                        names_all.extend(names[:avail])
                    stn['pump_names'] = names_all
                    if names_all:
                        stn['pump_name'] = names_all[0]
                elif stn.get('is_pump'):
                    names = stn.get('pump_names', [])
                    if len(names) < stn.get('max_pumps', 1):
                        names += [f"Pump {stn['name']} {i+1}" for i in range(len(names), stn.get('max_pumps', 1))]
                    stn['pump_names'] = names
                    if names:
                        stn['pump_name'] = names[0]
            if not stations_base:
                st.error("Please configure station data before running.")
                st.stop()
            term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}

            vol_df = st.session_state.get("linefill_vol_df", pd.DataFrame())
            if vol_df is None or len(vol_df) == 0:
                st.error("Please enter linefill (volumetric) data.")
                st.stop()

            flow_df = st.session_state.get("proj_flow_df", pd.DataFrame())
            if flow_df is None or len(flow_df) == 0:
                st.error("Please enter flow schedule data.")
                st.stop()
            prod_plan_df = st.session_state.get("proj_plan_df", pd.DataFrame())
            current_plan = prod_plan_df.copy()

            flow_df = flow_df.copy()
            flow_df["Start"] = pd.to_datetime(flow_df["Start"])
            flow_df["End"] = pd.to_datetime(flow_df["End"])
            flow_df = flow_df.sort_values("Start").reset_index(drop=True)

            current_vol = vol_df.copy()
            reports = []
            linefill_snaps = []
            dra_reach_km = 0.0

            for _, row in flow_df.iterrows():
                flow = float(row.get("Flow (m³/h)", row.get("Flow", 0.0)) or 0.0)
                start_ts = row["Start"]
                end_ts = row["End"]
                if flow <= 0 or end_ts <= start_ts:
                    continue
                seg_start = start_ts
                while seg_start < end_ts:
                    seg_end = min(seg_start + pd.Timedelta(hours=4), end_ts)
                    duration_hr = (seg_end - seg_start).total_seconds() / 3600.0
                    pumped_m3 = flow * duration_hr

                    try:
                        kv_now, rho_now = map_vol_linefill_to_segments(current_vol, stations_base)
                        future_vol, current_plan = shift_vol_linefill(current_vol.copy(), pumped_m3, current_plan)
                        kv_next, rho_next = map_vol_linefill_to_segments(future_vol, stations_base)
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()

                    kv_run = [max(a, b) for a, b in zip(kv_now, kv_next)]
                    rho_run = [max(a, b) for a, b in zip(rho_now, rho_next)]

                    stns_run = copy.deepcopy(stations_base)
                    res = solve_pipeline(
                        stns_run,
                        term_data,
                        flow,
                        kv_run,
                        rho_run,
                        RateDRA,
                        Price_HSD,
                        current_vol.to_dict(),
                        dra_reach_km,
                        st.session_state.get("MOP_kgcm2"),
                        hours=duration_hr,
                    )
                    if res.get("error"):
                        st.error(f"Optimization failed for interval starting {seg_start} -> {res.get('message','')}")
                        st.stop()

                    reports.append({"time": seg_start, "result": res})
                    linefill_snaps.append(current_vol.copy())
                    current_vol = future_vol
                    dra_reach_km = float(res.get("dra_front_km", dra_reach_km))
                    seg_start = seg_end

            if not reports:
                st.error("No valid intervals in projected plan.")
                st.stop()

            st.session_state["last_plan_start"] = flow_df["Start"].min()
            st.session_state["last_plan_hours"] = (flow_df["End"].max() - flow_df["Start"].min()).total_seconds() / 3600.0
            st.session_state["last_res"] = copy.deepcopy(reports[-1]["result"])
            st.session_state["last_stations_data"] = copy.deepcopy(stations_base)
            st.session_state["last_term_data"] = copy.deepcopy(term_data)
            st.session_state["last_linefill"] = copy.deepcopy(current_vol)

            station_tables = []
            for rec in reports:
                res = rec["result"]
                ts = rec["time"]
                df_int = build_station_table(res, stations_base)
                df_int.insert(0, "Time", ts.strftime("%d/%m %H:%M"))
                station_tables.append(df_int)
            df_plan = pd.concat(station_tables, ignore_index=True).fillna(0.0).round(2)

            num_cols = [c for c in df_plan.columns if c not in ["Time", "Station", "Pump Name"]]
            styled = df_plan.style.format({c: "{:.2f}" for c in num_cols}).background_gradient(
                subset=num_cols, cmap="Blues"
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Dynamic Plan Output data",
                df_plan.to_csv(index=False, float_format="%.2f"),
                file_name="dynamic_plan_results.csv",
            )

            combined = []
            for idx, df_line in enumerate(linefill_snaps):
                ts = reports[idx]["time"]
                temp = df_line.copy()
                temp["Time"] = ts.strftime("%d/%m %H:%M")
                combined.append(temp)
            lf_all = pd.concat(combined, ignore_index=True).round(2)
            st.download_button(
                "Download Dynamic Linefill Output",
                lf_all.to_csv(index=False, float_format="%.2f"),
                file_name="linefill_snapshots.csv",
            )


if not auto_batch and "last_res" in st.session_state:
    st.markdown("<div class='section-title'>Sensitivity Analysis</div>", unsafe_allow_html=True)
    st.write("Analyze how key outputs respond to variations in a parameter. Each run recalculates results based on set pipeline parameter and optimization metric.")
    param = st.selectbox("Parameter to vary", [
        "Flowrate (m³/hr)", "Viscosity (cSt)", "Drag Reduction (%)", "Diesel Price (INR/L)", "DRA Cost (INR/L)"
    ])
    output = st.selectbox("Output metric", [
        "Total Cost (INR)", "Power Cost (INR)", "DRA Cost (INR)",
        "Residual Head (m)", "Pump Efficiency (%)",
    ])
    if st.button("Run Sensitivity Analysis"):
        FLOW = st.session_state["FLOW"]
        RateDRA = st.session_state["RateDRA"]
        Price_HSD = st.session_state["Price_HSD"]
        linefill_df = st.session_state.get("linefill_df", pd.DataFrame())
        N = 10
        if param == "Flowrate (m³/hr)":
            pvals = np.linspace(max(10, 0.5*FLOW), 1.5*FLOW, N)
        elif param == "Viscosity (cSt)":
            first_visc = linefill_df.iloc[0]["Viscosity (cSt)"] if not linefill_df.empty else 10
            pvals = np.linspace(max(1, 0.5*first_visc), 2*first_visc, N)
        elif param == "Drag Reduction (%)":
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
        st.info("Running sensitivity... This may take a few seconds per parameter.")
        progress = st.progress(0)
        for i, val in enumerate(pvals):
            stations_data = [dict(s) for s in st.session_state['stations']]
            term_data = dict(st.session_state["last_term_data"])
            kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
            this_FLOW = FLOW
            this_RateDRA = RateDRA
            this_Price_HSD = Price_HSD
            this_linefill_df = linefill_df.copy()
            if param == "Flowrate (m³/hr)":
                this_FLOW = val
            elif param == "Viscosity (cSt)":
                this_linefill_df["Viscosity (cSt)"] = val
                kv_list, rho_list = map_linefill_to_segments(this_linefill_df, stations_data)
            elif param == "Drag Reduction (%)":
                for stn in stations_data:
                    if stn.get('is_pump', False):
                        stn['max_dr'] = max(stn.get('max_dr', val), val)
                        break
            elif param == "Diesel Price (INR/L)":
                this_Price_HSD = val
            elif param == "DRA Cost (INR/L)":
                this_RateDRA = val
            resi = solve_pipeline(stations_data, term_data, this_FLOW, kv_list, rho_list, this_RateDRA, this_Price_HSD, this_linefill_df.to_dict())
            total_cost = power_cost = dra_cost = rh = eff = 0
            for idx, stn in enumerate(stations_data):
                key = stn['name'].lower().replace(' ', '_')
                dra_cost_i = float(resi.get(f"dra_cost_{key}", 0.0) or 0.0)
                power_cost_i = float(resi.get(f"power_cost_{key}", 0.0) or 0.0)
                eff_i = float(resi.get(f"efficiency_{key}", 100.0))
                rh_i = float(resi.get(f"residual_head_{key}", 0.0) or 0.0)
                total_cost += dra_cost_i + power_cost_i
                power_cost += power_cost_i
                dra_cost += dra_cost_i
                rh += rh_i
                eff += eff_i if stn.get('is_pump', False) else 0
            if output == "Total Cost (INR)":
                yvals.append(total_cost)
            elif output == "Power Cost (INR)":
                yvals.append(power_cost)
            elif output == "DRA Cost (INR)":
                yvals.append(dra_cost)
            elif output == "Residual Head (m)":
                yvals.append(rh)
            elif output == "Pump Efficiency (%)":
                yvals.append(eff)
            progress.progress((i+1)/len(pvals))
        fig = px.line(x=pvals, y=yvals, labels={"x": param, "y": output}, title=f"{output} vs {param} (Sensitivity)")
        df_sens = pd.DataFrame({param: pvals, output: yvals})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_sens, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df_sens.to_csv(index=False).encode(), file_name="sensitivity.csv")

    st.markdown("<div class='section-title'>Benchmarking & Global Standards</div>", unsafe_allow_html=True)
    st.write("Compare pipeline performance with global/ custom benchmarks. Green indicates Pipeline operation match/exceed global standards while red means improvement is needed.")
    b_mode = st.radio("Benchmark Source", ["Global Standards", "Edit Benchmarks", "Upload CSV"])
    benchmarks = {}
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
        up = st.file_uploader("Upload benchmarks CSV", type="csv")
        if up:
            bdf = pd.read_csv(up)
            st.dataframe(bdf)
            benchmarks = dict(zip(bdf["Parameter"], bdf["Benchmark Value"]))
        if not benchmarks:
            st.warning("Please upload a CSV with columns [Parameter, Benchmark Value]")
    if st.button("Run Benchmarking"):
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        total_length = sum([s.get("L", 0.0) for s in stations_data])
        total_cost = 0
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
            total_power += power_cost
            eff = float(res.get(f"efficiency_{key}", 100.0))
            if stn.get('is_pump', False):
                effs.append(eff)
            if velocity > max_velocity:
                max_velocity = velocity
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
        df_bench = pd.DataFrame(rows, columns=["Parameter", "Pipeline", "Benchmark", "Status"])
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Annualized Savings Simulator</div>", unsafe_allow_html=True)
    st.write("Annual savings from efficiency improvements, energy cost and DRA optimizations.")
    FLOW = st.session_state["FLOW"]
    RateDRA = st.session_state["RateDRA"]
    Price_HSD = st.session_state["Price_HSD"]
    pump_eff_impr = st.slider("Pump Efficiency Improvement (%)", 0, 10, 3)
    dra_cost_impr = st.slider("DRA Price Reduction (%)", 0, 30, 5)
    flow_change = st.slider("Throughput Increase (%)", 0, 30, 0)
    if st.button("Run Savings Simulation"):
        stations_data = [dict(s) for s in st.session_state['stations']]
        term_data = dict(st.session_state["last_term_data"])
        linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
        new_RateDRA = RateDRA * (1 - dra_cost_impr / 100)
        new_FLOW = FLOW * (1 + flow_change / 100)
        kv_list, rho_list = map_linefill_to_segments(linefill_df, stations_data)
        res2 = solve_pipeline(stations_data, term_data, new_FLOW, kv_list, rho_list, new_RateDRA, Price_HSD, linefill_df.to_dict())
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
        st.info("Calculations are based on optimized values.")


st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
    &copy; 2025 Pipeline Optima™ v1.1.1. Developed by Parichay Das.
    </div>
    """,
    unsafe_allow_html=True
)
