import os
import sys
from pathlib import Path
import streamlit as st
import altair as alt
import pipeline_model
import datetime as dt

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
if "run_mode" not in st.session_state:
    st.session_state["run_mode"] = None
if "Fuel_density" not in st.session_state:
    st.session_state["Fuel_density"] = 820.0
if "Ambient_temp" not in st.session_state:
    st.session_state["Ambient_temp"] = 25.0
if "pump_shear_rate" not in st.session_state:
    st.session_state["pump_shear_rate"] = 0.0
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import isclose, pi, sqrt
import hashlib
import uuid
import json
import copy
from collections import OrderedDict
from plotly.colors import qualitative

# Ensure local modules are importable when the app is run from an arbitrary
# working directory (e.g. `streamlit run path/to/pipeline_optimization_app.py`).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hide Vega action buttons globally
alt.renderers.set_embed_options(actions=False)

from dra_utils import (
    get_ppm_for_dr,
    DRA_CURVE_DATA,
)


INIT_DRA_COL = "Initial DRA (ppm)"


def ensure_initial_dra_column(
    df: pd.DataFrame | None,
    *,
    default: float | None = 0.0,
    fill_blanks: bool = False,
) -> pd.DataFrame | None:
    """Ensure ``df`` exposes the user-editable DRA ppm column.

    ``fill_blanks`` only repopulates entries that are truly blank (``NaN`` or
    empty strings) so users can intentionally clear a cell to request
    ``0 ppm``.
    """

    if not isinstance(df, pd.DataFrame):
        return df

    if INIT_DRA_COL not in df.columns:
        if default is None:
            df[INIT_DRA_COL] = np.nan
        else:
            df[INIT_DRA_COL] = default
        return df

    if not fill_blanks or default is None:
        return df

    col = df[INIT_DRA_COL]
    blank_mask = col.isna()
    if col.dtype == object:
        blank_mask |= col.astype(str).str.strip() == ""
    if blank_mask.any():
        df.loc[blank_mask, INIT_DRA_COL] = default
    return df

st.set_page_config(page_title="Pipeline Optima™", layout="wide", initial_sidebar_state="expanded")

#Custom Styles
st.markdown("""
    <style>
    .stButton > button,
    .stDownloadButton > button {
        background: var(--primary-color);
        color: var(--secondary-background-color);
        font-weight: 600;
        border: 1px solid transparent;
        border-radius: 12px;
        box-shadow: 0 3px 18px #00000022;
        transition: filter 0.19s ease-in-out, transform 0.19s ease-in-out;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        filter: brightness(0.92);
    }
    .stButton > button:active,
    .stDownloadButton > button:active {
        filter: brightness(0.85);
        transform: translateY(1px);
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
users = {"pipeline_optima": hash_pwd("heteroscedasticity")}
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

if st.sidebar.button("Hydraulic feasibility check"):
    st.session_state["run_mode"] = "hydraulic"
    st.rerun()

if st.session_state.get("run_mode") == "hydraulic":
    from hydraulic_check import hydraulic_app
    hydraulic_app()
    st.stop()

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
    st.session_state['Fuel_density'] = loaded_data.get('Fuel_density', 820.0)
    st.session_state['Ambient_temp'] = loaded_data.get('Ambient_temp', 25.0)
    st.session_state['pump_shear_rate'] = loaded_data.get('pump_shear_rate', 0.0)
    st.session_state['MOP_kgcm2'] = loaded_data.get('MOP_kgcm2', 100.0)
    st.session_state['op_mode'] = loaded_data.get('op_mode', "Flow rate")
    if loaded_data.get("linefill_vol"):
        st.session_state["linefill_vol_df"] = pd.DataFrame(loaded_data["linefill_vol"])
        ensure_initial_dra_column(st.session_state["linefill_vol_df"], default=0.0, fill_blanks=True)
        # Keep a unified linefill table so subsequent logic and case saving
        # operate on the user-edited volumetric data instead of a stale
        # distance-based default.
        st.session_state["linefill_df"] = st.session_state["linefill_vol_df"].copy()
    if loaded_data.get("day_plan"):
        st.session_state["day_plan_df"] = pd.DataFrame(loaded_data["day_plan"])
        ensure_initial_dra_column(st.session_state["day_plan_df"], default=0.0, fill_blanks=True)
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
        ensure_initial_dra_column(st.session_state["linefill_df"], default=0.0, fill_blanks=True)
    for i, stn in enumerate(st.session_state['stations'], start=1):
        head_data = loaded_data.get(f"head_data_{i}", None)
        eff_data  = loaded_data.get(f"eff_data_{i}", None)
        peak_data = loaded_data.get(f"peak_data_{i}", None)
        loop_peak_data = loaded_data.get(f"loop_peak_data_{i}", None)
        if head_data is not None:
            df_head = pd.DataFrame(head_data)
            st.session_state[f"head_data_{i}"] = df_head
            st.session_state['stations'][i-1]['head_data'] = head_data
        if eff_data is not None:
            df_eff = pd.DataFrame(eff_data)
            st.session_state[f"eff_data_{i}"] = df_eff
            st.session_state['stations'][i-1]['eff_data'] = eff_data
        if peak_data is not None:
            df_peak = pd.DataFrame(peak_data)
            st.session_state[f"peak_data_{i}"] = df_peak
            st.session_state['stations'][i-1]['peak_data'] = peak_data
        if loop_peak_data is not None:
            df_lpeak = pd.DataFrame(loop_peak_data)
            st.session_state[f"loop_peak_data_{i}"] = df_lpeak
            st.session_state['stations'][i-1].setdefault('loopline', {})['peaks'] = loop_peak_data
        else:
            loop_peaks = st.session_state['stations'][i-1].get('loopline', {}).get('peaks')
            if loop_peaks is not None:
                st.session_state[f"loop_peak_data_{i}"] = pd.DataFrame(loop_peaks)

        # Load pump type-specific data if present
        for ptype in stn.get('pump_types', {}).keys():
            head_pt = loaded_data.get(f"head_data_{i}{ptype}", stn['pump_types'][ptype].get('head_data'))
            eff_pt  = loaded_data.get(f"eff_data_{i}{ptype}", stn['pump_types'][ptype].get('eff_data'))
            peak_pt = loaded_data.get(f"peak_data_{i}{ptype}", stn['pump_types'][ptype].get('peak_data'))
            if head_pt is not None:
                df_head_pt = pd.DataFrame(head_pt)
                st.session_state[f"head_data_{i}{ptype}"] = df_head_pt
                st.session_state['stations'][i-1].setdefault('pump_types', {}).setdefault(ptype, {})['head_data'] = head_pt
            if eff_pt is not None:
                df_eff_pt = pd.DataFrame(eff_pt)
                st.session_state[f"eff_data_{i}{ptype}"] = df_eff_pt
                st.session_state['stations'][i-1].setdefault('pump_types', {}).setdefault(ptype, {})['eff_data'] = eff_pt
            if peak_pt is not None:
                df_peak_pt = pd.DataFrame(peak_pt)
                st.session_state[f"peak_data_{i}{ptype}"] = df_peak_pt
                st.session_state['stations'][i-1].setdefault('pump_types', {}).setdefault(ptype, {})['peak_data'] = peak_pt

uploaded_case = st.sidebar.file_uploader("🔁 Load Case", type="json", key="casefile")
if uploaded_case is None:
    st.session_state["case_loaded"] = False
    st.session_state["case_loaded_name"] = None
else:
    file_id = getattr(uploaded_case, "name", None)
    if st.session_state.get("case_loaded_name") != file_id:
        loaded_data = json.load(uploaded_case)
        restore_case_dict(loaded_data)
        st.session_state["case_loaded"] = True
        st.session_state["case_loaded_name"] = file_id
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
        FLOW = st.number_input(
            "Flow rate (m³/hr)",
            value=st.session_state.get("FLOW", 1000.0),
            step=10.0,
            key="FLOW",
        )
        RateDRA = st.number_input(
            "DRA Cost (INR/L)",
            value=st.session_state.get("RateDRA", 500.0),
            step=1.0,
            key="RateDRA",
        )
        Price_HSD = st.number_input(
            "Fuel Price (INR/L)",
            value=st.session_state.get("Price_HSD", 70.0),
            step=0.5,
            key="Price_HSD",
        )
        Fuel_density = st.number_input(
            "Fuel density (kg/m³)",
            value=st.session_state.get("Fuel_density", 820.0),
            step=1.0,
            key="Fuel_density",
        )
        Ambient_temp = st.number_input(
            "Ambient temperature (°C)",
            value=st.session_state.get("Ambient_temp", 25.0),
            step=1.0,
            key="Ambient_temp",
        )
        st.slider(
            "Global pump shear fraction",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=float(st.session_state.get("pump_shear_rate", 0.0)),
            key="pump_shear_rate",
            help=(
                "Blends with per-station shear settings to attenuate DRA slugs. "
                "Set to 0.0 for 100% carry-over (no additional shear) or 1.0 for 0% carry-over."
            ),
        )
        st.caption(
            "**Carry-over reference:** 0.0 ⇒ downstream batches keep their DRA (100% carry-over); "
            "1.0 ⇒ pumps strip all DRA (0% carry-over)."
        )
        st.number_input(
            "MOP (kg/cm²)",
            value=st.session_state.get("MOP_kgcm2", 100.0),
            step=1.0,
            key="MOP_kgcm2",
        )

        rpm_step_default = getattr(pipeline_model, "RPM_STEP", 25)
        dra_step_default = getattr(pipeline_model, "DRA_STEP", 2)
        coarse_multiplier_default = getattr(pipeline_model, "COARSE_MULTIPLIER", 5.0)
        state_top_k_default = getattr(pipeline_model, "STATE_TOP_K", 50)
        state_cost_margin_default = getattr(pipeline_model, "STATE_COST_MARGIN", 5000.0)

    with st.expander("Advanced search depth", expanded=False):
        st.caption(
            "Adjust solver discretisation and dynamic-programming limits when you "
            "need a broader search of pump and DRA combinations."
        )
        st.number_input(
            "Refinement RPM step (rpm)",
            min_value=1,
            value=int(st.session_state.get("search_rpm_step", rpm_step_default)),
            step=1,
            key="search_rpm_step",
        )
        st.number_input(
            "Refinement DRA step (%DR)",
            min_value=1,
            value=int(st.session_state.get("search_dra_step", dra_step_default)),
            step=1,
            key="search_dra_step",
        )
        st.number_input(
            "Coarse step multiplier",
            min_value=0.1,
            value=float(st.session_state.get("search_coarse_multiplier", coarse_multiplier_default)),
            step=0.1,
            format="%.1f",
            key="search_coarse_multiplier",
        )
        st.number_input(
            "Max DP states retained",
            min_value=1,
            value=int(st.session_state.get("search_state_top_k", state_top_k_default)),
            step=1,
            key="search_state_top_k",
        )
        st.number_input(
            "DP cost margin (currency)",
            min_value=0.0,
            value=float(st.session_state.get("search_state_cost_margin", state_cost_margin_default)),
            step=100.0,
            key="search_state_cost_margin",
        )

    st.subheader("Operating Mode")
    if "linefill_df" not in st.session_state:
        st.session_state["linefill_df"] = pd.DataFrame({
            "Start (km)": [0.0],
            "End (km)": [100.0],
            "Viscosity (cSt)": [10.0],
            "Density (kg/m³)": [850.0],
            INIT_DRA_COL: [0.0],
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
                INIT_DRA_COL: [0.0],
            })
        else:
            ensure_initial_dra_column(st.session_state["linefill_vol_df"], default=0.0, fill_blanks=True)
        lf_df = st.data_editor(
            st.session_state["linefill_vol_df"],
            num_rows="dynamic",
            key="linefill_vol_editor",
        )
        lf_df = ensure_initial_dra_column(lf_df, default=0.0, fill_blanks=True)
        st.session_state["linefill_vol_df"] = lf_df
        # Ensure the generic linefill reference uses the volumetric table so
        # runs and saved cases reflect current edits.
        st.session_state["linefill_df"] = lf_df
    elif mode == "Daily Pumping Schedule":
        st.markdown("**Linefill at 07:00 Hrs (Volumetric)**")
        if "linefill_vol_df" not in st.session_state:
            st.session_state["linefill_vol_df"] = pd.DataFrame({
                "Product": ["Product-1", "Product-2", "Product-3"],
                "Volume (m³)": [50000.0, 40000.0, 15000.0],
                "Viscosity (cSt)": [5.0, 12.0, 15.0],
                "Density (kg/m³)": [810.0, 825.0, 865.0],
                INIT_DRA_COL: [0.0] * 3,
            })
        else:
            ensure_initial_dra_column(st.session_state["linefill_vol_df"], default=0.0, fill_blanks=True)
        lf_df = st.data_editor(
            st.session_state["linefill_vol_df"],
            num_rows="dynamic",
            key="linefill_vol_editor",
        )
        lf_df = ensure_initial_dra_column(lf_df, default=0.0, fill_blanks=True)
        st.session_state["linefill_vol_df"] = lf_df
        st.session_state["linefill_df"] = lf_df
        st.markdown("**Pumping Plan for the Day (Order of Pumping)**")
        if "day_plan_df" not in st.session_state:
            st.session_state["day_plan_df"] = pd.DataFrame({
                "Product": ["Product-4", "Product-5", "Product-6", "Product-7"],
                "Volume (m³)": [12000.0, 6000.0, 10000.0, 8000.0],
                "Viscosity (cSt)": [3.0, 10.0, 15.0, 4.0],
                "Density (kg/m³)": [800.0, 840.0, 880.0, 770.0],
                INIT_DRA_COL: [0.0] * 4,
            })
        else:
            ensure_initial_dra_column(st.session_state["day_plan_df"], default=0.0, fill_blanks=True)
        day_df = st.data_editor(
            st.session_state["day_plan_df"],
            num_rows="dynamic",
            key="day_plan_editor",
        )
        st.session_state["day_plan_df"] = ensure_initial_dra_column(day_df, default=0.0, fill_blanks=True)
        hourly_flow = st.number_input(
            "Hourly flow rate (m³/hr)",
            value=st.session_state.get("hourly_flow", 1000.0),
            step=10.0,
        )
        st.session_state["hourly_flow"] = hourly_flow
    else:
        st.markdown("**Linefill at 07:00 Hrs (Volumetric)**")
        if "linefill_vol_df" not in st.session_state:
            st.session_state["linefill_vol_df"] = pd.DataFrame({
                "Product": ["Product-1", "Product-2", "Product-3"],
                "Volume (m³)": [50000.0, 40000.0, 15000.0],
                "Viscosity (cSt)": [5.0, 12.0, 15.0],
                "Density (kg/m³)": [810.0, 825.0, 865.0],
                INIT_DRA_COL: [0.0] * 3,
            })
        else:
            ensure_initial_dra_column(st.session_state["linefill_vol_df"], default=0.0, fill_blanks=True)
        lf_df = st.data_editor(
            st.session_state["linefill_vol_df"],
            num_rows="dynamic",
            key="linefill_vol_editor",
        )
        lf_df = ensure_initial_dra_column(lf_df, default=0.0, fill_blanks=True)
        st.session_state["linefill_vol_df"] = lf_df
        st.session_state["linefill_df"] = lf_df
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
                "Flow (m³/h)": st.column_config.NumberColumn("Flow (m³/h)"),
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
        proj_df = st.data_editor(
            st.session_state["proj_plan_df"],
            num_rows="dynamic",
            key="proj_plan_editor",
            column_config={
                "Product": st.column_config.TextColumn("Product"),
                "Volume (m³)": st.column_config.NumberColumn("Volume (m³)"),
                "Viscosity (cSt)": st.column_config.NumberColumn("Viscosity (cSt)"),
                "Density (kg/m³)": st.column_config.NumberColumn("Density (kg/m³)"),
            },
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
        st.markdown("**Loopline (optional)**")
        has_loop = st.checkbox("Has Loopline?", value=bool(stn.get('loopline')), key=f"loopflag{idx}")
        if has_loop:
            loop = stn.setdefault('loopline', {})
            lcol1, lcol2, lcol3 = st.columns(3)
            with lcol1:
                loop['name'] = st.text_input("Name", value=loop.get('name', f"Loop {idx}"), key=f"loopname{idx}")
                loop['start_km'] = st.number_input("Start (km)", value=loop.get('start_km', 0.0), key=f"loopstart{idx}")
                loop['end_km'] = st.number_input("End (km)", value=loop.get('end_km', stn['L']), key=f"loopend{idx}")
                loop['L'] = st.number_input("Length (km)", value=loop.get('L', stn['L']), key=f"loopL{idx}")
            with lcol2:
                Dloop_in = st.number_input("OD (in)", value=loop.get('D', stn['D'])/0.0254, format="%.2f", step=0.01, key=f"loopD{idx}")
                tloop_in = st.number_input("Wall Thk (in)", value=loop.get('t', stn['t'])/0.0254, format="%.3f", step=0.001, key=f"loopt{idx}")
                loop['D'] = Dloop_in * 0.0254
                loop['t'] = tloop_in * 0.0254
                loop['SMYS'] = st.number_input("SMYS (psi)", value=loop.get('SMYS', stn['SMYS']), step=1000.0, key=f"loopSMYS{idx}")
            with lcol3:
                loop['rough'] = st.number_input("Pipe Roughness (m)", value=loop.get('rough', 0.00004), format="%.7f", step=0.0000001, key=f"looprough{idx}")
                loop['max_dr'] = st.number_input("Max Drag Reduction (%)", value=loop.get('max_dr', 0.0), key=f"loopmdr{idx}")
                loop['elev'] = st.number_input("Elevation (m)", value=loop.get('elev', stn.get('elev',0.0)), step=0.1, key=f"loopelev{idx}")

            loop_peak_key = f"loop_peak_data_{idx}"
            if loop_peak_key not in st.session_state or not isinstance(st.session_state[loop_peak_key], pd.DataFrame):
                st.session_state[loop_peak_key] = pd.DataFrame({
                    "Location (km)": [loop.get('L', stn['L'])/2.0],
                    "Elevation (m)": [loop.get('elev', stn.get('elev',0.0)) + 100.0]
                })
            loop_peak_df = st.data_editor(
                st.session_state[loop_peak_key],
                num_rows="dynamic",
                key=f"{loop_peak_key}_editor",
            )
            st.session_state[loop_peak_key] = loop_peak_df
            loop['peaks'] = loop_peak_df.to_dict(orient="records")
        else:
            stn.pop('loopline', None)

        tabs = st.tabs(["Pump", "Peaks"])
        with tabs[0]:
            if stn['is_pump']:
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
                                max_value=3,
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
                                min_label = "Min Pump RPM" if ptype_sel == "Diesel" else "Min RPM"
                                rated_label = "Rated Pump RPM" if ptype_sel == "Diesel" else "Rated RPM"
                                minrpm = st.number_input(min_label, value=pdata.get('MinRPM', 1000.0), key=f"minrpm{idx}{ptype}")
                                dol = st.number_input(rated_label, value=pdata.get('DOL', 1500.0), key=f"dol{idx}{ptype}")
                            with pcol3:
                                if ptype_sel == "Grid":
                                    tariff_mode = st.radio(
                                        "Tariff",
                                        ["Fixed", "Varying"],
                                        index=0 if not pdata.get('tariffs') else 1,
                                        key=f"tmode{idx}{ptype}"
                                    )
                                    if tariff_mode == "Fixed":
                                        rate = st.number_input("Elec Rate (INR/kWh)", value=pdata.get('rate', 9.0), key=f"rate{idx}{ptype}")
                                        tariffs = []
                                    else:
                                        default_rows = [
                                            {"rate": 9.0, "start": "07:00", "end": "11:00"},
                                            {"rate": 9.0, "start": "11:00", "end": "15:00"},
                                            {"rate": 9.0, "start": "15:00", "end": "19:00"},
                                            {"rate": 9.0, "start": "19:00", "end": "23:00"},
                                            {"rate": 9.0, "start": "23:00", "end": "03:00"},
                                            {"rate": 9.0, "start": "03:00", "end": "07:00"},
                                        ]
                                        raw_tariffs = pdata.get('tariffs') or default_rows
                                        for tr in raw_tariffs:
                                            if isinstance(tr.get('start'), str):
                                                tr['start'] = dt.datetime.strptime(tr['start'], "%H:%M").time()
                                            if isinstance(tr.get('end'), str):
                                                tr['end'] = dt.datetime.strptime(tr['end'], "%H:%M").time()
                                        tdf = st.data_editor(
                                            pd.DataFrame(raw_tariffs),
                                            num_rows="dynamic",
                                            key=f"tariff{idx}{ptype}",
                                            column_config={
                                                "rate": st.column_config.NumberColumn("Rate"),
                                                "start": st.column_config.TimeColumn("Start"),
                                                "end": st.column_config.TimeColumn("End"),
                                            },
                                        )
                                        tariffs = []
                                        intervals = []
                                        valid = True
                                        for _, row in tdf.iterrows():
                                            start_raw = row.get("start")
                                            end_raw = row.get("end")
                                            rate_val = row.get("rate")
                                            if pd.isna(start_raw) or pd.isna(end_raw):
                                                valid = False
                                                continue
                                            if not isinstance(start_raw, str):
                                                start_raw = start_raw.strftime("%H:%M")
                                            if not isinstance(end_raw, str):
                                                end_raw = end_raw.strftime("%H:%M")
                                            try:
                                                sdt = dt.datetime.strptime(start_raw, "%H:%M")
                                                edt = dt.datetime.strptime(end_raw, "%H:%M")
                                            except ValueError:
                                                valid = False
                                                continue
                                            if edt <= sdt:
                                                edt += dt.timedelta(days=1)
                                            duration = (edt - sdt).total_seconds() / 3600.0
                                            intervals.append((sdt, edt))
                                            tariffs.append({
                                                "rate": float(rate_val),
                                                "start": start_raw,
                                                "end": end_raw,
                                                "duration": duration,
                                            })
                                        intervals.sort(key=lambda x: x[0])
                                        for i in range(1, len(intervals)):
                                            if intervals[i][0] < intervals[i-1][1]:
                                                valid = False
                                                break
                                        if not valid:
                                            st.warning("Invalid tariff windows: ensure HH:MM format and no overlaps.")
                                        rate = pdata.get('rate', 9.0)
                                    sfc_mode = "none"
                                    sfc = 0.0
                                    engine_params = {}
                                else:
                                    sfc_mode = st.radio(
                                        "SFC Input",
                                        ["Enter manually", "System calculated (ISO 3046)"],
                                        index=0 if pdata.get('sfc_mode', 'manual') == 'manual' else 1,
                                        key=f"sfc_mode{idx}{ptype}"
                                    )
                                    if sfc_mode == "Enter manually":
                                        sfc = st.number_input("SFC (gm/bhp·hr)", value=pdata.get('sfc', 150.0), key=f"sfc{idx}{ptype}")
                                        engine_params = {}
                                    else:
                                        engine_make = st.text_input("Engine Make", value=pdata.get('engine_params', {}).get('make', ''), key=f"emake{idx}{ptype}")
                                        engine_model = st.text_input("Engine Model", value=pdata.get('engine_params', {}).get('model', ''), key=f"emodel{idx}{ptype}")
                                        rated_power = st.number_input("Engine Rated Power (kW)", value=pdata.get('engine_params', {}).get('rated_power', 0.0), key=f"epower{idx}{ptype}")
                                        sfc50 = st.number_input("SFC at 50% load", value=pdata.get('engine_params', {}).get('sfc50', 0.0), key=f"sfc50{idx}{ptype}")
                                        sfc75 = st.number_input("SFC at 75% load", value=pdata.get('engine_params', {}).get('sfc75', 0.0), key=f"sfc75{idx}{ptype}")
                                        sfc100 = st.number_input("SFC at 100% load", value=pdata.get('engine_params', {}).get('sfc100', 0.0), key=f"sfc100{idx}{ptype}")
                                        if st.button("Compute SFC", key=f"comp_sfc{idx}{ptype}"):
                                            pump_bkw = rated_power * 0.98
                                            sfc_calc = pipeline_model._compute_iso_sfc(
                                                {'engine_params': {'rated_power': rated_power, 'sfc50': sfc50, 'sfc75': sfc75, 'sfc100': sfc100}},
                                                dol,
                                                pump_bkw,
                                                dol,
                                                stn.get('elev', 0.0),
                                                st.session_state.get('Ambient_temp', 25.0),
                                            )
                                            st.session_state[f"sfc_display{idx}{ptype}"] = sfc_calc
                                        sfc = st.session_state.get(f"sfc_display{idx}{ptype}", 0.0)
                                        if sfc:
                                            st.write(f"Computed SFC at 100% load: {sfc:.2f} gm/bhp·hr")
                                        engine_params = {
                                            'make': engine_make,
                                            'model': engine_model,
                                            'rated_power': rated_power,
                                            'sfc50': sfc50,
                                            'sfc75': sfc75,
                                            'sfc100': sfc100,
                                        }
                                    rate = 0.0
                                    tariffs = []

                            stn.setdefault('pump_types', {})[ptype] = {
                                'names': names,
                                'name': names[0] if names else f'Pump {ptype}',
                                'head_data': df_head,
                                'eff_data': df_eff,
                                'power_type': ptype_sel,
                                'MinRPM': minrpm,
                                'DOL': dol,
                                'rate': rate,
                                'tariffs': tariffs,
                                'sfc_mode': 'manual' if sfc_mode == "Enter manually" else ('iso' if ptype_sel == 'Diesel' else 'none'),
                                'sfc': sfc,
                                'engine_params': engine_params,
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
            if 'pump_types' in stn:
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
                if isinstance(dfh, pd.DataFrame):
                    stn['head_data'] = dfh.to_dict(orient="records")
                if isinstance(dfe, pd.DataFrame):
                    stn['eff_data'] = dfe.to_dict(orient="records")

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
        proj_plan = plan_df.to_dict(orient="records")
    else:
        proj_plan = []

    stations = st.session_state.get('stations', [])
    first_station = stations[0] if stations else {}
    return {
        "stations": stations,
        "terminal": {
            "name": st.session_state.get('terminal_name', 'Terminal'),
            "elev": st.session_state.get('terminal_elev', 0.0),
            "min_residual": st.session_state.get('terminal_head', 50.0)
        },
        "FLOW": st.session_state.get('FLOW', 1000.0),
        "RateDRA": st.session_state.get('RateDRA', 500.0),
        "Price_HSD": st.session_state.get('Price_HSD', 70.0),
        "Fuel_density": st.session_state.get('Fuel_density', 820.0),
        "Ambient_temp": st.session_state.get('Ambient_temp', 25.0),
        "MOP_kgcm2": st.session_state.get('MOP_kgcm2', 100.0),
        "op_mode": st.session_state.get('op_mode', "Flow rate"),
        "linefill": st.session_state.get('linefill_df', pd.DataFrame()).to_dict(orient="records"),
        "linefill_vol": st.session_state.get('linefill_vol_df', pd.DataFrame()).to_dict(orient="records"),
        "day_plan": st.session_state.get('day_plan_df', pd.DataFrame()).to_dict(orient="records"),
        "proj_flow": proj_flow,
        "proj_plan": proj_plan,
        "planner_days": st.session_state.get('planner_days', 1.0),
        "pump_shear_rate": st.session_state.get('pump_shear_rate', 0.0),
        # --- Pump curve data ---
        **{
            f"head_data_{i+1}": (
                st.session_state.get(f"head_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"head_data_{i+1}"), pd.DataFrame)
                else stations[i].get('head_data')
            )
            for i in range(len(stations))
        },
        **{
            f"head_data_{1}{ptype}": (
                st.session_state.get(f"head_data_{1}{ptype}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"head_data_{1}{ptype}"), pd.DataFrame)
                else first_station.get('pump_types', {}).get(ptype, {}).get('head_data')
            )
            for ptype in ['A', 'B']
        },
        **{
            f"eff_data_{i+1}": (
                st.session_state.get(f"eff_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"eff_data_{i+1}"), pd.DataFrame)
                else stations[i].get('eff_data')
            )
            for i in range(len(stations))
        },
        **{
            f"eff_data_{1}{ptype}": (
                st.session_state.get(f"eff_data_{1}{ptype}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"eff_data_{1}{ptype}"), pd.DataFrame)
                else first_station.get('pump_types', {}).get(ptype, {}).get('eff_data')
            )
            for ptype in ['A', 'B']
        },
        **{
            f"peak_data_{i+1}": (
                st.session_state.get(f"peak_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"peak_data_{i+1}"), pd.DataFrame)
                else stations[i].get('peak_data')
            )
            for i in range(len(stations))
        },
        **{
            f"peak_data_{1}{ptype}": (
                st.session_state.get(f"peak_data_{1}{ptype}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"peak_data_{1}{ptype}"), pd.DataFrame)
                else first_station.get('pump_types', {}).get(ptype, {}).get('peak_data')
            )
            for ptype in ['A', 'B']
        },
        **{
            f"loop_peak_data_{i+1}": (
                st.session_state.get(f"loop_peak_data_{i+1}").to_dict(orient="records")
                if isinstance(st.session_state.get(f"loop_peak_data_{i+1}"), pd.DataFrame)
                else stations[i].get('loopline', {}).get('peaks')
            )
            for i in range(len(stations))
        }
    }


case_data = get_full_case_dict()
st.sidebar.download_button(
    label="💾 Save Case",
    data=json.dumps(case_data, indent=2),
    file_name="pipeline_case.json",
    mime="application/json"
)

def _default_segment_slices(
    stations: list[dict], kv_list: list[float], rho_list: list[float]
) -> list[list[dict]]:
    """Return single-slice segment profiles for legacy distance tables.

    Each station (segment) receives a single ``{"length_km", "kv", "rho"}``
    dictionary so downstream consumers such as the hydraulic solver receive a
    well-formed slice list even when only aggregate properties are available.
    """

    if not stations:
        return []

    fallback_kv = kv_list[0] if kv_list else 1.0
    fallback_rho = rho_list[0] if rho_list else 850.0
    slices: list[list[dict]] = []
    for idx, stn in enumerate(stations):
        length = float(stn.get("L", 0.0) or 0.0)
        kv = kv_list[idx] if idx < len(kv_list) else fallback_kv
        rho = rho_list[idx] if idx < len(rho_list) else fallback_rho
        slices.append(
            [
                {
                    "length_km": length,
                    "kv": float(kv),
                    "rho": float(rho),
                }
            ]
        )
    return slices


def derive_segment_profiles(
    linefill_df: pd.DataFrame | None, stations: list[dict]
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Return per-segment viscosity/density lists and batch slices."""

    kv_list, rho_list, segment_slices = map_linefill_to_segments(linefill_df, stations)
    return kv_list, rho_list, segment_slices


def map_linefill_to_segments(
    linefill_df, stations
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Map linefill properties onto each pipeline segment.

    Accepts either a tabular linefill with "Start/End (km)" columns or a
    volumetric table containing "Volume (m³)" information. In the latter case,
    :func:`map_vol_linefill_to_segments` is used to derive segment properties.
    """

    if linefill_df is None or len(linefill_df) == 0:
        # When no linefill information is provided, return conservative defaults
        # rather than zeros.  Zero viscosities and densities result in
        # non-physical conditions that cause the optimiser to reject all
        # scenarios.  Here we assume a light refined product of 1.0 cSt and
        # density 850 kg/m³ across all segments.
        kv_list = [1.0] * len(stations)
        rho_list = [850.0] * len(stations)
        segment_slices = _default_segment_slices(stations, kv_list, rho_list)
        return kv_list, rho_list, segment_slices

    cols = set(linefill_df.columns)

    # If Start/End columns are missing but volumetric info is present,
    # delegate to volumetric mapper and return directly.
    if "Start (km)" not in cols or "End (km)" not in cols:
        if "Volume (m³)" in cols or "Volume" in cols:
            return map_vol_linefill_to_segments(linefill_df, stations)
        # Fallback: assume uniform properties from the last row
        kv = float(linefill_df.iloc[-1].get("Viscosity (cSt)", 0.0))
        rho = float(linefill_df.iloc[-1].get("Density (kg/m³)", 0.0))
        kv_list = [kv] * len(stations)
        rho_list = [rho] * len(stations)
        segment_slices = _default_segment_slices(stations, kv_list, rho_list)
        return kv_list, rho_list, segment_slices

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
    segment_slices = _default_segment_slices(stations, viscs, dens)
    return viscs, dens, segment_slices

# ==== NEW: Volumetric linefill helpers ====
def pipe_cross_section_area_m2(stations: list[dict]) -> float:
    """Return pipe internal cross-sectional area (m²) using the first station's D/t."""
    if not stations:
        return 0.0
    D = float(stations[0].get("D", 0.711))
    t = float(stations[0].get("t", 0.007))
    d_inner = max(D - 2.0*t, 0.0)
    return float((pi * d_inner**2) / 4.0)

def map_vol_linefill_to_segments(
    vol_table: pd.DataFrame, stations: list[dict]
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Convert a volumetric linefill table to per-segment fluid properties.

    The returned tuple contains three parallel sequences with one entry per
    segment (station):

    ``kv_list``
        The viscosity of the upstream-most batch touching the segment.  This
        preserves the historical behaviour used by DRA lookups and other UI
        features that expect a single representative viscosity per segment.

    ``rho_list``
        A length-weighted average density across the batches occupying the
        segment.  This is used when converting heads to pressures.

    ``segment_slices``
        A list of ``{"length_km", "kv", "rho"}`` dictionaries describing the
        sequence of batches that span the segment.  These slices are later fed
        into the hydraulic solver so Darcy–Weisbach losses can be accumulated
        over the heterogeneous fluid profile instead of assuming a single
        viscosity.

    Assumes uniform diameter along the pipeline (uses first station ``D``/``t``)
    to convert volumes to kilometres.
    """
    A = pipe_cross_section_area_m2(stations)
    if A <= 0:
        raise ValueError("Invalid pipe area (check D and t).")
    d_inner = sqrt((4.0 * A) / pi)
    km_from_volume = pipeline_model._km_from_volume

    # Compute lengths occupied by each batch
    # Expected columns: Product, Volume (m³), Viscosity (cSt), Density (kg/m³)
    batches = []
    for _, r in vol_table.iterrows():
        vol = float(r.get("Volume (m³)", 0.0) or r.get("Volume", 0.0) or 0.0)
        if vol <= 0:
            continue
        length_km = km_from_volume(vol, d_inner)
        visc = float(r.get("Viscosity (cSt)", 0.0))
        dens = float(r.get("Density (kg/m³)", 0.0))
        batches.append({"len_km": length_km, "kv": visc, "rho": dens})

    # Map to segments (each station defines a segment length L)
    seg_kv: list[float] = []
    seg_rho: list[float] = []
    seg_slices: list[list[dict]] = []
    seg_lengths = [s.get("L", 0.0) for s in stations]
    i_batch = 0
    remaining = batches[0]["len_km"] if batches else 0.0
    kv_cur = batches[0]["kv"] if batches else 1.0
    rho_cur = batches[0]["rho"] if batches else 850.0

    for L in seg_lengths:
        need = L
        if L <= 0:
            seg_kv.append(kv_cur)
            seg_rho.append(rho_cur)
            seg_slices.append([])
            continue
        segment_entries: list[dict] = []
        # Consume from batches until we cover this segment upstream-to-downstream
        while need > 1e-9:
            if remaining <= 1e-9:
                i_batch += 1
                if i_batch >= len(batches):
                    # If we ran out, extend with last known properties
                    segment_entries.append({
                        "length_km": need,
                        "kv": kv_cur,
                        "rho": rho_cur,
                    })
                    need = 0.0
                    break
                else:
                    remaining = batches[i_batch]["len_km"]
                    kv_cur = batches[i_batch]["kv"]
                    rho_cur = batches[i_batch]["rho"]
                    if remaining <= 1e-9:
                        continue
            take = min(need, remaining)
            if take <= 0:
                break
            segment_entries.append({
                "length_km": take,
                "kv": kv_cur,
                "rho": rho_cur,
            })
            need -= take
            remaining -= take
        if not segment_entries:
            segment_entries.append({
                "length_km": L,
                "kv": kv_cur,
                "rho": rho_cur,
            })
        seg_slices.append(segment_entries)
        seg_kv.append(segment_entries[0]["kv"])
        if L > 0:
            avg_rho = sum(entry["length_km"] * entry["rho"] for entry in segment_entries) / L
        else:
            avg_rho = segment_entries[0]["rho"]
        seg_rho.append(avg_rho)

    return seg_kv, seg_rho, seg_slices


def _normalise_segment_entries(
    entries: list[dict] | None,
    seg_length: float,
    kv_fill: float,
    rho_fill: float,
) -> list[dict]:
    """Return cleaned slice entries that exactly span ``seg_length``."""

    tol = 1e-9
    if seg_length <= tol:
        return []

    cleaned: list[dict] = []
    total = 0.0
    for entry in entries or []:
        try:
            length = float(entry.get("length_km", 0.0) or 0.0)
        except (TypeError, ValueError):
            length = 0.0
        if length <= tol:
            continue
        remaining = seg_length - total
        if remaining <= tol:
            break
        take = min(length, remaining)
        try:
            kv_val = float(entry.get("kv", kv_fill))
        except (TypeError, ValueError):
            kv_val = kv_fill
        try:
            rho_val = float(entry.get("rho", rho_fill))
        except (TypeError, ValueError):
            rho_val = rho_fill
        cleaned.append({"length_km": take, "kv": kv_val, "rho": rho_val})
        total += take
        if total >= seg_length - tol:
            break

    remaining = seg_length - total
    if remaining > tol:
        cleaned.append({"length_km": remaining, "kv": kv_fill, "rho": rho_fill})

    return cleaned


def _compress_slice_entries(entries: list[dict], tol: float = 1e-9) -> list[dict]:
    """Merge adjacent slices with identical fluid properties."""

    if not entries:
        return []

    compressed: list[dict] = [dict(entries[0])]
    for entry in entries[1:]:
        if (
            isclose(entry.get("kv", 0.0), compressed[-1].get("kv", 0.0), rel_tol=0.0, abs_tol=1e-9)
            and isclose(entry.get("rho", 0.0), compressed[-1].get("rho", 0.0), rel_tol=0.0, abs_tol=1e-9)
        ):
            compressed[-1]["length_km"] += entry.get("length_km", 0.0)
        else:
            compressed.append(dict(entry))

    return [entry for entry in compressed if entry.get("length_km", 0.0) > tol]


def _merge_segment_profiles(
    current_entries: list[dict] | None,
    future_entries: list[dict] | None,
    kv_max: float,
    rho_max: float,
    seg_length: float,
) -> list[dict]:
    """Combine two slice sequences into a worst-case profile for a segment."""

    tol = 1e-9
    if seg_length <= tol:
        return []

    normalised_now = _normalise_segment_entries(current_entries, seg_length, kv_max, rho_max)
    normalised_next = _normalise_segment_entries(future_entries, seg_length, kv_max, rho_max)
    if not normalised_now and not normalised_next:
        return [{"length_km": seg_length, "kv": kv_max, "rho": rho_max}]

    idx_now = 0
    idx_next = 0
    rem_now = normalised_now[0]["length_km"] if normalised_now else seg_length
    kv_now = normalised_now[0]["kv"] if normalised_now else kv_max
    rho_now = normalised_now[0]["rho"] if normalised_now else rho_max
    rem_next = normalised_next[0]["length_km"] if normalised_next else seg_length
    kv_next = normalised_next[0]["kv"] if normalised_next else kv_max
    rho_next = normalised_next[0]["rho"] if normalised_next else rho_max
    total = 0.0
    merged: list[dict] = []

    while total < seg_length - tol:
        if rem_now <= tol:
            idx_now += 1
            if idx_now < len(normalised_now):
                current = normalised_now[idx_now]
                rem_now = current["length_km"]
                kv_now = current["kv"]
                rho_now = current["rho"]
            else:
                rem_now = seg_length - total
                kv_now = kv_max
                rho_now = rho_max
        if rem_next <= tol:
            idx_next += 1
            if idx_next < len(normalised_next):
                current = normalised_next[idx_next]
                rem_next = current["length_km"]
                kv_next = current["kv"]
                rho_next = current["rho"]
            else:
                rem_next = seg_length - total
                kv_next = kv_max
                rho_next = rho_max

        step = min(rem_now, rem_next, seg_length - total)
        if step <= tol:
            break

        kv_entry = max(kv_now, kv_next)
        rho_entry = max(rho_now, rho_next)
        merged.append({"length_km": step, "kv": kv_entry, "rho": rho_entry})
        rem_now -= step
        rem_next -= step
        total += step

    remaining = seg_length - total
    if remaining > tol:
        merged.append({"length_km": remaining, "kv": kv_max, "rho": rho_max})

    return _compress_slice_entries(merged)


def combine_volumetric_profiles(
    stations: list[dict],
    current_vol: pd.DataFrame,
    future_vol: pd.DataFrame,
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Return worst-case viscosity/density lists and slices for scheduling."""

    kv_now, rho_now, slices_now = map_vol_linefill_to_segments(current_vol, stations)
    kv_next, rho_next, slices_next = map_vol_linefill_to_segments(future_vol, stations)

    kv_list: list[float] = []
    rho_list: list[float] = []
    segment_slices: list[list[dict]] = []
    num_segments = len(stations)

    for idx in range(num_segments):
        kv_cur = kv_now[idx] if idx < len(kv_now) else (kv_now[-1] if kv_now else 1.0)
        kv_future = kv_next[idx] if idx < len(kv_next) else (kv_next[-1] if kv_next else kv_cur)
        rho_cur = rho_now[idx] if idx < len(rho_now) else (rho_now[-1] if rho_now else 850.0)
        rho_future = rho_next[idx] if idx < len(rho_next) else (rho_next[-1] if rho_next else rho_cur)
        kv_max = max(kv_cur, kv_future)
        rho_max = max(rho_cur, rho_future)
        seg_length = float(stations[idx].get("L", 0.0) or 0.0)
        entries_now = slices_now[idx] if idx < len(slices_now) else []
        entries_next = slices_next[idx] if idx < len(slices_next) else []
        merged = _merge_segment_profiles(entries_now, entries_next, kv_max, rho_max, seg_length)
        segment_slices.append(merged)
        kv_list.append(kv_max)
        rho_list.append(rho_max)

    return kv_list, rho_list, segment_slices


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
    vol_table = ensure_initial_dra_column(vol_table.copy(), default=0.0, fill_blanks=True)
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
        day_plan = ensure_initial_dra_column(day_plan.copy(), default=0.0, fill_blanks=True)
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
            ppm_value = day_plan.at[j, INIT_DRA_COL] if INIT_DRA_COL in day_plan.columns else 0.0
            try:
                ppm_value = float(ppm_value)
            except (TypeError, ValueError):
                ppm_value = 0.0
            if pd.isna(ppm_value):
                ppm_value = 0.0
            batch[INIT_DRA_COL] = ppm_value
            vol_table = pd.concat([pd.DataFrame([batch]), vol_table], ignore_index=True)
            day_plan.at[j, "Volume (m³)"] = v - take
            added -= take
            if day_plan.at[j, "Volume (m³)"] <= 1e-9:
                j += 1
        day_plan = day_plan.iloc[j:].reset_index(drop=True)

    return vol_table, day_plan


def df_to_dra_linefill(df: pd.DataFrame) -> list[dict]:
    """Convert a volumetric linefill dataframe to a list of DRA batches."""
    if df is None or len(df) == 0:
        return []
    col_vol = "Volume (m³)" if "Volume (m³)" in df.columns else "Volume"
    col_ppm = None
    for candidate in (INIT_DRA_COL, "DRA ppm", "dra_ppm"):
        if candidate in df.columns:
            col_ppm = candidate
            break
    batches: list[dict] = []
    for _, r in df.iterrows():
        vol = float(r.get(col_vol, 0.0) or 0.0)
        if vol <= 0:
            continue
        raw_ppm = r.get(col_ppm) if col_ppm else None
        ppm: float
        if raw_ppm is None:
            ppm = 0.0
        elif isinstance(raw_ppm, str):
            stripped = raw_ppm.strip()
            if stripped == "":
                ppm = 0.0
            else:
                try:
                    ppm = float(stripped)
                except Exception:
                    ppm = 0.0
        else:
            try:
                ppm = float(raw_ppm)
            except Exception:
                ppm = 0.0
            else:
                if pd.isna(ppm):
                    ppm = 0.0
        batches.append({"volume": vol, "dra_ppm": ppm})
    return batches


def apply_dra_ppm(df: pd.DataFrame, dra_batches: list[dict]) -> pd.DataFrame:
    """Assign ``dra_ppm`` values from ``dra_batches`` onto ``df`` by volume.

    The returned dataframe may include additional rows when a product batch spans
    multiple queue slices; each split row inherits the source product metadata
    but carries only the portion of volume covered by the corresponding DRA
    segment.
    """

    if df is None:
        return df

    df = ensure_initial_dra_column(df.copy(), default=0.0, fill_blanks=True)
    if "DRA ppm" not in df.columns:
        df["DRA ppm"] = df[INIT_DRA_COL]

    col_vol = "Volume (m³)" if "Volume (m³)" in df.columns else "Volume"

    queue: list[tuple[float, float]] = []
    for batch in dra_batches or []:
        try:
            vol = float(batch.get("volume", 0.0) or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol <= 1e-9:
            continue
        try:
            ppm_val = float(batch.get("dra_ppm", 0.0) or 0.0)
        except (TypeError, ValueError):
            ppm_val = 0.0
        if pd.isna(ppm_val):
            ppm_val = 0.0
        queue.append((vol, ppm_val))

    queue_idx = 0
    if queue:
        seg_remaining = queue[0][0]
        ppm_cur = queue[0][1]
    else:
        seg_remaining = float("inf")
        ppm_cur = 0.0

    def advance_segment() -> None:
        nonlocal queue_idx, seg_remaining, ppm_cur
        queue_idx += 1
        if queue_idx < len(queue):
            seg_remaining = queue[queue_idx][0]
            ppm_cur = queue[queue_idx][1]
        else:
            seg_remaining = float("inf")
            ppm_cur = 0.0

    rows_out: list[dict] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            row_vol = float(row_dict.get(col_vol, 0.0) or 0.0)
        except (TypeError, ValueError):
            row_vol = 0.0

        if row_vol <= 1e-9:
            row_dict["DRA ppm"] = ppm_cur
            row_dict[INIT_DRA_COL] = ppm_cur
            rows_out.append(row_dict)
            continue

        remaining_row = row_vol
        while remaining_row > 1e-9:
            while queue_idx < len(queue) and seg_remaining <= 1e-9:
                advance_segment()

            take = min(remaining_row, seg_remaining)
            if take <= 1e-9:
                take = remaining_row

            row_copy = row_dict.copy()
            row_copy[col_vol] = take
            row_copy["DRA ppm"] = ppm_cur
            row_copy[INIT_DRA_COL] = ppm_cur
            rows_out.append(row_copy)

            remaining_row -= take
            if queue_idx < len(queue):
                seg_remaining -= take
                if seg_remaining <= 1e-9:
                    advance_segment()

    if not rows_out:
        return df.iloc[0:0].copy()

    result_df = pd.DataFrame(rows_out, columns=df.columns)
    return result_df.reset_index(drop=True)


# Build a summary dataframe from solver results


def normalize_speed_suffix(label: str) -> str:
    """Normalise pump-type identifiers for use in speed keys."""

    if not isinstance(label, str):
        label = str(label or "")
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    cleaned = cleaned.strip("_")
    return cleaned.upper() if cleaned else "DEFAULT"


def get_speed_suffixes_and_values(
    res: dict,
    key: str,
    station: dict | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Return ordered suffixes and corresponding per-type speeds."""

    prefix = f"speed_{key}_"
    key_map: dict[str, str] = {}
    order_extra: list[str] = []
    for rkey in res.keys():
        if isinstance(rkey, str) and rkey.startswith(prefix):
            raw_suffix = rkey[len(prefix) :]
            norm = normalize_speed_suffix(raw_suffix)
            if norm not in key_map:
                key_map[norm] = rkey
                order_extra.append(norm)

    suffixes: list[str] = []
    seen: set[str] = set()
    if station:
        pump_types = station.get("pump_types")
        if isinstance(pump_types, dict) and pump_types:
            for ptype in pump_types.keys():
                norm = normalize_speed_suffix(ptype)
                if norm not in seen:
                    suffixes.append(norm)
                    seen.add(norm)
        else:
            pump_type_single = station.get("pump_type")
            if pump_type_single:
                norm = normalize_speed_suffix(pump_type_single)
                if norm not in seen:
                    suffixes.append(norm)
                    seen.add(norm)

    for norm in order_extra:
        if norm not in seen:
            suffixes.append(norm)
            seen.add(norm)

    values: dict[str, float] = {}
    for norm in suffixes:
        actual_key = key_map.get(norm)
        val = None
        if actual_key:
            val = res.get(actual_key)
        if val is None:
            val = res.get(f"{prefix}{norm}")
        if val is None and norm.lower() != norm:
            val = res.get(f"{prefix}{norm.lower()}")
        try:
            values[norm] = float(val)
        except (TypeError, ValueError):
            values[norm] = np.nan
    return suffixes, values


def get_speed_display_map(
    res: dict,
    key: str,
    station: dict | None = None,
) -> OrderedDict[str, float]:
    """Return an ordered mapping of per-type pump speeds for a station."""

    suffixes, values = get_speed_suffixes_and_values(res, key, station)
    aggregated = res.get(f"speed_{key}")
    aggregated_val: float | None
    try:
        aggregated_val = float(aggregated)
    except (TypeError, ValueError):
        aggregated_val = None
    else:
        if isinstance(aggregated_val, float) and np.isnan(aggregated_val):
            aggregated_val = None

    if not suffixes:
        fallback = None
        if station:
            pump_types = station.get("pump_types")
            if isinstance(pump_types, dict) and pump_types:
                fallback = next(iter(pump_types.keys()), None)
            if fallback is None:
                fallback = station.get("pump_type")
        if fallback is None:
            fallback = "DEFAULT"
        norm = normalize_speed_suffix(fallback)
        suffixes = [norm]
        if aggregated_val is not None:
            values[norm] = aggregated_val
        else:
            values.setdefault(norm, np.nan)
    elif aggregated_val is not None:
        for norm in suffixes:
            current = values.get(norm)
            if current is None or (isinstance(current, float) and np.isnan(current)):
                values[norm] = aggregated_val

    speed_map: OrderedDict[str, float] = OrderedDict()
    for norm in suffixes:
        speed_map[norm] = values.get(norm, np.nan)
    return speed_map


def add_speed_columns(row: dict, res: dict, station: dict | None, prefix: str = "Speed") -> None:
    """Populate ``row`` with per-type speed columns for ``station``."""

    if not station:
        return
    key_raw = station.get("name", "")
    key = key_raw.lower().replace(" ", "_")
    speed_map = get_speed_display_map(res, key, station)
    label_base = station.get("name", key_raw)
    for suffix, value in speed_map.items():
        column = f"{prefix} {label_base} ({suffix})"
        row[column] = value if not (isinstance(value, float) and np.isnan(value)) else np.nan


def build_summary_dataframe(
    res: dict,
    stations_data: list[dict],
    linefill_df: pd.DataFrame | None,
    terminal: dict | None = None,
    drop_unused: bool = True,
) -> pd.DataFrame:
    """Create station-wise summary table matching the Optimization Results view."""

    if linefill_df is not None and len(linefill_df):
        if "Start (km)" in linefill_df.columns:
            kv_list, _, _ = map_linefill_to_segments(linefill_df, stations_data)
        else:
            kv_list, _, _ = map_vol_linefill_to_segments(linefill_df, stations_data)
    else:
        kv_list = [0.0] * len(stations_data)

    names = [s['name'] for s in stations_data]
    if terminal is not None:
        names.append(terminal.get('name', 'Terminal'))
    keys = [n.lower().replace(' ', '_') for n in names]

    station_ppm = {}
    for idx, nm in enumerate(names):
        key = keys[idx]
        if idx < len(stations_data):
            station_ppm[key] = res.get(f"dra_ppm_{key}", 0.0)
        else:
            station_ppm[key] = np.nan

    segment_flows = [res.get(f"pipeline_flow_{k}", np.nan) for k in keys]
    loop_flows = [res.get(f"loopline_flow_{k}", np.nan) for k in keys]
    pump_flows = [res.get(f"pump_flow_{k}", np.nan) for k in keys]

    station_speed_maps: dict[str, OrderedDict[str, float]] = {}
    speed_suffixes: list[str] = []
    seen_suffixes: set[str] = set()
    for idx, stn in enumerate(stations_data):
        key = keys[idx]
        speed_map = get_speed_display_map(res, key, stn)
        station_speed_maps[key] = speed_map
        for suffix in speed_map.keys():
            if suffix not in seen_suffixes:
                speed_suffixes.append(suffix)
                seen_suffixes.add(suffix)

    params_pre = [
        "Pipeline Flow (m³/hr)", "Loopline Flow (m³/hr)", "Pump Flow (m³/hr)", "Bypass Next?",
        "Power & Fuel Cost (INR)", "DRA Cost (INR)", "DRA PPM", "No. of Pumps",
    ]
    params_post = [
        "Pump Eff (%)", "Pump BKW (kW)", "Motor Input (kW)", "Reynolds No.", "Head Loss (m)",
        "Head Loss (kg/cm²)", "Vel (m/s)", "Residual Head (m)", "Residual Head (kg/cm²)",
        "SDH (m)", "SDH (kg/cm²)", "MAOP (m)", "MAOP (kg/cm²)", "Drag Reduction (%)"
    ]
    params = params_pre + [f"Pump Speed {suffix} (rpm)" for suffix in speed_suffixes] + params_post
    summary = {"Parameters": params}

    for idx, nm in enumerate(names):
        key = keys[idx]
        pre_values = [
            segment_flows[idx],
            loop_flows[idx],
            pump_flows[idx] if idx < len(pump_flows) and not pd.isna(pump_flows[idx]) else np.nan,
            res.get(f"bypass_next_{key}", 0),
            res.get(f"power_cost_{key}", 0.0),
            res.get(f"dra_cost_{key}", 0.0),
            station_ppm.get(key, np.nan),
            int(res.get(f"num_pumps_{key}", 0)),
        ]
        if idx < len(stations_data):
            speed_map = station_speed_maps.get(key, OrderedDict())
        else:
            speed_map = OrderedDict()
        speed_values = [speed_map.get(suffix, np.nan) for suffix in speed_suffixes]
        post_values = [
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
        summary[nm] = pre_values + speed_values + post_values

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
    df_sum = df_sum.round(2)
    if 'DRA PPM' in df_sum['Parameters'].values:
        idx = df_sum[df_sum['Parameters'] == 'DRA PPM'].index[0]
        for col in df_sum.columns:
            if col == 'Parameters':
                continue
            val = df_sum.at[idx, col]
            df_sum.at[idx, col] = val if float(val or 0) > 0 else np.nan
    return df_sum


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
        n_pumps = int(res.get(f"num_pumps_{key}", 0) or 0)

        combo = None
        if isinstance(stn, dict):
            combo = stn.get('active_combo') or stn.get('pump_combo')
            pump_types = stn.get('pump_types') or base_stn.get('pump_types')
        else:
            pump_types = base_stn.get('pump_types') if base_stn else None
        if combo is None and base_stn:
            combo = base_stn.get('active_combo') or base_stn.get('pump_combo')

        pump_name = ""
        if combo and pump_types:
            names_used: list[str] = []
            for ptype, count in combo.items():
                if count <= 0:
                    continue
                pnames = pump_types.get(ptype, {}).get('names', [])
                names_used.extend(pnames[:count])
            pump_name = ", ".join(names_used)
        else:
            pump_list = None
            if isinstance(stn, dict):
                pump_list = stn.get('pump_names') or base_stn.get('pump_names')
            else:
                pump_list = base_stn.get('pump_names') if base_stn else None
            if pump_list and n_pumps > 0:
                pump_name = ", ".join(pump_list[:n_pumps])
            else:
                pump_name = ""

        if origin_name and name != origin_name and name.startswith(origin_name):
            station_display = origin_name

        row = {
            'Station': station_display,
            'Pump Name': pump_name,
            'Pipeline Flow (m³/hr)': float(res.get(f"pipeline_flow_{key}", 0.0) or 0.0),
            'Loopline Flow (m³/hr)': float(res.get(f"loopline_flow_{key}", 0.0) or 0.0),
            'Pump Flow (m³/hr)': float(res.get(f"pump_flow_{key}", 0.0) or 0.0),
            'Power & Fuel Cost (INR)': float(res.get(f"power_cost_{key}", 0.0) or 0.0),
            'DRA Cost (INR)': float(res.get(f"dra_cost_{key}", 0.0) or 0.0),
            'DRA PPM': res.get(f"dra_ppm_{key}", 0.0),
            'Loop DRA PPM': res.get(f"dra_ppm_loop_{key}", 0.0),
            'No. of Pumps': n_pumps,
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
            'Loop Drag Reduction (%)': float(res.get(f"drag_reduction_loop_{key}", 0.0) or 0.0),
        }

        speed_station = base_stn if base_stn else (stn if isinstance(stn, dict) else None)
        speed_map = get_speed_display_map(res, key, speed_station)
        for suffix, value in speed_map.items():
            col_name = f"Pump Speed {suffix} (rpm)"
            try:
                row[col_name] = float(value)
            except (TypeError, ValueError):
                row[col_name] = np.nan

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
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].round(2)
    if 'DRA PPM' in df.columns:
        df['DRA PPM'] = df['DRA PPM'].apply(lambda x: x if float(x or 0) > 0 else 'NIL')
    if 'Loop DRA PPM' in df.columns:
        df['Loop DRA PPM'] = df['Loop DRA PPM'].apply(lambda x: x if float(x or 0) > 0 else 'NIL')
    return df


def display_pump_type_details(res: dict, stations: list[dict], heading: str | None = None) -> bool:
    """Render a table with per-pump-type metrics for stations with multiple types."""
    multi: list[tuple[str, str, list[dict]]] = []
    station_lookup = {
        s.get('name', '').lower().replace(' ', '_'): s for s in stations if isinstance(s, dict)
    }
    for stn in stations:
        key_raw = stn.get('name', '')
        key = key_raw.lower().replace(' ', '_')
        details = res.get(f"pump_details_{key}")
        if details is None:
            details = res.get(f"pump_details_{key_raw}", [])
        if isinstance(details, list) and len(details) > 1:
            name = stn.get('orig_name', key_raw)
            multi.append((name, key, details))

    if not multi:
        return False

    title = heading or "Pump Details by Type"
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    for name, key, details in multi:
        stn_info = station_lookup.get(key, {})
        speed_map = get_speed_display_map(res, key, stn_info)
        speed_vals: list[float] = []
        for detail in details:
            suffix = normalize_speed_suffix(detail.get("ptype", ""))
            speed_val = speed_map.get(suffix)
            if speed_val is None or (isinstance(speed_val, float) and np.isnan(speed_val)):
                speed_val = detail.get("rpm", 0.0)
            try:
                speed_vals.append(float(speed_val))
            except (TypeError, ValueError):
                speed_vals.append(0.0)
        df_pump = pd.DataFrame({
            "Pump Type": [d.get("ptype", f"Type {i+1}") for i, d in enumerate(details)],
            "Count": [d.get("count", 0) for d in details],
            "Pump Speed (rpm)": speed_vals,
            "Pump Eff (%)": [d.get("eff", 0.0) for d in details],
            "Pump BKW (kW)": [d.get("pump_bkw", 0.0) for d in details],
            "Motor Input (kW)": [d.get("prime_kw", 0.0) for d in details],
        })
        fmt = {c: "{:.2f}" for c in df_pump.columns if c not in ["Pump Type", "Count"]}
        st.markdown(f"**{name}**")
        st.dataframe(df_pump.style.format(fmt), width='stretch', hide_index=True)
    return True

# Persisted DRA lock from the reference hourly run
def lock_dra_in_stations_from_result(stations: list[dict], res: dict, kv_list: list[float]) -> list[dict]:
    """Freeze per-station DRA (as %DR) based on ppm chosen at the reference hour for each station.

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

def _collect_search_depth_kwargs() -> dict[str, float | int]:
    """Return validated search-depth parameters for backend solvers."""

    rpm_step_default = getattr(pipeline_model, "RPM_STEP", 25)
    dra_step_default = getattr(pipeline_model, "DRA_STEP", 2)
    coarse_multiplier_default = getattr(pipeline_model, "COARSE_MULTIPLIER", 5.0)
    state_top_k_default = getattr(pipeline_model, "STATE_TOP_K", 50)
    state_cost_margin_default = getattr(pipeline_model, "STATE_COST_MARGIN", 5000.0)

    rpm_step = int(st.session_state.get("search_rpm_step", rpm_step_default) or rpm_step_default)
    if rpm_step <= 0:
        rpm_step = rpm_step_default

    dra_step = int(st.session_state.get("search_dra_step", dra_step_default) or dra_step_default)
    if dra_step <= 0:
        dra_step = dra_step_default

    coarse_multiplier = float(
        st.session_state.get("search_coarse_multiplier", coarse_multiplier_default)
        or coarse_multiplier_default
    )
    if coarse_multiplier <= 0:
        coarse_multiplier = coarse_multiplier_default

    state_top_k = int(
        st.session_state.get("search_state_top_k", state_top_k_default)
        or state_top_k_default
    )
    if state_top_k <= 0:
        state_top_k = state_top_k_default

    state_cost_margin = float(
        st.session_state.get("search_state_cost_margin", state_cost_margin_default)
        or state_cost_margin_default
    )
    if state_cost_margin < 0:
        state_cost_margin = 0.0

    return {
        "rpm_step": rpm_step,
        "dra_step": dra_step,
        "coarse_multiplier": coarse_multiplier,
        "state_top_k": state_top_k,
        "state_cost_margin": state_cost_margin,
    }

def solve_pipeline(
    stations,
    terminal,
    FLOW,
    KV_list,
    rho_list,
    segment_slices,
    RateDRA,
    Price_HSD,
    Fuel_density,
    Ambient_temp,
    linefill_dict,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
    start_time: str = "00:00",
    pump_shear_rate: float | None = None,
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

    if pump_shear_rate is None:
        pump_shear_rate = st.session_state.get("pump_shear_rate", 0.0)

    try:
        # Delegate to the backend optimiser
        search_kwargs = _collect_search_depth_kwargs()
        if any(s.get('pump_types') for s in stations):
            res = pipeline_model.solve_pipeline_with_types(
                stations,
                terminal,
                FLOW,
                KV_list,
                rho_list,
                segment_slices,
                RateDRA,
                Price_HSD,
                Fuel_density,
                Ambient_temp,
                linefill_dict,
                dra_reach_km,
                mop_kgcm2,
                hours,
                start_time=start_time,
                pump_shear_rate=pump_shear_rate,
                **search_kwargs,
            )
        else:
            res = pipeline_model.solve_pipeline(
                stations,
                terminal,
                FLOW,
                KV_list,
                rho_list,
                segment_slices,
                RateDRA,
                Price_HSD,
                Fuel_density,
                Ambient_temp,
                linefill_dict,
                dra_reach_km,
                mop_kgcm2,
                hours,
                start_time=start_time,
                pump_shear_rate=pump_shear_rate,
                **search_kwargs,
            )
        # Append a human-readable flow pattern name based on loop usage
        if not res.get("error"):
            usage = res.get("loop_usage", [])
            # Build segment-based descriptors for each looped section
            seg_names = []
            for idx, stn in enumerate(stations):
                # Looplines are associated with the segment connecting this station to the next
                if stn.get('loopline') and idx < len(stations) - 1:
                    uval = usage[idx] if idx < len(usage) else 0
                    seg_label = f"{stn['name']}–{stations[idx+1]['name']}"
                    if uval == 1:
                        seg_names.append(f"Parallel on {seg_label}")
                    elif uval == 2:
                        seg_names.append(f"Bypass on {seg_label}")
                    elif uval == 3:
                        seg_names.append(f"Loop only on {seg_label}")
            if not seg_names:
                pattern_name = "Mainline Only"
            elif len(seg_names) == sum(1 for stn in stations if stn.get('loopline')) and all('Parallel' in n for n in seg_names):
                pattern_name = "Parallel on all loop segments"
            elif len(seg_names) == sum(1 for stn in stations if stn.get('loopline')) and all('Loop only' in n for n in seg_names):
                pattern_name = "Loop only on all loop segments"
            else:
                pattern_name = ' & '.join(seg_names)
            res['flow_pattern_name'] = pattern_name
        return res
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
    Price_HSD = st.number_input("Fuel Price (INR/L)", value=st.session_state.get("Price_HSD", 70.0), step=0.5, key="batch_diesel")
    Fuel_density = st.number_input("Fuel density (kg/m³)", value=st.session_state.get("Fuel_density", 820.0), step=1.0, key="batch_fuel_density")
    Ambient_temp = st.number_input("Ambient temperature (°C)", value=st.session_state.get("Ambient_temp", 25.0), step=1.0, key="batch_amb_temp")
    st.session_state["FLOW"] = FLOW
    st.session_state["RateDRA"] = RateDRA
    st.session_state["Price_HSD"] = Price_HSD
    st.session_state["Fuel_density"] = Fuel_density
    st.session_state["Ambient_temp"] = Ambient_temp
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
                    res = solve_pipeline(
                        stations_data,
                        term_data,
                        FLOW,
                        kv_list,
                        rho_list,
                        None,
                        RateDRA,
                        Price_HSD,
                        st.session_state.get("Fuel_density", 820.0),
                        st.session_state.get("Ambient_temp", 25.0),
                        {},
                        pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                    )
                    row = {"Scenario": f"100% {product_table.iloc[0]['Product']}, 0% {product_table.iloc[1]['Product']}"}
                    for idx, stn in enumerate(stations_data, start=1):
                        key = stn['name'].lower().replace(' ', '_')
                        row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                        add_speed_columns(row, res, stn)
                        row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                        row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                        _ppm = res.get(f"dra_ppm_{key}", 0.0)
                        row[f"DRA PPM {stn['name']}"] = _ppm if float(_ppm or 0) > 0 else "NIL"
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
                    res = solve_pipeline(
                        stations_data,
                        term_data,
                        FLOW,
                        kv_list,
                        rho_list,
                        None,
                        RateDRA,
                        Price_HSD,
                        st.session_state.get("Fuel_density", 820.0),
                        st.session_state.get("Ambient_temp", 25.0),
                        {},
                        pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                    )
                    row = {"Scenario": f"0% {product_table.iloc[0]['Product']}, 100% {product_table.iloc[1]['Product']}"}
                    for idx, stn in enumerate(stations_data, start=1):
                        key = stn['name'].lower().replace(' ', '_')
                        row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                        add_speed_columns(row, res, stn)
                        row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                        row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                        _ppm = res.get(f"dra_ppm_{key}", 0.0)
                        row[f"DRA PPM {stn['name']}"] = _ppm if float(_ppm or 0) > 0 else "NIL"
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
                        res = solve_pipeline(
                            stations_data,
                            term_data,
                            FLOW,
                            kv_list,
                            rho_list,
                            None,
                            RateDRA,
                            Price_HSD,
                            st.session_state.get("Fuel_density", 820.0),
                            st.session_state.get("Ambient_temp", 25.0),
                            {},
                            pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                        )
                        row = {"Scenario": f"{pct_A}% {product_table.iloc[0]['Product']}, {pct_B}% {product_table.iloc[1]['Product']}"}
                        for idx, stn in enumerate(stations_data, start=1):
                            key = stn['name'].lower().replace(' ', '_')
                            row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                            add_speed_columns(row, res, stn)
                            row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                            row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                            _ppm = res.get(f"dra_ppm_{key}", 0.0)
                            row[f"DRA PPM {stn['name']}"] = _ppm if float(_ppm or 0) > 0 else "NIL"
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
                        res = solve_pipeline(
                            stations_data,
                            term_data,
                            FLOW,
                            kv_list,
                            rho_list,
                            None,
                            RateDRA,
                            Price_HSD,
                            st.session_state.get("Fuel_density", 820.0),
                            st.session_state.get("Ambient_temp", 25.0),
                            {},
                            pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                        )
                        for idx, stn in enumerate(stations_data, start=1):
                            key = stn['name'].lower().replace(' ', '_')
                            row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                            add_speed_columns(row, res, stn)
                            row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                            row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                            _ppm = res.get(f"dra_ppm_{key}", 0.0)
                            row[f"DRA PPM {stn['name']}"] = _ppm if float(_ppm or 0) > 0 else "NIL"
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
                        res = solve_pipeline(
                            stations_data,
                            term_data,
                            FLOW,
                            kv_list,
                            rho_list,
                            None,
                            RateDRA,
                            Price_HSD,
                            st.session_state.get("Fuel_density", 820.0),
                            st.session_state.get("Ambient_temp", 25.0),
                            {},
                            pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                        )
                        for idx, stn in enumerate(stations_data, start=1):
                            key = stn['name'].lower().replace(' ', '_')
                            row[f"Num Pumps {stn['name']}"] = res.get(f"num_pumps_{key}", "")
                            add_speed_columns(row, res, stn)
                            row[f"SDH {stn['name']}"] = fmt_pressure(res, f"sdh_{key}", f"sdh_kgcm2_{key}")
                            row[f"RH {stn['name']}"] = fmt_pressure(res, f"residual_head_{key}", f"rh_kgcm2_{key}")
                            _ppm = res.get(f"dra_ppm_{key}", 0.0)
                            row[f"DRA PPM {stn['name']}"] = _ppm if float(_ppm or 0) > 0 else "NIL"
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
        st.dataframe(df_batch, width='stretch')
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
            st.plotly_chart(fig, width='stretch')
            st.info("Each line = one scenario. Hover to see full parameter set for each scenario.")
else:
    st.session_state.pop('batch_df', None)

def invalidate_results():
    """Clear any cached optimisation results from session state."""
    for k in ("last_res", "last_stations_data", "last_term_data", "last_linefill", "last_station_table"):
        st.session_state.pop(k, None)


def run_all_updates():
    """Invalidate caches, rebuild station data and solve for the global optimum."""
    invalidate_results()
    stations_data = st.session_state.stations
    term_data = {
        "name": st.session_state.get("terminal_name", "Terminal"),
        "elev": st.session_state.get("terminal_elev", 0.0),
        "min_residual": st.session_state.get("terminal_head", 10.0),
    }
    linefill_df = st.session_state.get("linefill_df")
    if not isinstance(linefill_df, pd.DataFrame):
        linefill_df = pd.DataFrame()
    else:
        linefill_df = ensure_initial_dra_column(linefill_df, default=0.0, fill_blanks=True)

    vol_linefill = st.session_state.get("linefill_vol_df")
    if isinstance(vol_linefill, pd.DataFrame) and len(vol_linefill) > 0:
        vol_linefill = ensure_initial_dra_column(vol_linefill, default=0.0, fill_blanks=True)
        kv_list, rho_list, segment_slices = map_vol_linefill_to_segments(
            vol_linefill, stations_data
        )
        linefill_df = vol_linefill
    else:
        kv_list, rho_list, segment_slices = derive_segment_profiles(
            linefill_df, stations_data
        )

    for idx, stn in enumerate(stations_data, start=1):
        if stn.get("is_pump", False):
            if "pump_types" in stn:
                for ptype in ["A", "B"]:
                    if ptype not in stn["pump_types"]:
                        continue
                    if stn["pump_types"][ptype].get("available", 0) == 0:
                        continue
                    dfh = st.session_state.get(f"head_data_{idx}{ptype}")
                    dfe = st.session_state.get(f"eff_data_{idx}{ptype}")
                    stn["pump_types"][ptype]["head_data"] = dfh
                    stn["pump_types"][ptype]["eff_data"] = dfe
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
                    stn["A"], stn["B"], stn["C"] = float(coeff[0]), float(coeff[1]), float(coeff[2])
                if dfe is not None and len(dfe) >= 5:
                    Qe = dfe.iloc[:, 0].values
                    Ee = dfe.iloc[:, 1].values
                    coeff_e = np.polyfit(Qe, Ee, 4)
                    stn["P"], stn["Q"], stn["R"], stn["S"], stn["T"] = [float(c) for c in coeff_e]

    search_kwargs = _collect_search_depth_kwargs()
    with st.spinner("Solving optimization..."):
        res = pipeline_model.solve_pipeline_with_types(
            stations_data,
            term_data,
            st.session_state.get("FLOW", 1000.0),
            kv_list,
            rho_list,
            segment_slices,
            st.session_state.get("RateDRA", 500.0),
            st.session_state.get("Price_HSD", 70.0),
            st.session_state.get("Fuel_density", 820.0),
            st.session_state.get("Ambient_temp", 25.0),
            df_to_dra_linefill(linefill_df),
            200.0,
            st.session_state.get("MOP_kgcm2"),
            24.0,
            pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
            **search_kwargs,
        )
    if not res or res.get("error"):
        msg = res.get("message") if isinstance(res, dict) else "Optimization failed"
        st.error(msg)
        return
    import copy
    st.session_state["last_res"] = copy.deepcopy(res)
    st.session_state["last_stations_data"] = copy.deepcopy(res.get("stations_used", stations_data))
    st.session_state["last_term_data"] = copy.deepcopy(term_data)
    st.session_state["last_linefill"] = copy.deepcopy(linefill_df)
    st.session_state["last_station_table"] = build_station_table(res, stations_data)
    st.session_state["run_mode"] = "instantaneous"
    st.rerun()


if not auto_batch:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.button("Start task", key="start_task", type="primary", on_click=run_all_updates)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; margin-top: 0.6rem;'>", unsafe_allow_html=True)
    run_hour = st.button("Run Hourly Flow Rate Optimizer", key="run_hour_btn", type="primary")
    run_day = st.button("Run Daily Pumping Schedule Optimizer", key="run_day_btn", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_day or run_hour:
        is_hourly = bool(run_hour)
        st.session_state["run_mode"] = "hourly" if is_hourly else "daily"

        import copy
        stations_base = copy.deepcopy(st.session_state.stations)
        for stn in stations_base:
            if stn.get('pump_types'):
                names_all = []
                for pdata in stn['pump_types'].values():
                    avail = int(pdata.get('available', 0))
                    names = pdata.get('names', [])
                    if len(names) < avail:
                        names += [f"Pump {len(names_all)+i+1}" for i in range(avail - len(names))]
                    pdata['names'] = names
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
        if isinstance(vol_df, pd.DataFrame):
            vol_df = ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)
        if vol_df is None or len(vol_df) == 0:
            st.error("Please enter linefill (volumetric) data.")
            st.stop()

        # Determine FLOW for this mode
        plan_df = st.session_state.get("day_plan_df", pd.DataFrame())
        if isinstance(plan_df, pd.DataFrame):
            plan_df = ensure_initial_dra_column(plan_df, default=0.0, fill_blanks=True)
        if is_hourly:
            FLOW_sched = st.session_state.get("hourly_flow", st.session_state.get("FLOW", 1000.0))
        else:
            daily_m3 = float(plan_df["Volume (m³)"].astype(float).sum()) if len(plan_df) else 0.0
            FLOW_sched = daily_m3 / 24.0

        if is_hourly:
            hours = [7]
        else:
            hours = [(7 + h) % 24 for h in range(24)]
        sub_steps = 1
        total_runs = len(hours) * sub_steps if hours else 0
        first_label = f"{hours[0] % 24:02d}:00" if hours else "00:00"
        last_label = f"{hours[-1] % 24:02d}:00" if hours else "23:00"
        if is_hourly:
            spinner_msg = f"Running 1 optimization ({first_label})..."
        else:
            spinner_msg = f"Running {total_runs} optimizations ({first_label} to {last_label})..."
        reports = []
        linefill_snaps = []
        st.session_state["linefill_next_day"] = None
        total_length = sum(stn.get('L', 0.0) for stn in stations_base)
        dra_reach_km = 200.0

        current_vol = ensure_initial_dra_column(vol_df.copy(), default=0.0, fill_blanks=True)
        if "DRA ppm" not in current_vol.columns:
            current_vol["DRA ppm"] = current_vol[INIT_DRA_COL]
        else:
            ppm_col = current_vol["DRA ppm"]
            ppm_blank = ppm_col.isna()
            if ppm_col.dtype == object:
                ppm_blank |= ppm_col.astype(str).str.strip() == ""
            if ppm_blank.any():
                current_vol.loc[ppm_blank, "DRA ppm"] = current_vol.loc[ppm_blank, INIT_DRA_COL]
        dra_linefill = df_to_dra_linefill(current_vol)
        current_vol = apply_dra_ppm(current_vol, dra_linefill)
        error_msg = None

        with st.spinner(spinner_msg):
            for ti, hr in enumerate(hours):
                linefill_snaps.append(current_vol.copy())
                sdh_hourly = []
                res = {}
                block_cost = 0.0
                power_cost_acc: dict[str, float] = {}
                dra_cost_acc: dict[str, float] = {}
                for sub in range(sub_steps):
                    pumped_tmp = FLOW_sched * 1.0
                    future_vol, future_plan = shift_vol_linefill(
                        current_vol.copy(), pumped_tmp, plan_df.copy() if plan_df is not None else None
                    )
                    # Determine worst-case fluid properties over this 1h window
                    kv_list, rho_list, segment_slices = combine_volumetric_profiles(
                        stations_base, current_vol, future_vol
                    )

                    stns_run = copy.deepcopy(stations_base)

                    start_str = f"{int((hr + sub) % 24):02d}:00"
                    res = solve_pipeline(
                        stns_run,
                        term_data,
                        FLOW_sched,
                        kv_list,
                        rho_list,
                        segment_slices,
                        RateDRA,
                        Price_HSD,
                        st.session_state.get("Fuel_density", 820.0),
                        st.session_state.get("Ambient_temp", 25.0),
                        dra_linefill,
                        dra_reach_km,
                        st.session_state.get("MOP_kgcm2"),
                        hours=1.0,
                        start_time=start_str,
                        pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                    )

                    block_cost += res.get("total_cost", 0.0)

                    if res.get("error"):
                        cur_hr = (hr + sub) % 24
                        error_msg = f"Optimization failed at {cur_hr:02d}:00 -> {res.get('message','')}"
                        break

                    # Record SDH for objective calculation and accumulate per-station costs
                    term_key = term_data["name"].lower().replace(" ", "_")
                    keys = [s['name'].lower().replace(' ', '_') for s in stns_run]
                    for k in keys:
                        power_cost_acc[k] = power_cost_acc.get(k, 0.0) + float(res.get(f"power_cost_{k}", 0.0) or 0.0)
                        dra_cost_acc[k] = dra_cost_acc.get(k, 0.0) + float(res.get(f"dra_cost_{k}", 0.0) or 0.0)
                    sdh_vals = [float(res.get(f"sdh_{k}", 0.0) or 0.0) for k in keys]
                    sdh_vals.append(float(res.get(f"sdh_{term_key}", 0.0) or 0.0))
                    sdh_hourly.append(max(sdh_vals))

                    dra_linefill = res.get("linefill", dra_linefill)
                    current_vol, plan_df = future_vol, future_plan
                    current_vol = apply_dra_ppm(current_vol, dra_linefill)
                    dra_reach_km = res.get("dra_front_km", dra_reach_km)

                if error_msg:
                    break

                for k, val in power_cost_acc.items():
                    res[f"power_cost_{k}"] = val
                for k, val in dra_cost_acc.items():
                    res[f"dra_cost_{k}"] = val
                res["total_cost"] = block_cost
                reports.append(
                    {
                        "time": hr % 24,
                        "result": res,
                        "sdh_hourly": sdh_hourly,
                        "sdh_max": max(sdh_hourly) if sdh_hourly else 0.0,
                    }
                )

        if error_msg:
            st.error(error_msg)
            st.stop()

        st.session_state["linefill_next_day"] = current_vol.copy(deep=True)

        # Build a consolidated station-wise table with flow pattern names
        station_tables = []
        for rec in reports:
            res = rec["result"]
            hr = rec["time"]
            df_int = build_station_table(res, stations_base)
            # Insert human-readable pattern and time columns
            pattern = res.get('flow_pattern_name', '')
            df_int.insert(0, "Pattern", pattern)
            df_int.insert(0, "Time", f"{hr:02d}:00")
            station_tables.append(df_int)
        df_day = pd.concat(station_tables, ignore_index=True).fillna(0.0).round(2)

        # Ensure numeric columns are typed as numeric to avoid conversion errors when styling
        # Pandas may treat some columns as object if they contain NaN or are newly inserted.
        df_day_numeric = df_day.copy()
        # Identify columns eligible for numeric styling
        num_cols = [c for c in df_day_numeric.columns if c not in ["Time", "Station", "Pump Name", "Pattern"]]
        for c in num_cols:
            df_day_numeric[c] = pd.to_numeric(df_day_numeric[c], errors="coerce").fillna(0.0)

        # Persist results for reuse across Streamlit reruns
        st.session_state["day_df"] = df_day_numeric
        st.session_state["day_df_raw"] = df_day
        st.session_state["day_reports"] = reports
        st.session_state["day_linefill_snaps"] = linefill_snaps
        st.session_state["day_hours"] = hours
        st.session_state["day_stations"] = stations_base

    if st.session_state.get("run_mode") in ("hourly", "daily") and st.session_state.get("day_df") is not None:
        df_day_numeric = st.session_state["day_df"]
        reports = st.session_state.get("day_reports", [])
        stations_base = st.session_state.get("day_stations", [])
        linefill_snaps = st.session_state.get("day_linefill_snaps", [])
        hours = st.session_state.get("day_hours", [])
        df_day = st.session_state.get("day_df_raw", df_day_numeric)
        transpose_view = st.checkbox("Transpose output table", key="transpose_day")
        df_display = df_day_numeric.T if transpose_view else df_day_numeric
        if transpose_view:
            numeric_rows_mask = df_display.apply(
                lambda row: pd.to_numeric(row, errors="coerce").notna().all(), axis=1
            )
            num_rows_disp = df_display.index[numeric_rows_mask].tolist()
            df_display.loc[num_rows_disp] = df_display.loc[num_rows_disp].apply(
                pd.to_numeric, errors="coerce"
            )
            df_disp_style = df_display.style.format(
                "{:.2f}", subset=pd.IndexSlice[num_rows_disp, :]
            )
            if num_rows_disp:
                df_disp_style = df_disp_style.background_gradient(
                    cmap="Blues", subset=pd.IndexSlice[num_rows_disp, :]
                )
        else:
            num_cols_disp = [
                c for c in df_display.columns if c not in ["Time", "Pattern", "Station", "Pump Name"]
            ]
            fmt_disp = {c: "{:.2f}" for c in num_cols_disp}
            df_disp_style = df_display.style.format(fmt_disp)
            if num_cols_disp:
                df_disp_style = df_disp_style.background_gradient(
                    cmap="Blues", subset=num_cols_disp
                )
        st.dataframe(
            df_disp_style,
            width='stretch',
            hide_index=not transpose_view,
        )
        label_prefix = "Hourly" if st.session_state.get("run_mode") == "hourly" else "Daily"
        first_label = f"{hours[0] % 24:02d}:00" if hours else "00:00"
        last_label = f"{hours[-1] % 24:02d}:00" if hours else "23:00"
        st.download_button(
            f"Download {label_prefix} Optimizer Output data",
            df_day.to_csv(index=False, float_format="%.2f"),
            file_name="hourly_schedule_results.csv" if st.session_state.get("run_mode") == "hourly" else "daily_schedule_results.csv",
        )

        # Display total cost per time slice and global sum
        cost_rows = [
            {
                "Time": f"{rec['time']:02d}:00",
                "Pattern": rec["result"].get("flow_pattern_name", ""),
                "Total Cost (INR)": float(rec["result"].get("total_cost", 0.0)),
            }
            for rec in reports
        ]
        df_cost = pd.DataFrame(cost_rows)
        df_cost["Total Cost (INR)"] = pd.to_numeric(
            df_cost["Total Cost (INR)"], errors="coerce",
        )
        df_cost = df_cost.round(2)
        df_cost_style = df_cost.style.format({"Total Cost (INR)": "{:.2f}"})
        st.dataframe(df_cost_style, width='stretch', hide_index=True)
        if st.session_state.get("run_mode") == "hourly":
            total_label = f"1h ({first_label})" if hours else "1h"
        else:
            total_label = f"24h ({first_label} to {last_label})" if hours else "24h"
        total_cost_value = df_cost["Total Cost (INR)"].sum()
        st.markdown(
            f"**Total Optimized Cost ({total_label}): {total_cost_value:,.2f} INR**",
        )
        for rec in reports:
            display_pump_type_details(
                rec["result"],
                stations_base,
                heading=f"Pump Details by Type ({rec['time']:02d}:00)",
            )

        combined = []
        for idx, df_line in enumerate(linefill_snaps):
            hr = hours[idx] % 24
            temp = df_line.copy()
            temp['Time'] = f"{hr:02d}:00"
            combined.append(temp)
        lf_all = pd.concat(combined, ignore_index=True).round(2)
        st.download_button(
            f"Download {label_prefix} Dynamic Linefill Output",
            lf_all.to_csv(index=False, float_format="%.2f"),
            file_name="linefill_snapshots.csv",
        )

        next_day_linefill = st.session_state.get("linefill_next_day")
        if isinstance(next_day_linefill, pd.DataFrame) and not next_day_linefill.empty:
            st.download_button(
                "Download next day's Linefill and DRA state",
                next_day_linefill.to_csv(index=False, float_format="%.4f"),
                file_name="next_day_linefill.csv",
            )

    st.markdown("<div style='text-align:center; margin-top: 0.6rem;'>", unsafe_allow_html=True)
    run_plan = st.button("Run Dynamic Pumping Plan Optimizer", key="run_plan_btn", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_plan:
        st.session_state["run_mode"] = "plan"
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
                            names += [f"Pump {len(names_all)+i+1}" for i in range(avail - len(names))]
                        pdata['names'] = names
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

            if isinstance(vol_df, pd.DataFrame):
                vol_df = ensure_initial_dra_column(vol_df, default=0.0, fill_blanks=True)
            current_vol = ensure_initial_dra_column(vol_df.copy(), default=0.0, fill_blanks=True)
            if "DRA ppm" not in current_vol.columns:
                current_vol["DRA ppm"] = current_vol[INIT_DRA_COL]
            else:
                ppm_col = current_vol["DRA ppm"]
                ppm_blank = ppm_col.isna()
                if ppm_col.dtype == object:
                    ppm_blank |= ppm_col.astype(str).str.strip() == ""
                if ppm_blank.any():
                    current_vol.loc[ppm_blank, "DRA ppm"] = current_vol.loc[ppm_blank, INIT_DRA_COL]
            dra_linefill = df_to_dra_linefill(current_vol)
            current_vol = apply_dra_ppm(current_vol, dra_linefill)
            reports = []
            linefill_snaps = []
            dra_reach_km = 200.0

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
                        kv_now, rho_now, slices_now = map_vol_linefill_to_segments(current_vol, stations_base)
                        future_vol, current_plan = shift_vol_linefill(current_vol.copy(), pumped_m3, current_plan)
                        kv_next, rho_next, slices_next = map_vol_linefill_to_segments(future_vol, stations_base)
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
                        slices_now,
                        RateDRA,
                        Price_HSD,
                        st.session_state.get("Fuel_density", 820.0),
                        st.session_state.get("Ambient_temp", 25.0),
                        dra_linefill,
                        dra_reach_km,
                        st.session_state.get("MOP_kgcm2"),
                        hours=duration_hr,
                        pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                    )
                    if res.get("error"):
                        st.error(f"Optimization failed for interval starting {seg_start} -> {res.get('message','')}")
                        st.stop()

                    reports.append({"time": seg_start, "result": res})
                    linefill_snaps.append(current_vol.copy())
                    dra_linefill = res.get("linefill", dra_linefill)
                    current_vol = apply_dra_ppm(future_vol, dra_linefill)
                    # In the revised model DRA does not propagate downstream over time.
                    # Keep the DRA reach constant (zero).
                    dra_reach_km = res.get("dra_front_km", dra_reach_km)
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
                # Prepend pattern and time columns
                pattern = res.get('flow_pattern_name', '')
                df_int.insert(0, "Pattern", pattern)
                df_int.insert(0, "Time", ts.strftime("%d/%m %H:%M"))
                station_tables.append(df_int)
            df_plan = pd.concat(station_tables, ignore_index=True).fillna(0.0).round(2)

            # Exclude non-numeric columns including Pattern for gradient styling
            num_cols = [c for c in df_plan.columns if c not in ["Time", "Station", "Pump Name", "Pattern"]]
            df_plan_numeric = df_plan.copy()
            for c in num_cols:
                df_plan_numeric[c] = pd.to_numeric(df_plan_numeric[c], errors="coerce").fillna(0.0)
            fmt_dict = {c: "{:.2f}" for c in num_cols}
            df_plan_style = (
                df_plan_numeric.style.format(fmt_dict)
                .background_gradient(cmap="Blues", subset=num_cols)
            )
            st.dataframe(df_plan_style, width='stretch', hide_index=True)
            st.download_button(
                "Download Dynamic Plan Output data",
                df_plan.to_csv(index=False, float_format="%.2f"),
                file_name="dynamic_plan_results.csv",
            )

            # Display total cost per interval and overall sum
            cost_rows = [
                {
                    "Time": rec["time"].strftime("%d/%m %H:%M"),
                    "Pattern": rec["result"].get("flow_pattern_name", ""),
                    "Total Cost (INR)": rec["result"].get("total_cost", 0.0),
                }
                for rec in reports
            ]
            df_cost = pd.DataFrame(cost_rows).round(2)
            df_cost_style = df_cost.style.format({"Total Cost (INR)": "{:.2f}"})
            st.dataframe(df_cost_style, width='stretch', hide_index=True)
            st.markdown(
                f"**Total Optimized Cost: {df_cost['Total Cost (INR)'].sum():,.2f} INR**"
            )
            for rec in reports:
                display_pump_type_details(
                    rec["result"],
                    stations_base,
                    heading=f"Pump Details by Type ({rec['time'].strftime('%d/%m %H:%M')})",
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


if not auto_batch and st.session_state.get("run_mode") == "instantaneous":
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
    
            # --- Display plan timing and table if available ---
            plan_start = st.session_state.get("last_plan_start")
            plan_hours = st.session_state.get("last_plan_hours")
            if plan_start is not None and plan_hours:
                plan_end = plan_start + pd.Timedelta(hours=plan_hours)
                st.markdown(
                    f"**Pumping plan duration:** {plan_start.strftime('%d/%m/%y %H:%M')} to {plan_end.strftime('%d/%m/%y %H:%M')}**"
                )
                sched_df = st.session_state.get("proj_flow_df", pd.DataFrame()).copy()
                if not sched_df.empty and "Start" in sched_df and "End" in sched_df:
                    sched_disp = sched_df.copy()
                    sched_disp["Start"] = pd.to_datetime(sched_disp["Start"]).dt.strftime("%d/%m/%y %H:%M")
                    sched_disp["End"] = pd.to_datetime(sched_disp["End"]).dt.strftime("%d/%m/%y %H:%M")
                    st.dataframe(sched_disp, width='stretch')

            # --- Use flows from backend output only ---
            segment_flows = []
            pump_flows = []
            for nm in names:
                key = nm.lower().replace(' ', '_')
                segment_flows.append(res.get(f"pipeline_flow_{key}", np.nan))
                pump_flows.append(res.get(f"pump_flow_{key}", np.nan))

            linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
            names = [s['name'] for s in stations_data]
            keys = [n.lower().replace(' ', '_') for n in names]
            df_sum = build_summary_dataframe(res, stations_data, linefill_df, st.session_state["last_term_data"])
            st.session_state["summary_table"] = df_sum.copy()
            df_sum.replace("NIL", np.nan, inplace=True)
            fmt_cols = {col: "{:.2f}" for col in df_sum.columns if col != "Parameters"}
            numeric_cols = df_sum.select_dtypes(include=[np.number]).columns
            df_display = df_sum.style.format(fmt_cols, na_rep="NIL")
            st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
            st.dataframe(df_display, width='stretch', hide_index=True)
            st.download_button(
                "📥 Download CSV",
                df_sum.round(2).to_csv(index=False, float_format="%.2f").encode(),
                file_name="results.csv",
            )

            # --- Detailed pump information when multiple pump types run ---
            display_pump_type_details(res, stations_data)

            # --- Aggregate counts for display ---
            total_cost = float(res.get("total_cost", 0.0))
            total_pumps = 0
            effs = []
            speeds = []
            for stn in stations_data:
                key = stn['name'].lower().replace(' ','_')
                npump = int(res.get(f"num_pumps_{key}", 0))
                if npump > 0:
                    total_pumps += npump
                    eff_default = float(res.get(f"efficiency_{key}", 0.0))
                    details = res.get(f"pump_details_{key}", [])
                    if isinstance(details, list) and details:
                        for detail in details:
                            count = int(detail.get("count", 0) or 0)
                            if count <= 0:
                                continue
                            try:
                                rpm_val = float(detail.get("rpm", 0.0))
                            except (TypeError, ValueError):
                                rpm_val = 0.0
                            try:
                                eff_val = float(detail.get("eff", eff_default))
                            except (TypeError, ValueError):
                                eff_val = eff_default
                            speeds.extend([rpm_val] * count)
                            effs.extend([eff_val] * count)
                    else:
                        speed_map = get_speed_display_map(res, key, stn)
                        speed_values = list(speed_map.values())
                        if speed_values:
                            try:
                                base_speed = float(speed_values[0])
                            except (TypeError, ValueError):
                                base_speed = float(res.get(f"speed_{key}", 0.0) or 0.0)
                        else:
                            base_speed = float(res.get(f"speed_{key}", 0.0) or 0.0)
                        speeds.extend([base_speed] * npump)
                        effs.extend([eff_default] * npump)
            avg_eff = sum(effs)/len(effs) if effs else 0.0
            avg_speed = sum(speeds)/len(speeds) if speeds else 0.0
            
            pattern_name = res.get('flow_pattern_name', 'Mainline Only')
            st.markdown(
                f"""<br>
                <div style='font-size:1.1em;'>
                <b>Total Optimized Cost:</b> {total_cost:.2f} INR<br>
                <b>No. of operating Pumps:</b> {total_pumps}<br>
                <b>Average Pump Efficiency:</b> {avg_eff:.2f} %<br>
                <b>Average Pump Speed:</b> {avg_speed:.0f} rpm<br>
                <b>Flow Pattern:</b> {pattern_name}
                </div>
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
    
            power_costs = [float(res.get(f"power_cost_{k}", 0.0) or 0.0) for k in keys]
            dra_costs = [float(res.get(f"dra_cost_{k}", 0.0) or 0.0) for k in keys]
            total_costs = [p + d for p, d in zip(power_costs, dra_costs)]

            df_cost = pd.DataFrame({
                "Station": names,
                "Power & Fuel Cost (INR)": power_costs,
                "DRA Cost (INR)": dra_costs,
                "Total Cost (INR)": total_costs,
            })
    
            # --- Grouped bar chart (side by side) for Power+Fuel and DRA ---
            fig_grouped = go.Figure()
            fig_grouped.add_trace(go.Bar(
                x=df_cost["Station"],
                y=df_cost["Power & Fuel Cost (INR)"],
                name="Power+Fuel",
                marker_color="#1976D2",
                text=[f"{x:.2f}" for x in df_cost["Power & Fuel Cost (INR)"]],
                textposition='outside'
            ))
            fig_grouped.add_trace(go.Bar(
                x=df_cost["Station"],
                y=df_cost["DRA Cost (INR)"],
                name="DRA",
                marker_color="#FFA726",
                text=[f"{x:.2f}" for x in df_cost["DRA Cost (INR)"]],
                textposition='outside'
            ))
            fig_grouped.update_layout(
                barmode='group',
                title="Station 24h Cost: Power & Fuel and DRA",
                xaxis_title="Station",
                yaxis_title="Cost (INR)",
                font=dict(size=16),
                legend=dict(font=dict(size=14)),
                height=430,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_grouped, width='stretch')
    
            # DRA cost bar chart only ---
            st.markdown("<h4 style='font-weight:600; margin-top: 2em;'>DRA Cost</h4>", unsafe_allow_html=True)
            fig_dra = px.bar(
                df_cost,
                x="Station",
                y="DRA Cost (INR)",
                text="DRA Cost (INR)",
                color="DRA Cost (INR)",
                color_continuous_scale=px.colors.sequential.YlOrBr,
                height=320,
            )
            fig_dra.update_traces(texttemplate="%{text:.2f}", textposition='outside')
            fig_dra.update_layout(
                yaxis_title="DRA Cost (INR)",
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_dra, width='stretch')
    
            # --- Pie chart: Total cost distribution by station ---
            st.markdown("#### Cost Contribution by Station")
            fig_pie = px.pie(
                df_cost,
                values="Total Cost (INR)",
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
            st.plotly_chart(fig_pie, width='stretch')
    
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
                    yaxis_title="Cost (INR)",
                    font=dict(size=15),
                    height=350
                )
                st.plotly_chart(fig_line, width='stretch')
    
            # --- Table: All cost heads, 2-decimal formatted ---
            df_cost_fmt = df_cost.copy()
            for c in df_cost_fmt.columns:
                if c != "Station":
                    df_cost_fmt[c] = df_cost_fmt[c].apply(lambda x: f"{x:.2f}")
            st.markdown("#### Tabular Cost Summary")
            st.dataframe(df_cost_fmt, width='stretch', hide_index=True)
    
            st.download_button(
                "📥 Download Station Cost (CSV)",
                df_cost.to_csv(index=False).encode(),
                file_name="station_cost.csv"
            )
    
            # --- KPI highlights ---
            st.markdown(
                f"""<br>
                <div style='font-size:1.1em;'><b>Total Operating Cost:</b> {sum(total_costs):,.2f} INR<br>
                <b>Maximum Station Cost:</b> {max(total_costs):,.2f} INR ({df_cost.loc[df_cost['Total Cost (INR)'].idxmax(), 'Station']})</div>
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
                st.plotly_chart(fig_h, width='stretch', key=f"perf_headloss_{uuid.uuid4().hex[:6]}")
                st.dataframe(df_hloss.style.format({"Head Loss (m)": "{:.2f}"}), width='stretch', hide_index=True)
            
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
                st.plotly_chart(fig_v, width='stretch')
                # Data table
                st.dataframe(df_vel.style.format({"Velocity (m/s)":"{:.2f}", "Reynolds Number":"{:.0f}"}), width='stretch', hide_index=True)
            
            # --- 3. Pump Characteristic Curve (Head vs Flow at various Speeds) ---
            with char_tab:
                st.markdown("<div class='section-title'>Pump Characteristic Curves (Head vs Flow at various Speeds)</div>", unsafe_allow_html=True)
                for i, stn in enumerate(stations_data, start=1):
                    if not stn.get('is_pump', False):
                        continue
                    key = stn['name'].lower().replace(' ','_')
                    df_head = st.session_state.get(f"head_data_{i}")
                    if df_head is not None and "Flow (m³/hr)" in df_head.columns and len(df_head) > 1:
                        flow_user = np.array(df_head["Flow (m³/hr)"], dtype=float)
                        max_flow = np.max(flow_user)
                    else:
                        max_flow = st.session_state.get("FLOW", 1000.0)
                    flows = np.linspace(0, max_flow, 200)
                    A = res.get(f"coef_A_{key}",0)
                    B = res.get(f"coef_B_{key}",0)
                    C = res.get(f"coef_C_{key}",0)
                    N_min = int(res.get(f"min_rpm_{key}", 0))
                    N_max = int(res.get(f"dol_{key}", 0))
                    if N_max == 0:
                        st.warning(f"Pump DOL (max RPM) not set for {stn['name']} — cannot plot characteristic curves.")
                        continue
                    rpm_vals = list(range(N_min, N_max + 1, pipeline_model.RPM_STEP))
                    if rpm_vals and rpm_vals[-1] != N_max:
                        rpm_vals.append(N_max)
                    fig = go.Figure()
                    for rpm in rpm_vals:
                        if rpm == 0:
                            continue
                        Q_at_rpm = flows
                        Q_equiv_DOL = Q_at_rpm * N_max / rpm if rpm else Q_at_rpm
                        H_DOL = (A*Q_equiv_DOL**2 + B*Q_equiv_DOL + C)
                        H = H_DOL * (rpm/N_max)**2 if N_max else H_DOL
                        valid = H >= 0
                        fig.add_trace(go.Scatter(
                            x=Q_at_rpm[valid], y=H[valid], mode='lines', name=f"{rpm} rpm",
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
                    st.plotly_chart(fig, width='stretch', key=f"char_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")
    
            
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
                        flow_min, flow_max = 0.0, FLOW
                        max_user_eff = 100
                    # Polynomial coefficients at DOL (user input speed)
                    P = stn.get('P', 0); Qc = stn.get('Q', 0); R = stn.get('R', 0)
                    S = stn.get('S', 0); T = stn.get('T', 0)
                    N_min = int(res.get(f"min_rpm_{key}", 0))
                    N_max = int(res.get(f"dol_{key}", 0))
                    rpm_vals = list(range(N_min, N_max + 1, pipeline_model.RPM_STEP))
                    if rpm_vals and rpm_vals[-1] != N_max:
                        rpm_vals.append(N_max)
                    fig = go.Figure()
                    for rpm in rpm_vals:
                        q_upper = flow_max * (rpm/N_max) if N_max else flow_max
                        flows = np.linspace(0, q_upper, 100)
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
                    st.plotly_chart(fig, width='stretch')
            
            # --- 5. Pressure vs Pipeline Length ---
            with press_tab:
                import plotly.graph_objects as go
                st.markdown("<div class='section-title'>Pipeline Hydraulics Profile: Optimized Pressure, Elevation and MAOP</div>", unsafe_allow_html=True)
                stations_data = st.session_state["last_stations_data"]
                res = st.session_state["last_res"]
                terminal = st.session_state["last_term_data"]
                N = len(stations_data)
            
                # Chainage/cumulative length at each station
                lengths = [0]
                for stn in stations_data:
                    lengths.append(lengths[-1] + stn.get("L", 0.0))
                names = [s['name'] for s in stations_data] + [terminal["name"]]
                keys = [n.lower().replace(' ', '_') for n in names]
            
                # Elevation profile (stations + peaks) converted to kg/cm²
                elev_x, elev_y = [], []
                for i, stn in enumerate(stations_data):
                    rho_i = res.get(f"rho_{keys[i]}", 850.0)
                    elev_x.append(lengths[i])
                    elev_y.append(stn['elev'] * rho_i / 10000.0)
                    if 'peaks' in stn and stn['peaks']:
                        for pk in sorted(stn['peaks'], key=lambda x: x['loc']):
                            pk_x = lengths[i] + pk['loc']
                            elev_x.append(pk_x)
                            elev_y.append(pk['elev'] * rho_i / 10000.0)
                rho_term = res.get(f"rho_{keys[-1]}", 850.0)
                elev_x.append(lengths[-1])
                elev_y.append(terminal['elev'] * rho_term / 10000.0)

                # RH and SDH at stations/terminal in kg/cm²
                rh_list = [res.get(f"rh_kgcm2_{k}", 0.0) for k in keys]
                sdh_list = [res.get(f"sdh_kgcm2_{k}", 0.0) for k in keys]

                # MAOP: single line at max MAOP value across all segments (flat)
                maop_val = max([res.get(f"maop_kgcm2_{k}", 85.0) for k in keys[:-1]] + [85.0])
                maop_x = [lengths[0], lengths[-1]]
                maop_y = [maop_val, maop_val]
            
                # Build sawtooth pressure profile: 
                press_x, press_y = [], []
                for i in range(N):
                    # 1. Vertical from RH up to SDH at station
                    press_x.extend([lengths[i], lengths[i]])
                    press_y.extend([rh_list[i], sdh_list[i]])
                    # 2. Sloped from SDH at this station down to RH at next station
                    press_x.append(lengths[i+1])
                    press_y.append(rh_list[i+1])
                # Only one marker at terminal (no jump)
            
                # Peaks (diamond markers)
                peak_x, peak_y = [], []
                for i, stn in enumerate(stations_data):
                    seg_len = stn.get("L", 0.0)
                    if 'peaks' in stn and stn['peaks']:
                        for pk in sorted(stn['peaks'], key=lambda x: x['loc']):
                            pk_x = lengths[i] + pk['loc']
                            # Linear interpolate pressure head along segment
                            y0, y1 = sdh_list[i], rh_list[i+1]
                            frac = pk['loc']/seg_len if seg_len > 0 else 0
                            pk_head = y0 + (y1 - y0) * frac
                            peak_x.append(pk_x)
                            peak_y.append(pk_head)
            
                fig = go.Figure()
            
                # Elevation (very subtle green dotted, thin)
                fig.add_trace(go.Scatter(
                    x=elev_x, y=elev_y, mode='lines+markers',
                    line=dict(dash='dot', color='#2ab240', width=1.5),
                    marker=dict(symbol='circle', color='#2ab240', size=5),
                    name="Elevation"
                ))
            
                # MAOP envelope (flat, dashed, red, thin)
                fig.add_trace(go.Scatter(
                    x=maop_x, y=maop_y, mode='lines',
                    line=dict(dash='dash', color='#e0115f', width=2),
                    name="MAOP Envelope"
                ))
            
                # Pressure Profile (sawtooth, blue, not too thick)
                fig.add_trace(go.Scatter(
                    x=press_x, y=press_y, mode='lines',
                    line=dict(color='#1846d2', width=2.8),
                    name="Pressure Optimizer"
                ))
            
                # RH markers at stations (open circles, black border)
                fig.add_trace(go.Scatter(
                    x=[lengths[i] for i in range(N+1)],
                    y=[rh_list[i] for i in range(N+1)],
                    mode='markers+text',
                    marker=dict(symbol='circle-open', size=12, color='#222', line=dict(width=2, color='#1846d2')),
                    text=[f"{s}<br>{rh_list[i]:.1f}" for i,s in enumerate(names)],
                    textposition='top center',
                    name="Residual Head",
                    showlegend=True
                ))
            
                # Peaks (diamond, magenta, no text)
                if peak_x:
                    fig.add_trace(go.Scatter(
                        x=peak_x, y=peak_y, mode='markers',
                        marker=dict(symbol='diamond', size=14, color='#b501c9', line=dict(width=1.5, color='#222')),
                        name="Peaks"
                    ))
            
                # Layout (clean, minimal, not cluttered)
                fig.update_layout(
                    title="Pipeline Hydraulics Profile: Pressure Optimization & Elevation",
                    xaxis_title="Pipeline Length (km)",
                    yaxis_title="Elevation / Pressure (kg/cm²)",
                    font=dict(size=15, family="Segoe UI, Arial"),
                    legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.95)", bordercolor="#bbb", borderwidth=1, x=0.78, y=0.98),
                    height=560,
                    margin=dict(l=35, r=15, t=60, b=35),
                    plot_bgcolor='#fff',
                    paper_bgcolor='#fff'
                )
                fig.update_xaxes(gridcolor="#e0e0e0", zeroline=False, showline=True, linewidth=1.5, linecolor='#1846d2', mirror=True)
                fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False, showline=True, linewidth=1.5, linecolor='#1846d2', mirror=True)
            
                st.plotly_chart(fig, width='stretch')
            
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
                    if N_max > 0:
                        speeds = np.arange(N_min, N_max + 1, pipeline_model.RPM_STEP)
                        power_curve = [P1 * (rpm/N_max)**3 for rpm in speeds]
                        fig_pwr = go.Figure()
                        fig_pwr.add_trace(go.Scatter(
                            x=speeds, y=power_curve, mode='lines+markers',
                            name="Power vs Speed",
                            marker_color="#1976D2",
                            line=dict(width=3),
                            hovertemplate="Speed: %{x} rpm<br>Power: %{y:.2f} kW",
                        ))
                        fig_pwr.update_layout(
                            title=f"Power vs Speed (at Pump Flow = {pump_flow:.2f} m³/hr): {stn['name']}",
                            xaxis_title="Speed (rpm)",
                            yaxis_title="Power (kW)",
                            font=dict(size=16),
                            height=400
                        )
                        st.plotly_chart(fig_pwr, width='stretch')
                    else:
                        st.warning("DOL speed not specified; skipping Power vs Speed plot.")
                    # --- 2. Power vs Flow (various speeds) ---
                    df_head = st.session_state.get(f"head_data_{i}")
                    if df_head is not None and "Flow (m³/hr)" in df_head.columns and len(df_head) > 1:
                        flow_user = np.array(df_head["Flow (m³/hr)"], dtype=float)
                        flow_max = float(np.max(flow_user))
                    else:
                        flow_max = pump_flow
                    rpm_vals = list(range(N_min, N_max + 1, pipeline_model.RPM_STEP))
                    if rpm_vals and rpm_vals[-1] != N_max:
                        rpm_vals.append(N_max)
                    fig_pwr2 = go.Figure()
                    for rpm in rpm_vals:
                        q_upper = flow_max * (rpm/N_max) if N_max else flow_max
                        flows = np.linspace(0, q_upper, 100)
                        Q_equiv = flows * N_max / rpm if rpm else flows
                        H_DOL = A*Q_equiv**2 + B*Q_equiv + C
                        H = H_DOL * (rpm/N_max)**2 if N_max else H_DOL
                        eff_flows = (P4*Q_equiv**4 + Qc*Q_equiv**3 + R*Q_equiv**2 + S*Q_equiv + T)
                        eff_flows = np.clip(eff_flows/100, 0.01, 1.0)
                        power_flows = (rho * flows * 9.81 * H)/(3600.0*1000*eff_flows)
                        fig_pwr2.add_trace(go.Scatter(
                            x=flows, y=power_flows, mode='lines', name=f"{rpm} rpm",
                            hovertemplate="Flow: %{x:.2f} m³/hr<br>Power: %{y:.2f} kW",
                        ))
                    fig_pwr2.update_layout(
                        title=f"Power vs Flow at Various Speeds: {stn['name']}",
                        xaxis_title="Flow (m³/hr)",
                        yaxis_title="Power (kW)",
                        font=dict(size=16),
                        height=400,
                    )
                    st.plotly_chart(fig_pwr2, width='stretch')
    
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
                kv_list, _, _ = map_linefill_to_segments(linefill_df, stations_data)
                visc = kv_list[i-1]
                flows = np.linspace(0, st.session_state.get("FLOW", 1000.0), 101)
                v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
                if visc > 0:
                    Re_vals = v_vals * d_inner_i / (visc*1e-6)
                    Re_pow = np.where(Re_vals>0, Re_vals**0.9, np.inf)
                    term = rough/d_inner_i/3.7 + 5.74/Re_pow
                    f_vals = np.where(Re_vals>0, 0.25/(np.log10(term)**2), 0.0)
                else:
                    Re_vals = np.zeros_like(v_vals)
                    f_vals = np.zeros_like(v_vals)
                # Professional gradient: from blue to red
                n_curves = (max_dr // pipeline_model.DRA_STEP) + 1
                color_palette = [
                    "#1565C0", "#1976D2", "#1E88E5", "#3949AB", "#8E24AA",
                    "#D81B60", "#F4511E", "#F9A825", "#43A047", "#00897B"
                ]
                color_idx = np.linspace(0, len(color_palette)-1, n_curves).astype(int)
                fig_sys = go.Figure()
                for j, dra in enumerate(range(0, max_dr + pipeline_model.DRA_STEP, pipeline_model.DRA_STEP)):
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
                st.plotly_chart(fig_sys, width='stretch', key=f"sys_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")
    
    
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
                kv_list, _, _ = map_linefill_to_segments(linefill_df, stations_data)
                visc = kv_list[stn_idx]
    
                # --------- Begin Figure ---------
                fig = go.Figure()
    
                # -------- System Curves: All DRA, Turbo Colormap, Vivid and Bold --------
                system_dra_steps = list(range(0, max_dr + pipeline_model.DRA_STEP, pipeline_model.DRA_STEP))
                n_dra = len(system_dra_steps)
                for idx, dra in enumerate(system_dra_steps):
                    v_vals = flows/3600.0 / (pi*(d_inner**2)/4)
                    if visc > 0:
                        Re_vals = v_vals * d_inner / (visc*1e-6)
                        Re_pow = np.where(Re_vals>0, Re_vals**0.9, np.inf)
                        term = rough/d_inner/3.7 + 5.74/Re_pow
                        f_vals = np.where(Re_vals>0, 0.25/(np.log10(term)**2), 0.0)
                    else:
                        Re_vals = np.zeros_like(v_vals)
                        f_vals = np.zeros_like(v_vals)
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
                    rpm_steps = np.arange(N_min, N_max + 1, pipeline_model.RPM_STEP)
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
                            Q_equiv = flows * N_max / rpm if rpm else flows
                            H_DOL = A*Q_equiv**2 + B*Q_equiv + C
                            H_pump = npump * (H_DOL * (rpm/N_max)**2 if N_max else H_DOL)
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
    
                    # Optimized pump combination curve
                    base = stn['name'].split('_')[0]
                    combo_units = [s for s in stations_data if s.get('is_pump', False) and s['name'].startswith(base)]
                    if len(combo_units) > 1:
                        head_combo = np.zeros_like(flows)
                        for unit in combo_units:
                            ukey = unit['name'].lower().replace(' ', '_')
                            rpm_u = res.get(f"speed_{ukey}", N_max)
                            A_u = res.get(f"coef_A_{ukey}", 0)
                            B_u = res.get(f"coef_B_{ukey}", 0)
                            C_u = res.get(f"coef_C_{ukey}", 0)
                            Nmax_u = res.get(f"dol_{ukey}", N_max)
                            Q_equiv = flows * Nmax_u / rpm_u if rpm_u else flows
                            H_DOL = A_u*Q_equiv**2 + B_u*Q_equiv + C_u
                            H_u = H_DOL * (rpm_u/Nmax_u)**2 if Nmax_u else H_DOL
                            head_combo += H_u
                        fig.add_trace(go.Scatter(
                            x=flows, y=head_combo, mode='lines',
                            line=dict(color='black', width=4, dash='dash'),
                            name='Optimized Combo',
                        ))
                    else:
                        speed_opt = res.get(f"speed_{key}", N_max)
                        nopt = int(res.get(f"num_pumps_{key}", 1))
                        Q_equiv = flows * N_max / speed_opt if speed_opt else flows
                        H_DOL = A*Q_equiv**2 + B*Q_equiv + C
                        H_opt = H_DOL * (speed_opt/N_max)**2 if N_max else H_DOL
                        head_combo = nopt * H_opt
                        fig.add_trace(go.Scatter(
                            x=flows, y=head_combo, mode='lines',
                            line=dict(color='black', width=4, dash='dash'),
                            name=f'Optimized {nopt} pump{"s" if nopt>1 else ""}',
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
                    hovermode="closest",
                )
                st.plotly_chart(fig, width='stretch')
    
    
    
    
    # ---- Tab 6: DRA Curves ----
    with tab6:
        if "last_res" not in st.session_state or "last_stations_data" not in st.session_state:
            st.info("Please run optimization first to analyze DRA curves.")
            st.stop()
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
        kv_list, _, _ = map_linefill_to_segments(linefill_df, stations_data)
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
                    if np.isclose(upper, lower):
                        ppm_vals = ppm_lower
                        curve_label = f"{lower} cSt curve"
                    else:
                        # Interpolate each percent_dr value for given viscosity
                        ppm_vals = ppm_lower + (ppm_upper - ppm_lower) * ((viscosity - lower) / (upper - lower))
                        curve_label = f"Interpolated for {viscosity:.2f} cSt"
                opt_ppm = res.get(f"dra_ppm_{key}", 0.0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ppm_vals,
                    y=percent_dr,
                    mode='lines+markers',
                    name=curve_label
                ))
                fig.add_trace(go.Scatter(
                    x=[opt_ppm], y=[dr_opt],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name="Optimized Point",
                ))
                fig.update_layout(
                    title=f"DRA Curve for {stn['name']} (Viscosity: {viscosity:.2f} cSt)",
                    xaxis_title="PPM",
                    yaxis_title="% Drag Reduction",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, width='stretch')
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

        pump_idx = next((i for i,s in enumerate(stations_data) if s.get('is_pump', False)), None)
        if pump_idx is None:
            st.warning("No pump station available for 3D analysis.")
            st.stop()
        stn = stations_data[pump_idx]
        key = stn['name'].lower().replace(' ', '_')

        default_speed = pipeline_model._station_max_rpm(stn)
        if default_speed <= 0:
            default_speed = pipeline_model._station_min_rpm(stn)
        speed_opt = float(last_res.get(f"speed_{key}", default_speed))
        dra_opt = float(last_res.get(f"drag_reduction_{key}", 0.0))
        nopt_opt = int(last_res.get(f"num_pumps_{key}", 1))
        flow_opt = FLOW

        delta_speed = 150
        delta_dra = 10
        delta_nop = 1
        delta_flow = 150
        N = 9
        N_min = int(pipeline_model._station_min_rpm(stn, default=1000))
        N_max = int(pipeline_model._station_max_rpm(stn, default=1500))
        if N_max <= 0 and N_min > 0:
            N_max = N_min
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
        DOL = float(pipeline_model._station_max_rpm(stn, default=N_max) or N_max)
        linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
        kv_list, rho_list, _ = map_linefill_to_segments(linefill_df, stations_data)
        rho = rho_list[pump_idx]
        rate = stn.get('rate', 9.0)
        tariffs = stn.get('tariffs') or []
        g = 9.81

        def tariff_cost(kw, hours, start_time="00:00"):
            t0 = dt.datetime.strptime(start_time, "%H:%M")
            remaining = hours
            cost = 0.0
            while remaining > 0:
                current = t0 + dt.timedelta(hours=hours - remaining)
                applied = False
                for tr in tariffs:
                    s = dt.datetime.strptime(tr["start"], "%H:%M")
                    e = dt.datetime.strptime(tr["end"], "%H:%M")
                    if e <= s:
                        e += dt.timedelta(days=1)
                    if s <= current < e:
                        overlap = min(remaining, (e - current).total_seconds() / 3600.0)
                        cost += kw * overlap * float(tr["rate"])
                        remaining -= overlap
                        applied = True
                        break
                if not applied:
                    cost += kw * remaining * rate
                    break
            return cost

        def get_head(q, n): return (A*q**2 + B*q + Cc)*(n/DOL)**2
        def get_eff(q, n): q_adj = q * DOL/n if n > 0 else q; return (P*q_adj**4 + Qc*q_adj**3 + R*q_adj**2 + S*q_adj + T)
        def get_power_cost(q, n, d, npump=1):
            h = get_head(q, n)
            eff = max(get_eff(q, n)/100, 0.01)
            motor_eff = 0.98 if stn.get('power_type') == 'Diesel' else (0.95 if n >= DOL else 0.91)
            pwr = (rho*q*g*h*npump)/(3600.0*eff*motor_eff*1000)
            return tariff_cost(pwr, 24.0, "00:00")
        def get_system_head(q, d):
            d_inner = stn['D'] - 2*stn['t']
            rough = stn['rough']
            L_seg = stn['L']
            visc = kv_list[pump_idx]
            v = q/3600.0/(np.pi*(d_inner**2)/4)
            Re = v*d_inner/(visc*1e-6) if visc > 0 else 0
            if Re > 0:
                f = 0.25/(np.log10(rough/d_inner/3.7 + 5.74/(Re**0.9))**2)
            else:
                f = 0.0
            DH = f*((L_seg*1000.0)/d_inner)*(v**2/(2*g))*(1-d/100)
            return stn['elev'] + DH
            
        dr_opt = last_res.get(f"drag_reduction_{key}", 0.0)
        viscosity = kv_list[pump_idx]
        ppm_value = last_res.get(f"dra_ppm_{key}", 0.0)
    
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
            "PowerCost": "Z: Power Cost (INR)",
            "DRA": "Y: DRA (%)",
            "NOP": "X: No. of Pumps",
            "TotalCost": "Z: Total Cost (INR)",
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
    
        st.plotly_chart(fig, width='stretch')
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
            # ---- 1. Gather all points: stations and peaks ----
            res = st.session_state['last_res']
            stations_exp = st.session_state['last_stations_data']
            stations_phys = st.session_state['stations']
            terminal = st.session_state['last_term_data']

            chainages = [0]
            elevs = []
            rh = []
            names = []
            mesh_x, mesh_y, mesh_z, mesh_text, mesh_color = [], [], [], [], []
            peak_x, peak_y, peak_z, peak_label = [], [], [], []

            for i, stn in enumerate(stations_phys):
                chainages.append(chainages[-1] + stn.get('L', 0.0))
                elevs.append(stn['elev'])
                base = stn['name']
                base_key = base.lower().replace(' ', '_')
                if stn.get('pump_types'):
                    candidates = [s['name'].lower().replace(' ', '_') for s in stations_exp if s['name'].startswith(base)]
                    key = candidates[-1] if candidates else base_key
                else:
                    key = base_key
                rh_val = res.get(f'residual_head_{key}', 0.0)
                rh.append(rh_val)
                names.append(base)
                mesh_x.append(chainages[i])
                mesh_y.append(stn['elev'])
                mesh_z.append(rh_val)
                mesh_text.append(base)
                mesh_color.append(rh_val)
                if 'peaks' in stn and stn['peaks']:
                    for pk in stn['peaks']:
                        peak_x_val = chainages[i] + pk.get('loc', 0)
                        py = pk.get('elev', stn['elev'])
                        pz = rh_val
                        mesh_x.append(peak_x_val)
                        mesh_y.append(py)
                        mesh_z.append(pz)
                        mesh_text.append('Peak')
                        mesh_color.append(pz)
                        peak_x.append(peak_x_val)
                        peak_y.append(py)
                        peak_z.append(pz)
                        peak_label.append(f'Peak @ {base}')

            terminal_chainage = chainages[-1]
            key_term = terminal['name'].lower().replace(' ', '_')
            rh_term = res.get(f'residual_head_{key_term}', 0.0)
            mesh_x.append(terminal_chainage)
            mesh_y.append(terminal['elev'])
            mesh_z.append(rh_term)
            mesh_text.append(terminal['name'])
            mesh_color.append(rh_term)
            names.append(terminal['name'])
            elevs.append(terminal['elev'])
            rh.append(rh_term)

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
                x=chainages[:-1],
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
                x=chainages,
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
                    'text': "<b>3D Pressure Profile:</b>",
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
    
            st.plotly_chart(fig3d, width='stretch')
            st.markdown(
                "<div style='text-align:center;color:#888;margin-top:1.1em;'>"
                "Z-axis = Residual Head (mcl). Mesh surface interpolates between stations and peaks. <br>"
                "Stations, terminal, and peaks are all shown with dynamic coloring.</div>",
                unsafe_allow_html=True
            )
    
    with tab_sens:
        st.markdown("<div class='section-title'>Sensitivity Analysis</div>", unsafe_allow_html=True)
        st.write("Analyze how key outputs respond to variations in a parameter. Each run recalculates results based on set pipeline parameter and optimization metric.")
        if "last_res" not in st.session_state:
            st.info("Run optimization first to enable sensitivity analysis.")
        else:
            param = st.selectbox("Parameter to vary", [
                "Flowrate (m³/hr)", "Viscosity (cSt)", "Drag Reduction (%)", "Fuel Price (INR/L)", "DRA Cost (INR/L)"
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
                elif param == "Fuel Price (INR/L)":
                    pvals = np.linspace(0.5*Price_HSD, 2*Price_HSD, N)
                elif param == "DRA Cost (INR/L)":
                    pvals = np.linspace(0.5*RateDRA, 2*RateDRA, N)
                yvals = []
                st.info("Running sensitivity... This may take a few seconds per parameter.")
                progress = st.progress(0)
                for i, val in enumerate(pvals):
                    stations_data = [dict(s) for s in st.session_state['stations']]
                    term_data = dict(st.session_state["last_term_data"])
                    this_FLOW = FLOW
                    this_RateDRA = RateDRA
                    this_Price_HSD = Price_HSD
                    this_linefill_df = linefill_df.copy()
                    if param == "Flowrate (m³/hr)":
                        this_FLOW = val
                    elif param == "Viscosity (cSt)":
                        this_linefill_df["Viscosity (cSt)"] = val
                    elif param == "Drag Reduction (%)":
                        for stn in stations_data:
                            if stn.get('is_pump', False):
                                stn['max_dr'] = max(stn.get('max_dr', val), val)
                                break
                    elif param == "Fuel Price (INR/L)":
                        this_Price_HSD = val
                    elif param == "DRA Cost (INR/L)":
                        this_RateDRA = val
                    kv_list, rho_list, segment_slices = derive_segment_profiles(this_linefill_df, stations_data)
                    resi = solve_pipeline(
                        stations_data,
                        term_data,
                        this_FLOW,
                        kv_list,
                        rho_list,
                        segment_slices,
                        this_RateDRA,
                        this_Price_HSD,
                        st.session_state.get("Fuel_density", 820.0),
                        st.session_state.get("Ambient_temp", 25.0),
                        this_linefill_df.to_dict(),
                        pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                    )
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
                st.plotly_chart(fig, width='stretch')
                st.dataframe(df_sens, width='stretch', hide_index=True)
                st.download_button("Download CSV", df_sens.to_csv(index=False).encode(), file_name="sensitivity.csv")

    with tab_bench:
        st.markdown("<div class='section-title'>Benchmarking & Global Standards</div>", unsafe_allow_html=True)
        st.write("Compare pipeline performance with global/ custom benchmarks. Green indicates Pipeline operation match/exceed global standards while red means improvement is needed.")
        if "last_res" not in st.session_state:
            st.info("Run optimization to show benchmark analysis.")
        else:
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
                kv_list, rho_list, segment_slices = map_linefill_to_segments(linefill_df, stations_data)
                for idx, stn in enumerate(stations_data):
                    key = stn['name'].lower().replace(' ', '_')
                    dra_cost = float(res.get(f"dra_cost_{key}", 0.0) or 0.0)
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
                st.dataframe(df_bench, width='stretch', hide_index=True)

    with tab_sim:
        st.markdown("<div class='section-title'>Annualized Savings Simulator</div>", unsafe_allow_html=True)
        st.write("Annual savings from efficiency improvements, energy cost and DRA optimizations.")
        if "last_res" not in st.session_state:
            st.info("Run optimization first.")
        else:
            FLOW = st.session_state["FLOW"]
            RateDRA = st.session_state["RateDRA"]
            Price_HSD = st.session_state["Price_HSD"]
            st.write("Adjust improvement assumptions and see the impact over a year.")
            pump_eff_impr = st.slider("Pump Efficiency Improvement (%)", 0, 10, 3)
            dra_cost_impr = st.slider("DRA Price Reduction (%)", 0, 30, 5)
            flow_change = st.slider("Throughput Increase (%)", 0, 30, 0)
            if st.button("Run Savings Simulation"):
                linefill_df = st.session_state.get("last_linefill", st.session_state.get("linefill_df", pd.DataFrame()))
                stations_data = [dict(s) for s in st.session_state['stations']]
                term_data = dict(st.session_state["last_term_data"])
                for stn in stations_data:
                    if stn.get('is_pump', False) and "eff_data" in stn and pump_eff_impr > 0:
                        pass  # placeholder for efficiency adjustment
                new_RateDRA = RateDRA * (1 - dra_cost_impr / 100)
                new_FLOW = FLOW * (1 + flow_change / 100)
                kv_list, rho_list, segment_slices = map_linefill_to_segments(
                    linefill_df, stations_data
                )
                res2 = solve_pipeline(
                    stations_data,
                    term_data,
                    new_FLOW,
                    kv_list,
                    rho_list,
                    segment_slices,
                    new_RateDRA,
                    Price_HSD,
                    st.session_state.get("Fuel_density", 820.0),
                    st.session_state.get("Ambient_temp", 25.0),
                    linefill_df.to_dict(),
                    pump_shear_rate=st.session_state.get("pump_shear_rate", 0.0),
                )
                total_cost, new_cost = 0, 0
                for idx, stn in enumerate(stations_data):
                    key = stn['name'].lower().replace(' ', '_')
                    last = st.session_state["last_res"]
                    dra_cost = float(last.get(f"dra_cost_{key}", 0.0) or 0.0)
                    power_cost = float(last.get(f"power_cost_{key}", 0.0) or 0.0)
                    total_cost += dra_cost + power_cost
                    dra_cost2 = float(res2.get(f"dra_cost_{key}", 0.0) or 0.0)
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
