import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory

if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 NEOS_EMAIL not found in secrets. Please add it.")

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(
    page_title="Pipeline Optimization Application",
    layout="wide"
)

# ---------------------
# Custom CSS for styling
# ---------------------
st.markdown("""
    <style>
      .reportview-container, .main, .block-container, .sidebar .sidebar-content {
        background: none !important;
      }
      .stMetric > div {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(5px);
        border-radius: 8px;
        padding: 12px;
        color: var(--text-primary-color) !important;
        text-align: center;
      }
      .stMetric .metric-value, .stMetric .metric-label {
        display: block;
        width: 100%;
        text-align: center;
      }
      .color-adapt {
        color: var(--text-primary-color) !important;
      }
      .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary-color) !important;
        margin-top: 1rem;
      }
    </style>
    """, unsafe_allow_html=True)

# ---------------------
# Title
# ---------------------
st.markdown("<h1 class='color-adapt'>Mixed Integer Nonlinear Pipeline Optimization</h1>", unsafe_allow_html=True)

# ---------------------
# Solver wrapper
# ---------------------
def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    import pipeline_model  # the back-end module
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)

# ---------------------
# Sidebar: Inputs
# ---------------------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=2000.0, step=10.0)
        KV        = st.number_input("Kinematic Viscosity (cSt)", value=10.0, step=0.1)
        rho       = st.number_input("Fluid Density (kg/m³)", value=880.0, step=1.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=0.1)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)

    # Buttons to add/remove stations
    add_col, rem_col = st.columns(2)
    add_btn = add_col.button("➕ Add Station")
    rem_btn = rem_col.button("🗑️ Remove Station")
    if 'stations' not in st.session_state:
        # Initialize with one pump station
        st.session_state.stations = [{
            'name': 'Station 1',
            'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True,
            'power_type': 'Grid', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 3, 'MinRPM': 1200.0, 'DOL': 1500.0, 'max_dr': 40.0,
            'A': -2e-6, 'B': -0.0015, 'C': 179.14,
            'P': -4.161e-14, 'Q': 6.574e-10, 'R': -8.737e-06, 'S': 0.04924, 'T': -0.001754
        }]
    if add_btn:
        n = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name': f'Station {n}',
            'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True,
            'power_type': 'Diesel', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 2, 'MinRPM': 2750.0, 'DOL': 3437.0, 'max_dr': 40.0,
            'A': -1e-5, 'B': 0.00135, 'C': 270.08,
            'P': -4.07e-13, 'Q': 3.4657e-09, 'R': -1.9273e-05, 'S': 0.067033, 'T': -0.15043
        })
    if rem_btn and len(st.session_state.stations) > 1:
        st.session_state.stations.pop()

    # Render each station’s inputs
    for idx, stn in enumerate(st.session_state.stations, start=1):
        with st.expander(f"Station {idx}: {stn['name']}", expanded=True):
            stn['name']  = st.text_input("Name", value=stn['name'], key=f"name{idx}")
            stn['elev']  = st.number_input("Elevation (m)", value=stn['elev'], key=f"elev{idx}")
            if idx == 1:
                stn['min_residual'] = st.number_input("Required Residual Head (m)", 
                                                     value=stn.get('min_residual', 50.0), 
                                                     step=0.1, key=f"resid{idx}")
            stn['D']     = st.number_input("Outer Diameter (m)", value=stn['D'], step=0.00001, format="%.5f", key=f"D{idx}")
            stn['t']     = st.number_input("Wall Thickness (m)", value=stn['t'], step=0.00001, format="%.5f", key=f"t{idx}")
            stn['SMYS']  = st.number_input("SMYS (psi)", value=stn['SMYS'], key=f"SMYS{idx}")
            stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], step=0.00001, format="%.5f", key=f"rough{idx}")
            stn['L']     = st.number_input("Length to Next Station (km)", value=stn['L'], key=f"L{idx}")
            stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")
            if stn['is_pump']:
                stn['power_type'] = st.selectbox("Power Source", ["Grid", "Diesel"],
                    index=(0 if stn['power_type']=="Grid" else 1), key=f"ptype{idx}")
                if stn['power_type'] == "Grid":
                    stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", value=stn['rate'], key=f"rate{idx}")
                else:
                    stn['sfc'] = st.number_input("SFC (gm/bhp-hr)", value=stn['sfc'], key=f"sfc{idx}")
                stn['max_pumps'] = st.number_input("Available Pumps", min_value=1, value=stn['max_pumps'], step=1, key=f"mpumps{idx}")
                stn['MinRPM']    = st.number_input("Min RPM", value=stn['MinRPM'], key=f"minrpm{idx}")
                stn['DOL']       = st.number_input("Rated RPM (DOL)", value=stn['DOL'], key=f"dol{idx}")
                stn['max_dr']    = st.number_input("Max Drag Reduction (%)", value=stn['max_dr'], key=f"mdr{idx}")
                st.file_uploader("Pump Head Curve (image)", type=["png","jpg","jpeg"], key=f"headimg{idx}")
                st.file_uploader("Efficiency Curve (image)", type=["png","jpg","jpeg"], key=f"effimg{idx}")

    # ——— Terminal Station inputs ———
    st.markdown("---")
    st.subheader("🏁 Terminal Station")
    terminal_name = st.text_input("Terminal Station Name", value="Terminal")
    terminal_elev = st.number_input("Terminal Elevation (m)", value=0.0, step=0.00001, format="%.5f")
    terminal_min_res = st.number_input("Required Residual Head (m)", value=50.0, step=0.1)

    run = st.button("🚀 Run Optimization")

# --------------------------------------------------------

if run:
    with st.spinner("Solving pipeline optimization..."):
        stations_data = st.session_state.stations
        terminal_data = {
            "name": terminal_name,
            "elev": terminal_elev,
            "min_residual": terminal_min_res
        }
        # Solve model
        res = solve_pipeline(stations_data, terminal_data, FLOW, KV, rho, RateDRA, Price_HSD)

    # KPI Cards
    total_cost = res.get('total_cost', 0)
    total_pumps = sum(res.get(f"num_pumps_{s['name'].lower()}", 0) for s in stations_data)
    speeds = [res.get(f"speed_{s['name'].lower()}", 0) for s in stations_data]
    effs   = [res.get(f"efficiency_{s['name'].lower()}", 0) for s in stations_data]
    avg_speed = np.mean(speeds) if speeds else 0
    avg_eff   = np.mean(effs)   if effs else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost (INR/day)", f"₹{total_cost:,.2f}")
    c2.metric("Total Pumps", total_pumps)
    c3.metric("Avg Speed (RPM)", f"{avg_speed:.2f}")
    c4.metric("Avg Efficiency (%)", f"{avg_eff:.2f}")

    # Detailed results table
    station_names = [s['name'] for s in stations_data] + [terminal_name]
    rows = []
    for name in station_names:
        row = res.get(name.lower(), {})
        row['Station'] = name
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Station')
    st.markdown("<h2 class='section-title'>Station-wise Results</h2>", unsafe_allow_html=True)
    st.dataframe(df.style.format({
        'speed': "{:.2f}", 'efficiency': "{:.1f}%", 
        'power_cost': "{:.2f}", 'dra_cost': "{:.2f}", 
        'head_loss': "{:.2f}", 'residual_head': "{:.2f}"}))

    # Download button for results
    csv = df.to_csv().encode('utf-8')
    st.download_button("📥 Download Results (CSV)", csv, "pipeline_results.csv", "text/csv")
