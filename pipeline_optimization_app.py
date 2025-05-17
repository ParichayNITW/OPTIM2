import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from pipeline_model import solve_pipeline  # your back-end

if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 NEOS_EMAIL not found in secrets. Please add it in Streamlit secrets.")

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(
    page_title="Mixed Integer Non Linear Convex Optimization of Pipeline Operations",
    layout="wide"
)

# ---------------------
# Custom CSS
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
  .stMetric .metric-value,
  .stMetric .metric-label {
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
st.markdown(
    "<h1 class='color-adapt'>Mixed Integer Non Linear Convex Optimization of Pipeline Operations</h1>",
    unsafe_allow_html=True
)

# ---------------------
# Sidebar inputs
# ---------------------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")

    # Global fluid + cost
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=2000.0, step=10.0)
        KV        = st.number_input("Viscosity (cSt)",     value=10.0,    step=0.1)
        rho       = st.number_input("Density (kg/m³)",     value=880.0,   step=10.0)
        RateDRA   = st.number_input("DRA Rate (INR/L)",    value=500.0,   step=0.1)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0,    step=0.5)

    # Build as many intermediate stations as you like
    st.markdown("----")
    st.markdown("### 🚰 Define Pumping Stations")
    add_col, rem_col = st.columns(2)
    add_btn = add_col.button("➕ Add Station")
    rem_btn = rem_col.button("❌ Remove Last")

    if 'stations' not in st.session_state:
        st.session_state.stations = [
            {
              'name':     'Station 1',
              'elev':     0.0,
              'D':        0.71120,
              't':        0.00714,
              'SMYS':     52000.0,
              'rough':    0.00004,
              'L':        50.0,
              'is_pump':  True,
              'power_type': 'Grid',
              'rate':     9.0,
              'sfc':      150.0,
              'max_pumps': 3,
              'MinRPM':   1200.0,
              'DOL':      1500.0,
              'max_dr':   40.0,
              # pump‐curve & eff‐curve coeffs:
              'A': -2e-6, 'B': -0.0015, 'C': 179.14,
              'P': -4.161e-14, 'Q': 6.574e-10, 'R': -8.737e-06, 'S': 0.04924, 'T': -0.001754
            }
        ]

    if add_btn:
        n = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name':       f'Station {n}',
            'elev':       0.0,
            'D':          0.71120,
            't':          0.00714,
            'SMYS':       52000.0,
            'rough':      0.00004,
            'L':          50.0,
            'is_pump':    True,
            'power_type': 'Diesel',
            'rate':       9.0,
            'sfc':        150.0,
            'max_pumps':  2,
            'MinRPM':     2750.0,
            'DOL':        3437.0,
            'max_dr':     40.0,
            'A': -1e-5, 'B': 0.00135, 'C': 270.08,
            'P': -4.07e-13, 'Q': 3.4657e-09, 'R': -1.9273e-05, 'S': 0.067033, 'T': -0.15043
        })
    if rem_btn and len(st.session_state.stations)>1:
        st.session_state.stations.pop()

    # Per‐station inputs
    for idx, stn in enumerate(st.session_state.stations, start=1):
        with st.expander(f"Station {idx}: {stn['name']}", expanded=True):
            stn['name']  = st.text_input("Name",     value=stn['name'],  key=f"name{idx}")
            stn['elev']  = st.number_input(
                              "Elevation (m)",
                              value=stn['elev'],
                              key=f"elev{idx}",
                              step=0.00001,
                              format="%.5f"
                          )
            stn['D']     = st.number_input(
                              "Outer Diameter (m)",
                              value=stn['D'],
                              key=f"D{idx}",
                              step=0.00001,
                              format="%.5f"
                          )
            stn['t']     = st.number_input(
                              "Wall Thickness (m)",
                              value=stn['t'],
                              key=f"t{idx}",
                              step=0.00001,
                              format="%.5f"
                          )
            stn['SMYS']  = st.number_input("SMYS (psi)",  value=stn['SMYS'], key=f"SMYS{idx}")
            stn['rough'] = st.number_input(
                              "Pipe Roughness (m)",
                              value=stn['rough'],
                              key=f"rough{idx}",
                              step=0.00001,
                              format="%.5f"
                          )
            stn['L']     = st.number_input("Length to next (km)", value=stn['L'], key=f"L{idx}")
            stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")
            if stn['is_pump']:
                stn['power_type'] = st.selectbox(
                    "Power Source",
                    ["Grid", "Diesel"],
                    index=0 if stn['power_type']=='Grid' else 1,
                    key=f"ptype{idx}"
                )
                if stn['power_type']=='Grid':
                    stn['rate'] = st.number_input(
                                      "Electricity Rate (INR/kWh)",
                                      value=stn['rate'], key=f"rate{idx}"
                                  )
                else:
                    stn['sfc'] = st.number_input(
                                      "SFC (gm/bhp-hr)",
                                      value=stn['sfc'], key=f"sfc{idx}"
                                  )
                stn['max_pumps'] = st.number_input(
                                      "Available Pumps",
                                      min_value=1,
                                      value=stn['max_pumps'],
                                      step=1,
                                      key=f"mpumps{idx}"
                                  )
                stn['MinRPM'] = st.number_input("Min RPM",   value=stn['MinRPM'], key=f"minrpm{idx}")
                stn['DOL']    = st.number_input("Rated RPM", value=stn['DOL'],    key=f"dol{idx}")
                stn['max_dr'] = st.number_input(
                                      "Max Drag Reduction (%)",
                                      value=stn['max_dr'], key=f"mdr{idx}"
                                  )
                st.file_uploader("Pump Head Curve (img)",    type=["png","jpg","jpeg"], key=f"headimg{idx}")
                st.file_uploader("Efficiency Curve (img)",   type=["png","jpg","jpeg"], key=f"effimg{idx}")

    # ────────────────────────────────────────────────────────────────────────
    # Terminal Station
    st.markdown("----")
    st.subheader("🏁 Terminal Station")
    terminal_name     = st.text_input("Name",    value="Terminal")
    terminal_elev     = st.number_input(
                            "Elevation (m)",
                            value=0.0,
                            step=0.00001,
                            format="%.5f"
                        )
    terminal_residual = st.number_input(
                            "Required Residual Head (m)",
                            value=50.0,
                            step=0.1
                        )

    run = st.button("🚀 Run Optimization")

# ────────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Optimizing..."):
        res = solve_pipeline(
            st.session_state.stations,
            {
              'name':        terminal_name,
              'elevation':   terminal_elev,
              'min_residual': terminal_residual
            },
            FLOW, KV, rho, RateDRA, Price_HSD
        )

    # —– KPI cards —–
    total_cost  = res['total_cost']
    total_pumps = sum(res[f'num_pumps_{stn["name"].lower()}'] 
                      for stn in st.session_state.stations)
    speeds  = [res[f'speed_{stn["name"].lower()}']       for stn in st.session_state.stations]
    effs    = [res[f'efficiency_{stn["name"].lower()}'] for stn in st.session_state.stations]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost (INR)",          f"₹{total_cost:,.2f}")
    c2.metric("Total Pumps",               f"{total_pumps}")
    c3.metric("Avg Pump Speed (rpm)",      f"{np.mean(speeds):.2f}")
    c4.metric("Avg Pump Efficiency (%)",   f"{np.mean(effs):.2f}")

    # …continue with your summary table & charts as before…
