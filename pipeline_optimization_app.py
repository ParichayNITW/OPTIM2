import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO
from pipeline_model import solve_pipeline
from math import pi

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(
    page_title="Pipeline Optimization App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Cross-Country Pipeline Optimization")

# ---------------------
# Sidebar Inputs
# ---------------------
st.sidebar.title("Pipeline Parameters")
FLOW = st.sidebar.number_input("Flow rate (m³/hr)", value=3500.0)
KV = st.sidebar.number_input("Kinematic Viscosity (cSt)", value=10.0)
rho = st.sidebar.number_input("Density (kg/m³)", value=880.0)
RateDRA = st.sidebar.number_input("DRA Cost (INR/L)", value=60.0)
Price_HSD = st.sidebar.number_input("Diesel Price (INR/L)", value=70.0)

# ---------------------
# Station Inputs
# ---------------------
st.sidebar.markdown("---")
st.sidebar.title("Stations Setup")
num_stations = st.sidebar.number_input("Number of Stations", min_value=2, value=2, step=1)

stations = []
station_tabs = st.tabs([f"Station {i+1}" for i in range(num_stations)])

for i in range(num_stations):
    with station_tabs[i]:
        name = st.text_input(f"Station {i+1} Name", value=f"Station{i+1}")
        elev = st.number_input(f"Elevation (m) - Station {i+1}", value=0.0, key=f"elev_{i}")
        D = st.number_input(f"Outer Diameter (m) - Station {i+1}", value=0.7112, key=f"D_{i}")
        t = st.number_input(f"Wall Thickness (m) - Station {i+1}", value=0.0071, key=f"t_{i}")
        L = st.number_input(f"Length to next (km) - Station {i+1}", value=50.0, key=f"L_{i}")
        rough = st.number_input(f"Pipe Roughness (m) - Station {i+1}", value=0.00004, key=f"rough_{i}")
        SMYS = st.number_input(f"SMYS (psi) - Station {i+1}", value=52000, key=f"SMYS_{i}")
        DF = st.number_input(f"Design Factor - Station {i+1}", value=0.72, key=f"DF_{i}")

        is_pump = st.checkbox(f"Pumping Station?", key=f"pump_{i}")
        station = {
            'name': name, 'elev': elev, 'D': D, 't': t, 'L': L, 'rough': rough,
            'SMYS': SMYS, 'DF': DF, 'is_pump': is_pump
        }

        if i == 0:
            station['min_residual'] = st.number_input("Initial Residual Head (m)", value=50.0, key=f"RH_start")

        if is_pump:
            power_type = st.selectbox("Power Source", ["Grid", "Diesel"], key=f"ptype_{i}")
            station['power_type'] = power_type
            if power_type == "Grid":
                station['rate'] = st.number_input("Electricity Rate (INR/kWh)", value=9.0, key=f"rate_{i}")
            else:
                station['SFC'] = st.number_input("SFC (gm/bhp/hr)", value=210.0, key=f"sfc_{i}")

            station['max_pumps'] = st.number_input("Max Pumps", value=2, key=f"maxp_{i}", step=1)
            station['MinRPM'] = st.number_input("Min RPM", value=1200.0, key=f"minrpm_{i}")
            station['DOL'] = st.number_input("Rated RPM", value=1500.0, key=f"dol_{i}")
            station['max_dr'] = st.number_input("Max Drag Reduction (%)", value=40.0, key=f"dr_{i}")

            st.markdown("#### Enter Flow vs Head Data")
            df_head = st.data_editor(pd.DataFrame({"Flow (m³/hr)": [1000, 2000, 3000], "Head (m)": [100, 80, 60]}), key=f"head_{i}")
            st.markdown("#### Enter Flow vs Efficiency Data")
            df_eff = st.data_editor(pd.DataFrame({"Flow (m³/hr)": [1000, 2000, 3000], "Efficiency (%)": [70, 80, 75]}), key=f"eff_{i}")

            station['head_curve'] = df_head
            station['eff_curve'] = df_eff

            st.markdown("#### Enter Intermediate Elevation Peaks")
            df_peaks = st.data_editor(pd.DataFrame({"Location (km)": [], "Elevation (m)": []}), key=f"peaks_{i}")
            station['peaks'] = df_peaks.to_dict("records")

        stations.append(station)

# ---------------------
# Terminal Inputs
# ---------------------
with st.expander("Terminal Station"):
    terminal_name = st.text_input("Terminal Station Name", value="Terminal")
    term_elev = st.number_input("Elevation (m) - Terminal", value=0.0)
    min_residual = st.number_input("Required Residual Head (m)", value=50.0)

terminal = {
    'name': terminal_name,
    'elev': term_elev,
    'min_residual': min_residual
}

# ---------------------
# Run Optimization
# ---------------------
if st.button("Run Optimization"):
    with st.spinner("Running optimization via NEOS Couenne Solver..."):
        try:
            results = solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)
            st.success("Optimization completed successfully!")

            st.subheader("Summary Results")
            st.metric("Total Cost (INR/day)", f"₹{results['total_cost']:,.2f}")

            df_data = []
            for stn in stations:
                key = stn['name'].strip().lower().replace(' ','_')
                df_data.append([
                    stn['name'],
                    results.get(f'num_pumps_{key}', 0),
                    results.get(f'speed_{key}', 0),
                    results.get(f'efficiency_{key}', 0),
                    results.get(f'power_cost_{key}', 0),
                    results.get(f'dra_cost_{key}', 0),
                    results.get(f'drag_reduction_{key}', 0),
                    results.get(f'head_loss_{key}', 0),
                    results.get(f'residual_head_{key}', 0),
                    results.get(f'velocity_{key}', 0),
                    results.get(f'reynolds_{key}', 0)
                ])
            df = pd.DataFrame(df_data, columns=[
                "Station", "No. of Pumps", "Speed (RPM)", "Efficiency (%)",
                "Power Cost (INR)", "DRA Cost (INR)", "Drag Reduction (%)",
                "Head Loss (m)", "Residual Head (m)", "Velocity (m/s)", "Reynolds No."
            ])
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
