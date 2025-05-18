# pipeline_optimization_app.py (frontend)

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pyomo.opt import SolverManagerFactory

# Set NEOS email from secrets if provided
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 NEOS_EMAIL not found in secrets.")

st.set_page_config(
    page_title="Pipeline Optimization",
    layout="wide"
)

# Solver wrapper
def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)

# Sidebar: Inputs
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=2000.0, step=10.0)
        KV        = st.number_input("Viscosity (cSt)",    value=10.0,    step=0.1)
        rho       = st.number_input("Density (kg/m³)",    value=880.0,   step=10.0)
        RateDRA   = st.number_input("DRA Rate (INR/L)",   value=500.0,   step=0.1)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0,   step=0.5)

    # Buttons to add/remove stations
    add_col, rem_col = st.columns(2)
    add_btn = add_col.button("➕ Add Station")
    rem_btn = rem_col.button("🗑️ Remove Station")

    if 'stations' not in st.session_state:
        st.session_state.stations = [{
            'name': 'Station 1', 'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Grid', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 3, 'MinRPM': 1200.0, 'DOL': 1500.0,
            'max_dr': 40.0,
            'A': -2e-6, 'B': -0.0015, 'C': 179.14,
            'P': -4.161e-14, 'Q': 6.574e-10, 'R': -8.737e-06, 'S': 0.04924, 'T': -0.001754
        }]

    if add_btn:
        n = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name': f'Station {n}', 'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Diesel', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 2, 'MinRPM': 2750.0, 'DOL': 3437.0,
            'max_dr': 40.0,
            'A': -1e-5, 'B': 0.00135, 'C': 270.08,
            'P': -4.07e-13, 'Q': 3.4657e-09, 'R': -1.9273e-05, 'S': 0.067033, 'T': -0.15043
        })
    if rem_btn and len(st.session_state.stations) > 1:
        st.session_state.stations.pop()

    # Station inputs
    for idx, stn in enumerate(st.session_state.stations, start=1):
        with st.expander(f"Station {idx}: {stn['name']}", expanded=True):
            stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
            stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], key=f"elev{idx}")
            # **Origin residual head input for Station 1**
            if idx == 1:
                stn['min_residual'] = st.number_input(
                    "Initial Station Residual Head (m)",
                    value=stn.get('min_residual', 50.0),
                    step=0.1, key=f"minres{idx}"
                )
            stn['D'] = st.number_input("Outer Diameter (m)", value=stn['D'], step=0.00001, format="%.5f", key=f"D{idx}")
            stn['t'] = st.number_input("Wall Thickness (m)", value=stn['t'], step=0.00001, format="%.5f", key=f"t{idx}")
            stn['SMYS'] = st.number_input("SMYS (psi)", value=stn['SMYS'], key=f"SMYS{idx}")
            stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], step=0.00001, format="%.5f", key=f"rough{idx}")
            stn['L'] = st.number_input("Length to next (km)", value=stn['L'], key=f"L{idx}")
            stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")
            if stn['is_pump']:
                stn['power_type'] = st.selectbox("Power Source", ["Grid","Diesel"],
                                                index=0 if stn['power_type']=="Grid" else 1,
                                                key=f"ptype{idx}")
                if stn['power_type']=="Grid":
                    stn['rate'] = st.number_input("Electricity Rate (INR/kWh)",
                                                  value=stn['rate'], key=f"rate{idx}")
                else:
                    stn['sfc']  = st.number_input("SFC (gm/bhp-hr)",
                                                  value=stn['sfc'], key=f"sfc{idx}")
                stn['max_pumps'] = st.number_input("Available Pumps",
                                                   min_value=1, value=stn['max_pumps'],
                                                   step=1, key=f"mpumps{idx}")
                stn['MinRPM'] = st.number_input("Min RPM", value=stn['MinRPM'], key=f"minrpm{idx}")
                stn['DOL']    = st.number_input("Rated RPM", value=stn['DOL'], key=f"dol{idx}")
                stn['max_dr'] = st.number_input("Max Drag Reduction (%)", value=stn['max_dr'], key=f"mdr{idx}")
                st.file_uploader("Pump Head Curve (img)", key=f"headimg{idx}", type=["png","jpg","jpeg"])
                st.file_uploader("Efficiency Curve (img)", key=f"effimg{idx}", type=["png","jpg","jpeg"])

    # Terminal station inputs
    st.markdown("---")
    st.subheader("🏁 Terminal Station")
    terminal_name  = st.text_input("Terminal Station Name", value="Terminal")
    terminal_elev  = st.number_input("Terminal Elevation (m)", value=0.0, format="%.5f")
    residual_head  = st.number_input("Required Residual Head (m)", value=50.0, step=0.1)

    run = st.button("🚀 Run Optimization")

# Run solver when button clicked
if run:
    with st.spinner("Solving pipeline optimization..."):
        stations_data = st.session_state.stations
        terminal_data = {
            "name":        terminal_name,
            "elev":        terminal_elev,
            "min_residual": residual_head
        }
        res = solve_pipeline(stations_data, terminal_data, FLOW, KV, rho, RateDRA, Price_HSD)

    # KPI Cards
    total_cost = res.get('total_cost', 0)
    total_pumps = sum(res.get(f"num_pumps_{s['name'].lower()}", 0) for s in stations_data)
    speeds = [res.get(f"speed_{s['name'].lower()}", 0) for s in stations_data]
    effs = [res.get(f"efficiency_{s['name'].lower()}", 0) for s in stations_data]
    avg_speed = np.mean(speeds) if speeds else 0
    avg_eff   = np.mean(effs) if effs else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost (INR)", f"₹{total_cost:,.2f}")
    c2.metric("Total Pumps", total_pumps)
    c3.metric("Avg Speed (rpm)", f"{avg_speed:.2f}")
    c4.metric("Avg Efficiency (%)", f"{avg_eff:.2f}")

    # Summary Table
    station_names = [s['name'] for s in stations_data] + [terminal_name]
    summary = {"Process Particulars": [
        "Power & Fuel cost (INR/day)", "DRA cost (INR/day)", "No. of Pumps",
        "Pump Speed (rpm)", "Pump Efficiency (%)", "Reynold's No.",
        "Dynamic Head Loss (m)", "Velocity (m/s)", "Residual Head (m)",
        "SDH (m)", "Drag Reduction (%)"
    ]}
    for s in station_names:
        key = s.lower()
        num = int(res.get(f"num_pumps_{key}", 0))
        sp = res.get(f"speed_{key}", 0) if num > 0 else 0
        ef = res.get(f"efficiency_{key}", 0) if num > 0 else 0
        summary[s] = [
            round(res.get(f"power_cost_{key}", 0), 2),
            round(res.get(f"dra_cost_{key}", 0), 2),
            num,
            round(sp, 2),
            round(ef, 2),
            round(res.get(f"reynolds_{key}", 0), 2),
            round(res.get(f"head_loss_{key}", 0), 2),
            round(res.get(f"velocity_{key}", 0), 2),
            round(res.get(f"residual_head_{key}", 0), 2),
            round(res.get(f"sdh_{key}", 0), 2),
            round(res.get(f"drag_reduction_{key}", 0), 2)
        ]
    df_sum = pd.DataFrame(summary)
    fmt = {col: "{:.2f}" for col in df_sum.columns if col != "Process Particulars"}
    st.dataframe(df_sum.style.format(fmt).set_properties(**{'text-align':'center'}), use_container_width=True)
