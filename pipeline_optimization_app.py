import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# Solver wrapper to call the backend model
def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)

# Page setup
st.set_page_config(page_title="Pipeline Optimization", layout="wide")
st.title("Pipeline Network Optimization")

# Sidebar: Global inputs and station configuration
with st.sidebar:
    st.header("🔧 Global Inputs")
    FLOW      = st.number_input("Flow rate (m³/hr)", value=2000.0, step=10.0)
    KV        = st.number_input("Viscosity (cSt)",    value=10.0,    step=0.1)
    rho       = st.number_input("Density (kg/m³)",    value=880.0,   step=10.0)
    RateDRA   = st.number_input("DRA Rate (INR/L)",   value=500.0,   step=0.1)
    Price_HSD = st.number_input("Diesel Price (INR/L)",value=70.0,    step=0.5)

    # Add/Remove stations
    add_col, rem_col = st.columns(2)
    add_btn = add_col.button("➕ Add Station")
    rem_btn = rem_col.button("🗑️ Remove Station")
    if 'stations' not in st.session_state:
        # Initialize with one station
        st.session_state.stations = [{
            'name': 'Station 1', 'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Grid', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 2, 'MinRPM': 1000.0, 'DOL': 1200.0,
            'max_dr': 40.0,
            'A': -2e-6, 'B': -0.0015, 'C': 179.14,
            'P': -4.161e-14, 'Q': 6.574e-10, 'R': -8.737e-06, 'S': 0.04924, 'T': -0.001754
        }]
    if add_btn:
        n = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name': f'Station {n}', 'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Grid', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 2, 'MinRPM': 1000.0, 'DOL': 1200.0,
            'max_dr': 40.0,
            'A': -2e-6, 'B': -0.0015, 'C': 179.14,
            'P': -4.161e-14, 'Q': 6.574e-10, 'R': -8.737e-06, 'S': 0.04924, 'T': -0.001754
        })
    if rem_btn and len(st.session_state.stations) > 1:
        st.session_state.stations.pop()

    # Station details input form
    for idx, stn in enumerate(st.session_state.stations, start=1):
        with st.expander(f"Station {idx}: {stn['name']}", expanded=True):
            stn['name']  = st.text_input("Name", value=stn['name'], key=f"name{idx}")
            stn['elev']  = st.number_input("Elevation (m)", value=stn['elev'], key=f"elev{idx}")
            stn['D']     = st.number_input("Outer Diameter (m)", value=stn['D'], step=0.00001, format="%.5f", key=f"D{idx}")
            stn['t']     = st.number_input("Wall Thickness (m)", value=stn['t'], step=0.00001, format="%.5f", key=f"t{idx}")
            stn['SMYS']  = st.number_input("SMYS (psi)", value=stn['SMYS'], key=f"SMYS{idx}")
            stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], step=0.00001, format="%.5f", key=f"rough{idx}")
            stn['L']     = st.number_input("Length to next (km)", value=stn['L'], key=f"L{idx}")
            stn['is_pump'] = st.checkbox("Pumping Station?", value=stn.get('is_pump', False), key=f"pump{idx}")
            if stn['is_pump']:
                stn['power_type'] = st.selectbox("Power Source", ["Grid","Diesel"],
                                                index=(0 if stn.get('power_type')=="Grid" else 1), key=f"ptype{idx}")
                if stn['power_type'] == "Grid":
                    stn['rate'] = st.number_input("Electricity Rate (INR/kWh)",
                                                  value=stn.get('rate', 9.0), key=f"rate{idx}")
                else:
                    stn['sfc']  = st.number_input("SFC (gm/bhp-hr)",
                                                  value=stn.get('sfc', 150.0), key=f"sfc{idx}")
                stn['max_pumps'] = st.number_input("Available Pumps", min_value=1,
                                                   value=stn.get('max_pumps',1), step=1, key=f"mpumps{idx}")
                stn['MinRPM']    = st.number_input("Min RPM", value=stn.get('MinRPM',0.0), key=f"minrpm{idx}")
                stn['DOL']       = st.number_input("Rated RPM", value=stn.get('DOL',0.0), key=f"dol{idx}")
                stn['max_dr']    = st.number_input("Max Drag Reduction (%)",
                                                   value=stn.get('max_dr',0.0), key=f"mdr{idx}")

    st.markdown("---")
    st.subheader("🏁 Terminal Station")
    terminal_name   = st.text_input("Name", value="Terminal")
    terminal_elev   = st.number_input("Elevation (m)", value=0.0, step=0.00001, format="%.5f")
    run = st.button("🚀 Run Optimization")

# Run optimization and display results
if run:
    with st.spinner("Solving pipeline optimization..."):
        stations_data = st.session_state.stations
        terminal_data = {"name": terminal_name, "elevation": terminal_elev}
        res = solve_pipeline(stations_data, terminal_data, FLOW, KV, rho, RateDRA, Price_HSD)

    # KPI cards
    total_cost = res.get('total_cost', 0)
    total_pumps = sum(int(res.get(f"num_pumps_{s['name'].lower()}", 0)) for s in stations_data)
    speeds = [res.get(f"speed_{s['name'].lower()}", 0) for s in stations_data]
    effs   = [res.get(f"efficiency_{s['name'].lower()}", 0) for s in stations_data]
    avg_speed = np.mean(speeds) if speeds else 0
    avg_eff   = np.mean(effs) if effs else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cost (INR)", f"₹{total_cost:,.2f}")
    col2.metric("Total Pumps", total_pumps)
    col3.metric("Avg Speed (rpm)", f"{avg_speed:.2f}")
    col4.metric("Avg Efficiency (%)", f"{avg_eff:.2f}")

    # Prepare cost summary table
    station_names = [s['name'] for s in stations_data]
    cost_data = {
        "Station": station_names,
        "Power & Fuel (INR/day)": [res.get(f"power_cost_{name.lower()}", 0) for name in station_names],
        "DRA (INR/day)":            [res.get(f"dra_cost_{name.lower()}",  0) for name in station_names]
    }
    df_cost = pd.DataFrame(cost_data)

    # Display results in two tabs
    tab1, tab2 = st.tabs(["📋 Summary Table", "💰 Cost Charts"])
    with tab1:
        st.markdown("**Optimized Pipeline Costs by Station**")
        st.dataframe(df_cost.set_index("Station"), use_container_width=True)
        csv_bytes = df_cost.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary as CSV", csv_bytes, "pipeline_summary.csv", "text/csv")

    with tab2:
        st.markdown("**Cost Breakdown per Station**")
        df_melt = df_cost.melt(id_vars="Station", value_vars=["Power & Fuel (INR/day)", "DRA (INR/day)"],
                                var_name="Cost Type", value_name="Amount (INR)")
        fig_cost = px.bar(df_melt, x="Station", y="Amount (INR)", color="Cost Type", barmode="group",
                          title="Power & Fuel vs DRA Cost by Station")
        fig_cost.update_layout(xaxis_title="Station", yaxis_title="Cost (INR)")
        fig_cost.update_yaxes(tickformat=".2f")
        st.plotly_chart(fig_cost, use_container_width=True)
