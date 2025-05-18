import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline_model import solve_pipeline

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(
    page_title="Mixed Integer Non Linear Convex Optimization of Pipeline Operations",
    layout="wide"
)

# ---------------------
# User Inputs
# ---------------------
st.title("Mixed Integer Non Linear Convex Optimization of Pipeline Operations")
st.sidebar.header("Pipeline Inputs")

FLOW = st.sidebar.number_input("Flow rate (m³/hr)", value=3500.0)
KV = st.sidebar.number_input("Viscosity (cSt)", value=20.0)
rho = st.sidebar.number_input("Density (kg/m³)", value=880.0)
RateDRA = st.sidebar.number_input("DRA Rate (INR/L)", value=55.0)
Price_HSD = st.sidebar.number_input("Diesel Price (INR/L)", value=70.0)

stations = []
num_stations = st.sidebar.number_input("Number of Stations", min_value=1, max_value=10, value=2, step=1)

for i in range(num_stations):
    with st.expander(f"Station {i+1} Inputs"):
        name = st.text_input(f"Station {i+1} Name", value=f"Station{i+1}")
        elev = st.number_input(f"Elevation (m) - Station {i+1}", value=0.0, key=f"elev_{i}")
        D = st.number_input(f"Internal Diameter (m) - Station {i+1}", value=0.697, key=f"d_{i}")
        L = st.number_input(f"Length to next (km) - Station {i+1}", value=50.0, key=f"L_{i}")
        rough = st.number_input(f"Pipe Roughness (m) - Station {i+1}", value=0.00004, key=f"rough_{i}")
        is_pump = st.checkbox("Pumping Station?", key=f"pump_{i}")

        stn = {
            'name': name,
            'elev': elev,
            'D': D,
            'L': L,
            'rough': rough,
            'is_pump': is_pump
        }

        if is_pump:
            stn['power_type'] = st.selectbox("Power Source", ["Grid", "Diesel"], key=f"pwr_{i}")
            stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", value=9.0, key=f"rate_{i}")
            stn['max_pumps'] = st.number_input("Available Pumps", value=3, step=1, key=f"maxp_{i}")
            stn['MinRPM'] = st.number_input("Min RPM", value=1200.0, key=f"minrpm_{i}")
            stn['DOL'] = st.number_input("Rated RPM", value=1500.0, key=f"dol_{i}")
            stn['DR'] = st.number_input("Max Drag Reduction (%)", value=40.0, key=f"dr_{i}")

            # Pump curves
            stn['A'] = st.number_input("Head Curve A", value=0.0, key=f"A_{i}")
            stn['B'] = st.number_input("Head Curve B", value=0.0, key=f"B_{i}")
            stn['C'] = st.number_input("Head Curve C", value=0.0, key=f"C_{i}")
            stn['P'] = st.number_input("Eff Curve P", value=0.0, key=f"P_{i}")
            stn['Q'] = st.number_input("Eff Curve Q", value=0.0, key=f"Q_{i}")
            stn['R'] = st.number_input("Eff Curve R", value=0.0, key=f"R_{i}")
            stn['S'] = st.number_input("Eff Curve S", value=0.0, key=f"S_{i}")
            stn['T'] = st.number_input("Eff Curve T", value=0.0, key=f"T_{i}")
            if stn['power_type'] == "Diesel":
                stn['SFC'] = st.number_input("SFC (gm/bhp/hr)", value=210.0, key=f"sfc_{i}")

        stn['RH'] = st.number_input("Initial RH (m)", value=50.0, key=f"rh_{i}") if i == 0 else 50
        stations.append(stn)

with st.expander("Terminal Station"):
    term_name = st.text_input("Terminal Station Name", value="Terminal")
    term_elev = st.number_input("Terminal Elevation (m)", value=10.0)
    term_rh = st.number_input("Required Residual Head (m)", value=50.0)

terminal = {
    'name': term_name,
    'elev': term_elev,
    'min_residual': term_rh
}

# ---------------------
# Run Optimization
# ---------------------
if st.button("🚀 Run Optimization"):
    with st.spinner("Running Optimization using NEOS Couenne Solver..."):
        try:
            results = solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)
            st.success("Optimization completed successfully!")

            st.subheader("Summary Results")
            st.metric("Total Cost (INR/day)", f"₹{results['total_cost']:,.2f}")

            data = []
            cols = ['Station', 'No. of Pumps', 'Speed (RPM)', 'Efficiency (%)',
                    'Power Cost (INR)', 'DRA Cost (INR)', 'Drag Reduction (%)',
                    'Head Loss (m)', 'Residual Head (m)', 'Velocity (m/s)', 'Reynolds No.']
            for stn in stations:
                key = stn['name'].strip().lower()
                row = [
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
                    results.get(f'reynolds_number_{key}', 0),
                ]
                data.append(row)

            key = terminal['name'].strip().lower()
            data.append([
                terminal['name'], '-', '-', '-', '-', '-', '-', '-',
                results.get(f'residual_head_{key}', 0), '-', '-'
            ])

            df = pd.DataFrame(data, columns=cols)
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
