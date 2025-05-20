import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly import graph_objects as go3d
from math import pi
from io import BytesIO
from pyomo.opt import SolverManagerFactory

# NEOS email
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

# Page config
st.set_page_config(page_title="Pipeline Optimization", layout="wide")

# CSS
st.markdown("""
<style>
.section-title {
  font-size:1.2rem;
  font-weight:600;
  margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Mixed Integer Nonlinear Pipeline Optimization", unsafe_allow_html=True)

# Solver wrapper
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

# --- Sidebar Inputs & View Selector --------------------------------------
with st.sidebar:
    st.title("🔧 Inputs")
    with st.expander("Global Fluid & Cost", expanded=True):
        FLOW      = st.number_input("Flow (m³/hr)", 1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", 500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", 70.0, step=0.5)
    st.subheader("Stations")
    add_c, rem_c = st.columns(2)
    if add_c.button("➕ Add Station"): 
        st.session_state.stations.append({
            'name':'Station', 'elev':0.0,'D':0.711,'t':0.007,'SMYS':52000,'rough':4e-5,'L':50.0,
            'min_residual':50.0,'is_pump':False,'power_type':'Grid','rate':9.0,'sfc':150.0,
            'max_pumps':1,'MinRPM':1200,'DOL':1500,'max_dr':0
        })
    if rem_c.button("🗑️ Remove Station") and len(st.session_state.stations)>1:
        st.session_state.stations.pop()
    view = st.radio("View", [
        "Summary","Cost Breakdown","Performance",
        "System Curves","Pump-System Interaction",
        "Cost Landscape","Nonconvex Visuals"
    ])

# Initialize stations
if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name':'Station 1','elev':0.0,'D':0.711,'t':0.007,'SMYS':52000,'rough':4e-5,'L':50.0,
        'min_residual':50.0,'is_pump':False,'power_type':'Grid','rate':9.0,'sfc':150.0,
        'max_pumps':1,'MinRPM':1200,'DOL':1500,'max_dr':0
    }]

# Station inputs
for i, stn in enumerate(st.session_state.stations, 1):
    with st.expander(f"Station {i}"):
        stn['name'] = st.text_input("Name", stn.get('name', f"Station {i}"), key=f"name{i}")
        stn['elev'] = st.number_input("Elev (m)", stn.get('elev', 0.0), step=0.1, key=f"elev{i}")
        stn['KV']   = st.number_input("Viscosity (cSt)", stn.get('KV', 10.0), step=0.1, key=f"kv{i}")
        stn['rho']  = st.number_input("Density (kg/m³)", stn.get('rho', 850), step=1, key=f"rho{i}")
        if i == 1:
            stn['min_residual'] = st.number_input(
                "Min RH (m)", stn.get('min_residual', 50), step=0.1, key=f"res{i}"
            )
        stn['D']     = st.number_input("Outer D (m)", stn.get('D', 0.7), format="%.3f", step=0.001, key=f"D{i}")
        stn['t']     = st.number_input("t (m)", stn.get('t', 0.007), format="%.4f", step=1e-4, key=f"t{i}")
        stn['rough'] = st.number_input("Rough (m)", stn.get('rough', 4e-5), format="%.5f", step=1e-5, key=f"rough{i}")
        stn['L'] = st.number_input(
            "Length to next (km)",
            value=float(stn.get('L', 50.0)),
            min_value=0.0,
            step=1.0,
            key=f"L{i}"
        )
        stn['is_pump'] = st.checkbox("Pump?", stn.get('is_pump', False), key=f"pump{i}")

        if stn['is_pump']:
            stn['power_type'] = st.selectbox("Power", ["Grid", "Diesel"], key=f"ptype{i}")
            if stn['power_type'] == 'Grid':
                stn['rate'] = st.number_input("Rate (INR/kWh)", stn.get('rate', 9), key=f"rate{i}")
                stn['sfc'] = 0
            else:
                stn['sfc'] = st.number_input("SFC (gm/bhp·hr)", stn.get('sfc', 150), key=f"sfc{i}")
                stn['rate'] = 0
            stn['max_pumps'] = st.number_input(
                "Maximum Pumps Available",
                min_value=1,
                value=stn.get('max_pumps', 1),
                step=1,
                key=f"mpumps{i}"
            )
            stn['MinRPM'] = st.number_input("Min RPM", stn.get('MinRPM', 1200), key=f"minrpm{i}")
            stn['DOL']    = st.number_input("DOL RPM", stn.get('DOL', 1500), key=f"dol{i}")
            stn['max_dr'] = st.number_input("Max DRA (%)", stn.get('max_dr', 0), key=f"mdr{i}")
            # pump tables
            st.markdown("**Flow–Head**")
            dfh = st.data_editor(pd.DataFrame({"Q": [0.0], "H": [0.0]}), num_rows='dynamic', key=f"head{i}")
            st.session_state[f"head{i}"] = dfh
            st.markdown("**Flow–Eff**")
            dfe = st.data_editor(pd.DataFrame({"Q": [0.0], "η%": [0.0]}), num_rows='dynamic', key=f"eff{i}")
            st.session_state[f"eff{i}"] = dfe
        st.markdown("**Peaks**")
        pk = st.data_editor(
            pd.DataFrame({"loc_km": [stn['L']/2], "elev": [stn['elev']+100]}),
            num_rows='dynamic', key=f"peak{i}"
        )
        st.session_state[f"peak{i}"] = pk

# Terminal
st.markdown("---")
st.subheader("Terminal")
term_name = st.text_input("Name", "Terminal")
term_elev = st.number_input("Elev (m)", 0.0)
term_min  = st.number_input("Min RH (m)", 50)

# Run optimization
if st.button("🚀 Run"):
    with st.spinner("Solving..."):
        for i, stn in enumerate(st.session_state.stations, 1):
            if stn['is_pump']:
                Qh, Hh = st.session_state[f"head{i}"].values.T
                try:
                    a, b, c = np.polyfit(Qh, Hh, 2)
                except np.linalg.LinAlgError:
                    st.error(f"Station {i} head fit failed.")
                    st.stop()
                stn.update({'A': a, 'B': b, 'C': c})
                Qe, Ee = st.session_state[f"eff{i}"].values.T
                try:
                    P, Qc, R, S, T = np.polyfit(Qe, Ee, 4)
                except np.linalg.LinAlgError:
                    st.error(f"Station {i} eff fit failed.")
                    st.stop()
                stn.update({'P': P, 'Q': Qc, 'R': R, 'S': S, 'T': T})
            stn['peaks'] = [
                {'loc': row.loc_km, 'elev': row.elev}
                for _, row in st.session_state[f"peak{i}"].iterrows()
            ]
        res = solve_pipeline(
            st.session_state.stations,
            {'name': term_name, 'elev': term_elev, 'min_residual': term_min},
            FLOW, RateDRA, Price_HSD
        )
        st.session_state['res'] = res
        st.session_state['stations_data'] = st.session_state.stations

# Render results
if 'res' in st.session_state:
    res = st.session_state['res']
    sta = st.session_state['stations_data']

    # Downloads
    with st.sidebar:
        st.markdown("---")
        df_glob = pd.DataFrame([{
            'FLOW': FLOW, 'RateDRA': RateDRA, 'Price_HSD': Price_HSD,
            'Terminal': term_name, 'Elev': term_elev, 'MinRH': term_min
        }])
        st.download_button(
            "Scenario CSV",
            df_glob.to_csv(index=False).encode(),
            "scenario.csv", "text/csv"
        )
        df_sta = pd.DataFrame(sta)
        st.download_button(
            "Stations CSV",
            df_sta.to_csv(index=False).encode(),
            "stations.csv", "text/csv"
        )
    # Summary
    if view == "Summary":
        st.markdown("<div class='section-title'>Summary</div>", unsafe_allow_html=True)
        names = [s['name'] for s in sta] + [term_name]
        rows  = ["Cost","DRA","Pumps","Speed","Eff","Re","HLoss","Vel","ResRH","SDH","DRA%"]
        data  = {"Process": rows}
        for nm in names:
            k = nm.lower().replace(' ', '_')
            data[nm] = [
                res[f"power_cost_{k}"], res[f"dra_cost_{k}"],
                res[f"num_pumps_{k}"], res[f"speed_{k}"],
                res[f"efficiency_{k}"], res[f"reynolds_{k}"],
                res[f"head_loss_{k}"], res[f"velocity_{k}"],
                res[f"residual_head_{k}"], res[f"sdh_{k}"],
                res[f"drag_reduction_{k}"]
            ]
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    # Cost Breakdown
    elif view == "Cost Breakdown":
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        dfc = pd.DataFrame({
            "Station": [s['name'] for s in sta],
            "Cost": [res[f"power_cost_{s['name'].lower().replace(' ', '_')}" ] for s in sta],
            "DRA":  [res[f"dra_cost_{s['name'].lower().replace(' ', '_')}" ] for s in sta]
        })
        fig = px.bar(
            dfc.melt(id_vars="Station", value_vars=["Cost", "DRA"], var_name="Type", value_name="INR/day"),
            x="Station", y="INR/day", color="Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    # Performance
    elif view == 
