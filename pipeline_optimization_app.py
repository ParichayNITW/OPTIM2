import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly import graph_objects as go3d
from math import pi

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

st.markdown("# Mixed Integer Nonlinear Pipeline Optimization")

# Solver wrapper
@st.cache_data
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

# Sidebar inputs
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    st.subheader("Stations")
    add_col, rem_col = st.columns(2)
    if add_col.button("➕ Add Station"):
        idx = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name': f'Station {idx}', 'elev': 0.0, 'D': 0.711, 't': 0.007,
            'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'min_residual': 50.0, 'is_pump': False,
            'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
            'max_pumps': 1, 'MinRPM': 1200, 'DOL': 1500, 'max_dr': 0
        })
    if rem_col.button("🗑️ Remove Station") and len(st.session_state.stations) > 1:
        st.session_state.stations.pop()

    view = st.radio(
        "Show results for:",
        ["Summary", "Cost Breakdown", "Performance", "System Curves",
         "Pump-System Interaction", "Cost Landscape", "Nonconvex Visuals"]
    )

# Initialize stations in session
if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200, 'DOL': 1500, 'max_dr': 0
    }]

# Station inputs
for i, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {i}"):
        stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{i}")
        stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{i}")
        stn['KV']   = st.number_input("Viscosity (cSt)", value=stn.get('KV',10.0), step=0.1, key=f"kv{i}")
        stn['rho']  = st.number_input("Density (kg/m³)", value=stn.get('rho',850.0), step=1.0, key=f"rho{i}")
        if i == 1:
            stn['min_residual'] = st.number_input(
                "Required Residual Head (m)", value=stn['min_residual'], step=0.1, key=f"res{i}"
            )
        stn['D']     = st.number_input("Outer Diameter (m)", value=stn['D'], format="%.3f", step=0.001, key=f"D{i}")
        stn['t']     = st.number_input("Wall Thickness (m)", value=stn['t'], format="%.4f", step=1e-4, key=f"t{i}")
        stn['SMYS']  = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{i}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=1e-5, key=f"rough{i}")
        stn['L']     = st.number_input("Length to next (km)", value=stn['L'], min_value=0.0, step=1.0, key=f"L{i}")
        stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{i}")
        if stn['is_pump']:
            stn['power_type'] = st.selectbox("Power Source", ["Grid","Diesel"],
                                            index=0 if stn['power_type']=='Grid' else 1,
                                            key=f"ptype{i}")
            if stn['power_type']=='Grid':
                stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", value=stn['rate'], key=f"rate{i}")
                stn['sfc']  = 0.0
            else:
                stn['sfc']  = st.number_input("SFC (gm/bhp·hr)", value=stn['sfc'], key=f"sfc{i}")
                stn['rate'] = 0.0
            stn['max_pumps'] = st.number_input(
                "Maximum Pumps Available", min_value=1, value=int(stn['max_pumps']), step=1,
                key=f"mpumps{i}"
            )
            stn['MinRPM'] = st.number_input("Min RPM", value=int(stn['MinRPM']), step=100, key=f"minrpm{i}")
            stn['DOL']    = st.number_input("Rated RPM (DOL)", value=int(stn['DOL']), step=100, key=f"dol{i}")
            stn['max_dr'] = st.number_input("Max Drag Reduction (%)", value=int(stn['max_dr']), step=5, key=f"mdr{i}")
            dfh = st.data_editor(pd.DataFrame({"Flow": [0.0], "Head": [0.0]}), num_rows='dynamic', key=f"head{i}")
            st.session_state[f"head{i}"] = dfh
            dfe = st.data_editor(pd.DataFrame({"Flow": [0.0], "Eff": [0.0]}), num_rows='dynamic', key=f"eff{i}")
            st.session_state[f"eff{i}"] = dfe
        peaks_df = st.data_editor(
            pd.DataFrame({"Location": [stn['L']/2], "Elevation": [stn['elev']+100]}),
            num_rows='dynamic', key=f"peak{i}"
        )
        st.session_state[f"peak{i}"] = peaks_df

# Terminal
st.markdown("---")
st.subheader("🏁 Terminal Station")
term_name = st.text_input("Name", "Terminal")
term_elev = st.number_input("Elevation (m)", value=0.0)
term_min  = st.number_input("Required Residual Head (m)", value=50.0)

# Run
if st.button("🚀 Run Optimization"):
    with st.spinner("Solving..."):
        stations_data = st.session_state.stations
        term_data = {'name': term_name, 'elev': term_elev, 'min_residual': term_min}
        for i, stn in enumerate(stations_data, start=1):
            if stn['is_pump']:
                Qh, Hh = st.session_state[f"head{i}"].values.T
                stn['A'], stn['B'], stn['C'] = np.polyfit(Qh, Hh, 2)
                Qe, Ee = st.session_state[f"eff{i}"].values.T
                stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = np.polyfit(Qe, Ee, 4)
            stn['peaks'] = [ {'loc':r[0], 'elev':r[1]} for _, r in st.session_state[f"peak{i}"].iterrows() ]
        res = solve_pipeline(stations_data, term_data, FLOW, RateDRA, Price_HSD)
        st.session_state['res'] = res
        st.session_state['stations_data'] = stations_data

# Results
if 'res' in st.session_state:
    res = st.session_state['res']
    sta = st.session_state['stations_data']

    # create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary", "💰 Cost Breakdown", "⚙️ Performance",
        "🌀 System Curves", "🔄 Pump-System", "🌄 Cost Landscape",
        "🔍 Nonconvex Visuals"
    ])

    # Summary
    if view == "Summary":
        with tab1:
            st.markdown("<div class='section-title'>Summary</div>", unsafe_allow_html=True)
            names = [s['name'] for s in sta] + [term_name]
            rows = ["Cost+Fuel","DRA","No.Pumps","Speed","Eff%","Re","HeadLoss","Vel","ResHead","SDH","DRA%"]
            data = {"Process": rows}
            for nm in names:
                k=nm.lower().replace(' ', '_')
                data[nm] = [
                    res[f"power_cost_{k}"], res[f"dra_cost_{k}"],
                    res[f"num_pumps_{k}"],  res[f"speed_{k}"],
                    res[f"efficiency_{k}"], res[f"reynolds_{k}"],
                    res[f"head_loss_{k}"],   res[f"velocity_{k}"],
                    res[f"residual_head_{k}"], res[f"sdh_{k}"],
                    res[f"drag_reduction_{k}"]
                ]
            st.dataframe(pd.DataFrame(data), use_container_width=True)

    # Cost Breakdown
    elif view == "Cost Breakdown":
        with tab2:
            st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
            dfc = pd.DataFrame({
                "Station": [s['name'] for s in sta],
                "Power+Fuel": [res[f"power_cost_{s['name'].lower().replace(' ','_')}"] for s in sta],
                "DRA": [res[f"dra_cost_{s['name'].lower().replace(' ','_')}"] for s in sta]
            })
            fig = px.bar(dfc.melt(id_vars="Station", value_vars=["Power+Fuel","DRA"], var_name="Type", value_name="INR/day"),
                         x="Station", y="INR/day", color="Type")
            st.plotly_chart(fig, use_container_width=True)

    # Performance
    elif view == "Performance":
        with tab3:
            ptabs = st.tabs(["HeadLoss","Vel&Re","H-Q","Eff","Power-RPM","Power-Flow"])
            # implement each sub-tab similarly...
            st.write("Performance charts here")

    # System Curves
    elif view == "System Curves":
        with tab4:
            st.write("System curves here")

    # Pump-System Interaction
    elif view == "Pump-System Interaction":
        with tab5:
            st.write("Pump-System interaction here")

    # Cost Landscape
    elif view == "Cost Landscape":
        with tab6:
            st.write("3D cost landscape here")

    # Nonconvex Visuals
    elif view == "Nonconvex Visuals":
        with tab7:
            first = sta[0]
            max_dr = int(first.get('max_dr',0))
            opt_dr = int(res[f"drag_reduction_{first['name'].lower().replace(' ','_')}"])
            dr = st.slider("Drag Reduction (%)", 0, max_dr, opt_dr)
            pumps = list(range(1, first['max_pumps']+1))
            rpms  = list(range(first['MinRPM'], first['DOL']+1, 100))
            Z = np.zeros((len(pumps), len(rpms)))
            for i,npumps in enumerate(pumps):
                for j,rpm in enumerate(rpms):
                    stn = dict(first)
                    stn.update({'is_pump':True, 'max_pumps':npumps, 'MinRPM':rpm, 'DOL':rpm, 'max_dr':dr})
                    out = solve_pipeline([stn]+sta[1:],
                                         {'name':term_name,'elev':term_elev,'min_residual':term_min},
                                         FLOW,RateDRA,Price_HSD)
                    Z[i,j] = out['total_cost']
            fig = go3d.Figure(data=[go3d.Surface(x=rpms, y=pumps, z=Z)])
            fig.update_layout(scene=dict(xaxis_title="RPM", yaxis_title="#Pumps", zaxis_title="Cost"))
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Fill inputs and click 🚀 Run Optimization to display results.")

# Footer
st.markdown("---")
st.caption("© 2025 Parichay Das. All rights reserved.")