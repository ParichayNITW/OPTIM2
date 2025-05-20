# pipeline_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi

from io import BytesIO
from pyomo.opt import SolverManagerFactory

# NEOS email (stored in Streamlit secrets)
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")
st.set_page_config(page_title="Pipeline Optimization", layout="wide")

st.markdown("""
<style>
.section-title {
  font-size:1.2rem; font-weight:600; margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>Mixed Integer Non Linear Convex Optimization of Pipeline Operations</h1>", unsafe_allow_html=True)

# Solver call
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

    # Dynamic station list
    st.subheader("Stations")
    add_col, rem_col = st.columns(2)
    if add_col.button("➕ Add Station"):
        n = len(st.session_state.get('stations',[])) + 1
        default = {
            'name': f'Station {n}', 'elev': 0.0, 'D': 0.711, 't': 0.007,
            'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'min_residual': 50.0, 'is_pump': False,
            'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
            'max_pumps': 1, 'MinRPM': 1000.0, 'DOL': 1500.0,
            'max_dr': 0.0
        }
        st.session_state.stations.append(default)
    if rem_col.button("🗑️ Remove Station"):
        if st.session_state.get('stations'):
            st.session_state.stations.pop()

# Initialize station list if not present
if 'stations' not in st.session_state:
    st.session_state.stations = []
    # Add one default station
    st.session_state.stations.append({
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
        'max_dr': 0.0
    })

# Station inputs (dynamic)
for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}", expanded=True):
        stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
        stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{idx}")
        
        # Station-specific viscosity & density
        stn['KV'] = st.number_input(
            "Viscosity (cSt)",
            value=stn.get('KV', 10.0),
            step=0.1,
            key=f"kv_{idx}"
        )
        stn['rho'] = st.number_input(
            "Density (kg/m³)",
            value=stn.get('rho', 850.0),
            step=1.0,
            key=f"rho_{idx}"
        )

        if idx == 1:
            stn['min_residual'] = st.number_input("Residual Head at Station (m)", value=stn.get('min_residual',50.0), step=0.1, key=f"res{idx}")
        stn['D'] = st.number_input("Outer Diameter (m)", value=stn['D'], format="%.3f", step=0.001, key=f"D{idx}")
        stn['t'] = st.number_input("Wall Thickness (m)", value=stn['t'], format="%.4f", step=0.0001, key=f"t{idx}")
        stn['SMYS'] = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{idx}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=0.00001, key=f"rough{idx}")
        stn['L'] = st.number_input("Length to next (km)", value=stn['L'], step=1.0, key=f"L{idx}")
        stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")
        if stn['is_pump']:
            stn['power_type'] = st.selectbox("Power Source", ["Grid", "Diesel"],
                                            index=0 if stn['power_type']=="Grid" else 1, key=f"ptype{idx}")
            if stn['power_type']=="Grid":
                stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", value=stn.get('rate',9.0), key=f"rate{idx}")
                stn['sfc'] = 0.0
            else:
                stn['sfc'] = st.number_input("SFC (gm/bhp·hr)", value=stn.get('sfc',150.0), key=f"sfc{idx}")
                stn['rate'] = 0.0
            stn['max_pumps'] = st.number_input("Max Pumps Available", min_value=1, value=stn['max_pumps'], step=1, key=f"mpumps{idx}")
            stn['MinRPM'] = st.number_input("Min RPM", value=stn['MinRPM'], key=f"minrpm{idx}")
            stn['DOL'] = st.number_input("Rated RPM (DOL)", value=stn['DOL'], key=f"dol{idx}")
            stn['max_dr'] = st.number_input("Max Drag Reduction (%)", value=stn['max_dr'], key=f"mdr{idx}")
            # Pump performance data tables
            st.markdown("**Enter Pump Performance Data:**")
            st.write("Flow vs Head data (m³/hr, m)")
            df_head = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
            df_head = st.data_editor(df_head, num_rows="dynamic", key=f"head{idx}")
            st.write("Flow vs Efficiency data (m³/hr, %)")
            df_eff = pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})
            df_eff = st.data_editor(df_eff, num_rows="dynamic", key=f"eff{idx}")
            # Store these tables in session_state
            st.session_state[f"head_data_{idx}"] = df_head
            st.session_state[f"eff_data_{idx}"] = df_eff

        # Elevation peaks between this station and next
        st.markdown("**Intermediate Elevation Peaks (to next station):**")
        default_peak = pd.DataFrame({"Location (km)": [stn['L']/2.0], "Elevation (m)": [stn['elev']+100.0]})
        peak_df = st.data_editor(default_peak, num_rows="dynamic", key=f"peak{idx}")
        st.session_state[f"peak_data_{idx}"] = peak_df

# Terminal inputs
st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value="Terminal")
terminal_elev = st.number_input("Elevation (m)", value=0.0, step=0.1)
terminal_head = st.number_input("Required Residual Head (m)", value=50.0, step=1.0)

# Run optimization
run = st.button("🚀 Run Optimization")
if run:
    with st.spinner("Solving optimization..."):
        stations_data = st.session_state.stations
        term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}
        # Attach pump curve data to stations
        for idx, stn in enumerate(stations_data, start=1):
            if stn.get('is_pump', False):
                dfh = st.session_state.get(f"head_data_{idx}")
                dfe = st.session_state.get(f"eff_data_{idx}")
                # Validate data points
                if dfh is None or dfe is None or len(dfh)<3 or len(dfe)<5:
                    st.error(f"Station {idx}: At least 3 points for flow-head and 5 for flow-eff are required.")
                    st.stop()
                # Fit polynomials
                Qh = dfh.iloc[:,0].values; Hh = dfh.iloc[:,1].values
                coeff = np.polyfit(Qh, Hh, 2)
                stn['A'], stn['B'], stn['C'] = coeff[0], coeff[1], coeff[2]
                Qe = dfe.iloc[:,0].values; Ee = dfe.iloc[:,1].values
                coeff_e = np.polyfit(Qe, Ee, 4)
                stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = coeff_e
            # Attach peaks data
            peaks_df = st.session_state.get(f"peak_data_{idx}")
            peaks_list = []
            if peaks_df is not None:
                for _, row in peaks_df.iterrows():
                    try:
                        loc = float(row["Location (km)"])
                        elev_pk = float(row["Elevation (m)"])
                    except:
                        continue
                    if loc<0 or loc>stn['L']:
                        st.error(f"Station {idx}: Peak location must be between 0 and segment length.")
                        st.stop()
                    if elev_pk < stn['elev']:
                        st.error(f"Station {idx}: Peak elevation cannot be below station elevation.")
                        st.stop()
                    peaks_list.append({'loc': loc, 'elev': elev_pk})
            stn['peaks'] = peaks_list

        # Solve via Pyomo backend
        res = solve_pipeline(stations_data, term_data, FLOW, RateDRA, Price_HSD)

    # Display results
    total_cost = res.get('total_cost', 0.0)
    total_pumps = sum(int(res.get(f"num_pumps_{s['name'].lower().replace(' ','_')}",0)) for s in stations_data)
    speeds = [res.get(f"speed_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
    effs   = [res.get(f"efficiency_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
    avg_speed = np.mean([s for s in speeds if s]) if speeds else 0
    avg_eff = np.mean([e for e in effs if e]) if effs else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost (INR)", f"₹{total_cost:,.2f}")
    c2.metric("Total Pumps", total_pumps)
    c3.metric("Avg Speed (rpm)", f"{avg_speed:.1f}")
    c4.metric("Avg Efficiency (%)", f"{avg_eff:.1f}")

    # Summary table
    names = [s['name'] for s in stations_data] + [terminal_name]
    rows = ["Power+Fuel Cost", "DRA Cost", "No. Pumps", "Pump Speed (rpm)",
            "Pump Eff (%)", "Reynolds", "Head Loss (m)", "Vel (m/s)",
            "Residual Head (m)", "SDH (m)", "DRA (%)"]
    summary = {"Process": rows}
    for nm in names:
        key = nm.lower().replace(' ','_')
        summary[nm] = [
            res.get(f"power_cost_{key}",0.0),
            res.get(f"dra_cost_{key}",0.0),
            int(res.get(f"num_pumps_{key}",0)),
            res.get(f"speed_{key}",0.0),
            res.get(f"efficiency_{key}",0.0),
            res.get(f"reynolds_{key}",0.0),
            res.get(f"head_loss_{key}",0.0),
            res.get(f"velocity_{key}",0.0),
            res.get(f"residual_head_{key}",0.0),
            res.get(f"sdh_{key}",0.0),
            res.get(f"drag_reduction_{key}",0.0)
        ]
    df_sum = pd.DataFrame(summary)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves", "🔄 Pump-System"])
    with tab1:
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")

    with tab2:
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        df_cost = pd.DataFrame({
            "Station": [s['name'] for s in stations_data],
            "Power+Fuel": [res.get(f"power_cost_{s['name'].lower().replace(' ','_')}",0) for s in stations_data],
            "DRA":       [res.get(f"dra_cost_{s['name'].lower().replace(' ','_')}",0)    for s in stations_data]
        })
        fig_cost = px.bar(df_cost.melt(id_vars="Station", value_vars=["Power+Fuel","DRA"],
                                       var_name="Type", value_name="INR/day"),
                          x="Station", y="INR/day", color="Type",
                          title="Daily Cost by Station")
        fig_cost.update_layout(yaxis_title="Cost (INR)")
        st.plotly_chart(fig_cost, use_container_width=True)

    with tab3:
        perf_tab, head_tab = st.tabs(["Head Loss", "Velocity & Re"])
        with perf_tab:
            st.markdown("<div class='section-title'>Head Loss per Segment</div>", unsafe_allow_html=True)
            df_hloss = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Head Loss": [res.get(f"head_loss_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
            })
            fig_h = go.Figure(go.Bar(x=df_hloss["Station"], y=df_hloss["Head Loss"]))
            fig_h.update_layout(yaxis_title="Head Loss (m)")
            st.plotly_chart(fig_h, use_container_width=True)
        with head_tab:
            st.markdown("<div class='section-title'>Velocity & Reynolds</div>", unsafe_allow_html=True)
            df_vel = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Velocity (m/s)": [res.get(f"velocity_{s['name'].lower().replace(' ','_')}",0) for s in stations_data],
                "Reynolds": [res.get(f"reynolds_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
            })
            st.dataframe(df_vel.style.format({"Velocity (m/s)":"{:.2f}", "Reynolds":"{:.0f}"}))

    with tab4:
        st.markdown("<div class='section-title'>System Head Curves</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): 
                continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']; L_seg = stn['L']; elev_i = stn['elev']
            # Generate SDH vs flow for 0%,10%,...,max DR
            curves = []

            kv = stn.get('KV', 10.0)
            for dra in range(0, int(stn['max_dr'])+1, 10):
                v_vals = np.linspace(0, FLOW, 101)/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (KV*1e-6) if kv>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                curves.append(pd.DataFrame({"Flow": np.linspace(0, FLOW, 101), "SDH": SDH_vals, "DR": dra}))
            df_sys = pd.concat(curves)
            fig_sys = px.line(df_sys, x="Flow", y="SDH", color="DR", title=f"System Head ({stn['name']})")
            fig_sys.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig_sys, use_container_width=True)

    with tab5:
        st.markdown("<div class='section-title'>Pump vs System Interaction</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False):
                continue
            key = stn['name'].lower().replace(' ','_')
            # Plot pump curves at different RPMs against system curve (DRA steps)

            # station-specific viscosity
            kv = stn.get('KV', 10.0)
            flows = np.linspace(0, FLOW*1.5, 200)
            # System curve for 0% DRA (for simplicity)
            d_inner_i = stn['D'] - 2*stn['t']
            v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
            Re_vals = v_vals * d_inner_i / (KV*1e-6) if kv>0 else np.zeros_like(v_vals)
            f_vals = np.where(Re_vals>0,
                              0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
            DH = f_vals * ((stn['L']*1000.0)/d_inner_i) * (v_vals**2/(2*9.81))
            Hsys = stn['elev'] + DH
            # Pump head curves at some sample RPMs
            A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
            N_min = res.get(f"min_rpm_{key}", 0)
            N_max = res.get(f"dol_{key}", 0)
            fig_int = go.Figure()
            fig_int.add_trace(go.Scatter(x=flows, y=Hsys, mode='lines', name='System (0% DRA)'))
            for rpm in np.arange(N_min, N_max+1, 100):
                Hpump = (A*flows**2 + B*flows + C)*(rpm/N_max)**2
                fig_int.add_trace(go.Scatter(x=flows, y=Hpump, mode='lines', name=f'Pump {rpm} rpm'))
            fig_int.update_layout(title=f"Interaction ({stn['name']})", xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
            st.plotly_chart(fig_int, use_container_width=True)
