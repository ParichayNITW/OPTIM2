import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
import hashlib

st.set_page_config(page_title="Pipeline Optimization", layout="wide")

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

users = {
    "parichay_das": hash_pwd("heteroscedasticity")
}
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.markdown("""
        <style>
        .css-10trblm {text-align: left;}
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: left;'>🔒 Pipeline Optimization Login</h2>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and hash_pwd(password) == users[username]:
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.stop()
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
check_login()

if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

st.markdown("""
<style>
.justify-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.justify-title h1 {
    margin: 0;
    font-size: 2rem;
}
.section-title {
  font-size:1.2rem; font-weight:600; margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
<div class="justify-title">
  <h1>Mixed Integer Non-Linear Non-Convex Optimization of Pipeline Operations</h1>
</div>
""", unsafe_allow_html=True)

def solve_pipeline(stations, terminal, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, RateDRA, Price_HSD)

# Sidebar Inputs
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0, format="%.2f")
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)

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
            'max_dr': 0.0, 'rho': 850.0, 'KV': 10.0, 'FLOW': FLOW
        }
        st.session_state.stations.append(default)
    if rem_col.button("🗑️ Remove Station"):
        if st.session_state.get('stations'):
            st.session_state.stations.pop()

if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
        'max_dr': 0.0, 'rho': 850.0, 'KV': 10.0, 'FLOW': FLOW
    }]

for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}", expanded=True):
        stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
        stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{idx}")
        if idx == 1:
            stn['min_residual'] = st.number_input("Available suction head (m)", value=stn.get('min_residual',50.0), step=0.1, key=f"res{idx}")
        stn['D'] = st.number_input("Outer Diameter (m)", value=stn['D'], format="%.3f", step=0.001, key=f"D{idx}")
        stn['t'] = st.number_input("Wall Thickness (m)", value=stn['t'], format="%.4f", step=0.0001, key=f"t{idx}")
        stn['SMYS'] = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{idx}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=0.00001, key=f"rough{idx}")
        stn['L'] = st.number_input("Length to next station (km)", value=stn['L'], step=1.0, key=f"L{idx}")
        stn['rho'] = st.number_input("Density (kg/m³)", value=stn.get('rho', 850.0), step=1.0, key=f"rho{idx}")
        stn['KV'] = st.number_input("Viscosity (cSt)", value=stn.get('KV', 10.0), step=0.01, key=f"kv{idx}")
        stn['FLOW'] = FLOW
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
            st.markdown("**Enter Pump Performance Data:**")
            st.write("Flow vs Head data (m³/hr, m)")
            df_head = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
            df_head = st.data_editor(df_head, num_rows="dynamic", key=f"head{idx}")
            st.write("Flow vs Efficiency data (m³/hr, %)")
            df_eff = pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})
            df_eff = st.data_editor(df_eff, num_rows="dynamic", key=f"eff{idx}")
            st.session_state[f"head_data_{idx}"] = df_head
            st.session_state[f"eff_data_{idx}"] = df_eff

        st.markdown("**Intermediate Elevation Peaks (to next station):**")
        default_peak = pd.DataFrame({"Location (km)": [stn['L']/2.0], "Elevation (m)": [stn['elev']+100.0]})
        peak_df = st.data_editor(default_peak, num_rows="dynamic", key=f"peak{idx}")
        st.session_state[f"peak_data_{idx}"] = peak_df

st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value="Terminal")
terminal_elev = st.number_input("Elevation (m)", value=0.0, step=0.1)
terminal_head = st.number_input("Minimum Residual Head (m)", value=50.0, step=1.0)

run = st.button("🚀 Run Optimization")
if run:
    with st.spinner("Solving optimization..."):
        stations_data = st.session_state.stations
        term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}
        for idx, stn in enumerate(stations_data, start=1):
            if stn.get('is_pump', False):
                dfh = st.session_state.get(f"head_data_{idx}")
                dfe = st.session_state.get(f"eff_data_{idx}")
                if dfh is None or dfe is None or len(dfh)<3 or len(dfe)<5:
                    st.error(f"Station {idx}: At least 3 points for flow-head and 5 for flow-eff are required.")
                    st.stop()
                Qh = dfh.iloc[:,0].values; Hh = dfh.iloc[:,1].values
                coeff = np.polyfit(Qh, Hh, 2)
                stn['A'], stn['B'], stn['C'] = coeff[0], coeff[1], coeff[2]
                Qe = dfe.iloc[:,0].values; Ee = dfe.iloc[:,1].values
                coeff_e = np.polyfit(Qe, Ee, 4)
                stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = coeff_e
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
            stn['FLOW'] = FLOW  # pass flow for backend

        res = solve_pipeline(stations_data, term_data, RateDRA, Price_HSD)

    # Results Table with "Sl." and 2 decimal places
    # ----- Results Table with "Sl." and 2 decimal places formatting -----
    names = [s['name'] for s in stations_data] + [terminal_name]
    rows = ["Power+Fuel Cost", "DRA Cost", "No. Pumps", "Pump Speed (rpm)",
            "Pump Eff (%)", "Reynolds", "Head Loss (m)", "Vel (m/s)",
            "Residual Head (m)", "SDH (m)", "DRA (%)"]

    summary = {nm: [] for nm in names}
    
    for nm in names:
        key = nm.lower().replace(' ','_')
        vals = [
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
        vals_fmt = [
            f"{vals[0]:.2f}",
            f"{vals[1]:.2f}",
            vals[2],                # No. Pumps
            vals[3],                # Pump Speed
            f"{vals[4]:.2f}",
            f"{vals[5]:.2f}",
            f"{vals[6]:.2f}",
            f"{vals[7]:.2f}",
            f"{vals[8]:.2f}",
            f"{vals[9]:.2f}",
            f"{vals[10]:.2f}"
        ]
        summary[nm] = vals_fmt
    
    # Create DataFrame: Each row = metric, columns = stations
    df_sum = pd.DataFrame(summary, index=rows)
    df_sum.insert(0, "Sl.", range(1, len(rows)+1))
    df_sum = df_sum.reset_index().rename(columns={'index': 'Process'})
    df_sum = df_sum[["Sl.", "Process"] + names]
    st.session_state.df_sum = df_sum


st.download_button(
    "Download Output Results (CSV)", 
    df_sum.to_csv(index=False).encode(), 
    file_name="results.csv"
)
st.download_button(
    "Download Optimization Report (PDF)", 
    pdf_bytes,   # pdf_bytes is your PDF in bytes (see below for PDF code)
    file_name="Optimization_Report.pdf"
)
st.ca

    
# Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Summary", 
        "💰 Costs", 
        "⚙️ Performance", 
        "🌀 System Curves", 
        "🔄 Pump-System",
        "🎢 3D Objective Surface"
    ])
    with tab1:
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")

    with tab2:
        st.markdown("<div class='section-title'>Cost Breakdown (Station-wise)</div>", unsafe_allow_html=True)
        labels = [s['name'] for s in stations_data]
        values = [res.get(f"power_cost_{s['name'].lower().replace(' ','_')}",0) +
                  res.get(f"dra_cost_{s['name'].lower().replace(' ','_')}",0)
                  for s in stations_data]
        fig = px.pie(values=values, names=labels, title="Total Daily Cost Breakdown by Station")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-title'>Pressure vs Pipeline Length (Stepwise TDH and Pressure Loss)</div>", unsafe_allow_html=True)
        # Construct stepwise profile as per your specs
        x_vals = []
        y_vals = []
        chainage = [0]
        station_labels = [stations_data[0]['name']]
        cum_length = 0
        y = res.get(f"residual_head_{stations_data[0]['name'].lower().replace(' ','_')}",0.0)
        y_vals.append(y)
        x_vals.append(cum_length)
        for i, stn in enumerate(stations_data):
            rh = res.get(f"residual_head_{stn['name'].lower().replace(' ','_')}",0.0)
            sdh = res.get(f"sdh_{stn['name'].lower().replace(' ','_')}",0.0)
            l = stn.get('L', 0)
            # vertical line (TDH)
            x_vals.append(cum_length)
            y_vals.append(sdh)
            # sloped line (pressure loss)
            cum_length += l
            next_rh = res.get(f"residual_head_{stations_data[i+1]['name'].lower().replace(' ','_')}",0.0) if i+1 < len(stations_data) else res.get(f"residual_head_{terminal_name.lower().replace(' ','_')}",0.0)
            x_vals.append(cum_length)
            y_vals.append(next_rh)
            chainage.append(cum_length)
            if i+1 < len(stations_data):
                station_labels.append(stations_data[i+1]['name'])
            else:
                station_labels.append(terminal_name)
        # For terminal
        rh_terminal = res.get(f"residual_head_{terminal_name.lower().replace(' ','_')}",0.0)
        x_vals.append(cum_length)
        y_vals.append(rh_terminal)
        fig = go.Figure()
        # Now plot stepwise: vertical in red, sloped in blue
        for i in range(0, len(x_vals)-2, 2):
            fig.add_trace(go.Scatter(
                x=[x_vals[i], x_vals[i+1]], y=[y_vals[i], y_vals[i+1]],
                mode='lines+markers', line=dict(color='red', width=3), name='TDH of Pump' if i==0 else None, showlegend=(i==0)))
            fig.add_trace(go.Scatter(
                x=[x_vals[i+1], x_vals[i+2]], y=[y_vals[i+1], y_vals[i+2]],
                mode='lines+markers', line=dict(color='blue', width=3), name='Pressure Loss' if i==0 else None, showlegend=(i==0)))
        fig.update_layout(
            title="Pressure vs Pipeline Length",
            xaxis_title="Chainage (km)",
            yaxis_title="Pressure (mcl)",
            xaxis=dict(tickvals=chainage, ticktext=station_labels)
        )
        fig.add_annotation(text="TDH = Vertical Line, Pressure Loss = Sloped Line", xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("<div class='section-title'>System Head Curves at different %DRA</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): 
                continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']; L_seg = stn['L']; elev_i = stn['elev']
            KV_i = stn.get('KV', 10.0)
            max_dr = int(stn.get('max_dr', 40))
            curves = []
            for dra in range(0, max_dr+1, 5):
                flows = np.linspace(0, FLOW, 101)
                v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (KV_i*1e-6) if KV_i>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                curves.append(pd.DataFrame({"Flow": flows, "SDH": SDH_vals, "DRA": dra}))
            df_sys = pd.concat(curves)
            fig_sys = px.line(df_sys, x="Flow", y="SDH", color="DRA", title=f"System Head ({stn['name']}) at various % DRA")
            fig_sys.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig_sys, use_container_width=True)

    with tab5:
        st.markdown("<div class='section-title'>Pump vs System Interaction</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False):
                continue
            key = stn['name'].lower().replace(' ','_')
            flows = np.linspace(0, FLOW*1.5, 200)
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']
            KV_i = stn.get('KV', 10.0)
            # System curve for 0% DRA
            v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
            Re_vals = v_vals * d_inner_i / (KV_i*1e-6) if KV_i>0 else np.zeros_like(v_vals)
            f_vals = np.where(Re_vals>0,
                              0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
            DH = f_vals * ((stn['L']*1000.0)/d_inner_i) * (v_vals**2/(2*9.81))
            Hsys = stn['elev'] + DH
            A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
            N_min = int(res.get(f"min_rpm_{key}", 0))
            N_max = int(res.get(f"dol_{key}", 0))
            fig_int = go.Figure()
            fig_int.add_trace(go.Scatter(x=flows, y=Hsys, mode='lines', name='System (0% DRA)'))
            for n_pump in range(1, stn['max_pumps']+1):
                for rpm in np.arange(N_min, N_max+1, 100):
                    Hpump = n_pump*(A*flows**2 + B*flows + C)*(rpm/N_max)**2
                    fig_int.add_trace(go.Scatter(x=flows, y=Hpump, mode='lines', name=f'{n_pump} Pump(s) {rpm} rpm'))
            fig_int.update_layout(title=f"Interaction ({stn['name']})", xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
            st.plotly_chart(fig_int, use_container_width=True)

import numpy as np
import plotly.graph_objects as go

with tab6:
    st.markdown("## 🎢 3D Objective Function Surface (Non-Convexity Visualization)")
    stations_data = st.session_state.stations
    terminal_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}

    # Find first pump station
    pump_indices = [i for i, s in enumerate(stations_data) if s.get('is_pump', False)]
    if pump_indices:
        pump_idx = pump_indices[0]
        st.info(f"Visualizing at: {stations_data[pump_idx]['name']}")
    else:
        st.warning("No pump station found, using Station 1 for demo.")
        pump_idx = 0

    # Choose range for Pump Speed (DOL) and DRA (%) - centered around user value
    default_speed = stations_data[pump_idx].get('DOL', 1500)
    default_dr = stations_data[pump_idx].get('max_dr', 20)
    speed_min, speed_max = int(default_speed * 0.7), int(default_speed * 1.3)
    dra_min, dra_max = max(0, int(default_dr - 15)), min(100, int(default_dr + 15))

    speed_range = st.slider("Pump Speed (rpm)", min_value=500, max_value=4000, value=(speed_min, speed_max), step=50)
    dra_range = st.slider("DRA (%)", min_value=0, max_value=100, value=(dra_min, dra_max), step=1)

    pump_speeds = np.linspace(speed_range[0], speed_range[1], 28)
    dra_percents = np.linspace(dra_range[0], dra_range[1], 28)

    # Calculate surface
    from pipeline_model import evaluate_objective_for_grid
    with st.spinner("Evaluating objective surface (this may take up to a minute)..."):
        Z = evaluate_objective_for_grid(
            stations_data, terminal_data, RateDRA, Price_HSD,
            var1_name="DOL", var2_name="max_dr",
            var1_vals=pump_speeds, var2_vals=dra_percents
        )
    X, Y = np.meshgrid(pump_speeds, dra_percents, indexing='ij')

    fig = go.Figure(
        data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=True, opacity=0.95)],
    )
    fig.update_layout(
        title=f"3D Surface: Station Cost vs Pump Speed and DRA (%) at {stations_data[pump_idx]['name']}",
        scene=dict(
            xaxis_title='Pump Speed (rpm)',
            yaxis_title='DRA (%)',
            zaxis_title='Station Cost (INR/day)',
        ),
        autosize=True,
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Notice the valleys and peaks: this surface reveals the non-convex (wavy) nature of the optimization function."
    )


# End of file


st.markdown(
    """
    <hr style="margin-top:2em;">
    <div style="text-align:center; color:gray; font-size: 0.95em;">
    &copy; 2025 (R) Parichay Das. All rights reserved.<br>
    This software and its outputs are protected by copyright.<br>
    No part may be reproduced, distributed, or transmitted without prior written permission.
    </div>
    """,
    unsafe_allow_html=True
)
