import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
import hashlib
import uuid
import json
from plotly.colors import qualitative

palette = [c for c in qualitative.Plotly if 'yellow' not in c.lower() and '#FFD700' not in c and '#ffeb3b' not in c.lower()]

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
        st.title("🔒 Pipeline Optimization Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and hash_pwd(password) == users[username]:
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown(
            """
            <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
            &copy; 2025 Pipeline Optimizer v1.1.1. Developed by Parichay Das. All rights reserved.
            </div>
            """,
            unsafe_allow_html=True
        )
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
.section-title {
  font-size:1.2rem; font-weight:600; margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>Mixed Integer Non-Linear Non-Convex Optimization of Pipeline Operations</h1>", unsafe_allow_html=True)

def get_input_fingerprint():
    return hashlib.md5(json.dumps({
        "stations": st.session_state.get('stations', []),
        "terminal": {
            "name": st.session_state.get('terminal_name', 'Terminal'),
            "elev": st.session_state.get('terminal_elev', 0.0),
            "min_residual": st.session_state.get('terminal_head', 50.0),
        },
        "FLOW": st.session_state.get('FLOW', 1000.0),
        "RateDRA": st.session_state.get('RateDRA', 500.0),
        "Price_HSD": st.session_state.get('Price_HSD', 70.0)
    }, sort_keys=True, default=str).encode()).hexdigest()

# ----- SIDEBAR -----
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
        st.session_state["FLOW"] = FLOW
        st.session_state["RateDRA"] = RateDRA
        st.session_state["Price_HSD"] = Price_HSD

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

if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
        'max_dr': 0.0
    }]

for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}", expanded=True):
        stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
        stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{idx}")
        stn['rho'] = st.number_input("Density (kg/m³)", value=stn.get('rho', 850.0), step=10.0, key=f"rho{idx}")
        stn['KV'] = st.number_input("Viscosity (cSt)", value=stn.get('KV', 10.0), step=0.1, key=f"kv{idx}")
        if idx == 1:
            stn['min_residual'] = st.number_input("Available Suction Head (m)", value=stn.get('min_residual',50.0), step=0.1, key=f"res{idx}")
        stn['D'] = st.number_input("Outer Diameter (m)", value=stn['D'], format="%.3f", step=0.001, key=f"D{idx}")
        stn['t'] = st.number_input("Wall Thickness (m)", value=stn['t'], format="%.4f", step=0.0001, key=f"t{idx}")
        stn['SMYS'] = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{idx}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=0.00001, key=f"rough{idx}")
        stn['L'] = st.number_input("Length to next station (km)", value=stn['L'], step=1.0, key=f"L{idx}")
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

# Terminal
st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value="Terminal", key="terminal_name")
terminal_elev = st.number_input("Elevation (m)", value=0.0, step=0.1, key="terminal_elev")
terminal_head = st.number_input("Minimum Residual Head (m)", value=50.0, step=1.0, key="terminal_head")

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD)

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
        per_station_KV = [stn['KV'] for stn in stations_data]
        per_station_rho = [stn['rho'] for stn in stations_data]
        res = solve_pipeline(stations_data, term_data, FLOW, per_station_KV, per_station_rho, RateDRA, Price_HSD)
        import copy
        st.session_state["last_res"] = copy.deepcopy(res)
        st.session_state["last_stations_data"] = copy.deepcopy(stations_data)
        st.session_state["last_term_data"] = copy.deepcopy(term_data)
        st.session_state["last_input_fingerprint"] = get_input_fingerprint()

# ---------- TABS -----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Summary", 
    "💰 Costs", 
    "⚙️ Performance", 
    "🌀 System Curves", 
    "🔄 Pump-System",
    "🧊 3D Analysis"      
])

# ---- Tab 1 ----
with tab1:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        terminal_name = st.session_state["last_term_data"]["name"]
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
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")

# ---- Tab 2 ----
with tab2:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        df_cost = pd.DataFrame({
            "Station": [s['name'] for s in stations_data],
            "Power+Fuel": [res.get(f"power_cost_{s['name'].lower().replace(' ','_')}",0) for s in stations_data],
            "DRA":       [res.get(f"dra_cost_{s['name'].lower().replace(' ','_')}",0)    for s in stations_data]
        })
        df_cost['Total'] = df_cost['Power+Fuel'] + df_cost['DRA']
        fig_pie = px.pie(df_cost, names='Station', values='Total', title="Station-wise Cost Breakdown (Pie)")
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.download_button("Download CSV", df_cost.to_csv(index=False).encode(), file_name="cost_breakdown.csv")

# ---- Tab 3 ----
with tab3:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        perf_tab, head_tab, char_tab, eff_tab, press_tab, power_tab = st.tabs([
            "Head Loss", "Velocity & Re", 
            "Pump Characteristic Curve", "Pump Efficiency Curve",
            "Pressure vs Pipeline Length", "Power vs Speed/Flow"
        ])
        # Head Loss
        with perf_tab:
            st.markdown("<div class='section-title'>Head Loss per Segment</div>", unsafe_allow_html=True)
            df_hloss = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Head Loss": [res.get(f"head_loss_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
            })
            fig_h = go.Figure(go.Bar(x=df_hloss["Station"], y=df_hloss["Head Loss"]))
            fig_h.update_layout(yaxis_title="Head Loss (m)")
            st.plotly_chart(fig_h, use_container_width=True, key=f"perf_headloss_{uuid.uuid4().hex[:6]}")
        # Velocity & Reynolds
        with head_tab:
            st.markdown("<div class='section-title'>Velocity & Reynolds</div>", unsafe_allow_html=True)
            df_vel = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Velocity (m/s)": [res.get(f"velocity_{s['name'].lower().replace(' ','_')}",0) for s in stations_data],
                "Reynolds": [res.get(f"reynolds_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
            })
            st.dataframe(df_vel.style.format({"Velocity (m/s)":"{:.2f}", "Reynolds":"{:.0f}"}))
        # Pump Characteristic Curve (at multiple RPMs)
        with char_tab:
            st.markdown("<div class='section-title'>Pump Characteristic Curves (Head vs Flow at various Speeds)</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ','_')
                flows = np.linspace(0, st.session_state.get("FLOW",1000.0)*1.5, 200)
                A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                fig = go.Figure()
                for rpm in range(N_min, N_max+1, 100):
                    H = (A*flows**2 + B*flows + C)*(rpm/N_max)**2
                    fig.add_trace(go.Scatter(x=flows, y=H, mode='lines', name=f"{rpm} rpm"))
                fig.update_layout(title=f"Head vs Flow: {stn['name']}", xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
                st.plotly_chart(fig, use_container_width=True, key=f"char_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")
        # Pump Efficiency Curve (at multiple RPMs)
        with eff_tab:
            st.markdown("<div class='section-title'>Pump Efficiency Curves (Eff vs Flow at various Speeds)</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ','_')
                Qe = st.session_state.get(f"eff_data_{i}")
                FLOW = st.session_state.get("FLOW",1000.0)
                if Qe is not None and len(Qe) > 0:
                    flow_min, flow_max = np.min(Qe['Flow (m³/hr)']), np.max(Qe['Flow (m³/hr)'])
                    flows = np.linspace(flow_min, flow_max, 200)
                else:
                    flows = np.linspace(0.01, FLOW*1.5, 200)
                P = stn.get('P',0); Q = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                fig = go.Figure()
                for rpm in range(N_min, N_max+1, 100):
                    Q_adj = flows * N_max/rpm
                    eff = (P*Q_adj**4 + Q*Q_adj**3 + R*Q_adj**2 + S*Q_adj + T)
                    fig.add_trace(go.Scatter(x=flows, y=eff, mode='lines', name=f"{rpm} rpm"))
                fig.update_layout(title=f"Efficiency vs Flow: {stn['name']}", xaxis_title="Flow (m³/hr)", yaxis_title="Efficiency (%)")
                st.plotly_chart(fig, use_container_width=True, key=f"eff_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")
        # Pressure vs Pipeline Length
        with press_tab:
            st.markdown("<div class='section-title'>Pressure vs Pipeline Length</div>", unsafe_allow_html=True)
            lengths = [0]
            names_p = []
            for stn in stations_data:
                l = stn.get('L', 0)
                lengths.append(lengths[-1] + l)
                names_p.append(stn['name'])
            terminal_name = st.session_state["last_term_data"]["name"]
            names_p.append(terminal_name)
            n_stn = len(stations_data)
            available_suction_head = res.get(f"residual_head_{stations_data[0]['name'].lower().replace(' ','_')}", 0.0)
            sdh = [res.get(f"sdh_{s['name'].lower().replace(' ','_')}", 0.0) for s in stations_data]
            rh = [res.get(f"residual_head_{s['name'].lower().replace(' ','_')}", 0.0) for s in stations_data]
            rh.append(res.get(f"residual_head_{terminal_name.lower().replace(' ','_')}", 0.0))
            x_pts = []
            y_pts = []
            x_pts.extend([lengths[0], lengths[0]])
            y_pts.extend([available_suction_head, sdh[0]])
            for i in range(n_stn - 1):
                x_pts.extend([lengths[i], lengths[i+1]])
                y_pts.extend([sdh[i], rh[i+1]])
                x_pts.extend([lengths[i+1], lengths[i+1]])
                y_pts.extend([rh[i+1], sdh[i+1]])
            x_pts.extend([lengths[-2], lengths[-1]])
            y_pts.extend([sdh[-1], rh[-1]])
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=x_pts, y=y_pts, mode='lines+markers',
                name="Pressure Profile", line=dict(width=3)
            ))
            for idx, name in enumerate(names_p):
                y_annot = rh[idx] if idx < len(rh) else rh[-1]
                fig_p.add_annotation(x=lengths[idx], y=y_annot, text=name, showarrow=True, yshift=12)
            fig_p.update_layout(
                title="Pressure vs Pipeline Length",
                xaxis_title="Cumulative Length (km)",
                yaxis_title="Pressure Head (mcl)",
                showlegend=False
            )
            st.plotly_chart(fig_p, use_container_width=True)
        # Power vs Speed, Power vs Flow
        with power_tab:
            st.markdown("<div class='section-title'>Power vs Speed & Power vs Flow</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ','_')
                A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
                P = stn.get('P',0); Qc = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                flow = st.session_state.get("FLOW",1000.0)
                speeds = np.arange(N_min, N_max+1, 100)
                power = []
                for rpm in speeds:
                    H = (A*flow**2 + B*flow + C)*(rpm/N_max)**2
                    eff = (P*flow**4 + Qc*flow**3 + R*flow**2 + S*flow + T)
                    eff = max(0.01, eff/100)
                    pwr = (stn['rho'] * flow * 9.81 * H)/(3600.0*eff*0.95)
                    power.append(pwr)
                fig_pwr = go.Figure()
                fig_pwr.add_trace(go.Scatter(x=speeds, y=power, mode='lines+markers', name="Power vs Speed"))
                fig_pwr.update_layout(title=f"Power vs Speed: {stn['name']}", xaxis_title="Speed (rpm)", yaxis_title="Power (kW)")
                st.plotly_chart(fig_pwr, use_container_width=True)
                flows = np.linspace(0.01, flow*1.5, 100)
                power2 = []
                for q in flows:
                    H = (A*q**2 + B*q + C)
                    eff = (P*q**4 + Qc*q**3 + R*q**2 + S*q + T)
                    eff = max(0.01, eff/100)
                    pwr = (stn['rho'] * q * 9.81 * H)/(3600.0*eff*0.95)
                    power2.append(pwr)
                fig_pwr2 = go.Figure()
                fig_pwr2.add_trace(go.Scatter(x=flows, y=power2, mode='lines+markers', name="Power vs Flow"))
                fig_pwr2.update_layout(title=f"Power vs Flow: {stn['name']}", xaxis_title="Flow (m³/hr)", yaxis_title="Power (kW)")
                st.plotly_chart(fig_pwr2, use_container_width=True)

# ---- Tab 4 ----
with tab4:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): 
                continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']; L_seg = stn['L']; elev_i = stn['elev']
            max_dr = int(stn.get('max_dr', 40))
            curves = []
            for dra in range(0, max_dr+1, 5):
                flows = np.linspace(0, st.session_state.get("FLOW",1000.0), 101)
                v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (stn['KV']*1e-6) if stn['KV']>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                curves.append(pd.DataFrame({"Flow": flows, "SDH": SDH_vals, "DRA": dra}))
            df_sys = pd.concat(curves)
            fig_sys = px.line(df_sys, x="Flow", y="SDH", color="DRA", title=f"System Head ({stn['name']}) at various % DRA")
            fig_sys.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig_sys, use_container_width=True, key=f"sys_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")

# ---- Tab 5 ----
with tab5:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        palette = [c for c in qualitative.Plotly if 'yellow' not in c.lower() and '#FFD700' not in c and '#ffeb3b' not in c.lower()]
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False):
                continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']
            max_dr = int(stn.get('max_dr', 40))
            N_min = int(res.get(f"min_rpm_{key}", 0))
            N_max = int(res.get(f"dol_{key}", 0))
            num_pumps = max(1, int(res.get(f"num_pumps_{key}", 1)))
            flows = np.linspace(0, st.session_state.get("FLOW",1000.0)*1.5, 200)
            fig_int = go.Figure()
            dra_list = list(range(0, max_dr+1, 5))
            n_curves = max(len(dra_list), num_pumps * len(range(N_min, N_max+1, 100)))
            colors = (palette * ((n_curves // len(palette)) + 1))[:n_curves]
            for idx_dra, dra in enumerate(dra_list):
                v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (stn['KV']*1e-6) if stn['KV']>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((stn['L']*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                Hsys = stn['elev'] + DH
                fig_int.add_trace(go.Scatter(
                    x=flows, y=Hsys, mode='lines',
                    name=f'System {dra}% DRA',
                    line=dict(color=colors[idx_dra], width=2)
                ))
            A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
            pump_curve_idx = 0
            for pumps_in_series in range(1, num_pumps+1):
                for rpm in range(N_min, N_max+1, 100):
                    Hpump = (A*flows**2 + B*flows + C)*(rpm/N_max)**2 * pumps_in_series
                    color = colors[pump_curve_idx % len(colors)]
                    fig_int.add_trace(
                        go.Scatter(
                            x=flows, y=Hpump, mode='lines',
                            name=f'Pump {pumps_in_series}x @ {rpm}rpm',
                            line=dict(color=color, width=2)
                        )
                    )
                    pump_curve_idx += 1
            fig_int.update_layout(
                title=f"Interaction ({stn['name']})",
                xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)",
                legend_title_text="Curve"
            )
            st.plotly_chart(fig_int, use_container_width=True, key=f"interaction_{i}_{key}_{uuid.uuid4().hex[:6]}")


# --------------- Tab 6: 3D Plots -----------------
with tab6:
    # -- Ensure optimization run exists --
    if "last_res" not in st.session_state:
        st.warning("Please run optimization first.")
        st.stop()
    last_res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]
    terminal_data = st.session_state["last_term_data"]
    FLOW = st.session_state.get("FLOW", 1000.0)
    
    # -- Find first pump station for plotting --
    pump_idx = next((i for i, s in enumerate(stations_data) if s.get('is_pump', False)), None)
    if pump_idx is None:
        st.warning("No pump station found.")
        st.stop()
    stn = stations_data[pump_idx]
    key = stn['name'].lower().replace(' ', '_')
    N_min = int(last_res.get(f"min_rpm_{key}", 1000))
    N_max = int(last_res.get(f"dol_{key}", 1500))
    NOP_opt = int(last_res.get(f"num_pumps_{key}", 1))
    DRA_opt = float(last_res.get(f"drag_reduction_{key}", 0))
    Speed_opt = float(last_res.get(f"speed_{key}", N_max))
    Eff_opt = float(last_res.get(f"efficiency_{key}", 0))
    Flow_opt = FLOW
    PowerCost_opt = last_res.get(f"power_cost_{key}", 0)
    DRACost_opt = last_res.get(f"dra_cost_{key}", 0)
    Head_opt = last_res.get(f"sdh_{key}", 0)
    ResidualHead_opt = last_res.get(f"residual_head_{key}", 0)
    TotalCost_opt = last_res.get('total_cost', 0)
    # Use ±20% delta around optimum
    delta_flow = Flow_opt * 0.2 if Flow_opt else 200
    delta_speed = Speed_opt * 0.2 if Speed_opt else 200
    delta_dra = max(5, DRA_opt * 0.3)
    delta_nop = max(1, NOP_opt)
    DOL = N_max

    # -------------- Plot Category Dropdowns --------------
    plot_categories = {
        "Cost Plots": ["Total Cost", "Power Cost", "DRA Cost"],
        "Head Plots": ["System Head", "Residual Head"],
        "Efficiency Plots": ["Pump Efficiency"],
        "Power Plots": ["Pump Power"],
        "Performance Plots": ["Velocity", "Reynolds"],
        "System Head Plots": ["System Head vs Flow & DRA"]
    }
    plot_category = st.selectbox("Select Plot Category", list(plot_categories.keys()), key="tab6_plotcat")
    subplot_options = plot_categories[plot_category]
    subplot = st.selectbox("Select Subplot", subplot_options, key="tab6_subplot")

    # -------------- 3D Grid Setup --------------
    # (x, y axes depend on plot type)
    n_pts = 21  # resolution for meshgrid
    flows = np.linspace(max(0.1, Flow_opt - delta_flow), Flow_opt + delta_flow, n_pts)
    speeds = np.linspace(max(500, Speed_opt - delta_speed), Speed_opt + delta_speed, n_pts)
    dras = np.linspace(max(0, DRA_opt - delta_dra), DRA_opt + delta_dra, n_pts)
    nops = np.arange(max(1, NOP_opt - 2), min(stn.get('max_pumps', NOP_opt+2), NOP_opt + 2)+1, 1)

    # Head/Efficiency curve coefficients
    A = last_res.get(f"coef_A_{key}", 0)
    B = last_res.get(f"coef_B_{key}", 0)
    Cc = last_res.get(f"coef_C_{key}", 0)
    P = stn.get('P', 0)
    Qc = stn.get('Q', 0)
    R = stn.get('R', 0)
    S = stn.get('S', 0)
    T = stn.get('T', 0)
    rho = stn.get('rho', 850)
    rate = stn.get('rate', 9)
    # Plot generator functions

    def get_power(flow, speed, nop, dra):
        # System head for each (ignores intermediate peaks for this visualization)
        D_base = DOL
        tdh = (A*flow**2 + B*flow + Cc)*(speed/D_base)**2 if speed else 0
        q_adj = flow * D_base / speed if speed else 0
        eff = (P*q_adj**4 + Qc*q_adj**3 + R*q_adj**2 + S*q_adj + T)
        eff_pct = max(0.01, eff/100)
        pwr = (rho * flow * 9.81 * tdh * nop)/(3600.0 * eff_pct * 0.95)
        return pwr, tdh, eff

    # ------ Plot logic grouped by category -------
    if plot_category == "Cost Plots":
        if subplot == "Total Cost":
            X, Y = np.meshgrid(flows, dras, indexing='ij')
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    flow = X[i, j]
                    dra = Y[i, j]
                    power, tdh, eff = get_power(flow, Speed_opt, NOP_opt, dra)
                    power_cost = power * 24 * rate
                    dra_cost = (dra / 4) * (flow * 1000 * 24 / 1e6) * st.session_state.get('RateDRA', 500.0)
                    Z[i, j] = power_cost + dra_cost
            # Optimum point
            opt_x = Flow_opt
            opt_y = DRA_opt
            opt_z = PowerCost_opt + DRACost_opt
            labelstr = f"**Optimum Total Cost:** {opt_z:,.2f} INR/day  \n"
            labelstr += f"Flow: {Flow_opt:.1f} m³/hr | DRA: {DRA_opt:.1f} %"
            plot_title = "Total Cost Surface (Flow vs DRA)"
            xlab = "Flow (m³/hr)"; ylab = "DRA (%)"; zlab = "Total Cost (INR/day)"
        elif subplot == "Power Cost":
            X, Y = np.meshgrid(flows, speeds, indexing='ij')
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    flow = X[i, j]
                    speed = Y[i, j]
                    power, tdh, eff = get_power(flow, speed, NOP_opt, DRA_opt)
                    power_cost = power * 24 * rate
                    Z[i, j] = power_cost
            opt_x = Flow_opt
            opt_y = Speed_opt
            opt_z = PowerCost_opt
            labelstr = f"**Optimum Power Cost:** {opt_z:,.2f} INR/day  \n"
            labelstr += f"Flow: {Flow_opt:.1f} m³/hr | Speed: {Speed_opt:.1f} rpm"
            plot_title = "Power Cost Surface (Flow vs Speed)"
            xlab = "Flow (m³/hr)"; ylab = "Speed (rpm)"; zlab = "Power Cost (INR/day)"
        else:  # DRA Cost
            X, Y = np.meshgrid(flows, dras, indexing='ij')
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    flow = X[i, j]
                    dra = Y[i, j]
                    dra_cost = (dra / 4) * (flow * 1000 * 24 / 1e6) * st.session_state.get('RateDRA', 500.0)
                    Z[i, j] = dra_cost
            opt_x = Flow_opt
            opt_y = DRA_opt
            opt_z = DRACost_opt
            labelstr = f"**Optimum DRA Cost:** {opt_z:,.2f} INR/day  \n"
            labelstr += f"Flow: {Flow_opt:.1f} m³/hr | DRA: {DRA_opt:.1f} %"
            plot_title = "DRA Cost Surface (Flow vs DRA)"
            xlab = "Flow (m³/hr)"; ylab = "DRA (%)"; zlab = "DRA Cost (INR/day)"

    elif plot_category == "Head Plots":
        if subplot == "System Head":
            X, Y = np.meshgrid(flows, speeds, indexing='ij')
            Z = (A*X**2 + B*X + Cc)*(Y/DOL)**2
            opt_x = Flow_opt
            opt_y = Speed_opt
            opt_z = Head_opt
            labelstr = f"**Optimum System Head:** {opt_z:.2f} m  \n"
            labelstr += f"Flow: {Flow_opt:.1f} m³/hr | Speed: {Speed_opt:.1f} rpm"
            plot_title = "System Head Surface (Flow vs Speed)"
            xlab = "Flow (m³/hr)"; ylab = "Speed (rpm)"; zlab = "System Head (m)"
        else:  # Residual Head
            X, Y = np.meshgrid(flows, speeds, indexing='ij')
            Z = (A*X**2 + B*X + Cc)*(Y/DOL)**2 - (X/200.0)
            opt_x = Flow_opt
            opt_y = Speed_opt
            opt_z = ResidualHead_opt
            labelstr = f"**Optimum Residual Head:** {opt_z:.2f} m  \n"
            labelstr += f"Flow: {Flow_opt:.1f} m³/hr | Speed: {Speed_opt:.1f} rpm"
            plot_title = "Residual Head Surface (Flow vs Speed)"
            xlab = "Flow (m³/hr)"; ylab = "Speed (rpm)"; zlab = "Residual Head (m)"

    elif plot_category == "Efficiency Plots":
        # Efficiency vs Flow & Speed
        X, Y = np.meshgrid(flows, speeds, indexing='ij')
        D_base = DOL
        Q_adj = X * D_base / Y
        Z = (P*Q_adj**4 + Qc*Q_adj**3 + R*Q_adj**2 + S*Q_adj + T)
        opt_x = Flow_opt
        opt_y = Speed_opt
        opt_z = Eff_opt
        labelstr = f"**Optimum Efficiency:** {opt_z:.2f} %  \n"
        labelstr += f"Flow: {Flow_opt:.1f} m³/hr | Speed: {Speed_opt:.1f} rpm"
        plot_title = "Pump Efficiency Surface (Flow vs Speed)"
        xlab = "Flow (m³/hr)"; ylab = "Speed (rpm)"; zlab = "Efficiency (%)"

    elif plot_category == "Power Plots":
        X, Y = np.meshgrid(flows, speeds, indexing='ij')
        power, tdh, eff = get_power(X, Y, NOP_opt, DRA_opt)
        Z = power
        opt_x = Flow_opt
        opt_y = Speed_opt
        opt_z = get_power(opt_x, opt_y, NOP_opt, DRA_opt)[0]
        labelstr = f"**Optimum Pump Power:** {opt_z:.2f} kW  \n"
        labelstr += f"Flow: {Flow_opt:.1f} m³/hr | Speed: {Speed_opt:.1f} rpm"
        plot_title = "Pump Power Surface (Flow vs Speed)"
        xlab = "Flow (m³/hr)"; ylab = "Speed (rpm)"; zlab = "Power (kW)"

    elif plot_category == "Performance Plots":
        if subplot == "Velocity":
            D = stn['D'] - 2*stn['t']
            X, Y = np.meshgrid(flows, speeds, indexing='ij')
            area = np.pi * (D**2) / 4
            Z = X/3600.0 / area if area else np.zeros_like(X)
            opt_x = Flow_opt
            opt_y = Speed_opt
            opt_z = Flow_opt/3600.0 / area if area else 0.0
            labelstr = f"**Optimum Velocity:** {opt_z:.3f} m/s  \n"
            labelstr += f"Flow: {Flow_opt:.1f} m³/hr"
            plot_title = "Velocity Surface (Flow vs Speed)"
            xlab = "Flow (m³/hr)"; ylab = "Speed (rpm)"; zlab = "Velocity (m/s)"
        else:  # Reynolds
            D = stn['D'] - 2*stn['t']
            KV = stn['KV']
            X, Y = np.meshgrid(flows, speeds, indexing='ij')
            area = np.pi * (D**2) / 4
            v = X/3600.0 / area if area else np.zeros_like(X)
            Z = v * D / (KV * 1e-6) if KV > 0 else np.zeros_like(X)
            opt_x = Flow_opt
            opt_y = Speed_opt
            opt_z = (Flow_opt/3600.0 / area) * D / (KV * 1e-6) if (KV > 0 and area) else 0.0
            labelstr = f"**Optimum Reynolds:** {opt_z:,.0f}  \n"
            labelstr += f"Flow: {Flow_opt:.1f} m³/hr"
            plot_title = "Reynolds Surface (Flow vs Speed)"
            xlab = "Flow (m³/hr)"; ylab = "Speed (rpm)"; zlab = "Reynolds Number"

    elif plot_category == "System Head Plots":
        # Show System Head vs Flow & DRA (fixed speed at optimum)
        X, Y = np.meshgrid(flows, dras, indexing='ij')
        # System head curve formula
        D = stn['D'] - 2*stn['t']
        rough = stn['rough']
        L = stn['L']
        KV = stn['KV']
        v_vals = X/3600.0 / (np.pi*(D**2)/4)
        Re_vals = v_vals * D / (KV*1e-6) if KV > 0 else np.zeros_like(v_vals)
        f_vals = np.where(Re_vals>0,
                          0.25/(np.log10(rough/D/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
        DH = f_vals * ((L*1000.0)/D) * (v_vals**2/(2*9.81)) * (1-Y/100.0)
        Z = stn['elev'] + DH
        opt_x = Flow_opt
        opt_y = DRA_opt
        # Compute optimum Z at (Flow_opt, DRA_opt)
        vopt = opt_x/3600.0 / (np.pi*(D**2)/4)
        Reopt = vopt * D / (KV*1e-6) if KV > 0 else 0
        fopt = 0.25/(np.log10(rough/D/3.7 + 5.74/(Reopt**0.9))**2) if Reopt>0 else 0
        DHopt = fopt * ((L*1000.0)/D) * (vopt**2/(2*9.81)) * (1-opt_y/100.0)
        opt_z = stn['elev'] + DHopt
        labelstr = f"**Optimum System Head:** {opt_z:.2f} m  \n"
        labelstr += f"Flow: {Flow_opt:.1f} m³/hr | DRA: {DRA_opt:.1f} %"
        plot_title = "System Head Surface (Flow vs DRA)"
        xlab = "Flow (m³/hr)"; ylab = "DRA (%)"; zlab = "System Head (m)"

    else:
        st.warning("Unknown plot type.")
        st.stop()

    # ------------- Display optimum value as highlight -------------------
    st.markdown(f"""
    <div style="background: #222; color: #ffd700; border-radius: 8px; padding: 0.8em 1em; margin-bottom: 10px; font-size: 1.15em;">
        {labelstr}
    </div>
    """, unsafe_allow_html=True)

    # ------------- Plotting 3D surface -----------------
    fig = go.Figure(data=[
        go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=True)
    ])
    # Optimum point marker
    fig.add_trace(go.Scatter3d(
        x=[opt_x], y=[opt_y], z=[opt_z],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=["Optimum"],
        textposition='top right',
        name='Optimum'
    ))
    # Axes labeling
    fig.update_layout(
        scene=dict(
            xaxis_title=f"X: {xlab}",
            yaxis_title=f"Y: {ylab}",
            zaxis_title=f"Z: {zlab}"
        ),
        title=plot_title,
        height=750,
        margin=dict(l=30, r=30, b=30, t=80)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
    &copy; 2025 Pipeline Optimizer v1.1.1. Developed by Parichay Das. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
