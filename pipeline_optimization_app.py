# pipeline_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
from io import BytesIO
import hashlib
import uuid
from plotly.colors import qualitative
# Remove yellow from palette as before
palette = [c for c in qualitative.Plotly if 'yellow' not in c.lower() and '#FFD700' not in c and '#ffeb3b' not in c.lower()]

st.set_page_config(page_title="Pipeline Optimization", layout="wide")

# ---- USER AUTH ----
def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

# Only this username and password are allowed
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
        st.stop()  # Prevent the rest of the app from loading

    # Add a logout button in sidebar
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

# Call this before everything else!
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

# Solver call
def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD)

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
    st.session_state.stations = []
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

# Terminal inputs
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
        # Attach pump curve data to stations
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

    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary", 
        "💰 Costs", 
        "⚙️ Performance", 
        "🌀 System Curves", 
        "🔄 Pump-System"
    ])
    # === Tab 1 ===
    with tab1:
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")
    # === Tab 2 ===
    with tab2:
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        df_cost = pd.DataFrame({
            "Station": [s['name'] for s in stations_data],
            "Power+Fuel": [res.get(f"power_cost_{s['name'].lower().replace(' ','_')}",0) for s in stations_data],
            "DRA":       [res.get(f"dra_cost_{s['name'].lower().replace(' ','_')}",0)    for s in stations_data]
        })
        df_cost['Total'] = df_cost['Power+Fuel'] + df_cost['DRA']
        # Only pie chart, no bar chart
        fig_pie = px.pie(df_cost, names='Station', values='Total', title="Station-wise Cost Breakdown (Pie)")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.download_button("Download CSV", df_cost.to_csv(index=False).encode(), file_name="cost_breakdown.csv")
    
    # === Tab 3 (Performance) ===
    with tab3:
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
                flows = np.linspace(0, FLOW*1.5, 200)
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
                    # Scale flow to RPM
                    Q_adj = flows * N_max/rpm
                    eff = (P*Q_adj**4 + Q*Q_adj**3 + R*Q_adj**2 + S*Q_adj + T)
                    fig.add_trace(go.Scatter(x=flows, y=eff, mode='lines', name=f"{rpm} rpm"))
                fig.update_layout(title=f"Efficiency vs Flow: {stn['name']}", xaxis_title="Flow (m³/hr)", yaxis_title="Efficiency (%)")
                st.plotly_chart(fig, use_container_width=True, key=f"eff_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")
        # Pressure vs Pipeline Length
        with press_tab:
            st.markdown("<div class='section-title'>Pressure vs Pipeline Length</div>", unsafe_allow_html=True)
            # Prepare cumulative lengths and heads
            lengths = [0]
            names_p = []
            for stn in stations_data:
                l = stn.get('L', 0)
                lengths.append(lengths[-1] + l)
                names_p.append(stn['name'])
            names_p.append(terminal_name)
            n_stn = len(stations_data)
            # Collect all needed values
            available_suction_head = res.get(f"residual_head_{stations_data[0]['name'].lower().replace(' ','_')}", 0.0)
            sdh = [res.get(f"sdh_{s['name'].lower().replace(' ','_')}", 0.0) for s in stations_data]
            rh = [res.get(f"residual_head_{s['name'].lower().replace(' ','_')}", 0.0) for s in stations_data]
            rh.append(res.get(f"residual_head_{terminal_name.lower().replace(' ','_')}", 0.0))
        
            # Build points for custom curve: list of (x, y)
            x_pts = []
            y_pts = []
            # 1st: vertical from Available Suction Head at Station-1 (at 0) to SDH of Station-1 (at 0)
            x_pts.extend([lengths[0], lengths[0]])
            y_pts.extend([available_suction_head, sdh[0]])
            # Now for each station (except last)
            for i in range(n_stn - 1):
                # Sloped line from SDH[i] (at lengths[i]) to RH[i+1] (at lengths[i+1])
                x_pts.extend([lengths[i], lengths[i+1]])
                y_pts.extend([sdh[i], rh[i+1]])
                # Vertical line from RH[i+1] (at lengths[i+1]) to SDH[i+1] (at lengths[i+1])
                x_pts.extend([lengths[i+1], lengths[i+1]])
                y_pts.extend([rh[i+1], sdh[i+1]])
            # Final: from last SDH to terminal RH (NO vertical at terminal, so just a sloped line)
            x_pts.extend([lengths[-2], lengths[-1]])
            y_pts.extend([sdh[-1], rh[-1]])
        
            # Plot
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=x_pts, y=y_pts, mode='lines+markers',
                name="Pressure Profile", line=dict(width=3)
            ))
        
            # Annotate station names at corresponding x positions
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
                # For Power vs Speed
                A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
                P = stn.get('P',0); Qc = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                eff_ref = max(1, np.max([res.get(f"efficiency_{key}",0), 1]))
                # Fixed flow at design
                flow = FLOW
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
                # Power vs Flow
                flows = np.linspace(0.01, FLOW*1.5, 100)
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

    # === Tab 4 (System Curves at various DRA) ===
    with tab4:
        st.markdown("<div class='section-title'>System Head Curves at different %DRA</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): 
                continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']; L_seg = stn['L']; elev_i = stn['elev']
            max_dr = int(stn.get('max_dr', 40))
            curves = []
            for dra in range(0, max_dr+1, 5):
                flows = np.linspace(0, FLOW, 101)
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
    
    # === Tab 5 (Pump-System Interaction, 3D Total Cost plot) ===
    with tab5:
        st.markdown("<div class='section-title'>Pump vs System Interaction</div>", unsafe_allow_html=True)
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
            flows = np.linspace(0, FLOW*1.5, 200)
            fig_int = go.Figure()
            dra_list = list(range(0, max_dr+1, 5))
            n_curves = max(len(dra_list), num_pumps * len(range(N_min, N_max+1, 100)))
            colors = (palette * ((n_curves // len(palette)) + 1))[:n_curves]
    
            # --- Plot System Curves ---
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
    
            # --- Plot Pump Curves using the SAME colors ---
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
    
            # Download PNG
            png_bytes = fig_int.to_image(format="png")
            st.download_button(f"Download {stn['name']} Interaction Chart (PNG)", png_bytes, file_name=f"interaction_{key}.png", mime="image/png")
    
            # ========== SHOW 3D PLOTS ONLY FOR STATION-1 ==========
            if i == 1:
                # ---- 3D Plot (Cost vs Pump Speed vs No. of Pumps, Efficiency as color) ----
                st.markdown("<div class='section-title'>3D Surface: Total Cost vs Pump Speed vs No. of Pumps (Efficiency as Color)</div>", unsafe_allow_html=True)
                N_min3d = int(res.get(f"min_rpm_{key}", 1000))
                N_max3d = int(res.get(f"dol_{key}", 1500))
                max_pumps_3d = int(stn.get('max_pumps', 4))
                rpm_range = np.arange(N_min3d, N_max3d+1, 20)
                num_pumps_list = list(range(1, max_pumps_3d+1))
                X, Y = np.meshgrid(num_pumps_list, rpm_range)
                Z = np.zeros_like(X, dtype=float)
                eff_vals = np.zeros_like(X, dtype=float)
                for i_p, n_pumps in enumerate(num_pumps_list):
                    for j_r, rpm in enumerate(rpm_range):
                        H = (A*FLOW**2 + B*FLOW + C)*(rpm/N_max3d)**2
                        P_ = stn.get('P',0); Qc_ = stn.get('Q',0); R_ = stn.get('R',0); S_ = stn.get('S',0); T_ = stn.get('T',0)
                        eff = (P_*FLOW**4 + Qc_*FLOW**3 + R_*FLOW**2 + S_*FLOW + T_)
                        eff = max(0.01, eff/100)
                        pwr = (stn['rho'] * FLOW * 9.81 * H * n_pumps)/(3600.0*eff*0.95)
                        power_cost = pwr*24*stn.get('rate', 0)
                        dra = 0
                        dra_cost = (dra/4)*(FLOW*1000.0*24.0/1e6)*RateDRA
                        Z[j_r, i_p] = power_cost + dra_cost
                        eff_vals[j_r, i_p] = eff*100  # As percent
                fig_surface = go.Figure(data=[
                    go.Surface(
                        x=X, y=Y, z=Z, surfacecolor=eff_vals,
                        colorbar=dict(title="Efficiency (%)"),
                        colorscale="Viridis"
                    )
                ])
                fig_surface.update_layout(
                    title=f"Total Cost vs Pump Speed vs No. of Pumps at {stn['name']}",
                    scene = dict(
                        xaxis_title='No. of Pumps',
                        yaxis_title='Pump Speed (rpm)',
                        zaxis_title='Total Cost (INR/day)'
                    ),
                    margin=dict(l=30, r=30, b=30, t=50)
                )
                st.plotly_chart(fig_surface, use_container_width=True, key=f"cost_surface_{i}_{key}_{uuid.uuid4().hex[:6]}")

            # --- NON-CONVEXITY PLOTS: Only show after optimization run ---
            if run:  # Or you can use: if 'res' in locals() and res is not None:
                st.markdown("---")
                st.markdown("## Non-Convexity Visualization (Parameter Sweep)")
        
                stations_data = st.session_state.stations
                term_data = {
                    "name": terminal_name,
                    "elev": terminal_elev,
                    "min_residual": terminal_head
                }
                per_station_KV = [stn['KV'] for stn in stations_data]
                per_station_rho = [stn['rho'] for stn in stations_data]
        
                # --------- 1. Pump Speed vs Total Cost -----------
                pump_idx = None
                for i, stn in enumerate(stations_data):
                    if stn.get("is_pump", False):
                        pump_idx = i
                        break
        
                if pump_idx is not None:
                    st.markdown(f"### Pump Speed vs Total Cost ({stations_data[pump_idx]['name']})")
                    min_rpm = int(stations_data[pump_idx]['MinRPM'])
                    max_rpm = int(stations_data[pump_idx]['DOL'])
                    speeds = np.linspace(min_rpm, max_rpm, 20)
                    total_costs = []
                    for rpm in speeds:
                        stn_copy = [dict(s) for s in stations_data]
                        stn_copy[pump_idx]['fixed_speed'] = rpm  # If your backend supports this!
                        res_sweep = solve_pipeline(
                            stn_copy, term_data, FLOW, per_station_KV, per_station_rho, RateDRA, Price_HSD
                        )
                        total_costs.append(res_sweep.get('total_cost', np.nan))
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=speeds, y=total_costs, mode="lines+markers",
                        marker=dict(color="#1f77b4"), name="Total Cost"
                    ))
                    fig1.update_layout(
                        title="Pump Speed vs Total Cost",
                        xaxis_title="Pump Speed (rpm)",
                        yaxis_title="Total Cost (INR/day)"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.warning("No pumping station found to plot Pump Speed vs Total Cost.")

            # --------- 2. DRA % vs Total Cost -----------
                st.markdown("### DRA (%) vs Total Cost (First Pump Station)")
                max_dra = 40
                dra_range = np.linspace(0, max_dra, 20)
                dra_costs = []
                for dra_pct in dra_range:
                    stn_copy = [dict(s) for s in stations_data]
                    for stn in stn_copy:
                        if stn.get("is_pump", False):
                            stn["max_dr"] = dra_pct
                    res_sweep = solve_pipeline(
                        stn_copy, term_data, FLOW, per_station_KV, per_station_rho, RateDRA, Price_HSD
                    )
                    dra_costs.append(res_sweep.get('total_cost', np.nan))
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=dra_range, y=dra_costs, mode="lines+markers",
                    marker=dict(color="#d62728"), name="Total Cost"
                ))
                fig2.update_layout(
                    title="DRA (%) vs Total Cost",
                    xaxis_title="DRA Injection (%)",
                    yaxis_title="Total Cost (INR/day)"
                )
                st.plotly_chart(fig2, use_container_width=True)
               
                # 3D Plot (Pump Efficiency vs Pump Speed vs DRA)
                st.markdown("<div class='section-title'>3D Surface: Pump Efficiency vs DRA% vs Pump Speed (Station-1 Only)</div>", unsafe_allow_html=True)
                N_min = int(res.get(f"min_rpm_{key}", 1000))
                N_max = int(res.get(f"dol_{key}", 1500))
                max_dr = int(stn.get('max_dr', 40))
                rpm_range = np.arange(N_min, N_max+1, 20)
                dra_range = np.arange(0, max_dr+1, 5)
                X, Y = np.meshgrid(dra_range, rpm_range)
                Z = np.zeros_like(X, dtype=float)
                P = stn.get('P',0); Qc = stn.get('Q',0); Rcoef = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                FLOW_ = FLOW
                N_base = int(res.get(f"dol_{key}", 1500))
                for i_r, rpm in enumerate(rpm_range):
                    for i_d, dra in enumerate(dra_range):
                        flow = FLOW_
                        Q_adj = flow * N_base/rpm  # Flow at this speed (if you scale)
                        eff = (P*Q_adj**4 + Qc*Q_adj**3 + Rcoef*Q_adj**2 + S*Q_adj + T)
                        Z[i_r, i_d] = eff
                fig_eff = go.Figure(data=[
                    go.Surface(
                        x=X, y=Y, z=Z,
                        colorbar=dict(title="Efficiency (%)"),
                        colorscale="Viridis"
                    )
                ])
                fig_eff.update_layout(
                    title=f"Pump Efficiency vs DRA% vs Pump Speed at {stn['name']}",
                    scene = dict(
                        xaxis_title='DRA (%)',
                        yaxis_title='Pump Speed (rpm)',
                        zaxis_title='Pump Efficiency (%)'
                    ),
                    margin=dict(l=30, r=30, b=30, t=50)
                )
                st.plotly_chart(fig_eff, use_container_width=True, key=f"eff_surface_{i}_{key}_{uuid.uuid4().hex[:6]}")
    
                # 3D Plot (Cost vs Pump Speed vs DRA)
                st.markdown(f"**3D Total Cost vs Pump Speed & DRA for {stn['name']}**")
                rpm_range = np.arange(N_min, N_max+1, 100)
                dra_range = np.arange(0, int(stn.get('max_dr', 40))+1, 5)
                X, Y = np.meshgrid(rpm_range, dra_range)
                Z = np.zeros_like(X, dtype=float)
                for i_r, rpm in enumerate(rpm_range):
                    for i_d, dra in enumerate(dra_range):
                        flow = FLOW_
                        H = (A*flow**2 + B*flow + C)*(rpm/N_max)**2
                        P = stn.get('P',0); Qc = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                        eff = (P*flow**4 + Qc*flow**3 + R*flow**2 + S*flow + T)
                        eff = max(0.01, eff/100)
                        pwr = (stn['rho'] * flow * 9.81 * H)/(3600.0*eff*0.95)
                        dra_cost = (dra/4)*(flow*1000.0*24.0/1e6)*RateDRA
                        power_cost = pwr*24*stn.get('rate', 0)
                        Z[i_d, i_r] = power_cost + dra_cost
                fig3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
                fig3d.update_layout(title="Total Cost vs Speed & DRA", scene=dict(
                    xaxis_title="Speed (rpm)", yaxis_title="DRA (%)", zaxis_title="Total Cost (INR/day)"))
                st.plotly_chart(fig3d, use_container_width=True, key=f"cost3d_{i}_{key}_{uuid.uuid4().hex[:6]}")

st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
    &copy; 2025 Developed by Parichay Das. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
