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
def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)

# Sidebar inputs
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        KV        = st.number_input("Viscosity (cSt)", value=10.0, step=0.1)
        rho       = st.number_input("Density (kg/m³)", value=850.0, step=10.0)
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
terminal_head = st.number_input("Required Residual Head (m)", value=50.0, step=1.0)

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

        res = solve_pipeline(stations_data, term_data, FLOW, KV, rho, RateDRA, Price_HSD)

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
        fig_cost = px.bar(df_cost.melt(id_vars="Station", value_vars=["Power+Fuel","DRA"],
                                       var_name="Type", value_name="INR/day"),
                          x="Station", y="INR/day", color="Type",
                          title="Daily Cost by Station")
        fig_cost.update_layout(yaxis_title="Cost (INR)")
        st.plotly_chart(fig_cost, use_container_width=True)
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
            st.plotly_chart(fig_h, use_container_width=True)
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
                st.plotly_chart(fig, use_container_width=True)
        # Pump Efficiency Curve (at multiple RPMs)
        with eff_tab:
            st.markdown("<div class='section-title'>Pump Efficiency Curves (Eff vs Flow at various Speeds)</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].lower().replace(' ','_')
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
                st.plotly_chart(fig, use_container_width=True)
        # Pressure vs Pipeline Length
        with press_tab:
            st.markdown("<div class='section-title'>Pressure vs Pipeline Length</div>", unsafe_allow_html=True)
            # Prepare cumulative lengths and pressure (SDH, RH)
            lengths = [0]
            sdh = []
            rh = []
            names_p = []
            for i, stn in enumerate(stations_data, start=1):
                names_p.append(stn['name'])
                l = stn.get('L', 0)
                lengths.append(lengths[-1] + l)
                key = stn['name'].lower().replace(' ','_')
                sdh.append(res.get(f"sdh_{key}",0.0))
                rh.append(res.get(f"residual_head_{key}",0.0))
            # Terminal
            names_p.append(terminal_name)
            sdh.append(0.0)  # Or None, as SDH at terminal is not used
            rh.append(res.get(f"residual_head_{terminal_name.lower().replace(' ','_')}",0.0))
            # Plot
            x_vals = np.array(lengths)
            fig_p = go.Figure()
            # SDH and RH at stations
            fig_p.add_trace(go.Scatter(x=x_vals[:-1], y=sdh, mode='markers+lines+text', name="SDH", text=names_p, textposition='top center'))
            fig_p.add_trace(go.Scatter(x=x_vals, y=rh, mode='markers+lines+text', name="RH", text=names_p, textposition='bottom center'))
            # Show vertical jump for each pump station
            for i in range(1, len(x_vals)-1):
                key = stations_data[i-1]['name'].lower().replace(' ','_')
                if stations_data[i-1].get('is_pump', False):
                    # Vertical line at pump station
                    fig_p.add_trace(go.Scatter(
                        x=[x_vals[i], x_vals[i]],
                        y=[rh[i], sdh[i]],
                        mode='lines',
                        line=dict(dash='dot', color='red'),
                        name=f"Pump Jump @ {names_p[i]}"
                    ))
            fig_p.update_layout(title="Pressure vs Pipeline Length", xaxis_title="Cumulative Length (km)", yaxis_title="Pressure (mcl)")
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
                    pwr = (rho * flow * 9.81 * H)/(3600.0*eff*0.95)
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
                    pwr = (rho * q * 9.81 * H)/(3600.0*eff*0.95)
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
                Re_vals = v_vals * d_inner_i / (KV*1e-6) if KV>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                curves.append(pd.DataFrame({"Flow": flows, "SDH": SDH_vals, "DRA": dra}))
            df_sys = pd.concat(curves)
            fig_sys = px.line(df_sys, x="Flow", y="SDH", color="DRA", title=f"System Head ({stn['name']}) at various % DRA")
            fig_sys.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig_sys, use_container_width=True)
    # === Tab 5 (Pump-System Interaction, 3D Total Cost plot) ===
    with tab5:
        st.markdown("<div class='section-title'>Pump vs System Interaction & 3D Cost Analysis</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False):
                continue
            key = stn['name'].strip().lower().replace(' ','_')

            # Get optimized values from results
            opt_speed = res.get(f"speed_{key}", 0)
            opt_nop = res.get(f"num_pumps_{key}", 0)
            opt_dra = res.get(f"drag_reduction_{key}", 0)
            opt_cost = res.get(f"power_cost_{key}", 0) + res.get(f"dra_cost_{key}", 0)

            # Grid setup (coarse for speed, otherwise very slow)
            N_min = int(res.get(f"min_rpm_{key}", 0))
            N_max = int(res.get(f"dol_{key}", 0))
            max_nop = int(stn.get('max_pumps', 2))
            dra_slices = list(range(0, int(stn.get('max_dr', 40))+1, 10)) if stn.get('max_dr', 0) >= 10 else [0, int(stn.get('max_dr', 0))]

            # Needed for backend-accurate cost calculation
            FLOW = FLOW
            rho = stn.get('rho', 850.0)
            RateDRA = RateDRA
            Price_HSD = Price_HSD
            P = stn.get('P', 0); Q = stn.get('Q', 0); R = stn.get('R', 0); S = stn.get('S', 0); T = stn.get('T', 0)
            A = res.get(f"coef_A_{key}", 0); B = res.get(f"coef_B_{key}", 0); C = res.get(f"coef_C_{key}", 0)
            N_ref = int(res.get(f"dol_{key}", 0))

            rpm_range = np.arange(N_min, N_max+1, 100)
            nop_range = np.arange(1, max_nop+1, 1)

            surfaces = []

        for dra in dra_slices:
            Z = np.zeros((len(nop_range), len(rpm_range)))
            for ix, nop in enumerate(nop_range):
                for iy, rpm in enumerate(rpm_range):
                    # Get per-point cost using the backend formulas (same as in your model)
                    flow = FLOW
                    # Calculate head at this speed
                    H = (A*flow**2 + B*flow + C)*(rpm/N_ref)**2 if N_ref>0 else 0
                    eff = (P*flow**4 + Q*flow**3 + R*flow**2 + S*flow + T)
                    eff = max(0.01, eff/100)
                    # Power (kW)
                    if nop>0 and rpm>0 and eff>0:
                        pwr = (rho * flow * 9.81 * H * nop)/(3600.0*1000.0*eff*0.95)
                    else:
                        pwr = 0
                    # DRA cost
                    dra_cost = (dra/4)*(flow*1000.0*24.0/1e6)*RateDRA
                    # Power cost (assume grid for now; you can extend for diesel)
                    power_cost = pwr*24*stn.get('rate', 0)
                    Z[ix,iy] = power_cost + dra_cost
            # Add surface
            surfaces.append(go.Surface(
                x=rpm_range, y=nop_range, z=Z,
                name=f"DRA {dra}%",
                showscale=False,
                opacity=0.7,
                hovertemplate="Speed: %{x} rpm<br>NoP: %{y}<br>Cost: %{z:.0f}<br>DRA: "+str(dra)+"%"
            ))

        # Marker for optimized point
        # Find which DRA slice matches the optimizer (or nearest if not on grid)
        nearest_dra = min(dra_slices, key=lambda x: abs(x-opt_dra))
        marker_color = "red"

        marker_trace = go.Scatter3d(
            x=[opt_speed],
            y=[opt_nop],
            z=[opt_cost],
            mode='markers+text',
            marker=dict(size=10, color=marker_color, symbol='diamond'),
            name=f"Optimized Solution (Cost: {opt_cost:,.0f})",
            text=[f"Optimized<br>Cost: ₹{opt_cost:,.0f}"],
            textposition='top center'
        )

        fig3d = go.Figure(data=surfaces + [marker_trace])
        fig3d.update_layout(
            title=f"Total Cost vs Speed vs No. of Pumps ({stn['name']}) for DRA%",
            scene=dict(
                xaxis_title="Speed (rpm)",
                yaxis_title="No. of Pumps",
                zaxis_title="Total Cost (INR/day)"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend_title="DRA% Slices"
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.info(f"**Optimized cost for {stn['name']} (marked in red): ₹{opt_cost:,.2f} per day**\n"
                f"Speed: {opt_speed:.0f} rpm, No. of Pumps: {opt_nop}, DRA: {opt_dra:.1f}%")

