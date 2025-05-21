import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import pi
from io import BytesIO
import hashlib

st.set_page_config(page_title="Pipeline Optimization", layout="wide")

# ---- USER AUTH ----
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

def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

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
            'KV': 10.0, 'rho': 850.0,  # <-- add defaults
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
        'KV': 10.0, 'rho': 850.0,
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
        stn['KV'] = st.number_input("Viscosity (cSt)", value=stn.get('KV', 10.0), step=0.1, key=f"KV{idx}")
        stn['rho'] = st.number_input("Density (kg/m³)", value=stn.get('rho', 850.0), step=10.0, key=f"rho{idx}")
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Summary", 
    "💰 Costs", 
    "⚙️ Performance", 
    "🌀 System Curves", 
    "🔄 Pump-System"
])

if st.button("🚀 Run Optimization"):
    with st.spinner("Solving optimization..."):
        stations_data = st.session_state.stations
        term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}
        for idx, stn in enumerate(stations_data, start=1):
            if stn.get('is_pump', False):
                dfh = st.session_state.get(f"head_data_{idx}")
                dfe = st.session_state.get(f"eff_data_{idx}")
                if dfh is None or dfe is None or len(dfh) < 3 or len(dfe) < 5:
                    st.error(f"Station {idx}: At least 3 points for flow-head and 5 for flow-eff are required.")
                    st.stop()
                Qh = dfh.iloc[:, 0].values; Hh = dfh.iloc[:, 1].values
                coeff = np.polyfit(Qh, Hh, 2)
                stn['A'], stn['B'], stn['C'] = coeff[0], coeff[1], coeff[2]
                Qe = dfe.iloc[:, 0].values; Ee = dfe.iloc[:, 1].values
                coeff_e = np.polyfit(Qe, Ee, 4)
                stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = coeff_e
                stn['flow_head_range'] = (Qh.min(), Qh.max())
                stn['flow_eff_range'] = (Qe.min(), Qe.max())
            peaks_df = st.session_state.get(f"peak_data_{idx}")
            peaks_list = []
            if peaks_df is not None:
                for _, row in peaks_df.iterrows():
                    try:
                        loc = float(row["Location (km)"])
                        elev_pk = float(row["Elevation (m)"])
                    except:
                        continue
                    if loc < 0 or loc > stn['L']:
                        st.error(f"Station {idx}: Peak location must be between 0 and segment length.")
                        st.stop()
                    if elev_pk < stn['elev']:
                        st.error(f"Station {idx}: Peak elevation cannot be below station elevation.")
                        st.stop()
                    peaks_list.append({'loc': loc, 'elev': elev_pk})
            stn['peaks'] = peaks_list

        res = solve_pipeline(stations_data, term_data, FLOW, RateDRA, Price_HSD)
    
    # === Tab 1: Summary ===
    with tab1:
        st.markdown("<div class='section-title'>Summary Table (Key Results)</div>", unsafe_allow_html=True)
        summary_rows = []
        for i, stn in enumerate(stations_data, start=1):
            key = stn['name'].strip().lower().replace(' ','_')
            summary_rows.append({
                "Station": stn['name'],
                "Type": "Pump" if stn.get('is_pump', False) else "Intermediate",
                "Optimized Pumps": res.get(f"num_pumps_{key}", 0),
                "Optimized Speed (rpm)": res.get(f"speed_{key}", 0),
                "Drag Reduction (%)": res.get(f"drag_reduction_{key}", 0),
                "Eff. (%)": round(res.get(f"efficiency_{key}", 0),2),
                "Residual Head (m)": round(res.get(f"residual_head_{key}", 0),2),
                "Head Loss (m)": round(res.get(f"head_loss_{key}", 0),2),
                "Power Cost (₹/day)": round(res.get(f"power_cost_{key}",0),2),
                "DRA Cost (₹/day)": round(res.get(f"dra_cost_{key}",0),2)
            })
        # Terminal
        key_t = terminal_name.strip().lower().replace(' ','_')
        summary_rows.append({
            "Station": terminal_name,
            "Type": "Terminal",
            "Optimized Pumps": 0,
            "Optimized Speed (rpm)": 0,
            "Drag Reduction (%)": 0,
            "Eff. (%)": 0,
            "Residual Head (m)": round(res.get(f"residual_head_{key_t}",0),2),
            "Head Loss (m)": 0,
            "Power Cost (₹/day)": 0,
            "DRA Cost (₹/day)": 0
        })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)
        st.success(f"**Total Operating Cost: ₹{res.get('total_cost',0):,.2f} / day**")

    # === Tab 2: Detailed Costs by Type ===
    with tab2:
        st.markdown("<div class='section-title'>Cost Breakdown by Station</div>", unsafe_allow_html=True)
        cost_rows = []
        total_power = 0; total_dra = 0
        for i, stn in enumerate(stations_data, start=1):
            key = stn['name'].strip().lower().replace(' ','_')
            pcost = res.get(f"power_cost_{key}",0)
            dracost = res.get(f"dra_cost_{key}",0)
            total_power += pcost
            total_dra += dracost
            cost_rows.append({
                "Station": stn['name'],
                "Power/Fuel Cost (₹/day)": round(pcost,2),
                "DRA Cost (₹/day)": round(dracost,2)
            })
        cost_rows.append({
            "Station": "TOTAL",
            "Power/Fuel Cost (₹/day)": round(total_power,2),
            "DRA Cost (₹/day)": round(total_dra,2)
        })
        cost_df = pd.DataFrame(cost_rows)
        st.dataframe(cost_df, use_container_width=True)

        # Pie chart of total cost breakdown
        pie_labels = ['Power/Fuel Cost', 'DRA Cost']
        pie_vals = [total_power, total_dra]
        fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_vals, hole=.5)])
        fig_pie.update_layout(title="Cost Breakdown (All Stations)")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # === Tab 3 (Performance) ===
    with tab3:
        perf_tab, head_tab, char_tab, eff_tab, press_tab, power_tab = st.tabs([
            "Head Loss", "Velocity & Re", 
            "Pump Characteristic Curve", "Pump Efficiency Curve",
            "Pressure drop vs Pipeline Length", "Power vs Speed/Flow"
        ])
        # Head Loss
        with perf_tab:
            st.markdown("<div class='section-title'>Head Loss per Segment</div>", unsafe_allow_html=True)
            df_hloss = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Head Loss": [res.get(f"head_loss_{s['name'].strip().lower().replace(' ','_')}",0) for s in stations_data]
            })
            fig_h = go.Figure(go.Bar(x=df_hloss["Station"], y=df_hloss["Head Loss"]))
            fig_h.update_layout(yaxis_title="Head Loss (m)")
            st.plotly_chart(fig_h, use_container_width=True)

        # Velocity & Reynolds
        with head_tab:
            st.markdown("<div class='section-title'>Velocity & Reynolds</div>", unsafe_allow_html=True)
            df_vel = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Velocity (m/s)": [res.get(f"velocity_{s['name'].strip().lower().replace(' ','_')}",0) for s in stations_data],
                "Reynolds": [res.get(f"reynolds_{s['name'].strip().lower().replace(' ','_')}",0) for s in stations_data]
            })
            st.dataframe(df_vel.style.format({"Velocity (m/s)":"{:.2f}", "Reynolds":"{:.0f}"}))

        # Pump Characteristic Curve (at multiple RPMs)
        with char_tab:
            st.markdown("<div class='section-title'>Pump Characteristic Curves (Head vs Flow at various Speeds)</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].strip().lower().replace(' ','_')
                flows_user = st.session_state[f"head_data_{i}"].iloc[:,0].values
                flows = np.linspace(flows_user.min(), flows_user.max(), 200)
                A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                fig = go.Figure()
                for rpm in range(N_min, N_max+1, 100):
                    H = (A*flows**2 + B*flows + C)*(rpm/N_max)**2
                    fig.add_trace(go.Scatter(x=flows, y=H, mode='lines', name=f"{rpm} rpm"))
                fig.update_layout(title=f"Head vs Flow: {stn['name']}", xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
                st.plotly_chart(fig, use_container_width=True)
        # Pump Efficiency Curve (at multiple RPMs, no extrapolation)
        with eff_tab:
            st.markdown("<div class='section-title'>Pump Efficiency Curves (Eff vs Flow at various Speeds, User-Range Only)</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].strip().lower().replace(' ','_')
                flows_user = st.session_state[f"eff_data_{i}"].iloc[:,0].values
                flows = np.linspace(flows_user.min(), flows_user.max(), 200)
                P = stn.get('P',0); Q = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                fig = go.Figure()
                for rpm in range(N_min, N_max+1, 100):
                    Q_adj = flows * N_max/rpm
                    eff = (P*Q_adj**4 + Q*Q_adj**3 + R*Q_adj**2 + S*Q_adj + T)
                    fig.add_trace(go.Scatter(x=flows, y=eff, mode='lines', name=f"{rpm} rpm"))
                fig.update_layout(title=f"Efficiency vs Flow: {stn['name']}", xaxis_title="Flow (m³/hr)", yaxis_title="Efficiency (%)")
                st.plotly_chart(fig, use_container_width=True)

        # Pressure drop vs Pipeline Length
        with press_tab:
            st.markdown("<div class='section-title'>Pressure drop vs Pipeline Length</div>", unsafe_allow_html=True)
            lengths = [0]
            sdh = []
            rh = []
            names_p = []
            for i, stn in enumerate(stations_data, start=1):
                names_p.append(stn['name'])
                l = stn.get('L', 0)
                lengths.append(lengths[-1] + l)
                key = stn['name'].strip().lower().replace(' ','_')
                sdh.append(res.get(f"sdh_{key}",0.0))
                rh.append(res.get(f"residual_head_{key}",0.0))
            # Terminal
            names_p.append(terminal_name)
            sdh.append(0.0)
            rh.append(res.get(f"residual_head_{terminal_name.strip().lower().replace(' ','_')}",0.0))
            x_vals = np.array(lengths)
            fig_p = go.Figure()
            # Draw segment lines (SDH upstream -> RH downstream)
            for i in range(len(stations_data)):
                fig_p.add_trace(go.Scatter(
                    x=[x_vals[i], x_vals[i+1]], y=[sdh[i], rh[i+1]],
                    mode='lines+markers', name=f"{names_p[i]}→{names_p[i+1]} (drop)", line=dict(color="blue")
                ))
            # Draw vertical jump for each pump station
            for i in range(len(stations_data)):
                key = stations_data[i]['name'].strip().lower().replace(' ','_')
                if stations_data[i].get('is_pump', False):
                    # Vertical line: RH up to SDH at same station
                    fig_p.add_trace(go.Scatter(
                        x=[x_vals[i], x_vals[i]],
                        y=[rh[i], sdh[i]],
                        mode='lines+markers',
                        line=dict(color='red', dash='dot', width=3),
                        name=f"{names_p[i]}: Pump jump"
                    ))
            fig_p.update_layout(title="Pressure drop vs Pipeline Length", xaxis_title="Cumulative Length (km)", yaxis_title="Pressure (mcl)")
            st.plotly_chart(fig_p, use_container_width=True)

        # Power vs Speed, Power vs Flow
        with power_tab:
            st.markdown("<div class='section-title'>Power vs Speed & Power vs Flow</div>", unsafe_allow_html=True)
            for i, stn in enumerate(stations_data, start=1):
                if not stn.get('is_pump', False):
                    continue
                key = stn['name'].strip().lower().replace(' ','_')
                A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
                P = stn.get('P',0); Qc = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
                N_min = int(res.get(f"min_rpm_{key}", 0))
                N_max = int(res.get(f"dol_{key}", 0))
                flow = FLOW
                rho = stn.get('rho',850.0)
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
                flows_user = st.session_state[f"eff_data_{i}"].iloc[:,0].values
                flows = np.linspace(flows_user.min(), flows_user.max(), 100)
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

    # === Tab 4: Pump System Interaction Curves ===
    with tab4:
        st.markdown("<div class='section-title'>Pump-System Interaction Curves (System at various %DR, Pumps at various speeds, for optimized NOP)</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False):
                continue
            key = stn['name'].strip().lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']; L_seg = stn['L']; elev_i = stn['elev']
            max_dr = int(stn.get('max_dr', 40))
            FLOW_val = FLOW
            KV = stn.get('KV', 10.0)
            A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
            N_min = int(res.get(f"min_rpm_{key}", 0))
            N_max = int(res.get(f"dol_{key}", 0))
            opt_nop = int(res.get(f"num_pumps_{key}", 1))
            # System curves at different DR%
            fig = go.Figure()
            for dra in range(0, max_dr+1, 5):
                flows = np.linspace(0.01, FLOW_val*1.5, 100)
                v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (KV*1e-6) if KV>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                fig.add_trace(go.Scatter(x=flows, y=SDH_vals, mode='lines', name=f"System ({dra}% DR)"))
            # Pump curves at different speeds (for optimized NOP)
            for rpm in range(N_min, N_max+1, 100):
                flows_user = st.session_state[f"head_data_{i}"].iloc[:,0].values
                flows = np.linspace(flows_user.min(), flows_user.max(), 200)
                Hpump = (A*flows**2 + B*flows + C)*(rpm/N_max)**2 * opt_nop
                fig.add_trace(go.Scatter(x=flows, y=Hpump, mode='lines', line=dict(dash='dot'), name=f"Pump ({rpm} rpm, {opt_nop} NOP)"))
            fig.update_layout(title=f"System vs Pump Curves: {stn['name']}", xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
            st.plotly_chart(fig, use_container_width=True)

    # === Tab 5: 3D Cost Surface (One station at a time, with optimizer marker) ===
    with tab5:
        st.markdown("<div class='section-title'>3D Total Cost vs Speed vs No. of Pumps</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False):
                continue
            key = stn['name'].strip().lower().replace(' ','_')
            # Only show for stations with nonzero NOP
            opt_nop = int(res.get(f"num_pumps_{key}", 0))
            if opt_nop == 0:
                st.info(f"No pumps running at optimized point for {stn['name']}. Skipping 3D cost plot.")
                continue
            opt_speed = res.get(f"speed_{key}", 0)
            opt_dra = res.get(f"drag_reduction_{key}", 0)
            opt_cost = res.get(f"power_cost_{key}", 0) + res.get(f"dra_cost_{key}", 0)
            N_min = int(res.get(f"min_rpm_{key}", 0))
            N_max = int(res.get(f"dol_{key}", 0))
            max_nop = int(stn.get('max_pumps', 2))
            max_dr = int(stn.get('max_dr', 40))
            dra_slices = list(range(0, max_dr+1, 10)) if max_dr >= 10 else [0, max_dr]
            FLOW_val = FLOW
            rho = stn.get('rho',850.0)
            RateDRA = RateDRA
            Price_HSD = Price_HSD
            P = stn.get('P',0); Q = stn.get('Q',0); R = stn.get('R',0); S = stn.get('S',0); T = stn.get('T',0)
            A = res.get(f"coef_A_{key}",0); B = res.get(f"coef_B_{key}",0); C = res.get(f"coef_C_{key}",0)
            N_ref = int(res.get(f"dol_{key}", 0))
            rpm_range = np.arange(N_min, N_max+1, 100)
            nop_range = np.arange(1, max_nop+1, 1)
            surfaces = []
            for dra in dra_slices:
                Z = np.zeros((len(nop_range), len(rpm_range)))
                for ix, nop in enumerate(nop_range):
                    for iy, rpm in enumerate(rpm_range):
                        flow = FLOW_val
                        H = (A*flow**2 + B*flow + C)*(rpm/N_ref)**2 if N_ref>0 else 0
                        eff = (P*flow**4 + Q*flow**3 + R*flow**2 + S*flow + T)
                        eff = max(0.01, eff/100)
                        if nop>0 and rpm>0 and eff>0:
                            pwr = (rho * flow * 9.81 * H * nop)/(3600.0*1000.0*eff*0.95)
                        else:
                            pwr = 0
                        dra_cost = (dra/4)*(flow*1000.0*24.0/1e6)*RateDRA
                        power_cost = pwr*24*stn.get('rate', 0)
                        Z[ix,iy] = power_cost + dra_cost
                surfaces.append(go.Surface(
                    x=rpm_range, y=nop_range, z=Z,
                    name=f"DRA {dra}%",
                    showscale=False,
                    opacity=0.7,
                    hovertemplate="Speed: %{x} rpm<br>NoP: %{y}<br>Cost: %{z:.0f}<br>DRA: "+str(dra)+"%"
                ))
            marker_trace = go.Scatter3d(
                x=[opt_speed], y=[opt_nop], z=[opt_cost],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='diamond'),
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


