import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import pi
import uuid
import os
import json
from datetime import datetime

# --------- PRO STYLES ----------
st.markdown("""
<style>
body {background: #f6f8fb !important;}
.main-container {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 6px 24px rgba(20,60,120,0.07), 0 1.5px 4px rgba(15,55,130,0.12);
    margin: 0 auto;
    padding: 1.8em 2em 2.2em 2em;
    max-width: 1100px;
}
.card {
    background: #f7fafd;
    border-radius: 13px;
    padding: 1.2em 1.6em;
    margin-bottom: 1.6em;
    box-shadow: 0 2px 8px rgba(100,110,180,0.05);
}
.stTabs [data-baseweb="tab-list"] {justify-content: center;}
.stTabs [data-baseweb="tab"] {font-size: 1.16rem; padding: 1em 2em;}
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg,#2571b4,#19b18a 65%);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    min-width: 140px;
    min-height: 44px;
    box-shadow: 0 2px 8px rgba(36,99,167,0.08);
    margin-right: 1em;
}
.stButton>button:hover {filter: brightness(1.1);}
.input-label {font-weight: 550; color: #23456c;}
.section-title {font-size: 1.22em; color: #2a5078; font-weight: 650; margin-bottom: 0.45em;}
hr {border: 0; height: 1.5px; background: linear-gradient(90deg, #bdd7fa 50%, #eaf6ff 100%); margin-bottom: 1em;}
.logo-header {display: flex; align-items: center; margin-bottom: 1.3em;}
.logo-header img {height: 44px; margin-right: 1.1em;}
.brand-title {
    font-size: 2.4rem; font-weight: 700; color: #23456c; margin-top: 0.1em;
    letter-spacing: -0.5px;
}
.card-row {display: flex; gap: 18px;}
.card-row > div {flex: 1;}
::-webkit-scrollbar {height: 8px; width: 8px;}
::-webkit-scrollbar-thumb {background: #d1e1f1; border-radius: 7px;}
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://img.icons8.com/external-flaticons-flat-flat-icons/64/000000/external-pipeline-industry-flaticons-flat-flat-icons-3.png"

# --------- SIDEBAR: ONLY CASE MANAGEMENT ----------
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.markdown("### Case Management")
    if st.button("💾 Save Case"):
        case_data = {
            "stations": st.session_state.get("stations", []),
            "terminal": st.session_state.get("terminal", {}),
            "fluid": st.session_state.get("fluid", {}),
        }
        jstr = json.dumps(case_data)
        st.download_button("Download Case (.json)", data=jstr, file_name=f"pipeline_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    st.markdown("**Load Case**")
    case_file = st.file_uploader("Upload JSON", type="json", label_visibility="collapsed")
    if case_file is not None:
        try:
            loaded = json.load(case_file)
            st.session_state["stations"] = loaded.get("stations", [])
            st.session_state["terminal"] = loaded.get("terminal", {})
            st.session_state["fluid"] = loaded.get("fluid", {})
            st.success("Case loaded!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Could not load case: {e}")

# --------- MAIN CONTENT: CARDS & TABS -------------
st.markdown(f"""
<div class="main-container">
<div class="logo-header">
    <img src="{LOGO_URL}" />
    <span class="brand-title">Pipeline Optima</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Pipeline Inputs", "Optimization Results", "Visualization", "PDF Report"])

# --------- 1. PIPELINE INPUTS TAB -----------
with tabs[0]:
    st.markdown('<div class="section-title">Station Input Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Station Input Mode", ["Form", "Bulk Table"], horizontal=True, label_visibility="collapsed")
    
    # ---- Fluid & Cost Params ----
    with st.expander("🌊 Global Fluid & Cost Parameters", expanded=True):
        if "fluid" not in st.session_state:
            st.session_state["fluid"] = {"flow": 1000.0, "DRA": 500.0, "diesel": 70.0}
        f = st.session_state["fluid"]
        f["flow"] = st.number_input("Flow rate (m³/hr)", value=f.get("flow", 1000.0), step=10.0)
        f["DRA"] = st.number_input("DRA Cost (INR/L)", value=f.get("DRA", 500.0), step=1.0)
        f["diesel"] = st.number_input("Diesel Price (INR/L)", value=f.get("diesel", 70.0), step=0.5)
    
    # --------- Stations -------------
    if "stations" not in st.session_state:
        st.session_state["stations"] = [{
            "name": "Station 1", "elev": 0.0, "D": 0.711, "t": 0.007,
            "SMYS": 52000.0, "rough": 0.00004, "L": 50.0,
            "min_residual": 50.0, "is_pump": False,
            "power_type": "Grid", "rate": 9.0, "sfc": 150.0,
            "max_pumps": 1, "MinRPM": 1200.0, "DOL": 1500.0,
            "max_dr": 0.0, "rho": 850.0, "KV": 10.0
        }]
    
    if mode == "Form":
        station_names = [s["name"] for s in st.session_state["stations"]]
        selected_station = st.selectbox("Select/Edit Station", station_names, key="station_select")
        stn_idx = station_names.index(selected_station)
        stn = st.session_state["stations"][stn_idx]
        with st.form(f"station_{stn_idx}"):
            st.markdown('<div class="section-title">Edit Station Parameters</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                stn["name"] = st.text_input("Station Name", value=stn["name"])
                stn["elev"] = st.number_input("Elevation (m)", value=stn["elev"])
                stn["L"] = st.number_input("Length to Next (km)", value=stn["L"])
                stn["D"] = st.number_input("Diameter (OD, m)", value=stn["D"])
                stn["t"] = st.number_input("Wall Thickness (m)", value=stn["t"])
                stn["SMYS"] = st.number_input("SMYS (psi)", value=stn["SMYS"])
                stn["rough"] = st.number_input("Pipe Roughness (m)", value=stn["rough"])
            with col2:
                stn["is_pump"] = st.checkbox("Is Pump Station?", value=stn["is_pump"])
                stn["max_dr"] = st.number_input("Max Drag Reduction (%)", value=stn["max_dr"])
                stn["rho"] = st.number_input("Density (kg/m³)", value=stn["rho"])
                stn["KV"] = st.number_input("Viscosity (cSt)", value=stn["KV"])
                if stn["is_pump"]:
                    st.markdown('<hr><b>Pump Parameters</b>', unsafe_allow_html=True)
                    stn["power_type"] = st.selectbox("Power Source", ["Grid", "Diesel"], index=0 if stn["power_type"]=="Grid" else 1)
                    stn["MinRPM"] = st.number_input("Min RPM", value=stn["MinRPM"])
                    stn["DOL"] = st.number_input("Rated RPM", value=stn["DOL"])
                    stn["max_pumps"] = st.number_input("Max Pumps", min_value=1, value=stn["max_pumps"])
                    if stn["power_type"]=="Grid":
                        stn["rate"] = st.number_input("Electricity Rate (INR/kWh)", value=stn["rate"])
                    else:
                        stn["sfc"] = st.number_input("SFC (gm/bhp·hr)", value=stn["sfc"])
            st.markdown('<hr><b>Intermediate Peaks</b>', unsafe_allow_html=True)
            peaks_json = st.text_area("Peaks (JSON array, e.g., [{'loc':10,'elev':115}])", value=json.dumps(stn.get("peaks", [])))
            try:
                stn["peaks"] = json.loads(peaks_json)
            except:
                st.warning("Invalid JSON for peaks.")
            submitted = st.form_submit_button("Save Station")
            if submitted:
                st.success("Station updated.")
        # Add/remove station buttons
        c1, c2 = st.columns([1,1])
        if c1.button("+ Add Station"):
            st.session_state["stations"].append(st.session_state["stations"][-1].copy())
        if c2.button("🗑 Remove Station") and len(st.session_state["stations"]) > 1:
            st.session_state["stations"].pop(stn_idx)
    
    # -------- Bulk Table Mode -----------
    else:
        df = pd.DataFrame(st.session_state["stations"])
        edited = st.data_editor(df, key="stations_editor", num_rows="dynamic")
        if st.button("Apply Bulk Changes"):
            st.session_state["stations"] = edited.to_dict("records")
            st.success("Bulk station data updated.")
    
    # ------- Terminal Station -------
    with st.expander("🏁 Terminal Station", expanded=True):
        if "terminal" not in st.session_state:
            st.session_state["terminal"] = {"name": "Terminal", "elev": 0.0, "min_residual": 50.0}
        t = st.session_state["terminal"]
        t["name"] = st.text_input("Terminal Name", value=t["name"])
        t["elev"] = st.number_input("Terminal Elevation (m)", value=t["elev"])
        t["min_residual"] = st.number_input("Minimum Residual Head (m)", value=t["min_residual"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; color:gray; font-size:0.95em;">'
        '© 2025 Pipeline Optima v1.1.1. Developed by Parichay Das. All rights reserved.</div>',
        unsafe_allow_html=True
    )

# At this point, add the rest of your tabs, solver, report generation, and all remaining features.
st.markdown('<div class="footer">&copy; 2025 Pipeline Optima v1.1.1. Developed by Parichay Das. All rights reserved.</div>', unsafe_allow_html=True)


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
    "🧊 3D Analysis and Surface Plots"      
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
        params = [
            "Power+Fuel Cost", "DRA Cost", "No. of Pumps", "Pump Speed (rpm)", "Pump Eff (%)",
            "Reynolds No.", "Head Loss (m)", "Vel (m/s)", "Residual Head (m)", "SDH (m)", "DRA (%)"
        ]
        summary = {"Parameters": params}
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
        # Remove index column, format decimals, left align all
        fmt = {c: "{:.2f}" for c in df_sum.columns if c != "Parameters"}
        fmt["No. of Pumps"] = "{:.0f}"
        fmt["Pump Speed (rpm)"] = "{:.0f}"
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        styled = df_sum.style.format(fmt).set_properties(**{'text-align': 'left'})
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")
        # Show summary below the table
        st.markdown(
            f"""<br>
            <div style='font-size:1.1em;'><b>Total Optimized Cost:</b> {res.get('total_cost', 0):,.2f} INR/day<br>
            <b>No. of operating Pumps:</b> {int(res.get('num_pumps_'+names[0].lower().replace(' ','_'),0))}<br>
            <b>Average Pump Efficiency:</b> {res.get('efficiency_'+names[0].lower().replace(' ','_'),0.0):.2f} %<br>
            <b>Average Pump Speed:</b> {res.get('speed_'+names[0].lower().replace(' ','_'),0.0):.0f} rpm</div>
            """,
            unsafe_allow_html=True
        )

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
        fig_pie = px.pie(df_cost, names='Station', values='Total', title="Station-wise Cost Breakdown")
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
                    pwr = (stn['rho'] * flow * 9.81 * H)/(3600.0*eff*0.95*1000)
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
                    pwr = (stn['rho'] * q * 9.81 * H)/(3600.0*eff*0.95*1000)
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
    if "last_res" not in st.session_state or "last_stations_data" not in st.session_state:
        st.info("Please run optimization at least once to enable 3D analysis.")
        st.stop()
    last_res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]
    FLOW = st.session_state.get("FLOW", 1000.0)
    RateDRA = st.session_state.get("RateDRA", 500.0)
    Price_HSD = st.session_state.get("Price_HSD", 70.0)
    key = stations_data[0]['name'].lower().replace(' ', '_')

    speed_opt = float(last_res.get(f"speed_{key}", 1500.0))
    dra_opt = float(last_res.get(f"drag_reduction_{key}", 0.0))
    nopt_opt = int(last_res.get(f"num_pumps_{key}", 1))
    flow_opt = FLOW

    delta_speed = 150
    delta_dra = 10
    delta_nop = 1
    delta_flow = 150
    N = 9
    stn = stations_data[0]
    N_min = int(stn.get('MinRPM', 1000))
    N_max = int(stn.get('DOL', 1500))
    DRA_max = int(stn.get('max_dr', 40))
    max_pumps = int(stn.get('max_pumps', 4))

    speed_range = np.linspace(max(N_min, speed_opt - delta_speed), min(N_max, speed_opt + delta_speed), N)
    dra_range = np.linspace(max(0, dra_opt - delta_dra), min(DRA_max, dra_opt + delta_dra), N)
    nop_range = np.arange(max(1, nopt_opt - delta_nop), min(max_pumps, nopt_opt + delta_nop)+1)
    flow_range = np.linspace(max(0.01, flow_opt - delta_flow), flow_opt + delta_flow, N)

    groups = {
        "Pump Performance Surface Plots": {
            "Head vs Flow vs Speed": {"x": flow_range, "y": speed_range, "z": "Head"},
            "Efficiency vs Flow vs Speed": {"x": flow_range, "y": speed_range, "z": "Efficiency"},
        },
        "System Interaction Surface Plots": {
            "System Head vs Flow vs DRA": {"x": flow_range, "y": dra_range, "z": "SystemHead"},
        },
        "Cost Surface Plots": {
            "Power Cost vs Speed vs DRA": {"x": speed_range, "y": dra_range, "z": "PowerCost"},
            "Power Cost vs Flow vs Speed": {"x": flow_range, "y": speed_range, "z": "PowerCost"},
            "Total Cost vs NOP vs DRA": {"x": nop_range, "y": dra_range, "z": "TotalCost"},
        }
    }
    col1, col2 = st.columns(2)
    group = col1.selectbox("Plot Group", list(groups.keys()))
    plot_opt = col2.selectbox("Plot Type", list(groups[group].keys()))
    conf = groups[group][plot_opt]
    Xv, Yv = np.meshgrid(conf['x'], conf['y'], indexing='ij')
    Z = np.zeros_like(Xv, dtype=float)

    # --- Pump coefficients ---
    A = stn.get('A', 0); B = stn.get('B', 0); Cc = stn.get('C', 0)
    P = stn.get('P', 0); Qc = stn.get('Q', 0); R = stn.get('R', 0)
    S = stn.get('S', 0); T = stn.get('T', 0)
    DOL = float(stn.get('DOL', N_max))
    rho = stn.get('rho', 850.0)
    rate = stn.get('rate', 9.0)
    g = 9.81

    def get_head(q, n): return (A*q**2 + B*q + Cc)*(n/DOL)**2
    def get_eff(q, n): q_adj = q * DOL/n if n > 0 else q; return (P*q_adj**4 + Qc*q_adj**3 + R*q_adj**2 + S*q_adj + T)
    def get_power_cost(q, n, d, npump=1):
        h = get_head(q, n)
        eff = max(get_eff(q, n)/100, 0.01)
        pwr = (rho*q*g*h*npump)/(3600.0*eff*0.95*1000)
        return pwr*24*rate
    def get_system_head(q, d):
        d_inner = stn['D'] - 2*stn['t']
        rough = stn['rough']
        L_seg = stn['L']
        v = q/3600.0/(np.pi*(d_inner**2)/4)
        Re = v*d_inner/(stn['KV']*1e-6) if stn['KV'] > 0 else 0
        if Re > 0:
            f = 0.25/(np.log10(rough/d_inner/3.7 + 5.74/(Re**0.9))**2)
        else:
            f = 0.0
        DH = f*((L_seg*1000.0)/d_inner)*(v**2/(2*g))*(1-d/100)
        return stn['elev'] + DH
    def get_total_cost(q, n, d, npump):
        pcost = get_power_cost(q, n, d, npump)
        dracost = (d/4)*(q*1000.0*24.0/1e6)*RateDRA
        return pcost + dracost

    for i in range(Xv.shape[0]):
        for j in range(Xv.shape[1]):
            if plot_opt == "Head vs Flow vs Speed":
                Z[i,j] = get_head(Xv[i,j], Yv[i,j])
            elif plot_opt == "Efficiency vs Flow vs Speed":
                Z[i,j] = get_eff(Xv[i,j], Yv[i,j])
            elif plot_opt == "System Head vs Flow vs DRA":
                Z[i,j] = get_system_head(Xv[i,j], Yv[i,j])
            elif plot_opt == "Power Cost vs Speed vs DRA":
                Z[i,j] = get_power_cost(flow_opt, Xv[i,j], Yv[i,j], nopt_opt)
            elif plot_opt == "Power Cost vs Flow vs Speed":
                Z[i,j] = get_power_cost(Xv[i,j], Yv[i,j], dra_opt, nopt_opt)
            elif plot_opt == "Total Cost vs NOP vs DRA":
                Z[i,j] = get_total_cost(flow_opt, speed_opt, Yv[i,j], int(Xv[i,j]))

    axis_labels = {
        "Flow": "X: Flow (m³/hr)",
        "Speed": "Y: Pump Speed (rpm)",
        "Head": "Z: Head (m)",
        "Efficiency": "Z: Efficiency (%)",
        "SystemHead": "Z: System Head (m)",
        "PowerCost": "Z: Power Cost (INR/day)",
        "DRA": "Y: DRA (%)",
        "NOP": "X: No. of Pumps",
        "TotalCost": "Z: Total Cost (INR/day)",
    }
    label_map = {
        "Head vs Flow vs Speed": ["Flow", "Speed", "Head"],
        "Efficiency vs Flow vs Speed": ["Flow", "Speed", "Efficiency"],
        "System Head vs Flow vs DRA": ["Flow", "DRA", "SystemHead"],
        "Power Cost vs Speed vs DRA": ["Speed", "DRA", "PowerCost"],
        "Power Cost vs Flow vs Speed": ["Flow", "Speed", "PowerCost"],
        "Total Cost vs NOP vs DRA": ["NOP", "DRA", "TotalCost"]
    }
    xlab, ylab, zlab = [axis_labels[l] for l in label_map[plot_opt]]

    fig = go.Figure(data=[go.Surface(
        x=conf['x'], y=conf['y'], z=Z.T, colorscale='Viridis', colorbar=dict(title=zlab)
    )])

    # No diamond/Optimum marker is added here.

    fig.update_layout(
        scene=dict(
            xaxis_title=xlab,
            yaxis_title=ylab,
            zaxis_title=zlab
        ),
        title=f"{plot_opt}",
        height=750,
        margin=dict(l=30, r=30, b=30, t=80)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='centered-caption'>Surface shown for a small region (+/- delta) from the optimum point for clarity and hydraulic relevance.</div>",
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
    &copy; 2025 Pipeline Optima v1.1.1. Developed by Parichay Das. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
