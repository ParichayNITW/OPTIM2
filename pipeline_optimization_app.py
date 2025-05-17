import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory

if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 NEOS_EMAIL not found in secrets. Please add it.")

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(
    page_title="Mixed Integer Non Linear Convex Optimization of Pipeline Operations",
    layout="wide"
)

# ---------------------
# Custom CSS
# ---------------------
st.markdown(
    """
    <style>
      .reportview-container, .main, .block-container, .sidebar .sidebar-content {
        background: none !important;
      }
      .stMetric > div {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(5px);
        border-radius: 8px;
        padding: 12px;
        color: var(--text-primary-color) !important;
        text-align: center;
      }
      .stMetric .metric-value,
      .stMetric .metric-label {
        display: block;
        width: 100%;
        text-align: center;
      }
      .color-adapt {
        color: var(--text-primary-color) !important;
      }
      .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary-color) !important;
        margin-top: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------
# Title
# ---------------------
st.markdown("<h1 class='color-adapt'>Mixed Integer Non Linear Convex Optimization of Pipeline Operations</h1>", unsafe_allow_html=True)

# ---------------------
# Solver wrapper
# ---------------------
def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    import pipeline_model  # your back-end module
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)

# ---------------------
# Sidebar: global inputs + dynamic station builder
# ---------------------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW = st.number_input("Flow rate (m³/hr)", value=2000.0, step=10.0)
        KV = st.number_input("Viscosity (cSt)", value=10.0, step=0.1)
        rho = st.number_input("Density (kg/m³)", value=880.0, step=10.0)
        RateDRA = st.number_input("DRA Rate (INR/L)", value=500.0, step=0.1)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)

    # Buttons to add/remove stations
    add_col, rem_col = st.columns(2)
    add_btn = add_col.button("➕ Add Station")
    rem_btn = rem_col.button("🗑️ Remove Station")
    if 'stations' not in st.session_state:
        # initialize with one pump station
        st.session_state.stations = [{
            'name': 'Station 1', 'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Grid', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 3, 'MinRPM': 1200.0, 'DOL': 1500.0,
            'max_dr': 40.0,
            'A': -2e-6, 'B': -0.0015, 'C': 179.14,
            'P': -4.161e-14, 'Q': 6.574e-10, 'R': -8.737e-06, 'S': 0.04924, 'T': -0.001754
        }]
    if add_btn:
        n = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name': f'Station {n}', 'elev': 0.0,
            'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Diesel', 'rate': 9.0,
            'sfc': 150.0, 'max_pumps': 2, 'MinRPM': 2750.0, 'DOL': 3437.0,
            'max_dr': 40.0,
            'A': -1e-5, 'B': 0.00135, 'C': 270.08,
            'P': -4.07e-13, 'Q': 3.4657e-09, 'R': -1.9273e-05, 'S': 0.067033, 'T': -0.15043
        })
    if rem_btn and len(st.session_state.stations) > 1:
        st.session_state.stations.pop()

    # Render each station’s inputs
    for idx, stn in enumerate(st.session_state.stations, start=1):
        with st.expander(f"Station {idx}: {stn['name']}", expanded=True):
            stn['name'] = st.text_input("Name",     value=stn['name'], key=f"name{idx}")
            stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], key=f"elev{idx}")
            stn['D']   = st.number_input(
                "Outer Diameter (m)",
                value=stn['D'],
                step=0.00001,
                format="%.5f",
                key=f"D{idx}"
            )
            stn['t']   = st.number_input(
                "Wall Thickness (m)",
                value=stn['t'],
                step=0.00001,
                format="%.5f",
                key=f"t{idx}"
            )
            stn['SMYS']= st.number_input("SMYS (psi)", value=stn['SMYS'], key=f"SMYS{idx}")
            stn['rough'] = st.number_input(
                "Pipe Roughness (m)",
                value=stn['rough'],
                step=0.00001,
                format="%.5f",
                key=f"rough{idx}"
            )
            stn['L']    = st.number_input("Length to next (km)", value=stn['L'], key=f"L{idx}")
            stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")
            if stn['is_pump']:
                stn['power_type'] = st.selectbox("Power Source", ["Grid","Diesel"],
                                                 index=(0 if stn['power_type']=="Grid" else 1),
                                                 key=f"ptype{idx}")
                if stn['power_type']=="Grid":
                    stn['rate'] = st.number_input("Electricity Rate (INR/kWh)",
                                                  value=stn['rate'], key=f"rate{idx}")
                else:
                    stn['sfc'] = st.number_input("SFC (gm/bhp-hr)",
                                                 value=stn['sfc'], key=f"sfc{idx}")
                stn['max_pumps'] = st.number_input("Available Pumps", min_value=1,
                                                   value=stn['max_pumps'], step=1, key=f"mpumps{idx}")
                stn['MinRPM'] = st.number_input("Min RPM",      value=stn['MinRPM'], key=f"minrpm{idx}")
                stn['DOL']    = st.number_input("Rated RPM",    value=stn['DOL'],    key=f"dol{idx}")
                stn['max_dr'] = st.number_input("Max Drag Reduction (%)",
                                                value=stn['max_dr'], key=f"mdr{idx}")
                st.file_uploader("Pump Head Curve (img)", type=["png","jpg","jpeg"], key=f"headimg{idx}")
                st.file_uploader("Efficiency Curve (img)", type=["png","jpg","jpeg"], key=f"effimg{idx}")

run = st.button("🚀 Run Optimization")

if run:
    with st.spinner("Solving pipeline optimization..."):
        stations_data = st.session_state.stations
        terminal_data = {"name":"Terminal","elev":0.0}
        res = solve_pipeline(stations_data, terminal_data, FLOW, KV, rho, RateDRA, Price_HSD)

    # KPI Cards
    total_cost = res.get('total_cost',0)
    total_pumps = sum(res.get(f"num_pumps_{s['name'].lower()}",0) for s in stations_data)
    speeds = [res.get(f"speed_{s['name'].lower()}",0) for s in stations_data]
    effs   = [res.get(f"efficiency_{s['name'].lower()}",0) for s in stations_data]
    avg_speed = np.mean(speeds) if speeds else 0
    avg_eff   = np.mean(effs)   if effs   else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost (INR)", f"₹{total_cost:,.2f}")
    c2.metric("Total Pumps", total_pumps)
    c3.metric("Avg Speed (rpm)", f"{avg_speed:.2f}")
    c4.metric("Avg Efficiency (%)", f"{avg_eff:.2f}")

    # Summary table
    station_names = [s['name'] for s in stations_data] + [terminal_data['name']]
    summary = {"Process Particulars": [
        "Power & Fuel cost (INR/day)", "DRA cost (INR/day)", "No. of Pumps",
        "Pump Speed (rpm)", "Pump Efficiency (%)", "Reynold's No.",
        "Dynamic Head Loss (m)", "Velocity (m/s)", "Residual Head (m)",
        "SDH (m)", "Drag Reduction (%)"
    ]}
    for s in station_names:
        key = s.lower()
        num = int(res.get(f"num_pumps_{key}", 0))
        sp = res.get(f"speed_{key}", 0) if num > 0 else 0
        ef = res.get(f"efficiency_{key}", 0) if num > 0 else 0
        summary[s] = [
            round(res.get(f"power_cost_{key}", 0), 2),
            round(res.get(f"dra_cost_{key}", 0), 2),
            num,
            round(sp, 2),
            round(ef, 2),
            round(res.get(f"reynolds_{key}", 0), 2),
            round(res.get(f"head_loss_{key}", 0), 2),
            round(res.get(f"velocity_{key}", 0), 2),
            round(res.get(f"residual_head_{key}", 0), 2),
            round(res.get(f"sdh_{key}", 0), 2),
            round(res.get(f"drag_reduction_{key}", 0), 2)
        ]
    df_sum = pd.DataFrame(summary)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary Table", "💰 Cost Charts", "⚙️ Performance Charts", "🌀 System Curves", "🔄 Pump-System Interaction"])
    # Tab 1: Summary + download
    with tab1:
        st.markdown("<div class='section-title'>Optimized Parameters Summary</div>", unsafe_allow_html=True)
        fmt = {col: "{:.0f}" if col == "No. of Pumps" else "{:.2f}" for col in df_sum.columns if col != "Process Particulars"}
        st.dataframe(df_sum.style.format(fmt).set_properties(**{'text-align':'center'}), use_container_width=True)
        csv_bytes = df_sum.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary as CSV", csv_bytes, "pipeline_summary.csv", "text/csv")
    # Tab 2: Cost breakdown charts
    with tab2:
        st.markdown("<div class='section-title'>Cost Breakdown per Station</div>", unsafe_allow_html=True)
        df_cost = pd.DataFrame({
            "Station": station_names[:-1],  # exclude terminal (no cost)
            "Power & Fuel (INR/day)": [res.get(f"power_cost_{s.lower()}", 0) for s in station_names[:-1]],
            "DRA (INR/day)": [res.get(f"dra_cost_{s.lower()}", 0) for s in station_names[:-1]]
        })
        fig_cost = px.bar(
            df_cost.melt(id_vars="Station", value_vars=["Power & Fuel (INR/day)", "DRA (INR/day)"], var_name="Type", value_name="Amount"),
            x="Station", y="Amount", color="Type", barmode="group", title="Cost Components by Station"
        )
        fig_cost.update_layout(xaxis_title="Station", yaxis_title="Amount (INR)")
        fig_cost.update_yaxes(tickformat=".2f")
        st.plotly_chart(fig_cost, use_container_width=True)
    # Tab 3: Performance charts
    with tab3:
        perf_tab, pump_tab, eff_tab = st.tabs(["Performance Metrics", "Pump Characteristic Curves", "Pump Efficiency Curves"])
        with perf_tab:
            st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)
            df_perf = pd.DataFrame({
                "Station": station_names[:-1],
                "Head Loss (m)": [res.get(f"head_loss_{s.lower()}", 0) for s in station_names[:-1]]
            })
            fig_hl = go.Figure(go.Bar(x=df_perf["Station"], y=df_perf["Head Loss (m)"]))
            fig_hl.update_layout(title_text="Head Loss by Segment", xaxis_title="Station", yaxis_title="Head Loss (m)")
            fig_hl.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig_hl, use_container_width=True)
        with pump_tab:
            st.markdown("<div class='section-title'>Pump Characteristic Curves</div>", unsafe_allow_html=True)
            pump_stations = [s for s in station_names[:-1] if res.get(f"coef_A_{s.lower()}", None) is not None]
            selected = st.multiselect("Select stations", pump_stations, default=pump_stations)
            flow_range = np.arange(0, max(4500, FLOW+1), 100)
            for stn in selected:
                key = stn.lower()
                A = res.get(f"coef_A_{key}", None)
                B = res.get(f"coef_B_{key}", None)
                C = res.get(f"coef_C_{key}", None)
                dol = res.get(f"dol_{key}", None)
                mn = res.get(f"min_rpm_{key}", None)
                if None in [A, B, C, dol, mn]:
                    continue
                dfs = []
                for rpm in np.arange(mn, dol+1, 100):
                    H_curve = (A*flow_range**2 + B*flow_range + C) * (rpm/dol)**2
                    dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_range, "Head (m)": H_curve, "RPM": rpm}))
                df_all = pd.concat(dfs, ignore_index=True)
                fig_pc = px.line(df_all, x="Flow (m³/hr)", y="Head (m)", color="RPM", title=f"Pump Curves ({stn})")
                fig_pc.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
                fig_pc.update_yaxes(tickformat=".2f")
                st.plotly_chart(fig_pc, use_container_width=True)
        with eff_tab:
            st.markdown("<div class='section-title'>Pump Efficiency Curves</div>", unsafe_allow_html=True)
            flow_range = np.arange(0, max(4500, FLOW+1), 100)
            for stn in pump_stations:
                key = stn.lower()
                Pcoef = res.get(f"coef_P_{key}", None)
                Qcoef = res.get(f"coef_Q_{key}", None)
                Rcoef = res.get(f"coef_R_{key}", None)
                Scoef = res.get(f"coef_S_{key}", None)
                Tcoef = res.get(f"coef_T_{key}", None)
                dol = res.get(f"dol_{key}", None)
                mn = res.get(f"min_rpm_{key}", None)
                if None in [Pcoef, Qcoef, Rcoef, Scoef, Tcoef, dol, mn]:
                    continue
                dfs = []
                for rpm in np.arange(mn, dol+1, 100):
                    # Equivalent flow at rated speed
                    flow_eq = flow_range * dol / rpm
                    E_curve = (Pcoef*flow_eq**4 + Qcoef*flow_eq**3 + Rcoef*flow_eq**2 + Scoef*flow_eq + Tcoef) / 100
                    mask = E_curve > 0
                    if not mask.any():
                        continue
                    dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_range[mask], "Efficiency (%)": E_curve[mask]*100, "RPM": rpm}))
                if dfs:
                    df_eff = pd.concat(dfs, ignore_index=True)
                    fig_eff = px.line(df_eff, x="Flow (m³/hr)", y="Efficiency (%)", color="RPM", title=f"Efficiency Curves ({stn})")
                    fig_eff.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="Efficiency (%)")
                    fig_eff.update_yaxes(tickformat=".2f")
                    st.plotly_chart(fig_eff, use_container_width=True)
    # Tab 4: System curves (SDHR vs flow for various DRA)
    with tab4:
        st.markdown("<div class='section-title'>System Curves of SDHR</div>", unsafe_allow_html=True)
        flow_arr = np.arange(0, max(4500, FLOW+1), 100)
        for i, stn in enumerate(station_names[:-1], start=1):
            # skip non-pump stations for system curve plotting
            if not stations_data[i-1]['is_pump']:
                continue
            # static head diff of segment i
            sd = stations_data[i-1]['elev'] if i == 1 else (stations_data[i-1]['elev'] - (stations_data[i-2]['elev'] if i-2 >= 0 else 0))
            d = stations_data[i-1]['D'] - 2*stations_data[i-1]['t']
            rough = stations_data[i-1]['rough']
            L_seg = stations_data[i-1]['L']
            dfs = []
            for dra in range(0, int(stations_data[i-1]['max_dr'])+5, 5):
                v = flow_arr/(3.1416*(d**2)/4)/3600
                Re = v * d / (KV * 1e-6)
                f = 0.25/(np.log10((rough/d/3.7)+(5.74/(Re**0.9)))**2)
                DH = (f*(L_seg*1000/d)*(v**2/(2*9.81))) * (1 - dra/100)
                dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr, "SDHR (m)": sd + DH, "DRA (%)": dra}))
            df_sys = pd.concat(dfs, ignore_index=True)
            fig_sys = px.line(df_sys, x="Flow (m³/hr)", y="SDHR (m)", color="DRA (%)", title=f"SDHR Curves ({stn})")
            fig_sys.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="SDHR (m)")
            fig_sys.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig_sys, use_container_width=True)
    # Tab 5: Pump-System interaction
    with tab5:
        st.markdown("<div class='section-title'>Pump-System Interaction</div>", unsafe_allow_html=True)
        flow_arr = np.arange(0, max(4500, FLOW+1), 100)
        for stn in station_names[:-1]:
            if res.get(f"coef_A_{stn.lower()}", None) is None:
                continue
            # system curves
            p_idx = station_names.index(stn) + 1  # segment index for this station
            d = (stations_data[p_idx-1]['D'] - 2*stations_data[p_idx-1]['t']) if p_idx <= len(stations_data) else stations_data[-1]['D']
            rough = stations_data[p_idx-1]['rough']
            L_seg = stations_data[p_idx-1]['L']
            dfs = []
            for dra in range(0, int(stations_data[p_idx-1]['max_dr'])+1, 5):
                v = flow_arr/(3.1416*(d**2)/4)/3600
                Re = v * d / (KV * 1e-6)
                f = 0.25/(np.log10((rough/d/3.7)+(5.74/(Re**0.9)))**2)
                DH = (f*(L_seg*1000/d)*(v**2/(2*9.81))) * (1 - dra/100)
                dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr, "Head (m)": stations_data[p_idx-1]['elev'] + DH, "Curve": f"System DRA {dra}%"}))
            # pump curves for various series configurations
            key = stn.lower()
            A = res.get(f"coef_A_{key}"); B = res.get(f"coef_B_{key}"); C = res.get(f"coef_C_{key}")
            dol = res.get(f"dol_{key}"); mn = res.get(f"min_rpm_{key}")
            num_installed = int(res.get(f"num_pumps_{key}", 1))
            for rpm in np.arange(mn, dol+1, 100):
                H_curve = (A*flow_arr**2 + B*flow_arr + C) * (rpm/dol)**2
                dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr, "Head (m)": H_curve, "Curve": f"Pump {rpm} rpm"}))
                # current optimized series total
                if num_installed > 1:
                    dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr, "Head (m)": H_curve * num_installed, "Curve": f"Pump Total {rpm} x{num_installed}"}))
                # Hypothetical 2-pump series (for comparison)
                dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr, "Head (m)": H_curve * 2, "Curve": f"2 pumps in series {rpm} rpm"}))
            df_comb = pd.concat(dfs, ignore_index=True)
            fig_int = px.line(df_comb, x="Flow (m³/hr)", y="Head (m)", color="Curve", title=f"Pump-System Interaction ({stn})")
            fig_int.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)", height=600)
            fig_int.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig_int, use_container_width=True)
