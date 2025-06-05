# pipeline_app.py

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
from fpdf import FPDF
import tempfile

# PAGE CONFIG FIRST!
st.set_page_config(page_title="Pipeline Optima™", layout="wide")

# -------- SESSION STATE INIT -----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "stations" not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
        'max_dr': 0.0
    }]
if "terminal_name" not in st.session_state:
    st.session_state.terminal_name = "Terminal"
if "terminal_elev" not in st.session_state:
    st.session_state.terminal_elev = 0.0
if "terminal_head" not in st.session_state:
    st.session_state.terminal_head = 50.0
if "FLOW" not in st.session_state:
    st.session_state.FLOW = 1000.0
if "RateDRA" not in st.session_state:
    st.session_state.RateDRA = 500.0
if "Price_HSD" not in st.session_state:
    st.session_state.Price_HSD = 70.0
if "linefill_df" not in st.session_state or len(st.session_state["linefill_df"]) != len(st.session_state["stations"]):
    n = len(st.session_state.stations)
    st.session_state.linefill_df = pd.DataFrame({
        "Segment": list(range(1, n+1)),
        "Viscosity (cSt)": [10.0]*n,
        "Density (kg/m³)": [850.0]*n
    })

# USER LOGIN
def hash_pwd(pwd): return hashlib.sha256(pwd.encode()).hexdigest()
users = {"parichay_das": hash_pwd("heteroscedasticity")}
def check_login():
    if not st.session_state.authenticated:
        st.title("🔒 User Login")
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
            "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v1.2. Developed by Parichay Das. All rights reserved.</div>",
            unsafe_allow_html=True
        )
        st.stop()
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
check_login()

# PALETTE (Remove yellow)
palette = [c for c in qualitative.Plotly if 'yellow' not in c.lower() and '#FFD700' not in c and '#ffeb3b' not in c.lower()]

# DRA CURVES
DRA_CSV_FILES = {10: "10 cst.csv", 15: "15 cst.csv", 20: "20 cst.csv", 25: "25 cst.csv",
                 30: "30 cst.csv", 35: "35 cst.csv", 40: "40 cst.csv"}
DRA_CURVE_DATA = {}
for cst, fname in DRA_CSV_FILES.items():
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        DRA_CURVE_DATA[cst] = df
    else:
        x = np.linspace(0, 60, 7)
        y = 0.8*x + 10 + np.random.uniform(-2,2, size=x.shape)
        DRA_CURVE_DATA[cst] = pd.DataFrame({"%Drag Reduction": x, "PPM": y})

def _ppm_from_df(df, target_dr):
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    if target_dr <= x[0]: return y[0]
    elif target_dr >= x[-1]: return y[-1]
    return np.interp(target_dr, x, y)
def get_ppm_for_dr(viscosity, target_dr, dra_curve_data=DRA_CURVE_DATA):
    cst_list = sorted(dra_curve_data.keys())
    viscosity = float(viscosity)
    if viscosity <= cst_list[0]:
        df = dra_curve_data[cst_list[0]]
        return _ppm_from_df(df, target_dr)
    elif viscosity >= cst_list[-1]:
        df = dra_curve_data[cst_list[-1]]
        return _ppm_from_df(df, target_dr)
    else:
        lower = max([c for c in cst_list if c <= viscosity])
        upper = min([c for c in cst_list if c >= viscosity])
        df_lower = dra_curve_data[lower]
        df_upper = dra_curve_data[upper]
        ppm_lower = _ppm_from_df(df_lower, target_dr)
        ppm_upper = _ppm_from_df(df_upper, target_dr)
        ppm_interp = np.interp(viscosity, [lower, upper], [ppm_lower, ppm_upper])
        return ppm_interp

# LOAD/RESTORE CASE
uploaded_case = st.sidebar.file_uploader("🔁 Load Case", type="json")
if uploaded_case is not None:
    loaded_data = json.load(uploaded_case)
    st.session_state.stations = loaded_data.get('stations', [])
    st.session_state.terminal_name = loaded_data.get('terminal', {}).get('name', "Terminal")
    st.session_state.terminal_elev = loaded_data.get('terminal', {}).get('elev', 0.0)
    st.session_state.terminal_head = loaded_data.get('terminal', {}).get('min_residual', 50.0)
    st.session_state.FLOW = loaded_data.get('FLOW', 1000.0)
    st.session_state.RateDRA = loaded_data.get('RateDRA', 500.0)
    st.session_state.Price_HSD = loaded_data.get('Price_HSD', 70.0)
    st.success("Case loaded!")

# PAGE TITLE
st.markdown(
    "<h1 style='text-align:center;font-size:3.4rem;font-weight:700;color:#232733;margin-bottom:0.25em;margin-top:0.01em;letter-spacing:0.5px;font-family: inherit;'>Pipeline Optima™</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;font-size:2.05rem;font-weight:700;color:#232733;margin-bottom:0.15em;margin-top:0.02em;font-family: inherit;'>Mixed Integer Non-Linear Non-Convex Pipeline Optimization</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.6em; margin-bottom:1.2em; border: 1px solid #e1e5ec;'>", unsafe_allow_html=True)

# --------- GLOBAL PARAMETERS ---------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=st.session_state.FLOW, step=10.0, key="FLOW")
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=st.session_state.RateDRA, step=1.0, key="RateDRA")
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=st.session_state.Price_HSD, step=0.5, key="Price_HSD")
    st.subheader("Stations")
    add_col, rem_col = st.columns(2)
    if add_col.button("➕ Add Station"):
        n = len(st.session_state.stations) + 1
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
        if st.session_state.stations:
            st.session_state.stations.pop()

# LINEFILL ENTRY
with st.sidebar:
    st.markdown("#### Stepwise Linefill Segments")
    if "linefill_segments" not in st.session_state:
        st.session_state["linefill_segments"] = pd.DataFrame({
            "Start (km)": [0.0],
            "End (km)": [50.0],
            "Viscosity (cSt)": [5.0],
            "Density (kg/m³)": [850.0]
        })
    linefill_segments = st.data_editor(
        st.session_state["linefill_segments"], 
        num_rows="dynamic", key="linefill_segments_editor"
    )
    st.session_state["linefill_segments"] = linefill_segments


# STATION INPUTS
for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}: {stn['name']}", expanded=False):
        col1, col2, col3 = st.columns([1.5,1,1])
        with col1:
            stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
            stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{idx}")
            stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")
            stn['L'] = st.number_input("Length to next Station (km)", value=stn['L'], step=1.0, key=f"L{idx}")
            stn['max_dr'] = st.number_input("Max achievable Drag Reduction (%)", value=stn.get('max_dr', 0.0), key=f"mdr{idx}")
            if idx == 1:
                stn['min_residual'] = st.number_input("Available Suction Head (m)", value=stn.get('min_residual',50.0), step=0.1, key=f"res{idx}")
        with col2:
            D_in = st.number_input("OD (in)", value=stn['D']/0.0254, format="%.2f", step=0.01, key=f"D{idx}")
            t_in = st.number_input("Wall Thk (in)", value=stn['t']/0.0254, format="%.3f", step=0.001, key=f"t{idx}")
            stn['D'] = D_in * 0.0254
            stn['t'] = t_in * 0.0254
            stn['SMYS'] = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{idx}")
            stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=0.00001, key=f"rough{idx}")
        with col3:
            stn['max_pumps'] = st.number_input("Max Pumps available", min_value=1, value=stn.get('max_pumps',1), step=1, key=f"mpumps{idx}")
        tabs = st.tabs(["Pump", "Peaks"])
        with tabs[0]:
            if stn['is_pump']:
                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    stn['power_type'] = st.selectbox("Power Source", ["Grid", "Diesel"],
                                                    index=0 if stn['power_type']=="Grid" else 1, key=f"ptype{idx}")
                with pcol2:
                    stn['MinRPM'] = st.number_input("Min RPM", value=stn['MinRPM'], key=f"minrpm{idx}")
                    stn['DOL'] = st.number_input("Rated RPM", value=stn['DOL'], key=f"dol{idx}")
                with pcol3:
                    if stn['power_type']=="Grid":
                        stn['rate'] = st.number_input("Elec Rate (INR/kWh)", value=stn.get('rate',9.0), key=f"rate{idx}")
                        stn['sfc'] = 0.0
                    else:
                        stn['sfc'] = st.number_input("SFC (gm/bhp·hr)", value=stn.get('sfc',150.0), key=f"sfc{idx}")
                        stn['rate'] = 0.0
                st.markdown("**Pump Data at DOL:**")
                st.write("Flow vs Head data (m³/hr, m)")
                df_head = st.data_editor(st.session_state.get(f"head_data_{idx}", pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})), num_rows="dynamic", key=f"head{idx}")
                st.write("Flow vs Efficiency data (m³/hr, %)")
                df_eff = st.data_editor(st.session_state.get(f"eff_data_{idx}", pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})), num_rows="dynamic", key=f"eff{idx}")
                st.session_state[f"head_data_{idx}"] = df_head
                st.session_state[f"eff_data_{idx}"] = df_eff
            else:
                st.info("Not a pumping station. No pump data required.")
        with tabs[1]:
            st.markdown("Intermediate Elevation Peaks (to next station):")
            default_peak = st.session_state.get(f"peak_data_{idx}", pd.DataFrame({"Location (km)": [stn['L']/2.0], "Elevation (m)": [stn['elev']+100.0]}))
            peak_df = st.data_editor(default_peak, num_rows="dynamic", key=f"peak{idx}")
            st.session_state[f"peak_data_{idx}"] = peak_df

# TERMINAL INPUTS
st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value="Terminal", key="terminal_name")
terminal_elev = st.number_input("Elevation (m)", value=0.0, step=0.1, key="terminal_elev")
terminal_head = st.number_input("Minimum Residual Head (m)", value=50.0, step=1.0, key="terminal_head")


# SAVE CASE
def get_full_case_dict():
    return {
        "stations": st.session_state.stations,
        "terminal": {
            "name": st.session_state.terminal_name,
            "elev": st.session_state.terminal_elev,
            "min_residual": st.session_state.terminal_head
        },
        "FLOW": st.session_state.FLOW,
        "RateDRA": st.session_state.RateDRA,
        "Price_HSD": st.session_state.Price_HSD,
    }
case_data = get_full_case_dict()
st.sidebar.download_button(
    label="💾 Save Case",
    data=json.dumps(case_data, indent=2),
    file_name="pipeline_case.json",
    mime="application/json"
)

# OPTIMIZATION
def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD)
run = st.button("🚀 Run Optimization")
if run:
    with st.spinner("Solving optimization..."):
        stations_data = st.session_state.stations
        term_data = {"name": st.session_state.terminal_name, "elev": st.session_state.terminal_elev, "min_residual": st.session_state.terminal_head}
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
        linefill_df = st.session_state.linefill_df
        per_station_KV = linefill_df["Viscosity (cSt)"].tolist()
        per_station_rho = linefill_df["Density (kg/m³)"].tolist()
        res = solve_pipeline(stations_data, term_data, st.session_state.FLOW, per_station_KV, per_station_rho, st.session_state.RateDRA, st.session_state.Price_HSD)
        import copy
        st.session_state["last_res"] = copy.deepcopy(res)
        st.session_state["last_stations_data"] = copy.deepcopy(stations_data)
        st.session_state["last_term_data"] = copy.deepcopy(term_data)

# ========== TABS ==========
tabs = st.tabs([
    "📋 Summary", 
    "💰 Costs", 
    "⚙️ Performance", 
    "🌀 System Curves", 
    "🔄 Pump-System",
    "📉 DRA Curves",
    "🧊 3D Analysis",
    "📑 Optimization Report"
])

# === Tab 1: Summary Table ===
with tabs[0]:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        terminal_name = st.session_state["last_term_data"]["name"]
        names = [s['name'] for s in stations_data] + [terminal_name]
        params = [
            "Power+Fuel Cost", "DRA Cost", "No. of Pumps", "Pump Speed (rpm)",
            "Pump Eff (%)", "Head Loss (m)", "Vel (m/s)", "Residual Head (m)",
            "SDH (m)", "DRA (%)"
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
                res.get(f"head_loss_{key}",0.0),
                res.get(f"velocity_{key}",0.0),
                res.get(f"residual_head_{key}",0.0),
                res.get(f"sdh_{key}",0.0),
                res.get(f"drag_reduction_{key}",0.0)
            ]
        df_sum = pd.DataFrame(summary)
        st.session_state["df_sum"] = df_sum  # For PDF/report
        # Format
        fmt = {c: "{:.2f}" for c in df_sum.columns if c != "Parameters"}
        fmt["No. of Pumps"] = "{:.0f}"
        fmt["Pump Speed (rpm)"] = "{:.0f}"
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        styled = df_sum.style.format(fmt).set_properties(**{'text-align': 'left'})
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")
        # --- System-wide stats ---
        total_pumps = 0; effs = []; speeds = []
        for stn in stations_data:
            key = stn['name'].lower().replace(' ','_')
            npump = int(res.get(f"num_pumps_{key}", 0))
            if npump > 0:
                total_pumps += npump
                eff = float(res.get(f"efficiency_{key}", 0.0))
                speed = float(res.get(f"speed_{key}", 0.0))
                for _ in range(npump):
                    effs.append(eff)
                    speeds.append(speed)
        avg_eff = sum(effs)/len(effs) if effs else 0.0
        avg_speed = sum(speeds)/len(speeds) if speeds else 0.0
        total_cost = res.get('total_cost', 0)
        if isinstance(total_cost, str):
            total_cost = float(total_cost.replace(',', ''))
        st.markdown(
            f"""<br>
            <div style='font-size:1.1em;'><b>Total Optimized Cost:</b> {total_cost:.2f} INR/day<br>
            <b>No. of operating Pumps:</b> {total_pumps}<br>
            <b>Average Pump Efficiency:</b> {avg_eff:.2f} %<br>
            <b>Average Pump Speed:</b> {avg_speed:.0f} rpm</div>
            """,
            unsafe_allow_html=True
        )

# === Tab 2: Costs ===
with tabs[1]:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        df_cost = pd.DataFrame({
            "Station": [s['name'] for s in stations_data],
            "Power+Fuel": [res.get(f"power_cost_{s['name'].lower().replace(' ','_')}",0) for s in stations_data],
            "DRA": [res.get(f"dra_cost_{s['name'].lower().replace(' ','_')}", 0.0) for s in stations_data]
        })
        df_cost['Total'] = df_cost['Power+Fuel'] + df_cost['DRA']
        fig_pie = px.pie(df_cost, names='Station', values='Total', title="Station-wise Cost Breakdown")
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.download_button("Download CSV", df_cost.to_csv(index=False).encode(), file_name="cost_breakdown.csv")
        st.session_state["fig_pie"] = fig_pie

# === Tab 3: Performance ===
with tabs[2]:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        df_hloss = pd.DataFrame({
            "Station": [s['name'] for s in stations_data],
            "Head Loss": [res.get(f"head_loss_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
        })
        fig_h = go.Figure(go.Bar(x=df_hloss["Station"], y=df_hloss["Head Loss"]))
        fig_h.update_layout(yaxis_title="Head Loss (m)")
        st.plotly_chart(fig_h, use_container_width=True)
        st.session_state["fig_h"] = fig_h

# === Tab 4: System Curves ===
with tabs[3]:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']; L_seg = stn['L']; elev_i = stn['elev']
            max_dr = int(stn.get('max_dr', 40))
            curves = []
            for dra in range(0, max_dr+1, 5):
                flows = np.linspace(0, st.session_state.get("FLOW",1000.0), 101)
                v_vals = flows/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (stn.get('KV',10.0)*1e-6) if stn.get('KV',10.0)>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                curves.append(pd.DataFrame({"Flow": flows, "SDH": SDH_vals, "DRA": dra}))
            df_sys = pd.concat(curves)
            fig_sys = px.line(df_sys, x="Flow", y="SDH", color="DRA", title=f"System Head ({stn['name']}) at various % DRA")
            fig_sys.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig_sys, use_container_width=True, key=f"sys_curve_{i}_{key}_{uuid.uuid4().hex[:6]}")
            st.session_state[f"fig_sys_{i}_{key}"] = fig_sys

# === Tab 5: Pump-System Interaction ===
with tabs[4]:
    if "last_res" not in st.session_state:
        st.info("Please run optimization.")
    else:
        res = st.session_state["last_res"]
        stations_data = st.session_state["last_stations_data"]
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): continue
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
                Re_vals = v_vals * d_inner_i / (stn.get('KV',10.0)*1e-6) if stn.get('KV',10.0)>0 else np.zeros_like(v_vals)
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
            st.session_state[f"fig_int_{i}_{key}"] = fig_int

# === Tab 6: DRA Curves ===
with tabs[5]:
    if "last_res" not in st.session_state or "last_stations_data" not in st.session_state:
        st.info("Please run optimization first to analyze DRA curves.")
        st.stop()
    res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]

    st.markdown("<div class='section-title'>DRA Curve (PPM vs %Drag Reduction) for Each Station</div>", unsafe_allow_html=True)

    for idx, stn in enumerate(stations_data, start=1):
        key = stn['name'].lower().replace(' ', '_')
        dr_opt = res.get(f"drag_reduction_{key}", 0.0)
        if dr_opt > 0:
            viscosity = st.session_state['linefill_df'].iloc[idx-1]["Viscosity (cSt)"]
            cst_list = sorted(DRA_CURVE_DATA.keys())
            # Use interpolation logic as in get_ppm_for_dr
            if viscosity <= cst_list[0]:
                df_curve = DRA_CURVE_DATA[cst_list[0]]
                curve_label = f"{cst_list[0]} cSt curve"
                percent_dr = df_curve['%Drag Reduction'].values
                ppm_vals = df_curve['PPM'].values
            elif viscosity >= cst_list[-1]:
                df_curve = DRA_CURVE_DATA[cst_list[-1]]
                curve_label = f"{cst_list[-1]} cSt curve"
                percent_dr = df_curve['%Drag Reduction'].values
                ppm_vals = df_curve['PPM'].values
            else:
                lower = max([c for c in cst_list if c <= viscosity])
                upper = min([c for c in cst_list if c >= viscosity])
                df_lower = DRA_CURVE_DATA[lower]
                df_upper = DRA_CURVE_DATA[upper]
                percent_dr = np.linspace(
                    min(df_lower['%Drag Reduction'].min(), df_upper['%Drag Reduction'].min()),
                    max(df_lower['%Drag Reduction'].max(), df_upper['%Drag Reduction'].max()),
                    50
                )
                ppm_lower = np.interp(percent_dr, df_lower['%Drag Reduction'], df_lower['PPM'])
                ppm_upper = np.interp(percent_dr, df_upper['%Drag Reduction'], df_upper['PPM'])
                ppm_vals = np.interp(viscosity, [lower, upper], np.vstack([ppm_lower, ppm_upper]))
                curve_label = f"Interpolated for {viscosity:.2f} cSt"
            # Optimum point
            opt_ppm = get_ppm_for_dr(viscosity, dr_opt)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=percent_dr,
                y=ppm_vals,
                mode='lines+markers',
                name=curve_label
            ))
            fig.add_trace(go.Scatter(
                x=[dr_opt], y=[opt_ppm],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond'),
                name="Optimized Point"
            ))
            fig.update_layout(
                title=f"DRA Curve for {stn['name']} (Viscosity: {viscosity:.2f} cSt)",
                xaxis_title="% Drag Reduction",
                yaxis_title="PPM",
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No DRA applied at {stn['name']} (Optimal %DR = 0)")

# === Tab 7: 3D Analysis ===
with tabs[6]:
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
        Re = v*d_inner/(stn.get('KV',10.0)*1e-6) if stn.get('KV',10.0) > 0 else 0
        if Re > 0:
            f = 0.25/(np.log10(rough/d_inner/3.7 + 5.74/(Re**0.9))**2)
        else:
            f = 0.0
        DH = f*((L_seg*1000.0)/d_inner)*(v**2/(2*g))*(1-d/100)
        return stn['elev'] + DH
    def get_total_cost(q, n, d, npump):
        local_ppm = get_ppm_for_dr(stn.get('KV', 10.0), d)
        pcost = get_power_cost(q, n, d, npump)
        dracost = local_ppm * (q * 1000.0 * 24.0 / 1e6) * RateDRA
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
    st.session_state["fig_3d"] = fig

# === Tab 8: PDF Report Generation ===
with tabs[7]:
    st.markdown("<div class='section-title'>Optimization Report Generator</div>", unsafe_allow_html=True)
    if "last_res" not in st.session_state:
        st.info("Please run optimization before generating the report.")
        st.stop()
    if st.button("Generate & Download Optimization Report (PDF)", type="primary"):
        # --------- PDF GENERATION --------------
        with st.spinner("Building PDF Report..."):
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(33, 39, 51)
            pdf.cell(0, 10, "Pipeline Optima Optimization Report", ln=True, align="C")
            pdf.set_font("Arial", '', 12)
            pdf.ln(4)
            pdf.cell(0, 10, "This report summarizes the complete optimization results, all user inputs, cost breakdown, performance, and visualization.", ln=True, align="C")
            pdf.ln(10)
            # --- Inputs Table
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Input Parameters", ln=True)
            pdf.set_font("Arial", '', 10)
            inputs = [
                f"Flow Rate: {st.session_state.get('FLOW', 0):.2f} m³/hr",
                f"DRA Cost: {st.session_state.get('RateDRA', 0):.2f} INR/L",
                f"Diesel Price: {st.session_state.get('Price_HSD', 0):.2f} INR/L",
                f"Terminal: {st.session_state.get('terminal_name','Terminal')} | Elev: {st.session_state.get('terminal_elev',0)} | Head: {st.session_state.get('terminal_head',0)}"
            ]
            for inp in inputs:
                pdf.cell(0, 7, inp, ln=True)
            pdf.ln(4)
            pdf.cell(0, 7, "Linefill (Viscosity & Density):", ln=True)
            linefill_df = st.session_state['linefill_df']
            pdf.set_font("Arial", '', 9)
            for i, row in linefill_df.iterrows():
                pdf.cell(0, 6, f"Segment {int(row['Segment'])}: KV={row['Viscosity (cSt)']} | Rho={row['Density (kg/m³)']}", ln=True)
            pdf.ln(5)
            # --- Summary Table
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Optimization Summary Table", ln=True)
            pdf.set_font("Arial", '', 9)
            df_sum = st.session_state["df_sum"]
            pdf.set_fill_color(220, 230, 241)
            pdf.cell(38, 8, "Parameter", 1, 0, 'C', 1)
            for col in df_sum.columns[1:]:
                pdf.cell(38, 8, str(col), 1, 0, 'C', 1)
            pdf.ln()
            for idx, row in df_sum.iterrows():
                pdf.set_fill_color(245, 245, 245)
                pdf.cell(38, 8, str(row["Parameters"]), 1, 0, 'C', 1)
                for col in df_sum.columns[1:]:
                    val = row[col]
                    fmt = "{:.2f}" if isinstance(val, float) else "{}"
                    pdf.cell(38, 8, fmt.format(val), 1, 0, 'C', 1)
                pdf.ln()
            pdf.ln(6)
            # --- Pie/Bar/3D Plots
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Key Visualizations", ln=True)
            pdf.ln(4)
            # --- Helper: save Plotly as PNG to temp and insert
            import plotly.io as pio
            figs_to_save = [
                ("Cost Breakdown", st.session_state.get("fig_pie")),
                ("Head Loss", st.session_state.get("fig_h")),
                ("3D Analysis", st.session_state.get("fig_3d"))
            ]
            for desc, fig in figs_to_save:
                if fig:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        pio.write_image(fig, tmp.name, width=850, height=420, scale=2)
                        pdf.image(tmp.name, w=250)
                        pdf.ln(2)
            # --- Final Writeup
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Summary Writeup", ln=True)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 8, 
                "This report was automatically generated by Pipeline Optima™. "
                "All results, summary tables, and engineering curves reflect the "
                "current project input and optimized operation. Use this document "
                "for technical discussions, management reviews, or project documentation."
            )
            # --- PDF Download
            pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.output(pdf_file.name)
            pdf_file.seek(0)
            st.success("PDF report generated! Download below:")
            st.download_button(
                "Download Report (PDF)",
                pdf_file.read(),
                file_name="Pipeline_Optima_Optimization_Report.pdf",
                mime="application/pdf"
            )


# --------------- FOOTER ---------------
st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v1.2. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
