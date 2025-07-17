import streamlit as st
import pandas as pd
import numpy as np
import pipeline_model
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Pipeline Optima", layout="wide")
st.markdown("""
<style>
.big-red {font-size:2em; color:#e74c3c; font-weight:bold;}
.stTabs [data-baseweb="tab"] {font-size: 1.05em;}
thead tr th {background:#f7fafd;}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-red'>Batch Linefill Scenario Analysis</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1,4])
with col1:
    auto_batch = st.checkbox("Run Auto Linefill Generator (Batch Interface Scenarios)")
with col2:
    st.write("")

st.markdown("<br>", unsafe_allow_html=True)
run_col = st.container()
with run_col:
    run_btn = st.button("🚨 Run Optimization", key="runopt", use_container_width=True)

tab_labels = [
    "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves", "🛢️ Pump-System",
    "🧪 DRA Curves", "📊 3D Analysis and Surface Plots", "🧩 3D Pressure Profile",
    "📈 Sensitivity", "📚 Benchmarking", "🏆 Savings Simulation"
]
tabs = st.tabs(tab_labels)

if "last_output" not in st.session_state:
    st.session_state["last_output"] = None
if "last_data" not in st.session_state:
    st.session_state["last_data"] = {}

with st.sidebar:
    st.header("Pipeline Settings")
    FLOW = st.number_input("Pipeline Flow (m³/hr)", min_value=1.0, value=5000.0, step=100.0)
    RateDRA = st.number_input("DRA Cost per kg (INR)", min_value=0.0, value=300.0, step=1.0)
    Price_HSD = st.number_input("Fuel Price (INR/L)", min_value=0.0, value=100.0, step=1.0)
    rpm_step = st.selectbox("RPM Step", [10, 25, 50, 100], index=3)
    dra_step = st.selectbox("DRA% Step", [1, 2, 5, 10], index=0)
    st.header("Upload Data")
    stations_file = st.file_uploader("Stations CSV", type=['csv'])
    dra_files = st.file_uploader("DRA Curve CSVs (multiple files)", accept_multiple_files=True, type=['csv'])

def load_station_data(csv_file):
    df = pd.read_csv(csv_file)
    stations = df.to_dict(orient="records")
    return stations, df

def load_dra_curves(dra_files):
    dra_curve_dict = {}
    dra_curve_table = {}
    for file in dra_files:
        try:
            vis = float(''.join(filter(str.isdigit, file.name)))
            df = pd.read_csv(file)
            dra_curve_dict[vis] = df
            dra_curve_table[vis] = df
        except Exception:
            continue
    return dra_curve_dict, dra_curve_table

ready = stations_file and dra_files

if ready and run_btn:
    stations, stations_df = load_station_data(stations_file)
    dra_curve_dict, dra_curve_table = load_dra_curves(dra_files)
    N = len(stations)
    kv_list = []
    rho_list = []
    for idx in range(N):
        kv_list.append(stations[idx].get('kv', 10.0))
        rho_list.append(stations[idx].get('rho', 850.0))
    terminal = {"elev": stations[-1].get('elev', 0.0), "name": "Terminal"}
    try:
        with st.spinner("Running optimization..."):
            output = pipeline_model.optimize_pipeline(
                stations, terminal, FLOW, kv_list, rho_list, RateDRA, Price_HSD, dra_curve_dict,
                rpm_step=int(rpm_step), dra_step=int(dra_step)
            )
        st.session_state["last_output"] = output
        st.session_state["last_data"] = {
            "stations": stations,
            "stations_df": stations_df,
            "dra_curve_dict": dra_curve_dict,
            "dra_curve_table": dra_curve_table
        }
        st.success("Optimization completed.")
    except Exception as e:
        st.session_state["last_output"] = None
        st.error(f"Backend error: {e}")

elif st.session_state["last_output"]:
    output = st.session_state["last_output"]
    stations = st.session_state["last_data"]["stations"]
    stations_df = st.session_state["last_data"]["stations_df"]
    dra_curve_dict = st.session_state["last_data"]["dra_curve_dict"]
    dra_curve_table = st.session_state["last_data"]["dra_curve_table"]
else:
    for tab in tabs:
        with tab:
            st.info("Please run optimization.")
    st.stop()

# --- SUMMARY TAB ---
with tabs[0]:
    st.markdown("### Optimization Summary")
    summary_df = pd.DataFrame(output['summary_table'])
    st.dataframe(summary_df, use_container_width=True)
    st.markdown(f"#### <span style='color:#006400'>Total Optimized Cost: ₹{output['total_cost']:,.2f}</span>", unsafe_allow_html=True)

# --- COSTS TAB ---
with tabs[1]:
    st.markdown("### Cost Breakdown (All Stations)")
    summary_df = pd.DataFrame(output['summary_table'])
    cost_cols = ["Station","DRA Cost","Power Cost","Total Cost"]
    st.dataframe(summary_df[cost_cols], use_container_width=True)
    st.bar_chart(summary_df.set_index("Station")[["Total Cost"]])
    st.line_chart(summary_df.set_index("Station")[["DRA Cost","Power Cost"]])
    st.markdown("#### Cumulative Cost Pie")
    fig = px.pie(summary_df, values="Total Cost", names="Station", title="Share of Total Cost by Station")
    st.plotly_chart(fig, use_container_width=True)

# --- PERFORMANCE TAB ---
with tabs[2]:
    st.markdown("### Hydraulic & Pump Performance")
    perf_cols = ["Station","NOP","Eff (%)","Head (m)","SDH (m)"]
    st.dataframe(summary_df[perf_cols], use_container_width=True)
    st.line_chart(summary_df.set_index("Station")[["Eff (%)","SDH (m)","Head (m)"]])
    eff_df = summary_df[["Station", "Eff (%)"]]
    st.markdown("#### Efficiency by Station")
    fig = px.bar(eff_df, x="Station", y="Eff (%)", title="Pump Efficiency per Station")
    st.plotly_chart(fig, use_container_width=True)

# --- SYSTEM CURVES TAB ---
with tabs[3]:
    st.markdown("### System Curves")
    st.info("Below: System head for each station at each DRA% (best config highlighted).")
    for i, stn in enumerate(stations):
        name = stn['Station']
        df = output['station_tables'][name]
        if "%DR" in df.columns and "SDH" in df.columns:
            pivot = df.pivot(index="%DR", columns="NOP", values="SDH")
            fig = px.line(pivot, y=pivot.columns, x=pivot.index,
                labels={"value":"SDH (m)","%DR":"DRA%"},
                title=f"System Curve - {name} (varied NOP at each DRA%)")
            st.plotly_chart(fig, use_container_width=True)

# --- PUMP-SYSTEM TAB ---
with tabs[4]:
    st.markdown("### Pump-System Curve Overlays")
    for i, stn in enumerate(stations):
        name = stn['Station']
        df = output['station_tables'][name]
        if "%DR" in df.columns and "Head" in df.columns:
            fig = px.scatter(df, x="SDH", y="Head", color="NOP",
                labels={"SDH":"System Head (m)","Head":"Pump Head (m)"},
                title=f"Pump vs System Head - {name}")
            st.plotly_chart(fig, use_container_width=True)

# --- DRA CURVES TAB ---
with tabs[5]:
    st.markdown("### DRA Curves")
    for vis, df in dra_curve_table.items():
        fig = px.line(df, x="%Drag Reduction", y="PPM", title=f"DRA Curve for {vis} cSt")
        st.plotly_chart(fig, use_container_width=True)

# --- 3D ANALYSIS TAB ---
with tabs[6]:
    st.markdown("### 3D Cost & Performance Surface Plots")
    for i, stn in enumerate(stations):
        name = stn['Station']
        df = output['station_tables'][name]
        if "NOP" in df.columns and "%DR" in df.columns and "Total Cost" in df.columns:
            fig = px.density_heatmap(df, x="NOP", y="%DR", z="Total Cost", nbinsx=20, nbinsy=20, title=f"Cost Surface: {name}")
            st.plotly_chart(fig, use_container_width=True)
        if "NOP" in df.columns and "Eff" in df.columns and "%DR" in df.columns:
            fig2 = px.scatter_3d(df, x="NOP", y="%DR", z="Eff", color="Total Cost", title=f"Efficiency 3D: {name}")
            st.plotly_chart(fig2, use_container_width=True)

# --- 3D PRESSURE PROFILE TAB ---
with tabs[7]:
    st.markdown("### 3D Pressure/Head Profile")
    x = [row['Station'] for row in output['summary_table']]
    y = [row['SDH (m)'] for row in output['summary_table']]
    z = [row['Head (m)'] for row in output['summary_table']]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', name='Pressure Profile'))
    fig.update_layout(scene=dict(
        xaxis_title='Station',
        yaxis_title='SDH (m)',
        zaxis_title='Head (m)'
    ))
    st.plotly_chart(fig, use_container_width=True)

# --- SENSITIVITY TAB ---
with tabs[8]:
    st.markdown("### Sensitivity Analysis")
    st.info("Sensitivity with respect to DRA cost, Fuel price, and Flow rate.")
    base_cost = output['total_cost']
    for param, pct in [("DRA Cost per kg (INR)", RateDRA), ("Fuel Price (INR/L)", Price_HSD), ("Pipeline Flow (m³/hr)", FLOW)]:
        deltas = np.linspace(-0.3, 0.3, 7)
        costs = []
        for delta in deltas:
            if param == "DRA Cost per kg (INR)":
                val = RateDRA * (1+delta)
                results = pipeline_model.optimize_pipeline(
                    stations, FLOW, val, Price_HSD, dra_curve_dict
                )
            elif param == "Fuel Price (INR/L)":
                val = Price_HSD * (1+delta)
                results = pipeline_model.optimize_pipeline(
                    stations, FLOW, RateDRA, val, dra_curve_dict
                )
            else:
                val = FLOW * (1+delta)
                results = pipeline_model.optimize_pipeline(
                    stations, val, RateDRA, Price_HSD, dra_curve_dict
                )
            costs.append(results['total_cost'])
        fig = px.line(x=(100*deltas+100), y=costs, title=f"Sensitivity: {param}", labels={"x":param,"y":"Total Cost"})
        st.plotly_chart(fig, use_container_width=True)

# --- BENCHMARKING TAB ---
with tabs[9]:
    st.markdown("### Benchmarking")
    base_df = pd.DataFrame(output['summary_table'])
    st.info("You can upload a base case CSV for benchmarking if desired.")
    base_file = st.file_uploader("Upload Base Case (summary_table CSV)", type=["csv"], key="bench_csv")
    if base_file:
        base_case = pd.read_csv(base_file)
        merged = base_df.set_index("Station")[["Total Cost"]].join(base_case.set_index("Station")[["Total Cost"]], rsuffix="_base")
        merged['Savings'] = merged["Total Cost_base"] - merged["Total Cost"]
        st.dataframe(merged)
        st.bar_chart(merged[["Total Cost_base","Total Cost"]])

# --- SAVINGS SIMULATION TAB ---
with tabs[10]:
    st.markdown("### Savings Simulation")
    yearly_savings = 0.0
    base_case_file = st.file_uploader("Upload Base Case (summary_table CSV)", type=["csv"], key="savings_base_csv")
    if base_case_file:
        base_case = pd.read_csv(base_case_file)
        merged = pd.DataFrame(output['summary_table']).set_index("Station").join(
            base_case.set_index("Station"), rsuffix="_base"
        )
        merged['Savings'] = merged["Total Cost_base"] - merged["Total Cost"]
        yearly_savings = merged['Savings'].sum() * 365
        st.metric("Estimated Yearly Savings (INR)", f"{yearly_savings:,.0f}")
        st.bar_chart(merged[["Total Cost_base","Total Cost","Savings"]])

# --- Station-wise Export Tabs ---
st.markdown("### Station-wise Detailed Results")
station_tabs = st.tabs([f"{row['Station']}" for row in output['summary_table']])
for idx, (tab, row) in enumerate(zip(station_tabs, output['summary_table'])):
    name = row['Station']
    with tab:
        df = output['station_tables'].get(name, pd.DataFrame())
        st.dataframe(df, use_container_width=True)
        st.download_button(
            f"Download Results for {name}",
            output['csv_tables'].get(name, ""),
            file_name=f"{name}_results.csv",
            mime="text/csv"
        )
        if df.shape[1] == 1 and "No feasible configs" in df.columns[0]:
            st.warning(f"No feasible configs for {name}. See raw_results for rejection reasons.")

with st.expander("Show Raw Results (All Configs Tried, All Stations)"):
    st.write(output['raw_results'])

st.markdown("""
<hr style="margin-top:2em; margin-bottom:1em; border: 1px solid #aaa;">
<p style="text-align:right;font-size:0.9em;color:#888;">
&copy; 2025 Pipeline Optima | Built on world-class graphical optimization engine.
</p>
""", unsafe_allow_html=True)
