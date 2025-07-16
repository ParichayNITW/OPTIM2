import streamlit as st
import pandas as pd
import pipeline_model

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

# --- Tab Bar ---
tab_labels = [
    "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves", "🛢️ Pump-System",
    "🧪 DRA Curves", "📊 3D Analysis and Surface Plots", "🧩 3D Pressure Profile",
    "📈 Sensitivity", "📚 Benchmarking", "🏆 Savings Simulation"
]
tabs = st.tabs(tab_labels)

# --- Session state for result caching ---
if "last_output" not in st.session_state:
    st.session_state["last_output"] = None

# ----------- User Inputs Section -----------
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
    return stations

def load_dra_curves(dra_files):
    dra_curve_dict = {}
    for file in dra_files:
        try:
            vis = float(''.join(filter(str.isdigit, file.name)))
            df = pd.read_csv(file)
            dra_curve_dict[vis] = df
        except Exception:
            continue
    return dra_curve_dict

ready = stations_file and dra_files

if ready and run_btn:
    stations = load_station_data(stations_file)
    dra_curve_dict = load_dra_curves(dra_files)
    N = len(stations)
    kv_list = []
    rho_list = []
    for idx in range(N):
        kv_list.append(stations[idx].get('kv', 10.0))
        rho_list.append(stations[idx].get('rho', 850.0))
    terminal = {"elev": stations[-1].get('elev', 0.0), "name": "Terminal"}
    try:
        with st.spinner("Running optimization..."):
            results = pipeline_model.optimize_pipeline(
                stations, terminal, FLOW, kv_list, rho_list, RateDRA, Price_HSD, dra_curve_dict,
                rpm_step=int(rpm_step), dra_step=int(dra_step)
            )
        output = results
        st.session_state["last_output"] = output
        st.success("Optimization completed.")
    except Exception as e:
        st.session_state["last_output"] = None
        st.error(f"Backend error: {e}")

elif st.session_state["last_output"]:
    output = st.session_state["last_output"]
else:
    for tab in tabs:
        with tab:
            st.info("Please run optimization.")
    st.stop()

# -------- TABS CONTENT --------

# --- Summary Tab ---
with tabs[0]:
    st.markdown("### Optimization Summary")
    summary_df = pd.DataFrame(output['summary_table'])
    st.dataframe(summary_df, use_container_width=True)
    st.markdown(f"#### <span style='color:#006400'>Total Optimized Cost: ₹{output['total_cost']:,.2f}</span>", unsafe_allow_html=True)

# --- Costs Tab ---
with tabs[1]:
    st.markdown("### Cost Breakdown (All Stations)")
    summary_df = pd.DataFrame(output['summary_table'])
    cost_cols = ["Station","DRA_Cost","Power_Cost","Total_Cost"]
    st.dataframe(summary_df[cost_cols], use_container_width=True)
    st.bar_chart(summary_df.set_index("Station")[["Total_Cost"]])

# --- Performance Tab ---
with tabs[2]:
    st.markdown("### Pump Efficiency & Hydraulic Performance")
    perf_cols = ["Station","NOP","RPM","Eff (%)","Head_Required (m)","Head_Generated (m)"]
    st.dataframe(summary_df[perf_cols], use_container_width=True)
    st.line_chart(summary_df.set_index("Station")[["Eff (%)","Head_Required (m)","Head_Generated (m)"]])

# --- System Curves Tab ---
with tabs[3]:
    st.markdown("### System Curves (Upload pump/system data for custom plotting)")
    st.info("Feature: Overlay all system curves for selected DRA%, bypass scenarios, etc. (Custom plotting to be integrated as needed.)")

# --- Pump-System Tab ---
with tabs[4]:
    st.markdown("### Pump-System Curve Interactions")
    st.info("Feature: Overlay pump curves and system curves at various speeds/DRA%. (To be customized with your data/plots.)")

# --- DRA Curves Tab ---
with tabs[5]:
    st.markdown("### DRA Curves")
    st.info("Feature: DRA PPM vs Drag Reduction curve for each viscosity (custom plotting on demand).")

# --- 3D Analysis/Surface Plots Tab ---
with tabs[6]:
    st.markdown("### 3D Analysis & Surface Plots")
    st.info("Feature: 3D plots of cost, efficiency, DRA usage over (NOP, RPM, DRA%) grid. (To be integrated.)")

# --- 3D Pressure Profile Tab ---
with tabs[7]:
    st.markdown("### 3D Pressure Profile")
    st.info("Feature: Sawtooth pressure profile and spatial station-wise plot (to be integrated).")

# --- Sensitivity Tab ---
with tabs[8]:
    st.markdown("### Sensitivity Analysis")
    st.info("Feature: Sensitivity plots and tornado charts for all parameters (planned).")

# --- Benchmarking Tab ---
with tabs[9]:
    st.markdown("### Benchmarking")
    st.info("Feature: Compare results against other solutions/scenarios.")

# --- Savings Simulation Tab ---
with tabs[10]:
    st.markdown("### Savings Simulation")
    st.info("Feature: Simulate annual savings and present business case vs base case.")

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
