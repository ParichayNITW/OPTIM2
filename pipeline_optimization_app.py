import streamlit as st
import pandas as pd
import numpy as np
import json
from pipeline_model import solve_pipeline

st.set_page_config(page_title="Pipeline Optima", layout="wide")

# ---- LOGIN LOGIC ----
def login_widget():
    st.title("Pipeline Optima Login")
    userid = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if userid == "parichay_das" and password == "heteroscedasticity":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid UserID or Password")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    login_widget()
    st.stop()

# ---- STATION TABLE AND DYNAMIC PUMP CURVE TABLES ----
station_cols = [
    "Station","Distance (km)","Elevation (m)","Diameter (mm)","Roughness (mm)",
    "Is Pump Station","Pump Count","Head Limit (m)","Max Power (kW)","Fuel Rate (Rs/kWh)","MAOP (m)","Peaks"
]
station_defaults = [
    {"Station":"STN1","Distance (km)":0,"Elevation (m)":100,"Diameter (mm)":500,"Roughness (mm)":0.045,
     "Is Pump Station":True,"Pump Count":2,"Head Limit (m)":1500,"Max Power (kW)":3000,"Fuel Rate (Rs/kWh)":12.5,"MAOP (m)":900,"Peaks":[]},
    {"Station":"STN2","Distance (km)":50,"Elevation (m)":105,"Diameter (mm)":500,"Roughness (mm)":0.045,
     "Is Pump Station":False,"Pump Count":"","Head Limit (m)":"","Max Power (kW)":"","Fuel Rate (Rs/kWh)":"","MAOP (m)":900,"Peaks":[]},
]

def ensure_station_df(val=None):
    if val is None:
        df = pd.DataFrame(station_defaults)
    elif isinstance(val, list):
        df = pd.DataFrame(val)
    elif isinstance(val, pd.DataFrame):
        df = val
    else:
        df = pd.DataFrame(station_defaults)
    # fill any missing columns
    for col in station_cols:
        if col not in df.columns:
            df[col] = "" if "Pump" in col or "Head" in col or "Max" in col or "Fuel" in col else np.nan
    return df[station_cols]

def get_pump_curve_table(stn, curve_type, default):
    key = f"{stn}_{curve_type}"
    data = st.session_state.get(key, None)
    if data is None:
        df = pd.DataFrame(default)
        st.session_state[key] = df
    else:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        key=key
    )
    st.session_state[key] = edited
    return edited

def reset_all_pump_curves(station_df):
    for idx, row in station_df.iterrows():
        if bool(row.get("Is Pump Station", False)):
            stn = row["Station"]
            st.session_state[f"{stn}_qh"] = pd.DataFrame([{"Flow (m3/h)":100,"Head (m)":1000}])
            st.session_state[f"{stn}_qeff"] = pd.DataFrame([{"Flow (m3/h)":100,"Efficiency (%)":62}])

def clear_unused_pump_curves(valid_stations):
    all_keys = list(st.session_state.keys())
    for key in all_keys:
        if any(key.endswith(suffix) for suffix in ["_qh", "_qeff"]):
            stn = key[:-3] if key.endswith("_qh") else key[:-5]
            if stn not in valid_stations:
                del st.session_state[key]

# ---- INPUT TAB: ALL USER INPUTS ----
tabs = st.tabs(["Input", "Optimization", "Visualization", "Cost Breakdown", "Download Report"])

with tabs[0]:
    st.header("Stations (last row = terminal station)")
    val = st.session_state.get("station_table", station_defaults)
    df = ensure_station_df(val)
    edited_df = st.data_editor(df, num_rows="dynamic", key="station_table", column_order=station_cols)
    st.session_state["station_table"] = edited_df

    # Clean up per-station pump curves for deleted stations
    current_stations = set(edited_df["Station"].astype(str))
    clear_unused_pump_curves(current_stations)

    st.write("If you add/remove stations, click below to reset all pump curve tables (prevents widget errors).")
    if st.button("Reset Pump Curves"):
        reset_all_pump_curves(edited_df)
        st.success("All pump curve tables reset.")

    st.subheader("Pump Curves (per pump station)")
    for idx, row in edited_df.iterrows():
        stn = row["Station"]
        if bool(row.get("Is Pump Station", False)):
            st.markdown(f"**{stn}:**")
            # Q-H curve
            qh_default = [{"Flow (m3/h)":100,"Head (m)":1000},{"Flow (m3/h)":200,"Head (m)":900},{"Flow (m3/h)":300,"Head (m)":750}]
            qh_table = get_pump_curve_table(stn, "qh", qh_default)
            # Q-Eff curve
            qeff_default = [{"Flow (m3/h)":100,"Efficiency (%)":62},{"Flow (m3/h)":200,"Efficiency (%)":70},{"Flow (m3/h)":300,"Efficiency (%)":68}]
            qeff_table = get_pump_curve_table(stn, "qeff", qeff_default)

    # ---- SCENARIO SAVE/LOAD ----
    def get_scenario_json():
        scenario = {
            "station_table": edited_df.to_dict(orient="records"),
            "pump_curves": {}
        }
        for idx, row in edited_df.iterrows():
            stn = row["Station"]
            scenario["pump_curves"][stn] = {
                "qh": st.session_state.get(f"{stn}_qh", pd.DataFrame()).to_dict(orient="records"),
                "qeff": st.session_state.get(f"{stn}_qeff", pd.DataFrame()).to_dict(orient="records"),
            }
        return json.dumps(scenario, indent=2)

    st.download_button("Download Scenario as JSON", get_scenario_json(), file_name="pipeline_scenario.json", mime="application/json")
    uploaded = st.file_uploader("Upload Scenario (JSON)", type="json")
    if uploaded:
        data = json.load(uploaded)
        st.session_state["station_table"] = data["station_table"]
        # Restore all pump curve tables
        for stn, curves in data["pump_curves"].items():
            st.session_state[f"{stn}_qh"] = pd.DataFrame(curves.get("qh", []))
            st.session_state[f"{stn}_qeff"] = pd.DataFrame(curves.get("qeff", []))
        st.experimental_rerun()

# ---- OPTIMIZATION TAB ----
with tabs[1]:
    st.header("Pipeline Optima™ Optimization")
    flow_rate = st.number_input("Enter flow rate (m³/h)", min_value=1, value=200, step=10)
    viscosity = st.number_input("Enter kinematic viscosity (cSt)", min_value=1.0, value=20.0, step=0.1)
    if st.button("Run Optimization"):
        with st.spinner("Optimizing..."):
            # Build per-station pump curve dicts
            pump_curves = {}
            for idx, row in edited_df.iterrows():
                stn = row["Station"]
                if bool(row.get("Is Pump Station", False)):
                    pump_curves[stn] = {
                        "qh": st.session_state.get(f"{stn}_qh", pd.DataFrame()).to_dict(orient="records"),
                        "qeff": st.session_state.get(f"{stn}_qeff", pd.DataFrame()).to_dict(orient="records"),
                    }
            # Run backend
            result = solve_pipeline(
                edited_df.to_dict(orient="records"),
                pump_curves,
                flow_rate, viscosity
            )
            st.session_state["opt_result"] = result
            st.session_state["opt_summary"] = pd.DataFrame(result["summary_table"])
            st.session_state["opt_total_cost"] = result["total_cost"]
            st.success("Optimization complete")
            st.dataframe(st.session_state["opt_summary"], use_container_width=True)
            st.markdown(f"Total Operating Cost: ₹ {result['total_cost']:,.2f}")

# ---- VISUALIZATION TAB ----
with tabs[2]:
    st.header("Visualization")
    for idx, row in edited_df.iterrows():
        stn = row["Station"]
        if bool(row.get("Is Pump Station", False)):
            st.subheader(f"Pump Curves - {stn}")
            qh_df = st.session_state.get(f"{stn}_qh", pd.DataFrame())
            qeff_df = st.session_state.get(f"{stn}_qeff", pd.DataFrame())
            if isinstance(qh_df, pd.DataFrame) and not qh_df.empty:
                import matplotlib.pyplot as plt
                q_vals = np.linspace(min(qh_df["Flow (m3/h)"]), max(qh_df["Flow (m3/h)"]), 100)
                H = np.interp(q_vals, qh_df["Flow (m3/h)"], qh_df["Head (m)"])
                plt.figure(figsize=(7,3))
                plt.plot(q_vals, H, label=f"{stn} Head Curve")
                plt.xlabel("Flow (m³/h)")
                plt.ylabel("Head (m)")
                plt.title(f"{stn} Q vs Head")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf())
            if isinstance(qeff_df, pd.DataFrame) and not qeff_df.empty:
                q_vals = np.linspace(min(qeff_df["Flow (m3/h)"]), max(qeff_df["Flow (m3/h)"]), 100)
                Eff = np.interp(q_vals, qeff_df["Flow (m3/h)"], qeff_df["Efficiency (%)"])
                plt.figure(figsize=(7,3))
                plt.plot(q_vals, Eff, label=f"{stn} Efficiency Curve", color="green")
                plt.xlabel("Flow (m³/h)")
                plt.ylabel("Efficiency (%)")
                plt.title(f"{stn} Q vs Efficiency")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf())

# ---- COST BREAKDOWN TAB ----
with tabs[3]:
    st.header("Cost Breakdown")
    if "opt_summary" in st.session_state:
        df = st.session_state["opt_summary"]
        if "DRA Cost" in df.columns and "Power Cost" in df.columns and "Total Cost" in df.columns:
            import plotly.express as px
            st.dataframe(df[["Station", "DRA Cost", "Power Cost", "Total Cost"]], use_container_width=True)
            st.bar_chart(df.set_index("Station")[["Total Cost"]])
            fig = px.pie(df, values="Total Cost", names="Station", title="Share of Total Cost by Station")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run optimization to see cost breakdown.")

# ---- DOWNLOAD TAB ----
with tabs[4]:
    st.header("Download Reports")
    if "opt_summary" in st.session_state:
        summary_df = st.session_state["opt_summary"]
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Optimization Summary as CSV",
            data=csv,
            file_name="optimization_summary.csv",
            mime="text/csv"
        )
    st.markdown("Download Input Tables")
    # Download station and all pump curves as CSVs
    station_csv = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Station Table CSV",
        data=station_csv,
        file_name="station_data.csv",
        mime="text/csv"
    )
    for idx, row in edited_df.iterrows():
        stn = row["Station"]
        if bool(row.get("Is Pump Station", False)):
            qh_df = st.session_state.get(f"{stn}_qh", pd.DataFrame())
            qeff_df = st.session_state.get(f"{stn}_qeff", pd.DataFrame())
            st.download_button(
                label=f"Download {stn} Q-H Curve CSV",
                data=qh_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{stn}_q_h_curve.csv",
                mime="text/csv"
            )
            st.download_button(
                label=f"Download {stn} Q-Eff Curve CSV",
                data=qeff_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{stn}_q_eff_curve.csv",
                mime="text/csv"
            )
