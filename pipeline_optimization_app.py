import streamlit as st
import pandas as pd
import json
import uuid
from pipeline_model import solve_pipeline

# ----- Login Widget -----
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

st.set_page_config(page_title="Pipeline Optima", layout="wide")

tabs = st.tabs([
    "Input", "Optimization", "Visualization", "Cost Breakdown", "Download Report"
])

def get_unique_key(stn, typ):
    base = f"{stn}_{typ}_curve"
    return f"{base}_{st.session_state.get('stations_uid', 'init')}"

if "stations_uid" not in st.session_state:
    st.session_state["stations_uid"] = str(uuid.uuid4())

# ------------------- INPUT TAB -------------------
with tabs[0]:
    st.subheader("Stations (last row = terminal station)")
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
    # Robust DataFrame assignment:
    if "station_table" in st.session_state:
        val = st.session_state["station_table"]
        if isinstance(val, pd.DataFrame):
            df = val.copy()
        else:
            df = pd.DataFrame(val)
    else:
        df = pd.DataFrame(station_defaults)
    edited_df = st.data_editor(
        df, num_rows="dynamic", key="station_table", column_order=station_cols
    )

    # Hide/disable pump columns if not a pump station
    for idx, row in edited_df.iterrows():
        if not row.get("Is Pump Station", False):
            for col in ["Pump Count","Head Limit (m)","Max Power (kW)","Fuel Rate (Rs/kWh)"]:
                edited_df.at[idx, col] = ""

    st.markdown("If you add/remove stations, click below to reset all pump curve tables (prevents widget errors).")
    if st.button("Reset Pump Curves"):
        st.session_state["stations_uid"] = str(uuid.uuid4())
        st.experimental_rerun()

    st.subheader("Pump Curves (per pump station)")
    station_curve_dict = {}
    qh_defaults = [{"Flow (m3/h)":100,"Head (m)":1000},{"Flow (m3/h)":200,"Head (m)":900},{"Flow (m3/h)":300,"Head (m)":750}]
    qeff_defaults = [{"Flow (m3/h)":100,"Efficiency (%)":62},{"Flow (m3/h)":200,"Efficiency (%)":70},{"Flow (m3/h)":300,"Efficiency (%)":68}]
    for idx, row in edited_df.iterrows():
        stn = row["Station"]
        if row.get("Is Pump Station", False):
            st.markdown(f"**{stn}:**")
            qh_table = st.data_editor(
                pd.DataFrame(qh_defaults),
                num_rows="dynamic",
                key=get_unique_key(stn, "qh")
            )
            qeff_table = st.data_editor(
                pd.DataFrame(qeff_defaults),
                num_rows="dynamic",
                key=get_unique_key(stn, "qeff")
            )
            station_curve_dict[stn] = {
                "qh_curve": qh_table.to_dict(orient="records"),
                "qeff_curve": qeff_table.to_dict(orient="records")
            }
        else:
            station_curve_dict[stn] = {"qh_curve": [], "qeff_curve": []}

    # Download/Upload scenario
    if st.button("Download Scenario as JSON"):
        scenario = {
            "station_table": edited_df.to_dict(orient="records"),
            "station_curve_dict": station_curve_dict
        }
        j = json.dumps(scenario, indent=2)
        st.download_button("Save Scenario as JSON", j, file_name="pipeline_scenario.json", mime="application/json")

    uploaded = st.file_uploader("Upload Scenario (JSON)", type="json")
    if uploaded:
        data = json.load(uploaded)
        # Always force a DataFrame
        try:
            st.session_state["station_table"] = pd.DataFrame(data["station_table"])
        except:
            st.session_state["station_table"] = data["station_table"]
        st.session_state["stations_uid"] = str(uuid.uuid4())
        st.experimental_rerun()

# ------------------- OPTIMIZATION TAB -------------------
with tabs[1]:
    st.header("Pipeline Optima™ Optimization")
    flow_rate = st.number_input("Enter flow rate (m³/h)", min_value=1, value=200, step=10)
    viscosity = st.number_input("Enter kinematic viscosity (cSt)", min_value=1.0, value=20.0, step=0.1)
    if st.button("Run Optimization"):
        with st.spinner("Optimizing..."):
            per_stn_qh = {}
            per_stn_qeff = {}
            for idx, row in edited_df.iterrows():
                stn = row["Station"]
                if row.get("Is Pump Station", False):
                    qh_key = get_unique_key(stn, "qh")
                    qeff_key = get_unique_key(stn, "qeff")
                    qh_df = st.session_state[qh_key]
                    qeff_df = st.session_state[qeff_key]
                    qh = qh_df.to_dict(orient="records")
                    qeff = qeff_df.to_dict(orient="records")
                else:
                    qh, qeff = [], []
                per_stn_qh[stn] = qh
                per_stn_qeff[stn] = qeff
            result = solve_pipeline(
                edited_df.to_dict(orient="records"),
                per_stn_qh,
                per_stn_qeff,
                flow_rate, viscosity
            )
            st.session_state["opt_result"] = result
            st.session_state["opt_summary"] = pd.DataFrame(result["summary_table"])
            st.session_state["opt_total_cost"] = result["total_cost"]
            st.success("Optimization complete")
            st.dataframe(st.session_state["opt_summary"], use_container_width=True)
            st.markdown(f"Total Operating Cost: ₹ {result['total_cost']:,.2f}")

# ------------------- VISUALIZATION TAB -------------------
with tabs[2]:
    st.header("Visualization")
    for idx, row in edited_df.iterrows():
        stn = row["Station"]
        if row.get("Is Pump Station", False):
            st.subheader(f"Pump Curves - {stn}")
            qh_key = get_unique_key(stn, "qh")
            qeff_key = get_unique_key(stn, "qeff")
            qh_df = st.session_state.get(qh_key, pd.DataFrame())
            qeff_df = st.session_state.get(qeff_key, pd.DataFrame())
            if not qh_df.empty:
                import matplotlib.pyplot as plt
                import numpy as np
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
            if not qeff_df.empty:
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

# ------------------- COST BREAKDOWN TAB -------------------
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

# ------------------- DOWNLOAD REPORT TAB -------------------
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
    station_csv = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Station Table CSV",
        data=station_csv,
        file_name="station_data.csv",
        mime="text/csv"
    )
    # Download all pump curves per station
    for idx, row in edited_df.iterrows():
        stn = row["Station"]
        if row.get("Is Pump Station", False):
            qh_key = get_unique_key(stn, "qh")
            qeff_key = get_unique_key(stn, "qeff")
            qh_df = st.session_state.get(qh_key, pd.DataFrame())
            qeff_df = st.session_state.get(qeff_key, pd.DataFrame())
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
