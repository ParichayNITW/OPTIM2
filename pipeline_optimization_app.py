import streamlit as st
import pandas as pd
import json
from pipeline_model import solve_pipeline

def login_widget():
    st.title("Pipeline Optima Login")
    userid = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if userid == "admin" and password == "yourpassword":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid UserID or Password")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_widget()
    st.stop()

st.set_page_config(page_title="Pipeline Optima", layout="wide")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Input", "Optimization", "Visualization", "Cost Breakdown", "Download Report"
])

with tab1:
    st.subheader("Stations (last row = terminal station)")
    station_defaults = [
        {"Station":"STN1","Distance (km)":0,"Elevation (m)":100,"Diameter (mm)":500,"Roughness (mm)":0.045,
         "Is Pump Station":True,"Pump Count":2,"Head Limit (m)":1500,"Max Power (kW)":3000,"Fuel Rate (Rs/kWh)":12.5,"MAOP (m)":900,"Peaks":[]},
        {"Station":"STN2","Distance (km)":50,"Elevation (m)":105,"Diameter (mm)":500,"Roughness (mm)":0.045,
         "Is Pump Station":False,"Pump Count":"","Head Limit (m)":"","Max Power (kW)":"","Fuel Rate (Rs/kWh)":"","MAOP (m)":900,"Peaks":[]},
    ]
    station_cols = [
        "Station","Distance (km)","Elevation (m)","Diameter (mm)","Roughness (mm)",
        "Is Pump Station","Pump Count","Head Limit (m)","Max Power (kW)","Fuel Rate (Rs/kWh)","MAOP (m)","Peaks"
    ]
    df = pd.DataFrame(station_defaults)
    edited_df = st.data_editor(df, num_rows="dynamic", key="station_table", column_order=station_cols)

    # Hide/disable pump columns if Is Pump Station is not checked
    for idx, row in edited_df.iterrows():
        if not row.get("Is Pump Station", False):
            for col in ["Pump Count","Head Limit (m)","Max Power (kW)","Fuel Rate (Rs/kWh)"]:
                edited_df.at[idx, col] = ""

    st.subheader("Q vs Head Curve")
    qh_defaults = [{"Flow (m3/h)":100,"Head (m)":1000},{"Flow (m3/h)":200,"Head (m)":900},{"Flow (m3/h)":300,"Head (m)":750}]
    qh_df = st.data_editor(pd.DataFrame(qh_defaults), num_rows="dynamic", key="qh_curve")

    st.subheader("Q vs Efficiency Curve")
    qeff_defaults = [{"Flow (m3/h)":100,"Efficiency (%)":62},{"Flow (m3/h)":200,"Efficiency (%)":70},{"Flow (m3/h)":300,"Efficiency (%)":68}]
    qeff_df = st.data_editor(pd.DataFrame(qeff_defaults), num_rows="dynamic", key="qeff_curve")

    if st.button("Download Scenario as JSON"):
        scenario = {
            "station_table": edited_df.to_dict(orient="records"),
            "qh_curve": qh_df.to_dict(orient="records"),
            "qeff_curve": qeff_df.to_dict(orient="records"),
        }
        j = json.dumps(scenario, indent=2)
        st.download_button("Save Scenario as JSON", j, file_name="pipeline_scenario.json", mime="application/json")

    uploaded = st.file_uploader("Upload Scenario (JSON)", type="json")
    if uploaded:
        data = json.load(uploaded)
        st.session_state["station_table"] = pd.DataFrame(data["station_table"])
        st.session_state["qh_curve"] = pd.DataFrame(data["qh_curve"])
        st.session_state["qeff_curve"] = pd.DataFrame(data["qeff_curve"])
        st.experimental_rerun()

with tab2:
    st.header("Pipeline Optima™ Optimization")
    flow_rate = st.number_input("Enter flow rate (m³/h)", min_value=1, value=200, step=10)
    viscosity = st.number_input("Enter kinematic viscosity (cSt)", min_value=1.0, value=20.0, step=0.1)
    if st.button("Run Optimization"):
        with st.spinner("Optimizing..."):
            result = solve_pipeline(
                edited_df.to_dict(orient="records"),
                qh_df.to_dict(orient="records"),
                qeff_df.to_dict(orient="records"),
                flow_rate, viscosity
            )
            st.session_state["opt_result"] = result
            st.session_state["opt_summary"] = pd.DataFrame(result["summary_table"])
            st.session_state["opt_total_cost"] = result["total_cost"]
            st.success("Optimization complete")
            st.dataframe(st.session_state["opt_summary"], use_container_width=True)
            st.markdown(f"Total Operating Cost: ₹ {result['total_cost']:,.2f}")

with tab3:
    st.header("Visualization")
    if not qh_df.empty:
        import matplotlib.pyplot as plt
        import numpy as np
        qh = qh_df
        q_vals = np.linspace(min(qh["Flow (m3/h)"]), max(qh["Flow (m3/h)"]), 100)
        H = np.interp(q_vals, qh["Flow (m3/h)"], qh["Head (m)"])
        plt.figure(figsize=(8,4))
        plt.plot(q_vals, H, label="Pump Head Curve")
        plt.xlabel("Flow (m³/h)")
        plt.ylabel("Head (m)")
        plt.title("Q vs Head")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt.gcf())
    if not qeff_df.empty:
        qeff = qeff_df
        Eff = np.interp(q_vals, qeff["Flow (m3/h)"], qeff["Efficiency (%)"])
        plt.figure(figsize=(8,4))
        plt.plot(q_vals, Eff, label="Pump Efficiency Curve", color="green")
        plt.xlabel("Flow (m³/h)")
        plt.ylabel("Efficiency (%)")
        plt.title("Q vs Efficiency")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt.gcf())

with tab4:
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

with tab5:
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
    qh_csv = qh_df.to_csv(index=False).encode("utf-8")
    qeff_csv = qeff_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Station Table CSV",
        data=station_csv,
        file_name="station_data.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Q-H Curve CSV",
        data=qh_csv,
        file_name="q_h_curve.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Q-Eff Curve CSV",
        data=qeff_csv,
        file_name="q_eff_curve.csv",
        mime="text/csv"
    )
