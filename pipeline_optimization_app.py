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

def get_or_init_table(default, session_key):
    if session_key not in st.session_state:
        st.session_state[session_key] = pd.DataFrame(default)
    df = st.data_editor(st.session_state[session_key], num_rows="dynamic", key=session_key)
    if not df.equals(st.session_state[session_key]):
        st.session_state[session_key] = df
    return df

def download_json(obj, filename):
    j = json.dumps(obj, indent=2)
    st.download_button("Download Scenario as JSON", j, file_name=filename, mime="application/json")

def upload_json(session_keys):
    uploaded = st.file_uploader("Upload Saved Scenario (JSON)", type="json")
    if uploaded:
        data = json.load(uploaded)
        for k in session_keys:
            if k in data:
                st.session_state[k] = pd.DataFrame(data[k]) if isinstance(data[k], list) else data[k]
        st.success("Scenario loaded.")
        st.experimental_rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Input", "Optimization", "Visualization", "Cost Breakdown", "Download Report"
])

with tab1:
    stations_cols = ["Station","Distance (km)","Elevation (m)","Diameter (mm)","Roughness (mm)","Is Pump Station","Pump Count","Head Limit (m)","Max Power (kW)","Fuel Rate (Rs/kWh)","MAOP (m)"]
    station_defaults = [
        {"Station":"STN1","Distance (km)":0,"Elevation (m)":100,"Diameter (mm)":500,"Roughness (mm)":0.045,"Is Pump Station":True,"Pump Count":2,"Head Limit (m)":1500,"Max Power (kW)":3000,"Fuel Rate (Rs/kWh)":12.5,"MAOP (m)":900},
        {"Station":"STN2","Distance (km)":50,"Elevation (m)":105,"Diameter (mm)":500,"Roughness (mm)":0.045,"Is Pump Station":True,"Pump Count":2,"Head Limit (m)":1500,"Max Power (kW)":3000,"Fuel Rate (Rs/kWh)":12.5,"MAOP (m)":900},
    ]
    station_df = get_or_init_table(station_defaults, "station_table")
    qh_defaults = [{"Flow (m3/h)":100,"Head (m)":1000},{"Flow (m3/h)":200,"Head (m)":900},{"Flow (m3/h)":300,"Head (m)":750}]
    qh_df = get_or_init_table(qh_defaults, "qh_curve")
    qeff_defaults = [{"Flow (m3/h)":100,"Efficiency (%)":62},{"Flow (m3/h)":200,"Efficiency (%)":70},{"Flow (m3/h)":300,"Efficiency (%)":68}]
    qeff_df = get_or_init_table(qeff_defaults, "qeff_curve")
    scenario = {
        "station_table": station_df.to_dict(orient="records"),
        "qh_curve": qh_df.to_dict(orient="records"),
        "qeff_curve": qeff_df.to_dict(orient="records"),
    }
    download_json(scenario, "pipeline_scenario.json")
    upload_json(["station_table","qh_curve","qeff_curve"])

with tab2:
    st.header("Pipeline Optima™ Optimization")
    flow_rate = st.number_input("Enter flow rate (m³/h)", min_value=1, value=200, step=10)
    viscosity = st.number_input("Enter kinematic viscosity (cSt)", min_value=1.0, value=20.0, step=0.1)
    if st.button("Run Optimization"):
        with st.spinner("Optimizing..."):
            result = solve_pipeline(
                st.session_state["station_table"].to_dict(orient="records"),
                st.session_state["qh_curve"].to_dict(orient="records"),
                st.session_state["qeff_curve"].to_dict(orient="records"),
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
    if "qh_curve" in st.session_state:
        import matplotlib.pyplot as plt
        import numpy as np
        qh = st.session_state["qh_curve"]
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
    if "qeff_curve" in st.session_state:
        qeff = st.session_state["qeff_curve"]
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
    station_csv = st.session_state["station_table"].to_csv(index=False).encode("utf-8")
    qh_csv = st.session_state["qh_curve"].to_csv(index=False).encode("utf-8")
    qeff_csv = st.session_state["qeff_curve"].to_csv(index=False).encode("utf-8")
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
