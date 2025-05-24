import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
import hashlib

# --- CONFIG ---
palette = [c for c in px.colors.qualitative.Plotly if 'yellow' not in c.lower()]

st.set_page_config(page_title="Pipeline Optimization", layout="wide")

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

# --- DUMMY AUTH ---
users = {"parichay_das": hash_pwd("heteroscedasticity")}
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
        st.markdown("<div style='text-align:center;color:gray;margin-top:2em;font-size:0.9em;'>&copy; 2025 Pipeline Optimizer v1.1.1. Developed by Parichay Das.</div>", unsafe_allow_html=True)
        st.stop()
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
check_login()

# --- FAKE DATA for DEMO (Replace with your backend integration!) ---
if "last_res" not in st.session_state:
    st.session_state["last_res"] = {
        "power_cost_haldia": 267499.25,
        "dra_cost_haldia": 1000,
        "num_pumps_haldia": 2,
        "speed_haldia": 1480,
        "efficiency_haldia": 82.52,
        "reynolds_haldia": 1.13e5,
        "head_loss_haldia": 10.88,
        "velocity_haldia": 2.34,
        "residual_head_haldia": 54.12,
        "sdh_haldia": 64.5,
        "drag_reduction_haldia": 12.0,
        "total_cost": 267499.25 + 1000
    }
    st.session_state["last_stations_data"] = [{
        "name": "Haldia",
        "A": -2e-4, "B": 0.6, "C": 42, "P": -1e-11, "Q": 3e-8, "R": -3e-5, "S": 0.11, "T": 65,
        "DOL": 1500, "MinRPM": 1200, "max_dr": 40, "max_pumps": 4, "D": 0.7, "t": 0.008, "rough": 0.000045,
        "L": 100, "KV": 3.2, "elev": 10, "rho": 870, "rate": 9.0
    }]
    st.session_state["last_term_data"] = {"name": "Barauni", "elev": 6.0}
    st.session_state["FLOW"] = 1350.0
    st.session_state["RateDRA"] = 340.0
    st.session_state["Price_HSD"] = 87.0

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Summary", 
    "💰 Costs", 
    "⚙️ Performance", 
    "🌀 System Curves", 
    "🔄 Pump-System",
    "🧊 3D Analysis and Surface Plots"
])

# ------- TAB 1: Summary -------
with tab1:
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
    # Formatting, align left, two decimals except pumps/speed
    fmt = {c: "{:.2f}" for c in df_sum.columns if c != "Parameters"}
    fmt["No. of Pumps"] = "{:.0f}"; fmt["Pump Speed (rpm)"] = "{:.0f}"
    st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
    styled = df_sum.style.format(fmt).set_properties(**{'text-align': 'left'})
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")
    st.markdown(
        f"""<br>
        <div style='font-size:1.1em;'><b>Total Optimized Cost:</b> {res.get('total_cost', 0):,.2f} INR/day<br>
        <b>No. of operating Pumps:</b> {int(res.get('num_pumps_haldia',0))}<br>
        <b>Average Pump Efficiency:</b> {res.get('efficiency_haldia',0.0):.2f} %<br>
        <b>Average Pump Speed:</b> {res.get('speed_haldia',0.0):.0f} rpm</div>
        """,
        unsafe_allow_html=True
    )

# ------- TAB 2: Cost Breakdown -------
with tab2:
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

# ------- TAB 3: Performance (Demo, customize as needed) -------
with tab3:
    st.header("Performance")
    # Example: Power vs Flow
    flows = np.linspace(100, 2000, 20)
    power = np.maximum(700 + (flows-800)*0.5, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flows, y=power, mode='lines+markers', name='Power'))
    fig.update_layout(title="Power vs Flow: Haldia", xaxis_title="Flow (m³/hr)", yaxis_title="Power (kW)")
    st.plotly_chart(fig, use_container_width=True)

# ------- TAB 4: System Curves (Demo) -------
with tab4:
    st.header("System Curves")
    # Example system curve
    flows = np.linspace(100, 2000, 20)
    system_head = 50 + 0.01*(flows-100)**2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flows, y=system_head, mode='lines+markers', name='System Head'))
    fig.update_layout(title="System Head vs Flow", xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
    st.plotly_chart(fig, use_container_width=True)

# ------- TAB 5: Pump-System Interaction (Demo) -------
with tab5:
    st.header("Pump-System Interaction")
    # Dummy example
    flows = np.linspace(100, 2000, 20)
    pump_head = 90 - 0.02*(flows-900)
    system_head = 50 + 0.01*(flows-100)**2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flows, y=pump_head, mode='lines', name='Pump Head'))
    fig.add_trace(go.Scatter(x=flows, y=system_head, mode='lines', name='System Head'))
    fig.update_layout(title="Pump & System Head vs Flow", xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
    st.plotly_chart(fig, use_container_width=True)

# ------- TAB 6: 3D Analysis and Surface Plots -------
with tab6:
    last_res = st.session_state["last_res"]
    stations_data = st.session_state["last_stations_data"]
    FLOW = st.session_state["FLOW"]
    RateDRA = st.session_state["RateDRA"]
    Price_HSD = st.session_state["Price_HSD"]
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
    # Mark optimum
    if plot_opt == "Head vs Flow vs Speed":
        opt_z = get_head(flow_opt, speed_opt); opt_x, opt_y = flow_opt, speed_opt
    elif plot_opt == "Efficiency vs Flow vs Speed":
        opt_z = get_eff(flow_opt, speed_opt); opt_x, opt_y = flow_opt, speed_opt
    elif plot_opt == "System Head vs Flow vs DRA":
        opt_z = get_system_head(flow_opt, dra_opt); opt_x, opt_y = flow_opt, dra_opt
    elif plot_opt == "Power Cost vs Speed vs DRA":
        opt_z = get_power_cost(flow_opt, speed_opt, dra_opt, nopt_opt); opt_x, opt_y = speed_opt, dra_opt
    elif plot_opt == "Power Cost vs Flow vs Speed":
        opt_z = get_power_cost(flow_opt, speed_opt, dra_opt, nopt_opt); opt_x, opt_y = flow_opt, speed_opt
    elif plot_opt == "Total Cost vs NOP vs DRA":
        opt_z = get_total_cost(flow_opt, speed_opt, dra_opt, nopt_opt); opt_x, opt_y = nopt_opt, dra_opt
    fig.add_trace(go.Scatter3d(
        x=[opt_x], y=[opt_y], z=[opt_z], mode="markers+text", marker=dict(size=8, color='red'),
        text=["Optimum"], textposition="top center", name="Optimum"
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title=xlab,
            yaxis_title=ylab,
            zaxis_title=zlab,
        ),
        title=f"<b>{plot_opt}</b>",
        margin=dict(l=0, r=0, t=60, b=20),
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div style='text-align:center;font-size:1.0em;margin-top:10px;'>Surface shown for a small region (+/- delta) from the optimum point (marked in red) for clarity and hydraulic relevance.</div>",
        unsafe_allow_html=True
    )
