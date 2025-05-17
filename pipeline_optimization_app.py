import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import pytesseract
from PIL import Image
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory

if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 NEOS_EMAIL not found in secrets. Please add it.")

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(page_title="Mixed Integer Non Linear Convex Optimization of Pipeline Operations", layout="wide")

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
# Solver and Sidebar Inputs
# ---------------------
def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    """Solve pipeline optimization given dynamic station configuration."""
    N = len(stations)
    model = pyo.ConcreteModel()
    # Global parameters
    model.FLOW = pyo.Param(initialize=FLOW)
    model.KV = pyo.Param(initialize=KV)
    model.rho = pyo.Param(initialize=rho)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)
    # Index set for segments (1..N)
    model.I = pyo.RangeSet(1, N)
    # Pipeline parameters per segment
    elev = {i: stations[i-1]['elev'] for i in range(1, N+1)}
    elev[N+1] = terminal['elev']
    D_val = {i: stations[i-1]['D'] for i in range(1, N+1)}
    t_val = {i: stations[i-1]['t'] for i in range(1, N+1)}
    SMYS_val = {i: stations[i-1]['SMYS'] for i in range(1, N+1)}
    eps_val = {i: stations[i-1]['rough'] for i in range(1, N+1)}
    L_val = {i: stations[i-1]['L'] for i in range(1, N+1)}
    model.z = pyo.Param(range(1, N+2), initialize=lambda m,i: elev[i])
    model.D = pyo.Param(model.I, initialize=lambda m,i: D_val[i])
    model.t = pyo.Param(model.I, initialize=lambda m,i: t_val[i])
    model.SMYS = pyo.Param(model.I, initialize=lambda m,i: SMYS_val[i])
    model.eps = pyo.Param(model.I, initialize=lambda m,i: eps_val[i])
    model.L = pyo.Param(model.I, initialize=lambda m,i: L_val[i])
    model.d_int = pyo.Param(model.I, initialize=lambda m,i: max(1e-6, D_val[i] - 2*t_val[i]))
    # Pump parameters per station
    A_val = {i: stations[i-1]['A'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    B_val = {i: stations[i-1]['B'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    C_val = {i: stations[i-1]['C'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    DOL_val = {i: stations[i-1]['DOL'] if stations[i-1]['is_pump'] else 1 for i in range(1, N+1)}
    MinRPM_val = {i: stations[i-1]['MinRPM'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    P_val = {i: stations[i-1]['P'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    Q_val = {i: stations[i-1]['Q'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    R_val = {i: stations[i-1]['R'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    S_val = {i: stations[i-1]['S'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    T_val = {i: stations[i-1]['T'] if stations[i-1]['is_pump'] else 0 for i in range(1, N+1)}
    model.A = pyo.Param(model.I, initialize=lambda m,i: A_val[i])
    model.B = pyo.Param(model.I, initialize=lambda m,i: B_val[i])
    model.C = pyo.Param(model.I, initialize=lambda m,i: C_val[i])
    model.DOL = pyo.Param(model.I, initialize=lambda m,i: DOL_val[i])
    model.MinRPM = pyo.Param(model.I, initialize=lambda m,i: MinRPM_val[i])
    model.Pcoef = pyo.Param(model.I, initialize=lambda m,i: P_val[i])
    model.Qcoef = pyo.Param(model.I, initialize=lambda m,i: Q_val[i])
    model.Rcoef = pyo.Param(model.I, initialize=lambda m,i: R_val[i])
    model.Scoef = pyo.Param(model.I, initialize=lambda m,i: S_val[i])
    model.Tcoef = pyo.Param(model.I, initialize=lambda m,i: T_val[i])
    # Decision variables
    model.RH = pyo.Var(range(2, N+2), domain=pyo.NonNegativeReals, bounds=(50, None), initialize=50)
    def NOP_bounds(m, i):
        maxp = stations[i-1]['pumps'] if stations[i-1]['is_pump'] else 0
        lb = 0
        if i == 1 and stations[i-1]['is_pump'] and maxp > 0:
            lb = 1
        return (lb, maxp)
    model.NOP = pyo.Var(model.I, domain=pyo.NonNegativeIntegers, bounds=NOP_bounds, initialize=1)
    def N_bounds(m, i):
        return (MinRPM_val[i] if stations[i-1]['is_pump'] else 0, DOL_val[i] if stations[i-1]['is_pump'] else 1)
    model.N = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=N_bounds, initialize=lambda m,i: (MinRPM_val[i]+DOL_val[i])/2 if stations[i-1]['is_pump'] else 0)
    model.DR = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, 100), initialize=40)
    for i in range(1, N+1):
        if stations[i-1]['is_pump']:
            model.DR[i].setub(stations[i-1]['max_dr'])
        else:
            model.DR[i].fix(0)
    # Constraints: head balance for each segment
    def head_balance_rule(m, i):
        RH_i = 50 if i == 1 else m.RH[i]
        RH_next = m.RH[i+1] if i+1 <= N+1 else 50
        dz = elev[i+1] - elev[i]
        # flow velocity and friction
        v = (m.FLOW/3600) / (3.1416 * (m.d_int[i]**2) / 4)
        Re = v * m.d_int[i] / (m.KV * 1e-6)
        f = 0.25 / (pyo.log10((m.eps[i]/m.d_int[i]/3.7) + (5.74/(Re**0.9))))**2
        # determine which pump's injection applies
        j = i
        while j >= 1 and not stations[j-1]['is_pump']:
            j -= 1
        DR_frac = (1 - model.DR[j]/100) if j >= 1 and stations[j-1]['is_pump'] else 1
        # head losses
        DH = f * (m.L[i]*1000/m.d_int[i]) * (v**2/(2*9.81)) * DR_frac
        # pump head added
        TDH = (m.A[i]*m.FLOW**2 + m.B[i]*m.FLOW + m.C[i]) * (m.N[i]/m.DOL[i])**2 if stations[i-1]['is_pump'] else 0
        return RH_i + TDH * model.NOP[i] == RH_next + dz + DH
    model.head_balance = pyo.Constraint(model.I, rule=head_balance_rule)
    # Ensure pump speed is at least MinRPM when pump is on
    def min_speed_rule(m, i):
        if stations[i-1]['is_pump']:
            return m.N[i] >= m.MinRPM[i] * (model.NOP[i] > 0)
        else:
            return pyo.Constraint.Skip
    model.min_speed = pyo.Constraint(model.I, rule=min_speed_rule)
    # Objective: minimize daily power cost + DRA cost
    power_terms = []
    dra_terms = []
    for i in range(1, N+1):
        if stations[i-1]['is_pump']:
            # approximate power usage for cost (efficiency ~95% mechanical assumed)
            if stations[i-1]['power_type'] == "Grid":
                power_terms.append(((model.rho*model.FLOW*9.81*(A_val[i]*model.FLOW**2 + B_val[i]*model.FLOW + C_val[i]))/(3600*1000*0.95)) * 24 * stations[i-1]['rate'])
            else:
                power_terms.append(((model.rho*model.FLOW*9.81*(A_val[i]*model.FLOW**2 + B_val[i]*model.FLOW + C_val[i]))/(3600*1000*0.95)) * (stations[i-1]['sfc']*1.34102/820 * 24 * Price_HSD))
            dra_terms.append((model.DR[i]/1e6) * model.FLOW * 24 * 1000 * model.Rate_DRA)
    model.Objf = pyo.Objective(expr=sum(power_terms) + sum(dra_terms), sense=pyo.minimize)
    # Solve optimization
    try:
        results = SolverManagerFactory('neos').solve(model, opt='couenne')
        model.solutions.load_from(results)
    except Exception as e:
        st.error(f"Solver error: {e}")
    # Collect results into dictionary
    res = {}
    total_cost = 0.0
    for i in range(1, N+1):
        name = stations[i-1]['name']
        key = name.lower()
        if stations[i-1]['is_pump']:
            NOP_val = int(pyo.value(model.NOP[i]))
            res[f"num_pumps_{key}"] = NOP_val
            res[f"speed_{key}"] = pyo.value(model.N[i])
            # pump efficiency (in %)
            flow_eq = float(FLOW) * DOL_val[i] / max(pyo.value(model.N[i]), 1e-6)
            effp = (P_val[i]*flow_eq**4 + Q_val[i]*flow_eq**3 + R_val[i]*flow_eq**2 + S_val[i]*flow_eq + T_val[i]) / 100.0
            res[f"efficiency_{key}"] = effp * 100
            res[f"drag_reduction_{key}"] = pyo.value(model.DR[i])
            # compute cost contributions
            if stations[i-1]['power_type'] == "Grid":
                p_cost = ((rho * FLOW * 9.81 * ((A_val[i]*FLOW**2 + B_val[i]*FLOW + C_val[i]) * (pyo.value(model.N[i])/DOL_val[i])**2) * NOP_val) / (3600*1000 * max(effp,1e-3) * 0.95)) * 24 * stations[i-1]['rate']
            else:
                p_cost = ((rho * FLOW * 9.81 * ((A_val[i]*FLOW**2 + B_val[i]*FLOW + C_val[i]) * (pyo.value(model.N[i])/DOL_val[i])**2) * NOP_val) / (3600*1000 * max(effp,1e-3) * 0.95)) * (stations[i-1]['sfc']*1.34102/820 * 24 * Price_HSD)
            d_cost = (pyo.value(model.DR[i]) / 1e6) * FLOW * 24 * 1000 * RateDRA
            res[f"power_cost_{key}"] = p_cost
            res[f"dra_cost_{key}"] = d_cost
            res[f"station_cost_{key}"] = p_cost + d_cost
            total_cost += (p_cost + d_cost)
            res[f"sdh_{key}"] = 50 if i == 1 else pyo.value(model.RH[i])
        else:
            res[f"num_pumps_{key}"] = 0
        # pipeline metrics for segment i
        v_val = (FLOW/3600) / (3.1416 * ((D_val[i] - 2*t_val[i])**2) / 4)
        Re_val = v_val * (D_val[i] - 2*t_val[i]) / (KV * 1e-6)
        f_val = 0.25 / (np.log10((eps_val[i]/(D_val[i] - 2*t_val[i])/3.7) + (5.74/(Re_val**0.9))))**2
        j = i
        while j >= 1 and not stations[j-1]['is_pump']:
            j -= 1
        DR_frac_val = (1 - (pyo.value(model.DR[j]) / 100)) if j >= 1 and stations[j-1]['is_pump'] else 1
        DH_val = f_val * ((L_val[i] * 1000) / (D_val[i] - 2*t_val[i])) * (v_val**2 / (2*9.81)) * DR_frac_val
        res[f"reynolds_{key}"] = Re_val
        res[f"head_loss_{key}"] = DH_val
        res[f"velocity_{key}"] = v_val
        res[f"residual_head_{key}"] = 50 if i == 1 else pyo.value(model.RH[i])
        if stations[i-1]['is_pump']:
            res[f"coef_A_{key}"] = A_val[i]; res[f"coef_B_{key}"] = B_val[i]; res[f"coef_C_{key}"] = C_val[i]
            res[f"dol_{key}"] = DOL_val[i]; res[f"min_rpm_{key}"] = MinRPM_val[i]
            res[f"coef_P_{key}"] = P_val[i]; res[f"coef_Q_{key}"] = Q_val[i]; res[f"coef_R_{key}"] = R_val[i]; res[f"coef_S_{key}"] = S_val[i]; res[f"coef_T_{key}"] = T_val[i]
    term_key = terminal['name'].lower()
    res[f"residual_head_{term_key}"] = pyo.value(model.RH[N+1])
    res["total_cost"] = total_cost
    return res

with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Adjust Parameters", expanded=True):
        FLOW = st.number_input("Flow rate (m³/hr)", value=2000.0, step=10.0)
        KV = st.number_input("Viscosity (cSt)", value=10.0, step=0.1)
        rho = st.number_input("Density (kg/m³)", value=880.0, step=10.0)
        # Removed individual SFC inputs (now provided per station)
        RateDRA = st.number_input("DRA Rate (INR/L)", value=500.0, step=0.1)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    # Dynamic station configuration
    add_col, rem_col = st.columns(2)
    add_clicked = add_col.button("Add Station")
    rem_clicked = rem_col.button("Remove Station")
    if 'stations' not in st.session_state:
        st.session_state.stations = [{
            'name': 'Station 1', 'elev': 0.0, 'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0, 'pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
            'max_dr': 40.0, 'A': -2e-6, 'B': -0.0015, 'C': 179.14, 'P': -4.161e-14, 'Q': 6.574e-10, 'R': -8.737e-06, 'S': 0.04924, 'T': -0.001754
        }]
    if add_clicked:
        n = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name': f'Station {n}', 'elev': 0.0, 'D': 0.7112, 't': 0.0071374, 'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'is_pump': True, 'power_type': 'Diesel', 'rate': 9.0, 'sfc': 150.0, 'pumps': 1, 'MinRPM': 2750.0, 'DOL': 3437.0,
            'max_dr': 40.0, 'A': -1e-5, 'B': 0.00135, 'C': 270.08, 'P': -4.07e-13, 'Q': 3.4657e-09, 'R': -1.9273e-05, 'S': 0.067033, 'T': -0.15043
        })
    if rem_clicked and len(st.session_state.stations) > 1:
        st.session_state.stations.pop()
    # Station input fields
    for idx, station in enumerate(st.session_state.stations, start=1):
        exp_label = f"Station {idx}: " + (station['name'] if station['name'] else "")
        with st.expander(exp_label, expanded=True):
            station['name'] = st.text_input("Station name", value=station['name'], key=f"station{idx}_name")
            station['elev'] = st.number_input("Elevation (m)", value=float(station['elev']), key=f"station{idx}_elev")
            station['D'] = st.number_input("Pipe outer diameter (m)", value=float(station['D']), key=f"station{idx}_D")
            station['t'] = st.number_input("Wall thickness (m)", value=float(station['t']), key=f"station{idx}_t")
            station['SMYS'] = st.number_input("SMYS (psi)", value=float(station['SMYS']), key=f"station{idx}_SMYS")
            station['rough'] = st.number_input("Pipe roughness (m)", value=float(station['rough']), key=f"station{idx}_rough")
            station['L'] = st.number_input("Segment length to next (km)", value=float(station['L']), key=f"station{idx}_L")
            station['is_pump'] = st.checkbox("Pumping station", value=bool(station['is_pump']), key=f"station{idx}_pump")
            if station['is_pump']:
                station['power_type'] = st.selectbox("Power source", ["Grid", "Diesel"], index=(0 if station['power_type']=="Grid" else 1), key=f"station{idx}_power")
                if station['power_type'] == "Grid":
                    station['rate'] = st.number_input("Rate (Rs/kWh)", value=float(station['rate']), key=f"station{idx}_rate")
                else:
                    station['sfc'] = st.number_input("SFC (gm/bhp-hr)", value=float(station['sfc']), key=f"station{idx}_sfc")
                station['pumps'] = st.number_input("Number of pumps available", min_value=1, value=int(station['pumps']), step=1, key=f"station{idx}_pumps")
                station['MinRPM'] = st.number_input("Pump Min RPM", value=float(station['MinRPM']), key=f"station{idx}_MinRPM")
                station['DOL'] = st.number_input("Pump rated RPM", value=float(station['DOL']), key=f"station{idx}_DOL")
                st.file_uploader("Pump head curve image", type=["png","jpg","jpeg"], key=f"station{idx}_head_img")
                st.file_uploader("Pump efficiency curve image", type=["png","jpg","jpeg"], key=f"station{idx}_eff_img")
                station['max_dr'] = st.number_input("Maximum Drag Reduction (%)", value=float(station['max_dr']), key=f"station{idx}_maxdr")
    run = st.button("🚀 Run Optimization")

if run:
    with st.spinner("Solving pipeline optimization..."):
        # Prepare station and terminal data for solver
        stations_data = st.session_state.stations
        terminal_data = {"name": "Terminal", "elev": 0.0}
        res = solve_pipeline(stations_data, terminal_data, FLOW, KV, rho, RateDRA, Price_HSD)

    # KPI cards
    total_pumps = sum(int(res.get(f"num_pumps_{s['name'].lower()}", 0)) for s in stations_data)
    speeds = [res.get(f"speed_{s['name'].lower()}", 0) for s in stations_data if res.get(f"num_pumps_{s['name'].lower()}", 0) > 0]
    effs = [res.get(f"efficiency_{s['name'].lower()}", 0) for s in stations_data if res.get(f"num_pumps_{s['name'].lower()}", 0) > 0]
    avg_speed = np.mean(speeds) if speeds else 0
    avg_eff = np.mean(effs) if effs else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost (INR)", f"₹{res.get('total_cost',0):,.2f}")
    c2.metric("Total Pumps", total_pumps)
    c3.metric("Avg Speed (rpm)", f"{avg_speed:.2f}")
    c4.metric("Avg Pumping Efficiency (%)", f"{avg_eff:.2f}")

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
