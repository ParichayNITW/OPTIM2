import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi, log10
from io import BytesIO

# Ensure NEOS email is configured
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets to enable NEOS solver.")

# Page setup
st.set_page_config(page_title="Pipeline Optimization", layout="wide")
st.markdown("""
<style>
.section-title {
  font-size:1.2rem;
  font-weight:600;
  margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)
st.markdown("# Mixed Integer Non-Linear Convex Optimization of Pipeline Operations", unsafe_allow_html=True)

# Solver wrapper
@st.cache_data
def solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, KV, rho, RateDRA, Price_HSD)

# --- Sidebar: Inputs ---
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        KV        = st.number_input("Viscosity (cSt)", value=10.0, step=0.1)
        rho       = st.number_input("Density (kg/m³)", value=850.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)

    # Station management
    st.subheader("Stations List")
    add_col, rem_col = st.columns(2)
    if 'stations' not in st.session_state:
        st.session_state.stations = []
    if add_col.button("➕ Add Station"):
        st.session_state.stations.append({
            'name': 'Station {}'.format(len(st.session_state.stations)+1),
            'elev': 0.0, 'D': 0.711, 't': 0.007, 'SMYS':52000.0,
            'rough':0.00004, 'L':50.0, 'min_residual':50.0,
            'is_pump':False, 'power_type':'Grid', 'rate':9.0,
            'sfc':150.0, 'max_pumps':1, 'MinRPM':1200.0,
            'DOL':1500.0,'max_dr':0.0
        })
    if rem_col.button("🗑️ Remove Station") and st.session_state.stations:
        st.session_state.stations.pop()

# Station inputs
stations_data = []
for idx, stn in enumerate(st.session_state.stations):
    st.markdown(f"### Station {idx+1}")
    cols = st.columns(4)
    stn['name'] = cols[0].text_input("Name", value=stn.get('name',''))
    stn['elev'] = cols[1].number_input("Elevation (m)", value=stn.get('elev',0.0))
    stn['L']    = cols[2].number_input("Length from prev (km)", value=stn.get('L',50.0))
    stn['min_residual'] = cols[3].number_input("Min Residual Head (m)", value=stn.get('min_residual',50.0))
    cols2 = st.columns(4)
    stn['D']     = cols2[0].number_input("Diameter (m)", value=stn.get('D',0.711))
    stn['t']     = cols2[1].number_input("Thickness (m)", value=stn.get('t',0.007))
    stn['rough'] = cols2[2].number_input("Roughness (m)", value=stn.get('rough',0.00004))
    stn['SMYS']  = cols2[3].number_input("SMYS (Pa)", value=stn.get('SMYS',52000.0))
    stn['is_pump'] = st.checkbox("Is Pump Station?", value=stn.get('is_pump',False))
    if stn['is_pump']:
        pump_cols = st.columns(5)
        stn['power_type'] = pump_cols[0].selectbox("Power Type", ["Grid","Diesel"], index=["Grid","Diesel"].index(stn.get('power_type','Grid')))
        stn['rate']       = pump_cols[1].number_input("Energy Rate (INR/kWh)", value=stn.get('rate',9.0))
        stn['sfc']        = pump_cols[2].number_input("Specific Fuel Cons (g/kWh)", value=stn.get('sfc',150.0))
        stn['max_pumps']  = pump_cols[3].number_input("# Pumps", value=stn.get('max_pumps',1), step=1)
        stn['MinRPM']     = pump_cols[4].number_input("Min RPM", value=stn.get('MinRPM',1200.0))
        stn['DOL']        = pump_cols[4].number_input("DOL RPM", value=stn.get('DOL',1500.0))
        stn['max_dr']     = st.number_input("Max Drag Reduction (%)", value=stn.get('max_dr',0.0), step=1.0)

# Terminal inputs
st.sidebar.markdown("---")
terminal_name = st.sidebar.text_input("Terminal Name", value=st.session_state.get('terminal_name','Terminal'))
terminal_elev = st.sidebar.number_input("Terminal Elevation (m)", value=st.session_state.get('terminal_elev',0.0))
terminal_min  = st.sidebar.number_input("Terminal Min Residual (m)", value=st.session_state.get('terminal_min',50.0))
st.session_state.terminal_name  = terminal_name
st.session_state.terminal_elev  = terminal_elev
st.session_state.terminal_min   = terminal_min

# Run Optimization
if st.button("🚀 Run Optimization"):
    terminal = {
        'name': terminal_name, 'elev':terminal_elev, 'min_residual':terminal_min
    }
    res = solve_pipeline(stations_data, terminal, FLOW, KV, rho, RateDRA, Price_HSD)

    # Build DataFrames
    df_sum = pd.DataFrame(res['summary'])
    df_cost = pd.DataFrame(res['cost_breakdown'])
    df_h    = pd.DataFrame(res['head_loss'])
    df_vel  = pd.DataFrame(res['velocity_re'])

    # Create figure for cost
    fig_cost = px.bar(df_cost, x='station', y=['energy_cost','dra_cost'], barmode='group', title="Cost Breakdown per Station")

    # Summary of head loss
    fig_h = px.line(df_h, x='segment', y='head_loss', title="Head Loss per Segment")

    # --- Tabs Layout ---
    tabs = st.tabs([
        "📋 Summary", "💰 Costs", "⚙️ Performance",
        "🌀 System Curves", "🔄 Pump-System",
        "🌡️ Pressure Profile", "📊 Cost Surface"
    ])

    # Summary
    with tabs[0]:
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button("📥 Download CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")

    # Costs
    with tabs[1]:
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_cost, use_container_width=True)

    # Performance
    perf_tabs = tabs[2].tabs([
        "Head Loss", "Velocity & Re",
        "Pump Curve", "Pump Efficiency",
        "Power vs Speed", "Power vs Flow"
    ])
    with perf_tabs[0]: st.plotly_chart(fig_h, use_container_width=True)
    with perf_tabs[1]: st.dataframe(df_vel)
    with perf_tabs[2]:
        for idx, stn in enumerate(stations_data):
            if not stn['is_pump']: continue
            A,B,C = res['pump_coeffs'][idx]
            flows = np.linspace(0, FLOW, 200)
            fig = go.Figure()
            for rpm in np.arange(stn['MinRPM'], stn['DOL']+1, 100):
                Hp = (A*flows**2 + B*flows + C)*(rpm/stn['DOL'])**2
                fig.add_trace(go.Scatter(x=flows, y=Hp, mode='lines', name=f'{rpm} rpm'))
            fig.update_layout(title=f"Characteristic Curve - {stn['name']}")
            st.plotly_chart(fig, use_container_width=True)
    with perf_tabs[3]:
        for idx, stn in enumerate(stations_data):
            if not stn['is_pump']: continue
            P,Q,R,S,T = res['pump_eff_coeffs'][idx]
            flows = np.linspace(0, FLOW, 200)
            fig = go.Figure()
            for rpm in np.arange(stn['MinRPM'], stn['DOL']+1, 100):
                eff = (P*flows**4 + Q*flows**3 + R*flows**2 + S*flows + T)*(rpm/stn['DOL'])
                fig.add_trace(go.Scatter(x=flows, y=eff, mode='lines', name=f'{rpm} rpm'))
            fig.update_layout(title=f"Efficiency Curve - {stn['name']}")
            st.plotly_chart(fig, use_container_width=True)
    with perf_tabs[4]:
        for idx, stn in enumerate(stations_data):
            if not stn['is_pump']: continue
            rpms = np.arange(stn['MinRPM'], stn['DOL']+1, 100)
            power = [res['power_vs_speed'][idx][i] for i in range(len(rpms))]
            fig = go.Figure(go.Scatter(x=rpms, y=power, mode='lines+markers'))
            fig.update_layout(title=f"Power vs Speed - {stn['name']}")
            st.plotly_chart(fig, use_container_width=True)
    with perf_tabs[5]:
        for idx, stn in enumerate(stations_data):
            flows = np.linspace(0, FLOW, 200)
            power = res['power_vs_flow'][idx]
            fig = go.Figure(go.Scatter(x=flows, y=power, mode='lines'))
            fig.update_layout(title=f"Power vs Flow - {stn['name']}")
            st.plotly_chart(fig, use_container_width=True)

    # System Curves
    with tabs[3]:
        for idx, stn in enumerate(stations_data):
            if not stn['is_pump']: continue
            d_i = stn['D']-2*stn['t']
            flows = np.linspace(0, FLOW*1.5, 200)
            fig = go.Figure()
            for dra in np.arange(0, stn['max_dr']+1, 5):
                v = flows/3600/(pi*(d_i**2)/4)
                Re = v*d_i/(KV*1e-6)
                f = np.where(Re>0, 0.25/(np.log10(stn['rough']/d_i/3.7+5.74/(Re**0.9))**2), 0)
                Hsys = stn['elev'] + f*(stn['L']*1000/d_i)*(v**2/(2*9.81))*(1-dra/100)
                fig.add_trace(go.Scatter(x=flows, y=Hsys, name=f'{dra}% DRA'))
            fig.update_layout(title=f"System Curve - {stn['name']}")
            st.plotly_chart(fig, use_container_width=True)

    # Pump-System Interaction
    with tabs[4]:
        for idx, stn in enumerate(stations_data):
            if not stn['is_pump']: continue
            flows = np.linspace(0, FLOW*1.5, 200)
            fig = go.Figure()
            # system at 5% steps
            for dra in np.arange(0, stn['max_dr']+1, 5):
                # same Hsys calc as above
                pass  # boilerplate
            # pump
            A,B,C = res['pump_coeffs'][idx]
            for rpm in np.arange(stn['MinRPM'], stn['DOL']+1, 100):
                Hp = (A*flows**2+B*flows+C)*(rpm/stn['DOL'])**2
                fig.add_trace(go.Scatter(x=flows, y=Hp, name=f'{rpm} rpm'))
            fig.update_layout(title=f"Interaction - {stn['name']}")
            st.plotly_chart(fig, use_container_width=True)

    # Pressure Profile
    with tabs[5]:
        positions = [0] + list(np.cumsum([s['L'] for s in stations_data]))
        pressures = [res['sdh_heads'][idx] for idx in range(len(stations_data))] + [res['residual_terminal']]
        fig = go.Figure()
        for i in range(len(positions)-1):
            fig.add_trace(go.Scatter(x=[positions[i],positions[i+1]], y=[pressures[i],pressures[i+1]], mode='lines+markers'))
            if stations_data[i+1]['is_pump']:
                j=i+1
                fig.add_trace(go.Scatter(x=[positions[j],positions[j]], y=[res['residual_heads'][j],res['sdh_heads'][j]], mode='lines'))
        fig.update_layout(title="Pressure Profile Along Pipeline", xaxis_title="Length (km)", yaxis_title="Head (m)")
        st.plotly_chart(fig, use_container_width=True)

    # Cost Surface
    with tabs[6]:
        for idx, stn in enumerate(stations_data):
            if not stn['is_pump']: continue
            rpms = np.arange(stn['MinRPM'], stn['DOL']+1, 100)
            drs  = np.arange(0, stn['max_dr']+1, 5)
            X,Y = np.meshgrid(rpms, drs)
            Z = res['cost_surface'][idx]
            fig = go.Figure(data=[go.Surface(x=X,y=Y,z=Z)])
            fig.update_layout(title=f"Cost Surface - {stn['name']}", scene=dict(xaxis_title='RPM',yaxis_title='DRA %',zaxis_title='Cost'))
            st.plotly_chart(fig, use_container_width=True)
