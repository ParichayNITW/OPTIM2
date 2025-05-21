import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly import graph_objects as go3d
from math import pi

# Configure NEOS email
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

# Page settings
st.set_page_config(page_title="Pipeline Optimization", layout="wide")

# Styles
st.markdown("""
<style>
.section-title {
  font-size:1.2rem; font-weight:600; margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)

st.title("Pipeline Optimization Dashboard")

# Solver wrapper (cached)
@st.cache_data
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🔧 Inputs")
    # Global fluid & cost
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    # Station list controls
    st.subheader("Stations")
    add_col, rem_col = st.columns(2)
    if add_col.button("➕ Add Station"):
        idx = len(st.session_state.stations) + 1
        st.session_state.stations.append({
            'name': f'Station {idx}', 'elev': 0.0, 'D': 0.711, 't': 0.007,
            'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'min_residual': 50.0, 'is_pump': False,
            'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
            'max_pumps': 1, 'MinRPM': 1200, 'DOL': 1500, 'max_dr': 0
        })
    if rem_col.button("🗑️ Remove Station") and len(st.session_state.stations) > 1:
        st.session_state.stations.pop()
    # View selector
    view = st.radio("Select View:", [
        "Summary", "Cost Breakdown", "Performance",
        "System Curves", "Pump-System Interaction",
        "Cost Landscape", "Nonconvex Visuals"
    ])

# --- Initialize station list if empty ---
if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200, 'DOL': 1500, 'max_dr': 0
    }]

# --- Station Input Forms ---
for i, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {i}"):
        stn['name'] = st.text_input("Name", stn['name'], key=f"name{i}")
        stn['elev'] = st.number_input("Elevation (m)", stn['elev'], step=0.1, key=f"elev{i}")
        stn['KV']   = st.number_input("Viscosity (cSt)", stn.get('KV',10.0), step=0.1, key=f"kv{i}")
        stn['rho']  = st.number_input("Density (kg/m³)", stn.get('rho',850.0), step=1.0, key=f"rho{i}")
        if i==1:
            stn['min_residual'] = st.number_input("Required Residual Head (m)", stn['min_residual'], step=0.1, key=f"res{i}")
        stn['D']     = st.number_input("Outer Diameter (m)", stn['D'], format="%.3f", step=0.001, key=f"D{i}")
        stn['t']     = st.number_input("Wall Thickness (m)", stn['t'], format="%.4f", step=1e-4, key=f"t{i}")
        stn['SMYS']  = st.number_input("SMYS (psi)", stn['SMYS'], step=1000, key=f"SMYS{i}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", stn['rough'], format="%.5f", step=1e-5, key=f"rough{i}")
        stn['L']     = st.number_input("Length to next (km)", stn['L'], min_value=0.0, step=1.0, key=f"L{i}")
        stn['is_pump'] = st.checkbox("Pumping Station?", stn['is_pump'], key=f"pump{i}")
        # Pump-specific
        if stn['is_pump']:
            stn['power_type'] = st.selectbox("Power Source", ["Grid","Diesel"], key=f"ptype{i}")
            if stn['power_type']=='Grid':
                stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", stn['rate'], key=f"rate{i}")
                stn['sfc']  = 0.0
            else:
                stn['sfc']  = st.number_input("SFC (gm/bhp·hr)", stn['sfc'], key=f"sfc{i}")
                stn['rate'] = 0.0
            stn['max_pumps'] = st.number_input("Maximum Pumps Available", stn['max_pumps'], min_value=1, step=1, key=f"mpumps{i}")
            stn['MinRPM']    = st.number_input("Min RPM", stn['MinRPM'], step=100, key=f"minrpm{i}")
            stn['DOL']       = st.number_input("Rated RPM (DOL)", stn['DOL'], step=100, key=f"dol{i}")
            stn['max_dr']    = st.number_input("Max Drag Reduction (%)", stn['max_dr'], step=5, key=f"mdr{i}")
            # performance editors
            head_df = st.data_editor(pd.DataFrame({"Flow":[] ,"Head":[]}), num_rows='dynamic', key=f"head{i}")
            eff_df  = st.data_editor(pd.DataFrame({"Flow":[] ,"Eff":[]}),  num_rows='dynamic', key=f"eff{i}")
            st.session_state[f"head_data_{i}"] = head_df
            st.session_state[f"eff_data_{i}"]  = eff_df
        # peaks
        peak_df = st.data_editor(pd.DataFrame({"Location":[],"Elevation":[]}), num_rows='dynamic', key=f"peak{i}")
        st.session_state[f"peak_data_{i}"] = peak_df

# --- Terminal Input ---
st.markdown("---")
st.subheader("Terminal Station")
term_name = st.text_input("Name", "Terminal")
term_elev = st.number_input("Elevation (m)", 0.0)
term_min  = st.number_input("Required Residual Head (m)", 50.0)

# --- Solve Trigger ---
if st.button("Run Optimization 🚀"):
    with st.spinner("Optimizing..."):
        sts = st.session_state.stations
        for i, stn in enumerate(sts,1):
            if stn['is_pump']:
                dfh = st.session_state[f"head_data_{i}"]; Qh,Hh = dfh.values.T
                stn['A'],stn['B'],stn['C'] = np.polyfit(Qh,Hh,2)
                dfe = st.session_state[f"eff_data_{i}"]; Qe,Ee = dfe.values.T
                stn['P'],stn['Q'],stn['R'],stn['S'],stn['T'] = np.polyfit(Qe,Ee,4)
            peaks = st.session_state[f"peak_data_{i}"]
            stn['peaks'] = [{'loc':loc,'elev':elev} for loc,elev in peaks.values]
        term = {'name':term_name,'elev':term_elev,'min_residual':term_min}
        out = solve_pipeline(st.session_state.stations, term, FLOW, RateDRA, Price_HSD)
        st.session_state['res'] = out

# --- Display Results ---
if 'res' in st.session_state:
    res  = st.session_state['res']
    sta  = st.session_state['stations']
    # key metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost", f"₹{res['total_cost']:,.2f}")
    c2.metric("Total Pumps", sum(res[f"num_pumps_{s['name'].lower().replace(' ','_')}"] for s in sta))
    c3.metric("Avg Speed", f"{np.mean([res[f"speed_{s['name'].lower().replace(' ','_')}"] for s in sta]):.1f} rpm")
    c4.metric("Avg Eff", f"{np.mean([res[f"efficiency_{s['name'].lower().replace(' ','_')}"] for s in sta]):.1f}%")
    # tabs
    tabs = st.tabs(["Summary","Cost Breakdown","Performance","System Curves","Pump-System","Cost Landscape","Nonconvex"])
    # Summary
    if view=="Summary":
        with tabs[0]:
            df = pd.DataFrame({ 'Process':["Cost","DRA","Pumps","Speed","Eff","Re","HLoss","Vel","ResH","SDH","DR%"],
                **{s['name']:[res[k.format(s['name'].lower().replace(' ','_'))] for k in [
                    'power_cost_{}','dra_cost_{}','num_pumps_{}','speed_{}','efficiency_{}',
                    'reynolds_{}','head_loss_{}','velocity_{}','residual_head_{}','sdh_{}','drag_reduction_{}'
                ]]
                for s in sta }
            )
            st.dataframe(df, use_container_width=True)
    # Cost Breakdown
    elif view=="Cost Breakdown":
        with tabs[1]:
            dfc = pd.DataFrame({
                'Station':[s['name'] for s in sta],
                'Power+Fuel':[res[f"power_cost_{s['name'].lower().replace(' ','_')}"] for s in sta],
                'DRA':[res[f"dra_cost_{s['name'].lower().replace(' ','_')}"] for s in sta]
            })
            fig=px.bar(dfc.melt(id_vars='Station',value_vars=['Power+Fuel','DRA'],var_name='Type',value_name='INR/day'),x='Station',y='INR/day',color='Type')
            st.plotly_chart(fig, use_container_width=True)

elif view == "Performance":
    with tab3:
        # Add two more sub-tabs under Performance
        perf_tab, head_tab, pump_curve_tab, eff_curve_tab, pwr_speed_tab, pwr_flow_tab = st.tabs([
         "Head Loss", "Velocity & Re",
         "Pump Characteristic Curve", "Pump Efficiency Curve",
         "Power vs Speed", "Power vs Flow"
        ])
        with perf_tab:
            st.markdown("<div class='section-title'>Head Loss per Segment</div>", unsafe_allow_html=True)
            df_hloss = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Head Loss": [res.get(f"head_loss_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
            })
            fig_h = go.Figure(go.Bar(x=df_hloss["Station"], y=df_hloss["Head Loss"]))
            fig_h.update_layout(yaxis_title="Head Loss (m)")
            st.plotly_chart(fig_h, use_container_width=True)
        with head_tab:
            st.markdown("<div class='section-title'>Velocity & Reynolds</div>", unsafe_allow_html=True)
            df_vel = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Velocity (m/s)": [res.get(f"velocity_{s['name'].lower().replace(' ','_')}",0) for s in stations_data],
                "Reynolds": [res.get(f"reynolds_{s['name'].lower().replace(' ','_')}",0) for s in stations_data]
            })
            st.dataframe(df_vel.style.format({"Velocity (m/s)":"{:.2f}", "Reynolds":"{:.0f}"}))

        with pump_curve_tab:
            st.markdown("<div class='section-title'>Pump Characteristic Curves (Head vs Flow)</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn.get("is_pump"): continue
                key  = stn["name"].lower().replace(" ","_")
                A,B,C = stn["A"], stn["B"], stn["C"]
                DOL = res[f"dol_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                fig = go.Figure()
                # sweep rpm in 100-rpm steps
                for rpm in np.arange(stn["MinRPM"], DOL+1, 100):
                    Hcurve = (A*flows**2 + B*flows + C) * (rpm/DOL)**2
                    fig.add_trace(go.Scatter(x=flows, y=Hcurve, mode="lines", name=f"{rpm} rpm"))
                fig.update_layout(
                    title=f"{stn['name']}: Head vs Flow",
                    xaxis_title="Flow (m³/hr)",
                    yaxis_title="Head (m)"
                )
                st.plotly_chart(fig, use_container_width=True)

        with eff_curve_tab:
            st.markdown("<div class='section-title'>Pump Efficiency Curves (η vs Flow)</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn.get("is_pump"): continue
                key = stn["name"].lower().replace(" ","_")
                P,Q,R,S,T = stn["P"], stn["Q"], stn["R"], stn["S"], stn["T"]
                DOL = res[f"dol_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                fig = go.Figure()
                for rpm in np.arange(stn["MinRPM"], DOL+1, 100):
                    flow_eq = FLOW * DOL / rpm
                    η = (P*flow_eq**4 + Q*flow_eq**3 + R*flow_eq**2 + S*flow_eq + T)/100.0
                    fig.add_trace(go.Scatter(x=flows, y=[η]*len(flows), mode="lines", name=f"{rpm} rpm"))
                fig.update_layout(
                    title=f"{stn['name']}: Efficiency vs Flow",
                    xaxis_title="Flow (m³/hr)",
                    yaxis_title="Efficiency (%)"
                )
                st.plotly_chart(fig, use_container_width=True)

        with pwr_speed_tab:
            st.markdown("<div class='section-title'>Power (kW) vs Pump Speed</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn.get("is_pump"): continue
                key = stn["name"].lower().replace(" ","_")
                num = res[f"num_pumps_{key}"]
                A,B,C = stn["A"], stn["B"], stn["C"]
                P,Q,R,S,T = stn["P"], stn["Q"], stn["R"], stn["S"], stn["T"]
                DOL = res[f"dol_{key}"]
                rates = []
                rpms  = np.arange(stn["MinRPM"], DOL+1, 100)
                for rpm in rpms:
                    # head & eff at this rpm
                    H = (A*FLOW**2 + B*FLOW + C)*(rpm/DOL)**2
                    flow_eq = FLOW * DOL / rpm
                    η = (P*flow_eq**4 + Q*flow_eq**3 + R*flow_eq**2 + S*flow_eq + T)/100.0
                    power = (stn["rho"] * FLOW * 9.81 * H * num)/(3600*1000*η*0.95)
                    rates.append(power)
                fig = go.Figure(go.Scatter(x=rpms, y=rates, mode="lines+markers"))
                fig.update_layout(
                    title=f"{stn['name']}: Power vs Speed",
                    xaxis_title="RPM",
                    yaxis_title="Power (kW)"
                )
                st.plotly_chart(fig, use_container_width=True)

        with pwr_flow_tab:
            st.markdown("<div class='section-title'>Power (kW) vs Flow</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn.get("is_pump"): continue
                key = stn["name"].lower().replace(" ","_")
                num = res[f"num_pumps_{key}"]
                speed = res[f"speed_{key}"]
                A,B,C = stn["A"], stn["B"], stn["C"]
                P,Q,R,S,T = stn["P"], stn["Q"], stn["R"], stn["S"], stn["T"]
                flows = np.linspace(0, FLOW*1.2, 100)
                powers = []
                for q in flows:
                    H = (A*q**2 + B*q + C)*(speed/res[f"dol_{key}"])**2
                    flow_eq = q * res[f"dol_{key}"] / speed
                    η = (P*flow_eq**4 + Q*flow_eq**3 + R*flow_eq**2 + S*flow_eq + T)/100.0
                    power = (stn["rho"] * q * 9.81 * H * num)/(3600*1000*η*0.95)
                    powers.append(power)
                fig = go.Figure(go.Scatter(x=flows, y=powers, mode="lines"))
                fig.update_layout(
                    title=f"{stn['name']}: Power vs Flow",
                    xaxis_title="Flow (m³/hr)",
                    yaxis_title="Power (kW)"
                )
                st.plotly_chart(fig, use_container_width=True)


elif view == "System Curves":
    with tab4:
        st.markdown("<div class='section-title'>System Head Curves</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False): 
                continue
            key = stn['name'].lower().replace(' ','_')
            d_inner_i = stn['D'] - 2*stn['t']
            rough = stn['rough']; L_seg = stn['L']; elev_i = stn['elev']
            # Generate SDH vs flow for 0%,10%,...,max DR
            curves = []
            kv = stn.get('KV', 10.0)
            for dra in range(0, int(stn['max_dr'])+1, 5):
                v_vals = np.linspace(0, FLOW, 101)/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (kv*1e-6) if kv>0 else np.zeros_like(v_vals)
                f_vals = np.where(Re_vals>0,
                                  0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2), 0.0)
                DH = f_vals * ((L_seg*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1-dra/100.0)
                SDH_vals = elev_i + DH
                curves.append(pd.DataFrame({"Flow": np.linspace(0, FLOW, 101), "SDH": SDH_vals, "DR": dra}))
            df_sys = pd.concat(curves)
            fig_sys = px.line(df_sys, x="Flow", y="SDH", color="DR", title=f"System Head ({stn['name']})")
            fig_sys.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig_sys, use_container_width=True)

elif view == "Pump-System Interaction":
    with tab5:
        st.markdown("<div class='section-title'>Pump vs System Interaction</div>", unsafe_allow_html=True)
        for i, stn in enumerate(stations_data, start=1):
            if not stn.get('is_pump', False):
                continue

            key = stn['name'].lower().replace(' ', '_')
            A = stn['A']; B = stn['B']; C = stn['C']
            P = stn['P']; Q = stn['Q']; R = stn['R']; S = stn['S']; T = stn['T']
            num_pumps = res[f"num_pumps_{key}"]
            min_rpm   = res[f"min_rpm_{key}"]
            dol       = res[f"dol_{key}"]

            # station-specific viscosity & roughness
            kv    = stn.get('KV', 10.0)
            rough = stn['rough']

            # common flow array
            flows = np.linspace(0, FLOW*1.5, 200)

            # Inner diameter
            d_inner_i = stn['D'] - 2*stn['t']

            # --- Pump vs System Combined (fig_int) ---
            fig_int = go.Figure()

            # 1) System head curves at 0–max_dr in 5% steps
            for dra in np.arange(0, int(stn['max_dr'])+1, 5):
                v_vals  = flows/3600.0 / (pi*(d_inner_i**2)/4)
                Re_vals = v_vals * d_inner_i / (kv*1e-6) if kv>0 else np.zeros_like(flows)
                f_vals  = np.where(
                    Re_vals>0,
                    0.25/(np.log10(rough/d_inner_i/3.7 + 5.74/(Re_vals**0.9))**2),
                    0.0
                )
                DH = f_vals * ((stn['L']*1000.0)/d_inner_i) * (v_vals**2/(2*9.81)) * (1 - dra/100.0)
                Hsys = stn['elev'] + DH
                fig_int.add_trace(go.Scatter(
                    x=flows, y=Hsys, mode='lines',
                    name=f"System ({dra}% DRA)"
                ))

            # 2) Pump head curves at 100 rpm increments
            for rpm in np.arange(min_rpm, dol+1, 100):
                Hpump = (A*flows**2 + B*flows + C) * (rpm/dol)**2
                fig_int.add_trace(go.Scatter(
                    x=flows, y=Hpump, mode='lines',
                    name=f"Pump {rpm} rpm"
                ))

            fig_int.update_layout(
                title=f"{stn['name']}: System & Pump Interaction",
                xaxis_title="Flow (m³/hr)",
                yaxis_title="Head (m)"
            )
            st.plotly_chart(fig_int, use_container_width=True)

            # --- Pressure vs Pipeline Length (fig_pl) ---
            # build cumulative distances
            cum_dist = [0]
            for s in stations_data:
                cum_dist.append(cum_dist[-1] + s['L'])
            fig_pl = go.Figure()
            for seg in range(len(stations_data)):
                x0, x1 = cum_dist[seg], cum_dist[seg+1]
                k0 = stations_data[seg]['name'].lower().replace(' ','_')
                y0 = res[f"sdh_{k0}"]
                y1 = res[f"residual_head_{k0}"]
                fig_pl.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode='lines+markers',
                    name=f"{stations_data[seg]['name']}→{stations_data[seg+1]['name'] if seg+1<len(stations_data) else 'Terminal'}"
                ))
                # vertical jump if pump at next node
                if seg+1 < len(stations_data) and res.get(f"num_pumps_{stations_data[seg+1]['name'].lower().replace(' ','_')}",0)>0:
                    k1 = stations_data[seg+1]['name'].lower().replace(' ','_')
                    y2 = res[f"sdh_{k1}"]
                    fig_pl.add_trace(go.Scatter(
                        x=[x1, x1], y=[y1, y2],
                        mode='lines', line=dict(dash='dash'), showlegend=False
                    ))
            fig_pl.update_layout(
                title="Pressure vs Pipeline Length",
                xaxis_title="Distance (km)",
                yaxis_title="Head (m)"
            )
            st.plotly_chart(fig_pl, use_container_width=True)

            # --- 3D Cost vs Speed vs DRA (fig3d) ---
            from plotly import graph_objects as go3d
            speeds = np.arange(min_rpm, dol+1, 100)
            drs    = np.arange(0, int(stn['max_dr'])+1, 5)
            Z = np.zeros((len(drs), len(speeds)))
            for ii, dra in enumerate(drs):
                for jj, rpm in enumerate(speeds):
                    # head & eff
                    H    = (A*FLOW**2 + B*FLOW + C)*(rpm/dol)**2
                    feq  = FLOW * dol / rpm
                    η    = (P*feq**4 + Q*feq**3 + R*feq**2 + S*feq + T)/100.0
                    pwr  = (stn['rho']*FLOW*9.81*H*num_pumps)/(3600*1000*η*0.95)
                    fuel = (stn['rate'] if stn['power_type']=="Grid"
                            else (stn['sfc']*1.34102/820 * Price_HSD))
                    dra_cost = (dra/4)*(FLOW*1000*24/1e6)*RateDRA
                    Z[ii, jj] = pwr*24*fuel + dra_cost

            surf = go3d.Surface(x=speeds, y=drs, z=Z)
            fig3d = go3d.Figure(data=[surf])
            fig3d.update_layout(
                title=f"{stn['name']}: Cost vs RPM vs DRA",
                scene=dict(
                    xaxis_title="RPM",
                    yaxis_title="DRA (%)",
                    zaxis_title="Cost (INR/day)"
                )
            )
            st.plotly_chart(fig3d, use_container_width=True)

elif view == "Cost Landscape":
    with tab6:
        st.markdown("<div class='section-title'>Cost Landscape (RPM vs DRA)</div>", unsafe_allow_html=True)
        for stn in stations_data:
            if not stn.get("is_pump"): 
                continue
        key     = stn["name"].lower().replace(" ","_")
        A,B,C   = stn["A"], stn["B"], stn["C"]
        P,Q,R,S,T = stn["P"],stn["Q"],stn["R"],stn["S"],stn["T"]
        num     = res[f"num_pumps_{key}"]
        dol     = res[f"dol_{key}"]
        min_rpm = res[f"min_rpm_{key}"]
        rho     = stn["rho"]
        rate    = stn["rate"] if stn["power_type"]=="Grid" else (stn["sfc"]*1.34102/820 * Price_HSD)
        max_dr  = stn["max_dr"]

        # Define grids
        rpms = np.arange(min_rpm, dol+1, 100)
        drs  = np.arange(0, max_dr+1, 5)
        Z    = np.zeros((len(drs), len(rpms)))

        for i, dra in enumerate(drs):
            for j, rpm in enumerate(rpms):
                # head at this rpm
                H    = (A*FLOW**2 + B*FLOW + C)*(rpm/dol)**2
                # eq flow for efficiency
                feq  = FLOW * dol / rpm
                η    = (P*feq**4 + Q*feq**3 + R*feq**2 + S*feq + T)/100.0
                # power (kW)
                pwr  = (rho * FLOW * 9.81 * H * num)/(3600*1000*η*0.95)
                # daily energy or fuel cost
                e_cost = pwr * 24 * rate
                # DRA cost
                dra_cost = (dra/4)*(FLOW*1000*24/1e6)*RateDRA
                Z[i,j] = e_cost + dra_cost

        # Plot 3D surface
        from plotly import graph_objects as go3d
        surf = go3d.Surface(x=rpms, y=drs, z=Z)
        fig = go3d.Figure(data=[surf])
        fig.update_layout(
            title=f"{stn['name']}: Cost vs RPM vs DRA",
            scene=dict(
                xaxis_title="RPM",
                yaxis_title="DRA (%)",
                zaxis_title="Cost (INR/day)"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
