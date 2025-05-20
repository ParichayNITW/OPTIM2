import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
from io import BytesIO
from pyomo.opt import SolverManagerFactory

# NEOS email
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

st.set_page_config(page_title="Pipeline Optimization", layout="wide")

# CSS
st.markdown("""
<style>
.section-title {
  font-size:1.2rem; font-weight:600; margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Mixed Integer Nonlinear Optimization of Pipeline Operations</h1>", unsafe_allow_html=True)

# Solver function
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

# --- Sidebar Inputs --------------------------------------------------------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")

    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)

    st.subheader("Stations")
    add_col, rem_col = st.columns(2)
    if add_col.button("➕ Add Station"):
        n = len(st.session_state.get('stations', [])) + 1
        default = {
            'name': f'Station {n}', 'elev': 0.0, 'D': 0.711, 't': 0.007,
            'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
            'min_residual': 50.0, 'is_pump': False,
            'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
            'max_pumps': 1, 'MinRPM': 1000.0, 'DOL': 1500.0,
            'max_dr': 0.0
        }
        st.session_state.stations.append(default)
    if rem_col.button("🗑️ Remove Station"):
        if st.session_state.get('stations'):
            st.session_state.stations.pop()

# Initialize station list if missing
if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
        'max_dr': 0.0
    }]

# Station inputs
for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}", expanded=True):
        stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
        stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{idx}")
        stn['KV']  = st.number_input("Viscosity (cSt)", value=stn.get('KV',10.0), step=0.1, key=f"kv{idx}")
        stn['rho'] = st.number_input("Density (kg/m³)", value=stn.get('rho',850.0), step=1.0, key=f"rho{idx}")
        if idx == 1:
            stn['min_residual'] = st.number_input(
                "Residual Head at Station (m)", value=stn.get('min_residual',50.0), step=0.1, key=f"res{idx}"
            )
        stn['D']     = st.number_input("Outer Diameter (m)", value=stn['D'], format="%.3f", step=0.001, key=f"D{idx}")
        stn['t']     = st.number_input("Wall Thickness (m)", value=stn['t'], format="%.4f", step=1e-4, key=f"t{idx}")
        stn['SMYS']  = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{idx}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=1e-5, key=f"rough{idx}")
        stn['L']     = st.number_input("Length to next (km)", value=stn['L'], step=1.0, key=f"L{idx}")
        stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{idx}")

        if stn['is_pump']:
            stn['power_type'] = st.selectbox("Power Source", ["Grid","Diesel"], index=0 if stn['power_type']=="Grid" else 1, key=f"ptype{idx}")
            if stn['power_type']=="Grid":
                stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", value=stn.get('rate',9.0), key=f"rate{idx}")
                stn['sfc'] = 0.0
            else:
                stn['sfc'] = st.number_input("SFC (gm/bhp·hr)", value=stn.get('sfc',150.0), key=f"sfc{idx}")
                stn['rate'] = 0.0

            stn['max_pumps'] = st.number_input("Max Pumps", min_value=1, value=stn['max_pumps'], key=f"mpumps{idx}")
            stn['MinRPM']    = st.number_input("Min RPM", value=stn['MinRPM'], key=f"minrpm{idx}")
            stn['DOL']       = st.number_input("Rated RPM (DOL)", value=stn['DOL'], key=f"dol{idx}")
            stn['max_dr']    = st.number_input("Max Drag Reduction (%)", value=stn['max_dr'], key=f"mdr{idx}")

            st.markdown("**Enter Pump Performance Data:**")
            df_h = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
            df_h = st.data_editor(df_h, num_rows="dynamic", key=f"head{idx}")
            st.session_state[f"head_{idx}"] = df_h

            df_e = pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})
            df_e = st.data_editor(df_e, num_rows="dynamic", key=f"eff{idx}")
            st.session_state[f"eff_{idx}"] = df_e

        st.markdown("**Intermediate Elevation Peaks (to next station):**")
        pk_df = pd.DataFrame({"Location (km)": [stn['L']/2.0], "Elevation (m)": [stn['elev']+100.0]})
        pk_df = st.data_editor(pk_df, num_rows="dynamic", key=f"peak{idx}")
        st.session_state[f"peak_{idx}"] = pk_df

# Terminal inputs
st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value="Terminal")
terminal_elev = st.number_input("Elevation (m)", value=0.0, step=0.1)
terminal_head = st.number_input("Required Residual Head (m)", value=50.0, step=1.0)

# Run optimization button
run = st.button("🚀 Run Optimization")

if run:
    with st.spinner("Solving optimization..."):
        stations_data = st.session_state['stations']
        term_data = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_head}

        for idx, stn in enumerate(stations_data, start=1):
            if stn.get('is_pump', False):
                dfh = st.session_state[f"head_{idx}"]
                dfe = st.session_state[f"eff_{idx}"]
                if len(dfh) < 3 or len(dfe) < 5:
                    st.error("Each pump needs ≥3 head & ≥5 eff points.")
                    st.stop()
                Qh, Hh = dfh.iloc[:,0].values, dfh.iloc[:,1].values
                c2, c1, c0 = np.polyfit(Qh, Hh, 2)
                stn['A'], stn['B'], stn['C'] = c2, c1, c0
                Qe, Ee = dfe.iloc[:,0].values, dfe.iloc[:,1].values
                coeff_e = np.polyfit(Qe, Ee, 4)
                stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = coeff_e

            pk_df = st.session_state[f"peak_{idx}"]
            peaks = []
            for _, r in pk_df.iterrows():
                loc, elev_pk = float(r[0]), float(r[1])
                if loc < 0 or loc > stn['L'] or elev_pk < stn['elev']:
                    st.error("Invalid peak location or elevation.")
                    st.stop()
                peaks.append({'loc': loc, 'elev': elev_pk})
            stn['peaks'] = peaks

        res = solve_pipeline(stations_data, term_data, FLOW, RateDRA, Price_HSD)
        st.session_state['res'] = res
        st.session_state['stations_data'] = stations_data

# Sidebar: view selector
view = st.sidebar.radio("Show results for:", [
    "Summary", "Cost Breakdown", "Performance",
    "System Curves", "Pump-System Interaction", "Cost Landscape", "Nonconvex Visuals"
])

if 'res' in st.session_state:
    res = st.session_state['res']
    stations_data = st.session_state['stations_data']

    # Scenario download
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Download Scenario**")
        df_glob = pd.DataFrame([{"FLOW":FLOW, "RateDRA":RateDRA, "Price_HSD":Price_HSD,
                                 "Terminal":terminal_name, "Term_Elev":terminal_elev,
                                 "Term_Head":terminal_head}])
        df_sta = pd.DataFrame(stations_data)
        bio = BytesIO()
        with pd.ExcelWriter(bio) as writer:
            df_glob.to_excel(writer, sheet_name="Global", index=False)
            df_sta.to_excel(writer, sheet_name="Stations", index=False)
        st.download_button("📥 Download .xlsx", bio.getvalue(), file_name="scenario.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("📥 Download Stations CSV", df_sta.to_csv(index=False).encode(), file_name="stations.csv", mime="text/csv")

    # rest of rendering remains unchanged...


    # Summary
    if view == "Summary":
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        names = [s['name'] for s in stations_data] + [terminal_name]
        rows  = ["Power+Fuel Cost","DRA Cost","No. Pumps","Pump Speed (rpm)",
                 "Pump Eff (%)","Reynolds","Head Loss (m)","Vel (m/s)",
                 "Residual Head (m)","SDH (m)","DRA (%)"]
        summary = {"Process":rows}
        for nm in names:
            key = nm.lower().replace(' ','_')
            summary[nm] = [
                res[f"power_cost_{key}"], res[f"dra_cost_{key}"],
                int(res[f"num_pumps_{key}"]), res[f"speed_{key}"],
                res[f"efficiency_{key}"], res[f"reynolds_{key}"],
                res[f"head_loss_{key}"], res[f"velocity_{key}"],
                res[f"residual_head_{key}"], res[f"sdh_{key}"],
                res[f"drag_reduction_{key}"]
            ]
        df_sum = pd.DataFrame(summary)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button("📥 CSV", df_sum.to_csv(index=False).encode(), file_name="results.csv")

    # Cost Breakdown
    elif view == "Cost Breakdown":
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        df_cost = pd.DataFrame({
            "Station": [s['name'] for s in stations_data],
            "Power+Fuel": [res[f"power_cost_{s['name'].lower().replace(' ','_')}"] for s in stations_data],
            "DRA":       [res[f"dra_cost_{s['name'].lower().replace(' ','_')}"]       for s in stations_data]
        })
        fig = px.bar(
            df_cost.melt(id_vars="Station", value_vars=["Power+Fuel","DRA"],
                         var_name="Type", value_name="INR/day"),
            x="Station", y="INR/day", color="Type",
            title="Daily Cost by Station"
        )
        fig.update_layout(yaxis_title="Cost (INR)")
        st.plotly_chart(fig, use_container_width=True)

    # Performance
    elif view == "Performance":
        tab_hl, tab_vr, tab_hq, tab_eq, tab_ps, tab_pf = st.tabs([
            "Head Loss","Vel & Re","Pump H–Q","Pump η–Q","Power vs Speed","Power vs Flow"
        ])

        with tab_hl:
            st.markdown("<div class='section-title'>Head Loss per Segment</div>", unsafe_allow_html=True)
            df_hl = pd.DataFrame({
                "Station":[s['name'] for s in stations_data],
                "Head Loss":[res[f"head_loss_{s['name'].lower().replace(' ','_')}"] for s in stations_data]
            })
            fig_hl = go.Figure(go.Bar(x=df_hl["Station"], y=df_hl["Head Loss"]))
            fig_hl.update_layout(yaxis_title="Head Loss (m)")
            st.plotly_chart(fig_hl, use_container_width=True)

        with tab_vr:
            st.markdown("<div class='section-title'>Velocity & Reynolds</div>", unsafe_allow_html=True)
            df_vr = pd.DataFrame({
                "Station":[s['name'] for s in stations_data],
                "Velocity (m/s)":[res[f"velocity_{s['name'].lower().replace(' ','_')}"] for s in stations_data],
                "Reynolds":[res[f"reynolds_{s['name'].lower().replace(' ','_')}"] for s in stations_data]
            })
            st.dataframe(df_vr.style.format({"Velocity (m/s)":"{:.2f}","Reynolds":"{:.0f}"}))

        with tab_hq:
            st.markdown("<div class='section-title'>Pump Characteristic Curves (H vs Q)</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                A,B,C = stn['A'],stn['B'],stn['C']
                dol = res[f"dol_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                fig = go.Figure()
                for rpm in np.arange(stn['MinRPM'], dol+1, 100):
                    fig.add_trace(go.Scatter(
                        x=flows, y=(A*flows**2 + B*flows + C)*(rpm/dol)**2,
                        mode="lines", name=f"{rpm} rpm"
                    ))
                fig.update_layout(title=stn['name'], xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
                st.plotly_chart(fig, use_container_width=True)

        with tab_eq:
            st.markdown("<div class='section-title'>Pump Efficiency Curves (η vs Q)</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                P,Q,R,S,T = stn['P'],stn['Q'],stn['R'],stn['S'],stn['T']
                dol = res[f"dol_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                fig = go.Figure()
                for rpm in np.arange(stn['MinRPM'], dol+1, 100):
                    feq = FLOW * dol / rpm
                    η = (P*feq**4 + Q*feq**3 + R*feq**2 + S*feq + T)/100.0
                    fig.add_trace(go.Scatter(x=flows, y=[η]*len(flows), mode="lines", name=f"{rpm} rpm"))
                fig.update_layout(title=stn['name'], xaxis_title="Flow (m³/hr)", yaxis_title="Efficiency (%)")
                st.plotly_chart(fig, use_container_width=True)

        with tab_ps:
            st.markdown("<div class='section-title'>Power (kW) vs Pump Speed</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                dol = res[f"dol_{key}"]; num = res[f"num_pumps_{key}"]
                rpms = np.arange(stn['MinRPM'], dol+1, 100)
                rates = []
                for rpm in rpms:
                    H = (stn['A']*FLOW**2 + stn['B']*FLOW + stn['C'])*(rpm/dol)**2
                    feq = FLOW * dol / rpm
                    η = (stn['P']*feq**4 + stn['Q']*feq**3 + stn['R']*feq**2 + stn['S']*feq + stn['T'])/100.0
                    pwr = (stn['rho']*FLOW*9.81*H*num)/(3600*1000*η*0.95)
                    rates.append(pwr)
                fig = go.Figure(go.Scatter(x=rpms, y=rates, mode="lines+markers"))
                fig.update_layout(title=stn['name'], xaxis_title="RPM", yaxis_title="Power (kW)")
                st.plotly_chart(fig, use_container_width=True)

        with tab_pf:
            st.markdown("<div class='section-title'>Power (kW) vs Flow</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                speed = res[f"speed_{key}"]; num = res[f"num_pumps_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                powers = []
                for q in flows:
                    H = (stn['A']*q**2 + stn['B']*q + stn['C'])*(speed/res[f"dol_{key}"])**2
                    feq = q * res[f"dol_{key}"] / speed
                    η = (stn['P']*feq**4 + stn['Q']*feq**3 + stn['R']*feq**2 + stn['S']*feq + stn['T'])/100.0
                    pwr = (stn['rho']*q*9.81*H*num)/(3600*1000*η*0.95)
                    powers.append(pwr)
                fig = go.Figure(go.Scatter(x=flows, y=powers, mode="lines"))
                fig.update_layout(title=stn['name'], xaxis_title="Flow (m³/hr)", yaxis_title="Power (kW)")
                st.plotly_chart(fig, use_container_width=True)

    # System Curves
    elif view == "System Curves":
        st.markdown("<div class='section-title'>System Head Curves</div>", unsafe_allow_html=True)
        for stn in stations_data:
            if not stn['is_pump']: continue
            key    = stn['name'].lower().replace(' ','_')
            d_i    = stn['D'] - 2*stn['t']
            rough  = stn['rough']
            Lseg   = stn['L']
            elev_i = stn['elev']
            kv     = stn['KV']
            curves = []
            for dra in np.arange(0, stn['max_dr']+1, 5):
                v  = np.linspace(0, FLOW, 101)/3600.0/(pi*d_i**2/4)
                Re = v*d_i/(kv*1e-6) if kv>0 else np.zeros_like(v)
                f  = np.where(Re>0, 0.25/(np.log10(rough/d_i/3.7 + 5.74/Re**0.9)**2), 0)
                DH = f*((Lseg*1000)/d_i)*(v**2/(2*9.81))*(1-dra/100)
                curves.append(pd.DataFrame({"Flow":np.linspace(0,FLOW,101),"SDH":elev_i+DH,"DR":dra}))
            df_sys = pd.concat(curves)
            fig = px.line(df_sys, x="Flow", y="SDH", color="DR", title=stn['name'])
            fig.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig, use_container_width=True)

    # Pump-System Interaction
    elif view == "Pump-System Interaction":
        st.markdown("<div class='section-title'>Pump vs System Interaction</div>", unsafe_allow_html=True)
        for stn in stations_data:
            if not stn['is_pump']: continue
            key = stn['name'].lower().replace(' ','_')
            A,B,C = stn['A'],stn['B'],stn['C']
            min_r  = res[f"min_rpm_{key}"]; dol = res[f"dol_{key}"]
            num    = res[f"num_pumps_{key}"]
            kv     = stn['KV']; rough=stn['rough']; Lseg=stn['L']; elev=stn['elev']
            flows = np.linspace(0, FLOW*1.5, 200)
            fig_int = go.Figure()
            for dra in np.arange(0, stn['max_dr']+1, 5):
                v  = flows/3600.0/(pi*(stn['D']-2*stn['t'])**2/4)
                Re = v*(stn['D']-2*stn['t'])/(kv*1e-6) if kv>0 else np.zeros_like(v)
                f  = np.where(Re>0, 0.25/(np.log10(rough/(stn['D']-2*stn['t'])/3.7 + 5.74/Re**0.9)**2), 0)
                DH = f*((Lseg*1000)/(stn['D']-2*stn['t']))*(v**2/(2*9.81))*(1-dra/100)
                fig_int.add_trace(go.Scatter(x=flows, y=elev+DH, mode="lines", name=f"Sys {dra}% DRA"))
            for rpm in np.arange(min_r, dol+1, 100):
                Hp = (A*flows**2 + B*flows + C)*(rpm/dol)**2
                fig_int.add_trace(go.Scatter(x=flows, y=Hp, mode="lines", name=f"Pump {rpm} rpm"))
            fig_int.update_layout(xaxis_title="Flow", yaxis_title="Head")
            st.plotly_chart(fig_int, use_container_width=True)

    # Cost Landscape (pumps × RPM)
    elif view == "Cost Landscape":
        st.markdown("<div class='section-title'>Cost vs Pumps & Speed (fixed DRA)</div>", unsafe_allow_html=True)
        # pick first pumping station as example
        for stn in stations_data:
            if not stn['is_pump']: continue
            key   = stn['name'].lower().replace(' ','_')
            A,B,C = stn['A'],stn['B'],stn['C']
            P,Q,R,S,T = stn['P'],stn['Q'],stn['R'],stn['S'],stn['T']
            kv     = stn['KV']; rho=stn['rho']
            rate   = stn['rate'] if stn['power_type']=="Grid" else (stn['sfc']*1.34102/820 * Price_HSD)
            maxp   = stn['max_pumps']
            dr_opt = res[f"drag_reduction_{key}"]
            Qvol   = FLOW
            rpms   = np.arange(stn['MinRPM'], stn['DOL']+1, 100)
            pumps  = np.arange(1, maxp+1)
            Z = np.zeros((len(pumps), len(rpms)))

            for i, npumps in enumerate(pumps):
                for j, rpm in enumerate(rpms):
                    H    = (A*Qvol**2 + B*Qvol + C)*(rpm/stn['DOL'])**2
                    feq  = Qvol * stn['DOL'] / rpm
                    η    = (P*feq**4 + Q*feq**3 + R*feq**2 + S*feq + T)/100.0
                    pwr  = (rho * Qvol * 9.81 * H * npumps)/(3600*1000*η*0.95)
                    cost = pwr*24*rate + (dr_opt/4)*(Qvol*1000*24/1e6)*RateDRA
                    Z[i,j] = cost

            from plotly import graph_objects as go3d
            surf = go3d.Surface(x=rpms, y=pumps, z=Z)
            fig3 = go3d.Figure(data=[surf])
            fig3.update_layout(
                title=f"{stn['name']}: Cost vs Pumps & RPM",
                scene=dict(xaxis_title="RPM", yaxis_title="No. Pumps", zaxis_title="Cost (INR/day)")
            )
            st.plotly_chart(fig3, use_container_width=True)
            break  # only first station

# If no results yet:
else:
    st.info("🔹 Enter inputs and click 🚀 Run Optimization to see results.")

# Footer
st.markdown("---")
st.markdown("© 2025 Parichay Das. All rights reserved.")
