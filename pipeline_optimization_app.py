# pipeline_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import pi
from io import BytesIO
from pyomo.opt import SolverManagerFactory

# NEOS email (stored in Streamlit secrets)
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

st.set_page_config(page_title="Pipeline Optimization", layout="wide")

st.markdown("""
<style>
.section-title {
  font-size:1.2rem; font-weight:600; margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Mixed Integer Nonlinear Optimization of Pipeline Operations</h1>", unsafe_allow_html=True)

# Solver call
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

#
# --- Sidebar inputs --------------------------------------------------------
#
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

# Initialize station list if not present
if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200.0, 'DOL': 1500.0,
        'max_dr': 0.0
    }]

# Station inputs (dynamic)
for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}", expanded=True):
        stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{idx}")
        stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'],
                                      step=0.1, key=f"elev{idx}")

        # Station-specific viscosity & density
        stn['KV']  = st.number_input("Viscosity (cSt)",
                                     value=stn.get('KV', 10.0),
                                     step=0.1, key=f"kv_{idx}")
        stn['rho'] = st.number_input("Density (kg/m³)",
                                     value=stn.get('rho', 850.0),
                                     step=1.0, key=f"rho_{idx}")

        if idx == 1:
            stn['min_residual'] = st.number_input(
                "Residual Head at Station (m)",
                value=stn.get('min_residual',50.0),
                step=0.1, key=f"res{idx}"
            )
        stn['D'] = st.number_input("Outer Diameter (m)", value=stn['D'],
                                   format="%.3f", step=0.001, key=f"D{idx}")
        stn['t'] = st.number_input("Wall Thickness (m)", value=stn['t'],
                                   format="%.4f", step=0.0001, key=f"t{idx}")
        stn['SMYS'] = st.number_input("SMYS (psi)", value=stn['SMYS'],
                                      step=1000.0, key=f"SMYS{idx}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'],
                                       format="%.5f", step=1e-5, key=f"rough{idx}")
        stn['L'] = st.number_input("Length to next (km)", value=stn['L'],
                                   step=1.0, key=f"L{idx}")
        stn['is_pump'] = st.checkbox("Pumping Station?",
                                     value=stn['is_pump'], key=f"pump{idx}")

        if stn['is_pump']:
            stn['power_type'] = st.selectbox(
                "Power Source", ["Grid", "Diesel"],
                index=0 if stn['power_type']=="Grid" else 1,
                key=f"ptype{idx}"
            )
            if stn['power_type']=="Grid":
                stn['rate'] = st.number_input(
                    "Electricity Rate (INR/kWh)", value=stn.get('rate',9.0),
                    key=f"rate{idx}"
                )
                stn['sfc'] = 0.0
            else:
                stn['sfc'] = st.number_input(
                    "SFC (gm/bhp·hr)", value=stn.get('sfc',150.0),
                    key=f"sfc{idx}"
                )
                stn['rate'] = 0.0

            stn['max_pumps'] = st.number_input(
                "Max Pumps Available", min_value=1,
                value=stn['max_pumps'], step=1, key=f"mpumps{idx}"
            )
            stn['MinRPM'] = st.number_input(
                "Min RPM", value=stn['MinRPM'], key=f"minrpm{idx}"
            )
            stn['DOL'] = st.number_input(
                "Rated RPM (DOL)", value=stn['DOL'], key=f"dol{idx}"
            )
            stn['max_dr'] = st.number_input(
                "Max Drag Reduction (%)", value=stn['max_dr'], key=f"mdr{idx}"
            )

            st.markdown("**Enter Pump Performance Data:**")
            st.write("Flow vs Head data (m³/hr, m)")
            df_head = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
            df_head = st.data_editor(df_head, num_rows="dynamic", key=f"head{idx}")

            st.write("Flow vs Efficiency data (m³/hr, %)")
            df_eff = pd.DataFrame({"Flow (m³/hr)": [0.0], "Efficiency (%)": [0.0]})
            df_eff = st.data_editor(df_eff, num_rows="dynamic", key=f"eff{idx}")

            # store tables
            st.session_state[f"head_data_{idx}"] = df_head
            st.session_state[f"eff_data_{idx}"] = df_eff

        st.markdown("**Intermediate Elevation Peaks (to next station):**")
        default_peak = pd.DataFrame({
            "Location (km)": [stn['L']/2.0],
            "Elevation (m)": [stn['elev']+100.0]
        })
        peak_df = st.data_editor(default_peak, num_rows="dynamic", key=f"peak{idx}")
        st.session_state[f"peak_data_{idx}"] = peak_df

# Terminal inputs
st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", value="Terminal")
terminal_elev = st.number_input("Elevation (m)", value=0.0, step=0.1)
terminal_head = st.number_input("Required Residual Head (m)",
                                value=50.0, step=1.0)

# Run optimization
run = st.button("🚀 Run Optimization")

if run:
    with st.spinner("Solving optimization..."):
        stations_data = st.session_state.stations
        term_data = {
            "name": terminal_name,
            "elev": terminal_elev,
            "min_residual": terminal_head
        }

        # attach pump curves & peaks
        for idx, stn in enumerate(stations_data, start=1):
            if stn.get('is_pump', False):
                dfh = st.session_state.get(f"head_data_{idx}")
                dfe = st.session_state.get(f"eff_data_{idx}")
                if dfh is None or dfe is None or len(dfh)<3 or len(dfe)<5:
                    st.error(f"Station {idx}: Need ≥3 head & ≥5 eff points.")
                    st.stop()
                Qh, Hh = dfh.iloc[:,0].values, dfh.iloc[:,1].values
                coeff = np.polyfit(Qh, Hh, 2)
                stn['A'], stn['B'], stn['C'] = coeff

                Qe, Ee = dfe.iloc[:,0].values, dfe.iloc[:,1].values
                coeff_e = np.polyfit(Qe, Ee, 4)
                stn['P'], stn['Q'], stn['R'], stn['S'], stn['T'] = coeff_e

            peaks_df = st.session_state.get(f"peak_data_{idx}")
            pl = []
            if peaks_df is not None:
                for _, row in peaks_df.iterrows():
                    try:
                        loc = float(row["Location (km)"])
                        elev_pk = float(row["Elevation (m)"])
                    except:
                        continue
                    if not (0 <= loc <= stn['L']) or elev_pk < stn['elev']:
                        st.error(f"Station {idx}: Invalid peak.")
                        st.stop()
                    pl.append({'loc':loc, 'elev':elev_pk})
            stn['peaks'] = pl

        # solve
        res = solve_pipeline(stations_data, term_data, FLOW, RateDRA, Price_HSD)

    # show summary metrics
    total_cost   = res['total_cost']
    total_pumps  = sum(int(res[f"num_pumps_{s['name'].lower().replace(' ','_')}"]) for s in stations_data)
    speeds       = [res[f"speed_{s['name'].lower().replace(' ','_')}"] for s in stations_data]
    effs         = [res[f"efficiency_{s['name'].lower().replace(' ','_')}"] for s in stations_data]
    avg_speed    = np.mean([sp for sp in speeds if sp]) if speeds else 0
    avg_efficiency = np.mean([ef for ef in effs if ef]) if effs else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost (INR)", f"₹{total_cost:,.2f}")
    c2.metric("Total Pumps", total_pumps)
    c3.metric("Avg Speed (rpm)", f"{avg_speed:.1f}")
    c4.metric("Avg Efficiency (%)", f"{avg_efficiency:.1f}")

    # sidebar view picker
    view = st.sidebar.radio("Show results for:", [
        "Summary", "Cost Breakdown", "Performance",
        "System Curves", "Pump-System Interaction", "Cost Landscape"
    ])

    # --- VIEW: SUMMARY ------------------------------------------------------
    if view == "Summary":
        st.markdown("<div class='section-title'>Optimization Results</div>", unsafe_allow_html=True)
        # build summary DataFrame
        names = [s['name'] for s in stations_data] + [terminal_name]
        rows  = ["Power+Fuel Cost","DRA Cost","No. Pumps","Pump Speed (rpm)",
                 "Pump Eff (%)","Reynolds","Head Loss (m)","Vel (m/s)",
                 "Residual Head (m)","SDH (m)","DRA (%)"]
        summary = {"Process": rows}
        for nm in names:
            key = nm.lower().replace(' ','_')
            summary[nm] = [
                res[f"power_cost_{key}"], res[f"dra_cost_{key}"],
                int(res[f"num_pumps_{key}"]), res[f"speed_{key}"],
                res[f"efficiency_{key}"], res[f"reynolds_{key}"],
                res[f"head_loss_{key}"],   res[f"velocity_{key}"],
                res[f"residual_head_{key}"], res[f"sdh_{key}"],
                res[f"drag_reduction_{key}"]
            ]
        df_sum = pd.DataFrame(summary)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button(
            "📥 Download CSV", df_sum.to_csv(index=False).encode(),
            file_name="results.csv"
        )

    # --- VIEW: COST BREAKDOWN ----------------------------------------------
    elif view == "Cost Breakdown":
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        df_cost = pd.DataFrame({
            "Station": [s['name'] for s in stations_data],
            "Power+Fuel": [res[f"power_cost_{s['name'].lower().replace(' ','_')}"] for s in stations_data],
            "DRA":       [res[f"dra_cost_{s['name'].lower().replace(' ','_')}"]        for s in stations_data]
        })
        fig = px.bar(
            df_cost.melt(id_vars="Station", value_vars=["Power+Fuel","DRA"],
                         var_name="Type", value_name="INR/day"),
            x="Station", y="INR/day", color="Type",
            title="Daily Cost by Station"
        )
        fig.update_layout(yaxis_title="Cost (INR)")
        st.plotly_chart(fig, use_container_width=True)

    # --- VIEW: PERFORMANCE --------------------------------------------------
    elif view == "Performance":
        perf_tab, head_tab, pump_tab, eff_tab, ps_tab, pf_tab = st.tabs([
            "Head Loss","Vel & Re","Pump H–Q","Pump η–Q","Power vs Speed","Power vs Flow"
        ])

        with perf_tab:
            st.markdown("<div class='section-title'>Head Loss per Segment</div>", unsafe_allow_html=True)
            df_h = pd.DataFrame({
                "Station": [s['name'] for s in stations_data],
                "Head Loss": [res[f"head_loss_{s['name'].lower().replace(' ','_')}"] for s in stations_data]
            })
            fig_h = go.Figure(go.Bar(x=df_h["Station"], y=df_h["Head Loss"]))
            fig_h.update_layout(yaxis_title="Head Loss (m)")
            st.plotly_chart(fig_h, use_container_width=True)

        with head_tab:
            st.markdown("<div class='section-title'>Velocity & Reynolds</div>", unsafe_allow_html=True)
            df_v = pd.DataFrame({
                "Station":[s['name'] for s in stations_data],
                "Velocity (m/s)":[res[f"velocity_{s['name'].lower().replace(' ','_')}"] for s in stations_data],
                "Reynolds":[res[f"reynolds_{s['name'].lower().replace(' ','_')}"] for s in stations_data]
            })
            st.dataframe(df_v.style.format({"Velocity (m/s)":"{:.2f}","Reynolds":"{:.0f}"}))

        with pump_tab:
            st.markdown("<div class='section-title'>Pump Characteristic (H vs Q)</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                A,B,C = stn['A'],stn['B'],stn['C']
                dol = res[f"dol_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                fig = go.Figure()
                for rpm in np.arange(stn["MinRPM"], dol+1, 100):
                    Hcurve = (A*flows**2 + B*flows + C)*(rpm/dol)**2
                    fig.add_trace(go.Scatter(x=flows,y=Hcurve,mode="lines",name=f"{rpm} rpm"))
                fig.update_layout(xaxis_title="Flow",yaxis_title="Head",title=stn['name'])
                st.plotly_chart(fig,use_container_width=True)

        with eff_tab:
            st.markdown("<div class='section-title'>Pump Efficiency (η vs Q)</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                P,Q,R,S,T= stn['P'],stn['Q'],stn['R'],stn['S'],stn['T']
                dol = res[f"dol_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                fig = go.Figure()
                for rpm in np.arange(stn["MinRPM"], dol+1, 100):
                    feq = FLOW * dol / rpm
                    η = (P*feq**4 + Q*feq**3 + R*feq**2 + S*feq + T)/100.0
                    fig.add_trace(go.Scatter(x=flows,y=[η]*len(flows),mode="lines",name=f"{rpm} rpm"))
                fig.update_layout(xaxis_title="Flow",yaxis_title="Efficiency",title=stn['name'])
                st.plotly_chart(fig,use_container_width=True)

        with ps_tab:
            st.markdown("<div class='section-title'>Power vs Speed</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                num = res[f"num_pumps_{key}"]
                dol = res[f"dol_{key}"]
                flows = FLOW
                rpms = np.arange(stn["MinRPM"], dol+1, 100)
                rates = []
                for rpm in rpms:
                    H = (stn['A']*flows**2 + stn['B']*flows + stn['C'])*(rpm/dol)**2
                    feq = flows * dol / rpm
                    η = (stn['P']*feq**4 + stn['Q']*feq**3 + stn['R']*feq**2 + stn['S']*feq + stn['T'])/100.0
                    pwr = (stn['rho']*flows*9.81*H*num)/(3600*1000*η*0.95)
                    rates.append(pwr)
                fig = go.Figure(go.Scatter(x=rpms,y=rates,mode="lines+markers"))
                fig.update_layout(xaxis_title="RPM",yaxis_title="Power",title=stn['name'])
                st.plotly_chart(fig,use_container_width=True)

        with pf_tab:
            st.markdown("<div class='section-title'>Power vs Flow</div>", unsafe_allow_html=True)
            for stn in stations_data:
                if not stn['is_pump']: continue
                key = stn['name'].lower().replace(' ','_')
                num = res[f"num_pumps_{key}"]
                speed = res[f"speed_{key}"]
                flows = np.linspace(0, FLOW*1.2, 100)
                powers = []
                for q in flows:
                    H = (stn['A']*q**2 + stn['B']*q + stn['C'])*(speed/res[f"dol_{key}"])**2
                    feq = q * res[f"dol_{key}"] / speed
                    η = (stn['P']*feq**4 + stn['Q']*feq**3 + stn['R']*feq**2 + stn['S']*feq + stn['T'])/100.0
                    pwr = (stn['rho']*q*9.81*H*num)/(3600*1000*η*0.95)
                    powers.append(pwr)
                fig = go.Figure(go.Scatter(x=flows,y=powers,mode="lines"))
                fig.update_layout(xaxis_title="Flow",yaxis_title="Power",title=stn['name'])
                st.plotly_chart(fig,use_container_width=True)

    # --- VIEW: SYSTEM CURVES ------------------------------------------------
    elif view == "System Curves":
        st.markdown("<div class='section-title'>System Head Curves</div>", unsafe_allow_html=True)
        for stn in stations_data:
            if not stn['is_pump']: continue
            key     = stn['name'].lower().replace(' ','_')
            d_i     = stn['D'] - 2*stn['t']
            rough   = stn['rough']
            Lseg    = stn['L']
            elev_i  = stn['elev']
            kv      = stn['KV']
            curves = []
            for dra in np.arange(0, stn['max_dr']+1, 5):
                v  = np.linspace(0, FLOW, 101)/3600.0/(pi*d_i**2/4)
                Re = v*d_i/(kv*1e-6) if kv>0 else np.zeros_like(v)
                f  = np.where(Re>0, 0.25/(np.log10(rough/d_i/3.7+5.74/Re**0.9)**2),0)
                DH = f*((Lseg*1000)/d_i)*(v**2/(2*9.81))*(1-dra/100)
                curves.append(pd.DataFrame({"Flow":np.linspace(0,FLOW,101),"SDH":elev_i+DH,"DR":dra}))
            df_sys = pd.concat(curves)
            fig   = px.line(df_sys, x="Flow", y="SDH", color="DR", title=stn['name'])
            fig.update_layout(yaxis_title="Static+Dyn Head (m)")
            st.plotly_chart(fig,use_container_width=True)

    # --- VIEW: PUMP-SYSTEM INTERACTION --------------------------------------
    elif view == "Pump-System Interaction":
        st.markdown("<div class='section-title'>Pump vs System Interaction</div>", unsafe_allow_html=True)
        for stn in stations_data:
            if not stn['is_pump']: continue
            key   = stn['name'].lower().replace(' ','_')
            A,B,C = stn['A'],stn['B'],stn['C']
            min_r = res[f"min_rpm_{key}"]; dol = res[f"dol_{key}"]
            nv    = res[f"num_pumps_{key}"]
            kv    = stn['KV']; rough = stn['rough']; Lseg = stn['L']; elev = stn['elev']
            flows = np.linspace(0, FLOW*1.5, 200)
            fig_int = go.Figure()
            # system curves
            for dra in np.arange(0, stn['max_dr']+1, 5):
                v  = flows/3600.0/(pi*(stn['D']-2*stn['t'])**2/4)
                Re = v*(stn['D']-2*stn['t'])/(kv*1e-6) if kv>0 else np.zeros_like(v)
                f  = np.where(Re>0, 0.25/(np.log10(rough/(stn['D']-2*stn['t'])/3.7+5.74/Re**0.9)**2),0)
                DH = f*((Lseg*1000)/(stn['D']-2*stn['t']))*(v**2/(2*9.81))*(1-dra/100)
                fig_int.add_trace(go.Scatter(x=flows,y=elev+DH,mode="lines",name=f"Sys {dra}% DRA"))
            # pump curves
            for rpm in np.arange(min_r, dol+1, 100):
                Hp = (A*flows**2 + B*flows + C)*(rpm/dol)**2
                fig_int.add_trace(go.Scatter(x=flows,y=Hp,mode="lines",name=f"Pump {rpm} rpm"))
            fig_int.update_layout(xaxis_title="Flow",yaxis_title="Head")
            st.plotly_chart(fig_int,use_container_width=True)

    # --- VIEW: COST LANDSCAPE -----------------------------------------------
    elif view == "Cost Landscape":
        st.markdown("<div class='section-title'>Cost Landscape (RPM vs DRA)</div>", unsafe_allow_html=True)
        for stn in stations_data:
            if not stn['is_pump']: continue
            key  = stn['name'].lower().replace(' ','_')
            A,B,C = stn['A'],stn['B'],stn['C']
            P,Q,R,S,T = stn['P'],stn['Q'],stn['R'],stn['S'],stn['T']
            num  = res[f"num_pumps_{key}"]
            min_r= res[f"min_rpm_{key}"]; dol = res[f"dol_{key}"]
            rho  = stn['rho']
            rate = stn['rate'] if stn['power_type']=="Grid" else (stn['sfc']*1.34102/820*Price_HSD)
            drs  = np.arange(0, stn['max_dr']+1, 5)
            rpms = np.arange(min_r, dol+1, 100)
            Z    = np.zeros((len(drs), len(rpms)))
            for i, dra in enumerate(drs):
                for j, rpm in enumerate(rpms):
                    H   = (A*FLOW**2 + B*FLOW + C)*(rpm/dol)**2
                    feq = FLOW * dol / rpm
                    η   = (P*feq**4 + Q*feq**3 + R*feq**2 + S*feq + T)/100.0
                    pwr = (rho*FLOW*9.81*H*num)/(3600*1000*η*0.95)
                    e_c = pwr*24*rate
                    dra_cost = (dra/4)*(FLOW*1000*24/1e6)*RateDRA
                    Z[i,j] = e_c + dra_cost
            from plotly import graph_objects as go3d
            surf = go3d.Surface(x=rpms, y=drs, z=Z)
            fig3 = go3d.Figure(data=[surf])
            fig3.update_layout(
                title=f"{stn['name']}: Cost vs RPM vs DRA",
                scene=dict(xaxis_title="RPM", yaxis_title="DRA (%)", zaxis_title="Cost")
            )
            st.plotly_chart(fig3, use_container_width=True)

# COPYRIGHT
st.markdown("---")
st.markdown("© 2025 Parichay Das. All rights reserved.")
