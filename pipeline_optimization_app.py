import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly import graph_objects as go3d
from math import pi
from pyomo.opt import SolverManagerFactory

# Configure NEOS email
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

# Page layout
st.set_page_config(page_title="Pipeline Optimization", layout="wide")

# CSS styling
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

st.markdown("# Mixed Integer Nonlinear Pipeline Optimization", unsafe_allow_html=True)

# Backend solver wrapper
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

# Sidebar: inputs and view
with st.sidebar:
    st.title("🔧 Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
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
    view = st.radio(
        "Select View:",
        ["Summary", "Cost Breakdown", "Performance", "System Curves",
         "Pump-System Interaction", "Cost Landscape", "Nonconvex Visuals"]
    )

# Initialize station list
if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name': 'Station 1', 'elev': 0.0, 'D': 0.711, 't': 0.007,
        'SMYS': 52000.0, 'rough': 0.00004, 'L': 50.0,
        'min_residual': 50.0, 'is_pump': False,
        'power_type': 'Grid', 'rate': 9.0, 'sfc': 150.0,
        'max_pumps': 1, 'MinRPM': 1200, 'DOL': 1500, 'max_dr': 0
    }]

# Station configuration
for i, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {i}"):
        stn['name'] = st.text_input("Name", value=stn['name'], key=f"name{i}")
        stn['elev'] = st.number_input("Elevation (m)", value=stn['elev'], step=0.1, key=f"elev{i}")
        stn['KV']   = st.number_input("Viscosity (cSt)", value=stn.get('KV',10.0), step=0.1, key=f"kv{i}")
        stn['rho']  = st.number_input("Density (kg/m³)", value=stn.get('rho',850.0), step=1.0, key=f"rho{i}")
        if i == 1:
            stn['min_residual'] = st.number_input(
                "Required Residual Head (m)",
                value=stn['min_residual'], step=0.1, key=f"res{i}"
            )
        stn['D']     = st.number_input("Outer Diameter (m)", value=stn['D'], format="%.3f", step=0.001, key=f"D{i}")
        stn['t']     = st.number_input("Wall Thickness (m)", value=stn['t'], format="%.4f", step=1e-4, key=f"t{i}")
        stn['SMYS']  = st.number_input("SMYS (psi)", value=stn['SMYS'], step=1000.0, key=f"SMYS{i}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", value=stn['rough'], format="%.5f", step=1e-5, key=f"rough{i}")
        stn['L']     = st.number_input("Length to next (km)", value=stn['L'], min_value=0.0, step=1.0, key=f"L{i}")
        stn['is_pump'] = st.checkbox("Pumping Station?", value=stn['is_pump'], key=f"pump{i}")
        if stn['is_pump']:
            stn['power_type'] = st.selectbox("Power Source", ["Grid","Diesel"], index=0 if stn['power_type']=='Grid' else 1, key=f"ptype{i}")
            if stn['power_type'] == 'Grid':
                stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", value=stn['rate'], key=f"rate{i}")
                stn['sfc']  = 0.0
            else:
                stn['sfc']  = st.number_input("SFC (gm/bhp·hr)", value=stn['sfc'], key=f"sfc{i}")
                stn['rate'] = 0.0
            stn['max_pumps'] = st.number_input("Maximum Pumps Available", min_value=1, value=stn['max_pumps'], step=1, key=f"mpumps{i}")
            stn['MinRPM'] = st.number_input("Min RPM", value=stn['MinRPM'], key=f"minrpm{i}")
            stn['DOL']    = st.number_input("Rated RPM (DOL)", value=stn['DOL'], key=f"dol{i}")
            stn['max_dr'] = st.number_input("Max Drag Reduction (%)", value=stn['max_dr'], key=f"mdr{i}")
            # Pump performance curves
            dfh = st.data_editor(pd.DataFrame({"Flow": [0.0], "Head": [0.0]}), num_rows='dynamic', key=f"head{i}")
            st.session_state[f"head{i}"] = dfh
            dfe = st.data_editor(pd.DataFrame({"Flow": [0.0], "Eff": [0.0]}), num_rows='dynamic', key=f"eff{i}")
            st.session_state[f"eff{i}"] = dfe
        # Elevation peaks
        peaks_df = st.data_editor(pd.DataFrame({"Location": [stn['L']/2], "Elevation": [stn['elev']+100]}), num_rows='dynamic', key=f"peak{i}")
        st.session_state[f"peak{i}"] = peaks_df

# Terminal configuration
st.markdown("---")
st.subheader("🏁 Terminal Station")
term_name = st.text_input("Name", "Terminal")
term_elev = st.number_input("Elevation (m)", value=0.0)
term_min  = st.number_input("Required Residual Head (m)", value=50.0)

# Run optimization
if st.button("🚀 Run Optimization"):
    with st.spinner("Solving..."):
        stations_data = st.session_state.stations
        for i, stn in enumerate(stations_data, start=1):
            if stn['is_pump']:
                Qh, Hh = st.session_state[f"head{i}"].values.T
                A, B, C = np.polyfit(Qh, Hh, 2)
                stn.update({'A': A, 'B': B, 'C': C})
                Qe, Ee = st.session_state[f"eff{i}"].values.T
                P, Qc, R, S, T = np.polyfit(Qe, Ee, 4)
                stn.update({'P': P, 'Q': Qc, 'R': R, 'S': S, 'T': T})
            stn['peaks'] = [{'loc': r[0], 'elev': r[1]} for _, r in st.session_state[f"peak{i}"].iterrows()]
        res = solve_pipeline(stations_data, {'name': term_name, 'elev': term_elev, 'min_residual': term_min}, FLOW, RateDRA, Price_HSD)
        st.session_state['res'] = res
        st.session_state['stations_data'] = stations_data

# Display results based on selected view
if 'res' in st.session_state:
    res = st.session_state['res']
    sta = st.session_state['stations_data']
    # Download CSVs
    with st.sidebar:
        st.markdown("---")
        df0 = pd.DataFrame([{'FLOW':FLOW,'RateDRA':RateDRA,'Price_HSD':Price_HSD,'Terminal':term_name,'Elev':term_elev,'MinRH':term_min}])
        st.download_button("Download Scenario CSV", df0.to_csv(index=False).encode(), "scenario.csv", "text/csv")
        st.download_button("Download Stations CSV", pd.DataFrame(sta).to_csv(index=False).encode(), "stations.csv", "text/csv")
    # Summary
    if view == "Summary":
        st.markdown("<div class='section-title'>Summary</div>", unsafe_allow_html=True)
        names = [s['name'] for s in sta] + [term_name]
        rows  = ["Cost+Fuel","DRA Cost","No. Pumps","Speed","Eff","Re","Head Loss","Vel","Res Head","SDH","DRA%"]
        data  = {"Process": rows}
        for nm in names:
            k = nm.lower().replace(' ','_')
            data[nm] = [res[f"power_cost_{k}"], res[f"dra_cost_{k}"], res[f"num_pumps_{k}"], res[f"speed_{k}"], res[f"efficiency_{k}"], res[f"reynolds_{k}"], res[f"head_loss_{k}"], res[f"velocity_{k}"], res[f"residual_head_{k}"], res[f"sdh_{k}"], res[f"drag_reduction_{k}"]]
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    # Cost Breakdown
    elif view == "Cost Breakdown":
        st.markdown("<div class='section-title'>Cost Breakdown</div>", unsafe_allow_html=True)
        dfc = pd.DataFrame({"Station":[s['name'] for s in sta], "Cost":[res[f"power_cost_{s['name'].lower().replace(' ','_')}" ] for s in sta], "DRA":[res[f"dra_cost_{s['name'].lower().replace(' ','_')}" ] for s in sta]})
        fig = px.bar(dfc.melt(id_vars="Station", value_vars=["Cost","DRA"], var_name="Type", value_name="INR/day"), x="Station", y="INR/day", color="Type")
        st.plotly_chart(fig, use_container_width=True)
    # Performance Views
    elif view == "Performance":
        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Head Loss","Vel & Re","Pump H-Q","Pump Eff","Power-Speed","Power-Flow"])
        with tab1:
            dfhl = pd.DataFrame({"Station":[s['name'] for s in sta], "H Loss":[res[f"head_loss_{s['name'].lower().replace(' ','_')}" ] for s in sta]})
            fig = go.Figure(go.Bar(x=dfhl["Station"], y=dfhl["H Loss"]))
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            dfvr = pd.DataFrame({"Station":[s['name'] for s in sta], "Vel":[res[f"velocity_{s['name'].lower().replace(' ','_')}" ] for s in sta], "Re":[res[f"reynolds_{s['name'].lower().replace(' ','_')}" ] for s in sta]})
            st.dataframe(dfvr)
        with tab3:
            for s in sta:
                if not s['is_pump']: continue
                flows = np.linspace(0,FLOW*1.2,100)
                fig = go.Figure()
                for rpm in range(s['MinRPM'],s['DOL']+1,100):
                    H = (s['A']*flows**2 + s['B']*flows + s['C'])*(rpm/s['DOL'])**2
                    fig.add_trace(go.Scatter(x=flows,y=H,mode='lines',name=f"{rpm} rpm"))
                st.plotly_chart(fig, use_container_width=True)
        with tab4:
            for s in sta:
                if not s['is_pump']: continue
                flows = np.linspace(0,FLOW*1.2,100)
                fig = go.Figure()
                for rpm in range(s['MinRPM'],s['DOL']+1,100):
                    fe = FLOW*s['DOL']/rpm
                    η = (s['P']*fe**4 + s['Q']*fe**3 + s['R']*fe**2 + s['S']*fe + s['T'])/100
                    fig.add_trace(go.Scatter(x=flows,y=[η]*len(flows),mode='lines',name=f"{rpm} rpm"))
                st.plotly_chart(fig, use_container_width=True)
        with tab5:
            for s in sta:
                if not s['is_pump']: continue
                rpms = range(s['MinRPM'],s['DOL']+1,100)
                rates=[]
                for rpm in rpms:
                    H = (s['A']*FLOW**2 + s['B']*FLOW + s['C'])*(rpm/s['DOL'])**2
                    fe = FLOW*s['DOL']/rpm
                    η = (s['P']*fe**4 + s['Q']*fe**3 + s['R']*fe**2 + s['S']*fe + s['T'])/100
                    pwr = (s['rho']*FLOW*9.81*H*s['max_pumps'])/(3600*1000*η*0.95)
                    rates.append(pwr)
                fig = go.Figure(go.Scatter(x=list(rpms),y=rates,mode='lines+markers'))
                st.plotly_chart(fig, use_container_width=True)
        with tab6:
            for s in sta:
                if not s['is_pump']: continue
                flows = np.linspace(0,FLOW*1.2,100)
                powers=[]
                speed = res[f"speed_{s['name'].lower().replace(' ','_')}" ]
                for q in flows:
                    H = (s['A']*q**2 + s['B']*q + s['C'])*(speed/s['DOL'])**2
                    fe = q*s['DOL']/speed
                    η = (s['P']*fe**4 + s['Q']*fe**3 + s['R']*fe**2 + s['S']*fe + s['T'])/100
                    pwr = (s['rho']*q*9.81*H*s['max_pumps'])/(3600*1000*η*0.95)
                    powers.append(pwr)
                fig = go.Figure(go.Scatter(x=flows,y=powers,mode='lines'))
                st.plotly_chart(fig, use_container_width=True)
    # System Curves
    elif view == "System Curves":
        st.markdown("<div class='section-title'>System Head Curves</div>", unsafe_allow_html=True)
        for s in sta:
            if not s['is_pump']: continue
            d=s['D']-2*s['t']; kv=s['KV']; rough=s['rough']; L=s['L']; elev=s['elev']
            dfs=[]
            for dra in range(0,s['max_dr']+1,5):
                v=np.linspace(0,FLOW,100)/3600/(pi*(d**2)/4)
                Re=v*d/(kv*1e-6)
                f=np.where(Re>4000,0.25/(np.log10(rough/d/3.7+5.74/(Re**0.9))**2),64/Re)
                DH=f*((L*1000)/d)*(v**2/(2*9.81))*(1-dra/100)
                dfs.append(pd.DataFrame({"Flow":np.linspace(0,FLOW,100),"SDH":elev+DH,"DRA":dra}))
            dfc=pd.concat(dfs)
            fig=px.line(dfc,x="Flow",y="SDH",color="DRA",title=f"{s['name']} System Curve")
            st.plotly_chart(fig, use_container_width=True)
    # Pump-System Interaction
    elif view == "Pump-System Interaction":
        st.markdown("<div class='section-title'>Pump vs System Interaction</div>", unsafe_allow_html=True)
        for s in sta:
            if not s['is_pump']: continue
            flows=np.linspace(0,FLOW*1.5,200)
            fig=go.Figure()
            d=s['D']-2*s['t']; kv=s['KV']; rough=s['rough']
            for dra in range(0,s['max_dr']+1,5):
                v=flows/3600/(pi*(d**2)/4);Re=v*d/(kv*1e-6)
                f=np.where(Re>4000,0.25/(np.log10(rough/d/3.7+5.74/(Re**0.9))**2),64/Re)
                DH=f*((s['L']*1000)/d)*(v**2/(2*9.81))*(1-dra/100)
                fig.add_trace(go.Scatter(x=flows,y=elev+DH,mode='lines',name=f"Sys {dra}% DRA"))
            for rpm in range(s['MinRPM'],s['DOL']+1,100):
                fig.add_trace(go.Scatter(x=flows,y=(s['A']*flows**2+s['B']*flows+s['C'])*(rpm/s['DOL'])**2,mode='lines',name=f"Pump {rpm} rpm"))
            st.plotly_chart(fig, use_container_width=True)
    # Cost Landscape
    elif view == "Cost Landscape":
        st.markdown("<div class='section-title'>3D Cost Landscape</div>", unsafe_allow_html=True)
        for s in sta:
            if not s['is_pump']: continue
            rpms=np.arange(s['MinRPM'],s['DOL']+1,100); drs=np.arange(0,s['max_dr']+1,5)
            Z=np.zeros((len(drs),len(rpms)))
            for i,dra in enumerate(drs):
                for j,rpm in enumerate(rpms):
                    H=(s['A']*FLOW**2+s['B']*FLOW+s['C'])*(rpm/s['DOL'])**2
                    fe=FLOW*s['DOL']/rpm
                    η=(s['P']*fe**4+s['Q']*fe**3+s['R']*fe**2+s['S']*fe+s['T'])/100
                    pwr=(s['rho']*FLOW*9.81*H*s['max_pumps'])/(3600*1000*η*0.95)
                    cost=pwr*24*(s['rate'] if s['power_type']=='Grid' else s['sfc']*1.34102/820*Price_HSD)
                    cost+=dra/4*(FLOW*1000*24/1e6)*RateDRA
                    Z[i,j]=cost
            fig=go3d.Figure(data=[go3d.Surface(x=rpms,y=drs,z=Z)])
            fig.update_layout(scene=dict(xaxis_title="RPM",yaxis_title="DRA(%)",zaxis_title="Cost(INR/day)"))
            st.plotly_chart(fig, use_container_width=True)
    # Nonconvex Visualizations
    elif view == "Nonconvex Visuals":
        st.markdown("<div class='section-title'>Nonconvex Cost vs Pumps & RPM</div>", unsafe_allow_html=True)
        for s in sta:
            if not s['is_pump']: continue
            pumps=np.arange(1,s['max_pumps']+1); rpms=np.arange(s['MinRPM'],s['DOL']+1,100)
            Z=np.zeros((len(pumps),len(rpms)))
            for m,pm in enumerate(pumps):
                for n,rpm in enumerate(rpms):
                    H=(s['A']*FLOW**2+s['B']*FLOW+s['C'])*(rpm/s['DOL'])**2
                    fe=FLOW*s['DOL']/rpm; η=(s['P']*fe**4+s['Q']*fe**3+s['R']*fe**2+s['S']*fe+s['T'])/100
                    pwr=(s['rho']*FLOW*9.81*H*pm)/(3600*1000*η*0.95)
                    cost=pwr*24*(s['rate'] if s['power_type']=='Grid' else s['sfc']*1.34102/820*Price_HSD)
                    Z[m,n]=cost
            fig=go3d.Figure(data=[go3d.Surface(x=rpms,y=pumps,z=Z)])
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter all inputs then click 🚀 Run Optimization to view results.")
