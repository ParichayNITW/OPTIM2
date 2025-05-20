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

# Custom CSS
st.markdown("""
<style>
.section-title {
  font-size:1.2rem; font-weight:600; margin-top:1rem;
  color: var(--text-primary-color);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Mixed Integer Nonlinear Optimization of Pipeline Operations</h1>", unsafe_allow_html=True)

# Solver wrapper
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

# --- Sidebar: Global inputs and view selector --------------------------------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Global Fluid & Cost Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)", value=1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)

    st.subheader("Stations List")
    add_col, rem_col = st.columns(2)
    if add_col.button("➕ Add Station"):
        n = len(st.session_state.get('stations', [])) + 1
        default = {
            'name': f'Station {n}', 'elev':0.0, 'D':0.711, 't':0.007,
            'SMYS':52000.0, 'rough':0.00004, 'L':50.0,
            'min_residual':50.0, 'is_pump':False,
            'power_type':'Grid','rate':9.0,'sfc':150.0,
            'max_pumps':1,'MinRPM':1200.0,'DOL':1500.0,'max_dr':0.0
        }
        st.session_state.stations.append(default)
    if rem_col.button("🗑️ Remove Station"):
        if st.session_state.get('stations'):
            st.session_state.stations.pop()

    view = st.radio("Show results for:", [
        "Summary","Cost Breakdown","Performance",
        "System Curves","Pump-System Interaction",
        "Cost Landscape","Nonconvex Visuals"
    ])

# Initialize station list if needed
if 'stations' not in st.session_state:
    st.session_state.stations = [{
        'name':'Station 1','elev':0.0,'D':0.711,'t':0.007,
        'SMYS':52000.0,'rough':0.00004,'L':50.0,
        'min_residual':50.0,'is_pump':False,
        'power_type':'Grid','rate':9.0,'sfc':150.0,
        'max_pumps':1,'MinRPM':1200.0,'DOL':1500.0,'max_dr':0.0
    }]

# --- Main: Station parameter inputs ----------------------------------------
for idx, stn in enumerate(st.session_state.stations, start=1):
    with st.expander(f"Station {idx}", expanded=True):
        stn['name'] = st.text_input("Name", stn['name'], key=f"name{idx}")
        stn['elev'] = st.number_input("Elevation (m)", stn['elev'], step=0.1, key=f"elev{idx}")
        # station-specific viscosity & density
        stn['KV']  = st.number_input("Viscosity (cSt)", stn.get('KV',10.0), step=0.1, key=f"kv{idx}")
        stn['rho'] = st.number_input("Density (kg/m³)", stn.get('rho',850.0), step=1.0, key=f"rho{idx}")
        if idx == 1:
            stn['min_residual'] = st.number_input(
                "Residual Head at Station (m)", stn.get('min_residual',50.0), step=0.1, key=f"res{idx}"
            )
        stn['D']     = st.number_input("Outer Diameter (m)", stn['D'], format="%.3f", step=0.001, key=f"D{idx}")
        stn['t']     = st.number_input("Wall Thickness (m)", stn['t'], format="%.4f", step=1e-4, key=f"t{idx}")
        stn['SMYS']  = st.number_input("SMYS (psi)", stn['SMYS'], step=1000.0, key=f"SMYS{idx}")
        stn['rough'] = st.number_input("Pipe Roughness (m)", stn['rough'], format="%.5f", step=1e-5, key=f"rough{idx}")
        stn['L']     = st.number_input("Length to next (km)", stn['L'], step=1.0, key=f"L{idx}")
        stn['is_pump'] = st.checkbox("Pumping Station?", stn['is_pump'], key=f"pump{idx}")

        if stn['is_pump']:
            stn['power_type'] = st.selectbox(
                "Power Source", ["Grid","Diesel"],
                index=0 if stn['power_type']=="Grid" else 1,
                key=f"ptype{idx}"
            )
            if stn['power_type']=="Grid":
                stn['rate'] = st.number_input("Electricity Rate (INR/kWh)", stn['rate'], key=f"rate{idx}")
                stn['sfc'] = 0.0
            else:
                stn['sfc'] = st.number_input("SFC (gm/bhp·hr)", stn['sfc'], key=f"sfc{idx}")
                stn['rate'] = 0.0
            # pump hardware & performance
            stn['max_pumps'] = st.number_input("Max Pumps", 1, stn['max_pumps'], key=f"mpumps{idx}")
            stn['MinRPM']    = st.number_input("Min RPM", stn['MinRPM'], key=f"minrpm{idx}")
            stn['DOL']       = st.number_input("Rated RPM", stn['DOL'], key=f"dol{idx}")
            stn['max_dr']    = st.number_input("Max Drag Reduction (%)", stn['max_dr'], key=f"mdr{idx}")
            st.markdown("**Pump Performance Data:**")
            dfh = pd.DataFrame({"Flow (m³/hr)": [0.0], "Head (m)": [0.0]})
            dfh = st.data_editor(dfh, num_rows='dynamic', key=f"head{idx}")
            st.session_state[f"head_{idx}"] = dfh
            dfe = pd.DataFrame({"Flow (m³/hr)": [0.0], "Eff (%)": [0.0]})
            dfe = st.data_editor(dfe, num_rows='dynamic', key=f"eff{idx}")
            st.session_state[f"eff_{idx}"] = dfe
        # elevation peaks
        st.markdown("**Elevation Peaks:**")
        pk_df = pd.DataFrame({"Location (km)": [stn['L']/2], "Elevation (m)": [stn['elev']+100]})
        pk_df = st.data_editor(pk_df, num_rows='dynamic', key=f"peak{idx}")
        st.session_state[f"peak_{idx}"] = pk_df

# Terminal inputs
st.markdown("---")
st.subheader("🏁 Terminal Station")
terminal_name = st.text_input("Name", "Terminal")
terminal_elev = st.number_input("Elevation (m)", 0.0)
terminal_head = st.number_input("Required Residual Head (m)", 50.0)

# Run button
def run_optimization():
    stations_data = st.session_state['stations']
    # attach pump curves & peaks
    for idx, stn in enumerate(stations_data,1):
        if stn['is_pump']:
            dfh = st.session_state[f"head_{idx}"]
            dfe = st.session_state[f"eff_{idx}"]
            Qh,Hh = dfh.iloc[:,0].values, dfh.iloc[:,1].values
            a,b,c = np.polyfit(Qh,Hh,2)
            stn.update({'A':a,'B':b,'C':c})
            Qe,Ee = dfe.iloc[:,0].values, dfe.iloc[:,1].values
            P,Qc,R,S,T = np.polyfit(Qe,Ee,4)
            stn.update({'P':P,'Q':Qc,'R':R,'S':S,'T':T})
        peaks=[]
        for _,row in st.session_state[f"peak_{idx}"].iterrows():
            loc,e = float(row[0]),float(row[1])
            peaks.append({'loc':loc,'elev':e})
        stn['peaks']=peaks
    res = solve_pipeline(stations_data, {'name':terminal_name,'elev':terminal_elev,'min_residual':terminal_head}, FLOW, RateDRA, Price_HSD)
    st.session_state['res']=res
    st.session_state['stations_data']=stations_data

if st.button("🚀 Run Optimization"):
    with st.spinner("Solving..."):
        run_optimization()

# Render views
if 'res' in st.session_state:
    res = st.session_state['res']
    stations_data = st.session_state['stations_data']
    # downloads
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Download Scenario CSV**")
        df_glob=pd.DataFrame([{'FLOW':FLOW,'RateDRA':RateDRA,'Price_HSD':Price_HSD,'Terminal':terminal_name,'Term_Elev':terminal_elev,'Term_Head':terminal_head}])
        st.download_button("Scenario",df_glob.to_csv(index=False).encode(),"scenario.csv","text/csv")
        df_sta=pd.DataFrame(stations_data)
        st.download_button("Stations",df_sta.to_csv(index=False).encode(),"stations.csv","text/csv")
    # Summary
    if view=="Summary":
        st.markdown("<div class='section-title'>Summary</div>",unsafe_allow_html=True)
        names=[s['name'] for s in stations_data]+[terminal_name]
        rows=["Power+Fuel","DRA","#Pumps","Speed(rpm)","Eff(%)","Re","HeadLoss","Vel","ResHead","SDH","DRA%"]
        data={'Process':rows}
        for nm in names:
            k=nm.lower().replace(' ','_')
            data[nm]=[res[f"power_cost_{k}"],res[f"dra_cost_{k}"],res[f"num_pumps_{k}"],res[f"speed_{k}"],res[f"efficiency_{k}"],res[f"reynolds_{k}"],res[f"head_loss_{k}"],res[f"velocity_{k}"],res[f"residual_head_{k}"],res[f"sdh_{k}"],res[f"drag_reduction_{k}"]]
        st.dataframe(pd.DataFrame(data),use_container_width=True)
    # Cost Breakdown
    elif view=="Cost Breakdown":
        st.markdown("<div class='section-title'>Costs</div>",unsafe_allow_html=True)
        dfc=pd.DataFrame({"Station":[s['name'] for s in stations_data],"Power+Fuel":[res[f"power_cost_{s['name'].lower().replace(' ','_')} "] for s in stations_data],"DRA":[res[f"dra_cost_{s['name'].lower().replace(' ','_')}"] for s in stations_data]})
        fig=px.bar(dfc.melt(id_vars="Station",value_vars=["Power+Fuel","DRA"],var_name="Type"),x="Station",y="value",color="Type",title="Daily Costs")
        st.plotly_chart(fig,use_container_width=True)
    # Performance
    elif view=="Performance":
        tabs=st.tabs(["Head Loss","Vel+Re","Pump H-Q","Pump Eff","Power-Speed","Power-Flow"])
        # implement each tab similarly...
    # System Curves
    elif view=="System Curves":
        st.markdown("<div class='section-title'>System Curves</div>",unsafe_allow_html=True)
        # code for system curves
    # Pump-System Interaction
    elif view=="Pump-System Interaction":
        st.markdown("<div class='section-title'>Pump-System Interaction</div>",unsafe_allow_html=True)
        # code for interaction
    # Cost Landscape
    elif view=="Cost Landscape":
        st.markdown("<div class='section-title'>Cost Landscape</div>",unsafe_allow_html=True)
        # code for landscape
    # Nonconvex Visuals
    elif view=="Nonconvex Visuals":
        st.markdown("<div class='section-title'>Nonconvex Visuals</div>",unsafe_allow_html=True)
        # code for nonconvexity
else:
    st.info("🔹 Enter inputs above and click 🚀 Run Optimization to see results.")
