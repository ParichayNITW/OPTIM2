import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly import graph_objects as go3d
from math import pi
from io import BytesIO
from pyomo.opt import SolverManagerFactory

# NEOS email
if 'NEOS_EMAIL' in st.secrets:
    os.environ['NEOS_EMAIL'] = st.secrets['NEOS_EMAIL']
else:
    st.error("🛑 Please set NEOS_EMAIL in Streamlit secrets.")

# Page config
st.set_page_config(page_title="Pipeline Optimization", layout="wide")

# CSS
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

# Solver wrapper
def solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD):
    import pipeline_model
    return pipeline_model.solve_pipeline(stations, terminal, FLOW, RateDRA, Price_HSD)

# --- Sidebar Inputs & View Selector --------------------------------------
with st.sidebar:
    st.title("🔧 Inputs")
    with st.expander("Global Fluid & Cost", expanded=True):
        FLOW      = st.number_input("Flow (m³/hr)", 1000.0, step=10.0)
        RateDRA   = st.number_input("DRA Cost (INR/L)", 500.0, step=1.0)
        Price_HSD = st.number_input("Diesel Price (INR/L)", 70.0, step=0.5)
    st.subheader("Stations")
    add_c, rem_c = st.columns(2)
    if add_c.button("➕ Add Station"): st.session_state.stations.append({
        'name':'Station', 'elev':0.0,'D':0.711,'t':0.007,'SMYS':52000,'rough':4e-5,'L':50.0,
        'min_residual':50.0,'is_pump':False,'power_type':'Grid','rate':9.0,'sfc':150.0,
        'max_pumps':0,'MinRPM':1200,'DOL':1500,'max_dr':0
    })
    if rem_c.button("🗑️ Remove Station") and len(st.session_state.stations)>1:
        st.session_state.stations.pop()
    view = st.radio("View", [
        "Summary","Cost Breakdown","Performance",
        "System Curves","Pump-System Interaction",
        "Cost Landscape","Nonconvex Visuals"
    ])

# Initialize stations
if 'stations' not in st.session_state:
    st.session_state.stations = [{}]
    st.session_state.stations[0].update({
        'name':'Station 1','elev':0.0,'D':0.711,'t':0.007,'SMYS':52000,'rough':4e-5,'L':50.0,
        'min_residual':50.0,'is_pump':False,'power_type':'Grid','rate':9.0,'sfc':150.0,
        'max_pumps':0,'MinRPM':1200,'DOL':1500,'max_dr':0
    })

# Station inputs
for i, stn in enumerate(st.session_state.stations,1):
    with st.expander(f"Station {i}"):
        stn['name']=st.text_input("Name", stn.get('name',f"Station {i}"), key=f"name{i}")
        stn['elev']=st.number_input("Elev (m)", stn.get('elev',0.0), step=0.1, key=f"elev{i}")
        stn['KV']=st.number_input("Viscosity (cSt)", stn.get('KV',10.0), step=0.1, key=f"kv{i}")
        stn['rho']=st.number_input("Density (kg/m³)", stn.get('rho',850), step=1, key=f"rho{i}")
        if i==1: stn['min_residual']=st.number_input("Min RH (m)", stn.get('min_residual',50), step=0.1, key=f"res{i}")
        stn['D']=st.number_input("Outer D (m)", stn.get('D',0.7),format="%.3f",step=0.001, key=f"D{i}")
        stn['t']=st.number_input("t (m)", stn.get('t',0.007),format="%.4f",step=1e-4, key=f"t{i}")
        stn['rough']=st.number_input("Rough (m)", stn.get('rough',4e-5),format="%.5f",step=1e-5, key=f"rough{i}")
        stn['L']=st.number_input("Length to next (km)", stn.get('L',50),step=1, key=f"L{i}")
        stn['is_pump']=st.checkbox("Pump?", stn.get('is_pump',False), key=f"pump{i}")
        if stn['is_pump']:
            stn['power_type']=st.selectbox("Power",["Grid","Diesel"], key=f"ptype{i}")
            if stn['power_type']=='Grid':
                stn['rate']=st.number_input("Rate (INR/kWh)", stn.get('rate',9), key=f"rate{i}")
                stn['sfc']=0
            else:
                stn['sfc']=st.number_input("SFC (gm/bhp·hr)", stn.get('sfc',150), key=f"sfc{i}")
                stn['rate']=0
            stn['max_pumps']=st.number_input("Max Pumps",0,stn.get('max_pumps',0),step=1,key=f"mpumps{i}")
            stn['MinRPM']=st.number_input("Min RPM", stn.get('MinRPM',1200), key=f"minrpm{i}")
            stn['DOL']=st.number_input("DOL RPM", stn.get('DOL',1500), key=f"dol{i}")
            stn['max_dr']=st.number_input("Max DRA (%)", stn.get('max_dr',0), key=f"mdr{i}")
            # pump tables
            st.markdown("**Flow–Head**")
            dfh=st.data_editor(pd.DataFrame({"Q":[0.0],"H":[0.0]}), num_rows='dynamic', key=f"head{i}")
            st.session_state[f"head{i}"]=dfh
            st.markdown("**Flow–Eff**")
            dfe=st.data_editor(pd.DataFrame({"Q":[0.0],"η%":[0.0]}), num_rows='dynamic', key=f"eff{i}")
            st.session_state[f"eff{i}"]=dfe
        st.markdown("**Peaks**")
        pk=st.data_editor(pd.DataFrame({"loc_km":[stn['L']/2],"elev":[stn['elev']+100]}), num_rows='dynamic', key=f"peak{i}")
        st.session_state[f"peak{i}"]=pk

# Terminal
st.markdown("---")
st.subheader("Terminal")
term_name=st.text_input("Name","Terminal")
term_elev=st.number_input("Elev (m)",0.0)
term_min=st.number_input("Min RH (m)",50)

# Run optimization
if st.button("🚀 Run"):
    with st.spinner("Solving..."):
        for i,stn in enumerate(st.session_state.stations,1):
            if stn['is_pump']:
                Qh,Hh=st.session_state[f"head{i}"].values.T
                try: a,b,c=np.polyfit(Qh,Hh,2)
                except: st.error(f"Station {i} head fit fail"); st.stop()
                stn.update({'A':a,'B':b,'C':c})
                Qe,Ee=st.session_state[f"eff{i}"].values.T
                try: P,Qc,R,S,T=np.polyfit(Qe,Ee,4)
                except: st.error(f"Station {i} eff fit fail"); st.stop()
                stn.update({'P':P,'Q':Qc,'R':R,'S':S,'T':T})
            stn['peaks']=[{'loc':r.loc_km,'elev':r.elev} for _,r in st.session_state[f"peak{i}"].iterrows()]
        res=solve_pipeline(st.session_state.stations,{'name':term_name,'elev':term_elev,'min_residual':term_min},FLOW,RateDRA,Price_HSD)
        st.session_state['res']=res
        st.session_state['stations_data']=st.session_state.stations

# Render results
if 'res' in st.session_state:
    res=st.session_state['res']
    sta=st.session_state['stations_data']
    # downloads
    with st.sidebar:
        st.markdown("---")
        dfg=pd.DataFrame([{'FLOW':FLOW,'RateDRA':RateDRA,'Price_HSD':Price_HSD,'Terminal':term_name,'Elev':term_elev,'MinRH':term_min}])
        st.download_button("Scenario CSV",dfg.to_csv(index=False).encode(),"scenario.csv","text/csv")
        st.download_button("Stations CSV",pd.DataFrame(sta).to_csv(index=False).encode(),"stations.csv","text/csv")
    # Summary
    if view=="Summary":
        st.markdown("<div class='section-title'>Summary</div>",unsafe_allow_html=True)
        names=[s['name'] for s in sta]+[term_name]
        rows=["Cost","DRA","Pumps","Speed","Eff","Re","HLoss","Vel","ResRH","SDH","DRA%"]
        data={"Process":rows}
        for nm in names:
            k=nm.lower().replace(' ','_')
            data[nm]=[res[f"power_cost_{k}"],res[f"dra_cost_{k}"],res[f"num_pumps_{k}"],res[f"speed_{k}"],res[f"efficiency_{k}"],res[f"reynolds_{k}"],res[f"head_loss_{k}"],res[f"velocity_{k}"],res[f"residual_head_{k}"],res[f"sdh_{k}"],res[f"drag_reduction_{k}"]]
        st.dataframe(pd.DataFrame(data),use_container_width=True)
    # Cost Breakdown
    elif view=="Cost Breakdown":
        st.markdown("<div class='section-title'>Cost Breakdown</div>",unsafe_allow_html=True)
        dfc=pd.DataFrame({"Station":[s['name'] for s in sta],"Cost":[res[f"power_cost_{s['name'].lower().replace(' ','_')}" ] for s in sta],"DRA":[res[f"dra_cost_{s['name'].lower().replace(' ','_')}" ] for s in sta]})
        fig=px.bar(dfc.melt(id_vars="Station",value_vars=["Cost","DRA"],var_name="Type",value_name="INR/day"),x="Station",y="INR/day",color="Type")
        st.plotly_chart(fig,use_container_width=True)
    # Performance
    elif view=="Performance":
        tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["Head Loss","Vel & Re","Pump H-Q","Pump Eff","Pwr vs Speed","Pwr vs Flow"])
        with tab1:
            dfhl=pd.DataFrame({"Station":[s['name'] for s in sta],"H Loss":[res[f"head_loss_{s['name'].lower().replace(' ','_')}" ] for s in sta]})
            fig=go.Figure(go.Bar(x=dfhl["Station"],y=dfhl["H Loss"]))
            st.plotly_chart(fig,use_container_width=True)
        with tab2:
            dfvr=pd.DataFrame({"Station":[s['name'] for s in sta],"Vel":[res[f"velocity_{s['name'].lower().replace(' ','_')}" ] for s in sta],"Reyn":[res[f"reynolds_{s['name'].lower().replace(' ','_')}" ] for s in sta]})
            st.dataframe(dfvr)
        with tab3:
            for s in sta:
                if not s['is_pump']: continue
                key=s['name'].lower().replace(' ','_')
                flows=np.linspace(0,FLOW*1.2,100)
                fig=go.Figure()
                for rpm in np.arange(s['MinRPM'],s['DOL']+1,100): fig.add_trace(go.Scatter(x=flows,y=(s['A']*flows**2+s['B']*flows+s['C'])*(rpm/s['DOL'])**2,mode='lines',name=f"{rpm}rpm"))
                st.plotly_chart(fig,title=f"{s['name']} H-Q",use_container_width=True)
        with tab4:
            for s in sta:
                if not s['is_pump']: continue
                key=s['name'].lower().replace(' ','_')
                flows=np.linspace(0,FLOW*1.2,100)
                fig=go.Figure()
                for rpm in np.arange(s['MinRPM'],s['DOL']+1,100):
                    fe=FLOW*s['DOL']/rpm; eff=(s['P']*fe**4+s['Q']*fe**3+s['R']*fe**2+s['S']*fe+s['T'])/100
                    fig.add_trace(go.Scatter(x=flows,y=[eff]*len(flows),mode='lines',name=f"{rpm}rpm"))
                st.plotly_chart(fig,title=f"{s['name']} Eff-Q",use_container_width=True)
        with tab5:
            for s in sta:
                if not s['is_pump']: continue
                key=s['name'].lower().replace(' ','_')
                rpms=np.arange(s['MinRPM'],s['DOL']+1,100)
                rates=[]
                for rpm in rpms:
                    H=(s['A']*FLOW**2+s['B']*FLOW+s['C'])*(rpm/s['DOL'])**2
                    fe=FLOW*s['DOL']/rpm; eff=(s['P']*fe**4+s['Q']*fe**3+s['R']*fe**2+s['S']*fe+s['T'])/100
                    pwr=(s['rho']*FLOW*9.81*H*s['max_pumps'])/(3600*1000*eff*0.95)
                    rates.append(pwr)
                fig=go.Figure(go.Scatter(x=rpms,y=rates,mode='lines+markers'))
                st.plotly_chart(fig,title=f"{s['name']} Pwr-Speed",use_container_width=True)
        with tab6:
            for s in sta:
                if not s['is_pump']: continue
                flows=np.linspace(0,FLOW*1.2,100)
                powers=[]
                key=s['name'].lower().replace(' ','_')
                sp=FLOW
                for q in flows:
                    H=(s['A']*q**2+s['B']*q+s['C'])*(res[f"speed_{key}"]/s['DOL'])**2
                    fe=q*s['DOL']/res[f"speed_{key}"]; eff=(s['P']*fe**4+s['Q']*fe**3+s['R']*fe**2+s['S']*fe+s['T'])/100
                    pwr=(s['rho']*q*9.81*H*s['max_pumps'])/(3600*1000*eff*0.95)
                    powers.append(pwr)
                fig=go.Figure(go.Scatter(x=flows,y=powers,mode='lines'))
                st.plotly_chart(fig,title=f"{s['name']} Pwr-Flow",use_container_width=True)
    # System Curves
    elif view=="System Curves":
        st.markdown("<div class='section-title'>System Curves</div>",unsafe_allow_html=True)
        for s in sta:
            if not s['is_pump']: continue
            d=s['D']-2*s['t']; kv=s['KV']; rough=s['rough']; L=s['L']; elev=s['elev']
            dfs=[]
            for dra in range(0,int(s['max_dr'])+1,5):
                v=np.linspace(0,FLOW,100)/3600/(pi*(d**2)/4)
                Re=v*d/(kv*1e-6); f=np.where(Re>4000,0.25/(np.log10(rough/d/3.7+5.74/(Re**0.9))**2),64/Re)
                DH=f*((L*1000)/d)*(v**2/(2*9.81))*(1-dra/100); SDH=elev+DH
                dfs.append(pd.DataFrame({"Q":np.linspace(0,FLOW,100),"SDH":SDH,"DRA":dra}))
            dfc=pd.concat(dfs)
            fig=px.line(dfc,x="Q",y="SDH",color="DRA",title=f"{s['name']} Sys Curve")
            st.plotly_chart(fig,use_container_width=True)
    # Pump-System Interaction
    elif view=="Pump-System Interaction":
        st.markdown("<div class='section-title'>Pump-System Interaction</div>",unsafe_allow_html=True)
        for s in sta:
            if not s['is_pump']: continue
            flows=np.linspace(0,FLOW*1.5,200); d=s['D']-2*s['t']; kv=s['KV']; rough=s['rough']
            fig=go.Figure()
            for dra in range(0,int(s['max_dr'])+1,5):
                v=flows/3600/(pi*(d**2)/4);Re=v*d/(kv*1e-6)
                f=np.where(Re>4000,0.25/(np.log10(rough/d/3.7+5.74/(Re**0.9))**2),64/Re)
                DH=f*((s['L']*1000)/d)*(v**2/(2*9.81))*(1-dra/100); fig.add_trace(go.Scatter(x=flows,y=s['elev']+DH,mode='lines',name=f"Sys {dra}%"));
            for rpm in range(s['MinRPM'],s['DOL']+1,100): fig.add_trace(go.Scatter(x=flows,y=(s['A']*flows**2+s['B']*flows+s['C'])*(rpm/s['DOL'])**2,mode='lines',name=f"Pump {rpm}rpm"))
            fig.update_layout(title=f"{s['name']} Interaction",xaxis_title="Flow",yaxis_title="Head")
            st.plotly_chart(fig,use_container_width=True)
    # Cost Landscape
    elif view=="Cost Landscape":
        st.markdown("<div class='section-title'>3D Cost Landscape</div>",unsafe_allow_html=True)
        for s in sta:
            if not s['is_pump']: continue
            rpms=np.arange(s['MinRPM'],s['DOL']+1,100); drs=np.arange(0,int(s['max_dr'])+1,5)
            Z=np.zeros((len(drs),len(rpms)))
            for i,dra in enumerate(drs):
                for j,rpm in enumerate(rpms):
                    H=(s['A']*FLOW**2+s['B']*FLOW+s['C'])*(rpm/s['DOL'])**2
                    fe=FLOW*s['DOL']/rpm; eff=(s['P']*fe**4+s['Q']*fe**3+s['R']*fe**2+s['S']*fe+s['T'])/100
                    pwr=(s['rho']*FLOW*9.81*H*s['max_pumps'])/(3600*1000*eff*0.95)
                    cost=pwr*24*(s['rate'] if s['power_type']=='Grid' else s['sfc']*1.34102/820*Price_HSD)
                    cost+=dra/4*(FLOW*1000*24/1e6)*RateDRA
                    Z[i,j]=cost
            fig=go3d.Figure(data=[go3d.Surface(x=rpms,y=drs,z=Z)])
            fig.update_layout(title=f"{s['name']} Cost 3D",scene=dict(xaxis_title="RPM",yaxis_title="DRA%",zaxis_title="INR/day"))
            st.plotly_chart(fig,use_container_width=True)
    # Nonconvex Visuals
    elif view=="Nonconvex Visuals":
        st.markdown("<div class='section-title'>Nonconvexity</div>",unsafe_allow_html=True)
        # e.g. cost vs pumps & RPM
        for s in sta:
            if not s['is_pump']: continue
            pumps=np.arange(1,s['max_pumps']+1); rpms=np.arange(s['MinRPM'],s['DOL']+1,100)
            Z=np.zeros((len(pumps),len(rpms)))
            for i,npumps in enumerate(pumps):
                for j,rpm in enumerate(rpms):
                    H=(s['A']*FLOW**2+s['B']*FLOW+s['C'])*(rpm/s['DOL'])**2
                    fe=FLOW*s['DOL']/rpm; eff=(s['P']*fe**4+s['Q']*fe**3+s['R']*fe**2+s['S']*fe+s['T'])/100
                    pwr=(s['rho']*FLOW*9.81*H*npumps)/(3600*1000*eff*0.95)
                    cost=pwr*24*(s['rate'] if s['power_type']=='Grid' else s['sfc']*1.34102/820*Price_HSD)
                    Z[i,j]=cost
            fig=go3d.Figure(data=[go3d.Surface(x=rpms,y=pumps,z=Z)])
            fig.update_layout(title=f"{s['name']} Nonconvex",scene=dict(xaxis_title="RPM",yaxis_title="#Pumps",zaxis_title="Cost"))
            st.plotly_chart(fig,use_container_width=True)
else:
    st.info("Enter inputs and click 🚀 Run to view results.")
