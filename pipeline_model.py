import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import pandas as pd
import numpy as np

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL','youremail@example.com')

# --- Load DRA curves ---
DRA_CSV_FILES = {10:"10 cst.csv",15:"15 cst.csv",20:"20 cst.csv",25:"25 cst.csv",30:"30 cst.csv",35:"35 cst.csv",40:"40 cst.csv"}
DRA_CURVE = {}
for c,f in DRA_CSV_FILES.items():
    DRA_CURVE[c] = pd.read_csv(f) if os.path.exists(f) else None

def _interp_ppm(df,dr):
    x=df['%Drag Reduction']; y=df['PPM']
    return float(np.interp(dr, x, y, left=y.iloc[0], right=y.iloc[-1]))

def get_ppm(visc,dr):
    vs=sorted([c for c,df in DRA_CURVE.items() if df is not None])
    if not vs: return 0.0
    if visc<=vs[0]: return _interp_ppm(DRA_CURVE[vs[0]],dr)
    if visc>=vs[-1]:return _interp_ppm(DRA_CURVE[vs[-1]],dr)
    lo=max([c for c in vs if c<=visc]); hi=min([c for c in vs if c>=visc])
    p_lo=_interp_ppm(DRA_CURVE[lo],dr); p_hi=_interp_ppm(DRA_CURVE[hi],dr)
    return float(np.interp(visc,[lo,hi],[p_lo,p_hi]))


def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD, linefill=None):
    # Setup
    N=len(stations)
    pump_idx=[i+1 for i,s in enumerate(stations) if s.get('is_pump')]
    model=pyo.ConcreteModel()
    model.I=pyo.RangeSet(1,N)
    model.P=pyo.Set(initialize=pump_idx)
    model.Nodes=pyo.RangeSet(1,N+1)

    # Params
    model.F=pyo.Param(initialize=FLOW)
    model.KV=pyo.Param(model.I,initialize={i+1:KV_list[i] for i in range(N)})
    model.rho=pyo.Param(model.I,initialize={i+1:rho_list[i] for i in range(N)})
    model.Rate_DRA=pyo.Param(initialize=RateDRA)
    model.Price_HSD=pyo.Param(initialize=Price_HSD)

    # Segment flows
    seg=[FLOW]
    for s in stations: seg.append(seg[-1]-s.get('delivery',0)+s.get('supply',0))

    # Geometry & pumps
    length={}; d={}; rough={}; t={}; elev={}; peaks={}
    A=B=C=P=Q=R=S=T={}; sfc={}; ec={}; maxdr={}; visc={}
    for i,st in enumerate(stations,1):
        length[i]=st.get('L',0)
        D_out=st.get('D',st.get('d',0.7)); t[i]=st.get('t',0.007)
        d[i]=D_out-2*t[i]; rough[i]=st.get('rough',0.00004)
        elev[i]=st.get('elev',0); peaks[i]=st.get('peaks',[])
        if st.get('is_pump'):
            A[i]=st['A'];B[i]=st['B'];C[i]=st['C']
            P[i]=st['P'];Q[i]=st['Q'];R[i]=st['R'];S[i]=st['S'];T[i]=st['T']
            sfc[i]=st.get('sfc',0); ec[i]=st.get('rate',0)
            maxdr[i]=st.get('max_dr',40); visc[i]=st.get('viscosity',10)
    elev[N+1]=terminal.get('elev',0)

    model.L=pyo.Param(model.I,initialize=length)
    model.d=pyo.Param(model.I,initialize=d)
    model.e=pyo.Param(model.I,initialize=rough)
    model.z=pyo.Param(model.Nodes,initialize=elev)

    # Vars
    model.NOP=pyo.Var(model.P,domain=pyo.NonNegativeIntegers,
        bounds=lambda m,i:(1 if i==1 else 0,stations[i-1].get('max_pumps',2)),init=1)
    model.Nu=pyo.Var(model.P,domain=pyo.NonNegativeIntegers,
        bounds=lambda m,i:(1,(stations[i-1].get('DOL',1000)//10)),init=1)
    model.N=pyo.Expression(model.P,rule=lambda m,i:10*m.Nu[i])
    model.DRu=pyo.Var(model.P,domain=pyo.NonNegativeIntegers,
        bounds=lambda m,i:(0,int(maxdr[i]//10)),init=0)
    model.DR=pyo.Expression(model.P,rule=lambda m,i:10*m.DRu[i])
    model.RH=pyo.Var(model.Nodes,domain=pyo.NonNegativeReals)
    model.RH[1].fix(stations[0].get('min_residual',50))
    for j in range(2,N+2): model.RH[j].setlb(50)

    # Hydraulics
    g=9.81; vel={}; Re={}; fr={}
    for i in model.I:
        Qms=seg[i]/3600; A=pi*(d[i]**2)/4
        vel[i]=Qms/A if A>0 else 0
        Re[i]=vel[i]*d[i]/(model.KV[i]*1e-6) if model.KV[i]>0 else 0
        fr[i]=64/Re[i] if Re[i]<4000 and Re[i]>0 else 0.25/(log10((rough[i]/d[i]/3.7)+(5.74/(Re[i]**0.9)))**2)

    # SDH + pump
    model.SDH=pyo.Var(model.I,domain=pyo.NonNegativeReals)
    cts=pyo.ConstraintList(); TDH={}; EFF={}
    for i in model.I:
        drf=(model.DR[i]/100) if i in model.P else 0
        dh=fr[i]*(length[i]*1000/d[i])*(vel[i]**2/(2*g))*(1-drf)
        cts.add(model.SDH[i]>=model.RH[i+1]+(model.z[i+1]-model.z[i])+dh)
        if i in model.P:
            Qp=seg[i]/3600
            TDH[i]=(A[i]*Qp**2+B[i]*Qp+C[i])*((model.N[i]/stations[i-1]['DOL'])**2)
            r=Qp*stations[i-1]['DOL']/model.N[i]
            EFF[i]=(P[i]*r**4+Q[i]*r**3+R[i]*r**2+S[i]*r+T[i])/100
        else: TDH[i]=0;EFF[i]=1

    # Objective
    cost=0
    for i in model.P:
        rh=model.rho[i]; Qh=seg[i]
        Pkw=(rh*Qh*9.81*TDH[i]*model.NOP[i])/(3600*1000*EFF[i]*0.95)
        pc=Pkw*24*(ec[i] if i in electric_pumps else (sfc[i]*1.34102/820*Price_HSD))
        dc=get_ppm(visc[i],model.DR[i])*Qh*24/1000*model.Rate_DRA
        cost+=pc+dc
    model.Obj=pyo.Objective(expr=cost,sense=pyo.minimize)

    # Solve
    res=SolverManagerFactory('neos').solve(model,solver='couenne',tee=False)
    model.solutions.load_from(res)

    # Results
    out={}
    for i in model.P:
        out[f"power_cost_{i}"]=float(pc)
        out[f"dra_cost_{i}"]=float(dc)
    out['total_cost']=float(pyo.value(model.Obj))
    return out
