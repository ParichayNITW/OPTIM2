import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10

# Tell NEOS who you are
os.environ['NEOS_EMAIL'] = 'parichay.nitwarangal@gmail.com'

def solve_pipeline(FLOW, KV, rho, SFC_J, SFC_R, SFC_S, RateDRA, Price_HSD):
    """
    Full pipeline optimization model with initialization and solver checks.
    """
    model = pyo.ConcreteModel()

    # --------------------
    # PARAMETERS
    # --------------------
    # Section 1: Vadinar -> Jamnagar
    model.FLOW1   = pyo.Param(initialize=FLOW);   FLOW1   = model.FLOW1
    model.KV1     = pyo.Param(initialize=KV);     KV1     = model.KV1
    model.rho1    = pyo.Param(initialize=rho);    rho1    = model.rho1
    model.D1      = pyo.Param(initialize=0.7112); D1      = model.D1
    model.t1      = pyo.Param(initialize=0.0071374); t1   = model.t1
    model.SMYS1   = pyo.Param(initialize=52000);  SMYS1   = model.SMYS1
    model.e1      = pyo.Param(initialize=0.00004); e1     = model.e1
    model.L1      = pyo.Param(initialize=46.7);   L1      = model.L1
    model.z1      = pyo.Param(initialize=8);      z1      = model.z1
    model.d1      = pyo.Param(initialize=0.697);  d1      = model.d1
    model.DF1     = pyo.Param(initialize=0.72);   DF1     = model.DF1
    model.Rate1   = pyo.Param(initialize=9);      Rate1   = model.Rate1
    model.A1      = pyo.Param(initialize=-0.000002); A1   = model.A1
    model.B1      = pyo.Param(initialize=-0.0015);   B1   = model.B1
    model.C1      = pyo.Param(initialize=179.14);   C1    = model.C1
    model.DOL1    = pyo.Param(initialize=1500);     DOL1  = model.DOL1
    model.MinRPM1 = pyo.Param(initialize=1200);     MinRPM1 = model.MinRPM1
    model.P1      = pyo.Param(initialize=-4.161E-14); P1   = model.P1
    model.Q1      = pyo.Param(initialize=6.574E-10);  Q1   = model.Q1
    model.R1      = pyo.Param(initialize=-8.737E-06); R1 = model.R1
    model.S1      = pyo.Param(initialize=0.04924);   S1    = model.S1
    model.T1      = pyo.Param(initialize=-0.001754); T1    = model.T1
    model.RH1     = pyo.Param(initialize=50);        RH1   = model.RH1

    # Global rates & elevations
    model.Rate_DRA  = pyo.Param(initialize=RateDRA);  Rate_DRA  = model.Rate_DRA
    model.Price_HSD = pyo.Param(initialize=Price_HSD); Price_HSD = model.Price_HSD
    model.z2 = pyo.Param(initialize=24);   z2 = model.z2
    model.z3 = pyo.Param(initialize=113);  z3 = model.z3
    model.z4 = pyo.Param(initialize=232);  z4 = model.z4
    model.z5 = pyo.Param(initialize=80);   z5 = model.z5
    model.z6 = pyo.Param(initialize=23);   z6 = model.z6

    # Section 2: Jamnagar -> Rajkot
    model.FLOW2   = pyo.Param(initialize=FLOW);   FLOW2 = model.FLOW2
    model.KV2     = pyo.Param(initialize=KV);     KV2   = model.KV2
    model.rho2    = pyo.Param(initialize=rho);    rho2  = model.rho2
    model.D2      = pyo.Param(initialize=0.7112); D2    = model.D2
    model.t2      = pyo.Param(initialize=0.0071374); t2 = model.t2
    model.SMYS2   = pyo.Param(initialize=52000);  SMYS2 = model.SMYS2
    model.e2      = pyo.Param(initialize=0.00004); e2  = model.e2
    model.L2      = pyo.Param(initialize=67.9);   L2    = model.L2
    model.d2      = pyo.Param(initialize=0.697);  d2    = model.d2
    model.DF2     = pyo.Param(initialize=0.72);   DF2   = model.DF2
    model.SFC2    = pyo.Param(initialize=SFC_J);  SFC2  = model.SFC2
    model.A2      = pyo.Param(initialize=-1e-5);  A2    = model.A2
    model.B2      = pyo.Param(initialize=0.00135);B2    = model.B2
    model.C2      = pyo.Param(initialize=270.08); C2    = model.C2
    model.DOL2    = pyo.Param(initialize=3437);   DOL2 = model.DOL2
    model.MinRPM2 = pyo.Param(initialize=2750);   MinRPM2 = model.MinRPM2
    model.P2      = pyo.Param(initialize=-4.07033296e-13); P2 = model.P2
    model.Q2      = pyo.Param(initialize=3.4657688e-9);    Q2 = model.Q2
    model.R2      = pyo.Param(initialize=-1.92727273e-5); R2 = model.R2
    model.S2      = pyo.Param(initialize=6.7033189e-2);   S2 = model.S2
    model.T2      = pyo.Param(initialize=-0.1504329);     T2 = model.T2

    # Section 3: Rajkot -> Chotila
    model.FLOW3   = pyo.Param(initialize=FLOW);   FLOW3 = model.FLOW3
    model.KV3     = pyo.Param(initialize=KV);     KV3   = model.KV3
    model.rho3    = pyo.Param(initialize=rho);    rho3  = model.rho3
    model.D3      = pyo.Param(initialize=0.7112); D3    = model.D3
    model.t3      = pyo.Param(initialize=0.0071374); t3 = model.t3
    model.SMYS3   = pyo.Param(initialize=52000);  SMYS3 = model.SMYS3
    model.e3      = pyo.Param(initialize=0.00004); e3   = model.e3
    model.L3      = pyo.Param(initialize=40.2);   L3    = model.L3
    model.d3      = pyo.Param(initialize=0.697);  d3    = model.d3
    model.DF3     = pyo.Param(initialize=0.72);   DF3   = model.DF3
    model.SFC3    = pyo.Param(initialize=SFC_R);  SFC3  = model.SFC3
    model.A3      = pyo.Param(initialize=-1e-5);  A3    = model.A3
    model.B3      = pyo.Param(initialize=0.0192); B3    = model.B3
    model.C3      = pyo.Param(initialize=218.81);C3    = model.C3
    model.DOL3    = pyo.Param(initialize=2870);   DOL3 = model.DOL3
    model.MinRPM3 = pyo.Param(initialize=2296);   MinRPM3 = model.MinRPM3
    model.P3      = pyo.Param(initialize=-9.01972569e-13);P3= model.P3
    model.Q3      = pyo.Param(initialize=7.45948934e-9);  Q3= model.Q3
    model.R3      = pyo.Param(initialize=-3.19133266e-5); R3= model.R3
    model.S3      = pyo.Param(initialize=0.0815595446);  S3= model.S3
    model.T3      = pyo.Param(initialize=-0.00303811075);T3= model.T3

    # Section 4: Chotila -> Surendranagar (no pumps)
    model.FLOW4   = pyo.Param(initialize=FLOW);   FLOW4 = model.FLOW4
    model.KV4     = pyo.Param(initialize=KV);     KV4   = model.KV4
    model.rho4    = pyo.Param(initialize=rho);    rho4  = model.rho4
    model.D4      = pyo.Param(initialize=0.7112); D4    = model.D4
    model.t4      = pyo.Param(initialize=0.0071374);t4= model.t4
    model.SMYS4   = pyo.Param(initialize=52000);  SMYS4= model.SMYS4
    model.e4      = pyo.Param(initialize=0.00004); e4   = model.e4
    model.L4      = pyo.Param(initialize=60);     L4    = model.L4
    model.d4      = pyo.Param(initialize=0.697);  d4    = model.d4
    model.DF4     = pyo.Param(initialize=0.72);   DF4   = model.DF4

    # Section 5: Surendranagar -> Viramgam
    model.FLOW5   = pyo.Param(initialize=FLOW);   FLOW5 = model.FLOW5
    model.KV5     = pyo.Param(initialize=KV);     KV5   = model.KV5
    model.rho5    = pyo.Param(initialize=rho);    rho5  = model.rho5
    model.D5      = pyo.Param(initialize=0.7112); D5    = model.D5
    model.t5      = pyo.Param(initialize=0.0071374);t5= model.t5
    model.SMYS5   = pyo.Param(initialize=52000);  SMYS5= model.SMYS5
    model.e5      = pyo.Param(initialize=0.00004); e5   = model.e5
    model.L5      = pyo.Param(initialize=60);     L5    = model.L5
    model.d5      = pyo.Param(initialize=0.697);  d5    = model.d5
    model.DF5     = pyo.Param(initialize=0.72);   DF5   = model.DF5
    model.SFC5    = pyo.Param(initialize=SFC_S);  SFC5  = model.SFC5
    model.A5      = pyo.Param(initialize=-1e-5);  A5    = model.A5
    model.B5      = pyo.Param(initialize=0.0229); B5    = model.B5
    model.C5      = pyo.Param(initialize=183.59);C5    = model.C5
    model.DOL5    = pyo.Param(initialize=3437);   DOL5 = model.DOL5
    model.MinRPM5 = pyo.Param(initialize=2750);   MinRPM5 = model.MinRPM5
    model.P5      = pyo.Param(initialize=-7.24279835e-13);P5= model.P5
    model.Q5      = pyo.Param(initialize=5.08093278e-9);  Q5= model.Q5
    model.R5      = pyo.Param(initialize=-2.49506173e-5); R5= model.R5
    model.S5      = pyo.Param(initialize=0.0768906526);  S5= model.S5
    model.T5      = pyo.Param(initialize=-0.0912698413); T5= model.T5



    # --------------------
    # DECISION VARIABLES (discretized in 10‐unit steps)
    # --------------------

    # Section 1: Vadinar → Jamnagar
    model.NOP1   = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(1,3), initialize=1)
    NOP1 = model.NOP1

    # integer bounds for N1_u
    min1 = int(pyo.value(model.MinRPM1) + 9) // 10
    max1 = int(pyo.value(model.DOL1)) // 10

    model.DR1_u  = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=4)
    DR1 = model.DR1 = 10 * model.DR1_u

    model.N1_u   = pyo.Var(
        domain=pyo.NonNegativeIntegers,
        bounds=(min1, max1),
        initialize=(min1 + max1) // 2,
    )
    N1 = model.N1 = 10 * model.N1_u

    model.RH2    = pyo.Var(domain=pyo.NonNegativeReals, bounds=(50, None), initialize=50)
    RH2 = model.RH2


    # Section 2: Jamnagar → Rajkot
    model.NOP2   = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0,2), initialize=1)
    NOP2 = model.NOP2

    min2 = int(pyo.value(model.MinRPM2) + 9) // 10
    max2 = int(pyo.value(model.DOL2)) // 10

    model.DR2_u  = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=4)
    DR2 = model.DR2 = 10 * model.DR2_u

    model.N2_u   = pyo.Var(
        domain=pyo.NonNegativeIntegers,
        bounds=(min2, max2),
        initialize=(min2 + max2) // 2,
    )
    N2 = model.N2 = 10 * model.N2_u

    model.RH3    = pyo.Var(domain=pyo.NonNegativeReals, bounds=(50, None), initialize=50)
    RH3 = model.RH3


    # Section 3: Rajkot → Chotila
    model.NOP3   = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0,2), initialize=1)
    NOP3 = model.NOP3

    min3 = int(pyo.value(model.MinRPM3) + 9) // 10
    max3 = int(pyo.value(model.DOL3)) // 10

    model.DR3_u  = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=4)
    DR3 = model.DR3 = 10 * model.DR3_u

    model.N3_u   = pyo.Var(
        domain=pyo.NonNegativeIntegers,
        bounds=(min3, max3),
        initialize=(min3 + max3) // 2,
    )
    N3 = model.N3 = 10 * model.N3_u

    model.RH4    = pyo.Var(domain=pyo.NonNegativeReals, bounds=(50, None), initialize=50)
    RH4 = model.RH4


    # Section 4: Chotila → Surendranagar (no pumps, only head)
    model.RH5    = pyo.Var(domain=pyo.NonNegativeReals, bounds=(50, None), initialize=50)
    RH5 = model.RH5


    # Section 5: Surendranagar → Viramgam
    model.NOP5   = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0,2), initialize=1)
    NOP5 = model.NOP5

    min5 = int(pyo.value(model.MinRPM5) + 9) // 10
    max5 = int(pyo.value(model.DOL5)) // 10

    model.DR4_u  = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0,4), initialize=4)
    DR4 = model.DR4 = 10 * model.DR4_u

    model.N5_u   = pyo.Var(
        domain=pyo.NonNegativeIntegers,
        bounds=(min5, max5),
        initialize=(min5 + max5) // 2,
    )
    N5 = model.N5 = 10 * model.N5_u

    model.RH6    = pyo.Var(domain=pyo.NonNegativeReals, bounds=(50, None), initialize=50)
    RH6 = model.RH6




    # ----------------
    # HYDRAULIC & PUMP EQUATIONS
    # ----------------
    # Section 1: Vadinar→Jamnagar
    MAOP1 = (2*t1*(SMYS1*0.070307)*DF1/D1)*10000/rho1
    v1    = FLOW1/2/(3.414*d1*d1/4)/3600
    Re1   = v1*d1/(KV1*1e-6)
    f1    = 0.25/(log10((e1/d1/3.7)+(5.74/(Re1**0.9)))**2)
    SH1   = RH2 + (z2 - z1)
    DH1   = (f1*(L1*1000/d1)*(v1**2/(2*9.81))) * (1 - DR1/100)
    SDHR_1= SH1 + DH1
    TDHA_PUMP_1 = ((A1*FLOW1**2)+(B1*FLOW1)+C1)*(N1/DOL1)**2
    SDHA_1= RH1 + TDHA_PUMP_1*NOP1
    EFFM1 = 0.95
    PPM1  = DR1/4

    # Section 2: Jamnagar→Rajkot
    MAOP2 = (2*t2*(SMYS2*0.070307)*DF2/D2)*10000/rho2
    v2    = FLOW2/2/(3.414*d2*d2/4)/3600
    Re2   = v2*d2/(KV2*1e-6)
    f2    = 0.25/(log10((e2/d2/3.7)+(5.74/(Re2**0.9)))**2)
    SH2   = RH3 + (z3 - z2)
    DH2   = (f2*(L2*1000/d2)*(v2**2/(2*9.81))) * (1 - DR2/100)
    SDHR_2= SH2 + DH2
    TDHA_PUMP_2 = ((A2*FLOW2**2)+(B2*FLOW2)+C2)*(N2/DOL2)**2
    SDHA_2= RH2 + TDHA_PUMP_2*NOP2
    EFFM2 = 0.95
    PPM2  = DR2/4

    # Section 3: Rajkot→Chotila
    MAOP3 = (2*t3*(SMYS3*0.070307)*DF3/D3)*10000/rho3
    v3    = FLOW3/2/(3.414*d3*d3/4)/3600
    Re3   = v3*d3/(KV3*1e-6)
    f3    = 0.25/(log10((e3/d3/3.7)+(5.74/(Re3**0.9)))**2)
    SH3   = RH4 + (z4 - z3)
    DH3   = (f3*(L3*1000/d3)*(v3**2/(2*9.81))) * (1 - DR3/100)
    SDHR_3= SH3 + DH3
    TDHA_PUMP_3 = ((A3*FLOW3**2)+(B3*FLOW3)+C3)*(N3/DOL3)**2
    SDHA_3= RH3 + TDHA_PUMP_3*NOP3
    EFFM3 = 0.95
    PPM3  = DR3/4

    # Section 4: Chotila→Surendranagar
    MAOP4 = (2*t4*(SMYS4*0.070307)*DF4/D4)*10000/rho4
    v4    = FLOW4/2/(3.414*d4*d4/4)/3600
    Re4   = v4*d4/(KV4*1e-6)
    f4    = 0.25/(log10((e4/d4/3.7)+(5.74/(Re4**0.9)))**2)
    SH4   = RH5 + (z5 - z4)
    DH4   = (f4*(L4*1000/d4)*(v4**2/(2*9.81))) * (1 - DR4/100)
    SDHR_4= SH4 + DH4
    SDHA_4= RH4
    EFFM4 = 0.95
    PPM4  = PPM3

    # Section 5: Surendranagar→Viramgam
    MAOP5 = (2*t5*(SMYS5*0.070307)*DF5/D5)*10000/rho5
    v5    = FLOW5/2/(3.414*d5*d5/4)/3600
    Re5   = v5*d5/(KV5*1e-6)
    f5    = 0.25/(log10((e5/d5/3.7)+(5.74/(Re5**0.9)))**2)
    SH5   = RH6 + (z6 - z5)
    DH5   = (f5*(L5*1000/d5)*(v5**2/(2*9.81))) * (1 - DR4/100)
    SDHR_5= SH5 + DH5
    TDHA_PUMP_5 = ((A5*FLOW5**2)+(B5*FLOW5)+C5)*(N5/DOL5)**2
    SDHA_5= RH5 + TDHA_PUMP_5*NOP5
    EFFM5 = 0.95
    PPM5  = DR4/4



    # ----------------
    # PUMP EFFICIENCIES
    # ----------------
    FLOW1_EQ = FLOW1*DOL1/N1
    EFFP1 = (P1*FLOW1_EQ**4 + Q1*FLOW1_EQ**3 + R1*FLOW1_EQ**2 + S1*FLOW1_EQ + T1)/100

    FLOW2_EQ = FLOW2*DOL2/N2
    EFFP2 = (P2*FLOW2_EQ**4 + Q2*FLOW2_EQ**3 + R2*FLOW2_EQ**2 + S2*FLOW2_EQ + T2)/100

    FLOW3_EQ = FLOW3*DOL3/N3
    EFFP3 = (P3*FLOW3_EQ**4 + Q3*FLOW3_EQ**3 + R3*FLOW3_EQ**2 + S3*FLOW3_EQ + T3)/100

    FLOW5_EQ = FLOW5*DOL5/N5
    EFFP5 = (P5*FLOW5_EQ**4 + Q5*FLOW5_EQ**3 + R5*FLOW5_EQ**2 + S5*FLOW5_EQ + T5)/100


    # ----------------
    # OBJECTIVE
    # ----------------
    OF_POWER_1 = (rho1*FLOW1*9.81*TDHA_PUMP_1*NOP1)/(3600*1000*EFFP1*EFFM1)*24*Rate1
    OF_DRA_1   = (PPM1/1e6)*FLOW1*24*1000*Rate_DRA

    OF_POWER_2 = ((rho2*FLOW2*9.81*TDHA_PUMP_2*NOP2)/(3600*1000*EFFP2*EFFM2))*(SFC2*1.34102/1000/820)*1000*24*Price_HSD
    OF_DRA_2   = (PPM2/1e6)*FLOW2*24*1000*Rate_DRA

    OF_POWER_3 = ((rho3*FLOW3*9.81*TDHA_PUMP_3*NOP3)/(3600*1000*EFFP3*EFFM3))*(SFC3*1.34102/1000/820)*1000*24*Price_HSD
    OF_DRA_3   = (PPM3/1e6)*FLOW3*24*1000*Rate_DRA

    OF_POWER_4 = ((rho5*FLOW5*9.81*TDHA_PUMP_5*NOP5)/(3600*1000*EFFP5*EFFM5))*(SFC5*1.34102/1000/820)*1000*24*Price_HSD
    OF_DRA_4   = (PPM5/1e6)*FLOW5*24*1000*Rate_DRA

    def obj_rule(m):
        return OF_POWER_1 + OF_DRA_1 + OF_POWER_2 + OF_DRA_2 + OF_POWER_3 + OF_DRA_3 + OF_POWER_4 + OF_DRA_4
    model.Objf = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


    # ----------------
    # CONSTRAINTS
    # ----------------
    model.const1  = pyo.Constraint(expr=SDHA_1  >= SDHR_1)
    model.const2  = pyo.Constraint(expr=MAOP1   >= SDHA_1)
    model.const10 = pyo.Constraint(expr=SDHA_2  >= SDHR_2)
    model.const11 = pyo.Constraint(expr=MAOP2   >= SDHA_2)
    model.const19 = pyo.Constraint(expr=SDHA_3  >= SDHR_3)
    model.const20 = pyo.Constraint(expr=MAOP3   >= SDHA_3)
    model.const28 = pyo.Constraint(expr=SDHA_4  >= SDHR_4)
    model.const29 = pyo.Constraint(expr=MAOP4   >= SDHA_4)
    model.const31 = pyo.Constraint(expr=SDHA_5  >= SDHR_5)
    model.const32 = pyo.Constraint(expr=MAOP5   >= SDHA_5)




    # NEOS-only solve
    opts = {
       'tol': 1e-3,
       'acceptable_tol': 1e-3,
       'max_cpu_time': 3000,
       'max_iter': 100000
    }
    neos_mgr = SolverManagerFactory('neos')
    results = neos_mgr.solve(
        model,
        solver='couenne',    # or 'bonmin' 
        opt=opts,
        keepfiles=False,
	tee=True
    )
    model.solutions.load_from(results)


    # ----------------
    # EXTRACT
    # ----------------
    res = {
        'power_cost_vadinar':          pyo.value(OF_POWER_1),
        'dra_cost_vadinar':            pyo.value(OF_DRA_1),
        'power_cost_jamnagar':         pyo.value(OF_POWER_2),
        'dra_cost_jamnagar':           pyo.value(OF_DRA_2),
        'power_cost_rajkot':           pyo.value(OF_POWER_3),
        'dra_cost_rajkot':             pyo.value(OF_DRA_3),
        'power_cost_surendranagar':    pyo.value(OF_POWER_4),
        'dra_cost_surendranagar':      pyo.value(OF_DRA_4),
        'station_cost_vadinar':        pyo.value(OF_POWER_1+OF_DRA_1),
        'station_cost_jamnagar':       pyo.value(OF_POWER_2+OF_DRA_2),
        'station_cost_rajkot':         pyo.value(OF_POWER_3+OF_DRA_3),
        'station_cost_surendranagar':  pyo.value(OF_POWER_4+OF_DRA_4),
        'residual_head_vadinar':       pyo.value(RH1),
        'residual_head_jamnagar':      pyo.value(RH2),
        'residual_head_rajkot':        pyo.value(RH3),
        'residual_head_chotila':       pyo.value(RH4),
        'residual_head_surendranagar': pyo.value(RH5),
        'residual_head_viramgam':      pyo.value(RH6),
        'sdh_vadinar':                 pyo.value(SDHA_1),
        'sdh_jamnagar':      	       pyo.value(SDHA_2),
        'sdh_rajkot':                  pyo.value(SDHA_3),
        'sdh_chotila':                 pyo.value(SDHA_4),
        'sdh_surendranagar':           pyo.value(SDHA_5),
        'num_pumps_vadinar':           pyo.value(model.NOP1),
        'num_pumps_jamnagar':          pyo.value(model.NOP2),
        'num_pumps_rajkot':            pyo.value(model.NOP3),
        'num_pumps_surendranagar':     pyo.value(model.NOP5),
        'speed_vadinar':               pyo.value(model.N1),
        'speed_jamnagar':              pyo.value(model.N2),
        'speed_rajkot':                pyo.value(model.N3),
        'speed_surendranagar':         pyo.value(model.N5),
        'efficiency_vadinar':          pyo.value(EFFP1)*100,
        'efficiency_jamnagar':         pyo.value(EFFP2)*100,
        'efficiency_rajkot':           pyo.value(EFFP3)*100,
        'efficiency_surendranagar':    pyo.value(EFFP5)*100,
	'drag_reduction_vadinar':      pyo.value(DR1),
        'drag_reduction_jamnagar':     pyo.value(DR2),
        'drag_reduction_rajkot':       pyo.value(DR3),
        'drag_reduction_surendranagar':pyo.value(DR4),
        'reynolds_vadinar':            pyo.value(Re1),
        'reynolds_jamnagar':           pyo.value(Re2),
        'reynolds_rajkot':             pyo.value(Re3),
        'reynolds_surendranagar':      pyo.value(Re5),
        'head_loss_vadinar':           pyo.value(DH1),
        'head_loss_jamnagar':          pyo.value(DH2),
        'head_loss_rajkot':            pyo.value(DH3),
        'head_loss_surendranagar':     pyo.value(DH5),
        'velocity_vadinar':            pyo.value(v1),
        'velocity_jamnagar':           pyo.value(v2),
        'velocity_rajkot':             pyo.value(v3),
        'velocity_surendranagar':      pyo.value(v5),
        'total_cost':                  pyo.value(model.Objf),

        # Pump-curve parameters:
        'coef_A_vadinar':    pyo.value(A1),
        'coef_B_vadinar':    pyo.value(B1),
        'coef_C_vadinar':    pyo.value(C1),
        'dol_vadinar':       pyo.value(DOL1),
        'min_rpm_vadinar':   pyo.value(MinRPM1),
        'coef_P_vadinar':    pyo.value(P1),
        'coef_Q_vadinar':    pyo.value(Q1),
        'coef_R_vadinar':    pyo.value(R1),
        'coef_S_vadinar':    pyo.value(S1),
        'coef_T_vadinar':    pyo.value(T1),
        'coef_A_jamnagar':   pyo.value(A2),
        'coef_B_jamnagar':   pyo.value(B2),
        'coef_C_jamnagar':   pyo.value(C2),
        'dol_jamnagar':      pyo.value(DOL2),
        'min_rpm_jamnagar':  pyo.value(MinRPM2),
        'coef_P_jamnagar':   pyo.value(P2),
        'coef_Q_jamnagar':   pyo.value(Q2),
        'coef_R_jamnagar':   pyo.value(R2),
        'coef_S_jamnagar':   pyo.value(S2),
        'coef_T_jamnagar':   pyo.value(T2),
        'coef_A_rajkot':     pyo.value(A3),
        'coef_B_rajkot':     pyo.value(B3),
        'coef_C_rajkot':     pyo.value(C3),
        'dol_rajkot':        pyo.value(DOL3),
        'min_rpm_rajkot':    pyo.value(MinRPM3),
        'coef_P_rajkot':     pyo.value(P3),
        'coef_Q_rajkot':     pyo.value(Q3),
        'coef_R_rajkot':     pyo.value(R3),
        'coef_S_rajkot':     pyo.value(S3),
        'coef_T_rajkot':     pyo.value(T3),
        'coef_A_surendranagar':  pyo.value(A5),
        'coef_B_surendranagar':  pyo.value(B5),
        'coef_C_surendranagar':  pyo.value(C5),
        'dol_surendranagar':     pyo.value(DOL5),
        'min_rpm_surendranagar': pyo.value(MinRPM5),
        'coef_P_surendranagar':  pyo.value(P5),
        'coef_Q_surendranagar':  pyo.value(Q5),
        'coef_R_surendranagar':  pyo.value(R5),
        'coef_S_surendranagar':  pyo.value(S5),
        'coef_T_surendranagar':  pyo.value(T5),
    }
    return res
