import os
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10

# Ensure NEOS email is set
if 'NEOS_EMAIL' not in os.environ:
    raise RuntimeError("NEOS_EMAIL environment variable not set")

def solve_pipeline(stations, terminal, FLOW, KV, rho, Rate_DRA, Price_HSD):
    P = len(stations)
    if P < 1:
        raise ValueError("At least one pumping segment required")

    m = pyo.ConcreteModel()
    m.seg  = pyo.RangeSet(1, P)
    m.node = pyo.RangeSet(1, P+1)

    # Global parameters
    m.FLOW      = pyo.Param(initialize=FLOW)
    m.KV        = pyo.Param(initialize=KV)
    m.rho       = pyo.Param(initialize=rho)
    m.Rate_DRA  = pyo.Param(initialize=Rate_DRA)
    m.Price_HSD = pyo.Param(initialize=Price_HSD)

    # Input dictionaries
    L, Dout, thickness, eps, z = {},{},{},{},{}
    SMYS, DF = {},{}
    is_pump, max_pumps = {},{}
    Acoef, Bcoef, Ccoef = {},{},{}
    Pcoef, Qcoef, Rcoef, Scoef, Tcoef = {},{},{},{},{}
    minRPM, maxRPM = {},{}
    isGrid, ElecRt, SFC = {},{},{}

    # Read station inputs
    for i, s in enumerate(stations, start=1):
        L[i]          = s['L']
        Dout[i]       = s['D']
        thickness[i]  = s['t']
        eps[i]        = s['rough']
        z[i]          = s['elev']
        SMYS[i]       = s['SMYS']
        DF[i]         = s.get('DF', 0.72)
        if s.get('is_pump', False):
            is_pump[i]       = True
            max_pumps[i]     = s['max_pumps']
            Acoef[i] = s['A']
            Bcoef[i] = s['B']
            Ccoef[i] = s['C']
            Pcoef[i] = s['P']
            Qcoef[i] = s['Q']
            Rcoef[i] = s['R']
            Scoef[i] = s['S']
            Tcoef[i] = s['T']
            minRPM[i]       = s['MinRPM']
            maxRPM[i]       = s['DOL']
            isGrid[i]       = 1 if s['power_type']=='Grid' else 0
            ElecRt[i]       = s['rate']
            SFC[i]          = s.get('SFC',0)
        else:
            is_pump[i] = False

    z[P+1]      = terminal['elevation']
    min_res     = terminal['min_residual']

    m.L      = pyo.Param(m.seg,   initialize=L)
    m.Dout   = pyo.Param(m.seg,   initialize=Dout)
    m.t      = pyo.Param(m.seg,   initialize=thickness)
    m.eps    = pyo.Param(m.seg,   initialize=eps)
    m.SMYS   = pyo.Param(m.seg,   initialize=SMYS)
    m.DF     = pyo.Param(m.seg,   initialize=DF)
    m.z      = pyo.Param(m.node,  initialize=z)

    pumps = [i for i in m.seg if is_pump.get(i,False)]
    m.pumps = pyo.Set(initialize=pumps)
    if pumps:
        m.maxP   = pyo.Param(m.pumps, initialize=max_pumps)
        m.A      = pyo.Param(m.pumps, initialize=Acoef)
        m.B      = pyo.Param(m.pumps, initialize=Bcoef)
        m.C      = pyo.Param(m.pumps, initialize=Ccoef)
        m.Pp     = pyo.Param(m.pumps, initialize=Pcoef)
        m.Qp     = pyo.Param(m.pumps, initialize=Qcoef)
        m.Rp     = pyo.Param(m.pumps, initialize=Rcoef)
        m.Sp     = pyo.Param(m.pumps, initialize=Scoef)
        m.Tp     = pyo.Param(m.pumps, initialize=Tcoef)
        m.minRPM = pyo.Param(m.pumps, initialize=minRPM)
        m.maxRPM = pyo.Param(m.pumps, initialize=maxRPM)
        m.isGrid = pyo.Param(m.pumps, initialize=isGrid)
        m.ElecRt = pyo.Param(m.pumps, initialize=ElecRt)
        m.SFC    = pyo.Param(m.pumps, initialize=SFC)

    # [rest of your existing model code remains unchanged]
    # No need to repeat it here unless there's a bug to address there too
